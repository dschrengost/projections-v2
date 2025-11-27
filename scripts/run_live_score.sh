#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${PROJECTIONS_DATA_ROOT:-/home/daniel/projections-data}"
START_DATE="${LIVE_START_DATE:-$(date -I)}"
SEASON="${LIVE_SEASON:-$(date +%Y)}"
MONTH="${LIVE_MONTH:-$(date +%-m)}"
TIP_LEAD_MINUTES="${LIVE_TIP_LEAD_MINUTES:-60}"   # start scoring this many minutes before first tip
TIP_TAIL_MINUTES="${LIVE_TIP_TAIL_MINUTES:-180}"  # keep running this many minutes after last tip
LOCK_BUFFER_MINUTES="${LIVE_LOCK_BUFFER_MINUTES:-15}"
RECONCILE_MODE="${LIVE_RECONCILE_MODE:-none}"   # default off per operator request
MINUTES_OUTPUT_MODE="${LIVE_MINUTES_OUTPUT:-conditional}"
DISABLE_TIP_WINDOW="${LIVE_DISABLE_TIP_WINDOW:-0}" # set to 1 to bypass tip-window gating

cd /home/daniel/projects/projections-v2

# Determine whether we are within the scoring window based on today's schedule.
NOW_UTC=$(date -u +%s)
SCHEDULE_PATH="${DATA_ROOT}/silver/schedule/season=${SEASON}/month=$(printf "%02d" "${MONTH}")/schedule.parquet"
if [[ -f "${SCHEDULE_PATH}" ]]; then
  if [[ "${DISABLE_TIP_WINDOW}" == "1" ]]; then
    TIP_WINDOW="DISABLED"
    echo "[live] Tip window gating disabled via LIVE_DISABLE_TIP_WINDOW=1 (still using lead=${TIP_LEAD_MINUTES}m/tail=${TIP_TAIL_MINUTES}m for logging)."
  else
    TIP_WINDOW=$(SCHEDULE_PATH="${SCHEDULE_PATH}" START_DATE="${START_DATE}" TIP_LEAD_MINUTES="${TIP_LEAD_MINUTES}" TIP_TAIL_MINUTES="${TIP_TAIL_MINUTES}" /home/daniel/.local/bin/uv run python - <<'PY'
import os, sys
import pandas as pd
from datetime import datetime, timezone

path = os.environ["SCHEDULE_PATH"]
target = os.environ["START_DATE"]
lead = int(os.environ["TIP_LEAD_MINUTES"])
tail = int(os.environ["TIP_TAIL_MINUTES"])

df = pd.read_parquet(path)
df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
df = df[df["game_date"] == pd.to_datetime(target).date()]
if df.empty or "tip_ts" not in df.columns:
    print("MISSING", flush=True)
    sys.exit(0)
tips = pd.to_datetime(df["tip_ts"], utc=True, errors="coerce").dropna()
if tips.empty:
    print("MISSING", flush=True)
    sys.exit(0)
first_tip = tips.min()
last_tip = tips.max()
start_ts = first_tip - pd.Timedelta(minutes=lead)
end_ts = last_tip + pd.Timedelta(minutes=tail)
print(start_ts.timestamp(), end_ts.timestamp(), flush=True)
PY
)
  fi
  if [[ "${TIP_WINDOW}" != "MISSING" && "${TIP_WINDOW}" != "DISABLED" ]]; then
    START_TS=$(echo "${TIP_WINDOW}" | awk '{print $1}')
    END_TS=$(echo "${TIP_WINDOW}" | awk '{print $2}')
    if (( $(echo "${NOW_UTC} < ${START_TS}" | bc -l) )); then
      echo "[live] Skipping run: before pre-tip window (first tip lead ${TIP_LEAD_MINUTES}m)."
      exit 0
    fi
  else
    echo "[live] Schedule missing or gating disabled; proceeding without tip window gating."
  fi
else
  echo "[live] Schedule parquet not found at ${SCHEDULE_PATH}; proceeding without tip window gating."
fi

# Derive run_as_of_ts from the latest injury snapshot
RUN_AS_OF_TS=$(/home/daniel/.local/bin/uv run python - <<'PY'
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
import os

data_root = Path(os.environ.get("PROJECTIONS_DATA_ROOT", "/home/daniel/projections-data"))
season = os.environ.get("LIVE_SEASON", datetime.now(timezone.utc).year)
month = os.environ.get("LIVE_MONTH", f"{datetime.now(timezone.utc).month:02d}")
inj_path = data_root / "silver" / "injuries_snapshot" / f"season={season}" / f"month={month}" / "injuries_snapshot.parquet"
roster_path = data_root / "silver" / "roster_nightly" / f"season={season}" / f"month={month}" / "roster.parquet"

def latest_ts(path: Path, column: str) -> datetime | None:
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path, columns=[column])
        values = pd.to_datetime(df[column], utc=True, errors="coerce").dropna()
        if values.empty:
            return None
        return values.max().to_pydatetime()
    except Exception:
        return None

ts_candidates = [
    latest_ts(inj_path, "as_of_ts"),
    latest_ts(roster_path, "as_of_ts"),
]
ts_candidates = [ts for ts in ts_candidates if ts is not None]
ts = max(ts_candidates) if ts_candidates else datetime.now(timezone.utc)
print(ts.strftime("%Y-%m-%d %H:%M:%S"))
PY
)

echo "[live] Building features as of ${RUN_AS_OF_TS}..."
/home/daniel/.local/bin/uv run python -m projections.cli.build_minutes_live \
  --date "${START_DATE}" \
  --run-as-of-ts "${RUN_AS_OF_TS}" \
  --lock-buffer-minutes "${LOCK_BUFFER_MINUTES}"

LIVE_RUN_ID=$(date -u -d "${RUN_AS_OF_TS}" +%Y%m%dT%H%M%SZ)
echo "[live] Scoring model..."
SCORE_ARGS=(
  --date "${START_DATE}"
  --mode live
  --run-id "${LIVE_RUN_ID}"
  --reconcile-team-minutes "${RECONCILE_MODE}"
  --minutes-output "${MINUTES_OUTPUT_MODE}"
)
if [[ -n "${LIVE_BUNDLE_DIR:-}" ]]; then
  SCORE_ARGS+=(--bundle-dir "${LIVE_BUNDLE_DIR}")
fi

/home/daniel/.local/bin/uv run python -m projections.cli.score_minutes_v1 "${SCORE_ARGS[@]}"

echo "[live] scoring FPTS for ${START_DATE} using production bundle (minutes_run=${LIVE_RUN_ID})..."
if ! /home/daniel/.local/bin/uv run python -m projections.cli.score_fpts_v1 \
  --date "${START_DATE}" \
  --run-id "${LIVE_RUN_ID}" \
  --data-root "${DATA_ROOT}"
then
  echo "[live] warning: FPTS scoring failed; continuing without FPTS outputs." >&2
fi

echo "[live] Running health checks..."
/home/daniel/.local/bin/uv run python -m projections.cli.check_health
