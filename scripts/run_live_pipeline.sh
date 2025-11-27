#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${PROJECTIONS_DATA_ROOT:-/home/daniel/projections-data}"
START_DATE="${LIVE_START_DATE:-$(date -I)}"
END_DATE="${LIVE_END_DATE:-$START_DATE}"
SEASON="${LIVE_SEASON:-$(date +%Y)}"
MONTH="${LIVE_MONTH:-$(date +%-m)}"
FLAGS="${LIVE_FLAGS:-}"  # additional CLI flags
TIP_LEAD_MINUTES="${LIVE_TIP_LEAD_MINUTES:-60}"   # start scoring this many minutes before first tip
TIP_TAIL_MINUTES="${LIVE_TIP_TAIL_MINUTES:-180}"  # keep running this many minutes after last tip
LOCK_BUFFER_MINUTES="${LIVE_LOCK_BUFFER_MINUTES:-15}"
DISABLE_TIP_WINDOW="${LIVE_DISABLE_TIP_WINDOW:-0}"
RECONCILE_MODE="${LIVE_RECONCILE_MODE:-none}"   # default off per operator request
MINUTES_OUTPUT_MODE="${LIVE_MINUTES_OUTPUT:-conditional}"

cd /home/daniel/projects/projections-v2

# Determine whether we are within the scoring window based on today's schedule.
NOW_UTC=$(date -u +%s)
SCHEDULE_PATH="${DATA_ROOT}/silver/schedule/season=${SEASON}/month=$(printf "%02d" "${MONTH}")/schedule.parquet"
if [[ -f "${SCHEDULE_PATH}" ]]; then
  if [[ "${DISABLE_TIP_WINDOW}" == "1" ]]; then
    TIP_WINDOW="DISABLED"
    echo "[live] Tip window gating disabled via LIVE_DISABLE_TIP_WINDOW=1 (lead=${TIP_LEAD_MINUTES}m tail=${TIP_TAIL_MINUTES}m for logging)."
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

/home/daniel/.local/bin/uv run python -m projections.cli.live_pipeline \
  --start "${START_DATE}" \
  --end "${END_DATE}" \
  --season "${SEASON}" \
  --month "${MONTH}" \
  --data-root "${DATA_ROOT}" \
  --schedule "${DATA_ROOT}/silver/schedule/season=${SEASON}/month=$(printf "%02d" "${MONTH}")/schedule.parquet" \
  ${FLAGS}

if [[ "${LIVE_SCORE:-0}" != "1" ]]; then
  exit 0
fi

# Derive run_as_of_ts from the latest injury snapshot to avoid anti-leak filters
# dropping the feed; fallback to current UTC if snapshot missing.
RUN_AS_OF_TS=$(/home/daniel/.local/bin/uv run python - <<'PY'
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
import os

data_root = Path(os.environ.get("PROJECTIONS_DATA_ROOT", "/home/daniel/projections-data"))
season = os.environ.get("LIVE_SEASON", datetime.now(timezone.utc).year)
month = os.environ.get("LIVE_MONTH", f"{datetime.now(timezone.utc).month:02d}")
path = data_root / "silver" / "injuries_snapshot" / f"season={season}" / f"month={month}" / "injuries_snapshot.parquet"
ts = None
if path.exists():
    try:
        df = pd.read_parquet(path, columns=["as_of_ts"])
        values = pd.to_datetime(df["as_of_ts"], utc=True, errors="coerce").dropna()
        if not values.empty:
            ts = values.max()
    except Exception:
        ts = None
if ts is None:
    ts = datetime.now(timezone.utc)
print(ts.strftime("%Y-%m-%d %H:%M:%S"))
PY
)

/home/daniel/.local/bin/uv run python -m projections.cli.build_minutes_live \
  --date "${START_DATE}" \
  --run-as-of-ts "${RUN_AS_OF_TS}" \
  --lock-buffer-minutes "${LOCK_BUFFER_MINUTES}"

RUN_ID=$(date -u -d "${RUN_AS_OF_TS}" +%Y%m%dT%H%M%SZ)

SCORE_ARGS=(
  --date "${START_DATE}"
  --mode live
  --run-id "${RUN_ID}"
  --reconcile-team-minutes "${RECONCILE_MODE}"
  --minutes-output "${MINUTES_OUTPUT_MODE}"
)
if [[ -n "${LIVE_BUNDLE_DIR:-}" ]]; then
  SCORE_ARGS+=(--bundle-dir "${LIVE_BUNDLE_DIR}")
fi

/home/daniel/.local/bin/uv run python -m projections.cli.score_minutes_v1 "${SCORE_ARGS[@]}"

echo "[live] scoring FPTS for ${START_DATE} using production bundle (minutes_run=${RUN_ID})."
if ! /home/daniel/.local/bin/uv run python -m projections.cli.score_fpts_v1 \
  --date "${START_DATE}" \
  --run-id "${RUN_ID}" \
  --data-root "${DATA_ROOT}"
then
  echo "[live] warning: FPTS scoring failed; continuing without FPTS outputs." >&2
fi
