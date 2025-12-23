#!/usr/bin/env bash
set -euo pipefail

# Prevent overlapping runs - exit immediately if another instance is running
LOCKFILE="/tmp/live-score.lock"
exec 200>"${LOCKFILE}"
if ! flock -n 200; then
  echo "[live] Another instance is already running. Exiting."
  exit 0
fi

DATA_ROOT="${PROJECTIONS_DATA_ROOT:-/home/daniel/projections-data}"
START_DATE="${LIVE_START_DATE:-$(date -I)}"
END_DATE="${LIVE_END_DATE:-$START_DATE}"
SEASON="${LIVE_SEASON:-$(date +%Y)}"
MONTH="${LIVE_MONTH:-$(date +%-m)}"
TIP_LEAD_MINUTES="${LIVE_TIP_LEAD_MINUTES:-90}"         # start 5-min mode 1.5h before first tip
PRE_WINDOW_INTERVAL="${LIVE_PRE_WINDOW_INTERVAL:-1800}" # 30 minutes (in seconds) for pre-slate runs
LOCK_BUFFER_MINUTES="${LIVE_LOCK_BUFFER_MINUTES:-0}"
LAST_RUN_FILE="/tmp/live-score-last-run-${START_DATE}"
# Read reconcile mode from config JSON if present, else fall back to env var or default
CONFIG_RECONCILE=$(jq -r '.reconcile_team_minutes // empty' config/minutes_current_run.json 2>/dev/null || true)
# Default to 'none' - sim handles reconciliation internally via game script sampling
RECONCILE_MODE="${LIVE_RECONCILE_MODE:-${CONFIG_RECONCILE:-none}}"
MINUTES_OUTPUT_MODE="${LIVE_MINUTES_OUTPUT:-conditional}"
SIM_PROFILE="${LIVE_SIM_PROFILE:-sim_v3}"
SIM_WORLDS="${LIVE_SIM_WORLDS:-20000}"
RUN_SIM="${LIVE_RUN_SIM:-1}"
DISABLE_TIP_WINDOW="${LIVE_DISABLE_TIP_WINDOW:-0}" # enabled by default - respect game schedule
PIN_PROJECTIONS_RUN="${LIVE_PIN_PROJECTIONS_RUN:-0}" # optional: pin unified projections run for debugging
SCRAPE_FLAGS="${LIVE_SCRAPE_FLAGS:-}"

export PROJECTIONS_DATA_ROOT="${DATA_ROOT}"

cd /home/daniel/projects/projections-v2

# Determine phase: PRE_WINDOW (8am-1.5h before tip), SLATE (1.5h before - last tip), POST (after last tip)
NOW_UTC=$(date -u +%s)
SCHEDULE_PATH="${DATA_ROOT}/silver/schedule/season=${SEASON}/month=$(printf "%02d" "${MONTH}")/schedule.parquet"
PHASE="ALWAYS_RUN"  # default if no schedule

if [[ -f "${SCHEDULE_PATH}" ]] && [[ "${DISABLE_TIP_WINDOW}" != "1" ]]; then
  PHASE_INFO=$(SCHEDULE_PATH="${SCHEDULE_PATH}" START_DATE="${START_DATE}" TIP_LEAD_MINUTES="${TIP_LEAD_MINUTES}" NOW_UTC="${NOW_UTC}" /home/daniel/.local/bin/uv run python - <<'PY'
import os, sys
import pandas as pd
from datetime import datetime, timezone

path = os.environ["SCHEDULE_PATH"]
target = os.environ["START_DATE"]
lead = int(os.environ["TIP_LEAD_MINUTES"])
now_utc = float(os.environ["NOW_UTC"])

df = pd.read_parquet(path)
df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
df = df[df["game_date"] == pd.to_datetime(target).date()]
if df.empty or "tip_ts" not in df.columns:
    print("NO_GAMES", flush=True)
    sys.exit(0)
tips = pd.to_datetime(df["tip_ts"], utc=True, errors="coerce").dropna()
if tips.empty:
    print("NO_GAMES", flush=True)
    sys.exit(0)
first_tip = tips.min()
last_tip = tips.max()
slate_start = first_tip - pd.Timedelta(minutes=lead)
now = pd.Timestamp(now_utc, unit="s", tz="UTC")

if now < slate_start:
    print(f"PRE_WINDOW {slate_start.timestamp()} {last_tip.timestamp()}", flush=True)
elif now <= last_tip:
    print(f"SLATE {slate_start.timestamp()} {last_tip.timestamp()}", flush=True)
else:
    print(f"POST {slate_start.timestamp()} {last_tip.timestamp()}", flush=True)
PY
)
  PHASE=$(echo "${PHASE_INFO}" | awk '{print $1}')
  SLATE_START_TS=$(echo "${PHASE_INFO}" | awk '{print $2}')
  LAST_TIP_TS=$(echo "${PHASE_INFO}" | awk '{print $3}')
fi

# Handle each phase
case "${PHASE}" in
  PRE_WINDOW)
    # Before slate - run every 30 minutes, skip if ran recently
    if [[ -f "${LAST_RUN_FILE}" ]]; then
      LAST_RUN=$(cat "${LAST_RUN_FILE}")
      ELAPSED=$((NOW_UTC - LAST_RUN))
      if (( ELAPSED < PRE_WINDOW_INTERVAL )); then
        echo "[live] PRE_WINDOW: Last run ${ELAPSED}s ago (interval=${PRE_WINDOW_INTERVAL}s). Skipping."
        exit 0
      fi
    fi
    echo "[live] PRE_WINDOW: Running scoring (every $((PRE_WINDOW_INTERVAL / 60))m until $(date -u -d "@${SLATE_START_TS}" +%H:%M) UTC)..."
    ;;
  SLATE)
    # Slate window - run every 5 minutes (timer handles frequency)
    echo "[live] SLATE: Active scoring until $(date -u -d "@${LAST_TIP_TS}" +%H:%M) UTC (last tip)..."
    ;;
  POST)
    echo "[live] POST: All games have tipped. Done for today."
    exit 0
    ;;
  NO_GAMES)
    echo "[live] No games scheduled for ${START_DATE}. Skipping."
    exit 0
    ;;
  *)
    echo "[live] Schedule check disabled or failed; proceeding unconditionally."
    ;;
esac

# Record this run for PRE_WINDOW interval tracking
echo "${NOW_UTC}" > "${LAST_RUN_FILE}"

# === STEP 1: SCRAPE (injuries, odds, schedule) ===
if [[ "${LIVE_SKIP_SCRAPE:-0}" == "1" ]]; then
  echo "[live] Step 1: SKIPPED (LIVE_SKIP_SCRAPE=1) - using existing data"
else
  echo "[live] Step 1: Scraping live data..."
  /home/daniel/.local/bin/uv run python -m projections.cli.live_pipeline \
    --start "${START_DATE}" \
    --end "${END_DATE}" \
    --season "${SEASON}" \
    --month "${MONTH}" \
    --data-root "${DATA_ROOT}" \
    --schedule "${SCHEDULE_PATH}" \
    ${SCRAPE_FLAGS}
fi

# Use the current time for run_as_of_ts so faster feeds (odds) aren't filtered out
# between slower feeds (injuries). build_minutes_live still applies per-game tip_ts bounds.
RUN_AS_OF_TS="${LIVE_RUN_AS_OF_TS:-$(date -u '+%Y-%m-%d %H:%M:%S')}"

echo "[live] Building features as of ${RUN_AS_OF_TS}..."

# Generate run ID using current time (ensures dashboard always shows fresh data)
LIVE_RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
BACKFILL_MODE="${LIVE_BACKFILL_MODE:-0}"

BUILD_ARGS=(
  --date "${START_DATE}"
  --run-as-of-ts "${RUN_AS_OF_TS}"
  --run-id "${LIVE_RUN_ID}"
  --lock-buffer-minutes "${LOCK_BUFFER_MINUTES}"
)
if [[ "${BACKFILL_MODE}" == "1" ]]; then
  BUILD_ARGS+=(--backfill-mode)
  echo "[live] Backfill mode enabled - relaxed roster age checks"
fi

/home/daniel/.local/bin/uv run python -m projections.cli.build_minutes_live "${BUILD_ARGS[@]}"

echo "[live] Scoring model with run_id=${LIVE_RUN_ID}..."
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

# Copy minutes to gold for sim and downstream consumers
# Use the run we just created (LIVE_RUN_ID), not a directory search
MINUTES_ARTIFACTS="${DATA_ROOT}/artifacts/minutes_v1/daily/${START_DATE}"
if [[ -d "${MINUTES_ARTIFACTS}" ]]; then
  MINUTES_SRC="${MINUTES_ARTIFACTS}/run=${LIVE_RUN_ID}/minutes.parquet"
  MINUTES_DST="${DATA_ROOT}/gold/projections_minutes_v1/game_date=${START_DATE}"
  MINUTES_RUN_DST="${MINUTES_DST}/run=${LIVE_RUN_ID}"
  if [[ -f "${MINUTES_SRC}" ]]; then
    mkdir -p "${MINUTES_RUN_DST}"
    cp "${MINUTES_SRC}" "${MINUTES_RUN_DST}/minutes.parquet"
    # Legacy flat file for compatibility with older consumers.
    mkdir -p "${MINUTES_DST}"
    cp "${MINUTES_SRC}" "${MINUTES_DST}/minutes.parquet"
    cat > "${MINUTES_DST}/latest_run.json" <<EOF
{
  "run_id": "${LIVE_RUN_ID}",
  "run_as_of_ts": "${RUN_AS_OF_TS}",
  "updated_at": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
}
EOF
    echo "[live] Copied minutes to ${MINUTES_RUN_DST} (and updated latest_run.json)"
  fi
fi

echo "[live] scoring FPTS for ${START_DATE} using production bundle (minutes_run=${LIVE_RUN_ID})..."

# Build rates features from minutes features + season aggregates
echo "[live] Building rates features..."
/home/daniel/.local/bin/uv run python -m projections.cli.build_rates_features_live \
  --date "${START_DATE}" \
  --run-id "${LIVE_RUN_ID}" \
  --data-root "${DATA_ROOT}" \
  --no-strict

# Score rates_v1 predictions for live slate (required for FPTS).
/home/daniel/.local/bin/uv run python -m projections.cli.score_rates_live \
  --date "${START_DATE}" \
  --run-id "${LIVE_RUN_ID}" \
  --features-root "${DATA_ROOT}/live/features_rates_v1" \
  --out-root "${DATA_ROOT}/gold/rates_v1_live" \
  --no-strict

# Run feature sanity check (logs anomalies, saves report)
echo "[live] Running feature sanity check..."
if ! /home/daniel/.local/bin/uv run python -m scripts.diagnostics.feature_sanity_check \
  --date "${START_DATE}" \
  --data-root "${DATA_ROOT}" \
  --output-json "${DATA_ROOT}/artifacts/diagnostics/${START_DATE}/feature_sanity.json"
then
  echo "[live] warning: Feature sanity check failed or found issues." >&2
fi

if [[ "${RUN_SIM}" == "1" ]]; then
  echo "[live] Running sim_v2 worlds + aggregation (profile=${SIM_PROFILE}, worlds=${SIM_WORLDS})..."
  if ! /home/daniel/.local/bin/uv run python -m scripts.sim_v2.run_sim_live \
    --run-date "${START_DATE}" \
    --profile-name "${SIM_PROFILE}" \
    --num-worlds "${SIM_WORLDS}" \
    --data-root "${DATA_ROOT}" \
    --run-id "${LIVE_RUN_ID}" \
    --run-as-of-ts "${RUN_AS_OF_TS}" \
    --minutes-run-id "${LIVE_RUN_ID}" \
    --rates-run-id "${LIVE_RUN_ID}"
  then
    echo "[live] warning: sim_v2 live run failed; continuing without sim_v2 projections." >&2
  fi
else
  echo "[live] Skipping sim_v2 live run (LIVE_RUN_SIM=${RUN_SIM})."
fi

# === STEP 7: OWNERSHIP PREDICTIONS (skip if any game has locked) ===
# Check if the first game on the slate has already tipped
FIRST_TIP_LOCKED=$(SCHEDULE_PATH="${SCHEDULE_PATH}" START_DATE="${START_DATE}" /home/daniel/.local/bin/uv run python - <<'PY'
import os, sys
import pandas as pd

path = os.environ["SCHEDULE_PATH"]
target = os.environ["START_DATE"]

df = pd.read_parquet(path)
df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
df = df[df["game_date"] == pd.to_datetime(target).date()]
if df.empty or "tip_ts" not in df.columns:
    print("NO_GAMES", flush=True)
    sys.exit(0)
tips = pd.to_datetime(df["tip_ts"], utc=True, errors="coerce").dropna()
if tips.empty:
    print("NOT_LOCKED", flush=True)
    sys.exit(0)
first_tip = tips.min()
now = pd.Timestamp.utcnow()
if now >= first_tip:
    print("LOCKED", flush=True)
else:
    print("NOT_LOCKED", flush=True)
PY
)

if [[ "${FIRST_TIP_LOCKED}" == "LOCKED" ]]; then
  echo "[live] Ownership scoring SKIPPED - first game has locked, using cached predictions"
else
  echo "[live] Scoring ownership predictions for ${START_DATE}..."
  if ! /home/daniel/.local/bin/uv run python -m projections.cli.score_ownership_live \
    --date "${START_DATE}" \
    --run-id "${LIVE_RUN_ID}" \
    --data-root "${DATA_ROOT}"
  then
    echo "[live] warning: Ownership scoring failed; continuing without ownership predictions." >&2
  fi
fi

# === STEP 8: FINALIZE UNIFIED PROJECTIONS ===
# Determine main slate (largest by player count, prefer lowest ID if tied) for ownership merge
MAIN_DRAFT_GROUP=$(DATA_ROOT="${DATA_ROOT}" START_DATE="${START_DATE}" /home/daniel/.local/bin/uv run python - <<'PY'
import os
from pathlib import Path
import pandas as pd

data_root = Path(os.environ["DATA_ROOT"])
game_date = os.environ["START_DATE"]

# Find all draft groups for this date
base = data_root / "gold" / "dk_salaries" / "site=dk" / f"game_date={game_date}"
if not base.exists():
    print("", flush=True)
    exit(0)

# Find largest slate by player count (prefer lowest ID if tied - main slate is usually first)
slates = []
for dg_dir in base.glob("draft_group_id=*"):
    parquet_path = dg_dir / "salaries.parquet"
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        dg_id = dg_dir.name.split("=")[1]
        slates.append((int(dg_id), len(df)))

if not slates:
    print("", flush=True)
    exit(0)

# Sort by player count (desc), then by draft_group_id (asc) to get main slate
slates.sort(key=lambda x: (-x[1], x[0]))
best_dg_id = str(slates[0][0])

print(best_dg_id, flush=True)
PY
)

if [[ -z "${MAIN_DRAFT_GROUP}" ]]; then
  echo "[live] warning: Could not determine main draft group. Finalize skipped." >&2
else
  echo "[live] Finalizing unified projections artifact for ${START_DATE} (draft_group=${MAIN_DRAFT_GROUP})..."
  if /home/daniel/.local/bin/uv run python -m projections.cli.finalize_projections \
    --date "${START_DATE}" \
    --run-id "${LIVE_RUN_ID}" \
    --minutes-run-id "${LIVE_RUN_ID}" \
    --sim-run-id "${LIVE_RUN_ID}" \
    --ownership-run-id "${LIVE_RUN_ID}" \
    --draft-group-id "${MAIN_DRAFT_GROUP}" \
    --data-root "${DATA_ROOT}"
  then
    if [[ "${PIN_PROJECTIONS_RUN}" == "1" ]]; then
      echo "[live] Pinning projections run ${LIVE_RUN_ID} (LIVE_PIN_PROJECTIONS_RUN=1)..."
      if ! /home/daniel/.local/bin/uv run python -m projections.cli.check_health pin-projections-run \
        --date "${START_DATE}" \
        --run-id "${LIVE_RUN_ID}" \
        --data-root "${DATA_ROOT}"
      then
        echo "[live] warning: Failed to pin projections run ${LIVE_RUN_ID}." >&2
      fi
    fi
  else
    echo "[live] warning: Finalize projections failed." >&2
  fi
fi

echo "[live] Running health checks..."
/home/daniel/.local/bin/uv run python -m projections.cli.check_health check-rates-sanity --date "${START_DATE}"
