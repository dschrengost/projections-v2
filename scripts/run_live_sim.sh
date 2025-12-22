#!/usr/bin/env bash
# Run live simulation for a slate date.
#
# This script:
# 1. Ensures minutes predictions are in the expected location for the simulator
# 2. Runs simulation worlds generation
# 3. Aggregates worlds into projection quantiles
#
# Usage:
#   ./scripts/run_live_sim.sh [DATE]
#
# If DATE is not provided, uses today's date.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

# Default data root
DATA_ROOT="${PROJECTIONS_DATA_ROOT:-/home/daniel/projections-data}"

# Use provided date or today
if [[ -n "${1:-}" ]]; then
    SLATE_DATE="$1"
else
    SLATE_DATE="$(date +%Y-%m-%d)"
fi

# Configuration
PROFILE="${SIM_PROFILE:-sim_v3}"
NUM_WORLDS="${SIM_NUM_WORLDS:-2000}"

echo "[sim] Starting live simulation for ${SLATE_DATE}"
echo "[sim] Profile: ${PROFILE}, Worlds: ${NUM_WORLDS}"

cd "${PROJECT_ROOT}"

# Step 1: Ensure minutes predictions are available in expected location
# The simulator looks for: gold/projections_minutes_v1/game_date=YYYY-MM-DD/minutes.parquet
# But live scoring writes to: artifacts/minutes_v1/daily/YYYY-MM-DD/run=.../minutes.parquet
# We also check the non-partitioned path: gold/projections_minutes_v1/YYYY-MM-DD/minutes.parquet

MINUTES_DST_PARTITIONED="${DATA_ROOT}/gold/projections_minutes_v1/game_date=${SLATE_DATE}"
MINUTES_DST_PLAIN="${DATA_ROOT}/gold/projections_minutes_v1/${SLATE_DATE}"
MINUTES_SRC="${DATA_ROOT}/artifacts/minutes_v1/daily/${SLATE_DATE}"
MINUTES_LATEST_JSON="${MINUTES_SRC}/latest_run.json"
SIM_RUN_ID="${SIM_RUN_ID:-}"
SIM_RUN_AS_OF_TS=""

if [[ -z "${SIM_RUN_ID}" ]] && [[ -f "${MINUTES_LATEST_JSON}" ]]; then
    SIM_RUN_ID=$(jq -r '.run_id // empty' "${MINUTES_LATEST_JSON}" 2>/dev/null || true)
    SIM_RUN_AS_OF_TS=$(jq -r '.run_as_of_ts // empty' "${MINUTES_LATEST_JSON}" 2>/dev/null || true)
fi

if [[ -z "${SIM_RUN_ID}" ]] && [[ -d "${MINUTES_SRC}" ]]; then
    LATEST_RUN=$(ls -1t "${MINUTES_SRC}" 2>/dev/null | grep "^run=" | head -1 || true)
    if [[ -n "${LATEST_RUN}" ]]; then
        SIM_RUN_ID="${LATEST_RUN#run=}"
    fi
fi

if [[ -z "${SIM_RUN_ID}" ]]; then
    SIM_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
fi

# Check if minutes already exist in either expected location
if [[ -f "${MINUTES_DST_PARTITIONED}/minutes.parquet" ]]; then
    echo "[sim] Minutes already available at ${MINUTES_DST_PARTITIONED}"
elif [[ -f "${MINUTES_DST_PLAIN}/minutes.parquet" ]]; then
    echo "[sim] Minutes available at ${MINUTES_DST_PLAIN}, creating partition symlink..."
    mkdir -p "${MINUTES_DST_PARTITIONED}"
    cp "${MINUTES_DST_PLAIN}/minutes.parquet" "${MINUTES_DST_PARTITIONED}/"
elif [[ -d "${MINUTES_SRC}" ]]; then
    echo "[sim] Copying minutes from artifacts to gold..."
    # Find the latest run directory
    if [[ -f "${MINUTES_SRC}/run=${SIM_RUN_ID}/minutes.parquet" ]]; then
        mkdir -p "${MINUTES_DST_PARTITIONED}"
        cp "${MINUTES_SRC}/run=${SIM_RUN_ID}/minutes.parquet" "${MINUTES_DST_PARTITIONED}/"
        echo "[sim] Copied minutes from ${MINUTES_SRC}/run=${SIM_RUN_ID}"
    else
        echo "[sim] ERROR: No minutes.parquet found in ${MINUTES_SRC}/run=${SIM_RUN_ID}" >&2
        exit 1
    fi
else
    echo "[sim] ERROR: No minutes data found for ${SLATE_DATE}" >&2
    echo "[sim] Checked: ${MINUTES_DST_PARTITIONED}, ${MINUTES_DST_PLAIN}, ${MINUTES_SRC}" >&2
    exit 1
fi

# Step 2: Check that rates are available
RATES_PATH="${DATA_ROOT}/gold/rates_v1_live/${SLATE_DATE}"
if [[ ! -d "${RATES_PATH}" ]]; then
    echo "[sim] ERROR: No rates data found at ${RATES_PATH}" >&2
    echo "[sim] Run live-rates service first." >&2
    exit 1
fi
echo "[sim] Rates available at ${RATES_PATH}"

# Step 3: Run simulation
echo "[sim] Running simulation..."
/home/daniel/.local/bin/uv run python -m scripts.sim_v2.run_sim_live \
    --run-date "${SLATE_DATE}" \
    --profile "${PROFILE}" \
    --num-worlds "${NUM_WORLDS}" \
    --data-root "${DATA_ROOT}" \
    --run-id "${SIM_RUN_ID}" \
    ${SIM_RUN_AS_OF_TS:+--run-as-of-ts "${SIM_RUN_AS_OF_TS}"}

echo "[sim] Completed live simulation for ${SLATE_DATE}"
