#!/usr/bin/env bash
# Run live rates scoring for a slate date.
#
# This script:
# 1. Builds rates features from minutes features
# 2. Scores rates predictions
#
# Usage:
#   ./scripts/run_live_rates.sh [DATE]
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

echo "[rates] Starting live rates for ${SLATE_DATE}"

# Generate a run ID based on current time
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
echo "[rates] Run ID: ${RUN_ID}"

cd "${PROJECT_ROOT}"

# Compute NBA season token (e.g., 2024-25) based on slate date
# NBA season starts in October, so Oct-Dec = current year, Jan-Sep = previous year
MONTH=$(date -d "${SLATE_DATE}" +%-m)
YEAR=$(date -d "${SLATE_DATE}" +%Y)
if [[ ${MONTH} -ge 10 ]]; then
    SEASON_START="${YEAR}"
    SEASON_END="$((YEAR + 1))"
else
    SEASON_START="$((YEAR - 1))"
    SEASON_END="${YEAR}"
fi
NBA_SEASON="${SEASON_START}-${SEASON_END:2:2}"

echo "[rates] NBA Season: ${NBA_SEASON}"

# Step 0: Scrape tracking data (previous day since tracking has delay)
# Only run if we're not in a dry-run or testing mode
if [[ "${SKIP_TRACKING:-}" != "1" ]]; then
    TRACKING_DATE=$(date -d "${SLATE_DATE} - 1 day" +%Y-%m-%d)
    echo "[rates] Scraping tracking data for ${TRACKING_DATE}..."
    /home/daniel/.local/bin/uv run python scripts/tracking/scrape_tracking_raw.py run-day \
        --season "${NBA_SEASON}" \
        --date "${TRACKING_DATE}" || {
        echo "[rates] Warning: Tracking scrape failed for ${TRACKING_DATE}, continuing..."
    }
fi

# Step 1: Build rates features from minutes features
echo "[rates] Building rates features..."
/home/daniel/.local/bin/uv run python -m projections.cli.build_rates_features_live \
    --date "${SLATE_DATE}" \
    --run-id "${RUN_ID}" \
    --data-root "${DATA_ROOT}" \
    --strict

# Step 2: Score rates predictions
echo "[rates] Scoring rates predictions..."
/home/daniel/.local/bin/uv run python -m projections.cli.score_rates_live \
    --date "${SLATE_DATE}" \
    --run-id "${RUN_ID}" \
    --features-root "${DATA_ROOT}/live/features_rates_v1" \
    --out-root "${DATA_ROOT}/gold/rates_v1_live" \
    --strict

echo "[rates] Completed live rates for ${SLATE_DATE}"
