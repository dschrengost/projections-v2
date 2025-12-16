#!/usr/bin/env bash
# Run daily tracking data scrape and build roles
# Called by systemd/tracking-daily.timer

set -euo pipefail

cd /home/daniel/projects/projections-v2
source .venv/bin/activate

# Determine NBA season based on current month
# NBA season runs Oct-June: Oct 2025 is season "2025-26"
MONTH=$(date +%m)
YEAR=$(date +%Y)
if [[ "$MONTH" -ge 10 ]]; then
    # Oct-Dec: season starts this year (e.g., Oct 2025 = 2025-26)
    SEASON="${YEAR}-$(printf '%02d' $(((YEAR % 100) + 1)))"
    SEASON_START="${YEAR}-10-01"
else
    # Jan-Sep: season started last year (e.g., Jan 2026 = 2025-26)
    PREV_YEAR=$((YEAR - 1))
    SEASON="${PREV_YEAR}-$(printf '%02d' $((YEAR % 100)))"
    SEASON_START="${PREV_YEAR}-10-01"
fi

# Yesterday's date (tracking data is available day-after)
YESTERDAY=$(date -d "yesterday" +%Y-%m-%d)
TODAY=$(date +%Y-%m-%d)

echo "[tracking] Starting daily scrape for $YESTERDAY (season=$SEASON)"

# Step 1: Scrape bronze tracking data for yesterday
python -m scripts.tracking.scrape_tracking_raw run-day \
    --season "$SEASON" \
    --date "$YESTERDAY"

# Step 2: Build tracking roles (incremental update)
# Use full season range to ensure cumulative stats are correct
python -m scripts.tracking.build_tracking_roles \
    --start-date "$SEASON_START" \
    --end-date "$TODAY"

echo "[tracking] Completed daily tracking pipeline"
