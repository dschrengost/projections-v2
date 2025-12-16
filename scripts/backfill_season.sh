#!/bin/bash
# Backfill projections for a date range
# Usage: ./scripts/backfill_season.sh [START_DATE] [END_DATE]
#
# Example: ./scripts/backfill_season.sh 2025-10-21 2025-12-05

set -e

START="${1:-2025-10-21}"
END="${2:-2025-12-05}"

SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")

# Convert dates to seconds for iteration
start_sec=$(date -d "$START" +%s)
end_sec=$(date -d "$END" +%s)

# Count total days
total_days=$(( (end_sec - start_sec) / 86400 + 1 ))
current_day=0

echo "[backfill] Running projections for $START to $END ($total_days days)"
echo "[backfill] Using 10,000 worlds per day"
echo ""

# Iterate through each date
current_sec=$start_sec
while [ $current_sec -le $end_sec ]; do
    current_date=$(date -d "@$current_sec" +%Y-%m-%d)
    current_day=$((current_day + 1))
    
    echo "==========================================="
    echo "[backfill] [$current_day/$total_days] Processing $current_date"
    echo "==========================================="
    
    # Run the live score pipeline with backfill flags
    LIVE_START_DATE="$current_date" \
    LIVE_DISABLE_TIP_WINDOW=1 \
    LIVE_BACKFILL_MODE=1 \
    LIVE_LOCK_BUFFER_MINUTES=0 \
    LIVE_SKIP_SCRAPE="${LIVE_SKIP_SCRAPE:-0}" \
    "$SCRIPT_DIR/run_live_score.sh" 2>&1 || {
        echo "[backfill] WARNING: Failed for $current_date, continuing..."
    }
    
    echo ""
    
    # Move to next day
    current_sec=$((current_sec + 86400))
done

echo "==========================================="
echo "[backfill] Complete! Processed $total_days days"
echo "==========================================="
echo ""
echo "Run accuracy analysis:"
echo "  python -m scripts.analyze_accuracy --start $START --end $END"
