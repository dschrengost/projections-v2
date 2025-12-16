#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${PROJECTIONS_DATA_ROOT:-/home/daniel/projections-data}"
START_DATE="${LIVE_START_DATE:-$(date -I)}"
END_DATE="${LIVE_END_DATE:-$START_DATE}"
SEASON="${LIVE_SEASON:-$(date +%Y)}"
MONTH="${LIVE_MONTH:-$(date +%-m)}"
FLAGS="${LIVE_FLAGS:-}"  # additional CLI flags

cd /home/daniel/projects/projections-v2

# Run the scraping pipeline
/home/daniel/.local/bin/uv run python -m projections.cli.live_pipeline \
  --start "${START_DATE}" \
  --end "${END_DATE}" \
  --season "${SEASON}" \
  --month "${MONTH}" \
  --data-root "${DATA_ROOT}" \
  --schedule "${DATA_ROOT}/silver/schedule/season=${SEASON}/month=$(printf "%02d" "${MONTH}")/schedule.parquet" \
  ${FLAGS}
