#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${PROJECTIONS_DATA_ROOT:-/home/daniel/projections-data}"
START_DATE="${BOX_START_DATE:-$(date -I -d 'yesterday')}"
END_DATE="${BOX_END_DATE:-$START_DATE}"
SEASON="${BOX_SEASON:-$(date +%Y)}"
TIMEOUT="${BOX_TIMEOUT:-12}"

cd /home/daniel/projects/projections-v2

exec /home/daniel/.local/bin/uv run python -m projections.etl.boxscores \
  --start "${START_DATE}" \
  --end "${END_DATE}" \
  --season "${SEASON}" \
  --data-root "${DATA_ROOT}" \
  --timeout "${TIMEOUT}"
