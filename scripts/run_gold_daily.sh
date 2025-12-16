#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${PROJECTIONS_DATA_ROOT:-/home/daniel/projections-data}"
TARGET_DATE="${TARGET_DATE:-$(date -u -d 'yesterday' +%F)}"
SEASON="$(date -u -d "${TARGET_DATE}" +%Y)"
MONTH="$(date -u -d "${TARGET_DATE}" +%-m)"

cd /home/daniel/projects/projections-v2

/home/daniel/.local/bin/uv run python -m projections.pipelines.build_features_minutes_v1 \
  --start-date "${TARGET_DATE}" \
  --end-date "${TARGET_DATE}" \
  --season "${SEASON}" \
  --month "${MONTH}" \
  --data-root "${DATA_ROOT}"
