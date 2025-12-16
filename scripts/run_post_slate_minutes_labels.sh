#!/usr/bin/env bash
set -euo pipefail

# Determine the canonical data root (defaults to ./data when the env var is unset).
DATA_ROOT=${PROJECTIONS_DATA_ROOT:-"$PWD/data"}

# Compute "yesterday" relative to the host; Linux syntax shown here.
# On macOS use: date -v-1d +%F
YESTERDAY=$(date -d "yesterday" +%F)

echo "[post-slate-labels] Backfilling labels for ${YESTERDAY}"

UV_CACHE_DIR=${UV_CACHE_DIR:-/tmp/uv-cache} \
PROJECTIONS_DATA_ROOT="${DATA_ROOT}" \
uv run python -m projections.cli.backfill_minutes_labels \
  --start-date "${YESTERDAY}" \
  --end-date "${YESTERDAY}"
