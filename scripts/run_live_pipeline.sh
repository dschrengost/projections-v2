#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV_BIN="${UV_BIN:-/home/daniel/.local/bin/uv}"

DATA_ROOT="${PROJECTIONS_DATA_ROOT:-/home/daniel/projections-data}"
START_DATE="${LIVE_START_DATE:-$(date -I)}"
END_DATE="${LIVE_END_DATE:-$START_DATE}"
SEASON_VALUE="${LIVE_SEASON:-$(date +%Y)}"
MONTH_VALUE="${LIVE_MONTH:-$(date +%-m)}"
EXTRA_FLAGS="${LIVE_EXTRA_FLAGS:-}"

cd "$PROJECT_ROOT"

exec "$UV_BIN" run python -m projections.cli.live_pipeline \
  --start "$START_DATE" \
  --end "$END_DATE" \
  --season "$SEASON_VALUE" \
  --month "$MONTH_VALUE" \
  --data-root "$DATA_ROOT" \
  $EXTRA_FLAGS
