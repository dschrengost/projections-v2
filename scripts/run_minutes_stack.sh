#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/run_minutes_stack.sh [PORT]
API_PORT=${1:-8501}
UI_PORT=${2:-5173}

cd "$(dirname "$0")/.."

# Ensure uvicorn isn’t already running on that port
if lsof -i :"${API_PORT}" >/dev/null 2>&1; then
  echo "[stack] Port ${API_PORT} already in use. Kill the existing server first." >&2
  exit 1
fi

# Export env for the React dev server so fetch() hits the right API base.
export VITE_MINUTES_API="http://localhost:${API_PORT}"

# Start API (foreground) & dev server (background)
(
  PROJECTIONS_DATA_ROOT=/home/daniel/projections-data \
  MINUTES_DAILY_ROOT=/home/daniel/projects/projections-v2/artifacts/minutes_v1/daily \
  MINUTES_FPTS_ROOT=/home/daniel/projections-data/gold/projections_fpts_v1 \
  uv run uvicorn projections.api.minutes_api:create_app --host 0.0.0.0 --port "${API_PORT}"
) &
API_PID=$!

sleep 2
(
  cd web/minutes-dashboard
  npx vite --port "${UI_PORT}"
) &
UI_PID=$!

trap 'echo "[stack] Shutting down..."; kill ${API_PID} ${UI_PID}' INT TERM
wait
