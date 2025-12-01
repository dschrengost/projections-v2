#!/usr/bin/env bash
set -euo pipefail

# Unified launcher for the minutes API + dashboard.
# Usage: ./run_stack.sh [API_PORT] [UI_PORT]

API_PORT=${1:-8501}
UI_PORT=${2:-5173}
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
DATA_ROOT="${PROJECTIONS_DATA_ROOT:-/home/daniel/projections-data}"

cd "${REPO_ROOT}"

if lsof -i :"${API_PORT}" >/dev/null 2>&1; then
  echo "[stack] Port ${API_PORT} already in use. Kill the existing server first." >&2
  exit 1
fi

export VITE_MINUTES_API="http://localhost:${API_PORT}"

echo "[stack] starting API on ${API_PORT} (DATA_ROOT=${DATA_ROOT})"
(
  PROJECTIONS_DATA_ROOT="${DATA_ROOT}" \
  MINUTES_DAILY_ROOT="${REPO_ROOT}/artifacts/minutes_v1/daily" \
  MINUTES_FPTS_ROOT="${DATA_ROOT}/gold/projections_fpts_v1" \
  uv run uvicorn projections.api.minutes_api:create_app --host 0.0.0.0 --port "${API_PORT}"
) &
API_PID=$!

sleep 2
echo "[stack] starting dashboard on ${UI_PORT} (VITE_MINUTES_API=${VITE_MINUTES_API})"
(
  cd "${REPO_ROOT}/web/minutes-dashboard"
  npx vite --port "${UI_PORT}"
) &
UI_PID=$!

trap 'echo "[stack] Shutting down..."; kill ${API_PID} ${UI_PID}' INT TERM
wait
