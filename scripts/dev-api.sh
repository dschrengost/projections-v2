#!/usr/bin/env bash
# Development API server - uses port 8502 to avoid conflict with production (8501)
#
# Usage: ./scripts/dev-api.sh [--reload]
#
# IMPORTANT: Never run uvicorn directly on port 8501 - that's reserved for the
# systemd-managed production service. Running on 8501 will block the systemd
# service and cause "address already in use" errors.

set -euo pipefail

PORT=8502
RELOAD_FLAG=""

if [[ "${1:-}" == "--reload" ]]; then
    RELOAD_FLAG="--reload"
fi

# Check if something is already on this port
if ss -tlnp 2>/dev/null | grep -q ":${PORT} "; then
    echo "WARNING: Port ${PORT} already in use:"
    ss -tlnp | grep ":${PORT} "
    echo ""
    read -p "Kill existing process and continue? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        fuser -k ${PORT}/tcp 2>/dev/null || true
        sleep 1
    else
        exit 1
    fi
fi

echo "Starting dev API on port ${PORT} (production is 8501)"
echo "Access at: http://localhost:${PORT}"
echo ""

cd "$(dirname "$0")/.."
exec uv run uvicorn projections.api.minutes_api:create_app \
    --host 0.0.0.0 \
    --port ${PORT} \
    ${RELOAD_FLAG}
