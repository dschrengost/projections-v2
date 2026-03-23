#!/usr/bin/env bash
set -euo pipefail

# Bring the control plane online safely:
# - Prefect server (localhost only) + worker (limit=1)
# - Minutes dashboard API/UI (localhost only)
# - Expose both via Tailscale Serve
# - Deploy Prefect deployments and pause all schedules by default

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

API_URL="${PREFECT_API_URL:-http://127.0.0.1:4200/api}"
DATA_ROOT="${PROJECTIONS_DATA_ROOT:-/home/daniel/projections-data}"

echo "[bring-online] enabling user linger for ${USER} (start user services on boot)"
loginctl enable-linger "${USER}" >/dev/null 2>&1 || true

echo "[bring-online] ensuring Prefect DB directory exists"
mkdir -p "${DATA_ROOT}/control_plane/prefect"

echo "[bring-online] installing/updating systemd user units"
"${ROOT_DIR}/scripts/deploy/install_prefect_server_unit.sh" >/dev/null
"${ROOT_DIR}/scripts/deploy/install_prefect_worker_unit.sh" >/dev/null
"${ROOT_DIR}/scripts/deploy/install_minutes_dashboard_unit.sh" >/dev/null

echo "[bring-online] configuring Tailscale Serve (https ports 4200 + 8501)"
if command -v tailscale >/dev/null 2>&1; then
  # These commands configure tailscaled; they persist across reboots.
  if ! tailscale serve --bg --yes --https 4200 http://127.0.0.1:4200 >/dev/null 2>&1; then
    echo "[bring-online] WARNING: tailscale serve denied. Run once:" >&2
    echo "  sudo tailscale set --operator=${USER}" >&2
  fi
  if ! tailscale serve --bg --yes --https 8501 http://127.0.0.1:8501 >/dev/null 2>&1; then
    echo "[bring-online] WARNING: tailscale serve denied. After setting operator, run:" >&2
    echo "  tailscale serve --bg --yes --https 4200 http://127.0.0.1:4200" >&2
    echo "  tailscale serve --bg --yes --https 8501 http://127.0.0.1:8501" >&2
  fi
else
  echo "[bring-online] WARNING: tailscale not found; skipping tailscale serve config" >&2
fi

echo "[bring-online] waiting for Prefect API health: ${API_URL}/health"
for _ in $(seq 1 60); do
  if curl -fsS "${API_URL}/health" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
if ! curl -fsS "${API_URL}/health" >/dev/null 2>&1; then
  echo "[bring-online] ERROR: Prefect API failed health check: ${API_URL}/health" >&2
  systemctl --user status prefect-server.service --no-pager -l | sed -n '1,60p' || true
  exit 1
fi

echo "[bring-online] hardening Prefect runtime (deploy flows + overrides + worker restart)"
"${ROOT_DIR}/scripts/deploy/harden_prefect_runtime.sh" >/dev/null

echo "[bring-online] pausing all Prefect deployment schedules (safety default)"
(cd "${ROOT_DIR}" && PREFECT_API_URL="${API_URL}" uv run prefect deployment schedule pause --all >/dev/null) || true

echo "[bring-online] dashboard health: http://127.0.0.1:8501/api/version"
curl -fsS "http://127.0.0.1:8501/api/version" >/dev/null && echo "[bring-online] dashboard ok"

echo "[bring-online] tailscale serve status:"
if command -v tailscale >/dev/null 2>&1; then
  tailscale serve status || true
fi

echo "[bring-online] done"
