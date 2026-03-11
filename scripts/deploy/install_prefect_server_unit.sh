#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UNIT_SRC="${ROOT_DIR}/infra/systemd/prefect-server.service"
UNIT_DST="/etc/systemd/system/prefect-server.service"
ENV_DIR="/home/daniel/.config/projections"
ENV_FILE="${ENV_DIR}/prefect-server.env"

if [[ ! -f "${UNIT_SRC}" ]]; then
  echo "[prefect-server-unit] missing unit template: ${UNIT_SRC}" >&2
  exit 1
fi

if [[ "${EUID}" -ne 0 ]]; then
  echo "[prefect-server-unit] run with sudo: sudo ${ROOT_DIR}/scripts/deploy/install_prefect_server_unit.sh" >&2
  exit 2
fi

echo "[prefect-server-unit] installing ${UNIT_SRC} -> ${UNIT_DST}"
install -m 0644 "${UNIT_SRC}" "${UNIT_DST}"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "[prefect-server-unit] writing default ${ENV_FILE}"
  install -d -m 0755 -o daniel -g daniel "${ENV_DIR}"
  cat >"${ENV_FILE}" <<'EOF'
PREFECT_API_DATABASE_CONNECTION_URL=postgresql+asyncpg://prefect:prefect@127.0.0.1:55432/prefect
PREFECT_API_DATABASE_TIMEOUT=30.0
PREFECT_API_DATABASE_CONNECTION_TIMEOUT=30.0
EOF
  chown daniel:daniel "${ENV_FILE}"
  chmod 0600 "${ENV_FILE}"
fi

echo "[prefect-server-unit] reloading systemd and restarting prefect-server.service"
systemctl daemon-reload
systemctl enable prefect-server.service >/dev/null
systemctl restart prefect-server.service

echo "[prefect-server-unit] prefect-server.service status:"
systemctl status prefect-server.service --no-pager -n 20
