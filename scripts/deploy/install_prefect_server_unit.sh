#!/usr/bin/env bash
set -euo pipefail

# NOTE: This installs Prefect Server as a user service (no sudo) and binds to localhost.
# External access is provided via `tailscale serve`.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UNIT_SRC="${ROOT_DIR}/infra/systemd/prefect-server.service"
UNIT_DIR="${HOME}/.config/systemd/user"
UNIT_DST="${UNIT_DIR}/prefect-server.service"

ENV_DIR="${HOME}/.config/projections"
ENV_FILE="${ENV_DIR}/prefect-server.env"

if [[ ! -f "${UNIT_SRC}" ]]; then
  echo "[prefect-server-unit] missing unit template: ${UNIT_SRC}" >&2
  exit 1
fi

mkdir -p "${UNIT_DIR}"
install -m 0644 "${UNIT_SRC}" "${UNIT_DST}"

# Seed env file if missing. Defaults match the unit template (SQLite).
if [[ ! -f "${ENV_FILE}" ]]; then
  echo "[prefect-server-unit] writing default ${ENV_FILE}"
  install -d -m 0755 "${ENV_DIR}"
  cat >"${ENV_FILE}" <<'EOF'
# Override Prefect server DB settings here if you move to Postgres.
# Default (safe recovery baseline) is SQLite on the data volume:
#   sqlite+aiosqlite:////home/daniel/projections-data/control_plane/prefect/prefect.db
PREFECT_API_DATABASE_CONNECTION_URL=sqlite+aiosqlite:////home/daniel/projections-data/control_plane/prefect/prefect.db
PREFECT_API_DATABASE_TIMEOUT=30.0
PREFECT_API_DATABASE_CONNECTION_TIMEOUT=30.0
EOF
  chmod 0600 "${ENV_FILE}"
fi

echo "[prefect-server-unit] reloading user systemd and enabling prefect-server.service"
systemctl --user daemon-reload
systemctl --user enable prefect-server.service >/dev/null
systemctl --user restart prefect-server.service

echo "[prefect-server-unit] prefect-server.service status:"
systemctl --user status prefect-server.service --no-pager -n 20

