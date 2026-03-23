#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UNIT_SRC="${ROOT_DIR}/infra/systemd/minutes-dashboard.service"
UNIT_DIR="${HOME}/.config/systemd/user"
UNIT_DST="${UNIT_DIR}/minutes-dashboard.service"

if [[ ! -f "${UNIT_SRC}" ]]; then
  echo "[minutes-dashboard-unit] missing unit template: ${UNIT_SRC}" >&2
  exit 1
fi

mkdir -p "${UNIT_DIR}"
install -m 0644 "${UNIT_SRC}" "${UNIT_DST}"

echo "[minutes-dashboard-unit] reloading user systemd and enabling minutes-dashboard.service"
systemctl --user daemon-reload
systemctl --user enable minutes-dashboard.service >/dev/null
systemctl --user restart minutes-dashboard.service

echo "[minutes-dashboard-unit] minutes-dashboard.service status:"
systemctl --user status minutes-dashboard.service --no-pager -n 20

