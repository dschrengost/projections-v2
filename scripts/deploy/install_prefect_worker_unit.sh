#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UNIT_SRC="${ROOT_DIR}/infra/systemd/prefect-worker.service"
UNIT_DIR="${HOME}/.config/systemd/user"
UNIT_DST="${UNIT_DIR}/prefect-worker.service"

if [[ ! -f "${UNIT_SRC}" ]]; then
  echo "[prefect-worker-unit] missing unit template: ${UNIT_SRC}" >&2
  exit 1
fi

mkdir -p "${UNIT_DIR}"
install -m 0644 "${UNIT_SRC}" "${UNIT_DST}"

echo "[prefect-worker-unit] reloading user systemd and restarting prefect-worker.service"
systemctl --user daemon-reload
systemctl --user enable prefect-worker.service >/dev/null
systemctl --user restart prefect-worker.service

echo "[prefect-worker-unit] prefect-worker.service status:"
systemctl --user status prefect-worker.service --no-pager -n 20
