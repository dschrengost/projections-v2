#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UNIT_SRC="${ROOT_DIR}/infra/systemd/triton-gtv2.service"
UNIT_DIR="${HOME}/.config/systemd/user"
UNIT_DST="${UNIT_DIR}/triton-gtv2.service"

if [[ ! -f "${UNIT_SRC}" ]]; then
  echo "[triton-unit] missing unit template: ${UNIT_SRC}" >&2
  exit 1
fi

mkdir -p "${UNIT_DIR}"
install -m 0644 "${UNIT_SRC}" "${UNIT_DST}"

echo "[triton-unit] reloading user systemd and enabling triton-gtv2.service"
systemctl --user daemon-reload
systemctl --user enable triton-gtv2.service >/dev/null

echo "[triton-unit] NOTE: this unit requires Docker + NVIDIA runtime. Start with:"
echo "  systemctl --user start triton-gtv2.service"

