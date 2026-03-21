#!/usr/bin/env bash
set -euo pipefail

# Recover temporary two-SSD runtime wiring after boot, without enabling boot-time dependency.
# - Mount Samsung rescue NTFS partition at /mnt/relief (manual, non-fstab dependency)
# - Repoint /home/daniel/{projections-data,prod} to rescue tree symlinks
# - Run storage guard check

RELIEF_UUID="${RELIEF_UUID:-52F24914F248FE2B}"
RELIEF_MOUNT="${RELIEF_MOUNT:-/mnt/relief}"
RESCUE_ROOT="${RESCUE_ROOT:-/mnt/relief/linux-rescue}"
USER_HOME="${USER_HOME:-/home/daniel}"

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || { echo "missing command: $1" >&2; exit 1; }
}

require_cmd mount
require_cmd findmnt
require_cmd sudo

echo "[recover] ensuring mountpoint: ${RELIEF_MOUNT}"
sudo mkdir -p "${RELIEF_MOUNT}"

if findmnt -rno SOURCE,TARGET "${RELIEF_MOUNT}" >/dev/null 2>&1; then
  echo "[recover] mount already present at ${RELIEF_MOUNT}"
else
  echo "[recover] mounting UUID=${RELIEF_UUID} at ${RELIEF_MOUNT}"
  if ! sudo mount -t ntfs3 -o uid=1000,gid=1000,umask=022 "UUID=${RELIEF_UUID}" "${RELIEF_MOUNT}"; then
    echo "[recover] normal mount failed; retrying with ntfs3 force flag"
    sudo mount -t ntfs3 -o force,uid=1000,gid=1000,umask=022 "UUID=${RELIEF_UUID}" "${RELIEF_MOUNT}"
  fi
fi

if [[ ! -d "${RESCUE_ROOT}/projections-data" ]]; then
  echo "[recover] missing rescue path: ${RESCUE_ROOT}/projections-data" >&2
  exit 1
fi
if [[ ! -d "${RESCUE_ROOT}/prod" ]]; then
  echo "[recover] missing rescue path: ${RESCUE_ROOT}/prod" >&2
  exit 1
fi

for name in projections-data prod; do
  target="${USER_HOME}/${name}"
  link_to="${RESCUE_ROOT}/${name}"

  if [[ -L "${target}" ]]; then
    current="$(readlink -f "${target}" || true)"
    if [[ "${current}" == "${link_to}" ]]; then
      echo "[recover] symlink OK: ${target} -> ${link_to}"
      continue
    fi
    echo "[recover] updating symlink: ${target} -> ${link_to}"
    rm -f "${target}"
  elif [[ -e "${target}" ]]; then
    backup="${target}.bak.$(date -u +%Y%m%dT%H%M%SZ)"
    echo "[recover] backing up existing path: ${target} -> ${backup}"
    sudo mv "${target}" "${backup}"
  fi

  ln -s "${link_to}" "${target}"
  echo "[recover] linked: ${target} -> ${link_to}"
done

echo "[recover] storage guard check"
cd /home/daniel/projects/projections-v2
uv run python -m projections.cli.storage_guard --hot-root /home/daniel/projections-data || {
  echo "[recover] storage guard reported hard-stop (expected if mount/symlink unresolved)" >&2
  exit 2
}

echo "[recover] complete"
echo "[recover] restart services:" \
  "systemctl --user restart minutes-dashboard.service prefect-worker.service"
