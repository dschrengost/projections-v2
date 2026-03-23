#!/usr/bin/env bash
set -euo pipefail

# Host bootstrap for Triton on Debian 13 (trixie).
# Installs:
# - NVIDIA driver (Debian-packaged)
# - Docker
# - NVIDIA Container Toolkit (from NVIDIA libnvidia-container repo)
#
# This script is intentionally explicit and should be reviewed before running.
#
# Usage:
#   sudo ./scripts/deploy/bootstrap_triton_host_debian13.sh
#
# After it completes:
# - reboot (recommended after driver install)
# - verify: nvidia-smi
# - verify: docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi

if [[ "${EUID}" -ne 0 ]]; then
  echo "[bootstrap-triton] run as root (sudo)" >&2
  exit 2
fi

if [[ ! -f /etc/os-release ]]; then
  echo "[bootstrap-triton] missing /etc/os-release" >&2
  exit 1
fi

. /etc/os-release
if [[ "${ID:-}" != "debian" || "${VERSION_ID:-}" != "13" ]]; then
  echo "[bootstrap-triton] expected Debian 13; got ID=${ID:-?} VERSION_ID=${VERSION_ID:-?}" >&2
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive

echo "[bootstrap-triton] apt update"
apt-get update -y

echo "[bootstrap-triton] base deps"
apt-get install -y --no-install-recommends \
  ca-certificates curl gnupg \
  linux-headers-amd64 dkms build-essential

echo "[bootstrap-triton] installing NVIDIA driver (Debian package)"
apt-get install -y nvidia-driver

echo "[bootstrap-triton] installing Docker"
apt-get install -y docker.io
systemctl enable --now docker

echo "[bootstrap-triton] adding user 'daniel' to docker group (requires re-login)"
if getent group docker >/dev/null 2>&1; then
  usermod -aG docker daniel || true
fi

echo "[bootstrap-triton] installing NVIDIA Container Toolkit repo + package"
distribution=$(. /etc/os-release; echo ${ID}${VERSION_ID})
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/${distribution}/libnvidia-container.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  > /etc/apt/sources.list.d/nvidia-container-toolkit.list

apt-get update -y
apt-get install -y nvidia-container-toolkit

echo "[bootstrap-triton] configuring Docker runtime"
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

echo "[bootstrap-triton] done. Recommended next steps:"
echo "  1) reboot"
echo "  2) nvidia-smi"
echo "  3) docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi"

