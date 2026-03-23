#!/usr/bin/env bash
set -euo pipefail

# Bootstrap a Debian host for running the GTV2 Triton container:
# - Docker (daemon + cli)
# - NVIDIA driver + nvidia-smi (host kernel module)
# - NVIDIA container toolkit (enables `docker run --gpus all`)
#
# Notes:
# - This script must be run as root (use sudo).
# - A reboot is typically required after installing the NVIDIA driver and/or
#   after adding your user to the `docker` group.

if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
  echo "[bootstrap] error: must run as root."
  echo "[bootstrap] run: sudo $0"
  exit 2
fi

TARGET_USER="${SUDO_USER:-}"
if [[ -z "${TARGET_USER}" ]]; then
  # If invoked as root directly, fall back to the default operator user.
  TARGET_USER="daniel"
fi

export DEBIAN_FRONTEND=noninteractive

need_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "[bootstrap] error: missing required command: $cmd" >&2
    exit 1
  fi
}

need_cmd apt-get
need_cmd curl

echo "[bootstrap] host=$(hostname) distro=$( . /etc/os-release; echo ${PRETTY_NAME} )"
echo "[bootstrap] target_user=${TARGET_USER}"

cleanup_nvidia_apt_sources() {
  # If a previous bootstrap attempt left NVIDIA/CUDA repo stubs behind, apt (3.x +
  # sqv) can fail hard during `apt-get update` before we even get to the install
  # steps. Remove them by default so the script can be rerun safely.
  #
  # Set KEEP_NVIDIA_APT_REPO=1 to keep any existing NVIDIA-related repo entries.
  if [[ "${KEEP_NVIDIA_APT_REPO:-0}" == "1" ]]; then
    return 0
  fi

  local f base
  shopt -s nullglob
  for f in /etc/apt/sources.list.d/*.list /etc/apt/sources.list.d/*.sources; do
    base="$(basename "$f")"
    if [[ "$base" =~ (nvidia|cuda|libnvidia|triton) ]]; then
      rm -f "$f" || true
      continue
    fi
    if grep -Eq 'developer\.download\.nvidia\.com|nvidia\.github\.io|libnvidia-container|compute/cuda/repos' "$f" 2>/dev/null; then
      rm -f "$f" || true
      continue
    fi
  done
  for f in /etc/apt/preferences.d/*nvidia* /etc/apt/preferences.d/*cuda*; do
    rm -f "$f" || true
  done
  shopt -u nullglob
}

cleanup_nvidia_apt_sources

apt_update() {
  # Keep the normal security posture for Debian repos. If this fails, it's
  # typically because there is still a broken third-party repo configured.
  if apt-get update -y; then
    return 0
  fi
  return 1
}

apt_install() {
  apt-get install -y --no-install-recommends "$@"
}

echo "[bootstrap] apt update"
apt_update

echo "[bootstrap] installing base deps"
apt_install \
  ca-certificates \
  curl \
  gpg \
  gnupg \
  dkms \
  linux-headers-$(uname -r)

echo "[bootstrap] installing docker (debian docker.io)"
apt_install docker.io docker-buildx
systemctl enable --now docker

echo "[bootstrap] adding ${TARGET_USER} to docker group (if user exists)"
if id "${TARGET_USER}" >/dev/null 2>&1; then
  usermod -aG docker "${TARGET_USER}"
else
  echo "[bootstrap] warning: user not found, skipping docker group add: ${TARGET_USER}" >&2
fi

echo "[bootstrap] installing nvidia driver + nvidia-smi (debian non-free)"
apt_install nvidia-driver nvidia-smi

# NVIDIA Container Toolkit:
# Debian (including trixie) does not ship `nvidia-container-toolkit` in the main
# archive, so we install it from NVIDIA's CUDA repo.
#
# Debian 13 (apt 3.x + sqv) rejects the CUDA repo signing key chain due to SHA1
# certifications (policy cutoff 2026-02-01). To keep the host's default security
# posture, we:
# 1) Run a strict `apt-get update` against Debian repos (above).
# 2) Temporarily add the NVIDIA repo as `trusted=yes` for a one-time install.
# 3) Remove the repo list entry after install (unless KEEP_NVIDIA_APT_REPO=1).
CUDA_REPO_SUITE="${CUDA_REPO_SUITE:-debian12}"
CUDA_REPO_ARCH="${CUDA_REPO_ARCH:-x86_64}"
CUDA_REPO_BASE="https://developer.download.nvidia.com/compute/cuda/repos/${CUDA_REPO_SUITE}/${CUDA_REPO_ARCH}"
CUDA_LIST_PATH="/etc/apt/sources.list.d/nvidia-cuda.list"

echo "[bootstrap] configuring nvidia container toolkit apt repo (suite=${CUDA_REPO_SUITE})"
cat >"${CUDA_LIST_PATH}" <<EOF
deb [trusted=yes] ${CUDA_REPO_BASE}/ /
EOF

echo "[bootstrap] apt update (allowing insecure repo for NVIDIA toolkit install only)"
apt-get \
  -o Acquire::AllowInsecureRepositories=true \
  -o Acquire::AllowDowngradeToInsecureRepositories=true \
  update -y

if ! apt_install nvidia-container-toolkit; then
  echo "[bootstrap] error: failed to install nvidia-container-toolkit" >&2
  exit 1
fi

if command -v nvidia-ctk >/dev/null 2>&1; then
  echo "[bootstrap] configuring docker runtime for nvidia-container-toolkit"
  nvidia-ctk runtime configure --runtime=docker
  systemctl restart docker
else
  echo "[bootstrap] warning: nvidia-ctk not found after install; toolkit install may have failed" >&2
fi

echo ""
echo "[bootstrap] ============================================================"
echo "[bootstrap] DONE (reboot recommended)"
echo "[bootstrap] ============================================================"
echo "[bootstrap] Next checks (after reboot):"
echo "  nvidia-smi -L"
echo "  docker info"
echo "  docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi"
echo "  /home/daniel/prod/projections-v2/scripts/triton/check_prereqs.sh"
echo ""
echo "[bootstrap] If you added ${TARGET_USER} to the docker group, you must re-login or reboot."

# Avoid breaking future `apt-get update` runs if NVIDIA's repo remains "weak".
# Keep the keyring (harmless), but remove the repo list unless explicitly requested.
if [[ "${KEEP_NVIDIA_APT_REPO:-0}" != "1" ]]; then
  rm -f "${CUDA_LIST_PATH}" || true
fi
