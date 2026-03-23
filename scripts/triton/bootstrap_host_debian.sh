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
need_cmd gpg

echo "[bootstrap] host=$(hostname) distro=$( . /etc/os-release; echo ${PRETTY_NAME} )"
echo "[bootstrap] target_user=${TARGET_USER}"

apt_update() {
  # Debian 13 (apt 3.x + sqv) rejects some third-party repos whose signing keys
  # rely on SHA1 certifications. NVIDIA's debian12 CUDA repo currently hits this.
  #
  # We keep a best-effort strict update first, then retry allowing weak repos only
  # when necessary. This preserves normal security posture for Debian repos while
  # still enabling one-time installation of nvidia-container-toolkit.
  if apt-get update -y; then
    return 0
  fi
  echo "[bootstrap] WARNING: apt update failed; retrying with AllowWeakRepositories=1"
  if apt-get -o Acquire::AllowWeakRepositories=true update -y; then
    return 0
  fi
  return 1
}

apt_install() {
  # Use the same weak-repo retry strategy for installs.
  if apt-get install -y --no-install-recommends "$@"; then
    return 0
  fi
  echo "[bootstrap] WARNING: apt install failed; retrying with AllowWeakRepositories=1"
  if apt-get -o Acquire::AllowWeakRepositories=true install -y --no-install-recommends "$@"; then
    return 0
  fi
  return 1
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
# GitHub pages repo endpoints are unreliable in some environments; use the
# CUDA repo. Debian 13 repo may not always carry container toolkit packages,
# so we default to the Debian 12 CUDA repo which is known to include them.
CUDA_REPO_SUITE="${CUDA_REPO_SUITE:-debian12}"
CUDA_REPO_ARCH="${CUDA_REPO_ARCH:-x86_64}"
CUDA_REPO_BASE="https://developer.download.nvidia.com/compute/cuda/repos/${CUDA_REPO_SUITE}/${CUDA_REPO_ARCH}"
CUDA_LIST_PATH="/etc/apt/sources.list.d/nvidia-cuda.list"
CUDA_KEYRING_PATH="/usr/share/keyrings/nvidia-cuda-archive-keyring.gpg"

echo "[bootstrap] configuring nvidia container toolkit apt repo (suite=${CUDA_REPO_SUITE})"
# Key filename differs across suites; try known candidates.
CUDA_KEY_URL=""
for candidate in 3bf863cc.pub 8793F200.pub; do
  if curl -fsSI "${CUDA_REPO_BASE}/${candidate}" >/dev/null 2>&1; then
    CUDA_KEY_URL="${CUDA_REPO_BASE}/${candidate}"
    break
  fi
done
if [[ -z "${CUDA_KEY_URL}" ]]; then
  echo "[bootstrap] error: could not locate CUDA repo key under ${CUDA_REPO_BASE}" >&2
  echo "[bootstrap] try setting CUDA_REPO_SUITE=debian12 explicitly." >&2
  exit 1
fi
echo "[bootstrap] using CUDA repo key: ${CUDA_KEY_URL}"
curl -fsSL "${CUDA_KEY_URL}" | gpg --dearmor -o "${CUDA_KEYRING_PATH}"
chmod 0644 "${CUDA_KEYRING_PATH}"
cat >"${CUDA_LIST_PATH}" <<EOF
deb [signed-by=${CUDA_KEYRING_PATH}] ${CUDA_REPO_BASE}/ /
EOF

if ! apt_update; then
  echo "[bootstrap] WARNING: NVIDIA repo signature rejected by apt policy."
  echo "[bootstrap] WARNING: falling back to trusted=yes for one-time toolkit install."
  cat >"${CUDA_LIST_PATH}" <<EOF
deb [trusted=yes] ${CUDA_REPO_BASE}/ /
EOF
  apt-get update -y
fi

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
