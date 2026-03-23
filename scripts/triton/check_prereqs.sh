#!/usr/bin/env bash
set -euo pipefail

echo "[triton-prereqs] host=$(hostname) user=$(whoami)"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[triton-prereqs] nvidia-smi: OK"
  nvidia-smi -L || true
else
  echo "[triton-prereqs] nvidia-smi: MISSING"
fi

if command -v nvidia-ctk >/dev/null 2>&1; then
  echo "[triton-prereqs] nvidia-ctk: OK ($(nvidia-ctk --version 2>/dev/null || true))"
else
  echo "[triton-prereqs] nvidia-ctk: MISSING (install nvidia-container-toolkit)"
fi

if id -nG "$(whoami)" 2>/dev/null | tr ' ' '\n' | grep -qx docker; then
  echo "[triton-prereqs] docker group: OK"
else
  echo "[triton-prereqs] docker group: MISSING (if docker is installed, run: sudo usermod -aG docker $(whoami))"
fi

if command -v docker >/dev/null 2>&1; then
  echo "[triton-prereqs] docker: OK ($(docker --version))"
  set +e
  docker_info_out="$(docker info 2>&1)"
  docker_info_rc="$?"
  set -e
  if [[ "$docker_info_rc" -eq 0 ]]; then
    echo "[triton-prereqs] docker daemon: OK"
  else
    if echo "$docker_info_out" | grep -qi "permission denied"; then
      echo "[triton-prereqs] docker daemon: PERMISSION DENIED (add user to docker group, then re-login/reboot)"
    else
      echo "[triton-prereqs] docker daemon: NOT READY (try: sudo systemctl enable --now docker)"
    fi
  fi
else
  echo "[triton-prereqs] docker: MISSING"
fi

echo "[triton-prereqs] triton endpoint should be: http://127.0.0.1:18000/v2/health/ready"
echo "[triton-prereqs] model repo should be: /home/daniel/projections-data/triton_models"
