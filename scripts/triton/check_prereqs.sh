#!/usr/bin/env bash
set -euo pipefail

echo "[triton-prereqs] host=$(hostname) user=$(whoami)"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[triton-prereqs] nvidia-smi: OK"
  nvidia-smi -L || true
else
  echo "[triton-prereqs] nvidia-smi: MISSING"
fi

if command -v docker >/dev/null 2>&1; then
  echo "[triton-prereqs] docker: OK ($(docker --version))"
  if docker info >/dev/null 2>&1; then
    echo "[triton-prereqs] docker daemon: OK"
  else
    echo "[triton-prereqs] docker daemon: NOT READY (try: sudo systemctl start docker)"
  fi
else
  echo "[triton-prereqs] docker: MISSING"
fi

echo "[triton-prereqs] triton endpoint should be: http://127.0.0.1:18000/v2/health/ready"
echo "[triton-prereqs] model repo should be: /home/daniel/projections-data/triton_models"

