#!/usr/bin/env bash
set -euo pipefail

# Bring up the GTV2 Triton server as a user-level service.
#
# Prereqs (see bootstrap_host_debian.sh):
# - NVIDIA driver installed and loaded (nvidia-smi works)
# - Docker installed and daemon running
# - NVIDIA container toolkit installed/configured (docker --gpus all works)
# - Current user can run docker without sudo (docker group + re-login)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROD_ROOT="${PROJECTIONS_PROD_ROOT:-/home/daniel/prod/projections-v2}"
DATA_ROOT="${PROJECTIONS_DATA_ROOT:-/home/daniel/projections-data}"

PY="${PROJECTIONS_PYTHON:-${PROD_ROOT}/.venv/bin/python3}"
if [[ ! -x "$PY" ]]; then
  PY="$(command -v python3 || true)"
fi
if [[ -z "$PY" ]]; then
  echo "[triton-bringup] error: python3 not found" >&2
  exit 1
fi

echo "[triton-bringup] checking prereqs..."
"${ROOT_DIR}/scripts/triton/check_prereqs.sh"

echo "[triton-bringup] ensuring model repo config exists..."
"$PY" "${PROD_ROOT}/scripts/triton/setup_gtv2_model_repo.py" \
  --model-repo "${DATA_ROOT}/triton_models" \
  --project-root "${PROD_ROOT}" \
  --bundle-dir "${DATA_ROOT}/artifacts/game_transformer_v2/bundle_current" \
  --device "cuda:0" \
  --num-worlds 25000 \
  --world-chunk-size 5000 \
  --force

echo "[triton-bringup] starting user service: triton-gtv2.service"
systemctl --user daemon-reload
systemctl --user start triton-gtv2.service

echo "[triton-bringup] waiting for health..."
for _ in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:18000/v2/health/ready" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
if ! curl -fsS "http://127.0.0.1:18000/v2/health/ready" >/dev/null 2>&1; then
  echo "[triton-bringup] error: triton not ready on http://127.0.0.1:18000" >&2
  systemctl --user status triton-gtv2.service --no-pager -l | sed -n '1,80p' || true
  exit 1
fi

echo "[triton-bringup] triton ready: http://127.0.0.1:18000"
echo "[triton-bringup] next: run smoke test"
echo "  ${PROD_ROOT}/.venv/bin/python3 ${PROD_ROOT}/scripts/triton/smoke_test_gtv2.py --triton-endpoint localhost:18000 --game-date 2026-03-10 --num-worlds 256"

