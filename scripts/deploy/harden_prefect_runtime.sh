#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

API_URL="${PREFECT_API_URL:-http://127.0.0.1:4200/api}"

PREFECT_ENV_DIR="${HOME}/.config/projections"
PREFECT_ENV_FILE="${PREFECT_ENV_DIR}/prefect-server.env"

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "[prefect-harden] missing required command: $1" >&2
    exit 1
  fi
}

need_cmd curl
need_cmd systemctl
need_cmd uv

maybe_enable_postgres() {
  # If Docker is available, prefer Postgres over SQLite for Prefect metadata.
  if ! command -v docker >/dev/null 2>&1; then
    return 0
  fi
  if ! docker info >/dev/null 2>&1; then
    echo "[prefect-harden] docker found but daemon not ready; skipping postgres bootstrap"
    return 0
  fi

  local container="prefect-postgres"
  local pg_user="prefect"
  local pg_password="prefect"
  local pg_db="prefect"
  local pg_port="55432"
  local pg_url="postgresql+asyncpg://${pg_user}:${pg_password}@127.0.0.1:${pg_port}/${pg_db}"

  echo "[prefect-harden] ensuring ${container} container is running..."
  if docker ps -a --format '{{.Names}}' | grep -qx "${container}"; then
    docker start "${container}" >/dev/null
  else
    docker run -d \
      --name "${container}" \
      --restart unless-stopped \
      -e "POSTGRES_USER=${pg_user}" \
      -e "POSTGRES_PASSWORD=${pg_password}" \
      -e "POSTGRES_DB=${pg_db}" \
      -p "127.0.0.1:${pg_port}:5432" \
      -v prefect-postgres-data:/var/lib/postgresql/data \
      postgres:16-alpine >/dev/null
  fi

  for _ in $(seq 1 60); do
    if docker exec "${container}" pg_isready -U "${pg_user}" -d "${pg_db}" >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
  if ! docker exec "${container}" pg_isready -U "${pg_user}" -d "${pg_db}" >/dev/null 2>&1; then
    echo "[prefect-harden] WARNING: postgres readiness check failed; keeping existing DB config"
    return 0
  fi

  echo "[prefect-harden] writing ${PREFECT_ENV_FILE} (Postgres)"
  install -d -m 0755 "${PREFECT_ENV_DIR}"
  cat >"${PREFECT_ENV_FILE}" <<EOF
PREFECT_API_DATABASE_CONNECTION_URL=${pg_url}
PREFECT_API_DATABASE_TIMEOUT=30.0
PREFECT_API_DATABASE_CONNECTION_TIMEOUT=30.0
EOF
  chmod 0600 "${PREFECT_ENV_FILE}"
}

maybe_enable_postgres

echo "[prefect-harden] restarting prefect-server.service (user unit)..."
systemctl --user daemon-reload
systemctl --user restart prefect-server.service

echo "[prefect-harden] waiting for Prefect API health..."
for _ in $(seq 1 60); do
  if curl -fsS "${API_URL}/health" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
if ! curl -fsS "${API_URL}/health" >/dev/null 2>&1; then
  echo "[prefect-harden] prefect API failed health check: ${API_URL}/health" >&2
  exit 1
fi

echo "[prefect-harden] ensuring work pool exists: projections-local"
set +e
(cd "${ROOT_DIR}" && PREFECT_API_URL="${API_URL}" uv run prefect work-pool inspect projections-local >/dev/null 2>&1)
pool_ok=$?
set -e
if [[ "${pool_ok}" -ne 0 ]]; then
  (cd "${ROOT_DIR}" && PREFECT_API_URL="${API_URL}" uv run prefect work-pool create projections-local --type process)
fi

echo "[prefect-harden] deploying all deployments from prefect.yaml..."
deploy_exit=1
for attempt in 1 2 3; do
  set +e
  (cd "${ROOT_DIR}" && PREFECT_API_URL="${API_URL}" uv run prefect deploy --all)
  deploy_exit=$?
  set -e
  if [[ "${deploy_exit}" -eq 0 ]]; then
    break
  fi
  echo "[prefect-harden] warning: deploy attempt ${attempt}/3 failed with exit code ${deploy_exit}"
  if [[ "${attempt}" -lt 3 ]]; then
    sleep 3
  fi
done
if [[ "${deploy_exit}" -ne 0 ]]; then
  echo "[prefect-harden] warning: 'prefect deploy --all' failed after retries; continuing after verification"
fi

echo "[prefect-harden] applying deployment concurrency overrides..."
(cd "${ROOT_DIR}" && PREFECT_API_URL="${API_URL}" uv run python tools/prefect_apply_deployment_overrides.py \
  --deployment nba-live-pipeline-v3/nba-live-pipeline \
  --collision-strategy CANCEL_NEW \
  --concurrency-limit 1)

echo "[prefect-harden] restarting prefect worker..."
systemctl --user restart prefect-worker.service

echo "[prefect-harden] live deployment schedule:"
(cd "${ROOT_DIR}" && PREFECT_API_URL="${API_URL}" uv run prefect deployment inspect nba-live-pipeline-v3/nba-live-pipeline | sed -n '1,80p') || true

echo "[prefect-harden] prefect-server.service status:"
systemctl --user status prefect-server.service --no-pager -l | sed -n '1,18p'

echo "[prefect-harden] prefect-worker.service status:"
systemctl --user status prefect-worker.service --no-pager -l | sed -n '1,18p'
