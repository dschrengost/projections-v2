#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

API_URL="${PREFECT_API_URL:-http://100.78.180.34:4200/api}"
SERVER_PREFECT_BIN="${SERVER_PREFECT_BIN:-/home/daniel/prod/projections-v2/.venv311/bin/prefect}"
DEPLOY_PREFECT_BIN="${DEPLOY_PREFECT_BIN:-/home/daniel/prod/projections-v2/.venv311/bin/prefect}"
DEPLOY_PYTHON_BIN="${DEPLOY_PYTHON_BIN:-/home/daniel/prod/projections-v2/.venv311/bin/python}"

PG_CONTAINER="${PREFECT_PG_CONTAINER:-prefect-postgres}"
PG_USER="${PREFECT_PG_USER:-prefect}"
PG_PASSWORD="${PREFECT_PG_PASSWORD:-prefect}"
PG_DB="${PREFECT_PG_DB:-prefect}"
PG_PORT="${PREFECT_PG_PORT:-55432}"

PG_URL="postgresql+asyncpg://${PG_USER}:${PG_PASSWORD}@127.0.0.1:${PG_PORT}/${PG_DB}"

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "[prefect-harden] missing required command: $1" >&2
    exit 1
  fi
}

need_cmd docker
need_cmd curl
need_cmd jq
need_cmd systemctl

if [[ ! -x "${SERVER_PREFECT_BIN}" ]]; then
  echo "[prefect-harden] missing executable: ${SERVER_PREFECT_BIN}" >&2
  exit 1
fi
if [[ ! -x "${DEPLOY_PREFECT_BIN}" ]]; then
  echo "[prefect-harden] missing executable: ${DEPLOY_PREFECT_BIN}" >&2
  exit 1
fi
if [[ ! -x "${DEPLOY_PYTHON_BIN}" ]]; then
  echo "[prefect-harden] missing executable: ${DEPLOY_PYTHON_BIN}" >&2
  exit 1
fi

echo "[prefect-harden] ensuring postgres container '${PG_CONTAINER}' is running..."
if docker ps -a --format '{{.Names}}' | grep -qx "${PG_CONTAINER}"; then
  docker start "${PG_CONTAINER}" >/dev/null
else
  docker run -d \
    --name "${PG_CONTAINER}" \
    --restart unless-stopped \
    -e "POSTGRES_USER=${PG_USER}" \
    -e "POSTGRES_PASSWORD=${PG_PASSWORD}" \
    -e "POSTGRES_DB=${PG_DB}" \
    -p "127.0.0.1:${PG_PORT}:5432" \
    -v prefect-postgres-data:/var/lib/postgresql/data \
    postgres:16-alpine >/dev/null
fi

for _ in $(seq 1 60); do
  if docker exec "${PG_CONTAINER}" pg_isready -U "${PG_USER}" -d "${PG_DB}" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
if ! docker exec "${PG_CONTAINER}" pg_isready -U "${PG_USER}" -d "${PG_DB}" >/dev/null 2>&1; then
  echo "[prefect-harden] postgres readiness check failed" >&2
  exit 1
fi

echo "[prefect-harden] setting selfhost profile DB settings..."
PREFECT_PROFILE=selfhost "${SERVER_PREFECT_BIN}" config set "PREFECT_API_DATABASE_CONNECTION_URL=${PG_URL}" >/dev/null
PREFECT_PROFILE=selfhost "${SERVER_PREFECT_BIN}" config set "PREFECT_API_DATABASE_TIMEOUT=30.0" >/dev/null
PREFECT_PROFILE=selfhost "${SERVER_PREFECT_BIN}" config set "PREFECT_API_DATABASE_CONNECTION_TIMEOUT=30.0" >/dev/null

server_pid="$(systemctl show -p MainPID --value prefect-server.service 2>/dev/null || echo 0)"
if [[ "${server_pid}" != "0" ]]; then
  echo "[prefect-harden] restarting prefect-server.service by signaling pid ${server_pid}..."
  kill -TERM "${server_pid}" || true
fi

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

echo "[prefect-harden] deploying all deployments from prefect.yaml..."
deploy_exit=1
for attempt in 1 2 3; do
  set +e
  (cd "${ROOT_DIR}" && PREFECT_API_URL="${API_URL}" "${DEPLOY_PREFECT_BIN}" deploy --all)
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

deployments_payload=""
for _ in $(seq 1 60); do
  deployments_payload="$(curl -fsS -X POST "${API_URL}/deployments/filter" \
    -H 'content-type: application/json' \
    -d '{"limit":200}' 2>/dev/null || true)"
  if [[ -n "${deployments_payload}" ]]; then
    break
  fi
  sleep 1
done
if [[ -z "${deployments_payload}" ]]; then
  echo "[prefect-harden] deployment verification failed: Prefect API unreachable after deploy" >&2
  exit 1
fi

if ! jq -e '.[] | select(.name=="nba-live-pipeline")' <<<"${deployments_payload}" >/dev/null; then
  echo "[prefect-harden] deployment verification failed: nba-live-pipeline missing" >&2
  exit 1
fi

echo "[prefect-harden] applying deployment concurrency overrides..."
(cd "${ROOT_DIR}" && PREFECT_API_URL="${API_URL}" "${DEPLOY_PYTHON_BIN}" tools/prefect_apply_deployment_overrides.py \
  --deployment nba-live-pipeline-v3/nba-live-pipeline \
  --collision-strategy CANCEL_NEW \
  --concurrency-limit 1)

echo "[prefect-harden] restarting prefect worker..."
systemctl --user restart prefect-worker.service

echo "[prefect-harden] live deployment schedule:"
curl -fsS -X POST "${API_URL}/deployments/filter" \
  -H 'content-type: application/json' \
  -d '{"deployments":{"operator":"and_","name":{"any_":["nba-live-pipeline"]}},"limit":1}' \
  | jq '.[0] | {name, paused, schedules: [.schedules[].schedule.cron], global_limit: (.global_concurrency_limit.limit // null), collision_strategy: .concurrency_options.collision_strategy}'

echo "[prefect-harden] prefect-server.service status:"
systemctl status prefect-server.service --no-pager -l | sed -n '1,18p'

echo "[prefect-harden] prefect-worker.service status:"
systemctl --user status prefect-worker.service --no-pager -l | sed -n '1,18p'
