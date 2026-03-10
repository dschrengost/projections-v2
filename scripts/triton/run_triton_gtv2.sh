#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-start}"

TRITON_IMAGE="${TRITON_IMAGE:-projections-gtv2-triton:24.12-py3}"
TRITON_BASE_IMAGE="${TRITON_BASE_IMAGE:-nvcr.io/nvidia/tritonserver:24.12-py3}"
TRITON_CONTAINER_NAME="${TRITON_CONTAINER_NAME:-gtv2-triton}"
TRITON_MODEL_REPO="${TRITON_MODEL_REPO:-/home/daniel/projections-data/triton_models}"
TRITON_PROJECT_ROOT="${TRITON_PROJECT_ROOT:-/home/daniel/projects/projections-v2}"
TRITON_DATA_ROOT="${TRITON_DATA_ROOT:-/home/daniel/projections-data}"
TRITON_HTTP_PORT="${TRITON_HTTP_PORT:-18000}"
TRITON_GRPC_PORT="${TRITON_GRPC_PORT:-18001}"
TRITON_METRICS_PORT="${TRITON_METRICS_PORT:-18002}"
TRITON_MODEL_NAME="${TRITON_MODEL_NAME:-gtv2_scorer}"
TRITON_UID="${TRITON_UID:-$(id -u)}"
TRITON_GID="${TRITON_GID:-$(id -g)}"
TRITON_READY_ATTEMPTS="${TRITON_READY_ATTEMPTS:-60}"
TRITON_READY_SLEEP_SECONDS="${TRITON_READY_SLEEP_SECONDS:-1}"

require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "error: required command not found: $cmd" >&2
    exit 1
  fi
}

ensure_paths() {
  if [[ ! -d "$TRITON_MODEL_REPO" ]]; then
    echo "error: model repository not found: $TRITON_MODEL_REPO" >&2
    exit 1
  fi
  if [[ ! -d "$TRITON_PROJECT_ROOT" ]]; then
    echo "error: project root not found: $TRITON_PROJECT_ROOT" >&2
    exit 1
  fi
  if [[ ! -d "$TRITON_DATA_ROOT" ]]; then
    echo "error: data root not found: $TRITON_DATA_ROOT" >&2
    exit 1
  fi
}

image_exists() {
  docker image inspect "$TRITON_IMAGE" >/dev/null 2>&1
}

wait_until_ready() {
  require_cmd docker
  require_cmd curl

  local i
  for ((i = 1; i <= TRITON_READY_ATTEMPTS; i++)); do
    if curl -fsS "http://127.0.0.1:${TRITON_HTTP_PORT}/v2/health/ready" >/dev/null 2>&1 \
      && curl -fsS "http://127.0.0.1:${TRITON_HTTP_PORT}/v2/models/${TRITON_MODEL_NAME}" >/dev/null 2>&1; then
      return 0
    fi
    if ! docker ps --filter "name=^/${TRITON_CONTAINER_NAME}$" --format "{{.Names}}" | grep -Fxq "$TRITON_CONTAINER_NAME"; then
      echo "error: $TRITON_CONTAINER_NAME exited before reaching ready state" >&2
      docker logs --tail 200 "$TRITON_CONTAINER_NAME" || true
      return 1
    fi
    sleep "$TRITON_READY_SLEEP_SECONDS"
  done

  echo "error: triton did not reach ready state in ${TRITON_READY_ATTEMPTS}s" >&2
  docker logs --tail 200 "$TRITON_CONTAINER_NAME" || true
  return 1
}

build_image() {
  require_cmd docker
  docker build \
    -f "$TRITON_PROJECT_ROOT/scripts/triton/Dockerfile" \
    -t "$TRITON_IMAGE" \
    --build-arg BASE_IMAGE="$TRITON_BASE_IMAGE" \
    "$TRITON_PROJECT_ROOT"
}

start_container() {
  require_cmd docker
  ensure_paths

  if ! image_exists; then
    echo "image $TRITON_IMAGE missing; building now..."
    build_image
  fi

  docker rm -f "$TRITON_CONTAINER_NAME" >/dev/null 2>&1 || true
  docker run -d \
    --name "$TRITON_CONTAINER_NAME" \
    --restart unless-stopped \
    --user "${TRITON_UID}:${TRITON_GID}" \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -p "${TRITON_HTTP_PORT}:8000" \
    -p "${TRITON_GRPC_PORT}:8001" \
    -p "${TRITON_METRICS_PORT}:8002" \
    -e PYTHONPATH="$TRITON_PROJECT_ROOT" \
    -e PROJECTIONS_DATA_ROOT="$TRITON_DATA_ROOT" \
    -v "$TRITON_MODEL_REPO:/models:rw" \
    -v "$TRITON_PROJECT_ROOT:$TRITON_PROJECT_ROOT:ro" \
    -v "$TRITON_DATA_ROOT:$TRITON_DATA_ROOT:rw" \
    "$TRITON_IMAGE" \
    tritonserver \
      --model-repository=/models \
      --http-port=8000 \
      --grpc-port=8001 \
      --metrics-port=8002 \
      --model-control-mode=explicit \
      --load-model="$TRITON_MODEL_NAME" \
      --strict-readiness=true

  echo "started container: $TRITON_CONTAINER_NAME"
  wait_until_ready
  echo "triton ready: http://127.0.0.1:${TRITON_HTTP_PORT} model=${TRITON_MODEL_NAME}"
}

stop_container() {
  require_cmd docker
  docker rm -f "$TRITON_CONTAINER_NAME" >/dev/null 2>&1 || true
  echo "stopped container: $TRITON_CONTAINER_NAME"
}

status_container() {
  require_cmd docker
  docker ps --filter "name=^/${TRITON_CONTAINER_NAME}$"
}

logs_container() {
  require_cmd docker
  docker logs --tail 200 -f "$TRITON_CONTAINER_NAME"
}

smoke_container() {
  wait_until_ready
  echo "triton ready: http://127.0.0.1:${TRITON_HTTP_PORT} model=${TRITON_MODEL_NAME}"
}

case "$ACTION" in
  build)
    build_image
    ;;
  start)
    start_container
    ;;
  stop)
    stop_container
    ;;
  restart)
    stop_container
    start_container
    ;;
  status)
    status_container
    ;;
  logs)
    logs_container
    ;;
  smoke)
    smoke_container
    ;;
  *)
    echo "usage: $0 {build|start|stop|restart|status|logs|smoke}" >&2
    exit 2
    ;;
esac
