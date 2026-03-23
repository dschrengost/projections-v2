#!/usr/bin/env bash
set -euo pipefail

# Nightly storage retention planner / optional executor.
#
# Defaults to dry-run mode and writes reports under:
#   $PROJECTIONS_DATA_ROOT/artifacts/retention/reports/
#
# To actually delete payloads, set:
#   PROJECTIONS_RETENTION_EXECUTE=1

PROD_ROOT="${PROJECTIONS_PROD_ROOT:-/home/daniel/prod/projections-v2}"
DATA_ROOT="${PROJECTIONS_DATA_ROOT:-/home/daniel/projections-data}"

PY="${PROJECTIONS_PYTHON:-${PROD_ROOT}/.venv/bin/python3}"
if [[ ! -x "$PY" ]]; then
  PY="$(command -v python3 || true)"
fi
if [[ -z "${PY}" ]]; then
  echo "[retention] error: python3 not found" >&2
  exit 1
fi

FAMILIES="${PROJECTIONS_RETENTION_FAMILIES:-gtv2_worlds}"
LOOKBACK_DAYS="${PROJECTIONS_RETENTION_LOOKBACK_DAYS:-30}"
MIN_AGE_HOURS="${PROJECTIONS_RETENTION_MIN_AGE_HOURS:-24}"
MAX_DELETE_FILES="${PROJECTIONS_RETENTION_MAX_DELETE_FILES:-}"
MAX_DELETE_BYTES="${PROJECTIONS_RETENTION_MAX_DELETE_BYTES:-}"
EXECUTE="${PROJECTIONS_RETENTION_EXECUTE:-0}"

END_DATE="$(date -u +%F)"
START_DATE="$(date -u -d "${END_DATE} - ${LOOKBACK_DAYS} days" +%F)"

cd "$PROD_ROOT"
export PROJECTIONS_DATA_ROOT="$DATA_ROOT"

echo "[retention] families=${FAMILIES} lookback_days=${LOOKBACK_DAYS} start=${START_DATE} end=${END_DATE}"
echo "[retention] mode=$([[ "$EXECUTE" == "1" ]] && echo execute || echo dry-run) min_age_hours=${MIN_AGE_HOURS}"

select_out="$("$PY" -m projections.cli.storage_select_canonical \
  --families "$FAMILIES" \
  --start-date "$START_DATE" \
  --end-date "$END_DATE" \
  --write-decisions)"
echo "$select_out"

canonical_map="$(echo "$select_out" | sed -nE 's/.* map=([^ ]+).*/\1/p' | tail -n 1)"
if [[ -z "$canonical_map" ]]; then
  echo "[retention] error: failed to parse canonical map path from output" >&2
  exit 1
fi

prune_args=(
  -m projections.cli.storage_prune
  --canonical-json "$canonical_map"
  --min-age-hours "$MIN_AGE_HOURS"
)
if [[ -n "$MAX_DELETE_FILES" ]]; then
  prune_args+=(--max-delete-files "$MAX_DELETE_FILES")
fi
if [[ -n "$MAX_DELETE_BYTES" ]]; then
  prune_args+=(--max-delete-bytes "$MAX_DELETE_BYTES")
fi
if [[ "$EXECUTE" == "1" ]]; then
  prune_args+=(--execute)
else
  prune_args+=(--dry-run)
fi

"$PY" "${prune_args[@]}"
