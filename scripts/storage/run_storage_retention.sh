#!/usr/bin/env bash
set -euo pipefail

# Nightly storage retention planner / optional executor.
#
# Defaults to dry-run mode and writes reports under:
#   $PROJECTIONS_DATA_ROOT/artifacts/retention/reports/
#
# To execute archive+prune actions, set:
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
INCLUDE_CLASSIFICATIONS="${PROJECTIONS_RETENTION_INCLUDE_CLASSIFICATIONS:-noncanonical}"
INCLUDE_PROTECTED_ARCHIVE="${PROJECTIONS_RETENTION_INCLUDE_PROTECTED_ARCHIVE:-0}"
REQUIRE_ARCHIVE_RECEIPT="${PROJECTIONS_RETENTION_REQUIRE_ARCHIVE_RECEIPT:-1}"
MAX_ARCHIVE_FILES="${PROJECTIONS_RETENTION_MAX_ARCHIVE_FILES:-}"
MAX_ARCHIVE_BYTES="${PROJECTIONS_RETENTION_MAX_ARCHIVE_BYTES:-}"
MAX_DELETE_FILES="${PROJECTIONS_RETENTION_MAX_DELETE_FILES:-}"
MAX_DELETE_BYTES="${PROJECTIONS_RETENTION_MAX_DELETE_BYTES:-}"
EXECUTE="${PROJECTIONS_RETENTION_EXECUTE:-0}"
ARCHIVE_ROOT="${PROJECTIONS_ARCHIVE_ROOT:-/mnt/archive/projections-data-archive/warm}"

END_DATE="$(date -u +%F)"
START_DATE="$(date -u -d "${END_DATE} - ${LOOKBACK_DAYS} days" +%F)"

cd "$PROD_ROOT"
export PROJECTIONS_DATA_ROOT="$DATA_ROOT"

echo "[retention] families=${FAMILIES} lookback_days=${LOOKBACK_DAYS} start=${START_DATE} end=${END_DATE} archive_root=${ARCHIVE_ROOT}"
echo "[retention] mode=$([[ "$EXECUTE" == "1" ]] && echo execute || echo dry-run) min_prune_age_hours=${MIN_AGE_HOURS}"

weekly_args=(
  -m projections.cli.storage_retention_weekly
  --data-root "$DATA_ROOT"
  --hot-root "$DATA_ROOT"
  --archive-root "$ARCHIVE_ROOT"
  --families "$FAMILIES"
  --start-date "$START_DATE"
  --end-date "$END_DATE"
  --min-prune-age-hours "$MIN_AGE_HOURS"
  --include-classifications "$INCLUDE_CLASSIFICATIONS"
)

if [[ "$INCLUDE_PROTECTED_ARCHIVE" == "1" ]]; then
  weekly_args+=(--include-protected-archive)
else
  weekly_args+=(--no-include-protected-archive)
fi

if [[ "$REQUIRE_ARCHIVE_RECEIPT" == "1" ]]; then
  weekly_args+=(--require-archive-receipt-for-prune)
else
  weekly_args+=(--allow-prune-without-archive-receipt)
fi

if [[ -n "$MAX_ARCHIVE_FILES" ]]; then
  weekly_args+=(--max-archive-files "$MAX_ARCHIVE_FILES")
fi
if [[ -n "$MAX_ARCHIVE_BYTES" ]]; then
  weekly_args+=(--max-archive-bytes "$MAX_ARCHIVE_BYTES")
fi
if [[ -n "$MAX_DELETE_FILES" ]]; then
  weekly_args+=(--max-delete-files "$MAX_DELETE_FILES")
fi
if [[ -n "$MAX_DELETE_BYTES" ]]; then
  weekly_args+=(--max-delete-bytes "$MAX_DELETE_BYTES")
fi
if [[ "$EXECUTE" == "1" ]]; then
  weekly_args+=(--execute)
else
  weekly_args+=(--dry-run)
fi

"$PY" "${weekly_args[@]}"
