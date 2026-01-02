#!/usr/bin/env bash
set -euo pipefail

if [[ "${PROJECTIONS_ALLOW_LEGACY_SHELL_RUNNERS:-}" != "1" ]]; then
  echo "[legacy-runner] ERROR: scripts/run_live_scrape.sh is disabled in production." >&2
  echo "[legacy-runner] Use Prefect: /home/daniel/.local/bin/uv run prefect deployment run nba-live-pipeline/nba-live-pipeline" >&2
  exit 2
fi

DATA_ROOT="${PROJECTIONS_DATA_ROOT:-/home/daniel/projections-data}"
START_DATE="${LIVE_START_DATE:-$(date -I)}"
END_DATE="${LIVE_END_DATE:-$START_DATE}"
SEASON="${LIVE_SEASON:-$(date +%Y)}"
MONTH="${LIVE_MONTH:-$(date +%-m)}"
FLAGS="${LIVE_FLAGS:-}"  # additional CLI flags

cd /home/daniel/projects/projections-v2

# Some runners accidentally pass quoted strings (e.g. "'2025'"). Typer expects integers.
strip_wrapping_quotes() {
  local s="$1"
  s="${s#\'}"; s="${s%\'}"
  s="${s#\"}"; s="${s%\"}"
  printf "%s" "$s"
}
RAW_SEASON="${SEASON}"
RAW_MONTH="${MONTH}"
SEASON="$(strip_wrapping_quotes "${SEASON}")"
MONTH="$(strip_wrapping_quotes "${MONTH}")"
if [[ "${SEASON}" != "${RAW_SEASON}" ]]; then
  echo "[live] WARNING: Stripped wrapping quotes from LIVE_SEASON=${RAW_SEASON} -> ${SEASON}" >&2
fi
if [[ "${MONTH}" != "${RAW_MONTH}" ]]; then
  echo "[live] WARNING: Stripped wrapping quotes from LIVE_MONTH=${RAW_MONTH} -> ${MONTH}" >&2
fi
if ! [[ "${SEASON}" =~ ^[0-9]+$ ]]; then
  echo "[live] ERROR: LIVE_SEASON must be an integer (got ${RAW_SEASON})" >&2
  exit 2
fi
if ! [[ "${MONTH}" =~ ^[0-9]+$ ]]; then
  echo "[live] ERROR: LIVE_MONTH must be an integer (got ${RAW_MONTH})" >&2
  exit 2
fi

# Run the scraping pipeline
/home/daniel/.local/bin/uv run python -m projections.cli.live_pipeline \
  --start "${START_DATE}" \
  --end "${END_DATE}" \
  --season "${SEASON}" \
  --month "${MONTH}" \
  --data-root "${DATA_ROOT}" \
  --schedule "${DATA_ROOT}/silver/schedule/season=${SEASON}/month=$(printf "%02d" "${MONTH}")/schedule.parquet" \
  ${FLAGS}
