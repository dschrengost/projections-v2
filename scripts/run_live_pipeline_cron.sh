#!/bin/bash
# DEPRECATED: cron is not allowed to orchestrate the pipeline.
# Prefect is the only orchestrator; remove any crontab entries that invoke this script.

set -euo pipefail

export PROJECTIONS_DATA_ROOT=/home/daniel/projections-data
cd /home/daniel/projects/projections-v2

echo "[cron] ERROR: scripts/run_live_pipeline_cron.sh is disabled. Use Prefect deployment schedules instead." >&2
echo "[cron] Suggested: /home/daniel/.local/bin/uv run prefect deployment run nba-live-pipeline/nba-live-pipeline" >&2
exit 2
