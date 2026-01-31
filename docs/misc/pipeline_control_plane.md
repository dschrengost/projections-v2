# Pipeline Control Plane (NBA Live) — Current State & One True Path

This document describes **what runs the live NBA pipeline today**, where it writes, where overrides can enter, and the **new canonical production path**.

## Goals (production)

- Prefect is the **single source of truth** for orchestration and scheduling.
- Every run is **run_id-scoped**, has a **manifest** written at start, and is **atomically published** via pointers.
- Manual overrides are applied only via the **GameView** path; legacy override paths are disabled so they cannot affect production outputs.

## Current entrypoints (as found in-repo)

### Prefect

- Deployments are configured in `prefect.yaml`.
- Canonical flow code lives in `prefect_flows/live_nba_pipeline.py`.
- Legacy flow code lives in `prefect_flows/live_pipeline.py` (wrappers around bash scripts / subprocess calls).

Key deployments (current):
- `prefect_flows/live_nba_pipeline.py:nba_live_pipeline_flow` (**canonical**, end-to-end)
- `prefect_flows/boxscores_etl.py:boxscores_etl_flow` (yesterday boxscores ETL; scheduled before nightly eval)
- `prefect_flows/nightly_eval.py:nightly_eval_flow` (nightly evaluation; ensures boxscores exist before evaluating)
- `prefect_flows/live_pipeline.py:*` (legacy wrappers; deprecated and should remain unscheduled)

### systemd (in repo)

Systemd unit files live in `systemd/` and included timers/services that historically ran pipeline jobs directly.
To enforce Prefect as the only orchestrator:

- `systemd/live-*.service` now trigger the Prefect deployment `nba-live-pipeline/nba-live-pipeline` (no direct pipeline work).
- `systemd/live-*.timer` are disabled (set to a far-future `OnCalendar=`) so systemd does not schedule pipeline work.
- Other scheduled ETL jobs that touch pipeline inputs/labels:
  - `systemd/dk-salaries.timer` → **DEPRECATED** (DK salaries are fetched inside the canonical pipeline; do not schedule separately)
  - `systemd/freeze-slates.timer` → `projections.cli.freeze_slates`
  - `systemd/live-boxscores.timer` → **DEPRECATED** (Prefect `boxscores-etl` deployment)
  - `systemd/tracking-daily.timer` → `scripts/run_tracking_daily.sh`
  - `systemd/nightly-eval.timer` → **DEPRECATED** (Prefect `nightly-eval` deployment)

**Important:** Running these timers/services alongside Prefect creates multiple overlapping schedulers and duplicate triggers.

### Cron-like scripts (in repo)

- `scripts/run_live_pipeline_cron.sh` — disabled (exits non-zero) to prevent cron from orchestrating pipeline work.

### API-triggered runs

- `POST /api/trigger` in `projections/api/minutes_api.py` triggers the Prefect deployment `nba-live-pipeline/nba-live-pipeline` (no direct pipeline work).

## Where the live pipeline writes today (high-level)

Default `data_root` is `PROJECTIONS_DATA_ROOT` (commonly `/home/daniel/projections-data`).

### Scrape / ETL outputs

- Injuries/lineups/odds/roster/schedule are written under bronze/silver layers (examples):
  - `silver/schedule/season=.../month=.../schedule.parquet`
  - `silver/roster_nightly/.../roster.parquet`
  - `silver/espn_injuries/date=YYYY-MM-DD/injuries.parquet`
  - `silver/rotowire_lineups/date=YYYY-MM-DD/lineups.parquet`

### Live scoring artifacts (run-scoped)

- Minutes features: `live/features_minutes_v1/YYYY-MM-DD/run=<run_id>/features.parquet` (+ `latest_run.json`)
- Minutes predictions: `artifacts/minutes_v1/daily/YYYY-MM-DD/run=<run_id>/minutes.parquet` (+ `latest_run.json`)
- Rates features: `live/features_rates_v1/YYYY-MM-DD/run=<run_id>/features.parquet` (+ `latest_run.json`)
- Rates predictions: `gold/rates_v1_live/YYYY-MM-DD/run=<run_id>/rates.parquet` (+ `latest_run.json`)
- Sim worlds/projections: `artifacts/sim_v2/worlds_fpts_v2/game_date=YYYY-MM-DD/run=<run_id>/...` (+ `latest_run.json`)
- Unified projections: `artifacts/projections/YYYY-MM-DD/run=<run_id>/projections.parquet` (+ `latest_run.json`, plus `blessed_run.json` / `pinned_run.json` pointers)

### Non-atomic publishes (risk)

- Some jobs still write mutable “flat” files (e.g., copying minutes into `gold/projections_minutes_v1/.../minutes.parquet`), which allows downstream readers to observe partial writes.

## Override surfaces (current)

### Production-impacting overrides (must be controlled)

- **GameView (authorized) overrides**: stored in `artifacts/ops/overrides_v1/game_date=YYYY-MM-DD/overrides.json`
  - Applied exactly once during the pipeline into `effective_minutes.parquet` via `projections/pipeline/effective_inputs.py`.
  - Downstream consumers prefer `effective_minutes.parquet` (no re-application at sim/finalize time).

### Non-production overrides (should not affect pipeline outputs)

- Optimizer “My Proj” overrides: `data_root/user_overrides/...` via `projections/api/optimizer_api.py`
  - **Disabled by default** via `PROJECTIONS_ALLOW_LEGACY_USER_OVERRIDES=0` (must be explicitly enabled for local debugging).
  - Pipeline inputs are not allowed to consume this path.

### Run selection overrides

- `pinned_run.json` under `artifacts/projections/YYYY-MM-DD/` can override what the API serves (priority is blessed > pinned > latest).

## The new “one true path” (target state)

Production path becomes:

1. **Prefect deployment** runs the canonical flow entrypoint (end-to-end): `prefect_flows/live_nba_pipeline.py:nba_live_pipeline_flow`.
2. Flow acquires a **single-writer guard** and writes a **run manifest** at start.
3. Each stage writes to **run-scoped output directories** only.
4. After each stage, **health checks** validate contracts; failures hard-stop the run.
5. After success, outputs are **atomically promoted** via pointer files:
   - preferred: `.../LATEST/current.json`
   - legacy/back-compat: `.../latest_run.json`
6. Overrides are applied only via the **GameView** path and materialized into a single “effective inputs” layer consumed downstream.

When this migration is complete:
- systemd timers/services and cron runners no longer execute pipeline steps directly.
- systemd is used only to run Prefect worker/service(s).
 - legacy shell entrypoints (`scripts/run_live_*.sh`) are gated behind `PROJECTIONS_ALLOW_LEGACY_SHELL_RUNNERS=1` and should remain OFF in production.

## Safety toggles (debug only)

- `PROJECTIONS_ALLOW_UNPROMOTED_RUN_READS=1`: allows readers to scan `run=*` directories when pointers are missing (default OFF).
- `PROJECTIONS_ALLOW_LEGACY_USER_OVERRIDES=1`: enables optimizer “My Proj” overrides path (default OFF).

## Diagnostics

- Trigger a one-off run: `uv run prefect deployment run nba-live-pipeline/nba-live-pipeline --param game_date=YYYY-MM-DD`
- Trace lineage for an output: `uv run python -m projections.cli.trace_output <path> --data-root /home/daniel/projections-data`
