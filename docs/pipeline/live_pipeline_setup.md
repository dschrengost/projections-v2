# Live Pipeline Setup

Use this checklist before wiring the ETLs into cron/systemd. It ensures the runtime environment has the correct dependencies, storage layout, and configuration for aggressive live polling.

## 1. Python environment

```bash
# From the repo root
uv sync  # installs runtime + dev dependencies from pyproject.toml
```

All live commands (`projections.etl.*`, `projections.scrape.*`, etc.) should be invoked via `uv run ...` so they inherit the managed virtualenv.

## 2. System packages

The injuries scraper parses NBA PDF grids via `tabula-py`, which in turn requires Java and JPype:

1. Install a headless JRE (Debian/Ubuntu example):

   ```bash
   sudo apt-get update
   sudo apt-get install -y openjdk-17-jre-headless
   ```

2. Verify Java is on PATH:

   ```bash
   java -version
   ```

3. Ensure JPype is installed (`uv sync` already brings in `jpype1 >= 1.5.0`).

## 3. Data root

Set `PROJECTIONS_DATA_ROOT` to the mounted data volume (default `/home/daniel/projections-data`) and create the root folders:

```bash
export PROJECTIONS_DATA_ROOT=/home/daniel/projections-data
mkdir -p "$PROJECTIONS_DATA_ROOT"/{bronze,silver,labels}
```

This env var should be configured for the systemd service/cron job as well.

## 4. Environment check script

Run the helper to confirm Java, tabula-py, JPype, and the data root are all usable:

```bash
uv run python scripts/check_live_env.py
```

It will exit with a non-zero status if any dependency is missing.

## 5. Smoke test the ETLs

Before enabling automation, run the ETLs for a single day to make sure bronze/silver partitions materialize under the new layout:

```bash
uv run python -m projections.etl.injuries --start 2025-11-16 --end 2025-11-16 --season 2025 --month 11
uv run python -m projections.etl.daily_lineups --start 2025-11-16 --end 2025-11-16 --season 2025
uv run python -m projections.etl.roster_nightly --start 2025-11-16 --end 2025-11-16 --season 2025 --month 11
uv run python -m projections.etl.odds --start 2025-11-16 --end 2025-11-16 --season 2025 --month 11
```

If any command fails, inspect the logs in `/home/daniel/projections-data` and fix the dependency/permission issue before wiring up timers.

## 6. Orchestrator CLI

After verifying the individual ETLs, use the orchestrator to run them in sequence:

```bash
uv run python -m projections.cli.live_pipeline run \
  --start 2025-11-16 --end 2025-11-16 \
  --season 2025 --month 11
```

This command is the one you’ll wrap in systemd timers/cron jobs. Use `--skip-*` flags to disable specific stages or `--schedule/--roster` options to point at existing parquet inputs.

## 7. Systemd timers

Sample unit files live under `systemd/`:

- `live-pipeline.service`: base oneshot service that runs the CLI once.
- `live-pipeline-hourly.timer`: baseline hourly runs (08:00–23:00).
- `live-pipeline-evening.service/.timer`: every 10 minutes on weeknights (16:00–23:00).
- `live-pipeline-weekend.service/.timer`: every 15 minutes on weekends (12:00–23:00) with odds skipped for matinees.

Install them with:

```bash
sudo cp systemd/live-pipeline*.service systemd/live-pipeline*.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now live-pipeline.service
sudo systemctl enable --now live-pipeline-hourly.timer
sudo systemctl enable --now live-pipeline-evening.timer
sudo systemctl enable --now live-pipeline-weekend.timer
```

Adjust `Environment=LIVE_FLAGS=...` in the service files (season/month) as the year progresses, and use `journalctl -u live-pipeline.service -f` to tail outputs.

---

Once the steps above succeed consistently, it’s safe to rely on the timers for your desired cadence.
