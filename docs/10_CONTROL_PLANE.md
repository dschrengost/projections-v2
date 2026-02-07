# Control Plane

Pipeline orchestration, Prefect flows, and systemd services for `projections-v2`.

## Prefect Architecture

### Main Flow

The primary pipeline is `nba_live_pipeline_flow` in `prefect_flows/live_nba_pipeline.py`. It orchestrates:

1. **Data scraping** - Injuries, lineups, odds, salaries
2. **Feature building** - Minutes features, rates features
3. **Model scoring** - Minutes, ownership, rates predictions
4. **Projection finalization** - Unified projections artifact
5. **Simulation** - Monte Carlo world generation

### Deployment

Prefect deployments are defined in `prefect.yaml`:

```bash
# Deploy all flows
uv run prefect deploy --all

# Prefect CLI does not currently persist all deployment fields from `prefect.yaml`
# (notably `concurrency_options`). Re-apply overrides after deploy.
uv run python tools/prefect_apply_deployment_overrides.py

# View deployments
uv run prefect deployment ls
```

### Flow Schedules

| Flow | Schedule | Purpose |
|------|----------|---------|
| `nba-live-pipeline` | Every 5 min (game days) | Main prediction pipeline |
| `boxscores-etl` | 3:30 AM daily | Scrape previous-day boxscores to bronze + legacy labels |
| `minutes-labels-refresh` | 3:40 AM daily | Materialize `gold/labels_minutes_v1` from boxscore raw partitions |
| `rates-training-base-refresh` | 4:05 AM daily | Refresh rates training base partitions |
| `gamerotation-scrape` | 4:15 AM daily | Scrape NBA Stats GameRotation feed |
| `rotation-priors-update` | 4:45 AM daily | Rebuild rotation priors after scrape |
| `nightly-eval` | 3 AM daily | Model evaluation |
| `minutes-retrain-pipeline` | Weekly (Tue 10 AM ET) | Minutes recency retrain + head-to-head eval vs prod |
| `rates-retrain-pipeline` | Weekly trigger (Tue 10 AM ET, biweekly gate in flow) | Rates recency retrain + calibration diagnostics + head-to-head guardrails + auto-promotion |
| `rates-calibration-monitor` | Weekly (Tue 9 AM ET) | Calibration diagnostics (decile curves / efficiency heads) for current production rates run |

## Systemd Services

Services live in `infra/systemd/` and are installed to `~/.config/systemd/user/`.

### Core Services

| Service | Purpose |
|---------|---------|
| `prefect-worker.service` | Prefect work pool worker |
| `live-api.service` | FastAPI backend |
| `minutes-dashboard.service` | React frontend |
| `mlflow.service` | MLflow tracking server |

### Management Commands

```bash
# Enable and start a service
systemctl --user enable --now prefect-worker.service

# Check status
systemctl --user status live-api.service

# View logs
journalctl --user -u prefect-worker.service -f

# Restart all services
systemctl --user restart prefect-worker live-api minutes-dashboard
```

## Pipeline Timers

Timers trigger pipeline runs on schedule:

| Timer | Triggers | Schedule |
|-------|----------|----------|
| `live-pipeline.timer` | `live-pipeline.service` | Game-time schedule |
| `nightly-eval.timer` | `nightly-eval.service` | 3 AM daily |

## Monitoring

- **Prefect UI**: http://localhost:4200
- **MLflow UI**: http://localhost:5000
- **API docs**: http://localhost:8000/docs
- **Dashboard**: http://localhost:3000

## RMH Shadow Mode (Minutes)

The live pipeline can optionally run the Rotation Minutes Hurdle model (RMH v1.1) in **shadow mode** alongside production minutes scoring.

- Enable: `RMH_SHADOW_ENABLED=1`
- Configure bundle: `RMH_ARTIFACT_DIR=/full/path/to/artifacts/rotation_minutes_hurdle_v1/<run_id>`
- Optional label: `RMH_MODEL_LABEL="RMH v1.1"`
- Output (dev namespace): `$PROJECTIONS_DATA_ROOT/artifacts/minutes_models/daily/model_id=rmh_v1_1/<game_date>/run=<run_id>/minutes.parquet`

Shadow failures are non-blocking and will log with `[rmh-shadow]`.

## See Also

- [00_REPO_MAP.md](./00_REPO_MAP.md) - Repository structure
- [30_DEV_PLAYBOOK.md](./30_DEV_PLAYBOOK.md) - Local development setup
