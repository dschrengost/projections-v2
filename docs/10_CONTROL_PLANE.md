# Control Plane

Pipeline orchestration, Prefect flows, and systemd services for `projections-v2`.

## Prefect Architecture

### Main Flow

The primary pipeline is `live_pipeline_flow` in `prefect_flows/live_pipeline.py`. It orchestrates:

1. **Data scraping** - Injuries, lineups, odds, salaries
2. **Feature building** - Minutes features, rates features
3. **Model scoring** - Minutes, ownership, rates predictions
4. **Projection finalization** - Unified projections artifact
5. **Simulation** - Monte Carlo world generation

### Deployment

Prefect deployments are defined in `prefect.yaml`:

```bash
# Deploy all flows
prefect deploy --all

# View deployments
prefect deployment ls
```

### Flow Schedules

| Flow | Schedule | Purpose |
|------|----------|---------|
| `live-pipeline` | Every 5 min (game days) | Main prediction pipeline |
| `nightly-eval` | 3 AM daily | Model evaluation |

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

## See Also

- [00_REPO_MAP.md](./00_REPO_MAP.md) - Repository structure
- [30_DEV_PLAYBOOK.md](./30_DEV_PLAYBOOK.md) - Local development setup
