# Repository Map

Overview of the `projections-v2` repository structure and key entry points.

## Directory Structure

```
projections-v2/
├── projections/          # Core Python package
│   ├── api/              # REST API endpoints (FastAPI)
│   ├── cli/              # Command-line tools
│   ├── contest_sim/      # Contest simulation engine
│   ├── etl/              # Data transformation pipelines
│   ├── fpts_v2/          # Fantasy points calculations
│   ├── minutes_v1/       # Minutes projection model
│   ├── models/           # ML models and feature contracts
│   ├── optimizer/        # Lineup optimization
│   └── worlds/           # Monte Carlo simulation worlds
├── prefect_flows/        # Prefect workflow definitions
├── scrapers/             # External data scrapers
├── scripts/              # Operational scripts
│   ├── sim_v2/           # Simulation scripts
│   └── run_*.sh          # Pipeline runners
├── systemd/              # Systemd service/timer units
├── tests/                # Pytest test suite
├── tools/                # One-off inspection/debugging scripts
├── web/                  # Frontend dashboard (React)
├── config/               # Runtime configuration files
├── docs/                 # Documentation
│   ├── archive/          # Historical planning docs
│   └── [topic]/          # Topic-specific docs
└── obsolete/             # Deprecated code (reference only)
```

## Key Entry Points

| Purpose | Location |
|---------|----------|
| Live pipeline flow | `prefect_flows/live_pipeline.py` |
| Minutes API | `projections/api/minutes_api.py` |
| Optimizer service | `projections/api/optimizer_service.py` |
| CLI commands | `projections/cli/` |
| Frontend | `web/` |

## Configuration

- `config/minutes_current_run.json` - Active minutes model bundle
- `config/sim_v2_profiles.json` - Simulation profiles
- `prefect.yaml` - Prefect deployment configuration

## Data Locations

Production data lives **outside this repo** at `/home/daniel/projections-data/`:
- `artifacts/` - Pipeline outputs
- `bronze/` - Raw scraped data
- `silver/` - Transformed data
- `gold/` - Feature-engineered data

## See Also

- [10_CONTROL_PLANE.md](./10_CONTROL_PLANE.md) - Pipeline orchestration
- [20_DATA_CONTRACTS.md](./20_DATA_CONTRACTS.md) - Data schemas
- [30_DEV_PLAYBOOK.md](./30_DEV_PLAYBOOK.md) - Developer workflows
