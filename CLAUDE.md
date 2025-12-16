# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development Commands

```bash
# Install dependencies (requires Python ≥3.11)
uv sync

# Run tests
uv run pytest -q

# Lint
uv run ruff check .

# Run a single test file
uv run pytest tests/test_minutes_v1_quickstart.py -q
```

## Key CLI Commands

```bash
# Score a daily slate (writes to artifacts/minutes_v1/daily/YYYY-MM-DD/)
uv run python -m projections.cli.score_minutes_v1 --date 2025-12-07

# Build live features (same-day, pre-tip)
uv run python -m projections.cli.build_minutes_live --date 2025-12-07 --run-as-of-ts 2025-12-07T15:00:00Z

# Build a full month (bronze → gold pipeline)
uv run python -m projections.cli.build_month 2024 12

# Run simulator
python -m scripts.sim_v2.generate_worlds_fpts_v2 --start-date 2025-12-01 --end-date 2025-12-07 --n-worlds 10000 --profile baseline

# Dashboard development
uv run uvicorn projections.api.minutes_api:create_app --reload
cd web/minutes-dashboard && npm install && npm run dev
```

## Architecture Overview

This is an NBA player minutes and fantasy points projection system with a Medallion data architecture (Bronze → Silver → Gold).

### Core Pipeline Flow

```
Raw Data (Injuries/Odds/Roster/Schedule)
  ↓ [ETL]
Bronze (time-series parquet)
  ↓ [Snapshots - latest as_of_ts ≤ tip_ts]
Silver (per-game snapshots)
  ↓ [MinutesFeatureBuilder]
Gold Features
  ↓ [LightGBM quantile regression]
Predictions (p10/p50/p90 with conformal calibration)
  ↓ [L2 Reconciliation]
Reconciled Minutes (team totals ≈ 240)
```

### Module Structure

- `projections/` - Main Python package
  - `minutes_v1/` - Minutes prediction (core product)
  - `fpts_v1/`, `fpts_v2/` - Fantasy points prediction
  - `rates_v1/` - Per-minute stat rate predictions
  - `sim_v2/` - Monte Carlo FPTS simulator
  - `optimizer/` - Lineup optimizer (CP-SAT solver)
  - `etl/` - Extract/Transform/Load (schedule, injuries, odds, roster, boxscores)
  - `features/` - Feature engineering modules (role, rest, trend, depth, etc.)
  - `models/` - Model training (Ridge, LightGBM, LSTM)
  - `api/` - FastAPI serving predictions + React dashboard
  - `cli/` - Typer CLI commands
- `scrapers/` - Web/API scraping modules
- `scripts/` - Shell and Python orchestration (run_*.sh)
- `web/minutes-dashboard/` - React + Vite frontend

### Data Paths

Set `PROJECTIONS_DATA_ROOT` to override the default `./data` location.

```
data/
├── bronze/          # Raw extracted data
├── silver/          # Normalized snapshots
├── gold/            # Final features
├── labels/          # Frozen box score labels (immutable)
├── live/            # Per-run live predictions
└── preds/           # Model predictions
```

## Critical Patterns

### Anti-Leak Enforcement

All pipelines enforce strict temporal ordering: `as_of_ts ≤ tip_ts`. Features, snapshots, and labels are timestamped and validated to prevent future information leakage.

### Feature Builder

`MinutesFeatureBuilder` chains feature modules with anti-leak validation at each stage:
- `_attach_schedule()` → `_attach_injuries()` → `_attach_odds()` → `_attach_depth()` → etc.

### Configuration

- YAML/JSON configs drive behavior (see `config/`)
- CLI flags override config defaults
- Pydantic validates config schemas

### Live vs Gold

- **Live**: Fresh per-run snapshots in `data/live/features_minutes_v1/<date>/run=<ts>/` → Dashboard
- **Gold**: Monthly backfill in `data/gold/features_minutes_v1/season=YYYY/month=MM/` → Analytics

## Code Style

- Python 3.11+, 4-space indentation
- Type hints required for public APIs
- `snake_case` for functions/variables, `PascalCase` for classes
- Use `ruff` for linting
- Mock network calls in tests

## Commit Guidelines

- Short imperative subject with scope: `scraper: add NBA.com daily lineups`
- Keep changes minimal and targeted
- Scrapers may require Java (tabula-py) and `jpype1`
