# Repository Guidelines

## Documentation

Start with these canonical docs:

- [docs/00_REPO_MAP.md](docs/00_REPO_MAP.md) — Repository structure and entry points
- [docs/10_CONTROL_PLANE.md](docs/10_CONTROL_PLANE.md) — Prefect flows and systemd services
- [docs/20_DATA_CONTRACTS.md](docs/20_DATA_CONTRACTS.md) — Data schemas and feature contracts
- [docs/30_DEV_PLAYBOOK.md](docs/30_DEV_PLAYBOOK.md) — Developer setup and workflows

> [!NOTE]
> `docs/archive/` and `obsolete/` contain historical/reference material.
> Only `docs/00_REPO_MAP.md`, `10_...`, `20_...`, `30_...` and `README.md` are authoritative.

## Project Structure

- `projections/`: Core Python package
  - `api/`: FastAPI backend serving predictions + React dashboard
  - `builders/`: Feature builders (minutes, rates, DNP history, etc.)
  - `cli/`: Typer CLI commands
  - `etl/`: Extract/Transform/Load (schedule, injuries, odds, roster, boxscores)
  - `minutes_v1/`: Minutes prediction (core product)
  - `fpts_v1/`, `fpts_v2/`: Fantasy points prediction
  - `rates_v1/`: Per-minute stat rate predictions
  - `sim_v2/`: Monte Carlo FPTS simulator
  - `optimizer/`: Lineup optimizer (CP-SAT solver + late swap support)
- `scrapers/`: Web/API scrapers (NBA injuries, schedule, box scores, odds, props).
- `prefect_flows/`: Prefect workflow definitions (`nba-live-pipeline` is the canonical flow).
- `web/minutes-dashboard/`: React + Vite frontend.
- `tests/`: Pytest suite; mirrors module layout.
- `tools/`: One-off inspection and debugging scripts.
- `docs/`: Project documentation.

### Key Features

- **DNP History**: Tracks consecutive DNP-CD streaks to adjust `play_prob` predictions.
- **Contest Simulation**: Async job queue for simulating DFS contests with ROI/exposure analysis.
- **Late Swap**: Optimizer supports locked players and slot-aware swaps for live contests.
- **Props Analysis** (side project): Scrapes RotoWire props for over/under predictions—does not affect DFS flow.

### Production Models & Configs

**Current run configs** (what the live pipeline uses):

- `config/minutes_current_run.json` — Minutes model bundle path and reconciliation settings
- `config/rates_current_run.json` — Rates model run ID
- `config/rotation_set_minutes_live.json` — Rotation-share overlay settings
- `config/sim_v2_profiles.json` — Simulation profiles (baseline, ceiling, etc.)
- `config/optimizer.yaml` — Optimizer constraints and scoring rules

**Model artifacts** (trained model bundles):

- `artifacts/minutes_lgbm/` — LightGBM minutes models (bundle referenced by `minutes_current_run.json`)
- `artifacts/experiments/lgbm_rotalloc_final_v1/` — Rotation allocation model
- `models/ownership_v1/` — Ownership prediction model

To change the production minutes model, update `bundle_dir` in `config/minutes_current_run.json`.

## Data Paths

Set `PROJECTIONS_DATA_ROOT` to override the default `./data` location (production uses `/home/daniel/projections-data`).

```
$PROJECTIONS_DATA_ROOT/
├── bronze/          # Raw extracted data (time-series parquet)
├── silver/          # Normalized per-game snapshots
├── gold/            # Monthly backfill features (for analytics)
├── live/            # Per-run live predictions (dashboard reads from here)
│   └── features_minutes_v1/<date>/run=<ts>/
├── preds/           # Model predictions
└── labels/          # Frozen box score labels (immutable)
```

**Live vs Gold**: Live outputs are scoped by `run_id` and use "latest" pointer files. Gold is monthly backfill for analytics/training.

## Entry Points

**Canonical orchestrator** (source of truth for the production pipeline):

```bash
# Prefect flow - runs scrapers → features → scoring → sim → finalize
uv run python -m prefect_flows.live_nba_pipeline  # or trigger via Prefect UI
```

**Manual CLI commands** (useful for debugging):

```bash
uv run python -m projections.cli.build_minutes_live --date 2026-01-18 --run-as-of-ts 2026-01-18T20:00:00Z
uv run python -m projections.cli.score_minutes_v1 --date 2026-01-18
```

## API & Dashboard

**Backend** (FastAPI): `projections/api/minutes_api.py` → `create_app()`

```bash
# Development
uv run uvicorn projections.api.minutes_api:create_app --reload --port 8501

# Production (systemd)
systemctl --user status minutes-dashboard.service
systemctl --user restart minutes-dashboard.service
```

**Frontend** (React + Vite): `web/minutes-dashboard/`

```bash
cd web/minutes-dashboard
npm install && npm run build   # Build static assets (served by FastAPI)
npm run dev                    # Local dev server (port 5173)
```

> [!IMPORTANT]
> The production dashboard serves pre-built static assets from `web/minutes-dashboard/dist/`. After frontend changes, run `npm run build` and restart the API.

## Systemd Services

| Service | Status | Purpose |
|---------|--------|---------|
| `minutes-dashboard.service` | **Active** | FastAPI + static dashboard on port 8501 (with `--reload`) |
| `prefect-worker.service` | On-demand | Prefect worker for `projections-local` pool |

> [!TIP]
> The dashboard service uses `--reload`, so Python changes auto-reload without restarting. Frontend changes still require `npm run build`.

```bash
# Common commands
systemctl --user start prefect-worker.service
systemctl --user restart minutes-dashboard.service
journalctl --user -u minutes-dashboard.service -f  # Tail logs
```

## Debugging Patterns

**Pipeline failures**: Check Prefect UI (`http://localhost:4200`) or `journalctl --user -u prefect-worker.service`.

**Feature inspection**: Look at pointer files in `$PROJECTIONS_DATA_ROOT/live/features_minutes_v1/<date>/`.

**Player ID mismatches**: Check alias dictionaries in `projections/cli/finalize_projections.py`.

**Model not updating**: Verify `bundle_dir` in `config/minutes_current_run.json` points to the correct artifact.

## Common Gotchas

- **Anti-leak enforcement**: All pipelines enforce `as_of_ts ≤ tip_ts`. Features and labels are timestamped.
- **Player name normalization**: DK names may differ from internal IDs (e.g., "Alexandre Sarr" vs "Alex Sarr"). Check alias dicts.
- **run_id scoping**: Live outputs use `run=<timestamp>` directories. The "latest" pointer symlinks to the most recent run.
- **DNP-CD handling**: Players with consecutive DNP-CD streaks get adjusted `play_prob`. Check `projections/builders/dnp_history.py`.

## Build & Test

```bash
uv sync                    # Install dependencies
uv run pytest -q           # Run tests
uv run ruff check .        # Lint
```

## Coding Style

- Python 3.11+, 4-space indentation, type hints for public APIs.
- Naming: `snake_case` for functions/variables, `PascalCase` for classes.
- Use `ruff` for linting; keep code formatted and imports ordered.

## Testing Guidelines

- Use `pytest`; place tests under `tests/` with names like `test_<module>.py`.
- Mock network calls; avoid hitting live endpoints in unit tests.
- Add tests for new scrapers, CLI paths, and edge cases.

## Commit & PR Guidelines

- Commits: short, imperative subject; include scope when helpful (e.g., `scraper:`).
- PRs: describe motivation, summary of changes, testing notes, breaking changes.
- Link related issues; include sample commands or before/after snippets.

## Agent-Specific Notes

- Keep changes minimal and targeted; avoid drive-by refactors.
- Update docs and tests alongside code changes.
- Scrapers may require Java (tabula-py); call out prerequisites in PRs.
- Prefer resilient parsing and clear error messages.

## Tools

- Serena MCP memories — create a memory when ending a session or making major changes.
- Chrome DevTools MCP — use for web research and scraper development.