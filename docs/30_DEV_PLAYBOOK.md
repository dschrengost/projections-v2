# Developer Playbook

Setup, workflows, and common tasks for developing `projections-v2`.

## Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- Java (for tabula-py PDF parsing)
- Node.js 18+ (for frontend)
- Playwright Chromium (for RealGM depth chart scraper)

## Initial Setup

```bash
# Clone and enter repo
cd ~/projects/projections-v2

# Install Python dependencies
uv sync

# Install Playwright browser runtime used by RealGM scraper
uv run playwright install chromium

# Verify installation
uv run python -c "import projections; print('OK')"

# Run tests
uv run pytest -q
```

## Common Commands

### Testing

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_scrapers/test_injuries.py

# Run with coverage
uv run pytest --cov=projections --cov-report=html
```

### Linting

```bash
# Check code style
uv run ruff check .

# Auto-fix issues
uv run ruff check --fix .

# Format code
uv run ruff format .
```

### CLI Tools (Development Only)

These commands are for local development and debugging. **Production execution
must go through Prefect** (see below).

```bash
# Score minutes model (dev only)
uv run python -m projections.cli.score_minutes_v1 --date 2025-01-01

# Build features (dev only)
uv run python -m projections.cli.build_minutes_live --date 2025-01-01

# Finalize projections (dev only)
uv run python -m projections.cli.finalize_projections --date 2025-01-01
```

### RealGM Depth Chart Prior Ops

```bash
# Manually scrape + persist RealGM depth charts into bronze
uv run python -m projections.cli.scrape_realgm_depth_charts scrape --date 2026-01-18

# Rebuild/update RealGM->canonical player crosswalk from a minutes run
uv run python -m projections.cli.build_realgm_player_crosswalk run --date 2026-01-18
```

Operator notes:
- Manual crosswalk overrides live at `bronze/realgm/player_id_crosswalk_overrides.csv`.
- Required override columns: `realgm_player_id`, `player_id`.
- Live pipeline logs include `[dc-prior]`, `[dc-cap]`, `[dc-disagree]`, `[dc-crosswalk]`, and `[dc-alert]`.
- `PROJECTIONS_DC_CROSSWALK_WARN_MIN_MATCH_RATE` controls crosswalk warning threshold (default `0.30`).

### Rotation Eval (rot_eval)

```bash
# Baseline: prior_topn candidate pool, no gate
uv run python -m projections.cli.rot_eval \
  --rot-bundle $ROT_BUNDLE \
  --run-id prior_topn_nogate \
  --minutes-prior-parquet $PRIORS \
  --candidate-pool prior_topn \
  --candidate-top-n 11 \
  --no-gate

# Candidate pool: predictor_threshold, no gate
uv run python -m projections.cli.rot_eval \
  --rot-bundle $ROT_BUNDLE \
  --run-id predictor_pool_nogate \
  --minutes-prior-parquet $PRIORS \
  --rotation-predictor-bundle $PRED_BUNDLE \
  --gate-feature-source cached_all \
  --candidate-pool predictor_threshold \
  --pool-max-size 11 \
  --t-ge15 0.35 \
  --t-ge5 0.35 \
  --always-include-top-n 8 \
  --no-gate \
  --baseline-out-dir $BASELINE_OUT_DIR

# Candidate pool: predictor_threshold, with gate
uv run python -m projections.cli.rot_eval \
  --rot-bundle $ROT_BUNDLE \
  --run-id predictor_pool_gate \
  --minutes-prior-parquet $PRIORS \
  --rotation-predictor-bundle $PRED_BUNDLE \
  --gate-feature-source cached_all \
  --candidate-pool predictor_threshold \
  --pool-max-size 11 \
  --t-ge15 0.35 \
  --t-ge5 0.35 \
  --always-include-top-n 8 \
  --gate \
  --baseline-out-dir $BASELINE_OUT_DIR
```

### Production Pipeline (Prefect)

**Prefect is the single source of truth** for production orchestration.

```bash
# Trigger the canonical live pipeline
uv run prefect deployment run nba-live-pipeline/nba-live-pipeline

# Trigger for a specific date
uv run prefect deployment run nba-live-pipeline/nba-live-pipeline \
    --param game_date=2025-01-01

# Check deployment status
uv run prefect deployment ls

# View recent flow runs
uv run prefect flow-run ls --limit 10
```

Legacy shell scripts (`scripts/run_live_*.sh`) are gated and will refuse to run
unless `PROJECTIONS_ALLOW_LEGACY_SHELL_RUNNERS=1` is set. This gate exists to
prevent accidental direct execution in production.

### Running Services Locally

```bash
# Start API server
uv run uvicorn projections.api.minutes_api:app --reload --port 8000

# Start frontend (in web/ directory)
cd web && npm run dev
```

## Directory Conventions

| Directory | Purpose |
|-----------|---------|
| `scratch/` | Temporary work (gitignored) |
| `tools/` | One-off debugging scripts |
| `obsolete/` | Deprecated code for reference |
| `docs/archive/` | Historical planning documents |

## Commit Guidelines

- Use imperative mood: "Add feature" not "Added feature"
- Prefix with scope when helpful: `scraper: add Rotowire support`
- Keep commits focused and atomic
- Include test updates with code changes

## PR Checklist

- [ ] Tests pass (`uv run pytest`)
- [ ] Linting passes (`uv run ruff check .`)
- [ ] No new warnings introduced
- [ ] Documentation updated if needed
- [ ] Breaking changes called out

## Troubleshooting

### Import Errors

If you get import errors, ensure you're using `uv run`:
```bash
uv run python -c "import projections"
```

### Java/Tabula Issues

Some scrapers need Java. Install OpenJDK:
```bash
sudo apt install openjdk-11-jre-headless
```

### Prefect Connection

Ensure Prefect API is running:
```bash
prefect server status
# or
systemctl --user status prefect-worker
```

## See Also

- [00_REPO_MAP.md](./00_REPO_MAP.md) - Repository structure
- [10_CONTROL_PLANE.md](./10_CONTROL_PLANE.md) - Service management
- [AGENTS.md](../AGENTS.md) - AI agent guidelines
