# Developer Playbook

Setup, workflows, and common tasks for developing `projections-v2`.

## Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- Java (for tabula-py PDF parsing)
- Node.js 18+ (for frontend)

## Initial Setup

```bash
# Clone and enter repo
cd ~/projects/projections-v2

# Install Python dependencies
uv sync

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
