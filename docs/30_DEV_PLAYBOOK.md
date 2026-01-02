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

### CLI Tools

```bash
# Score minutes model
uv run python -m projections.cli.score_minutes_v1 --game-date 2025-01-01

# Build features
uv run python -m projections.cli.build_minutes_features_live --game-date 2025-01-01

# Finalize projections
uv run python -m projections.cli.finalize_projections --game-date 2025-01-01
```

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
