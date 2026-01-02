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

- `projections/`: Core Python package (CLI, API, models, ETL).
- `scrapers/`: Web/API scrapers (NBA injuries, schedule, box scores, odds).
- `prefect_flows/`: Prefect workflow definitions.
- `tests/`: Pytest suite; mirrors module layout.
- `tools/`: One-off inspection and debugging scripts.
- `docs/`: Project documentation.

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