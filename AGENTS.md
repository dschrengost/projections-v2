# Repository Guidelines

## Project Structure & Module Organization
- `projections/`: CLI and pipeline entry points (e.g., `scrape.py`).
- `scrapers/`: Web/API scrapers (NBA injuries, schedule, box scores, odds).
- `tests/`: Pytest suite; mirrors module layout (e.g., `tests/test_scrapers/`).
- `data/`: Local artifacts and cached outputs (JSON/Parquet).
- `docs/`: Project reports and design notes (e.g., `minutes_model_report.md`).

## Build, Test, and Development Commands
- Install deps: `uv sync` (requires Python ≥ 3.11).
- Run tests: `uv run pytest -q`.
- Lint (ruff): `uv run ruff check .`.
- CLI examples:
  - Injuries (range): `uv run projections/scrape.py injuries --mode range --start 2024-10-21T00:30-04:00 --end 2025-04-13T23:30-04:00 --out data/injuries.json`.
  - Schedule (season): `uv run projections/scrape.py schedule --mode season --season 2025-26 --out data/schedule.json`.

## Coding Style & Naming Conventions
- Python, 4-space indentation, type hints required for public APIs.
- Naming: `snake_case` for functions/variables, `PascalCase` for classes.
- Keep modules focused; prefer dataclasses for simple models.
- Use `ruff` for linting; keep code formatted and imports ordered.

## Testing Guidelines
- Use `pytest`; place tests under `tests/` with names like `test_<module>.py`.
- Mock network calls; avoid hitting live endpoints in unit tests.
- Add tests for new scrapers, CLI paths, and edge cases (empty/malformed responses).

## Commit & Pull Request Guidelines
- Commits: short, imperative subject; include scope when helpful (e.g., `scraper:`).
- PRs: describe motivation, summary of changes, testing notes, and any breaking changes.
- Link related issues; include sample commands or before/after snippets.

## Agent-Specific Notes
- Keep changes minimal and targeted; avoid drive-by refactors.
- Update docs and tests alongside code changes.
- Scrapers may require Java (tabula-py) and `jpype1`; call out prerequisites in PRs.
- Prefer resilient parsing and clear error messages; guard against schema drift.

## Tools
- Serena mcp memories -- create a memory when ending a session, or when making major changes / refactors
- Chrome Devtools mcp -- use this when researching anything on the web, creating scrapers. It is a very powerful web