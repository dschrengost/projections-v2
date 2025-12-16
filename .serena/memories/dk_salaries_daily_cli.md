Added daily DK salaries workflow (feature/dk-salaries-gold).

Key components:
- projections/dk/normalize.py: shared draftables_json_to_df helper (flatten DK draftables payload), normalization to gold salaries unchanged; raw_data stored as JSON string.
- scripts/dk/run_daily_salaries.py: Typer CLI to resolve slates (via slates.list_draft_groups_for_date with lobby payload), ensure bronze draftables (fetch if missing or force refresh), normalize to gold dk_salaries, and log summary per slate; exits non-zero on missing slates/rows.
- scripts/dk/fetch_draftables.py now uses draftables_json_to_df for CSV output; bronze path unchanged.
- Gold path helper: projections/dk/salaries_schema.dk_salaries_gold_path.

Usage example:
UV_PROJECT_ENVIRONMENT=.venv-user PROJECTIONS_DATA_ROOT=/home/daniel/projections-data \
  uv run python -m scripts.dk.run_daily_salaries --game-date 2025-11-30 --site dk --slate-type main --slate-type night
