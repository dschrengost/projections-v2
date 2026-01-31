# DK Salaries Daily Flow

Manual flow for producing gold DraftKings salaries and using them in the optimizer.

Environment setup:
- `cd ~/projects/projections-v2`
- `export PROJECTIONS_DATA_ROOT=/home/daniel/projections-data`
- `export GAME_DATE=2025-11-30`
- `export SITE=dk`

Steps:
1) Discover slates (optional sanity)
```
UV_PROJECT_ENVIRONMENT=.venv-user uv run \
  python -m scripts.dk.list_slates \
    --game-date "$GAME_DATE" \
    --slate-type all
```

2) Fetch draftables (bronze)
```
UV_PROJECT_ENVIRONMENT=.venv-user uv run \
  python -m scripts.dk.fetch_draftables \
    --draft-group-id 138112
```

3) Normalize to gold salaries
```
UV_PROJECT_ENVIRONMENT=.venv-user uv run \
  python -m scripts.dk.normalize_draftables_to_salaries \
    --game-date "$GAME_DATE" \
    --site "$SITE" \
    --draft-group-id 138112
```

Gold output: `${PROJECTIONS_DATA_ROOT}/gold/dk_salaries/site=dk/game_date=YYYY-MM-DD/draft_group_id=<id>/salaries.parquet`

4) Build lineups from gold salaries (draft_group_id arg to be added to build_lineups_from_gold in follow-up)
```
UV_PROJECT_ENVIRONMENT=.venv-user PROJECTIONS_DATA_ROOT=/home/daniel/projections-data uv run \
  python scripts/optimizer/build_lineups_from_gold.py \
    --game-date "$GAME_DATE" \
    --site "$SITE" \
    --draft-group-id 138112 \
    --num-lineups 20
```

Notes:
- Bronze draftables JSON default: `${PROJECTIONS_DATA_ROOT}/bronze/dk/draftables/draftables_raw_<draft_group_id>.json`.
- `normalize_draftables_to_salaries` will fetch live from the DK API if the bronze JSON is missing (unless `--no-fetch-live`).
- Schema is defined in `projections/dk/salaries_schema.py`.
