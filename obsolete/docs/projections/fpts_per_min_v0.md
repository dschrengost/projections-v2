# FPTS per Minute v0

## Overview

- Predicts `fpts_per_min = actual_fpts / actual_minutes` using LightGBM regression.
- Relies on DraftKings scoring via the shared boxscore payloads.
- Consumes existing gold minutes feature parquet slices and augments with player priors, usage, and roster context.

## Feature Blocks

1. **Minutes / Role**
   - Consumes `minutes_p10/p50/p90` (predicted or actual fallback) plus volatility spans and play probability.
   - Keeps `starter_flag`, `is_projected_starter`, lineup role tiers, roster/ramp flags, and existing minutes trend features.
2. **Player Priors**
   - Rolling, minutes-weighted FPTS/min (5 & 10 games) and season-to-date priors.
   - Rolling usage per minute, assists per minute, rebounds per minute, and cumulative games/minutes.
3. **Game Context**
   - Spread, total, team/opponent implied totals, blowout index, home/away, rest flags, season phase.
4. **Roster Context**
   - Team-level counts for `out`/`questionable`, same-position outs, and minutes-weighted sum of missing teammate priors.

## Minutes Source

Two modes wired into `FptsDatasetBuilder` and the CLI:

- `predicted` (default): Reads historical runs from `gold/prediction_logs_minutes/run=<minutes_run>` (falling back to the legacy `prediction_logs_minutes_v1` tree) and deduplicates by the latest run timestamp.
- `actual`: Uses realized minutes for experimentation; clearly tagged and logs a warning when filling gaps.

> ⚠️ Training with `minutes_source="actual"` is optimistic/leaky and should be limited to experiments. The CLI now emits a warning to call this out explicitly.

## Training CLI

```
uv run python -m projections.models.fpts_lgbm \
  --run-id fpts_lgbm_v2 \
  --data-root /home/daniel/projections-data \
  --artifact-root artifacts/fpts_lgbm \
  --train-start 2023-10-24 --train-end 2023-12-10 \
  --cal-start 2023-12-11 --cal-end 2023-12-20 \
  --val-start 2023-12-21 --val-end 2023-12-31 \
  --minutes-source actual --scoring-system dk
```

Artifacts per run: `model.joblib`, `metrics.json`, `config.json`, and `report.md` with bucketed breakdowns (starters vs bench, high vs low minutes, injury-struck teams, favorites vs underdogs).

## Inference

- `projections/fpts_v1/production.py` exposes `load_fpts_model`, `load_production_fpts_bundle`, `predict_fpts_per_min`, and `predict_fpts`.
- Assumes a slate dataframe already contains minutes_v2 outputs; returns per-minute and projected FPTS (`p50 * fpts_per_min_pred`).

## Production Config

- `config/fpts_current_run.json` points to the canonical production run:

  ```json
  {
    "run_id": "fpts_lgbm_v2",
    "artifact_root": "artifacts/fpts_lgbm/fpts_lgbm_v2",
    "scoring_system": "dk",
    "minutes_source": "actual"
  }
  ```

- The live pipeline loads this config via `load_production_fpts_bundle`. Update the JSON (and commit it) whenever a new run is promoted. Optional env vars (`FPTS_PRODUCTION_DIR`, `FPTS_PRODUCTION_RUN_ID`, etc.) override the JSON for ad hoc runs.
- Keep the scoring system field in sync with the run’s metadata so downstream consumers know whether DK or FD points are being served.

## Next Steps / Notes

- Minutes backfill currently expects populated `gold/prediction_logs_minutes/run=<minutes_run>` partitions; schedule a historical replay job if gaps exist (legacy `prediction_logs_minutes_v1` is still honored as a fallback).
- Future FanDuel support: extend `SCORING_SYSTEMS` with an FD calculator and pass `--scoring-system fd` through the CLI.
- The dataset builder warns when predicted minutes are missing; add monitoring once live backfills are in place.

## FPTS v1 – Operations

- **Train a run**:

  ```bash
  PROJECTIONS_DATA_ROOT=/home/daniel/projections-data \
  uv run python -m projections.models.fpts_lgbm \
    --run-id fpts_lgbm_v1 \
    --data-root /home/daniel/projections-data \
    --artifact-root artifacts/fpts_lgbm \
    --train-start 2022-10-01 --train-end 2024-02-28 \
    --cal-start 2024-03-01 --cal-end 2024-06-30 \
    --val-start 2024-07-01 --val-end 2024-12-31 \
    --minutes-source predicted --scoring-system dk
  ```

- **Promote a run to production**:

  ```bash
  cat config/fpts_current_run.json   # edit run_id, artifact_root, scoring_system
  ```

- **Backfill FPTS for a window**:

  ```bash
  PROJECTIONS_DATA_ROOT=/home/daniel/projections-data \
  uv run python -m projections.cli.backfill_fpts_v1 \
    --start-date 2023-10-01 \
    --end-date 2024-04-15 \
    --minutes-run-id <minutes_run_id> \
    --fpts-run-id fpts_lgbm_v2 \
    --overwrite false
  ```

- **Live scoring / API verification**:

  ```bash
  PROJECTIONS_DATA_ROOT=/home/daniel/projections-data LIVE_SCORE=1 \
  ./scripts/run_live_pipeline.sh

  uv run uvicorn projections.api.minutes_api:create_app --host 0.0.0.0 --port 8000
  curl 'http://localhost:8000/api/minutes?date=2024-01-15' \
    | jq '.players[0] | {minutes_p50_cond, fpts_per_min_pred, proj_fpts}'
  ```

## Current production status

- `fpts_lgbm_v2` (minutes_source=`actual`) is the promoted FPTS v0 head listed in `config/fpts_current_run.json`.
- The live pipeline (`LIVE_SCORE=1 ./scripts/run_live_pipeline.sh`) reads that config, scores FPTS for each slate, and writes outputs to `gold/projections_fpts_v1/date=*/run=*`.
- The minutes API/dashboard surface `proj_fpts` and `fpts_per_min_pred` directly from those gold files and display `fpts_meta` so operators can confirm the active run.
