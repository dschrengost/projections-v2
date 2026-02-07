# Minutes V1 Retrain (Recency Decay H=35)

## Run IDs

- Full retrain run: `minutes_v1_recency_h35_20260207T110500Z`
- Smoke run: `minutes_v1_recency_h35_smoke_20260207T101500Z`
- Dataset-only run: `minutes_v1_recency_h35_dataset_only_20260207T111500Z`

## Windows Used

Requested:
- Train: `2025-02-01` → `2026-01-31`
- Cal/Val: `2026-02-01` → `2026-02-05`

Effective (label/feature clamped):
- Train: `2025-10-21` → `2026-01-31`
- Cal/Val: `2026-02-01` → `2026-02-05`

Reason for clamp:
- Label bounds in `/home/daniel/projections-data/labels/season=2025/boxscore_labels.parquet` start at `2025-10-21`.
- Gold features for `season=2025` were also only available starting `2025-10-21`.

## Recency Weighting

- Formula: `w = 2 ** (-age_days / 35.0)`
- Age anchor: `train_end_date = 2026-01-31`
- Applied only to TRAIN rows via column `weight_recency`

Train weight distribution (`dataset meta`):
- min: `0.1327`
- p05: `0.1408`
- p50: `0.2092`
- p95: `0.5202`
- max: `1.0000`

## Dataset Build Summary

Dataset artifact:
- `/home/daniel/projections-data/artifacts/minutes_retrain_runs/minutes_v1_recency_h35_20260207T110500Z/dataset.parquet`

Meta artifact:
- `/home/daniel/projections-data/artifacts/minutes_retrain_runs/minutes_v1_recency_h35_20260207T110500Z/meta.json`

Counts:
- Total rows: `12,380`
- Train rows (`split=train`): `11,027`
- Cal rows (`split=cal`): `1,353`
- Dropped missing labels: `0`
- Leakage violations (`feature_as_of_ts > tip_ts`): `0`

## Model Bundle Output

Bundle:
- `/home/daniel/projections-data/artifacts/minutes_lgbm/minutes_v1_recency_h35_20260207T110500Z/`

Key files:
- `lgbm_quantiles.joblib`
- `metrics.json`
- `meta.json`
- `conformal_offsets.json`

`meta.json` includes:
- `recency_decay` settings
- `train_cal_windows`
- `retrain_dataset` pointers

## Validation Metrics (Cal/Val Window 2026-02-01..2026-02-05)

From `metrics.json`:
- Play-prob false active rate (`pred>=0.5`, actual inactive): `0.09682`
- Play-prob false inactive rate (`pred<=0.2`, actual active): `0.04213`
- Play-prob Brier score: `0.13297`
- Conditional MAE (`p50` vs actual, active rows): `8.35077`
- Bench smear proxy (`p50>10`, actual<1): `0.18551`

Additional context:
- Train rows used by trainer after filters: `10,413`
- Cal rows used by trainer: `1,353`
- Val rows used by trainer: `1,353`
- `sample_weight_col`: `weight_recency`
- `train_sample_weight_used_play_prob_head`: `true`
- `train_sample_weight_used_conditional_head`: `true`

## Issues Encountered

- Initial full run failed guardrails (`bench|>18 conditional P90`) with strict guard behavior.
- Final run completed with `--allow-guard-failure` to produce artifacts and metrics for review.
- MLflow logging raised a non-fatal warning: `float() argument must be a string or a real number, not 'NoneType'`.
- Gold features were stale (only through `2025-11-30`); rebuilt canonical features for:
  - `2025-12-01..2025-12-31`
  - `2026-01-01..2026-01-31`
  - `2026-02-01..2026-02-05`

## Commands Executed

- Lint/tests (touched files):
  - `uv run ruff check projections/minutes_v1/retrain_dataset.py projections/cli/retrain_minutes_v1_recency.py projections/minutes_v1/modeling.py projections/models/minutes_lgbm.py tests/test_minutes_v1_retrain_dataset.py`
  - `uv run pytest -q tests/test_minutes_v1_retrain_dataset.py tests/test_minutes_v1_modeling.py tests/test_minutes_v1_datasets.py`

- Feature backfill (canonical builder):
  - `uv run python -m projections.pipelines.build_features_minutes_v1 --start-date 2025-12-01 --end-date 2025-12-31 --season 2025 --month 12`
  - `uv run python -m projections.pipelines.build_features_minutes_v1 --start-date 2026-01-01 --end-date 2026-01-31 --season 2025 --month 1`
  - `uv run python -m projections.pipelines.build_features_minutes_v1 --start-date 2026-02-01 --end-date 2026-02-05 --season 2025 --month 2`

- Dataset build command:
  - `uv run python -m projections.cli.retrain_minutes_v1_recency build-dataset --run-id minutes_v1_recency_h35_dataset_only_20260207T111500Z --train-start-date 2025-02-01 --train-end-date 2026-01-31 --cal-start-date 2026-02-01 --cal-end-date 2026-02-05`

- Smoke run:
  - `uv run python -m projections.cli.retrain_minutes_v1_recency run --run-id minutes_v1_recency_h35_smoke_20260207T101500Z --train-start-date 2026-01-31 --train-end-date 2026-01-31 --cal-start-date 2026-02-01 --cal-end-date 2026-02-02`

- Full run:
  - `uv run python -m projections.cli.retrain_minutes_v1_recency run --run-id minutes_v1_recency_h35_20260207T110500Z --train-start-date 2025-02-01 --train-end-date 2026-01-31 --cal-start-date 2026-02-01 --cal-end-date 2026-02-05 --allow-guard-failure`
