# Dual Minutes Models (Early + Late)

This repo supports a two-model minutes system:

- **Early** model: trained on earlier-in-day snapshots (higher uncertainty)
- **Late** model: trained on near-lock snapshots (highest accuracy priority)

## Training (dual)

### 0) Build a snapshot-rich labeled parquet (recommended)

Dual training requires a dataset with **multiple snapshots per game/player** (many `feature_as_of_ts` values) plus the final `minutes` label. If you don’t already have one, build it from live feature runs + gold labels:

```bash
uv run python -m projections.cli.build_minutes_snapshot_dataset build \
  --start-date 2025-10-22 \
  --end-date 2025-12-03
```

This writes `.../training/snapshots_minutes_v1/run=<timestamp>/features.parquet` plus a `manifest.json`.

### 1) Train early + late bundles

The dual trainer builds a horizonized dataset, assigns grouped splits by `game_id` using `tip_ts`, then trains two LightGBM bundles:

```bash
uv run python -m projections.cli.train_minutes_dual \
  --features-path <PARQUET_WITH_MULTIPLE_SNAPSHOTS_PER_GAME> \
  --early-run-id <RUN_ID_EARLY> \
  --late-run-id <RUN_ID_LATE> \
  --train-end 2025-02-28T23:59:59 \
  --cal-end 2025-04-30T23:59:59 \
  --val-end 2025-11-14T23:59:59
```

Optional one-shot:

```bash
uv run python -m projections.cli.build_minutes_snapshot_dataset train-dual \
  --start-date 2025-10-22 \
  --end-date 2025-12-03 \
  --early-run-id <RUN_ID_EARLY> \
  --late-run-id <RUN_ID_LATE> \
  --train-end 2025-02-28T23:59:59 \
  --cal-end 2025-04-30T23:59:59 \
  --val-end 2025-11-14T23:59:59
```

Outputs:
- Horizon datasets under `data/training/horizons_minutes_v1/run=<timestamp>/`
- Model bundles under `artifacts/minutes_lgbm/<run_id>/`

Notes:
- Horizon rows are selected by `snapshot_ts = max(snapshot_ts) <= tip_ts - horizon_min`.
- The trainer computes `time_to_tip_min` and an `odds_missing` indicator at inference time too, so models can include them as features without changing the base feature schema.
- Splits are grouped by `game_id` (all horizons for the same game stay in the same split).

## Production config

Legacy single-bundle config still works:

```json
{
  "mode": "single",
  "bundle_dir": "artifacts/minutes_lgbm/<run_id>",
  "run_id": "<run_id>"
}
```

Dual mode is supported via `config/minutes_current_run.json` (see `config/minutes_current_run_dual_example.json`):

```json
{
  "mode": "dual",
  "early_run_id": "<early_run_id>",
  "late_run_id": "<late_run_id>",
  "late_threshold_min": 60,
  "blend_band_min": 30
}
```

## Inference routing

At scoring time we compute:

- `time_to_tip_min = (tip_ts - feature_as_of_ts) / 60`

Routing:
- If `time_to_tip_min <= late_threshold_min`: use **late** model
- Else: use **early** model
- If `blend_band_min > 0`, blend in `(late_threshold_min, late_threshold_min + blend_band_min)`:
  `pred = w * late + (1 - w) * early`, where `w` increases as tip approaches.

Debug outputs:
- `minutes_model_used` in `{early, late, blend}`
- `minutes_model_late_weight` in `[0, 1]`
