# Rates Retrain Window + Slice Eval (2026-02-07)

- Current run: `rates_v1_stage5_fta_20260107_231545`
- Retrain run: `rates_v1_stage5_recency_h75_20260207_182546`
- Train window: 2025-01-10 .. 2026-01-09 (cutoff 2026-01-10)
- Cal window: 2026-01-10 .. 2026-01-23
- Val window: 2026-01-24 .. 2026-02-06
- Recency weighting: half-life=75 days, floor=0.0

## normal_pre_deadline
- Rows scored: 2225 (raw: 2236)
- Avg MAE current: 0.086848
- Avg MAE retrain: 0.084993
- Delta (retrain-current): -0.001855

## chaos_deadline
- Rows scored: 1338 (raw: 1351)
- Avg MAE current: 0.085857
- Avg MAE retrain: 0.084354
- Delta (retrain-current): -0.001502

## parity_retrain_followup
- Current run: `rates_v1_stage5_recency_h75_20260207_182546`
- Retrain run: `rates_v1_stage5_recency_h75_20260207_184701`
- Training base refreshed with nullable tracking fields (no pre-fill), then retrained with same window/config.
- New bundle persists train-time tracking imputation in `meta.json` under `preprocess.tracking_fill_values`.
- Detailed head-to-head JSON: `/home/daniel/projections-data/artifacts/rates_v1/runs/rates_v1_stage5_recency_h75_20260207_184701/head_to_head_eval_normal_vs_chaos.json`

### normal_pre_deadline (2026-01-10..2026-01-24)
- Rows scored: 2237 (raw: 2237)
- Avg MAE current: 0.086449
- Avg MAE retrain: 0.084957
- Delta (retrain-current): -0.001492

### chaos_deadline (2026-01-29..2026-02-06)
- Rows scored: 1371 (raw: 1371)
- Avg MAE current: 0.085650
- Avg MAE retrain: 0.084665
- Delta (retrain-current): -0.000986

### per-head check
- 17/24 head-slice MAE comparisons improved.
- Remaining 7/24 were small regressions (largest +0.000868 MAE).
