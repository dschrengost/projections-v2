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
