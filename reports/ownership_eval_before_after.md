# Ownership Model Eval — Before/After (Fixed Slice)

- Fixed validation slice: `config/ownership_eval_slice.json` (12 DK slates; `2025-11-24..2025-12-06`)
- Target sum reference: DK classic `800%`

## Before (Production Run: dk_only_v4)

- Slice: `dk_2025-11-24_to_2025-12-06_fixed` (n_slates=12, n_rows=1205, dates=2025-11-24..2025-12-06)
- Target sum: 800.0%

### Error
- Raw MAE/RMSE: 3.7133 / 6.7422 (pct points)
- Raw MAE/RMSE (logit): 0.6013 / 0.8364
- Scaled-to-sum MAE/RMSE: 3.4083 / 5.9944 (pct points)

### Ranking
- Spearman pooled raw/scaled: 0.8456 / 0.8493
- Spearman per-slate mean±std raw/scaled: 0.8640 ± 0.0432 / 0.8640 ± 0.0432
- Spearman top10/top20 actual raw/scaled: 0.4232 / 0.6474 / 0.4232 / 0.6474
- Recall@10/20 raw/scaled: 0.7500 / 0.7625 / 0.7500 / 0.7625

### Calibration
- ECE raw/scaled: 1.5974 / 1.0414 (pct points)
- Top10/top20 mean bias raw/scaled: -1.5111 / 0.4704 / -3.7655 / -1.2358 (pct points)
- Tail <=5% / <=1% mean bias raw/scaled: 0.9202 / 0.2833 / 0.7290 / 0.2876 (pct points)

### Sum Constraint
- Actual sum mean±std: 752.2860 ± 107.7921 (min=459.4788, max=847.1646)
- Actual mean |sum - target|: 67.5589
- Raw sum(pred) mean±std: 857.6196 ± 262.9349 (min=480.7504, max=1488.5712)
- Raw mean |sum - target|: 190.7866
- Raw max(pred) mean/p95: 54.9260 / 78.1340
- Raw count pred>60 / pred>70 / pred>100: 9 / 3 / 0
- Scaled sum(pred) mean±std: 800.0000 ± 0.0000 (min=800.0000, max=800.0000)
- Scaled mean |sum - target|: 0.0000
- Scaled max(pred) mean/p95: 51.5951 / 59.5581
- Scaled count pred>60 / pred>70 / pred>100: 1 / 0 / 0

## After (TBD)

- Pending Phase 2+ changes. This section will contain the same metrics on the same fixed slice.
