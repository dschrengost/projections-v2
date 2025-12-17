# Ownership Model Eval — Before/After (Fixed Slice)

- Fixed validation slice: `config/ownership_eval_slice.json` (10 DK slates; `2025-11-24..2025-12-06`)
- Target sum reference: DK classic `800%`

## Before (Production Run: dk_only_v4)

- Slice: `dk_2025-11-24_to_2025-12-06_fixed_supported` (n_slates=10, n_rows=1080, dates=2025-11-24..2025-12-06)
- Target sum: 800.0%

### Error
- Raw MAE/RMSE: 3.4034 / 6.1279 (pct points)
- Raw MAE/RMSE (logit): 0.5782 / 0.8029
- Scaled-to-sum MAE/RMSE: 3.1123 / 5.4634 (pct points)

### Ranking
- Spearman pooled raw/scaled: 0.8578 / 0.8596
- Spearman per-slate mean±std raw/scaled: 0.8690 ± 0.0453 / 0.8690 ± 0.0453
- Spearman top10/top20 actual raw/scaled: 0.4921 / 0.6844 / 0.4921 / 0.6844
- Recall@10/20 raw/scaled: 0.7800 / 0.7600 / 0.7800 / 0.7600

### Calibration
- ECE raw/scaled: 1.3883 / 0.7612 (pct points)
- Top10/top20 mean bias raw/scaled: -2.0413 / -0.1578 / -4.8649 / -2.2616 (pct points)
- Tail <=5% / <=1% mean bias raw/scaled: 1.1249 / 0.4127 / 0.9246 / 0.4178 (pct points)

### Sum Constraint
- Actual sum mean±std: 766.3635 ± 113.4787 (min=459.4788, max=847.1646)
- Actual mean |sum - target|: 57.4504
- Raw sum(pred) mean±std: 872.3790 ± 267.7436 (min=480.7504, max=1488.5712)
- Raw mean |sum - target|: 183.7262
- Raw max(pred) mean/p95: 54.6906 / 73.7671
- Raw count pred>60 / pred>70 / pred>100: 6 / 1 / 0
- Scaled sum(pred) mean±std: 800.0000 ± 0.0000 (min=800.0000, max=800.0000)
- Scaled mean |sum - target|: 0.0000
- Scaled max(pred) mean/p95: 50.7427 / 57.8836
- Scaled count pred>60 / pred>70 / pred>100: 0 / 0 / 0

## After (TBD)

- Pending Phase 2+ changes. This section will contain the same metrics on the same fixed slice.
