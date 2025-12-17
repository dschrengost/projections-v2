# Ownership Model Eval — Before/After (Fixed Slice)

- Fixed validation slice: `config/ownership_eval_slice.json` (10 DK slates; `2025-11-24..2025-12-06`)
- Target sum reference: DK classic `800%`

## Legacy Baseline (Production Run: dk_only_v4)

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

## Before (Clean DK Base Baseline: dk_only_v4_cleanbase_seed1337)

- Slice: `dk_2025-11-24_to_2025-12-06_fixed_supported` (n_slates=10, n_rows=1155, dates=2025-11-24..2025-12-06)
- Target sum: 800.0%

### Error
- Raw MAE/RMSE: 2.9839 / 5.4306 (pct points)
- Raw MAE/RMSE (logit): 0.5212 / 0.7206
- Scaled-to-sum MAE/RMSE: 2.7428 / 5.0205 (pct points)

### Ranking
- Spearman pooled raw/scaled: 0.8835 / 0.8925
- Spearman per-slate mean±std raw/scaled: 0.8903 ± 0.0389 / 0.8903 ± 0.0389
- Spearman top10/top20 actual raw/scaled: 0.4655 / 0.6669 / 0.4655 / 0.6669
- Recall@10/20 raw/scaled: 0.7600 / 0.7500 / 0.7600 / 0.7500

### Calibration
- ECE raw/scaled: 0.9520 / 0.4561 (pct points)
- Top10/top20 mean bias raw/scaled: -1.9765 / -1.2879 / -4.1234 / -2.8378 (pct points)
- Tail <=5% / <=1% mean bias raw/scaled: 0.9363 / 0.5044 / 0.7940 / 0.4505 (pct points)

### Sum Constraint
- Actual sum mean±std: 793.8969 ± 4.1716 (min=783.9559, max=798.2284)
- Actual mean |sum - target|: 6.1031
- Raw sum(pred) mean±std: 843.7149 ± 167.5997 (min=585.8941, max=1089.0342)
- Raw mean |sum - target|: 144.7382
- Raw max(pred) mean/p95: 56.6488 / 71.9071
- Raw count pred>60 / pred>70 / pred>100: 5 / 1 / 0
- Scaled sum(pred) mean±std: 800.0000 ± 0.0000 (min=800.0000, max=800.0000)
- Scaled mean |sum - target|: 0.0000
- Scaled max(pred) mean/p95: 54.1435 / 61.2324
- Scaled count pred>60 / pred>70 / pred>100: 1 / 0 / 0

## After (dk_only_v6_logit_chalk5_cleanbase_seed1337)

- Slice: `dk_2025-11-24_to_2025-12-06_fixed_supported` (n_slates=10, n_rows=1155, dates=2025-11-24..2025-12-06)
- Target sum: 800.0%

### Error
- Raw MAE/RMSE: 2.6013 / 4.9357 (pct points)
- Raw MAE/RMSE (logit): 0.4383 / 0.6008
- Scaled-to-sum MAE/RMSE: 2.5151 / 4.6249 (pct points)

### Ranking
- Spearman pooled raw/scaled: 0.9206 / 0.9184
- Spearman per-slate mean±std raw/scaled: 0.9181 ± 0.0199 / 0.9181 ± 0.0199
- Spearman top10/top20 actual raw/scaled: 0.5661 / 0.6650 / 0.5661 / 0.6650
- Recall@10/20 raw/scaled: 0.7500 / 0.7750 / 0.7500 / 0.7750

### Calibration
- ECE raw/scaled: 0.9205 / 0.6195 (pct points)
- Top10/top20 mean bias raw/scaled: -3.3762 / -2.6423 / -2.3695 / -1.9432 (pct points)
- Tail <=5% / <=1% mean bias raw/scaled: 0.4976 / 0.4219 / 0.6119 / 0.4804 (pct points)

### Sum Constraint
- Actual sum mean±std: 793.8969 ± 4.1716 (min=783.9559, max=798.2284)
- Actual mean |sum - target|: 6.1031
- Raw sum(pred) mean±std: 762.7442 ± 144.4675 (min=536.7044, max=969.8067)
- Raw mean |sum - target|: 123.7524
- Raw max(pred) mean/p95: 56.2224 / 70.8775
- Raw count pred>60 / pred>70 / pred>100: 7 / 1 / 0
- Scaled sum(pred) mean±std: 800.0000 ± 0.0000 (min=800.0000, max=800.0000)
- Scaled mean |sum - target|: 0.0000
- Scaled max(pred) mean/p95: 59.5921 / 77.2449
- Scaled count pred>60 / pred>70 / pred>100: 5 / 3 / 0
