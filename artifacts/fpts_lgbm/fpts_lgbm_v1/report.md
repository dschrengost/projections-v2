# FPTS per Minute LightGBM — fpts_lgbm_v1

## Model vs Baseline

| Split | Rows | Baseline MAE/min | Model MAE/min | Δ MAE/min | Baseline MAE FPTS | Model MAE FPTS | Δ MAE FPTS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 965 | 0.3440 | 0.0123 | 0.3317 | 6.317 | 0.114 | 6.204 |
| cal | 215 | 0.3199 | 0.0210 | 0.2988 | 6.313 | 0.395 | 5.918 |
| val | 206 | 0.3053 | 0.0168 | 0.2885 | 6.147 | 0.380 | 5.767 |

Model beats the baseline on val MAE by 5.767 DK points.
**Warning:** `minutes_source="actual"` leaks realized minutes and should only be used for experiments.

## Bucket Summaries
### Train

| Bucket | Rows | Baseline MAE/min | Model MAE/min | Δ MAE/min | Baseline MAE FPTS | Model MAE FPTS | Δ MAE FPTS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bench | 965 | 0.3440 | 0.0123 | 0.3317 | 6.317 | 0.114 | 6.204 |
| high_minutes | 349 | 0.2758 | 0.0038 | 0.2720 | 9.441 | 0.131 | 9.310 |
| low_minutes | 616 | 0.3827 | 0.0172 | 0.3655 | 4.547 | 0.104 | 4.444 |
| favorites | 137 | 0.3496 | 0.0192 | 0.3304 | 6.281 | 0.118 | 6.163 |
| underdogs | 140 | 0.3480 | 0.0087 | 0.3392 | 6.805 | 0.149 | 6.656 |
| full_strength | 965 | 0.3440 | 0.0123 | 0.3317 | 6.317 | 0.114 | 6.204 |

### Cal

| Bucket | Rows | Baseline MAE/min | Model MAE/min | Δ MAE/min | Baseline MAE FPTS | Model MAE FPTS | Δ MAE FPTS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bench | 215 | 0.3199 | 0.0210 | 0.2988 | 6.313 | 0.395 | 5.918 |
| high_minutes | 81 | 0.2912 | 0.0197 | 0.2714 | 9.537 | 0.663 | 8.874 |
| low_minutes | 134 | 0.3372 | 0.0218 | 0.3154 | 4.365 | 0.233 | 4.132 |
| favorites | 63 | 0.3348 | 0.0161 | 0.3187 | 6.710 | 0.378 | 6.332 |
| underdogs | 64 | 0.3646 | 0.0364 | 0.3283 | 6.532 | 0.560 | 5.972 |
| full_strength | 215 | 0.3199 | 0.0210 | 0.2988 | 6.313 | 0.395 | 5.918 |

### Val

| Bucket | Rows | Baseline MAE/min | Model MAE/min | Δ MAE/min | Baseline MAE FPTS | Model MAE FPTS | Δ MAE FPTS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bench | 206 | 0.3053 | 0.0168 | 0.2885 | 6.147 | 0.380 | 5.767 |
| high_minutes | 82 | 0.2403 | 0.0164 | 0.2239 | 8.381 | 0.596 | 7.785 |
| low_minutes | 124 | 0.3483 | 0.0171 | 0.3312 | 4.670 | 0.237 | 4.433 |
| favorites | 38 | 0.2443 | 0.0170 | 0.2273 | 5.303 | 0.385 | 4.918 |
| underdogs | 42 | 0.3525 | 0.0144 | 0.3380 | 7.234 | 0.296 | 6.938 |
| full_strength | 206 | 0.3053 | 0.0168 | 0.2885 | 6.147 | 0.380 | 5.767 |
