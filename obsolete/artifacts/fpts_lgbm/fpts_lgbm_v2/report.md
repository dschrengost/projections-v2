# FPTS per Minute LightGBM — fpts_lgbm_v2

## Model vs Baseline

| Split | Rows | Baseline MAE/min | Model MAE/min | Δ MAE/min | Baseline MAE FPTS | Model MAE FPTS | Δ MAE FPTS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 965 | 0.3440 | 0.1374 | 0.2067 | 6.317 | 2.308 | 4.009 |
| cal | 215 | 0.3199 | 0.3146 | 0.0053 | 6.313 | 5.977 | 0.336 |
| val | 206 | 0.3053 | 0.2902 | 0.0151 | 6.147 | 6.358 | -0.211 |

Model trails the baseline on val MAE by 0.211 DK points.
**Warning:** `minutes_source="actual"` leaks realized minutes and should only be used for experiments.

## Bucket Summaries
### Train

| Bucket | Rows | Baseline MAE/min | Model MAE/min | Δ MAE/min | Baseline MAE FPTS | Model MAE FPTS | Δ MAE FPTS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bench | 965 | 0.3440 | 0.1374 | 0.2067 | 6.317 | 2.308 | 4.009 |
| high_minutes | 349 | 0.2758 | 0.0971 | 0.1786 | 9.441 | 3.359 | 6.083 |
| low_minutes | 616 | 0.3827 | 0.1602 | 0.2225 | 4.547 | 1.713 | 2.834 |
| favorites | 137 | 0.3496 | 0.1242 | 0.2254 | 6.281 | 2.000 | 4.280 |
| underdogs | 140 | 0.3480 | 0.1129 | 0.2351 | 6.805 | 2.243 | 4.562 |
| full_strength | 965 | 0.3440 | 0.1374 | 0.2067 | 6.317 | 2.308 | 4.009 |

### Cal

| Bucket | Rows | Baseline MAE/min | Model MAE/min | Δ MAE/min | Baseline MAE FPTS | Model MAE FPTS | Δ MAE FPTS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bench | 215 | 0.3199 | 0.3146 | 0.0053 | 6.313 | 5.977 | 0.336 |
| high_minutes | 81 | 0.2912 | 0.2642 | 0.0270 | 9.537 | 8.653 | 0.884 |
| low_minutes | 134 | 0.3372 | 0.3451 | -0.0078 | 4.365 | 4.359 | 0.005 |
| favorites | 63 | 0.3348 | 0.3318 | 0.0030 | 6.710 | 6.310 | 0.400 |
| underdogs | 64 | 0.3646 | 0.3472 | 0.0175 | 6.532 | 5.830 | 0.702 |
| full_strength | 215 | 0.3199 | 0.3146 | 0.0053 | 6.313 | 5.977 | 0.336 |

### Val

| Bucket | Rows | Baseline MAE/min | Model MAE/min | Δ MAE/min | Baseline MAE FPTS | Model MAE FPTS | Δ MAE FPTS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bench | 206 | 0.3053 | 0.2902 | 0.0151 | 6.147 | 6.358 | -0.211 |
| high_minutes | 82 | 0.2403 | 0.2573 | -0.0170 | 8.381 | 8.956 | -0.575 |
| low_minutes | 124 | 0.3483 | 0.3120 | 0.0363 | 4.670 | 4.640 | 0.029 |
| favorites | 38 | 0.2443 | 0.2517 | -0.0074 | 5.303 | 5.856 | -0.553 |
| underdogs | 42 | 0.3525 | 0.3389 | 0.0135 | 7.234 | 7.317 | -0.083 |
| full_strength | 206 | 0.3053 | 0.2902 | 0.0151 | 6.147 | 6.358 | -0.211 |
