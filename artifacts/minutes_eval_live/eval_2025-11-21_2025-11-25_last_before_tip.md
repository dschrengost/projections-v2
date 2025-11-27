# Minutes Live Evaluation

- Date window: 2025-11-21 → 2025-11-25
- Snapshot mode: last_before_tip
- Rows evaluated: 603
- Games evaluated: 23
- Artifact root: /home/daniel/projects/projections-v2/artifacts/minutes_eval_live
- Data root: /home/daniel/projections-data

## Snapshot Coverage

- Games in window: 37
- Games with snapshots: 23
- Games skipped: 14

| game_id | game_date | reason |
| ---: | --- | --- |
| 22500048 | 2025-11-21 | no_logs |
| 22500049 | 2025-11-21 | no_logs |
| 22500050 | 2025-11-21 | no_logs |
| 22500051 | 2025-11-21 | no_logs |
| 22500052 | 2025-11-21 | no_logs |
| 22500053 | 2025-11-21 | no_logs |
| 22500054 | 2025-11-21 | no_logs |
| 22500055 | 2025-11-21 | no_logs |
| 22500056 | 2025-11-21 | no_logs |
| 22500268 | 2025-11-22 | no_snapshot_before_tip |
| 22500269 | 2025-11-22 | no_snapshot_before_tip |
| 22500270 | 2025-11-22 | no_snapshot_before_tip |
| 22500271 | 2025-11-22 | no_snapshot_before_tip |
| 22500272 | 2025-11-22 | no_snapshot_before_tip |

## Overall Metrics

| metric | value |
| --- | ---: |
| mae_minutes | 5.755 |
| rmse_minutes | 7.788 |
| smape_minutes | 0.672 |
| coverage_p10_p90 | 0.328 |
| under_rate_p10 | 0.272 |
| over_rate_p90 | 0.400 |

### Conditional vs Unconditional Coverage

| metric | value |
| --- | ---: |
| coverage_p10_p90 | 0.328 |
| under_rate_p10 | 0.272 |
| over_rate_p90 | 0.400 |
| coverage_p10_p90_cond | 0.394 |
| under_rate_p10_cond | 0.127 |
| over_rate_p90_cond | 0.479 |

### Starter vs bench

| bucket | rows | mae | rmse | coverage | under | over |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| starter | 75 | 4.59 | 6.01 | 0.533 | 0.307 | 0.160 |
| bench | 528 | 5.92 | 8.01 | 0.299 | 0.267 | 0.434 |

### Spread buckets

| bucket | rows | mae | rmse | coverage | under | over |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| <=3 | 127 | 5.23 | 7.17 | 0.260 | 0.213 | 0.528 |
| 3-8 | 235 | 6.54 | 8.94 | 0.315 | 0.298 | 0.387 |
| >8 | 241 | 5.26 | 6.84 | 0.378 | 0.278 | 0.344 |

### Minutes projection buckets

| bucket | rows | mae | rmse | coverage | under | over |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| <20 | 350 | 6.24 | 8.58 | 0.306 | 0.337 | 0.357 |
| 20-30 | 175 | 5.56 | 7.07 | 0.314 | 0.183 | 0.503 |
| >30 | 78 | 4.04 | 5.11 | 0.462 | 0.179 | 0.359 |

### DFS Rotation Slices

| bucket | rows | mae | rmse | coverage | coverage_cond | under | over | under_cond | over_cond |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rotation_all | 250 | 4.88 | 6.08 | 0.364 | 0.364 | 0.172 | 0.464 | 0.172 | 0.464 |
| rotation_starters_30_plus | 47 | 4.30 | 5.59 | 0.553 | 0.553 | 0.277 | 0.170 | 0.277 | 0.170 |
| rotation_mid_minutes | 172 | 5.26 | 6.47 | 0.320 | 0.320 | 0.169 | 0.512 | 0.169 | 0.512 |

### Status Buckets (normalized)

| bucket | rows | mae | rmse | coverage | coverage_cond | under | over | under_cond | over_cond |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CLEAN | 603 | 5.75 | 7.79 | 0.328 | 0.394 | 0.272 | 0.400 | 0.127 | 0.479 |

### Injury Return Slices

| bucket | rows | mae | rmse | coverage | coverage_cond | under | over | under_cond | over_cond |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| non_injury_return | 503 | 6.07 | 8.00 | 0.394 | 0.394 | 0.127 | 0.479 | 0.127 | 0.479 |
