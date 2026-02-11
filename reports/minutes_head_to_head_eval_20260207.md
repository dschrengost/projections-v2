# Minutes Bundle Head-to-Head Eval (2026-02-07)

- Eval run id: `occupancy_sparse_tuned_candidate_20260210T002122Z`
- Current bundle: `/home/daniel/projects/projections-v2/artifacts/minutes_lgbm/minutes_v1_safe_starter_20260127_dnp_playprob_dedicated`
- Retrain bundle: `/home/daniel/projections-data/artifacts/minutes_lgbm/minutes_v1_recency_h35_20260207T110500Z`

## deadline_chaos

- Requested window: `2026-02-01` .. `2026-02-05`
- Effective window: `2026-02-01` .. `2026-02-05`
- Eval rows: `1353`

| metric | current | retrain | delta (retrain-current) |
| --- | ---: | ---: | ---: |
| rows | 1353 | 1353 | 0.000000 |
| positive_rows | 844 | 844 | 0.000000 |
| brier_play_prob | 0.148791 | 0.132089 | -0.016702 |
| false_active_rate_p_ge_0_5 | 0.070953 | 0.096822 | 0.025868 |
| false_inactive_rate_p_le_0_2 | 0.030303 | 0.037694 | 0.007391 |
| mae_p50_conditional | 8.154554 | 6.359971 | -1.794583 |
| bench_smear_proxy | 0.210643 | 0.187731 | -0.022912 |
| p10_coverage_leq | 0.411678 | 0.467110 | 0.055432 |
| p90_coverage_leq | 0.894309 | 0.977827 | 0.083518 |

| metric | retrain | retrain_occupancy_v0 | delta (occupancy-retrain) |
| --- | ---: | ---: | ---: |
| rows | 1353 | 1353 | 0.000000 |
| positive_rows | 844 | 844 | 0.000000 |
| brier_play_prob | 0.132089 | 0.097880 | -0.034209 |
| false_active_rate_p_ge_0_5 | 0.096822 | 0.053954 | -0.042868 |
| false_inactive_rate_p_le_0_2 | 0.037694 | 0.056911 | 0.019217 |
| mae_p50_conditional | 6.359971 | 7.281063 | 0.921093 |
| bench_smear_proxy | 0.187731 | 0.049520 | -0.138211 |
| p10_coverage_leq | 0.467110 | 0.463415 | -0.003695 |
| p90_coverage_leq | 0.977827 | 0.949741 | -0.028086 |

## pre_deadline_stability

- Requested window: `2026-01-15` .. `2026-01-31`
- Effective window: `2026-01-15` .. `2026-01-24`
- Eval rows: `2635`

| metric | current | retrain | delta (retrain-current) |
| --- | ---: | ---: | ---: |
| rows | 2635 | 2635 | 0.000000 |
| positive_rows | 1597 | 1597 | 0.000000 |
| brier_play_prob | 0.163873 | 0.115784 | -0.048089 |
| false_active_rate_p_ge_0_5 | 0.185579 | 0.133207 | -0.052372 |
| false_inactive_rate_p_le_0_2 | 0.000380 | 0.000000 | -0.000380 |
| mae_p50_conditional | 6.562460 | 5.572484 | -0.989976 |
| bench_smear_proxy | 0.310057 | 0.127514 | -0.182543 |
| p10_coverage_leq | 0.432258 | 0.481594 | 0.049336 |
| p90_coverage_leq | 0.924478 | 0.984061 | 0.059583 |

| metric | retrain | retrain_occupancy_v0 | delta (occupancy-retrain) |
| --- | ---: | ---: | ---: |
| rows | 2635 | 2635 | 0.000000 |
| positive_rows | 1597 | 1597 | 0.000000 |
| brier_play_prob | 0.115784 | 0.076156 | -0.039628 |
| false_active_rate_p_ge_0_5 | 0.133207 | 0.046300 | -0.086907 |
| false_inactive_rate_p_le_0_2 | 0.000000 | 0.037951 | 0.037951 |
| mae_p50_conditional | 5.572484 | 5.653516 | 0.081032 |
| bench_smear_proxy | 0.127514 | 0.046679 | -0.080835 |
| p10_coverage_leq | 0.481594 | 0.455787 | -0.025806 |
| p90_coverage_leq | 0.984061 | 0.978748 | -0.005313 |
