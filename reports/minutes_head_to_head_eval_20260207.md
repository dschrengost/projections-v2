# Minutes Bundle Head-to-Head Eval (2026-02-07)

- Eval run id: `minutes_head_to_head_20260207`
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
| brier_play_prob | 0.191569 | 0.285862 | 0.094293 |
| false_active_rate_p_ge_0_5 | 0.206948 | 0.376201 | 0.169254 |
| false_inactive_rate_p_le_0_2 | 0.001478 | 0.000000 | -0.001478 |
| mae_p50_conditional | 8.079947 | 8.051194 | -0.028754 |
| bench_smear_proxy | 0.218034 | 0.185514 | -0.032520 |
| p10_coverage_leq | 0.394678 | 0.447154 | 0.052476 |
| p90_coverage_leq | 0.907613 | 0.932742 | 0.025129 |

## pre_deadline_stability

- Requested window: `2026-01-15` .. `2026-01-31`
- Effective window: `2026-01-26` .. `2026-01-31`
- Eval rows: `212`

| metric | current | retrain | delta (retrain-current) |
| --- | ---: | ---: | ---: |
| rows | 212 | 212 | 0.000000 |
| positive_rows | 118 | 118 | 0.000000 |
| brier_play_prob | 0.250420 | 0.342281 | 0.091860 |
| false_active_rate_p_ge_0_5 | 0.443396 | 0.443396 | 0.000000 |
| false_inactive_rate_p_le_0_2 | 0.000000 | 0.000000 | 0.000000 |
| mae_p50_conditional | 10.704334 | 9.085647 | -1.618687 |
| bench_smear_proxy | 0.443396 | 0.429245 | -0.014151 |
| p10_coverage_leq | 0.443396 | 0.457547 | 0.014151 |
| p90_coverage_leq | 0.900943 | 0.976415 | 0.075472 |
