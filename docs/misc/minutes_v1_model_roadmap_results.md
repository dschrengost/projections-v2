## Minutes V1 Rebuilt Data Experiments (2025-11-21)

Summary of the training-window experiments requested in `docs/minutes_v1_model_roadmap.md` using rebuilt gold features at `/home/daniel/projections-data/gold/features_minutes_v1` (feature hash `ac92bebab5ccfcd74e7a788d859f382bae61f4021f4cd9dc099d3889f79e10ff`). Conformal guardrails were left in warn mode (`--allow-guard-failure`) so runs could complete even when coverage drifted.

### Data coverage and deviations
- Season coverage for gold features: 2022-10-18→2023-04-08, 2023-10-24→2024-04-14, 2024-10-22→2025-04-13, 2025-10-21→2025-11-20.
- The roadmap cal window for the “very recent” run (2025-04-14→2025-05-31) has zero rows; the run below uses train_end=2025-03-31 and cal=2025-04-01→2025-04-13 instead.
- The “recent-only” run keeps cal=2024-05-01→2025-05-31, so train was capped at 2024-04-30 to avoid overlap.

### Training windows executed
| run_id | train window | cal window | val window | val_rows |
| --- | --- | --- | --- | --- |
| v1_full_calibration | 2022-10-01 → 2024-04-30 | 2024-05-01 → 2025-05-31 | 2025-06-01 → 2025-11-18 | 4,867 |
| v1_rebuilt_prodconfig_20251121 | 2022-10-01 → 2024-04-30 | 2024-05-01 → 2025-05-31 | 2025-06-01 → 2025-11-18 | 3,665 |
| v1_full_history_rebuilt_20251121 | 2022-10-01 → 2025-02-28 | 2025-03-01 → 2025-04-30 | 2025-10-01 → 2025-11-14 | 3,209 |
| v1_recent_history_rebuilt_20251121 | 2023-10-24 → 2024-04-30 | 2024-05-01 → 2025-05-31 | 2025-06-01 → 2025-11-18 | 3,665 |
| v1_very_recent_rebuilt_20251121 | 2024-10-22 → 2025-03-31 | 2025-04-01 → 2025-04-13 | 2025-10-21 → 2025-11-18 | 3,665 |
| **v1_full_history_noplayoff_20251121** | 2022-10-01 → 2025-03-31 | 2025-04-01 → 2025-04-13 | 2025-10-21 → 2025-11-18 | 3,665 |
| **v1_recent_history_noplayoff_20251121** | 2023-10-24 → 2025-03-31 | 2025-04-01 → 2025-04-13 | 2025-10-21 → 2025-11-18 | 3,665 |
| **v1_very_recent_noplayoff_20251121** | 2024-10-22 → 2025-03-31 | 2025-04-01 → 2025-04-13 | 2025-10-21 → 2025-11-18 | 3,665 |

### Validation metrics (minutes target)
| run_id | val_mae | mae_0_10 | mae_10_20 | mae_20_30 | mae_30_plus | p10_cond | p90_cond | winkler_cond |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v1_full_calibration | 6.836 | 10.367 | 5.372 | 4.249 | 5.541 | 0.106 | 0.921 | 22.680 |
| v1_rebuilt_prodconfig_20251121 | 11.895 | 17.733 | 6.127 | 4.144 | 6.862 | 0.119 | 0.902 | 25.626 |
| v1_full_history_rebuilt_20251121 | 11.391 | 16.875 | 5.757 | 4.416 | 5.929 | 0.149 | 0.899 | 24.347 |
| v1_recent_history_rebuilt_20251121 | 11.824 | 17.192 | 6.426 | 4.229 | 7.840 | 0.149 | 0.877 | 27.143 |
| v1_very_recent_rebuilt_20251121 | 11.715 | 17.020 | 6.593 | 4.398 | 7.364 | 0.158 | 0.888 | 26.498 |
| **v1_full_history_noplayoff_20251121** | 11.818 | 17.543 | 6.206 | 4.177 | 6.890 | 0.139 | 0.906 | 25.035 |
| **v1_recent_history_noplayoff_20251121** | 11.738 | 17.051 | 6.614 | 4.435 | 7.347 | 0.158 | 0.894 | 26.099 |
| **v1_very_recent_noplayoff_20251121** | 11.715 | 17.020 | 6.593 | 4.398 | 7.364 | 0.158 | 0.888 | 26.498 |

### Sequential backtest (2025-10-01 → 2025-11-18)
| run_id | p10_overall | p90_overall |
| --- | --- | --- |
| v1_full_calibration | 0.441 | 0.930 |
| v1_rebuilt_prodconfig_20251121 | 0.446 | 0.921 |
| v1_full_history_rebuilt_20251121 | 0.446 | 0.924 |
| v1_recent_history_rebuilt_20251121 | 0.462 | 0.902 |
| v1_very_recent_rebuilt_20251121 | 0.457 | 0.898 |
| **v1_full_history_noplayoff_20251121** | 0.445 | 0.923 |
| **v1_recent_history_noplayoff_20251121** | 0.452 | 0.906 |
| **v1_very_recent_noplayoff_20251121** | 0.457 | 0.898 |

### Notes and next steps
- `v1_full_calibration` remains far better on MAE (6.8 vs 11–12 for rebuilt runs) and has the strongest p90 conditional coverage (0.921). All rebuilt variants under-cover p90 and over-cover p10, with notably worse Winkler on playable rows.
- Dropping 2022 data or narrowing to the most recent season further erodes coverage (p90_cond as low as 0.877) and increases tail width (Winkler 26–27).
- Sequential backtests show p10 coverage around 0.45 for every bundle (including baseline), so rolling calibration remains highly conservative on the lower tail.
- Open question: do we want to rebuild labels/features beyond 2025-04-13 so the planned cal window (2025-04-14 → 2025-05-31) is usable, or keep using the shortened cal window used here?
- Suggested next experiments (playoffs excluded): lower `conformal_k` (100–150) and/or add `injury_snapshot` bucket, try shallower trees (`max_depth=5–6`, ~700–800 trees), and re-run the backtest to see if p10 overcoverage can be reduced without blowing up p90. Current windows stop at 2025-03-31 for training and 2025-04-01→2025-04-13 for calibration to avoid playoffs entirely.
