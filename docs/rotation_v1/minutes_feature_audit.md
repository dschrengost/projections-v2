# Minutes V1 Feature Audit (minutes_v1_safe_starter_20251214)

## Dataset discovery
- Bundle: `artifacts/minutes_lgbm/minutes_v1_safe_starter_20251214`
- Feature columns manifest: `artifacts/minutes_lgbm/minutes_v1_safe_starter_20251214/feature_columns.json`
- Selected training dataset dir: `/home/daniel/projections-data/training/datasets/v1_enriched_20251214`
- Features parquet: `/home/daniel/projections-data/training/datasets/v1_enriched_20251214/features.parquet`
- Labels parquet: `/home/daniel/projections-data/training/datasets/v1_enriched_20251214/labels.parquet`
- Rows loaded: 15,643
- Manifest created_at: 2025-12-14T05:29:21.848169+00:00
- Manifest date_range: {'end': '2025-12-12', 'start': '2022-10-01'}
- Snapshot type: pretip

## Column inventory
- Feature column count: 58
- Grouped by prefix:
  - `arch` (6): arch_delta_max_role, arch_delta_min_role, arch_delta_same_pos, arch_delta_sum, arch_missing_same_pos_count, arch_missing_total_count
  - `available` (3): available_B, available_G, available_W
  - `away` (1): away_team_id
  - `blowout` (2): blowout_index, blowout_risk_score
  - `close` (1): close_game_score
  - `days` (2): days_since_last, days_since_return
  - `depth` (1): depth_same_pos_active
  - `games` (1): games_since_return
  - `home` (2): home_flag, home_team_id
  - `injury` (1): injury_snapshot_missing
  - `is` (8): is_3in4, is_4in6, is_b2b, is_confirmed_starter, is_out, is_prob, is_projected_starter, is_q
  - `min` (3): min_last1, min_last3, min_last5
  - `opp` (2): opp_def_rtg_szn, opp_pace_szn
  - `ramp` (1): ramp_flag
  - `recent` (1): recent_start_pct_10
  - `restriction` (1): restriction_flag
  - `role` (1): role_change_rate_10g
  - `roll` (4): roll_iqr_5, roll_mean_10, roll_mean_3, roll_mean_5
  - `rotation` (1): rotation_minutes_std_5g
  - `same` (1): same_archetype_overlap
  - `season` (1): season_phase
  - `spread` (1): spread_home
  - `starter` (2): starter_flag, starter_prev_game_asof
  - `sum` (1): sum_min_7d
  - `team` (4): team_def_rtg_szn, team_minutes_dispersion_prior, team_off_rtg_szn, team_pace_szn
  - `total` (1): total
  - `vac` (4): vac_min_big_szn, vac_min_guard_szn, vac_min_szn, vac_min_wing_szn
  - `z` (1): z_vs_10

## Missingness (top 30)
- column | missing_pct
  - `spread_home` | 70.22%
  - `total` | 70.22%
  - `team_minutes_dispersion_prior` | 4.25%
  - `roll_mean_3` | 3.92%
  - `min_last1` | 3.92%
  - `roll_mean_5` | 3.92%
  - `min_last3` | 3.92%
  - `min_last5` | 3.92%
  - `days_since_last` | 3.92%
  - `roll_mean_10` | 3.92%
  - `games_since_return` | 0.47%
  - `days_since_return` | 0.47%
  - `available_G` | 0.00%
  - `available_W` | 0.00%
  - `close_game_score` | 0.00%
  - `blowout_risk_score` | 0.00%
  - `depth_same_pos_active` | 0.00%
  - `blowout_index` | 0.00%
  - `away_team_id` | 0.00%
  - `arch_delta_same_pos` | 0.00%
  - `arch_missing_same_pos_count` | 0.00%
  - `arch_missing_total_count` | 0.00%
  - `available_B` | 0.00%
  - `arch_delta_sum` | 0.00%
  - `arch_delta_max_role` | 0.00%
  - `arch_delta_min_role` | 0.00%
  - `is_prob` | 0.00%
  - `is_out` | 0.00%
  - `is_confirmed_starter` | 0.00%
  - `is_b2b` | 0.00%

## Low-cardinality columns (<= 10 unique)
- `arch_delta_max_role`: 1
- `arch_delta_min_role`: 1
- `arch_delta_same_pos`: 1
- `arch_delta_sum`: 1
- `arch_missing_same_pos_count`: 1
- `arch_missing_total_count`: 1
- `role_change_rate_10g`: 1
- `rotation_minutes_std_5g`: 1
- `season_phase`: 1
- `starter_flag`: 1
- `days_since_return`: 2
- `games_since_return`: 2
- `home_flag`: 2
- `injury_snapshot_missing`: 2
- `is_3in4`: 2
- `is_4in6`: 2
- `is_b2b`: 2
- `is_confirmed_starter`: 2
- `is_out`: 2
- `is_prob`: 2
- `is_projected_starter`: 2
- `is_q`: 2
- `ramp_flag`: 2
- `recent_start_pct_10`: 2
- `restriction_flag`: 2
- `same_archetype_overlap`: 2
- `starter_prev_game_asof`: 2
- `available_B`: 7
- `available_G`: 8
- `available_W`: 9

## Constant / near-constant columns
- Near-constant threshold: max_freq >= 0.995
- Constant columns:
  - `arch_delta_max_role`
  - `arch_delta_min_role`
  - `arch_delta_same_pos`
  - `arch_delta_sum`
  - `arch_missing_same_pos_count`
  - `arch_missing_total_count`
  - `role_change_rate_10g`
  - `rotation_minutes_std_5g`
  - `season_phase`
  - `starter_flag`
- Near-constant columns:
  - `arch_delta_max_role` (max_freq=1.000)
  - `arch_delta_min_role` (max_freq=1.000)
  - `arch_delta_same_pos` (max_freq=1.000)
  - `arch_delta_sum` (max_freq=1.000)
  - `arch_missing_same_pos_count` (max_freq=1.000)
  - `arch_missing_total_count` (max_freq=1.000)
  - `days_since_return` (max_freq=0.995)
  - `games_since_return` (max_freq=0.995)
  - `is_prob` (max_freq=0.999)
  - `is_q` (max_freq=0.999)
  - `restriction_flag` (max_freq=0.999)
  - `role_change_rate_10g` (max_freq=1.000)
  - `rotation_minutes_std_5g` (max_freq=1.000)
  - `season_phase` (max_freq=1.000)
  - `starter_flag` (max_freq=1.000)

## Correlation clusters (abs(corr) > 0.995)
- Keep `away_team_id`; cluster: `away_team_id`, `home_team_id`
- Keep `roll_mean_3`; cluster: `min_last3`, `roll_mean_3`
- Keep `roll_mean_5`; cluster: `min_last5`, `roll_mean_5`
- Keep `recent_start_pct_10`; cluster: `recent_start_pct_10`, `starter_prev_game_asof`

## Starter / injury feature sanity
- Starter flag missing rates:
  - `is_confirmed_starter`: 0.00%
  - `is_projected_starter`: 0.00%
  - `starter_flag`: 0.00%
- Injury flag non-zero rates:
  - `is_out`: 22.33%
  - `is_prob`: 0.05%
  - `is_q`: 0.12%

## Recommended pruning
### DROP_NOW
- `arch_delta_max_role`
- `arch_delta_min_role`
- `arch_delta_same_pos`
- `arch_delta_sum`
- `arch_missing_same_pos_count`
- `arch_missing_total_count`
- `days_since_return`
- `games_since_return`
- `home_team_id`
- `is_prob`
- `is_q`
- `min_last3`
- `min_last5`
- `restriction_flag`
- `role_change_rate_10g`
- `rotation_minutes_std_5g`
- `season_phase`
- `starter_flag`
- `starter_prev_game_asof`

### KEEP_CORE
- `away_team_id`
- `depth_same_pos_active`
- `home_flag`
- `injury_snapshot_missing`
- `is_confirmed_starter`
- `is_projected_starter`
- `min_last1`
- `opp_def_rtg_szn`
- `opp_pace_szn`
- `roll_iqr_5`
- `roll_mean_10`
- `roll_mean_3`
- `roll_mean_5`
- `spread_home`
- `sum_min_7d`
- `team_def_rtg_szn`
- `team_minutes_dispersion_prior`
- `team_off_rtg_szn`
- `team_pace_szn`
- `vac_min_big_szn`
- `vac_min_guard_szn`
- `vac_min_szn`
- `vac_min_wing_szn`

### KEEP_TEMP
- `available_B`
- `available_G`
- `available_W`
- `blowout_index`
- `blowout_risk_score`
- `close_game_score`
- `days_since_last`
- `is_3in4`
- `is_4in6`
- `is_b2b`
- `is_out`
- `ramp_flag`
- `recent_start_pct_10`
- `same_archetype_overlap`
- `total`
- `z_vs_10`

## Notes
- Minutes_v1 training uses the feature allowlist from `feature_columns.json`; target is `minutes`.
- No MLflow metadata for this run was found under `mlruns/`; model artifacts live in `artifacts/minutes_lgbm`.