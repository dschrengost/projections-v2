# Data Contracts

Data schemas, feature contracts, and artifact formats for `projections-v2`.

## Feature Contracts

Feature contracts define required columns and types for ML model inputs. Located in `projections/models/feature_contract.py`.

### Minutes Model Features

The minutes model (`minutes_v1`) requires these feature groups:

| Group | Examples | Source |
|-------|----------|--------|
| Player identity | `player_id`, `player_name`, `team_abbr` | Roster data |
| Game context | `game_date`, `is_home`, `opponent_abbr` | Schedule |
| Historical stats | `l5_minutes_mean`, `l10_minutes_mean` | Boxscores |
| Injury status | `is_out`, `is_gtd`, `is_questionable` | Injury reports |
| Lineup status | `is_starter`, `starter_status` | Daily lineups |
| Team context | `team_pace`, `opponent_pace` | Team stats |

### Validating Features

```python
from projections.models.feature_contract import validate_features

# Check a DataFrame meets contract
validate_features(df, model_name="minutes_v1")
```

## Artifact Formats

### Unified Projections

Primary output artifact with combined predictions:

```
artifacts/unified_projections/game_date=YYYY-MM-DD/
└── projections.parquet
```

Columns:
- `player_id`, `player_name`, `team_abbr`
- `minutes_mean`, `minutes_std`
- `pts_mean`, `reb_mean`, `ast_mean`, ...
- `dk_fpts_mean`, `dk_fpts_std`
- `ownership_proj`

### Worlds Tensor

Monte Carlo simulation outputs:

```
artifacts/sim_v2/worlds_fpts_v2/game_date=YYYY-MM-DD/
├── projections.parquet    # Player projections
├── world_*.parquet        # Per-world FPTS samples
└── index.parquet          # Player index mapping
```

`projections.parquet` includes two semantics for world-aggregated metrics:
- **Conditional-on-playing** (historical default): `dk_fpts_*`, `minutes_sim_*` are computed over only worlds where the player is active.
- **Unconditional (DNP=0)**: `*_uncond` variants (e.g., `dk_fpts_mean_uncond`, `minutes_sim_p50_uncond`) include inactive/DNP worlds as 0 and should be preferred for decision metrics.

### Rotation Priors (Minutes Quantile Priors)

Rotation generation and `rot_eval_v1` can consume a “minutes prior” parquet with (at minimum):

- `game_id` (string)
- `team_id` (int)
- `player_id` (int)
- `minutes_prior` (float; typically p50-ish)
- `play_prob` (float in [0,1])
- Optional quantiles: `minutes_p10`, `minutes_p50`, `minutes_p90`

Important notes:

- `play_prob` in current internal priors is a placeholder (often constant `1.0`) and must not be treated as meaningful availability.
- In-memory only (not required on disk): rotation prior heuristics may add optional columns:
  - `p_ge5_prior_heur`: heuristic `P(minutes >= 5)` derived from `minutes_prior` + quantiles
  - `p_eq0_prior_heur`: heuristic `P(minutes == 0)` / DNP-ish derived from `minutes_prior` + quantiles

### Depth Chart Prior (Inference-Only)

`effective_minutes.parquet` may include RealGM-derived depth-chart fields that are applied at inference time only
(after model scoring/overrides, before world generation). The prior does not add new model heads and is not part of training data.

Derived columns (when available):

- `dc_present` (bool)
- `dc_role` (`starter` | `rotation` | `limited` | `not_listed`)
- `dc_role_priority` (int)
- `dc_order_in_role` (nullable int)
- `dc_ahead_global` (int)
- `dc_is_primary_backup` (bool)
- `dc_snapshot_ts` (timestamp)

Diagnostics are written to `effective_inputs_summary.json` under `depth_chart_prior`.
Key diagnostics include:

- `matched_total`, `players_total`, `matched_rate`
- `snapshot_age_minutes`
- `alert_flags` (e.g., `low_match_rate`, `stale_snapshot`, `prior_not_applied`)
- `has_alerts` (bool)
- `dnp_guardrail` (diagnostics for inference-time DNP history penalty/caps)

Crosswalk diagnostics are written under `depth_chart_crosswalk` and include:

- `matched_rows`, `unmatched_snapshot_rows`
- `snapshot_unique_players`, `match_rate`

Crosswalk artifact used by the prior:

- `bronze/realgm/player_id_crosswalk.parquet`
  - `realgm_player_id` (int)
  - `player_id` (int; canonical NBA person id used by live minutes/sim)
  - `updated_at` (UTC timestamp)
  - `match_method` (`team_name` or `override`)
  - `source_snapshot_ts` (UTC timestamp, nullable)
  - `note` (string, nullable)

Optional manual overrides file:

- `bronze/realgm/player_id_crosswalk_overrides.csv`
  - required: `realgm_player_id`, `player_id`
  - optional: `note`, `updated_at`

Optional alert tuning knobs (in `config/depth_chart_prior.json`):

- `warn_min_match_rate` (default `0.25`)
- `warn_max_snapshot_age_minutes` (default `360.0`)
- DNP guardrail knobs:
  - `dnp_guardrail_enabled`
  - `dnp_streak_threshold`, `dnp_rate_threshold`, `dnp_inactive_streak_threshold`
  - `dnp_k_streak`, `dnp_k_rate`, `dnp_k_inactive_streak`, `dnp_rotation_scale`
  - `dnp_penalty_min`, `dnp_guardrail_max_p50`, `dnp_require_non_starter`
  - `dnp_severe_*` caps for `minutes_p50/p90/p95`

## Schema Evolution

When modifying schemas:

1. Update the feature contract in `projections/models/feature_contract.py`
2. Add migration logic for downstream consumers
3. Update relevant tests in `tests/`
4. Document breaking changes in PR

## See Also

- [00_REPO_MAP.md](./00_REPO_MAP.md) - Repository structure
- [10_CONTROL_PLANE.md](./10_CONTROL_PLANE.md) - Pipeline orchestration
- `docs/minutes_v1_schema_plan.md` - Minutes schema details
