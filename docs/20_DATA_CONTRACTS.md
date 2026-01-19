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
