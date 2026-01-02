# DNP & Teammate Injury Features Implementation Plan

> **Goal**: Improve minutes model accuracy by properly handling DNP games and modeling teammate injury effects.

---

## Current Production Features (49 total)

The current `minutes_v1` model uses the following features. New features should integrate with this existing set.

### Injury/Availability Features
| Feature | Description |
|---------|-------------|
| `is_out` | 1 if player status is OUT |
| `is_q` | 1 if player status is QUESTIONABLE |
| `is_prob` | 1 if player status is PROBABLE |
| `restriction_flag` | 1 if player has minutes restriction |
| `ramp_flag` | 1 if player is ramping up from injury |
| `games_since_return` | Games played since returning from injury |
| `days_since_return` | Days since returning from injury |
| `injury_snapshot_missing` | 1 if no injury data available |

### Historical Minutes Features
| Feature | Description |
|---------|-------------|
| `min_last1` | Minutes in most recent game (excluding DNPs*) |
| `min_last3` | Average minutes over last 3 games (excluding DNPs*) |
| `min_last5` | Average minutes over last 5 games (excluding DNPs*) |
| `roll_mean_3` | Rolling 3-game mean minutes (excluding DNPs*) |
| `roll_mean_5` | Rolling 5-game mean minutes (excluding DNPs*) |
| `roll_mean_10` | Rolling 10-game mean minutes (excluding DNPs*) |
| `roll_iqr_5` | IQR of last 5 games (variance signal) |
| `sum_min_7d` | Total minutes played in last 7 calendar days |
| `z_vs_10` | Z-score of last game vs rolling 10-game mean |
| `rotation_minutes_std_5g` | Std dev of team's rotation minutes (last 5 games) |

*Note: DNP exclusion was added Dec 2025 to fix Aaron Wiggins issue

### Role/Starter Features
| Feature | Description |
|---------|-------------|
| `starter_flag` | 1 if player started most recent game |
| `starter_prev_game_asof` | Starter flag as of previous game (lag) |
| `recent_start_pct_10` | % of last 10 games player started |
| `role_change_rate_10g` | How often role changed in last 10 games |
| `is_projected_starter` | 1 if projected to start (from lineups) |
| `is_confirmed_starter` | 1 if confirmed starter (official lineup) |

### Game Environment Features
| Feature | Description |
|---------|-------------|
| `spread_home` | Vegas spread for home team |
| `total` | Vegas over/under total |
| `blowout_index` | Composite blowout risk (from spread/total) |
| `blowout_risk_score` | Probability of blowout |
| `close_game_score` | Probability of close game |
| `home_flag` | 1 if player's team is home |
| `home_team_id` | ID of home team |
| `away_team_id` | ID of away team |

### Schedule/Rest Features
| Feature | Description |
|---------|-------------|
| `is_b2b` | 1 if back-to-back game |
| `is_3in4` | 1 if 3rd game in 4 days |
| `is_4in6` | 1 if 4th game in 6 days |
| `days_since_last` | Days since last game |
| `season_phase` | 0-1 normalized season progress |

### Team Depth/Archetype Features
| Feature | Description |
|---------|-------------|
| `available_B` | Count of available "Big" archetype players |
| `available_G` | Count of available "Guard" archetype players |
| `available_W` | Count of available "Wing" archetype players |
| `depth_same_pos_active` | Active players at same position |
| `same_archetype_overlap` | Players sharing same archetype |
| `arch_delta_sum` | Total minutes vacuum from OUT archetypes |
| `arch_delta_same_pos` | Minutes vacuum at player's position |
| `arch_delta_max_role` | Max single archetype delta |
| `arch_delta_min_role` | Min single archetype delta |
| `arch_missing_same_pos_count` | OUT players at same position |
| `arch_missing_total_count` | Total OUT players on team |
| `team_minutes_dispersion_prior` | Historical team minutes concentration |

---

## Conflict Analysis: Existing vs Proposed Features

Before implementing, we identified overlaps with existing features:

| Existing Feature | Proposed Feature | Status | Resolution |
|------------------|------------------|--------|------------|
| `games_since_return` | `games_since_injury_return` | ⚠️ DUPLICATE | **Remove from plan** - existing does same thing |
| `ramp_flag` | `is_post_injury_ramp` | ⚠️ SIMILAR | **Keep both** - `ramp_flag` is text-based, new one is count-based |
| `arch_delta_sum` | `minutes_vacuum_team` | ⚠️ SIMILAR | **Audit first** - may already work |
| `arch_delta_same_pos` | `minutes_vacuum_same_pos` | ⚠️ SIMILAR | **Audit first** - may already work |

### Features REMOVED from plan (already exist):
- ~~`games_since_injury_return`~~ → Use existing `games_since_return`
- ~~`minutes_vacuum_team`~~ → Audit existing `arch_delta_sum` first
- ~~`minutes_vacuum_same_pos`~~ → Audit existing `arch_delta_same_pos` first

### Features KEPT in plan (truly new):
- `dnp_injury` / `dnp_cd` / `dnp_rest` labels
- `dnp_count_7d` / `dnp_injury_count_7d` rolling counts
- `is_post_injury_ramp` (first 5 games after return)
- `injury_stint_length` (how long was the absence)
- `is_next_man_up` (more specific than existing archetype features)
- `historical_boost_sum` (empirical teammate effects)

**need to remember that in current minutes production model, we dropped all DNP rows from training***


---

## Phase 0: Audit Existing Features (FIRST)

Before adding new features, verify existing ones work correctly.

### 0.1 Audit `games_since_return`

**Question**: Does this reset properly when a player returns from an injury stint?

**Test case**: Aaron Wiggins
- Was OUT Nov 8-28 (6 games)
- Returned Nov 30
- On Dec 10, `games_since_return` should be ~5-6

```bash
# Check Aaron Wiggins' games_since_return for Dec 10
uv run python -c "
import pandas as pd
labels = pd.read_parquet('gold/features_minutes_v1/season=2025/')
wiggins = labels[labels['player_name'].str.contains('Aaron Wiggins', na=False)]
print(wiggins[['game_date', 'games_since_return', 'ramp_flag', 'status']].tail(10))
"
```

### 0.2 Audit `arch_delta_sum` and `arch_delta_same_pos`

**Question**: Are these correctly capturing the "minutes vacuum" when starters are out?

**Test case**: When a 30mpg starter is OUT, do their teammates show higher `arch_delta_same_pos`?

```bash
# Check if arch_delta features are populated and sensible
uv run python -c "
import pandas as pd
features = pd.read_parquet('gold/features_minutes_v1/season=2025/')
print('arch_delta_sum stats:')
print(features['arch_delta_sum'].describe())
print()
print('arch_delta_same_pos stats:')
print(features['arch_delta_same_pos'].describe())
"
```

**If audit passes**: Existing features are sufficient, skip Phase 3.
**If audit fails**: Proceed with Phase 3 to fix/enhance vacuum features.

---

## Phase 1: DNP Labels in Boxscore Data

### 1.1 Create DNP Label ETL

**File**: `projections/etl/dnp_labels.py` (NEW)

```python
"""Add DNP reason labels to boxscore data."""

def classify_dnp_reason(
    boxscore_df: pd.DataFrame,
    injury_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each 0-minute game, classify the DNP reason.
    
    Returns boxscore_df with new columns:
    - dnp_flag: 1 if minutes == 0, else 0
    - dnp_injury: 1 if player was on injury report as OUT
    - dnp_rest: 1 if reason contains "Rest" or "Load Management"
    - dnp_cd: 1 if 0 minutes but no injury record (coach decision)
    """
```

**Logic**:
```
For each row where minutes == 0:
    1. Look up (player_id, game_date) in injury_snapshot
    2. If status = 'OUT' and reason contains 'Injury':
         dnp_injury = True
    3. Elif status = 'OUT' and reason contains 'Rest|Load':
         dnp_rest = True
    4. Elif no injury match or status != 'OUT':
         dnp_cd = True (coach decision / unknown)
```

**Output schema changes** to `BOX_SCORE_LABELS_SCHEMA`:
```python
# Add to boxscore_labels.parquet
"dnp_flag": INT_DTYPE,           # 1 if minutes == 0
"dnp_injury": INT_DTYPE,         # 1 if injury-related DNP
"dnp_rest": INT_DTYPE,           # 1 if rest/load management
"dnp_cd": INT_DTYPE,             # 1 if coach decision
"dnp_reason_raw": STRING_DTYPE,  # Original reason text (nullable)
```

### 1.2 Backfill Historical DNP Labels

**Script**: `scripts/minutes/backfill_dnp_labels.py` (NEW)

```
For each season:
    1. Load boxscore_labels.parquet
    2. Load injury_snapshot data for season
    3. Run classify_dnp_reason()
    4. Write updated boxscore_labels.parquet
```

**Estimated time**: 10-15 min per season

### 1.3 Update Boxscore ETL Pipeline

**File**: `projections/etl/boxscores.py` (MODIFY)

Add DNP classification step to the ingestion pipeline so new games automatically get DNP labels.

---

## Phase 2: DNP Rolling Features

### 2.1 Add DNP Features to Feature Builder

**File**: `projections/minutes_v1/features.py` (MODIFY)

Add new method `_attach_dnp_features()`:

```python
def _attach_dnp_features(self, df: pd.DataFrame) -> pd.DataFrame:
    """
    Add DNP-related rolling features.
    
    New features:
    - dnp_count_7d: Total DNPs in last 7 days
    - dnp_injury_count_7d: Injury DNPs in last 7 days  
    - dnp_cd_count_7d: Coach decision DNPs in last 7 days
    - games_since_last_dnp: Games played since last DNP
    - is_post_injury_ramp: True for first 5 games after injury return
    - injury_stint_length: How many games was player out?
    """
```

### 2.2 Update Live Feature Builder

**File**: `projections/cli/build_minutes_live.py` (MODIFY)

Add same DNP feature computation for live inference path (around line 830 where trend features are computed).

### 2.3 Update Feature Schema

**File**: `projections/minutes_v1/schemas.py` (MODIFY)

Add new feature columns to `FEATURES_MINUTES_V1_SCHEMA`:
```python
"dnp_count_7d": INT_DTYPE,
"dnp_injury_count_7d": INT_DTYPE,
"dnp_cd_count_7d": INT_DTYPE,
"games_since_last_dnp": INT_DTYPE,
"is_post_injury_ramp": INT_DTYPE,
"injury_stint_length": INT_DTYPE,
```

---

## Phase 3: Teammate Injury Interaction Features (CONDITIONAL)

> **Note**: Only proceed if Phase 0 audit shows `arch_delta_*` features are not working correctly.

### 3.1 Build Historical Minutes Baseline Table

**Script**: `scripts/minutes/build_player_minutes_baseline.py` (NEW)

Create a lookup table of each player's "baseline minutes":
```
player_minutes_baseline.parquet:
- player_id
- team_id  
- season
- avg_minutes_when_healthy: float
- role_rank: int  # 1 = most minutes, 2 = second, etc.
- position: str
```

### 3.2 Add Enhanced Vacuum Features

**File**: `projections/minutes_v1/features.py` (MODIFY)

Add method `_attach_enhanced_vacuum_features()`:

```python
def _attach_enhanced_vacuum_features(self, df: pd.DataFrame) -> pd.DataFrame:
    """
    Enhanced vacuum features (only if arch_delta_* insufficient).
    
    New features:
    - is_next_man_up: True if player is primary backup to an OUT starter
    - out_starter_count: Number of normal starters marked OUT
    """
```

### 3.3 Build Historical Interaction Lookup (Advanced)

**Script**: `scripts/minutes/build_teammate_interactions.py` (NEW)

Pre-compute empirical "when X was out, Y gained Z minutes":
```
For each team/season:
    For each player X who missed games:
        For each other player Y on roster:
            games_X_out = games where X was OUT
            games_X_in = games where X played
            Y_boost = mean(Y.minutes | X out) - mean(Y.minutes | X in)
            if Y_boost > 0: record (X, Y, boost)
```

Output: `teammate_interaction_effects.parquet`

---

## Phase 4: Training Pipeline Updates

### 4.1 Update Training Data Loading

**File**: `projections/models/minutes_lgbm.py` (MODIFY)

Add option to include DNP games:
```python
# New parameter
include_dnp_games: bool = typer.Option(
    False,
    "--include-dnp-games",
    help="Include 0-minute games in training data."
)

# Update filtering logic
if include_dnp_games:
    # Keep all rows, let model learn DNP patterns
    train_df = feature_df[...]
else:
    # Original behavior: filter to played games
    train_cond_df = train_df[train_df["plays_target"] == 1]
```

### 4.2 Add Play Probability Head Updates

The existing `play_prob` head should learn DNP patterns better with the new features. Ensure it's using:
- `dnp_count_7d`
- `is_post_injury_ramp`
- Existing `games_since_return`

### 4.3 Model Validation

Add validation checks:
```python
# Validate DNP labels are populated
assert (df["dnp_flag"].notna()).all()
```

---

## Phase 5: Backfill & Validation

### 5.1 Full Historical Backfill

```bash
# Step 1: Add DNP labels to boxscores
python scripts/minutes/backfill_dnp_labels.py --start-season 2022 --end-season 2025

# Step 2: Rebuild features
python -m projections.pipelines.build_features_minutes_v1 --start-date 2022-10-01 --end-date 2025-12-31
```

### 5.2 Validation Tests

Create `tests/minutes/test_dnp_features.py`:
```python
def test_dnp_labels_populated():
    """Verify all 0-minute games have DNP labels."""
    
def test_is_post_injury_ramp():
    """Aaron Wiggins case: verify first 5 games after return are flagged."""
    
def test_injury_stint_length():
    """Verify stint length is correctly computed."""
```

---

## Implementation Checklist

### Phase 0: Audit (Est: 1-2 hours)
- [ ] Verify `games_since_return` resets correctly after injury
- [ ] Verify `arch_delta_sum` captures minutes vacuum
- [ ] Document findings and decide if Phase 3 needed

### Phase 1: DNP Labels (Est: 3-4 hours)
- [ ] Create `projections/etl/dnp_labels.py`
- [ ] Add DNP columns to `BOX_SCORE_LABELS_SCHEMA`
- [ ] Create `scripts/minutes/backfill_dnp_labels.py`
- [ ] Run backfill for seasons 2022-2025
- [ ] Update `projections/etl/boxscores.py` to add DNP labels on ingest

### Phase 2: DNP Features (Est: 3-4 hours)
- [ ] Add `_attach_dnp_features()` to `features.py`
- [ ] Update `build_minutes_live.py` for live path
- [ ] Add columns to `FEATURES_MINUTES_V1_SCHEMA`
- [ ] Test with Aaron Wiggins case
- [ ] Rebuild features for validation window

### Phase 3: Enhanced Vacuum (Est: 4-6 hours) - CONDITIONAL
- [ ] Create `scripts/minutes/build_player_minutes_baseline.py`
- [ ] Add `is_next_man_up` feature
- [ ] (Optional) Build historical interaction lookup
- [ ] Test with specific injury scenarios

### Phase 4: Training Updates (Est: 2-3 hours)
- [ ] Add `--include-dnp-games` flag to training CLI
- [ ] Update play_prob head feature list
- [ ] Add validation assertions
- [ ] Run validation experiments

### Phase 5: Backfill & Validation (Est: 2-4 hours)
- [ ] Run full historical backfill
- [ ] Create validation test suite
- [ ] Validate Aaron Wiggins case is fixed
- [ ] Compare model metrics before/after

---

## Success Criteria

1. **Aaron Wiggins Test**: After implementation, re-running 2025-12-10 should give Aaron Wiggins p50 ≈ 20 minutes (not 0)

2. **DNP Labels Coverage**: 100% of 0-minute games should have DNP reason labels

3. **Ramp Flag**: First 5 games after injury return should have `is_post_injury_ramp = 1`

4. **Model Improvement**: Play probability should better predict DNP games; conditional minutes MAE should remain similar or improve

---

## File Change Summary

| File | Change Type | Effort |
|------|-------------|--------|
| `projections/etl/dnp_labels.py` | NEW | Medium |
| `projections/minutes_v1/schemas.py` | MODIFY | Low |
| `projections/etl/boxscores.py` | MODIFY | Low |
| `projections/minutes_v1/features.py` | MODIFY | Medium |
| `projections/cli/build_minutes_live.py` | MODIFY | Medium |
| `projections/models/minutes_lgbm.py` | MODIFY | Low |
| `scripts/minutes/backfill_dnp_labels.py` | NEW | Medium |
| `scripts/minutes/build_player_minutes_baseline.py` | NEW (Conditional) | Medium |
| `scripts/minutes/build_teammate_interactions.py` | NEW (Optional) | High |
| `tests/minutes/test_dnp_features.py` | NEW | Medium |

**Total Estimated Effort**: 12-18 hours (1.5-2.5 focused days)
- Phase 0: 1-2 hours
- Phases 1-2: 6-8 hours (core work)
- Phase 3: 4-6 hours (conditional)
- Phases 4-5: 4-6 hours
