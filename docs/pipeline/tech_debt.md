# Pipeline Tech Debt

This document tracks known technical debt in the data pipelines that should be addressed when time permits.

---

## Odds Snapshot Gap (December 2025)

**Added**: 2025-12-15  
**Priority**: Medium  
**Affected Systems**: `rates_training_base`, `usage_shares_training_base`

### Problem

The `silver/odds_snapshot` partitions have a gap for games from approximately December 1-11, 2025 (game_ids 22500306-22501208). This means that when `rates_training_base` is built for these dates, the vegas features (`spread_close`, `total_close`, `team_itt`, `opp_itt`) are null even though the live pipeline had odds during scoring.

The root cause is likely that:
1. The odds ETL was not running during this period, OR
2. The month rollover logic didn't properly partition December data

### Current Workaround

The `usage_shares_v1/build_training_base.py` script has a fallback that:
1. Detects when >50% of vegas features are missing
2. Loads odds from `minutes artifacts` or `gold/projections_minutes_v1`
3. Merges the fallback odds into the training data

This works because the live minutes pipeline always has odds during scoring via `game_env.attach_game_environment_features()`.

### Proper Fix

1. **Backfill `silver/odds_snapshot`** for December 2025 games using historical oddstrader data
2. **Ensure daily ETL** properly writes to month partitions across month boundaries
3. **Rebuild `rates_training_base`** for affected dates
4. **Remove the fallback** from `usage_shares_v1/build_training_base.py`

### Related Code

- `scripts/usage_shares_v1/build_training_base.py` - contains the fallback logic
- `scripts/rates/build_training_base.py` - reads from `odds_snapshot` in `load_odds()`
- `projections/etl/odds.py` - daily odds ETL that writes to `silver/odds_snapshot`

---

## Template for New Entries

```markdown
## [Issue Title]

**Added**: YYYY-MM-DD  
**Priority**: High/Medium/Low  
**Affected Systems**: list of affected modules

### Problem
Description of the issue.

### Current Workaround
Description of any workarounds in place.

### Proper Fix
Steps to properly resolve the issue.

### Related Code
- List of relevant files
```
