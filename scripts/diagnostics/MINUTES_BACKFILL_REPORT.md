# Minutes Backfill Feasibility Report

**Date**: 2025-12-15
**Target Window**: 2024-10-22 to 2025-02-01 (usage_shares_v1 training range)

## Executive Summary

**STATUS: PARTIAL GO**

Only **2 dates** require backfill in the target window:
- **2024-12-17**: NBA Cup Final (1 game) - **FIXED** in pilot run
- **2025-02-01**: Regular season (9 games) - **BLOCKED** due to missing injury data

## Inventory of Existing Scripts

### Usable Scripts

| Script | Status | Purpose |
|--------|--------|---------|
| `scripts/minutes/backfill_projections_minutes_v1.py` | Usable | Main backfill orchestrator for scoring features |
| `projections/cli/build_minutes_live.py --backfill-mode` | Usable | Builds features for historical dates |
| `projections/cli/score_minutes_v1.py` | Usable | Scores features to get minutes predictions |
| `scripts/diagnostics/minutes_backfill_audit.py` | **NEW** | Created for this audit |
| `scripts/minutes/check_projections_minutes_v1_coverage.py` | Usable | Checks gold projection coverage |
| `scripts/minutes/find_feature_desert_dates.py` | Usable | Identifies dates with poor feature quality |

### Stale/Not Needed
None - all relevant scripts are functional.

## Coverage Audit Results

### Date Range Analysis (2024-10-22 to 2025-02-01)

```
Total dates with games: 99
Features missing: 2
Projections empty: 2
Total needing backfill: 2
```

### Input Coverage Summary

| Input | Full Coverage |
|-------|---------------|
| `injuries_snapshot` | 98/99 dates (missing: 2025-02-01) |
| `roster_nightly` | 99/99 dates |
| `odds_snapshot` | 99/99 dates |

### Dates Needing Backfill

| Date | Games | Injuries | Roster | Odds | Features | Issue |
|------|-------|----------|--------|------|----------|-------|
| 2024-12-17 | 1 | 1/1 | 1/1 | 1/1 | False | NBA Cup Final - features never built |
| 2025-02-01 | 9 | 0/9 | 9/9 | 9/9 | False | Injury snapshots missing for all games |

## Pilot Run Results

### 2024-12-17 (NBA Cup Final - Thunder vs Bucks)

**Result: SUCCESS**

Commands executed:
```bash
# Step 1: Build features with backfill mode
uv run python -m projections.cli.build_minutes_live \
    --date 2024-12-17 \
    --data-root /home/daniel/projections-data \
    --out-root /tmp/backfill_pilot_features \
    --run-as-of-ts "2024-12-18T01:00:00" \
    --backfill-mode \
    --skip-active-roster

# Step 2: Score features
uv run python -m projections.cli.score_minutes_v1 \
    --date 2024-12-17 \
    --features-path /tmp/backfill_pilot_features/2024-12-17/run=20241218T010000Z/features.parquet \
    --artifact-root /tmp/backfill_pilot_predictions

# Step 3: Copy to gold
cp /tmp/backfill_pilot_predictions/2024-12-17/minutes.parquet \
   /home/daniel/projections-data/gold/projections_minutes_v1/2024-12-17/minutes.parquet
```

**Output:**
- 36 rows (2 teams × 18 players)
- All key columns present: `minutes_p50`, `play_prob`, `minutes_p10`, `minutes_p90`
- Sample predictions look reasonable (Giannis ~35 min, SGA ~34 min)

### Bug Fixed During Pilot

Fixed `projections/minutes_v1/pos.py:13` - `canonical_pos_bucket()` didn't handle pandas NA values properly. Added `pd.isna(value)` check.

## Blockers

### 2025-02-01: Missing Injury Snapshots

The `silver/injuries_snapshot` does not contain game_ids for Feb 1, 2025 (22400686-22400694). The max game_id in injuries is 62400001 (NBA Cup Final).

**Root Cause**: The injury snapshot ETL pipeline stopped before Feb 2025 games or has gaps.

**Resolution Options**:
1. Backfill `silver/injuries_snapshot` from `bronze/injuries_raw` for Feb 2025
2. Use `--injuries-path` override pointing to a custom injury snapshot
3. Accept missing minutes predictions for this date (only 9 games, minimal impact on training)

## Recommended Actions

### Immediate (2024-12-17 - DONE)

The pilot run has already fixed 2024-12-17. The output is in:
```
/home/daniel/projections-data/gold/projections_minutes_v1/2024-12-17/minutes.parquet
```

### Post-Fix: Rebuild Downstream Data

After all dates are backfilled:

```bash
# Rebuild minutes_for_rates for affected dates
uv run python -m scripts.minutes.build_minutes_for_rates \
    --data-root /home/daniel/projections-data \
    --start-date 2024-12-17 \
    --end-date 2024-12-17

# Rebuild rates_training_base for affected dates
uv run python -m scripts.rates.build_training_base \
    --data-root /home/daniel/projections-data \
    --start-date 2024-12-17 \
    --end-date 2024-12-17 \
    --overwrite

# Rebuild usage_shares_training_base for affected dates
uv run python -m scripts.usage_shares_v1.build_training_base \
    --data-root /home/daniel/projections-data \
    --start-date 2024-12-17 \
    --end-date 2024-12-17
```

### For 2025-02-01 (Blocked)

Either:
1. Skip this date (minimal impact - 9 games out of 99 dates)
2. Fix injury ETL and backfill injuries first:
   ```bash
   # Check bronze injuries for Feb 2025
   ls /home/daniel/projections-data/bronze/injuries_raw/season=2024/

   # Backfill silver injuries from bronze (if data exists)
   # [Script TBD based on bronze data availability]
   ```

## Performance Notes

- Feature building: ~5-10 seconds per date
- Scoring: ~2-3 seconds per date
- **Total for full window**: ~15-20 minutes for 99 dates (if all inputs present)

## Files Created/Modified

### Created
- `scripts/diagnostics/minutes_backfill_audit.py` - Coverage audit tool

### Modified
- `projections/minutes_v1/pos.py` - Fixed NA handling bug

### Output Written
- `/home/daniel/projections-data/gold/projections_minutes_v1/2024-12-17/minutes.parquet`

## Next Steps

1. **Verify downstream impact**: Rebuild `rates_training_base` for 2024-12-17 and verify `minutes_pred_p50` coverage improves
2. **Decide on 2025-02-01**: Either fix injury ETL or accept missing data
3. **Re-run usage_shares model training**: Once downstream data is rebuilt
