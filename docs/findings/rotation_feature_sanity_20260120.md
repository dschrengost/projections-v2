# Rotation Feature Sanity Audit — 2026-01-20

## Executive Summary

**Root Cause Identified**: The rotation SetTransformer minutes overlay is flattening star minutes because `recent_start_pct_10` is **ALL ZEROS** for every player in every team-game. This feature is a critical signal telling the model who historically starts games.

Without this signal, the model rationally spreads minutes more evenly across the roster, causing stars to lose 5-6 minutes while bench players gain.

## Evidence

### All 14 Team-Games Have Degenerate Starter Features

| Metric | Value | Expected |
|--------|-------|----------|
| Teams with `recent_start_pct_10` = 100% zeros | **14/14** | 0/14 |
| Teams with `is_confirmed_starter` = 0 for all players | **14/14** | Should have 5 per team when lineups confirmed |
| Teams with `is_projected_starter` sum = 5 | 14/14 | ✅ Expected |

### Lakers Example (team_id=1610612747)

| Player | Baseline p50 | Final p50 | Delta | is_proj_starter | recent_start_pct_10 |
|--------|-------------|-----------|-------|-----------------|---------------------|
| Luka Dončić | 34.0 | 28.5 | **-5.5** | 1 | 0.0 |
| LeBron James | 32.9 | 27.5 | **-5.4** | 1 | 0.0 |
| Deandre Ayton | 31.6 | 25.5 | **-6.1** | 1 | 0.0 |
| Jake LaRavia | 31.0 | 26.6 | **-4.4** | 1 | 0.0 |
| Marcus Smart | 30.9 | 25.9 | **-5.0** | 1 | 0.0 |
| Rui Hachimura | 16.9 | 20.9 | **+4.0** | 0 | 0.0 |
| Gabe Vincent | 11.0 | 16.0 | **+5.0** | 0 | 0.0 |

> [!WARNING]
> Stars are losing ~5 minutes each while bench players gain ~4-5 minutes. This is the model "spreading" minutes due to missing starter signals.

## Root Cause Analysis

### Bug Location

[build_minutes_live.py:1518](file:///home/daniel/projects/projections-v2/projections/cli/build_minutes_live.py#L1518)

The code recomputes `recent_start_pct_10` from historical labels:

```python
# BEFORE (BUG):
if not history_work.empty and "starter_flag" in history_work.columns:
    history_work["starter_flag"] = pd.to_numeric(history_work["starter_flag"], errors="coerce").fillna(0)
```

### The Problem

In `labels/season=2025/boxscore_labels.parquet`:

| Column | Distribution |
|--------|-------------|
| `starter_flag` | **ALL 1s** (22,528/22,528) — CORRUPT! |
| `starter_flag_label` | 6,430 ones, 16,098 zeros — CORRECT |

The code uses `starter_flag` (corrupt, all 1s) instead of `starter_flag_label` (correct).

Since every player has `starter_flag=1` in history, the computed `recent_start_pct_10` would be 1.0 for everyone. However, due to join semantics, this actually results in 0.0 being propagated.

## Fix Applied

### 1. Fixed `recent_start_pct_10` Recomputation

[build_minutes_live.py](file:///home/daniel/projects/projections-v2/projections/cli/build_minutes_live.py#L1510-L1575)

```python
# AFTER (FIXED):
# Priority: starter_flag_label (ground truth) > starter_flag (may be corrupt)
starter_col = None
if "starter_flag_label" in history_work.columns:
    sfl = pd.to_numeric(history_work["starter_flag_label"], errors="coerce").fillna(0)
    if sfl.std() > 0.01:  # Has variance (not all same value)
        starter_col = "starter_flag_label"

if starter_col is None and "starter_flag" in history_work.columns:
    sf = pd.to_numeric(history_work["starter_flag"], errors="coerce").fillna(0)
    if sf.mean() > 0.95:  # All 1s - corrupt!
        logger.warning("starter_flag is corrupt (all 1s)")
        # Fall back to starter_flag_label
```

### 2. Added Sanity Check Warnings

[live_features_v1.py](file:///home/daniel/projects/projections-v2/projections/rotation/live_features_v1.py#L540-L565)

The rotation feature builder now logs warnings when:
- `recent_start_pct_10` is ~100% zeros for a team-game
- `is_projected_starter` sum is 0 for a team-game

### 3. Unit Tests Added

[test_recent_start_pct_recompute.py](file:///home/daniel/projects/projections-v2/tests/cli/test_recent_start_pct_recompute.py)

Tests verify:
- Corrupt `starter_flag` (all 1s) triggers fallback to `starter_flag_label`
- Valid `starter_flag` is used when it has variance
- Missing columns are handled gracefully

## Verification

### Run Diagnostic Script

```bash
cd /home/daniel/projects/projections-v2
uv run python scripts/diagnostics/audit_rotation_feature_sanity_20260120.py
```

### Run Unit Tests

```bash
uv run pytest tests/cli/test_recent_start_pct_recompute.py -v
```

### Rebuild Minutes Features (After Fix)

```bash
# This should now show nonzero recent_start_pct_10 values
PROJECTIONS_DATA_ROOT=/home/daniel/projections-data \
PROJECTIONS_ALLOW_UNSAFE_POINTER_WRITES=1 \
uv run python -m projections.cli.build_minutes_live \
  --date 2026-01-20 \
  --run-id fix-validation
```

Expected log output after fix:
```
[minutes-live] recent_start_pct_10 recomputed from starter_flag_label: nonzero=70/245, mean=0.285
```

## Files Changed

| File | Change |
|------|--------|
| `projections/cli/build_minutes_live.py` | Fixed `recent_start_pct_10` recompute to use `starter_flag_label` |
| `projections/rotation/live_features_v1.py` | Added sanity check warnings for degenerate features |
| `tests/cli/test_recent_start_pct_recompute.py` | Unit tests for the fix |
| `scripts/diagnostics/audit_rotation_feature_sanity_20260120.py` | Diagnostic script |

## Recommendations

1. **Fix the labels pipeline** to populate `starter_flag` correctly (currently all 1s)
2. **Run the fixed feature builder** to regenerate minutes features
3. **Re-run rotation scoring** to verify star minutes are no longer flattened
4. **Monitor Prefect logs** for the new sanity warning messages
