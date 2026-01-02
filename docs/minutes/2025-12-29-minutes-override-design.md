# Minutes Override Design Analysis

**Date**: 2025-12-29
**Status**: Implemented

## Problem Statement

The current minutes override system allows setting `minutes_p10`, `minutes_p50`, `minutes_p90` directly, but:
1. **Overriding p10/p90 has little effect** - these are only used as clamp bounds, not to determine variance
2. **Bug**: When `sigma_minutes_mu` is computed (lines 1835-1839 in `generate_worlds_fpts_v2.py`), p10/p90 overrides are overwritten
3. **Semantic mismatch**: Operators think in terms of "this player should play more/less" but the system requires precise quantile values
4. **Team constraint complexity**: Changing one player's minutes affects all teammates

## Root Cause Analysis

### Current Flow
```
Override p50 → apply_overrides_to_minutes_df → minutes_df["minutes_p50"] updated
                                              ↓
                                    team reconciliation (locked players preserved)
                                              ↓
                               gs_minutes_p50 = overridden value ✓
                               gs_minutes_p10/p90 = overridden values...
                                              ↓
                               BUG: if sigma_minutes_mu is set, p10/p90 are OVERWRITTEN!
                                              ↓
                               sample_minutes_noise_per_world():
                                 - noise = Normal(0, CV × p50)  # CV from position, not quantiles
                                 - bounds = [p10 - pad, p90 + pad]  # only used for clamping!
                                              ↓
                               team-240 projection (redistributes minutes)
```

### Why p50 overrides have "little effect"
1. The noise sampling uses position-aware CV (8-28% depending on role), not the quantile spread
2. Team-240 constraint redistributes the delta to/from teammates
3. Per-world sampling adds noise that may partially undo the override

### Why p10/p90 overrides are ignored
1. Lines 1835-1839 in `generate_worlds_fpts_v2.py` overwrite them when `sigma_minutes_mu` is computed
2. Even when not overwritten, they only affect clamp bounds, not variance

## Proposed Solution: "Minutes Delta" Override

Instead of raw quantile overrides, add a **relative adjustment** semantic that's more intuitive.

### New Override Field: `minutes_delta`

| Field | Type | Description |
|-------|------|-------------|
| `minutes_delta` | float | Additive adjustment to model's p50 (e.g., +5 or -3) |

### Behavior
1. **Apply delta to p50**: `adjusted_p50 = model_p50 + minutes_delta`
2. **Shift quantiles proportionally**: `adjusted_p10 = model_p10 + minutes_delta`, same for p90
3. **Clamp to valid range**: `[0, 48]` for all quantiles
4. **Lock the adjusted player**: prevent team reconciliation from changing them
5. **Redistribute delta to teammates**: non-locked players absorb the change

### Why This Works Better
- **Preserves model variance**: The shape of the distribution is unchanged, just shifted
- **Intuitive operator UX**: "Give Jokic 5 more minutes" vs "Set Jokic to 35.5 p50"
- **Predictable impact**: +5 for one player means -5 distributed among teammates
- **Compatible with sim**: The adjusted quantiles flow through correctly

## Implementation Plan

### Phase 1: Fix the Bug (Critical)
**File**: `scripts/sim_v2/generate_worlds_fpts_v2.py`

At lines 1835-1839, change from unconditionally overwriting p10/p90 to only computing sigma-based quantiles when no override exists:

```python
# Only recompute p10/p90 from sigma if NOT overridden
if sigma_minutes_mu is not None:
    z90 = 1.2815515655446004
    sigma = np.maximum(sigma_minutes_mu, 0.1)
    # Check if quantiles were overridden per player
    p10_overridden = ...  # track which players had p10 override
    p90_overridden = ...  # track which players had p90 override
    gs_minutes_p10 = np.where(p10_overridden, gs_minutes_p10, np.maximum(gs_minutes_p50 - z90 * sigma, 0.0))
    gs_minutes_p90 = np.where(p90_overridden, gs_minutes_p90, np.maximum(gs_minutes_p50 + z90 * sigma, gs_minutes_p10 + 0.01))
```

### Phase 2: Add `minutes_delta` Field
**File**: `projections/ops/overrides.py`

1. Add `"minutes_delta"` to `MINUTES_FIELDS` tuple
2. Add coercion logic in `_coerce_override_field()`:
   ```python
   if name == "minutes_delta":
       return float(val)  # allow negative values, clip to [-48, 48]
   ```
3. Modify `apply_overrides_to_minutes_df()` to apply delta before reconciliation:
   ```python
   # Apply minutes_delta as additive adjustment
   delta_col = merged.get("minutes_delta_ops")
   if delta_col is not None:
       has_delta = delta_col.notna()
       for qcol in ("minutes_p10", "minutes_p50", "minutes_p90", ...):
           if qcol in merged.columns:
               merged.loc[has_delta, qcol] = (merged.loc[has_delta, qcol] + delta_col[has_delta]).clip(0, 48)
       locked_mask = locked_mask | has_delta
   ```

### Phase 3: API & UI Updates
**Files**:
- `projections/api/ops_api.py`: Add `minutes_delta` to `OpsPlayerOverrideUpdate` model
- `web/minutes-dashboard/src/pages/OptimizerPage.tsx`: Add slider/input for delta adjustment

### Phase 4: Impact Preview (Optional Enhancement)
**Files**:
- `projections/api/ops_api.py`: New endpoint `POST /api/ops/preview-override`
- Returns: simulated impact on all teammates before committing

## Files to Modify

| File | Changes |
|------|---------|
| `scripts/sim_v2/generate_worlds_fpts_v2.py` | Fix p10/p90 override bug |
| `projections/ops/overrides.py` | Add `minutes_delta` field and application logic |
| `projections/api/ops_api.py` | Add field to Pydantic model |
| `web/minutes-dashboard/src/pages/OptimizerPage.tsx` | UI for delta input |

## Alternative Considered: Minutes Share

Instead of delta, allow operators to set "14.5% of team minutes" (share-based). Rejected because:
- Less intuitive than additive delta
- Harder to reason about when multiple players are locked
- More complex conversion logic

## Open Questions

1. Should `minutes_delta` replace or coexist with raw `minutes_p50` overrides?
   - Recommendation: Coexist. If both set, `minutes_delta` takes precedence.

2. When multiple teammates are locked, how to handle impossible constraints (e.g., locked sum > 240)?
   - Recommendation: Warn and cap proportionally.

3. Should variance be reduced when an override is applied (operator confidence)?
   - Current thinking: Preserve model variance, but could add optional `minutes_confidence` flag.
