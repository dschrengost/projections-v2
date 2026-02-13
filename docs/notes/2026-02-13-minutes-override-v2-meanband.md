# Minutes Override v2: Mean-Bound Semantics (2026-02-13)

## Summary
- `lock` now means **mean lock** (resolved team mean `mu_minutes` is fixed), not world-by-world lock.
- `band` now means **mean band** (`mu_minutes` constrained to `[L, U]`), not per-world truncation.
- World generation no longer clamps to mean lock/band bounds.

## Constraint Split
- **Mean bounds** (`mean_lb_minutes`, `mean_ub_minutes`):
  - Applied only during pre-world team mean resolution (`Project240` on baseline means).
- **World bounds** (`world_lb_minutes`, `world_ub_minutes`):
  - Applied only during per-world clamp + team projection.
  - Default world bounds are typically `[0, 48]`.

## Still Supported
- **Zero-lock / ineligible**: hard 0 in mean and world bounds (`eligible=false`, `force_inactive=true`).
- **Hard cap**: applied to `world_ub_minutes` for per-world physical enforcement.
- **Force active/inactive** gate behavior remains, but force-active is not converted into a world minutes floor.

## Why this change
- Prevents variance collapse caused by applying mean bands per world.
- Keeps manual edits deterministic and fast at the mean layer while preserving realistic world tails (`p10/p50/p90`).

## Artifacts
`override_resolved_minutes.parquet` now includes:
- `b_minutes`, `mu_minutes`
- `mean_lb_minutes`, `mean_ub_minutes`
- `world_lb_minutes`, `world_ub_minutes`
- `eligible`, `weight`
- compatibility aliases: `lb_minutes`, `ub_minutes` (mapped to mean bounds)

`override_diag.json` now includes both totals:
- `sum_mean_lb`, `sum_mean_ub`
- `sum_world_lb`, `sum_world_ub`
- plus `mean_world_bounds_differ` diagnostics when the sets intentionally diverge.
