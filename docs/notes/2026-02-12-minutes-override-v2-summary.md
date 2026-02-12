# Minutes Override V2 Summary

## What changed

- Added a new bounded projection utility: `projections/alloc/bounded_projection.py`.
  - `project_sum_with_bounds(x, target_sum, lb, ub, weights=None)`
  - deterministic weighted projection with explicit infeasible error (`ProjectionInfeasibleError`).

- Added v2 override engine: `projections/overrides/minutes_overrides_v2.py`.
  - Compiles raw/legacy override payloads into per-player constraints (`lb/ub`, lock, cap, floor, zero-lock, force active/inactive, eligibility).
  - Legacy `minutes_delta`/`minutes_target` are backward-compatible and compile to lock or target band (default ±2 when not exact).
  - Produces deterministic team-level `mu_minutes` via bounded projection and emits per-team diagnostics.

- Wired worlds generation with a feature-flagged mode switch in `scripts/sim_v2/generate_worlds_fpts_v2.py`:
  - `--minutes-override-mode {legacy,v2}` (default `legacy`)
  - `--override-infeasible {error,relax,ignore}`
  - `legacy` path is unchanged.
  - `v2` path restores model-baseline values on override rows (prevents double-apply), compiles constraints once, and enforces them during per-world minutes projection.

- Exposed flags in `scripts/sim_v2/run_sim_live.py` and passed through in `prefect_flows/live_nba_pipeline.py` (`run_sim_task`, defaults remain legacy/error).

## How to enable v2

```bash
uv run python -m scripts.sim_v2.run_sim_live \
  --run-date 2026-01-18 \
  --profile-name sim_v3 \
  --num-worlds 2000 \
  --minutes-override-mode v2 \
  --override-infeasible error
```

## Why this fixes "delta caused weird allocator outcomes"

- Overrides are treated as explicit constraints (not just soft deltas).
- Locked and zero-locked players are enforced inside world generation after sampling (`sample -> clamp -> project`), so allocator redistribution cannot break them.
- Force-active/force-inactive constraints are applied to availability gating so membership/bench-zero gates cannot silently flip intended locks.
- Bounds (`lb/ub`) prevent uncontrolled reallocations (e.g., 48-minute spikes) when caps/floors are present.
- Legacy behavior remains default and untouched unless `--minutes-override-mode v2` is set.

## V2 artifacts and diagnostics

For each worlds run in v2 mode, under:
`artifacts/sim_v2/worlds_fpts_v2/game_date=<date>/run=<run_id>/` (or day dir when no run id)

- `overrides_input.json`
- `overrides_compiled_v2.json`
- `override_resolved_minutes.parquet`
- `override_diag.json`

Logs include per-team lines prefixed with `[override-diag]`.

## Next Agent Handoff: GameView Redesign

The next agent should redesign the GameView manual override UX to match v2 semantics.

Primary target files:

- `web/minutes-dashboard/src/components/GameView.tsx`
- `web/minutes-dashboard/src/components/PlayerOpsPanel.tsx`

Goals:

1. Move UI away from delta-only mental model toward explicit constraints:
   - lock (exact minutes)
   - floor
   - cap
   - zero-lock / force inactive
   - force active
2. Keep backward compatibility with existing payload shape accepted by `/api/ops/overrides`.
3. Show team-level feasibility and projected redistribution preview before save.
4. Surface solver/compile diagnostics (caps hit, floors hit, infeasible reason) from v2 artifacts where available.
5. Make mode explicit in UI copy:
   - legacy (current default)
   - v2 constraints mode (when enabled by operator/runtime)

Design constraints:

- Do not break current production default behavior.
- Avoid double-apply semantics in UI messaging.
- Keep save path compatible with current backend endpoint contracts.
