# Gameview v2 Patch-Layer UI (2026-02-12)

## What changed

The Gameview UI now uses a constraint-based minutes override model (v2 semantics) instead of legacy delta-first editing.

Key UX updates:
- Game tabs for fast switching between games.
- Each game shows both teams side-by-side.
- Player row click opens a right-side drawer with baseline vs resolved projections.
- Override controls now edit constraints (`none`, `lock`, `band`, `cap`, `floor`, `zero`, `force_active`, `force_inactive`) rather than ad-hoc deltas.

## Apply vs Apply & Run Worlds

### Apply (fast)
- Writes v2 override payload for the selected game.
- Compiles constraints via `apply_minutes_overrides_v2()`.
- Returns resolved minutes (`mu_minutes`, `lb_minutes`, `ub_minutes`) and team diagnostics.
- Does **not** run full worlds.

### Apply & Run Worlds
- First performs the same v2 apply/compile step.
- Then runs the existing worlds patch flow using:
  - `minutes_override_mode=v2`
  - `override_infeasible` policy (currently `error` in UI)
- Polls ops pipeline status and refreshes projections after completion.

## Legacy conflict avoidance

To avoid mixed-format writes:
- The v2 apply endpoint rewrites touched player records to remove legacy minutes-control fields for that player before persisting v2 fields.
- UI writes only v2 shape for override controls.
- Existing legacy fields can still be shown as read-only conflict indicators (`legacy_fields_present`) but are not written by v2 UI controls.

This keeps Gameview as a patch layer, not a source-of-truth projection engine:
- UI submits override constraints.
- Server compiles/applies constraints and returns resolved outputs.
- Client does not double-apply minutes adjustments.
