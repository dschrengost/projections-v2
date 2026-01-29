# Projections Contract (Canonical)

This document defines the **canonical projection bundle** consumed by:
- Dashboard (Minutes page + GameView)
- Optimizer (QuickBuild)
- Contest simulation

Non-negotiable goal: **the primary numbers shown in the UI must match what we optimize/simulate**.
If the UI shows additional diagnostics (e.g. conditional-on-playing model outputs), they must be clearly labeled.

## Canonical Artifact + Assembly

### Canonical source
The canonical per-slate artifact is:

- `$PROJECTIONS_DATA_ROOT/artifacts/projections/<YYYY-MM-DD>/run=<run_id>/projections.parquet`

This is the **unified projections** dataset assembled by the live pipeline (minutes + sim summaries + ownership).

### Canonical assembly module
All consumers must load and normalize projections via:

- `projections/projections_bundle.py`

That module:
- resolves the correct `run_id` (blessed/pinned/promoted/latest)
- loads the unified projections parquet
- adds **canonical, explicitly-named fields** (additive; legacy columns remain)

## Conditioning Terminology

We use explicit naming for conditioning:

- `*_cond_*`: **conditional on playing/active** (worlds where the player is inactive are excluded)
- `*_uncond_*`: **unconditional** (inactive/DNP worlds contribute value `0`)

Decision metrics (optimizer objective and contest sim) must use `*_uncond_*`.

## Canonical Fields

### Identity / join keys
Required for joining across consumers:

- `player_id` (string/int; join key across projections + worlds + optimizer)
- `game_id`, `team_id` (when available)
- `player_name`, `team_tricode`, `opponent_team_tricode` (display / salary join)

### Play probability

- `p_play_raw`
  - Definition: raw play probability from the minutes model layer (`play_prob`)
  - Source: minutes model output (`minutes_v1`)
  - Intended use: diagnostic only

- `minutes_sim_p_active`
  - Definition: realized active rate across sim worlds (fraction of worlds with player active)
  - Source: sim_v2 worlds aggregation (`sim_p_active`)
  - Intended use: decision-relevant; used to interpret unconditional moments

- `p_play_eff`
  - Definition: effective play probability used for downstream decision metrics
  - Invariant: when sim outputs exist, `p_play_eff == minutes_sim_p_active`
  - Intended use: fallback conversion when only conditional moments exist

### Minutes

#### Model minutes (diagnostic)
- `minutes_cond_p10`, `minutes_cond_p50`, `minutes_cond_p90`
  - Definition: minutes distribution quantiles **given the player plays**
  - Source: minutes model output (`minutes_pXX_cond`)
  - Use: UI diagnostics; not used for optimizer objective

#### Sim minutes (decision)
- `minutes_sim_uncond_mean`, `minutes_sim_uncond_p10`, `minutes_sim_uncond_p50`, `minutes_sim_uncond_p90`, `minutes_sim_uncond_std`
  - Definition: minutes distribution moments with **DNP=0** (unconditional across all worlds)
  - Source: sim_v2 worlds aggregation (legacy columns `minutes_sim_*_uncond`)
  - Use: decision metric minutes (e.g., UI \"used by downstream\", override baselines)

- `minutes_sim_cond_mean`, `minutes_sim_cond_p10`, `minutes_sim_cond_p50`, `minutes_sim_cond_p90`, `minutes_sim_cond_std`
  - Definition: minutes distribution moments **conditional on playing**
  - Source: sim_v2 worlds aggregation (legacy columns `minutes_sim_*`)
  - Use: diagnostics only

### Fantasy Points (DK)

#### Sim FPTS (decision)
- `fpts_sim_uncond_mean`, `fpts_sim_uncond_p05`, `fpts_sim_uncond_p50`, `fpts_sim_uncond_p90`, `fpts_sim_uncond_p95`, `fpts_sim_uncond_std`
  - Definition: DK FPTS distribution moments with **DNP=0** (unconditional across all worlds)
  - Source: sim_v2 worlds aggregation (legacy columns `dk_fpts_*_uncond`)
  - Use: **optimizer objective** and decision displays

#### Sim FPTS (diagnostic)
- `fpts_sim_cond_mean`, `fpts_sim_cond_p05`, `fpts_sim_cond_p50`, `fpts_sim_cond_p90`, `fpts_sim_cond_p95`, `fpts_sim_cond_std`
  - Definition: DK FPTS distribution moments **conditional on playing**
  - Source: sim_v2 worlds aggregation (legacy columns `dk_fpts_*`)
  - Use: diagnostics only

## Field Mapping (Legacy -> Canonical)

The unified projections parquet contains legacy sim_v2 names. The canonical bundle adds aliases:

- `play_prob` -> `p_play_raw`
- `sim_p_active` -> `minutes_sim_p_active` and (when present) `p_play_eff`
- `minutes_p50_cond` -> `minutes_cond_p50` (and p10/p90)
- `minutes_sim_mean_uncond` -> `minutes_sim_uncond_mean` (and p10/p50/p90/std)
- `minutes_sim_mean` -> `minutes_sim_cond_mean` (and p10/p50/p90/std)
- `dk_fpts_mean_uncond` -> `fpts_sim_uncond_mean` (and p05/p10/p25/p50/p75/p90/p95/std)
- `dk_fpts_mean` -> `fpts_sim_cond_mean` (and p05/p10/p25/p50/p75/p90/p95/std)

Note: the canonical contract uses `*_uncond_*` ordering consistently; legacy columns are kept for backwards compatibility.

## Consumer Requirements

### Dashboard (Minutes page + GameView)
- Primary displayed decision metrics must come from:
  - `fpts_sim_uncond_mean`
  - `minutes_sim_uncond_mean` (and optionally `minutes_sim_uncond_p50`)
  - `minutes_sim_p_active`
- If showing conditional values, label explicitly as conditional:
  - `fpts_sim_cond_mean`, `minutes_cond_p50`, etc.

### Optimizer (QuickBuild)
- Objective projection must use `fpts_sim_uncond_mean`.
- Optional distribution fields:
  - `fpts_sim_uncond_std` (for stddev)
  - `fpts_sim_uncond_p90` (for upside)
- If unconditional fields are missing (older runs), fall back to:
  - `fpts_sim_cond_mean * p_play_eff` (best-effort; `p_play_eff` prefers `minutes_sim_p_active`)

### Contest Simulation
- Must use the **worlds matrix** corresponding to the same projection bundle:
  - resolve `sim_run_id` from the unified projections bundle and load worlds under
    `artifacts/sim_v2/worlds_fpts_v2/game_date=<date>/run=<sim_run_id>/...`

## Invariants (Assertions)

All consumers may assume:

1. Probabilities are clipped to `[0, 1]`:
   - `p_play_raw`, `p_play_eff`, `minutes_sim_p_active`
2. Unconditional decision metrics are non-negative:
   - `minutes_sim_uncond_* >= 0`, `fpts_sim_uncond_* >= 0`
3. When sim outputs exist:
   - `p_play_eff == minutes_sim_p_active`
4. DNP implies zeros in unconditional moments:
   - if `minutes_sim_p_active == 0`, then `minutes_sim_uncond_mean == 0` and `fpts_sim_uncond_mean == 0`

## Implementation References

- Canonical assembly: `projections/projections_bundle.py`
- Dashboard endpoint: `projections/api/minutes_api.py` (`GET /api/minutes`)
- Optimizer projections loader: `projections/api/optimizer_service.py` (`load_projections_for_date`, `build_player_pool`)
- Contest sim worlds selection: `projections/contest_sim/contest_sim_service.py` (`run_contest_simulation`)
