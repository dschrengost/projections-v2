# Play probability & active-worlds coherence (sim_v2 / sim_v3)

## What we were seeing

- Many rotation players were “active” in ~100% of worlds (desired).
- Some fringe players showed apparent contradictions (e.g. `play_prob≈0.76` but only ~13% of worlds “active”).
- Some players had `play_prob≈0.76`, `minutes_mean_uncond≈23`, yet ~99.5% “active” (suggesting an override / different prob being used).

## Key finding: sim_v3 is intentionally two-regime (plus a third gate)

In the production sim profile (`sim_v3` in `config/sim_v2_profiles.json`), the simulator is *explicitly two-stage*:

1. **Availability sampling (Bernoulli)**  
   Sample per-player availability using an **effective** probability `play_prob_eff`:
   - derived from minutes inputs’ `play_prob` (`play_prob_raw`)
   - then modified by **play-prob policy** (rotation locks floored to ~0.995, probable floor, OUT→0).  
   Code: `projections/sim_v2/play_prob_policy.py` via `apply_play_prob_policy_with_diagnostics()` and
   `scripts/sim_v2/generate_worlds_fpts_v2.py`.

2. **Core players (“rotation locks”) are protected**  
   “Core” players are identified by a deterministic heuristic (`compute_rotation_lock_mask`):
   - starters, top‑K by baseline minutes, or minutes ≥ threshold.  
   These locks are used by the play-prob policy and (optionally) feasibility gating.  
   Code: `projections/sim_v2/minutes_physics.py#compute_rotation_lock_mask`,
   `projections/sim_v2/play_prob_policy.py`.

3. **Bench-zero mixture (fringe gate) conditional on availability**  
   For low-minute players (`minutes_target < minutes_threshold`, default 10 in `sim_v3`), the sim applies a **mass-at-zero** drop with high probability (default `p_zero_base=0.85`, `p_zero_slope=0.35`).  
   This is why you can see:
   - `play_prob≈0.76` (availability) but `p(MIN≥1)≈0.13` (played)  
   because overall `P(played) ≈ P(available) * (1 - p_zero)` (plus feasibility restores).  
   Code: `projections/sim_v2/bench_zero_mixture.py#apply_bench_zero_mixture`,
   called from `scripts/sim_v2/generate_worlds_fpts_v2.py`.

## “Active” definitions (where they live)

| Component | “Active” definition | Field(s) | Threshold |
|---|---|---|---|
| Sim world generation (sim_v3) | “played” worlds used for conditional moments | `sim_p_active` | `minutes >= PLAY_THRESHOLD_MINUTES` (currently 1.0) |
| Sim availability (pre bench-zero) | “available after policy + feasibility” | `sim_p_available` | none (Bernoulli via `play_prob_eff`) |
| Sim meaningful rotation rate | “meaningful minutes” | `sim_p_rotation` | `minutes >= ROTATION_THRESHOLD_MINUTES` (currently 5.0) |
| Dashboard | Displays sim played rate | `minutes_sim_p_active` | shown as **p(MIN≥1)** |

## Play-prob-like fields (end-to-end)

**Minutes inputs**
- `play_prob` (minutes artifact / minutes API)  
  Origin: minutes model output. In sim_v3 it is treated as a *base availability input* (then policy + bench-zero apply).
- `rotation_prob` (minutes artifact / minutes API; if present)  
  Origin: minutes pipeline–derived “in-rotation” probability (often based on minutes quantiles / thresholds). In sim_v3 it is **not** the Bernoulli “available” gate; it is used for rotation/bench classification and some noise/shares heuristics.

**Sim policy / inputs**
- `play_prob_raw`: clipped version of `play_prob` used as policy input.
- `play_prob_eff`: probability used by the sim to sample availability (after policy).  
- `rotation_lock`: “core” boolean computed by `compute_rotation_lock_mask`.
- `play_prob_policy_reason`: best-effort reason string for policy action.

**Sim realized outputs**
- `sim_p_active`: realized “played” rate (share of worlds with minutes ≥ 1.0).  
- `sim_p_available`: realized availability draw rate (pre bench-zero).  
- `sim_p_rotation`: realized meaningful-minutes rate (minutes ≥ 5.0).

**Canonical unified fields**
- `minutes_sim_p_active`: “played” rate used in the dashboard.
- `p_play_raw` / `p_play_eff`: canonicalized play-prob columns in `projections/projections_bundle.py`
  (`p_play_eff` is treated as the downstream decision probability when sim outputs exist).

## Changes made to make things coherent

1. **Define “played/active” consistently for sim outputs**  
   `sim_p_active` is now computed from `minutes >= PLAY_THRESHOLD_MINUTES` (and conditional moments use the same mask),
   rather than relying on an internal mutable `active_mask`.

2. **Expose gating diagnostics in outputs** (and carry through to unified projections)
   - `play_prob_raw`, `play_prob_eff`, `rotation_lock`, `play_prob_policy_reason`
   - `sim_p_available`, `sim_p_rotation`
   - `bench_zero_p_zero`, `bench_zero_threshold_minutes`

3. **Dashboard labels + columns updated**  
   - `play_prob` shown as **p(avail)** (with tooltip explaining extra gates)
   - sim columns include **p(avail used)**, **p(drop|avail)**, **p(MIN≥5)**, **p(MIN≥1)**.

4. **Sanity-check invariants added** (warning logs)
   - availability draws vs `play_prob_eff`
   - `p_played` should not exceed `p_available`
   - `minutes_mean_uncond <= minutes_mean_cond`
   - “non-core nearly always played despite `play_prob_eff < 0.9`” warning

## Repro / diagnostic tooling

- Script: `tools/diagnose_sim_play_prob_coherence.py`  
  Loads a slate’s `minutes_matrix.parquet` + sim projections and prints a sample across the minutes distribution:
  `play_prob_display`, `play_prob_used`, `bench_zero_p_zero`, realized `p(MIN≥1)` / `p(MIN≥T)` and conditional/unconditional means.
