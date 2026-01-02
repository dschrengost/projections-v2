# Minutes v3 roadmap

Objective: solve “next man up” realism by predicting (1) who is in the rotation and (2) the minutes distribution of that rotation, while producing **team-feasible 240-minute outputs** that remain consistent under simulation.

This doc is the canonical minutes_v3 north star and decision log. If we add knobs, they must map to an explicit modeling assumption (not allocator folklore).

## Problem statement (what we are fixing)

Current failure modes (minutes_v1 + allocators/reconcile):
- **Bench minutes smear** when starters are OUT: too many bench/end-of-bench players get small non-zero minutes, stealing from true rotation guys.
- **End-of-rotation uncertainty** is handled inconsistently: per-player quantiles are sampled independently, then projected to team=240; projection changes the implied quantiles and often harms rotation concentration.

What we care about most:
- **Top ~9 accuracy**: if the top rotation players are right, we can tolerate ambiguity about who gets the last 0–12 “tail” minutes.
- **Tail minutes existence**: end-of-bench minutes should appear when called for (blowout risk, deep injuries, etc.) because they directly impact the top rotation minutes.

## Principles / invariants

- **No leakage**: only pre-lock / `as_of_ts` features.
- **Team feasibility**: minutes must sum to **240 per team-game** in each simulated world (and optionally in p50 outputs).
- **Consistency**: the mechanism used to generate quantiles must match the mechanism used in simulation.
- **Deterministic RNG**: given seed + inputs, outputs should be reproducible.
- **Minimal deps**: use existing stack (numpy/pandas/lightgbm/joblib); avoid new heavy probabilistic libs.

## Current state (as of Dec 2025)

- minutes_v1: quantile LightGBM; reconciliation happens downstream (sim).
- rotshare prototype + injury-regime evaluation exists:
  - Eval CLI: `projections/cli/eval_minutes_injury_regime.py`
  - Note: `docs/minutes/2025-12-29-rotshare-injury-regime-eval.md`
  - Summary: rotshare improves injury-regime bench-core / DNP errors; top-7 sum error largely unchanged.

## Proposed minutes_v3 direction: “quantiles from simulation”

Replace “independent per-player quantiles + later reconciliation” with a **team-joint generative model** that:
- samples **25k+ worlds per game** with team sum = 240 by construction
- produces `minutes_p10/p50/p90` as *summaries* of those same worlds

This makes quantiles and simulation consistent, avoids allocator hacks, and forces the model to learn rotation inclusion + concentration.

## Model sketch (minimal, shippable)

We model each **team-game** as having two latent “controls”:

1) **Tail minutes mass** (how much of 240 goes to end-of-bench / garbage / spot minutes)
   - Let `r ∈ [0, 1]` be tail fraction, `tail_minutes = 240 * r`, `core_minutes = 240 * (1 - r)`
   - `r` depends on team-game features:
     - injuries/starter_out counts
     - spread / total / blowout risk (pre-lock)
     - rest / back-to-back / travel proxies if available
   - We sample `r` per world (e.g., logistic-normal or Beta-like sampling using numpy), so tail minutes can vary across worlds.

2) **Rotation tightness** (how concentrated core minutes are among rotation players)
   - Predict per-player “core preference” logits `ℓ_i = f(x_i)` and a team-game temperature `τ = g(x_team)`
   - In a given world: `shares = softmax((ℓ + ε) / τ)` and `core_minutes_i = core_minutes * shares_i`
   - Smaller `τ` ⇒ tighter 7–8 man concentration; larger `τ` ⇒ deeper rotation.

Tail allocation (who gets tail minutes):
- We do **not** overfit tail identity. Allocate tail minutes across non-core players using a diffuse rule:
  - weights proportional to an “eligibility” score (e.g. play probability / in-rotation probability), optionally with mild exponent
  - or sample a simple Dirichlet-like draw via normalized Gamma (no new deps)

Optional “who is in rotation” component:
- Train a classifier `p_i = P(in_rotation)` and use it as a *soft prior* for core shares (not a hard top-N cut):
  - e.g., multiply softmax weights by `p_i^λ` or add `λ*logit(p_i)` to logits.

Key property: **no hard top-N** is required at inference. Rotation length emerges from `(r, τ, p_i)` and their dependence on features.

## Supervision / targets (aligned with top-9 objective)

We need explicit labels for the two team-level controls:

Tail minutes target (recommended first definition):
- For each team-game: `tail_minutes_actual = 240 - sum(actual minutes of actual top9)`
- This directly teaches: “don’t steal from the top9; if slack exists, put it in tail”.

Rotation tightness target:
- Not trained as a direct label; learn `τ` by minimizing distribution loss on core minutes:
  - normalize actual minutes into a target share vector (on the core portion)
  - train with KL / cross-entropy style loss to penalize smearing

Inclusion prior label (optional):
- `in_rotation = 1{minutes ≥ 10}` or `in_top9 = 1{player in actual top9}`
- Use this to shape which players get meaningful mass.

## Sampling → quantiles workflow

Per team-game, for W worlds (W≈25,000):
- sample `r_w` and compute `core_minutes_w`, `tail_minutes_w`
- sample logits noise `ε_w` and compute core shares → core minutes
- allocate tail minutes diffusely
- combine, clip to [0, 48] if needed, renormalize within team (preserving 240)

Quantiles are empirical percentiles across worlds:
- `minutes_p10/p50/p90` per player
- derived `play_prob` can be `P(minutes > 0)` or `P(minutes ≥ 10)` depending on downstream usage

Implementation note: sampling must be chunked (e.g., 512–2048 worlds per chunk) to keep memory bounded while still supporting 25k+ worlds.

## Evaluation (how we decide it works)

Primary metrics (aligned with objective):
- **Top-9 MAE**: MAE computed only on the 9 players with highest actual minutes (per team-game).
- **Top-9 sum error**: `abs(sum_pred_on_actual_top9 - sum_actual_top9)` (measures minutes stolen from core).
- **Tail minutes calibration**: MAE/bias for `tail_minutes` (as defined above).

Secondary metrics (diagnostics):
- injury-regime slices + non-injury slice (strictly healthy)
  - `projections/cli/eval_minutes_injury_regime.py` (extend/keep aligned)
- rotation depth metrics, bench concentration (Gini/HHI), DNP errors
- bucketed error by starters_out, by actual minutes bins

Validation protocol:
- Walk-forward monthly evaluation (train up to month-start, score month, report metrics).
- Explicit injury-regime focus, but guardrail: no meaningful regression on strict non-injury slice.

## Milestones

M0 — Metric alignment
- Add “top-9” and “tail_minutes” metrics to injury-regime eval and walk-forward runner.

M1 — Tail mass head
- Train a team-level model for `tail_minutes` (or tail fraction `r`) using pre-lock team features (including spread/total/blowout).
- Add sampling (variance) and validate tail calibration + top-9 sum error improvements.

M2 — Tightness head (`τ_team`)
- Predict `τ_team` from team features; train jointly to reduce smearing / improve top-9 MAE.

M3 — Inclusion prior (soft)
- Add `p_in_rotation` classifier (e.g. in_rotation minutes ≥ 10) as a soft prior for shares.
- Evaluate rotation membership / overlap for top-9.

M4 — Productionization
- Implement a minutes_v3 sampler interface used by sim (25k worlds) and a quantile-export mode for APIs/dashboards.
- Add deterministic RNG and artifact versioning.
- Shadow run and revert plan (flagged rollout).

## “Ship today” checklist (single-day execution plan)

The goal for today is to make **minutes quantiles be an output of simulation**, without rebuilding the entire pipeline.

**Step 1 — Add a minutes_v3 sampler (rotshare-backed)**
- Implement a Monte Carlo sampler that uses the existing rotshare artifacts (`RotationShareArtifacts`) to generate per-world minutes with team sum = 240.
- Output empirical `minutes_p10/p50/p90` per player from the sampled worlds.
- Deterministic: fixed seed + `(game_id, team_id)` ⇒ identical outputs.
- Guardrails: per-world team sums = 240 (within eps), caps respected.

**Step 2 — Wire sampler into scoring behind a flag**
- Extend `projections/cli/score_minutes_v1.py` to recognize `rotation_share_model.joblib` bundles.
- Add an opt-in switch (default off) so production behavior is unchanged:
  - `--rotshare-quantiles-mode point|mc` (default: `point`)
  - `--rotshare-n-worlds 25000` (default: 25000)
  - `--rotshare-concentration 60` (default: 60)
  - `--rotshare-seed 42` (default: 42)

**Step 3 — Add unit tests**
- Bundle loader recognizes rotshare bundles.
- MC sampler:
  - `p10 <= p50 <= p90` and all finite
  - deterministic for a fixed seed
  - per-world team totals sum to 240 (within 1e-6)
- Guardrail: rotshare scoring bypasses reconcile/upside (no double-reconciliation).

**Step 4 — Produce a live-slate artifact (shadow)**
- Run scoring with `--rotshare-quantiles-mode mc` for the target `--date`.
- Run the existing injury-regime eval / sanity check utilities on a short recent window.
- Spot check a few team outputs for realism (top-8/9 + tail minutes existence).

Example commands (local):
- Train (if needed): `uv run python -m projections.cli.train_rotation_share --help`
- Score daily minutes (MC tails): `uv run python -m projections.cli.score_minutes_v1 --date 2025-12-30 --bundle-dir <rotshare_bundle_dir> --rotshare-quantiles-mode mc --rotshare-n-worlds 25000`

## Open questions / decisions to record

- Tail label definition: “outside top9” vs “minutes < 10”. Default: outside top9 (more aligned).
- What downstream needs to consume:
  - per-world minutes for sim
  - quantiles for display / API
  - both from the same sampler (preferred)
- How to handle players marked OUT (hard zero at sampling time).
- Whether to model OT explicitly or treat OT as noise (initially treat as noise; keep 240 target).
