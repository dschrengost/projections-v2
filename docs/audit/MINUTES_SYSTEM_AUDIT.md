# Minutes System Audit (minutes_v1 → sim_v2) — Structural Realism Focus

Goal: explain how minutes flow through the system today, identify *where semantics drift* (conditional vs unconditional) and *why the 240-minute constraint introduces systematic distortions*, then propose fixes ranked by intrusiveness.

This is an audit, not an implementation. References below point to the current code paths that define behavior.

---

## Executive Summary (What’s Actually Happening)

### Key semantic fact
`minutes_v1` predicts **minutes conditional on playing** (minutes \| active), plus a separate `play_prob` (probability of playing). The simulator (`sim_v2`) correctly uses `play_prob` to sample who is active in each world, but then **reports minutes and FPTS aggregates conditional on being active** (i.e., E[· \| plays]).

That is internally consistent, but several downstream consumers treat those conditional aggregates as if they were unconditional, which creates the “low play-prob guys still have real minutes” symptom.

### Key mechanical fact
In `sim_v2`, the 240-minute enforcement is not a symmetric “scale everyone” step. With the current call pattern it:
- Caps each team to **10 players per world** (hard-coded in the driver).
- When a team is over 240 minutes, it **keeps starters fixed and scales only non-starters** to hit 240.

This creates a *systematic bench compression*, especially for high-minute bench players (6th men) whose raw conditional p50 is ~28–32 but whose simulated minutes are frequently scaled down to whatever “leftover after starters” happens to be.

---

## Glossary (The Source of Most Confusion)

- **Conditional minutes**: minutes distribution *given the player is active* (plays / suits up). In the codebase this is what `minutes_p10/p50/p90` represent for `minutes_v1`.
- **Unconditional minutes**: minutes distribution across all outcomes, including 0 when the player is inactive. For a player with `play_prob < 0.5`, the unconditional p50 is typically 0.
- **Rotation vs active**: a player can be “active” (not OUT) but still record 0 minutes (coach decision / not in rotation). This is distinct from `play_prob` and is only partially modeled today (heuristics like `rotation_prob`, plus caps).

---

## 1) How It Works — Minutes End-to-End

### 1.1 minutes_v1 scoring outputs (raw model)
Primary entrypoint: `projections/cli/score_minutes_v1.py`

Per player-row, the scorer produces:
- **Conditional minutes quantiles**:
  - `minutes_p10`, `minutes_p50`, `minutes_p90`
  - and aliases: `minutes_p10_cond`, `minutes_p50_cond`, `minutes_p90_cond`
- **Play probability**: `play_prob` (separate head; forced to 0 for known OUT)
- **Starter flag**: `is_starter` derived from starter signals

Important details:
- The scorer explicitly treats minutes quantiles as **minutes | plays**:
  - It logs that “p_play mixing removed: minutes quantiles represent minutes | plays” (see `enable_play_prob_mixing` handling).
- The “unconditional” columns in the minutes artifact are currently **not truly unconditional**:
  - `_attach_unconditional_minutes()` simply copies conditional quantiles to `minutes_pXX_uncond`.
  - There is no `minutes_uncond = play_prob * minutes_cond` conversion in the scorer today.
- There is an optional per-team p50 L2 reconciler (`projections/minutes_v1/reconcile.py`) designed to enforce ~240 at the p50 level via a small convex QP. It is configurable but typically disabled (`config/minutes_l2_reconcile.yaml` says “Reconciliation disabled by request”).

### 1.2 Where `play_prob` enters (and where it doesn’t)
`play_prob` exists as a column but is not automatically combined with the conditional minutes quantiles into an unconditional quantity in `minutes_v1` outputs.

So downstream systems must decide:
- Do they want **conditional** projections (E[· | plays]) and keep `play_prob` separate?
- Or do they want **unconditional** projections (E[·]) and multiply by `play_prob` (or model the 0-mass explicitly)?

This distinction is not enforced by the schema today; it’s a consumer responsibility.

### 1.3 sim_v2 loads minutes and rates and builds a baseline mean
Primary entrypoint: `scripts/sim_v2/generate_worlds_fpts_v2.py` (called by `scripts/sim_v2/run_sim_live.py`)

High level:
1. Load minutes projections for the date (prefers `artifacts/minutes_v1/daily/<date>/run=<id>/minutes.parquet`, else gold).
2. Choose a minutes column using `_resolve_minutes_column()`:
   - prefers `minutes_p50_cond`, then `minutes_p50`, then `minutes_pred_p50`.
   - This means sim baseline minutes are typically **conditional p50**.
3. Load rates (`rates_v1_live`) and join to build per-player mean stat totals and mean FPTS.
4. Filter out players with `play_prob < min_play_prob` (default 0.05).

### 1.4 sim_v2: per-world simulation flow (where 240 happens)
Inside the sim loop (`scripts/sim_v2/generate_worlds_fpts_v2.py`), for each chunk of worlds:

1) **Sample availability**
- `active_mask[w, i] ~ Bernoulli(play_prob_i)`

2) **Sample conditional minutes**
- If game scripts enabled (default), use `projections/sim_v2/game_script.py:sample_minutes_with_scripts()`:
  - chooses a target quantile per world based on script (close/blowout/etc)
  - samples from a split-normal fit to (p10,p50,p90)
  - treats “high-minute bench” (p50 ≥ rotation threshold, default 20) as “rotation” for script selection

3) **Convert conditional minutes → unconditional minutes (world-level)**
- `minutes_worlds *= active_mask`
- This is the *explicit* conditional→unconditional conversion point in the simulator.

4) **Enforce team total minutes ≈ 240**
- If `enforce_team_240` is enabled (baseline profiles: true), call:
  - `projections/sim_v2/minutes_noise.py:enforce_team_240_minutes()`
- This is the key distortion step (see Section 2).

5) **Convert minutes to stat totals and FPTS**
- stat totals are computed as `(rate_per_min * minutes_worlds)` plus noise.
- (optional) usage shares redistribution can further couple teammate outcomes, but minutes totals are already fixed by step (4).

### 1.5 sim_v2 aggregation: what’s written out (conditional again)
At the end, `generate_worlds_fpts_v2.py` aggregates across worlds *only where the player is active*:
- It computes **conditional mean/quantiles**:
  - `minutes_sim_mean`, `minutes_sim_p10/p50/p90` are computed over `mins_active = all_minutes[active_worlds_p, p]`
  - `dk_fpts_mean` and `dk_fpts_pXX` are computed over `fpts_active`
- It also writes `play_prob` alongside these conditional aggregates.

This is explicitly stated in the code comments: “Compute CONDITIONAL statistics (only worlds where player is active)… This is what DFS lineup builders want: E[FPTS | plays].”

Implication:
- A player with `play_prob = 0.10` can legitimately have `minutes_sim_p50 ≈ 20` (because it is p50 **conditional on playing**).
- The *unconditional* median minutes for that player would be 0, but that is not what `minutes_sim_p50` represents today.

Downstream note: several consumers treat `minutes_sim_*` / `dk_fpts_*` as if they were unconditional inputs (e.g., optimizer selection and some evaluation scripts). If that’s intentional, the missing step is the combination `E[X] = play_prob * E[X | plays]` (or explicit inclusion of the 0-mass in percentiles).

---

## 2) How the 240-Minute Enforcement Works (and how bench minutes “compete”)

Implementation: `projections/sim_v2/minutes_noise.py:enforce_team_240_minutes()`

### 2.1 Important driver behavior: hard-coded rotation cap
In `scripts/sim_v2/generate_worlds_fpts_v2.py`, the call always passes:
- `max_rotation_size=DEFAULT_MAX_ROTATION_SIZE` where `DEFAULT_MAX_ROTATION_SIZE = 10`

This means the function always runs in “rotation-capped reconciliation” mode (not the legacy proportional scaling path), regardless of profile config.

### 2.2 Rotation-capped algorithm (per team, per world)
For each team in each world:

1) Determine active players (from `active_mask`)
2) **Select who is allowed to receive minutes** (“kept rotation”):
   - Keep *all starters*.
   - Fill remaining slots up to K (default 10) with the top non-starters by baseline minutes (`baseline_minutes`, typically minutes p50).
   - Non-kept active players are hard-zeroed.

This is the first “competition”: fringe bench players compete for the last rotation slots.

Implementation note: in this capped mode, the `rotation_mask` / `bench_mask` arguments are effectively unused; “who is in rotation” is determined by starter status + baseline-minute rank plus the hard cap.

3) Compute:
   - `sum_starters = Σ minutes(starters_kept)`
   - `sum_bench = Σ minutes(nonstarters_kept)`
   - `sum_total = sum_starters + sum_bench`

4) If `sum_total > 240` (oversubscribed):
   - If `sum_starters < 240`, **starters are left unchanged** and bench is scaled to fit:
     - `bench_scale = (240 - sum_starters) / sum_bench`
     - `minutes_bench *= bench_scale`
   - Only if starters alone exceed 240 (rare) does it scale starters down.

This is the second “competition”: all non-starters compete for the *residual minutes after starters*.

5) If `sum_total < 240` (undersubscribed):
   - Scales everyone up by `240/sum_total`, but clamps scale to at most 1.3.

### 2.3 Consequences of the starter/bench split
This enforcement is asymmetric:
- On oversubscription, starters are protected and bench absorbs the full adjustment.
- On undersubscription, everyone is scaled up, but the “make-up” is limited by the 1.3 clamp (so the system can fail to reach 240 exactly in some worlds).

This asymmetry is the core reason the enforcement step can create systematic bench compression even if the underlying minutes model is reasonable.

---

## 3) Failure Mode Analysis (Mechanisms → Symptoms)

### 3.1 Bench rotation minutes flattening
**Symptom:** high-minute bench pieces with raw conditional p50 ~28–32 end up around ~20–22 in simulated p50 outcomes.

**Primary mechanism:** starter-protected oversubscription resolution.
- Worlds frequently sample `sum_total > 240` because minutes are sampled independently per player (plus game scripts often push starters upward in close games).
- When oversubscribed and `sum_starters < 240`, the reconciler computes:
  - `bench_scale = (240 - sum_starters)/sum_bench`
  - and applies it *uniformly to all non-starters*.
- If starters already “claim” ~180–205 minutes, the bench is mechanically forced into the remaining ~35–60 minutes, regardless of whether the true rotation has a heavy 6th man.

**Why this flattens specifically at ~20–22:**
- If a 6th man’s sampled conditional minutes is ~30 and bench_scale frequently lands around ~0.65–0.75, you get ~19.5–22.5 as a typical post-enforce outcome.

**Secondary amplifiers:**
- Game scripts treat high-minute bench as “rotation” for *sampling* (they can sample from starter quantiles in close games), but the 240 enforcement still treats them as non-starters for *scaling*.
- `max_rotation_size=10` hard-caps who can receive minutes, which can increase the frequency of “bench minutes as a pooled residual” behavior (the tail is removed, leaving fewer bench players to absorb scaling).

### 3.2 Why high-minute bench players lose minutes disproportionately vs starters
**Mechanism:** the reconciliation objective is effectively “don’t change starters unless forced”.
- Starters are never dropped by the cap, and are not scaled down when oversubscribed unless starters alone exceed 240.
- Any over-allocation created by both starter and bench sampling is paid for exclusively by non-starters.

This encodes an assumption that starters are rigid and bench is the only flexible source of reduction, which is not true for many real rotations (e.g., staggered stars, 6th men closing games, foul trouble variance).

### 3.3 Conditional vs unconditional minutes mishandling (and “ghost minutes”)
There are two separate issues that look similar:

#### A) Output semantics: sim_v2 aggregates are conditional
`generate_worlds_fpts_v2.py` outputs `minutes_sim_*` and `dk_fpts_*` as **conditional on being active**.

If a downstream consumer interprets `minutes_sim_p50` as an unconditional median, then:
- Players with low `play_prob` will appear to have meaningful minutes even when the true unconditional p50 is 0.

This shows up explicitly in evaluation tooling:
- `scripts/analyze_accuracy.py` uses `minutes_sim_p50 >= 10` as a DNP false-positive detector, which will over-flag low-`play_prob` players because it is using a conditional statistic.

This also matters in lineup/optimizer plumbing if it treats `minutes_sim_*` / `dk_fpts_*` as unconditional EV inputs without multiplying by `play_prob` (see `projections/api/optimizer_service.py` minutes-column selection).

#### B) Missing explicit unconditional columns
The minutes scorer produces `minutes_pXX_uncond`, but currently those columns are copies of conditional quantiles (no `* play_prob` adjustment).

So even “uncond”-named columns can be accidentally treated as unconditional while still being conditional.

### 3.4 “240 minutes enforced mechanically” without preserving role hierarchy / tail risk
The current approach is:
- sample independent conditional minutes per player
- hard-zero inactives
- hard-cap rotation size
- enforce 240 with a fixed starter/bench split rule

What it does *not* encode:
- lineup/position constraints (who substitutes for whom)
- correlated minutes substitution (if X plays more, Y plays less)
- bench depth asymmetry (some teams run 11–12; others run 8–9)
- realistic tail behavior (garbage time units, spot stints)

As a result, the enforcement step becomes the de facto “rotation model”, and the particular choice of enforcement heuristic materially shapes the distribution.

---

## 4) Recommendations (Ranked by Intrusiveness)

Each recommendation lists:
- **Assumption changed**
- **Why it addresses the failure modes**
- **Tradeoffs**

### 4.1 Minimal / surgical changes

1) **Make conditional vs unconditional explicit in simulator outputs**
- Assumption changed: downstream consumers no longer infer semantics from column names.
- Why: fixes “low play-prob players have 10+ minutes” by exposing *both*:
  - `minutes_sim_*_cond` (current behavior)
  - `minutes_sim_*_uncond` (include inactive worlds as 0 and compute true unconditional quantiles/means)
  - similarly for `dk_fpts_*`
- Tradeoffs: changes numbers in dashboards/optimizer unless they’re updated; requires clear migration and backward-compat mapping.

2) **Compute true unconditional minutes columns in minutes_v1 artifacts**
- Assumption changed: `*_uncond` columns actually mean unconditional.
- Why: prevents accidental misuse and enables consistent “expected minutes” calculations across the system.
- Tradeoffs: depends on `play_prob` calibration; if `play_prob` is imperfect, unconditional minutes inherit that bias.

3) **Stop hard-coding `max_rotation_size=10` in the sim driver**
- Assumption changed: rotation-cap behavior becomes profile-driven (or can be disabled).
- Why: removes a hidden lever that shapes bench depth and minutes redistribution; allows A/B of “cap vs no cap”.
- Tradeoffs: more configuration complexity; disabling cap may increase tail minutes and require stronger rotation modeling elsewhere.

4) **Change the oversubscription rule from “starters fixed, bench pays” to a role-aware but not starter-binary scheme**
Examples of “still surgical” options:
- protect top-N players by baseline minutes (includes 6th men), not just starters
- or apply a weighted scaling / projection that allows some starter minutes to move when needed
- Why: directly targets bench flattening; restores heavy-bench realism where appropriate.
- Tradeoffs: can move starters more than today; might reduce “DFS conditional if plays” sharpness for starters, but improves structural realism.

### 4.2 Structural modeling changes (minutes model / semantics)

1) **Add an explicit “in_rotation | active” probability (separate from play_prob)**
- Assumption changed: “active” is not equivalent to “will play minutes”.
- Why: prevents non-rotation actives (coach DNPs) from receiving non-trivial simulated minutes and from competing in the 240 step.
- Tradeoffs: requires labels and retraining; may increase variance if poorly calibrated early.

2) **Model minutes as a compositional allocation (shares) rather than independent absolute draws**
- Assumption changed: minutes are inherently constrained and substitution-coupled within a team.
- Why: produces 240 by construction and reduces reliance on ad-hoc reconciliation; can preserve hierarchy via priors/weights.
- Tradeoffs: larger model change; may reduce interpretability and require new calibration strategy for tails.

3) **Joint calibration of (play_prob, minutes_cond) so unconditional expectations are stable**
- Assumption changed: unconditional minutes/fpts are first-class outputs, not an afterthought.
- Why: improves EV realism for risky players; makes downstream “multiply by play_prob” less brittle.
- Tradeoffs: needs careful evaluation to avoid over-penalizing Q-tag players (calibration risk).

### 4.3 Simulator-level architectural changes

1) **Replace the current 240 enforcement with a per-world convex projection (QP)**
Sketch: for each team/world, solve:
`min Σ w_i (m'_i - m_i)^2  s.t. Σ m'_i = 240,  0 ≤ m'_i ≤ 48, (optional floors/caps)`
- Assumption changed: minutes enforcement preserves relative roles via weights rather than a hard starter/bench split.
- Why: eliminates systematic bench flattening while still enforcing 240; matches the philosophy of the minutes_v1 L2 reconciler, but applied per world.
- Tradeoffs: more compute (but team sizes are small; can be optimized/cached); requires weight design and guardrails.

2) **Hierarchical allocation: sample team “starter vs bench budget” then allocate within groups**
- Assumption changed: substitution happens at the group level (units/roles), not uniformly across all non-starters.
- Why: preserves realistic bench depth and allows heavy-bench scenarios (6th men closing) without being crushed by a starter-fixed rule.
- Tradeoffs: adds modeling assumptions; needs tuning per team and may require depth chart inputs.

3) **Replacement modeling for absences (vacancy-driven minutes substitution)**
- Assumption changed: missing minutes flow to specific replacements, not proportionally to “whoever is active”.
- Why: aligns tail behavior and bench asymmetry with real coaching patterns; reduces the need for large global scaling that flattens.
- Tradeoffs: requires depth chart / position mapping and can be operationally heavy.

---

## 5) Practical Next Diagnostic Questions (Before Changing Anything)

1) How often is a team oversubscribed pre-enforce (sum minutes > 240) vs undersubscribed, and what is the distribution of `bench_scale`?
2) For the teams where 6th-man flattening occurs, what is the typical `sum_starters` pre-enforce? Is the bench consistently acting as the residual sink?
3) Are downstream consumers (optimizer, eval scripts) intentionally using conditional projections, or do they actually want unconditional EV? If unconditional, where is the combination step with `play_prob` supposed to live?

The repo already has a purpose-built diagnostic script that can help quantify (1)–(2):
- `scripts/diagnostics/minutes_sim_vs_pred_audit.py`
