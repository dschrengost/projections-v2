# PR5 Spec: Minutes Worlds (Transformer-First, Sim-Integrated)

This is an implementation-ready spec + task list for PR5, built from the de-risk findings in:
- `docs/minutes/minutes_worlds_pr5_derisk.md`

Non-goals for this spec:
- No Prefect default behavior changes.
- No deployment steps.
- No new standalone minutes-world generator outside sim (avoid double-noise).

## Goals

1) Generate minutes worlds using the rotation_set transformer (allocator of record) and integrate into sim as the single minutes-worlds source of truth.
2) Publish minutes quantiles that are worlds-derived for UI/ops compatibility, with explicit semantics (uncond vs cond).
3) Preserve sim determinism and avoid double-sampling minutes noise.
4) Establish calibration guardrails (coverage gating) and robust degrade behavior.

## PR5 Must-Haves (verbatim)

1. Add a "Minutes Quantiles Contract" section:
- minutes_pXX: unconditional from worlds (includes DNP zeros)
- minutes_pXX_cond: conditional given plays
- UI chooses which to display; default display should be unconditional.

2. Require retrain (or explicit calibration mapping) for gate semantics:
- gate head target must match ROTATION_THRESHOLD_MINUTES (5.0) and/or expose a second head for play (1.0)
- temperature scaling is acceptable only as a short-term patch.

3. Make play_prob missingness a hard degradation:
- no fillna(1.0) for PR5 backend; instead degrade or fail.

4. Bench-zero mixture decision becomes empirical:
- keep it on until you show the new backend reproduces cameo/zero mass.

5. Calibration prerequisite:
- odds coverage must exceed a threshold (e.g., >=80% games in window) OR use a fallback calibration that doesn't depend on odds.

## Minutes Quantiles Contract (authoritative)

Definitions (per player, per slate/run):
- `minutes_p10`, `minutes_p50`, `minutes_p90`: UNCONDITIONAL minutes quantiles from minutes worlds, where DNP worlds contribute 0 minutes.
- `minutes_p10_cond`, `minutes_p50_cond`, `minutes_p90_cond`: CONDITIONAL minutes quantiles from minutes worlds, conditioning on "plays" (active worlds).

Required invariants:
- `minutes_pXX` must be computed from the exact minutes worlds used for FPTS worlds (no separate generator).
- If "plays" is sampled (active mask), then:
  - conditional quantiles computed over active worlds only
  - unconditional quantiles computed over all worlds (inactive => 0)

Consumer guidance:
- Dashboard default display should be unconditional quantiles (more decision-relevant, matches DNP-as-0 semantics).
- UI can optionally show conditional quantiles as an alternate view for role analysis.

## Integration Architecture (single source of truth)

Integration point (existing sim, new backend):
- Add a new minutes sampling backend inside `scripts/sim_v2/generate_worlds_fpts_v2.py` at the existing minutes sampling switch.
- Implement the new backend in a new module:
  - `projections/sim_v2/minutes_worlds_model_space_v1.py`

High-level backend behavior:
- Input: rotation_set model outputs (minutes mean allocation) + aux (gate logits/prob, share logits, router probs) + play_prob + game context.
- Output: `minutes_worlds` array shaped (W, P) for the sim chunk.
- The backend MUST be mutually exclusive with:
  - structured minutes noise (`sample_minutes_noise_per_world`)
  - game scripts (`sample_minutes_with_scripts`)
  - fallback split-normal

Determinism rules:
- Only use the per-chunk RNG created by sim (`rng = np.random.default_rng(date_seed + chunk_start)`).
- Never touch global RNG.
- Do not change RNG draw counts for other backends.

## Gate Semantics Requirements

We need explicit semantics for two distinct probabilities:
- `p_play`: probability player appears in the game at all (minutes >= 1.0).
- `p_rotation`: probability player is "in rotation" (minutes >= ROTATION_THRESHOLD_MINUTES).

Requirement:
- Rotation-set model must either:
  1) retrain gate head target to match `ROTATION_THRESHOLD_MINUTES=5.0`, and add a separate play head for `>=1.0`, or
  2) provide an explicit, documented calibration mapping from existing gate logits to these semantics (short-term only).

Short-term allowed:
- Temperature scaling on a held-out window is acceptable only as a bridge. It does not replace retraining if we change target semantics.

## play_prob Missingness (hard degradation)

PR5 backend must not silently assume `play_prob=1.0`.

Policy:
- If required play probability inputs are missing/invalid for a slate:
  - either fail the minutes-worlds backend and fall back to a legacy minutes sampler
  - or emit a degraded flag in outputs and skip publishing minutes_pXX as authoritative

This must be wired so the failure is visible and testable (not a silent fillna).

## Bench-Zero Mixture (empirical decision)

Default for PR5:
- Keep `bench_zero_mixture` enabled until the new backend demonstrably reproduces cameo/zero mass (by comparing distributions).

Evaluation criteria (minimum):
- fraction of zero-minute worlds for low-minute bench players is within a tolerance band vs historical sim behavior
- top8 minutes share and tail minutes metrics are not systematically shifted

## Calibration Prerequisite (coverage gating)

We will not use odds-dependent calibration unless odds coverage is high.

Coverage check:
- Define `odds_coverage = (# games with pre-tip spread+total) / (# schedule games in window)`.
- Require `odds_coverage >= 0.80` to fit/refresh spread->margin and OT models from odds.

Fallback calibration (no odds required):
- Use a margin distribution calibrated on realized margins only (season and recency-bucketed), or
- Use fixed priors (current sim defaults) with explicit degraded flag until odds coverage is sufficient.

## Implementation Boundaries (module responsibilities)

1) `projections/rotation/set_model.py`
- Add an opt-in inference path to expose aux outputs (gate logits, share logits, router probs).
- Keep current default runtime behavior unchanged unless explicitly enabled by caller/config.

2) `projections/sim_v2/minutes_worlds_model_space_v1.py` (new)
- Pure function(s) to sample minutes worlds given:
  - baseline minutes mean allocation
  - aux tensors/columns
  - play_prob and active_mask
  - game context + calibration knobs
  - rng

3) `scripts/sim_v2/generate_worlds_fpts_v2.py`
- Add a new backend selection branch, wired to sim profile config.
- Compute and publish both unconditional and conditional minutes quantiles according to the contract.

4) Unified outputs + UI (follow-up)
- Ensure `minutes_pXX` and `minutes_pXX_cond` semantics are respected across:
  - `projections/cli/finalize_projections.py`
  - `projections/api/minutes_api.py`
  - `web/minutes-dashboard/`

## Task List (actionable)

### 0) Config + constants
- Define `ROTATION_THRESHOLD_MINUTES = 5.0` in a single shared place (new module or existing config), and ensure sim/UI/ops all import or mirror it intentionally.
- Add sim profile option: `minutes_worlds_mode: "model_space_v1"` and a nested config block for knobs + calibration policy.

### 1) Expose transformer aux outputs (opt-in)
- Add `return_aux` (or equivalent) to `RotationSetMinutesPredictor.predict`:
  - Default `False` to avoid runtime behavior changes.
- Return columns:
  - `gate_logit`, `gate_prob` (sigmoid), `share_logit`
- Return group-level router outputs in a structured way (decide one):
  - separate dataframe keyed by `(game_id_norm, team_id)`, or
  - per-row broadcast with `group_id` and `router_pi_*` JSON blobs

Tests:
- Unit test: aux outputs are present when enabled and absent when disabled.

### 2) Add new sim backend (no double-noise)
- Create `projections/sim_v2/minutes_worlds_model_space_v1.py` with:
  - API: `sample_minutes_worlds_model_space_v1(..., rng) -> minutes_worlds`
  - Zero side effects; deterministic given rng.
- Wire it into `scripts/sim_v2/generate_worlds_fpts_v2.py`:
  - backend selection at existing minutes sampling branch
  - ensure mutual exclusion with existing backends

Tests:
- Determinism test: same seed + inputs -> identical minutes_worlds.
- "No double-noise" test: when model_space_v1 enabled, structured noise/game scripts/fallback not invoked.

### 3) Minutes quantiles contract in sim outputs
- In sim aggregation step, compute and publish:
  - unconditional minutes p10/p50/p90 from all worlds (including zeros)
  - conditional minutes p10/p50/p90 conditioned on "plays"
- Confirm the output schema includes both variants with stable names:
  - `minutes_p10/p50/p90` and `minutes_p10_cond/p50_cond/p90_cond` (or a consistent naming scheme).

Tests:
- Contract test on a tiny synthetic example:
  - known worlds matrix + known active mask -> exact quantiles.

### 4) play_prob missingness hard degradation
- Remove `fillna(1.0)` behavior for PR5 backend path in sim availability sampling.
- Implement degradation policy:
  - fail fast for PR5 backend, or
  - fall back to legacy backend + set a degraded flag.

Tests:
- If `play_prob` is missing: PR5 backend does not silently proceed with ones.

### 5) Gate semantics retrain or explicit mapping
- Decide approach:
  1) Retrain rotation_set with:
     - gate head target `minutes >= 5.0` (rotation)
     - second head for `minutes >= 1.0` (play)
     - recency weighting (half-life)
  2) Temporary mapping:
     - temperature scaling on held-out labels for the chosen thresholds

Artifacts:
- Manifest must record:
  - thresholds used for targets
  - recency half-life
  - calibration method (if any)

### 6) Bench-zero mixture empirical evaluation
- Keep `bench_zero_mixture` enabled by default during rollout.
- Add audit metrics in sim manifest (or logs) comparing:
  - zero-mass rate for low-minute players
  - top8 share, tail minutes, entropy

### 7) Calibration gating + fallback
- Implement odds coverage audit:
  - compute `odds_coverage` for the calibration window before fitting odds-dependent parameters
- If below threshold:
  - use fallback calibration and set degraded flag

### 8) UI + API alignment (follow-up tasks)
- UI:
  - default to unconditional minutes (minutes_pXX) display
  - add toggle to show conditional minutes (minutes_pXX_cond)
- API/unified:
  - ensure the published `minutes_pXX` fields reflect the contract, not legacy model-authored tails.

## Rollout / Safety

- Feature-gate the new backend via sim profile config.
- Keep existing backends available for fallback.
- Add explicit degraded flags so ops/UI can see when PR5 backend is not active.

