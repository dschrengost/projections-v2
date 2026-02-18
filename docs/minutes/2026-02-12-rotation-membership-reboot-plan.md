# Rotation Membership + Bench Minutes Reboot Plan (2026-02-12)

## Why This Exists

We are not blocked on injury/OUT handling.
The failure mode is among active players:

- Wrong rotation membership (who actually plays vs active-DNP).
- Wrong bench tail allocation (over-smear vs over-zero).
- Too much reliance on hard heuristics instead of model outputs.

This plan tracks the redesign so we stop iterating blindly.

## North Star

Train and serve a minutes model that is explicit about:

1. Rotation membership probability per player (`p_rot`).
2. Team rotation size expectation per game (`k_hat_team`).
3. Conditional share allocation among in-rotation players (`share_logits` -> minutes, team sum = 240).

Monte Carlo worlds must consume these same objects directly (not separate ad-hoc heuristics).

## Success Metrics (Primary)

- `ghost_active_rate`: projected meaningful run among actives, actual near-DNP.
- `missed_promo_rate`: projected near-zero among actives, actual meaningful run.
- `rotation_size_mae`: `|K_pred - K_actual|` at team-game level.
- `bench_tail_mass_error`: predicted vs actual minutes in low-minute tail bucket.
- `top8_share_error`: predicted vs actual team concentration.

All metrics are evaluated on last snapshot before tip.

## Architecture Decisions

### Model outputs (single contract)

- Player level:
  - `gate_logit`, `gate_prob` (`p_rot`).
  - `share_logit` (within-rotation allocation signal).
  - `pred_minutes` (point estimate, team-240 constrained).
- Team-game level:
  - `rotation_size_hat_team = sum(gate_prob)` (`k_hat_team`).

### Sim contract

- Model-space worlds mode consumes:
  - `play_prob`, `gate_logit/gate_prob`, `share_logit`, `minutes_mean`.
- Worlds sampler supports:
  - configurable gate temperature,
  - configurable bench-zero mixture,
  - share-logit-driven within-team demand sampling (not baseline minutes copy only).

## Execution Phases

### Phase 1: Contract + plumbing (start now)

- [x] Persist this plan and keep it updated.
- [x] Surface `rotation_size_hat_team` from rotation-set scoring artifacts.
- [x] Remove hardcoded model-space worlds constants in sim backend wiring.
- [x] Make model-space worlds consume `share_logit` for demand variation.
- [x] Keep behavior backward-compatible when aux cols are missing.

Acceptance:

- Rotation outputs include `rotation_size_hat_team`.
- Sim `minutes_worlds.mode=model_space_v1` behavior is config-driven.
- No regression in lint/tests touched by these files.

### Phase 2: Inference semantics cleanup

- [ ] Reduce reliance on heuristic `alloc_mask` as hard gate (degrade to soft fallback).
- [ ] Add explicit diagnostics for heuristic-vs-model membership disagreement.
- [ ] Calibrate gate probabilities on close-to-tip validation slice.

Acceptance:

- Lower catastrophic misses without forcing fixed K.
- Better calibration of `gate_prob` by minutes bucket.

### Phase 3: Feature enrichment + embeddings

- [x] Add player trust embeddings (`player_id`, optional `player_id x team_id`).
- [ ] Add coach/role stability features (lineup continuity, bench order, prior substitution role).
- [x] Add explicit `k_hat` supervision and team cardinality regularization.

Acceptance:

- Material improvement on rotation membership metrics.
- Improvement stable across injury and non-injury team-games.

### Phase 4: Sim-native integration

- [ ] Move to cardinality-aware membership draws (sample with `k_hat_team` constraint).
- [ ] Add explicit garbage-time pool latent for deep bench worlds.
- [ ] Ensure optimizer/dashboard consume worlds quantiles consistently.

Acceptance:

- Bench tail only appears in plausible worlds.
- Reduced need for arbitrary post-processing rules.

## Immediate Work Log

### 2026-02-12 (phase 1 completed)

- Implemented `rotation_size_hat_team` in `score_minutes_rotation_set_v1` and surfaced summary diagnostics.
- Model-space minutes worlds now support share-logit demand sampling with configurable blending/noise.
- Removed hardcoded model-space defaults in `generate_worlds_fpts_v2`; all knobs come from `sim_v2` profile.
- Validation status: targeted `ruff` checks passed and targeted pytest suite passed (`17 passed`).

### 2026-02-12 (phase 3 scaffolding started)

- Added optional player identity embeddings (`player_id`) and hashed player-team embeddings (`player_id x team_id`) to rotation-set model/trainer.
- Added configurable `k_target_source` (`fixed|label|blend`) so K regularization can use label-derived per-team rotation size targets.
- Added K diagnostics (`rotation_size_label_mean`, `k_hat_label_mae`, `k_hat_target_mae`) to eval/train history.
- Added unit tests for tensor-valued K targets and player-embedding inference fallback for unknown players.

### 2026-02-12 (phase 3 ablation readout)

- Ran 4-way ablation (`baseline_fixed`, `emb_fixed`, `emb_labelk`, `emb_blendk`) on `rotation_train_v1_20260103`.
- Offline: baseline remained best (`mae_team_w=7.9940`); embedding variants regressed.
- Live close-to-tip (`2026-01-02`) showed no meaningful improvement on active-only catastrophic rates:
  - `ghost_rate_active_only` = `0.1412` for all variants.
  - `missed_rate_active_only` = `0.2118` for all variants.
- Restricting to very close snapshots (`--max-minutes-to-tip 30`) did not change ranking:
  - `ghost_rate_active_only` = `0.1322` for all variants.
  - `missed_rate_active_only` = `0.2101` for all variants.
- Conclusion: embedding + K-target changes alone are insufficient; keep scaffolding but move to role-aware membership features and calibration.

### 2026-02-12 (eval tooling hardening)

- `eval_minutes_live_injury_regime` now reports `catastrophic_active_only` (filters OUT-like rows) in addition to all-status catastrophic metrics.
- Added tip-window diagnostics (`minutes_to_tip` p50/p90) and optional `--max-minutes-to-tip` filter.
- Fixed candidate-eval edge case where empty candidate coverage with `--allow-missing-candidate` raised `KeyError(run_id)` instead of returning empty-schema predictions.

### 2026-02-12 (strict vs not_out inference mask probe)

- Added runtime inference switch in rotation scorer/model (`--alloc-mask-mode strict|not_out`) to probe membership constraints without retraining.
- Probe result (`baseline_fixed`, `2026-01-02`):
  - `strict`: exactly 9 positive-minute players per team-game, high missed promotions.
  - `not_out`: model outputs become overly flat, triggering pathology fallback on all team-games.
- Takeaway: hard mask is currently compensating for weak membership learning; removing it naively causes smear/fallback.
