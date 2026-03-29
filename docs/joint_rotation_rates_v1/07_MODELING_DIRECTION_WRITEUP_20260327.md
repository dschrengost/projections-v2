# GTv2 Modeling Direction Write-Up

**Date:** 2026-03-27

## Purpose

This memo summarizes the current state of the GameTransformerV2 (GTv2) modeling work,
what has been tested recently, what improved, what failed, and the concrete decision
points where outside modeling opinions would be useful.

This is meant to be a review document, not a full historical log.

## Executive Summary

Current view:

1. The core GTv2 idea is still viable.
2. The largest upstream issue was objective drift from an overly coupled training stack.
3. The minutes / rotation side is still not strong enough to be considered solved, but it
   is good enough to use as a research conditioning layer.
4. On the downstream flow/stat side, the first genuinely promising structural change is:
   - `flow_target_schema=v2`
   - separate efficiency head retained
   - stronger direct supervision on opportunity stats than on box-score makes
   - `beta_binomial_all` decode at inference using learned efficiency parameters
5. That path materially improved high-usage / star player allocation and player-level
   point error without breaking the rotation side.
6. The remaining weakness is calibration:
   - possession calibration
   - interval coverage
   - to a lesser extent FG% calibration

## Original Intent

The intended role of GTv2 was:

- predict active set
- predict minutes
- generate a joint player/team stat distribution that captures real game correlation

The point was to model game-level correlation structure without building a full
possession-by-possession Markov simulator.

That goal still makes sense.

## Current Architecture

High-level order:

1. `JointActiveSetHead`
2. `JointMinutesHead`
3. `JointGameFlow`
4. optional:
   - `EfficiencyHead`
   - `UsageShareHead`
   - possession backbone / team-event backbone / 3PA share head

Relevant code:

- `projections/rotation/game_transformer_v2.py`
- `projections/rotation/joint_game_flow.py`
- `scripts/rotation/train_game_transformer_v2.py`

## What We Learned About Training

The trainer had become a research harness with too many simultaneous objectives:

- minutes losses
- active-count / membership losses
- flow NLL
- decision / FPTS losses
- efficiency losses
- usage-share losses
- possession / backbone losses
- emergent share auxiliaries
- spread / total auxiliaries
- props auxiliaries
- direct stat auxiliaries

That produced two practical problems:

1. minutes/rotation behavior drifted while optimizing downstream objectives
2. train/inference mismatch increased because minutes and flow can be teacher-forced
   during training but rely on predicted state at inference

The result is that the current trainer should not be treated as one coherent production
recipe.

## Minutes / Rotation Findings

Recent work on the minutes side established:

- clean scratch retrains with simplified objectives are better than the later
  drifted mixed-objective setup
- but they still do not fully solve sparse-starter / next-man-up failures
- a promotion-expert hybrid was directionally useful offline, but not yet ready for
  production live use

Current practical conclusion:

- minutes/rotation is good enough to freeze for downstream research isolation
- minutes/rotation is not good enough to declare final

## Flow / Stat Modeling Findings

### 1. Predicted vs oracle rotation-state diagnostic

Replacing predicted rotation/minutes state with oracle label-derived rotation state
improved overall environment calibration, but did **not** fix star/high-usage player
allocation.

Interpretation:

- upstream minutes error is part of the problem
- but downstream stat allocation is also a real independent problem

### 2. Direct-stat supervision

We added grouped direct-stat supervision to the trainer:

- `direct_boxscore_aux`
  - `PTS`, `REB`, `AST`, `STL`, `BLK`, `3PM`, `FTM`, `TOV`
- `direct_opportunity_aux`
  - `FGA`, `FTA`

This was added in:

- `scripts/rotation/train_game_transformer_v2.py`

### 3. `v1` flow target schema result

Using grouped direct-stat supervision with `flow_target_schema=v1` was not good enough.

It helped some high-usage slices, but:

- overall point calibration was not clearly better
- possession calibration worsened
- coverage worsened

Conclusion:

- `v1` with stronger direct stat supervision is not the right path

### 4. `v2` flow target schema result

The first promising result came from:

- `flow_target_schema=v2`
- `EfficiencyHead` retained
- direct opportunity loss > direct box-score loss
- inference decode with `make_model=beta_binomial_all`
- learned efficiency used during make sampling

Best training recipe so far:

- `w_direct_boxscore_aux=0.05`
- `w_direct_opportunity_aux=0.15`

Best inference recipe so far:

- `allocation_source=emergent`
- `make_model=beta_binomial_all`
- `bb_use_learned_efficiency=1`

## Best Experimental Result So Far

Reference run:

- `gtv2_flow_v2_directstats_ft_20260327T035212Z`

Useful artifacts:

- `compare_vs_start.json`
- `allocation_variant_summary.json`

Against the current baseline bundle on a 12-game / 64-world comparison:

Improved:

- overall `PTS` MAE
- `high_usage` point MAE
- `star` point MAE
- `elite` point MAE
- `high_usage` / `star` / `elite` FGA allocation
- top-1 and top-2 point-share errors
- FT% calibration
- total vs actual stayed approximately flat

Worsened:

- possession calibration
- interval coverage
- FG% calibration modestly

This is the first result I would call a meaningful structural improvement rather than
noise.

## Inference-Side Allocation Sweep

We also tested inference-only alternatives on the promising `v2` checkpoint:

- `allocation_source=emergent`
- `allocation_source=blend, alpha=0.25`
- `allocation_source=blend, alpha=0.50`
- `allocation_source=usage_head`

What happened:

- moving toward `usage_head` improved player-level high-usage / star means further
- but coverage and team-total behavior worsened

Interpretation:

- the model is capable of expressing stronger star concentration
- but the current downstream world generation becomes under-dispersed or miscalibrated
  when pushed too far toward usage-head allocation

So for now:

- `emergent + beta_binomial_all` is the best balanced decode
- `usage_head` remains an aggressive experimental variant, not the main path

## Negative Result: Possession-Focused Follow-Up Fine-Tune

We also tried a follow-up fine-tune that:

- increased possession-related loss weights
- slightly unfroze the game-level projection path
- started from the promising `v2` checkpoint

Result:

- no improvement in the ex-possession total objective
- no evidence that it solved the actual calibration problem

Conclusion:

- simply turning up possession/backbone pressure is not the next answer

## Current Modeling Interpretation

The model now looks like this:

### What is likely fixed or partly fixed

- direct supervision on opportunity stats helps
- separating attempt/opportunity modeling from make-rate modeling helps
- `v2 + efficiency + beta_binomial_all` is a better factorization than the earlier
  `v1` setup

### What is still missing

- better distribution calibration
- better possession/game-environment calibration
- possibly better coupling between flow outputs and world-generation variance

### What does **not** currently look like the main bottleneck

- lack of raw next-man-up signal in the dataset
- lack of expressiveness in the flow block itself
- lack of ability to concentrate usage on stars

The latest experiments suggest the remaining problem is more about:

- calibration
- decode
- variance structure

than about feature omission.

## Decision Options

### Option A: Stay on the current `v2` branch and focus on calibration

This is my current recommendation.

Pros:

- already showed a real improvement
- keeps the current architecture family intact
- isolates the remaining problem to calibration/coverage

Cons:

- may require additional inference/decode work
- may require cleaner likelihood or calibration losses rather than simple aux tuning

### Option B: Push harder on usage-head allocation

Pros:

- improves star/high-usage slices further

Cons:

- worsens coverage and total-game behavior
- looks too aggressive for the mainline model right now

My view:

- useful as an upside-profile experiment
- not the main branch

### Option C: Revisit training of the full stack from scratch on the new factorization

Meaning:

- train a full `v2 + efficiency + opportunity-heavy supervision` recipe from a fresh
  initialization once the recipe is stable enough

Pros:

- could remove warm-start bias from the older `v1` production bundle

Cons:

- expensive
- harder to isolate whether improvements come from factorization or full retraining

My view:

- worth doing, but after the calibration path is better defined

## My Current Recommendation

I would continue with:

1. keep the `v2 + efficiency + direct opportunity aux` branch as the main experimental
   line
2. keep `beta_binomial_all` as the current best decode
3. focus next on calibration rather than feature expansion or allocation aggression
4. only after that run a cleaner from-scratch full retrain of the same factorization

## Questions For External Review

These are the questions I would ask other modelers:

1. Is the current factorization correct?
   - flow models attempts/opportunity/peripherals
   - efficiency head models make rates
   - decode uses conditional make sampling

2. What is the best way to fix coverage and calibration from here?
   - sharper likelihood?
   - explicit calibration loss?
   - better decode?
   - different uncertainty parameterization?

3. Should the possession/game-environment component remain a separate backbone, or be
   absorbed more directly into the flow/world generation path?

4. Is `beta_binomial_all` the right inference-time reconstruction, or is it covering
   for a training mismatch that should be addressed in-model?

5. At this point, would you:
   - continue iterating on the current `v2` branch
   - do a full from-scratch retrain on the new factorization
   - or redesign the downstream generative head again?

## Artifacts To Share

If someone wants the concrete outputs rather than just this memo, the most useful files
to hand them are:

- `/home/daniel/projects/projections-v2/docs/joint_rotation_rates_v1/GAME_TRANSFORMER_SPEC.md`
- `/home/daniel/projections-data/training/runs/gtv2_flow_v2_directstats_ft_20260327T035212Z/compare_vs_start.json`
- `/home/daniel/projections-data/training/runs/gtv2_flow_v2_directstats_ft_20260327T035212Z/allocation_variant_summary.json`
- `/home/daniel/projections-data/training/runs/gtv2_flow_v2_directstats_ft_20260327T035212Z/candidate_eval_12g64w_beta/make_rate_eval.json`
- `/home/daniel/projections-data/training/runs/gtv2_flow_directstats_ft_20260327T034643Z/compare_vs_start.json`

## Follow-up Diagnostic Addendum

After external review, the next diagnostic pass split the remaining "calibration"
problem into three pieces:

1. possession / game-environment calibration
2. decode-side dispersion / coverage
3. minutes-uncertainty propagation into worlds

### Diagnostic 1: Possession / environment calibration

What was tested:

- a follow-up fine-tune from the promising `v2` checkpoint with stronger:
  - `w_poss_nll`
  - `w_backbone_nll`
  - `w_poss_regression`
- plus a slight unfreeze of the game-level projection path

Artifact:

- `/home/daniel/projections-data/training/runs/gtv2_flow_v2_poss_ft_20260327T040459Z`

Result:

- this was a negative result
- the follow-up did not improve the ex-possession objective
- stronger possession/backbone weighting alone does not appear to be the right
  mechanism

Interpretation:

- possession calibration still matters
- but simple loss-weight escalation is not the fix

### Diagnostic 2: Decode-side dispersion / coverage

What was tested:

- decode variants on the promising `v2` checkpoint:
  - `allocation_source=emergent` + `beta_binomial_all`
  - `blend`
  - `usage_head`
- wider beta-binomial concentration settings at inference

Artifacts:

- `/home/daniel/projections-data/training/runs/gtv2_flow_v2_directstats_ft_20260327T035212Z/allocation_variant_summary.json`
- `/home/daniel/projections-data/training/runs/gtv2_flow_v2_directstats_ft_20260327T035212Z/candidate_eval_12g64w_beta/make_rate_eval.json`
- `/home/daniel/projections-data/training/runs/gtv2_flow_v2_directstats_ft_20260327T035212Z/candidate_eval_12g64w_beta_wider/make_rate_eval.json`

Result:

- `emergent + beta_binomial_all` remains the best balanced decode
- `usage_head` improves star/high-usage means further, but gives back too much in:
  - coverage
  - total-game accuracy
- simply lowering beta-binomial concentration did not materially improve coverage

Interpretation:

- `beta_binomial_all` is useful, but not sufficient to solve under-dispersion by itself
- the remaining coverage issue is not just a concentration hyperparameter problem

### Diagnostic 3: Minutes-uncertainty propagation

What was tested:

- world-level variance decomposition on:
  - baseline worlds
  - `v2_beta`
  - `v2_usage_beta`
- comparison of:
  - total minutes variance
  - within-active minutes variance
  - variance due to active/inactive branching
- then a stricter test:
  - group worlds by exact team active-mask signature
  - measure player-minute variance within identical signatures

Artifacts:

- `/home/daniel/projections-data/training/runs/gtv2_flow_v2_directstats_ft_20260327T035212Z/minutes_uncertainty_diagnostic.json`
- `/home/daniel/projections-data/training/runs/gtv2_flow_v2_directstats_ft_20260327T035212Z/minutes_uncertainty_active_signature_diagnostic.json`

Key finding:

- minutes are deterministic once the sampled team active mask is fixed

Evidence:

- for meaningful players, active-only minute variance exists at the world level
- but within an identical team active-mask signature, weighted minute std is exactly `0.0`
  across:
  - baseline
  - `v2_beta`
  - `v2_usage_beta`
- meaningful-player median active-only minute std is about:
  - `1.46` on baseline
  - `1.35` on `v2_beta`
- but median same-signature minute std is `0.0`

Interpretation:

- current world spread in minutes comes from discrete active-set branching
- it does **not** come from a continuous minutes distribution conditioned on the same
  rotation state
- this is likely one reason interval coverage remains too tight even after the `v2`
  factorization improved player-share means

### First inference-side minutes-uncertainty experiment

What was tested:

- inference-only Gaussian minute noise for active players in `sample_worlds_v2.py`
- hurdle `sigma` when available, otherwise trailing `minutes_from_stints_std_prior_*`
  priors
- noisy minute seeds projected back to the capped team-minute simplex
- no trainer changes

Artifacts:

- `/home/daniel/projections-data/training/runs/gtv2_flow_v2_directstats_ft_20260327T035212Z/candidate_eval_12g64w_beta_minunc_s1p0/make_rate_eval.json`
- `/home/daniel/projections-data/training/runs/gtv2_flow_v2_directstats_ft_20260327T035212Z/candidate_eval_12g64w_beta_minunc_s1p5/make_rate_eval.json`
- `/home/daniel/projections-data/training/runs/gtv2_flow_v2_directstats_ft_20260327T035212Z/minutes_uncertainty_backtest_compare.json`

Result:

- `p90_coverage` improved slightly:
  - `0.8583 -> 0.8639` at scale `1.0`
  - `0.8583 -> 0.8667` at scale `1.5`
- `p95_coverage` stayed flat at `0.9111`
- possession metrics improved slightly
- but:
  - `PTS` MAE worsened by about `+0.32` to `+0.35`
  - star-slice `PTS` MAE worsened by about `+0.63`
  - total-game MAE vs actual worsened by about `+0.20`

Interpretation:

- minutes uncertainty is a real lever
- the naive independent-Gaussian implementation is not a clean improvement
- any next version should be more selective or more coherent at the team level
  instead of perturbing all active-player minutes the same way

### Follow-up: selective and residual-share uncertainty variants

Two more inference-only variants were tested after the naive Gaussian pass:

1. selective Gaussian
   - preserve top-3 minute players per team
   - taper uncertainty down as predicted minutes rise
   - exact re-imposition of protected player minutes after projection

2. residual-share Dirichlet
   - preserve top-3 minute players per team exactly
   - sample the remaining team minutes across the residual active pool from a
     Dirichlet-style share distribution centered on the baseline allocation

Artifacts:

- `/home/daniel/projections-data/training/runs/gtv2_flow_v2_directstats_ft_20260327T035212Z/minutes_uncertainty_backtest_compare_v2.json`
- `/home/daniel/projections-data/training/runs/gtv2_flow_v2_directstats_ft_20260327T035212Z/minutes_uncertainty_dirichlet_compare.json`

Result:

- selective Gaussian was only marginally different from naive Gaussian:
  - same slight `p90` gain
  - same flat `p95`
  - same degradation in `PTS` MAE and star-slice `PTS` MAE
- residual-share Dirichlet was worse:
  - `p90` and `p95` improved slightly
  - player `PTS` MAE improved slightly
  - but team total MAE versus actual worsened badly (about `+2.38`)

Interpretation:

- inference-only uncertainty injection can move the right metrics
- but the tradeoff is not acceptable with these post-hoc schemes
- this is strong evidence that the next serious step should be a training-side
  minutes-distribution model or another learned uncertainty mechanism, not more
  sampler-only hacks

## Updated Recommendation

The current best branch is still:

- `flow_target_schema=v2`
- separate efficiency head
- heavier direct opportunity supervision than box-score supervision
- `allocation_source=emergent`
- `make_model=beta_binomial_all`

But the next work should now be framed more narrowly:

1. keep the current `v2` factorization
2. treat possession calibration and world dispersion as separate problems
3. investigate calibrated minutes uncertainty, not just more decode tweaks
4. only revisit a full from-scratch retrain after the calibration path is clearer

## Follow-up: learned minutes distribution

The next iteration moved from inference-only uncertainty injection to a learned
minutes-distribution path.

Training run:

- `/home/daniel/projections-data/training/runs/gtv2_flow_v2_minutesdist_ft_20260327T050500Z`

Key modeling changes:

- enabled the minutes hurdle head with learned `sigma`
- enabled `flow_use_minutes_conditioning=true`
- warm-started from the prior `v2` direct-stats run
- kept the shared encoder / active head frozen so the experiment isolated the
  minutes-conditioned downstream path

The most important control was to evaluate the trained checkpoint in two modes:

1. no sampler-side minutes uncertainty
2. sampler-side Gaussian minutes uncertainty using learned hurdle `sigma`

Artifacts:

- no sampler uncertainty:
  - `/home/daniel/projections-data/training/runs/gtv2_flow_v2_minutesdist_ft_20260327T050500Z/candidate_eval_12g64w_nominunc/make_rate_eval.json`
- learned-sigma sampler path:
  - `/home/daniel/projections-data/training/runs/gtv2_flow_v2_minutesdist_ft_20260327T050500Z/candidate_eval_12g64w_learnedsigma/make_rate_eval.json`
- comparison table:
  - `/home/daniel/projections-data/training/runs/gtv2_flow_v2_minutesdist_ft_20260327T050500Z/compare_baseline_vs_minutesdist.json`
- same-signature variance diagnostic:
  - `/home/daniel/projections-data/training/runs/gtv2_flow_v2_minutesdist_ft_20260327T050500Z/learned_minutes_same_signature_diagnostic.json`

Result versus prior `v2_beta` control:

- no sampler uncertainty:
  - `pts_mae`: `10.95 -> 9.78`
  - `total_mae_vs_actual`: `16.93 -> 13.17`
  - `total_mae_vs_vegas`: `9.56 -> 5.23`
  - `p90_coverage`: `0.858 -> 0.864`
  - `p95_coverage`: `0.911 -> 0.919`
  - `poss_mae`: `4.69 -> 4.09`
  - `game_poss_mae_vs_est`: `1.74 -> 1.44`
  - `top1_share_mae_pts`: `0.0442 -> 0.0333`
  - `top2_share_mae_pts`: `0.0645 -> 0.0489`

- learned-sigma sampler path:
  - `pts_mae`: `10.95 -> 9.84`
  - `total_mae_vs_actual`: `16.93 -> 12.96`
  - `total_mae_vs_vegas`: `9.56 -> 4.28`
  - `p90_coverage`: `0.858 -> 0.861`
  - `p95_coverage`: `0.911 -> 0.925`
  - `poss_mae`: `4.69 -> 4.00`
  - `game_poss_mae_vs_est`: `1.74 -> 0.86`

Tradeoffs:

- the trained minutes-distribution model is a clear improvement even with no
  sampler-side uncertainty
- learned-sigma sampling improves:
  - `p95` coverage
  - total-game calibration
  - possession calibration
- but learned-sigma sampling also gives back some player-level mean quality,
  especially on the `25-34` actual-point star slice

The key structural diagnostic also changed:

- prior control:
  - same-signature minute std was exactly `0.0`
- learned minutes-distribution run:
  - mean same-signature minute std: `0.89`
  - `32.5%` of meaningful player/signature groups had positive within-signature
    minute variance

Interpretation:

- the learned distribution fixed the original structural defect:
  minutes are no longer deterministic conditional on active signature
- most of the gain appears to come from:
  - training with minutes conditioning turned on
  - learning the minutes-distribution representation
- sampler-side uncertainty is now optional and should be treated as a calibration
  knob, not the core mechanism

Updated recommendation:

1. promote the learned minutes-distribution branch to the main experimental line
2. use the `no_minunc` decode as the default research control
3. keep learned-sigma sampling available as a calibration variant
4. next work should target the remaining star-slice tradeoff and calibration,
   not revert to sampler-only uncertainty hacks

## Follow-up: flow minutes teacher-forcing anneal

The next targeted experiment attacked the remaining mismatch directly:

- training: flow conditioned on true minutes
- inference: flow conditioned on predicted minutes

Instead of another architecture change, the experiment annealed flow minutes
teacher forcing from `1.0 -> 0.0` over training.

Run:

- `/home/daniel/projections-data/training/runs/gtv2_flow_v2_minutesdist_mtfanneal_20260327T050033Z`

Artifact:

- `/home/daniel/projections-data/training/runs/gtv2_flow_v2_minutesdist_mtfanneal_20260327T050033Z/compare_vs_minutesdist_control.json`

Result versus the prior learned-distribution control (`minutesdist_no_minunc`):

- `pts_mae`: `9.78 -> 9.41`
- `total_mae_vs_actual`: `13.17 -> 12.99`
- `total_mae_vs_vegas`: `5.23 -> 4.71`
- `high_usage_mae_pts_18plus`: `5.88 -> 5.54`
- `star_mae_pts_25_34`: `7.08 -> 6.37`
- `elite_mae_pts_35plus`: `12.96 -> 12.65`
- `p90_coverage`: `0.864 -> 0.881`
- `p95_coverage`: `0.919 -> 0.931`
- `poss_mae`: `4.09 -> 4.01`

Minor tradeoff:

- top-share MAE worsened slightly versus `minutesdist_no_minunc`, but remained
  much better than the original `v2_beta` control

Interpretation:

- the star-slice regression was largely a train/inference conditioning mismatch
- annealing flow minutes teacher forcing is a cleaner fix than more sampler-side
  uncertainty injection
- this now becomes the best mainline branch so far

Updated recommendation:

1. promote `minutesdist_mtfanneal` to the main experimental control
2. treat learned-sigma sampling as a secondary calibration variant
3. continue from this branch with smaller calibration / share-allocation tuning,
   not another large structural reset

## Follow-up: 60-game production-aligned backtest

The next step was a broader production-style backtest using the live-aligned
world-generation harness:

- script: `scripts/rotation/run_gtv2_promotion_alignment.py`
- window: `60` games from the 60-day validation slice
- worlds: `128` per game
- variants:
  - `prod_live_exact`
  - `minutesdist_mtfanneal`
  - `minutesdist_mtfanneal_learnedsigma`

Artifacts:

- `/home/daniel/projections-data/training/runs/gtv2_minutesdist_backtest_20260327T050908Z/summary.csv`
- `/home/daniel/projections-data/training/runs/gtv2_minutesdist_backtest_20260327T050908Z/compare_vs_baseline.csv`

Main result:

`minutesdist_mtfanneal` held up as the best promotion candidate.

Versus `prod_live_exact`:

- `dk_fpts_mae`: `5.650 -> 5.613`
- `minutes_mae`: `3.718 -> 3.721` essentially flat/slightly worse
- `active_acc_at4`: `0.9206 -> 0.9183` slightly worse
- `pts_mae_player`: `3.548 -> 3.494`
- `pts_mae_team`: `10.777 -> 10.226`
- `spread_mae_vs_vegas`: `5.577 -> 5.023`
- `total_mae_vs_vegas`: `4.391 -> 3.651`
- `poss_mae`: `5.116 -> 4.269`

Important caveats:

- the sampled 60-game window had `0` starter-promotion slice rows under the
  current definition, so this backtest does not validate the sparse surprise-
  starter failure mode directly
- `p90` calibration improved, but `p95` calibration error was slightly worse
  than production on this broader sample
- `REB`, `AST`, `STL`, and `active_acc_at4` were flat to slightly worse

Learned-sigma variant:

- improved `dk_fpts_mae` and `pts_mae_player` slightly more
- but worsened minutes/active slightly more than `minutesdist_mtfanneal`
- and gave back some spread/total quality

Interpretation:

- the new branch is now broad-backtest credible
- the default candidate should remain `minutesdist_mtfanneal`
- learned-sigma remains a secondary calibration A/B, not the mainline default

Updated promotion stance:

1. this branch is ready for a production shadow / flag-gated promotion candidate
2. do not fully replace production until the sparse surprise-starter cases are
   checked explicitly on a targeted slice or replay set

## Follow-up: targeted sparse-starter replay and hybrid expert

The targeted replay benchmark was added specifically to answer the remaining
open question: does the new branch actually fix the original sparse-starter
failure mode?

Target case source:

- `/home/daniel/projections-data/training/runs/gtv2_minutesdist_backtest_20260327T050908Z/target_sparse_starter_cases.csv`

Full replay run:

- `/home/daniel/projections-data/training/runs/gtv2_sparse_replay_full_20260327T052101Z`

Result:

- broad `minutesdist_mtfanneal` branch is still worse than `prod_live_exact` on
  this sparse-starter replay set:
  - starter-promotion predicted minutes mean:
    - `12.44 -> 11.99`
  - starter-promotion minutes MAE:
    - `9.86 -> 10.43`
  - starter-promotion active recall @4:
    - `0.909 -> 0.818`

This confirmed the broad branch had not solved the original problem, even
though it was much better in aggregate.

### Sparse expert follow-up

A specialist sparse-starter expert was then fine-tuned from the broad branch:

- run:
  - `/home/daniel/projections-data/training/runs/gtv2_flow_v2_minutesdist_sparseexpert_20260327T052217Z`
- trained with:
  - broader sparse mask
  - `w_sparse_starter_underpred_loss=0.10`
  - intended use as a gated uplift expert, not a replacement model

Hybrid replay run:

- `/home/daniel/projections-data/training/runs/gtv2_sparse_replay_hybrid_20260327T052313Z`

Hybrid sparse replay result versus base `minutesdist_mtfanneal`:

- starter-promotion predicted minutes mean:
  - `11.99 -> 13.64`
- starter-promotion minutes MAE:
  - `10.43 -> 10.06`
- starter-promotion active recall @4:
  - unchanged at `0.818`
- starter-promotion low-8 rate:
  - `0.364 -> 0.273`

Case-level uplift examples:

- `Guerschon Yabusele`: `+2.67` predicted minutes
- `Elfrid Payton`: `+3.18`
- `Hunter Tyson`: `+1.35`
- `Jaden Hardy` (`2026-01-14`): `+2.24`

Artifact:

- `/home/daniel/projections-data/training/runs/gtv2_sparse_replay_hybrid_20260327T052313Z/target_case_compare.csv`

### Broad-window check for the hybrid

Broad 60-game check:

- `/home/daniel/projections-data/training/runs/gtv2_broad_hybrid_20260327T052345Z`

Result:

- hybrid was identical to base on the broad 60-game window
- reason:
  - that 60-game sample contained no promotion-slice rows, so the gated expert
    never activated

Interpretation:

- the branch story is now cleaner:
  - `minutesdist_mtfanneal` is the best broad base model
  - a gated sparse expert improves the original failure class
  - and does not disturb the broad backtest when the slice is absent

Updated recommendation:

1. keep `minutesdist_mtfanneal` as the main experimental base
2. carry the sparse expert as an optional hybrid overlay for the promotion slice
3. future work should tune the promotion gate and expert quality, not destabilize
   the broad branch again

### Narrowed sparse expert: active/minutes heads only

Follow-up tuning confirmed that the remaining sparse-starter gap was not a blend
policy problem. It was an expert-quality problem.

Policy sweep artifact:

- `/home/daniel/projections-data/training/runs/gtv2_sparse_hybrid_policy_sweep_20260327T053500Z/summary.csv`

Key result from the policy sweep:

- `uplift_only` remained the best application policy
- forcing active candidates did not improve sparse recall
- full expert replacement was worse than uplift-only

That motivated a narrower specialist:

- training run:
  - `/home/daniel/projections-data/training/runs/gtv2_flow_v2_minutesdist_sparseexpert_heads_v14_20260327T054500Z`
- setup:
  - warm-start from the first sparse expert
  - freeze everything except `active_head` and `minutes_head`
  - keep the base `minutesdist_mtfanneal` branch as the downstream flow/stat model
  - train only the rotation/minutes decision for the sparse slice

Replay comparison:

- `/home/daniel/projections-data/training/runs/gtv2_sparse_hybrid_heads_compare_v14_20260327T054650Z/summary.csv`

Targeted replay result versus base:

- starter-promotion predicted minutes mean:
  - `11.95 -> 13.73`
- starter-promotion minutes MAE:
  - `10.57 -> 10.01`
- starter-promotion active recall @4:
  - `0.818 -> 0.909`
- starter-promotion low-8 rate:
  - `0.364 -> 0.273`

Interpretation:

- the best current structure is now:
  - broad base: `minutesdist_mtfanneal`
  - sparse overlay: heads-only gated sparse expert
- this closes most of the remaining gap to the production sparse replay without
  reopening the broad branch

### Broader sparse replay validation

The narrowed sparse expert was then re-evaluated on a widened replay slice built
from the same inferable gate features:

- `lineup_starter_announced=1`
- `minutes_from_stints_prior_20 <= 14`
- `max(recent_start_pct_10, started_proxy_rate_prior_10, started_proxy_rate_prior_20) <= 0.25`
- actual minutes `>= 16`

Artifacts:

- `/home/daniel/projections-data/training/runs/gtv2_sparse_hybrid_broadreplay_20260327T055000Z/summary.csv`
- `/home/daniel/projections-data/training/runs/gtv2_sparse_hybrid_broadreplay_20260327T055000Z/target_rows.csv`

This widened replay covered:

- `22` player rows
- `18` games

Result versus base:

- starter-promotion predicted minutes mean:
  - `10.37 -> 12.27`
- starter-promotion minutes MAE:
  - `8.85 -> 8.43`
- starter-promotion active recall @4:
  - unchanged at `0.818`
- starter-promotion low-8 rate:
  - `0.364 -> 0.273`
- overall replay `dk_fpts_mae`:
  - `5.864 -> 5.817`
- overall replay `minutes_mae`:
  - `4.378 -> 4.362`

Interpretation:

- the heads-only sparse overlay generalizes better than the original 9-game
  replay suggested
- it is also slightly better than the first sparse expert on the widened replay
- current best sparse path is now clearly:
  - `minutesdist_mtfanneal` base
  - `hybrid_heads_v14` sparse overlay

### Oversampled promotion specialist

After confirming that true positives are scarce (`22` broad positive rows in
train) and that the broad gate pool mixes both real promotions and many
low-minute outcomes, the next iteration changed the specialist training recipe:

- add a broad sparse-candidate sampler in the trainer
- oversample games containing at least one pre-tip promotion candidate
- enable the existing starter-promotion delta head
- keep the freeze pattern narrow (`active_head` + `minutes_head` trainable only)

Trainer code path:

- `/home/daniel/projects/projections-v2/scripts/rotation/train_game_transformer_v2.py`

Promotion specialist run:

- `/home/daniel/projections-data/training/runs/gtv2_flow_v2_minutesdist_sparseexpert_promote_20260327T055700Z`

Replay comparison:

- `/home/daniel/projections-data/training/runs/gtv2_sparse_hybrid_promote_compare_20260327T055900Z/summary.csv`

On the broadened replay benchmark, the new hybrid (`hybrid_promote`) was best:

- starter-promotion slice size:
  - `19 -> 31`
- starter-promotion next-up rows:
  - `11 -> 20`
- starter-promotion predicted minutes mean:
  - `10.37 -> 18.25`
- starter-promotion minutes MAE:
  - `8.85 -> 7.87`
- starter-promotion active recall @4:
  - `0.818 -> 0.950`
- starter-promotion under-10 rate:
  - `0.364 -> 0.200`
- starter-promotion low-8 rate:
  - `0.364 -> 0.100`

Replay-level tradeoff:

- `dk_fpts_mae` improved:
  - `5.864 -> 5.812`
- `minutes_mae` improved:
  - `4.378 -> 4.355`
- `poss_mae` worsened:
  - `3.763 -> 4.101`

Interpretation:

- this is the first specialist that moves the sparse replay materially, not
  incrementally
- the broad gate + sampler + promotion-delta head is the right training shape
- the main caution is that this version is more aggressive and should still be
  treated as experimental until it is checked on a longer shadow window

### Fair same-gate comparison

To separate the effect of training from the effect of simply widening the gate,
the two sparse overlays were compared again under the same replay gate
(`prior_minutes<=14`, `hist_start_rate<=0.25`):

- `/home/daniel/projections-data/training/runs/gtv2_sparse_hybrid_promote_faircompare_20260327T060100Z/summary.csv`

Result:

- `hybrid_promote_gate14` was best on sparse minutes behavior:
  - starter-promotion predicted minutes mean:
    - `18.25` vs `16.25` for `hybrid_heads_v14_gate14`
  - starter-promotion minutes MAE:
    - `7.87` vs `8.11`
  - replay-level `minutes_mae`:
    - `4.355` vs `4.375`
- `hybrid_heads_v14_gate14` remained slightly better on replay-level FPTS:
  - `dk_fpts_mae`:
    - `5.799` vs `5.812`

Updated interpretation:

- if the objective is the sparse-starter minutes failure specifically, the new
  oversampled promotion specialist is the best branch
- if the objective is the best mixed replay-level FPTS on this benchmark, the
  earlier heads-only overlay still has a small edge
- both remain improvements over the base model on the broadened sparse replay

### Longer replay on the full candidate pool

Final replay before locking the sparse path:

- `/home/daniel/projections-data/training/runs/gtv2_sparse_hybrid_longreplay_20260327T060500Z/summary.csv`
- candidate universe:
  - all rows with
    - `lineup_starter_announced=1`
    - `minutes_from_stints_prior_20 <= 14`
    - `max(recent_start_pct_10, started_proxy_rate_prior_10, started_proxy_rate_prior_20) <= 0.25`
- coverage:
  - `88` candidate rows
  - `38` games

Result:

- `hybrid_promote_gate14` was the best activation overlay:
  - starter-promotion active recall @4:
    - `1.00`
  - starter-promotion low-8 rate:
    - `0.10`
  - starter-promotion predicted minutes mean:
    - `11.83`
- `hybrid_heads_v14_gate14` was next:
  - starter-promotion active recall @4:
    - `0.95`
  - starter-promotion low-8 rate:
    - `0.15`
- base:
  - starter-promotion active recall @4:
    - `0.90`
  - starter-promotion low-8 rate:
    - `0.20`

Tradeoff on this candidate-heavy replay:

- base remained best on overall replay `dk_fpts_mae` and `minutes_mae`
- both overlays gave back some overall replay quality in exchange for much
  stronger activation behavior on the sparse candidate slice

Decision:

- for the stated workflow goal
  - “get sparse surprise starters alive in worlds and projectable, then manual
    boost is acceptable”
- the sparse path is now locked as:
  - broad base: `minutesdist_mtfanneal`
  - experimental sparse overlay: `hybrid_promote_gate14`

## Bench-Riser Follow-Up

Outside the sparse-starter layer, the biggest remaining minutes misses were:

- high-minute non-starters
- spot starters
- the messy `4-12` minute fringe bucket

A new evaluator bucket was added for bench risers:

- `bench_riser_candidate`
- `bench_riser_next_up`
- `bench_core_next_up`
- plus underprediction rates for `20+` and `32+` minute non-starters

Base branch 60-day diagnostics:

- run:
  - `/home/daniel/projections-data/training/runs/gtv2_flow_v2_minutesdist_mtfanneal_20260327T050033Z/eval_slices_60d.json`
- result:
  - `bench_riser_underprediction_rate = 0.174`
  - `bench_core_underprediction_rate = 0.043`
  - `bench_riser_next_up` predicted mean:
    - `20.73` vs actual `25.63`
  - `bench_core_next_up` predicted mean:
    - `25.92` vs actual `35.19`

Training follow-up:

- added a direct `bench_riser_underpred_loss` to the trainer
- tested warm-start head-only fine-tunes from `minutesdist_mtfanneal`

Runs:

- `/home/daniel/projections-data/training/runs/gtv2_flow_v2_minutesdist_benchriser_w010_20260327T124800Z`
- `/home/daniel/projections-data/training/runs/gtv2_flow_v2_minutesdist_benchriser_w025_20260327T124800Z`
- `/home/daniel/projections-data/training/runs/gtv2_flow_v2_minutesdist_benchriser_narrow_w005_20260327T125500Z`

Conclusion:

- the monolithic bench-riser penalty is not a keeper
- it reduced bench-riser underprediction rates
- but it catastrophically over-lifted the broader non-starter pool and doubled
  overall minutes MAE
- even the narrower version was not safe

Working conclusion:

- keep the new bench-riser eval slices
- do not keep the direct bench-riser underprediction loss in the main recipe
- if this bucket becomes the next priority, it should likely be handled the same
  way sparse starters were handled:
  - a gated specialist or overlay
  - not a broad loss applied to the whole base model

### Bench-Riser Specialist Overlay

Follow-up experiment:

- specialist run:
  - `/home/daniel/projections-data/training/runs/gtv2_flow_v2_minutesdist_benchspecialist_20260327T131500Z`
- training shape:
  - warm-start from `minutesdist_mtfanneal`
  - freeze everything except `active_head` and `minutes_head`
  - oversample games with narrow bench-riser candidates
  - bench-riser loss only on narrow pre-tip candidates

The specialist was only usable with a much tighter inference gate than the
training-side broad bench bucket:

- non-starter
- `hist_start_rate <= 0.35`
- `minutes_from_stints_prior_20 >= 12`
- `prior_play_prob >= 0.80`
- `an_implied_minutes >= 12`
- uplift-only blend into the base model

Artifact:

- `/home/daniel/projections-data/training/runs/gtv2_flow_v2_minutesdist_benchspecialist_20260327T131500Z/hybrid_gate_eval.json`

Result on the 60-day player-level eval:

- overall minutes MAE:
  - base `4.116`
  - hybrid `4.197`
- gated `20+` bench-riser rows:
  - MAE `3.699 -> 3.568`
  - bias `-2.084 -> +1.319`
- gated `32+` bench-riser rows:
  - MAE `7.921 -> 6.439`
  - bias `-7.835 -> -6.353`

Interpretation:

- the specialist route is viable
- the gate has to be tight
- broad non-starter gates still over-lift too many rows
- a narrow bench-riser overlay now looks much more promising than a monolithic
  base-loss change

### Overlay Interaction Check

The sparse-starter and bench-riser overlays were then wired into the same
world-generation path.

Implementation note:

- the bench specialist was trained with the same feature schema but different
  player-feature normalization than the base branch
- sampler integration now renormalizes player features from the base model
  normalization into the bench expert normalization before the bench expert
  forward pass

Interaction replay:

- replay root:
  - `/home/daniel/projections-data/training/runs/gtv2_hybrid_interaction_20260327T133500Z`
- replay set:
  - 60-day validation union of games where either overlay gate would fire
  - `237` games selected, `236` evaluated
  - gate counts in that window:
    - starter gate rows: `21`
    - bench gate rows: `553`

Variants:

- `base`
- `starter_only`
- `bench_only`
- `starter_and_bench`

Key result:

- the two overlays do not materially fight each other
- `starter_and_bench` preserved the starter overlay gain and the bench overlay
  gain at the same time

Relevant metrics:

- starter slice:
  - base:
    - predicted minutes mean `16.45`
    - minutes MAE `8.60`
    - active recall `0.857`
    - under-10 rate `0.143`
  - starter only:
    - predicted minutes mean `23.90`
    - minutes MAE `7.01`
    - active recall `1.00`
    - under-10 rate `0.063`
  - starter and bench:
    - predicted minutes mean `23.32`
    - minutes MAE `6.97`
    - active recall `1.00`
    - under-10 rate `0.063`

- bench slice:
  - base:
    - predicted minutes mean `22.69`
    - minutes MAE `4.55`
    - under-16 rate `0.0056`
  - bench only:
    - predicted minutes mean `26.22`
    - minutes MAE `5.69`
    - under-16 rate `0.0`
  - starter and bench:
    - predicted minutes mean `26.21`
    - minutes MAE `5.68`
    - under-16 rate `0.0`

Broad replay tradeoff:

- the combined overlay slightly worsened aggregate replay minutes MAE
  (`4.156 -> 4.248`)
- but that is consistent with the overlays being targeted activation/uplift
  layers, not broad replacements for the base model

Working conclusion:

- keep `minutesdist_mtfanneal` as the base branch
- sparse overlay and bench overlay can coexist in the experimental hybrid path
- precedence should remain:
  - starter overlay first
  - bench overlay second
  - bench overlay never applies to starter rows

### Minutes Status Before Switching to Rates

Current broad minutes assessment on the base branch
(`/home/daniel/projections-data/training/runs/gtv2_flow_v2_minutesdist_mtfanneal_20260327T050033Z/eval_slices_60d.json`):

- stable cores / normal starters are strong
  - starters:
    - `n = 1110`
    - actual minutes mean `29.82`
    - predicted minutes mean `29.80`
    - active recall `0.995`
    - predicted low-minute rate `<8`: `0.0054`
  - starter next-up underprediction rate:
    - `0.0039`

- remaining broad minutes weaknesses are concentrated in:
  - sparse / promoted starters
  - high-minute non-starters / bench risers

So the current base branch is no longer broadly weak on stable cores. The
remaining minutes work is regime-specific, not a general starter-minutes
problem.

### Rates / Game Context Training Map

Current GTv2 rates/game-context stack is still trained as one coupled objective.

Model order:

- active set head
- minutes head
- joint flow head
- optional efficiency head
- optional possession head + team event backbone + optional 3PA share head
- optional usage-share head

Current responsibilities in code:

- `flow_target_schema=v2`:
  - flow owns attempts / opportunity / peripheral count generation
  - efficiency head can separately own make-rate uncertainty
- efficiency head:
  - predicts Beta-Binomial parameters for `FT`, `FG2`, `FG3` make rates
- possession backbone:
  - predicts game possessions and team event rates under a possession identity
- usage-share head:
  - predicts per-player within-team `FGA`, `FTA`, `TOV` share logits

Current training loss stack is still heavily coupled:

- core rotation losses:
  - count CE
  - member BCE
  - minutes MAE
  - minutes NLL / hurdle NLL
  - optional role / promotion / sparse / bench losses
- flow / decision losses:
  - flow NLL
  - CRPS over sampled FPTS
  - team energy
- possession / backbone losses:
  - possessions Student-t NLL
  - possession regression
  - backbone NLL
  - 3PA share NLL
- efficiency / usage losses:
  - efficiency Beta-Binomial NLL
  - efficiency mean aux
  - usage-share CE
- emergent auxiliary losses from zero-latent flow decode:
  - emergent share CE
  - AST / REB structure aux
  - spread / total aux
  - props aux
  - direct stat aux
  - grouped direct boxscore / opportunity aux

This means the current trainer still asks one shared representation to do all of
the following simultaneously:

- stabilize rotation/minutes
- fit joint player count distributions
- fit game totals / spreads
- fit possession environment
- fit make rates
- fit share allocation
- fit player props and direct stat reconstruction

That is too much coupling for clean diagnosis.

Current assessment:

- the `v2` factorization is still the right one:
  - flow for attempts / opportunity / peripherals
  - efficiency head for make rates
- possession backbone should remain separate from flow, not merged into it
- usage-share head is currently optional and likely redundant with emergent share
  supervision in the early recipe
- market / props / direct-stat auxiliaries should not be in the first rates
  retrain recipe

Recommended stripped rates recipe:

- keep the current minutes base fixed:
  - `minutesdist_mtfanneal`
- train rates/game context with:
  - `flow_target_schema=v2`
  - `enable_efficiency_head=true`
  - `enable_usage_share_head=false`
  - `enable_possession_backbone=false`
  - `w_flow_nll > 0`
  - `w_direct_opportunity_aux > 0`
  - `w_direct_boxscore_aux` small but non-zero
  - `w_crps_fpts = 0`
  - `w_team_energy = 0`
  - `w_spread_aux = 0`
  - `w_total_aux = 0`
  - `w_props_* = 0`
  - `w_direct_pts/reb/ast/stl/blk/tov = 0`

Recommended sequencing from here:

1. `v2 flow + efficiency` only
2. verify player means / share structure / coverage
3. optionally add possession backbone back as a separate calibration problem
4. only after that revisit usage-share head or market/props auxiliaries

Working judgment:

- the next rates step should be simplification, not another broad coupled
  retrain
- the current trainer is still closer to a research harness than a single clean
  production recipe on the rates side

### First Stripped Rates Run

First stripped rates/game-context experiment:

- run:
  - `/home/daniel/projections-data/training/runs/gtv2_flow_v2_effonly_stripped_20260327T135311Z`
- recipe:
  - warm-start from `minutesdist_mtfanneal`
  - freeze upstream minutes stack and shared representation:
    - `player_proj`
    - `game_proj`
    - `game_token`
    - `team_tokens`
    - `token_type_embedding`
    - `side_embedding`
    - `encoder`
    - `final_norm`
    - `active_head`
    - `minutes_head`
  - also freeze optional team-context heads to keep the run structurally narrow:
    - `possession_head`
    - `event_backbone`
    - `three_pa_share_head`
    - `usage_share_head`
  - trainable path was effectively:
    - `flow_head`
    - `efficiency_head`
  - active settings:
    - `flow_target_schema=v2`
    - `flow_use_minutes_conditioning=true`
    - `enable_efficiency_head=true`
    - `w_flow_nll=1.0`
    - `w_direct_opportunity_aux=0.15`
    - `w_direct_boxscore_aux=0.05`
    - `w_efficiency_nll=1.0`
  - explicitly zeroed:
    - usage-share loss
    - possession / backbone losses
    - CRPS / team-energy losses
    - spread / total aux
    - props aux
    - direct point-stat auxiliaries

Training behavior:

- run was stable
- minutes metrics stayed fixed, confirming the frozen upstream contract
- flow NLL improved as flow-minutes teacher forcing annealed out
- best checkpoint:
  - epoch `8`
  - `best_val_total = 7.7164`

Quick 12-game / 64-world diagnostic:

- root:
  - `/home/daniel/projections-data/training/runs/gtv2_flow_v2_effonly_stripped_20260327T135311Z/quick_eval_12g64w`
- compared against:
  - `minutesdist_mtfanneal`
- directional result:
  - small player-level gains
  - small coverage gains
  - mixed team-context behavior

Broader 60-game / 128-world replay:

- root:
  - `/home/daniel/projections-data/training/runs/gtv2_flow_v2_effonly_stripped_20260327T135311Z/broad_eval_60g128w`

`minutesdist_mtfanneal` -> `effonly_stripped`:

- improved:
  - `dk_fpts_mae`: `5.613 -> 5.564`
  - `pts_mae_player`: `3.494 -> 3.462`
  - `reb_mae_player`: `1.692 -> 1.673`
  - `stl_mae_player`: `0.533 -> 0.522`
  - `blk_mae_player`: `0.457 -> 0.431`
  - `spread_mae_vs_vegas`: `5.023 -> 4.894`
  - `p90 calibration error abs`: `0.0078 -> 0.0033`
  - `p95 calibration error abs`: `0.0133 -> 0.0044`
  - `top1 share bias pts`: `-0.0072 -> -0.0021`
  - `top2 share bias pts`: `-0.0144 -> -0.0056`

- essentially unchanged:
  - `minutes_mae`
  - `active_acc_at4`
  - `poss_mae`

- worse:
  - `ast_mae_player`: `1.102 -> 1.141`
  - `pts_mae_team`: `10.226 -> 10.503`
  - `total_mae_vs_vegas`: `3.651 -> 5.585`
  - team points bias became much more negative:
    - `-1.20 -> -2.47`

Interpretation:

- the stripped `flow + efficiency` recipe does improve player-level means and
  share calibration
- but it gives back too much team/game-context calibration once spread / total /
  possession-facing structure is removed from training
- so the next rates iteration should not stay fully stripped

Updated rates conclusion:

- `v2 + efficiency` remains the correct factorization
- but some game-context supervision is load-bearing
- the next experiment should add back a small amount of explicit team/game
  structure without reintroducing the full coupled harness

Most likely next candidate:

- keep:
  - stripped `flow + efficiency` base
- add back only one narrow context family at a time:
  - either light spread / total aux
  - or light possession / backbone supervision
- do not re-enable usage-share, props, or broad direct-stat stacks yet

### Narrow Re-Coupling Follow-Up

Two narrow follow-up variants were then tested.

#### 1. Light Spread / Total Aux

- run:
  - `/home/daniel/projections-data/training/runs/gtv2_flow_v2_eff_spreadtotal_20260327T135852Z`
- recipe:
  - same frozen upstream / `flow + efficiency` path as the stripped run
  - add only:
    - `w_spread_aux=0.05`
    - `w_total_aux=0.10`
    - `spread_total_aux_ramp_epochs=3`

Broad 60-game / 128-world replay:

- root:
  - `/home/daniel/projections-data/training/runs/gtv2_flow_v2_eff_spreadtotal_20260327T135852Z/broad_eval_60g128w`

Result:

- this variant was worse than the fully stripped branch
- especially bad on team/game calibration:
  - `total_mae_vs_vegas`: `5.585 -> 7.610`
  - `pts_mae_team`: `10.503 -> 10.738`
- conclusion:
  - light spread/total aux is not the right path
  - the missing structure is not recoverable through market-facing aux alone

#### 2. Light Possession / Backbone Supervision

First attempt:

- run:
  - `/home/daniel/projections-data/training/runs/gtv2_flow_v2_eff_posslight_20260327T140101Z`
- recipe:
  - unfreeze possession backbone family
  - add:
    - `w_poss_nll=0.10`
    - `w_backbone_nll=0.10`
    - `w_three_pa_nll=0.05`
    - `w_poss_regression=0.05`

Result:

- partly recovered team-context metrics relative to the stripped branch:
  - `spread_mae_vs_vegas`: `4.894 -> 4.730`
  - `total_mae_vs_vegas`: `5.585 -> 5.096`
- but possession quality degraded:
  - `poss_mae`: `4.269 -> 5.189`
- conclusion:
  - full possession-light retraining is too unstable

Second attempt: freeze possession head, train only event backbone / 3PA structure.

- run:
  - `/home/daniel/projections-data/training/runs/gtv2_flow_v2_eff_backbonelight_20260327T140309Z`
- recipe:
  - keep `possession_head` frozen
  - train:
    - `flow_head`
    - `efficiency_head`
    - `event_backbone`
    - `three_pa_share_head`
  - weights:
    - `w_backbone_nll=0.10`
    - `w_three_pa_nll=0.05`
    - `w_poss_nll=0.0`
    - `w_poss_regression=0.0`

Broad 60-game / 128-world replay:

- root:
  - `/home/daniel/projections-data/training/runs/gtv2_flow_v2_eff_backbonelight_20260327T140309Z/broad_eval_60g128w`

Compared with `minutesdist_mtfanneal`:

- improved:
  - `dk_fpts_mae`: `5.613 -> 5.564`
  - `pts_mae_player`: `3.494 -> 3.463`
  - `spread_mae_vs_vegas`: `5.023 -> 4.727`
  - `total_mae_vs_vegas`: `3.651 -> 4.927` still worse than base, but much better than stripped
  - coverage errors:
    - `p90`: `0.0078 -> 0.0022`
    - `p95`: `0.0133 -> 0.0033`

- importantly, it preserved possession quality:
  - `poss_mae`: `4.269 -> 4.269` essentially unchanged

- still worse than base on some team/game means:
  - `pts_mae_team`: `10.226 -> 10.445`

Compared with the fully stripped branch:

- it kept almost all of the player-level gain
- recovered a meaningful amount of team/game context
- did so without the possession drift seen in the full possession-light run

Updated working conclusion:

- current best rates branch is now:
  - `eff_backbonelight`
- best structural interpretation:
  - `v2 flow + efficiency` is the correct core
  - some team-event backbone structure is useful
  - the possession head itself should remain fixed for now
  - spread/total aux alone is not a sufficient substitute

### Isolated Possession Retrain

Because `eff_backbonelight` still lagged the base branch on total-game
calibration, an isolated possession-context retrain was run next.

- run:
  - `/home/daniel/projections-data/training/runs/gtv2_flow_v2_possisolated_20260327T141358Z`
- init checkpoint:
  - `eff_backbonelight`
- freeze:
  - shared representation
  - active/minutes
  - flow head
  - efficiency head
  - usage-share head
- train only:
  - `possession_head`
  - `event_backbone`
  - `three_pa_share_head`
- active losses:
  - `w_poss_nll=0.20`
  - `w_backbone_nll=0.15`
  - `w_three_pa_nll=0.05`
  - `w_poss_regression=0.10`
- everything else was zeroed
- checkpoint selection used:
  - `val_total_ex_possreg`

Training behavior:

- early stopped quickly
- best checkpoint was still epoch `1`
- this suggests the environment module was already close to its local optimum
  under the current supervision

Broad 60-game / 128-world replay:

- root:
  - `/home/daniel/projections-data/training/runs/gtv2_flow_v2_possisolated_20260327T141358Z/broad_eval_60g128w`

Compared with `eff_backbonelight`:

- improved:
  - `total_mae_vs_vegas`: `4.927 -> 4.667`
  - `total_corr_vs_vegas`: `0.907 -> 0.940`
  - team points bias moved closer to base:
    - `-2.074 -> -1.291`
  - coverage improved further:
    - `p90`: `0.0022 -> 0.0006`
    - `p95`: `0.0033 -> 0.0017`

- worse:
  - `dk_fpts_mae`: `5.564 -> 5.570`
  - `pts_mae_player`: `3.463 -> 3.475`
  - `pts_mae_team`: `10.445 -> 10.593`
  - `poss_mae`: `4.269 -> 4.734`

Interpretation:

- isolated possession retraining does recover some total-game calibration
- but it does so by degrading pace quality and slightly softening player/team
  means
- so it is not a clean replacement for `eff_backbonelight`

Current rates conclusion after the possession retrain:

- `eff_backbonelight` remains the best mainline rates branch
- `poss_isolated` is a useful calibration variant, especially if total-game
  alignment is prioritized more than pace accuracy
- the environment problem is narrower now:
  - total calibration can be improved
  - but naive possession-head retraining introduces new pace error

## Environment Feature Pass

Question:

- are the remaining game-environment issues partly due to under-specified GTv2
  game/team context, not just loss design?

Feature inventory check:

- the joint dataset already contains usable environment columns:
  - `is_b2b`
  - `team_pace_szn`
  - `team_off_rtg_szn`
  - `team_def_rtg_szn`
  - `opp_pace_szn`
  - `opp_def_rtg_szn`
- but the current GTv2 recipe only fed these as player features
- explicit context inputs were still too thin:
  - `game_feature_columns`:
    - `vegas_total`
    - `vegas_spread`
    - `estimated_possessions`
    - missing flags
  - `team_feature_columns`:
    - empty

Implementation note:

- enabling `team_feature_columns` surfaced a real bug in
  `build_game_level_examples`
- the per-game boolean team mask was being applied against the full
  `team_feats_df` index instead of the per-game slice
- fixed in:
  - `/home/daniel/projects/projections-v2/projections/rotation/game_transformer_v2.py`
- regression test added in:
  - `/home/daniel/projects/projections-v2/tests/rotation/test_game_transformer_v2.py`

Experimental enriched-context recipe:

- run:
  - `/home/daniel/projections-data/training/runs/gtv2_flow_v2_eff_envctx_20260327T150500Z`
- same recipe as `eff_backbonelight`, but with explicit team context:
  - `--team-feature-cols is_b2b,team_pace_szn,team_off_rtg_szn,team_def_rtg_szn,opp_pace_szn,opp_def_rtg_szn`
- broad replay:
  - `/home/daniel/projections-data/training/runs/gtv2_flow_v2_eff_envctx_20260327T150500Z/broad_eval_60g128w`

Broad 60-game / 128-world replay vs `minutesdist_mtfanneal`:

- improved:
  - `pts_mae_team`: `10.226 -> 9.985`
  - `total_mae_vs_vegas`: `3.651 -> 2.591`
  - `total_corr_vs_vegas`: `0.901 -> 0.956`
  - `poss_mae`: `4.269 -> 4.256`
- worse:
  - `dk_fpts_mae`: `5.613 -> 5.699`
  - `pts_mae_player`: `3.494 -> 3.508`
  - `minutes_mae`: `3.721 -> 3.883`
  - `active_acc_at4`: `0.918 -> 0.898`
  - `spread_mae_vs_vegas`: `5.023 -> 5.648`

Interpretation:

- the extra environment context is clearly real signal
- but feeding it through the shared GTv2 sequence is too broad:
  - team totals and possession environment improve materially
  - player/FPTS quality degrades
  - even frozen active/minutes heads drift because their upstream token states
    moved

Follow-up check:

- reran the same recipe with `team_tokens` frozen:
  - `/home/daniel/projections-data/training/runs/gtv2_flow_v2_eff_envctx_projonly_20260327T152000Z`
- result was numerically almost identical during training
- conclusion:
  - the tradeoff is not caused by unfreezing `team_tokens`
  - it comes from routing richer team context through the shared encoder path at
    all

Current conclusion:

- explicit team/game context is probably missing signal, but the right place for
  it is not a blanket shared-sequence injection
- next environment-context work should be narrower:
  - head-specific conditioning for possession / event backbone / flow
  - or another targeted late-fusion path
- for now:
  - `eff_backbonelight` remains the best mainline rates branch
  - `eff_envctx` is evidence that environment features matter, not yet a
    production candidate

## Environment Context Follow-Up (2026-03-27 Late)

We tested three late-fused environment variants after the shared-sequence context pass regressed player quality.

Variants:
- `eff_envadapter`: late-fused env adapter MLP into possession/backbone only
- `eff_envlate_trainposs`: corrected raw late-fused branch with trainable possession head
- `eff_envrich`: raw late-fused branch with richer derived environment features (implied totals, spread magnitude, matchup deltas)

Common setup:
- shared encoder / active / minutes / flow / efficiency frozen
- train only possession/backbone/3PA path
- same 60-game / 128-world replay window as `eff_backbonelight`

Results:
- `eff_envadapter` was a hard regression:
  - `dk_fpts_mae`: `5.588`
  - `pts_mae_team`: `14.13`
  - `total_mae_vs_vegas`: `22.50`
  - `poss_mae`: `9.64`
- `eff_envlate_trainposs` was also a hard regression:
  - `dk_fpts_mae`: `5.594`
  - `pts_mae_team`: `13.78`
  - `total_mae_vs_vegas`: `21.23`
  - `poss_mae`: `9.08`
- `eff_envrich` improved training-side possession NLL but still failed badly on replay:
  - `dk_fpts_mae`: `5.580`
  - `pts_mae_team`: `13.13`
  - `total_mae_vs_vegas`: `18.60`
  - `poss_mae`: `8.02`

Conclusion:
- Shared-sequence env injection is too blunt.
- Late-fused env routing into the possession/backbone stack is still not viable under the current architecture/training contract.
- Richer derived env features improved the training proxy a bit, but not enough to survive replay.
- `eff_backbonelight` remains the best rates/game-context branch.
- Environment-context work should pause unless we are willing to redesign the possession/game-context pathway more substantially.

## Environment Side-Channel Redesign (2026-03-27 Evening)

After the late-fused environment branch failed, we tested a larger redesign
motivated by the idea that game-environment prediction needs its own pathway
instead of depending on the frozen shared encoder.

Architecture change:
- added a standalone `env_side_channel_encoder` MLP in `GameTransformerV2`
- the side-channel consumes the late-fused environment block directly
- its embedding conditions:
  - `flow_head`
  - `possession_head`
  - `event_backbone`
  - `three_pa_share_head`
- shared encoder, active head, and minutes head remained frozen

Training run:
- `/home/daniel/projections-data/training/runs/gtv2_flow_v2_envside_20260327T181500Z`
- best checkpoint metric:
  - `best_val_total = 7.6613`
- this was the best training-side result among all environment-routing variants

Replay result:
- broad replay:
  - `/home/daniel/projections-data/training/runs/gtv2_flow_v2_envside_20260327T181500Z/broad_eval_60g128w_rerun`
- versus `base_minutesdist_mtfanneal`:
  - `dk_fpts_mae`: `5.613 -> 5.627`
  - `pts_mae_player`: `3.494 -> 3.499`
  - `pts_mae_team`: `10.226 -> 13.339`
  - `spread_mae_vs_vegas`: `5.023 -> 5.318`
  - `total_mae_vs_vegas`: `3.651 -> 19.333`
  - `poss_mae`: `4.269 -> 9.672`
- versus `eff_backbonelight`:
  - `dk_fpts_mae`: `5.564 -> 5.627`
  - `pts_mae_team`: `10.445 -> 13.339`
  - `total_mae_vs_vegas`: `4.927 -> 19.333`
  - `poss_mae`: `4.269 -> 9.672`

Interpretation:
- the architectural diagnosis may still be directionally correct
- but this first side-channel implementation does not survive replay
- training-side improvement did not translate to production-aligned world
  generation
- the failure is too large to justify more local tuning on this design

Updated conclusion:
- `eff_backbonelight` still remains the best current rates/game-context branch
- the environment problem is now clearly a larger architecture problem, not a
  local loss-weight or routing tweak problem
- if we revisit this area, it should be treated as a fresh redesign effort with
  explicit replay-first validation

## Full Retrain With Shared Environment Features (2026-03-27 Night)

We then ran the cleanest remaining test of the shared-environment hypothesis:
retrain from the best current rates branch, add explicit team environment
features to the shared model inputs from epoch 1, and let the encoder adapt
instead of freezing around the new signal.

Run:
- `/home/daniel/projections-data/training/runs/gtv2_flow_v2_envfullretrain_20260327T190500Z`

Recipe:
- init from:
  - `/home/daniel/projections-data/training/runs/gtv2_flow_v2_eff_backbonelight_20260327T140309Z/model.pt`
- kept current factorization:
  - `flow_target_schema=v2`
  - `enable_efficiency_head=true`
  - `enable_possession_backbone=true`
  - `enable_three_pa_share=true`
- shared environment inputs:
  - `team_feature_columns = [is_b2b, team_pace_szn, team_off_rtg_szn, team_def_rtg_szn, opp_pace_szn, opp_def_rtg_szn]`
- losses stayed stripped:
  - no spread/total aux
  - no CRPS/team-energy
  - no usage-share loss
- encoder was trainable with a reduced LR multiplier (`0.5`)

Training result:
- best epoch: `11`
- best `val_total = 7.5844`
- this was better than `eff_backbonelight` on the training proxy

Broad replay:
- `/home/daniel/projections-data/training/runs/gtv2_flow_v2_envfullretrain_20260327T190500Z/broad_eval_60g128w`

Versus `base_minutesdist_mtfanneal`:
- `dk_fpts_mae`: `5.613 -> 5.576`
- `pts_mae_player`: `3.494 -> 3.437`
- `reb_mae_player`: `1.692 -> 1.684`
- but:
  - `pts_mae_team`: `10.226 -> 12.704`
  - `spread_mae_vs_vegas`: `5.023 -> 5.324`
  - `total_mae_vs_vegas`: `3.651 -> 17.313`
  - `poss_mae`: `4.269 -> 7.981`
  - `active_acc_at4`: `0.918 -> 0.911`

Versus `eff_backbonelight`:
- slightly better player points:
  - `pts_mae_player`: `3.463 -> 3.437`
- but materially worse game environment:
  - `pts_mae_team`: `10.445 -> 12.704`
  - `total_mae_vs_vegas`: `4.927 -> 17.313`
  - `poss_mae`: `4.269 -> 7.981`

Interpretation:
- this is the strongest negative result so far on game context
- it rules out the simple story that environment features only failed because
  they were injected into a frozen representation
- even when the shared encoder is allowed to adapt from the beginning of the
  rates retrain, the production-aligned world path still fails badly on game
  environment

Current conclusion:
- explicit environment features are still real signal
- but the current GTv2/shared-world-generation contract is not able to absorb
  them cleanly
- `eff_backbonelight` remains the best retained rates/game-context branch
- further game-context work should be treated as a larger redesign, not another
  incremental retrain on this path

### Follow-Up Diagnostic: Where `envfullretrain` Fails

We explicitly compared `envfullretrain` before and after the production-style
world post-processing stack.

Result:
- the branch is already bad in `raw_worlds.parquet`
- realism controls / contract repair only change the numbers slightly

Representative comparison:
- `eff_backbonelight`
  - raw:
    - `team_pts_mae = 10.439`
    - `poss_mae = 4.267`
    - `total_mae_vs_vegas = 4.924`
  - post:
    - `team_pts_mae = 10.438`
    - `poss_mae = 4.269`
    - `total_mae_vs_vegas = 5.119`
- `envfullretrain`
  - raw:
    - `team_pts_mae = 12.701`
    - `poss_mae = 7.991`
    - `total_mae_vs_vegas = 17.299`
  - post:
    - `team_pts_mae = 12.791`
    - `poss_mae = 7.981`
    - `total_mae_vs_vegas = 17.632`

Direct head diagnostic on the same games:
- `eff_backbonelight`
  - possession-head mean vs estimated possessions:
    - `MAE = 1.94`
    - predicted mean `102.03` vs estimate mean `103.51`
- `envfullretrain`
  - possession-head mean vs estimated possessions:
    - `MAE = 7.58`
    - predicted mean `95.97` vs estimate mean `103.51`

Interpretation:
- `envfullretrain` does not mainly fail in post-processing
- it fails earlier, at the possession/backbone prediction stage
- the branch was pushed into a low-possession regime before world sampling

### Cleaner Shared-Team-Context Retrain (2026-03-27 Night)

The previous `envfullretrain` was confounded because it also changed
`backbone_env_feature_cols`, which reinitialized the possession/backbone heads.

We reran a cleaner test:
- add `team_feature_columns`
- keep `backbone_env_feature_columns = []`
- warm-start the existing possession/backbone heads unchanged

Run:
- `/home/daniel/projections-data/training/runs/gtv2_flow_v2_envteamonly_20260327T193500Z`

Training:
- only missing warm-start params:
  - `team_proj.weight`
  - `team_proj.bias`
- best `val_total = 7.5643`

Broad replay:
- `/home/daniel/projections-data/training/runs/gtv2_flow_v2_envteamonly_20260327T193500Z/broad_eval_60g128w`

Versus `base_minutesdist_mtfanneal`:
- improved:
  - `dk_fpts_mae`: `5.613 -> 5.576`
  - `minutes_mae`: `3.721 -> 3.705`
  - `pts_mae_player`: `3.494 -> 3.458`
  - `pts_mae_team`: `10.226 -> 10.090`
  - `total_mae_vs_vegas`: `3.651 -> 3.278`
  - `total_corr_vs_vegas`: `0.901 -> 0.922`
- worse:
  - `active_acc_at4`: `0.918 -> 0.908`
  - `spread_mae_vs_vegas`: `5.023 -> 5.282`
  - `poss_mae`: `4.269 -> 4.366`

Versus `eff_backbonelight`:
- improved:
  - `pts_mae_team`: `10.445 -> 10.090`
  - `total_mae_vs_vegas`: `4.927 -> 3.278`
  - `pts_mae_player`: `3.463 -> 3.458`
- slightly worse:
  - `dk_fpts_mae`: `5.564 -> 5.576`
  - `poss_mae`: `4.269 -> 4.366`
  - `spread_mae_vs_vegas`: `4.727 -> 5.282`

Structural check:
- direct possession-head mean vs estimated possessions:
  - `eff_backbonelight`: `MAE = 1.94`, mean `102.03`
  - `envteamonly`: `MAE = 2.66`, mean `101.57`
- this is much closer to the healthy branch than `envfullretrain`

Updated conclusion:
- the shared-team-context idea is not dead
- the failure in `envfullretrain` was at least partly caused by changing the
  backbone input contract and effectively resetting the environment heads
- `envteamonly` is the first game-context branch that survives broad replay
- it is now the best candidate on this path, with a real tradeoff:
  - better totals and team points
  - slightly worse spread and active accuracy

### Narrow Spread Follow-Up (`envteamonly_spread001`)

We then ran the smallest plausible follow-up from `envteamonly`:
- same architecture
- same shared team-context inputs
- only a tiny spread auxiliary (`w_spread_aux=0.01`)

Run:
- `/home/daniel/projections-data/training/runs/gtv2_flow_v2_envteamonly_spread001_20260327T201500Z`

Broad replay:
- `/home/daniel/projections-data/training/runs/gtv2_flow_v2_envteamonly_spread001_20260327T201500Z/broad_eval_60g128w`

Result versus `envteamonly`:
- slight improvements:
  - `minutes_mae`: `3.705 -> 3.675`
  - `active_acc_at4`: `0.908 -> 0.912`
  - `spread_mae_vs_vegas`: `5.282 -> 5.214`
- but meaningful regressions:
  - `dk_fpts_mae`: `5.576 -> 5.598`
  - `pts_mae_player`: `3.458 -> 3.465`
  - `pts_mae_team`: `10.090 -> 10.281`
  - `total_mae_vs_vegas`: `3.278 -> 5.588`
  - `poss_mae`: `4.366 -> 4.381`
  - team points bias became much more negative

Tail audit on actual ceiling outcomes:
- artifact:
  - `/home/daniel/projections-data/training/runs/gtv2_flow_v2_envteamonly_spread001_20260327T201500Z/broad_eval_60g128w/tail_slice_eval.csv`
- actual DK `>= 45`, `p95` coverage:
  - `eff_backbonelight`: `0.713`
  - `envteamonly`: `0.694`
  - `envteamonly_spread001`: `0.593`
- actual DK `>= 55`, `p95` coverage:
  - `eff_backbonelight`: `0.469`
  - `envteamonly`: `0.438`
  - `envteamonly_spread001`: `0.281`

Interpretation:
- overall p90 calibration looked better only because the branch got more conservative broadly
- on true ceiling outcomes, tails got materially worse
- this is not a useful spread recovery path

Decision:
- retain `envteamonly`
- drop `envteamonly_spread001`

### Clean Scratch Check Before Live Consideration

We then tested the obvious remaining question: is `envteamonly` only strong
because it sits on top of a warm-start chain?

#### Attempt A: direct scratch under the continuation recipe

Run:
- `/home/daniel/projections-data/training/runs/gtv2_flow_v2_envteamonly_fullscratch_20260327T174308Z`

Setup:
- same high-level contract as `envteamonly`
- no checkpoint init
- immediate phase-2 flow

Result:
- failed in epoch 1
- `train_flow_nll = 48.69`
- three phase-2 backoffs in six batches
- rollback triggered
- validation metrics became `NaN`

Interpretation:
- the continuation recipe is not self-starting from random initialization

#### Attempt B: staged scratch with a saner curriculum

Run:
- `/home/daniel/projections-data/training/runs/gtv2_flow_v2_envteamonly_fullscratch_stable_20260327T174455Z`

Changes:
- lower LR: `5e-4`
- delayed flow:
  - `phase2_flow_delay_epochs = 2`
  - `phase2_flow_warmup_epochs = 8`
  - `phase2_anchor_end_weight = 0.75`
- relaxed guard:
  - `phase2_nll_guard_abs = 250`
  - `phase2_max_backoffs_before_rollback = 20`
- no checkpoint init
- kept the same final branch contract:
  - `v2` flow
  - minutes hurdle head
  - flow minutes conditioning with `1.0 -> 0.0` TF anneal
  - efficiency head
  - possession backbone
  - shared team-context columns
  - grouped direct opportunity / box-score losses

Result:
- trained cleanly
- best checkpoint:
  - epoch `3`
  - `best_val_total = 8.4672`

Comparison:
- retained warm-start `envteamonly`:
  - epoch `12`
  - `best_val_total = 7.5643`

Conclusion:
- scratch training is possible with a curriculum
- but the current scratch recipe still leaves substantial quality on the table
- the retained `envteamonly` branch is still materially better than the clean
  scratch alternative


## 2026-03-28 Live Deployment Follow-Up

### What We Confirmed
- Live input schema is aligned with the promoted bundle contract.
- The live builder is emitting the expected player/game/team features.
- Priors are generally present; the main live failures are not explained by missing priors.
- A live surface semantics bug existed: GTv2-facing `*_mean` columns were exposing conditional-on-active means. That is now fixed; unconditional means are the default surfaced values and conditional values are preserved under `*_mean_cond` aliases.
- A runtime availability contract bug existed: `is_out=1` players could still receive minutes/world mass. That is now fixed in inference/world generation with a pre-mask plus final hard-zero safety check.

### What Failed In Live / Live-Aligned Evaluation
- The retained `envteamonly` branch is still materially under-allocating top-end player stats, especially assists and high-end scoring.
- Example failure on the 2026-03-27 live slate:
  - Andrew Nembhard remained far below his AST market line even after the surface semantics fix.
- High-end slices remain suppressed:
  - AST line `>= 7.0`: materially undercalled
  - PTS line `>= 25.0`: materially undercalled
  - Top-end REB also appears suppressed
- This is not primarily a feature-contract bug.
- This is not primarily a missing-priors bug.
- The current issue is modeling / objective behavior: the branch is flattening top-end allocation.

### What We Tried
1. Generic share supervision
- `share-only` and `share + usage-share` retries were not keepers.
- They did not fix the real top-end failure mode and worsened broader quality.

2. Strict targeted top-end supervision
- First successful branch:
  - `gtv2_flow_v2_envteamonly_topendprops_20260328T001500Z`
- Settings:
  - `w_props_ast_aux = 0.05`
  - `w_props_pts_aux = 0.03`
  - `props_ast_aux_min_line = 6.0`
  - `props_pts_aux_min_line = 20.0`
- This is the first approach that moved the real live failure mode in the right direction.

3. AST-only follow-up
- `gtv2_flow_v2_envteamonly_topendast_20260327T211200Z`
- It moved assists harder, but broke scoring and game totals.
- Conclusion: a scoring anchor is required; pure AST supervision is too destabilizing.

4. Nearby mixed follow-ups
- Stronger AST + smaller PTS anchor did not beat `topendprops` on the training proxy.
- Current best local balance remains the original `topendprops` setting.

### Current Best Read
- `topendprops` is the best branch on this specific problem so far.
- It improves high-end AST/PTS allocation relative to retained `envteamonly`.
- It is still not good enough.
- We are still materially off on the top end, especially for playmakers and likely high-end rebounders.
- REB should be handled as a separate targeted problem, not folded into AST/PTS tuning until the current branch is more stable.

### Practical Conclusion
- Current evidence says the live issue is not a builder/input contract failure.
- The main remaining problem is top-end allocation flattening in the model.
- Generic structural share losses were not enough.
- Targeted top-end supervision is the right direction, but the current branch has not closed the gap yet.


### Decode-Time Top-Usage Reweighting Proof Of Concept
Research-only change added to the worlds sampler:
- opt-in decode multipliers for top implied-usage players after emergent allocation weights are computed
- parameters:
  - `allocation_top_usage_top1_scale`
  - `allocation_top_usage_top2_scale`
- defaults are `1.0`, so production behavior is unchanged unless explicitly enabled

Live-slate POC on retained `envteamonly` branch:
- Artifact:
  - `/home/daniel/projections-data/training/runs/gtv2_flow_v2_envteamonly_20260327T193500Z/decode_top_usage_reweight_live_slate_eval_20260327T190000Z.json`
- Compared:
  - base `1.00 / 1.00`
  - mild `1.08 / 1.04`
  - moderate `1.12 / 1.06`

Findings:
- Top-end scoring improved modestly:
  - PTS line `>= 25.0` mean diff improved from `-7.72` to `-7.16` (`1.08/1.04`) and `-6.89` (`1.12/1.06`)
  - overall PTS mean diff vs props improved slightly from `-0.811` to `-0.765` / `-0.741`
- Team totals worsened slightly:
  - mean absolute total error vs market moved from `3.95` to `4.09` / `4.13`
- AST did not move at all:
  - Andrew Nembhard AST stayed `3.03`
  - overall AST mean diff vs props stayed `-0.736`
  - AST line `>= 7.0` mean diff stayed `-3.04`

Interpretation:
- Decode-time top-usage reweighting is a real lever for top-end scoring concentration.
- It does not address the assist suppression problem because AST is not reconstructed through the current FGA/FTA/TOV budget-allocation path.
- This is an important localization result:
  - high-end PTS flattening is at least partly an opportunity-share magnitude problem
  - high-end AST suppression is elsewhere, likely in direct flow stat generation / AST structure rather than the decode allocator
- Therefore, Opus's decode-side idea is directionally right for scoring, but it does not explain the current AST miss by itself.

Current decision:
- keep the decode reweighting path as a research tool
- do not promote it to live
- treat PTS and AST as partially separate mechanisms from here


### First AST-Factorized Branch
Implemented a first explicit AST factorization path in the model/trainer:
- `TeamAstBudgetHead`
- `AssistShareHead`
- three optional losses:
  - `w_team_ast_budget_aux`
  - `w_assist_share_aux`
  - `w_assist_share_recon_aux`

Rationale:
- decode-time top-usage reweighting helped PTS but did nothing for AST
- generic share supervision was not enough
- AST appears to need an explicit:
  - team assist budget
  - passer allocation within that budget

First stable continuation run:
- `/home/daniel/projections-data/training/runs/gtv2_flow_v2_envteamonly_astfactor_stable_20260328T014500Z`

Recipe:
- warm-start from retained `envteamonly`
- keep game/team context contract unchanged
- freeze rotation/environment blocks:
  - encoder / projections / tokens
  - active/minutes
  - possession/event backbone
  - efficiency
  - usage-share
- train only:
  - flow head
  - new AST heads
- stabilize phase 2:
  - lower LR
  - longer flow warmup
  - AST losses only
  - backbone/effect head losses set to zero

Training result:
- `envteamonly best_val_total = 7.5643`
- `astfactor_stable best_val_total = 7.2214`

Live-slate raw-world eval:
- Artifact:
  - `/home/daniel/projections-data/training/runs/gtv2_flow_v2_envteamonly_astfactor_stable_20260328T014500Z/live_slate_eval_20260327T190000Z.json`
- Compared against `envteamonly` on the same raw-world evaluation path:
  - AST overall mean diff vs props:
    - `-0.757 -> -0.589`
  - AST overall MAE:
    - `1.072 -> 1.019`
  - AST line `>= 7.0` mean diff:
    - `-3.165 -> -2.800`
  - AST line `>= 7.0` MAE:
    - `3.165 -> 2.958`
  - total absolute error vs market:
    - `4.182 -> 3.733`

Other slice effects:
- PTS overall mean diff vs props improved modestly:
  - `-0.823 -> -0.662`
- PTS line `>= 25.0` improved modestly:
  - `-7.665 -> -7.441`
- High-end REB got worse:
  - REB line `>= 10.0` mean diff:
    - `-4.317 -> -5.242`

Named example:
- Andrew Nembhard (`player_id=1629614`, IND):
  - live market AST line: `7.67`
  - raw-world AST:
    - `envteamonly`: `3.05`
    - `astfactor_stable`: `3.06`

Interpretation:
- The first AST factorization branch is a real improvement on AST slices in aggregate.
- It does not move the named high-end playmaker example enough.
- This means:
  - the factorization is directionally useful
  - but the first pass is still too weak or too constrained to solve the real top-end AST miss
- It is a better base for AST work than `topendprops`, because it improves AST without reintroducing the earlier total-collapse failure mode.

Current decision:
- keep `astfactor_stable` as the active AST research branch
- do not touch REB inside this branch yet
- next AST iteration should stay on this factorization path, not return to generic share or props aux losses

## AST Follow-up: Playmaker-Conditioned AssistShareHead

Date: 2026-03-28

Run:
- `/home/daniel/projections-data/training/runs/gtv2_flow_v2_astfactor_conditioned_cuda_20260328T025500Z`
- Eval:
  - `/home/daniel/projections-data/training/runs/gtv2_flow_v2_astfactor_conditioned_cuda_20260328T025500Z/live_slate_eval_20260327T190000Z.json`

What changed:
- Held the `astfactor_stable` recipe constant.
- Added explicit playmaker-conditioning features to `AssistShareHead`:
  - `an_ast_line`
  - `an_implied_minutes`
  - `prior_play_prob`
  - `started_proxy_rate_prior_20`
- The features are unnormalized inside GTv2 and routed only into the assist-share head.

Result:
- Training proxy improved:
  - `7.2214 -> 7.1465`
- Named high-end playmaker improved modestly:
  - Nembhard AST `3.14 -> 3.28`
- Aggregate AST changed only slightly:
  - AST overall mean diff vs props `-0.589 -> -0.570`
  - AST line `>= 7.0` mean diff `-2.800 -> -2.794`
  - AST line `>= 7.0` MAE `2.958 -> 3.042`
- Side effects:
  - total absolute error vs market worsened `3.733 -> 4.096`
  - PTS overall mean diff worsened `-0.662 -> -0.843`

Conclusion:
- This confirms the AST head benefits from explicit playmaker-side inputs.
- But the effect size is still too small on the actual high-end AST slice.
- The branch is not a keeper.
- The remaining gap likely requires stronger structural integration of AST factorization with the flow path rather than further light head-local conditioning.

## AST Follow-up: Replace Flow AST With Reconstructed AST

Date: 2026-03-28

Run:
- `/home/daniel/projections-data/training/runs/gtv2_flow_v2_astfactor_replaceast_cuda_20260328T031500Z`
- Eval:
  - `/home/daniel/projections-data/training/runs/gtv2_flow_v2_astfactor_replaceast_cuda_20260328T031500Z/live_slate_eval_20260327T190000Z.json`

What changed:
- Held the `astfactor_conditioned` recipe constant.
- Replaced the projected flow AST channel with the explicit AST reconstruction:
  - `team_ast_budget * assist_share`
- This is a stronger structural test than the prior AST auxiliary branches because the reconstructed AST now directly feeds the boxscore/world contract.

Result:
- Training proxy was unchanged:
  - `7.1465 -> 7.1465`
- But the raw-world AST behavior changed materially.
- Named high-end playmaker:
  - Nembhard AST `3.14 -> 5.59`
- Aggregate AST:
  - AST overall mean diff vs props `-0.589 -> -0.503`
  - AST overall MAE `1.019 -> 1.144`
  - AST line `>= 7.0` mean diff `-2.800 -> +1.914`
  - AST line `>= 7.0` MAE `2.958 -> 2.508`
- Side effects:
  - total absolute error vs market worsened `3.733 -> 4.229`
  - PTS overall mean diff worsened `-0.662 -> -0.855`
  - PTS line `>= 25.0` mean diff worsened `-7.441 -> -8.596`

Conclusion:
- The localization is useful.
- Stronger structural AST integration does move the high-end playmaker problem.
- But this specific mechanism is too blunt:
  - it overshoots the high-AST slice
  - and it gives back points/totals quality
- This is not a keeper, but it is better evidence than the prior auxiliary attempts that the remaining issue is architectural, not just a missing loss term.

## AST Follow-up: Remove AST From Flow Supervision

Date: 2026-03-28

Run:
- `/home/daniel/projections-data/training/runs/gtv2_flow_v2_astfactor_factorized_v2_20260328T034500Z`
- Eval:
  - `/home/daniel/projections-data/training/runs/gtv2_flow_v2_astfactor_factorized_v2_20260328T034500Z/live_slate_eval_20260327T190000Z.json`

What changed:
- Started from `astfactor_stable`.
- Added `assist_share_factorized_ast` mode.
- AST is removed from flow supervision:
  - masked out of `flow_targets`
  - masked out of `flow_observed_mask`
  - removed from direct flow-stat AST losses
- Raw-world AST is still sourced from:
  - `team_ast_budget * assist_share`

Result:
- Training proxy improved slightly:
  - `7.2214 -> 7.2012`
- High-end AST improved materially without the overshoot of `replaceast`:
  - Nembhard AST `3.14 -> 3.63`
  - AST overall mean diff vs props `-0.589 -> -0.456`
  - AST line `>= 7.0` mean diff `-2.800 -> -1.782`
  - AST line `>= 7.0` MAE `2.958 -> 2.186`
- Side effects:
  - total absolute error vs market worsened `3.733 -> 3.976`
  - PTS overall mean diff worsened `-0.662 -> -0.974`
  - PTS line `>= 25.0` mean diff worsened `-7.441 -> -8.631`

Conclusion:
- This is the best AST branch so far in terms of solving the actual high-AST slice without blowing through into overprediction.
- It validates the structural direction:
  - AST should not be a standard flow-supervised channel.
- But the result is still not strong enough to keep:
  - the named playmaker gap remains too large
  - scoring quality gives back too much
- The next AST step should be deeper architecture:
  - remove AST from the flow architecture itself, not just from its supervision
  - inject factorized AST as a first-class generative path before stat-budget reconciliation

## AST Follow-up: RQS Coupling Ablation

Date: 2026-03-28

Run:
- `/home/daniel/projections-data/training/runs/gtv2_flow_v2_astfactor_factorized_rqs_20260328T040500Z`
- Eval:
  - `/home/daniel/projections-data/training/runs/gtv2_flow_v2_astfactor_factorized_rqs_20260328T040500Z/live_slate_eval_20260327T190000Z.json`

What changed:
- Held the `astfactor_factorized_v2` recipe constant.
- Switched only the flow coupling type:
  - `affine -> rqs`

Result:
- Training proxy improved:
  - `7.2012 -> 7.1243`
- High-end AST improved further:
  - Nembhard AST `3.63 -> 3.78`
  - AST overall mean diff vs props `-0.456 -> -0.414`
  - AST line `>= 7.0` mean diff `-1.782 -> -0.916`
  - AST line `>= 7.0` MAE `2.186 -> 1.674`
- Tail slices outside AST also improved modestly:
  - PTS line `>= 25.0` mean diff `-8.631 -> -8.090`
  - REB line `>= 10.0` mean diff `-5.467 -> -5.003`
- Side effects:
  - AST overall MAE worsened `1.079 -> 1.162`
  - PTS overall MAE worsened slightly `3.025 -> 3.055`
  - total absolute error vs market worsened `3.976 -> 4.281`

Conclusion:
- RQS does seem to help tail expression mechanically.
- It is a better coupling choice than affine for this AST structural branch.
- But it is not enough by itself:
  - the named playmaker gap is still large
  - broad calibration still gives back too much
- The recommendation remains the same:
  - deeper architectural AST integration is still required

## AST Runtime Calibration Sweep

Date: 2026-03-28

Base branch:
- `/home/daniel/projections-data/training/runs/gtv2_flow_v2_astfactor_factorized_rqs_20260328T040500Z`

Sweep artifact:
- `/home/daniel/projections-data/training/runs/gtv2_flow_v2_astfactor_factorized_rqs_20260328T040500Z/ast_runtime_calibration_sweep_live_slate_eval_20260327T190000Z.json`

What changed:
- Did not retrain.
- Added three research-only AST runtime knobs on top of the current RQS factorized-AST branch:
  - blend between flow AST and factorized AST
  - assist-share temperature
  - blend between explicit team AST budget and flow-implied team AST

Best setting:
- `ast_blend_alpha = 0.75`
- `assist_share_temperature = 0.85`
- `team_ast_budget_blend_alpha = 1.0`

Result vs base:
- Nembhard AST `3.78 -> 3.81`
- AST overall mean diff vs props `-0.416 -> -0.392`
- AST line `>= 7.0` mean diff `-0.916 -> -0.511`
- AST line `>= 7.0` MAE `1.674 -> 1.570`
- total absolute error vs market `4.473 -> 2.760`

Interpretation:
- The practical wins are:
  - partial AST blend
  - sharper assist-share allocation
- Team AST budget blending did not help in this pass.
- This does not remove the need for deeper architecture, but it is a strong hint about what that architecture should look like:
  - the next model should learn this AST-flow blend rather than hard-replacing AST.

## Small RQS Hyperparameter Sweep

Date: 2026-03-28

Run:
- `/home/daniel/projections-data/training/runs/gtv2_flow_v2_astfactor_factorized_rqs_b12_tb60_20260328T042500Z`

What changed:
- Held the RQS factorized-AST recipe constant.
- Increased only:
  - `num_bins: 8 -> 12`
  - `tail_bound: 40 -> 60`

Result:
- baseline RQS factorized AST:
  - `best_val_total = 7.1243`
- `bins=12, tail=60`:
  - `best_val_total = 7.1644`

Conclusion:
- The larger spline setting was not better.
- The useful gains are coming from AST-specific calibration, not from more RQS capacity.

## Learned AST Blend Gate

Date: 2026-03-28

Implementation:
- Added a new `AstBlendGateHead` and wired it through:
  - [assist_heads.py](/home/daniel/projects/projections-v2/projections/rotation/assist_heads.py)
  - [game_transformer_v2.py](/home/daniel/projects/projections-v2/projections/rotation/game_transformer_v2.py)
  - [sample_worlds_v2.py](/home/daniel/projects/projections-v2/projections/rotation/sample_worlds_v2.py)
  - [train_game_transformer_v2.py](/home/daniel/projects/projections-v2/scripts/rotation/train_game_transformer_v2.py)

Validation:
- `78 passed`
- `ruff` clean

Run:
- `/home/daniel/projections-data/training/runs/gtv2_flow_v2_astfactor_learnedgate_rqs_20260328T050900Z`

Training result:
- baseline RQS factorized AST:
  - `best_val_total = 7.1243`
- learned-gate:
  - `best_val_total = 7.2552`

Live-slate raw-world eval:
- `/home/daniel/projections-data/training/runs/gtv2_flow_v2_astfactor_learnedgate_rqs_20260328T050900Z/live_slate_eval_20260327T190000Z.json`

Result vs `astfactor_factorized_rqs`:
- Nembhard AST:
  - `3.78 -> 4.21`
- AST overall mean diff vs props:
  - `-0.414 -> +0.109`
- AST line `>= 7.0` mean diff:
  - `-0.916 -> -2.294`
- total absolute error vs market:
  - `4.281 -> 13.365`

Most important diagnostic:
- The gate stayed constant at its initialization:
  - mean / p25 / p50 / p75 / max all `~= 0.75`
- Nembhard gate was also `~= 0.75`

Interpretation:
- The gate did not learn a player- or context-specific blend.
- Under the current loss recipe, it effectively had no meaningful training signal.
- So this branch mostly tested a fixed `0.75` runtime blend, not a trained learned-gate policy.

Decision:
- Reject this learned-gate branch as currently trained.
- If learned gating is revisited, add an explicit supervision path for the gate itself:
  - phase-3 / world-level training,
  - or gated AST losses on the emergent-flow path.

## Supervised AST Blend Gate

Date: 2026-03-28

Follow-up implementation:
- Added explicit gate supervision in:
  - [train_game_transformer_v2.py](/home/daniel/projects/projections-v2/scripts/rotation/train_game_transformer_v2.py)
- New pieces:
  - `w_ast_blend_gate_aux`
  - `ast_blend_gate_target_eps`
  - `_ast_blend_gate_targets(...)`

Important implementation notes:
- The first supervised-gate attempt exposed two real issues:
  - the gate loss path was not active because `w_ast_blend_gate_aux` was missing from the emergent-flow guard
  - the initial target builder could still produce unstable divides on unsolved rows
- Both were fixed before the retained run.

Retained run:
- `/home/daniel/projections-data/training/runs/gtv2_flow_v2_astfactor_learnedgate_supervised_rqs_fix2_20260328T055200Z`

Training result:
- `best_epoch = 11`
- `best_val_total = 7.1944`
- baseline RQS factorized AST remained better:
  - `7.1243`

Critical training diagnostic:
- The gate loss was active and nonzero after the fixes.
- Example epochs from `history.json`:
  - epoch `3`: `train_ast_blend_gate_aux = 0.6613`, `val_ast_blend_gate_aux = 0.6327`
  - epoch `11`: `train_ast_blend_gate_aux = 0.2036`, `val_ast_blend_gate_aux = 0.1786`

Gate distribution on the live slate:
- mean: `0.9489`
- p25/p50/p75: `0.9339 / 0.9572 / 0.9746`
- min/max: `0.8325 / 0.9981`
- Nembhard gate: `0.9247`

Live-slate raw-world eval:
- `/home/daniel/projections-data/training/runs/gtv2_flow_v2_astfactor_learnedgate_supervised_rqs_fix2_20260328T055200Z/live_slate_eval_20260327T190000Z.json`

Result vs `astfactor_factorized_rqs`:
- Nembhard AST:
  - `3.78 -> 4.84`
- AST overall mean diff vs props:
  - `-0.414 -> +0.512`
- AST line `>= 7.0` mean diff:
  - `-0.916 -> -0.892`
- AST line `>= 7.0` MAE:
  - `1.674 -> 1.417`
- total absolute error vs market:
  - `4.281 -> 16.697`
- overall PTS mean diff vs props:
  - `-0.924 -> -1.450`

Interpretation:
- Explicit supervision does make the gate learn.
- The learned gate is too aggressive and effectively collapses toward factorized AST for most players.
- That improves the named playmaker miss and modestly improves the high-AST slice, but it destroys totals and hurts scoring.

Decision:
- Reject the supervised learned-gate branch as a keeper.
- Learned gating is trainable, but the current supervision target is not the right one.
- If we come back to gating, it needs a more coupled downstream objective so totals and scoring constrain the gate directly.

## AST Follow-up: Compare Phase-3 Gate Supervision vs AST Reconciliation

Date: 2026-03-28

Two deeper follow-ups were evaluated against the retained RQS AST baseline:
- baseline:
  - `/home/daniel/projections-data/training/runs/gtv2_flow_v2_astfactor_factorized_rqs_20260328T040500Z`
- candidate 1, phase-3/world-level learned gate:
  - `/home/daniel/projections-data/training/runs/gtv2_flow_v2_astfactor_learnedgate_phase3_rqs_20260328T070800Z`
- candidate 2, AST reconciliation into the team budget:
  - `/home/daniel/projections-data/training/runs/gtv2_flow_v2_astreconcile_rqs_20260328T071800Z`

Implementation notes:
- Phase-3 gate branch:
  - kept the learned AST-flow blend gate
  - enabled phase-3 decision/world losses
  - removed direct gate-target supervision
- AST reconciliation branch:
  - added `assist_share_reconcile_ast_budget`
  - reconciles player AST to the exact team AST budget using a blended share:
    - emergent flow AST share
    - assist-share head weights
  - uses:
    - `assist_share_reconcile_alpha = 0.75`
    - `assist_share_reconcile_temperature = 0.85`

Training result:
- baseline RQS factorized AST:
  - `best_val_total = 7.1243`
- phase-3 gate:
  - `best_val_total = 7.4595`
- AST reconciliation:
  - `best_val_total = 6.8576`

Live-slate raw-world eval:
- baseline:
  - `/home/daniel/projections-data/training/runs/gtv2_flow_v2_astfactor_factorized_rqs_20260328T040500Z/live_slate_eval_20260327T190000Z.json`
- phase-3 gate:
  - `/home/daniel/projections-data/training/runs/gtv2_flow_v2_astfactor_learnedgate_phase3_rqs_20260328T070800Z/live_slate_eval_20260327T190000Z.json`
- AST reconciliation:
  - `/home/daniel/projections-data/training/runs/gtv2_flow_v2_astreconcile_rqs_20260328T071800Z/live_slate_eval_20260327T190000Z.json`

Key comparison:
- baseline:
  - Nembhard AST: `3.78`
  - AST overall mean diff vs props: `-0.414`
  - AST line `>= 7.0` mean diff: `-0.916`
  - AST line `>= 7.0` MAE: `1.674`
  - PTS line `>= 25` mean diff: `-8.090`
  - total absolute error vs market: `4.281`
- phase-3 gate:
  - Nembhard AST: `2.56`
  - AST overall mean diff vs props: `-1.659`
  - AST line `>= 7.0` mean diff: `-4.026`
  - AST line `>= 7.0` MAE: `4.026`
  - PTS line `>= 25` mean diff: `-7.549`
  - total absolute error vs market: `16.033`
- AST reconciliation:
  - Nembhard AST: `5.72`
  - AST overall mean diff vs props: `-0.014`
  - AST line `>= 7.0` mean diff: `+1.896`
  - AST line `>= 7.0` MAE: `2.578`
  - PTS line `>= 25` mean diff: `-6.966`
  - total absolute error vs market: `15.433`

Interpretation:
- Phase-3/world-level gate supervision is not the right path in the current setup.
  - It is worse than baseline on both the training proxy and the live-slate AST slices.
  - It also blows up team-total calibration.
- AST reconciliation is the first branch that moves the named playmaker miss hard while also nearly eliminating aggregate AST underprediction.
  - But it currently overshoots the high-AST slice and collapses total calibration.

Current conclusion:
- The reconciliation mechanism is the more promising deep direction.
- The phase-3 gate branch is rejected.
- The next AST architecture work should build from reconciliation, but it must introduce a way for totals/scoring to constrain the AST reallocation.

## AST Reconciliation Follow-up: Constraining Training Coupling

Date: 2026-03-28

Three follow-ups were tested to determine whether the reconciliation branch was failing because AST supervision was distorting the scoring path, or because the reconciliation mechanism itself was still unconstrained.

### 1. Reconciliation + Direct Losses

Run:
- `/home/daniel/projections-data/training/runs/gtv2_flow_v2_astreconcile_rqs_direct_20260328T082500Z`

What changed:
- kept AST reconciliation active
- restored `envteamonly`-style direct losses:
  - `w_direct_boxscore_aux = 0.05`
  - `w_direct_opportunity_aux = 0.15`
- turned off phase-3 losses:
  - `w_crps_fpts = 0.0`
  - `w_team_energy = 0.0`

Training result:
- `best_val_total = 6.9587`
- better than AST baseline `7.1243`
- worse than unconstrained reconciliation `6.8576`

Live-slate raw-world eval:
- `/home/daniel/projections-data/training/runs/gtv2_flow_v2_astreconcile_rqs_direct_20260328T082500Z/live_slate_eval_20260327T190000Z.json`

Result vs AST baseline:
- Nembhard AST:
  - `3.78 -> 6.24`
- AST overall mean diff vs props:
  - `-0.414 -> -0.091`
- AST line `>= 7.0` mean diff:
  - `-0.916 -> +2.578`
- total absolute error vs market:
  - `4.281 -> 16.181`

Interpretation:
- Restoring direct boxscore/opportunity losses does not rescue reconciliation.
- The branch still over-pushes AST and still breaks totals.

### 2. Reconciliation + Actual Phase-3 / World-Level Supervision

Run:
- `/home/daniel/projections-data/training/runs/gtv2_flow_v2_astreconcile_phase3_rqs_20260328T084500Z`

What changed:
- kept AST reconciliation active
- enabled phase-3 decision training
- used light world-level losses:
  - `w_crps_fpts = 0.05`
  - `w_team_energy = 0.02`
  - `phase3_num_samples = 8`

Training result:
- `best_val_total = 7.6018`

Interpretation:
- This is clearly worse than:
  - AST baseline `7.1243`
  - unconstrained reconciliation `6.8576`
- Light phase-3/world-level supervision is not enough to stabilize the reconciliation branch.

### 3. Reconciliation With Flow Head Frozen

Run:
- `/home/daniel/projections-data/training/runs/gtv2_flow_v2_astreconcile_rqs_astonly_20260328T090000Z`

What changed:
- kept AST reconciliation active
- froze the scoring path more aggressively:
  - added `final_norm.` and `flow_head.` to `freeze_prefixes`
- only AST heads remained trainable:
  - `team_ast_budget_head`
  - `assist_share_head`

Training result:
- `best_val_total = 6.7649`
- best training proxy seen so far for the AST line of work

Live-slate raw-world eval:
- `/home/daniel/projections-data/training/runs/gtv2_flow_v2_astreconcile_rqs_astonly_20260328T090000Z/live_slate_eval_20260327T190000Z.json`

Result vs AST baseline:
- Nembhard AST:
  - `3.78 -> 7.03`
- AST overall mean diff vs props:
  - `-0.414 -> -0.119`
- AST overall MAE:
  - `1.162 -> 1.024`
- AST line `>= 7.0` mean diff:
  - `-0.916 -> +3.094`
- total absolute error vs market:
  - `4.281 -> 15.239`

Interpretation:
- Freezing the flow head does improve the training proxy materially.
- It does not fix replay behavior.
- The AST slice still overshoots badly and totals remain far off.

Current conclusion:
- The remaining failure is not just "AST supervision is distorting the flow head".
- Even when the flow head is frozen, the reconciliation mechanism still overshoots AST and replay totals stay broken.
- That means the next step needs a deeper structural coupling between AST reconciliation and the rest of the generated stat budget, not just a different loss mix or a narrower freeze set.

## AST Reconciliation Correction: Checkpoint-Compatibility Confound

Date: 2026-03-28

A material confound was identified in the earlier AST reconciliation comparisons.

What was wrong:
- Several AST branches warm-started from:
  - `/home/daniel/projections-data/training/runs/gtv2_flow_v2_astfactor_factorized_rqs_20260328T040500Z`
- but changed both:
  - `backbone_env_feature_cols`
  - `assist_share_condition_feature_cols`
- That caused shape mismatches on frozen heads during warm-start, leaving these heads randomly initialized while still participating in world generation:
  - `possession_head`
  - `event_backbone`
  - `three_pa_share_head`
  - parts of `assist_share_head`

This invalidates the earlier interpretation that reconciliation itself was causing the extreme team-total failures seen in those incompatible runs.

### Clean compatible reconciliation branch

Run:
- `/home/daniel/projections-data/training/runs/gtv2_flow_v2_astreconcile_compatible_rqs_20260328T093500Z`

Setup:
- warm-start from the RQS AST baseline checkpoint
- kept the checkpoint contract compatible:
  - `backbone_env_feature_cols = ""`
  - `assist_share_condition_feature_cols = ""`
- enabled only:
  - `assist_share_reconcile_ast_budget = true`
  - `assist_share_reconcile_alpha = 0.75`
  - `assist_share_reconcile_temperature = 0.85`

Warm-start diagnostic:
- no shape-mismatched keys
- no missing frozen backbone/environment heads

Training result:
- `best_val_total = 6.9505`
- better than baseline RQS AST:
  - `7.1243`

Live-slate raw-world eval:
- `/home/daniel/projections-data/training/runs/gtv2_flow_v2_astreconcile_compatible_rqs_20260328T093500Z/live_slate_eval_20260327T190000Z.json`

Result vs baseline RQS AST:
- Nembhard AST:
  - `3.78 -> 4.50`
- AST overall mean diff vs props:
  - `-0.414 -> -0.033`
- AST line `>= 7.0` mean diff:
  - `-0.916 -> +0.355`
- AST line `>= 7.0` MAE:
  - `1.674 -> 1.724`
- total absolute error vs market:
  - `4.684 -> 4.751`

Interpretation:
- The clean compatible reconciliation branch materially improves aggregate AST calibration.
- It improves the named playmaker miss without blowing up team-total calibration.
- It slightly overshoots the high-AST slice, but the magnitude is modest rather than catastrophic.

### Runtime sweep on the compatible branch

Artifact:
- `/home/daniel/projections-data/training/runs/gtv2_flow_v2_astreconcile_compatible_rqs_20260328T093500Z/runtime_sweep_live_slate_eval_20260327T190000Z.json`

Sweep summary:
- best balanced setting remained close to the training default:
  - `alpha = 0.75`
  - `temperature = 0.85`
- raising `alpha` or lowering `temperature` pushed AST further, but moved the high-AST slice into clearer overprediction.
- totals and PTS stayed essentially unchanged across the sweep, confirming the remaining effect is local to AST allocation.

Examples from the sweep:
- `a0.75_t0.85`
  - Nembhard AST: `4.49`
  - AST overall mean diff: `-0.034`
  - AST `>= 7` mean diff: `+0.357`
  - total abs err: `4.915`
- `a0.90_t0.85`
  - Nembhard AST: `4.53`
  - AST overall mean diff: `-0.020`
  - AST `>= 7` mean diff: `+0.521`
  - total abs err: `4.915`
- `a0.85_t0.75`
  - Nembhard AST: `4.71`
  - AST overall mean diff: `+0.038`
  - AST `>= 7` mean diff: `+1.278`
  - total abs err: `4.915`

### Compatible conditioned reconciliation branch

Run:
- `/home/daniel/projections-data/training/runs/gtv2_flow_v2_astreconcile_compatible_conditioned_rqs_20260328T103500Z`

What changed:
- same compatible reconciliation setup
- reintroduced `assist_share_condition_feature_cols`:
  - `an_ast_line,an_implied_minutes,prior_play_prob,started_proxy_rate_prior_20`
- only the assist-share head changed shape; backbone/environment heads remained compatible

Training result:
- `best_val_total = 7.1402`
- worse than the unconditioned compatible branch `6.9505`

Decision:
- Not replayed.
- The conditioned compatible branch is not better than the clean unconditioned compatible reconciliation branch on the training proxy.

Current conclusion:
- The earlier "reconciliation breaks totals" conclusion was overstated due to a checkpoint-compatibility confound.
- The clean compatible reconciliation branch is now the best AST direction.
- The next work should continue from this branch, not from the earlier incompatible AST branches.

### Conditioned compatible branch: replay was better than the training proxy suggested

The conditioned compatible branch was replayed after the checkpoint-compatibility issue was corrected.

Run:
- `/home/daniel/projections-data/training/runs/gtv2_flow_v2_astreconcile_compatible_conditioned_rqs_20260328T103500Z`

Replay result at the trained runtime setting:
- Nembhard AST:
  - `4.52 -> 6.91` vs the unconditioned compatible branch
- AST overall mean diff vs props:
  - `-0.029 -> -0.113`
- AST overall MAE:
  - `1.307 -> 1.101`
- AST line `>= 7.0` mean diff:
  - `+0.345 -> +3.085`
- AST line `>= 7.0` MAE:
  - `1.711 -> 3.302`
- PTS overall mean diff:
  - `-1.063 -> -0.851`
- total absolute error vs market:
  - `3.851 -> 4.171`

This changed the interpretation:
- the conditioned assist-share head is not a dead branch
- the training proxy understated its value on named playmaker misses
- the real issue is calibration of the AST reconciliation strength

### Runtime calibration sweep on the conditioned branch

An explicit runtime sweep was run on the conditioned compatible branch.

Sweep artifact:
- `/tmp/conditioned_runtime_sweep.jsonl`

Main result:
- lower AST reconciliation `alpha` reduces the high-AST overshoot while preserving most of the Nembhard gain
- `temperature` had a smaller effect than `alpha` in the useful region
- totals stayed effectively fixed across this sweep

Representative settings (`temperature = 1.0`):
- `alpha = 0.35`
  - Nembhard AST: `6.15`
  - AST overall mean diff: `-0.276`
  - AST overall MAE: `0.898`
  - AST `>= 7` mean diff: `+1.408`
  - AST `>= 7` MAE: `1.841`
  - total abs err: `4.171`
- `alpha = 0.50`
  - Nembhard AST: `6.22`
  - AST overall mean diff: `-0.261`
  - AST overall MAE: `0.915`
  - AST `>= 7` mean diff: `+1.567`
  - AST `>= 7` MAE: `1.979`
  - total abs err: `4.171`
- `alpha = 0.75`
  - Nembhard AST: `6.41`
  - AST overall mean diff: `-0.220`
  - AST `>= 7` mean diff: `+2.003`
  - total abs err: `4.171`

Updated read:
- the safe retained AST base is still the unconditioned compatible reconciliation branch
- the most promising next direction is the conditioned compatible branch with a lower AST reconciliation `alpha`
- that direction no longer requires a larger architecture jump before another calibration pass

### Trained conditioned follow-up with lower built-in reconciliation

The next direct follow-up trained the lower reconciliation setting into the model instead of applying it only at runtime.

Run:
- `/home/daniel/projections-data/training/runs/gtv2_flow_v2_astreconcile_compatible_conditioned_rqs_a035_t100_20260328T120500Z`

Changes:
- init from the conditioned compatible checkpoint
- `assist_share_reconcile_alpha = 0.35`
- `assist_share_reconcile_temperature = 1.0`

Training:
- `best_val_total = 7.0379`
- improved vs the original conditioned branch (`7.1402`)
- still slightly behind the unconditioned compatible branch (`6.9505`)

Replay:
- Nembhard AST:
  - `6.91 -> 7.57`
- AST overall mean diff:
  - `-0.113 -> -0.126`
- AST overall MAE:
  - `1.101 -> 0.810`
- AST `>= 7` mean diff:
  - `+3.085 -> +2.484`
- AST `>= 7` MAE:
  - `3.302 -> 2.511`
- total abs err:
  - `4.171 -> 3.672`

But the branch still is not clean:
- PTS overall mean diff worsened:
  - `-0.851 -> -1.212`
- REB bias drifted positive:
  - `-0.807 -> +0.451`

Conclusion:
- baking the lower AST reconciliation strength into training helps the conditioned branch
- it is still too aggressive on named playmakers
- the retained AST base remains the clean unconditioned compatible reconciliation branch
- if iteration continues, the next work should focus on cross-stat stabilization of the conditioned branch rather than more AST pressure

### Follow-up experiments: simple gate and direct `PTS/TOV` stabilization

Two immediate experiments were run after the conditioned `a035_t100` branch.

#### Simple threshold gate between unconditioned and conditioned AST

Research-only artifact:
- `/tmp/ast_gate_blend_eval.json`

Setup:
- use the unconditioned compatible branch as the base world state
- replace AST with the conditioned branch only for gated players
- gates tested around `an_ast_line`, `an_implied_minutes`, and `prior_play_prob`

Result:
- not useful
- simple gates mostly selected the same high-AST players and reproduced the conditioned overshoot

Examples:
- base unconditioned:
  - Nembhard AST: `4.52`
  - AST `>= 7` mean diff: `+0.345`
- full conditioned:
  - Nembhard AST: `7.57`
  - AST `>= 7` mean diff: `+2.484`
- `gate_ast7`:
  - Nembhard AST: `7.57`
  - AST `>= 7` mean diff: `+2.484`

This says the gating idea is not dead in principle, but the simple threshold version is not selective enough.

#### Direct `PTS/TOV` stabilization on the conditioned branch

Run:
- `/home/daniel/projections-data/training/runs/gtv2_flow_v2_astreconcile_compatible_conditioned_stabilized_20260328T123500Z`

Added:
- `w_direct_pts_aux = 0.02`
- `w_direct_tov_aux = 0.01`

Training result:
- `best_val_total = 7.2859`
- worse than the conditioned `a035_t100` branch (`7.0379`)
- worse than the original conditioned branch (`7.1402`)

Read:
- direct `PTS/TOV` stabilization in this form hurts the branch
- this is not the right stabilization path

Updated recommendation:
- abandon the simple threshold gate
- abandon the direct `PTS/TOV` stabilization branch
- if work continues, the next stabilization attempt should be narrower and more creator-role specific

### Follow-up experiment: non-AST flow-anchor stabilization

One additional stabilization branch was run after the direct `PTS/TOV` attempt.

Run:
- `/home/daniel/projections-data/training/runs/gtv2_flow_v2_astreconcile_conditioned_flowanchor_20260328T130500Z`

Setup:
- student:
  - conditioned compatible reconciliation with baked-in lower strength
  - `/home/daniel/projections-data/training/runs/gtv2_flow_v2_astreconcile_compatible_conditioned_rqs_a035_t100_20260328T120500Z`
- frozen teacher:
  - clean unconditioned compatible reconciliation
  - `/home/daniel/projections-data/training/runs/gtv2_flow_v2_astreconcile_compatible_rqs_20260328T093500Z`
- added loss:
  - `w_flow_anchor_nonast_aux = 0.01`
  - anchor only non-AST emergent flow channels to the teacher branch

Training result:
- `best_val_total = 7.0613`
- slightly worse than conditioned `a035_t100`:
  - `7.0379`
- still better than the original conditioned branch:
  - `7.1402`

Replay comparison:

Unconditioned compatible:
- Nembhard AST: `4.52`
- AST overall mean diff: `-0.029`
- AST overall MAE: `1.307`
- AST `>= 7` mean diff: `+0.345`
- AST `>= 7` MAE: `1.711`
- PTS overall mean diff: `-1.063`
- REB overall mean diff: `-0.019`
- total abs err: `3.851`

Conditioned `a035_t100`:
- Nembhard AST: `7.57`
- AST overall mean diff: `-0.126`
- AST overall MAE: `0.810`
- AST `>= 7` mean diff: `+2.484`
- AST `>= 7` MAE: `2.511`
- PTS overall mean diff: `-1.212`
- REB overall mean diff: `+0.451`
- total abs err: `3.672`

Flow-anchor:
- Nembhard AST: `7.52`
- AST overall mean diff: `-0.040`
- AST overall MAE: `0.859`
- AST `>= 7` mean diff: `+2.795`
- AST `>= 7` MAE: `2.837`
- PTS overall mean diff: `-1.122`
- REB overall mean diff: `+0.176`
- total abs err: `3.705`

Read:
- the anchor reduced some cross-stat drift versus the conditioned branch
  - PTS bias improved
  - REB bias improved materially
- it did not improve the high-AST overshoot
- it did not meaningfully improve totals versus the conditioned branch

Updated recommendation:
- do not retain the flow-anchor branch as the new AST base
- keep:
  - unconditioned compatible reconciliation as the safe AST base
  - conditioned `a035_t100` as the stronger playmaker mechanism
- if iteration continues, the next stabilization step should be more creator-specific than a generic non-AST anchor

### Follow-up experiment: creator-alpha gating inside AST reconciliation

After the flow-anchor result, two creator-gating variants were tested directly inside the AST reconciliation path:

1. absolute creator alpha
2. team-relative creator alpha

Implementation path:
- [sample_worlds_v2.py](/home/daniel/projects/projections-v2/projections/rotation/sample_worlds_v2.py)

Result:
- both were effectively no-ops on replay
- the conditioned branch metrics were unchanged to numerical noise

Read:
- that means the remaining AST problem is probably not about who gets the passes within a team
- the conditioned branch already appears concentrated enough on the lead creator
- the remaining miss is more likely:
  - AST budget magnitude
  - or creator-channel coupling after AST moves

Updated recommendation:
- stop iterating on creator/share gating for now
- move the next AST experiment to:
  1. team AST budget calibration/cap
  2. AST/TOV-style creator-channel coupling

### 60-day validation backtest changed the AST read

The single-slate AST loop was useful for mechanism discovery, but it overstated the broad problem.

A full 60-day validation backtest was run using the latest retained AST branches:

Artifact root:
- `/home/daniel/projections-data/training/runs/ast_60d_eval_20260328T144241Z`

Branches compared:
- unconditioned compatible AST reconciliation:
  - `/home/daniel/projections-data/training/runs/gtv2_flow_v2_astreconcile_compatible_rqs_20260328T093500Z`
- conditioned AST reconciliation:
  - `/home/daniel/projections-data/training/runs/gtv2_flow_v2_astreconcile_compatible_conditioned_rqs_a035_t100_20260328T120500Z`

Backtest scope:
- 60 validation dates:
  - `2025-12-09` through `2026-02-11`
- `4,989` AST prop-bearing player-games
- `206` rows with `an_ast_line >= 7`
- `25` distinct high-AST players

Market baseline on `AST >= 7`:
- line mean: `8.16`
- actual mean: `7.53`
- line minus actual: `+0.63`
- line MAE vs actual: `2.96`

So the market itself was already high on this broad slice.

Unconditioned compatible branch:
- overall:
  - pred mean: `3.35`
  - actual mean: `2.86`
  - pred minus actual: `+0.49`
  - pred MAE vs actual: `1.89`
  - market MAE vs actual: `1.56`
- `AST >= 7`:
  - pred mean: `8.88`
  - line mean: `8.16`
  - actual mean: `7.53`
  - pred minus line: `+0.72`
  - pred minus actual: `+1.35`
  - pred MAE vs actual: `3.46`
  - market MAE vs actual: `2.96`
  - over-line rate: `55.8%`
  - over-actual rate: `61.7%`

Conditioned branch:
- overall:
  - pred mean: `3.28`
  - actual mean: `2.86`
  - pred minus actual: `+0.42`
  - pred MAE vs actual: `1.58`
  - market MAE vs actual: `1.56`
- `AST >= 7`:
  - pred mean: `10.59`
  - line mean: `8.16`
  - actual mean: `7.53`
  - pred minus line: `+2.43`
  - pred minus actual: `+3.06`
  - pred MAE vs actual: `3.80`
  - over-line rate: `89.8%`
  - over-actual rate: `81.6%`

This changed the conclusion materially:
- there is **not** a broad high-AST underprediction problem
- the Nembhard-type miss is real, but it is not representative of the broad 60-day high-AST cohort
- the conditioned branch is not a general solution; it is an overcorrection on the broad slice

The residual underpredicted names on the unconditioned branch are concentrated in a smaller set:
- Ja Morant
- Davion Mitchell
- Isaiah Collier
- Jamal Shead
- Andrew Nembhard
- James Harden

That points much more toward:
- selective archetype/context misses
- or feature staleness / short-window context misses

than toward a broad AST architecture failure.

### Updated AST recommendation

- stop broad AST escalation
- keep the unconditioned compatible reconciliation branch as the retained AST base
- if AST work resumes later, it should start from:
  - player/archetype analysis
  - short-window feature/context work
  - teammate-absence / role-change effects

not from more global AST-head iterations

### Methodology to carry into REB work

The AST work established the process we should reuse for rebounds:

1. run the broad 60-day validation slice first
2. check whether the high-REB slice is actually systematically wrong in aggregate
3. only then decide whether a REB-specific factorization is warranted

This is important because the AST line consumed a large amount of iteration before the broad backtest showed the main problem was narrower than the single-slate read suggested.

Working expectation for REB:
- REB may be more tractable than AST because it has a cleaner structure:
  - team rebound-opportunity budget
  - player rebound share
- but we should not assume a factorization branch is needed until the 60-day slice confirms a broad miss

So the next modeling step should be:
- move on from AST for now
- start REB with the same broad-slice validation harness first

## Rebound Status Update (2026-03-28)

The 60-day REB baseline confirmed that rebounds are a broad structural miss, not just a
named-player issue.

Baseline read on the retained broad slice:

- high-REB slice (`an_reb_line >= 10`) was materially underpredicted
- team rebounds were too high overall
- predicted team DREB had almost no coupling to opponent missed shots

That justified explicit REB factorization work. The first useful lesson from the
follow-up branches was asymmetric:

- explicit `DREB` opportunity coupling helps
- shared `OREB/DREB` treatment is too blunt and can break OREB

What happened in the first REB branches:

- `dreb_only` reconciliation with stronger share blend improved player concentration and
  stayed the best broad player-facing result so far
- direct `dreb_rate` parameterization improved `pred team dreb vs opp missed FG` sharply,
  but overcorrected team totals unless heavily blended back toward flow
- learned rebound budget blend gates improved DREB structure somewhat, but did not beat the
  simpler fixed-blend branch on the high-REB slice

Current modeling conclusion:

- the residual DREB path improved high-REB concentration, but it still fought a
  systematically wrong base team DREB budget
- the core miss is structural: flow DREB still has too little dependence on opponent missed
  shots, so a residual head has to undo that error before it can add useful signal
- the next iteration should make **team DREB deterministic from the sampled environment**:
  - derive `team_dreb_budget ~= opp_missed_fg - opp_oreb` inside each sampled world
  - leave `OREB` on the existing flow path for now
  - keep the learned piece focused on `DREB` player share allocation

Follow-up result from that deterministic branch:

- the structural target was achieved:
  - predicted team `DREB` vs opponent missed FG correlation jumped to `0.717`
- but the branch over-assigned player rebounds at the team level:
  - team REB mean `52.53` vs actual `44.07` (`+8.46` bias)
  - predicted mean `DREB` capture rate `0.754` vs actual `0.688`
  - high-REB slice still undercalled (`8.46` vs `10.41`)

Updated interpretation:

- making `DREB = opp_missed_fg - opp_oreb` by construction is too hard because it assumes
  essentially every missed FG becomes a player rebound
- the missing piece is a **dead-ball / non-player rebound slack** or equivalent capture prior
  on top of the deterministic environment budget
- next step should be a deterministic DREB budget multiplied by a bounded player-rebound
  capture factor, rather than full one-for-one conversion of misses into player DREB

Follow-up result with a fixed empirical discount:

- estimated train-set player `DREB` capture rate against
  `opp_missed_fg - opp_oreb` was `0.9054` weighted (`0.9078` unweighted mean)
- applying that fixed scalar before DREB share allocation corrected the level problem:
  - predicted mean `DREB` capture rate moved to `0.681` vs actual `0.688`
  - team REB bias improved to `+4.76`
  - eval-only `share_alpha=1.0` improved team REB bias further to `+4.31` with team REB
    MAE `6.68`
- the structural DREB win held:
  - predicted team `DREB` vs opponent missed FG correlation stayed high at `0.693`

What remains:

- high-REB concentration is still too weak even after the budget fix
- the problem is no longer the team DREB budget
- the next lever should shift from budget design to **player DREB share concentration**

First DREB share-concentration follow-up:

- added explicit rebound-share conditioning on:
  - `an_reb_line`
  - `an_implied_minutes`
  - `prior_play_prob`
  - `started_proxy_rate_prior_20`
- kept the discounted deterministic DREB budget fixed and increased share reconciliation /
  share supervision

Result:

- high-REB slice improved materially:
  - predicted mean `7.87 -> 8.95`
  - bias `-2.54 -> -1.46`
  - MAE `3.98 -> 3.39`
  - over-line `1.2% -> 4.7%`
- broad player metrics improved too:
  - overall REB MAE `2.19 -> 2.06`
  - rotation-20 player REB corr improved to `0.501`
- team-level REB stayed effectively flat:
  - team REB bias `+4.76 -> +4.77`
  - team REB MAE `6.93 -> 6.96`
- structural DREB coupling held:
  - `pred team dreb vs opp missed FG corr = 0.704`

Mechanism check:

- the remaining high-REB miss after the conditioned `DREB` branch was mostly `OREB`, not
  `DREB`
  - on the `line >= 10` slice:
    - predicted `DREB = 6.94` vs actual `7.46`
    - predicted `OREB = 1.93` vs actual `2.95`
  - mean high-line share gaps were:
    - `DREB share gap = -0.015`
    - `OREB share gap = -0.089`
- that pointed to an inference/training mismatch:
  - the branch was still running `rebound_factor_reconcile_mode=dreb_only`
  - so the new rebound-share head was improving `DREB`, but `OREB` stayed on the old
    flow allocation path

Next iteration: OREB share-only reconcile on top of the existing team OREB total

- added a narrow `OREB` reconcile option that keeps the current team `OREB` total and only
  redistributes it via the rebound-share head
- eval-only enablement was worse, which confirmed that the branch needed retraining with the
  `OREB` reconcile path active
- after retraining with:
  - discounted deterministic `DREB`
  - `rebound_factor_reconcile_mode=both`
  - `rebound_oreb_reconcile_use_flow_budget=true`

Result:

- this is the best REB branch so far
- high-REB slice improved materially again:
  - predicted mean `8.95 -> 10.15`
  - bias `-1.46 -> -0.27`
  - MAE `3.39 -> 3.10`
  - over-line `4.7% -> 33.7%`
- broad REB metrics improved too:
  - overall REB MAE `2.06 -> 2.01`
  - overall REB corr `0.647 -> 0.665`
- team-level REB improved despite turning on `OREB` reconcile:
  - team REB bias `+4.77 -> +4.00`
  - team REB MAE `6.96 -> 6.53`
- structural `DREB` coupling stayed strong:
  - `pred team dreb vs opp missed FG corr = 0.697`

Mechanism confirmation:

- high-line `OREB` allocation was the missing piece
  - predicted mean `OREB` moved from `1.93 -> 3.14` vs actual `2.95`
  - mean high-line `OREB` share gap moved from `-0.089 -> +0.008`
- `DREB` allocation stayed effectively intact
  - predicted mean `DREB = 6.91` vs actual `7.46`
  - mean high-line `DREB` share gap stayed near flat at `-0.014`

Updated conclusion:

- discounted deterministic `DREB` budget plus conditioned rebound-share allocation on both
  `OREB` and `DREB` is now the leading REB branch
- the next REB question is no longer whether to factorize rebounds; that is settled
- the next question is whether `OREB` budget itself deserves a stronger environment-coupled
  prior, or whether the current team-total anchor is already sufficient

- high-line players' mean predicted DREB share moved from `0.183` to `0.216`
  against actual `0.221`
- the average DREB-share gap on the high-line slice shrank from `-0.039` to `-0.005`

Current conclusion:

- the isolated DREB share problem is real and the conditioned share head is the first branch
  that attacks it directly
- this is the new best REB branch so far

## Flow Minutes Teacher-Forcing Fix on Rebound Branch (2026-03-28)

### Problem identified

After promoting the rebound branch (`reb_sharecond_disc905_orebflow`) to production,
live predictions were materially low on player-level stats (pts, reb, ast, stl).

Root cause: the rebound branch was trained with
`flow_minutes_teacher_forcing_prob=1.0` throughout — the flow head always saw
ground-truth minutes during training, but at inference it receives predicted
minutes. This is the same train/inference mismatch that `minutesdist_mtfanneal`
fixed earlier, but the rebound branch forked before that fix was applied.

### Retrain

Warm-started from the promoted rebound checkpoint with the only material change
being flow minutes teacher-forcing anneal:

- `flow_minutes_teacher_forcing_prob_start=1.0`
- `flow_minutes_teacher_forcing_prob_end=0.0`
- `flow_minutes_teacher_forcing_ramp_epochs=12`
- 20 epochs total, gentler phase2 NLL guard (`35.0`, 6 backoffs)

All rebound/assist head settings were identical to the parent branch.

Training run:

- `/home/daniel/projects/projections-v2/data/training/runs/game_transformer_v2_20260328T200632Z`
- best epoch: `9` (flow_mtf=0.273)
- best `val_total = 11.33` vs parent `11.61`
- zero instability events across all 20 epochs

### 60-game / 128-world production-aligned replay

Artifacts:

- `/home/daniel/projections-data/training/runs/gtv2_mtfanneal_reb_60day_eval/summary.csv`
- `/home/daniel/projections-data/training/runs/gtv2_mtfanneal_reb_60day_eval/compare_vs_baseline.csv`

Result versus `prod_current` (the rebound branch without mtf anneal):

Improved:

- `dk_fpts_mae`: `5.910 -> 5.601`
- `pts_mae_player`: `3.544 -> 3.389`
- `ast_mae_player`: `1.375 -> 1.277`
- `blk_mae_player`: `0.411 -> 0.375`
- `stl_mae_player`: `0.534 -> 0.532`
- `active_acc_at4`: `0.907 -> 0.914`
- `minutes_mae`: `3.662 -> 3.607`
- `poss_mae`: `4.026 -> 3.447`
- `spread_mae_vs_vegas`: `5.533 -> 5.461`
- `top1_share_bias_pts`: `-0.021 -> +0.004` (near zero)
- `top2_share_bias_pts`: `-0.034 -> +0.003` (near zero)

Worse:

- `reb_mae_player`: `1.579 -> 1.667`
- `total_mae_vs_vegas`: `2.409 -> 4.529`
- `pts_mae_team`: `9.830 -> 9.863`

Interpretation:

- the player-level improvement is broad and substantial
- the star-share underprediction bias (the original symptom) is resolved
- the `total_mae_vs_vegas` regression is expected: the actuals-conditioned flow
  implicitly used oracle minutes information to produce tighter team totals;
  removing that oracle information degrades team totals but fixes the
  player-level calibration that matters for production use
- `reb_mae` regression is small (+0.09) and likely from noisier minutes
  conditioning slightly degrading the rebound allocation path

Decision:

- promote `mtfanneal_reb` to production
- the player-level and FPTS improvements outweigh the team-total regression
- future work on team-total calibration should be a separate effort, not a
  reason to keep the train/inference mismatch

## Spread/Total Auxiliary Loss Fine-Tune Experiment (2026-03-28)

### Motivation

After deploying `mtfanneal_reb`, team implied totals were not differentiated by
spread. For example, PHX favored by 16.5 still produced ~114 implied total for
both teams. The hypothesis was that adding auxiliary losses on predicted game
spread and total (vs Vegas lines) could push the model to differentiate team
scoring without regressing player-level accuracy.

### Setup

Warm-started from `mtfanneal_reb` (epoch 9 checkpoint) with two new aux losses:

- `w_spread_aux=0.03`: penalizes deviation of predicted home-away point
  differential from Vegas spread
- `w_total_aux=0.05`: penalizes deviation of predicted combined score from
  Vegas total
- 3-epoch loss ramp (`aux_ramp_epochs=3`) to avoid early instability
- 5 total epochs, lr `3e-5`

Training run:

- `/home/daniel/projects/projections-v2/data/training/runs/game_transformer_v2_20260328T204601Z`
- best epoch: `2`, best `val_total = 11.32`
- `val_spread_aux` decreased `0.39 -> 0.25` over training (learning spread signal)
- `val_total_aux` volatile (`0.17 -> 1.16 -> 0.12`)
- zero instability events

### 60-day production-aligned eval (3-way comparison)

Artifacts:

- `/home/daniel/projections-data/tmp/gtv2_eval_spreadtotal/summary.csv`
- `/home/daniel/projections-data/tmp/gtv2_eval_spreadtotal/compare_vs_baseline.csv`

Results (delta vs `prod_current` baseline):

| Metric | prod_current | mtfanneal_reb | spreadtotal |
|--------|-------------|---------------|-------------|
| dk_fpts_mae | 5.94 | **5.62** (-0.32) | 5.98 (+0.04) |
| pts_mae | 3.55 | **3.40** (-0.15) | 3.53 (-0.02) |
| reb_mae | **1.58** | 1.67 (+0.08) | 1.91 (+0.32) |
| ast_mae | 1.38 | **1.28** (-0.10) | 1.29 (-0.09) |
| spread_mae | **5.15** | 5.33 (+0.18) | 5.17 (+0.02) |
| spread_corr | **0.454** | 0.386 (-0.07) | 0.425 (-0.03) |
| total_mae | **2.25** | 4.19 (+1.95) | 4.49 (+2.24) |
| total_corr | **0.930** | 0.704 (-0.23) | 0.678 (-0.25) |
| poss_mae | 4.04 | 3.41 (-0.63) | **2.98** (-1.06) |

### Follow-up probe: stronger backbone conditioning without a new budget head

Two follow-up probes were run after the live issue was confirmed again on
`reb_mtfanneal_live_20260328`:

- `backbone_env_enrich_features=true` on the live branch
- explicit side-specific market context added to the backbone team state
  (`backbone_side_market_context`)

The first probe did not help. On the 2026-03-28 live slate replay, implied
team margins stayed near ties and actually compressed slightly further.

The second probe was directionally better but still not good enough to retain.
Explicit side-market context roughly doubled mean absolute implied margin on the
same live slate replay (`~0.98 -> ~1.98` on the common 5 games), but it still
fell far short of market spreads and pulled total scoring down too much
(`~230.9 -> ~223.5`). In other words: explicit home/away market conditioning is
necessary, but simply injecting it into the backbone team state is not
sufficient.

Conclusion: the next branch should stop trying to get team asymmetry to emerge
indirectly from shared game-volume latents. It should add an explicit team-level
budget split target or latent before player allocation.

### Follow-up probe: explicit team-points budget latent

A first explicit team budget split probe has now been run:

- new supervised head: `team_points_budget_head`
- target: side-specific implied totals from `vegas_total` and `vegas_spread`
- latent injection: predicted home/away team-point budgets encoded back into the
  backbone team states (`team_points_budget_to_backbone=true`)
- no separate side-market context branch in this probe

Training result:

- run: `/home/daniel/projections-data/training/runs/gtv2_team_points_budget_liveprobe_20260328T214029Z`
- best epoch: `7`
- best `val_total = 11.68`
- the new head fit the implied-total target very easily:
  - `val_team_points_budget_aux: 3.19 -> 0.02`

Live-slate local replay result:

- artifact: `/home/daniel/projections-data/tmp/gtv2_team_points_budget_liveprobe_20260328T214029Z/team_margin_summary.json`
- mean absolute implied margin improved only slightly:
  - current live: `0.93`
  - explicit team-points latent probe: `1.24`
  - market: `10.08`
- mean total points stayed in the right neighborhood:
  - current live: `230.72`
  - probe: `228.75`

Interpretation:

- the model does **not** lack access to a team-split signal
- the new head can learn the implied team totals directly
- but adding that split only as a latent perturbation to team state is still too
  indirect to control downstream event budgets

Updated conclusion:

- explicit team budget supervision alone is not enough if it only enters as a
  soft latent
- the next branch should make the team split operative in the generative path:
  - either explicit team-level points budgets before player allocation
  - or explicit team opportunity budgets conditioned on a sampled home/away
    points split
- in short: the team split likely needs to be a **harder budget object**, not
  just another conditioning feature

### Follow-up probe: operative market-implied team budget

The next probe made the side split a real budget instead of a learned latent:

- new config path: `team_points_budget_parameterization=market_implied`
- budget source: direct home/away implied totals from `vegas_total` and
  `vegas_spread`
- operative constraint: reconcile player scoring makes toward that side-specific
  budget (`team_points_reconcile_budget=true`)
- optional conditioning: encode the same market-implied team budget back into
  backbone team state (`team_points_budget_to_backbone=true`)

Eval-only replay on the current live bundle established the key architectural
point immediately:

- artifact:
  `/home/daniel/projections-data/tmp/live_bundle_team_points_market_reconcile_20260328/summary.json`
- current live mean absolute implied margin: `0.93`
- market-implied reconcile:
  - `alpha=0.50`: `5.31`
  - `alpha=0.75`: `7.69`
  - `alpha=1.00`: `10.07`
- sign correctness on the 6-game live slate improved to `6/6` for all three
  alphas

Interpretation:

- the downstream generator **can** carry a real home/away split once the split
  is made operative
- the tie-like live behavior is not a publish-layer bug and not an unavoidable
  property of player-first aggregation
- the failure is upstream: the model was never producing an operative team split
  budget on its own

Training-consistent probe:

- run:
  `/home/daniel/projections-data/training/runs/gtv2_market_team_points_reconcile_a075_train_20260328T215918Z`
- live replay artifact:
  `/home/daniel/projections-data/tmp/gtv2_market_team_points_reconcile_a075_train_20260328T215918Z/team_margin_summary.json`
- best epoch: `6`
- best `val_total = 11.46`
- live-slate replay:
  - mean absolute implied margin: `7.67`
  - mean total points: `233.46`
  - spread sign correctness: `6/6`
  - team-total MAE vs current live publish: `3.91`

Updated team-split conclusion:

- a deterministic market-implied team budget is the first branch that actually
  fixes the tie-collapse behavior
- the viable operating region is the partial-reconcile setting, not the
  full-parrot setting:
- `alpha=1.0` reproduces market spreads almost exactly
- `alpha≈0.75` preserves most of the spread recovery while leaving room for
  model-side deviation
- this is a materially stronger direction than soft spread/total auxiliaries or
  learned split latents

### 60-day gate on the operative market-budget branch

We ran the same 60-game production-aligned validation envelope used for the
current live branch:

- `alpha=0.75` trained branch:
  - `/home/daniel/projections-data/training/runs/gtv2_market_team_points_60day_eval_20260328T2200Z/summary.csv`
- `alpha=0.50` eval-only fallback:
  - `/home/daniel/projections-data/training/runs/gtv2_market_team_points_60day_eval_a050_20260328T2204Z/summary.csv`

Baseline (`current_live`) on that packet:

- `dk_fpts_mae = 5.60`
- `pts_mae_player = 3.39`
- `pts_mae_team = 9.86`
- `spread_mae_vs_vegas = 5.46`
- `total_mae_vs_vegas = 4.53`

`alpha=0.75` result:

- `spread_mae_vs_vegas = 1.62`
- `total_mae_vs_vegas = 2.59`
- but `pts_mae_team = 16.01`
- and `pts_bias_mean_team = -8.08`
- plus `pts_mae_player = 3.68`, `dk_fpts_mae = 5.91`

`alpha=0.50` result:

- `spread_mae_vs_vegas = 2.82`
- `total_mae_vs_vegas = 3.70`
- but `pts_mae_team = 13.23`
- and `pts_bias_mean_team = -5.52`
- plus `pts_mae_player = 3.55`, `dk_fpts_mae = 5.80`

Decision:

- do **not** promote the market-implied team-budget branch yet
- it solves the spread-collapse problem, but does so by over-anchoring team
  points and degrading broad bundle quality on the 60-day gate
- the next iteration should keep the explicit team split, but make it a softer
  residual or opportunity-budget constraint rather than a direct points-budget
  anchor

### Interpretation

- **Spread differentiation did not materially improve.** `spread_corr` went
  from 0.386 (mtfanneal_reb) to 0.425 (spreadtotal), still below prod_current's
  0.454. The aux loss nudged spread predictions slightly but not enough to
  solve the undifferentiated team totals problem.
- **Rebound accuracy regressed significantly** (+0.32 MAE vs prod, +0.24 vs
  mtfanneal_reb). The fine-tune disrupted the learned rebound allocation
  without the rebound-specific conditioning being retrained.
- **Possession accuracy improved markedly** (poss_mae 2.98, best of all three).
  The total aux loss did help anchor game pace/possessions.
- **Player-level FPTS accuracy regressed** back to roughly prod_current levels,
  losing the mtfanneal_reb gains.
- **Both retrained models show large total_mae regression** vs prod_current
  (~+2 pts). This confirms the earlier hypothesis: prod_current's tight team
  totals were an artifact of teacher-forcing leak (oracle minutes → better
  implied totals), not genuine team-total modeling strength.

### Decision

- **Do not promote spreadtotal fine-tune.** It regresses player accuracy and
  rebounds without delivering meaningful spread differentiation.
- **Stay on `mtfanneal_reb` in production.**

### Lessons and next directions for team differentiation

1. **Aux losses on emergent outputs are weak levers.** Spread and total are
   downstream of many interacting heads (active set, minutes, efficiency, flow).
   A small aux loss cannot easily steer this chain without disrupting calibration
   elsewhere.
2. **The teacher-forcing leak masks the real team-total problem.** Any future
   work on team totals must be evaluated against mtfanneal models, not the
   TF-leaked baseline whose total accuracy was artificially inflated.
3. **More promising directions:**
   - Condition the game flow / possession head more directly on spread and
     total features (input-side, not loss-side)
   - Separate team-level pace/efficiency priors that create asymmetry before
     the player allocation step
   - Team-level fine-grained efficiency conditioning (e.g., eFG% differential
     as an input feature rather than an emergent prediction target)

### 60-day gate on market-implied opportunity split

We ran the next softer branch as an eval-only config perturbation on the current
live lineage:

- output:
  `/home/daniel/projections-data/training/runs/gtv2_market_team_opp_60day_eval_20260328T2232Z/summary.csv`
- delta vs baseline:
  `/home/daniel/projections-data/training/runs/gtv2_market_team_opp_60day_eval_20260328T2232Z/compare_vs_baseline.csv`

Setup:

- keep the current live checkpoint
- turn on
  `team_opportunity_budget_parameterization=market_implied_share`
- reconcile side-specific `FGA` and `FTA` opportunity split only
- do **not** reconcile direct team points
- sweep `alpha ∈ {0.25, 0.40, 0.50, 0.60}`

Best broad result was `alpha=0.50`:

- `spread_mae_vs_vegas = 2.54` vs `5.46` baseline
- `spread_corr_vs_vegas = 0.925` vs `0.338`
- `total_mae_vs_vegas = 4.51` vs `4.53`
- `pts_mae_team = 9.75` vs `9.86`
- `pts_mae_player = 3.388` vs `3.389`
- `dk_fpts_mae = 5.600` vs `5.601`

So unlike direct team-points anchoring, the opportunity split branch does **not**
blow up broad player or team points accuracy.

But it introduces a new structural failure:

- possession symmetry breaks badly after reconcile
- `poss_sym_abs_p95` jumps from `0.323` baseline to:
  - `5.27` at `alpha=0.25`
  - `8.43` at `alpha=0.40`
  - `10.54` at `alpha=0.50`
  - `12.67` at `alpha=0.60`
- raw world contract diagnostics show large home/away possession deltas after
  reconcile even though basketball possessions should stay near symmetric by
  construction

Decision:

- do **not** promote the market-implied opportunity-split branch yet
- this is a better direction than direct team-points anchoring because it
  preserves broad accuracy while recovering spread
- but the current implementation is not generative-safe because it perturbs side
  `FGA/FTA` independently after the possession process is already sampled

Next step:

- move the team split earlier, into an explicit side-specific possession /
  opportunity budget latent that preserves home-away possession symmetry
- do **not** continue with post-hoc side `FGA/FTA` reconcile on top of already
  sampled possessions

### Early-chain opportunity-context probe

We implemented the first earlier-chain attempt by injecting market-implied
home/away opportunity-share context into `backbone_team_states` before
`TeamEventBackbone`, rather than rescaling side `FGA/FTA` after the fact.

Code path:

- `team_opportunity_budget_to_backbone`
- `team_opportunity_budget_backbone_alpha`
- market-implied home/away share encoded into the backbone team state before
  event generation

Train-consistent probes:

- encoder-only warm-start:
  `/home/daniel/projections-data/training/runs/gtv2_team_opp_backbone_enconly_20260328T2248Z`
- encoder + `event_backbone` + `three_pa_share_head` warm-start:
  `/home/daniel/projections-data/training/runs/gtv2_team_opp_backbone_eventprobe_20260328T2250Z`

Result:

- both probes regressed validation relative to the current live lineage
- current live branch: `best_val_total = 11.33`
- encoder-only probe: `best_val_total = 12.79`
- event-backbone probe: `best_val_total = 12.80`
- neither short probe showed evidence that the additive context path was
  unlocking meaningful side differentiation early in the chain

Interpretation:

- simply adding a learned side-opportunity context encoder on top of the
  existing backbone is too weak
- the model still treats team split as a perturbation of the shared process,
  not as a first-class generative budget
- the next earlier-chain branch should therefore be a **true side-specific
  opportunity / possession latent**, not another additive context encoder

### Side-specific possession split scaffold

We then started the next branch directly:

- possession head now supports an optional side-specific home/away possession
  split output
- backbone can consume `(home_poss, away_poss)` directly instead of only a
  shared scalar possession total
- training now supports a direct per-team possession supervision target from
  box score truth

Focused smoke runs:

- frozen-possession smoke:
  `/home/daniel/projections-data/training/runs/gtv2_team_possession_split_smoke_20260328T2310Z`
  - `best_val_total = 17.66`
- trainable possession/backbone smoke:
  `/home/daniel/projections-data/training/runs/gtv2_team_possession_split_smoke2_20260328T2313Z`
  - `best_val_total = 12.70`

Interpretation:

- the branch executes end-to-end and the right modules are now in place
- freezing the possession head is not viable because the new side-split output
  block is randomly initialized
- the first trainable smoke is materially better than the frozen version, but
  still worse than the current live lineage (`11.33`)
- so this is a valid next branch, but it needs a real warm-started probe rather
  than 1-epoch smoke judgments

Short warm-start probe:

- run:
  `/home/daniel/projections-data/training/runs/gtv2_team_possession_split_probe_20260328T2317Z`
- setup:
  - side-specific possession split enabled
  - direct per-team possession supervision enabled
  - `possession_head`, `event_backbone`, and `three_pa_share_head` trainable
  - rest of the live lineage frozen

Result:

- best epoch: `1`
- best `val_total = 12.70`
- then validation degraded across the remaining epochs:
  - epoch 2: `12.92`
  - epoch 3: `13.16`
  - epoch 4: `13.57`
  - epoch 5: `13.63`
  - epoch 6: `13.61`

Decision:

- do **not** run the 60-game alignment gate on this branch yet
- the side-specific possession split scaffold is valid, but the current
  formulation is not learning stably enough to justify broader evaluation

Stabilization sweeps:

- possession-head-only, tighter cap + stronger aux:
  `/home/daniel/projections-data/training/runs/gtv2_team_possession_split_possonly_20260328T2322Z`
  - `team_possession_max_delta = 4.0`
  - `w_team_possession_aux = 1.0`
  - `lr = 1e-4`
  - best `val_total = 14.05`
- possession + event backbone, same tighter cap / lower LR:
  `/home/daniel/projections-data/training/runs/gtv2_team_possession_split_eventstable_20260328T2324Z`
  - best `val_total = 14.08`

Interpretation:

- lowering LR and constraining the side split does not recover the branch
- training only the possession head is too weak
- letting the event backbone move under the same stabilized settings is only
  marginally different and still far from the current live lineage

Updated conclusion:

- the current side-specific possession formulation is not just noisy; it is
  directionally wrong enough that simple stabilization sweeps do not fix it
- the next iteration should change the parameterization, not just the optimizer
  settings

### Efficiency-split probe

We tested a lighter alternative to the possession split:

- keep possessions shared
- add market-implied home/away context directly into the efficiency head path
- supervise side-specific team PPP from box score truth

Implementation:

- `efficiency_market_context`
- `efficiency_market_hidden`
- `efficiency_market_alpha`
- `w_team_efficiency_ppp_aux`

First probe:

- run:
  `/home/daniel/projections-data/training/runs/gtv2_efficiency_market_probe_20260328T234741Z`
- trainable modules:
  - `efficiency_head`
  - `efficiency_team_market_encoder`
- result:
  - `best_val_total = 11.3328`, effectively flat to the current live lineage
  - `val_team_efficiency_ppp_aux = 0.0` throughout the run

We then fixed a trainer bug where the new PPP aux was computed inside the
possession block and later overwritten back to zero, and reran the same probe:

- run:
  `/home/daniel/projections-data/training/runs/gtv2_efficiency_market_probe_fix_20260328T235043Z`
- result:
  - `best_val_total = 11.7010`
  - `val_team_efficiency_ppp_aux = 0.0` throughout the run, even after the
    reset bug was removed

Interpretation:

- this is not a useful team-split branch in its current form
- the PPP aux is effectively redundant with the existing per-player efficiency
  objective because it uses true attempts as the aggregation base
- so the branch does not create operative side-specific team differentiation;
  it only perturbs an already-trained efficiency head

Updated conclusion:

- if we revisit an efficiency-split branch, it must target **emergent team
  PPP / points** before player allocation, not a downstream player make-rate
  head evaluated on true attempts
- the current efficiency-head-only path should be considered closed

### Team PPP latent probe

We then tested the more direct version of that idea: a dedicated learned
`team_ppp` head supervised on observed team PPP, with the resulting latent
injected into the backbone and efficiency paths before event generation.

Implementation:

- `enable_team_ppp_head`
- `team_ppp_to_backbone`
- `team_ppp_to_efficiency`
- `w_team_ppp_aux`

First probe, with only the new PPP head and its encoders trainable:

- run:
  `/home/daniel/projections-data/training/runs/gtv2_team_ppp_probe_fix_20260329T002013Z`
- trainable modules:
  - `team_ppp_head`
  - `backbone_team_ppp_encoder`
  - `efficiency_team_ppp_encoder`
- result:
  - `best_val_total = 11.7208`
  - `val_team_ppp_aux` was now live and non-zero (`~0.39` to `0.44`)

Second probe, allowing the downstream path to adapt:

- run:
  `/home/daniel/projections-data/training/runs/gtv2_team_ppp_eventprobe_20260329T002128Z`
- additionally trainable modules:
  - `event_backbone`
  - `efficiency_head`
  - `three_pa_share_head`
- result:
  - `best_val_total = 11.7350`
  - `val_team_ppp_aux` remained non-zero (`~0.39` to `0.44`)

Interpretation:

- this is a cleaner result than the earlier efficiency-market probe because the
  PPP head is actually learning
- but even with a learned PPP latent and partial downstream adaptation, broad
  validation is still worse than the live lineage (`11.33`)
- so the problem is not just “missing team PPP supervision”; it is that the
  current latent injection path is still too indirect to improve the generator

Updated conclusion:

- a learned team PPP latent by itself is not enough
- the next branch should turn team split into a more operative budget/rate
  mechanism inside generation, rather than another additive latent on top of the
  current backbone

We then tested a stricter version where the learned PPP split becomes a direct
input to the team event backbone rather than an additive latent only:

- run:
  `/home/daniel/projections-data/training/runs/gtv2_team_ppp_directctx_eventprobe_fix_20260329T003002Z`
- change:
  - pass the derived `(own_ppp, opp_ppp, gap, abs_gap)` context directly into
    `TeamEventBackbone` and `ThreePAShareHead`
- result:
  - `best_val_total = 11.5594`
  - `best_epoch = 2`
  - `val_team_ppp_aux` remained non-zero (`~0.39` to `0.43`)
- interpretation:
  - this was the best team-PPP branch so far
  - it materially improved the operative path versus additive PPP latent
    injection, but still did not beat the live lineage (`11.3327`)

We then pushed the same PPP context directly into the efficiency head as well:

- run:
  `/home/daniel/projects/projections-v2/data/training/runs/game_transformer_v2_20260329T003745Z`
- change:
  - add direct per-team PPP context as an explicit efficiency-head input, while
    keeping the direct backbone context on
- result:
  - `best_val_total = 13.4565`
  - `best_epoch = 1`
  - the branch deteriorated monotonically after the first epoch
- interpretation:
  - direct PPP conditioning in the efficiency head is too destabilizing in this
    form
  - the useful signal is in the event-generation path, not in forcing the same
    split through player make-rate estimation

Refined conclusion:

- direct team-PPP context into the backbone is directionally right, but still
  not enough to beat baseline
- direct team-PPP context into the efficiency head is a clear regression and
  should not be pursued in the current form
- the next branch should keep team split operative in the generator, but move
  toward a harder team budget/rate mechanism rather than broader PPP latent
  injection

We also tested the most direct version of that idea available without adding a
new head: use learned `team_ppp` with backbone possessions to form a soft team
points budget, then apply the existing points reconcile path.

- run:
  `/home/daniel/projects/projections-v2/data/training/runs/game_transformer_v2_20260329T004527Z`
- config change:
  - `team_points_budget_parameterization = team_ppp_implied`
  - `team_points_reconcile_budget = true`
  - `team_points_reconcile_alpha = 0.35`
  - budget source is `pred_team_ppp * pred_possessions`
- result:
  - `best_val_total = 13.4820`
  - `best_epoch = 1`
  - the branch deteriorated monotonically after epoch 1
- interpretation:
  - making learned PPP operative as a post-flow points reconcile is also too
    destabilizing
  - this failure pattern matches the earlier market-implied points-budget
    branch: direct scoring-budget anchoring is too late in the chain

Updated boundary:

- team split should not be reintroduced primarily as a post-flow points
  reconcile, even when the budget comes from a learned PPP head
- the remaining credible direction is to move the split into earlier team event
  generation, not player scoring reconciliation

Quick diagnostic on the training labels confirms where actual game margin is
coming from. On per-game team differentials, point margin correlates most
strongly with:

- `eFG differential`: `0.8137`
- `DREB differential`: `0.6095`
- `TOV differential`: `-0.3377`
- `FGA differential`: `0.1448`
- `FTA differential`: `0.0943`
- `OREB differential`: `0.0438`

A simple standardized OLS decomposition says the largest drivers are:

- `eFG`: `1.0102`
- `FGA`: `0.5765`
- `FTA`: `0.3996`

So the margin problem is not mainly an `OREB/TOV` problem. It is much more
about shot quality and, secondarily, shot volume / free throws.

We still tested a world-advantage latent directly inside the event generator:

- run:
  `/home/daniel/projects/projections-v2/data/training/runs/game_transformer_v2_20260329T010755Z`
- change:
  - add a sampled game-level `team_advantage` head
  - inject its antisymmetric projection directly into `TeamEventBackbone` rate
    logits and `ThreePAShareHead` logits
  - supervise the latent mean on true team point margin
- result:
  - `best_val_total = 14.4014`
  - `best_epoch = 1`
  - `val_team_advantage_aux` was live (`~0.94`), but broad validation was much
    worse than the direct PPP-backbone branch

Interpretation:

- the new latent does learn a margin-like signal
- but a backbone-only advantage latent is still too weak, because actual margin
  is dominated by `eFG`
- this means the next credible branch is not more event-side latent work; it is
  an operative **scoring-rate latent** or make-rate bias mechanism that can move
  shot quality directly without using late points reconcile

We then added the first shooting-match feature expansion to the shared
train/live dataset path:

- player shooting priors from box score history:
  - `fg2_pct_prior_*`, `fg3_pct_prior_*`, `ft_pct_prior_*`,
    `efg_pct_prior_*`
  - `fg2a_per_min_prior_*`, `fg3a_per_min_prior_*`,
    `fta_per_min_prior_*`, `three_pa_share_prior_*`
- opponent defensive allowance priors:
  - `opp_fg2_pct_allowed_prior_*`, `opp_fg3_pct_allowed_prior_*`,
    `opp_fta_rate_allowed_prior_*`, `opp_efg_pct_allowed_prior_*`,
    `opp_three_pa_share_allowed_prior_*`

That feature work is now live in the priors and dataset builders:

- `scripts/rotation/build_rotation_priors_v1.py`
- `projections/rotation/rotation_set_minutes_features_v1.py`
- `projections/rotation/live_features_v1.py`

Rebuilt artifacts:

- rotation dataset:
  `/home/daniel/projections-data/training/datasets/rotation_train_v1_shootmatch_20260329T014610Z`
- joint GTv2 dataset:
  `/home/daniel/projections-data/training/datasets/joint_rotation_rates_v1_shootmatch_20260329T014610Z`

The first GTv2 probe on the rebuilt dataset regressed:

- run:
  `/home/daniel/projections-data/training/runs/gtv2_shootmatch_probe2_20260329T020051Z`
- result:
  - `best_val_total = 11.9214`
  - retained live lineage reference remains about `11.3327`

The important point is that this was **not** a clean read on the new priors
themselves. Adding the new columns to the generic feature stack widened
`player_proj` and the flow conditioner inputs, so the warm-start skipped
`85` shape-mismatched keys. In practice, the probe partly reinitialized the
same flow-conditioning path we were trying to evaluate.

Updated conclusion on the shooting-match branch:

- the data work is worth keeping
- the rebuilt train/live datasets now contain the right first-pass matchup
  priors
- but a generic “just add the columns to the main feature stack” probe is too
  confounded by warm-start breakage to answer the modeling question

Updated conclusion on full retrain:

- a blind full retrain is **not** the best immediate next step
- it would remove the warm-start shape-mismatch issue, but it would also mix
  together two questions:
  - whether the new shooting/matchup priors are useful
  - whether a scratch run on the current architecture can recover the retained
    warm-start quality
- recent GTv2 history has generally favored staged warm-start lineages over
  scratch retrains for broad validation

Recommended next step:

- keep the rebuilt dataset
- route the new shooting and opponent-allowance priors into a dedicated
  efficiency-side residual / sidecar path, so they can influence make-rate /
  shot-quality estimation without widening and reinitializing the generic
  flow-conditioning stack
- only after that branch shows real lift should we consider a cleaner full
  retrain of the retained architecture with the new features integrated

### Shooting-match sidecar status update (2026-03-29)

We implemented the first dedicated efficiency-side sidecar path for the rebuilt
shooting / matchup priors.

Implementation summary:

- `GameLevelExample` / batch collation now support a separate
  `efficiency_sidecar_features` tensor
- GTv2 now has an `efficiency_player_sidecar_encoder` that injects those
  features only into the efficiency branch
- trainer now supports:
  - `--efficiency-sidecar-feature-cols`
  - `--feature-columns-json` to lock the generic player feature stack to a
    retained bundle contract

This was specifically meant to avoid widening `player_proj` just to test the
new shooting-match features.

First clean-contract sidecar probe:

- run:
  `/home/daniel/projections-data/training/runs/gtv2_shootmatch_sidecar_probe3_20260329T022417Z`
- dataset:
  `/home/daniel/projections-data/training/datasets/joint_rotation_rates_v1_shootmatch_20260329T014610Z`
- retained feature contract:
  `/home/daniel/projections-data/artifacts/game_transformer_v2/bundles/reb_mtfanneal_live_20260328/config.json`
- result:
  - `best_val_total = 14.3401`
  - still much worse than the retained live lineage (`~11.3327`)

Important interpretation:

- the sidecar path itself is now doing the right architectural thing:
  - new shooting/matchup priors are no longer forcing `player_proj` to widen
- however, this still did **not** produce a clean no-mismatch replay of the
  retained branch
- `player_proj.weight` mismatch disappeared, but the warm-start still skipped
  `84` flow-conditioner tensors
- so this first sidecar probe is cleaner than the generic-stack probe, but it
  is still not a true “same branch plus sidecar only” test

Updated conclusion:

- the first sidecar formulation is not a keeper on broad validation
- it also does **not** justify jumping to a full retrain
- the remaining problem is now narrower:
  - either the current research recipe still does not match the retained live
    branch closely enough for a fair sidecar test
  - or the shoot-match priors are not adding enough incremental signal in this
    form

Best next step if this line is continued:

- run an exact-config replay of the retained live branch with only the new
  sidecar encoder added
- do not interpret the current sidecar probe as a final verdict on the feature
  family until the remaining flow-conditioner warm-start mismatch is removed

We then fixed the remaining architecture drift by matching the retained live
branch on:

- `flow_target_schema = v2`
- `flow_coupling_type = rqs`
- exact retained generic `feature_columns` contract via
  `--feature-columns-json`

That produced the first truly clean sidecar read:

- run:
  `/home/daniel/projections-data/training/runs/gtv2_shootmatch_sidecar_probe4_20260329T022835Z`
- warm-start result:
  - only missing keys were the 6 new sidecar encoder tensors
  - no `player_proj` mismatch
  - no flow-conditioner mismatch
- validation result:
  - `best_val_total = 11.8556`

Interpretation:

- this is the correct apples-to-apples read on the feature family
- the sidecar helps substantially relative to the earlier broken probes
- but it still does **not** beat the retained live lineage (`~11.3327`)

We also ran the scratch/full-retrain version of the same branch:

- run:
  `/home/daniel/projections-data/training/runs/gtv2_shootmatch_sidecar_scratch_20260329T022947Z`
- result:
  - phase-2 instability rollback in epoch 1
  - `best_val_total = inf`

Updated conclusion on full retrain:

- there is some real downside, even if the wall-clock cost is acceptable
- for this branch, the warm-start path is not just cheaper; it is materially
  more stable
- a scratch retrain is still a valid parallel experiment when we want the read,
  but it should not be assumed to be a harmless substitute for a clean
  warm-start branch

### Shooting-match interaction-sidecar update (2026-03-29)

We then tested the next obvious refinement: keep the exact clean sidecar replay
setup from `probe4`, but add engineered offense-vs-defense interaction deltas
instead of relying only on raw player shooting priors and raw opponent allowed
priors.

Implementation summary:

- trainer now supports `--efficiency-sidecar-add-interactions`
- when enabled, it derives matchup-delta sidecar features such as:
  - `fg2_pct_matchup_delta_*`
  - `fg3_pct_matchup_delta_*`
  - `efg_pct_matchup_delta_*`
  - `fta_rate_matchup_delta_*`
  - `three_pa_share_matchup_delta_*`
  - `team_off_vs_opp_def_delta`
- these are added only to the efficiency sidecar path; the retained generic
  feature contract still stays fixed

Result:

- run:
  `/home/daniel/projections-data/training/runs/gtv2_shootmatch_sidecar_interactions_20260329T024134Z`
- warm-start:
  - still clean
  - only the 6 sidecar encoder tensors were missing
  - no `player_proj` mismatch
  - no flow-conditioner mismatch
- validation:
  - `best_val_total = 11.8546`

Interpretation:

- this is effectively flat versus the prior clean sidecar replay
  (`11.8556 -> 11.8546`)
- the engineered matchup deltas do not materially improve the branch
- the feature family is not completely disproven, but “raw priors + simple
  engineered deltas through a small efficiency sidecar” is not enough to beat
  the retained live lineage (`~11.3327`)

Updated next-step boundary:

- do not spend more cycles on small sidecar feature engineering variants in
  this form
- if the shooting/matchup line continues, the next branch should be a more
  operative team shot-quality / efficiency residual mechanism rather than more
  input tweaks to the current sidecar MLP
