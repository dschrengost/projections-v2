# 01 Architecture

## Design Goals

1. Keep the current strengths of rotation-set minutes modeling (set encoder + 240-minute allocation).
2. Add rates prediction in the same model so rates can condition on learned rotation/role context.
3. Preserve current downstream contracts (`minutes` and `rates_v1_live` style outputs).
4. Provide a path to learned covariance for world generation.

## Current Baseline (Reference)

- Minutes/rotation model:
  - `projections/rotation/set_model.py`
  - `scripts/rotation/train_rotation_set_model_v1.py`
- Rates model:
  - `scripts/rates/train_rates_v1.py`
- Sim combines minutes and rates late:
  - `scripts/sim_v2/generate_worlds_fpts_v2.py`

## Proposed Joint Model

Unit of prediction remains a team-game with variable roster size (set input).

### Shared Encoder

- Reuse `SetTransformerMinutesModel` style encoder:
  - player-level input embedding,
  - permutation-invariant attention over roster slots,
  - optional team/opponent + player + player-team embeddings.

Encoder output:

- `h_i` per player slot `i`,
- optional pooled context `h_team`.

### Heads

1. Rotation Gate Head
- Output: `gate_logit_i`
- Meaning: player in-rotation likelihood.

2. Minutes Share Head
- Output: `share_logit_i`
- Combined with gate logits via existing allocation:
  - `minutes_i = 240 * normalize(sigmoid(gate_i) * simplex(share_i))`
- Reuse current alloc activation options (entmax/softmax/etc.).

3. Rates Mean Heads (multi-target)
- Outputs per player:
  - `fga2_per_min, fga3_per_min, fta_per_min, ast_per_min, tov_per_min, oreb_per_min, dreb_per_min, stl_per_min, blk_per_min`
- One joint multi-output MLP head or small per-target heads off a shared rates trunk.

4. Efficiency Heads
- Outputs per player:
  - `fg2_pct, fg3_pct, ft_pct`
- Constrained to sane ranges via sigmoid + affine bounds.

5. Optional Covariance Head (Phase 2)
- Low-rank latent loadings for player-stat residual correlation.
- Intended to replace/augment post-hoc `game_factor`/independent noise in sim.

## Forward Contract

Given team-game roster set:

1. Encoder computes `h`.
2. Gate/share heads produce rotation + minutes.
3. Rates/efficiency heads produce per-minute means.
4. Derived expected counts for consistency loss:
   - `count_hat = minutes_hat * rate_hat`.

Inference outputs (MVP):

- minutes block (current columns):
  - `rotation_minutes_p50`, `gate_prob`, `gate_logit`, `share_logit` (as available).
- rates block (current columns):
  - `pred_*_per_min`, `pred_fg2_pct`, `pred_fg3_pct`, `pred_ft_pct`.

## Why This Should Improve Over Separate Models

1. Shared latent context means rates can react to projected role, not only handcrafted proxy features.
2. Injury/vacancy regimes become internal conditioning, not purely external feature engineering.
3. Less reliance on post-hoc covariance knobs because dependencies can be learned in-model.

## Risks

1. Identifiability: minutes-vs-rate error tradeoff can drift.
2. Training instability with many losses and targets.
3. Over-coupling could hurt one task while helping another.

Mitigation: staged training and guarded loss weights (details in `03_TRAINING_AND_ROLLOUT.md`).
