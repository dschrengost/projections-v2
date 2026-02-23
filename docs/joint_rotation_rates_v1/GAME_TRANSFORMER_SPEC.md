# Game Transformer v2: Joint Game Distribution Model

## Spec Status: DRAFT v0.2 (2026-02-23)

---

## 1. Motivation

### 1.1 Current gap

The current production stack is still a staged pipeline:

```
Minutes model -> rates model -> sim_v2 worlds -> quantiles -> QuickBuild optimizer
```

That design makes correlation quality an emergent property of many hand-tuned modules.
Even with strong tuning, we still rely on assumptions about availability, minutes coupling,
rate noise, and game-level covariance.

### 1.2 What this spec changes

This revision makes one hard decision:

1. Keep cross-team attention in the backbone.
2. Replace per-player independent sampling with a true joint game distribution head.
3. Remove independent Bernoulli gate sampling (joint team rotation sampling instead).
4. Enforce feasibility by construction (not by large post-hoc correction).

### 1.3 Non-negotiable requirements

- Learn intra-team and intra-game correlations from data.
- Produce feasible team minutes in every world.
- Preserve downstream `sim_v2` output contract so QuickBuild and dashboard integrations stay stable.
- Keep exact likelihood training for the generative head.

---

## 2. Design Principles

1. No fake "joint" model:
   if sampling is independent per player, this is not acceptable.

2. Constraints are mostly parameterized, not patched:
   projection can do small numeric correction only.

3. Discrete and continuous structure are modeled explicitly:
   active set, minutes, and box-score lines are different random objects.

4. Start with the smallest high-signal target space:
   v1 optimizes DFS-relevant stats first.

---

## 3. Architecture Overview

## 3.1 High-level factorization

For each game `g` with both teams:

- `X`: input features for up to 30 players + game/team context
- `A`: active set indicators (who plays meaningful minutes)
- `M`: minutes vector for all players
- `Y`: box-score target vector for all players

We model:

```
p(A, M, Y | X) = p(A | X) * p(M | A, X) * p_flow(Y | A, M, X)
```

where `p_flow` is a **joint game-level flow**, not per-player independent flows.

## 3.2 Backbone: cross-team game transformer

Sequence layout per game:

```
[GAME] [TEAM_H] [home_players...] [TEAM_A] [away_players...]
```

- max length: 33 tokens (1 + 1 + 15 + 1 + 15)
- no positional encoding
- learned team-side embedding for player tokens
- full self-attention across both teams

Backbone output:

- `H_player`: `(B, P, d_model)` player states
- `H_team`: `(B, 2, d_model)` team token states
- `H_game`: `(B, d_model)` game token state

## 3.3 Joint active-set head (replaces independent gate sampling)

### Team-wise active count

For each team `t`, predict a categorical distribution over active count:

```
K_t in {5, 6, ..., 13}
```

using `H_team[t]`.

### Team-wise player selection without replacement

Given `K_t`, score players on that team and sample a subset jointly with
Gumbel-TopK / Plackett-Luce style sampling.

Properties:

- sampled active sets are correlated by construction
- impossible states (for example 2 active players) are unreachable
- competition for slots is learned directly

Training:

- supervised active labels from minutes threshold (for example `minutes >= 1.0`)
- count loss + set-membership loss

## 3.4 Joint minutes head (team-constrained)

Given `A` and context, produce team minutes with exact constraints.

For each team:

1. Predict unconstrained minute preference `u_i` for active players.
2. Solve capped-simplex projection:

```
argmin_m ||m - u||^2
subject to:
  sum_i m_i = 240
  0 <= m_i <= 48
  m_i = 0 for inactive players
```

Feasibility is guaranteed because `K_t >= 5`.

This head yields realistic negative teammate minute correlation naturally,
without bench/starter hand-coded realloc rules.

## 3.5 Joint game flow head (core change)

### Target tensor

Let `Y` be `(P, S)` for each game.

v1 target dimensions per player (`S = 14`):

1. `fga2`
2. `fg2m`
3. `fga3`
4. `fg3m`
5. `fta`
6. `ftm`
7. `oreb`
8. `dreb`
9. `ast`
10. `stl`
11. `blk`
12. `tov`
13. `pf`
14. `minutes` (jointly modeled with stats)

Notes:

- `plus_minus` is excluded in v1 to reduce noise and data pressure.
- percentages are derived downstream (`fg2_pct`, `fg3_pct`, `ft_pct`),
  avoiding undefined-label edge cases at zero attempts.

### Flow structure

Use a **set-equivariant coupling flow** over full game tensors.

Each coupling block:

1. partitions `(P, S)` into transformed subset and conditioning subset
2. computes shift/scale with an equivariant conditioner that uses:
   - conditioning subset values
   - all player states `H_player`
   - team/game states `H_team`, `H_game`
3. applies invertible affine or spline transform

Because transformations depend on other players in the same game,
the resulting `Y` distribution has learned cross-player covariance.

No separate global latent branch is required.

### Base noise

`Z ~ N(0, I)` with shape `(P, S)`.

Correlation comes from joint invertible transforms, not from independent
per-player heads.

## 3.6 Constraint handling

Primary constraints are handled in-model. Lightweight projection is kept only
for numeric cleanup.

Hard requirements for every sampled world:

- team minutes sum exactly 240
- `0 <= minutes <= 48`
- counting stats non-negative
- makes do not exceed attempts:
  - `fg2m <= fga2`
  - `fg3m <= fga3`
  - `ftm <= fta`
- derived stats are internally consistent:
  - `fga = fga2 + fga3`
  - `fgm = fg2m + fg3m`
  - `pts = 2*fg2m + 3*fg3m + ftm`
  - `reb = oreb + dreb`

---

## 4. Data and Labels

## 4.1 Training unit

A training example is a full game (both teams), not a team-game row.

## 4.2 Label source

Use box-score-derived labels where possible to increase sample size.

- mandatory: attempts, makes, rebounds, assists, steals, blocks, turnovers, fouls, minutes
- derived in pipeline: points, percentages, totals

## 4.3 Dequantization

Apply dequantization noise on-the-fly during training for integer-like targets,
not as a fixed dataset column.

```
y_tilde = y + Uniform(-0.5, 0.5)
```

This prevents memorization of one fixed jitter realization.

## 4.4 Masks and eligibility

- active-set supervision mask: roster-valid players
- minutes supervision mask: players with observed minutes
- flow-dimension mask:
  - DNP rows: no stat likelihood terms
  - low-min rows: optionally downweight volatile dims

---

## 5. Losses and Training

## 5.1 Loss decomposition

```
L_total = a1 * L_backbone
        + a2 * L_joint_gen
        + a3 * L_decision
```

### Tier 1: deterministic anchors (`L_backbone`)

Keep existing supervised anchors (minutes/rates style losses) to stabilize training.

### Tier 2: joint generative losses (`L_joint_gen`)

```
L_joint_gen = L_active_count
            + L_active_set
            + L_minutes_nll
            + L_flow_nll
            + w_c * L_constraint_soft
```

- `L_active_count`: CE for team active count distribution
- `L_active_set`: listwise/set loss for selected players
- `L_minutes_nll`: likelihood-style objective for minutes before projection
- `L_flow_nll`: exact NLL from joint game flow
- `L_constraint_soft`: light penalty for residual violations pre-cleanup

### Tier 3: decision losses (`L_decision`)

```
L_decision = L_crps_fpts + w_e * L_team_energy
```

- `L_crps_fpts`: CRPS on sampled DK FPTS
- `L_team_energy`: energy score on team FPTS vectors

CRPS definition (reference implementation should use this form directly):

```
CRPS = (1/K) * sum_i |x_i - y|
     - (1/(2*K*K)) * sum_i sum_j |x_i - x_j|
```

Start with the explicit pairwise form for correctness at `K <= 32`.
Optimize later only after parity tests.

## 5.2 Phase schedule

- Phase 1 (epochs 1-8): `a1=1.0, a2=0.0, a3=0.0`
- Phase 2 (epochs 9-18): `a1=0.5, a2=1.0 (warmup over 4 epochs), a3=0.0`
- Phase 3 (epochs 19-30): `a1=0.25, a2=0.5, a3=1.0`

## 5.3 Stability guards

- separate grad clipping:
  - backbone: `max_norm=1.0`
  - joint flow: `max_norm=5.0`
- NLL guard:
  - if generative NLL explodes for 2 consecutive batches, halve `a2`
  - if repeated, stop and revert to last stable checkpoint
- minutes regression guard:
  - if val minutes MAE > `1.2x` Phase 1 best for 3 epochs, abort Phase 2/3

---

## 6. Inference and World Generation

## 6.1 Sampling algorithm (single game)

1. Run backbone, get context states.
2. Sample team active counts `K_home`, `K_away`.
3. Sample active sets jointly per team.
4. Sample team minutes jointly with capped-simplex constraints.
5. Sample full game stat tensor from joint flow conditioned on context, `A`, `M`.
6. Apply lightweight numeric cleanup (should be near-zero adjustment).
7. Derive box-score totals and DK FPTS.

## 6.2 Performance target

- generate 25k worlds with batching (`5k` chunks)
- target latency:
  - GPU: < 8s per game
  - CPU: < 45s per game

(Updated target acknowledges joint-flow cost.)

---

## 7. Output Contract and Integration

## 7.1 Contract requirement

Output schema must remain compatible with `scripts/sim_v2/generate_worlds_fpts_v2.py`
summary columns consumed by API and QuickBuild paths.

## 7.2 Explicit semantics

- `dk_fpts_*`: conditional on active (`A=1`)
- `dk_fpts_*_uncond`: unconditional over active/inactive outcomes
- `minutes_sim_*`: conditional on active
- `minutes_sim_*_uncond`: unconditional
- `sim_p_active`: marginal active probability from sampled active sets
- `sim_p_rotation`: probability of being in the rotation set

These definitions must be asserted in tests and documented in output manifest.

---

## 8. Evaluation Plan

## 8.1 Offline metrics

- minutes MAE vs current baseline (must be no worse)
- FPTS CRPS vs sim_v2 (must improve)
- FPTS tail coverage (`p90`, `p95`) calibration
- teammate correlation diagnostics:
  - pairwise correlation RMSE
  - team-stack variance calibration
- active-rate calibration by role tier
- DD/TD rate calibration

## 8.2 Backtest metrics (QuickBuild)

Across 20-30 historical slates:

- lineup mean realized FPTS
- top-lineup realized FPTS
- ROI proxy metrics
- exposure and stacking pattern shifts

## 8.3 Go/no-go criteria

Ship candidate only if all hold:

1. minutes MAE <= current production baseline
2. FPTS CRPS < sim_v2
3. tail calibration error <= 3% on p90/p95
4. backtest does not regress on aggregate ROI proxy

---

## 9. Risk Analysis

## 9.1 Main technical risks

1. Joint-flow optimization instability.
2. Overfitting due to limited games.
3. Runtime cost for 25k-world generation.

## 9.2 Mitigations

- start with 3-4 coupling blocks and small conditioner width
- strong regularization and early stopping on val NLL/CRPS
- remove low-value targets in v1 (`plus_minus`)
- chunked world generation and mixed precision inference

## 9.3 Fallback hierarchy

If full joint flow is unstable:

1. keep joint active-set + joint minutes heads
2. simplify flow to lower-capacity coupling blocks
3. fallback to conditional Gaussian copula over game tensor

Do not fallback to independent per-player sampling.

---

## 10. Implementation Roadmap

### Phase 1: Foundation

- [ ] Build game-level dataset (both teams per game)
- [ ] Add box-score label builder in make/attempt space
- [ ] Implement `GameTransformerV2` backbone with cross-team attention
- [ ] Implement joint active-set head:
  - [ ] team active-count classifier
  - [ ] set selection sampler (without replacement)
- [ ] Implement joint team-minutes head with capped-simplex solver
- [ ] Train deterministic/joint-discrete Phase 1 model
- [ ] Validate minutes parity vs current baseline

### Phase 2: Joint generative model

- [ ] Implement `JointGameFlow` module (equivariant coupling blocks)
- [ ] Wire exact `L_flow_nll` over game tensors
- [ ] Add mixed objective (`L_active* + L_minutes_nll + L_flow_nll`)
- [ ] Add stability guards and checkpoint rollback logic
- [ ] Build sampling module `projections/rotation/sample_worlds_v2.py`
- [ ] Implement schema-compatible summary writer

### Phase 3: Decision fine-tuning

- [ ] Add `L_crps_fpts` and `L_team_energy`
- [ ] Run offline eval suite vs sim_v2
- [ ] Run historical slate backtests with QuickBuild
- [ ] Produce go/no-go report

### Phase 4: Integration

- [ ] Add feature-flagged dispatch in `prefect_flows/live_nba_pipeline.py`
- [ ] Keep sim_v2 as immediate fallback path
- [ ] Run short shadow period (3-5 slates) with automated parity checks
- [ ] Promote to default if go/no-go thresholds remain satisfied

---

## 11. Concrete Module Plan

## 11.1 New files

- `projections/rotation/game_transformer_v2.py`
- `projections/rotation/joint_active_set.py`
- `projections/rotation/joint_minutes.py`
- `projections/rotation/joint_game_flow.py`
- `projections/rotation/sample_worlds_v2.py`
- `scripts/rotation/train_game_transformer_v2.py`
- `scripts/rotation/eval_game_transformer_v2.py`
- `scripts/rotation/generate_worlds_game_transformer_v2.py`

## 11.2 Existing files to reuse

- `projections/rotation/joint_set_model_v1.py`
- `projections/rotation/set_model.py`
- `projections/rotation/training_losses.py`
- `scripts/rotation/build_joint_rotation_rates_dataset_v1.py`
- `scripts/sim_v2/generate_worlds_fpts_v2.py` (contract reference)

---

## 12. Open Questions

1. Coupling type in v1 default: affine vs spline.
2. Active-label threshold definition (`>= 1` min vs `>= 4` min).
3. Whether to include `pf` in v1 target space.
4. Whether to include explicit blowout token beyond spread/total features.

---

## 13. Summary

This v0.2 rewrite keeps cross-team attention and upgrades the generative side from
independent per-player sampling to a true joint game distribution.

The core shift is architectural, not cosmetic:

- joint active-set sampling
- joint team-constrained minutes
- joint game flow for box-score worlds

That is the minimum design that can genuinely learn realistic intra-team and
intra-game correlations from data while staying feasible and production-integrable.
