# Game Transformer: End-to-End FPTS Distribution Model

## Spec Status: DRAFT v0.1 (2026-02-22)

---

## 1. Motivation

### What we have today

The current production pipeline is a **five-stage cascade**:

```
Minutes Model (LightGBM)
  → Rates Model (LightGBM)
    → Monte Carlo Sim (25K worlds, ~15 modules of hand-tuned physics)
      → FPTS Quantiles
        → Optimizer
```

Each stage introduces compounding assumptions:

| Stage | Hand-coded assumptions |
|-------|----------------------|
| Minutes sim | Student-t noise (σ per starter/bench), bench-zero mixture, absorption caps by depth rank, game-script quantile shifts |
| Availability | `play_prob_policy` with 15+ tuning knobs (starter floor, core lock, DNP blockers, depth blockers) |
| Team-240 | Iterative projection with priority weights, bisection Lagrange multiplier |
| Rates noise | Residual buckets (6 min bins × starter/bench × 4 injury status), team/player σ scales |
| Game factor | Additive shock (σ=20) by minutes share; game-script margin sampling from Vegas spread |
| Correlations | Implicitly from game factor only; no within-stat or cross-player stat correlation |

**The core problem:** every one of these hand-coded distributions and constraints is a
*designer's prior* about NBA variance. If that prior is wrong (and it always is, somewhat),
the tails are wrong, the correlations are wrong, and the optimizer makes suboptimal decisions.

### What we want

A single model that:

1. **Inputs** a full game's worth of players (both teams) plus context
2. **Outputs** a joint distribution over all box-score stats for all players in that game
3. **Naturally learns** the 240-minute constraint, rotation patterns, intra-team correlations, game-script effects, and realistic tails — from data, not hand-coded rules
4. **Can be sampled** to produce correlated "worlds" for downstream optimization

---

## 2. Architecture Overview

### 2.1 High-Level Design: Game-Level Set Transformer + Normalizing Flow

```
                  ┌─────────────────────────────┐
                  │  Per-Player Feature Encoder  │
                  │  (reuse existing joint_set   │
                  │   feature projection + MLP)  │
                  └──────────┬──────────────────┘
                             │ (B, P, d_embed)
                  ┌──────────▼──────────────────┐
                  │   Game-Level Transformer     │
                  │  (cross-team attention with  │
                  │   team/role/matchup tokens)  │
                  └──────────┬──────────────────┘
                             │ (B, P, d_model)
              ┌──────────────┼──────────────────┐
              ▼              ▼                   ▼
    ┌─────────────┐  ┌──────────────┐  ┌────────────────┐
    │ Gate Head   │  │ Share Head   │  │ Box-Score Head │
    │ (rotation   │  │ (minutes     │  │ (per-minute    │
    │  membership)│  │  allocation) │  │  stat rates +  │
    │             │  │              │  │  efficiencies) │
    └──────┬──────┘  └──────┬───────┘  └───────┬────────┘
           │                │                   │
           ▼                ▼                   ▼
    ┌──────────────────────────────────────────────────┐
    │         Conditional Flow (per-player)            │
    │  Maps deterministic predictions → learnable      │
    │  distribution. Conditioned on transformer h.     │
    │  Output: full joint sample of                    │
    │    (minutes, pts, reb, ast, stl, blk, tov,       │
    │     fgm, fga, fg3m, fg3a, ftm, fta, oreb, dreb) │
    └──────────────────────────────────────────────────┘
              │
              ▼
    ┌──────────────────────────────────────────────────┐
    │  Constraint Projection Layer                     │
    │  - Team minutes → 240 (differentiable Sinkhorn  │
    │    or entmax re-normalization)                   │
    │  - Box score consistency (pts = 2*fg2m + 3*fg3m  │
    │    + ftm, reb = oreb + dreb)                     │
    │  - Per-player minute cap [0, 48]                 │
    └──────────────────────────────────────────────────┘
              │
              ▼
         (B, P, S) sampled box-score lines
              │
              ▼
         DK FPTS scoring (deterministic, differentiable)
```

### 2.2 Why This Architecture

**Set Transformer backbone (reused from joint_set_model_v1):** Already proven to learn rotation
structure. We keep the permutation-invariant design but extend from single-team to full-game.

**Conditional normalizing flow:** This is the key innovation. Instead of hand-coding noise
distributions, we *learn* the residual distribution conditioned on context. The flow can
capture:
- Heavy tails (a Gaussian can't; the flow can learn Student-t-like or asymmetric tails)
- Correlation between stats (assists and turnovers are correlated; pts and minutes are correlated)
- State-dependent variance (high-minute starters have lower relative variance than bench players)
- The zero-inflated nature of bench players (mass at exactly 0 minutes)

**Constraint projection:** Hard constraints (team-240, box-score arithmetic) are enforced
*after* sampling but *before* loss computation, so the model never produces infeasible outputs
but still receives gradients through the projection.

---

## 3. Detailed Architecture

### 3.1 Input Representation

**Per-player features** (reuse existing `build_joint_rotation_rates_dataset_v1.py` spine):

| Category | Examples | Dim |
|----------|----------|-----|
| Rolling stats | `roll_mean_5`, `started_proxy_rate_prior_5`, etc. | ~30 |
| Team context | pace, off_rtg, def_rtg (own + opp) | ~6 |
| Availability | `is_out`, `is_gtd`, `is_prob`, `play_prob` | ~4 |
| Salary / position | DK salary, position one-hot | ~8 |
| Vegas | spread, total, team ITT, opp ITT | ~4 |
| Props (optional) | Action Network lines for pts/reb/ast/etc. | ~10 |

**Embedding indices** (reuse existing):
- `team_id_idx` → 8-dim team embedding
- `opp_id_idx` → 8-dim opponent embedding
- `player_id_idx` → 16-dim player embedding (optional)
- `player_team_hash_idx` → 8-dim player×team embedding

**Game-level tokens** (new):
- **[GAME]** token: Vegas spread, total, time-of-day, day-of-week, rest days
- **[TEAM_A]**, **[TEAM_B]** tokens: team-level aggregates (pace, rating, injury load)
- These participate in attention but don't produce box-score outputs

**Per-player input dim:** ~70 features + embeddings → projected to `d_embed=128`

### 3.2 Game-Level Transformer

**Why cross-team attention matters:** The current joint_set model processes each team
independently. But NBA games have inter-team correlations:
- Opponent defensive quality affects shooting efficiency
- Pace matchup affects total possessions (and thus counting stats)
- Blowout dynamics affect both teams' rotations simultaneously
- Star matchups affect usage patterns

```python
class GameTransformer(nn.Module):
    """Full-game transformer processing both teams + game context tokens."""

    def __init__(
        self,
        d_embed: int = 128,
        d_model: int = 256,
        n_layers: int = 4,         # deeper than single-team (was 2)
        n_heads: int = 8,          # more heads for richer attention
        d_ff: int = 512,
        dropout: float = 0.1,
    ):
        # Input projection: per-player features → d_embed
        self.player_proj = MLP(d_embed, d_model, d_model, num_layers=2)

        # Game/team token projections
        self.game_token_proj = nn.Linear(game_context_dim, d_model)
        self.team_token_proj = nn.Linear(team_context_dim, d_model)

        # Team indicator embedding (disambiguate home/away within attention)
        self.team_side_embed = nn.Embedding(2, d_model)  # 0=home, 1=away

        # Transformer encoder (no positional encoding — set invariant)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
```

**Sequence layout per game:**

```
[GAME] [TEAM_H] [player_h_1] ... [player_h_15] [TEAM_A] [player_a_1] ... [player_a_15]
```

Total sequence length: 1 + 1 + 15 + 1 + 15 = 33 tokens (small enough for full attention).

**Team disambiguation:** Instead of positional encodings, we add a learned `team_side_embed`
(home=0, away=1) to every player token. This preserves set invariance within each team
while letting the model distinguish which team a player is on.

### 3.3 Prediction Heads (Deterministic)

These produce the **conditional mean** (point prediction) — the "center" of the distribution.

#### Gate Head (reused from joint_set_model_v1)
```
h_player → Linear(d_model, 1) → gate_logit
gate_prob = sigmoid(gate_logit)
```
Predicts P(in rotation) per player.

#### Share Head (reused from joint_set_model_v1)
```
h_player → Linear(d_model, 1) → share_logit
minutes = 240 * entmax(gate_prob * share_logit) per team
```
Predicts minute allocation conditioned on rotation membership.

#### Box-Score Rate Head (extended from joint_set_model_v1)
```
h_player → MLP(d_model, 128, 128) → rates_trunk
rates_trunk → Linear(128, n_rate_targets) → per-minute rates
rates_trunk → Linear(128, n_eff_targets) → sigmoid → bounded efficiencies
```

**Rate targets (9):** fga2_per_min, fga3_per_min, fta_per_min, ast_per_min, tov_per_min,
oreb_per_min, dreb_per_min, stl_per_min, blk_per_min

**Efficiency targets (3):** fg2_pct [0.3, 0.75], fg3_pct [0.2, 0.55], ft_pct [0.5, 0.95]

These heads are trained with the same MAE losses as today (§5.1), providing a stable
"backbone" prediction. The flow head (§3.4) learns to model *deviations* from these means.

### 3.4 Conditional Normalizing Flow (Distribution Head)

This is the core innovation — replacing all of sim_v2's hand-coded noise/correlation logic.

#### What is a conditional normalizing flow?

A normalizing flow learns an invertible mapping `f: z → x` where `z ~ N(0,I)` is simple
noise and `x` is the complex target distribution. "Conditional" means the mapping parameters
depend on context (the transformer's hidden state for each player).

#### Architecture: Masked Autoregressive Flow (MAF)

```python
class PlayerConditionalFlow(nn.Module):
    """Conditional MAF that maps N(0,I) → box-score residuals.

    Conditioned on:
      - h_player (d_model): transformer hidden state for this player
      - pred_minutes: deterministic minutes prediction
      - pred_rates: deterministic rate predictions
    """

    def __init__(
        self,
        stat_dim: int = 15,      # minutes + 14 box-score stats
        context_dim: int = 256,   # d_model
        n_flow_layers: int = 6,
        hidden_dim: int = 128,
    ):
        self.flows = nn.ModuleList([
            ConditionalAffineAutoregressive(
                stat_dim=stat_dim,
                context_dim=context_dim,
                hidden_dim=hidden_dim,
            )
            for _ in range(n_flow_layers)
        ])
```

**Output space (15 dimensions per player):**

| Dim | Stat | Notes |
|-----|------|-------|
| 0 | minutes | 0–48, zero-inflated for bench |
| 1 | fga2 | 2pt field goal attempts |
| 2 | fg2_pct | 2pt field goal % |
| 3 | fga3 | 3pt field goal attempts |
| 4 | fg3_pct | 3pt field goal % |
| 5 | fta | free throw attempts |
| 6 | ft_pct | free throw % |
| 7 | oreb | offensive rebounds |
| 8 | dreb | defensive rebounds |
| 9 | ast | assists |
| 10 | stl | steals |
| 11 | blk | blocks |
| 12 | tov | turnovers |
| 13 | pf | personal fouls |
| 14 | plus_minus | game +/- |

**Why MAF over other flow architectures:**
- **Autoregressive structure** naturally captures the dependency chain:
  minutes → attempts → makes → assists/boards/etc.
- **Exact log-likelihood** training (no ELBO approximation like VAEs)
- **Fast sampling** in the forward direction (parallel)
- **Conditional** on transformer context makes it player-specific

#### Zero-inflation for DNP players

The flow needs to handle the mass-at-zero for players who don't play. We use a
**mixture model**:

```
P(box_score | context) = (1 - gate_prob) * δ(0)  +  gate_prob * flow(z | context)
```

- With probability `(1 - gate_prob)`, the player gets a zero line (DNP)
- With probability `gate_prob`, sample from the flow conditioned on being in rotation
- The gate head (§3.3) provides `gate_prob`; it's trained jointly

This eliminates all of sim_v2's `play_prob_policy` machinery (15+ hand-tuned knobs).

### 3.5 Constraint Projection Layer

After sampling from the flow, we enforce hard constraints differentiably.

#### Team-240 Minutes Projection

```python
def project_team_240(minutes_sample, team_mask, temperature=0.1):
    """Differentiable projection of sampled minutes to team-240 constraint.

    Uses Sinkhorn-like iterative scaling:
    1. For each team, compute team_total = sum(minutes[team])
    2. Scale: minutes[team] *= 240 / team_total
    3. Clamp to [0, 48] per player
    4. Repeat until convergence (typically 3 iterations)

    Gradients flow through the scaling operation.
    """
```

This replaces `_enforce_team_240_simple()` and the full `minutes_allocator.py`.

#### Box-Score Consistency

Deterministic post-projection (no learned params, just arithmetic constraints):

```
pts = (fga2 * fg2_pct) * 2 + (fga3 * fg3_pct) * 3 + (fta * ft_pct) * 1
reb = oreb + dreb
fgm = fga2 * fg2_pct + fga3 * fg3_pct
fg3m = fga3 * fg3_pct
ftm = fta * ft_pct
fga = fga2 + fga3
```

The model predicts *attempts* and *percentages* independently; makes/points are
derived deterministically. This guarantees internally consistent box scores.

---

## 4. What We Reuse vs. Build New

### 4.1 Reuse from existing codebase

| Component | Source | Reuse strategy |
|-----------|--------|----------------|
| Feature engineering | `build_joint_rotation_rates_dataset_v1.py` | Extend to include both teams per game |
| Feature projection + normalization | `joint_set_model_v1.py` | Lift `SetTransformerMinutesModel.forward_embeddings()` |
| Gate head architecture | `set_model.py` gate_head | Keep identical |
| Share head + entmax allocation | `set_model.py` share_head + `minutes_from_gate_and_share_logits` | Keep identical |
| Rate targets + efficiency bounds | `JointRotationRatesModelConfig` | Keep identical |
| Training losses (minutes) | `training_losses.py` | Keep all 5 loss components |
| DK FPTS scoring | `fpts_v2/scoring.py` | Use as-is for evaluation |
| Optimizer interface | `optimizer/nba_optimizer.py` | Output format-compatible | *** we use quickbuild optimizer for production***
| Labels | `gold/rates_training_base/` + box score labels | Extend to full box-score lines |



### 4.2 Build new

| Component | Effort | Priority |
|-----------|--------|----------|
| Game-level transformer (cross-team attention) | Medium | P0 |
| Conditional normalizing flow | Medium | P0 |
| Full box-score label pipeline | Low | P0 |
| Differentiable constraint projection | Low | P1 |
| Game-level dataset builder (both teams per game) | Medium | P0 |
| Flow training loop (NLL loss) | Medium | P0 |
| Sampling / worlds generation | Low | P1 |
| Evaluation harness (calibration, tail metrics) | Medium | P1 |

### 4.3 Retire (eventually)

| Component | Replaced by |
|-----------|-------------|
| `sim_v2/noise.py` | Flow's learned residual distribution |
| `sim_v2/minutes_noise.py` | Flow's minutes distribution |
| `sim_v2/minutes_stabilization.py` | Constraint projection layer |
| `sim_v2/game_factor.py` | Cross-team attention + flow correlations |
| `sim_v2/game_script.py` | Transformer context (Vegas features) |
| `sim_v2/bench_zero_mixture.py` | Gate head zero-inflation mixture |
| `sim_v2/play_prob_policy.py` (15+ knobs) | Gate head (learned from data) |
| `sim_v2/minutes_allocator.py` | Differentiable Sinkhorn projection |
| `sim_v2/residuals.py` (bucketed Student-t) | Flow (learned tails, per-context) |
| `sim_v2/minutes_worlds_model_space_v1.py` | Direct flow sampling |

---

## 5. Training

### 5.1 Loss Function: Design Philosophy

The loss design follows three principles:

1. **Losses should form a curriculum, not a soup.** Each phase of training has a clear
   primary objective. Losses are added only when earlier phases have converged.
2. **Every hand-coded sim component we retire must be replaced by a learned signal,
   not just hoped-for.** Each sim module we remove corresponds to a specific loss
   component that teaches the model to capture that behavior.
3. **The terminal loss must be decision-relevant.** We care about FPTS distribution
   quality for DFS optimization, not density estimation for its own sake.

### 5.2 Loss Tier Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  TIER 1: BACKBONE SUPERVISION (deterministic heads)             │
│  Purpose: Accurate conditional means                            │
│  Active: All phases                                             │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ L_min_mae  L_rates_mae  L_eff_mae  L_gate_bce           │  │
│  │ L_minutes_out  L_k_reg  L_anti_smear  L_rot_bce         │  │
│  └───────────────────────────────────────────────────────────┘  │
├──────────────────────────────────────────────────────────────────┤
│  TIER 2: DISTRIBUTIONAL (flow NLL + mixture)                    │
│  Purpose: Accurate conditional densities                        │
│  Active: Phase 2+                                               │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ L_nll_mixture (gate × flow joint likelihood)              │  │
│  │ L_constraint_soft (team-240, non-negativity)              │  │
│  └───────────────────────────────────────────────────────────┘  │
├──────────────────────────────────────────────────────────────────┤
│  TIER 3: DECISION-RELEVANT (sample-based)                       │
│  Purpose: FPTS distribution quality for DFS                     │
│  Active: Phase 3+                                               │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ L_crps_fpts (FPTS distribution accuracy)                  │  │
│  │ L_correlation (teammate FPTS correlation structure)        │  │
│  └───────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

---

### 5.3 Tier 1: Backbone Supervision (Unchanged from Today)

These losses train the deterministic prediction heads. They are **identical** to the
current `train_joint_rotation_rates_model_v1.py` training loop (lines 1093-1102).
They remain active at all phases — even when the flow is training — because they
anchor the backbone's conditional means.

```
L_tier1 = λ_min   * L_minutes_mae
        + λ_rates  * L_rates_mae
        + λ_eff    * L_eff_mae
        + λ_rot    * L_rot_bce
        + w_gate   * L_gate_bce
        + w_out    * L_minutes_out
        + w_k      * L_k_reg
        + w_smear  * L_anti_smear
```

| Loss | Formula | What it teaches | Replaces in sim |
|------|---------|-----------------|-----------------|
| `L_minutes_mae` | `MAE(pred_minutes, y_minutes)` masked by roster | Accurate minute allocation per player | `minutes_stabilization.py` baseline |
| `L_rates_mae` | `MAE(pred_rates, y_rates)` masked by rates_eligible | Per-minute counting stat rates | `noise.py` rate baselines |
| `L_eff_mae` | `MAE(pred_eff, y_eff)` masked by eff_eligible | Shooting percentages (bounded sigmoid) | `noise.py` efficiency baselines |
| `L_rot_bce` | `BCE(gate_logits, minutes ≥ 8)` | Soft rotation membership | `play_prob_policy.py` rotation locks |
| `L_gate_bce` | `BCE(gate_logits, minutes ≥ 6)` with dynamic pos_weight | Hard rotation membership (balanced) | `play_prob_policy.py` all 15+ knobs |
| `L_minutes_out` | `SmoothL1(pred_min * (1-in_rot), 0)` | Push DNP minutes → 0 | `bench_zero_mixture.py` |
| `L_k_reg` | `MSE(sum(sigmoid(gate)), k_target)` | Team rotation size ≈ 9.5 | `play_prob_policy.py` core_lock_topk |
| `L_anti_smear` | `mean(relu(pred_min - 4) * (1-gate_prob_detached))` | No smeared minutes across bench | `minutes_allocator.py` priority weights |

**Weights (carry over from current training):**

```python
λ_min   = 1.0    # Primary signal — do not reduce
λ_rates = 0.6    # Secondary to minutes
λ_eff   = 0.2    # Least certain labels (low FGA → noisy %)
λ_rot   = 0.4    # Soft rotation signal
w_gate  = 1.0    # Hard rotation signal, balanced pos_weight
w_out   = 0.25   # Gentle push (avoid overfitting to coach randomness)
w_k     = 0.05   # Regularizer, not hard constraint
w_smear = 0.05   # Regularizer, not hard constraint
```

**Why keep these when we have the flow?** The flow models a *distribution*. These
losses ensure the *center* of that distribution is accurate. Without them, the flow
could learn a well-calibrated but imprecise distribution (correct spread, wrong mean).
In DFS, the mean projection matters enormously — it's the optimizer's primary objective.

**Weight schedule across phases:**

| Phase | Tier 1 weight | Rationale |
|-------|--------------|-----------|
| Phase 1 (epochs 1-8) | 1.0× | Backbone-only, full weight |
| Phase 2 (epochs 9-18) | 0.5× | Halve as flow NLL takes over distributional learning |
| Phase 3 (epochs 19-30) | 0.25× | Keep as anchor, but flow + CRPS dominate |

The halving prevents the backbone from "fighting" the flow. If the backbone losses
remain at full weight while the flow is also training, the gradients through the shared
transformer can conflict (backbone wants sharper means; flow wants wider distributions
to capture tails). Reducing backbone weight lets the flow "have the wheel" for
distribution shape while the backbone anchors the center.

---

### 5.4 Tier 2: Distributional Losses (Flow NLL + Mixture)

#### 5.4.1 The Mixture NLL: Joint Gate + Flow Log-Likelihood

This is the core distributional loss. It jointly trains the gate head (rotation
probability) and the flow (conditional box-score distribution) through a single
principled probabilistic objective.

**The generative model:**

```
For each player i in game g:
    1. Sample rotation membership:  r_i ~ Bernoulli(gate_prob_i)
    2. If r_i = 0 (DNP):           y_i = 0  (zero box-score line)
    3. If r_i = 1 (plays):         y_i ~ Flow(z | h_i)  where z ~ N(0, I)
```

**The likelihood:**

```
p(y_i | context) = (1 - g_i) · δ(y_i = 0)  +  g_i · p_flow(y_i | h_i)
```

where `g_i = sigmoid(gate_logit_i)`.

**The log-likelihood (what we maximize):**

For a DNP player (y_i = 0):
```
log p(y_i = 0) = log(1 - g_i) = log_sigmoid(-gate_logit_i)
```

For a playing player (y_i > 0):
```
log p(y_i) = log(g_i) + log p_flow(y_i | h_i)
           = log_sigmoid(gate_logit_i) + flow_log_prob_i
```

**Implementation:**

```python
def compute_mixture_nll(
    flow_log_prob: Tensor,   # (B, P) log p_flow(y | h) for ALL players
    gate_logits: Tensor,     # (B, P) raw gate logits
    played: Tensor,          # (B, P) bool: minutes > 0
    mask: Tensor,            # (B, P) bool: valid roster slot
) -> Tensor:
    """Mixture NLL: trains gate + flow jointly.

    Numerically stable via logsigmoid (avoids log(sigmoid(x)) underflow).
    """
    # log(gate_prob) and log(1 - gate_prob), numerically stable
    log_g = F.logsigmoid(gate_logits)        # log σ(x)
    log_1mg = F.logsigmoid(-gate_logits)     # log(1 - σ(x))

    # DNP branch: -log(1 - gate_prob)
    nll_dnp = -log_1mg

    # Playing branch: -log(gate_prob) - log p_flow(y)
    nll_play = -log_g - flow_log_prob

    nll = torch.where(played, nll_play, nll_dnp)

    # Mask invalid slots and normalize
    nll = nll * mask.float()
    return nll.sum() / mask.float().sum().clamp(min=1.0)
```

**Why this is better than separate gate BCE + flow NLL:**

The separate approach (current spec draft) trains the gate via BCE and the flow via
NLL independently. But these objectives can disagree: the BCE loss says "this player
had 0 minutes, so gate should be low" while the flow says "if this player *had*
played, here's the density." The mixture NLL unifies them — if a player DNPs, the
loss only asks "was gate_prob low?" and never backprops through the flow. If a player
plays, the loss asks "was gate_prob high AND was the flow density correct?" This avoids
the pathological case where the gate learns to predict 0.5 for everyone (satisfying
BCE in expectation) while the flow overconfidently models the conditional distribution.

**We still keep `L_gate_bce` from Tier 1** as additional supervision on the gate head.
The mixture NLL gradient through the gate is relatively weak (it only sees
log(sigmoid(x))), while the BCE provides a stronger, more direct gradient with
pos_weight balancing. Think of Tier 1 gate BCE as a "prior" and the mixture NLL as the
"likelihood" — both inform the gate head, but through different channels.

#### 5.4.2 Flow NLL: What Distribution Does the Flow Model?

**Target space:** The flow models a **standardized residual space**:

```
y_flow_i = (y_raw_i - μ_stat) / σ_stat
```

where `μ_stat` and `σ_stat` are **dataset-level** (not per-context) statistics for each
of the 15 output dimensions. This standardization means:

- All dimensions have comparable scale in flow space (the flow doesn't need to learn
  that minutes has range [0, 48] while blk has range [0, 5])
- The base distribution N(0, I) is a reasonable starting point
- The flow's capacity is spent on learning the *shape* of the conditional distribution,
  not the unconditional location/scale

The autoregressive ordering within the flow is:

```
minutes → fga2 → fga3 → fta → fg2_pct → fg3_pct → ft_pct → oreb → dreb → ast → stl → blk → tov → pf → plus_minus
```

**Why this ordering matters:** MAF factors the joint density as:

```
p(y) = p(y_0) · p(y_1|y_0) · p(y_2|y_0,y_1) · ... · p(y_14|y_0,...,y_13)
```

Minutes comes first because it's the "master" variable — all counting stats scale
with it. Attempts come next because they determine the opportunity set. Efficiencies
come after their corresponding attempts (fg2_pct after fga2, etc.) because the
percentage is conditioned on having attempts. This ordering aligns the autoregressive
factorization with the causal structure of basketball statistics.

**Masking within the flow NLL:**

Not all 15 dimensions should contribute equally to every player's NLL:

| Player state | Dims contributing to NLL | Rationale |
|---|---|---|
| DNP (0 minutes) | **None** (handled by gate mixture) | No box-score data to fit |
| Played < 4 min | minutes only (dim 0) | Rate/eff labels too noisy below 4 min |
| Played ≥ 4 min | All 15 dims | Full box-score data available |

```python
# Per-dimension mask: (B, P, 15)
dim_mask = torch.ones(B, P, 15, device=device)
low_minutes = played & (y_minutes < 4.0)
dim_mask[low_minutes, 1:] = 0.0  # mask all stats except minutes
dim_mask[~played] = 0.0          # DNP: all dims masked (gate handles it)
```

The flow's NLL is computed per-dimension (each autoregressive step produces a
log-density contribution), and we apply `dim_mask` before summing.

#### 5.4.3 Soft Constraint Loss

The constraint projection layer (§3.5) enforces hard constraints at sample time.
But we also add a soft loss to encourage the flow to *naturally* produce near-feasible
samples, reducing the distortion from the projection layer.

```python
def compute_constraint_loss(
    flow_samples: Tensor,  # (K, B, P, 15) K samples per example
    team_mask: Tensor,     # (B, P) team assignment
    played_samples: Tensor,  # (K, B, P) gate samples (relaxed Bernoulli)
) -> Tensor:
    """Soft penalties for constraint violations in raw flow samples."""

    minutes_samples = flow_samples[..., 0]  # (K, B, P)
    minutes_active = minutes_samples * played_samples  # zero out DNP

    # 1. Team-240 violation: (team_total - 240)²
    #    Compute per-team sum, penalize deviation from 240
    team_totals = scatter_sum(minutes_active, team_mask)  # (K, B, T)
    L_team_240 = ((team_totals - 240.0) ** 2).mean()

    # 2. Non-negativity: all counting stats should be ≥ 0
    counting_stats = flow_samples[..., [0,1,3,5,7,8,9,10,11,12,13]]
    L_nonneg = torch.relu(-counting_stats).mean()

    # 3. Percentage bounds: fg2%, fg3%, ft% in [0, 1]
    pct_dims = flow_samples[..., [4, 5, 6]]  # fg2%, fg3%, ft%
    L_pct_bounds = (torch.relu(-pct_dims) + torch.relu(pct_dims - 1.0)).mean()

    # 4. Per-player minute cap: [0, 48]
    L_minute_cap = torch.relu(minutes_samples - 48.0).mean()

    return 0.1 * L_team_240 + 0.05 * L_nonneg + 0.05 * L_pct_bounds + 0.02 * L_minute_cap
```

**Why soft + hard?** The hard projection guarantees valid outputs. The soft loss
improves training dynamics: if the flow produces wildly infeasible samples, the
projection layer introduces large distortions, and the CRPS/NLL gradients become noisy
because they're measuring a heavily-corrected distribution. The soft loss keeps the
flow "close enough" to feasible that the projection is a minor correction, not a
major transformation.

**Weight schedule:** Start at full weight in Phase 2; reduce by 0.5× in Phase 3
(the flow should have internalized constraints by then).

---

### 5.5 Tier 3: Decision-Relevant Losses (Sample-Based)

These are the most novel and most important losses. They directly optimize what
we care about: **the quality of FPTS distributions for DFS lineup optimization**.

#### 5.5.1 CRPS on DK FPTS

**CRPS** (Continuous Ranked Probability Score) is a proper scoring rule that measures
the full distributional accuracy:

```
CRPS(F, y) = E_F[|X - y|] - ½ E_F[|X - X'|]
```

where X, X' are independent draws from the predicted distribution F and y is the
actual outcome. The first term penalizes **bias** (samples far from the truth) and
the second rewards **sharpness** (samples concentrated, not spread out). CRPS is
strictly proper — it's minimized only when F equals the true conditional distribution.

**Why CRPS on FPTS specifically (not per-stat)?**

DFS FPTS is a nonlinear function of the box-score stats:
```
FPTS = pts + 1.25*reb + 1.5*ast + 2*stl + 2*blk - 0.5*tov + DD_bonus + TD_bonus
```

Key properties of this function:
- **Heterogeneous weights**: steals and blocks are worth 2× per event; a steal is
  worth 4× a rebound in FPTS. The NLL doesn't know this.
- **Nonlinear bonuses**: The DD/TD bonuses create thresholds at exactly 10 in each
  counting category. A player with 9 rebounds and 10 rebounds has a 1.25-FPTS
  difference from the rebound plus potentially a 1.5 FPTS DD bonus — a 2.75 FPTS
  swing from one rebound. The flow needs to accurately model P(reb ≥ 10).
- **Correlation amplification**: If minutes and fg3_pct are positively correlated in
  the flow's output, the FPTS ceiling increases super-linearly. CRPS on FPTS captures
  this; per-stat NLL does not.

By computing CRPS on the *derived* FPTS (not the raw stats), we let the gradient flow
back through the DK scoring function into the flow, naturally weighting each stat
dimension by its FPTS importance.

**Implementation:**

```python
def compute_crps_fpts(
    flow: ConditionalFlow,
    h: Tensor,               # (B, P, d_model) transformer hidden states
    gate_logits: Tensor,     # (B, P) gate logits
    team_mask: Tensor,       # (B, P) team assignment
    y_fpts: Tensor,          # (B, P) actual DK FPTS
    played: Tensor,          # (B, P) bool
    mask: Tensor,            # (B, P) bool
    n_samples: int = 32,
) -> Tensor:
    """Sample-based CRPS on DraftKings FPTS.

    Uses reparameterization trick for differentiability:
      z ~ N(0,I)  →  x = flow.forward(z, context=h)  →  fpts = dk_score(x)

    Gumbel-softmax for differentiable gate sampling.
    """
    B, P = y_fpts.shape

    # 1. Sample z ~ N(0, I): shape (K, B, P, 15)
    z = torch.randn(n_samples, B, P, STAT_DIM, device=h.device)

    # 2. Transform through flow: (K, B, P, 15)
    h_expanded = h.unsqueeze(0).expand(n_samples, -1, -1, -1)
    raw_samples = flow.forward(z, context=h_expanded)

    # 3. Un-standardize: raw physical units
    samples = raw_samples * σ_stat + μ_stat  # broadcast (15,)

    # 4. Differentiable gate sampling via Gumbel-sigmoid
    gate_prob = torch.sigmoid(gate_logits)  # (B, P)
    # Straight-through estimator: sample hard 0/1 but use soft grad
    u = torch.rand(n_samples, B, P, device=h.device)
    gate_hard = (u < gate_prob.unsqueeze(0)).float()
    gate_soft = gate_prob.unsqueeze(0)
    gate_sample = gate_hard - gate_soft.detach() + gate_soft  # STE

    # 5. Zero out DNP players' samples
    samples = samples * gate_sample.unsqueeze(-1)  # (K, B, P, 15)

    # 6. Project: team-240, non-negativity, box-score consistency
    samples = project_constraints(samples, team_mask)

    # 7. Compute DK FPTS from box-score samples: (K, B, P)
    fpts_samples = compute_dk_fpts_differentiable(samples)

    # 8. CRPS = E|X-y| - 0.5 * E|X-X'|
    y_exp = y_fpts.unsqueeze(0)  # (1, B, P)

    # Term 1: mean absolute error between samples and truth
    term1 = (fpts_samples - y_exp).abs().mean(dim=0)  # (B, P)

    # Term 2: mean pairwise absolute difference between samples
    # Efficient: sort samples along K dim, use order statistic formula
    sorted_samples, _ = fpts_samples.sort(dim=0)  # (K, B, P)
    K = n_samples
    # CRPS via sorted samples: (2i - K - 1) / K^2 * x_(i)
    weights = (2 * torch.arange(1, K+1, device=h.device).float() - K - 1) / (K * K)
    term2 = (weights[:, None, None] * sorted_samples).sum(dim=0)  # (B, P)
    # term2 is actually the full CRPS via the PWM formula, so:
    # CRPS = term1 + term2 (term2 is negative, representing sharpness credit)

    crps = term1 + term2

    # Only count played players (DNP FPTS = 0 for both predicted and actual,
    # so CRPS contribution is just from the gate probability — already in Tier 2)
    crps = crps * played.float() * mask.float()
    return crps.sum() / (played.float() * mask.float()).sum().clamp(min=1.0)
```

**Computational cost:** K=32 samples × B=16 games × P=30 players × 15 stats =
~230K floats per batch. The flow forward pass is the bottleneck but is parallelizable.
Total overhead: ~3× the cost of a single forward pass (32 flow evaluations vs. 1).

**The sorted-samples CRPS formula** (Probability Weighted Moments) avoids the O(K²)
pairwise computation. The formula is:

```
CRPS = (1/K) Σᵢ |x₍ᵢ₎ - y| + (1/K²) Σᵢ (2i - K - 1) x₍ᵢ₎
```

This is O(K log K) (dominated by the sort) instead of O(K²).

#### 5.5.2 Why Not Tail-Weighted CRPS?

Standard CRPS weights all quantile levels equally. For DFS, we care more about tails
(the upside of a player determines their stacking value). We considered:

**Tail-weighted CRPS (twCRPS):**
```
twCRPS = ∫₀¹ w(τ) · 2(I(y ≤ qτ) - τ)(qτ - y) dτ
```

with w(τ) upweighting tails: w(0.05) = w(0.95) = 2.0, w(0.50) = 1.0.

**Decision:** Start with standard CRPS. If offline evaluation (§7.1) shows poor tail
calibration specifically, add tail-weighting as a tuning knob. Premature tail-weighting
can cause the flow to over-disperse (widen tails to reduce tail CRPS at the expense
of overall calibration). Better to get the full distribution right first, then adjust.

#### 5.5.3 Correlation-Aware Energy Score

CRPS is a *marginal* scoring rule — it evaluates each player's FPTS distribution
independently. But for DFS, **teammate correlations matter** (stacking decisions
depend on whether Player A and Player B are positively correlated).

The **Energy Score** is the multivariate generalization:

```
ES(F, y) = E_F[||X - y||] - ½ E_F[||X - X'||]
```

where ||·|| is the Euclidean norm over a vector of player FPTS values.

We compute this on **team-level FPTS vectors** (5-8 players per team):

```python
def compute_team_energy_score(
    fpts_samples: Tensor,   # (K, B, P) sampled FPTS per player
    y_fpts: Tensor,          # (B, P) actual FPTS
    team_mask: Tensor,       # (B, P) team assignment
    mask: Tensor,            # (B, P) valid player mask
    n_samples: int = 32,
) -> Tensor:
    """Energy score on per-team FPTS vectors.

    Captures whether the model correctly correlates teammates' performances.
    """
    # For each team in each game, gather the vector of player FPTS values
    # Compute energy score on these vectors (not individual players)
    # This rewards models that correctly produce:
    #   - High total-team FPTS in some worlds (everyone pops)
    #   - Low total-team FPTS in other worlds (team has bad game)
    # Rather than independently sampling each player
    ...
```

**Weight:** Start at 0, introduce in Phase 3 with weight 0.1. This loss is secondary
to the per-player CRPS — it fine-tunes the correlation structure.

**Why not just per-player CRPS?** Consider two models:
- Model A: correctly predicts each player's marginal FPTS distribution but samples
  independently (no correlation). Team stacks would have incorrect variance.
- Model B: correctly predicts marginal distributions AND correlations. Team stacks
  have correct variance.

Per-player CRPS cannot distinguish A and B. The energy score can.

However, the shared transformer context *should* induce most correlations naturally
(teammates share team-level representations). The energy score is a validation signal
to confirm this, and a gentle nudge if it's insufficient.

---

### 5.6 Complete Loss Formula

```
L_total(phase) = α₁(phase) · L_tier1
               + α₂(phase) · (L_nll_mixture + w_constraint · L_constraint)
               + α₃(phase) · (L_crps_fpts + w_energy · L_energy_score)
```

**Phase schedule:**

| Phase | Epochs | α₁ | α₂ | α₃ | Primary objective |
|-------|--------|-----|-----|-----|-------------------|
| **1: Backbone** | 1–8 | 1.0 | 0.0 | 0.0 | Accurate conditional means (identical to current training) |
| **2: Distribution** | 9–18 | 0.5 | 1.0 | 0.0 | Flow learns conditional density around stable means |
| **3: Decision** | 19–30 | 0.25 | 0.5 | 1.0 | FPTS distribution quality for DFS optimization |

**Component weights within tiers:**

```python
# Tier 1 (internal weights unchanged from current training)
λ_min   = 1.0;  λ_rates = 0.6;  λ_eff   = 0.2;  λ_rot = 0.4
w_gate  = 1.0;  w_out   = 0.25; w_k     = 0.05; w_smear = 0.05

# Tier 2
w_constraint = 0.1  # soft constraint penalty

# Tier 3
w_energy = 0.1  # energy score (correlation)
```

**Why reduce α₁ and α₂ when adding later tiers?**

The losses interact through the shared transformer backbone. In Phase 3, three tiers
produce gradients for the transformer parameters:
- Tier 1 wants the transformer to produce features that predict accurate means
- Tier 2 wants the transformer to produce features that condition an accurate density
- Tier 3 wants the transformer to produce features that yield good FPTS samples

These are aligned but not identical. If all three have full weight, the gradient is
dominated by whichever tier has the largest magnitude, making training noisy. The
schedule ensures each tier's contribution is proportional to its importance at that
phase, with the terminal objective (FPTS quality) getting the highest weight.

---

### 5.7 Training Stability Provisions

#### 5.7.1 Flow NLL Warmup

Normalizing flows can produce NaN gradients early in training when the flow
transformation is near-degenerate (Jacobian close to singular). Mitigations:

1. **Dequantization:** For integer-valued counting stats (pts, reb, ast, etc.), add
   uniform noise in [-0.5, 0.5] before standardization. This prevents the flow from
   trying to place point masses at integers (which produces infinite log-density).

2. **NLL clamping:** Clamp per-player NLL to [-20, 20] before averaging. This prevents
   a single extreme log-density from dominating the batch gradient:
   ```python
   nll_per_player = nll_per_player.clamp(-20.0, 20.0)
   ```

3. **Gradient scaling for flow vs. backbone:** The flow's gradients can be much larger
   than the backbone's. Use separate gradient clipping:
   ```python
   torch.nn.utils.clip_grad_norm_(backbone_params, max_norm=1.0)
   torch.nn.utils.clip_grad_norm_(flow_params, max_norm=5.0)
   ```
   Higher clip norm for the flow because flow transforms have naturally larger
   gradients (chain rule through many affine layers).

4. **Warmup within Phase 2:** Don't jump to full α₂=1.0 at epoch 9. Linear ramp
   α₂ from 0 to 1.0 over epochs 9-12 (4-epoch warmup):
   ```python
   α₂ = min(1.0, (epoch - 8) / 4)  # for epochs 9-12
   ```

#### 5.7.2 CRPS Gradient Variance

Sample-based CRPS with K=32 has non-trivial gradient variance. Mitigations:

1. **Accumulate over mini-batches:** Instead of K=32 per batch, use K=8 per batch and
   accumulate gradient across 4 micro-batches before stepping. Same effective K=32 but
   smoother gradient per step.

2. **Baseline subtraction:** For the STE gate sampling, subtract a per-player baseline
   (running mean of CRPS) to reduce variance:
   ```python
   crps_centered = crps - crps_running_mean.detach()
   ```

3. **Phase 3 uses lower learning rate:** Drop LR by 2× when entering Phase 3 to
   compensate for the noisier gradients:
   ```
   Phase 1-2: OneCycleLR with peak 1e-3
   Phase 3:   CosineAnnealingLR starting at 5e-4, decaying to 1e-5
   ```

#### 5.7.3 Abort Conditions

If training goes off the rails, these conditions trigger early stopping:

1. **Minutes MAE regression:** If val minutes MAE exceeds 1.2× the Phase 1 best by
   more than 3 consecutive epochs, abort flow training and fall back to backbone-only
   model. The backbone predictions are our safety net.

2. **NLL explosion:** If L_nll_mixture exceeds 100 for 2 consecutive batches, reduce
   α₂ by 50% and retry. If it happens again, abort.

3. **CRPS divergence:** If L_crps_fpts increases for 5 consecutive epochs in Phase 3,
   revert to Phase 2 checkpoint (flow-trained but no CRPS fine-tuning).

---

### 5.8 What Each Sim Module Is Replaced By

Here's the mapping from retired sim components to the specific loss components that
teach the model to capture that behavior:

| Sim component | What it hand-codes | Loss that learns it | Mechanism |
|---|---|---|---|
| `residuals.py` (bucketed Student-t) | Heavy-tailed FPTS noise, 6 min bins × starter/bench × injury | `L_nll_mixture` | Flow learns context-dependent tails from data; no binning needed |
| `game_factor.py` (σ=20 additive) | Cross-team within-game correlation | `L_energy_score` + cross-team attention | Transformer context induces correlation; energy score validates it |
| `game_script.py` (5 scripts, quantile shifts) | Blowout → bench plays more | `L_nll_mixture` + Vegas features | Flow conditions on spread/total; learns margin-dependent minutes variance |
| `bench_zero_mixture.py` (p_zero curves) | Low-minute players sometimes DNP | `L_nll_mixture` (gate branch) | Gate head learns P(DNP) from data; no hand-coded p_zero_base/p_zero_slope |
| `play_prob_policy.py` (15+ knobs) | Starter floors, core locks, DNP blockers | `L_gate_bce` + `L_k_reg` | Gate head learns all these patterns from supervised labels |
| `minutes_allocator.py` (Lagrange bisection) | Team-240 constraint with priority weights | `L_constraint_soft` + hard projection | Flow learns near-240 naturally; projection handles the residual |
| `minutes_noise.py` (σ_starter=2, σ_bench=3) | Per-role minutes variance | `L_nll_mixture` (minutes dim) | Flow learns heteroscedastic minutes variance conditioned on context |
| `noise.py` (team σ, player σ per rate) | Per-rate noise with team/player components | `L_nll_mixture` (rate dims) | Flow learns context-dependent rate variance with correct covariance |

---

### 5.9 Dataset

**Source:** Same `rotation_train_v1` dataset + full box-score labels.

**Key change from current dataset:** Instead of one team per example, each example is a **full game**
(both teams, up to 30 players).

```
game_example = {
    "home_features": (15, n_feat),     # home team roster features
    "away_features": (15, n_feat),     # away team roster features
    "game_context": (game_feat_dim,),  # Vegas, schedule, rest
    "home_labels_minutes": (15,),      # actual minutes home
    "away_labels_minutes": (15,),      # actual minutes away
    "home_labels_boxscore": (15, 14),  # full box score home
    "away_labels_boxscore": (15, 14),  # full box score away
    "home_mask": (15,),                # valid roster slots home
    "away_mask": (15,),                # valid roster slots away
}
```

**Box-score label columns (14):**

```python
BOXSCORE_TARGETS = [
    "fga2",    # 2-pt FGA (derived: fga - fg3a)
    "fg2_pct", # 2-pt FG% (derived: fg2m / fga2)
    "fga3",    # 3-pt FGA (direct: fg3a)
    "fg3_pct", # 3-pt FG% (direct: fg3m / fg3a)
    "fta",     # FT attempts
    "ft_pct",  # FT% (ftm / fta)
    "oreb",    # offensive rebounds
    "dreb",    # defensive rebounds
    "ast",     # assists
    "stl",     # steals
    "blk",     # blocks
    "tov",     # turnovers
    "pf",      # personal fouls
    "plus_minus",  # game +/-
]
```

With minutes as the 15th target, the flow models a 15-dimensional distribution per player.

**Temporal split:** Same policy (last 14 days as validation, remainder as training).

### 5.10 Training Schedule

```
Optimizer: AdamW, weight_decay=1e-4
Phase 1-2 LR: OneCycleLR (peak 1e-3, warmup 10%, cosine decay)
Phase 3 LR: CosineAnnealingLR (start 5e-4, end 1e-5)
Precision: bfloat16 (torch.autocast)
Batch size: 16 games (≈480 players per batch)
Gradient clipping: backbone max_norm=1.0, flow max_norm=5.0
Epochs: 30 (Phase 1: 1-8, Phase 2: 9-18, Phase 3: 19-30)
Checkpointing: Save best model per phase (separate checkpoints)
```

---

## 6. Inference & Sampling

### 6.1 Generating Worlds (Replaces sim_v2)

```python
def sample_game_worlds(
    model: GameTransformer,
    features: dict,          # game features for both teams
    n_worlds: int = 25_000,
    device: str = "cuda",
) -> np.ndarray:
    """Sample correlated box-score worlds for a full game.

    Returns: (n_worlds, n_players, 15) array of box-score samples
             where dim 0 of the last axis is minutes, dims 1-14 are stats.
    """
    model.eval()
    with torch.no_grad():
        # 1. Forward pass to get deterministic predictions + transformer hidden states
        h, pred_minutes, gate_logits, share_logits, pred_rates, pred_eff = model(features)

        # 2. Sample z ~ N(0, I) for each player in each world
        z = torch.randn(n_worlds, n_players, stat_dim, device=device)

        # 3. Transform through conditional flow
        #    (conditioned on h, pred_minutes, pred_rates per player)
        samples = model.flow.forward(z, context=h)  # (W, P, 15)

        # 4. Apply zero-inflation mask (gate sampling)
        gate_prob = torch.sigmoid(gate_logits)
        active = torch.bernoulli(gate_prob.expand(n_worlds, -1))
        samples = samples * active.unsqueeze(-1)

        # 5. Enforce team-240 constraint on minutes dimension
        samples[:, :, 0] = project_team_240(samples[:, :, 0], team_mask)

        # 6. Enforce box-score consistency
        samples = enforce_boxscore_consistency(samples)

        # 7. Compute FPTS
        fpts = compute_dk_fpts_tensor(samples)  # (W, P)

    return samples.cpu().numpy(), fpts.cpu().numpy()
```

**Batching:** 25K worlds at ~30 players × 15 stats = ~11M floats per game.
At float32, that's ~44MB per game. With 8 games on a slate, ~350MB total — fits
comfortably in GPU memory. Sample in one forward pass, no batching needed.

### 6.2 Output Contract (Drop-in Replacement)

The sampler produces exactly the same output schema as today's sim_v2:

```python
# Per-player quantiles (same as sim_v2/worlds_summary.py)
output_columns = {
    # FPTS quantiles (unconditional)
    "dk_fpts_mean", "dk_fpts_std",
    "dk_fpts_p05", "dk_fpts_p10", "dk_fpts_p25",
    "dk_fpts_p50", "dk_fpts_p75", "dk_fpts_p90", "dk_fpts_p95",
    # FPTS quantiles (unconditional, DNP=0)
    "dk_fpts_mean_uncond", "dk_fpts_std_uncond", ...,
    # Minutes sim quantiles
    "minutes_sim_mean", "minutes_sim_std",
    "minutes_sim_p10", "minutes_sim_p50", "minutes_sim_p90",
    # Box score means
    "pts_mean", "reb_mean", "ast_mean", "stl_mean", "blk_mean", "tov_mean",
    # Sim diagnostics
    "sim_p_active", "sim_p_rotation",
}
```

This means the **optimizer, dashboard, and finalize_projections.py all work unchanged**.

---

## 7. Evaluation Plan

### 7.1 Offline Metrics (vs. sim_v2 baseline)

| Metric | What it measures | Target |
|--------|-----------------|--------|
| **Minutes MAE** | Point prediction accuracy | ≤ current (4.5 min MAE) |
| **Minutes calibration** | P(actual ≤ predicted_p10) ≈ 10%? | Each quantile within 2% |
| **FPTS CRPS** | Full distribution accuracy | < sim_v2 CRPS |
| **FPTS tail calibration** | P(actual > predicted_p90) ≈ 10%? | Each quantile within 2% |
| **Correlation capture** | Pearson(sampled_teammate_fpts) vs actual | Closer to actual than sim_v2 |
| **Team-total accuracy** | sampled team FPTS total vs actual | Lower RMSE than sim_v2 |
| **Zero-inflation calibration** | predicted DNP rate vs actual DNP rate | Within 3% |
| **Double-double rate** | frequency of DD in samples vs actual | Within 5% |

### 7.2 Online A/B (DFS Backtest)

Run the optimizer on both:
- **Control:** Current sim_v2 pipeline → optimizer
- **Treatment:** Game transformer → optimizer

Compare:
- **ROI** on historical DFS slates
- **Lineup diversity** (entropy of ownership-weighted returns)
- **Tail capture** (how often the model's ceiling players actually hit)

### 7.3 Diagnostic Visualizations

1. **Flow samples vs. actuals** scatter plots (per stat, per minutes bucket)
2. **Correlation heatmap** of sampled stats (does the model learn ast↔tov correlation?)
3. **Team-total distribution** (should be tight around 240 for minutes)
4. **Gate calibration curve** (predicted rotation prob vs. actual rotation rate)
5. **Per-quantile reliability diagram** (p10/p50/p90 for minutes and FPTS)

---

## 8. Risk Analysis & Mitigations

### 8.1 Technical Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Flow training instability (NaN, mode collapse) | High | Phase 1 warm-start; gradient clipping; flow depth tuning. Start with simple affine flows, add complexity only if needed. |
| Flow underestimates tails (learned distribution too tight) | Medium | Evaluate CRPS tail components explicitly. Can add tail-aware loss (weighted NLL on extreme residuals). |
| Cross-team attention adds complexity but no lift | Low | Ablation: train with/without cross-team tokens. If no lift, fall back to per-team transformer (already works). |
| Overfitting with full game context | Medium | ~3 seasons × ~1230 games × 2 teams = ~7400 team-games. Same order as current training set. Dropout + weight decay should suffice. |
| Slow sampling (flow inversion) | Low | MAF forward pass is parallel (fast). 25K worlds × 30 players should be <5 sec on GPU. |

### 8.2 Product Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Model produces unrealistic box scores | Medium | Constraint projection layer guarantees feasibility. Sanity-check assertions at inference time. |
| Regression on minutes accuracy (our best signal) | High | Phase 1 warm-start ensures minutes head converges first. If minutes MAE regresses, abort flow training and use deterministic heads only. |
| Loss of manual override capability | Low | GameView overrides can still be applied post-sampling (same as today's effective_minutes layer). |

---

## 9. Implementation Roadmap

### Phase 1: Foundation (P0)

1. **Game-level dataset builder** — extend `build_joint_rotation_rates_dataset_v1.py`
   to produce game-level examples with both teams and full box-score labels
2. **Game transformer module** — `projections/rotation/game_transformer_v1.py`
   with cross-team attention, reusing feature projection from `set_model.py`
3. **Deterministic training** — train the game transformer with existing supervised
   losses (no flow yet). Validate that cross-team attention doesn't hurt minutes MAE.
4. **Evaluation harness** — build calibration + CRPS evaluation for distributions

### Phase 2: Flow Integration (P0)

5. **Conditional flow module** — `projections/rotation/conditional_flow.py`
   implementing MAF conditioned on transformer hidden states
6. **Flow training loop** — extend training script with NLL loss, curriculum schedule
7. **Constraint projection** — differentiable team-240 and box-score consistency
8. **Sampling pipeline** — `projections/rotation/sample_worlds.py` replacing
   `sim_v2/generate_worlds_fpts_v2.py`

### Phase 3: Validation & Integration (P1)

9. **Offline evaluation** — full evaluation suite (§7.1) comparing to sim_v2 baseline
10. **Output adapter** — produce same parquet schema as sim_v2 for downstream compatibility
11. **DFS backtest** — run optimizer on historical slates (§7.2)
12. **Live pipeline integration** — add as alternative sim backend in Prefect flow,
    gated by config flag

### Phase 4: Production (P2)

13. **Shadow mode** — run alongside sim_v2, log diagnostics, compare live
14. **Gradual rollover** — switch optimizer input from sim_v2 → game transformer
15. **Deprecate sim_v2** — remove after sustained positive results

---

## 10. Alternatives Considered

### 10.1 Diffusion Model Instead of Normalizing Flow

**Pros:** More expressive; better at complex multimodal distributions.
**Cons:** Requires iterative denoising (slow sampling); harder to get exact log-likelihood
for training. NBA box scores aren't highly multimodal — the main complexity is
correlation and tails, which flows handle well.

**Verdict:** Start with flow (simpler, faster sampling). Revisit diffusion if flow
can't capture the distribution complexity.

### 10.2 Quantile Regression Network (No Generative Model)

**Pros:** Much simpler; just predict p10/p25/p50/p75/p90 per stat.
**Cons:** No joint sampling — you can't draw correlated worlds from independent quantiles.
This is the fundamental limitation of the current approach (independent minutes quantiles +
independent rate noise + hand-coded game factor).

**Verdict:** Rejected. Joint sampling is the whole point.

### 10.3 Copula-Based Approach

**Pros:** Well-understood statistical framework; explicit correlation modeling.
**Cons:** Parametric copulas (Gaussian, t) may not capture the complex dependency
structure of NBA stats. Vine copulas are more flexible but harder to fit and sample from.

**Verdict:** The conditional flow is essentially a learned copula with unlimited
flexibility. If we find the flow is overkill, we could simplify to a Gaussian copula
conditioned on the transformer output (simpler but less expressive).

### 10.4 VAE Instead of Flow

**Pros:** More forgiving to train (ELBO objective is smoother).
**Cons:** Posterior collapse risk; blurry samples (underdispersion); no exact
log-likelihood. For DFS optimization, we need sharp, calibrated tails — VAE's
tendency to blur is exactly the wrong failure mode.

**Verdict:** Rejected for this use case.

### 10.5 Keep Sim, Replace Components Piecemeal

**Pros:** Lower risk; incremental improvement.
**Cons:** The fundamental issue is that hand-coded correlation structure and noise
distributions can't be validated or improved systematically. Every knob tuning is
a guess. The sim has ~15 modules with ~50+ configurable parameters — the interaction
effects are untestable.

**Verdict:** This is the status quo. The spec proposes replacing it wholesale, but
the phased approach (§9) provides natural off-ramps if the game transformer doesn't
deliver.

---

## 11. Open Questions

1. **Flow architecture choice:** MAF vs. Neural Spline Flow vs. RealNVP? NSF may give
   better tail behavior with fewer layers. Worth ablating.

2. **Correlation modeling scope:** Should the flow model cross-player correlations
   (one big joint distribution over all 30 players × 15 stats = 450 dims)? Or per-player
   conditional on team context (15 dims per player, with correlation induced via shared
   transformer context)? The per-player approach is much more tractable and the shared
   context should capture most correlations.

3. **Training data volume:** ~3 seasons × ~1230 games = ~3690 games. Is this enough
   for a 450-dim flow? The per-player factorization (15 dims, conditioned on context)
   gives us ~3690 × 30 ≈ 110K player-game examples for the flow, which should be
   sufficient.

4. **Handling of blowouts:** The current sim has explicit game-script logic (blowout
   → bench plays more). Should we add explicit game-script tokens, or trust the
   transformer to learn this from Vegas spread features? Probably start with features
   only and add explicit tokens if calibration suffers for extreme spreads.

5. **Props integration:** Action Network props provide strong signals for individual
   stat lines. Should they condition the flow directly (as features), or only the
   backbone (as today)? Probably both — they're a form of market information about the
   distribution shape.

---

## 12. Summary

Replace the fragile 15-module Monte Carlo simulation with a single learned model:

**Input:** Game features (both teams, 30 players + context)

**Model:** Game-level set transformer → conditional normalizing flow

**Output:** 25K correlated box-score samples per game, constrained to valid NBA outcomes

**Key advantage:** The model learns correlations, tails, and constraints from data instead
of relying on 50+ hand-tuned parameters. The flow's learned distribution is validated via
log-likelihood, not eyeballed.

**Migration path:** Drop-in replacement for sim_v2's output contract — everything downstream
(optimizer, dashboard, finalize) works unchanged.
