# World-Keyed Lineup Generation

## Spec Status: DRAFT v0.1 (2026-03-02)

---

## 1. Motivation

The current lineup generation pipeline has a structural clustering problem.
QuickBuild workers sample random worlds per solve cycle and optimize under
each world's FPTS vector, but the downstream portfolio pipeline selects
lineups by expected value — which collapses back to mean-projection
dominance.  The decorrelated portfolio selector (`build_decorrelated_portfolio`)
can reduce variance by swapping correlated lineups, but it can only choose
from candidates that already exist in the pool.  Since the pool is generated
by a single multi-cycle optimizer targeting many lineups per worker, the
candidate set clusters heavily around high-mean players regardless of
world-sampling or diversity constraints.

**The goal:** generate lineups that are *optimal for specific game-flow
scenarios*, so the portfolio naturally reflects diverse outcome states rather
than converging on the mean.

### 1.1 Concrete Example (2026-02-28 Slate)

On the 2/28 five-game slate, the GTV2 produced 25,000 correlated worlds
across 150 players.  The top-8-player ceiling (unconstrained by salary/
position) varies from 335 to 528 DK FPTS across worlds — a 57% spread
from floor to ceiling.  Yet QuickBuild-generated portfolios cluster within a
narrow band of player exposures because the pool is filtered on mean EV.

Players like Jeremiah Fears (actual 43.25, predicted mean 24.77, 96th
percentile of world distribution) and Jakob Poeltl (actual 44.0, predicted
27.3, 94th percentile) delivered ceiling games that would appear in
world-keyed lineups optimized for those specific scenarios but would be
underweighted in mean-EV portfolios.

---

## 2. Design Principles

1. **One lineup per world.**  Each selected world produces exactly one
   salary-legal optimal lineup.  No jitter, no streaming, no multi-cycle
   exploration within a world — the solver finds the deterministic best.
2. **Scenario coverage over mean optimization.**  The portfolio's diversity
   comes from the worlds themselves, not from post-hoc decorrelation of a
   mean-clustered pool.
3. **Robustness signal from redundancy.**  When many worlds produce the same
   (or overlapping) optimal lineup, that lineup is robust.  The count of
   worlds where a lineup is optimal is a first-class metric.
4. **Composable with existing portfolio selection.**  The decorrelation
   layer remains available as an optional fine-tuning step, not a required
   crutch.
5. **Minimal solver overhead.**  Each per-world solve is a single-shot
   CP-SAT optimization with a short timeout — parallelizable and fast.

---

## 3. Architecture Overview

```
┌──────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│  World Selection  │────>│  Per-World Solver    │────>│  Dedup & Candidate  │
│  (N ⊂ 25k worlds)│     │  (N × CP-SAT solve) │     │  Construction       │
└──────────────────┘     └─────────────────────┘     └─────────────────────┘
                                                              │
                                                              v
                                                     ┌─────────────────────┐
                                                     │  Portfolio Selection │
                                                     │  (existing layer)   │
                                                     └─────────────────────┘
```

### 3.1 Data Flow

1. Load `PlayerWorlds` from GTV2 artifacts (same path as today).
2. Select N world indices via configurable strategy (§4).
3. For each selected world, solve the DK salary-legal 8-player lineup that
   maximizes that world's FPTS vector.  Parallel across workers.
4. Deduplicate optimal lineups (exact + near-dup by Jaccard), recording
   per-lineup world frequency counts and objective values.
5. Score deduplicated lineups against the full worlds matrix for distribution
   stats (mean, p10, p90, ceiling_upside — same as today's
   `compute_lineup_distribution_stats`).
6. Optionally pass candidates to `build_decorrelated_portfolio` or
   `build_portfolio` for final selection.

### 3.2 New Module

`projections/optimizer/world_keyed_build.py`

Depends on:
- `projections/optimizer/cpsat_solver.py` — `build_cpsat_model`, `build_cpsat_counts`
- `projections/optimizer/quick_build.py` — reuses `_build_spec_from_payload` and `InMemoryPool`
- `projections/contest_sim/contest_sim_service.py` — `load_player_worlds`

### 3.3 Integration Points

| Component                  | Change                                             |
|----------------------------|----------------------------------------------------|
| `optimizer_service.py`     | New build mode `world_keyed` alongside `quick_build` |
| `optimizer_api.py`         | New request field `build_mode: Literal["quick_build", "world_keyed"]` |
| `portfolio_optimizer.py`   | Accept `world_count` as a candidate metric          |
| `contest_sim_api.py`       | Portfolio endpoint accepts world-keyed builds       |
| Frontend (OptimizerPage)   | Toggle for build mode, world count config           |

---

## 4. World Selection Strategies

The GTV2 generates W=25,000 worlds per slate.  We select N << W worlds to
optimize against.  The selection strategy determines portfolio character.

### 4.1 Uniform Random (v0 — default)

Sample N world indices uniformly without replacement.  Simple, unbiased,
sufficient for initial deployment.

**Config:** `world_selection: "uniform"`, `n_worlds: 200`

### 4.2 Stratified by Game Total

Bin worlds by total-slate FPTS into Q quantiles.  Sample N/Q from each bin.
Ensures representation of high-scoring and low-scoring environments.

**Config:** `world_selection: "stratified_total"`, `n_worlds: 200`, `n_strata: 10`

### 4.3 K-Means on Game-Level Features

Cluster worlds by a feature vector (per-game total pts, spread, pace proxy).
Sample one world per cluster (nearest to centroid).

**Config:** `world_selection: "kmeans"`, `n_clusters: 200`

### 4.4 Enriched Tail Sampling

Over-sample from the tails of the game-total distribution.  Useful for
GPP contests where ceiling lineups matter disproportionately.

**Config:** `world_selection: "enriched_tails"`, `n_worlds: 200`, `tail_weight: 2.0`

### 4.5 Recommended Starting Point

Uniform random with N=200.  Revisit after initial evaluation shows whether
tail enrichment or stratification meaningfully changes portfolio quality.

---

## 5. Per-World Solver

### 5.1 Model Construction

Build the CP-SAT model once (constraints only, no objective) via
`build_cpsat_model(spec, optimize=False)`.  The model encodes:

- DK salary cap ($50,000)
- Position eligibility (PG, SG, SF, PF, C, G, F, UTIL)
- 8 roster slots
- Player availability (exclude locked/out players)
- Optional: game stacking rules, max-per-team, min-teams

### 5.2 Per-World Objective

For world index `w_i`, set objective:

```
Maximize  Σ_j  fpts_matrix[w_i, j] × x_j × SCALE
```

where `x_j ∈ {0, 1}` is the player selection variable and SCALE=1000 for
integer precision.

Then solve with a tight timeout (0.5–1.0s).  The solver returns the optimal
(or best-found) lineup for that world.

### 5.3 Parallelization

Distribute N world solves across `min(N, cpu_count)` worker processes.
Each worker receives a batch of world indices, builds its own model copy,
and iterates through its assigned worlds — swapping the objective
coefficients between solves (same pattern as the current `_worker_main`
world-sampling loop, but one solve per world rather than streaming).

**Expected performance:**

| N worlds | Workers | Solves/worker | Time/solve | Wall time |
|----------|---------|---------------|------------|-----------|
| 200      | 8       | 25            | 0.5s       | ~12.5s    |
| 500      | 16      | 31            | 0.5s       | ~15.5s    |

### 5.4 Tie-Breaking

When multiple lineups tie for optimal under a world (common with integer
FPTS values after scaling), use deterministic jitter ε ∈ (0, 0.001) seeded
by `(world_idx, player_id)` to break ties consistently.

---

## 6. Deduplication and Candidate Construction

### 6.1 Exact Dedup

Many worlds will produce identical optimal lineups — especially worlds where
the same high-projection player set dominates.  Collapse identical lineups
and record:

- `world_count`: number of worlds where this lineup was optimal
- `world_indices`: which worlds produced it
- `best_objective`: highest world-level objective among producing worlds
- `worst_objective`: lowest (indicates robustness of the optimal status)

### 6.2 Near-Dup Bucketing (Jaccard)

Lineups differing by 1 player (Jaccard ~0.75 for 8-player lineups) may
represent minor substitutions.  Group near-dups and keep the
representative with the highest `world_count`.

**Threshold:** Jaccard ≥ 0.75 (same as current `near_dup_jaccard` default).

### 6.3 Expected Yield

From N=200 world solves on a 5-game slate:
- Raw lineups: 200
- After exact dedup: ~80–120 unique (highly dependent on slate depth)
- After near-dup: ~40–80

On deeper slates (8+ games, more viable players), uniqueness will be higher.

### 6.4 Candidate Metrics

Each deduplicated lineup becomes a `PortfolioCandidate` with:

| Metric            | Source                                                  |
|-------------------|---------------------------------------------------------|
| `expected_value`  | Mean lineup FPTS across all W worlds (same as today)     |
| `world_count`     | Worlds where this lineup was optimal (new)               |
| `world_frequency` | `world_count / N` — normalized robustness (new)          |
| `mean`, `p90`     | From `compute_lineup_distribution_stats` (existing)      |
| `total_own`       | Sum of ownership projections (existing)                  |
| `ceiling_upside`  | `p90 - mean` (existing)                                  |

---

## 7. Portfolio Selection

### 7.1 Greedy by World Frequency

Sort by `world_frequency` descending, greedily select respecting min_uniques
and exposure caps.  Lineups that are optimal in more worlds are prioritized
— they are robust across scenarios.

### 7.2 Decorrelated Selection (Existing)

Feed world-keyed candidates to `build_decorrelated_portfolio`.  The
covariance matrix is still computed from the full worlds matrix.  The key
difference: the candidate pool already contains structurally diverse lineups
(ceiling-game lineups, blowout lineups, pace-variance lineups) that the
current mean-clustered pool lacks.

### 7.3 Hybrid: World Frequency + Decorrelation

Use `world_frequency` as the EV metric in decorrelated selection.  The
`ev_retention` parameter then controls: "how much robustness am I willing
to sacrifice for lower portfolio variance?"

### 7.4 When Decorrelation Becomes Optional

If the world-keyed pool is sufficiently diverse (>60 unique lineups from
200 solves), simple greedy selection with min_uniques may produce portfolios
with adequate variance reduction — making the decorrelation swap pass
unnecessary.  This should be evaluated empirically.

---

## 8. Calibration Audit (2026-02-28 Slate)

Baseline calibration of the GTV2 worlds that will feed the world-keyed
optimizer.  Five-game slate (NOP/CHA, LAL/HOU, MIA/POR, TOR/GSW, UTA/WAS),
25,000 worlds, 150 players.

### 8.1 Player-Level FPTS Accuracy

| Metric      | Value   |
|-------------|---------|
| Mean error  | +1.09   |
| MAE         | 7.13    |
| RMSE        | 9.18    |
| Correlation | 0.747   |

Bias is slightly positive (+1.1 FPTS) — the model slightly underpredicts on
average for active players.

### 8.2 Percentile Calibration

For the top 40 predicted players (active), where does their actual FPTS
fall in the 25,000-world distribution?

| Threshold   | Fraction below | Ideal |
|-------------|----------------|-------|
| p10         | 5%             | 10%   |
| p25         | 15%            | 25%   |
| p50         | 45%            | 50%   |
| p75         | 82%            | 75%   |
| p90         | 92%            | 90%   |

**Interpretation:** The distribution is moderately well-calibrated at the
median but slightly overconfident in the right tail — too many actuals land
in the 50th–75th percentile range (82% below p75 vs ideal 75%).  This means
the model's upside distribution is somewhat compressed: it places too much
mass in the "good but not great" region and not enough in the true ceiling.

This is relevant for world-keyed builds because the ceiling-world lineups
depend on accurate tail modeling.  If the p90 is too conservative, the
"ceiling" lineups won't be extreme enough.

### 8.3 Notable Outliers

| Player            | Actual | Predicted | Error   | Percentile | Notes                         |
|-------------------|--------|-----------|---------|------------|-------------------------------|
| Jeremiah Fears    | 43.25  | 24.77     | +18.48  | 96.0%      | Breakout game, model low      |
| Jakob Poeltl      | 44.00  | 27.31     | +16.69  | 94.0%      | Double-double, model low      |
| Immanuel Quickley | 48.00  | 32.19     | +15.81  | 91.7%      | Career-type game              |
| Zion Williamson   | 8.25   | 41.61     | −33.36  | 0.4%       | Injury exit, model had high   |
| Alperen Sengun    | 27.00  | 47.99     | −21.00  | 5.6%       | Off night, well below floor   |
| Scoot Henderson   | 18.25  | 30.27     | −12.02  | 13.8%      | Limited role game             |
| Austin Reaves     | 24.25  | 36.59     | −12.34  | 15.1%      | Quiet game                    |

Zion at 0.4th percentile suggests the model's play_prob / injury model
didn't capture his in-game exit risk.

### 8.4 Team Total Points

| Team | Actual | Predicted | Error  |
|------|--------|-----------|--------|
| NOP  | 115    | 115.3     | −0.3   |
| GSW  | 101    | 115.0     | −14.0  |
| HOU  | 105    | 116.2     | −11.2  |
| LAL  | 129    | 114.6     | +14.4  |
| MIA  | 115    | 115.7     | −0.7   |
| POR  | 93     | 113.4     | −20.4  |
| TOR  | 134    | 116.5     | +17.5  |
| UTA  | 105    | 115.4     | −10.4  |
| WAS  | 125    | 112.8     | +12.2  |
| CHA  | 109    | 114.3     | −5.3   |

**Key finding:** Team total predictions cluster around 113–117 regardless of
the actual game (range 93–134).  The worlds spread (σ ≈ 23.5 pts per game)
captures *some* game-to-game variance, but the **means are too
homogeneous** — the model doesn't condition strongly enough on Vegas
lines/totals to differentiate a POR (93 actual) from a TOR (134 actual).

This directly impacts world-keyed builds: if the model's game-total
distribution doesn't discriminate between high-scoring and low-scoring game
environments, the "high-scoring world" and "low-scoring world" lineups will
differ less than they should.

### 8.5 Correlation Structure

| Property                          | Value               |
|-----------------------------------|---------------------|
| Cross-game pts correlation        | r ≈ 0.001 (none)    |
| Within-team player FPTS corr      | r ≈ −0.05 (slight)  |
| World ceiling (top-8 sum) mean    | 418.9 ± 29.6        |
| World ceiling range               | [335.4, 527.8]      |

Cross-game independence is correct by construction (games are independent
entities in the GTV2 forward pass).  The slight negative within-team
correlation reflects the minutes/usage sharing constraint.

The ceiling range (335–528) shows meaningful world diversity exists.
World-keyed builds should capture this range.

### 8.6 Calibration Implications for World-Keyed Builds

1. **Compressed right tail:** The p75 overcount (82% vs 75%) suggests the
   model's high-end scenarios are too conservative.  World-keyed ceiling
   lineups will be slightly less extreme than reality.  This is acceptable
   for v0 — addressing it requires GTV2 model changes, not optimizer changes.

2. **Homogeneous game totals:** The mean game total varies by only ~3 points
   across 5 very different games.  Stratifying worlds by game total will
   have less effect than expected because the model's game-environment
   conditioning is weak.  This is a GTV2 improvement area.

3. **Active-rate calibration is good:** DNP players have low active_rate
   (0.111 avg) and low predicted FPTS (0.80).  The model is appropriately
   cautious about bench players.

4. **Negative within-team correlation is healthy:** World-keyed lineups
   should naturally produce game stacks (when a world has a high-scoring
   game, multiple players from those teams get elevated FPTS), and the
   negative within-team correlation ensures those stacks don't just load
   all five starters from one team.

---

## 9. Ownership Integration

### 9.1 Current State

Ownership projections exist in the player pool (`own_proj` column from
LineStarr or model-based predictions).  The current QuickBuild applies
an optional ownership penalty to the objective.

### 9.2 World-Keyed Approach

For world-keyed builds, ownership handling has two options:

**Option A: Pure scenario optimization (recommended for v0).**
Each world solve maximizes FPTS only, ignoring ownership.  Ownership enters
at portfolio selection: prefer lineups with lower total ownership when
choosing between candidates of similar world-frequency.

**Option B: Ownership-penalized per-world objective.**
Apply the existing `_calculate_ownership_penalty_term` to each per-world
objective.  This biases every world's optimal lineup toward contrarian
plays, potentially missing the "correct" optimal lineup for that scenario.

Option A is cleaner because it separates "what's the best lineup if this
scenario happens?" from "which scenarios should I weight in my portfolio?"

---

## 10. Failure Modes and Mitigations

### 10.1 Low Unique Lineup Yield

**Risk:** On shallow slates (2–3 games), most worlds produce the same
optimal lineup because the player pool is small.

**Mitigation:** Fall back to QuickBuild if unique count < `min_unique_threshold`
(default: 30).  Add a small per-player jitter (ε=0.1 FPTS) to break ties
when the raw yield is too low.

**Diagnostic:** `unique_lineup_count / n_worlds_solved` ratio.  Healthy: >0.3.
Degenerate: <0.1.

### 10.2 Solver Timeouts Under Extreme Worlds

**Risk:** Some worlds have degenerate FPTS vectors (bench player at 80 FPTS)
that make the solver explore pathological branches.

**Mitigation:** Hard timeout of 1.0s per solve.  Accept best-found solution
even if not proven optimal.  Log timeout rate as a diagnostic.

### 10.3 Stale Worlds

**Risk:** GTV2 worlds were generated hours before lock.  Lineup news changes
player availability after world generation.

**Mitigation:** Before solving, mask out players flagged as OUT/DTD in the
latest injury report.  Their world-FPTS values become 0 for the solve.
This is the same approach as the current pipeline.

### 10.4 Correlated World Selection Bias

**Risk:** If world selection accidentally over-samples a particular game
scenario (e.g., all sampled worlds have Game A as a blowout), the portfolio
will be skewed toward that scenario.

**Mitigation:** Stratified sampling (§4.2) or just N≥200 with uniform
sampling (law of large numbers).

### 10.5 Loss of "In-Between" Lineups

**Risk:** Lineups that are mediocre in every individual world but good *on
average* (the current mean-optimal lineups) won't appear in the world-keyed
pool.

**Mitigation:** Optionally include K mean-optimized lineups (K=5–10) as
"anchor" candidates alongside the world-keyed set.  These serve as a
baseline the portfolio can fall back to.

---

## 11. Implementation Roadmap

### Phase 1: Core Per-World Solver

- New module: `projections/optimizer/world_keyed_build.py`
- `WorldKeyedConfig` dataclass (n_worlds, selection strategy, timeout, jitter)
- `solve_world_keyed_lineups(spec, worlds_matrix, player_index, config)` →
  `WorldKeyedBuildResult`
- Multiprocess parallelization via `multiprocessing.Pool`
- Unit tests with synthetic worlds

### Phase 2: Dedup and Candidate Pipeline

- Exact + Jaccard near-dedup
- `PortfolioCandidate` construction with `world_count` metric
- Integration with `compute_lineup_distribution_stats`
- Diagnostic summary (unique rate, timeout rate, world coverage)

### Phase 3: API and Service Integration

- `optimizer_service.py`: new `run_world_keyed_build` function
- `optimizer_api.py`: `build_mode` field in `QuickBuildRequest`
- Wire through job store, progress reporting, build saving

### Phase 4: Portfolio Selection Adaptation

- Add `world_frequency` sort key to `build_portfolio`
- Test `build_decorrelated_portfolio` with world-keyed candidates
- Evaluate whether decorrelation adds value over simple greedy

### Phase 5: Frontend and Evaluation

- OptimizerPage toggle between QuickBuild and World-Keyed
- World selection config controls
- A/B evaluation framework: generate both QuickBuild and World-Keyed
  portfolios for the same slate, compare simulated ROI

---

## 12. Open Questions

1. **Optimal N?**  200 is a starting guess.  Need to empirically measure the
   unique yield curve (unique lineups as a function of N) across different
   slate sizes.

2. **Should worlds be pre-filtered?**  E.g., exclude worlds where a player
   known to be OUT still appears active.  Currently the solve handles this
   by zeroing out those players, but pre-filtering would reduce wasted
   solves.

3. **World-keyed builds for late swap?**  The per-world solver could
   incorporate locked-player constraints for live contests.  This is
   natural but adds complexity (different constraint sets per lineup slot
   state).

4. **Interaction with strategy overrides?**  Strategy overrides currently
   adjust the worlds matrix before QuickBuild.  They should similarly
   adjust worlds before per-world solves, using the same
   `apply_strategy_overrides_to_worlds` path.

---

## 13. Summary

World-keyed lineup generation replaces the "generate massive pool then
select by EV" paradigm with "generate one optimal lineup per scenario then
select by scenario coverage."  The approach is:

- **Simpler** at the solver level (deterministic single-shot vs multi-cycle streaming)
- **Faster** for modest N (200 solves × 0.5s ≈ 12s vs current 30–60s QuickBuild)
- **More diverse** by construction (each lineup is optimal for a different world)
- **Composable** with existing portfolio selection and ownership layers

The main dependency is GTV2 world quality — calibration audit (§8) shows
the worlds are usable but have compressed tails and homogeneous game totals
that limit scenario differentiation.  These are GTV2 model issues, not
optimizer issues, and can be addressed independently.
