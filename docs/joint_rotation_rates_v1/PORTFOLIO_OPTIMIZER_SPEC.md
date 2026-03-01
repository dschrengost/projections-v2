# Portfolio Optimizer: Hardening and Production Spec

## Spec Status: DRAFT v0.1 (2026-03-01)

---

## 1. Motivation

### 1.1 Current gap

The repo currently has three distinct layers for multi-entry lineup construction:

1. candidate generation in QuickBuild / optimizer
2. lineup evaluation in contest sim
3. final-set / portfolio selection in UI helpers and an experimental backend portfolio module

That split is directionally correct, but the current portfolio-selection layer is not yet
a hardened production system.

Current issues:

1. the main contest-sim UI still relies on local selection helpers rather than the
   backend portfolio optimizer as the canonical selector
2. the backend portfolio optimizer is still experimental in API and behavior
3. contract details are underspecified:
   - what objective is authoritative
   - whether min exposure is supported or not
   - whether selection must use train/holdout world splits
   - whether live selection must use `gtv2` worlds only
4. candidate generation and portfolio selection are not yet treated as one coherent
   end-to-end system

### 1.2 What this spec changes

This spec makes the following hard decisions:

1. treat portfolio optimization as a post-contest-sim decision layer, not a replacement
   for contest sim
2. make `gtv2` worlds the default and authoritative live selection source
3. define a single backend portfolio optimizer contract and route UI selection through it
4. harden the logic around exposure bounds, metric handling, diagnostics, and
   train/holdout separation
5. allow a substantial rewrite of the current experimental module if that is the
   cleanest path

### 1.3 Why this matters

For multi-entry DFS, "top lineups by EV" is not enough.

A production portfolio optimizer must solve two separate problems:

1. preserve enough expected value / upside
2. avoid selecting lineups that all fail together

The correct place to solve that is after contest sim has already scored candidate lineups
against a realistic field and a realistic worlds distribution.

---

## 2. Scope

In scope:

1. backend portfolio-selection logic
2. authoritative selection objectives and constraints
3. live-vs-backtest worlds-source policy
4. API and saved-build contract for portfolio selection
5. UI integration for final-set / portfolio construction
6. diagnostics and hardening tests

Out of scope for this spec:

1. replacing QuickBuild candidate generation entirely
2. changing the game-transformer model itself
3. redesigning contest payout modeling
4. operator boosts / nerfs beyond how they appear in candidate inputs

---

## 3. Design Principles

1. Contest sim is the truth source for lineup quality.
   Portfolio optimization consumes contest-sim outputs; it does not invent a parallel
   notion of EV.

2. Live selection must be explicit about worlds source.
   Silent fallback between `gtv2` and `sim_v2` is unacceptable for live decision-making.

3. Constraints must be honest.
   If the system exposes min/max exposure, uniqueness, or duplication controls,
   those controls must either be enforced or rejected. "Accepted but ignored" is not allowed.

4. Train/selection leakage must be controlled.
   A portfolio should not be tuned solely on the exact same worlds used to produce
   its reported decision metrics without an explicit policy.

5. Diagnostics are first-class.
   The selector must explain why a portfolio was chosen:
   EV retained, risk reduced, exposures, overlap, and worlds source must all be visible.

6. The product should support simple and advanced modes.
   Most users need a robust default. Advanced users may still want risk sliders,
   weighted allocations, or explicit constraints.

---

## 4. Current-State Audit

### 4.1 Candidate generation

Current QuickBuild behavior:

1. generates lineups from mean projections or one sampled world at a time
2. does not solve a true multi-world contest objective
3. should be treated as candidate generation only

### 4.2 Contest sim

Contest sim already performs the true multi-world evaluation step:

1. scores lineups across all worlds
2. compares them against a field
3. computes EV / ROI / win rate / cash rate / tail metrics / dupe-adjusted metrics

That is the correct scoring layer to feed the portfolio optimizer.

### 4.3 Existing backend portfolio optimizer

`projections/contest_sim/portfolio_optimizer.py` already contains:

1. a greedy constrained selector
2. a covariance-penalized de-correlation selector

But it is still experimental because:

1. it is not the main UI authority
2. the public contract advertises `min` and `max` exposure while only `max` is enforced
3. missing metric handling is underspecified for ascending sorts
4. weighted duplicate allocation across contests is not implemented in the active path
5. train/holdout split behavior exists only as a low-level option, not a production policy

### 4.4 Existing UI selection

The contest-sim page currently uses local selection heuristics for:

1. top-N filtering
2. min-uniques filtering
3. exposure-bound filtering
4. final-set selection
5. "set and forget" construction

This creates a contract split between backend and frontend selection logic.

That split should be removed.

---

## 5. Product Goals

### 5.1 MVP production goal

Given a candidate lineup pool that has already been scored by contest sim, choose a final
portfolio that:

1. retains most of the best available contest-sim EV
2. materially lowers portfolio concentration / covariance
3. respects explicit exposure and uniqueness constraints
4. is reproducible and auditable

### 5.2 Desired user modes

The product should support three modes:

1. `greedy_constraints`
   - rank by a chosen contest-sim metric
   - apply min uniques / max ownership / exposure bounds
   - simple and fast baseline

2. `decorrelated_ev`
   - start from best-EV feasible set
   - reduce covariance subject to EV retention floor
   - default recommended portfolio mode

3. `weighted_allocations`
   - integer weights over fewer unique lineups
   - useful for larger entry counts / duplicate-allowed strategies
   - later phase, but included in the contract now

---

## 6. Canonical Inputs

The portfolio optimizer consumes:

### 6.1 Candidate lineups

For each lineup:

1. `lineup_id`
2. `player_ids`
3. optional metadata:
   - total ownership
   - stack tags
   - game/team exposures
   - originating build id

### 6.2 Contest-sim metrics

For each lineup:

1. `expected_value`
2. `roi`
3. `win_rate`
4. `top_1pct_rate`
5. `cash_rate`
6. `tail_score`
7. `select_score`
8. `robust_floor`

### 6.3 World-level inputs

For de-correlation / risk-aware modes:

1. `worlds_matrix`
2. `player_index`
3. optional explicit `world_indices`
4. optional train/holdout split config

### 6.4 World source contract

Allowed values:

1. `gtv2`
2. `sim_v2`
3. `auto` (backtest/debug only, not live default)

Policy:

1. live product surfaces default to `gtv2`
2. if `gtv2` is requested and unavailable, fail loudly
3. `sim_v2` may be used only by explicit request for backtests / comparisons

---

## 7. Canonical Outputs

The selector returns:

### 7.1 Selected portfolio

1. ordered selected lineup ids
2. selected lineup payloads
3. optional per-lineup integer weights

### 7.2 Diagnostics

At minimum:

1. `mode`
2. `sort_key`
3. `sort_dir`
4. `worlds_source`
5. `worlds_used`
6. `world_selection_policy`
7. `candidate_count`
8. `portfolio_size`
9. `ev_best`
10. `ev_target`
11. `ev_selected`
12. `risk_var_total_baseline`
13. `risk_var_total_selected`
14. `risk_var_total_reduction_pct`
15. `passes`
16. `swaps_made`
17. exposure summary
18. overlap summary

### 7.3 Saved-build metadata

Saved portfolio builds must record:

1. source contest-sim build id(s)
2. run id
3. worlds source
4. worlds run id if available
5. selection config
6. diagnostics summary

---

## 8. Objective Logic

## 8.1 Baseline constrained portfolio

The baseline selector is greedy and deterministic:

1. filter invalid candidates
2. order by the selected metric
3. apply hard constraints during selection:
   - min uniques
   - max total ownership if requested
   - player exposure caps
4. stop when `portfolio_size` is reached

This mode is useful as:

1. fallback behavior
2. explainable baseline
3. comparison point for more advanced selectors

## 8.2 Decorrelated EV mode

This is the primary production selector.

Given:

1. candidate EV vector `EV_i`
2. player/world score matrix from the worlds source
3. a retained-EV threshold `ev_retention`

Algorithm:

1. choose the best-EV feasible baseline portfolio
2. compute player covariance from selected worlds
3. represent each lineup as a player indicator vector
4. iteratively swap lineups to reduce total portfolio variance while maintaining:
   - `EV(selected) >= ev_retention * EV(best_feasible)`
   - exposure caps
   - any other active hard constraints

This mode explicitly trades a small amount of EV for lower portfolio covariance.

## 8.3 Weighted allocation mode

This mode is allowed in the final architecture and should be planned now, even if
not implemented in Phase 1.

Decision object:

1. lineup selection
2. integer weight per lineup

Use cases:

1. 20-max / 150-max entries where controlled duplication is acceptable
2. contest-specific allocation later through entry manager

This mode should not block the initial hardening release.

---

## 9. Constraint Logic

## 9.1 Exposure bounds

Exposure semantics must be explicit:

1. `max` exposure is a hard cap
2. `min` exposure is either:
   - fully enforced, or
   - rejected as unsupported

Current "accept `min` but ignore it" behavior is not acceptable.

Recommended Phase 1 decision:

1. support `max` only in the hardened backend
2. reject non-null `min` with a validation error
3. add `min` in a later phase only if we truly need it

## 9.2 Min uniques

Min uniques should remain a hard optional guardrail.

Policy:

1. enforce in greedy baseline mode
2. optional in decorrelated mode; if enabled, it must be enforced explicitly
3. if a requested combination is infeasible, return a structured constraint-exhaustion error

## 9.3 Metric validity

Missing metrics must never silently dominate selection order.

Rules:

1. for descending sorts, missing values sort to the bottom
2. for ascending sorts, missing values also sort to the bottom
3. if the chosen metric is missing for too many candidates, surface a warning

## 9.4 Finite-input requirement

Portfolio selection should drop or reject:

1. lineups with non-finite primary sort metric
2. lineups with no mapped players in `player_index` when decorrelated mode is requested
3. worlds inputs with fewer than 2 usable worlds

---

## 10. World-Split Policy

### 10.1 Problem

If portfolio optimization uses the same exact worlds both to choose and to report
the final portfolio, the selector can overfit the evaluation sample.

### 10.2 Production policy

For risk-aware modes, one of the following must be used:

1. explicit `world_indices` train subset, or
2. deterministic `worlds_train_frac` split

Holdout evaluation should be supported in diagnostics.

### 10.3 Recommended Phase 1 default

For live:

1. use deterministic train subset with `worlds_train_frac` in `(0, 1)`
2. report selection on train worlds
3. optionally compute holdout diagnostics on the complement

The exact fraction can be tuned later, but the policy itself should be explicit.

---

## 11. Candidate-Set Policy

The portfolio optimizer is not responsible for generating all diversity by itself.

Candidate quality depends on upstream generation.

Production guidance:

1. merge multiple candidate build families before selection:
   - mean objective
   - world-sample objective
   - randomness variants
   - constraint variants
2. de-duplicate lineups before portfolio selection
3. label candidate origin in diagnostics

The selector should assume candidate diversity exists, but it must also expose
when the candidate set is too homogeneous:

1. overlap histogram
2. player exposure concentration before selection
3. variance-reduction ceiling diagnostics

---

## 12. API Contract

## 12.1 New backend selector API

Add a dedicated service / endpoint layer rather than keeping all logic embedded in the page.

Suggested request fields:

1. `game_date`
2. `draft_group_id`
3. `source_build_id`
4. `mode`
5. `worlds_source`
6. `sort_key`
7. `sort_dir`
8. `portfolio_size`
9. `ev_retention`
10. `worlds_sample`
11. `worlds_train_frac`
12. `min_uniques`
13. `max_total_own`
14. `exposure_bounds`
15. `allow_weighted_duplicates`

Suggested response fields:

1. selected lineups
2. weights if any
3. diagnostics
4. warnings / constraint exhaustion info

## 12.2 UI contract

The contest-sim page should stop re-implementing core selection logic locally.

UI responsibilities:

1. collect config
2. send request to backend selector
3. display diagnostics
4. allow manual include/exclude edits on top of the selected set

Backend responsibilities:

1. execute selection logic
2. validate config
3. enforce constraints
4. return diagnostics

---

## 13. Implementation Plan

### 13.1 Phase 1: Harden the existing backend selector

Files:

1. `projections/contest_sim/portfolio_optimizer.py`
2. `tests/contest_sim/test_portfolio_optimizer.py`

Tasks:

1. fix missing-value sort behavior for ascending sorts
2. make exposure-bound behavior honest:
   - reject unsupported min bounds, or
   - fully implement them
3. add explicit config / result schemas for selector modes
4. strengthen diagnostics object and serialization
5. add tests for:
   - ascending sort with missing values
   - rejected min exposure
   - deterministic train-split behavior
   - worlds-source diagnostics propagation

### 13.2 Phase 2: Add a dedicated selection service/API

Files:

1. `projections/api/contest_sim_api.py`
2. `projections/contest_sim/portfolio_optimizer.py`
3. new service helper if needed under `projections/contest_sim/`

Tasks:

1. add portfolio-selection request/response models
2. expose `mode`, `worlds_source`, and train-split config explicitly
3. consume saved contest-sim builds as candidate input
4. save selected-portfolio builds with diagnostics

### 13.3 Phase 3: Replace local UI heuristics

Files:

1. `web/minutes-dashboard/src/pages/ContestSimPage.tsx`
2. optional new API client in `web/minutes-dashboard/src/api/contest_sim.ts`

Tasks:

1. remove local duplicate selection logic as authoritative path
2. call backend selector for final-set construction
3. display diagnostics:
   - EV retained
   - risk reduction
   - worlds source
   - train/holdout policy
4. keep manual include/exclude as a thin post-selector layer

### 13.4 Phase 4: Weighted allocations and entry-manager integration

Files:

1. `projections/contest_sim/portfolio_optimizer.py`
2. `projections/api/entry_manager_api.py`
3. `web/minutes-dashboard/src/pages/ContestSimPage.tsx`
4. `web/minutes-dashboard/src/pages/EntryManagerPage.tsx`

Tasks:

1. support integer lineup weights
2. support contest-aware allocation later if needed
3. emit allocation-ready output for entry manager

---

## 14. Hardening Requirements

The hardened selector must satisfy all of the following:

1. deterministic given:
   - candidate set
   - worlds source
   - worlds split
   - seed
   - config
2. explicit failure on missing required worlds
3. explicit failure or warning on infeasible constraints
4. no silent ignoring of requested constraints
5. no frontend/backend divergence in authoritative selection logic
6. saved outputs include enough metadata to reconstruct what happened

---

## 15. Success Criteria

The hardened portfolio optimizer is acceptable when:

1. it is the authoritative final-set selector for contest sim
2. live selection uses `gtv2` worlds by default and surfaces that fact in diagnostics
3. selection logic is no longer duplicated in frontend-only heuristics
4. historical replay shows:
   - similar or slightly lower EV than best-EV greedy baseline
   - materially lower concentration / covariance
   - more stable downside behavior
5. tests cover the main contract edges

---

## 16. Recommended Immediate Decisions

To avoid ambiguity, adopt these decisions now:

1. `decorrelated_ev` becomes the default production portfolio mode
2. `gtv2` is the default live worlds source
3. `min` exposure is rejected until implemented properly
4. train/holdout split is required for risk-aware selection in production
5. the frontend final-set logic is transitional and should be replaced by the backend selector

---

## 17. File-by-File Task List

### Backend logic

- [ ] `projections/contest_sim/portfolio_optimizer.py`
  - harden sort semantics
  - harden exposure semantics
  - formalize selector modes and diagnostics
  - add train/holdout policy handling

### API

- [ ] `projections/api/contest_sim_api.py`
  - add portfolio selection endpoint
  - save/load selected portfolio builds

### Frontend

- [ ] `web/minutes-dashboard/src/pages/ContestSimPage.tsx`
  - replace local final-set logic with backend call
  - surface diagnostics and warnings

### Tests

- [ ] `tests/contest_sim/test_portfolio_optimizer.py`
  - extend for hardening cases
- [ ] new API tests under `tests/api/`
  - request validation
  - worlds source propagation
  - diagnostics persistence

---

## 18. Final Position

The right long-term architecture is:

```
candidate generation -> contest sim -> portfolio optimizer -> entry manager
```

not:

```
candidate generation -> ad hoc UI heuristics -> entry manager
```

The current experimental backend module is a reasonable starting point, but it should
now be treated as a production hardening project, not as a side utility.
