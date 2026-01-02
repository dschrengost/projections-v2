# DFS Portfolio Optimizer (NBA DK) — Implementation Plan

## Goal
Treat multi-entry DFS as a portfolio: give up a little EV to materially reduce downside risk by de-correlating entries while retaining upside.

This runs **after contest sim** (true EV inputs), and supports:
- NBA DraftKings Classic only
- Any multi-entry GPP (up to 150 per contest)
- Multiple contests per slate (any total number of entries)
- Two portfolio modes:
  - **Unique**: each entry is a unique lineup
  - **Weighted**: allocate integer weights across fewer unique lineups (duplicate lineups allowed)

## What we optimize
We choose a set of lineups (and optionally weights) across contests that:
1) has high expected profit (true EV from contest sim), and
2) has better left-tail outcomes (smaller losing nights) than “top EV lineups independently”.

## Inputs (data contract)
### Contest list
For each contest `c`:
- `contest_id`, `contest_name`
- `entry_fee`
- `field_size`
- `entry_max` (max entries allowed)
- payout definition (archetype or explicit tiers)
- `N_c` entries to allocate (from entry manager / user choice)

### Candidate lineups
For each lineup `i`:
- `lineup_id`
- `player_ids` (8 DK classic slots)
- optional tags: game stack, team stack, total_own, salary, etc.

### Contest sim results (true EV)
For each `(contest c, lineup i)`:
- `expected_value` (expected_payout − entry_fee)
- `roi`, `win_rate`, `top_1pct_rate`, etc.
- dupe metrics if available (`dupe_penalty`, `adjusted_expected_payout`)

### Scenario data (for true risk optimization)
To optimize downside directly we need per-world profits:
- `profit[c, i, w]`: profit per *single entry* of lineup `i` in contest `c` under world `w`
  - `profit = payout - entry_fee`
  - Worlds come from `sim_v2` player outcome simulations
  - Field modeling should match contest sim (weighted field library + payout tiers)

If scenario capture is too heavy for large candidate sets, we will compute it only for a reduced set `K` (see “Candidate set strategy”).

## Outputs
Portfolio allocation:
- per contest: list of `(lineup_id, weight)` pairs, where `weight` is integer entries
- option to enforce global uniqueness across all contests or allow repeats across contests

Diagnostics:
- portfolio EV / ROI (sum over contests)
- downside summary from worlds: mean, P10/P50/P90, worst-1%/5%, `P(loss)`
- exposure tables by player/team/game/stack
- overlap/correlation proxies (distribution of shared-player overlap, concentration metrics)

## Core optimization model
Let contests `c ∈ C`, candidate lineups `i ∈ {1..K}`.

Decision variables:
- `w[c,i] ∈ Z_{≥0}`: number of entries of lineup `i` in contest `c`

Hard constraints:
- **Per contest entries**: `Σ_i w[c,i] = N_c`
- **Entry max**: `N_c ≤ entry_max_c` (validated up front)
- **Unique vs weighted mode**:
  - unique-per-contest: `w[c,i] ∈ {0,1}`
  - global unique: `Σ_c w[c,i] ≤ 1`
  - max duplicates per lineup: `Σ_c w[c,i] ≤ d_i` (toggle; default `d_i = N_total`)
- **Exposure caps (global)**:
  - for each player `p`: `Σ_{c,i} w[c,i] * 1[p ∈ lineup_i] ≤ cap_p * N_total`
  - similarly for team/game/stack exposures (configurable)

Objective options (choose per phase):
1) **Baseline EV**: maximize `Σ_{c,i} w[c,i] * EV[c,i]`
2) **EV floor + diversification** (fast, linear):
   - enforce `EV(portfolio) ≥ (1−ε) * EV_best`, then minimize:
     - max player exposure, or
     - total exposure deviation from targets, or
     - overlap proxy penalties (see below)
3) **Mean–CVaR (recommended)**:
   - per world `w`: portfolio profit `P[w] = Σ_{c,i} w[c,i] * profit[c,i,w]`
   - maximize `E[P] − λ * CVaR_α(loss)` where `loss[w] = −P[w]`
   - standard linearization:
     - variables `t` (VaR) and `u[w] ≥ 0`
     - `u[w] ≥ loss[w] − t`
     - `CVaR_α(loss) = t + (1/(αW)) * Σ_w u[w]`

Notes:
- Scale currency to cents (integers) for solver stability.
- A practical interface is a “risk slider” over `λ` (or an EV floor `ε`) and `α` (tail fraction).

## Correlation / diversification proxies (when scenario capture isn’t available)
Even with true EV, we still need to avoid “all lineups share the same fate”.
We can do that with fast, explainable constraints/penalties:
- max player exposure, max team exposure, max game exposure
- minimum lineup uniqueness (max shared players) against already-selected lineups (greedy) or via clustering
- limit “core” size: number of players above X% exposure

This is a good MVP and can coexist with CVaR (still useful as guardrails).

## Candidate set strategy (keep the solve tractable)
Contest sim may score 5k–50k lineups; CVaR needs scenario profits which are expensive to materialize.

Practical two-stage approach:
1) Start from contest sim results; keep a superset `M`.
2) Reduce to `K` (e.g., 500–3000) via:
   - top EV cutoff
   - include some “upside” lineups (high win/top1% rates)
   - cluster by lineup composition / stack tags and keep top EV per cluster
3) Materialize `profit[c,i,w]` only for the reduced `K`.

## Solver plan
Use OR-Tools (already in repo):
- Phase 1 (MVP): linear/integer diversification model with CP-SAT or MIP
- Phase 2 (CVaR): MILP with many world constraints; prefer OR-Tools MIP (`pywraplp`) if CP-SAT struggles

## Integration points (where this plugs in)
- **Contest sim** provides `EV[c,i]` and (Phase 2) `profit[c,i,w]`.
- **Entry manager** supplies `N_c` per contest and consumes portfolio allocations to fill entries and export DK CSV.

## Phased roadmap
### Phase 0 — Plumbing
- Define portfolio config + result schemas (contests, allocation mode, caps, risk params).
- Load contest sim saved builds and candidate lineup libraries.

### Phase 1 — Usable MVP (fast)
- Optimize EV subject to exposure caps + uniqueness toggles.
- Add an “EV floor” mode to explicitly trade EV for diversification.
- Emit allocation + exposure report.

### Phase 2 — True downside optimization (CVaR)
- Extend contest sim to optionally emit per-world user profits for candidate lineups (reduced `K`).
- Implement mean–CVaR optimization with `α`/`λ` controls.
- Add “efficient frontier” generation (vary `λ` or `ε`) and pick a point.

### Phase 3 — Product wiring
- FastAPI endpoint(s) + UI:
  - choose contests + entry counts
  - choose mode (unique/weighted)
  - choose risk slider / EV floor
  - preview portfolio distribution + exposures
  - apply to entry manager + export

## Success criteria
On historical slates:
- EV decreases slightly (e.g., ≤ 1–3%) while worst 5% outcomes improve materially (smaller losing nights).
- Exposure concentration decreases (lower max player/team exposure, fewer “all-in” cores).
- Runtime stays practical:
  - Phase 1: < 1s for large candidate sets
  - Phase 2: interactive or async for `K ≈ 500–1500`, `W ≈ 5k–10k`

