# Late Swap V2 Spec

This document is the living redesign spec for late swap in `projections-v2`.

It replaces the old "per-entry optimize and auto-pick best projection" model
with a portfolio-aware late swap system that treats late swap as one of the
highest-EV operator workflows in NBA DFS.

It complements, but does not replace:

- `docs/late-swap/late-swap.md`
- `docs/entry-manager/entry-manager-plan.md`
- `docs/optimizer/portfolio_optimizer_plan.md`
- `docs/joint_rotation_rates_v1/PORTFOLIO_OPTIMIZER_SPEC.md`

## Implementation Status (2026-03-09)

This spec is now partially implemented end-to-end in production code.

### Implemented in this revision

1. New backend package `projections/late_swap/` with:
   - `models.py`
   - `session_store.py`
   - `lock_state.py`
   - `candidate_generation.py`
   - `scoring.py`
   - `portfolio_selector.py`
   - `diagnostics.py`
2. New sessionized API endpoints:
   - `POST /api/entry-manager/late-swap/sessions`
   - `GET /api/entry-manager/late-swap/sessions?date=...`
   - `GET /api/entry-manager/late-swap/sessions/{session_id}`
   - `POST /api/entry-manager/late-swap/sessions/{session_id}/preview`
   - `POST /api/entry-manager/late-swap/sessions/{session_id}/pin-candidates`
   - `POST /api/entry-manager/late-swap/sessions/{session_id}/policy`
   - `POST /api/entry-manager/late-swap/sessions/{session_id}/commit`
   - `POST /api/entry-manager/late-swap/sessions/{session_id}/export`
3. Legacy compatibility path updated:
   - `POST /api/entry-manager/entries/{contest_id}/late-swap` now runs through
     session create -> preview -> auto-commit (legacy behavior compatibility).
4. `EntryFileState` lineage fields added:
   - `source_late_swap_session_id`
   - `source_late_swap_mode`
   - `source_late_swap_committed_at`
5. Frontend Late Swap workbench added:
   - `web/minutes-dashboard/src/pages/LateSwapPage.tsx`
   - `web/minutes-dashboard/src/api/late_swap.ts`
   - `web/minutes-dashboard/src/components/lateSwap/*`
   - New top-level app tab/route: `late-swap`
6. Backend tests added for:
   - session lifecycle (`create -> preview -> pin -> commit -> export`)
   - lock-floor cap infeasibility surfaced in diagnostics

### Follow-on work (not in this revision)

1. Full remaining-slate simulation-backed candidate scoring.
2. Contest-position-aware objective variants (requires live contest state feed).
3. Additional frontend UX polish for segmented policies and advanced candidate
   rebuild controls.

## 1. Purpose

Build a late swap system that:

1. preserves the operator's pre-lock portfolio thesis when possible
2. respects DraftKings slot-lock semantics exactly
3. controls exposure drift across all affected entries and contests
4. provides a robust preview -> review -> commit workflow
5. supports progressively better scoring, up to remaining-slate simulation and
   contest-aware modes

The target architecture should make late swap a first-class product surface, not
just a thin endpoint attached to Entry Manager.

## 2. Current Problems

The current implementation is directionally useful but structurally limited:

1. late swap is still fundamentally per-entry
2. alternatives are generated per entry and the best projected alternative is
   auto-applied
3. selected contests are processed independently instead of as one portfolio
4. there is no hard or soft exposure control layer
5. there is no target-preservation concept tied to the source portfolio build
6. there is no preview session with first-class diagnostics before commit
7. `only_out_lineups` is too narrow for real late-news decision-making

That creates the main failure mode we want to eliminate:

- late news opens the same value for many entries
- each entry independently converges to the same best local solve
- final exposures spike far beyond the operator's intended risk

## 3. Goals

### 3.1 Core product goals

1. select late swaps across one or more contests as a grouped portfolio problem
2. support explicit player exposure bounds before and after swap
3. preserve original target exposures by default when entries came from a
   portfolio build
4. provide deterministic, explainable diagnostics for why a swap set was chosen
5. keep the current lineup as a valid "hold" candidate for every entry
6. allow low-friction manual intervention without destroying portfolio controls

### 3.2 Backend goals

1. move late swap logic out of one large API handler into dedicated modules
2. persist swap sessions and lineage for audit/debugging
3. separate candidate generation from portfolio selection
4. reuse existing portfolio selection semantics where appropriate
5. keep scale comfortably within the expected max of about 300 entries

### 3.3 Frontend goals

1. promote late swap to a dedicated operator workbench
2. make exposure state visible before and after swap
3. make infeasibility obvious, especially lock-driven overexposure
4. allow preview without mutating saved entry files
5. support batch review and per-entry drilldown

## 4. Non-goals

1. direct automation against DraftKings web actions
2. guaranteed ingestion of live DK contest standings in V1
3. replacing the entire optimizer stack
4. solving pre-lock MME generation from scratch inside late swap
5. multi-user collaborative editing in the first release

## 5. Design Principles

### 5.1 Portfolio-first, not entry-first

Late swap should choose one candidate per entry only after evaluating the full
set of entries together.

### 5.2 Locks create floors

If a player is already locked in 42% of entries, that 42% is an immutable floor.
Any requested cap below that is infeasible and must be surfaced explicitly.

### 5.3 Hard caps and soft targets are different

- Hard caps are do-not-exceed constraints unless locks already violate them.
- Soft targets express portfolio intent and should be optimized toward, not
  treated as binary pass/fail rules.

### 5.4 Current lineup is always a candidate

The system must never force a swap simply because a new solve exists. "Hold"
must remain a valid option, and swap cost should be part of the decision layer.

### 5.5 Candidate diversity matters

A selector cannot fix exposure if every candidate still contains the same chalk.
Alternative generation must actively create cap-relief and target-recovery
options.

### 5.6 Preview before commit

The operator should be able to inspect exposures, warnings, and per-entry diffs
before any saved entry file is mutated.

### 5.7 Repeated runs are normal

Late swap is a repeated workflow across lock windows. Sessions, targets, and
diagnostics must support reruns cleanly.

## 6. Final Product Position

The correct architecture is:

```text
entry states
  -> lock detection
  -> per-entry candidate generation
  -> grouped portfolio selection
  -> preview diagnostics
  -> explicit commit/export
```

not:

```text
entry states
  -> per-entry optimize
  -> auto-pick best local projection
  -> hope exposures stay sane
```

This is a redesign of the late swap selection layer, not a full rewrite of the
lineup solver.

## 7. Product Modes

The system should ship with explicit late swap modes, not hidden numeric knobs.

### 7.1 Recommended modes

1. `preserve_targets`
   - default
   - maximize score while minimizing deviation from target exposures
   - moderate swap-cost penalty
   - good for most MME late swap runs

2. `best_ev`
   - maximize projected or remaining-slate EV under hard caps
   - weakest target-preservation penalty
   - useful when operator wants pure re-optimization with guardrails

3. `decorrelated_ev`
   - maximize EV subject to hard caps, then reduce portfolio covariance/overlap
   - recommended for larger MME sets

4. `catch_up`
   - more willing to move off chalk
   - more weight on leverage/upside among remaining players
   - weaker swap-cost penalty

5. `block`
   - tighter target preservation
   - higher swap-cost penalty
   - more conservative selection for protecting strong early positions

### 7.2 Manual mode

`manual_review` is not a scoring mode. It is a workflow mode that:

1. generates candidates
2. computes exposure diagnostics
3. recommends a selected portfolio
4. lets the operator lock specific candidate choices before re-solving the rest

## 8. Core Architecture

### 8.1 Major stages

1. create late swap session
2. load one or more entry files and consolidate them into one swap universe
3. refresh live player pool and draftables
4. detect lock state per slot
5. compute baseline exposure state
6. generate per-entry alternatives
7. score alternatives
8. run grouped selector
9. surface preview diagnostics
10. commit selected alternatives to entry files
11. export updated CSVs

### 8.2 Separation of concerns

Backend logic should be split into dedicated modules:

- `projections/late_swap/models.py`
- `projections/late_swap/session_store.py`
- `projections/late_swap/lock_state.py`
- `projections/late_swap/candidate_generation.py`
- `projections/late_swap/scoring.py`
- `projections/late_swap/portfolio_selector.py`
- `projections/late_swap/diagnostics.py`

The API layer should orchestrate these modules, not contain the full algorithm.

## 9. Data Model

### 9.1 Existing objects to keep

Keep `EntryFileState` as the source of saved entry file truth.

Keep existing lineage fields such as:

- `source_build_source`
- `source_build_id`
- `source_build_kind`
- `source_build_name`
- `source_portfolio_build_id`
- `source_run_build_id`
- `source_selection_mode`

Those are important because they let late swap infer the original portfolio
intent when available.

### 9.2 New persisted object: `LateSwapSession`

Persist sessions under:

```text
$PROJECTIONS_DATA_ROOT/late_swap/{date}/dk/session={session_id}/
```

Session payload:

```python
class LateSwapSession(BaseModel):
    session_id: str
    game_date: str
    site: str = "dk"
    contest_ids: list[str]
    draft_group_ids: list[int]
    created_at: str
    updated_at: str
    status: Literal["draft", "preview_ready", "committed", "stale", "failed"]
    source_entry_revisions: dict[str, int]
    source_profile: LateSwapSourceProfile
    policy: LateSwapPolicy
    lock_state: LateSwapLockStateSummary
    candidate_summary: LateSwapCandidateSummary
    selection_summary: LateSwapSelectionSummary | None
    diagnostics: LateSwapDiagnostics
    selected_candidates_by_entry_id: dict[str, str]
    pinned_candidates_by_entry_id: dict[str, str]
    warnings: list[str]
```

### 9.3 New object: `LateSwapPolicy`

```python
class LateSwapPolicy(BaseModel):
    mode: Literal[
        "preserve_targets",
        "best_ev",
        "decorrelated_ev",
        "catch_up",
        "block",
        "manual_review",
    ] = "preserve_targets"
    target_source: Literal[
        "source_portfolio",
        "current_entries",
        "explicit",
        "none",
    ] = "source_portfolio"
    exposure_bounds: dict[str, ExposureBoundsPct] = {}
    team_exposure_bounds: dict[str, ExposureBoundsPct] = {}
    game_exposure_bounds: dict[str, ExposureBoundsPct] = {}
    min_uniques: int = 0
    max_duplicate_lineups: int = 1
    candidate_count_per_entry: int = 10
    max_swaps_per_entry: int | None = None
    max_total_swaps: int | None = None
    min_gain_to_swap: float = 0.0
    swap_cost_lambda: float = 0.0
    target_deviation_lambda: float = 0.0
    overlap_penalty_lambda: float = 0.0
    ownership_penalty_lambda: float = 0.0
    leverage_boost_lambda: float = 0.0
    segment_mode: Literal["global", "by_contest", "by_tag"] = "global"
    segment_overrides: dict[str, SegmentPolicyOverride] = {}
    rerun_anchor: Literal[
        "source_portfolio",
        "last_committed",
        "session_start",
    ] = "source_portfolio"
```

### 9.4 New object: `LateSwapCandidate`

Each candidate should be a complete proposed final lineup for one entry.

```python
class LateSwapCandidate(BaseModel):
    candidate_id: str
    entry_id: str
    contest_id: str
    locked_slots: list[str]
    slot_values: dict[str, str]
    player_ids: list[str]
    unlocked_player_ids: list[str]
    generated_by: Literal[
        "hold",
        "best_ev",
        "randomized",
        "cap_relief",
        "target_recovery",
        "low_own",
        "manual",
    ]
    projected_score: float | None
    remaining_score_mean: float | None
    remaining_score_p90: float | None
    expected_value: float | None
    roi: float | None
    total_own: float | None
    swap_count: int
    added_player_ids: list[str]
    removed_player_ids: list[str]
    tags: list[str]
    reason_codes: list[str]
```

### 9.5 Exposure state objects

The UI and backend should reason about four exposure states:

1. `source_target`
   - the original exposure intent from the source portfolio or explicit policy
2. `locked_floor`
   - immutable current exposure from already-locked players
3. `current_committed`
   - the currently saved entry exposure state before preview
4. `proposed_final`
   - the exposure state implied by the current selected preview

This four-way split is essential. It prevents confusion between "what I wanted",
"what is already locked", "what is saved now", and "what this preview would do".

## 10. Exposure Semantics

### 10.1 Hard caps

Hard caps apply to final selected entries.

For player `p`:

```text
final_count[p] <= cap_count[p]
```

unless:

```text
locked_floor[p] > cap_count[p]
```

In that case the cap is infeasible. The system must:

1. flag the player as `forced_over_cap_by_locks`
2. prevent adding more exposure unless the operator explicitly allows it
3. report the minimum achievable final exposure

### 10.2 Hard mins

Hard mins are allowed, but must be validated against:

1. available candidate coverage
2. remaining unlocked entries
3. lock state

If a requested min is impossible, the preview should fail clearly rather than
pretend the policy was respected.

### 10.3 Soft targets

Targets are optimization anchors, not hard feasibility rules.

For player `p`, target deviation is:

```text
abs(final_count[p] - target_count[p])
```

This should be represented with linear deviation variables so the selector can
trade small deviations against EV and swap cost.

### 10.4 Default target source

Default target precedence:

1. explicit operator targets
2. source portfolio build exposures
3. current committed entry exposures
4. no targets

## 11. Candidate Generation

### 11.1 Required behavior

For every entry:

1. detect locked slots
2. compute the set of unlockable slots
3. include the current lineup as a `hold` candidate
4. generate additional candidates under live pool and lock constraints
5. deduplicate candidate lineups
6. ensure every candidate is DK-assignable and export-safe

### 11.2 Candidate generation passes

Candidate generation should not be a single call with `N_lineups=K`.
It should be a multi-pass process designed to produce useful diversity.

Recommended passes:

1. `hold`
   - exact current lineup

2. `best_ev`
   - strongest remaining-lineup solve under current constraints

3. `randomized`
   - additional solves with randomness / jitter

4. `cap_relief`
   - targeted solves banning players currently trending over cap

5. `target_recovery`
   - targeted solves boosting or requiring players below target when feasible

6. `low_own`
   - lower-ownership or higher-leverage solves for `catch_up`

7. `limited_swap`
   - optional pass restricting max swaps from current lineup to 1, 2, or 3

### 11.3 Why targeted generation is mandatory

If a late value piece projects as the best play by a wide margin, plain top-K
generation will still overproduce candidates containing that player. Exposure
control then looks broken even though the selector had no real alternatives.

Targeted generation is what gives the selector actual room to diversify.

### 11.4 Candidate counts

Recommended defaults:

- default `candidate_count_per_entry = 10`
- allow `6..20`
- allow targeted expansion for entries whose candidates do not provide enough
  feasible exposure coverage

### 11.5 Candidate diagnostics

Per entry, record:

- number of candidates requested
- number generated
- number deduplicated away
- number rejected as unassignable
- number rejected as salary invalid
- number rejected by swap-limit filters
- whether coverage for constrained players was adequate

## 12. Candidate Scoring

### 12.1 Baseline scoring

Every candidate should have at least:

- projected remaining score
- total projected ownership
- swap count
- added and removed players

### 12.2 Recommended scoring layer

The recommended long-term scoring layer is remaining-slate simulation:

1. use world-based remaining player outcomes when available
2. derive candidate-level mean, p90, EV, ROI, or tail metrics
3. feed those into grouped portfolio selection

This allows late swap to optimize for more than raw projected points.

### 12.3 V1 scoring fallback

If remaining-slate sim data is not available, use:

1. projected remaining points
2. ownership and leverage proxies
3. overlap penalties
4. swap cost

This is weaker than simulation-backed scoring but still much stronger than pure
per-entry best projection.

### 12.4 Scoring by mode

Mode defaults:

1. `preserve_targets`
   - score on projected remaining points or EV
   - moderate target penalty
   - moderate swap-cost penalty

2. `best_ev`
   - strongest score weight on EV/projection
   - low target penalty
   - low swap-cost penalty

3. `decorrelated_ev`
   - same baseline score as `best_ev`
   - followed by decorrelation repair/search

4. `catch_up`
   - more weight on upside, leverage, and reduced ownership
   - less penalty for larger swaps

5. `block`
   - more weight on median/floor
   - stronger target and swap-cost penalties

## 13. Grouped Portfolio Selector

### 13.1 Problem definition

Late swap selection is a grouped assignment problem.

For each entry `e`, choose exactly one candidate `c` from the candidate set
`C_e`.

Binary decision variable:

```text
x[e,c] in {0,1}
```

Required constraint:

```text
sum_c x[e,c] = 1 for each entry e
```

### 13.2 Hard constraints

The selector must support:

1. exact one-candidate-per-entry selection
2. player min/max exposure bounds
3. team exposure bounds
4. game exposure bounds
5. duplicate-lineup count limit
6. optional max swaps per entry
7. optional max total swaps

### 13.3 Overlap control

Overlap is important but expensive to encode exactly across all candidate pairs.
Implementation should be two-stage:

1. baseline CP-SAT assignment with linear hard constraints
2. optional local-search repair that reduces overlap/covariance while preserving
   the baseline EV threshold and all hard constraints

This mirrors the existing portfolio optimizer architecture and is a better fit
than forcing a large pairwise MIP from day one.

### 13.4 Objective

Baseline linear objective:

```text
maximize
  sum(score[e,c] * x[e,c])
  - lambda_target * total_target_deviation
  - lambda_swap * total_swap_count
```

Optional second-stage decorrelation:

1. compute baseline feasible portfolio
2. enforce EV retention threshold
3. iteratively swap candidate choices entry by entry to reduce portfolio
   covariance / overlap

### 13.5 Selector outputs

Return:

1. selected candidate per entry
2. unselected alternatives for drilldown
3. exposure summary before and after
4. infeasibility warnings
5. policy used
6. objective diagnostics
7. constraint-relaxation or fallback notes if any

## 14. Session Lifecycle

### 14.1 Create session

Operator chooses one or more contests and creates a session. This does not
mutate entry files.

### 14.2 Build preview

Backend:

1. refreshes live pool
2. computes locks
3. generates candidates
4. runs selector
5. persists preview result

### 14.3 Review and iterate

Operator may:

1. change policy
2. pin a candidate for one or more entries
3. set a player cap or target override
4. hold specific entries
5. rerun selection

### 14.4 Commit

Commit writes selected candidates back into the relevant `EntryFileState`
objects, bumps revisions, records lineage, and makes export available.

### 14.5 Export

Export continues to use existing CSV export flows, but should include late swap
session lineage in the export manifest.

## 15. API Contract

### 15.1 New endpoints

1. `POST /api/entry-manager/late-swap/sessions`
   - create a session from selected contests

2. `GET /api/entry-manager/late-swap/sessions?date=...`
   - list recent sessions

3. `GET /api/entry-manager/late-swap/sessions/{session_id}`
   - load full session state

4. `POST /api/entry-manager/late-swap/sessions/{session_id}/preview`
   - generate or regenerate candidates and selected preview

5. `POST /api/entry-manager/late-swap/sessions/{session_id}/pin-candidates`
   - pin one or more entry -> candidate choices

6. `POST /api/entry-manager/late-swap/sessions/{session_id}/policy`
   - update policy and mark session stale

7. `POST /api/entry-manager/late-swap/sessions/{session_id}/commit`
   - apply selected preview back to entry files

8. `POST /api/entry-manager/late-swap/sessions/{session_id}/export`
   - export committed or selected preview entries

### 15.2 Legacy endpoint compatibility

Existing per-contest `late-swap` endpoint should become a compatibility wrapper:

1. create a single-contest session
2. run preview with default policy
3. optionally auto-commit only in legacy mode

The new product surface should use the session endpoints directly.

### 15.3 Request model: create session

```json
{
  "date": "2026-03-08",
  "contest_ids": ["188511762", "188511763"],
  "policy": {
    "mode": "preserve_targets",
    "target_source": "source_portfolio",
    "candidate_count_per_entry": 10,
    "min_uniques": 1,
    "max_duplicate_lineups": 1,
    "min_gain_to_swap": 0.25,
    "swap_cost_lambda": 0.15,
    "exposure_bounds": {
      "203507": {"max": 45.0},
      "1629029": {"max": 25.0}
    }
  }
}
```

### 15.4 Response model essentials

Preview response should include:

- session metadata
- selected candidate ids
- per-entry candidate bundles
- exposure tables for all four states
- warnings and infeasibilities
- aggregate swap statistics
- selector diagnostics

## 16. Persistence and Lineage

### 16.1 Session artifacts

Persist:

- `request.json`
- `session_state.json`
- `candidates.parquet` or `candidates.json`
- `selection_result.json`
- `diagnostics.json`
- optional `preview_export.csv`

### 16.2 Entry state lineage

Add new optional fields to `EntryFileState`:

- `source_late_swap_session_id`
- `source_late_swap_mode`
- `source_late_swap_committed_at`

### 16.3 Export lineage

Export manifests should include:

- session id
- policy mode
- target source
- exposure summary
- warnings present at commit time

## 17. Frontend Product Spec

### 17.1 Page structure

Do not keep expanding `EntryManagerPage.tsx` as the only late swap surface.

Add a dedicated top-level page:

- `LateSwapPage.tsx`

Entry Manager remains the place for:

- upload
- entry file list
- apply build
- validation
- raw export

Late Swap becomes its own workbench.

### 17.2 Navigation

Add a distinct `late-swap` tab in `web/minutes-dashboard/src/App.tsx`.

`LivePage` should route "Open Late Swap" to this new page instead of sending the
operator into the generic entry-manager screen.

### 17.3 Frontend layout

The workbench should use a three-column desktop layout:

1. left rail
   - date selector
   - contest/session selector
   - mode presets
   - quick policy controls

2. center pane
   - exposure dashboard
   - swap summary
   - entry table

3. right pane
   - selected entry detail
   - candidate alternatives
   - warnings / reason codes / manual controls

On narrower screens, collapse to stacked panels in this order:

1. session and policy
2. exposure summary
3. entries table
4. entry detail drawer

### 17.4 Main frontend sections

#### A. Session Header

Show:

- date
- selected contests
- entry counts
- locked slots count
- last refreshed time
- stale session indicator
- preview vs committed status

#### B. Mode Presets

Preset cards for:

- Preserve Targets
- Best EV
- Decorrelated
- Catch Up
- Block

Each card should show a one-line summary of what it optimizes.

#### C. Exposure Policy Panel

Controls:

- target source
- player max/min exposure editor
- team/game exposure editors
- duplicate limit
- min uniques
- swap threshold
- max swaps per entry
- global vs segmented exposure scope

The panel must show effective caps after lock floors are applied.

#### D. Exposure Dashboard

Core table columns:

- player
- source target %
- locked floor %
- current committed %
- proposed final %
- delta vs target
- status

Statuses:

- `within_target`
- `over_target`
- `forced_over_by_locks`
- `under_target`
- `added_post_lock`
- `removed_post_lock`

Add quick filters:

- only over cap
- only forced by locks
- only changed players
- only top 20 exposures

#### E. Swap Summary Panel

Show:

- entries selected
- entries swapped
- held entries
- total swaps
- average swaps per changed entry
- projected or EV delta
- max exposure before
- max exposure after
- duplicate count before/after
- infeasibility count

#### F. Entry Review Table

Each row represents one entry.

Columns:

- contest
- entry id
- locked slots
- current lineup summary
- selected candidate summary
- swap count
- projected/EV delta
- reason chips
- state

States:

- `held`
- `swapped`
- `pinned`
- `warning`
- `infeasible`
- `manual_override`

#### G. Entry Detail Drawer

Selecting an entry opens a drawer with:

1. current lineup
2. selected candidate
3. all available candidates
4. player-in / player-out diffs
5. candidate metrics
6. reason codes
7. actions:
   - pin candidate
   - hold entry
   - exclude candidate
   - rebuild candidates for entry

#### H. Diagnostics Panel

Show:

- unmapped locked players
- draftable/start-time mismatches
- cap infeasibilities
- candidate coverage gaps
- stale session or stale projections

#### I. Commit / Export Bar

Persistent footer actions:

- Refresh Preview
- Commit Preview
- Export CSV
- Revert to Last Committed

Commit must require an explicit click. Preview generation alone cannot mutate the
entry file state.

### 17.5 Frontend state model

Do not keep late swap state in one giant page component.

Split into:

- `api/late_swap.ts`
- `pages/LateSwapPage.tsx`
- `components/lateSwap/LateSwapHeader.tsx`
- `components/lateSwap/LateSwapPolicyPanel.tsx`
- `components/lateSwap/ExposureDashboard.tsx`
- `components/lateSwap/SwapSummaryPanel.tsx`
- `components/lateSwap/EntryReviewTable.tsx`
- `components/lateSwap/EntryDetailDrawer.tsx`
- `components/lateSwap/DiagnosticsPanel.tsx`

State management should use:

1. server-backed session state from API
2. local draft policy edits
3. explicit preview refresh

Avoid hidden auto-runs on every small UI edit.

### 17.6 Manual operator controls

Manual controls should be session-local by default, not persistent global
strategy overrides.

Session-local controls:

- pin candidate for an entry
- hold an entry
- temporary ban player
- temporary boost player
- temporary per-player cap
- temporary target override

Persistent strategy overrides remain a separate system.

## 18. Backend Implementation Plan

### 18.1 New package

Create:

```text
projections/late_swap/
```

Initial files:

- `models.py`
- `session_store.py`
- `lock_state.py`
- `candidate_generation.py`
- `scoring.py`
- `portfolio_selector.py`
- `diagnostics.py`

### 18.2 Existing files to update

Backend:

- `projections/api/entry_manager_api.py`
- `projections/api/minutes_api.py`
- optionally `projections/api/contest_sim_api.py` for shared scoring helpers

Frontend:

- `web/minutes-dashboard/src/App.tsx`
- new `web/minutes-dashboard/src/pages/LateSwapPage.tsx`
- new `web/minutes-dashboard/src/api/late_swap.ts`
- new `web/minutes-dashboard/src/components/lateSwap/*`

### 18.3 Reuse opportunities

Reuse concepts and utilities from:

- `projections/contest_sim/portfolio_optimizer.py`
- existing DK assignment / export validation logic
- existing entry upload / persistence flows
- existing late swap slot-lock logic
- existing late-swap bonus and slot placement logic upstream of lock

## 19. Edge Cases and Required Behavior

### 19.1 Lock-driven overexposure

If a player is already locked above the requested cap:

1. mark as infeasible
2. prevent adding new exposure by default
3. show effective minimum exposure equal to current locked floor

### 19.2 Repeated late-swap runs

Operators may run late swap at 7:05, 7:55, and 8:25. The system must support:

1. session history
2. re-anchor policy behavior
3. clear distinction between committed state and new preview

### 19.3 News without `OUT`

Lineup changes, minute-limit news, role changes, and teammate news should still
justify swap preview. Selection cannot rely on `only_out_lineups` alone.

### 19.4 Contest segmentation

The operator may want different exposure behavior for:

- high-dollar single-entry contests
- small-field 3-max
- large-field MME

Support either:

1. separate sessions
2. segmented policies within one session

### 19.5 Duplicate lineup risk

When late value opens, duplicate final lineups can explode. Duplicate limits
must be explicit, not left to chance.

### 19.6 Candidate starvation

If a player is over cap but every candidate still includes him, the system must
report coverage failure rather than silently accepting bad exposure drift.

### 19.7 Mapping gaps

If a locked slot cannot be mapped to a player:

1. mark the entry as degraded
2. keep it out of auto-swap if necessary
3. surface it prominently in diagnostics

### 19.8 Draftable vs start-time disagreement

When DK lock markers, draftable metadata, and schedule start times disagree, the
system must show which source forced the lock.

### 19.9 Dead locked lineups

If a locked player is later ruled out or effectively dead, the system cannot fix
that slot. It should still optimize the rest of the lineup and show a warning.

### 19.10 Mixed source entries

If some contests came from a portfolio build and others came from manual entry
edits, target-source fallback must be deterministic and visible.

## 20. Testing Plan

### 20.1 Backend unit tests

Add tests for:

1. lock-state computation
2. target exposure derivation
3. lock-driven cap infeasibility
4. candidate generation diversity passes
5. grouped selector exact-one-per-entry behavior
6. hard exposure min/max enforcement
7. duplicate-lineup enforcement
8. session re-anchor behavior
9. commit lineage persistence

### 20.2 Backend API tests

Add tests for:

1. session creation
2. preview generation
3. policy updates
4. pinning and rerun behavior
5. commit behavior
6. export lineage
7. legacy endpoint compatibility

### 20.3 Frontend tests

Add tests for:

1. mode preset behavior
2. exposure dashboard rendering and filtering
3. pin / hold interactions
4. stale-session warnings
5. commit gating
6. entry detail drawer candidate selection

### 20.4 Replay and evaluation tests

Build historical replay workflows where possible:

1. load archived entry files
2. reconstruct live-like swap windows
3. compare:
   - old per-entry late swap
   - new grouped late swap
4. measure:
   - max exposure
   - exposure deviation from target
   - duplicate rate
   - projected/EV retained
   - post-contest realized ROI when available

## 21. Rollout Plan

### Phase 1: Sessionized preview workflow

Deliver:

1. late swap session model
2. preview without auto-commit
3. grouped hard-cap selector
4. exposure dashboard
5. commit/export flow

This phase removes the biggest product flaw even without simulation-backed
scoring.

### Phase 2: Target-preserving portfolio selection

Deliver:

1. source-target derivation
2. target deviation penalties
3. pin / hold workflow
4. duplicate controls
5. segmented contest policies

### Phase 3: Decorrelated remaining-slate selection

Deliver:

1. remaining-slate candidate scoring
2. decorrelated portfolio repair/search
3. `decorrelated_ev`, `catch_up`, and `block` mode hardening

### Phase 4: Contest-aware late swap

Deliver when live contest state is available:

1. current score / rank ingestion
2. payout-aware objective variants
3. contest-position-sensitive mode defaults

## 22. Success Criteria

The redesign is successful when:

1. late swap no longer auto-creates accidental all-in exposures
2. operators can preview exact post-swap exposure state before commit
3. lock-driven infeasibility is surfaced clearly
4. repeated late swap runs remain auditable and understandable
5. large MME late swap runs stay performant and operationally calm

## 23. Final Recommendation

Go all in on a portfolio-aware late swap redesign.

Specifically:

1. keep the existing lineup solver and DK slot logic
2. replace the current auto-pick layer with a grouped sessionized selector
3. give late swap its own frontend workbench
4. treat exposure state, infeasibility, and preview diagnostics as first-class
5. phase in remaining-slate simulation and contest-aware selection after the
   sessionized grouped-selector foundation is in place
