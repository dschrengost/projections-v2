# Live Operations UI Spec

This document is the living redesign spec for the live operator UI in
`projections-v2`.

It is the canonical product/design spec for the operator-facing dashboard and
control surface. It should be updated as the UI direction changes, as backend
contracts become clearer, or as operational requirements are refined.

It complements, but does not replace:

- `docs/pipeline/LIVE_PIPELINE_PRODUCTION_SPEC.md`
- `docs/pipeline/LIVE_OPERATIONS_UX_REQUIREMENTS.md`
- `docs/pipeline/MANUAL_OVERRIDE_CONTRACT.md`

## 1. Purpose

The current dashboard is doing too many jobs at once:

- daily projection browsing
- live pipeline monitoring
- optimizer and contest workflows
- diagnostics browsing
- props browsing

That makes it harder to reason about the live operator experience as a product.
We need one living spec that answers:

- who the live UI is for
- what the primary screens should be
- which states matter operationally
- what data contracts the UI needs from the backend
- how redesign work should be sequenced

## 2. Goals

1. Make the operator's current truth obvious at a glance.
2. Make the latest published run and the next candidate run impossible to
   confuse.
3. Surface blocked, stale, waiting, and superseded states clearly.
4. Make per-game freshness and provenance visible without opening raw JSON.
5. Support audited manual `OUT` / `IN` actions once the backend path is ready.
6. Keep lineup-building and decision support available even while the next run
   is blocked or still processing.

## 3. Non-goals

1. This spec does not redesign every dashboard tab immediately.
2. This spec does not define the entire optimizer UX.
3. This spec does not replace pipeline policy docs.
4. This spec does not assume we collapse all product surfaces into one page.

## 4. Current Problems

### 4.1 Product boundaries are unclear

The current React app mixes:

- live projections
- pipeline state
- evaluation
- optimizer
- entry manager
- contest sim
- diagnostics
- props

Those are related, but they are not one workflow. The redesign should impose a
clear hierarchy instead of treating all tabs as peers.

### 4.2 The live control plane is still too hidden

The pipeline now emits strong diagnostics:

- `source_freshness`
- `freshness_gates`
- `input_change_set`
- `rerun_plan`
- `publish_status`

But operators still have to infer too much from logs and artifacts.

### 4.3 The current UI is optimized for browsing rows, not running operations

The existing minutes table is useful, but it is not yet a strong operator
surface for:

- run-state interpretation
- late-news monitoring
- per-game intervention
- safe manual override workflows

## 5. Primary Users

### 5.1 Live operator

Needs:

- confidence about what is currently published
- visibility into what is changing
- quick answers under time pressure
- safe control actions with audit trail

### 5.2 Analyst / reviewer

Needs:

- run history
- source provenance
- before/after comparison
- diagnostic detail for debugging incidents

### 5.3 Lineup builder / consumer

Needs:

- stable current projections
- awareness when a newer run is pending
- enough context to judge whether current outputs are safe to use

## 6. Design Principles

### 6.1 Current truth first

The UI should lead with:

- what run is currently published
- whether a newer run is pending
- whether anything is blocked or stale

### 6.2 Game-first live monitoring

Live operations are usually resolved at the game level, not the full-table
level. The UI should make games the primary operational unit.

### 6.3 State before detail

The operator should be able to answer these questions without opening a drawer:

- Is the slate healthy?
- Which games are risky?
- Is the next run publishable?
- Did anything get superseded?

### 6.4 Safe controls only

Canonical live controls should stay narrow:

- rebuild game
- manual `OUT`
- clear manual `OUT` / manual `IN`

Minute boosts and arbitrary output edits should not be part of the canonical
live operations UI.

## 7. Information Architecture

Recommended top-level product structure:

1. `Live`
   - the operator home for active slates
2. `Runs`
   - run history, publish history, diagnostics drill-down
3. `Projections`
   - player-level table browsing and comparison
4. `Strategy`
   - optimizer / entry-manager / contest-sim surfaces
5. `Diagnostics`
   - deeper technical drill-downs and lower-frequency debugging tools

This is intentionally not the same as the current tab list.

## 8. Core Live Screen

The `Live` screen should be the canonical operator home.

It should contain four zones.

### 8.1 Slate status rail

Must show:

- slate date
- current published run id
- current published `as_of_ts`
- candidate run id, if any
- candidate state
- publish status
- last successful update time

Candidate state values should include:

- `published`
- `in_progress`
- `blocked`
- `waiting_for_fresh_input`
- `superseded`
- `stale_relative_to_newer_input`

### 8.2 Game board

The main operational surface should be a board of per-game cards or rows.

Each game card should show at minimum:

- matchup
- tip time
- minutes to tip
- injury freshness
- lineup freshness
- odds freshness
- props freshness
- current risk / warning badges
- latest effective status source summary

### 8.3 Selected game detail

Selecting a game should open a richer detail surface with:

- affected players
- status provenance
- change-set summary
- latest source timestamps
- latest run impact summary
- manual override controls when supported

### 8.4 Run event strip

A compact run timeline should show:

- published runs
- blocked runs
- superseded runs
- current candidate run

This should help answer "what just happened?" without opening logs.

## 9. Required Backend Contracts For UI

The UI redesign should be driven by explicit API payloads, not by teaching the
frontend to read artifact directories directly.

Needed API surfaces:

### 9.1 Live slate summary

Must include:

- `game_date`
- `latest_published_run_id`
- `latest_published_as_of_ts`
- `candidate_run_id`
- `candidate_status`
- `candidate_status_reason`
- `publish_status`
- `updated_at`

### 9.2 Per-game live status

Must include:

- `game_id`
- `tip_ts`
- `minutes_to_tip`
- `source_freshness`
- `freshness_gates`
- `affected_by_change_set`
- `rerun_targeted`
- `manual_override_active`

### 9.3 Run detail

Must include:

- `run_id`
- `as_of_ts`
- `rerun_plan`
- `input_change_set`
- `preflight_report`
- `postflight_report`
- `publish_precheck`
- `publish_superseded`

## 10. Manual Override UX Direction

The UI should eventually support the policy already documented elsewhere:

- manual `OUT`
- manual clear / `IN`

Required display behavior:

- active overrides are always visible in the game and player surfaces
- override provenance is visible
- override-driven rebuild state is visible
- clearing an override is explicit and auditable

This spec does not require the override UI to ship before the diagnostics
surface.

## 11. Visual Direction

The redesign should move away from "single dense table plus tabs" as the only
mental model.

Recommended direction:

- stronger slate status framing
- game cards / grouped rows as the primary live monitor
- drill-down panels for one game at a time
- clear state colors with restrained use of red/yellow
- typography that distinguishes run state, timestamps, and player info

Avoid:

- generic admin-dashboard look
- hidden operational state behind tooltips only
- forcing the operator to correlate multiple tabs manually

## 12. Rollout Sequence

### 12.1 Phase A: Live diagnostics shell

Ship first:

- slate status rail
- game board
- published vs candidate run state
- per-game freshness summaries

### 12.2 Phase B: Run detail and history

Ship next:

- run detail drawer/page
- rerun-plan and change-set visibility
- blocked / superseded explanation

### 12.3 Phase C: Manual override controls

Ship after backend support is ready:

- manual `OUT`
- clear / `IN`
- override audit fields

### 12.4 Phase D: Visual cleanup of adjacent tabs

After the live operator home is solid:

- decide what remains in the same app
- decide what should move into `Strategy` or `Diagnostics`

## 13. Open Questions

1. Should `Live` and `Projections` remain in the same app shell, or should
   operator views become their own route group?
2. How much of run history belongs in the main UI versus a diagnostics-only
   page?
3. Do we want one canonical game detail drawer shared by pipeline state and
   projections state, or two separate views?
4. When a candidate run is superseded, should the UI keep it visible by
   default or collapse it into history?

## 14. Immediate Next Steps

1. Freeze this spec as the canonical redesign doc before implementation work.
2. Add a lightweight API contract doc for the new live-status payloads.
3. Build the `Live` home first, not a broad visual refresh of all current tabs.
4. Keep `LIVE_OPERATIONS_UX_REQUIREMENTS.md` as the requirement/policy layer
   and update it only when operator rules change.
