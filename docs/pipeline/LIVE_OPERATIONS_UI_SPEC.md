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
5. This spec does not optimize for a multi-operator workflow yet.

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

### 5.4 Single-operator default

For the current implementation phase, assume one primary operator.

That means the UI should optimize for:

- speed of interpretation
- low-friction actions
- obvious current truth
- minimal workflow ceremony

It does not need to optimize yet for:

- role-based permissions
- multi-user conflict resolution
- approval chains
- handoff workflows between operators

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

### 6.5 Control-layer separation

The redesign should make a hard distinction between:

- canonical live projection controls
- downstream strategy controls

Canonical live projection controls belong in `Live` and `GameView` and are
allowed to change the published projection state.

Downstream strategy controls belong in `Strategy` surfaces such as optimizer
and contest sim and must not mutate the canonical live projection state.

This means:

- `GameView` may mark players `OUT`
- `GameView` may clear or explicitly mark players back `IN`
- `GameView` may trigger a rebuild for the affected game
- `GameView` should not expose boost / nerf controls for projections
- boost / nerf controls should live in optimizer and/or contest sim

### 6.6 Single-operator defaults

Until there is a real multi-operator need, use these defaults:

- manual availability overrides apply immediately
- a lightweight confirmation is enough for destructive actions
- no approval queue is required
- no role-based gating is required in the UI
- desktop/laptop is the primary operating environment
- mobile should be usable, but it is not the primary design target

This keeps the first implementation focused on operational clarity instead of
workflow machinery.

## 7. Information Architecture

Recommended top-level product structure:

1. `Live`
   - the default landing screen and operator home for active slates
2. `Runs`
   - run history, publish history, diagnostics drill-down
3. `Projections`
   - player-level table browsing and comparison
   - this is where the legacy main minutes table behavior moves
4. `Strategy`
   - optimizer / entry-manager / contest-sim surfaces
5. `Diagnostics`
   - deeper technical drill-downs and lower-frequency debugging tools

This is intentionally not the same as the current tab list.

### 7.1 Legacy main minutes page

The legacy main minutes page should be removed as the default home.

It is still valuable as a player-table browsing surface, but it is the wrong
default for live operations because it answers the wrong first question. The
default landing screen should answer:

- what is currently published
- what is changing
- which games are risky
- whether the operator needs to act

The old row-heavy minutes view should be retained only as part of
`Projections`, not as the main operator landing page.

### 7.2 Navigation model

The redesign should establish a simple hierarchy:

- `Live` is the slate-level triage surface
- game cards on `Live` are the primary entry point into per-game work
- `GameView` is the canonical per-game operations workspace
- `Projections` is the player-table analysis surface, not the live home

This keeps slate monitoring and per-game intervention distinct instead of
forcing one screen to do both.

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

The rail should always frame the currently published run as the default truth.

Candidate state values should include:

- `published`
- `in_progress`
- `blocked`
- `waiting_for_fresh_input`
- `superseded`
- `stale_relative_to_newer_input`

### 8.2 Game board

The main operational surface should be a board of per-game cards or rows.

Each card should be clearly clickable and route into the relevant `GameView`
for that game.

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

Recommended interaction behavior:

- single click opens `GameView`
- quick peek affordance may show a compact summary without leaving `Live`
- card styling should make the highest-risk games visually obvious without
  relying only on color

### 8.3 GameView as per-game operations page

`GameView` should be the canonical page for detailed per-game operations.

It should contain:

- affected players
- status provenance
- change-set summary
- latest source timestamps
- latest run impact summary
- current published vs candidate values where applicable
- manual override controls when supported

Recommended first implementation shape:

- desktop: full-page route
- mobile: full-page route
- defer drawer/modal variants until the core page is working well

Allowed canonical actions in `GameView`:

- mark player `OUT`
- clear manual `OUT`
- mark player back `IN` where the override contract supports it
- request or observe rebuild state for the game

Recommended action semantics for the first version:

- actions apply immediately after submit
- use a simple confirmation for `OUT` and clear actions
- `IN` should usually act as a practical recovery action, not a complex
  independent workflow
- the page should show the game as `rebuild requested`, `in progress`,
  `published`, `blocked`, or `superseded` after an action

Not allowed in canonical `GameView`:

- projection boosts
- projection nerfs
- direct minutes edits
- direct fantasy-point edits
- optimizer exposure tuning

Those controls belong in `Strategy` surfaces because they represent lineup or
portfolio preferences rather than source-of-truth live availability.

`GameView` should not try to replace the slate-level `Live` board. Its job is
to answer:

- what is happening in this game
- why this game is risky or blocked
- what changed in the latest candidate run
- whether the operator needs to intervene

Display priority in `GameView`:

- current published values first
- candidate values second
- override state always visible when active

The operator should never have to guess whether they are looking at published
truth or pending state.

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

Manual override placement:

- `GameView` is the canonical place for live availability overrides
- `Live` may expose read-only override state and lightweight entry points into
  `GameView`
- optimizer and contest sim may expose boosts / nerfs or other strategy
  controls, but those should be visually and architecturally distinct from
  live availability overrides

This spec does not require the override UI to ship before the diagnostics
surface.

Recommended first-pass override behavior:

- one-click entry from `Live` into `GameView`
- simple override form with reason/source fields
- immediate persistence on submit
- visible active-override state on the player row and game header
- no approval workflow
- no attempt to support non-availability override types in the canonical path

## 11. Visual Direction

The redesign should move away from "single dense table plus tabs" as the only
mental model.

Recommended direction:

- stronger slate status framing
- game cards / grouped rows as the primary live monitor
- `GameView` as the full per-game workspace reached from game cards
- clear state colors with restrained use of red/yellow
- typography that distinguishes run state, timestamps, and player info

Avoid:

- generic admin-dashboard look
- hidden operational state behind tooltips only
- forcing the operator to correlate multiple tabs manually
- keeping the legacy minutes table as the default home

## 12. Rollout Sequence

### 12.1 Phase A: Live diagnostics shell

Ship first:

- slate status rail
- game board
- published vs candidate run state
- per-game freshness summaries
- routing from game cards into `GameView`

### 12.2 Phase B: GameView and run detail

Ship next:

- `GameView` per-game detail page
- run detail drawer/page
- rerun-plan and change-set visibility
- blocked / superseded explanation
- clear migration of the legacy main minutes page into `Projections`

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
3. When a candidate run is superseded, should the UI keep it visible by
   default or collapse it into history?
4. When a true multi-operator workflow appears, which controls need stronger
   audit or permission boundaries first?

## 14. Immediate Next Steps

1. Freeze this spec as the canonical redesign doc before implementation work.
2. Add a lightweight API contract doc for the new live-status payloads.
3. Build the `Live` home first, not a broad visual refresh of all current tabs.
4. Treat the legacy main minutes page as deprecated home behavior and move that
   table UX under `Projections`.
5. Keep `LIVE_OPERATIONS_UX_REQUIREMENTS.md` as the requirement/policy layer
   and update it only when operator rules change.

## 15. Agent Handoff

The UI spec is now specific enough to begin implementation.

The first implementation pass should assume:

- one primary operator
- desktop/laptop as the main target
- `Live` as the default landing page
- `GameView` as a full-page per-game route
- published state as the default truth
- candidate state as visible but secondary
- immediate manual availability actions with lightweight confirmation
- no boost / nerf controls in the canonical live control surface

Build scope for the first pass:

1. Replace the legacy default home with `Live`.
2. Render a slate status rail with published vs candidate run state.
3. Render a game board with clickable game cards.
4. Route each game card into `GameView`.
5. In `GameView`, show:
   - game header and run state
   - player list
   - source provenance / freshness context
   - override state if active
6. Do not implement manual override submission yet unless the backend path is
   ready in the same pass.
7. Do not redesign optimizer / contest sim in this pass.

What should not be revisited during the first build unless implementation
forces it:

- whether the old minutes table remains the home
- whether boosts / nerfs belong in `GameView`
- whether the first version needs multi-operator workflow support
- whether candidate state should replace published state as the main view

If implementation pressure forces tradeoffs, preserve this priority order:

1. correct published vs candidate framing
2. clear slate-level `Live` triage
3. clean route into `GameView`
4. per-game provenance and override visibility
5. visual polish
