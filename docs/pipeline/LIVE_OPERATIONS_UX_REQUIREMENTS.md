# Live Operations UX Requirements

This document captures operator-facing UX decisions that should be treated as
requirements for the live pipeline and control plane.

It complements:

- `docs/pipeline/LIVE_PIPELINE_PRODUCTION_SPEC.md`
- `docs/pipeline/LIVE_OPERATIONS_UI_SPEC.md`
- `docs/pipeline/MANUAL_OVERRIDE_CONTRACT.md`

## 1. Core Principles

1. Operators must always know which published run they are building from.
2. Freshness must be visible per game, not only per slate.
3. Blocked or in-progress runs must not make the current published state
   ambiguous.
4. Manual availability actions must be auditable and easy to understand under
   time pressure.
5. Strategy controls must stay separate from canonical live projection state.
6. The first implementation may assume a single primary operator and should
   optimize for speed, clarity, and low-friction actions over workflow
   ceremony.

## 2. Latest Published Run Behavior

If the next run is blocked, stale-gated, or still in progress:

1. the latest published run remains available for lineup building
2. the UI must explicitly tell the operator that the next run is not yet
   published
3. the UI must show why the next run is blocked or in progress

Required operator states:

- `published`
- `in_progress`
- `blocked`
- `waiting_for_fresh_input`
- `stale_relative_to_newer_input`

Required blocker/in-progress fields:

- `game_date`
- `candidate_run_id`
- `latest_published_run_id`
- `status_reason`
- `affected_games`
- `updated_at`

## 3. Per-Game Freshness UX

The operator surface should show per-game freshness at minimum:

- injury `as_of`
- lineup `as_of`
- odds `as_of`
- props `as_of`, if used by the path
- freshness age in minutes
- whether a manual override is active

This should be visible in a game-level row/card, not hidden behind logs.

## 4. Source Provenance And Disagreement

For each player with a material status:

- show the effective status
- show the source used
- show whether sources disagree

Required provenance values:

- `nba_official`
- `rotowire`
- `manual_override`

Policy requirement:

- Rotowire takes precedence for explicit `OUT` signals in the live pipeline
  when official NBA injury data is lagging.

Recommended disagreement badges:

- `official_q_rotowire_out`
- `manual_out_unconfirmed`
- `newer_input_than_published`

## 5. Manual Override UX

### 5.1 Allowed live actions

Canonical live projection actions:

- manual `OUT`
- manual clear / `IN`

Not canonical live projection actions:

- minute boosts
- fantasy-point boosts
- projection nerfs
- direct model-output edits

Those belong in optimizer / contest-sim controls.

Placement requirement:

- canonical live availability actions must live in `GameView`
- slate-level `Live` can link into `GameView` and display override state
- boosts / nerfs must live in optimizer and/or contest sim, not in the
  canonical live control surface

### 5.2 Required manual override display

For each active manual override, show:

- player
- game
- override type
- who entered it
- reason/source
- created time
- expiry/clear state
- whether it has been confirmed by source data yet

### 5.3 First-pass interaction defaults

For the initial implementation:

- override actions apply immediately on submit
- a lightweight confirmation is sufficient for destructive actions
- no approval workflow is required
- desktop/laptop is the primary operating environment
- the UI should always frame published state as the current truth and candidate
  state as secondary

### 5.4 Reset behavior

The operator UI should provide a single reset action for manual strategy inputs
that are not manual `OUT` overrides.

Required rule:

- `reset manual inputs` must not silently clear active manual `OUT` overrides

Recommended split:

- `Reset strategy inputs`
- `Clear manual OUT overrides`

## 6. Single-Game Controls

Operators should be able to trigger single-game pipeline / inference runs.

Required control:

- `Rebuild game`

Recommended supporting fields:

- game id / matchup
- trigger timestamp
- trigger source (`operator`)
- queued/running/completed/failed state

This control should be available from the same game-level surface that shows
freshness and blocker state.

## 7. Late-News Operator Flow

The late-news flow should be minimal:

1. operator identifies player
2. operator marks player `OUT` with source/reason
3. UI immediately shows override active
4. affected game enters `rebuild requested` / `in progress`
5. latest published run remains usable until the new run publishes
6. UI updates to the new published run once complete

## 8. Implementation Notes

Likely code surfaces affected:

- `web/minutes-dashboard/`
- `projections/api/ops_api.py`
- `projections/api/minutes_api.py`
- `projections/cli/build_minutes_live.py`
- `prefect_flows/live_nba_pipeline_v3.py`

This doc should be updated before frontend implementation begins if additional
operator states or controls are introduced.
