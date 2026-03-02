# Strategy Overrides + GameView V2 Spec

This document defines the V2 product and technical spec for integrating
downstream strategy overrides into the game-level operator workflow while
keeping canonical live controls policy-safe.

Companion docs:

- `docs/pipeline/STRATEGY_OVERRIDE_CONTRACT.md`
- `docs/pipeline/MANUAL_OVERRIDE_CONTRACT.md`
- `docs/pipeline/LIVE_OPERATIONS_UI_SPEC.md`

## 1. Problem Statement

V1 strategy overrides are functional but fragmented:

- strategy deltas (`minutes_delta`, `fpts_delta`) live in optimizer surfaces
- manual canonical availability (`force_out`, `force_in`) lives in `GameView`
- operators cannot manage both layers from one game-centric workflow
- current game cards are too large/diffuse for dense slate monitoring

V2 goal: provide a compact, side-by-side game board where the operator can
quickly manage both layers with clear policy separation and optimizer-relevant
metrics visible in-context.

## 2. Product Goals

1. Unify game-level operator workflow:
   - manual availability actions
   - strategy override actions
2. Improve slate scanning density:
   - compact game cards
   - more games visible side-by-side
3. Surface optimizer metrics in GameView context:
   - minutes p50
   - FPPM
   - salary
   - value
   - projection/effective projection
4. Preserve policy boundary:
   - manual availability mutates canonical live layer
   - strategy overrides remain downstream-only
5. Keep operator flow fast and low-friction during lock window.

## 3. Non-Goals

- No mutation of canonical live model artifacts from strategy actions.
- No lineup-group/stack/exposure-rule editing in GameView.
- No new multi-operator approvals/permissions workflow in V2.
- No replacement of optimizer page for full build configuration.

## 4. UX Information Architecture

## 4.1 Screen model

Use a dual-level model:

1. `Live` page remains the slate entry point.
2. `GameView V2` becomes a board-capable workspace with two display modes:
   - `Board (Dense)` for side-by-side multi-game operations
   - `Focus` for one-game deep-dive (existing current behavior refined)

## 4.2 Layer separation in UI

Each game surface must show two distinct control bands:

- `Canonical Availability` (force in/out, clear)
- `Strategy Overrides (Downstream Only)` (minutes/fpts deltas)

The strategy band must include persistent copy:

- "Affects optimizer/contest sim only"
- "Does not change published live projections"

## 5. Dense Board Layout (Primary V2 Deliverable)

## 5.1 Card density targets

Desktop target:

- 2-4 cards visible per row depending viewport
- significantly reduced vertical padding
- compact table rows for player operations

Recommended CSS direction:

- game grid: `repeat(auto-fit, minmax(420px, 1fr))`
- card padding: `12px` (down from current larger rhythm)
- row height: `28-32px`
- summary chips: single-line, minimal vertical margin

## 5.2 Game card content

Each game card should include:

- header: matchup, tip, minutes-to-tip, risk chips
- mini game stats row: total/spread/implied
- compact player table with optimizer metrics + control inputs

Player table columns (dense mode):

- Player
- Status
- Min p50
- FPPM
- Salary
- Value
- Model FPTS
- Eff FPTS
- Manual (IN/OUT/Clear)
- Strategy (`Min Δ`, `FPTS Δ`, Clear)

## 5.3 Side-by-side usability constraints

- Sticky card header for each game when card scrolls.
- Horizontal overflow allowed only inside player table body, not whole page.
- All primary actions reachable in <= 2 clicks from board view.
- No mandatory drawer open for routine actions.

## 6. Data Contract V2 (UI payload)

Introduce a GameView-oriented aggregate endpoint to avoid N+1 frontend fetches.

Recommended endpoint:

- `GET /api/ops/game-board?date=YYYY-MM-DD&draft_group_id=...&run_id=...`

Returns per game:

- game metadata (matchup, tip, status/risk chips)
- player rows with:
  - manual availability state (active override info)
  - strategy override state (`minutes_delta`, `fpts_delta`, `has_override`)
  - optimizer metrics:
    - `model_minutes` / `effective_minutes`
    - `fppm`
    - `salary`
    - `value`
    - `model_proj` / `effective_proj`
    - optional `own_proj`, `stddev`, `p90`
- slate strategy metadata:
  - `client_revision`
  - `updated_at`
  - `strategy_override_count`

Implementation note:

- Backend can compose existing sources:
  - `/api/ops/game` semantics
  - optimizer pool (`use_strategy_overrides=true`)
  - slate override store (`/api/optimizer/overrides`)

## 7. Write Semantics

## 7.1 Manual availability actions

- Continue immediate write-through to:
  - `POST /api/ops/manual-availability-overrides`
  - `DELETE /api/ops/manual-availability-overrides/{override_id}`
- Optimistic UI update with per-player busy state.

## 7.2 Strategy actions

- Buffered edit model per slate (not immediate per keystroke write).
- Save/discard controls at board-level and per-card quick-save option.
- Persist via existing endpoint:
  - `PUT /api/optimizer/overrides` with `expected_revision`
- Revision conflict handling:
  - show conflict banner
  - reload latest overrides
  - preserve user draft locally for merge/retry

## 7.3 Cross-layer conflict handling

When player is manually forced out:

- keep strategy override persisted but visually muted
- show "inactive due to manual OUT" badge
- effective fpts/minutes display should reflect inactive state

## 8. Interaction Model

## 8.1 Operator quick flow

1. Open `Live` -> `Board (Dense)` mode.
2. Scan all games side-by-side.
3. For a player:
   - set manual OUT/IN if source-of-truth availability changed
   - apply strategy deltas if downstream belief differs from model
4. Save strategy changes once per slate (or per game quick-save).
5. Launch optimizer/contest sim with `use_strategy_overrides=true`.

## 8.2 Affordances for intuition

- Inline delta inputs with signed formatting (`+2.5`, `-1.0`).
- Immediate derived metric feedback (`Eff Min`, `Eff FPTS`, `Value`).
- Override chips:
  - `MANUAL OUT`, `MANUAL IN`
  - `STRAT MINΔ`
  - `STRAT FPTSΔ`
- Unsaved change counter pinned in toolbar.

## 9. Validation and Guardrails

- Reuse strategy contract validation:
  - finite numeric deltas
  - null+null means clear
- UI soft warnings:
  - implausible `minutes_delta` range breach (e.g. > +/-12)
  - large `fpts_delta` (e.g. > +/-10)
  - player marked OUT with active positive strategy delta

No hard block for soft warnings in V2; warnings are advisory.

## 10. Performance Requirements

- Dense board initial load target: < 1.5s for typical slate payload.
- Strategy delta typing latency: < 50ms local response.
- Save roundtrip target: < 500ms for normal slate override payload.
- Avoid per-game sequential calls from frontend.

## 11. Observability

Log and expose:

- strategy save attempts/success/conflicts
- manual override writes by operator
- board payload build timing
- count of games/players loaded
- per-request flag whether strategy effective values were used

## 12. Rollout Plan

## Phase A: Backend aggregation + compatibility

- Add `game-board` aggregate read endpoint.
- Keep existing endpoints unchanged.
- Add response diagnostics (`strategy_override_count`, revision).

## Phase B: Frontend dense board UI

- Add board mode toggle to Live/GameView surface.
- Implement compact side-by-side game cards.
- Add per-player strategy inputs and manual availability controls in same row.
- Add save/discard and conflict handling.

## Phase C: Operator polish

- Keyboard navigation and bulk clear actions.
- Filter chips: only overridden, only risky, only starters.
- Persist view preferences (dense/focus, column visibility).

## 13. Acceptance Criteria

1. Operator can view all slate games side-by-side in compact cards on desktop.
2. Each player row shows minutes p50, fppm, salary, value, and proj/effective proj.
3. Operator can apply manual IN/OUT and strategy deltas from the same game card.
4. Strategy controls clearly indicate downstream-only scope.
5. Saving strategy changes is revision-safe and conflict-aware.
6. Optimizer and contest sim both consume the same saved overrides.
7. No canonical live projection artifacts are mutated by strategy writes.

## 14. File-Level Implementation Targets

Frontend:

- `web/minutes-dashboard/src/pages/LivePage.tsx`
- `web/minutes-dashboard/src/components/GameView.tsx`
- `web/minutes-dashboard/src/components/gameview.css`
- `web/minutes-dashboard/src/pages/live.css`
- `web/minutes-dashboard/src/api/manualAvailability.ts`
- `web/minutes-dashboard/src/api/optimizer.ts`

Backend:

- `projections/api/ops_api.py` (new aggregate game-board endpoint)
- `projections/api/optimizer_api.py` (reuse existing overrides endpoints)
- `projections/api/optimizer_service.py` (metric field parity for UI payload)

Contracts:

- `docs/pipeline/STRATEGY_OVERRIDE_CONTRACT.md` (add reference to this V2 UI spec)
- `docs/pipeline/LIVE_OPERATIONS_UI_SPEC.md` (note GameView V2 dense mode placement)
