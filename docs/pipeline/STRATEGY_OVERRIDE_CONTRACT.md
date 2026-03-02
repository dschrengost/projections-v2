# Strategy Override Contract

This document defines the practical implementation contract for persistent
operator strategy overrides in `projections-v2`.

It is the companion to:

- `docs/pipeline/MANUAL_OVERRIDE_CONTRACT.md` for canonical live availability
  overrides (`force_out` / `force_in`)
- `docs/pipeline/LIVE_PIPELINE_PRODUCTION_SPEC.md` for the policy boundary:
  boost/preference logic belongs in optimizer / contest sim, not in the live
  projection pipeline

## 1. Purpose

This contract exists to support a downstream-only operator layer that can bias
optimizer and contest-sim decisions without mutating canonical live model
inputs or published live projections.

Primary operator use cases:

- boost or nerf a player by changing effective minutes
- boost or nerf a player by changing effective fantasy points directly
- persist those adjustments across subsequent model reruns
- clear one player or reset the slate at any time

This contract treats strategy overrides as a **private belief layer** applied
after baseline projection/world selection, not as a mutation of model inputs or
published artifacts.

## 2. Scope

This contract covers:

- per-slate persistent strategy overrides for optimizer / contest sim
- API and CLI interfaces
- persistence format
- application semantics on top of the selected projection run and worlds set
- diagnostics exposed to strategy surfaces
- first implementation support for both `minutes_delta` and `fpts_delta`

This contract does not cover:

- canonical live availability overrides
- mutation of minutes/rates model inputs
- mutation of published live projections
- lineup-group rules, stack rules, or portfolio-selection rules
- ownership overrides in the first implementation

## 3. Design Principles

1. Downstream-only.
   Strategy overrides must never feed back into `build_minutes_live`,
   `score_minutes_v1`, rates scoring, or published live projection artifacts.

2. Persistent across reruns.
   Overrides are slate-scoped, not run-scoped. A new model run should recompute
   effective values against the new baseline, not discard the override layer.

3. Applied after run selection.
   Consumers first resolve the projection bundle / worlds source, then apply
   strategy overrides to the selected run.

4. World-consistent.
   If overrides are enabled, optimizer and contest sim must agree on the
   adjusted player distribution. Mean-only overlays that leave contest-sim
   worlds untouched are not acceptable.

5. Clear provenance.
   Strategy surfaces must show both model values and effective values so the
   operator can see exactly what was changed.

6. Resettable at any time.
   A player override can be removed individually, or the whole slate can be
   cleared in one action.

## 4. Current Repo Starting Point

The repo already contains a usable seed implementation in:

- `projections/api/user_overrides.py`
- `projections/api/optimizer_api.py`
- `projections/api/optimizer_service.py`
- `web/minutes-dashboard/src/api/optimizer.ts`

Current useful behavior already present:

- slate-scoped persistence keyed by `game_date` + `draft_group_id`
- atomic JSON writes with backup
- effective minutes / FPTS recomputation
- clear-one and clear-all semantics

Current gaps relative to this contract:

- naming still reflects "legacy user overrides"
- contest sim does not yet consume the same layer
- semantics are underspecified in docs
- current implementation is mean-based rather than world-based
- diagnostics are thinner than they should be

## 5. Target Policy

Allowed to affect downstream strategy surfaces:

- effective minutes used by optimizer decision metrics
- effective FPTS used by optimizer decision metrics
- adjusted player worlds and world summaries used by optimizer / contest sim
- downstream displays derived from those effective values

Not allowed to affect canonical live outputs:

- live minutes features
- live minutes/rates/fpts model inputs
- published projection parquet files
- live run manifests except optional read-only metadata that strategy overrides
  are active

## 6. Persistence Contract

### 6.1 Short-term canonical storage

Short term, reuse the current slate-level path:

`$PROJECTIONS_DATA_ROOT/user_overrides/<DATE>/dg_<DRAFT_GROUP_ID>.json`

This is intentionally:

- slate-scoped
- persistent across reruns
- independent of projection `run_id`

### 6.2 Recommended long-term naming

The implementation may continue using the existing path initially, but the
product and docs should refer to this as the **strategy override layer** rather
than "legacy user overrides".

If a storage rename is done later, preserve backward compatibility with the
existing file shape during migration.

### 6.3 Slate lifecycle

Strategy overrides remain active until one of these happens:

- the operator clears the player override
- the operator clears all overrides for the slate
- an optional expiry policy is reached

They must not disappear merely because:

- a new live run was published
- the selected `run_id` changed
- the API process restarted

## 7. Data Model

### 7.1 Per-player override record

Required fields:

- `player_id`
- `minutes_delta`
- `fpts_delta`
- `updated_at`

Recommended metadata fields for the next revision:

- `player_name`
- `team`
- `entered_by`
- `reason_text`
- `apply_to`
  - `optimizer`
  - `contest_sim`
  - `both`

### 7.2 Semantics

The persisted operator intent should be **delta-based**, not target-based:

- `minutes_delta`: additive adjustment to the player's active-world minutes
- `fpts_delta`: additive adjustment to the player's active-world fantasy-point
  output

Why deltas:

- they survive reruns better than absolute targets
- `+3 minutes` still means `+3 minutes` after a new model run
- they express operator disagreement with the model more naturally

UI copy may present these as "boost / nerf" controls.

### 7.3 Supported combinations

The first implementation must support all three combinations below:

1. `minutes_delta` only
   - adjust active-world minutes by the delta
   - scale active-world FPTS from the minutes change

2. `fpts_delta` only
   - keep minutes worlds unchanged
   - scale active-world FPTS to hit the new adjusted mean

3. both `minutes_delta` and `fpts_delta`
   - apply `minutes_delta` first
   - then apply the direct FPTS delta on top of the already adjusted active
     worlds

This gives the operator two useful control modes immediately in v1:

- minutes-based adjustment that scales with FPPM
- direct FPTS adjustment

## 8. Runtime Semantics

### 8.1 Baseline values

Each consumer must first load the selected baseline run and derive:

- `model_minutes`
- `model_fpts`
- `model_own` if available
- `base_fppm`
- baseline player worlds for the selected `worlds_source`

`base_fppm` should be derived from the same decision-metric columns used by the
consumer, preferring unconditional means when available.

### 8.2 Adjustment order

When overrides are enabled:

1. resolve the selected baseline run
2. load baseline worlds
3. apply per-player strategy overrides to those worlds
4. derive adjusted world summaries from the transformed worlds
5. use adjusted summaries/worlds in optimizer and contest sim

### 8.3 Active-world-only rule

Strategy overrides must preserve the baseline activity structure:

- worlds where the player is inactive or has zero fantasy points remain zero
- overrides do not create new active worlds
- overrides do not change play probability unless a separate availability
  override exists

This keeps the strategy layer separate from canonical `force_out` / `force_in`
logic.

### 8.4 Minutes-based override

When `minutes_delta` is set:

- adjust only active worlds
- preserve zero/DNP worlds
- update fantasy-point worlds proportionally so tails move with minutes

Recommended transformation for a player with baseline per-world minutes
`m_w` and fantasy points `f_w`:

- define active worlds as those with `f_w > 0` or `m_w > 0`
- for active worlds:
  - `m'_w = max(0, m_w + minutes_delta)`
  - `f'_w = f_w * (m'_w / max(m_w, floor_minutes))`
- for inactive worlds:
  - `m'_w = m_w`
  - `f'_w = f_w`

Recommended `floor_minutes`: small positive constant such as `1.0`.

This is intentionally a player-level transform. It does **not** redistribute
minutes to teammates or attempt to re-solve team feasibility.

### 8.5 Direct FPTS override

When `fpts_delta` is set:

- minutes worlds remain unchanged unless `minutes_delta` is also set
- active fantasy-point worlds are rescaled so the adjusted mean reflects the
  direct belief shift
- zero/DNP worlds remain zero

Recommended transformation:

- compute baseline active-world mean `mu_active`
- target `mu'_active = max(0, mu_active + fpts_delta)`
- active-world scale `s = mu'_active / max(mu_active, floor_fpts)`
- for active worlds:
  - `f'_w = f_w * s`
- for inactive worlds:
  - `f'_w = f_w`

Recommended `floor_fpts`: small positive constant such as `1.0`.

### 8.6 Tail behavior

Strategy overrides should affect tails.

Rationale:

- if the operator believes a player should play more minutes, that belief should
  raise not only the mean but also the active-game ceiling/floor distribution
- contest sim and optimizer should see the same shifted player distribution

Therefore:

- `minutes_delta` changes active-world tails
- `fpts_delta` changes active-world tails
- inactive/DNP mass is preserved unless availability is separately overridden

### 8.7 No live-model mutation

Strategy overrides must not:

- write back into feature frames
- mutate live parquet artifacts
- trigger minutes/rates rescoring
- change published run pointers

### 8.8 World summaries

After transforming worlds, consumers should recompute adjusted summaries from
the adjusted worlds, for example:

- adjusted mean
- adjusted standard deviation
- adjusted `p10/p50/p90/p95`

Optimizer mean-mode may use these adjusted summary metrics. Optimizer
world-sample mode and contest sim should use the adjusted worlds directly.

### 8.9 Contest-sim behavior

Contest sim should consume the same adjusted-world layer when strategy overrides
are enabled.

In the first implementation, this means:

- candidate lineups are evaluated against the selected worlds source **after**
  world transformation
- player-level effective means shown in the UI are derived from those adjusted
  worlds
- optimizer and contest sim must not disagree on the adjusted player means/tails

This transform is private to the requesting surface and must not overwrite the
stored baseline worlds artifact.

## 9. API Contract

### 9.1 Short-term API

Short term, reuse the existing optimizer override endpoints:

1. `GET /api/optimizer/overrides`
2. `PUT /api/optimizer/overrides`
3. `DELETE /api/optimizer/overrides/{player_id}`
4. `DELETE /api/optimizer/overrides`

Recommended product rename in UI/API copy:

- "Strategy Overrides"
- not "My Proj"
- not "manual live overrides"

The first implementation must accept and persist both:

- `minutes_delta`
- `fpts_delta`

### 9.2 Request semantics

For each player:

- `minutes_delta` sets an additive active-world minutes adjustment
- `fpts_delta` sets an additive active-world fantasy-points adjustment
- both null removes that player override

Example request body:

```json
{
  "date": "2026-03-02",
  "draft_group_id": 140001,
  "overrides": [
    {
      "player_id": "203999",
      "minutes_delta": 3.0,
      "fpts_delta": null
    },
    {
      "player_id": "1629029",
      "minutes_delta": null,
      "fpts_delta": 4.5
    }
  ]
}
```

### 9.3 Response semantics

Responses should expose enough information for the UI to render both baseline
and effective values:

- `model_minutes`
- `model_fpts`
- `effective_minutes`
- `effective_fpts`
- `override_minutes_delta`
- `override_fpts_delta`
- `has_override`
- `used_fppm_fallback`

## 10. CLI Contract

Recommended CLI entry point:

`uv run python -m projections.cli.strategy_overrides`

Recommended commands:

```bash
uv run python -m projections.cli.strategy_overrides set-minutes \
  --date 2026-03-02 \
  --draft-group-id 140001 \
  --player-id 203999 \
  --minutes-delta 3.0

uv run python -m projections.cli.strategy_overrides set-fpts \
  --date 2026-03-02 \
  --draft-group-id 140001 \
  --player-id 1629029 \
  --fpts-delta 4.5

uv run python -m projections.cli.strategy_overrides list \
  --date 2026-03-02 \
  --draft-group-id 140001

uv run python -m projections.cli.strategy_overrides clear-player \
  --date 2026-03-02 \
  --draft-group-id 140001 \
  --player-id 203999

uv run python -m projections.cli.strategy_overrides clear-slate \
  --date 2026-03-02 \
  --draft-group-id 140001
```

## 11. Consumer Integration Points

### 11.1 Optimizer

Primary integration point:

- `projections/api/optimizer_service.py`

Required behavior:

- load slate strategy overrides when requested
- load adjusted worlds/world summaries when overrides are enabled
- expose both model and effective values in the player pool
- use adjusted summary metrics in mean-mode optimization
- use adjusted worlds in world-sample optimization
- leave canonical run artifacts unchanged

### 11.2 Contest sim

Primary integration points:

- `projections/api/contest_sim_api.py`
- `projections/contest_sim/contest_sim_service.py`

Required behavior:

- consume the same strategy override layer when enabled
- use the same slate/date/draft-group scope as optimizer
- keep worlds-source selection explicit (`gtv2` vs `sim_v2`)
- apply overrides to the loaded player worlds before lineup scoring
- avoid an optimizer/contest-sim mismatch where one surface applies the layer
  and the other does not

### 11.3 Entry flows

Important integration points:

- `projections/api/entry_manager_api.py`
- saved-build loaders for optimizer and contest sim

Required behavior:

- preserve effective values in saved build metadata where practical
- make it obvious whether a build was generated with strategy overrides active

## 12. UI Contract

The operator needs a separate downstream control surface, distinct from
`GameView`.

Required behaviors:

- show baseline minutes and FPTS
- show effective minutes and FPTS after override
- show that tails/EV are computed from adjusted worlds when overrides are on
- allow editing `minutes_delta` or `fpts_delta`
- support using both fields on the same player in the same slate
- allow per-player clear
- allow slate-wide reset
- indicate that changes persist across model reruns until cleared
- indicate that these controls affect optimizer / contest sim only

Recommended labels:

- "Strategy Overrides"
- "Effective Minutes"
- "Effective FPTS"
- "Reset Slate"

## 13. Validation Rules

Required validation:

- player must exist in the slate player pool
- override must be scoped to one player and one slate
- duplicate writes for the same player should upsert
- `minutes_delta` and `fpts_delta` must be finite
- if both are null, treat the write as a clear action

Recommended soft guards:

- warn when implied active-world minutes become implausible
- warn when direct FPTS deltas are implausibly large
- warn when the player is inactive or not on the current slate

## 14. Diagnostics

Every strategy-enabled surface should expose:

- override count
- per-player override state
- model vs effective values
- `used_fppm_fallback`
- active override revision / updated timestamp
- whether adjusted worlds are active for the current run/request

Suggested metadata on saved builds and job payloads:

- `strategy_overrides_active`
- `strategy_override_count`
- `strategy_override_revision`

## 15. Tests

Minimum test cases:

1. `minutes_delta` persists across reload and adjusts active worlds only
2. `fpts_delta` persists across reload and rescales active FPTS worlds only
3. combined `minutes_delta` + `fpts_delta` on the same player applies in the
   specified order
4. inactive/DNP worlds remain zero under strategy overrides
5. new projection run changes baseline but does not clear the override
6. clear-player removes only the targeted override
7. clear-slate removes all overrides
8. optimizer and contest sim apply the same adjusted-world layer when enabled
9. strategy overrides do not mutate canonical live outputs
