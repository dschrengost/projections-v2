# Manual Override Contract

This document defines the practical implementation contract for manual live
overrides in `projections-v2`.

It is the companion to
`docs/pipeline/LIVE_PIPELINE_PRODUCTION_SPEC.md`, which defines the policy:

- only audited manual `IN` / `OUT` overrides should affect canonical live
  projections
- arbitrary minute or fantasy-point boosts should not affect canonical live
  outputs
- boost/preference logic belongs in optimizer / contest sim surfaces

## 1. Scope

This contract covers:

- operator-entered live availability overrides
- API and CLI interfaces
- persistence format
- pipeline integration points
- downstream metadata requirements

This contract does not cover:

- optimizer-only exposure boosts
- ownership overrides
- arbitrary point-estimate minute edits
- contest strategy controls

## 2. Current State In Repo

Existing override surfaces:

- authorized GameView / ops API:
  `projections/api/ops_api.py`
- current override persistence and application helpers:
  `projections/ops/overrides.py`
- legacy effective inputs layer:
  `projections/pipeline/effective_inputs.py`
- legacy live flow that materializes effective inputs:
  `prefect_flows/live_nba_pipeline.py`
- v3 live flow:
  `prefect_flows/live_nba_pipeline_v3.py`

Important current-state constraint:

- the existing ops override system is broader than the desired production
  policy. It supports minute deltas, targets, rates edits, and other fields
  that we do not want in the canonical live projection path.

## 3. Target Policy

Allowed to affect canonical live projections:

- `force_out`
- `force_in`

Not allowed to affect canonical live projections:

- `minutes_delta`
- `minutes_target`
- direct `minutes_p50` edits
- direct rate/fpts edits
- optimizer-specific ownership or exposure changes

Operator judgment that is not availability-related should remain in:

- optimizer controls
- contest sim portfolio controls
- downstream build preferences

## 4. Shipping Strategy

### 4.1 Stage 1: Reuse current ops/GameView path

Shortest path to production:

- keep the existing GameView API entry point
- keep current override persistence location
- narrow the live effect to availability-only semantics

That means:

- `projections/api/ops_api.py` continues to accept writes from the current UI
- `projections/ops/overrides.py` remains the persistence layer
- `projections/cli/build_minutes_live.py` becomes the canonical place where
  active live availability overrides are applied for v3

### 4.2 Stage 2: Clean contract split

Once the v3 path is stable, consider splitting:

- live manual availability overrides
- optimizer / contest-sim preference controls
- legacy broad ops overrides

That cleanup can introduce a dedicated API/path if the current `overrides_v1`
shape proves too permissive.

## 5. Persistence Contract

### 5.1 Short-term canonical storage

Short term, reuse the existing path:

`$PROJECTIONS_DATA_ROOT/artifacts/ops/overrides_v1/game_date=<DATE>/overrides.json`

But treat only availability semantics as production-authoritative for the live
pipeline.

### 5.2 Canonical logical fields

Every active live availability override should resolve to these logical fields:

- `override_id`
- `game_date`
- `game_id`
- `player_id`
- `player_name`
- `team_id`
- `team_tricode`
- `override_type`
  - `force_out`
  - `force_in`
- `reason_code`
- `reason_text`
- `source_label`
- `entered_by`
- `created_ts`
- `effective_ts`
- `expires_ts`
- `active`
- `cleared_ts`
- `cleared_by`

### 5.3 Mapping from current ops payload

Short-term mapping from existing `overrides_v1` payload:

- `status="out"` or `ops_depth_role="out"` maps to `force_out`
- explicit clear/removal maps to clearing an existing override
- `force_active=true` may temporarily map to `force_in`, but should be treated
  as an operator action that clears a manual `force_out`, not as a general
  source override

Fields like `minutes_delta`, `minutes_target`, `pred_*`, or direct minute
quantiles should not be consumed by the v3 live projection path.

## 6. API Contract

### 6.1 Short-term API

Reuse the current GameView endpoint in
`projections/api/ops_api.py` for the first implementation wave.

Practical rule:

- the endpoint may still accept the broader payload for compatibility
- the live v3 pipeline should only honor availability fields

### 6.2 Recommended target API

Recommended explicit API surface:

1. `POST /api/ops/manual-availability-overrides`
   - create or update a `force_out` / `force_in`
2. `GET /api/ops/manual-availability-overrides`
   - list active and recent overrides for a date
3. `DELETE /api/ops/manual-availability-overrides/{override_id}`
   - clear a specific override

Recommended request body for create/update:

```json
{
  "game_date": "2026-02-27",
  "game_id": "22500858",
  "player_id": "201935",
  "player_name": "James Harden",
  "team_tricode": "CLE",
  "override_type": "force_out",
  "reason_code": "operator_report",
  "reason_text": "Beat writer reported player out before official feed.",
  "source_label": "twitter",
  "entered_by": "daniel",
  "expires_ts": "2026-02-28T00:00:00Z"
}
```

Recommended response fields:

- normalized override payload
- `override_id`
- `active`
- `created_ts`
- `effective_ts`
- `material_change_detected`

## 7. CLI Contract

Recommended CLI entry point:

`python -m projections.cli.manual_overrides`

Recommended commands:

```bash
uv run python -m projections.cli.manual_overrides add-out \
  --date 2026-02-27 \
  --player-id 201935 \
  --game-id 22500858 \
  --reason-code operator_report \
  --reason-text "Beat writer reported player out" \
  --source-label twitter \
  --entered-by daniel

uv run python -m projections.cli.manual_overrides add-in \
  --date 2026-02-27 \
  --player-id 201935 \
  --game-id 22500858 \
  --reason-code source_correction \
  --reason-text "Manual out cleared after official active confirmation" \
  --source-label rotowire \
  --entered-by daniel

uv run python -m projections.cli.manual_overrides list --date 2026-02-27

uv run python -m projections.cli.manual_overrides clear \
  --date 2026-02-27 \
  --override-id <override_id> \
  --cleared-by daniel
```

## 8. Pipeline Integration Points

### 8.1 Canonical integration point for v3

Primary target:

- `projections/cli/build_minutes_live.py`

Required behavior:

1. load active manual availability overrides for the slate
2. apply them after source ingestion but before final availability flags are
   written
3. write explicit override metadata columns or diagnostics so the scorer and
   downstream readers know the player was manually forced

Suggested output fields on the live minutes features frame:

- `manual_override_type`
- `manual_override_reason_code`
- `manual_override_source_label`
- `manual_override_entered_by`
- `manual_override_active`
- `manual_override_used`

### 8.2 Legacy compatibility path

Existing legacy/effective-inputs path:

- `projections/pipeline/effective_inputs.py`
- `prefect_flows/live_nba_pipeline.py`

Required near-term rule:

- keep this path working for compatibility
- stop treating broad minute/rate override fields as acceptable production live
  behavior once v3 manual availability is live

### 8.3 Flow-level wiring

Files that should reflect override state in manifests and digests:

- `prefect_flows/live_nba_pipeline_v3.py`
- `prefect_flows/live_nba_pipeline.py`

Required behavior:

- include active override digest in run manifest
- treat add/clear of an active override as a material game change
- surface override usage in run summaries and diagnostics

## 9. Downstream Contract

Downstream consumers should read override metadata, not re-apply business logic.

Important consumers:

- `projections/cli/finalize_projections.py`
- `projections/api/optimizer_service.py`
- dashboard/GameView surfaces

Required behavior:

- expose `manual_override_used`
- expose the override type and note/source where appropriate
- do not re-run override mutation logic independently in each consumer

## 10. Runtime Semantics

### 10.1 `force_out`

When active:

- status becomes `OUT`
- `is_out=1`
- `play_prob=0`
- minute distributions are zeroed
- downstream optimizer/sim eligibility reflects the out status

### 10.2 `force_in`

When active:

- clear a prior manual `force_out`
- rerun normal source precedence
- do not use it as a blanket "trust manual over all sources forever" switch

### 10.3 Expiry

Overrides must not disappear on rerun.

They should remain active until:

- explicitly cleared, or
- `expires_ts` / lock expiry is reached

## 11. Validation Rules

Required validation:

- player must exist on the slate
- game must exist on the slate
- override must be scoped to one player and one game
- duplicate active overrides for the same player/game should upsert, not fork
- unsupported fields should be rejected or ignored in the live v3 path

## 12. Audit And Diagnostics

Every override-driven run should record:

- active override count
- affected games
- override payload digest
- override rows used in the run
- operator/source metadata

Suggested run-level files:

- `manual_override_report.json`
- override block in `manifest.json`
- override block in run summary / dashboard metadata

## 13. Tests

Minimum test cases:

1. `force_out` zeroes player in live features
2. `force_out` survives rerun and is not silently overwritten
3. clear action removes the manual out and restores source precedence
4. override expiry at lock works
5. unsupported boost fields do not affect v3 live outputs
6. material-change detection fires when override is added or cleared
