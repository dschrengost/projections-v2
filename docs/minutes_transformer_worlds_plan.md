# Minutes: Transformer + Worlds Quantiles Plan

This is the “concrete, do-the-work” plan to fix minutes predictions by making the **rotation_set transformer** the single canonical allocator for production minutes (no RMH), and treating **worlds-derived minutes quantiles** as the only authoritative quantiles for UI/optimizer.

This document is intentionally implementation-oriented: contracts, invariants, file touch-points, config knobs, rollout/rollback.

## What’s running now (and where it’s decided)

- **Minutes scoring selection (RMH vs rotation_set)**
  - `prefect_flows/live_nba_pipeline.py:345` (`score_minutes_task`)
  - RMH enablement is read from `config/rmh_current_run.json` via `prefect_flows/live_nba_pipeline.py:332` (`_is_rmh_enabled`).
  - If RMH is enabled, the flow runs `projections.cli.score_minutes_rmh_v1`; otherwise it runs `projections.cli.score_minutes_rotation_set_v1` (`prefect_flows/live_nba_pipeline.py:375` / `prefect_flows/live_nba_pipeline.py:395`).
- **Effective minutes layer**
  - Built immediately after scoring in `prefect_flows/live_nba_pipeline.py:451` (`effective_minutes_task`), using `projections.pipeline.effective_inputs.write_effective_minutes_layer(...)`.
- **Sim worlds (source of minutes_sim_pXX)**
  - Invoked in `prefect_flows/live_nba_pipeline.py:817` (`run_sim_task`) via `scripts.sim_v2.run_sim_live`.
  - Worlds generator: `scripts/sim_v2/generate_worlds_fpts_v2.py` already computes:
    - `minutes_sim_p10/p50/p90` (conditional-on-playing worlds) and `_uncond` variants (DNP=0).
- **Unified projections (what the dashboard prefers)**
  - Produced by `projections.cli.finalize_projections`.
  - Includes minutes quantiles from sim (`minutes_sim_p10/p50/p90`) and minutes artifacts (`minutes_p10/p50/p90`) (see `projections/cli/finalize_projections.py:33` for `MINUTES_COLUMNS` and `SIM_COLUMNS`).

## North Star (end state)

- **Single canonical allocator**: rotation_set transformer produces minutes mean allocation per team (sums to 240 by construction).
- **Explicit gating**: model produces `in_rotation_prob` (or `plays_prob`) and a derived `in_rotation_flag` with a clearly defined threshold and semantics.
- **No RMH in production**: RMH can remain as a shadow job for monitoring only.
- **Quantiles come from worlds**: `minutes_pXX` in UI/optimizer should be sourced from sim-derived quantiles (`minutes_sim_pXX`), not model-emitted tails or baseline tail mapping.
- **No silent nonsense**: parity audits, provenance stamps, and invariants block publishing when inputs or outputs violate contracts.

## Non-negotiable contracts (hard requirements)

### 1) Provenance stamps (required to publish)

Published minutes artifacts must carry enough information to answer: “which model ran, with which feature schema, producing which contract version?”

**Required fields (either as parquet columns or a sidecar JSON next to minutes artifacts):**
- `minutes_source="rotation_set"`
- `rotation_set_model_dir` and a stable `rotation_set_model_run_id` (e.g. model folder name or manifest id)
- `feature_schema_hash` (hash of required feature column list + dtype expectations)
- `model_schema_hash` (hash of model config + feature_columns.json + normalization stats files)
- `minutes_contract_version` and `minutes_contract_hash`
- `worlds_quantiles_version` (because minutes quantiles are worlds-derived)

Publishing should **fail** if these stamps are missing.

### 2) Gating contract is explicit

Every player row must include:
- `in_rotation_prob` (0–1), derived from the model’s gate head (preferred) or from a documented deterministic proxy (temporary fallback only).
- `in_rotation_flag` derived from `in_rotation_prob` via a configured threshold.

Rules:
- OUT players must be hard-zeroed:
  - `in_rotation_prob = 0`, `in_rotation_flag = 0`, `minutes_mean = 0`.
- The definition of “rotation” must be written down once and reused everywhere (minutes, sim, UI, optimizer):
  - recommended: `in_rotation_flag := in_rotation_prob >= 0.5` (or threshold from config)
  - avoid mixing “>=5 minutes” semantics with probability semantics without a translation layer.

### 3) Integrity invariants (abort publish or degrade loudly)

These are not “nice to have”; they prevent bad slates from reaching the dashboard.

**Hard-fail invariants (abort publish):**
- Team sums for `minutes_mean` (or `minutes_p50`) equal 240 within tight tolerance.
- OUT players have 0 minutes.
- `game_date` exists and matches the run date (fixes `KeyError: game_date` class regressions).

**Soft-fail invariants (publish with explicit degraded flags + diagnostics):**
- Rotation size (`K(minutes >= X)` or `K(in_rotation_flag)`) outside a reasonable band (team-game level).
- Distribution sanity (top1/top8 share, entropy) outside plausible ranges.

### 4) Worlds-derived minutes quantiles are authoritative

Downstream consumers must prefer:
- `minutes_sim_p10/p50/p90` (and `_uncond`) from sim outputs
over
- `minutes_p10/p50/p90` from the minutes scorer.

This is already largely true in the dashboard path (unified projections include `minutes_sim_*`). The plan is to make this a documented contract and ensure any legacy consumers are updated.

## Implementation plan (minimal, safe changes)

### A) Transformer-only minutes scoring (no RMH)

**Goal:** the production flow always uses rotation_set scoring; RMH stays shadow-only.

**Edits**
- `prefect_flows/live_nba_pipeline.py`
  - Change `score_minutes_task` to refuse RMH as a primary path in production:
    - If `config/rmh_current_run.json.enabled=true`, log a warning and either:
      - hard-fail (preferred in PROD), or
      - ignore RMH and continue with rotation_set (safer short-term).
  - Keep `rmh_shadow_minutes_task` gated behind `RMH_SHADOW_ENABLED=1` as today.
- `config/rmh_current_run.json`
  - Set `"enabled": false` in production deployments (keep file for shadow workflows).

### B) Make rotation_set the primary scorer (not an overlay)

**Goal:** stop running baseline minutes + blending; output is transformer minutes.

**Edits**
- `projections/cli/score_minutes_rotation_set_v1.py`
  - Add a new mode/flag (e.g. `--transformer-only`) that:
    - skips baseline minutes_v1 scoring entirely
    - builds rotation_set live features
    - runs transformer inference
    - writes `minutes.parquet` in the canonical location
  - Maintain a guarded fallback path only when parity checks fail (see Workstream E).

### C) Expose gate outputs and publish gating fields

**Goal:** produce `in_rotation_prob`/`in_rotation_flag` from the transformer gate head.

**Edits**
- `projections/rotation/set_model.py`
  - Extend the predictor to optionally return aux outputs from `forward_with_aux(...)`:
    - `gate_logits` (or already-mixed gate logits for MoE)
    - `share_logits`
    - router weights (optional)
  - Public API sketch:
    - `predict_minutes(..., with_aux: bool = False) -> pd.DataFrame`
    - or a new helper `predict_minutes_with_aux(...)`.
- `projections/cli/score_minutes_rotation_set_v1.py`
  - Compute:
    - `in_rotation_prob = sigmoid(gate_logits)` (clipped)
    - `in_rotation_flag = in_rotation_prob >= threshold`
  - Enforce OUT hard-zeroing for minutes + gating fields.

### D) Worlds-derived quantiles publishing (use what we already compute)

**Goal:** minutes quantiles shown in UI/optimizer come from sim worlds.

**Current state:** sim projections already include `minutes_sim_p10/p50/p90` (and `_uncond`) via `scripts/sim_v2/generate_worlds_fpts_v2.py`.

**Minimal change:** document + enforce consumer preference
- `projections/api/minutes_api.py` and `projections/api/optimizer_service.py` already expose these fields.
- Add a contract note + validation:
  - if unified projections exist, `minutes_sim_*` should be present; if missing, emit a loud warning and include a degraded flag in the response payload.

Optional improvement (later): write a small “minutes worlds quantiles” parquet next to minutes artifacts for non-unified consumers, but keep unified projections as the canonical UI path.

### E) Parity + coverage audit tooling (hard gate)

**Goal:** inference must match training feature schema exactly; no silent fill-and-continue.

**Edits**
- Add a parity report generator (script + shared helper), e.g.:
  - `scripts/diagnostics/rotation_set_minutes_parity_audit.py`
  - reusable helper in `projections/rotation/` (to share with scoring CLI)

**Parity checks**
- Missing required columns (hard fail).
- Dtype mismatches (hard fail unless explicitly allowed).
- Missingness rates over thresholds on key joins (injuries, odds, starter flags, priors) (soft/hard depending on severity).

**Outputs**
- Write `parity_report.json` next to `minutes.parquet` and copy into gold run dir.
- If parity fails:
  - preferred: abort publish (production-safe, avoids bad dashboards)
  - fallback option: publish with `minutes_degraded=true` and explicit reason codes.

### F) Guardrails that don’t mask failures

**Goal:** keep “contract boundary” guardrails, but ensure they do not flatten or hide gating/feature failures.

**Edits**
- `projections/pipeline/health.py`
  - Add a stricter check for “rotation_set minutes contract”, separate from `require_minutes_sanity`:
    - team sum tolerance tighter
    - OUT minutes = 0
    - rotation size plausible band
    - top8 share plausible band
- Call this check in:
  - `projections/cli/score_minutes_rotation_set_v1.py` before writing/publishing outputs
  - `prefect_flows/live_nba_pipeline.py` before publishing to gold

### G) Config knobs (new / updated)

**Rotation-set live config** (`config/rotation_set_minutes_live.json`)
- `enabled` (existing)
- `model_dir` (existing)
- `mode` (new): `"primary"` vs `"overlay"` (default `"primary"` for this plan)
- `in_rotation_prob_threshold` (new): default `0.5`
- `parity` block (new):
  - `fail_on_missing_columns` (default `true`)
  - `max_missingness_injuries` / `max_missingness_odds` (defaults tuned conservatively)
  - `max_missingness_priors` (if priors are required)
- `minutes_contract_version` (new)
- `worlds_quantiles_version` (new; ties to sim profile version)

**Sim profile config** (`config/sim_v2_profiles.json`)
- Confirm we are using `sim_v3` profile and `preserve_input_rotation=true`.
- Add a note that `minutes_sim_*` fields are the authoritative minutes quantiles.

## Tests + commands

### Tests to add

- `tests/rotation/test_rotation_set_contract.py`
  - team sums to 240 for synthetic toy inputs
  - OUT players hard-zero
  - gating outputs in [0,1] and derived flag matches threshold
- `tests/rotation/test_rotation_set_parity_gate.py`
  - missing required feature columns → hard fail (no publish)
- `tests/sim_v2/test_worlds_minutes_quantiles_present.py` (light integration)
  - run a small-n_worlds sim on fixture data and assert `minutes_sim_p50` columns exist in sim projections output

### Commands

- Lint: `uv run ruff check .`
- Tests: `uv run pytest -q`
- Local sim run (small): `uv run python -m scripts.sim_v2.run_sim_live --run-date 2026-01-24 --profile sim_v3 --num-worlds 2000 --run-id <run_id> --minutes-run-id <run_id> --rates-run-id <run_id>`

## Rollout + rollback

### Rollout

1) Land the transformer-only scorer behind config:
   - `rotation_set_minutes_live.json.mode="primary"`
   - `rmh_current_run.json.enabled=false`
2) Deploy via the standard deploy script:
   - `scripts/deploy/deploy_live.sh`
3) Monitor:
   - Prefect logs for `[minutes-contract]`, `[minutes-parity]`
   - Dashboard sanity (team names, Vegas lines, rotation realism)

### Rollback

If parity/invariants fail in production:
- Immediate: set `rotation_set_minutes_live.json.enabled=false` (or `mode="overlay"`) and redeploy.
- If needed: temporarily re-enable RMH scoring (not desired long-term), but only with explicit logging and stamps so it’s never ambiguous what ran.

## Gotchas to address explicitly (do not hand-wave)

- Do not “fill missing continuous features with 0 and proceed” in production scoring.
- Define `in_rotation` semantics once and enforce it across minutes, sim, UI, and optimizer.
- Ensure `game_date` exists in all published artifacts (prevents `KeyError: game_date` class regressions).
- Keep reconciliation as a boundary constraint (240), but emit diagnostics showing whether it’s a no-op vs masking a failure.

