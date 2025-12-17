# Ownership Codex Log

## 2025-12-17 — Phase 1: Baseline + Data Audit

### Repo Mapping (Ownership)
- Training data builders:
  - `scripts/ownership/build_ownership_training_base.py` (LineStar proj + LineStar actual; adds injury features)
  - `scrapers/dk_contests/build_ownership_data.py` (DK contest results → `bronze/dk_contests/ownership_by_slate/*.parquet`)
  - `scripts/ownership/build_ownership_dk_base.py` (DK actual + LineStar features via name match → `gold/ownership_dk_base/ownership_dk_base.parquet`)
  - `scripts/ownership/merge_ownership_bases.py` (DK base preferred on overlapping dates → `gold/ownership_merged_base/ownership_merged_base.parquet`)
- Model code:
  - Training: `scripts/ownership/train_ownership_v1.py` (LightGBM regressor; feature sets v1–v6; optional logit target)
  - Inference: `projections/ownership_v1/*` + `projections/cli/score_ownership_live.py`
- Production model run:
  - `projections/cli/score_ownership_live.py` currently uses `PRODUCTION_MODEL_RUN = "dk_only_v4"` (docs under `docs/ownership/README.md` still reference an older run id).

### Fixed Validation Slice (Do Not Move)
- Slice config committed at `config/ownership_eval_slice.json`.
- Slice definition: 12 DK slates (slate_ids) spanning `2025-11-24..2025-12-06`.
- Target sum for constraint checks: DK classic `800%`.

### Baseline Repro Command (Train + Eval)
- Train (DK base, fixed date split):
  - `uv run python scripts/ownership/train_ownership_v1.py --run-id <RUN_ID> --training-base /home/daniel/projections-data/gold/ownership_dk_base/ownership_dk_base.parquet --feature-set v4 --sample-weighting --val-start-date 2025-11-24`
- Evaluate (fixed slice):
  - `uv run python scripts/ownership/evaluate_ownership_v1.py --run-id <RUN_ID> --out-md reports/ownership_eval_before_after.md --out-json reports/ownership_eval_runs/<RUN_ID>.json`

### Baseline Metrics (Before)
- Baseline metrics written to `reports/ownership_eval_before_after.md` for run `dk_only_v4`.

### Data Audit Findings (High-Leverage Issues)

#### 1) DK ownership labels do not obey the 800% sum constraint
- `scrapers/dk_contests/build_ownership_data.py` aggregates ownership by entry-weighted average **only over contests where a player appears**.
- When contest player pools differ within a slate cluster, “missing” players are implicitly dropped rather than treated as 0% in that contest.
- Result: label sums per slate can exceed 800% materially (example: `2025-11-26_0` raw sum ≈ 912.7%).

#### 2) DK base feature join has two major failure modes
- Current join in `scripts/ownership/build_ownership_dk_base.py`: `(game_date, player_name_norm)` only.
- **Multi-slate ambiguity (same date, multiple slates)**:
  - 9/36 dates have >1 slate in DK base.
  - ~14.7% of DK base rows are players appearing in >1 slate on the same date (ambiguous join key).
- **Date alignment/timezone drift**:
  - For some slates, LineStar `game_date` appears shifted by +1 day vs DK (likely UTC vs local date semantics).
  - This causes missing *high-owned* players (not just long-tail) for affected slates.
  - Example (validation slice): `2025-11-26_0` has ~50% ownership-mass coverage after joining; missing players include multiple 20–40% owned plays.

#### 3) Coverage varies by slate; several slates have large missing ownership mass
- Across 45 DK base slates: 11 slates have <90% ownership-mass coverage vs raw DK; 4 slates are <60%.
- This distorts both training (targets/features) and evaluation (actual sums & chalk set composition).

#### 4) Injury snapshots have `as_of_ts` but training builders don’t enforce “<= lock”
- `bronze/injuries_raw/*/injuries.parquet` includes `as_of_ts`.
- `scripts/ownership/build_ownership_training_base.py` joins injuries by `report_date` only (no lock-time gating).
- Live scoring keeps the latest snapshot for the day, but does not gate by lock time (relying on “run before lock” and lock persistence).

### Tests / Determinism Notes
- Full `uv run pytest -q` currently errors during collection due to unrelated minutes_v1 modules missing (`projections.minutes_v1.reconciliation`, `RollingCalibrationConfig`, etc.).
- Ownership-specific tests pass (`tests/ownership_v1/test_calibration.py`, `tests/ownership_v1/test_evaluation.py`).

### Phase 2 Priority Hypotheses
1) Fix DK label aggregation to treat absent players as 0% per contest (restores 800%-sum property per contest/slate).
2) Fix DK↔LineStar alignment:
   - Resolve game_date timezone mismatch.
   - Replace `(game_date, player_name_norm)` join with a slate-aware join (use slate overlap / slate_size matching).
3) After data is sane, iterate model improvements (logit target, monotonic constraints, calibration/normalization with caps).

## 2025-12-17 — Phase 2: Data Fixes

### Step 1: Fix DK slate aggregation (zero-fill across contests)
- Change: `scrapers/dk_contests/build_ownership_data.py` now treats players missing from a contest as **0%** when aggregating ownership across contests in a slate cluster.
- Why: without zero-fill, per-player denominators exclude contests where a player is absent, inflating ownership and causing slate sums to exceed 800%.
- Added: `slate_entries`, `slate_num_contests` metadata to the aggregated per-slate parquet rows.
- Tests: `tests/test_scrapers/test_build_ownership_data.py` covers zero-fill + entry weighting behavior.

