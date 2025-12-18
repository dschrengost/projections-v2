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

### Step 2: Slate-aware DK↔LineStar matching (subset-aware overlap)
- Change: `scripts/ownership/build_ownership_dk_base.py` now maps each DK `slate_id` to the best matching LineStar `slate_id` using player-pool overlap (subset-aware `overlap_coeff = |A∩B| / min(|A|,|B|)`), then joins players within that mapped slate by normalized name.
- Why: the previous `(game_date, player_name_norm)` join was ambiguous on multi-slate dates and brittle to date drift; it also produced severe chalk-mass dropouts on some slates.
- Result (current local lake): `76` DK slates → `36` matched to LineStar; `3,999` joined rows written to `gold/ownership_dk_base/ownership_dk_base.parquet` (dates `2025-10-25..2025-12-06`).
- Tests: added `tests/ownership_v1/test_dk_linestar_matching.py`; ran focused suite (`tests/ownership_v1/*` + `tests/test_scrapers/test_build_ownership_data.py`).

### Step 3: Validation slice refinement (feature coverage)
- Finding: two DK slates in the original fixed slice (`2025-12-02_1`, `2025-12-05_1`) do **not** have corresponding LineStar projection coverage (LineStar has only one slate for each of those dates) and also don’t have DK salary coverage in the local lake (only one DK `draft_group_id` per date captured).
- Action: updated the fixed eval slice to exclude those unsupported slates and renamed it to `dk_2025-11-24_to_2025-12-06_fixed_supported` (now `n_slates=10`) in `config/ownership_eval_slice.json`.
- Note: `reports/ownership_eval_before_after.md` baseline section was updated to reflect the new slice definition (still using the production run’s stored `val_predictions.csv`).

### Step 4: Deterministic training
- Change: `scripts/ownership/train_ownership_v1.py` now supports `--seed` (default `1337`) and `--num-threads` (default `1`) and injects LightGBM deterministic params (`seed`, `feature_fraction_seed`, `bagging_seed`, `data_random_seed`, `deterministic=True`, `force_row_wise=True`).
- Why: we want reproducible training/eval runs while iterating on features/targets and comparing ranking/top-chalk metrics.

### Step 5: Rebaseline on the fixed DK base (post label/join fixes)
- Trained a clean baseline with the production feature set + weighting:
  - Run: `dk_only_v4_cleanbase_seed1337` (`--feature-set v4 --sample-weighting --val-start-date 2025-11-24 --seed 1337`)
  - Eval JSON: `reports/ownership_eval_runs/dk_only_v4_cleanbase_seed1337.json`
  - Key outcomes vs legacy `dk_only_v4` (not apples-to-apples due to earlier label/join issues): stable actual sums (~794%), improved error and pooled Spearman.

### Step 6: Model iteration focused on ranking + chalk
- Tried richer feature set `v6` (adds value leverage + popularity + interactions):
  - `dk_only_v6_cleanbase_seed1337`: big lift in top10 Spearman (chalk ordering), but more spike risk after sum-scaling (more >60% preds).
- Tried logit target (better handling of tail + extremes):
  - `dk_only_v6_logit_cleanbase_seed1337`: major lift in pooled Spearman and error; trade-off was worse ECE after sum-scaling (sharper outputs).
- Tuned chalk weighting to recover top-chalk recall without losing ranking:
  - `dk_only_v6_logit_chalk5_cleanbase_seed1337`: improved Recall@10 and reduced top10/top20 underprediction after sum-scaling.
  - This run is the current “After” candidate in `reports/ownership_eval_before_after.md`.

### Step 7: Calibration/normalization experiments
- Added optional “softmax slate calibrator” plumbing (fit on train, evaluate on val) to explore distribution-shape calibration, but the naive softmax layer can allocate >100% to a single player (invalid) and performed poorly on the fixed slice without additional caps/redistribution.
- Next: if we want to use softmax/IPF-style normalization in production, we need support-aware caps + smooth redistribution (no >100%, no spikes), then re-evaluate.

## 2025-12-18 — Phase 3: Output + Integration

### Step 1: Production inference supports v6 + logit runs
- Updated inference (`projections/ownership_v1/score.py`) to invert `target_transform=logit` using sigmoid and return percent space.
- Expanded inference-time feature builder (`compute_ownership_features`) to include v5/v6 computed features (slate structure, value leverage, interactions) with safe defaults.
- Updated schema defaults to allow v6 “player popularity” features when present and to default them to 0 when missing.
- Live scoring now enriches with historical player priors (`player_own_avg_10`, `player_own_median`, `player_own_variance` (std), `player_chalk_rate`) from DK contest labels.
- Explicit intent: **no LineStar dependency at runtime**. Live scoring uses DK salaries + sim_v2 projections + optional injuries + historical DK priors.

### Step 2: Enforce slate sum constraints in live scoring
- Added `pred_own_pct_raw` column and applied playable filter first, then normalized `pred_own_pct` to the site total (default DK classic `800%`) via proportional scale-to-sum with an optional 100% cap (config: `config/ownership_calibration.yaml` → `normalization`).
- Smoke check: `score_ownership_live.py` on `2025-12-08` produced `sum(pred_own_pct)=800%` (raw sum was ~718%).

### Step 3: Downstream consumer robustness
- `projections/api/contest_sim_api.py` now loads ownership from the new per-slate directory format (`silver/ownership_predictions/<date>/*.parquet`) when present, falling back to the legacy single-file path.

### Docs / Tests
- Added `docs/ownership_model.md` and updated `docs/ownership/README.md` to reflect the current training/eval/inference pipeline and the “no LineStar at runtime” constraint.
- Focused test command remains: `uv run pytest -q tests/ownership_v1 tests/test_scrapers/test_build_ownership_data.py` (full suite still fails due to unrelated minutes_v1 collection issues).

### Step 4: Leak-safety for historical runs (injuries + lock gating)
- Problem: our `silver/injuries_snapshot/.../injuries.parquet` files can contain multiple snapshots for the same `game_date` (including post-lock updates). For historical backtests, “load latest for date” can leak future injury status into pre-lock features.
- Change: `projections/cli/score_ownership_live.py` now:
  - uses schedule `tip_ts` (UTC) for lock detection and gating,
  - threads an `injuries_cutoff_ts` into scoring, and
  - filters injuries to `as_of_ts <= min(now_utc, slate_first_tip_utc)` when `as_of_ts` exists.
- Outcome: scoring is now safe to run on historical dates without pulling post-lock injury snapshots for that slate.
- Follow-up: fixed `slates.json` metadata generation to use `datetime.now(tz=UTC)` so lock detection doesn’t compare tz-naive vs tz-aware timestamps.

### Step 5: Production-path backtest evaluator (DK actuals ↔ live preds)
- Added: `scripts/ownership/evaluate_ownership_production_path.py`
  - Maps DK `slate_id` (contest exports) ↔ DK `draft_group_id` (salary slates) via player-pool overlap, then joins players by normalized name.
  - Supports `--slate-selector largest_entries` (default) to approximate “main slate only”.
  - Supports `--pred-snapshot locked` (default) to evaluate the first pre-lock snapshot saved by live scoring.
  - Emits the same core metrics as our fixed-slice evaluator (MAE/RMSE, Spearman, top-chalk rank/recall, ECE, sum checks).
- Tests: `tests/ownership_v1/test_production_path_eval.py` covers overlap mapping + suffix normalization.

## 2025-12-18 — Calibration/Normalization Iteration

### Step 1: Add a “power” calibration option (rank-preserving)
- Added: `PowerCalibrator` (`projections/ownership_v1/calibration.py`) implementing `(s+eps)^gamma` within-slate allocation; preserves ranking by construction.
- Added: `scripts/ownership/fit_power_calibrator.py` to fit `gamma` on a joined production-path dataset, prioritizing top-chalk MAE.
- Updated: `projections/cli/score_ownership_live.py` to support `calibration.method: power` and to enforce post-calibration cap/sum guardrails via `normalize_ownership_to_target_sum` (prevents >100% allocations).
- Updated: `config/ownership_calibration.yaml` to document the new calibration method.
- Tests: extended `tests/ownership_v1/test_calibration.py` with `PowerCalibrator` unit coverage.
