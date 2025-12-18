# Ownership Model (`ownership_v1`)

This doc describes the NBA DFS ownership projection model and how to train/evaluate it using only the data already in this repo + the local data lake (`~/projections-data` or `$PROJECTIONS_DATA_ROOT`).

## Goals / Non-Goals

- Goal: accurate *ranking* and top-chalk ordering (top 10/20) for DK NBA slates, with sane slate-level sum constraints.
- Non-goal: using LineStar in production inference. LineStar can be used for training-data assembly only.

## Runtime (Production) Inference — No LineStar

**Entry point**: `projections/cli/score_ownership_live.py`

**Inputs (pre-lock only)**:
- DK salaries: `gold/dk_salaries/site=dk/game_date=YYYY-MM-DD/draft_group_id=*/salaries.parquet`
- Sim projections: `artifacts/sim_v2/projections/(game_date|date)=YYYY-MM-DD/**/sim_v2_projections.parquet` (or legacy `projections.parquet`)
- Injuries snapshot (optional): `silver/injuries_snapshot/season=*/month=*/injuries_snapshot.parquet`
- Historical DK ownership labels (for priors only): `bronze/dk_contests/ownership_by_slate/all_ownership.parquet` (filtered to `< game_date`)

**Output**:
- Per-slate predictions written to `silver/ownership_predictions/YYYY-MM-DD/<draft_group_id>.parquet`
- Cached lock snapshot: `silver/ownership_predictions/YYYY-MM-DD/<draft_group_id>_locked.parquet`
- Slate metadata: `silver/ownership_predictions/YYYY-MM-DD/slates.json`

**Key columns**:
- `pred_own_pct`: normalized ownership % (defaults to DK classic `800%` total)
- `pred_own_pct_raw`: raw model output before filtering/normalization
- `model_run`, `run_id`, `draft_group_id`, `is_locked`

**Sum constraint / normalization**:
- Controlled by `config/ownership_calibration.yaml`.
- Default behavior (when `calibration.enabled: false`) is `normalization.enabled: true`, which scales `pred_own_pct` to `calibration.R * 100` (DK classic = `800%`), after the playable filter.

## Training Data (Labels)

### DK “actual ownership” labels (post-contest)

**Builder**: `scrapers/dk_contests/build_ownership_data.py`

**Input**: `bronze/dk_contests/nba_gpp_data/<date>/results/contest_*_results.csv`  
**Output**:
- `bronze/dk_contests/ownership_by_slate/<date>_<slate_idx>.parquet`
- `bronze/dk_contests/ownership_by_slate/all_ownership.parquet`

Notes:
- Contest clusters are grouped into slates by player-pool overlap.
- Slate aggregation treats players missing from a contest as `0%` for that contest to avoid inflated sums.

### DK training base (features + DK labels)

**Builder**: `scripts/ownership/build_ownership_dk_base.py`  
**Output**: `gold/ownership_dk_base/ownership_dk_base.parquet`

This step matches DK slates to LineStar slates *only to borrow historical pre-lock features* when needed during dataset assembly. Production inference does not use LineStar.

## Model Training

**Trainer**: `scripts/ownership/train_ownership_v1.py`  
**Artifacts**: `artifacts/ownership_v1/runs/<run_id>/` (`model.txt`, `feature_cols.json`, `meta.json`, `val_predictions.csv`, etc.)

Recommended flags for reproducibility:
- `--seed 1337 --num-threads 1`

Target transform:
- `--target-transform logit` models ownership in logit space (with safe clipping internally) and inverts back to percent at inference time.

Chalk emphasis:
- `--sample-weighting --chalk-weight 5` increases loss weight for chalk plays to improve top-10/20 ordering.

## Evaluation (Fixed Slice — Do Not Move Goalposts)

**Slice config**: `config/ownership_eval_slice.json`  
**Evaluator**: `scripts/ownership/evaluate_ownership_v1.py`  
**Before/after report**: `reports/ownership_eval_before_after.md`

Metrics reported include:
- MAE/RMSE in percent space (and logit space internally)
- Spearman rank correlation (overall + top-chalk subsets)
- Calibration (ECE + segment bias for top 10/20 and long tail)
- Sum constraint checks under normalization

## Commands

All commands assume `uv sync` has been run and your data lake is at `~/projections-data` (or `$PROJECTIONS_DATA_ROOT`).

Rebuild DK ownership labels:
- `uv run python scrapers/dk_contests/build_ownership_data.py`

Build DK training base:
- `uv run python scripts/ownership/build_ownership_dk_base.py --dk-ownership-path ~/projections-data/bronze/dk_contests/ownership_by_slate --linestar-path ~/projections-data/gold/ownership_training_base/ownership_training_base.parquet --output ~/projections-data/gold/ownership_dk_base/ownership_dk_base.parquet`

Train:
- `uv run python scripts/ownership/train_ownership_v1.py --run-id <RUN_ID> --training-base ~/projections-data/gold/ownership_dk_base/ownership_dk_base.parquet --feature-set v6 --target-transform logit --sample-weighting --chalk-weight 5 --val-start-date 2025-11-24 --seed 1337 --num-threads 1`

Evaluate on the fixed slice:
- `uv run python scripts/ownership/evaluate_ownership_v1.py --run-id <RUN_ID> --slice-config config/ownership_eval_slice.json --out-json reports/ownership_eval_runs/<RUN_ID>.json`

Score a live slate date:
- `uv run python projections/cli/score_ownership_live.py --date YYYY-MM-DD --run-id <PIPELINE_RUN_ID> --model-run <RUN_ID>`

## Current Production Model Run

`projections/cli/score_ownership_live.py` defaults to:
- `dk_only_v6_logit_chalk5_cleanbase_seed1337`

