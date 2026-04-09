# Rates / GBM Inventory

Generated: 2026-04-09 UTC

Scope:
- Rates and GBM-specific inventory only.
- Only file added in this phase: `RATES_INVENTORY.md`.
- Inventory is based on direct code/config/file inspection in this checkout plus on-disk artifact inspection under `/home/daniel/projections-data`.
- No dependencies were installed and no project code was executed. Where parquet schema details are discussed, they are inferred from builder code and manifests rather than from runtime parquet introspection.

Important split note:
- The repo currently contains three different "rates" tracks:
  - `rates_v1`: the older LightGBM per-minute rates stack.
  - GTv2 joint rate heads inside the transformer stack.
  - `tree_rate_bundle`: a newer tree-model override path that can blend into GTv2 worlds.
- These paths coexist in code, but they do not all run in the same live path today.

## 1. The `rates_v1` Package and Related Modules

### Core `projections/rates_v1/`

`projections/rates_v1/__init__.py`
- Purpose: package marker only. The docstring still describes the package as "stage-0 GBM for per-minute rates."
- Role: none.
- State: non-substantive.

| Path | Purpose | Role | Direct internal imports / coupling | External deps | State |
| --- | --- | --- | --- | --- | --- |
| `projections/rates_v1/current.py` | Resolves the active rates selector JSON and returns the selected `run_id` / bundle. | Inference utility | Imports `projections.model_selectors.active_rates_selector_path` and `projections.rates_v1.loader`. No `scripts/` import. | stdlib only | Active |
| `projections/rates_v1/loader.py` | Loads a persisted rates bundle from `$DATA_ROOT/artifacts/rates_v1/runs/<run_id>`, including `model_<target>.txt`, `feature_cols.json`, and `meta.json`. | Inference utility | Imports `projections.paths.data_path`. No `scripts/` import. | `lightgbm` | Active |
| `projections/rates_v1/features.py` | Canonical feature-set definitions for stage0 through stage6. This is the shared feature contract between training and inference. | Shared utility | Imports `projections.features.action_props.ACTION_MARKET_FEATURE_COLUMNS`. No `scripts/` import. | none beyond local code | Active |
| `projections/rates_v1/preprocess.py` | Train/serve parity helpers for odds fill values and tracking fill values. | Shared utility | No internal imports. | `numpy`, `pandas` | Active |
| `projections/rates_v1/score.py` | Scores a DataFrame with a loaded bundle. Applies fill-value metadata from `meta.json`, then runs all per-target boosters. | Inference | Imports `projections.rates_v1.loader` and `projections.rates_v1.preprocess`. No `scripts/` import. | `pandas` | Active |
| `projections/rates_v1/production.py` | Thin wrapper around `current.py` + `loader.py` for "load the production bundle". | Inference utility | Imports `projections.rates_v1.current` and `projections.rates_v1.loader`. No `scripts/` import. | stdlib only | Active |
| `projections/rates_v1/schemas.py` | Schema contract for live rates features and scored outputs. Defines `RATES_PREDICTIONS_SCHEMA` and `EFFICIENCY_TARGETS`. | Shared utility / validation | Imports `projections.minutes_v1.schemas` and `projections.rates_v1.features`. This is a direct `minutes_v1` coupling. No `scripts/` import. | none beyond local code | Active, but stale relative to current promoted bundle |
| `projections/rates_v1/training_base_schema.py` | Minimal schema guardrail for `gold/rates_training_base`. Protects required keys/timestamps/labels. | Shared utility / validation | Imports `projections.minutes_v1.schemas`. This is a direct `minutes_v1` coupling. No `scripts/` import. | stdlib only | Active |

Direct findings:
- The serving half of `rates_v1` is compact. `loader.py` + `score.py` + `preprocess.py` are the real scoring core.
- The package itself does not import from `scripts/`.
- The schema layer is not fully aligned with the current promoted artifact:
  - `projections/rates_v1/schemas.py` still treats stage3-context as "the production feature set".
  - The current promoted bundle is stage6-action-props.

### Live rates CLIs and pipeline-adjacent modules

| Path | Purpose | Role | Direct internal imports / coupling | External deps | State |
| --- | --- | --- | --- | --- | --- |
| `projections/cli/build_rates_features_live.py` | Builds live `features_rates_v1/.../features.parquet` for a slate. Mirrors the training-base feature build, but against live data sources. | Inference feature build | Imports `projections.features.action_props`, `projections.minutes_v1.pos`, `projections.minutes_v1.season_dataset._parse_minutes_iso`, `projections.rates_v1.schemas`, `projections.pipeline.status`, `projections.runtime_safety`. The private `_parse_minutes_iso` import is a real `minutes_v1` coupling. | `numpy`, `pandas`, `typer` | Active on the legacy live path; not called by the canonical v3 flow |
| `projections/cli/score_rates_live.py` | Loads the active bundle, scores live features, writes `gold/rates_v1_live/.../rates.parquet`, `summary.json`, and `latest_run.json`. | Inference scoring | Imports `projections.rates_v1.production`, `projections.rates_v1.score`, `projections.rates_v1.schemas`, `projections.pipeline.status`, `projections.runtime_safety`. No `scripts/` import. | `pandas`, `typer` | Active on the legacy live path; not called by the canonical v3 flow |
| `projections/pipeline/live_orchestrator.py` | Legacy live orchestrator. It still shells out to `build_rates_features_live` and `score_rates_live`, then passes rates into older sim paths. | Live orchestration | No direct `rates_v1` import block, but it runs the rates CLIs as subprocess modules. | `pandas` | Legacy-but-real |
| `projections/pipeline/effective_inputs.py` | Writes `effective_rates.parquet` by applying authorized manual overrides to raw rates outputs. | Post-inference reconciliation | Imports `projections.ops.overrides`; no GTv2 coupling. | `pandas` | Active where the legacy rates path is still used |
| `projections/pipeline/guardrails.py` | Output sanity checks for rates/minutes/etc. | Validation | No rates-specific import coupling. | `pandas` | Active utility |
| `projections/pipeline/run_manifest.py` | Captures rates live output paths in a run manifest. | Manifesting | No `scripts/` coupling. | `pandas` | Active utility |
| `projections/pipeline/gtv2_live_features.py` | GTv2 live feature build. Not part of `rates_v1`, but it is rate-adjacent because it prepares the GTv2 live contract and imports script modules directly. | GTv2 inference feature build | Imports `scripts.rotation.build_joint_rotation_rates_dataset_v1` and `projections.cli.score_minutes_rotation_set_v1`. This is explicit script/CLI coupling. | `numpy`, `pandas` | Active GTv2 path |

Direct findings:
- `score_rates_live.py` is clean.
- `build_rates_features_live.py` is the main live-coupling surface:
  - it depends on `live/features_minutes_v1`,
  - historical `gold/rates_training_base`,
  - `gold/tracking_roles`,
  - raw bronze boxscores,
  - Action Network props snapshots,
  - and a private helper from `minutes_v1`.

### Training / evaluation scripts under `scripts/rates/`

| Path | Purpose | Role | Direct internal imports / coupling | External deps | State |
| --- | --- | --- | --- | --- | --- |
| `scripts/rates/build_training_base.py` | Builds `gold/rates_training_base/season=YYYY/game_date=YYYY-MM-DD/rates_training_base.parquet` from labels, boxscores, odds, roster, injuries, optional tracking/minutes-for-rates, and Action props. | Training data prep | Imports `projections.features.action_props`, `projections.minutes_v1.pos`, `projections.rates_v1.preprocess`. No `scripts/` import. | `numpy`, `pandas`, `typer` | Active |
| `scripts/rates/train_rates_v1.py` | Trains one LightGBM regressor per rate target and per efficiency target; writes a bundle under `artifacts/rates_v1/runs/<run_id>`. | Training | Imports `projections.rates_v1.features`, `projections.rates_v1.preprocess`, `projections.rates_v1.schemas`, `projections.registry.manifest`, `projections.lgbm_device`. No `scripts/` import. | `lightgbm`, `mlflow`, `numpy`, `pandas`, `typer` | Active, but not the center of March 2026 work |
| `scripts/rates/eval_efficiency_heads.py` | Evaluates the efficiency heads against holdout data and writes JSON/CSV diagnostics. | Evaluation | Imports `projections.rates_v1.features` and `projections.rates_v1.schemas`. No `scripts/` import. | `lightgbm`, `numpy`, `pandas`, `typer` | Active support script |
| `scripts/rates/compare_baselines.py` | Compares current bundle predictions with simple season/role baselines. | Evaluation | Imports `projections.rates_v1.current`, `loader`, and `score`. No `scripts/` import. | `numpy`, `pandas`, `typer` | Active support script |
| `scripts/rates/compute_rates_residuals.py` | Computes train/cal/val residual summaries for the current bundle. | Evaluation / diagnostics | Imports private helpers `_load_training_base` and `_split_by_date` from `scripts.rates.train_rates_v1`. This is explicit script-to-script coupling. | `pandas`, `typer` | Brittle support script |
| `scripts/rates/debug_efficiency_heads.py` | Ad hoc inspection/debug for efficiency outputs on a selected bundle. | Debug | Imports `projections.rates_v1.production`, `loader`, `score`, `schemas`. | `numpy`, `pandas`, `typer` | Debug-only |
| `scripts/rates/debug_training_base.py` | Sanity checks for `rates_training_base`. | Debug | Imports `projections.paths.data_path`. | `numpy`, `pandas`, `typer` | Debug-only |
| `scripts/rates/summarize_rates_runs.py` | Summarizes bundle metrics across `artifacts/rates_v1/runs/*`. | Reporting | Imports `projections.paths.data_path`. | `typer` | Ops/reporting utility |
| `scripts/rates/check_minutes_for_rates_coverage.py` | Coverage audit for `gold/minutes_for_rates`. | Validation | Imports `projections.paths.data_path`. | `pandas`, `typer` | Validation utility |
| `scripts/rates/check_rates_training_base_coverage.py` | Coverage audit for `gold/rates_training_base`. | Validation | Imports `projections.paths.data_path`. | `pandas`, `typer` | Validation utility |

### Related rate infrastructure outside `scripts/rates/`

| Path | Purpose | Role | Direct internal imports / coupling | External deps | State |
| --- | --- | --- | --- | --- | --- |
| `prefect_flows/rates_retrain.py` | Automated retrain / eval / calibration / promotion flow for `rates_v1`. | Training orchestration | Imports `projections.rates_v1.loader`, `projections.rates_v1.score`, and private helpers/constants from `scripts.rates.train_rates_v1`; also shells out to `scripts.rates.build_training_base`, `scripts.rates.train_rates_v1`, and `scripts.rates.eval_efficiency_heads`. This is direct script coupling. | `prefect`, `numpy`, `pandas` | Active-ish, but default config lags current promoted stage |
| `projections/rotation/tree_rate_bundle.py` | Separate tree-bundle format for GTv2 world overrides. Trains/loads/scored LightGBM or XGBoost bundles off joint datasets. | Training + inference, but not `rates_v1` | No `rates_v1` imports. Consumes `training/datasets/joint_rotation_rates_v1_*/labels_rates.parquet`. | `lightgbm`, `xgboost`, `numpy`, `pandas` | Active experimental GTv2-side path |
| `scripts/rotation/train_tree_rate_bundle.py` | CLI wrapper for `tree_rate_bundle.py`. | Training | Imports `projections.rotation.tree_rate_bundle`. | `pandas` | Active experimental |

Related absences:
- No dedicated rates module was found under `projections/models/`.
- There is no separate `projections/pipeline/*` training package for `rates_v1`; the training logic lives in `scripts/rates/*` plus `prefect_flows/rates_retrain.py`.

## 2. Trained Rates Artifacts

### Active selector state

Repo selector:
- Path: `config/rates_current_run.json`
- Contents:

```json
{
  "run_id": "rates_v1_stage6_action_props_h75_propsfix_20260218_194555"
}
```

- Mtime: `2026-02-28 21:18:19 UTC`

Runtime selector:
- Path: `/home/daniel/projections-data/control_plane/model_selectors/rates_current_run.json`
- Contents: same `run_id` as repo selector.
- Mtime: `2026-03-31 12:32:10 UTC`

Selector behavior:
- `projections.model_selectors.active_rates_selector_path()` prefers the runtime selector under `$PROJECTIONS_DATA_ROOT/control_plane/model_selectors` when present.
- There is no `current`, `bundle_current`, or symlink-based selector under `/home/daniel/projections-data/artifacts/rates_v1`. The active pointer is JSON-selector driven.

### Current promoted `rates_v1` bundle

- Path: `/home/daniel/projections-data/artifacts/rates_v1/runs/rates_v1_stage6_action_props_h75_propsfix_20260218_194555`
- Referenced by live config: yes.
  - `config/rates_current_run.json`
  - `/home/daniel/projections-data/control_plane/model_selectors/rates_current_run.json`
- Loaded by inference code: yes.
  - `projections.rates_v1.loader.load_rates_bundle()` resolves exactly `$DATA_ROOT/artifacts/rates_v1/runs/<run_id>`.
  - `score_rates_live.py` reaches it through `load_production_rates_bundle()`.
- Bundle directory mtime: `2026-02-24 13:59:59 UTC`
- File mtimes inside the bundle show the actual training writeout on `2026-02-18 19:46:05` through `19:47:42 UTC`; the later directory mtime likely reflects a later touch (the bundle also contains a `calibration_monitor/` directory created later).

Contents:

```text
model_fga2_per_min.txt      464314 bytes
model_fga3_per_min.txt      479724 bytes
model_fta_per_min.txt       399820 bytes
model_ast_per_min.txt       399788 bytes
model_tov_per_min.txt       373345 bytes
model_oreb_per_min.txt      291884 bytes
model_dreb_per_min.txt      356528 bytes
model_stl_per_min.txt       250568 bytes
model_blk_per_min.txt       441488 bytes
model_fg2_pct.txt           310202 bytes
model_fg3_pct.txt           259988 bytes
model_ft_pct.txt            216542 bytes
feature_cols.json             3418 bytes
meta.json                     8190 bytes
metrics.json                  3785 bytes
calibration_monitor/          empty directory at inspection time
```

Bundle metadata highlights:
- `feature_set`: `stage6_action_props`
- `targets`: 12 total
  - rate heads: `fga2_per_min`, `fga3_per_min`, `fta_per_min`, `ast_per_min`, `tov_per_min`, `oreb_per_min`, `dreb_per_min`, `stl_per_min`, `blk_per_min`
  - efficiency heads: `fg2_pct`, `fg3_pct`, `ft_pct`
- date window:
  - `start`: `2025-01-10`
  - `end`: `2026-02-06`
  - `train_end`: `2026-01-10`
  - `cal_end`: `2026-01-24`
- recency weighting: enabled, half-life `75.0`
- train/cal/val rows:
  - `train_rows`: `12546`
  - `cal_rows`: `2109`
  - `val_rows`: `1939`

### `rates_v1` artifact root state

- Path: `/home/daniel/projections-data/artifacts/rates_v1`
- Top-level subdirs found:
  - `runs/`
  - `analysis/`
  - `residuals/`
  - `walk_forward/`
- No `promotions/` directory exists at this path as of inspection, even though `prefect_flows/rates_retrain.py` has code to create one.
- Run chronology under `runs/` shows active development from November 2025 through February 2026, with stage progression:
  - stage0/stage1 in late November 2025
  - stage2/stage3 in late November through December 2025
  - stage4 recency in late December 2025
  - stage5 FTA / recency in January and February 2026
  - stage6 action-props in February 2026
- The newest bundle on disk is the active stage6 bundle from `2026-02-18`.

### Sibling rate bundle: GTv2 tree-rate override

- Path: `/home/daniel/projections-data/artifacts/tree_rate_bundles/tree_rate_astreb_lgbm_livev1_20260329T1620Z`
- Referenced by live config: no.
  - It is mentioned in docs.
  - `config/gtv2_inference_current.json` currently sets `"tree_rate_bundle_dir": null` and `"tree_rate_predictions_csv": null`.
- Loaded by inference code today: no.
  - `prefect_flows/live_nba_pipeline_v3.py` can load it if configured.
  - Current config leaves the override disabled.
- Directory mtime: `2026-03-29 16:48:23 UTC`

Contents:

```text
bundle_meta.json             12412 bytes
models/ast_per_min.txt      757359 bytes
models/oreb_per_min.txt     304151 bytes
models/dreb_per_min.txt     526796 bytes
```

Bundle metadata highlights:
- `bundle_type`: `tree_rate_v1`
- `model_type`: `lgbm`
- `target_set`: `astreb`
- Training source (from docs): `joint_rotation_rates_v1_shootmatch_20260329T014610Z`

### Sibling artifact root: joint rotation/rates model artifacts

- Path: `/home/daniel/projections-data/artifacts/joint_rotation_rates_v1`
- Subdirs found:
  - `runs/`
  - `sweeps/`
  - `pytorch_tabular_runs/`
- Referenced by live config: no.
  - Current GTv2 config points to `/home/daniel/projections-data/artifacts/game_transformer_v2/bundle_current`, not this root.
- Loaded by `rates_v1` inference code: no.
- State:
  - this is a historical joint-model experiment root with many February 2026 sweep runs,
  - not a selector-backed current production bundle root.

### `/home/daniel/projections-data/training/runs/` rate-related directories

This tree contains many rate-related experiment/eval outputs, but not selector-backed live bundles. Representative March 2026 directories include:

- `gtv2_tree_*`
- `lgbm_rate_hybrid_*`
- `tree_rate_hybrid_*`
- `reb_60d_eval_dreb_rate_*`
- `gtv2_effhead_*`
- `gtv2_team_possession_*`
- `game_transformer_v2_usage_share_*`

Read:
- These are experiment/evaluation output directories.
- They are evidence of active March 2026 GTv2-side rate work.
- They are not the load target for `projections.rates_v1.loader`.

## 3. Training Data and Labels

### `rates_v1` training source

`rates_v1` training reads from:
- `/home/daniel/projections-data/gold/rates_training_base/season=YYYY/game_date=YYYY-MM-DD/rates_training_base.parquet`

Freshness:
- Latest partition found: `/home/daniel/projections-data/gold/rates_training_base/season=2025/game_date=2026-04-08/rates_training_base.parquet`
- File mtime: `2026-04-09 08:05:06 UTC`

This training base is not stale. It is still being built daily into April 2026.

### What builds `rates_training_base`

`scripts/rates/build_training_base.py` builds it from:
- `gold/labels_minutes_v1/.../labels.parquet`
- `bronze/boxscores_raw/.../boxscores_raw.parquet`
- `silver/odds_snapshot/.../odds_snapshot.parquet`
- `silver/roster_nightly/.../roster.parquet`
- `silver/injuries_snapshot/...`
- optional `gold/minutes_for_rates/.../minutes_for_rates.parquet`
- optional `gold/tracking_roles/.../tracking_roles.parquet`
- optional Action Network props snapshots under `bronze/action_network/props`

Labels created directly in the build:
- `minutes_actual`
- `fga2_per_min`
- `fga3_per_min`
- `fta_per_min`
- `ast_per_min`
- `tov_per_min`
- `oreb_per_min`
- `dreb_per_min`
- `stl_per_min`
- `blk_per_min`
- `fg2_pct_label`
- `fg3_pct_label`
- `ft_pct_label`

Important label rules in the builder:
- minimum minutes for inclusion: `4.0`
- efficiency labels only if attempts meet thresholds:
  - FG2 attempts >= `3`
  - FG3 attempts >= `3`
  - FT attempts >= `2`

### `labels_rates.parquet` in joint datasets

Yes. Many `joint_rotation_rates_v1_*` datasets under `/home/daniel/projections-data/training/datasets/` contain `labels_rates.parquet`.

Observed dataset lineage:
- earliest found: `joint_rotation_rates_v1_20260221T163500Z`
- many February 2026 variants
- many March 2026 variants including:
  - `joint_rotation_rates_v1_trackingctx_prodparity_20260326T152634Z`
  - `joint_rotation_rates_v1_trackingctx_prodparity_lineupavailfix_20260326T181500Z`
  - `joint_rotation_rates_v1_shootmatch_20260329T014610Z`
  - `joint_rotation_rates_v1_20260331_livefill`
  - `joint_rotation_rates_v1_20260331_teamimplied`

Latest inspected example:
- Path: `/home/daniel/projections-data/training/datasets/joint_rotation_rates_v1_20260331_teamimplied/labels_rates.parquet`
- Mtime: `2026-03-31 10:56:44 UTC`
- Matching manifest reports:
  - `rows_labels_rates`: `72474`
  - `labels_rates_columns`: `22`
  - `rows_with_any_rate_labels`: `17520`
  - `rows_with_all_rate_targets`: `17520`
  - `rows_loss_eligible`: `17520`
- The same manifest records `rates_partition_count: 288` and names `/home/daniel/projections-data/gold/rates_training_base` as the label source root.

### `labels_rates.parquet` schema

The schema below is inferred from `scripts/rotation/build_joint_rotation_rates_dataset_v1.py`, not from parquet footer inspection.

Persisted columns (22 total):

```text
game_id
team_id
player_id
game_date
minutes_actual
fga2_per_min
fga3_per_min
fta_per_min
ast_per_min
tov_per_min
oreb_per_min
dreb_per_min
stl_per_min
blk_per_min
fg2_pct_label
fg3_pct_label
ft_pct_label
rates_non_null_count
efficiency_non_null_count
rates_label_available_any
rates_label_available_all_rate_targets
rates_loss_eligible
```

Inferred dtypes from the builder:
- `game_id`, `team_id`, `player_id`: integer-like (`Int64` in builder coercion)
- `game_date`: normalized date/timestamp
- label columns (`minutes_actual`, rate targets, efficiency labels): numeric float
- `rates_non_null_count`, `efficiency_non_null_count`: small integer counts (`int16` in builder)
- availability flags (`rates_label_available_any`, `rates_label_available_all_rate_targets`, `rates_loss_eligible`): `int8`

### Does GTv2 share the same rate label source?

Yes.

The GTv2 joint dataset builder does not create an independent rate-label system. It reads `gold/rates_training_base/.../rates_training_base.parquet`, extracts:
- `minutes_actual`
- the 9 rate targets
- the 3 efficiency labels

and then aligns those labels to the joint rotation feature spine, adding the availability/eligibility flags listed above.

So:
- `rates_v1` training uses `gold/rates_training_base` directly.
- GTv2 `labels_rates.parquet` is a derived packaging of the same underlying label source.

This is the cleanest architectural finding in the inventory. The label source is shared even though the training systems are different.

### Freshness / maintenance read

- `gold/rates_training_base` is actively maintained into April 2026.
- GTv2 joint datasets are also active, with March 31 2026 snapshots present.
- One caveat from the latest joint manifests:
  - both `joint_rotation_rates_v1_20260331_livefill` and `joint_rotation_rates_v1_20260331_teamimplied` record one missing rates date: `2026-03-02`.
  - That is a concrete coverage gap, not a guessed one.

## 4. Inference Path

### Exact entry points

Library-level scoring:
- `projections.rates_v1.score.predict_rates(features_df, bundle)`

Legacy live feature build:
- `python -m projections.cli.build_rates_features_live --date YYYY-MM-DD ...`

Legacy live scoring:
- `python -m projections.cli.score_rates_live --date YYYY-MM-DD ...`

### What live scoring expects as input

`score_rates_live.py` expects:
- a live features parquet at `live/features_rates_v1/<date>/run=<run_id>/features.parquet`, or an explicit `--features-path`
- a bundle resolved from the active rates selector unless `--bundle-config` is supplied

True required feature contract:
- not the stage3 schema in `projections/rates_v1/schemas.py`
- the actual required columns are `bundle.feature_cols`, which for the current promoted artifact is the full `stage6_action_props` list from `feature_cols.json`

Current promoted bundle feature groups:
- minutes predictions: `minutes_pred_p50`, `minutes_pred_spread`, `minutes_pred_play_prob`
- starter / home / rest / position flags
- season per-minute priors and season shooting priors
- odds context: `spread_close`, `total_close`, `team_itt`, `opp_itt`, `has_odds`
- tracking role and extended tracking features
- vacancy features
- team / opponent context
- stage4 recency windows (`last1`, `last3`, `last5`, `last10`)
- sample-size features (`n_games_season`, `season_minutes_sum`)
- Action Network props-derived features (`an_*`)

Feature-build inputs for the legacy end-to-end live path:
- `live/features_minutes_v1/<date>/run=<run_id>/features.parquet`
- historical `gold/rates_training_base`
- `gold/tracking_roles`
- `bronze/boxscores_raw`
- Action Network props snapshots

### What it outputs

`predict_rates()` returns raw target columns:
- `fga2_per_min`
- `fga3_per_min`
- `fta_per_min`
- `ast_per_min`
- `tov_per_min`
- `oreb_per_min`
- `dreb_per_min`
- `stl_per_min`
- `blk_per_min`
- `fg2_pct`
- `fg3_pct`
- `ft_pct`

`score_rates_live.py` then:
- renames them to `pred_*`
- clamps efficiency outputs:
  - `pred_fg2_pct` to `[0.30, 0.75]`
  - `pred_fg3_pct` to `[0.20, 0.55]`
  - `pred_ft_pct` to `[0.50, 0.95]`
- writes a parquet with:
  - `game_id`
  - `team_id`
  - `player_id`
  - the 12 `pred_*` columns
  - `game_date`
- also writes:
  - `summary.json`
  - `latest_run.json`

Output meanings:
- `pred_*_per_min`: predicted per-minute rate for the named event
- `pred_fg2_pct`, `pred_fg3_pct`, `pred_ft_pct`: predicted conversion rates for those shot types

### Minimum file set to run rates inference standalone

Smallest scoring-only file set:
- code:
  - `projections/rates_v1/loader.py`
  - `projections/rates_v1/preprocess.py`
  - `projections/rates_v1/score.py`
- artifacts:
  - the full promoted bundle directory
    - all 12 `model_*.txt` files
    - `feature_cols.json`
    - `meta.json`

If selector-driven "load whatever prod uses" behavior is required, add:
- `projections/rates_v1/current.py`
- `projections/rates_v1/production.py`
- `projections/model_selectors.py`
- `projections/paths.py`
- the selector JSON (`config/rates_current_run.json` or the runtime selector copy)

If the exact legacy live CLI is required, add:
- `projections/cli/score_rates_live.py`
- `projections/rates_v1/schemas.py`
- `projections/runtime_safety.py`
- `projections/pipeline/status.py`

If the full legacy live feature+score path is required, add:
- `projections/cli/build_rates_features_live.py`
- `projections/features/action_props.py`
- `projections/minutes_v1/pos.py`
- `projections/minutes_v1/season_dataset.py` (for `_parse_minutes_iso`)
- plus all data-source dependencies listed above

### Dependencies imported by the minimum file set

For scoring-only inference:
- `lightgbm`
- `pandas`
- `numpy`
- stdlib (`json`, `pathlib`, `dataclasses`, `typing`)

For the exact CLI:
- everything above
- `typer`

### Hidden coupling

Clean part:
- bundle loading and raw scoring are clean, self-contained, and do not import `scripts/`

Coupled part:
- `build_rates_features_live.py` depends on the older minutes system and on historical gold tables
- it imports a private `minutes_v1` helper: `projections.minutes_v1.season_dataset._parse_minutes_iso`
- it expects the older minutes live feature contract as its starting point
- it backfills context from `gold/rates_training_base`, so the live builder is not independent of the training base
- schema validation is weaker than the promoted artifact contract:
  - `validate_rates_features()` enforces the stage3 contract
  - the current promoted model needs stage6 Action-props features
- `score_rates_live.py --no-strict` will silently zero-fill missing bundle features, which is convenient operationally but dangerous for a reset/port

Bottom line on cleanliness:
- The scoring half of `rates_v1` is clean.
- The feature-building half is not clean.
- If "inference" means "given a fully-populated feature frame, score the artifact", the path is straightforward.
- If "inference" means "reproduce the whole live feature+score pipeline", the path inherits several legacy couplings.

## 5. Training Path

### Exact entry points

Training-base build:
- `python -m scripts.rates.build_training_base --start-date ... --end-date ...`

Bundle training:
- `python -m scripts.rates.train_rates_v1 --start-date ... --end-date ... --train-end-date ... --cal-end-date ...`

Orchestrated retrain:
- `prefect_flows/rates_retrain.py`

### What input data it requires

`scripts/rates/train_rates_v1.py` itself requires only:
- `gold/rates_training_base/season=YYYY/game_date=YYYY-MM-DD/rates_training_base.parquet`

It does not do its own raw-data preparation.

`prefect_flows/rates_retrain.py` can refresh the training base first by calling:
- `scripts.rates.build_training_base`

### What artifacts it produces

`scripts/rates/train_rates_v1.py` writes:
- `artifacts/rates_v1/runs/<run_id>/model_<target>.txt` for all 12 targets
- `feature_cols.json`
- `meta.json`
- `metrics.json`

It also logs to:
- MLflow experiment: `rates_v1`
- registry manifest via `projections.registry.manifest`

`prefect_flows/rates_retrain.py` can additionally produce:
- `calibration/efficiency_eval_<run_id>.json`
- `head_to_head_eval_normal_vs_chaos.json`
- runtime selector updates and promotion records under `artifacts/rates_v1/promotions/` if auto-promotion runs

At inspection time:
- the promotion directory does not exist under `/home/daniel/projections-data/artifacts/rates_v1`
- so the code supports promotion history, but I did not find on-disk evidence that this flow has written promotion records for `rates_v1`

### Data preparation: inline or pre-built?

Pre-built.

The training script consumes `rates_training_base` as its dataset. It does not rebuild labels or raw features itself.

That split is explicit in the codebase:
- `scripts/rates/build_training_base.py` = data prep
- `scripts/rates/train_rates_v1.py` = model fit

### Hyperparameters and experiment tracking

Hardcoded LightGBM defaults in `train_rates_v1.py`:
- `objective=regression`
- `metric=l2`
- `boosting_type=gbdt`
- `num_leaves=64`
- `learning_rate=0.05`
- `feature_fraction=0.8`
- `bagging_fraction=0.8`
- `bagging_freq=1`
- `min_data_in_leaf=50`
- `max_depth=-1`
- `lambda_l2=1.0`

Train-time switches:
- `feature_set`: `stage0`, `stage1`, `stage2_tracking`, `stage3_context`, `stage4_recency`, `stage5_fta_tracking`, `stage6_action_props`
- recency half-life / min weight
- LightGBM device / thread count
- whether missing predicted minutes may fall back to `minutes_actual`

Experiment tracking:
- MLflow is first-class in `train_rates_v1.py`
- rates-specific selector/registry integration exists
- there is no separate YAML sweep framework for `rates_v1`; the main variation is driven by CLI flags and run tags

### How long a typical training run takes

Hard evidence is limited.

What I found:
- no persisted `rates_v1` train log was present inside the inspected bundle directories
- the active stage6 bundle wrote its 12 model files from `2026-02-18 19:46:05 UTC` through `19:47:40 UTC`, then wrote `feature_cols.json`, `meta.json`, and `metrics.json` at `19:47:42 UTC`
- the preceding stage5 propsfix run on the same day shows a nearly identical pattern: `19:43:39 UTC` through `19:45:16 UTC`

That supports a bounded statement:
- once the fit/write loop starts, a 12-target rates bundle lands in roughly 1.5 to 2 minutes on this machine
- there is not enough persisted evidence to recover full end-to-end wall clock including dataset load, split, any preceding training-base refresh, or MLflow overhead

Relevant flow timeouts:
- training-base refresh timeout: `1 hour`
- rates train timeout: `2 hours`
- calibration diagnostics timeout: `30 minutes`

### Is the training path currently runnable or bit-rotted?

Evidence in favor of runnable:
- current promoted `rates_v1` bundle was trained on `2026-02-18`
- training base is fresh through `2026-04-08`
- retrain Prefect flow still exists and still points at the training script

Evidence of drift:
- `prefect_flows/rates_retrain.py` defaults to:
  - `feature_set = stage5_fta_tracking`
  - `run_tag = rates_v1_stage5_recency_h75`
- current promoted bundle is:
  - `feature_set = stage6_action_props`
  - `run_id = rates_v1_stage6_action_props_h75_propsfix_20260218_194555`
- `projections/rates_v1/schemas.py` still treats stage3 as the production contract

Read:
- the training path is not obviously dead
- it is also not obviously "freshly exercised in current form" beyond February 2026
- March 2026 rate work shifted heavily toward GTv2 joint-rate and tree-rate experiments

## 6. Relationship to GTv2 Rate Heads

### What rate-related heads exist in GTv2

Confirmed in `projections/rotation/game_transformer_v2.py` and related modules:
- `EfficiencyHead`
- `UsageShareHead`
- `AssistShareHead`
- `TeamAstBudgetHead`
- `ReboundShareHead`
- `TeamReboundBudgetHead`
- `ReboundBudgetBlendGateHead`
- `TeamPointsBudgetHead`
- `TeamPPPHead`
- `TeamAdvantageHead`
- `PossessionHead`
- `TeamEventBackbone`
- `ThreePAShareHead`
- `JointGameFlow` / flow contract reconstruction

This is broader than a simple drop-in replacement for `rates_v1`. GTv2 has both per-player and team-budget/share heads around the same rate/count surface.

### What the live pipeline currently consumes

Legacy live path:
- `prefect_flows/live_nba_pipeline.py`
- `projections/pipeline/live_orchestrator.py`

These still do:
- build minutes features
- score minutes
- build rates features
- score `rates_v1`
- optionally write `effective_rates.parquet`
- run sim against `rates_v1_live`

Downstream legacy consumers still present:
- `scripts/sim_v2/generate_worlds_fpts_v2.py`
- `projections/api/optimizer_service.py`
- API/ops loaders that read `gold/rates_v1_live/.../rates.parquet`

Canonical current flow:
- `prefect_flows/live_nba_pipeline_v3.py`

This flow:
- loads GTv2 config from `config/gtv2_inference_current.json`
- builds GTv2 live features
- scores GTv2
- generates GTv2 worlds
- does not call `build_rates_features_live`
- does not call `score_rates_live`

### Is the live system using `rates_v1`, GTv2 heads, or both?

Both exist in the repo.

But not in one unified live execution path:
- the older live path consumes `rates_v1`
- the canonical v3 flow consumes GTv2 worlds

That is the current split.

### If both exist, where are they merged/reconciled?

There is no generic `rates_v1` vs GTv2 merge layer.

Old path:
- raw rates live output: `rates.parquet`
- post-override layer: `effective_rates.parquet`
- `scripts/sim_v2/generate_worlds_fpts_v2.py` prefers `effective_rates.parquet` over `rates.parquet`
- so in the older path, manual ops overrides win over raw `rates_v1` output

GTv2 v3 path:
- no `rates_v1` load
- optional `tree_rate_bundle` override can blend tree predictions into GTv2 worlds before publish

Conflict rule for the tree-rate override:
- target mean per player/stat is:
  - `(1 - alpha) * current_gtv2_world_mean + alpha * (minutes_mean * predicted_per_min_rate)`
- AST uses mean rescaling
- DREB always uses a share-based override path
- OREB uses the share override only if `tree_rate_oreb_share_override_enabled` is true

Current live config state:
- `config/gtv2_inference_current.json` sets:
  - `"tree_rate_bundle_dir": null`
  - `"tree_rate_predictions_csv": null`
  - `"tree_rate_blend_alpha": 0.75`
- because the bundle dir and predictions CSV are null, the override is effectively disabled right now

### Documented design notes about `rates_v1` vs GTv2

Relevant docs:
- `docs/joint_rotation_rates_v1/01_ARCHITECTURE.md`
- `docs/joint_rotation_rates_v1/03_TRAINING_AND_ROLLOUT.md`
- `docs/joint_rotation_rates_v1/07_MODELING_DIRECTION_WRITEUP_20260327.md`

What those docs say:
- the joint model was explicitly intended to preserve current `rates_v1_live`-style downstream contracts
- rollout was supposed to be shadow-first, then contract-compatible cutover
- by late March 2026, tree-rate override work was active enough that the docs name a first real live bundle:
  - `/home/daniel/projections-data/artifacts/tree_rate_bundles/tree_rate_astreb_lgbm_livev1_20260329T1620Z`

Read:
- `rates_v1` is still the separate legacy tree path with real downstream readers
- GTv2 is the main current modeling direction
- the codebase is in a transition state, not a finished single-path architecture

## 7. What's Dead or Superseded

Conservative flags only.

### Clearly stale or drifted contracts

- `projections/rates_v1/schemas.py`
  - still encodes stage3-context as "the production feature set"
  - current promoted artifact is stage6-action-props
  - this is a real contract drift, not a guess

- `prefect_flows/rates_retrain.py` defaults
  - still default to stage5 (`feature_set=stage5_fta_tracking`, `run_tag=rates_v1_stage5_recency_h75`)
  - current promoted artifact is stage6
  - the retrain flow still exists, but its defaults lag the selector-backed promoted bundle

### Legacy live orchestration still present beside v3

- `projections/pipeline/live_orchestrator.py`
- `prefect_flows/live_nba_pipeline.py`

These still carry the full `rates_v1` live path, but the authoritative docs and current GTv2 config center on `prefect_flows/live_nba_pipeline_v3.py`.

Read:
- not deleted
- still operational
- but no longer the canonical path

### Brittle support code

- `scripts/rates/compute_rates_residuals.py`
  - imports private helpers from `scripts.rates.train_rates_v1`

- `prefect_flows/rates_retrain.py`
  - imports private helpers/constants from `scripts.rates.train_rates_v1`
  - shells out to `scripts.rates.*` modules instead of using a stable library boundary

These files may still work, but they are not cleanly decoupled.

### Historical / experimental artifact trees

- `/home/daniel/projections-data/artifacts/joint_rotation_rates_v1`
  - appears to be a historical joint-model artifact root from February 2026 sweeps
  - current live GTv2 config does not point here

- `/home/daniel/projections-data/training/runs/*rate*`
  - March 2026 contains many rate-hybrid, rebound-rate, tree-rate, and efficiency-head experiment directories
  - these are active experiment outputs, but not selector-backed production bundles

- `/home/daniel/projections-data/artifacts/rates_v1/analysis`
- `/home/daniel/projections-data/artifacts/rates_v1/residuals`
- `/home/daniel/projections-data/artifacts/rates_v1/walk_forward`
  - clearly historical analysis output, not live serving state

### Not found where an older inventory might have implied them

- No dedicated active `projections/models/*rates*` module was found.
- The actual `rates_v1` GBM implementation lives in:
  - `scripts/rates/train_rates_v1.py`
  - `projections/rates_v1/*`

## Reset-oriented takeaways

- A usable promoted `rates_v1` artifact exists now, and the raw scorer around it is small and clean. The coupling is in live feature construction, not in bundle loading or prediction.
- `rates_v1` and GTv2 do share the same underlying rate-label source: `gold/rates_training_base`. GTv2 `labels_rates.parquet` is a derived packaging of that base, not a separate label system.
- The legacy rates path is still real downstream state (`rates_v1_live`, `effective_rates.parquet`, sim/optimizer fallbacks), but the canonical v3 flow has already moved to GTv2 worlds and does not call the legacy rates scorer.
- March 2026 rate experimentation shifted toward GTv2-side heads and the separate `tree_rate_bundle` override path. Those artifacts exist, but current live config leaves the tree-rate override disabled.
- The `rates_v1` training base is fresh, and the training script was successfully used for the current promoted bundle in February 2026. The retrain flow still exists, but its default feature-set/run-tag configuration lags the currently promoted stage6 bundle.
