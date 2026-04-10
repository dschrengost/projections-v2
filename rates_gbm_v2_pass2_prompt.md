# Pass 2: Port rates GBM v2 experiment primitives and script into nba-dfs-v2

You are working in two repos:

- **Source repo (read-only):** `/home/daniel/projects/projections-v2`
- **Destination repo:** `/home/daniel/projects/nba-dfs-v2`

You are also reading data from `/home/daniel/projections-data/` (read-only).

## Goal

Extract the modeling primitives from `projections-v2/scripts/experiments/lgbm_scoring_rates_v2.py` into clean library modules in `nba-dfs-v2`, then write a self-contained experiment script that composes those primitives to reproduce the v2 experiment.

The success criteria are:
1. All new library code parses cleanly (`ast.parse` or `ruff check`).
2. A v1-compatible rates bundle is produced by the trainer (loadable by the existing `nba_dfs.rates.load_rates_bundle` with zero changes to the scorer).
3. The experiment script is a thin composition layer (~150-250 lines) that imports library primitives, not a monolithic rewrite.
4. All new code respects the existing repo disciplines: no `pd.read_parquet` outside `nba_dfs.data`, no script-module imports from the package, config through `nba_dfs.config`.

## Critical context from Pass 1

Read `/home/daniel/projects/nba-dfs-v2/docs/RATES_V2_BUNDLE_FORMAT_INVESTIGATION.md` before starting. The key findings that constrain this pass:

- **Bundle format:** v2-trained bundles must conform to the v1 bundle layout. Root-level `model_<target>.txt` files, `feature_cols.json` with `{"feature_cols": [...]}`, and `meta.json` with `targets`, `feature_cols`, and `preprocess.*_fill_values`. The existing `nba_dfs.rates.load_rates_bundle` and `predict_rates` must work unchanged.
- **Preprocessing parity (Option A):** The trainer must fit odds and tracking fill values on the train split before training, apply them to all splits, and persist them in `meta.json["preprocess"]`. This matches the v1 scorer's behavior at inference time.
- **Efficiency priors are not in the bundle.** They are a runtime computation, not a trained artifact. They live in a separate library primitive.
- **9 targets, not 12.** The v2 approach trains opportunity rates (fga2_per_min, fga3_per_min, fta_per_min) and counting rates (ast_per_min, oreb_per_min, dreb_per_min, stl_per_min, blk_per_min, tov_per_min). Efficiency is handled via rolling priors.

## Constraints

1. **Do not modify any file in `projections-v2`.** Read-only source.
2. **Do not modify the existing `nba_dfs.rates` scorer** (`loader.py`, `score.py`, `preprocess.py`). The whole point of Pass 1 was to confirm the scorer works unchanged. If you find yourself needing to change the scorer, **stop and report**.
3. **Do not modify `nba_dfs.minutes` or `nba_dfs.features`.** This pass adds new modules only.
4. **Do not install dependencies or run any code.** The user will validate separately.
5. **Do not read parquet files directly from Python.** You may use shell commands (`ls`, `stat`, `head`, `wc`, `jq`) to inspect JSON manifests and directory contents for verification.
6. **All reads from the data directory go through `nba_dfs.data`.** New data-loading functions belong in `nba_dfs.data`, not in the training module or the experiment script.
7. **The experiment script lives in `scripts/`, not in the package.** Scripts import from the package; the package never imports from scripts.

## New dependencies

Add to `pyproject.toml` if not already present:
- `pyarrow>=10.0.1,<21.0` — needed for `pd.read_parquet` in the data loaders

Do not add `mlflow`, `typer`, `prefect`, or any other dependency from the old repo that isn't needed for the primitives being built.

## Files to create

### Overview

```
src/nba_dfs/
├── config.py                          # MODIFY: add NBA_DFS_JOINT_DATASET_DIR
├── data/
│   ├── __init__.py                    # MODIFY: add re-exports
│   ├── joint_dataset.py               # NEW: joint dataset loader
│   └── boxscore_labels.py             # NEW: boxscore count labels loader
├── rates/
│   ├── priors.py                      # NEW: rolling efficiency priors
│   └── ...                            # existing files unchanged
├── training/
│   ├── __init__.py                    # NEW: package init
│   ├── README.md                      # NEW: module docs
│   ├── split.py                       # NEW: time-based train/val/test split
│   ├── weights.py                     # NEW: sample weighting utilities
│   └── train_rates.py                 # NEW: rates GBM trainer with v1-compatible bundle output
├── eval/
│   ├── __init__.py                    # NEW: package init
│   ├── README.md                      # NEW: module docs
│   └── metrics.py                     # NEW: evaluation metrics (MAE, bias, correlation, DK FPTS)
scripts/
└── experiments/
    └── rates_gbm_v2.py               # NEW: the experiment script
tests/
└── test_rates_training_smoke.py       # NEW: placeholder smoke test
docs/
└── PORT_NOTES.md                      # MODIFY: append port log
```

### File-by-file specifications

---

#### `src/nba_dfs/config.py` — MODIFY

Add a new env var `NBA_DFS_JOINT_DATASET_DIR` with default `/home/daniel/projections-data/training/datasets/joint_rotation_rates_v1_20260331_teamimplied`. Same pattern as the other config vars: read at import time, raise `FileNotFoundError` if the path doesn't exist. This is the canonical joint dataset identified in the prior inventory.

Add a docstring note that this default can be overridden to point at different dataset builds for experimentation.

---

#### `src/nba_dfs/data/joint_dataset.py` — NEW

Purpose: load the joint rotation/rates training dataset from a dataset directory.

Public API:

```python
@dataclass(frozen=True)
class JointDataset:
    """A loaded joint rotation/rates training dataset."""
    features: pd.DataFrame        # features.parquet contents
    labels_rates: pd.DataFrame    # labels_rates.parquet contents
    labels_boxscore: pd.DataFrame # labels_boxscore_counts.parquet contents
    manifest: dict                # manifest.json contents
    dataset_dir: Path             # where it was loaded from

JOIN_KEYS: list[str] = ["game_id", "team_id", "player_id", "game_date"]

def load_joint_dataset(dataset_dir: Path | None = None) -> JointDataset:
    """Load a joint rotation/rates dataset.
    
    If dataset_dir is None, uses NBA_DFS_JOINT_DATASET_DIR from config.
    
    Reads features.parquet, labels_rates.parquet, labels_boxscore_counts.parquet,
    and manifest.json from the dataset directory. Validates that all expected
    files exist and that row counts match the manifest.
    
    Returns a JointDataset with all four components loaded.
    """
```

Implementation notes:
- Import `NBA_DFS_JOINT_DATASET_DIR` from `nba_dfs.config` for the default.
- Read all three parquet files with `pd.read_parquet`.
- Read `manifest.json` with `json.load`.
- Validate that all four files exist before reading any of them. Raise `FileNotFoundError` with a helpful message listing the missing file(s).
- Normalize `game_date` to `pd.Timestamp` in both features and labels after loading.
- Do NOT join the DataFrames together. Return them separately. The consumer decides how to join.
- This is a data-layer module. It reads from disk and returns DataFrames. No business logic.

---

#### `src/nba_dfs/data/boxscore_labels.py` — NEW

Purpose: helper for accessing boxscore count labels from a JointDataset.

Public API:

```python
BOXSCORE_COUNT_COLS: list[str] = [
    "fg2m", "fg3m", "ftm", "fga2", "fga3", "fta", "minutes", "played",
    "ast", "oreb", "dreb", "stl", "blk", "tov",
]

def select_boxscore_counts(labels_boxscore: pd.DataFrame) -> pd.DataFrame:
    """Select join keys + available boxscore count columns from the labels frame.
    
    Returns a DataFrame with JOIN_KEYS plus whichever BOXSCORE_COUNT_COLS
    are actually present in the input. Does not fail if some columns are missing.
    """
```

This is intentionally small. It exists to centralize the column-name contract for boxscore counts rather than hardcoding column names in the experiment script.

---

#### `src/nba_dfs/data/__init__.py` — MODIFY

Add re-exports for the new public functions:

```python
from nba_dfs.data.joint_dataset import load_joint_dataset, JointDataset, JOIN_KEYS
from nba_dfs.data.boxscore_labels import select_boxscore_counts, BOXSCORE_COUNT_COLS
```

Keep existing exports. Extend `__all__` if it exists.

---

#### `src/nba_dfs/rates/priors.py` — NEW

Purpose: compute rolling Bayesian-shrunk efficiency priors from historical box score data.

Port the `_compute_rolling_efficiency` function from the v2 experiment script. Clean it up:

```python
# League average fallbacks (2024-25 NBA season approx)
LEAGUE_FG2_PCT: float = 0.545
LEAGUE_FG3_PCT: float = 0.365
LEAGUE_FT_PCT: float = 0.780

PRIOR_COLS: list[str] = [
    "prior_fg2_pct", "prior_fg3_pct", "prior_ft_pct",
    "prior_fg2_n", "prior_fg3_n", "prior_ft_n",
]

def compute_rolling_efficiency_priors(
    boxscore_df: pd.DataFrame,
    *,
    min_attempts: int = 10,
    join_keys: list[str] | None = None,
) -> pd.DataFrame:
    """Compute cumulative season-average shooting percentages per player.
    
    For each game, the prior is computed from ALL previous games in the dataset
    (strictly before the current game_date). This is anti-leak by construction.
    
    For players with fewer than `min_attempts`, shrinks toward league average
    using a Bayesian blend: prior = (player_makes + k*league) / (player_att + k)
    where k = min_attempts.
    
    Args:
        boxscore_df: must contain join_keys + fg2m, fg3m, ftm, fga2, fga3, fta columns.
        min_attempts: shrinkage strength (higher = more shrinkage toward league avg).
        join_keys: defaults to JOIN_KEYS from nba_dfs.data.
    
    Returns:
        DataFrame with join_keys + PRIOR_COLS (6 columns: 3 pct priors + 3 attempt counts).
    """
```

Implementation: port the logic from `lgbm_scoring_rates_v2.py:100-133` with these cleanups:
- Use `JOIN_KEYS` from `nba_dfs.data` as the default join keys.
- Keep the Bayesian shrinkage formula exactly as-is — this is scoring math, not a cleanup target.
- Add type hints.
- Add a module docstring explaining what rolling priors are and why they're used instead of trained efficiency models.

---

#### `src/nba_dfs/training/__init__.py` — NEW

Package init. Brief docstring:

```python
"""Training utilities for nba-dfs-v2.

This package provides dataset loading, splitting, weighting, and model
training primitives. It is consumed by experiment scripts in scripts/,
not by the inference pipeline.

The package does not import from scripts/. Experiments are scripts that
import from this package.
"""
```

No re-exports needed initially. Each module has its own import path.

---

#### `src/nba_dfs/training/README.md` — NEW

Document:
- What this module is: training utilities consumed by experiment scripts.
- The three modules and their roles: `split.py` (train/val/test splitting), `weights.py` (sample weighting), `train_rates.py` (rates-specific GBM trainer).
- The discipline: experiments live in `scripts/experiments/`, not in this package. This package provides stable primitives.
- The bundle format constraint: `train_rates.py` produces v1-compatible bundles loadable by the existing `nba_dfs.rates` scorer. See `docs/RATES_V2_BUNDLE_FORMAT_INVESTIGATION.md` for the format spec.

---

#### `src/nba_dfs/training/split.py` — NEW

Purpose: time-based splitting of training data.

```python
def time_split(
    df: pd.DataFrame,
    *,
    val_frac: float = 0.15,
    test_frac: float = 0.10,
    date_col: str = "game_date",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into train/val/test by date.
    
    Sorts unique dates, then assigns the last test_frac of dates to test,
    the preceding val_frac to val, and the remainder to train.
    
    Returns (train, val, test) as copies.
    """
```

Port from `lgbm_scoring_rates_v2.py:88-97`. Add type hints, docstring, date_col parameter.

---

#### `src/nba_dfs/training/weights.py` — NEW

Purpose: sample weighting utilities.

Two functions:

```python
def compute_recency_weights(
    df: pd.DataFrame,
    *,
    anchor_date: pd.Timestamp | str,
    half_life_days: float = 75.0,
    min_weight: float = 0.0,
    date_col: str = "game_date",
) -> np.ndarray:
    """Exponential decay weights anchored on anchor_date.
    
    weight = max(0.5^(age_days / half_life_days), min_weight)
    
    Ported from train_rates_v1.py's recency weighting logic (v1 has this,
    v2 experiment does not — but we want to support it for experimentation).
    """

def upweight_by_threshold(
    df: pd.DataFrame,
    *,
    column: str,
    threshold: float,
    multiplier: float,
) -> np.ndarray | None:
    """Upweight rows where column >= threshold by multiplier.
    
    Returns weight array, or None if multiplier <= 1.0 or column is missing.
    Ported from lgbm_scoring_rates_v2.py's _upweight_high_usage.
    """
```

Port the recency weighting from `train_rates_v1.py:_compute_recency_sample_weights` and the star upweighting from `lgbm_scoring_rates_v2.py:168-175`. Both are small, self-contained helpers.

---

#### `src/nba_dfs/training/train_rates.py` — NEW

Purpose: train LightGBM rate models and write a v1-compatible bundle.

This is the core of Pass 2. It combines the v2 experiment's training approach with the v1 bundle format.

Public API:

```python
@dataclass(frozen=True)
class RatesTrainingResult:
    """Result of a rates training run."""
    bundle_dir: Path
    models: dict[str, lgb.Booster]
    feature_columns: list[str]
    metrics: dict[str, dict[str, float]]  # per-target metrics
    meta: dict  # the full meta.json contents

# Default targets (v2 approach: opportunity + counting, no efficiency)
OPPORTUNITY_TARGETS: list[str] = ["fga2_per_min", "fga3_per_min", "fta_per_min"]
COUNTING_TARGETS: list[str] = [
    "ast_per_min", "oreb_per_min", "dreb_per_min",
    "stl_per_min", "blk_per_min", "tov_per_min",
]
DEFAULT_TARGETS: list[str] = OPPORTUNITY_TARGETS + COUNTING_TARGETS

DEFAULT_PARAMS: dict = {
    "objective": "regression",
    "metric": "mae",
    "num_leaves": 64,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_data_in_leaf": 50,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "seed": 42,
    "n_jobs": -1,
}

def train_rates_bundle(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    feature_columns: list[str],
    targets: list[str] = DEFAULT_TARGETS,
    params: dict | None = None,
    num_boost_round: int = 3000,
    early_stopping_rounds: int = 100,
    output_dir: Path,
    sample_weights_train: np.ndarray | None = None,
    sample_weights_val: np.ndarray | None = None,
    preprocess_meta: dict | None = None,
    run_id: str | None = None,
) -> RatesTrainingResult:
    """Train LightGBM rate models and write a v1-compatible bundle.
    
    For each target:
      1. Filter to rows where the target label is not null.
      2. Train a LightGBM booster with early stopping on val.
      3. Evaluate on val (MAE, bias, correlation).
      4. Save model as model_<target>.txt in output_dir.
    
    After all targets:
      5. Write feature_cols.json with {"feature_cols": feature_columns}.
      6. Write meta.json with targets, feature_cols, params, preprocess metadata,
         row counts, and run metadata.
      7. Write metrics.json with per-target val metrics.
    
    Args:
        train_df: training data with feature_columns + target columns.
        val_df: validation data with same schema.
        feature_columns: ordered list of feature column names.
        targets: list of target column names to train.
        params: LightGBM params dict. Defaults to DEFAULT_PARAMS.
        num_boost_round: max boosting rounds.
        early_stopping_rounds: early stopping patience.
        output_dir: where to write the bundle.
        sample_weights_train: optional per-row weights for training.
        sample_weights_val: optional per-row weights for validation.
        preprocess_meta: dict with "odds_fill_values" and "tracking_fill_values"
            to persist in meta.json. Required for v1 scorer compatibility.
        run_id: optional run identifier for meta.json.
    
    Returns:
        RatesTrainingResult with the trained models, feature columns,
        per-target metrics, and the full meta dict.
    """
```

Implementation notes:

- The per-target training loop should be sequential (same as v2 experiment and v1 trainer). No parallelism.
- Use `lgb.Dataset`, `lgb.train`, `lgb.early_stopping`, `lgb.log_evaluation(500)` — same as the v2 experiment.
- Save models with `model.save_model(str(output_dir / f"model_{target}.txt"), num_iteration=model.best_iteration)`. Note: root-level, `model_` prefix, `.txt` extension. This is the v1 convention the loader expects.
- `feature_cols.json` structure: `{"feature_cols": feature_columns}`. Exact key name matters.
- `meta.json` must contain at minimum: `targets`, `feature_cols`, `params`, `preprocess` (with `odds_fill_values` and `tracking_fill_values` sub-dicts), `train_rows`, `val_rows`, `run_id`, `created_at`. Other fields are optional stubs for documentation.
- `metrics.json`: per-target `{"best_iteration": int, "val_mae": float, "val_bias": float, "val_corr": float}`.
- `preprocess_meta` is passed in by the caller (the experiment script), not computed inside this function. The trainer doesn't know about the data layer or preprocessing; it just persists what it's told. This keeps the boundary clean.
- If `params` is None, use `DEFAULT_PARAMS`.
- If `run_id` is None, generate one from the current UTC timestamp.

**Do not** include MLflow logging, registry registration, or any operational infrastructure. This is a pure training function.

---

#### `src/nba_dfs/eval/__init__.py` — NEW

Package init with brief docstring:

```python
"""Evaluation utilities for nba-dfs-v2 experiments."""
```

---

#### `src/nba_dfs/eval/README.md` — NEW

One paragraph: this module provides evaluation metric helpers (MAE, bias, correlation) and DFS-specific evaluation (DK FPTS scoring) used by experiment scripts. Not part of the inference pipeline.

---

#### `src/nba_dfs/eval/metrics.py` — NEW

Port the evaluation helpers from the v2 experiment. Clean up into library functions:

```python
def metric_row(actual: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """Compute MAE, bias, and correlation for a prediction vector."""

def safe_corr(actual: np.ndarray, pred: np.ndarray) -> float:
    """Pearson correlation with safe handling of constant/empty arrays."""

def clip_non_negative(values: np.ndarray) -> np.ndarray:
    """Clip to [0, inf)."""

def clip_pct(values: np.ndarray, fallback: float = 0.0) -> np.ndarray:
    """Clip to [0, 1] with non-finite fallback."""

DK_SCORING: dict[str, float] = {
    "pts": 1.0,
    "reb": 1.25,
    "ast": 1.5,
    "stl": 2.0,
    "blk": 2.0,
    "tov": -0.5,
    "fg3m": 0.5,  # three-point bonus
}

def compute_dk_fpts(
    pts: np.ndarray,
    reb: np.ndarray,
    ast: np.ndarray,
    stl: np.ndarray,
    blk: np.ndarray,
    tov: np.ndarray,
    fg3m: np.ndarray,
) -> np.ndarray:
    """Compute DraftKings fantasy points from counting stats."""
```

Port from `lgbm_scoring_rates_v2.py:178-195` (metric helpers) and the DK FPTS computation at lines 479-481. Keep the math exactly as-is.

---

#### `scripts/experiments/rates_gbm_v2.py` — NEW

**This is an experiment script, not a library module.** It lives outside the package in `scripts/experiments/`. It imports from `nba_dfs.*` and composes the primitives into the full v2 experiment.

The script should:

1. Parse CLI args: `--dataset-dir` (optional, defaults to config), `--out-dir` (optional, auto-generated), `--n-rounds`, `--lr`, `--upweight-stars`, `--eff-min-attempts`.

2. Load the joint dataset via `nba_dfs.data.load_joint_dataset`.

3. Compute rolling efficiency priors via `nba_dfs.rates.priors.compute_rolling_efficiency_priors` from the boxscore labels.

4. Join features + labels_rates + selected boxscore counts + efficiency priors on JOIN_KEYS.

5. Filter to active rows (`minutes_actual >= 4.0`).

6. Select feature columns. Port `_select_features` from the v2 experiment. This logic stays in the script because it's experiment-specific (the exclude/include lists may change between experiments). **Do not put this in the library.**

7. Split into train/val/test via `nba_dfs.training.split.time_split`.

8. **Fit preprocessing fill values on the train split.** This is the Option A decision from Pass 1. Port the `fit_odds_fill_values` and `fit_tracking_fill_values` logic from the existing `nba_dfs.rates.preprocess` module. Call the functions from `nba_dfs.rates.preprocess` directly — they already exist in the ported rates_v1 code. Apply the fill values to train/val/test splits. Store the fitted values for later persistence into the bundle.

9. Compute star-upweighting via `nba_dfs.training.weights.upweight_by_threshold` per target.

10. Train via `nba_dfs.training.train_rates.train_rates_bundle`, passing the fitted preprocess metadata.

11. Evaluate: compute per-target rate metrics, rolling prior metrics, counting stats from decomposed predictions (LightGBM opportunity rates × priors), DK FPTS. Use `nba_dfs.eval.metrics` helpers.

12. Save evaluation report and test predictions alongside the bundle.

**Important:** The preprocessing fill step (step 8) imports from `nba_dfs.rates.preprocess`, which is part of the existing ported rates_v1 code. Verify that the functions `fit_odds_fill_values`, `apply_odds_fill_values`, `fit_tracking_fill_values`, `apply_tracking_fill_values` exist in the ported `nba_dfs.rates.preprocess` module before using them. If they don't exist (because the refactor-as-you-go port dropped them), **stop and report** — this is a real dependency that needs to be resolved.

**Important:** Steps 11-12 (evaluation) should be comprehensive but are not the core deliverable. If you need to simplify to keep the script under ~250 lines, the training and bundle-writing path (steps 1-10) takes priority over the evaluation detail. A minimal evaluation that reports per-target MAE and overall DK FPTS MAE is sufficient. The full bucketed analysis from the source v2 script is a nice-to-have, not a must-have.

---

#### `tests/test_rates_training_smoke.py` — NEW

Placeholder smoke test, same pattern as the other smoke tests:

```python
"""Smoke test for the rates training pipeline.

This test validates that:
1. The joint dataset loads from NBA_DFS_JOINT_DATASET_DIR
2. Rolling efficiency priors compute without error
3. The trainer produces a v1-compatible bundle
4. The existing nba_dfs.rates.load_rates_bundle can load the trained bundle

The user will fill in the validation logic after running the experiment.
"""
import pytest

@pytest.mark.requires_data
def test_rates_training_smoke():
    pytest.skip("Training smoke test not yet implemented; run the experiment first.")
```

---

#### `docs/PORT_NOTES.md` — MODIFY

Append a new section:

```markdown
## Port: rates GBM v2 experiment primitives

Date: <today>
Source: projections-v2 @ <git sha of source repo HEAD>
Reference script: scripts/experiments/lgbm_scoring_rates_v2.py

### What was ported

Library primitives extracted from the v2 experiment script:
- nba_dfs.data.joint_dataset — joint dataset loader
- nba_dfs.data.boxscore_labels — boxscore count column contract
- nba_dfs.rates.priors — rolling Bayesian-shrunk efficiency priors
- nba_dfs.training.split — time-based train/val/test splitting
- nba_dfs.training.weights — recency weighting and star upweighting
- nba_dfs.training.train_rates — rates GBM trainer with v1-compatible bundle output
- nba_dfs.eval.metrics — evaluation metric helpers and DK FPTS computation

Experiment script: scripts/experiments/rates_gbm_v2.py

### Key design decisions

- Bundle format: v1-compatible per RATES_V2_BUNDLE_FORMAT_INVESTIGATION.md (category B).
- Preprocessing: Option A — fit odds/tracking fill values on train split, apply to all splits, persist in meta.json. Uses existing nba_dfs.rates.preprocess functions.
- Efficiency: NOT in the bundle. Rolling priors are computed at experiment time from boxscore history. Production scoring of the decomposed model is deferred.
- Feature selection: remains in the experiment script, not in the library. Feature exclude/include lists are experiment-specific.
- Evaluation: DK FPTS is the primary evaluation metric, matching the v2 experiment's approach.

### Scoring math preserved exactly
- Bayesian shrinkage formula for efficiency priors
- LightGBM training loop and hyperparameters
- DK FPTS computation
- Efficiency clamp ranges (via existing scorer)
- Star upweighting thresholds and multipliers

### What was NOT ported
- MLflow integration
- Prefect orchestration
- Registry/manifest writes
- Auto-promotion logic
- Full bucketed evaluation (simplified in the experiment script if needed for length)

### Open questions
- The experiment has not been run yet in either the old or new repo against the canonical 20260331_teamimplied dataset. Baseline metrics are not established.
- Production scoring of the decomposed v2 model (opportunity rates × priors → points) is not yet wired into the inference pipeline. This is deferred until the experiment proves the approach is worth productionizing.
- The exact post-filter feature count from _select_features on the canonical dataset is not known without running the code.
```

---

## Commit

Make a single commit with the message:

```
Port rates GBM v2 experiment primitives and script
```

Do not push.

## Stop conditions

Stop and report (do not improvise) if any of the following:

1. `nba_dfs.rates.preprocess` does not contain `fit_odds_fill_values` / `apply_odds_fill_values` / `fit_tracking_fill_values` / `apply_tracking_fill_values`. The experiment script depends on these.
2. The existing `nna_dfs.rates.loader` or `nba_dfs.rates.score` needs changes to load a 9-target bundle. Pass 1 said it wouldn't, but if it does, that's a finding.
3. `pyarrow` is not installable or conflicts with existing deps. Check `pyproject.toml` for conflicts before adding.
4. You need to import from `scripts/` in any library module. This violates the repo discipline.
5. Any library module exceeds ~300 lines. If it does, it's doing too much and should be split.
6. The experiment script exceeds ~300 lines. If it does, push more logic into library primitives.
7. You find yourself making scoring-math decisions (changing clamp values, changing the shrinkage formula, changing DK weights). Math stays exactly as-is from the source.

## What success looks like

- New modules: `nba_dfs.data.joint_dataset`, `nba_dfs.data.boxscore_labels`, `nba_dfs.rates.priors`, `nna_dfs.training.split`, `nba_dfs.training.weights`, `nba_dfs.training.train_rates`, `nba_dfs.eval.metrics`
- New experiment script: `scripts/experiments/rates_gbm_v2.py`
- Updated `config.py` with `NBA_DFS_JOINT_DATASET_DIR`
- Updated `pyproject.toml` with `pyarrow`
- Updated `docs/PORT_NOTES.md`
- Placeholder smoke test
- All new Python files pass `ast.parse` or `ruff check`
- The existing `nba_dfs.rates` scorer is completely untouched
- One clean commit
- A clear report of any anomalies or stop conditions encountered
