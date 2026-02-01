#!/usr/bin/env python3
"""
Conservative rotation predictor classifier (v1).

Trains p_ge5 / p_ge15 classifiers for DFS risk control.
Goal: prevent fringe players from being over-promoted while catching real promotions.

Usage:
    uv run python -m projections.cli.train_rotation_predictor_v1 \
      --rotation-set-artifact artifacts/rotation_set_minutes/rotation_set_minutes_v1_settransformer_20260126T024705Z \
      --run-id <timestamp>
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import typer

# ==============================================================================
# CONFIGURABLE SAMPLE WEIGHTS (DFS RISK CONTROL)
# ==============================================================================
WEIGHT_BASE = 1.0
WEIGHT_MULT_LE_5 = 6.0       # multiply by 6 if minutes_actual <= 5
WEIGHT_MULT_LE_1 = 10.0      # multiply by 10 if minutes_actual <= 1
WEIGHT_MULT_CHAOS_LE_5 = 1.5 # extra multiplier for chaos/deep-bench + low min

# ==============================================================================
# CONSTANTS
# ==============================================================================
RANDOM_STATE = 42
UTC = timezone.utc

# Columns to explicitly exclude from features
EXCLUDE_PATTERNS = [
    "game_id", "player_id", "team_id", "season_id",
    "actual", "label", "target", "minutes_actual",
    "player_name", "team_name", "team_tricode", "opponent_team_name",
    "opponent_team_tricode", "season", "game_date", "tip_ts", "home_team_id",
    "away_team_id", "opponent_team_id", "status", "injury_as_of_ts", "odds_as_of_ts",
    "archetype", "pos_bucket", "roster_as_of_ts", "lineup_role", "lineup_status",
    "lineup_roster_status", "lineup_timestamp", "feature_as_of_ts", "snapshot_type",
    "snapshot_ts", "frozen_at", "source", "time_unit_detected",
    "team_prior_source_max_game_date_5", "team_prior_source_max_game_date_10",
    "team_prior_source_max_game_date_20", "player_prior_source_max_game_date_5",
    "player_prior_source_max_game_date_10", "player_prior_source_max_game_date_20",
]

app = typer.Typer()


def _load_manifest(artifact_dir: Path) -> dict[str, Any]:
    """Load manifest.json from rotation-set artifact."""
    manifest_path = artifact_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found in {artifact_dir}")
    with open(manifest_path) as f:
        return json.load(f)


def _load_dataset(manifest: dict[str, Any]) -> pd.DataFrame:
    """Load features + labels from dataset specified in manifest."""
    features_path = Path(manifest["inputs"]["features"])
    labels_path = Path(manifest["inputs"]["labels"])

    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    features = pd.read_parquet(features_path)
    labels = pd.read_parquet(labels_path)

    # Rename labels 'minutes' column to 'minutes_actual' to avoid confusion
    if "minutes" in labels.columns:
        labels = labels.rename(columns={"minutes": "minutes_actual"})

    # Merge on keys
    merge_keys = ["game_id", "team_id", "player_id"]
    df = features.merge(labels[merge_keys + ["minutes_actual"]], on=merge_keys, how="inner")

    return df


def _select_feature_columns(df: pd.DataFrame, manifest: dict[str, Any]) -> list[str]:
    """Select feature columns from manifest, excluding identifiers and labels."""
    # Use feature_columns from manifest.model
    feature_cols = manifest.get("model", {}).get("feature_columns", [])

    if not feature_cols:
        raise ValueError("No feature_columns found in manifest.model")

    # Filter to only columns present in df
    available = [c for c in feature_cols if c in df.columns]

    # Additional safety: exclude any that match exclude patterns
    final = []
    for c in available:
        exclude = False
        for pattern in EXCLUDE_PATTERNS:
            if pattern in c.lower() or c == pattern:
                exclude = True
                break
        if not exclude:
            final.append(c)

    return final


def _derive_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Derive binary classification labels from minutes_actual."""
    df = df.copy()
    df["y_ge5"] = (df["minutes_actual"] >= 5).astype(int)
    df["y_ge15"] = (df["minutes_actual"] >= 15).astype(int)
    return df


def _compute_sample_weights(
    df: pd.DataFrame,
    weight_base: float = WEIGHT_BASE,
    weight_mult_le5: float = WEIGHT_MULT_LE_5,
    weight_mult_le1: float = WEIGHT_MULT_LE_1,
    weight_mult_chaos_le5: float = WEIGHT_MULT_CHAOS_LE_5,
) -> np.ndarray:
    """Compute DFS risk-control sample weights."""
    weights = np.full(len(df), weight_base, dtype=float)

    # Up-weight low-minute outcomes
    le5_mask = df["minutes_actual"].values <= 5
    le1_mask = df["minutes_actual"].values <= 1

    weights[le5_mask] *= weight_mult_le5
    weights[le1_mask] *= weight_mult_le1

    # Extra weight for chaos/deep-bench regime (if column exists)
    if "regime_label" in df.columns:
        chaos_mask = df["regime_label"].str.lower().isin(["chaos", "deep_bench", "deep bench"])
        chaos_le5_mask = chaos_mask.values & le5_mask
        weights[chaos_le5_mask] *= weight_mult_chaos_le5

    return weights


def _split_by_game_date(
    df: pd.DataFrame,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Time-based split by game_date (by game, not row)."""
    # Get unique games sorted by date
    if "game_date" in df.columns:
        game_key = df.groupby("game_id")["game_date"].first().sort_values()
    else:
        # Fallback to game_id order
        game_key = pd.Series(df["game_id"].unique(), index=df["game_id"].unique())
        game_key = game_key.sort_index()

    game_ids_sorted = game_key.index.tolist()
    n_games = len(game_ids_sorted)

    train_end = int(n_games * train_frac)
    val_end = int(n_games * (train_frac + val_frac))

    train_games = set(game_ids_sorted[:train_end])
    val_games = set(game_ids_sorted[train_end:val_end])
    test_games = set(game_ids_sorted[val_end:])

    train_df = df[df["game_id"].isin(train_games)].copy()
    val_df = df[df["game_id"].isin(val_games)].copy()
    test_df = df[df["game_id"].isin(test_games)].copy()

    return train_df, val_df, test_df


def _train_classifier(
    X: pd.DataFrame,
    y: np.ndarray,
    weights: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    random_state: int = RANDOM_STATE,
) -> lgb.Booster:
    """Train LightGBM classifier with conservative hyperparameters."""
    train_data = lgb.Dataset(X, label=y, weight=weights, free_raw_data=False)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, free_raw_data=False)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 100,  # Conservative
        "min_data_in_leaf": 100,
        "max_depth": 6,
        "seed": random_state,
        "deterministic": True,
        "force_row_wise": True,
        "verbose": -1,
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, val_data],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100),
        ],
    )

    return model


def _brier_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Brier score (lower is better)."""
    return float(np.mean((y_pred - y_true) ** 2))


def _calibration_bins(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
) -> list[dict[str, float]]:
    """Compute calibration bin statistics."""
    bins = []
    edges = np.linspace(0, 1, n_bins + 1)

    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (y_pred >= lo) & (y_pred < hi)
        if i == n_bins - 1:
            mask = (y_pred >= lo) & (y_pred <= hi)

        n = int(mask.sum())
        if n == 0:
            bins.append({"bin": f"{lo:.1f}-{hi:.1f}", "count": 0, "p_mean": 0.0, "y_mean": 0.0})
        else:
            bins.append({
                "bin": f"{lo:.1f}-{hi:.1f}",
                "count": n,
                "p_mean": float(y_pred[mask].mean()),
                "y_mean": float(y_true[mask].mean()),
            })

    return bins


def _fpr_at_thresholds(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    minutes_actual: np.ndarray,
    minutes_threshold: float,
    thresholds: list[float],
) -> dict[float, float]:
    """Compute FPR for predictions where truth minutes <= threshold."""
    result = {}
    low_min_mask = minutes_actual <= minutes_threshold

    for th in thresholds:
        pred_pos = y_pred >= th
        # FP = pred_pos & low_min
        fp = (pred_pos & low_min_mask).sum()
        # All low-min cases
        total = low_min_mask.sum()
        fpr = float(fp / total) if total > 0 else 0.0
        result[th] = fpr

    return result


def _recall_at_thresholds(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    minutes_actual: np.ndarray,
    minutes_threshold: float,
    thresholds: list[float],
) -> dict[float, float]:
    """Compute recall for predictions where truth minutes >= threshold."""
    result = {}
    high_min_mask = minutes_actual >= minutes_threshold

    for th in thresholds:
        pred_pos = y_pred >= th
        # TP = pred_pos & high_min
        tp = (pred_pos & high_min_mask).sum()
        # All high-min cases
        total = high_min_mask.sum()
        recall = float(tp / total) if total > 0 else 0.0
        result[th] = recall

    return result


def _find_ghosts(
    test_df: pd.DataFrame,
    p_col: str,
    n: int = 25,
) -> pd.DataFrame:
    """Find top-N by p where minutes_actual <= 5."""
    low_min = test_df[test_df["minutes_actual"] <= 5].copy()
    low_min = low_min.nlargest(n, p_col)
    cols = ["game_id", "player_id", "player_name", "team_name", "minutes_actual", p_col]
    cols = [c for c in cols if c in low_min.columns]
    return low_min[cols]


def _find_missed_promotions(
    test_df: pd.DataFrame,
    p_col: str,
    minutes_threshold: float,
    prob_threshold: float = 0.2,
    n: int = 25,
) -> pd.DataFrame:
    """Find top-N actual minutes where p < prob_threshold."""
    high_min = test_df[test_df["minutes_actual"] >= minutes_threshold].copy()
    missed = high_min[high_min[p_col] < prob_threshold]
    missed = missed.nlargest(n, "minutes_actual")
    cols = ["game_id", "player_id", "player_name", "team_name", "minutes_actual", p_col]
    cols = [c for c in cols if c in missed.columns]
    return missed[cols]


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Convert DataFrame to markdown table without tabulate dependency."""
    if df.empty:
        return "_Empty table._"

    lines = []
    # Header row
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    lines.append(header)
    # Separator
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    lines.append(sep)
    # Data rows
    for _, row in df.iterrows():
        cells = []
        for val in row:
            if isinstance(val, float):
                cells.append(f"{val:.3f}")
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def _generate_report(
    test_df: pd.DataFrame,
    metrics_ge5: dict[str, Any],
    metrics_ge15: dict[str, Any],
    split_info: dict[str, Any],
    feature_columns: list[str],
) -> str:
    """Generate markdown report."""
    lines = ["# Rotation Predictor v1 Report", ""]

    # Split info
    lines.append("## Dataset Split")
    lines.append(f"- Train rows: {split_info['train_rows']:,}")
    lines.append(f"- Val rows: {split_info['val_rows']:,}")
    lines.append(f"- Test rows: {split_info['test_rows']:,}")
    lines.append(f"- Train date range: {split_info['train_date_min']} to {split_info['train_date_max']}")
    lines.append(f"- Val date range: {split_info['val_date_min']} to {split_info['val_date_max']}")
    lines.append(f"- Test date range: {split_info['test_date_min']} to {split_info['test_date_max']}")
    lines.append("")

    # Features
    lines.append("## Features")
    lines.append(f"- Number of features: {len(feature_columns)}")
    lines.append("")

    # Metrics for ge5
    lines.append("## p_ge5 (minutes >= 5)")
    lines.append(f"- Brier score: {metrics_ge5['brier']:.4f}")
    lines.append("")

    lines.append("### Calibration")
    lines.append("| Bin | Count | P_mean | Y_mean |")
    lines.append("|-----|-------|--------|--------|")
    for b in metrics_ge5["calibration_bins"]:
        lines.append(f"| {b['bin']} | {b['count']:,} | {b['p_mean']:.3f} | {b['y_mean']:.3f} |")
    lines.append("")

    lines.append("### False Positive Rate (truth min <= 5)")
    lines.append("| Threshold | FPR |")
    lines.append("|-----------|-----|")
    for th, fpr in sorted(metrics_ge5["fpr"].items()):
        lines.append(f"| {th:.1f} | {fpr:.3f} |")
    lines.append("")

    # Metrics for ge15
    lines.append("## p_ge15 (minutes >= 15)")
    lines.append(f"- Brier score: {metrics_ge15['brier']:.4f}")
    lines.append("")

    lines.append("### Calibration")
    lines.append("| Bin | Count | P_mean | Y_mean |")
    lines.append("|-----|-------|--------|--------|")
    for b in metrics_ge15["calibration_bins"]:
        lines.append(f"| {b['bin']} | {b['count']:,} | {b['p_mean']:.3f} | {b['y_mean']:.3f} |")
    lines.append("")

    lines.append("### Recall for truth >= 15")
    lines.append("| Threshold | Recall |")
    lines.append("|-----------|--------|")
    for th, recall in sorted(metrics_ge15["recall"].items()):
        lines.append(f"| {th:.1f} | {recall:.3f} |")
    lines.append("")

    # Ghosts
    lines.append("## Ghosts (top 25 by p_ge5 where minutes <= 5)")
    ghosts = _find_ghosts(test_df, "p_ge5", 25)
    if len(ghosts) > 0:
        lines.append(_df_to_markdown(ghosts))
    else:
        lines.append("_No ghosts found._")
    lines.append("")

    # Missed promotions
    lines.append("## Missed Promotions (top 25 actual minutes where p_ge15 < 0.2)")
    missed = _find_missed_promotions(test_df, "p_ge15", 15.0, 0.2, 25)
    if len(missed) > 0:
        lines.append(_df_to_markdown(missed))
    else:
        lines.append("_No missed promotions found._")
    lines.append("")

    return "\n".join(lines)


@app.command()
def main(
    rotation_set_artifact: Path = typer.Option(
        ...,
        help="Path to rotation-set transformer artifact directory (with manifest.json).",
    ),
    run_id: str = typer.Option(
        ...,
        help="Run ID for output artifact (e.g., timestamp).",
    ),
    artifact_root: Path = typer.Option(
        Path("artifacts/rotation_predictor_v1"),
        help="Root directory for output artifacts.",
    ),
    overwrite: bool = typer.Option(
        False,
        help="Overwrite existing artifact if present.",
    ),
    max_rows: int | None = typer.Option(
        None,
        help="Limit rows for smoke testing (optional).",
    ),
    random_state: int = typer.Option(
        RANDOM_STATE,
        help="Random seed for reproducibility.",
    ),
) -> None:
    """Train conservative rotation predictor classifiers (p_ge5, p_ge15)."""
    rotation_set_artifact = rotation_set_artifact.expanduser().resolve()
    artifact_root = artifact_root.expanduser().resolve()
    run_dir = artifact_root / run_id

    if run_dir.exists() and not overwrite:
        raise FileExistsError(f"Artifact directory already exists: {run_dir}. Use --overwrite to replace.")

    run_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(f"[load] Loading manifest from {rotation_set_artifact}")
    manifest = _load_manifest(rotation_set_artifact)

    dataset_dir = manifest.get("dataset_dir", "")
    typer.echo(f"[info] Dataset dir: {dataset_dir}")

    typer.echo("[load] Loading dataset (features + labels)")
    df = _load_dataset(manifest)
    typer.echo(f"[info] Loaded {len(df):,} rows")

    if max_rows is not None and max_rows < len(df):
        typer.echo(f"[limit] Sampling first {max_rows:,} rows for smoke test")
        # Sample by game to preserve structure
        game_ids = df["game_id"].unique()
        n_games_sample = max(1, int(len(game_ids) * (max_rows / len(df))))
        sample_games = game_ids[:n_games_sample]
        df = df[df["game_id"].isin(sample_games)]
        typer.echo(f"[limit] After limiting: {len(df):,} rows")

    # Select features
    typer.echo("[features] Selecting feature columns from manifest")
    feature_columns = _select_feature_columns(df, manifest)
    typer.echo(f"[features] {len(feature_columns)} feature columns selected")

    # Derive labels
    typer.echo("[labels] Deriving y_ge5 and y_ge15")
    df = _derive_labels(df)

    # Split
    typer.echo("[split] Time-based split (80% train, 10% val, 10% test)")
    train_df, val_df, test_df = _split_by_game_date(df)
    typer.echo(f"[split] Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")

    # Compute sample weights (train only)
    typer.echo("[weights] Computing DFS risk-control sample weights")
    weights = _compute_sample_weights(train_df)

    # Prepare feature matrices
    X_train = train_df[feature_columns].astype(float)
    X_val = val_df[feature_columns].astype(float)
    X_test = test_df[feature_columns].astype(float)

    # Train p_ge5
    typer.echo("[train] Training p_ge5 classifier")
    y_train_ge5 = train_df["y_ge5"].values
    y_val_ge5 = val_df["y_ge5"].values
    model_ge5 = _train_classifier(X_train, y_train_ge5, weights, X_val, y_val_ge5, random_state)

    # Train p_ge15
    typer.echo("[train] Training p_ge15 classifier")
    y_train_ge15 = train_df["y_ge15"].values
    y_val_ge15 = val_df["y_ge15"].values
    model_ge15 = _train_classifier(X_train, y_train_ge15, weights, X_val, y_val_ge15, random_state)

    # Predict on test
    typer.echo("[predict] Generating predictions on test set")
    test_df = test_df.copy()
    test_df["p_ge5"] = model_ge5.predict(X_test)
    test_df["p_ge15"] = model_ge15.predict(X_test)

    # Compute metrics
    y_test_ge5 = test_df["y_ge5"].values
    y_test_ge15 = test_df["y_ge15"].values
    p_ge5 = test_df["p_ge5"].values
    p_ge15 = test_df["p_ge15"].values
    minutes_actual = test_df["minutes_actual"].values

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]

    metrics_ge5 = {
        "brier": _brier_score(y_test_ge5, p_ge5),
        "calibration_bins": _calibration_bins(y_test_ge5, p_ge5),
        "fpr": _fpr_at_thresholds(y_test_ge5, p_ge5, minutes_actual, 5.0, thresholds),
    }

    metrics_ge15 = {
        "brier": _brier_score(y_test_ge15, p_ge15),
        "calibration_bins": _calibration_bins(y_test_ge15, p_ge15),
        "recall": _recall_at_thresholds(y_test_ge15, p_ge15, minutes_actual, 15.0, thresholds),
    }

    # Split info
    def _date_range(ddf: pd.DataFrame) -> tuple[str, str]:
        if "game_date" in ddf.columns and len(ddf) > 0:
            dates = pd.to_datetime(ddf["game_date"])
            return str(dates.min().date()), str(dates.max().date())
        return "N/A", "N/A"

    train_min, train_max = _date_range(train_df)
    val_min, val_max = _date_range(val_df)
    test_min, test_max = _date_range(test_df)

    split_info = {
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "train_date_min": train_min,
        "train_date_max": train_max,
        "val_date_min": val_min,
        "val_date_max": val_max,
        "test_date_min": test_min,
        "test_date_max": test_max,
    }

    # Generate report
    typer.echo("[report] Generating report.md")
    report = _generate_report(test_df, metrics_ge5, metrics_ge15, split_info, feature_columns)
    (run_dir / "report.md").write_text(report, encoding="utf-8")

    # Save predictions
    typer.echo("[save] Saving predictions_test.parquet")
    pred_cols = ["game_id", "team_id", "player_id", "minutes_actual", "y_ge5", "y_ge15", "p_ge5", "p_ge15"]
    if "player_name" in test_df.columns:
        pred_cols.insert(3, "player_name")
    if "game_date" in test_df.columns:
        pred_cols.insert(1, "game_date")
    test_df[pred_cols].to_parquet(run_dir / "predictions_test.parquet", index=False)

    # Save models
    typer.echo("[save] Saving models")
    model_ge5.save_model(str(run_dir / "model_ge5.lgb"))
    model_ge15.save_model(str(run_dir / "model_ge15.lgb"))

    # Save metadata
    meta = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "source_artifact": str(rotation_set_artifact),
        "dataset_dir": dataset_dir,
        "feature_columns": feature_columns,
        "n_features": len(feature_columns),
        "random_state": random_state,
        "weight_config": {
            "base": WEIGHT_BASE,
            "mult_le5": WEIGHT_MULT_LE_5,
            "mult_le1": WEIGHT_MULT_LE_1,
            "mult_chaos_le5": WEIGHT_MULT_CHAOS_LE_5,
        },
        "split_info": split_info,
        "metrics": {
            "ge5": metrics_ge5,
            "ge15": metrics_ge15,
        },
    }
    with open(run_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    typer.echo(f"[done] Artifacts written to: {run_dir}")
    typer.echo(f"  - model_ge5.lgb")
    typer.echo(f"  - model_ge15.lgb")
    typer.echo(f"  - predictions_test.parquet")
    typer.echo(f"  - report.md")
    typer.echo(f"  - meta.json")
    typer.echo("")
    typer.echo(f"Brier ge5: {metrics_ge5['brier']:.4f}")
    typer.echo(f"Brier ge15: {metrics_ge15['brier']:.4f}")


if __name__ == "__main__":
    app()
