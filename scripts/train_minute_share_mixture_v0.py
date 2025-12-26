#!/usr/bin/env python
"""Train the minute share mixture model v0.

Two-stage model:
1. State classifier (multiclass LightGBM) predicts p(state | features)
2. Per-state regressors predict E[minutes | state, features]

Combined expected minutes:
    E[minutes] = Σ_k p_state[k] * μ_k

Example usage:
    uv run python scripts/train_minute_share_mixture_v0.py \
        --start-date 2024-01-01 \
        --end-date 2024-06-30 \
        --out-bundle artifacts/minute_share_mixture/v0 \
        --val-days 14 \
        --seed 42
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import typer
from sklearn.metrics import confusion_matrix, mean_absolute_error

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from projections.minutes_v1.datasets import (
    KEY_COLUMNS,
    ensure_columns,
    load_feature_frame,
)
from projections.models.feature_contract import assert_no_leakage
from projections.models.minute_share_mixture import (
    NUM_STATES,
    STATE_NAMES,
    MixtureBundle,
    get_state_counts,
    minutes_to_state,
    predict_expected_minutes,
    train_mixture_model,
)
from projections.models.minutes_features import infer_feature_columns

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
LOGGER = logging.getLogger(__name__)

app = typer.Typer(help=__doc__)

UTC = timezone.utc


def _print_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Print confusion matrix with state labels."""
    cm = confusion_matrix(y_true, y_pred, labels=range(NUM_STATES))
    
    typer.echo("\n📊 Confusion Matrix (rows=actual, cols=predicted):")
    typer.echo("-" * 60)
    
    # Header
    header = "        " + "  ".join(f"S{i:d}".rjust(6) for i in range(NUM_STATES))
    typer.echo(header)
    
    # Rows
    for i in range(NUM_STATES):
        row_label = f"S{i} ({STATE_NAMES[i][:8]})".ljust(12)
        row_vals = "  ".join(f"{cm[i, j]:6d}" for j in range(NUM_STATES))
        typer.echo(f"{row_label} {row_vals}")
    
    # Per-class accuracy
    typer.echo("\nPer-state accuracy:")
    total_correct = 0
    total_samples = 0
    for i in range(NUM_STATES):
        n_actual = cm[i, :].sum()
        n_correct = cm[i, i]
        acc = n_correct / n_actual if n_actual > 0 else 0.0
        total_correct += n_correct
        total_samples += n_actual
        typer.echo(f"  S{i} ({STATE_NAMES[i]}): {acc:.1%} ({n_correct}/{n_actual})")
    
    overall_acc = total_correct / total_samples if total_samples > 0 else 0.0
    typer.echo(f"\nOverall classification accuracy: {overall_acc:.1%}")
    
    # Key states (S2 and S3) accuracy
    s2_acc = cm[2, 2] / cm[2, :].sum() if cm[2, :].sum() > 0 else 0.0
    s3_acc = cm[3, 3] / cm[3, :].sum() if cm[3, :].sum() > 0 else 0.0
    typer.echo(f"\n🎯 Key buckets: S2 (10-20) = {s2_acc:.1%}, S3 (20-30) = {s3_acc:.1%}")


def _compute_bucket_mae(
    actual_minutes: np.ndarray,
    predicted_minutes: np.ndarray,
) -> dict[str, float | None]:
    """Compute MAE by minute bucket."""
    buckets = {
        "0-10": (0, 10),
        "10-20": (10, 20),
        "20-30": (20, 30),
        "30+": (30, 100),
    }
    
    result = {}
    for name, (lo, hi) in buckets.items():
        mask = (actual_minutes >= lo) & (actual_minutes < hi)
        if name == "30+":
            mask = actual_minutes >= 30
        n = mask.sum()
        if n > 0:
            mae = mean_absolute_error(actual_minutes[mask], predicted_minutes[mask])
            result[name] = round(float(mae), 3)
        else:
            result[name] = None
    
    return result


# Default training dataset path
DEFAULT_TRAINING_DATASET = Path("data/training/datasets/full_contract_v2/features.parquet")


@app.command()
def main(
    start_date: datetime = typer.Option(
        ...,
        "--start-date",
        help="Training data start date (UTC).",
    ),
    end_date: datetime = typer.Option(
        ...,
        "--end-date",
        help="Training data end date (UTC).",
    ),
    out_bundle: Path = typer.Option(
        Path("artifacts/minute_share_mixture/v0"),
        "--out-bundle",
        help="Output bundle directory.",
    ),
    val_days: int = typer.Option(
        14,
        "--val-days",
        help="Number of days at end of window for validation.",
    ),
    features: Path | None = typer.Option(
        None,
        "--features",
        help="Path to features parquet file or directory. If not provided, uses canonical training dataset.",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Random seed.",
    ),
    target_col: str = typer.Option(
        "minutes",
        "--target",
        help="Target column name.",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Overwrite existing bundle if it exists.",
    ),
) -> None:
    """Train minute share mixture model."""

    # Check if output already exists
    if out_bundle.exists() and not overwrite:
        raise typer.BadParameter(
            f"Output bundle already exists: {out_bundle}\n"
            "Use --overwrite to replace it."
        )

    # Normalize dates
    start_date = start_date.replace(tzinfo=UTC)
    end_date = end_date.replace(tzinfo=UTC)
    
    # Compute validation split
    val_start = end_date - timedelta(days=val_days - 1)
    train_end = val_start - timedelta(days=1)
    
    typer.echo(f"[train] Training: {start_date.date()} to {train_end.date()}")
    typer.echo(f"[train] Validation: {val_start.date()} to {end_date.date()}")

    # Resolve features path
    if features is None:
        if DEFAULT_TRAINING_DATASET.exists():
            features = DEFAULT_TRAINING_DATASET
            typer.echo(f"[train] Using default training dataset: {features}")
        else:
            raise typer.BadParameter(
                f"No --features provided and default dataset not found: {DEFAULT_TRAINING_DATASET}"
            )

    # Load features
    typer.echo(f"[train] Loading features from {features}...")
    if features.is_dir():
        # Load all parquet files in directory
        parquet_files = sorted(features.rglob("*.parquet"))
        if not parquet_files:
            raise typer.BadParameter(f"No parquet files found in {features}")
        dfs = [pd.read_parquet(f) for f in parquet_files]
        feature_df = pd.concat(dfs, ignore_index=True)
    else:
        feature_df = pd.read_parquet(features)
    
    ensure_columns(
        feature_df,
        [target_col, "game_date", "team_id", *KEY_COLUMNS],
    )
    
    # Infer feature columns
    feature_columns = infer_feature_columns(
        feature_df,
        target_col=target_col,
        excluded={"prior_play_prob", "play_prob", "play_probability", "p_play"},
    )
    
    # Leakage guard
    assert_no_leakage(feature_columns)
    typer.echo(f"[train] ✓ Leakage guard passed: {len(feature_columns)} features")
    
    # Parse dates
    if "game_date" in feature_df.columns:
        feature_df["_game_date"] = pd.to_datetime(feature_df["game_date"], utc=True, errors="coerce")
    elif "tip_ts" in feature_df.columns:
        feature_df["_game_date"] = pd.to_datetime(feature_df["tip_ts"], utc=True, errors="coerce")
    else:
        raise ValueError("Need game_date or tip_ts column")
    
    # Split
    train_mask = (feature_df["_game_date"] >= start_date) & (feature_df["_game_date"] <= train_end)
    val_mask = (feature_df["_game_date"] >= val_start) & (feature_df["_game_date"] <= end_date)
    
    train_df = feature_df[train_mask].copy()
    val_df = feature_df[val_mask].copy()
    
    if train_df.empty:
        raise ValueError(f"No training data in range {start_date} - {train_end}")
    if val_df.empty:
        raise ValueError(f"No validation data in range {val_start} - {end_date}")
    
    typer.echo(f"[train] Training: {len(train_df):,} rows")
    typer.echo(f"[train] Validation: {len(val_df):,} rows")
    
    # Show state distribution
    train_counts = get_state_counts(train_df[target_col].values)
    typer.echo("\n[train] State distribution:")
    for s in range(NUM_STATES):
        pct = 100 * train_counts[s] / len(train_df)
        typer.echo(f"  S{s} ({STATE_NAMES[s]}): {train_counts[s]:,} ({pct:.1f}%)")
    
    # Train model
    typer.echo("\n[train] Training mixture model...")
    X_train = train_df[feature_columns]
    y_train = train_df[target_col].values
    
    bundle = train_mixture_model(
        X_train,
        y_train,
        random_state=seed,
    )
    
    # Update training metadata
    bundle.train_meta.update({
        "train_start": start_date.isoformat(),
        "train_end": train_end.isoformat(),
        "val_start": val_start.isoformat(),
        "val_end": end_date.isoformat(),
        "val_days": val_days,
    })
    
    # Evaluate on validation set
    typer.echo("\n[eval] Evaluating on validation set...")
    X_val = val_df[feature_columns]
    y_val = val_df[target_col].values
    
    # Get predictions
    pred_minutes = predict_expected_minutes(X_val, bundle)
    
    # Classification evaluation
    y_states_true = minutes_to_state(y_val)
    y_states_pred = minutes_to_state(pred_minutes)
    _print_confusion_matrix(y_states_true, y_states_pred)
    
    # Regression evaluation
    mae_overall = mean_absolute_error(y_val, pred_minutes)
    typer.echo(f"\n📈 Overall MAE: {mae_overall:.3f} minutes")
    
    bucket_mae = _compute_bucket_mae(y_val, pred_minutes)
    typer.echo("\nMAE by bucket:")
    for bucket, mae in bucket_mae.items():
        mae_str = f"{mae:.3f}" if mae is not None else "N/A"
        typer.echo(f"  {bucket}: {mae_str}")
    
    # Save bundle
    bundle.save(out_bundle)
    
    # Save evaluation metrics
    eval_metrics = {
        "mae_overall": round(mae_overall, 4),
        "bucket_mae": bucket_mae,
        "n_train": len(train_df),
        "n_val": len(val_df),
    }
    (out_bundle / "eval_metrics.json").write_text(json.dumps(eval_metrics, indent=2))
    
    typer.echo(f"\n✓ Bundle saved to {out_bundle}")


if __name__ == "__main__":
    app()
