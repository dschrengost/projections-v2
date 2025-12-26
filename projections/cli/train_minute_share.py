"""CLI for training minute share LightGBM model.

Example usage:
    uv run python -m projections.cli.train_minute_share \
        --run-id minute_share_v1 \
        --train-start 2024-10-01 \
        --train-end 2025-01-31 \
        --val-start 2025-02-01 \
        --val-end 2025-02-28
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import typer

from projections import paths
from projections.minutes_v1.artifacts import write_json
from projections.minutes_v1.datasets import (
    KEY_COLUMNS,
    ensure_columns,
    load_feature_frame,
)
from projections.minutes_v1.minute_share import (
    TEAM_TOTAL_MINUTES,
    evaluate_minute_share_model,
    predict_minutes,
    save_artifacts,
    train_minute_share_model,
)
from projections.models.feature_contract import (
    assert_no_leakage,
    save_feature_contract,
)
from projections.models.minutes_features import infer_feature_columns
from projections.utils import ensure_directory

# Configure logging for evaluation output
logging.basicConfig(level=logging.INFO, format="%(message)s")

app = typer.Typer(help="Train minute share prediction model")

UTC = timezone.utc


def _raise_empty_split_error(
    split_name: str,
    df: pd.DataFrame,
    start: datetime,
    end: datetime,
    date_col: str,
) -> None:
    """Raise detailed ValueError when a train/val split is empty.
    
    Provides debug information to diagnose datetime mismatches.
    """
    date_series = df[date_col]
    
    # Gather diagnostics
    n_rows = len(df)
    n_non_null = date_series.notna().sum()
    dtype = str(date_series.dtype)
    
    # Get sample tzinfo if datetime
    tzinfo_sample = None
    if n_non_null > 0:
        first_valid = date_series.dropna().iloc[0]
        if hasattr(first_valid, 'tzinfo'):
            tzinfo_sample = str(first_valid.tzinfo)
        min_date = date_series.min()
        max_date = date_series.max()
    else:
        min_date = max_date = None
    
    error_msg = f"""No {split_name} rows after date filter.

DEBUG INFO:
-----------
Filter range: {start} to {end}
  start type: {type(start).__name__}, tzinfo={start.tzinfo}
  end type: {type(end).__name__}, tzinfo={end.tzinfo}

Data column '{date_col}':
  total rows: {n_rows:,}
  non-null: {n_non_null:,}
  dtype: {dtype}
  tzinfo sample: {tzinfo_sample}
  min value: {min_date}
  max value: {max_date}

LIKELY CAUSES:
1. Validation period falls in NBA off-season/playoffs (no games in features)
2. Datetime timezone mismatch (tz-aware vs tz-naive)
3. Date vs datetime comparison issue
"""
    raise ValueError(error_msg)


def _filter_out_players(df: pd.DataFrame, target_col: str = "minutes") -> pd.DataFrame:
    """Filter to players who actually played (minutes > 0)."""
    return df[df[target_col] > 0].copy()


@app.command()
def main(
    run_id: str = typer.Option(..., help="Unique identifier for the training run."),
    train_start: datetime = typer.Option(
        ...,
        "--train-start",
        help="Training window start (UTC, inclusive).",
    ),
    train_end: datetime = typer.Option(
        ...,
        "--train-end",
        help="Training window end (UTC, inclusive).",
    ),
    val_start: datetime = typer.Option(
        None,
        "--val-start",
        help="Validation window start (UTC, inclusive). Defaults to day after --train-end.",
    ),
    val_end: datetime = typer.Option(
        ...,
        "--val-end",
        help="Validation window end (UTC, inclusive).",
    ),
    data_root: Path = typer.Option(
        paths.get_data_root(),
        help="Root directory containing data/* (defaults to PROJECTIONS_DATA_ROOT or ./data).",
    ),
    season: int | None = typer.Option(None, help="Season year for default features path (e.g., 2024)."),
    month: int | None = typer.Option(None, help="Month partition (1-12) for default path."),
    features: Path | None = typer.Option(None, help="Explicit feature parquet path."),
    artifact_root: Path = typer.Option(
        Path("artifacts/minute_share"),
        help="Where to store run artifacts.",
    ),
    target_col: str = typer.Option("minutes", help="Target column containing raw minutes."),
    random_state: int = typer.Option(42, help="Random seed for LightGBM."),
    lgbm_n_estimators: int | None = typer.Option(None, help="Override LightGBM n_estimators."),
    lgbm_learning_rate: float | None = typer.Option(None, help="Override LightGBM learning_rate."),
    lgbm_num_leaves: int | None = typer.Option(None, help="Override LightGBM num_leaves."),
    lgbm_max_depth: int | None = typer.Option(None, help="Override LightGBM max_depth."),
) -> None:
    """Train a minute share prediction model."""

    # Derive val_start if not provided
    if val_start is None:
        val_start = train_end + timedelta(days=1)

    # Normalize timestamps
    train_start = train_start.replace(tzinfo=UTC)
    train_end = train_end.replace(tzinfo=UTC)
    val_start = val_start.replace(tzinfo=UTC)
    val_end = val_end.replace(tzinfo=UTC)

    typer.echo(f"[train] Loading features from {data_root}")

    # Load features
    feature_df = load_feature_frame(
        features_path=features,
        data_root=data_root,
        season=season,
        month=month,
    )
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

    # Leakage guard: ensure no forbidden features made it through
    assert_no_leakage(feature_columns)
    typer.echo(f"[train] ✓ Leakage guard passed: {len(feature_columns)} features")

    # Parse game_date for splitting
    if "game_date" in feature_df.columns:
        feature_df["_game_date"] = pd.to_datetime(feature_df["game_date"], utc=True, errors="coerce")
    elif "tip_ts" in feature_df.columns:
        feature_df["_game_date"] = pd.to_datetime(feature_df["tip_ts"], utc=True, errors="coerce")
    else:
        raise ValueError("Need game_date or tip_ts column for date-based splits")

    train_mask = (feature_df["_game_date"] >= train_start) & (feature_df["_game_date"] <= train_end)
    val_mask = (feature_df["_game_date"] >= val_start) & (feature_df["_game_date"] <= val_end)

    train_df = _filter_out_players(feature_df[train_mask], target_col)
    val_df = _filter_out_players(feature_df[val_mask], target_col)

    if train_df.empty:
        _raise_empty_split_error("training", feature_df, train_start, train_end, "_game_date")
    if val_df.empty:
        _raise_empty_split_error("validation", feature_df, val_start, val_end, "_game_date")

    typer.echo(
        f"[train] Training on {len(train_df):,} rows, validating on {len(val_df):,} rows "
        f"with {len(feature_columns)} features"
    )

    # Build LightGBM params
    lgbm_params: dict[str, Any] = {}
    if lgbm_n_estimators is not None:
        lgbm_params["n_estimators"] = lgbm_n_estimators
    if lgbm_learning_rate is not None:
        lgbm_params["learning_rate"] = lgbm_learning_rate
    if lgbm_num_leaves is not None:
        lgbm_params["num_leaves"] = lgbm_num_leaves
    if lgbm_max_depth is not None:
        lgbm_params["max_depth"] = lgbm_max_depth

    # Train model
    X_train = train_df[feature_columns]
    y_train = train_df[target_col]
    
    # Extract keys for dynamic normalization
    train_game_ids = train_df["game_id"]
    train_team_ids = train_df["team_id"]

    typer.echo("[train] Training minute share model...")
    artifacts = train_minute_share_model(
        X_train,
        y_train,
        game_ids=train_game_ids,
        team_ids=train_team_ids,
        random_state=random_state,
        params=lgbm_params or None,
    )

    # Evaluate on validation set
    typer.echo("[eval] Evaluating on validation set...")
    X_val = val_df[feature_columns]
    y_val = val_df[target_col]
    
    val_game_ids = val_df["game_id"]
    val_team_ids = val_df["team_id"]

    predictions = predict_minutes(artifacts, X_val, team_ids=val_team_ids, game_ids=val_game_ids)
    
    # Add keys back for validation
    predictions["game_id"] = val_game_ids.values
    predictions["team_id"] = val_team_ids.values

    # Comprehensive evaluation
    evaluation = evaluate_minute_share_model(
        actual_minutes=y_val,
        predicted_shares=predictions["raw_share"],
        predicted_minutes=predictions["predicted_minutes"],
        game_ids=val_game_ids,
        team_ids=val_team_ids,
        verbose=True,
    )

    # Convert evaluation to metrics dict for JSON serialization
    metrics = evaluation.to_dict()

    # Check team sums (Grouping by Game/Team)
    team_minute_sums = predictions.groupby(["game_id", "team_id"])["predicted_minutes"].sum()
    typer.echo(f"[eval] Team minute sums: min={team_minute_sums.min():.1f}, max={team_minute_sums.max():.1f}")
    assert np.allclose(team_minute_sums.to_numpy(), TEAM_TOTAL_MINUTES, atol=0.1), \
        "Team minutes should sum to 240"

    # Save artifacts
    run_dir = ensure_directory(artifact_root / run_id)
    save_artifacts(artifacts, run_dir / "minute_share_model.joblib")

    # Save metadata
    meta = {
        "model": "minute_share_lgbm",
        "run_id": run_id,
        "feature_columns": feature_columns,
        "params": artifacts.params,
        "windows": {
            "train": {
                "start": train_start.isoformat(),
                "end": train_end.isoformat(),
            },
            "val": {
                "start": val_start.isoformat(),
                "end": val_end.isoformat(),
            },
        },
        "train_rows": len(train_df),
        "val_rows": len(val_df),
    }
    write_json(run_dir / "meta.json", meta)
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "feature_columns.json", {"columns": feature_columns})

    # Save feature contract for parity checking
    save_feature_contract(
        feature_columns,
        run_dir / "feature_contract.json",
        metadata={
            "run_id": run_id,
            "train_rows": len(train_df),
            "val_rows": len(val_df),
        },
    )
    typer.echo(f"[train] ✓ Feature contract saved with {len(feature_columns)} features")

    typer.echo(f"[done] Artifacts saved to {run_dir}")


if __name__ == "__main__":
    app()
