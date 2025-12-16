"""
Train LightGBM models for usage shares prediction.

Trains 3 independent regression models predicting log-weights:
    y_x = log(x + alpha) where x in {fga, fta, tov}

Usage:
    uv run python -m scripts.usage_shares_v1.train_lgbm \\
        --data-root /home/daniel/projections-data \\
        --targets fga \\
        --start-date 2024-11-01 \\
        --end-date 2025-02-01
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import typer

from projections.paths import data_path
from projections.usage_shares_v1.features import (
    CATEGORICAL_COLS,
    FEATURE_COLS,
    NUMERIC_COLS,
    add_derived_features,
)
from projections.usage_shares_v1.metrics import (
    TargetMetrics,
    check_odds_leakage,
    compute_metrics,
)

app = typer.Typer(add_completion=False, help=__doc__)


# =============================================================================
# Data Loading
# =============================================================================


def load_training_data(
    data_root: Path,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """Load usage_shares_training_base partitions for the date range."""
    root = data_root / "gold" / "usage_shares_training_base"
    frames: list[pd.DataFrame] = []
    
    for season_dir in root.glob("season=*"):
        for day_dir in season_dir.glob("game_date=*"):
            try:
                day_str = day_dir.name.split("=", 1)[1]
                day = pd.Timestamp(day_str).normalize()
            except (ValueError, IndexError):
                continue
            if day < start_date or day > end_date:
                continue
            path = day_dir / "usage_shares_training_base.parquet"
            if path.exists():
                frames.append(pd.read_parquet(path))
    
    if not frames:
        raise FileNotFoundError(
            f"No usage_shares_training_base partitions found for {start_date.date()}..{end_date.date()}"
        )
    
    df = pd.concat(frames, ignore_index=True)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.normalize()
    return df


def get_git_sha() -> str | None:
    """Get current git SHA if available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]
    except Exception:
        pass
    return None


# =============================================================================
# Training
# =============================================================================


@dataclass
class TrainingResult:
    """Result from training a single target model."""
    target: str
    model: lgb.LGBMRegressor
    train_metrics: TargetMetrics
    val_metrics: TargetMetrics
    feature_cols: list[str]


def train_target_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target: str,
    feature_cols: list[str],
    alpha: float,
    seed: int,
    num_threads: int,
) -> TrainingResult:
    """Train a single LightGBM model for one target."""
    
    # Filter to valid rows for this target
    # NaN validity is treated as "valid" if share value is finite (handles newer data without explicit flags)
    share_col = f"share_{target}"
    valid_col = f"share_{target}_valid"
    
    def get_valid_mask(df: pd.DataFrame) -> pd.Series:
        if valid_col in df.columns:
            explicit_valid = df[valid_col].fillna(True)  # NaN = assume valid
        else:
            explicit_valid = pd.Series(True, index=df.index)
        share_finite = df[share_col].notna() & np.isfinite(df[share_col])
        return explicit_valid & share_finite
    
    train_mask = get_valid_mask(train_df)
    train_df = train_df[train_mask].copy()
    val_mask = get_valid_mask(val_df)
    val_df = val_df[val_mask].copy()
    
    # Also filter to finite target values
    train_df = train_df[np.isfinite(train_df[target])].copy()
    val_df = val_df[np.isfinite(val_df[target])].copy()
    
    # Compute log-weight labels
    y_train = np.log(train_df[target].values + alpha)
    
    # Prepare features
    X_train = train_df[feature_cols].copy()
    X_val = val_df[feature_cols].copy()
    
    # Fill missing numeric features with 0
    for col in feature_cols:
        if col not in CATEGORICAL_COLS:
            X_train[col] = pd.to_numeric(X_train[col], errors="coerce").fillna(0.0)
            X_val[col] = pd.to_numeric(X_val[col], errors="coerce").fillna(0.0)
        else:
            # Categorical: fill with -1 for unknown
            X_train[col] = X_train[col].fillna(-1).astype(int)
            X_val[col] = X_val[col].fillna(-1).astype(int)
    
    # Train model
    params = {
        "n_estimators": 400,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 50,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 0.1,
        "random_state": seed,
        "n_jobs": num_threads,
        "verbose": -1,
    }
    
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    
    # Predict
    pred_train = model.predict(X_train)
    pred_val = model.predict(X_val)
    
    # Compute metrics using shared function
    train_metrics = compute_metrics(train_df, pred_train, target, alpha)
    val_metrics = compute_metrics(val_df, pred_val, target, alpha)
    
    return TrainingResult(
        target=target,
        model=model,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        feature_cols=feature_cols,
    )


# =============================================================================
# Main Command
# =============================================================================


@app.command()
def main(
    data_root: Path = typer.Option(
        None,
        help="Root containing gold/usage_shares_training_base (defaults to PROJECTIONS_DATA_ROOT).",
    ),
    run_id: str = typer.Option(
        None,
        help="Run ID for artifacts (default: timestamp).",
    ),
    start_date: str = typer.Option(
        ...,
        help="Start date (YYYY-MM-DD) inclusive.",
    ),
    end_date: str = typer.Option(
        ...,
        help="End date (YYYY-MM-DD) inclusive.",
    ),
    targets: str = typer.Option(
        "fga,fta,tov",
        help="Comma-separated list of targets to train.",
    ),
    alpha: float = typer.Option(
        0.5,
        help="Alpha for log transformation: y = log(x + alpha).",
    ),
    min_minutes_actual: float = typer.Option(
        4.0,
        help="Minimum minutes_actual to include in training.",
    ),
    val_days: int = typer.Option(
        30,
        help="Number of trailing days to use for validation.",
    ),
    seed: int = typer.Option(
        1337,
        help="Random seed for reproducibility.",
    ),
    num_threads: int = typer.Option(
        4,
        help="Number of threads for LightGBM.",
    ),
) -> None:
    """Train LightGBM models for usage shares prediction."""
    
    root = data_root or data_path()
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    target_list = [t.strip() for t in targets.split(",")]
    run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    
    typer.echo(f"[lgbm] Loading data from {start.date()} to {end.date()}...")
    df = load_training_data(root, start, end)
    typer.echo(f"[lgbm] Loaded {len(df):,} rows")
    
    # Add derived features
    df = add_derived_features(df)
    
    # Leakage check
    n_leaky, n_checked, missing_frac = check_odds_leakage(df)
    typer.echo(f"[lgbm] Odds leakage check: {n_leaky}/{n_checked} rows have odds_as_of_ts > tip_ts")
    typer.echo(f"[lgbm] Odds timestamp missing for {missing_frac:.1%} of rows")
    if n_leaky > 0:
        typer.echo("[lgbm] ERROR: Detected leaky odds! Aborting.")
        raise typer.Exit(1)
    
    # Filter by min_minutes_actual if column exists
    if "minutes_actual" in df.columns:
        df = df[df["minutes_actual"] >= min_minutes_actual].copy()
        typer.echo(f"[lgbm] Filtered to {len(df):,} rows with minutes_actual >= {min_minutes_actual}")
    
    # Determine feature columns (only use those that exist in data)
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    missing_features = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_features:
        typer.echo(f"[lgbm] Warning: missing features (will fill with 0): {missing_features}")
        for col in missing_features:
            df[col] = 0.0 if col in NUMERIC_COLS else -1
    
    feature_cols = available_features + [c for c in missing_features if c in FEATURE_COLS]
    typer.echo(f"[lgbm] Using {len(feature_cols)} features")
    
    # Date-based train/val split: last N days = val
    unique_dates = sorted(df["game_date"].unique())
    if len(unique_dates) <= val_days:
        raise ValueError(f"Not enough dates for val split: {len(unique_dates)} <= {val_days}")
    
    val_start_date = unique_dates[-val_days]
    train_df = df[df["game_date"] < val_start_date].copy()
    val_df = df[df["game_date"] >= val_start_date].copy()
    
    val_dates = sorted(val_df["game_date"].unique())
    typer.echo(f"[lgbm] Train: {len(train_df):,} rows, Val: {len(val_df):,} rows")
    typer.echo(f"[lgbm] Val dates: {val_dates[0].date()} to {val_dates[-1].date()} ({len(val_dates)} days)")
    
    # Train models
    results: dict[str, TrainingResult] = {}
    all_metrics: dict[str, dict[str, Any]] = {}
    
    for target in target_list:
        if target not in df.columns:
            typer.echo(f"[lgbm] Skipping target {target}: column not found")
            continue
        
        typer.echo(f"[lgbm] Training {target}...")
        result = train_target_model(
            train_df, val_df, target, feature_cols, alpha, seed, num_threads
        )
        results[target] = result
        
        # Log metrics
        val_m = result.val_metrics
        typer.echo(
            f"[lgbm] {target} val: MAE={val_m.share_MAE:.4f} "
            f"(baseline={val_m.share_MAE_baseline:.4f}) "
            f"KL={val_m.KL:.4f} top1={val_m.top1_acc:.2%}"
        )
        
        if val_m.share_MAE >= val_m.share_MAE_baseline:
            typer.echo(f"[lgbm] WARNING: {target} model does NOT beat baseline!")
        
        all_metrics[target] = {
            "train": result.train_metrics.to_dict(),
            "val": result.val_metrics.to_dict(),
        }
    
    # Save artifacts
    artifacts_dir = root / "artifacts" / "usage_shares_v1" / "runs" / run_id
    lgbm_dir = artifacts_dir / "lgbm"
    lgbm_dir.mkdir(parents=True, exist_ok=True)
    
    # Save models
    for target, result in results.items():
        model_path = lgbm_dir / f"model_{target}.txt"
        result.model.booster_.save_model(str(model_path))
        typer.echo(f"[lgbm] Saved model to {model_path}")
    
    # Save params
    params_path = lgbm_dir / "params.json"
    params_dict = {
        "n_estimators": 400,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 50,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 0.1,
    }
    params_path.write_text(json.dumps(params_dict, indent=2))
    
    # Save feature columns
    feature_cols_path = artifacts_dir / "feature_columns.json"
    feature_cols_payload = {
        "feature_cols": feature_cols,
        "numeric_cols": [c for c in feature_cols if c in NUMERIC_COLS],
        "categorical_cols": [c for c in feature_cols if c in CATEGORICAL_COLS],
    }
    feature_cols_path.write_text(json.dumps(feature_cols_payload, indent=2))
    
    # Save metrics
    metrics_path = artifacts_dir / "metrics.json"
    metrics_path.write_text(json.dumps(all_metrics, indent=2))
    
    # Save meta
    meta = {
        "run_id": run_id,
        "backend": "lgbm",
        "git_sha": get_git_sha(),
        "date_range": [start.date().isoformat(), end.date().isoformat()],
        "val_split": {
            "method": "tail_days",
            "n_days": val_days,
            "val_dates": [d.date().isoformat() for d in val_dates],
        },
        "alpha": alpha,
        "min_minutes_actual": min_minutes_actual,
        "seed": seed,
        "targets": list(results.keys()),
        "n_train_rows": len(train_df),
        "n_val_rows": len(val_df),
        "leakage_check": {
            "n_leaky": n_leaky,
            "n_checked": n_checked,
            "missing_frac": round(missing_frac, 4),
        },
        "created_at": datetime.now().isoformat(),
    }
    meta_path = artifacts_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    
    typer.echo(f"[lgbm] Artifacts saved to {artifacts_dir}")
    
    # Summary
    typer.echo("\n=== SUMMARY ===")
    for target, result in results.items():
        val_m = result.val_metrics
        status = "✓" if val_m.share_MAE < val_m.share_MAE_baseline else "✗"
        improvement = (1 - val_m.share_MAE / val_m.share_MAE_baseline) * 100
        typer.echo(
            f"{target}: {status} MAE={val_m.share_MAE:.4f} "
            f"(baseline={val_m.share_MAE_baseline:.4f}, {improvement:+.1f}%) "
            f"KL={val_m.KL:.4f} top1={val_m.top1_acc:.2%}"
        )


if __name__ == "__main__":
    app()
