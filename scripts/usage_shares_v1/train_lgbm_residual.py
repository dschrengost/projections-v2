"""
Train LightGBM RESIDUAL models for usage shares prediction.

Instead of predicting raw log-weights, this trains to predict the DELTA from baseline:
  delta = y_true_logw - y_baseline_logw

At inference: logw_pred = y_baseline_logw + shrink * delta_pred

This leverages the strong rate-weighted baseline while learning corrections.

Usage:
    uv run python -m scripts.usage_shares_v1.train_lgbm_residual \
        --data-root /home/daniel/projections-data \
        --targets fga,tov \
        --start-date 2024-10-22 \
        --end-date 2025-11-28 \
        --shrink 0.5 \
        --seed 1337
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
    add_derived_features,
)
from projections.usage_shares_v1.metrics import (
    TargetMetrics,
    check_odds_leakage,
    compute_baseline_log_weights,
    compute_metrics,
)

app = typer.Typer(add_completion=False, help=__doc__)


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


@dataclass
class ResidualTrainingResult:
    """Result from residual training."""
    target: str
    model: lgb.LGBMRegressor
    shrink: float
    train_metrics: TargetMetrics
    val_metrics: TargetMetrics
    feature_cols: list[str]


def train_residual_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target: str,
    feature_cols: list[str],
    alpha: float,
    shrink: float,
    seed: int,
    num_threads: int,
) -> ResidualTrainingResult:
    """Train a residual LightGBM model for one target."""
    
    # Filter to valid rows
    share_col = f"share_{target}"
    valid_col = f"share_{target}_valid"
    
    def get_valid_mask(df: pd.DataFrame) -> pd.Series:
        if valid_col in df.columns:
            explicit_valid = df[valid_col].fillna(True)
        else:
            explicit_valid = pd.Series(True, index=df.index)
        share_finite = df[share_col].notna() & np.isfinite(df[share_col])
        return explicit_valid & share_finite
    
    train_mask = get_valid_mask(train_df)
    train_df = train_df[train_mask].copy()
    val_mask = get_valid_mask(val_df)
    val_df = val_df[val_mask].copy()
    
    train_df = train_df[np.isfinite(train_df[target])].copy()
    val_df = val_df[np.isfinite(val_df[target])].copy()
    
    # Compute TRUE log-weights
    y_train_true = np.log(train_df[target].values + alpha)
    
    # Compute BASELINE log-weights
    y_train_baseline = compute_baseline_log_weights(train_df, target, alpha)
    y_val_baseline = compute_baseline_log_weights(val_df, target, alpha)
    
    # RESIDUAL = true - baseline
    y_train_residual = y_train_true - y_train_baseline
    
    # Prepare features
    X_train = train_df[feature_cols].copy()
    X_val = val_df[feature_cols].copy()
    
    for col in feature_cols:
        if col not in CATEGORICAL_COLS:
            X_train[col] = pd.to_numeric(X_train[col], errors="coerce").fillna(0.0)
            X_val[col] = pd.to_numeric(X_val[col], errors="coerce").fillna(0.0)
        else:
            X_train[col] = X_train[col].fillna(-1).astype(int)
            X_val[col] = X_val[col].fillna(-1).astype(int)
    
    # Train model to predict RESIDUAL
    params = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 50,
        "max_depth": 5,  # Shallower for residual learning
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 0.5,  # More regularization
        "reg_alpha": 0.1,
        "random_state": seed,
        "n_jobs": num_threads,
        "verbose": -1,
    }
    
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train_residual)
    
    # Predict on train and val
    delta_pred_train = model.predict(X_train)
    delta_pred_val = model.predict(X_val)
    
    # Final log-weights = baseline + shrink * delta
    logw_train = y_train_baseline + shrink * delta_pred_train
    logw_val = y_val_baseline + shrink * delta_pred_val
    
    # Compute metrics
    train_metrics = compute_metrics(train_df, logw_train, target, alpha)
    val_metrics = compute_metrics(val_df, logw_val, target, alpha)
    
    return ResidualTrainingResult(
        target=target,
        model=model,
        shrink=shrink,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        feature_cols=feature_cols,
    )


@app.command()
def main(
    data_root: Path = typer.Option(None),
    run_id: str = typer.Option(None),
    start_date: str = typer.Option(...),
    end_date: str = typer.Option(...),
    targets: str = typer.Option("fga,tov"),
    alpha: float = typer.Option(0.5),
    shrink: str = typer.Option("0.5"),  # Can be comma-separated for grid search
    min_minutes_actual: float = typer.Option(4.0),
    val_days: int = typer.Option(30),
    seed: int = typer.Option(1337),
    num_threads: int = typer.Option(-1),
) -> None:
    """Train LightGBM residual-on-baseline models."""
    
    np.random.seed(seed)
    root = data_root or data_path()
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    target_list = [t.strip() for t in targets.split(",")]
    shrink_list = [float(s.strip()) for s in shrink.split(",")]
    run_id = run_id or f"residual_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    typer.echo(f"[lgbm-residual] Loading data from {start.date()} to {end.date()}...")
    df = load_training_data(root, start, end)
    typer.echo(f"[lgbm-residual] Loaded {len(df):,} rows")
    
    df = add_derived_features(df)
    
    # Leakage check
    n_leaky, n_checked, missing_frac = check_odds_leakage(df)
    typer.echo(f"[lgbm-residual] Odds leakage check: {n_leaky}/{n_checked} rows have odds_as_of_ts > tip_ts")
    if n_leaky > 0:
        typer.echo("[lgbm-residual] ERROR: Detected leaky odds! Aborting.")
        raise typer.Exit(1)
    
    # Filter by minutes
    if "minutes_actual" in df.columns:
        df = df[df["minutes_actual"] >= min_minutes_actual].copy()
        typer.echo(f"[lgbm-residual] Filtered to {len(df):,} rows with minutes_actual >= {min_minutes_actual}")
    
    # Get available features
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    missing_features = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_features:
        typer.echo(f"[lgbm-residual] Warning: missing features: {missing_features}")
    typer.echo(f"[lgbm-residual] Using {len(available_features)} features")
    
    # Date-based split
    unique_dates = sorted(df["game_date"].unique())
    if len(unique_dates) <= val_days:
        raise ValueError(f"Not enough dates: {len(unique_dates)} <= {val_days}")
    
    val_start_date = unique_dates[-val_days]
    train_df = df[df["game_date"] < val_start_date].copy()
    val_df = df[df["game_date"] >= val_start_date].copy()
    
    typer.echo(f"[lgbm-residual] Train: {len(train_df):,} rows, Val: {len(val_df):,} rows")
    typer.echo(f"[lgbm-residual] Val dates: {unique_dates[-val_days].date()} to {unique_dates[-1].date()}")
    
    # Train models
    all_results: dict[str, dict[str, Any]] = {}
    best_models: dict[str, tuple[ResidualTrainingResult, float]] = {}
    
    for target in target_list:
        typer.echo(f"\n[lgbm-residual] Training {target}...")
        
        # Compute baseline metrics first
        share_col = f"share_{target}"
        valid_col = f"share_{target}_valid"
        
        def get_valid_mask(df_: pd.DataFrame) -> pd.Series:
            if valid_col in df_.columns:
                explicit_valid = df_[valid_col].fillna(True)
            else:
                explicit_valid = pd.Series(True, index=df_.index)
            share_finite = df_[share_col].notna() & np.isfinite(df_[share_col])
            return explicit_valid & share_finite
        
        val_valid = val_df[get_valid_mask(val_df)].copy()
        baseline_logw = compute_baseline_log_weights(val_valid, target, alpha)
        baseline_metrics = compute_metrics(val_valid, baseline_logw, target, alpha)
        
        typer.echo(f"[lgbm-residual] Baseline val: MAE={baseline_metrics.share_MAE:.4f}")
        
        # Try each shrink value
        best_result = None
        best_shrink = None
        best_val_mae = float("inf")
        
        for s in shrink_list:
            result = train_residual_model(
                train_df, val_df, target, available_features, alpha, s, seed, num_threads
            )
            
            improvement = (1 - result.val_metrics.share_MAE / baseline_metrics.share_MAE) * 100
            typer.echo(
                f"  shrink={s}: val MAE={result.val_metrics.share_MAE:.4f} "
                f"({improvement:+.1f}% vs baseline) KL={result.val_metrics.KL:.4f}"
            )
            
            if result.val_metrics.share_MAE < best_val_mae:
                best_val_mae = result.val_metrics.share_MAE
                best_result = result
                best_shrink = s
        
        if best_result is not None:
            best_models[target] = (best_result, best_shrink)
            status = "✓" if best_result.val_metrics.share_MAE < baseline_metrics.share_MAE_baseline else "✗"
            improvement = (1 - best_result.val_metrics.share_MAE / baseline_metrics.share_MAE) * 100
            typer.echo(
                f"[lgbm-residual] {target}: {status} Best shrink={best_shrink} "
                f"MAE={best_result.val_metrics.share_MAE:.4f} ({improvement:+.1f}% vs baseline)"
            )
            all_results[target] = {
                "val": best_result.val_metrics.to_dict(),
                "train": best_result.train_metrics.to_dict(),
                "shrink": best_shrink,
            }
    
    # Save artifacts
    artifacts_dir = root / "artifacts" / "usage_shares_v1" / "runs" / run_id
    lgbm_dir = artifacts_dir / "lgbm_residual"
    lgbm_dir.mkdir(parents=True, exist_ok=True)
    
    for target, (result, shrink_val) in best_models.items():
        model_path = lgbm_dir / f"model_{target}.txt"
        result.model.booster_.save_model(str(model_path))
        typer.echo(f"[lgbm-residual] Saved model to {model_path}")
    
    # Save config
    config = {
        "mode": "residual",
        "shrink_values": {t: best_models[t][1] for t in best_models},
        "alpha": alpha,
        "feature_cols": available_features,
    }
    (lgbm_dir / "config.json").write_text(json.dumps(config, indent=2))
    
    # Save metrics
    (artifacts_dir / "metrics_residual.json").write_text(json.dumps(all_results, indent=2))
    
    # Save meta
    meta = {
        "run_id": run_id,
        "backend": "lgbm_residual",
        "mode": "residual_on_baseline",
        "git_sha": get_git_sha(),
        "date_range": [start.date().isoformat(), end.date().isoformat()],
        "val_split": {"method": "tail_days", "n_days": val_days},
        "alpha": alpha,
        "shrink_values": {t: best_models[t][1] for t in best_models},
        "min_minutes_actual": min_minutes_actual,
        "seed": seed,
        "targets": target_list,
        "leakage_check": {"n_leaky": n_leaky, "n_checked": n_checked, "missing_frac": round(missing_frac, 4)},
        "created_at": datetime.now().isoformat(),
    }
    (lgbm_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    
    typer.echo(f"\n[lgbm-residual] Artifacts saved to {artifacts_dir}")
    
    # Summary
    typer.echo("\n=== SUMMARY (Residual Training) ===")
    for target in target_list:
        if target in all_results:
            m = all_results[target]["val"]
            s = all_results[target]["shrink"]
            status = "✓" if m["share_MAE"] < m["share_MAE_baseline"] else "✗"
            improvement = (1 - m["share_MAE"] / m["share_MAE_baseline"]) * 100
            typer.echo(
                f"{target}: {status} shrink={s} MAE={m['share_MAE']:.4f} "
                f"(baseline={m['share_MAE_baseline']:.4f}, {improvement:+.1f}%) "
                f"KL={m['KL']:.4f} top1={m['top1_acc']:.2%}"
            )


if __name__ == "__main__":
    app()
