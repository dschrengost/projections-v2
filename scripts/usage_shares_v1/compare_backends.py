"""
Compare LightGBM and NN backends for usage shares prediction.

Trains or loads both backends and prints a side-by-side metrics comparison table.

Usage:
    uv run python -m scripts.usage_shares_v1.compare_backends \\
        --data-root /home/daniel/projections-data \\
        --targets fga \\
        --start-date 2024-11-01 \\
        --end-date 2025-02-01
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from projections.paths import data_path
from projections.usage_shares_v1.features import (
    GROUP_COLS,
    add_derived_features,
)

app = typer.Typer(add_completion=False, help=__doc__)
console = Console()


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


# =============================================================================
# Metrics Computation
# =============================================================================


def compute_baseline_metrics(
    df: pd.DataFrame,
    target: str,
) -> dict[str, float]:
    """
    Compute rate-weighted baseline metrics.
    
    Baseline: season_{target}_per_min * minutes_pred_p50
    """
    share_col = f"share_{target}"
    valid_col = f"share_{target}_valid"
    season_rate_col = f"season_{target}_per_min"
    eps = 1e-9
    
    # Filter to valid rows
    if valid_col in df.columns:
        df = df[df[valid_col].fillna(False)].copy()
    
    if len(df) == 0:
        return {}
    
    # Compute baseline weights
    if season_rate_col in df.columns and "minutes_pred_p50" in df.columns:
        rate = df[season_rate_col].fillna(0.0)
        mins = df["minutes_pred_p50"].fillna(0.0)
        weights = rate * mins
    else:
        weights = df["minutes_pred_p50"].fillna(1.0) if "minutes_pred_p50" in df.columns else pd.Series(1.0, index=df.index)
    
    weights = weights.clip(lower=eps)
    
    # Normalize within groups
    working = df[GROUP_COLS + [share_col]].copy()
    working["weight"] = weights
    group_sums = working.groupby(GROUP_COLS)["weight"].transform("sum")
    working["share_pred"] = working["weight"] / group_sums.clip(lower=eps)
    working["share_true"] = working[share_col]
    
    # Compute metrics
    working["abs_err"] = (working["share_pred"] - working["share_true"]).abs()
    working["share_pred_sq"] = working["share_pred"] ** 2
    working["share_true_sq"] = working["share_true"] ** 2
    
    # Group-level aggregations
    group_agg = working.groupby(GROUP_COLS).agg(
        mae=("abs_err", "mean"),
        H_pred=("share_pred_sq", "sum"),
        H_true=("share_true_sq", "sum"),
    ).reset_index()
    
    # KL per group
    def kl_per_group(g: pd.DataFrame) -> float:
        s_true = g["share_true"].values + eps
        s_pred = g["share_pred"].values + eps
        return float(np.sum(s_true * np.log(s_true / s_pred)))
    
    kl_series = working.groupby(GROUP_COLS).apply(kl_per_group, include_groups=False)
    
    # Top-1 accuracy
    def top1_match(g: pd.DataFrame) -> float:
        pred_top = g["share_pred"].idxmax()
        true_top = g["share_true"].idxmax()
        return 1.0 if pred_top == true_top else 0.0
    
    top1_series = working.groupby(GROUP_COLS).apply(top1_match, include_groups=False)
    
    return {
        "share_MAE": float(group_agg["mae"].mean()),
        "KL": float(kl_series.mean()),
        "top1_acc": float(top1_series.mean()),
        "H_mean_pred": float(group_agg["H_pred"].mean()),
        "H_mean_true": float(group_agg["H_true"].mean()),
    }


def load_metrics_from_artifacts(
    data_root: Path,
    run_id: str,
    backend: str,
) -> dict[str, dict[str, Any]]:
    """Load metrics from saved artifacts."""
    run_dir = data_root / "artifacts" / "usage_shares_v1" / "runs" / run_id
    
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return {}
    
    all_metrics = json.loads(metrics_path.read_text())
    
    # Normalize to expected format
    result = {}
    for target, target_metrics in all_metrics.items():
        if backend == "lgbm" and "val" in target_metrics:
            result[target] = target_metrics["val"]
        elif backend == "nn" and "nn_val" in target_metrics:
            result[target] = target_metrics["nn_val"]
        elif backend == "nn" and "val" in target_metrics:
            result[target] = target_metrics["val"]
        elif "val" in target_metrics:
            result[target] = target_metrics["val"]
    
    return result


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
        help="Run ID to load (if provided, loads existing artifacts instead of training).",
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
        help="Comma-separated list of targets to compare.",
    ),
    val_days: int = typer.Option(
        30,
        help="Number of trailing days to use for validation (for baseline computation).",
    ),
    train: bool = typer.Option(
        False,
        "--train/--no-train",
        help="Train both backends before comparing (default: load existing artifacts).",
    ),
) -> None:
    """Compare LightGBM and NN backends for usage shares prediction."""
    
    root = data_root or data_path()
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    target_list = [t.strip() for t in targets.split(",")]
    
    console.print("[bold]Comparing usage shares backends[/bold]")
    console.print(f"Date range: {start.date()} to {end.date()}")
    console.print(f"Targets: {target_list}")
    
    if train:
        # Train both backends
        import subprocess
        
        run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        console.print("\n[yellow]Training LGBM...[/yellow]")
        subprocess.run([
            "uv", "run", "python", "-m", "scripts.usage_shares_v1.train_lgbm",
            "--data-root", str(root),
            "--run-id", run_id,
            "--targets", targets,
            "--start-date", start_date,
            "--end-date", end_date,
        ], check=True)
        
        console.print("\n[yellow]Training NN...[/yellow]")
        subprocess.run([
            "uv", "run", "python", "-m", "scripts.usage_shares_v1.train_nn",
            "--data-root", str(root),
            "--run-id", run_id,
            "--targets", targets,
            "--start-date", start_date,
            "--end-date", end_date,
            "--epochs", "20",
        ], check=True)
    
    # Find latest run if not specified
    if run_id is None:
        runs_dir = root / "artifacts" / "usage_shares_v1" / "runs"
        if runs_dir.exists():
            runs = sorted([d.name for d in runs_dir.iterdir() if d.is_dir()])
            if runs:
                run_id = runs[-1]
                console.print(f"Using latest run: {run_id}")
    
    if run_id is None:
        console.print("[red]No run ID specified and no existing runs found. Use --train to train models.[/red]")
        raise typer.Exit(1)
    
    # Load data for baseline computation
    console.print("\nLoading validation data...")
    df = load_training_data(root, start, end)
    df = add_derived_features(df)
    
    unique_dates = sorted(df["game_date"].unique())
    val_start_date = unique_dates[-val_days]
    val_df = df[df["game_date"] >= val_start_date].copy()
    console.print(f"Validation set: {len(val_df):,} rows from {val_days} days")
    
    # Load metrics
    lgbm_metrics = load_metrics_from_artifacts(root, run_id, "lgbm")
    nn_metrics = load_metrics_from_artifacts(root, run_id, "nn")
    
    # Compute baseline metrics
    baseline_metrics = {}
    for target in target_list:
        if f"share_{target}" in val_df.columns:
            baseline_metrics[target] = compute_baseline_metrics(val_df, target)
    
    # Build comparison table
    table = Table(title=f"Usage Shares Backend Comparison (run: {run_id})")
    table.add_column("Target", style="cyan")
    table.add_column("Backend", style="magenta")
    table.add_column("share_MAE", justify="right")
    table.add_column("KL", justify="right")
    table.add_column("top1_acc", justify="right")
    table.add_column("Beats Baseline", justify="center")
    
    for target in target_list:
        # Baseline row
        if target in baseline_metrics:
            bm = baseline_metrics[target]
            table.add_row(
                target,
                "rate_weighted",
                f"{bm.get('share_MAE', 0):.4f}",
                f"{bm.get('KL', 0):.4f}",
                f"{bm.get('top1_acc', 0):.2%}",
                "-",
            )
        
        # LGBM row
        if target in lgbm_metrics:
            lm = lgbm_metrics[target]
            baseline_mae = baseline_metrics.get(target, {}).get("share_MAE", 999)
            beats = "✓" if lm.get("share_MAE", 999) < baseline_mae else "✗"
            table.add_row(
                target,
                "lgbm",
                f"{lm.get('share_MAE', 0):.4f}",
                f"{lm.get('KL', 0):.4f}",
                f"{lm.get('top1_acc', 0):.2%}",
                beats,
            )
        
        # NN row
        if target in nn_metrics:
            nm = nn_metrics[target]
            baseline_mae = baseline_metrics.get(target, {}).get("share_MAE", 999)
            beats = "✓" if nm.get("share_MAE", 999) < baseline_mae else "✗"
            table.add_row(
                target,
                "nn",
                f"{nm.get('share_MAE', 0):.4f}",
                f"{nm.get('KL', 0):.4f}",
                f"{nm.get('top1_acc', 0):.2%}",
                beats,
            )
        
        # Separator between targets
        if target != target_list[-1]:
            table.add_row("", "", "", "", "", "")
    
    console.print()
    console.print(table)
    
    # Summary
    console.print("\n[bold]Summary:[/bold]")
    for target in target_list:
        lgbm_mae = lgbm_metrics.get(target, {}).get("share_MAE", 999)
        nn_mae = nn_metrics.get(target, {}).get("share_MAE", 999)
        base_mae = baseline_metrics.get(target, {}).get("share_MAE", 999)
        
        if lgbm_mae < nn_mae and lgbm_mae < base_mae:
            best = "lgbm"
        elif nn_mae < lgbm_mae and nn_mae < base_mae:
            best = "nn"
        elif base_mae <= lgbm_mae and base_mae <= nn_mae:
            best = "rate_weighted (baseline)"
        else:
            best = "unknown"
        
        console.print(f"  {target}: Best backend = [green]{best}[/green]")


if __name__ == "__main__":
    app()
