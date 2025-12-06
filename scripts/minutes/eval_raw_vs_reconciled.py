"""Evaluate raw model outputs vs L2-reconciled minutes predictions.

This script:
1. Loads historical features and actuals
2. Scores with the production model (raw quantiles)
3. Applies L2 reconciliation
4. Compares metrics: MAE, coverage, team-level error

Usage:
    uv run python scripts/minutes/eval_raw_vs_reconciled.py \
        --start-date 2024-01-01 \
        --end-date 2024-06-30 \
        --bundle-dir artifacts/minutes_lgbm/<run_id>
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from projections import paths
from projections.minutes_v1.reconcile import (
    ReconcileConfig,
    load_reconcile_config,
    reconcile_minutes_p50_all,
)
from projections.cli.score_minutes_v1 import _load_bundle, _score_rows

console = Console()
app = typer.Typer(help=__doc__)

DEFAULT_RECONCILE_CONFIG = Path("config/minutes_l2_reconcile.yaml")


def _compute_metrics(
    df: pd.DataFrame,
    *,
    pred_col: str,
    actual_col: str = "minutes",
    group_cols: list[str] | None = None,
) -> dict:
    """Compute MAE, coverage, and team-level metrics."""
    valid = df[df[actual_col].notna() & df[pred_col].notna()].copy()
    if valid.empty:
        return {"mae": None, "n": 0}
    
    actual = valid[actual_col].to_numpy(dtype=float)
    pred = valid[pred_col].to_numpy(dtype=float)
    
    mae = float(np.abs(pred - actual).mean())
    rmse = float(np.sqrt(np.mean((pred - actual) ** 2)))
    
    # Coverage for p10/p90 if available
    p10_col = pred_col.replace("p50", "p10")
    p90_col = pred_col.replace("p50", "p90")
    p10_cov = p90_cov = None
    if p10_col in valid.columns and p90_col in valid.columns:
        p10 = valid[p10_col].to_numpy(dtype=float)
        p90 = valid[p90_col].to_numpy(dtype=float)
        p10_cov = float((actual >= p10).mean())
        p90_cov = float((actual <= p90).mean())
    
    # Team-level total error
    team_error = None
    if group_cols and all(c in valid.columns for c in group_cols):
        team_totals = valid.groupby(group_cols).agg({
            pred_col: "sum",
            actual_col: "sum",
        })
        team_error = float(np.abs(team_totals[pred_col] - team_totals[actual_col]).mean())
    
    return {
        "mae": mae,
        "rmse": rmse,
        "n": len(valid),
        "p10_coverage": p10_cov,
        "p90_coverage": p90_cov,
        "team_total_mae": team_error,
    }


def _compute_by_bucket(
    df: pd.DataFrame,
    *,
    pred_col: str,
    actual_col: str = "minutes",
    bucket_col: str = "minutes_bucket",
) -> dict[str, dict]:
    """Compute metrics by minutes bucket (0-10, 10-20, etc.)."""
    valid = df[df[actual_col].notna()].copy()
    if bucket_col not in valid.columns:
        actual = valid[actual_col].to_numpy(dtype=float)
        valid[bucket_col] = pd.cut(
            actual,
            bins=[0, 10, 20, 30, 40, 50],
            labels=["0-10", "10-20", "20-30", "30-40", "40+"],
            include_lowest=True,
        )
    
    results = {}
    for bucket, group in valid.groupby(bucket_col, observed=True):
        if len(group) < 10:
            continue
        actual = group[actual_col].to_numpy(dtype=float)
        pred = group[pred_col].to_numpy(dtype=float)
        results[str(bucket)] = {
            "mae": float(np.abs(pred - actual).mean()),
            "n": len(group),
        }
    return results


def _rotation_vs_bench_split(
    df: pd.DataFrame,
    *,
    pred_col: str,
    actual_col: str = "minutes",
    rotation_threshold: float = 4.0,
) -> dict:
    """Compare metrics for rotation players vs deep bench."""
    valid = df[df[actual_col].notna()].copy()
    
    # Define rotation as actual >= threshold or starter
    starter_col = None
    for col in ("is_projected_starter", "starter_flag", "is_starter"):
        if col in valid.columns:
            starter_col = col
            break
    
    actual = valid[actual_col].to_numpy(dtype=float)
    is_rotation = actual >= rotation_threshold
    if starter_col:
        is_rotation = is_rotation | valid[starter_col].fillna(0).astype(bool).to_numpy()
    
    rotation = valid[is_rotation]
    bench = valid[~is_rotation]
    
    pred_rotation = rotation[pred_col].to_numpy(dtype=float)
    actual_rotation = rotation[actual_col].to_numpy(dtype=float)
    pred_bench = bench[pred_col].to_numpy(dtype=float)
    actual_bench = bench[actual_col].to_numpy(dtype=float)
    
    return {
        "rotation": {
            "mae": float(np.abs(pred_rotation - actual_rotation).mean()) if len(rotation) > 0 else None,
            "n": len(rotation),
            "pred_mean": float(pred_rotation.mean()) if len(rotation) > 0 else None,
            "actual_mean": float(actual_rotation.mean()) if len(rotation) > 0 else None,
        },
        "bench": {
            "mae": float(np.abs(pred_bench - actual_bench).mean()) if len(bench) > 0 else None,
            "n": len(bench),
            "pred_mean": float(pred_bench.mean()) if len(bench) > 0 else None,
            "actual_mean": float(actual_bench.mean()) if len(bench) > 0 else None,
            "minutes_leaked": float(pred_bench.sum()) if len(bench) > 0 else 0,
        },
    }


@app.command()
def main(
    start_date: str = typer.Option(..., help="Start date YYYY-MM-DD."),
    end_date: str = typer.Option(..., help="End date YYYY-MM-DD."),
    bundle_dir: Optional[Path] = typer.Option(
        None, help="Path to model bundle (defaults to production)."
    ),
    data_root: Path = typer.Option(
        paths.get_data_root(), help="Data root containing features."
    ),
    reconcile_config: Path = typer.Option(
        DEFAULT_RECONCILE_CONFIG, help="Path to L2 reconcile config YAML."
    ),
    output_json: Optional[Path] = typer.Option(
        None, help="Write detailed results to JSON file."
    ),
    season: Optional[int] = typer.Option(None, help="Filter to specific season."),
) -> None:
    """Compare raw model outputs vs L2-reconciled predictions."""
    
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    
    console.print(f"[bold]Evaluating raw vs reconciled: {start.date()} → {end.date()}[/bold]")
    
    # Load features from all relevant partitions
    console.print("[dim]Loading features...[/dim]")
    features_root = data_root / "gold" / "features_minutes_v1"
    if not features_root.exists():
        console.print(f"[red]Features root not found: {features_root}[/red]")
        raise typer.Exit(1)
    
    # Collect all parquet files in date range
    frames = []
    for season_dir in features_root.glob("season=*"):
        season_val = int(season_dir.name.split("=")[1])
        if season and season_val != season:
            continue
        for month_dir in season_dir.glob("month=*"):
            parquet_path = month_dir / "features.parquet"
            if parquet_path.exists():
                frames.append(pd.read_parquet(parquet_path))
    
    if not frames:
        console.print("[red]No feature files found.[/red]")
        raise typer.Exit(1)
    
    df = pd.concat(frames, ignore_index=True)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.normalize()
    df = df[(df["game_date"] >= start) & (df["game_date"] <= end)]
    
    if df.empty:
        console.print("[red]No data in date range.[/red]")
        raise typer.Exit(1)
    
    console.print(f"[dim]Loaded {len(df):,} rows across {df['game_date'].nunique()} dates[/dim]")
    
    # Load model bundle
    if bundle_dir is None:
        # Try to find production bundle from config
        current_run_path = Path("config/minutes_current_run.json")
        if current_run_path.exists():
            with current_run_path.open() as f:
                run_info = json.load(f)
            bundle_dir = Path(run_info.get("artifact_root", "artifacts/minutes_lgbm")) / run_info["run_id"]
        else:
            console.print("[red]No bundle specified and no production config found.[/red]")
            raise typer.Exit(1)
    
    console.print(f"[dim]Loading bundle from {bundle_dir}[/dim]")
    bundle = _load_bundle(bundle_dir)
    
    # Score (raw predictions)
    console.print("[dim]Scoring with model...[/dim]")
    scored = _score_rows(df, bundle)
    
    # Rename raw columns for comparison
    scored["minutes_p50_raw"] = scored["minutes_p50"].copy()
    scored["minutes_p10_raw"] = scored["minutes_p10"].copy()
    scored["minutes_p90_raw"] = scored["minutes_p90"].copy()
    
    # Apply reconciliation
    console.print("[dim]Applying L2 reconciliation...[/dim]")
    if reconcile_config.exists():
        cfg = load_reconcile_config(reconcile_config)
    else:
        console.print(f"[yellow]Reconcile config not found at {reconcile_config}, using defaults[/yellow]")
        cfg = ReconcileConfig()
    
    reconciled = reconcile_minutes_p50_all(scored, cfg)
    
    # Compute metrics
    console.print("\n[bold cyan]═══ Results ═══[/bold cyan]\n")
    
    raw_metrics = _compute_metrics(
        scored, pred_col="minutes_p50_raw", group_cols=["game_id", "team_id"]
    )
    recon_metrics = _compute_metrics(
        reconciled, pred_col="minutes_p50", group_cols=["game_id", "team_id"]
    )
    
    # Main comparison table
    table = Table(title="Overall Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Raw", justify="right")
    table.add_column("Reconciled", justify="right")
    table.add_column("Δ", justify="right")
    
    def _fmt(v, precision=3):
        return f"{v:.{precision}f}" if v is not None else "-"
    
    def _delta(raw, recon):
        if raw is None or recon is None:
            return "-"
        d = recon - raw
        color = "green" if d < 0 else "red" if d > 0 else ""
        return f"[{color}]{d:+.3f}[/{color}]" if color else f"{d:+.3f}"
    
    table.add_row("MAE (p50)", _fmt(raw_metrics["mae"]), _fmt(recon_metrics["mae"]), 
                  _delta(raw_metrics["mae"], recon_metrics["mae"]))
    table.add_row("RMSE", _fmt(raw_metrics["rmse"]), _fmt(recon_metrics["rmse"]),
                  _delta(raw_metrics["rmse"], recon_metrics["rmse"]))
    table.add_row("Team Total MAE", _fmt(raw_metrics["team_total_mae"]), _fmt(recon_metrics["team_total_mae"]),
                  _delta(raw_metrics["team_total_mae"], recon_metrics["team_total_mae"]))
    table.add_row("P10 Coverage", _fmt(raw_metrics["p10_coverage"]), _fmt(recon_metrics["p10_coverage"]), "-")
    table.add_row("P90 Coverage", _fmt(raw_metrics["p90_coverage"]), _fmt(recon_metrics["p90_coverage"]), "-")
    table.add_row("N", str(raw_metrics["n"]), str(recon_metrics["n"]), "-")
    
    console.print(table)
    
    # Rotation vs bench
    raw_split = _rotation_vs_bench_split(scored, pred_col="minutes_p50_raw")
    recon_split = _rotation_vs_bench_split(reconciled, pred_col="minutes_p50")
    
    console.print("\n")
    split_table = Table(title="Rotation vs Deep Bench")
    split_table.add_column("Segment", style="cyan")
    split_table.add_column("N", justify="right")
    split_table.add_column("Raw MAE", justify="right")
    split_table.add_column("Recon MAE", justify="right")
    split_table.add_column("Raw Pred Mean", justify="right")
    split_table.add_column("Actual Mean", justify="right")
    
    split_table.add_row(
        "Rotation (≥4min)",
        str(raw_split["rotation"]["n"]),
        _fmt(raw_split["rotation"]["mae"]),
        _fmt(recon_split["rotation"]["mae"]),
        _fmt(raw_split["rotation"]["pred_mean"], 1),
        _fmt(raw_split["rotation"]["actual_mean"], 1),
    )
    split_table.add_row(
        "Deep Bench (<4min)",
        str(raw_split["bench"]["n"]),
        _fmt(raw_split["bench"]["mae"]),
        _fmt(recon_split["bench"]["mae"]),
        _fmt(raw_split["bench"]["pred_mean"], 1),
        _fmt(raw_split["bench"]["actual_mean"], 1),
    )
    
    console.print(split_table)
    
    # Minutes leaked to bench
    console.print(f"\n[yellow]Minutes 'leaked' to deep bench (raw): {raw_split['bench']['minutes_leaked']:,.0f} total[/yellow]")
    console.print(f"[green]Minutes 'leaked' to deep bench (reconciled): {recon_split['bench']['minutes_leaked']:,.0f} total[/green]")
    
    # By minutes bucket
    raw_buckets = _compute_by_bucket(scored, pred_col="minutes_p50_raw")
    recon_buckets = _compute_by_bucket(reconciled, pred_col="minutes_p50")
    
    console.print("\n")
    bucket_table = Table(title="MAE by Actual Minutes Bucket")
    bucket_table.add_column("Bucket", style="cyan")
    bucket_table.add_column("N", justify="right")
    bucket_table.add_column("Raw MAE", justify="right")
    bucket_table.add_column("Recon MAE", justify="right")
    bucket_table.add_column("Δ", justify="right")
    
    for bucket in ["0-10", "10-20", "20-30", "30-40", "40+"]:
        raw_b = raw_buckets.get(bucket, {})
        recon_b = recon_buckets.get(bucket, {})
        bucket_table.add_row(
            bucket,
            str(raw_b.get("n", 0)),
            _fmt(raw_b.get("mae")),
            _fmt(recon_b.get("mae")),
            _delta(raw_b.get("mae"), recon_b.get("mae")),
        )
    
    console.print(bucket_table)
    
    # Summary
    console.print("\n[bold]Summary:[/bold]")
    mae_diff = (recon_metrics["mae"] or 0) - (raw_metrics["mae"] or 0)
    if mae_diff < -0.1:
        console.print(f"[green]✓ Reconciliation improves MAE by {abs(mae_diff):.3f} minutes[/green]")
    elif mae_diff > 0.1:
        console.print(f"[red]✗ Reconciliation hurts MAE by {mae_diff:.3f} minutes[/red]")
    else:
        console.print("[yellow]→ Reconciliation has minimal effect on MAE[/yellow]")
    
    # Save JSON output
    if output_json:
        results = {
            "date_range": {"start": start_date, "end": end_date},
            "raw": raw_metrics,
            "reconciled": recon_metrics,
            "raw_split": raw_split,
            "recon_split": recon_split,
            "raw_buckets": raw_buckets,
            "recon_buckets": recon_buckets,
        }
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with output_json.open("w") as f:
            json.dump(results, f, indent=2)
        console.print(f"\n[dim]Results written to {output_json}[/dim]")


if __name__ == "__main__":
    app()
