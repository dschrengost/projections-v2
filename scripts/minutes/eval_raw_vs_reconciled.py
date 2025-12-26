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
from projections.minutes_v1.production import resolve_production_run_dir

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
            bins=[0, 10, 20, 30, 1e9],
            labels=["0-10", "10-20", "20-30", "30+"],
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


def _starter_vs_bench_split(
    df: pd.DataFrame,
    *,
    pred_col: str,
    actual_col: str = "minutes",
) -> dict[str, dict]:
    """Compute MAE/bias split by starter vs non-starter."""
    valid = df[df[actual_col].notna() & df[pred_col].notna()].copy()
    if valid.empty:
        return {}

    starter_col = None
    for col in ("is_starter", "is_confirmed_starter", "is_projected_starter", "starter_flag"):
        if col in valid.columns:
            starter_col = col
            break
    if starter_col is None:
        return {}

    is_starter = valid[starter_col].fillna(0).astype(bool).to_numpy()
    actual = valid[actual_col].to_numpy(dtype=float)
    pred = valid[pred_col].to_numpy(dtype=float)
    err = pred - actual

    def _block(mask: np.ndarray) -> dict:
        if mask.sum() == 0:
            return {"n": 0, "mae": None, "bias": None, "pred_mean": None, "actual_mean": None}
        return {
            "n": int(mask.sum()),
            "mae": float(np.abs(err[mask]).mean()),
            "bias": float(err[mask].mean()),
            "pred_mean": float(pred[mask].mean()),
            "actual_mean": float(actual[mask].mean()),
        }

    return {
        "starter_col": starter_col,
        "starters": _block(is_starter),
        "bench": _block(~is_starter),
    }


def _topk_vs_rest_split(
    df: pd.DataFrame,
    *,
    pred_col: str,
    actual_col: str = "minutes",
    group_cols: list[str] | None = None,
    k: int = 8,
) -> dict[str, dict]:
    """Compute MAE/bias split by actual top-k minutes per team-game."""
    if group_cols is None:
        group_cols = ["game_id", "team_id"]

    valid = df[df[actual_col].notna() & df[pred_col].notna()].copy()
    if valid.empty:
        return {}

    actual = valid[actual_col].to_numpy(dtype=float)
    pred = valid[pred_col].to_numpy(dtype=float)
    err = pred - actual

    top_mask = np.zeros(len(valid), dtype=bool)
    for _, g in valid.groupby(group_cols, sort=False):
        if len(g) == 0:
            continue
        g_actual = pd.to_numeric(g[actual_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if g_actual.size == 0:
            continue
        order = np.argsort(-g_actual, kind="mergesort")
        take = min(int(k), int(order.size))
        idx = g.index.to_numpy()[order[:take]]
        top_mask[valid.index.get_indexer(idx)] = True

    def _block(mask: np.ndarray) -> dict:
        if mask.sum() == 0:
            return {"n": 0, "mae": None, "bias": None}
        return {
            "n": int(mask.sum()),
            "mae": float(np.abs(err[mask]).mean()),
            "bias": float(err[mask].mean()),
        }

    return {
        "k": int(k),
        "topk": _block(top_mask),
        "rest": _block(~top_mask),
    }


def _gini_coefficient(values: np.ndarray) -> float:
    """Compute Gini coefficient for a non-negative distribution."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0.0)]
    if arr.size < 2:
        return 0.0
    arr = np.sort(arr)
    total = float(arr.sum())
    if total <= 0.0:
        return 0.0
    cumsum = np.cumsum(arr)
    n = float(arr.size)
    gini = (n + 1.0 - 2.0 * float(cumsum.sum()) / total) / n
    return float(max(0.0, min(1.0, gini)))


def _hhi(values: np.ndarray, *, total: float = 240.0) -> float:
    """Compute HHI on minute shares."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0.0)]
    if arr.size == 0:
        return 0.0
    shares = arr / float(total)
    return float(np.sum(shares**2))


def _sixth_man_mae(
    df: pd.DataFrame,
    *,
    pred_col: str,
    actual_col: str = "minutes",
    group_cols: list[str] | None = None,
) -> dict[str, float | int | None]:
    """Compute absolute error for the actual 6th man (by actual minutes)."""
    if group_cols is None:
        group_cols = ["game_id", "team_id"]
    valid = df[df[actual_col].notna() & df[pred_col].notna()].copy()
    if valid.empty:
        return {"sixth_man_mae": None, "n_teams": 0}

    errs: list[float] = []
    for _, g in valid.groupby(group_cols, sort=False):
        if len(g) < 6:
            continue
        actual = pd.to_numeric(g[actual_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        pred = pd.to_numeric(g[pred_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        order = np.argsort(-actual, kind="mergesort")
        idx6 = int(order[5])
        errs.append(abs(float(pred[idx6] - actual[idx6])))
    return {
        "sixth_man_mae": float(np.mean(errs)) if errs else None,
        "n_teams": int(len(errs)),
    }


def _team_realism_metrics(
    df: pd.DataFrame,
    *,
    minutes_col: str,
    actual_col: str = "minutes",
    group_cols: list[str] | None = None,
) -> dict[str, float]:
    """Compute team-level distribution metrics for predicted and actual minutes."""
    if group_cols is None:
        group_cols = ["game_id", "team_id"]

    valid = df[df[actual_col].notna() & df[minutes_col].notna()].copy()
    if valid.empty:
        return {}

    pred_gini: list[float] = []
    act_gini: list[float] = []
    pred_hhi: list[float] = []
    act_hhi: list[float] = []
    pred_top6: list[float] = []
    act_top6: list[float] = []
    pred_top8: list[float] = []
    act_top8: list[float] = []
    pred_roster: list[float] = []
    act_roster: list[float] = []
    pred_bench_max: list[float] = []
    act_bench_max: list[float] = []

    for _, g in valid.groupby(group_cols, sort=False):
        pred = pd.to_numeric(g[minutes_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        act = pd.to_numeric(g[actual_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)

        pred_sorted = np.sort(pred)[::-1]
        act_sorted = np.sort(act)[::-1]
        n = int(len(pred_sorted))
        half = n // 2

        pred_gini.append(_gini_coefficient(pred_sorted))
        act_gini.append(_gini_coefficient(act_sorted))
        pred_hhi.append(_hhi(pred_sorted))
        act_hhi.append(_hhi(act_sorted))

        pred_top6.append(float(pred_sorted[:6].sum() / 240.0) if n else 0.0)
        act_top6.append(float(act_sorted[:6].sum() / 240.0) if n else 0.0)
        pred_top8.append(float(pred_sorted[:8].sum() / 240.0) if n else 0.0)
        act_top8.append(float(act_sorted[:8].sum() / 240.0) if n else 0.0)

        pred_roster.append(float((pred >= 1.0).sum()))
        act_roster.append(float((act >= 1.0).sum()))

        pred_bench_max.append(float(pred_sorted[half:].max()) if half < n else 0.0)
        act_bench_max.append(float(act_sorted[half:].max()) if half < n else 0.0)

    def _mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else float("nan")

    return {
        "pred_top6_share_mean": _mean(pred_top6),
        "act_top6_share_mean": _mean(act_top6),
        "pred_top8_share_mean": _mean(pred_top8),
        "act_top8_share_mean": _mean(act_top8),
        "pred_roster_size_mean": _mean(pred_roster),
        "act_roster_size_mean": _mean(act_roster),
        "pred_bench_max_mean": _mean(pred_bench_max),
        "act_bench_max_mean": _mean(act_bench_max),
        "pred_gini_mean": _mean(pred_gini),
        "act_gini_mean": _mean(act_gini),
        "pred_hhi_mean": _mean(pred_hhi),
        "act_hhi_mean": _mean(act_hhi),
    }


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
        bundle_dir, _ = resolve_production_run_dir()
    
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
    
    for bucket in ["0-10", "10-20", "20-30", "30+"]:
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

    # Starter vs bench split
    raw_starters = _starter_vs_bench_split(scored, pred_col="minutes_p50_raw")
    recon_starters = _starter_vs_bench_split(reconciled, pred_col="minutes_p50")
    if raw_starters:
        console.print("\n")
        starter_table = Table(title=f"Starters vs Bench (starter_col={raw_starters.get('starter_col')})")
        starter_table.add_column("Segment", style="cyan")
        starter_table.add_column("N", justify="right")
        starter_table.add_column("Raw MAE", justify="right")
        starter_table.add_column("Recon MAE", justify="right")
        starter_table.add_column("Raw Bias", justify="right")
        starter_table.add_column("Recon Bias", justify="right")
        for key, label in (("starters", "Starters"), ("bench", "Bench")):
            starter_table.add_row(
                label,
                str(raw_starters[key]["n"]),
                _fmt(raw_starters[key]["mae"]),
                _fmt(recon_starters.get(key, {}).get("mae")),
                _fmt(raw_starters[key]["bias"]),
                _fmt(recon_starters.get(key, {}).get("bias")),
            )
        console.print(starter_table)

    # Top-8 rotation vs rest (by actual minutes rank)
    raw_top8 = _topk_vs_rest_split(scored, pred_col="minutes_p50_raw", k=8)
    recon_top8 = _topk_vs_rest_split(reconciled, pred_col="minutes_p50", k=8)
    if raw_top8:
        console.print("\n")
        top8_table = Table(title="Top-8 (by actual minutes) vs Rest")
        top8_table.add_column("Segment", style="cyan")
        top8_table.add_column("N", justify="right")
        top8_table.add_column("Raw MAE", justify="right")
        top8_table.add_column("Recon MAE", justify="right")
        top8_table.add_column("Raw Bias", justify="right")
        top8_table.add_column("Recon Bias", justify="right")
        for key, label in (("topk", "Top-8"), ("rest", "Rest")):
            top8_table.add_row(
                label,
                str(raw_top8[key]["n"]),
                _fmt(raw_top8[key]["mae"]),
                _fmt(recon_top8.get(key, {}).get("mae")),
                _fmt(raw_top8[key]["bias"]),
                _fmt(recon_top8.get(key, {}).get("bias")),
            )
        console.print(top8_table)

    # Team-level realism metrics (pred vs actual)
    raw_realism = _team_realism_metrics(scored, minutes_col="minutes_p50_raw")
    recon_realism = _team_realism_metrics(reconciled, minutes_col="minutes_p50")
    if raw_realism:
        console.print("\n")
        realism_table = Table(title="Team-Level Realism (Means)")
        realism_table.add_column("Metric", style="cyan")
        realism_table.add_column("Pred Raw", justify="right")
        realism_table.add_column("Pred Recon", justify="right")
        realism_table.add_column("Actual", justify="right")
        for metric, raw_key, recon_key, act_key, precision in [
            ("Top6 share", "pred_top6_share_mean", "pred_top6_share_mean", "act_top6_share_mean", 3),
            ("Top8 share", "pred_top8_share_mean", "pred_top8_share_mean", "act_top8_share_mean", 3),
            ("Roster size (>=1)", "pred_roster_size_mean", "pred_roster_size_mean", "act_roster_size_mean", 2),
            ("Bench max (bottom-half)", "pred_bench_max_mean", "pred_bench_max_mean", "act_bench_max_mean", 2),
            ("Gini", "pred_gini_mean", "pred_gini_mean", "act_gini_mean", 3),
            ("HHI", "pred_hhi_mean", "pred_hhi_mean", "act_hhi_mean", 4),
        ]:
            realism_table.add_row(
                metric,
                _fmt(raw_realism.get(raw_key), precision),
                _fmt(recon_realism.get(recon_key), precision),
                _fmt(raw_realism.get(act_key), precision),
            )
        console.print(realism_table)

    # Sixth-man minutes error
    raw_sixth = _sixth_man_mae(scored, pred_col="minutes_p50_raw")
    recon_sixth = _sixth_man_mae(reconciled, pred_col="minutes_p50")
    if raw_sixth.get("sixth_man_mae") is not None:
        recon_val = recon_sixth.get("sixth_man_mae")
        recon_mae = float(recon_val) if recon_val is not None else float("nan")
        console.print(
            f"\n[dim]6th-man MAE (actual 6th man by minutes): raw={raw_sixth['sixth_man_mae']:.3f} "
            f"(n_teams={raw_sixth['n_teams']}), recon={recon_mae:.3f}[/dim]"
        )
    
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
            "raw_starters": raw_starters,
            "recon_starters": recon_starters,
            "raw_top8": raw_top8,
            "recon_top8": recon_top8,
            "raw_realism": raw_realism,
            "recon_realism": recon_realism,
            "raw_sixth": raw_sixth,
            "recon_sixth": recon_sixth,
        }
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with output_json.open("w") as f:
            json.dump(results, f, indent=2)
        console.print(f"\n[dim]Results written to {output_json}[/dim]")


if __name__ == "__main__":
    app()
