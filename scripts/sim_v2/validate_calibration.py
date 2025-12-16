"""
Validate simulator tail calibration by comparing predicted percentiles to actual FPTS.

This script:
1. Loads simulation percentile projections (p05, p10, p25, p50, p75, p95)
2. Joins with actual FPTS labels from fpts_training_base
3. Computes coverage: what % of actuals fall below each percentile threshold
4. Expected: p10 should have ~10% below, p50 ~50% below, etc.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import typer

from projections.paths import data_path

app = typer.Typer(add_completion=False)


def _load_sim_projections(root: Path) -> pd.DataFrame:
    """Load all simulation projection summaries."""
    sim_base = root / "artifacts" / "sim_v2" / "projections"
    if not sim_base.exists():
        raise FileNotFoundError(f"No sim projections at {sim_base}")
    
    frames = []
    for date_dir in sim_base.glob("date=*"):
        date_str = date_dir.name.replace("date=", "")
        # Get the latest run for each date
        runs = sorted(date_dir.glob("run=*"))
        if not runs:
            continue
        latest_run = runs[-1]
        parquet_file = latest_run / "sim_v2_projections.parquet"
        if parquet_file.exists():
            df = pd.read_parquet(parquet_file)
            df["game_date"] = pd.to_datetime(date_str)
            frames.append(df)
    
    if not frames:
        raise FileNotFoundError("No projection parquet files found")
    
    return pd.concat(frames, ignore_index=True)


def _load_actuals(root: Path, dates: list[pd.Timestamp]) -> pd.DataFrame:
    """Load actual FPTS from fpts_training_base for specific dates."""
    base = root / "gold" / "fpts_training_base"
    if not base.exists():
        raise FileNotFoundError(f"No fpts_training_base at {base}")
    
    frames = []
    for season_dir in base.glob("season=*"):
        for date_dir in season_dir.glob("game_date=*"):
            date_str = date_dir.name.replace("game_date=", "")
            date_ts = pd.Timestamp(date_str).normalize()
            if date_ts not in dates:
                continue
            parquet_file = date_dir / "fpts_training_base.parquet"
            if parquet_file.exists():
                df = pd.read_parquet(parquet_file)
                df["game_date"] = date_ts
                frames.append(df)
    
    if not frames:
        return pd.DataFrame()
    
    return pd.concat(frames, ignore_index=True)


def _compute_coverage(merged: pd.DataFrame, percentile_col: str, actual_col: str = "dk_fpts_actual") -> dict:
    """Compute coverage statistics for a given percentile column."""
    valid = merged[[percentile_col, actual_col]].dropna()
    if valid.empty:
        return {"n": 0, "below_pct": None, "above_pct": None}
    
    n = len(valid)
    below = (valid[actual_col] < valid[percentile_col]).sum()
    above = (valid[actual_col] > valid[percentile_col]).sum()
    
    return {
        "n": n,
        "below_pct": float(below / n * 100),
        "above_pct": float(above / n * 100),
    }


def _compute_interval_coverage(merged: pd.DataFrame, low_col: str, high_col: str, actual_col: str = "dk_fpts_actual") -> dict:
    """Compute interval coverage (what % of actuals fall within the interval)."""
    valid = merged[[low_col, high_col, actual_col]].dropna()
    if valid.empty:
        return {"n": 0, "in_interval_pct": None, "below_low_pct": None, "above_high_pct": None}
    
    n = len(valid)
    in_interval = ((valid[actual_col] >= valid[low_col]) & (valid[actual_col] <= valid[high_col])).sum()
    below_low = (valid[actual_col] < valid[low_col]).sum()
    above_high = (valid[actual_col] > valid[high_col]).sum()
    
    return {
        "n": n,
        "in_interval_pct": float(in_interval / n * 100),
        "below_low_pct": float(below_low / n * 100),
        "above_high_pct": float(above_high / n * 100),
    }


@app.command()
def main(
    data_root: Optional[Path] = typer.Option(None, "--data-root", help="Data root."),
    output_path: Optional[Path] = typer.Option(None, "--output", help="Output JSON path."),
    min_minutes: float = typer.Option(5.0, "--min-minutes", help="Minimum actual minutes to include."),
) -> None:
    root = data_root or data_path()
    
    typer.echo("[calibration] Loading simulation projections...")
    sim_df = _load_sim_projections(root)
    typer.echo(f"[calibration] Loaded {len(sim_df):,} simulation rows from {sim_df['game_date'].nunique()} dates")
    
    # Get unique dates from simulations
    sim_dates = set(sim_df["game_date"].dt.normalize().unique())
    
    typer.echo("[calibration] Loading actuals...")
    actuals_df = _load_actuals(root, list(sim_dates))
    typer.echo(f"[calibration] Loaded {len(actuals_df):,} actual rows")
    
    if actuals_df.empty:
        typer.echo("[calibration] No matching actuals found!", err=True)
        raise typer.Exit(1)
    
    # Prepare for join
    sim_df["game_date"] = pd.to_datetime(sim_df["game_date"]).dt.normalize()
    actuals_df["game_date"] = pd.to_datetime(actuals_df["game_date"]).dt.normalize()
    
    for col in ["game_id", "team_id", "player_id"]:
        sim_df[col] = pd.to_numeric(sim_df[col], errors="coerce")
        actuals_df[col] = pd.to_numeric(actuals_df[col], errors="coerce")
    
    # Join on game_date, game_id, team_id, player_id
    merged = pd.merge(
        sim_df,
        actuals_df[["game_date", "game_id", "team_id", "player_id", "dk_fpts_actual", "minutes_actual"]],
        on=["game_date", "game_id", "team_id", "player_id"],
        how="inner",
    )
    
    typer.echo(f"[calibration] Merged: {len(merged):,} rows")
    
    # Filter by minimum minutes
    if min_minutes > 0:
        merged = merged[merged["minutes_actual"] >= min_minutes]
        typer.echo(f"[calibration] After min_minutes={min_minutes} filter: {len(merged):,} rows")
    
    if merged.empty:
        typer.echo("[calibration] No matched rows after filtering!", err=True)
        raise typer.Exit(1)
    
    # Compute coverage for each percentile
    percentile_cols = {
        "p05": "dk_fpts_p05",
        "p10": "dk_fpts_p10",
        "p25": "dk_fpts_p25",
        "p50": "dk_fpts_p50",
        "p75": "dk_fpts_p75",
        "p95": "dk_fpts_p95",
    }
    
    results = {
        "n_rows": len(merged),
        "n_dates": int(merged["game_date"].nunique()),
        "percentile_coverage": {},
        "interval_coverage": {},
    }
    
    typer.echo("\n=== Percentile Coverage ===")
    typer.echo("(Expected: p10 should have ~10% below, p50 should have ~50% below, etc.)\n")
    
    for name, col in percentile_cols.items():
        if col not in merged.columns:
            typer.echo(f"  {name}: column {col} not found")
            continue
        
        coverage = _compute_coverage(merged, col)
        results["percentile_coverage"][name] = coverage
        expected_below = int(name.replace("p", ""))
        actual_below = coverage["below_pct"]
        diff = actual_below - expected_below if actual_below else None
        
        status = ""
        if diff is not None:
            if abs(diff) <= 3:
                status = "✓ well calibrated"
            elif diff < -3:
                status = "⚠ conservative (tails too narrow)"
            else:
                status = "⚠ aggressive (tails too wide)"
        
        typer.echo(f"  {name}: {actual_below:.1f}% below (expected {expected_below}%) {status}")
    
    # Compute interval coverage
    typer.echo("\n=== Interval Coverage ===")
    intervals = [
        ("p10-p95", "dk_fpts_p10", "dk_fpts_p95", 85),  # Expected 85% inside
        ("p25-p75", "dk_fpts_p25", "dk_fpts_p75", 50),  # Expected 50% inside
        ("p05-p95", "dk_fpts_p05", "dk_fpts_p95", 90),  # Expected 90% inside
    ]
    
    for name, low_col, high_col, expected_in in intervals:
        if low_col not in merged.columns or high_col not in merged.columns:
            continue
        
        interval_cov = _compute_interval_coverage(merged, low_col, high_col)
        results["interval_coverage"][name] = interval_cov
        actual_in = interval_cov["in_interval_pct"]
        diff = actual_in - expected_in if actual_in else None
        
        status = ""
        if diff is not None:
            if abs(diff) <= 5:
                status = "✓ well calibrated"
            elif diff > 5:
                status = "⚠ conservative (intervals too wide)"
            else:
                status = "⚠ aggressive (intervals too narrow)"
        
        below_pct = interval_cov["below_low_pct"]
        above_pct = interval_cov["above_high_pct"]
        typer.echo(f"  {name}: {actual_in:.1f}% inside (expected {expected_in}%), {below_pct:.1f}% below, {above_pct:.1f}% above {status}")
    
    # Stratify by minutes bucket
    typer.echo("\n=== Coverage by Minutes Bucket ===")
    merged["minutes_bucket"] = pd.cut(
        merged["dk_fpts_mean"],
        bins=[0, 15, 25, 35, 45, 100],
        labels=["0-15", "15-25", "25-35", "35-45", "45+"],
    )
    
    bucket_results = {}
    for bucket, group in merged.groupby("minutes_bucket", observed=True):
        if len(group) < 20:
            continue
        p10_cov = _compute_coverage(group, "dk_fpts_p10")
        p95_cov = _compute_coverage(group, "dk_fpts_p95") if "dk_fpts_p95" in group.columns else {"above_pct": None}
        bucket_results[str(bucket)] = {
            "n": len(group),
            "p10_below_pct": p10_cov["below_pct"],
            "p95_above_pct": 100 - (p95_cov.get("below_pct") or 0) if p95_cov.get("below_pct") else None,
        }
        typer.echo(
            f"  FPTS mean {bucket}: n={len(group):,}, "
            f"p10 has {p10_cov['below_pct']:.1f}% below (exp 10%), "
            f"p95 has {100 - (p95_cov.get('below_pct') or 0):.1f}% above (exp 5%)"
        )
    
    results["by_fpts_bucket"] = bucket_results
    
    # Save results
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        typer.echo(f"\n[calibration] Results saved to {output_path}")
    else:
        default_out = root / "artifacts" / "sim_v2" / "calibration_results.json"
        default_out.parent.mkdir(parents=True, exist_ok=True)
        default_out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        typer.echo(f"\n[calibration] Results saved to {default_out}")


if __name__ == "__main__":
    app()
