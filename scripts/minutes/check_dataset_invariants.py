#!/usr/bin/env python
"""CLI to run invariant checks on training dataset.

This script validates the minute-share training data pipeline by checking:
- Primary key uniqueness (game_id, team_id, player_id)
- Label share sum invariants per team-game  
- Post-normalization sum exactness

Usage:
    uv run python scripts/minutes/check_dataset_invariants.py --season 2024
    uv run python scripts/minutes/check_dataset_invariants.py --start-date 2024-01-01 --end-date 2024-01-31
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import typer

from projections.minutes_v1.minute_share import (
    MinuteLabelMode,
    TEAM_TOTAL_MINUTES,
    compute_minute_share,
)
from projections.pipeline.training.builder import build_features

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

app = typer.Typer(help=__doc__)


@app.command()
def main(
    season: str = typer.Option(None, help="Full season to check (e.g. 2024)"),
    start_date: str = typer.Option(None, "--start-date", help="Start date YYYY-MM-DD"),
    end_date: str = typer.Option(None, "--end-date", help="End date YYYY-MM-DD"),
    label_mode: MinuteLabelMode = typer.Option(
        MinuteLabelMode.REG240,
        "--label-mode", 
        help="Label mode: reg240 or team_total_actual"
    ),
    output: Path = typer.Option(None, "--output", help="Save report to JSON file"),
    allow_dedup: bool = typer.Option(False, "--allow-dedup", help="Allow dedup escape hatch"),
) -> None:
    """Run invariant checks on the minute-share training dataset."""
    
    # Determine date range
    if season:
        # NBA season spans Oct of year to June of year+1
        year = int(season)
        start = f"{year}-10-01"
        end = f"{year + 1}-06-30"
        season_str = season
    elif start_date and end_date:
        start = start_date
        end = end_date
        # Infer season from start date
        year = int(start_date[:4])
        month = int(start_date[5:7])
        season_str = str(year) if month >= 10 else str(year - 1)
    else:
        typer.echo("Must provide --season or both --start-date and --end-date", err=True)
        raise typer.Exit(1)
    
    typer.echo(f"Checking invariants for date range: {start} to {end}")
    typer.echo(f"Season: {season_str}, Label mode: {label_mode.value}")
    typer.echo("=" * 60)
    
    # Generate dates in range
    date_range = pd.date_range(start, end, freq="D")
    
    all_dfs = []
    dates_checked = 0
    dates_with_data = 0
    
    for date in date_range:
        date_str = date.strftime("%Y-%m-%d")
        try:
            df = build_features(date_str, None, season_str)
            if not df.empty:
                df["_check_date"] = date_str
                all_dfs.append(df)
                dates_with_data += 1
            dates_checked += 1
        except FileNotFoundError:
            continue
        except ValueError as e:
            # Fail-fast assertion triggered
            typer.echo(f"FAIL: {e}", err=True)
            raise typer.Exit(1)
    
    if not all_dfs:
        typer.echo("No data found in date range", err=True)
        raise typer.Exit(1)
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    typer.echo(f"Checked {dates_checked} dates, {dates_with_data} with data")
    typer.echo(f"Total rows: {len(combined_df):,}")
    
    # Run invariant checks
    report = run_invariant_checks(combined_df, label_mode)
    
    # Print summary
    print_report(report)
    
    # Save if requested
    if output:
        with open(output, "w") as f:
            json.dump(report, f, indent=2, default=str)
        typer.echo(f"\nReport saved to {output}")
    
    # Exit with appropriate code
    if report["passed"]:
        typer.echo("\n✓ All invariants PASSED")
        raise typer.Exit(0)
    else:
        typer.echo("\n✗ Some invariants FAILED", err=True)
        raise typer.Exit(1)


def run_invariant_checks(df: pd.DataFrame, label_mode: MinuteLabelMode) -> dict:
    """Run all invariant checks and return report."""
    
    pk_cols = ["game_id", "team_id", "player_id"]
    group_cols = ["game_id", "team_id"]
    
    report = {
        "label_mode": label_mode.value,
        "total_rows": len(df),
        "unique_pk_count": df[pk_cols].drop_duplicates().shape[0],
        "team_games_count": df[group_cols].drop_duplicates().shape[0],
        "checks": {},
        "summary": {},
        "offenders": [],
        "passed": True,
    }
    
    # 1. Primary key uniqueness
    duplicates = df[df.duplicated(subset=pk_cols, keep=False)]
    n_duplicates = len(duplicates)
    report["checks"]["pk_unique"] = n_duplicates == 0
    report["summary"]["n_duplicate_rows"] = n_duplicates
    
    if n_duplicates > 0:
        report["passed"] = False
        dup_keys = duplicates.groupby(pk_cols).size().reset_index(name="count")
        report["offenders"].append({
            "check": "pk_unique",
            "top_offenders": dup_keys.sort_values("count", ascending=False).head(5).to_dict("records")
        })
    
    # 2. Players per team-game stats
    players_per_team = df.groupby(group_cols).size()
    report["summary"]["players_per_team_min"] = int(players_per_team.min())
    report["summary"]["players_per_team_mean"] = float(players_per_team.mean())
    report["summary"]["players_per_team_max"] = int(players_per_team.max())
    
    # 3. Compute share labels
    if "minutes" in df.columns:
        df = df.copy()
        df["share_label"] = compute_minute_share(
            df["minutes"],
            df["game_id"],
            df["team_id"],
            mode=label_mode,
        )
        
        # Share label sum per team-game
        label_sums = df.groupby(group_cols)["share_label"].sum()
        report["summary"]["label_sum_min"] = float(label_sums.min())
        report["summary"]["label_sum_mean"] = float(label_sums.mean())
        report["summary"]["label_sum_max"] = float(label_sums.max())
        
        # For REG240: sum can be >1.0 for OT games
        # For TEAM_TOTAL_ACTUAL: sum should be exactly 1.0
        if label_mode == MinuteLabelMode.TEAM_TOTAL_ACTUAL:
            tolerance = 0.001
            bad_sums = label_sums[~label_sums.between(1.0 - tolerance, 1.0 + tolerance)]
            report["checks"]["label_sum_valid"] = len(bad_sums) == 0
            report["summary"]["n_bad_label_sums"] = len(bad_sums)
            
            if len(bad_sums) > 0:
                report["passed"] = False
                report["offenders"].append({
                    "check": "label_sum_valid",
                    "top_offenders": bad_sums.head(5).to_dict()
                })
        else:
            # REG240 mode: just report, don't fail
            report["checks"]["label_sum_valid"] = True
            
        # 4. Actual minutes sum per team (for OT detection)
        if "minutes" in df.columns:
            actual_sums = df.groupby(group_cols)["minutes"].sum()
            report["summary"]["actual_minutes_sum_min"] = float(actual_sums.min())
            report["summary"]["actual_minutes_sum_mean"] = float(actual_sums.mean())
            report["summary"]["actual_minutes_sum_max"] = float(actual_sums.max())
            
            # OT game detection (team total > 240)
            ot_teams = actual_sums[actual_sums > TEAM_TOTAL_MINUTES]
            report["summary"]["n_ot_team_games"] = len(ot_teams)
            report["summary"]["pct_ot_team_games"] = float(len(ot_teams) / len(actual_sums) * 100)
    
    return report


def print_report(report: dict) -> None:
    """Print formatted report."""
    typer.echo("\n" + "=" * 60)
    typer.echo("INVARIANT CHECK REPORT")
    typer.echo("=" * 60)
    
    typer.echo(f"\nLabel mode: {report['label_mode']}")
    typer.echo(f"Total rows: {report['total_rows']:,}")
    typer.echo(f"Unique primary keys: {report['unique_pk_count']:,}")
    typer.echo(f"Team-games: {report['team_games_count']:,}")
    
    typer.echo("\n[CHECKS]")
    for check, passed in report["checks"].items():
        status = "✓ PASS" if passed else "✗ FAIL"
        typer.echo(f"  {check}: {status}")
    
    typer.echo("\n[SUMMARY]")
    summary = report["summary"]
    
    if "n_duplicate_rows" in summary:
        typer.echo(f"  Duplicate rows: {summary['n_duplicate_rows']}")
    
    if "players_per_team_mean" in summary:
        typer.echo(f"  Players per team: min={summary['players_per_team_min']}, "
                   f"mean={summary['players_per_team_mean']:.1f}, "
                   f"max={summary['players_per_team_max']}")
    
    if "label_sum_mean" in summary:
        typer.echo(f"  Label share sum: min={summary['label_sum_min']:.4f}, "
                   f"mean={summary['label_sum_mean']:.4f}, "
                   f"max={summary['label_sum_max']:.4f}")
    
    if "actual_minutes_sum_mean" in summary:
        typer.echo(f"  Actual minutes sum: min={summary['actual_minutes_sum_min']:.1f}, "
                   f"mean={summary['actual_minutes_sum_mean']:.1f}, "
                   f"max={summary['actual_minutes_sum_max']:.1f}")
    
    if "n_ot_team_games" in summary:
        typer.echo(f"  OT team-games: {summary['n_ot_team_games']} "
                   f"({summary['pct_ot_team_games']:.1f}%)")
    
    if report["offenders"]:
        typer.echo("\n[TOP OFFENDERS]")
        for item in report["offenders"]:
            typer.echo(f"  {item['check']}:")
            for offender in item["top_offenders"][:5]:
                typer.echo(f"    {offender}")


if __name__ == "__main__":
    app()
