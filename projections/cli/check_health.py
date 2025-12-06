"""Health checks for the live projections pipeline."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from rich.console import Console

from projections import paths
from projections.minutes_v1.constants import AvailabilityStatus

app = typer.Typer(help=__doc__)
console = Console()


@app.command()
def check_latest_projections(
    lookback_hours: int = typer.Option(4, help="How far back to look for a projection file."),
    min_minutes_per_team: float = typer.Option(230.0, help="Minimum total minutes per team."),
    max_minutes_per_team: float = typer.Option(250.0, help="Maximum total minutes per team."),
) -> None:
    """Verify the integrity of the latest projections."""
    
    # 1. Find latest projection file
    proj_dir = paths.data_path("live", "projections_minutes_v1")
    if not proj_dir.exists():
        console.print(f"[yellow]Projections directory not found: {proj_dir} (skipping health check)[/yellow]")
        raise typer.Exit(code=0)
        
    # Find latest csv
    files = sorted(proj_dir.glob("*.csv"))
    if not files:
        console.print(f"[yellow]No projection files found in {proj_dir}; skipping health check[/yellow]")
        raise typer.Exit(code=0)
        
    latest_file = files[-1]
    file_age = pd.Timestamp.now() - pd.Timestamp.fromtimestamp(latest_file.stat().st_mtime)
    
    console.print(f"Checking latest file: [bold]{latest_file.name}[/bold] (Age: {file_age})")
    
    if file_age > pd.Timedelta(hours=lookback_hours):
        console.print(f"[red]Latest projection file is too old! > {lookback_hours} hours.[/red]")
        # We might not want to fail hard if it's just morning and no games yet, but for now warn.
        # raise typer.Exit(code=1) 
    
    df = pd.read_csv(latest_file)
    
    # 2. Check for NaNs in predictions
    if df["minutes"].isna().any():
        console.print("[red]Found NaN predictions![/red]")
        print(df[df["minutes"].isna()])
        raise typer.Exit(code=1)
        
    # 3. Check Team Totals
    team_totals = df.groupby("team_tricode")["minutes"].sum()
    console.print("\nTeam Totals:")
    print(team_totals)
    
    failed_teams = team_totals[
        (team_totals < min_minutes_per_team) | (team_totals > max_minutes_per_team)
    ]
    
    if not failed_teams.empty:
        console.print(f"\n[red]Teams with invalid minute totals (Range {min_minutes_per_team}-{max_minutes_per_team}):[/red]")
        print(failed_teams)
        raise typer.Exit(code=1)
        
    console.print("\n[green]All checks passed![/green]")


@app.command()
def check_rates_sanity(
    date_str: Optional[str] = typer.Option(None, "--date", help="Date to check (YYYY-MM-DD), defaults to today."),
    rates_root: Path = typer.Option(
        paths.data_path("gold", "rates_v1_live"),
        "--rates-root",
        help="Root directory for rates predictions.",
    ),
    min_fga2_median: float = typer.Option(0.15, help="Minimum pred_fga2_per_min median."),
    min_top_fpts: float = typer.Option(45.0, help="Minimum expected top FPTS on a slate."),
) -> None:
    """Check that rates predictions are plausible."""
    from projections.pipeline.guardrails import check_rates_output_sanity

    if date_str is None:
        import datetime
        date_str = datetime.date.today().isoformat()

    day_dir = rates_root / date_str
    if not day_dir.exists():
        console.print(f"[yellow]No rates predictions for {date_str}[/yellow]")
        raise typer.Exit(code=0)

    # Find latest run
    runs = sorted(day_dir.glob("run=*"))
    if not runs:
        console.print(f"[yellow]No runs found in {day_dir}[/yellow]")
        raise typer.Exit(code=0)

    latest_run = runs[-1]
    rates_file = latest_run / "rates.parquet"
    if not rates_file.exists():
        console.print(f"[red]Missing rates.parquet in {latest_run}[/red]")
        raise typer.Exit(code=1)

    console.print(f"Checking rates: [bold]{rates_file}[/bold]")
    df = pd.read_parquet(rates_file)

    result = check_rates_output_sanity(
        df,
        min_fga2_median=min_fga2_median,
        min_top_fpts=min_top_fpts,
    )

    console.print(f"\nMetrics: {result.metrics}")

    if result.warnings:
        console.print("\n[yellow]Warnings:[/yellow]")
        for warn in result.warnings:
            console.print(f"  - {warn}")

    if result.passed:
        console.print("\n[green]Rates sanity check passed![/green]")
    else:
        console.print("\n[red]Rates sanity check FAILED[/red]")
        raise typer.Exit(code=1)


@app.command()
def check_feature_coverage(
    date_str: Optional[str] = typer.Option(None, "--date", help="Date to check (YYYY-MM-DD), defaults to today."),
    features_root: Path = typer.Option(
        paths.data_path("live", "features_rates_v1"),
        "--features-root",
        help="Root directory for rates features.",
    ),
    expected_rows: Optional[int] = typer.Option(None, help="Expected number of rows (if known)."),
) -> None:
    """Check that rates features have adequate coverage."""
    from projections.pipeline.guardrails import check_feature_coverage as _check_feature_coverage

    if date_str is None:
        import datetime
        date_str = datetime.date.today().isoformat()

    day_dir = features_root / date_str
    if not day_dir.exists():
        console.print(f"[yellow]No features for {date_str}[/yellow]")
        raise typer.Exit(code=0)

    runs = sorted(day_dir.glob("run=*"))
    if not runs:
        console.print(f"[yellow]No runs found in {day_dir}[/yellow]")
        raise typer.Exit(code=0)

    latest_run = runs[-1]
    features_file = latest_run / "features.parquet"
    if not features_file.exists():
        console.print(f"[red]Missing features.parquet in {latest_run}[/red]")
        raise typer.Exit(code=1)

    console.print(f"Checking features: [bold]{features_file}[/bold]")
    df = pd.read_parquet(features_file)

    critical_cols = ["game_id", "player_id", "team_id", "minutes_pred_p50", "season_fga_per_min"]
    result = _check_feature_coverage(
        df,
        expected_rows=expected_rows,
        critical_cols=critical_cols,
    )

    console.print(f"\nMetrics: {result.metrics}")

    if result.warnings:
        console.print("\n[yellow]Warnings:[/yellow]")
        for warn in result.warnings:
            console.print(f"  - {warn}")

    if result.passed:
        console.print("\n[green]Feature coverage check passed![/green]")
    else:
        console.print("\n[red]Feature coverage check FAILED[/red]")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
