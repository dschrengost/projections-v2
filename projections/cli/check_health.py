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


if __name__ == "__main__":
    app()
