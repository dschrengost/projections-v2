"""CLI for scraping and archiving NBA player props from RotoWire.

Usage:
    uv run python -m projections.cli.scrape_props --date 2025-12-29
    uv run python -m projections.cli.scrape_props  # defaults to today
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import typer

app = typer.Typer(help="Scrape NBA player props from RotoWire")


@app.command()
def scrape(
    game_date: str = typer.Option(
        None,
        "--date",
        "-d",
        help="Game date (YYYY-MM-DD). Defaults to today.",
    ),
    output_format: str = typer.Option(
        "long",
        "--format",
        "-f",
        help="Output format: 'long' (one row per line) or 'wide' (one row per player)",
    ),
    save: bool = typer.Option(
        True,
        "--save/--no-save",
        help="Save to bronze layer as parquet",
    ),
    show_sample: bool = typer.Option(
        True,
        "--sample/--no-sample",
        help="Print sample of scraped data",
    ),
) -> None:
    """Scrape player props and optionally save to bronze layer."""
    from scrapers.rotowire_props import (
        save_props_bronze,
        scrape_props_wide,
        scrape_rotowire_props,
    )

    # Parse date
    if game_date:
        target_date = date.fromisoformat(game_date)
    else:
        target_date = date.today()

    typer.echo(f"Scraping props for {target_date}...")

    # Scrape
    if output_format == "wide":
        df = scrape_props_wide()
    else:
        df = scrape_rotowire_props()

    if df.empty:
        typer.echo("No props data found", err=True)
        raise typer.Exit(1)

    n_players = df["player_name"].nunique() if "player_name" in df.columns else len(df)
    typer.echo(f"Found {len(df)} rows for {n_players} players")

    # Show sample
    if show_sample:
        typer.echo("\n=== Sample (DraftKings points) ===")
        if output_format == "wide":
            cols = ["player_name", "team", "opponent", "draftkings_pts", "draftkings_pts_over", "draftkings_pts_under"]
            cols = [c for c in cols if c in df.columns]
            sample = df[cols].head(10)
        else:
            sample = df[
                (df["book"] == "draftkings") & (df["prop_type"] == "pts")
            ][["player_name", "team", "opponent", "line", "over_odds", "under_odds"]].head(10)
        typer.echo(sample.to_string(index=False))

    # Save
    if save:
        data_root = Path(os.environ.get("PROJECTIONS_DATA_ROOT", "data"))
        # For saving, always use long format
        if output_format == "wide":
            save_df = scrape_rotowire_props()
        else:
            save_df = df
        out_path = save_props_bronze(save_df, target_date, data_root)
        typer.echo(f"\nSaved to: {out_path}")


@app.command()
def compare(
    game_date: str = typer.Option(
        None,
        "--date",
        "-d",
        help="Game date (YYYY-MM-DD). Defaults to today.",
    ),
    prop_type: str = typer.Option(
        "pts",
        "--prop",
        "-p",
        help="Prop type to compare (pts, reb, ast, etc.)",
    ),
    min_diff: float = typer.Option(
        0.5,
        "--min-diff",
        help="Minimum line difference to highlight",
    ),
) -> None:
    """Compare lines across sportsbooks to find discrepancies."""
    from scrapers.rotowire_props import scrape_rotowire_props

    if game_date:
        target_date = date.fromisoformat(game_date)
    else:
        target_date = date.today()

    typer.echo(f"Comparing {prop_type} lines for {target_date}...")

    df = scrape_rotowire_props()
    if df.empty:
        typer.echo("No props data found", err=True)
        raise typer.Exit(1)

    # Filter to prop type
    prop_df = df[df["prop_type"] == prop_type].copy()
    if prop_df.empty:
        typer.echo(f"No {prop_type} props found", err=True)
        raise typer.Exit(1)

    # Pivot to get lines by book
    pivot = prop_df.pivot_table(
        index=["player_name", "team"],
        columns="book",
        values="line",
        aggfunc="first",
    )

    # Find players with line differences
    pivot["max_line"] = pivot.max(axis=1)
    pivot["min_line"] = pivot.min(axis=1)
    pivot["diff"] = pivot["max_line"] - pivot["min_line"]

    # Filter and sort
    diff_df = pivot[pivot["diff"] >= min_diff].sort_values("diff", ascending=False)

    if diff_df.empty:
        typer.echo(f"No line differences >= {min_diff} found")
        return

    typer.echo(f"\n=== Players with line diff >= {min_diff} ({prop_type}) ===")
    typer.echo(diff_df.head(20).to_string())


if __name__ == "__main__":
    app()
