"""CLI for scraping NBA play-by-play data.

Usage:
    # Scrape a single game
    python -m projections.cli.scrape_pbp game 0022400510 --out pbp_game.parquet
    
    # Scrape all games for a date range
    python -m projections.cli.scrape_pbp daily --start 2025-01-01 --end 2025-01-02 --out pbp_daily.parquet
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.progress import track

from scrapers.nba_playbyplay import NbaPlayByPlayScraper, GamePlayByPlay

console = Console()
app = typer.Typer(help="Scrape NBA play-by-play data using the nba_api v3 endpoint.")


def _parse_date_field(field: str, value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise typer.BadParameter(
            f"Invalid date for {field}: {value}", param_hint=field
        ) from exc


@app.command()
def game(
    game_id: str = typer.Argument(..., help="NBA game ID (10-digit string)."),
    out: Optional[Path] = typer.Option(
        None,
        "--out",
        "-o",
        help="Output path (.parquet, .csv, or .json).",
    ),
    start_period: int = typer.Option(
        1, "--start-period", help="First period to include."
    ),
    end_period: int = typer.Option(
        10, "--end-period", help="Last period to include (10 covers OT games)."
    ),
) -> None:
    """Scrape play-by-play data for a single game."""
    
    scraper = NbaPlayByPlayScraper()
    
    try:
        pbp = scraper.fetch_game_pbp(
            game_id,
            start_period=start_period,
            end_period=end_period,
        )
    except RuntimeError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)
    
    console.print(f"Fetched [green]{len(pbp.actions)}[/green] actions for game {game_id}")
    
    if out:
        df = pbp.to_dataframe()
        _save_output(df, out)
    else:
        # Pretty print a summary
        df = pbp.to_dataframe()
        if df.empty:
            console.print("No actions found.")
            return
        
        console.print(f"\nSample actions:")
        for _, row in df.head(10).iterrows():
            console.print(f"  {row['period']}Q {row['clock']}: {row['description']}")


@app.command()
def daily(
    start: str = typer.Option(..., "--start", help="Start date (YYYY-MM-DD)."),
    end: str = typer.Option(..., "--end", help="End date inclusive (YYYY-MM-DD)."),
    out: Optional[Path] = typer.Option(
        None,
        "--out",
        "-o",
        help="Output path (.parquet, .csv, or .json).",
    ),
    completed_only: bool = typer.Option(
        True,
        "--completed-only/--all-games",
        help="Only scrape completed games.",
    ),
    season: Optional[str] = typer.Option(
        None,
        "--season",
        help="Season string (e.g., '2024-25'). Auto-detected if not specified.",
    ),
) -> None:
    """Scrape play-by-play data for all games in a date range."""
    
    start_date = _parse_date_field("start", start)
    end_date = _parse_date_field("end", end)
    
    if end_date < start_date:
        raise typer.BadParameter(
            "end date must be >= start date", param_hint="end"
        )
    
    scraper = NbaPlayByPlayScraper()
    
    # Build date list
    dates: List[date] = []
    cursor = start_date
    while cursor <= end_date:
        dates.append(cursor)
        cursor += timedelta(days=1)
    
    all_games: List[GamePlayByPlay] = []
    
    for target_date in track(dates, description="Scraping play-by-play..."):
        console.print(f"\n[dim]Processing {target_date}...[/dim]")
        try:
            games = scraper.fetch_daily_pbp(
                target_date,
                season=season,
                completed_only=completed_only,
            )
            all_games.extend(games)
            console.print(f"  Found {len(games)} games")
        except Exception as exc:
            console.print(f"[yellow]Warning: Failed to fetch {target_date}: {exc}[/yellow]")
            continue
    
    total_actions = sum(len(g.actions) for g in all_games)
    console.print(
        f"\n[green]Scraped {len(all_games)} games with {total_actions:,} total actions[/green]"
    )
    
    if out and all_games:
        import pandas as pd
        dfs = [g.to_dataframe() for g in all_games]
        combined = pd.concat(dfs, ignore_index=True)
        _save_output(combined, out)
    elif not all_games:
        console.print("[yellow]No games found for the specified range.[/yellow]")


def _save_output(df, out: Path) -> None:
    """Save DataFrame to the specified output format."""
    import pandas as pd
    
    out.parent.mkdir(parents=True, exist_ok=True)
    suffix = out.suffix.lower()
    
    if suffix == ".parquet":
        df.to_parquet(out, index=False)
    elif suffix == ".csv":
        df.to_csv(out, index=False)
    elif suffix == ".json":
        df.to_json(out, orient="records", indent=2)
    else:
        console.print(f"[yellow]Unknown format {suffix}, defaulting to parquet[/yellow]")
        out = out.with_suffix(".parquet")
        df.to_parquet(out, index=False)
    
    console.print(f"Saved to [blue]{out}[/blue]")


def main() -> None:
    """Entry-point for `python -m projections.cli.scrape_pbp`."""
    app()


if __name__ == "__main__":
    main()
