from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import typer
from rich.console import Console
from rich.progress import track

from projections import paths
from projections.etl import daily_lineups as daily_lineups_etl
from scrapers.oddstrader import EventOdds, MarketLine, OddstraderScraper

console = Console()
app = typer.Typer(help="Scrape NBA data feeds (odds, daily lineups) for inspection or ETL backfills.")


def _serialize_market_line(line: MarketLine) -> Dict[str, object]:
    return {
        "market": line.market,
        "selection": line.selection,
        "price": line.price,
        "point": line.point,
        "book": line.book,
        "updated_at": line.updated_at.isoformat(),
    }


def _serialize_event(event: EventOdds) -> Dict[str, object]:
    return {
        "event_id": event.event_id,
        "scheduled": event.scheduled.isoformat(),
        "home_team": event.home_team,
        "away_team": event.away_team,
        "markets": {
            market: {
                selection: _serialize_market_line(line)
                for selection, line in selections.items()
            }
            for market, selections in event.markets.items()
        },
    }


def _parse_date_field(field: str, value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise typer.BadParameter(
            f"Invalid date for {field}: {value}", param_hint=field
        ) from exc


@app.command()
def oddstrader(
    start: str = typer.Option(..., "--start", help="Start date inclusive (YYYY-MM-DD)."),
    end: str = typer.Option(..., "--end", help="End date inclusive (YYYY-MM-DD)."),
    out: Optional[Path] = typer.Option(
        None,
        "--out",
        "-o",
        help="Optional path to write JSON output (list of events).",
    ),
    pretty: bool = typer.Option(
        False,
        "--pretty/--compact",
        help="Pretty-print JSON output when writing to --out.",
    ),
) -> None:
    """Fetch odds from Oddstrader across the requested date range."""

    start_date = _parse_date_field("start", start)
    end_date = _parse_date_field("end", end)
    if end_date < start_date:
        raise typer.BadParameter(
            "end date must be greater than or equal to start", param_hint="end"
        )

    scraper = OddstraderScraper()
    dates: List[date] = []
    cursor = start_date
    while cursor <= end_date:
        dates.append(cursor)
        cursor += timedelta(days=1)

    events: List[EventOdds] = []
    for target_date in track(dates, description="Scraping Oddstrader"):
        events.extend(scraper.fetch_daily_odds(target_date))

    if out:
        payload: List[Dict[str, object]] = [_serialize_event(event) for event in events]
        text = json.dumps(payload, indent=2 if pretty else None)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + ("\n" if pretty else ""), encoding="utf-8")
        console.print(f"Wrote {len(events)} events to {out}")
    else:
        if not events:
            console.print("No events found for the requested range.")
            return
        for event in events:
            console.print(
                f"{event.scheduled.date()}: {event.away_team} @ {event.home_team} "
                f"({', '.join(sorted(event.markets.keys()))})"
            )


@app.command("nba-daily-lineups")
def nba_daily_lineups(
    start: str = typer.Option(..., "--start", help="Start date inclusive (YYYY-MM-DD)."),
    end: str = typer.Option(..., "--end", help="End date inclusive (YYYY-MM-DD)."),
    season: int = typer.Option(..., "--season", help="Season start year for partitioning (e.g., 2025)."),
    data_root: Path = typer.Option(
        paths.get_data_root(),
        "--data-root",
        help="Base data directory (defaults to PROJECTIONS_DATA_ROOT or ./data).",
    ),
    timeout: float = typer.Option(10.0, "--timeout", help="stats.nba.com HTTP timeout (seconds)."),
    out: Optional[Path] = typer.Option(
        None,
        "--out",
        "-o",
        help="Optional JSON dump of the scraped payloads (list of {date, payload}).",
    ),
    pretty: bool = typer.Option(
        False,
        "--pretty/--compact",
        help="Pretty-print JSON output when writing to --out.",
    ),
) -> None:
    """Scrape NBA.com daily lineups, run the ETL, and optionally dump raw JSON."""

    start_date = _parse_date_field("start", start)
    end_date = _parse_date_field("end", end)
    if end_date < start_date:
        raise typer.BadParameter(
            "end date must be greater than or equal to start", param_hint="end"
        )
    data_root = data_root.resolve()
    capture_payloads = out is not None
    results = daily_lineups_etl.ingest_range(
        start_day=pd.Timestamp(start_date),
        end_day=pd.Timestamp(end_date),
        season=season,
        data_root=data_root,
        timeout=timeout,
        capture_payloads=capture_payloads,
    )
    silver_rows = sum(result.silver_rows for result in results)
    console.print(
        f"Wrote {silver_rows} lineup rows across {len(results)} day(s) -> "
        f"{data_root / 'silver' / 'nba_daily_lineups'}"
    )
    if out:
        payload = [
            {"date": result.target_date.isoformat(), "payload": result.payload or {}}
            for result in results
        ]
        text = json.dumps(payload, indent=2 if pretty else None)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + ("\n" if pretty else ""), encoding="utf-8")
        console.print(f"Wrote {len(payload)} payload(s) to {out}")


def main() -> None:
    """Entry-point for `python -m projections.scrape`."""

    app()


if __name__ == "__main__":
    main()
