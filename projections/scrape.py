from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import typer
from rich.console import Console
from rich.progress import track

from scrapers.oddstrader import EventOdds, MarketLine, OddstraderScraper

console = Console()
app = typer.Typer(help="Scrape external betting feeds and persist normalized odds.")


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


def main() -> None:
    """Entry-point for `python -m projections.scrape`."""

    app()


if __name__ == "__main__":
    main()
