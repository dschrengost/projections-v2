"""Scrape NBA.com daily lineups and persist bronze/silver parquet outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import List

import pandas as pd
import typer

from projections import paths
from projections.etl import storage
from scrapers.nba_daily_lineups import NbaDailyLineupsScraper, normalize_daily_lineups

app = typer.Typer(help=__doc__)


@dataclass
class DailyLineupIngestResult:
    target_date: date
    bronze_rows: int
    silver_rows: int
    bronze_path: Path
    silver_path: Path
    payload: dict | None = None


def _default_silver_path(root: Path, season: int, target_day: date) -> Path:
    return (
        root
        / f"season={season}"
        / f"date={target_day.isoformat()}"
        / "lineups.parquet"
    )


def _build_bronze_frame(
    *,
    payload: dict,
    target_day: date,
    season: int,
    source_url: str,
) -> pd.DataFrame:
    ingested_ts = pd.Timestamp.utcnow()
    payload_json = json.dumps(payload, sort_keys=True)
    records = [
        {
            "date": pd.Timestamp(target_day),
            "season_start": season,
            "payload_json": payload_json,
            "source_url": source_url,
            "ingested_ts": ingested_ts,
            "source": "stats.nba.com/daily_lineups",
        }
    ]
    columns = ["date", "season_start", "payload_json", "source_url", "ingested_ts", "source"]
    return pd.DataFrame.from_records(records, columns=columns)


def ingest_range(
    *,
    start_day: pd.Timestamp,
    end_day: pd.Timestamp,
    season: int,
    data_root: Path,
    timeout: float,
    bronze_root: Path | None = None,
    silver_root: Path | None = None,
    capture_payloads: bool = False,
) -> List[DailyLineupIngestResult]:
    start_norm = start_day.normalize()
    end_norm = end_day.normalize()
    if end_norm < start_norm:
        raise typer.BadParameter("--end must be on/after --start.")

    data_root = data_root.resolve()
    base_bronze = (bronze_root or storage.default_bronze_root("daily_lineups", data_root)).resolve()
    base_silver = (silver_root or (data_root / "silver" / "nba_daily_lineups")).resolve()
    scraper = NbaDailyLineupsScraper(timeout=timeout)

    results: List[DailyLineupIngestResult] = []
    for cursor in storage.iter_days(start_norm, end_norm):
        target = cursor.date()
        typer.echo(f"[daily_lineups] fetching {target} (season={season})")
        payload = scraper.fetch_daily_lineups(target)
        source_url = scraper.url_template.format(date=target.strftime("%Y%m%d"))
        bronze_df = _build_bronze_frame(
            payload=payload,
            target_day=target,
            season=season,
            source_url=source_url,
        )
        silver_df = normalize_daily_lineups(
            payload,
            target_date=target,
            season_start=season,
        )
        bronze_result = storage.write_bronze_partition(
            bronze_df,
            dataset="daily_lineups",
            data_root=data_root,
            season=season,
            target_date=target,
            bronze_root=base_bronze,
        )
        silver_path = _default_silver_path(base_silver, season, target)
        silver_path.parent.mkdir(parents=True, exist_ok=True)
        silver_df.to_parquet(silver_path, index=False)
        typer.echo(
            f"[daily_lineups] wrote bronze_rows={len(bronze_df)} -> {bronze_result.path} "
            f"and silver_rows={len(silver_df)} -> {silver_path}"
        )
        results.append(
            DailyLineupIngestResult(
                target_date=target,
                bronze_rows=len(bronze_df),
                silver_rows=len(silver_df),
                bronze_path=bronze_result.path,
                silver_path=silver_path,
                payload=payload if capture_payloads else None,
            )
        )
    typer.echo(
        f"[daily_lineups] completed {len(results)} day(s); "
        f"silver root={base_silver}"
    )
    return results


@app.command()
def run(
    start: datetime = typer.Option(..., "--start", help="Start date inclusive (YYYY-MM-DD)."),
    end: datetime = typer.Option(..., "--end", help="End date inclusive (YYYY-MM-DD)."),
    season: int = typer.Option(..., "--season", help="Season start year (e.g., 2025)."),
    data_root: Path = typer.Option(
        paths.get_data_root(),
        "--data-root",
        help="Base data directory (defaults to PROJECTIONS_DATA_ROOT or ./data).",
    ),
    bronze_root: Path | None = typer.Option(None, help="Optional override for bronze output root."),
    silver_root: Path | None = typer.Option(None, help="Optional override for silver output root."),
    timeout: float = typer.Option(10.0, "--timeout", help="HTTP timeout for stats.nba.com (seconds)."),
) -> None:
    """Typer entry point for scraping daily lineups over a date range."""
    start_day = pd.Timestamp(start).normalize()
    end_day = pd.Timestamp(end).normalize()
    ingest_range(
        start_day=start_day,
        end_day=end_day,
        season=season,
        data_root=data_root,
        timeout=timeout,
        bronze_root=bronze_root,
        silver_root=silver_root,
    )


if __name__ == "__main__":  # pragma: no cover
    app()
