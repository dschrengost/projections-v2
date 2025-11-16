"""Ingest Oddstrader odds snapshots into bronze/silver parquet outputs."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import typer
from zoneinfo import ZoneInfo

from projections import paths
from projections.etl.common import load_schedule_data, month_slug
from projections.minutes_v1.schemas import (
    ODDS_RAW_SCHEMA,
    ODDS_SNAPSHOT_SCHEMA,
    enforce_schema,
    validate_with_pandera,
)
from projections.minutes_v1.snapshots import latest_pre_tip_snapshot
from projections.minutes_v1.smoke_dataset import TeamResolver
from scrapers.oddstrader import EventOdds, OddstraderScraper

app = typer.Typer(help=__doc__)
ODDSTRADER_TZ = ZoneInfo("America/New_York")


def _iter_dates(start: pd.Timestamp, end: pd.Timestamp) -> Iterable[pd.Timestamp]:
    current = start
    while current <= end:
        yield current
        current += pd.Timedelta(days=1)


def _build_odds_raw(
    events: List[EventOdds],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    resolver: TeamResolver,
) -> pd.DataFrame:
    ingested_ts = pd.Timestamp.now(tz="UTC")
    records: list[dict] = []
    for event in events:
        scheduled = pd.Timestamp(event.scheduled)
        if scheduled.tzinfo is None:
            scheduled = scheduled.tz_localize("UTC")
        else:
            scheduled = scheduled.tz_convert("UTC")
        event_day = scheduled.tz_convert(ODDSTRADER_TZ).tz_localize(None).normalize()
        if not (start <= event_day <= end):
            continue
        home_tri = resolver.resolve_tricode(event.home_team)
        away_tri = resolver.resolve_tricode(event.away_team)
        if not (home_tri and away_tri):
            continue
        game_id = resolver.lookup_game_id(
            event_day.strftime("%Y-%m-%d"), away_tri, home_tri, tip_ts=scheduled
        )
        if not game_id:
            continue
        markets = event.markets or {}
        spread = markets.get("spread", {})
        total = markets.get("total", {})
        home_spread = spread.get("home")
        over_total = total.get("over")
        timestamps: list[pd.Timestamp] = []
        for line in (home_spread, over_total):
            if line is None or line.updated_at is None:
                continue
            ts = pd.Timestamp(line.updated_at)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")
            timestamps.append(ts)
        if not timestamps:
            continue
        as_of_ts = min(max(timestamps), scheduled)
        records.append(
            {
                "game_id": int(game_id),
                "home_team_id": resolver.resolve_team_id(event.home_team),
                "away_team_id": resolver.resolve_team_id(event.away_team),
                "spread_home": home_spread.point if home_spread else None,
                "total": over_total.point if over_total else None,
                "book": (home_spread or over_total).book if (home_spread or over_total) else None,
                "market": "spread_total",
                "as_of_ts": as_of_ts,
                "ingested_ts": ingested_ts,
                "source": "oddstrader",
            }
        )
    columns = list(ODDS_RAW_SCHEMA.columns)
    return pd.DataFrame(records, columns=columns)


def _build_odds_snapshot(odds_raw: pd.DataFrame, schedule_df: pd.DataFrame) -> pd.DataFrame:
    if odds_raw.empty:
        return pd.DataFrame(columns=ODDS_SNAPSHOT_SCHEMA.columns)
    merged = odds_raw.merge(schedule_df[["game_id", "tip_ts"]], on="game_id", how="left")
    merged.loc[merged["as_of_ts"] > merged["tip_ts"], "as_of_ts"] = merged["tip_ts"]
    snapshot = latest_pre_tip_snapshot(
        merged,
        group_cols=["game_id"],
        tip_ts_col="tip_ts",
        as_of_col="as_of_ts",
    )
    snapshot["book_pref"] = snapshot["book"]
    return snapshot[
        ["game_id", "as_of_ts", "spread_home", "total", "book", "book_pref", "ingested_ts", "source"]
    ]


@app.command()
def main(
    start: datetime = typer.Option(..., help="Start date inclusive (YYYY-MM-DD)."),
    end: datetime = typer.Option(..., help="End date inclusive (YYYY-MM-DD)."),
    season: int = typer.Option(..., help="Season partition label (e.g., 2025)."),
    month: int = typer.Option(..., help="Month partition (1-12)."),
    schedule: List[str] = typer.Option(
        [],
        "--schedule",
        help="Optional schedule parquet glob(s). Falls back to live NBA API when omitted.",
    ),
    data_root: Path = typer.Option(
        paths.get_data_root(),
        help="Base data directory (defaults to PROJECTIONS_DATA_ROOT or ./data).",
    ),
    bronze_out: Path | None = typer.Option(
        None,
        "--bronze-out",
        help="Optional explicit output path for odds_raw parquet.",
    ),
    silver_out: Path | None = typer.Option(
        None,
        "--silver-out",
        help="Optional explicit output path for odds_snapshot parquet.",
    ),
    scraper_timeout: float = typer.Option(10.0, "--timeout", help="Oddstrader HTTP timeout (seconds)."),
    schedule_timeout: float = typer.Option(
        10.0,
        "--schedule-timeout",
        help="Timeout (seconds) for NBA schedule API fallback.",
    ),
) -> None:
    start_day = pd.Timestamp(start).normalize()
    end_day = pd.Timestamp(end).normalize()
    if end_day < start_day:
        raise typer.BadParameter("--end must be on/after --start.")

    schedule_df = load_schedule_data(schedule, start_day, end_day, schedule_timeout)
    resolver = TeamResolver(schedule_df)

    scraper = OddstraderScraper(timeout=scraper_timeout)
    events: list[EventOdds] = []
    for cursor in _iter_dates(start_day, end_day):
        events.extend(scraper.fetch_daily_odds(cursor.date()))

    odds_raw = _build_odds_raw(events, start=start_day, end=end_day, resolver=resolver)
    odds_snapshot = _build_odds_snapshot(odds_raw.copy(), schedule_df)

    odds_raw = enforce_schema(odds_raw, ODDS_RAW_SCHEMA)
    validate_with_pandera(odds_raw, ODDS_RAW_SCHEMA)
    odds_snapshot = enforce_schema(odds_snapshot, ODDS_SNAPSHOT_SCHEMA, allow_missing_optional=True)
    validate_with_pandera(odds_snapshot, ODDS_SNAPSHOT_SCHEMA)

    month_token = month_slug(start_day)
    bronze_default = (
        data_root / "bronze" / "odds_raw" / f"season={season}" / f"odds_{month_token}.parquet"
    )
    silver_default = (
        data_root
        / "silver"
        / "odds_snapshot"
        / f"season={season}"
        / f"month={month:02d}"
        / "odds_snapshot.parquet"
    )

    bronze_path = bronze_out or bronze_default
    silver_path = silver_out or silver_default
    bronze_path.parent.mkdir(parents=True, exist_ok=True)
    silver_path.parent.mkdir(parents=True, exist_ok=True)
    odds_raw.to_parquet(bronze_path, index=False)
    odds_snapshot.to_parquet(silver_path, index=False)

    typer.echo(
        f"[odds] wrote {len(odds_raw):,} raw rows -> {bronze_path} "
        f"and {len(odds_snapshot):,} snapshot rows -> {silver_path}"
    )


if __name__ == "__main__":  # pragma: no cover
    app()
