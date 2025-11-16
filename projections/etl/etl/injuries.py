"""Ingest NBA.com injury reports into bronze/silver parquet snapshots."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd
import typer

from projections import paths
from projections.etl.common import load_schedule_data, month_slug as _month_slug
from projections.minutes_v1.schemas import (
    INJURIES_RAW_SCHEMA,
    INJURIES_SNAPSHOT_SCHEMA,
    enforce_schema,
    validate_with_pandera,
)
from projections.minutes_v1.snapshots import select_injury_snapshot
from projections.minutes_v1.smoke_dataset import (
    PlayerResolver,
    TeamResolver,
    _normalize_key,
    _restriction_flag,
    _ramp_flag,
    _status_from_raw,
)
from scrapers.nba_players import NbaPlayersScraper
app = typer.Typer(help=__doc__)


def _read_json(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _build_player_resolver(timeout: float) -> PlayerResolver:
    scraper = NbaPlayersScraper(timeout=timeout)
    profiles = scraper.fetch_players(active_only=False)
    lookup: dict[str, int] = {}
    for profile in profiles:
        aliases = {
            profile.player_slug.replace("-", " "),
            f"{profile.first_name} {profile.last_name}",
            f"{profile.last_name}, {profile.first_name}",
            profile.first_name,
            profile.last_name,
        }
        for alias in aliases:
            norm = _normalize_key(alias)
            if norm and norm not in lookup:
                lookup[norm] = profile.person_id
    return PlayerResolver(lookup=lookup)


def _parse_matchup_text(matchup: str | None) -> tuple[str | None, str | None]:
    if not matchup:
        return None, None
    if "@" in matchup:
        away, home = matchup.split("@", 1)
        return away.strip(), home.strip()
    if "vs" in matchup:
        home, away = matchup.split("vs", 1)
        return away.strip(), home.strip()
    return None, None


def _build_injuries_raw(
    records: list[dict],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    resolver: TeamResolver,
    player_resolver: PlayerResolver,
) -> pd.DataFrame:
    start_pad = start - pd.Timedelta(days=1)
    end_pad = end + pd.Timedelta(days=1)
    ingested_ts = pd.Timestamp.utcnow()
    data: list[dict] = []
    for idx, row in enumerate(records):
        report_time = pd.Timestamp(row["report_time"]).tz_convert("UTC")
        report_day = report_time.tz_convert(None).normalize()
        if not (start_pad <= report_day <= end_pad):
            continue
        matchup = row.get("matchup", "")
        away_tri, home_tri = _parse_matchup_text(matchup)
        game_date_str = row.get("game_date")
        game_id = (
            resolver.lookup_game_id(game_date_str, away_tri, home_tri)
            if away_tri and home_tri
            else None
        )
        player_id = player_resolver.resolve(row.get("player_name"))
        team_id = resolver.resolve_team_id(row.get("team"))
        data.append(
            {
                "report_date": report_day,
                "as_of_ts": report_time,
                "team_id": team_id,
                "player_name": row.get("player_name"),
                "player_id": player_id,
                "status_raw": row.get("current_status"),
                "notes_raw": row.get("reason"),
                "game_id": game_id,
                "ingested_ts": ingested_ts,
                "source": row.get("report_url"),
                "source_row_id": f"{int(report_time.timestamp())}_{idx}",
            }
        )
    columns = [
        "report_date",
        "as_of_ts",
        "team_id",
        "player_name",
        "player_id",
        "status_raw",
        "notes_raw",
        "game_id",
        "ingested_ts",
        "source",
        "source_row_id",
        "status",
        "restriction_flag",
        "ramp_flag",
        "games_since_return",
        "days_since_return",
    ]
    injuries = pd.DataFrame(data, columns=columns)
    if injuries.empty:
        return injuries
    injuries["status"] = injuries["status_raw"].apply(_status_from_raw).astype(str)
    injuries["restriction_flag"] = injuries["notes_raw"].apply(_restriction_flag)
    injuries["ramp_flag"] = injuries["notes_raw"].apply(_ramp_flag)
    injuries["games_since_return"] = pd.NA
    injuries["days_since_return"] = pd.NA
    return injuries


def _build_injury_snapshot(injuries_raw: pd.DataFrame, schedule_df: pd.DataFrame) -> pd.DataFrame:
    if injuries_raw.empty:
        return pd.DataFrame(columns=INJURIES_SNAPSHOT_SCHEMA.columns)
    merged = injuries_raw.dropna(subset=["game_id", "player_id"]).merge(
        schedule_df.loc[:, ["game_id", "tip_ts"]],
        on="game_id",
        how="left",
    )
    snapshot = select_injury_snapshot(merged)
    return snapshot


@app.command()
def main(
    injuries_json: Path = typer.Option(..., "--injuries-json", help="JSON file from `projections.scrape injuries --out`."),
    schedule: List[str] = typer.Option(
        [],
        "--schedule",
        help="Optional parquet glob(s) for schedule data (silver). When omitted or empty, falls back to live NBA API.",
    ),
    start: datetime = typer.Option(..., "--start", help="Inclusive start date (YYYY-MM-DD)."),
    end: datetime = typer.Option(..., "--end", help="Inclusive end date (YYYY-MM-DD)."),
    season: int = typer.Option(..., "--season", help="Season partition (e.g., 2025)."),
    month: int = typer.Option(..., "--month", help="Month partition (1-12)."),
    data_root: Path = typer.Option(
        paths.get_data_root(),
        "--data-root",
        help="Base data directory (defaults to PROJECTIONS_DATA_ROOT or ./data).",
    ),
    bronze_out: Path | None = typer.Option(
        None,
        "--bronze-out",
        help="Optional explicit path for injuries_raw parquet.",
    ),
    silver_out: Path | None = typer.Option(
        None,
        "--silver-out",
        help="Optional explicit path for injuries_snapshot parquet.",
    ),
    scraper_timeout: float = typer.Option(10.0, "--timeout", help="NBA roster scraper timeout (seconds)."),
    schedule_timeout: float = typer.Option(10.0, "--schedule-timeout", help="Timeout (seconds) for NBA schedule API fallback."),
) -> None:
    start_day = pd.Timestamp(start).normalize()
    end_day = pd.Timestamp(end).normalize()
    if end_day < start_day:
        raise typer.BadParameter("--end must be on/after --start.")

    records = _read_json(injuries_json)
    schedule_df = load_schedule_data(schedule, start_day, end_day, schedule_timeout)
    resolver = TeamResolver(schedule_df)
    player_resolver = _build_player_resolver(scraper_timeout)

    injuries_raw = _build_injuries_raw(
        records,
        start=start_day,
        end=end_day,
        resolver=resolver,
        player_resolver=player_resolver,
    )
    injuries_raw = enforce_schema(injuries_raw, INJURIES_RAW_SCHEMA)
    validate_with_pandera(injuries_raw, INJURIES_RAW_SCHEMA)

    injuries_snapshot = _build_injury_snapshot(injuries_raw, schedule_df)
    injuries_snapshot = enforce_schema(injuries_snapshot, INJURIES_SNAPSHOT_SCHEMA)
    validate_with_pandera(injuries_snapshot, INJURIES_SNAPSHOT_SCHEMA)

    month_slug = _month_slug(start_day)
    default_bronze = (
        data_root
        / "bronze"
        / "injuries_raw"
        / f"season={season}"
        / f"injuries_{month_slug}.parquet"
    )
    default_silver = (
        data_root
        / "silver"
        / "injuries_snapshot"
        / f"season={season}"
        / f"month={month:02d}"
        / "injuries_snapshot.parquet"
    )
    bronze_path = bronze_out or default_bronze
    silver_path = silver_out or default_silver
    bronze_path.parent.mkdir(parents=True, exist_ok=True)
    silver_path.parent.mkdir(parents=True, exist_ok=True)
    injuries_raw.to_parquet(bronze_path, index=False)
    injuries_snapshot.to_parquet(silver_path, index=False)

    typer.echo(
        f"[injuries] wrote {len(injuries_raw):,} raw rows -> {bronze_path} "
        f"and {len(injuries_snapshot):,} snapshot rows -> {silver_path}"
    )


if __name__ == "__main__":  # pragma: no cover
    app()
