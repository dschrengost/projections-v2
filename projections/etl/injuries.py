"""Ingest NBA.com injury reports into bronze/silver parquet snapshots."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List

import pandas as pd
import typer

from projections import paths
from projections.etl import storage
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
from projections.pipeline.status import JobStatus, write_status
from scrapers.nba_injuries import InjuryRecord, NBAInjuryScraper
from scrapers.nba_players import NbaPlayersScraper

app = typer.Typer(help=__doc__)


def _status_target(start_day: pd.Timestamp, end_day: pd.Timestamp) -> str:
    if start_day == end_day:
        return start_day.date().isoformat()
    return f"{start_day.date()}_{end_day.date()}"


def _nan_rate(df: pd.DataFrame, cols: list[str]) -> float | None:
    present = [col for col in cols if col in df.columns]
    if not present or df.empty:
        return 0.0
    return float(df[present].isna().mean().mean())


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


def _records_from_scraper_payload(records: list[InjuryRecord]) -> list[dict[str, Any]]:
    """Convert dataclass records to dictionaries compatible with _build_injuries_raw."""
    payload: list[dict[str, Any]] = []
    for record in records:
        raw = asdict(record)
        raw["report_time"] = record.report_time.isoformat()
        raw["game_date"] = record.game_date.isoformat()
        payload.append(raw)
    return payload


def _scrape_injury_records(
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    timeout: float,
) -> list[dict[str, Any]]:
    """Scrape NBA injury PDFs for the requested window and return normalized records."""
    # Fetch through the last minute of the requested end day to ensure complete coverage.
    start_dt = start.to_pydatetime()
    end_dt = (end + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)).to_pydatetime()
    with NBAInjuryScraper(timeout=timeout) as scraper:
        typer.echo(
            f"[injuries] scraping NBA reports from {start_dt.date()} to {end_dt.date()} (ET)."
        )
        records = scraper.fetch_range(start_dt, end_dt)
    return _records_from_scraper_payload(records)


def _load_injury_records(
    *,
    injuries_json: Path | None,
    use_scraper: bool,
    start: pd.Timestamp,
    end: pd.Timestamp,
    timeout: float,
) -> list[dict[str, Any]]:
    if injuries_json is not None:
        typer.echo(f"[injuries] loading JSON override from {injuries_json}")
        return _read_json(injuries_json)
    if not use_scraper:
        raise typer.BadParameter(
            "Provide --injuries-json or enable --use-scraper to fetch data."
        )
    return _scrape_injury_records(start=start, end=end, timeout=timeout)


def _snapshot_partition_key(target_day: pd.Timestamp, month_override: int | None = None) -> str:
    """Return the folder token for snapshots (month buckets for now, hook for day partitions)."""
    partition_month = month_override or target_day.month
    return f"month={partition_month:02d}"


def _default_silver_path(data_root: Path, season: int, partition_key: str) -> Path:
    return (
        data_root
        / "silver"
        / "injuries_snapshot"
        / f"season={season}"
        / partition_key
        / "injuries_snapshot.parquet"
    )


@app.command()
def main(
    injuries_json: Path | None = typer.Option(
        None,
        "--injuries-json",
        help="Optional JSON override (e.g., from an older scrape) when bypassing the scraper.",
    ),
    schedule: List[str] = typer.Option(
        [],
        "--schedule",
        help="Optional parquet glob(s) for schedule data (silver). When omitted or empty, falls back to live NBA API.",
    ),
    use_scraper: bool = typer.Option(
        True,
        "--use-scraper/--no-use-scraper",
        help="Scrape NBA.com injury PDFs directly (default). Disable to rely solely on --injuries-json.",
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
    bronze_root: Path | None = typer.Option(
        None,
        "--bronze-root",
        help="Optional override for injuries_raw bronze root (defaults to the standard contract).",
    ),
    bronze_out: Path | None = typer.Option(
        None,
        "--bronze-out",
        help="[deprecated] Write a single parquet instead of date partitions.",
    ),
    silver_out: Path | None = typer.Option(
        None,
        "--silver-out",
        help="Optional explicit path for injuries_snapshot parquet.",
    ),
    scraper_timeout: float = typer.Option(10.0, "--timeout", help="NBA player resolver scraper timeout (seconds)."),
    schedule_timeout: float = typer.Option(10.0, "--schedule-timeout", help="Timeout (seconds) for NBA schedule API fallback."),
    injury_timeout: float = typer.Option(
        15.0,
        "--injury-timeout",
        help="HTTP timeout (seconds) for NBA injury PDF scraping.",
    ),
) -> None:
    """Main injuries ETL entry point; intended usage relies on the live scraper + bronze/silver sinks."""
    start_day = pd.Timestamp(start).normalize()
    end_day = pd.Timestamp(end).normalize()
    if end_day < start_day:
        raise typer.BadParameter("--end must be on/after --start.")

    target_date = _status_target(start_day, end_day)
    run_ts = datetime.now(timezone.utc).isoformat()
    rows_written = 0
    try:
        data_root = data_root.resolve()
        records = _load_injury_records(
            injuries_json=injuries_json,
            use_scraper=use_scraper,
            start=start_day,
            end=end_day,
            timeout=injury_timeout,
        )
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
        partition_key = _snapshot_partition_key(start_day, month_override=month)
        default_silver = _default_silver_path(data_root, season, partition_key)
        silver_path = silver_out or default_silver
        silver_path.parent.mkdir(parents=True, exist_ok=True)

        bronze_rows_written = 0
        bronze_partitions = 0
        bronze_root_path = (bronze_root or storage.default_bronze_root("injuries_raw", data_root)).resolve()
        if bronze_out:
            bronze_out.parent.mkdir(parents=True, exist_ok=True)
            injuries_raw.to_parquet(bronze_out, index=False)
            bronze_rows_written = len(injuries_raw)
            bronze_partitions = 1
            typer.echo(
                f"[injuries] wrote {len(injuries_raw):,} rows to legacy bronze_out={bronze_out} "
                "(consider using --bronze-root for partitioned writes)."
            )
        else:
            if injuries_raw.empty:
                typer.echo("[injuries] no rows to persist for the requested window.")
            else:
                normalized_dates = injuries_raw["report_date"].dt.normalize()
                for cursor in storage.iter_days(start_day, end_day):
                    mask = normalized_dates == cursor
                    if not mask.any():
                        continue
                    day_frame = injuries_raw.loc[mask].copy()
                    result = storage.write_bronze_partition(
                        day_frame,
                        dataset="injuries_raw",
                        data_root=data_root,
                        season=season,
                        target_date=cursor.date(),
                        bronze_root=bronze_root_path,
                    )
                    bronze_rows_written += result.rows
                    bronze_partitions += 1
                    typer.echo(
                        f"[injuries] bronze partition {result.target_date}: "
                        f"{result.rows} rows -> {result.path}"
                    )

        injuries_snapshot.to_parquet(silver_path, index=False)
        rows_written = len(injuries_snapshot)

        typer.echo(
            f"[injuries] window={start_day.date()}->{end_day.date()} "
            f"bronze_partitions={bronze_partitions} bronze_rows={bronze_rows_written} "
            f"silver_rows={len(injuries_snapshot)} -> {silver_path}"
        )
        write_status(
            JobStatus(
                job_name="injuries_live",
                stage="silver",
                target_date=target_date,
                run_ts=run_ts,
                status="success",
                rows_written=rows_written,
                expected_rows=None,
                nan_rate_key_cols=_nan_rate(injuries_snapshot, ["game_id", "player_id", "status"]),
            )
        )
    except Exception as exc:  # noqa: BLE001
        write_status(
            JobStatus(
                job_name="injuries_live",
                stage="silver",
                target_date=target_date,
                run_ts=run_ts,
                status="error",
                rows_written=rows_written,
                expected_rows=None,
                message=str(exc),
            )
        )
        raise


if __name__ == "__main__":  # pragma: no cover
    app()
