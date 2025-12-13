#!/usr/bin/env python3
"""Backfill NBA injury report PDFs into append-only bronze partitions.

This script fetches hourly NBA injury reports (typically published at :30 ET),
normalizes them into the canonical ``injuries_raw`` schema, and writes:

  - canonical history: ``date=YYYY-MM-DD/hour=HH/injuries.parquet``
  - transitional latest view: ``date=YYYY-MM-DD/injuries.parquet`` (overwritten)

Status is tracked under ``<DATA_ROOT>/bronze/injuries_raw/_backfill_status.json`` so
long backfills can be resumed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
import os
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import typer

from projections import paths
from projections.etl import storage
from projections.etl.common import load_schedule_data
from projections.etl.injuries import (
    _build_injuries_raw,
    _build_player_resolver,
    _records_from_scraper_payload,
)
from projections.minutes_v1.schemas import INJURIES_RAW_SCHEMA, enforce_schema, validate_with_pandera
from projections.minutes_v1.season_dataset import PlayerResolver, TeamResolver
from scrapers.nba_injuries import ET_TZ, NBAInjuryScraper

app = typer.Typer(help=__doc__)


class NoScheduleForDateError(RuntimeError):
    """Raised when schedule data cannot be loaded for a date (often: no games)."""


def _iter_dates(start: date, end: date) -> Iterable[date]:
    cursor = start
    while cursor <= end:
        yield cursor
        cursor += timedelta(days=1)


def _atomic_write_text(path: Path, payload: str, *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        tmp.write_text(payload, encoding=encoding)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _status_path(data_root: Path) -> Path:
    return storage.default_bronze_root("injuries_raw", data_root) / "_backfill_status.json"


def _load_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "last_run": None,
            "dates_completed": [],
            "dates_failed": [],
            "hours_fetched": 0,
            "hours_missing": 0,
            "dates": {},
        }
    return json.loads(path.read_text(encoding="utf-8"))


def _write_status(path: Path, status: dict[str, Any]) -> None:
    status["last_run"] = datetime.now(timezone.utc).isoformat()
    _atomic_write_text(path, json.dumps(status, indent=2, sort_keys=True))


def _hour_partition_exists(
    *,
    data_root: Path,
    season: int,
    target_date: date,
    hour: int,
    bronze_root: Path | None = None,
) -> bool:
    output_name = storage.DEFAULT_BRONZE_FILENAMES.get("injuries_raw", "data.parquet")
    day_dir = storage.bronze_partition_dir(
        "injuries_raw",
        data_root=data_root,
        season=season,
        target_date=target_date,
        bronze_root=bronze_root,
    )
    return (day_dir / f"hour={hour:02d}" / output_name).exists()


@dataclass(frozen=True)
class HourResult:
    hour: int
    status: str
    rows: int
    error: str | None = None


def _backfill_day(
    target_date: date,
    *,
    season: int,
    data_root: Path,
    schedule_paths: list[str],
    schedule_timeout: float,
    player_resolver: PlayerResolver,
    injury_timeout: float,
    start_hour: int,
    end_hour: int,
    dry_run: bool,
    force: bool,
) -> tuple[list[HourResult], int, int]:
    if start_hour < 0 or start_hour > 23:
        raise typer.BadParameter("--start-hour must be in [0, 23]")
    if end_hour < 0 or end_hour > 23 or end_hour < start_hour:
        raise typer.BadParameter("--end-hour must be in [0, 23] and >= --start-hour")

    target_day = pd.Timestamp(target_date).normalize()
    schedule_start = target_day - pd.Timedelta(days=1)
    schedule_end = target_day + pd.Timedelta(days=1)

    effective_schedule = list(schedule_paths)
    if not effective_schedule:
        # Prefer local silver schedule partitions over NBA API fallback (API can be flaky/blocked).
        months: set[int] = {int(schedule_start.month), int(schedule_end.month)}
        for month in sorted(months):
            candidate = (
                data_root
                / "silver"
                / "schedule"
                / f"season={season}"
                / f"month={month:02d}"
                / "schedule.parquet"
            )
            if candidate.exists():
                effective_schedule.append(str(candidate))

    try:
        schedule_df = load_schedule_data(
            effective_schedule, schedule_start, schedule_end, schedule_timeout
        )
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        no_games_markers = (
            "NBA schedule API did not return any games",
            "Schedule filter removed all rows",
            "No games found in the requested window",
        )
        if any(marker in msg for marker in no_games_markers):
            raise NoScheduleForDateError(
                f"No schedule rows for {target_date.isoformat()} (season={season}); likely no games."
            ) from exc
        raise NoScheduleForDateError(
            f"Unable to load schedule rows for {target_date.isoformat()} (season={season}). "
            "Provide --schedule pointing at local schedule.parquet files, or run the schedule ETL "
            "to populate <data_root>/silver/schedule/season=YYYY/month=MM/schedule.parquet."
        ) from exc
    resolver = TeamResolver(schedule_df)

    bronze_root_path = storage.default_bronze_root("injuries_raw", data_root)

    hour_results: list[HourResult] = []
    frames: list[pd.DataFrame] = []
    hours_fetched = 0
    hours_missing = 0

    with NBAInjuryScraper(timeout=injury_timeout) as scraper:
        for hour in range(start_hour, end_hour + 1):
            if not force and _hour_partition_exists(
                data_root=data_root,
                season=season,
                target_date=target_date,
                hour=hour,
                bronze_root=bronze_root_path,
            ):
                hour_results.append(HourResult(hour=hour, status="skipped_exists", rows=0))
                continue

            report_time = datetime.combine(target_date, time(hour, 30), tzinfo=ET_TZ)
            try:
                if not scraper.report_exists(report_time):
                    hours_missing += 1
                    hour_results.append(HourResult(hour=hour, status="missing", rows=0))
                    continue

                records = scraper.fetch_report(report_time)
                if isinstance(records, pd.DataFrame):
                    raise RuntimeError("Expected InjuryRecord list, received DataFrame.")
                payload = _records_from_scraper_payload(records)
                injuries_raw = _build_injuries_raw(
                    payload,
                    start=target_day,
                    end=target_day,
                    resolver=resolver,
                    player_resolver=player_resolver,
                )
                injuries_raw = enforce_schema(injuries_raw, INJURIES_RAW_SCHEMA)
                validate_with_pandera(injuries_raw, INJURIES_RAW_SCHEMA)
                frames.append(injuries_raw)
                if not dry_run:
                    storage.write_bronze_partition_hourly(
                        injuries_raw,
                        dataset="injuries_raw",
                        data_root=data_root,
                        season=season,
                        target_date=target_date,
                        hour=hour,
                        bronze_root=bronze_root_path,
                    )
                hours_fetched += 1
                hour_results.append(HourResult(hour=hour, status="success", rows=len(injuries_raw)))
            except Exception as exc:  # noqa: BLE001
                hour_results.append(HourResult(hour=hour, status="error", rows=0, error=str(exc)))

    if frames and not dry_run:
        combined = pd.concat(frames, ignore_index=True)
        storage.write_bronze_partition(
            combined,
            dataset="injuries_raw",
            data_root=data_root,
            season=season,
            target_date=target_date,
            bronze_root=bronze_root_path,
        )

    return hour_results, hours_fetched, hours_missing


@app.command()
def backfill(
    season: int | None = typer.Option(
        None,
        "--season",
        help=(
            "Season partition label (e.g., 2025). When omitted, derived from each date "
            "(month>=8 -> same year, else year-1)."
        ),
    ),
    date_str: str | None = typer.Option(None, "--date", help="Single ET date (YYYY-MM-DD)."),
    start: str | None = typer.Option(None, "--start", help="Start ET date (YYYY-MM-DD)."),
    end: str | None = typer.Option(None, "--end", help="End ET date (YYYY-MM-DD)."),
    schedule: list[str] = typer.Option(
        [],
        "--schedule",
        help="Optional schedule parquet glob(s). Falls back to NBA API when omitted.",
    ),
    data_root: Path = typer.Option(
        paths.get_data_root(),
        "--data-root",
        help="Base data directory (defaults to PROJECTIONS_DATA_ROOT or ./data).",
    ),
    dry_run: bool = typer.Option(True, "--dry-run/--no-dry-run", help="Don't modify data."),
    force: bool = typer.Option(False, "--force", help="Re-fetch and overwrite existing hour partitions."),
    start_hour: int = typer.Option(8, "--start-hour", min=0, max=23, help="First ET hour to probe."),
    end_hour: int = typer.Option(23, "--end-hour", min=0, max=23, help="Last ET hour to probe."),
    schedule_timeout: float = typer.Option(10.0, "--schedule-timeout", help="Schedule API timeout (seconds)."),
    player_timeout: float = typer.Option(10.0, "--player-timeout", help="NBA player resolver timeout (seconds)."),
    injury_timeout: float = typer.Option(30.0, "--injury-timeout", help="NBA injury PDF timeout (seconds)."),
    resume: bool = typer.Option(True, "--resume/--no-resume", help="Skip dates already marked completed."),
) -> None:
    """Backfill hourly injury report snapshots into bronze."""
    if date_str:
        start_date = end_date = date.fromisoformat(date_str)
    elif start and end:
        start_date = date.fromisoformat(start)
        end_date = date.fromisoformat(end)
    else:
        raise typer.BadParameter("Provide --date or both --start/--end.")
    if end_date < start_date:
        raise typer.BadParameter("--end must be on/after --start.")

    data_root = data_root.resolve()
    status_file = _status_path(data_root)
    status = _load_status(status_file)

    typer.echo(
        f"[injuries-backfill] season={season or 'auto'} dates={start_date.isoformat()}..{end_date.isoformat()} "
        f"hours={start_hour:02d}-{end_hour:02d} dry_run={dry_run} force={force} resume={resume}"
    )

    typer.echo("[injuries-backfill] building player resolver...")
    player_resolver = _build_player_resolver(player_timeout)

    for target_date in _iter_dates(start_date, end_date):
        season_value = season or (target_date.year if target_date.month >= 8 else target_date.year - 1)
        key = target_date.isoformat()
        if resume and not force and key in set(status.get("dates_completed", [])):
            typer.echo(f"[injuries-backfill] {key}: skipping (already completed).")
            continue

        typer.echo(f"[injuries-backfill] {key}: probing hourly PDFs...")
        try:
            hour_results, hours_fetched, hours_missing = _backfill_day(
                target_date,
                season=season_value,
                data_root=data_root,
                schedule_paths=schedule,
                schedule_timeout=schedule_timeout,
                player_resolver=player_resolver,
                injury_timeout=injury_timeout,
                start_hour=start_hour,
                end_hour=end_hour,
                dry_run=dry_run,
                force=force,
            )
        except NoScheduleForDateError as exc:
            typer.echo(f"[injuries-backfill] {key}: {exc}")
            status.setdefault("dates", {})[key] = {
                "no_schedule": True,
                "message": str(exc),
                "hours": {},
            }
            status.setdefault("dates_completed", []).append(key)
            _write_status(status_file, status)
            continue
        status.setdefault("dates", {})[key] = {
            "hours": {
                f"{result.hour:02d}": {
                    "status": result.status,
                    "rows": result.rows,
                    "error": result.error,
                }
                for result in hour_results
            },
        }
        status["hours_fetched"] = int(status.get("hours_fetched", 0)) + int(hours_fetched)
        status["hours_missing"] = int(status.get("hours_missing", 0)) + int(hours_missing)

        errors = [result for result in hour_results if result.status == "error"]
        if errors:
            status.setdefault("dates_failed", []).append(
                {"date": key, "error": errors[0].error or "unknown"}
            )
            typer.echo(f"[injuries-backfill] {key}: completed with {len(errors)} error(s).", err=True)
        else:
            status.setdefault("dates_completed", []).append(key)
            typer.echo(
                f"[injuries-backfill] {key}: done (hours_fetched={hours_fetched}, hours_missing={hours_missing})."
            )

        _write_status(status_file, status)


if __name__ == "__main__":  # pragma: no cover
    app()
