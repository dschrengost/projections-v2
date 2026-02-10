#!/usr/bin/env python3
"""Rebuild silver odds/injuries snapshots from immutable bronze history."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import os
from pathlib import Path
from uuid import uuid4

import pandas as pd
import typer

from projections import paths
from projections.etl import storage
from projections.etl import injuries as injuries_etl
from projections.etl import odds as odds_etl
from projections.etl.snapshot_guard import enforce_non_regression
from projections.minutes_v1.schemas import (
    INJURIES_SNAPSHOT_SCHEMA,
    ODDS_SNAPSHOT_SCHEMA,
    enforce_schema,
    validate_with_pandera,
)

app = typer.Typer(help=__doc__)


def _normalize_day(value: datetime) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.normalize()


def _iter_days(start_day: pd.Timestamp, end_day: pd.Timestamp) -> list[pd.Timestamp]:
    out: list[pd.Timestamp] = []
    cursor = start_day
    while cursor <= end_day:
        out.append(cursor)
        cursor += pd.Timedelta(days=1)
    return out


def _resolve_season(day: pd.Timestamp, season_override: int | None) -> int:
    if season_override is not None:
        return int(season_override)
    return int(day.year) if int(day.month) >= 8 else int(day.year) - 1


def _schedule_path(data_root: Path, season: int, month: int) -> Path:
    return (
        data_root
        / "silver"
        / "schedule"
        / f"season={season}"
        / f"month={month:02d}"
        / "schedule.parquet"
    )


def _load_schedule_map(
    *,
    data_root: Path,
    days: list[pd.Timestamp],
    season_override: int | None,
) -> tuple[dict[tuple[int, int], pd.DataFrame], pd.DataFrame]:
    month_frames: dict[tuple[int, int], pd.DataFrame] = {}
    requested_keys = {
        (_resolve_season(day, season_override), int(day.month))
        for day in days
    }

    for season, month in sorted(requested_keys):
        path = _schedule_path(data_root, season, month)
        if not path.exists():
            raise FileNotFoundError(f"Missing schedule parquet: {path}")
        frame = pd.read_parquet(path)
        frame["game_date"] = pd.to_datetime(frame["game_date"], errors="coerce").dt.normalize()
        month_frames[(season, month)] = frame

    combined = pd.concat(month_frames.values(), ignore_index=True)
    return month_frames, combined


def _atomic_write_parquet(df: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        df.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, destination)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def _odds_snapshot_target(data_root: Path, season: int, month: int) -> Path:
    return (
        data_root
        / "silver"
        / "odds_snapshot"
        / f"season={season}"
        / f"month={month:02d}"
        / "odds_snapshot.parquet"
    )


def _injuries_snapshot_target(data_root: Path, season: int, month: int) -> Path:
    return (
        data_root
        / "silver"
        / "injuries_snapshot"
        / f"season={season}"
        / f"month={month:02d}"
        / "injuries_snapshot.parquet"
    )


def _dedupe_injuries(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    work = df.copy()
    work["as_of_ts"] = pd.to_datetime(work["as_of_ts"], utc=True, errors="coerce")
    work = work.sort_values(["as_of_ts"], ascending=True)
    work = work.drop_duplicates(subset=["game_id", "player_id"], keep="last")
    return work


@app.command()
def main(
    start_date: datetime = typer.Option(..., "--start-date", "--start", help="Start date inclusive (YYYY-MM-DD)."),
    end_date: datetime = typer.Option(..., "--end-date", "--end", help="End date inclusive (YYYY-MM-DD)."),
    data_root: Path = typer.Option(
        paths.get_data_root(),
        "--data-root",
        help="Base data directory (defaults to PROJECTIONS_DATA_ROOT or ./data).",
    ),
    season: int | None = typer.Option(
        None,
        "--season",
        help="Optional season override. Defaults to NBA season rule per date (month>=8 => same year else year-1).",
    ),
    rebuild_odds: bool = typer.Option(True, "--odds/--skip-odds", help="Rebuild odds_snapshot."),
    rebuild_injuries: bool = typer.Option(
        True,
        "--injuries/--skip-injuries",
        help="Rebuild injuries_snapshot.",
    ),
    allow_snapshot_regression: bool = typer.Option(
        False,
        "--allow-snapshot-regression/--no-allow-snapshot-regression",
        help="Allow rebuilt silver snapshots to have lower key coverage than existing files.",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Compute and report outputs without writing."),
) -> None:
    start_day = _normalize_day(start_date)
    end_day = _normalize_day(end_date)
    if end_day < start_day:
        raise typer.BadParameter("--end-date must be on/after --start-date")
    if not rebuild_odds and not rebuild_injuries:
        raise typer.BadParameter("At least one of --odds or --injuries must be enabled.")

    data_root = data_root.resolve()
    days = _iter_days(start_day, end_day)
    month_schedule_map, full_schedule = _load_schedule_map(
        data_root=data_root,
        days=days,
        season_override=season,
    )
    full_schedule = full_schedule[
        (full_schedule["game_date"] >= start_day) & (full_schedule["game_date"] <= end_day)
    ].copy()
    if full_schedule.empty:
        raise RuntimeError("No scheduled games found for the requested range.")

    odds_month_frames: dict[tuple[int, int], list[pd.DataFrame]] = defaultdict(list)
    injuries_month_frames: dict[tuple[int, int], list[pd.DataFrame]] = defaultdict(list)

    odds_days_with_data = 0
    injuries_days_with_data = 0

    for day in days:
        season_value = _resolve_season(day, season)
        month_value = int(day.month)
        key = (season_value, month_value)

        schedule_month = month_schedule_map.get(key)
        if schedule_month is None or schedule_month.empty:
            continue
        schedule_day = schedule_month[schedule_month["game_date"] == day].copy()
        if schedule_day.empty:
            continue

        if rebuild_odds:
            odds_raw = storage.read_bronze_day(
                "odds_raw",
                data_root=data_root,
                season=season_value,
                target_date=day.date(),
                include_runs=True,
                prefer_history=True,
            )
            if not odds_raw.empty:
                schedule_valid = schedule_month.copy()
                schedule_valid["tip_ts"] = pd.to_datetime(
                    schedule_valid["tip_ts"], utc=True, errors="coerce"
                )
                schedule_valid = schedule_valid[schedule_valid["tip_ts"].notna()].copy()
                if schedule_valid.empty:
                    continue
                valid_game_ids = set(
                    pd.to_numeric(schedule_valid["game_id"], errors="coerce")
                    .dropna()
                    .astype(int)
                    .tolist()
                )
                odds_raw["game_id"] = pd.to_numeric(
                    odds_raw.get("game_id"), errors="coerce"
                ).astype("Int64")
                odds_raw = odds_raw[odds_raw["game_id"].isin(valid_game_ids)].copy()
                if odds_raw.empty:
                    continue
                odds_snapshot = odds_etl._build_odds_snapshot(odds_raw.copy(), schedule_month.copy())
                odds_snapshot = enforce_schema(
                    odds_snapshot, ODDS_SNAPSHOT_SCHEMA, allow_missing_optional=True
                )
                validate_with_pandera(odds_snapshot, ODDS_SNAPSHOT_SCHEMA)
                if not odds_snapshot.empty:
                    odds_month_frames[key].append(odds_snapshot)
                    odds_days_with_data += 1

    if rebuild_injuries:
        injuries_scan_days = _iter_days(start_day - pd.Timedelta(days=1), end_day)
        injuries_raw_frames: list[pd.DataFrame] = []
        for day in injuries_scan_days:
            season_value = _resolve_season(day, season)
            injuries_raw = storage.read_bronze_day(
                "injuries_raw",
                data_root=data_root,
                season=season_value,
                target_date=day.date(),
                include_runs=True,
                prefer_history=True,
            )
            if injuries_raw.empty:
                continue
            injuries_raw["game_id"] = pd.to_numeric(
                injuries_raw.get("game_id"), errors="coerce"
            ).astype("Int64")
            injuries_raw_frames.append(injuries_raw)
            injuries_days_with_data += 1

        injuries_raw_all = (
            pd.concat(injuries_raw_frames, ignore_index=True)
            if injuries_raw_frames
            else pd.DataFrame()
        )

        if not injuries_raw_all.empty:
            for key, schedule_month in month_schedule_map.items():
                season_value, month_value = key
                schedule_window = schedule_month[
                    (schedule_month["game_date"] >= start_day)
                    & (schedule_month["game_date"] <= end_day)
                ].copy()
                if schedule_window.empty:
                    continue
                valid_game_ids = set(
                    pd.to_numeric(schedule_window["game_id"], errors="coerce")
                    .dropna()
                    .astype(int)
                    .tolist()
                )
                raw_month = injuries_raw_all[injuries_raw_all["game_id"].isin(valid_game_ids)].copy()
                if raw_month.empty:
                    continue
                if "report_ts" not in raw_month.columns:
                    raw_month["report_ts"] = pd.to_datetime(
                        raw_month.get("as_of_ts"), utc=True, errors="coerce"
                    )
                injuries_snapshot = injuries_etl._build_injury_snapshot(
                    raw_month,
                    schedule_window,
                )
                injuries_snapshot = enforce_schema(injuries_snapshot, INJURIES_SNAPSHOT_SCHEMA)
                validate_with_pandera(injuries_snapshot, INJURIES_SNAPSHOT_SCHEMA)
                if not injuries_snapshot.empty:
                    injuries_month_frames[(season_value, month_value)].append(injuries_snapshot)

    game_date_map = (
        full_schedule[["game_id", "game_date"]]
        .dropna(subset=["game_id", "game_date"])
        .drop_duplicates(subset=["game_id"], keep="last")
    )
    game_date_map["game_id"] = pd.to_numeric(game_date_map["game_id"], errors="coerce").astype("Int64")
    game_date_lookup = game_date_map.set_index("game_id")["game_date"]

    if rebuild_odds:
        for key, frames in sorted(odds_month_frames.items()):
            season_value, month_value = key
            rebuilt = pd.concat(frames, ignore_index=True)
            rebuilt = odds_etl._dedupe_snapshot_by_game(rebuilt)
            rebuilt = enforce_schema(rebuilt, ODDS_SNAPSHOT_SCHEMA, allow_missing_optional=True)
            validate_with_pandera(rebuilt, ODDS_SNAPSHOT_SCHEMA)

            target = _odds_snapshot_target(data_root, season_value, month_value)
            existing_rows = 0
            if target.exists():
                existing = pd.read_parquet(target)
                existing_rows = len(existing)
                existing["game_id"] = pd.to_numeric(existing["game_id"], errors="coerce").astype("Int64")
                existing["_game_date"] = existing["game_id"].map(game_date_lookup)
                keep_mask = (
                    existing["_game_date"].isna()
                    | (existing["_game_date"] < start_day)
                    | (existing["_game_date"] > end_day)
                )
                keep_existing = existing.loc[keep_mask].drop(columns=["_game_date"], errors="ignore")
                rebuilt = pd.concat([keep_existing, rebuilt], ignore_index=True, sort=False)
                rebuilt = odds_etl._dedupe_snapshot_by_game(rebuilt)
                rebuilt = enforce_schema(rebuilt, ODDS_SNAPSHOT_SCHEMA, allow_missing_optional=True)
                validate_with_pandera(rebuilt, ODDS_SNAPSHOT_SCHEMA)
                enforce_non_regression(
                    dataset_name="odds_rebuild",
                    existing=existing,
                    candidate=rebuilt,
                    key_cols=("game_id",),
                    allow_regression=allow_snapshot_regression,
                )

            if dry_run:
                typer.echo(
                    f"[rebuild][odds] season={season_value} month={month_value:02d}: "
                    f"existing_rows={existing_rows} rebuilt_rows={len(rebuilt)} target={target}"
                )
            else:
                _atomic_write_parquet(rebuilt, target)
                typer.echo(
                    f"[rebuild][odds] wrote {len(rebuilt):,} rows -> {target} "
                    f"(existing_rows={existing_rows:,})"
                )

    if rebuild_injuries:
        for key, frames in sorted(injuries_month_frames.items()):
            season_value, month_value = key
            rebuilt = pd.concat(frames, ignore_index=True)
            rebuilt = _dedupe_injuries(rebuilt)
            rebuilt = enforce_schema(rebuilt, INJURIES_SNAPSHOT_SCHEMA)
            validate_with_pandera(rebuilt, INJURIES_SNAPSHOT_SCHEMA)

            target = _injuries_snapshot_target(data_root, season_value, month_value)
            existing_rows = 0
            if target.exists():
                existing = pd.read_parquet(target)
                existing_rows = len(existing)
                existing["game_id"] = pd.to_numeric(existing["game_id"], errors="coerce").astype("Int64")
                existing["_game_date"] = existing["game_id"].map(game_date_lookup)
                keep_mask = (
                    existing["_game_date"].isna()
                    | (existing["_game_date"] < start_day)
                    | (existing["_game_date"] > end_day)
                )
                keep_existing = existing.loc[keep_mask].drop(columns=["_game_date"], errors="ignore")
                rebuilt = pd.concat([keep_existing, rebuilt], ignore_index=True, sort=False)
                rebuilt = _dedupe_injuries(rebuilt)
                rebuilt = enforce_schema(rebuilt, INJURIES_SNAPSHOT_SCHEMA)
                validate_with_pandera(rebuilt, INJURIES_SNAPSHOT_SCHEMA)
                enforce_non_regression(
                    dataset_name="injuries_rebuild",
                    existing=existing,
                    candidate=rebuilt,
                    key_cols=("game_id", "player_id"),
                    allow_regression=allow_snapshot_regression,
                )

            if dry_run:
                typer.echo(
                    f"[rebuild][injuries] season={season_value} month={month_value:02d}: "
                    f"existing_rows={existing_rows} rebuilt_rows={len(rebuilt)} target={target}"
                )
            else:
                _atomic_write_parquet(rebuilt, target)
                typer.echo(
                    f"[rebuild][injuries] wrote {len(rebuilt):,} rows -> {target} "
                    f"(existing_rows={existing_rows:,})"
                )

    typer.echo(
        "[rebuild] complete "
        f"(odds_days_with_data={odds_days_with_data}, injuries_days_with_data={injuries_days_with_data}, "
        f"dry_run={dry_run})"
    )


if __name__ == "__main__":  # pragma: no cover
    app()
