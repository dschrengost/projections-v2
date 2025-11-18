"""Build pre-tip roster snapshots per team-game."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Sequence

import pandas as pd
import typer

from projections import paths
from projections.etl import storage
from projections.etl.common import load_schedule_data
from projections.minutes_v1.schemas import (
    ROSTER_NIGHTLY_RAW_SCHEMA,
    ROSTER_NIGHTLY_SCHEMA,
    enforce_schema,
    validate_with_pandera,
)
from projections.utils import asof_left_join, ensure_directory
from scrapers.nba_players import NbaPlayersScraper, PlayerProfile

app = typer.Typer(help=__doc__)

REQUIRED_ROSTER_COLUMNS: tuple[str, ...] = (
    "game_id",
    "team_id",
    "player_id",
    "player_name",
    "as_of_ts",
    "active_flag",
)

SCHEDULE_COLUMNS: tuple[str, ...] = ("game_id", "game_date", "tip_ts")

OUTPUT_COLUMNS: tuple[str, ...] = (
    "game_id",
    "team_id",
    "player_id",
    "player_name",
    "game_date",
    "as_of_ts",
    "active_flag",
    "starter_flag",
    "lineup_role",
    "lineup_status",
    "lineup_roster_status",
    "lineup_timestamp",
    "is_projected_starter",
    "is_confirmed_starter",
    "listed_pos",
    "height_in",
    "weight_lb",
    "age",
    "ingested_ts",
    "source",
)


@dataclass(frozen=True)
class SnapshotConfig:
    start_date: pd.Timestamp | None
    end_date: pd.Timestamp | None


def _read_parquet_sources(patterns: Sequence[str | Path], *, allow_empty: bool = False) -> pd.DataFrame:
    from glob import glob

    paths: list[Path] = []
    for pattern in patterns:
        pattern_path = Path(pattern)
        if pattern_path.exists() and pattern_path.suffix == ".parquet":
            paths.append(pattern_path)
            continue
        matches = [Path(p) for p in glob(str(pattern), recursive=True)]
        paths.extend(p for p in matches if p.suffix == ".parquet" and p.exists())
    if not paths:
        if allow_empty:
            return pd.DataFrame()
        raise FileNotFoundError(f"No parquet files found for inputs: {patterns}")
    frames = [pd.read_parquet(path) for path in paths]
    return pd.concat(frames, ignore_index=True)


def _normalize_date(value: datetime | str | pd.Timestamp | None) -> pd.Timestamp | None:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.normalize()


def _roster_from_players(
    schedule: pd.DataFrame,
    players: Sequence[PlayerProfile],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    source: str = "nba.com/players",
    now: pd.Timestamp | None = None,
) -> pd.DataFrame:
    if now is None:
        now = pd.Timestamp.now(tz="UTC")
    if not players:
        return pd.DataFrame(columns=list(ROSTER_NIGHTLY_RAW_SCHEMA.columns))
    schedule_window = schedule[
        (schedule["game_date"] >= start) & (schedule["game_date"] <= end)
    ].copy()
    if schedule_window.empty:
        return pd.DataFrame(columns=list(ROSTER_NIGHTLY_RAW_SCHEMA.columns))
    schedule_window["tip_ts"] = pd.to_datetime(schedule_window["tip_ts"], utc=True, errors="coerce")
    player_map: dict[int, list[PlayerProfile]] = {}
    for profile in players:
        if profile.team_id is None:
            continue
        player_map.setdefault(int(profile.team_id), []).append(profile)
    records: list[dict] = []
    for row in schedule_window.itertuples():
        tip_ts = row.tip_ts
        tip_ts = pd.Timestamp(tip_ts) if pd.notna(tip_ts) else None
        if tip_ts is not None and tip_ts.tzinfo is None:
            tip_ts = tip_ts.tz_localize("UTC")
        for team_id in (row.home_team_id, row.away_team_id):
            roster_players = player_map.get(int(team_id))
            if not roster_players:
                continue
            as_of_ts = min(now, tip_ts) if tip_ts is not None else now
            for profile in roster_players:
                full_name = " ".join(part for part in (profile.first_name, profile.last_name) if part)
                records.append(
                    {
                        "game_id": int(row.game_id),
                        "team_id": int(team_id),
                        "game_date": pd.Timestamp(row.game_date).normalize(),
                        "player_id": int(profile.person_id),
                        "player_name": full_name or profile.player_slug,
                        "active_flag": True,
                        "starter_flag": False,
                        "listed_pos": profile.position,
                        "ingested_ts": now,
                        "source": source,
                        "as_of_ts": as_of_ts,
                    }
                )
    return pd.DataFrame(records, columns=list(ROSTER_NIGHTLY_RAW_SCHEMA.columns))


def _scrape_roster_poll(
    schedule: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    timeout: float,
) -> pd.DataFrame:
    scraper = NbaPlayersScraper(timeout=timeout)
    players = scraper.fetch_players(active_only=True)
    return _roster_from_players(schedule, players, start=start, end=end)


def _attach_schedule(roster: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    missing = set(REQUIRED_ROSTER_COLUMNS) - set(roster.columns)
    if missing:
        raise ValueError(f"Roster dataframe missing required columns: {', '.join(sorted(missing))}")
    schedule_missing = set(SCHEDULE_COLUMNS) - set(schedule.columns)
    if schedule_missing:
        raise ValueError(f"Schedule dataframe missing columns: {', '.join(sorted(schedule_missing))}")

    schedule_work = schedule.loc[:, SCHEDULE_COLUMNS].copy()
    schedule_work["tip_ts"] = pd.to_datetime(schedule_work["tip_ts"], utc=True)
    schedule_work["game_date"] = pd.to_datetime(schedule_work["game_date"]).dt.normalize()

    merged = roster.merge(schedule_work, on="game_id", how="left", validate="many_to_one")
    if merged["tip_ts"].isna().any():
        raise ValueError("Roster rows found without matching schedule/tip_ts.")

    if "game_date_y" in merged.columns:
        merged["game_date"] = merged["game_date_y"].fillna(merged.get("game_date_x"))
    elif "game_date_x" in merged.columns:
        merged["game_date"] = merged["game_date_x"]
    else:
        merged["game_date"] = pd.to_datetime(merged["tip_ts"]).dt.tz_convert(None)
    merged = merged.drop(columns=[col for col in ("game_date_x", "game_date_y") if col in merged.columns])
    return merged


def build_roster_snapshot(
    roster_raw: pd.DataFrame,
    schedule: pd.DataFrame,
    *,
    config: SnapshotConfig,
    lineups: pd.DataFrame | None = None,
) -> pd.DataFrame:
    working = _attach_schedule(roster_raw, schedule)
    working["game_date"] = pd.to_datetime(working["game_date"]).dt.normalize()
    lineup_frame = lineups.copy() if lineups is not None else pd.DataFrame()
    if not lineup_frame.empty:
        if "game_date" in lineup_frame.columns:
            date_series = lineup_frame["game_date"]
        elif "date" in lineup_frame.columns:
            date_series = lineup_frame["date"]
        else:
            raise KeyError("Lineup snapshot missing required date column (expected 'game_date' or 'date').")
        lineup_frame["game_date"] = pd.to_datetime(date_series).dt.normalize()
        if "as_of_ts" in lineup_frame.columns:
            asof_series = lineup_frame["as_of_ts"]
        elif "lineup_timestamp" in lineup_frame.columns:
            asof_series = lineup_frame["lineup_timestamp"]
        else:
            raise KeyError("Lineup snapshot missing as-of column (expected 'as_of_ts' or 'lineup_timestamp').")
        lineup_frame["as_of_ts"] = pd.to_datetime(asof_series, utc=True, errors="coerce")
        for column in ("game_id", "team_id", "player_id"):
            lineup_frame[column] = lineup_frame[column].astype("Int64")
            if column in working.columns:
                working[column] = working[column].astype("Int64")
        lineup_frame.sort_values(["game_id", "as_of_ts"], inplace=True)
        working = asof_left_join(
            working,
            lineup_frame,
            on=["game_id", "team_id", "player_id"],
            left_time_col="as_of_ts",
            right_time_col="as_of_ts",
        )
    else:
        missing_cols = set(OUTPUT_COLUMNS) - set(working.columns)
        for col in missing_cols:
            working[col] = pd.NA
    if config.start_date is not None:
        working = working[working["game_date"] >= config.start_date]
    if config.end_date is not None:
        working = working[working["game_date"] <= config.end_date]
    working = enforce_schema(working, ROSTER_NIGHTLY_SCHEMA, allow_missing_optional=True)
    validate_with_pandera(working, ROSTER_NIGHTLY_SCHEMA)
    return working


def _load_lineups_tree(root: Path, *, season: int, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    start_norm = start.normalize()
    end_norm = end.normalize()
    frames: list[pd.DataFrame] = []
    for cursor in storage.iter_days(start_norm, end_norm):
        partition = root / f"season={season}" / f"date={cursor.date():%Y-%m-%d}" / "lineups.parquet"
        if partition.exists():
            frames.append(pd.read_parquet(partition))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _default_silver_path(data_root: Path, season: int, month: int) -> Path:
    dest = ensure_directory(
        data_root
        / "silver"
        / "roster_nightly"
        / f"season={season}"
        / f"month={month:02d}"
    )
    return dest / "roster.parquet"


@app.command()
def main(
    roster: List[str] = typer.Option(
        [],
        "--roster",
        help="Input roster parquet(s) or glob patterns. When omitted, relies on NBA.com fallback.",
    ),
    schedule: List[str] = typer.Option(
        [],
        "--schedule",
        help="Schedule parquet(s) providing tip timestamps. Falls back to the live NBA API when empty.",
    ),
    start: datetime = typer.Option(..., help="Inclusive start date (YYYY-MM-DD)."),
    end: datetime = typer.Option(..., help="Inclusive end date (YYYY-MM-DD)."),
    season: int = typer.Option(..., help="Season used for output partitioning."),
    month: int = typer.Option(..., help="Month used for output partitioning."),
    data_root: Path = typer.Option(
        paths.get_data_root(),
        help="Data root containing silver outputs (defaults to PROJECTIONS_DATA_ROOT or ./data).",
    ),
    bronze_root: Path | None = typer.Option(
        None,
        "--bronze-root",
        help="Optional override for roster_nightly_raw bronze root (defaults to the standard contract).",
    ),
    bronze_out: Path | None = typer.Option(
        None,
        "--bronze-out",
        help="[deprecated] Write a single parquet instead of partitioned bronze outputs.",
    ),
    out: Path | None = typer.Option(None, help="Optional explicit roster snapshot parquet path."),
    lineups_dir: Path | None = typer.Option(
        None,
        "--lineups-dir",
        help="Optional override for normalized nba_daily_lineups parquet root (defaults to <data_root>/silver/nba_daily_lineups).",
    ),
    scrape_missing: bool = typer.Option(
        True,
        "--scrape-missing/--no-scrape-missing",
        help="Fetch NBA.com active roster data when --roster inputs resolve to zero rows.",
    ),
    roster_timeout: float = typer.Option(10.0, "--roster-timeout", help="NBA.com roster scraper timeout (seconds)."),
    schedule_timeout: float = typer.Option(10.0, "--schedule-timeout", help="Schedule API fallback timeout (seconds)."),
) -> None:
    start_day = pd.Timestamp(start).normalize()
    end_day = pd.Timestamp(end).normalize()
    if end_day < start_day:
        raise typer.BadParameter("--end must be on/after --start.")

    schedule_df = load_schedule_data(schedule, start_day, end_day, schedule_timeout)
    roster_df = pd.DataFrame()
    if roster:
        roster_df = _read_parquet_sources(roster, allow_empty=True)
    if roster_df.empty and scrape_missing:
        roster_df = _scrape_roster_poll(schedule_df, start=start_day, end=end_day, timeout=roster_timeout)
    if roster_df.empty:
        raise typer.BadParameter(
            "No roster rows were loaded. Provide --roster inputs or enable --scrape-missing."
        )

    roster_raw = enforce_schema(roster_df, ROSTER_NIGHTLY_RAW_SCHEMA)
    validate_with_pandera(roster_raw, ROSTER_NIGHTLY_RAW_SCHEMA)
    cfg = SnapshotConfig(start_date=start_day, end_date=end_day)
    schedule_slice = schedule_df.loc[:, ["game_id", "game_date", "tip_ts"]]
    lineup_root = (lineups_dir or (data_root / "silver" / "nba_daily_lineups")).resolve()
    if lineup_root.exists():
        lineup_df = _load_lineups_tree(lineup_root, season=season, start=start_day, end=end_day)
        typer.echo(
            f"[roster] loaded {len(lineup_df):,} lineup rows from {lineup_root}"
            if not lineup_df.empty
            else f"[roster] lineup partitions present at {lineup_root} but no rows matched window"
        )
    else:
        typer.echo(f"[roster] lineup directory {lineup_root} not found; continuing without lineup metadata.")
        lineup_df = pd.DataFrame()
    snapshot = build_roster_snapshot(roster_raw.copy(), schedule_slice, config=cfg, lineups=lineup_df)

    silver_path = out or _default_silver_path(data_root, season, month)
    silver_path.parent.mkdir(parents=True, exist_ok=True)
    bronze_root_path = (bronze_root or storage.default_bronze_root("roster_nightly_raw", data_root)).resolve()
    if bronze_out:
        bronze_out.parent.mkdir(parents=True, exist_ok=True)
        roster_raw.to_parquet(bronze_out, index=False)
        typer.echo(
            f"[roster] wrote {len(roster_raw):,} raw rows -> {bronze_out} (legacy bronze_out path)."
        )
    else:
        normalized_dates = pd.to_datetime(roster_raw["game_date"]).dt.normalize()
        for cursor in storage.iter_days(start_day, end_day):
            mask = normalized_dates == cursor
            if not mask.any():
                continue
            day_frame = roster_raw.loc[mask].copy()
            result = storage.write_bronze_partition(
                day_frame,
                dataset="roster_nightly_raw",
                data_root=data_root,
                season=season,
                target_date=cursor.date(),
                bronze_root=bronze_root_path,
            )
            typer.echo(
                f"[roster] bronze partition {result.target_date}: "
                f"{result.rows} rows -> {result.path}"
            )
    snapshot.to_parquet(silver_path, index=False)
    typer.echo(
        f"[roster] wrote {len(snapshot):,} snapshot rows -> {silver_path}"
    )


if __name__ == "__main__":  # pragma: no cover
    app()
