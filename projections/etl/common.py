"""Shared helpers for ETL CLIs (schedule loading, partitions, etc.)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List

import pandas as pd
from zoneinfo import ZoneInfo

from scrapers.nba_schedule import NbaScheduleScraper


REQUIRED_SCHEDULE_COLUMNS: tuple[str, ...] = (
    "game_id",
    "game_code",
    "season",
    "game_date",
    "tip_ts",
    "home_team_id",
    "home_team_name",
    "home_team_city",
    "home_team_tricode",
    "away_team_id",
    "away_team_name",
    "away_team_city",
    "away_team_tricode",
)

LOCAL_RUN_TZ = ZoneInfo("America/New_York")


def month_slug(day: pd.Timestamp) -> str:
    """Return the lowercase month abbreviation for partition naming."""

    return day.strftime("%b").lower()


def _resolve_schedule_paths(parquet_paths: Iterable[str]) -> List[Path]:
    resolved: list[Path] = []
    for pattern in parquet_paths:
        path = Path(pattern)
        if path.is_file():
            resolved.append(path)
            continue
        parent = path.parent if path.parent != Path("") else Path(".")
        resolved.extend(parent.glob(path.name))
    return resolved


def read_schedule(parquet_paths: Iterable[str]) -> pd.DataFrame:
    """Load one or more parquet files containing the schedule silver dataset."""

    files = _resolve_schedule_paths(parquet_paths)
    if not files:
        raise FileNotFoundError("No schedule parquet files found for provided patterns.")
    frames = [pd.read_parquet(file) for file in files]
    df = pd.concat(frames, ignore_index=True)
    missing = set(REQUIRED_SCHEDULE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Schedule dataframe missing columns: {', '.join(sorted(missing))}")
    return df


def schedule_from_api(start: pd.Timestamp, end: pd.Timestamp, timeout: float) -> pd.DataFrame:
    """Fetch schedule rows from the live NBA API for the requested window."""

    scraper = NbaScheduleScraper(timeout=timeout)
    season_label = f"{start.year}-{(start.year + 1) % 100:02d}"
    records: list[dict] = []
    current = start
    while current <= end:
        # Don't pass season param - that triggers mobile schedule which misses NBA Cup games
        games = scraper.fetch_daily_schedule(current.date())
        for game in games:
            if not game.game_id:
                continue
            tip_source = game.game_time_utc or game.local_game_date
            if isinstance(tip_source, datetime):
                base_dt = tip_source
            elif tip_source is not None:
                base_dt = datetime.combine(pd.Timestamp(tip_source).date(), datetime.min.time())
            else:
                base_dt = current.to_pydatetime()
            if base_dt.tzinfo is None:
                base_dt = base_dt.replace(tzinfo=timezone.utc)
            else:
                base_dt = base_dt.astimezone(timezone.utc)
            tip_ts = pd.Timestamp(base_dt)
            tip_day = tip_ts.tz_convert(LOCAL_RUN_TZ).tz_localize(None).normalize()
            home = game.home_team
            away = game.away_team
            if not (home and away):
                continue
            local_day = (
                pd.Timestamp(game.local_game_date).normalize()
                if game.local_game_date
                else tip_day
            )
            records.append(
                {
                    "game_id": int(game.game_id),
                    "game_code": game.game_code,
                    "season": game.season_year or f"{start.year}-{(start.year + 1) % 100:02d}",
                    "game_date": local_day,
                    "tip_day": tip_day,
                    "local_game_date": local_day,
                    "tip_ts": tip_ts,
                    "home_team_id": home.team_id,
                    "home_team_name": home.team_name,
                    "home_team_city": home.team_city,
                    "home_team_tricode": home.team_tricode,
                    "away_team_id": away.team_id,
                    "away_team_name": away.team_name,
                    "away_team_city": away.team_city,
                    "away_team_tricode": away.team_tricode,
                    "arena_id": None,
                    "arena_name": None,
                    "arena_city": None,
                    "arena_state": None,
                }
            )
        current += pd.Timedelta(days=1)
    if not records:
        raise RuntimeError("NBA schedule API did not return any games for requested window.")
    return pd.DataFrame(records)


def normalize_schedule_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure timestamp columns are timezone-aware and add helper columns."""

    tip_ts = pd.to_datetime(df["tip_ts"], errors="coerce", utc=True)
    df = df.copy()
    df["tip_ts"] = tip_ts
    local_tip = tip_ts.dt.tz_convert(LOCAL_RUN_TZ)
    df["tip_local_ts"] = local_tip.dt.tz_localize(None)
    df["tip_day"] = local_tip.dt.normalize()
    if "game_date" in df.columns:
        game_dates = pd.to_datetime(df["game_date"], errors="coerce")
        if getattr(game_dates.dtype, "tz", None) is not None:
            game_dates = game_dates.dt.tz_convert(LOCAL_RUN_TZ).dt.tz_localize(None)
        df["game_date"] = game_dates.dt.normalize()
    else:
        df["game_date"] = df["tip_day"]
    if "local_game_date" in df.columns:
        local_dates = pd.to_datetime(df["local_game_date"], errors="coerce")
        if getattr(local_dates.dtype, "tz", None) is not None:
            local_dates = local_dates.dt.tz_convert(LOCAL_RUN_TZ).dt.tz_localize(None)
        df["local_game_date"] = local_dates.dt.normalize()
    df["game_id"] = pd.to_numeric(df["game_id"], errors="coerce").astype("Int64")
    return df


def load_schedule_data(
    schedule_paths: Iterable[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    timeout: float,
) -> pd.DataFrame:
    """Load schedule data from parquet with API fallback when globs are empty."""

    def _dedupe(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.drop_duplicates(subset="game_id", keep="last")
        frame.sort_values("tip_ts", inplace=True)
        frame.reset_index(drop=True, inplace=True)
        return frame

    if schedule_paths:
        try:
            df = read_schedule(schedule_paths)
        except FileNotFoundError:
            df = pd.DataFrame()
        if not df.empty:
            df = normalize_schedule_frame(df)
            game_dates = df["game_date"]
            if getattr(game_dates.dtype, "tz", None) is not None:
                game_dates = game_dates.dt.tz_convert(LOCAL_RUN_TZ).dt.tz_localize(None)
            mask = (game_dates >= start) & (game_dates <= end)
            if "local_game_date" in df.columns:
                local_dates = df["local_game_date"]
                if getattr(local_dates.dtype, "tz", None) is not None:
                    local_dates = local_dates.dt.tz_convert(LOCAL_RUN_TZ).dt.tz_localize(None)
                mask |= (local_dates >= start) & (local_dates <= end)
            tip_days = df["tip_day"]
            if getattr(tip_days.dtype, "tz", None) is not None:
                tip_days = tip_days.dt.tz_convert(LOCAL_RUN_TZ).dt.tz_localize(None)
            mask |= (tip_days >= start) & (tip_days <= end)
            slice_df = df.loc[mask].copy()
            if not slice_df.empty:
                return _dedupe(slice_df)

    api_df = schedule_from_api(start, end, timeout)
    normalized = normalize_schedule_frame(api_df)
    game_dates = normalized["game_date"]
    if getattr(game_dates.dtype, "tz", None) is not None:
        game_dates = game_dates.dt.tz_convert(LOCAL_RUN_TZ).dt.tz_localize(None)
    mask = (game_dates >= start) & (game_dates <= end)
    tip_days = normalized["tip_day"]
    if getattr(tip_days.dtype, "tz", None) is not None:
        tip_days = tip_days.dt.tz_convert(LOCAL_RUN_TZ).dt.tz_localize(None)
    mask |= (tip_days >= start) & (tip_days <= end)
    if "local_game_date" in normalized.columns:
        local_dates = normalized["local_game_date"]
        if getattr(local_dates.dtype, "tz", None) is not None:
            local_dates = local_dates.dt.tz_convert(LOCAL_RUN_TZ).dt.tz_localize(None)
        mask |= (local_dates >= start) & (local_dates <= end)
    normalized = normalized.loc[mask].copy()
    if normalized.empty:
        raise RuntimeError("Schedule filter removed all rows for requested window.")
    return _dedupe(normalized)


__all__ = [
    "month_slug",
    "load_schedule_data",
    "read_schedule",
    "schedule_from_api",
    "normalize_schedule_frame",
]
