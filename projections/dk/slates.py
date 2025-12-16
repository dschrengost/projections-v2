from __future__ import annotations

"""DraftKings slate discovery and filtering."""

import datetime as dt
import re
from typing import Dict, Literal, Optional

import pandas as pd
from zoneinfo import ZoneInfo

from .api import fetch_nba_contests

SlateType = Literal["main", "night", "turbo", "early", "showdown", "all"]

_EASTERN = ZoneInfo("America/New_York")


def _infer_slate_type_from_name(name: str) -> str:
    name_lower = name.lower()
    if "turbo" in name_lower:
        return "turbo"
    if "late" in name_lower or "night" in name_lower:
        return "night"
    if "early" in name_lower:
        return "early"
    if "showdown" in name_lower or "single game" in name_lower:
        return "showdown"
    return "main"


def _parse_start_datetime(value: object) -> Optional[dt.datetime]:
    if value is None:
        return None
    if isinstance(value, str):
        match = re.search(r"/Date\((\d+)\)/", value)
        if match:
            millis = int(match.group(1))
            return dt.datetime.fromtimestamp(millis / 1000, tz=dt.timezone.utc)
        try:
            cleaned = value.replace("Z", "+00:00")
            parsed = dt.datetime.fromisoformat(cleaned)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=dt.timezone.utc)
            return parsed
        except ValueError:
            return None
    if isinstance(value, (int, float)):
        try:
            millis = float(value)
        except (TypeError, ValueError):
            return None
        if pd.isna(millis):
            return None
        if millis > 1e12:
            millis /= 1000
        return dt.datetime.fromtimestamp(millis, tz=dt.timezone.utc)
    return None


def _resolve_start_times(df: pd.DataFrame) -> pd.Series:
    candidate_columns = (
        "sd",
        "StartDate",
        "startDate",
        "sdUtc",
        "ssd",
        "startTime",
        "StartTime",
    )
    for column in candidate_columns:
        if column not in df.columns:
            continue
        parsed = df[column].apply(_parse_start_datetime)
        if parsed.notna().any():
            normalized = pd.to_datetime(parsed, utc=True)
            return normalized
    raise RuntimeError("Unable to parse contest start times; DraftKings payload changed?")


def _resolve_name_column(df: pd.DataFrame) -> str:
    for candidate in ("n", "ContestName", "contestName", "Name", "name"):
        if candidate in df.columns:
            return candidate
    raise RuntimeError("Contest name column not found; DraftKings payload changed?")


def load_contests_df(contests_payload: Optional[Dict[str, object]] = None) -> pd.DataFrame:
    raw = contests_payload if contests_payload is not None else fetch_nba_contests()
    contests = raw.get("Contests", []) if isinstance(raw, dict) else []
    if not isinstance(contests, list):
        raise RuntimeError("Unexpected contests payload; 'Contests' is not a list")
    df = pd.DataFrame(contests)
    if df.empty:
        return df
    if "dg" not in df.columns:
        raise RuntimeError("No 'dg' field in contests; DraftKings payload changed?")
    return df


def list_draft_groups_for_date(
    game_date: str,
    slate_type: SlateType = "all",
    *,
    contests_payload: Optional[Dict[str, object]] = None,
) -> pd.DataFrame:
    """
    Return a DataFrame with one row per (draft_group_id, slate_type).

    Columns include:
      - game_date (YYYY-MM-DD, America/New_York)
      - draft_group_id (int)
      - slate_type (str)
      - n_contests
      - earliest_start
      - latest_start
      - example_contest_name
    """

    try:
        target_date = dt.date.fromisoformat(game_date)
    except ValueError as exc:
        raise ValueError(f"Invalid game_date format (expected YYYY-MM-DD): {game_date}") from exc

    df = load_contests_df(contests_payload)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "game_date",
                "slate_type",
                "draft_group_id",
                "n_contests",
                "earliest_start",
                "latest_start",
                "example_contest_name",
            ]
        )

    start_times_utc = _resolve_start_times(df)
    start_times_local = start_times_utc.dt.tz_convert(_EASTERN)

    name_column = _resolve_name_column(df)
    df = df.assign(
        start_time=start_times_local,
        contest_name=df[name_column].astype(str),
        draft_group_id=pd.to_numeric(df["dg"], errors="coerce").astype("Int64"),
    )

    df = df[df["start_time"].notna() & df["draft_group_id"].notna()].copy()
    df.loc[:, "game_date"] = df["start_time"].dt.date
    df.loc[:, "slate_type"] = df["contest_name"].apply(_infer_slate_type_from_name)

    df = df[df["game_date"] == target_date]
    if slate_type != "all":
        df = df[df["slate_type"] == slate_type]

    if df.empty:
        return pd.DataFrame(
            columns=[
                "game_date",
                "slate_type",
                "draft_group_id",
                "n_contests",
                "earliest_start",
                "latest_start",
                "example_contest_name",
            ]
        )

    grouped = (
        df.groupby(["draft_group_id", "slate_type"], as_index=False)
        .agg(
            n_contests=("draft_group_id", "size"),
            earliest_start=("start_time", "min"),
            latest_start=("start_time", "max"),
            example_contest_name=("contest_name", "first"),
        )
        .assign(game_date=target_date)
    )

    ordered_columns = [
        "game_date",
        "slate_type",
        "draft_group_id",
        "n_contests",
        "earliest_start",
        "latest_start",
        "example_contest_name",
    ]

    result = grouped[ordered_columns].sort_values(
        by=["earliest_start", "draft_group_id"]
    )
    return result.reset_index(drop=True)

