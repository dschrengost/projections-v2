"""Snapshot helpers enforcing as-of semantics for Minutes V1."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Iterable
from zoneinfo import ZoneInfo

import pandas as pd

from projections.minutes_v1.constants import AvailabilityStatus
from projections.utils import asof_left_join


# Pattern to extract timestamp from NBA injury PDF URLs
# Format: Injury-Report_2026-01-19_05_00PM.pdf
_INJURY_REPORT_URL_PATTERN = re.compile(
    r"Injury-Report_(\d{4}-\d{2}-\d{2})_(\d{2})_(\d{2})(AM|PM)\.pdf$"
)


def _parse_report_ts_from_source(source: str | None) -> pd.Timestamp | None:
    """Extract report timestamp from NBA injury PDF URL.
    
    URL format: https://ak-static.cms.nba.com/referee/injury/Injury-Report_2026-01-19_05_00PM.pdf
    Returns a timezone-aware (ET) timestamp, or None if parsing fails.
    """
    if not source or not isinstance(source, str):
        return None
    match = _INJURY_REPORT_URL_PATTERN.search(source)
    if not match:
        return None
    date_str, hour_str, minute_str, ampm = match.groups()
    hour = int(hour_str)
    minute = int(minute_str)
    if ampm == "PM" and hour != 12:
        hour += 12
    elif ampm == "AM" and hour == 12:
        hour = 0
    try:
        et = ZoneInfo("America/New_York")
        dt = datetime.strptime(date_str, "%Y-%m-%d").replace(
            hour=hour, minute=minute, second=0, microsecond=0, tzinfo=et
        )
        return pd.Timestamp(dt).tz_convert("UTC")
    except (ValueError, OverflowError):
        return None


def ensure_as_of_column(
    df: pd.DataFrame,
    *,
    column: str = "as_of_ts",
    timestamp: datetime | str | None = None,
) -> pd.DataFrame:
    """Guarantee that the dataframe includes an ``as_of_ts`` column.

    Parameters
    ----------
    df:
        Input dataframe that may or may not already have the as-of column.
    column:
        Column name to enforce (defaults to ``as_of_ts``).
    timestamp:
        Optional timestamp applied when the column is missing. When omitted the
        current UTC timestamp is used.
    """

    result = df.copy()
    if column not in result.columns:
        ts_value = pd.Timestamp.utcnow() if timestamp is None else pd.to_datetime(timestamp, utc=True)
        result[column] = ts_value
    else:
        result[column] = pd.to_datetime(result[column], utc=True)
    return result


def latest_pre_tip_snapshot(
    df: pd.DataFrame,
    *,
    group_cols: Iterable[str],
    tip_ts_col: str,
    as_of_col: str = "as_of_ts",
) -> pd.DataFrame:
    """Select the latest snapshot with ``as_of`` ≤ ``tip_ts`` for each entity."""

    required_cols = set(group_cols) | {tip_ts_col, as_of_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    working = df.copy()
    working[tip_ts_col] = pd.to_datetime(working[tip_ts_col], utc=True)
    working[as_of_col] = pd.to_datetime(working[as_of_col], utc=True)

    left_cols = list(group_cols) + [tip_ts_col]
    left = working[left_cols].drop_duplicates(subset=list(group_cols)).copy()
    right_cols = list(group_cols) + [as_of_col]
    extra_cols = [col for col in working.columns if col not in set(right_cols + [tip_ts_col])]
    right = working[right_cols + extra_cols]

    merged = asof_left_join(
        left,
        right,
        on=list(group_cols),
        left_time_col=tip_ts_col,
        right_time_col=as_of_col,
    )
    merged = merged.dropna(subset=[as_of_col])
    return merged.drop(columns=[tip_ts_col]).reset_index(drop=True)


def select_latest_before(
    df: pd.DataFrame,
    cutoff_ts: datetime | str | pd.Timestamp,
    *,
    group_cols: Iterable[str],
    as_of_col: str = "as_of_ts",
    ingested_col: str = "ingested_ts",
) -> pd.DataFrame:
    """Select the latest row per group with ``as_of`` ≤ cutoff (ties by ``ingested_ts``).

    This is the core "time travel" primitive used by gold slate freezing:
      - Prefer ``as_of_ts`` as the semantic timestamp.
      - Fall back to ``ingested_ts`` only when ``as_of_ts`` is unavailable.
      - Break ties deterministically by sorting on ``(as_of_ts, ingested_ts)``.
    """

    if df.empty:
        return df.copy()

    missing = set(group_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required group columns: {', '.join(sorted(missing))}")

    cutoff = pd.to_datetime(cutoff_ts, utc=True)
    working = df.copy()

    added_cols: list[str] = []
    if as_of_col in working.columns:
        working[as_of_col] = pd.to_datetime(working[as_of_col], utc=True, errors="coerce")
    else:
        working[as_of_col] = pd.NaT
        added_cols.append(as_of_col)

    if ingested_col in working.columns:
        working[ingested_col] = pd.to_datetime(working[ingested_col], utc=True, errors="coerce")
    else:
        working[ingested_col] = pd.NaT
        added_cols.append(ingested_col)

    use_ingested = working[as_of_col].isna().all() and working[ingested_col].notna().any()
    primary_col = ingested_col if use_ingested else as_of_col

    eligible = working[primary_col].notna() & (working[primary_col] <= cutoff)
    filtered = working.loc[eligible].copy()
    if filtered.empty:
        return pd.DataFrame(columns=df.columns)

    sort_cols = list(group_cols) + [primary_col, ingested_col]
    filtered.sort_values(sort_cols, kind="mergesort", na_position="first", inplace=True)
    selected = filtered.groupby(list(group_cols), as_index=False).tail(1)
    if added_cols:
        selected = selected.drop(columns=added_cols, errors="ignore")
    return selected.reset_index(drop=True)


def select_injury_snapshot(
    df: pd.DataFrame,
    *,
    group_cols: Iterable[str] = ("game_id", "player_id"),
    tip_ts_col: str = "tip_ts",
    as_of_col: str = "as_of_ts",
    report_ts_col: str = "report_ts",
) -> pd.DataFrame:
    """Strict injury snapshot selection with placeholder rows for missing pre-tip data.
    
    Uses report_ts (actual PDF report timestamp) for ordering when available,
    falling back to as_of_ts. This ensures that when a player's status changes
    (e.g., Questionable -> Available), the most recent report is selected.
    """

    required = set(group_cols) | {tip_ts_col, as_of_col, "status", "restriction_flag", "ramp_flag"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    working = df.copy()
    working[tip_ts_col] = pd.to_datetime(working[tip_ts_col], utc=True)
    working[as_of_col] = pd.to_datetime(working[as_of_col], utc=True, errors="coerce")
    
    # Use report_ts for ordering and pre-tip eligibility if available, else try to
    # derive it from the source URL. The raw injuries feed may be ingested after
    # tip even when the underlying PDF report was published on time.
    working["_effective_as_of_ts"] = working[as_of_col]
    has_report_ts = report_ts_col in working.columns and working[report_ts_col].notna().any()
    if has_report_ts:
        working[report_ts_col] = pd.to_datetime(working[report_ts_col], utc=True, errors="coerce")
        working["_effective_as_of_ts"] = working[report_ts_col].fillna(working["_effective_as_of_ts"])
    elif "source" in working.columns:
        # Derive report_ts from source URL for backwards compatibility.
        working["_derived_report_ts"] = working["source"].apply(_parse_report_ts_from_source)
        working["_effective_as_of_ts"] = working["_derived_report_ts"].fillna(working["_effective_as_of_ts"])

    valid_mask = working["_effective_as_of_ts"].notna() & (working["_effective_as_of_ts"] <= working[tip_ts_col])
    valid = working.loc[valid_mask].copy()

    latest = pd.DataFrame(columns=working.columns)
    if not valid.empty:
        # Select the entry with the latest effective report/as-of time per player/game.
        latest_idx = valid.groupby(list(group_cols))["_effective_as_of_ts"].idxmax()
        latest = valid.loc[latest_idx].copy()
        latest[as_of_col] = latest["_effective_as_of_ts"]
        latest["selection_rule"] = "latest_leq_tip"
        latest["snapshot_missing"] = 0

    all_keys = working[list(group_cols)].drop_duplicates()
    if latest.empty:
        selected_keys = all_keys.iloc[0:0]
    else:
        selected_keys = latest[list(group_cols)].drop_duplicates()
    missing_keys = all_keys.merge(selected_keys, on=list(group_cols), how="left", indicator=True)
    missing_keys = missing_keys[missing_keys["_merge"] == "left_only"].drop(columns="_merge")

    placeholders: list[dict[str, object]] = []
    for key in missing_keys.itertuples(index=False):
        placeholder: dict[str, object] = {col: getattr(key, col) for col in group_cols}
        placeholder.update(
            {
                as_of_col: pd.NaT,
                "status": AvailabilityStatus.UNKNOWN.value,
                "restriction_flag": False,
                "ramp_flag": False,
                "games_since_return": pd.NA,
                "days_since_return": pd.NA,
                "ingested_ts": pd.NaT,
                "source": "missing_pre_tip_snapshot",
                "selection_rule": "no_pre_tip_snapshot",
                "snapshot_missing": 1,
            }
        )
        placeholders.append(placeholder)

    placeholder_df = pd.DataFrame(placeholders, columns=list(group_cols) + [
        as_of_col,
        "status",
        "restriction_flag",
        "ramp_flag",
        "games_since_return",
        "days_since_return",
        "ingested_ts",
        "source",
        "selection_rule",
        "snapshot_missing",
    ])

    combined = pd.concat([latest, placeholder_df], ignore_index=True, sort=False)
    combined["snapshot_missing"] = combined["snapshot_missing"].fillna(0).astype(int)

    if (combined[as_of_col].notna() & (combined[as_of_col] > combined[tip_ts_col])).any():
        raise AssertionError("Detected injury snapshots with as_of_ts after tip_ts")

    return combined.drop(columns=[tip_ts_col, "_effective_as_of_ts", "_derived_report_ts"], errors="ignore")
