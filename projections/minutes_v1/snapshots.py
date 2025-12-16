"""Snapshot helpers enforcing as-of semantics for Minutes V1."""

from __future__ import annotations

from datetime import datetime
from typing import Iterable

import pandas as pd

from projections.minutes_v1.constants import AvailabilityStatus
from projections.utils import asof_left_join


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
) -> pd.DataFrame:
    """Strict injury snapshot selection with placeholder rows for missing pre-tip data."""

    required = set(group_cols) | {tip_ts_col, as_of_col, "status", "restriction_flag", "ramp_flag"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    working = df.copy()
    working[tip_ts_col] = pd.to_datetime(working[tip_ts_col], utc=True)
    working[as_of_col] = pd.to_datetime(working[as_of_col], utc=True, errors="coerce")

    valid_mask = working[as_of_col].notna() & (working[as_of_col] <= working[tip_ts_col])
    valid = working.loc[valid_mask].copy()

    latest = pd.DataFrame(columns=working.columns)
    if not valid.empty:
        latest_idx = valid.groupby(list(group_cols))[as_of_col].idxmax()
        latest = valid.loc[latest_idx].copy()
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

    return combined.drop(columns=[tip_ts_col])
