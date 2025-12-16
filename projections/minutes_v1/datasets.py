"""Feature dataset loading and slicing utilities for Minutes V1."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

KEY_COLUMNS: tuple[str, str, str] = ("game_id", "player_id", "team_id")


def _read_parquet_tree(path: Path) -> pd.DataFrame:
    """Read a parquet file or directory (recursively) into a dataframe."""

    if not path.exists():
        raise FileNotFoundError(f"Missing parquet input at {path}")
    if path.is_file():
        return pd.read_parquet(path)
    files = sorted(path.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files discovered under {path}")
    frames = [pd.read_parquet(file) for file in files]
    return pd.concat(frames, ignore_index=True)


def _normalize_date(value: datetime | str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.normalize()


def default_features_path(data_root: Path, season: int, month: int) -> Path:
    """Return the canonical gold feature path for a season/month partition."""

    return (
        data_root
        / "gold"
        / "features_minutes_v1"
        / f"season={season}"
        / f"month={month:02d}"
        / "features.parquet"
    )


def load_feature_frame(
    *,
    features_path: Path | None,
    data_root: Path,
    season: int | None,
    month: int | None,
) -> pd.DataFrame:
    """Load the feature parquet for training/inference.

    If `features_path` is provided it takes precedence; otherwise the season/month
    partition must be supplied so the canonical path can be derived.
    """

    if features_path is None:
        if season is None or month is None:
            raise ValueError("Either --features or both --season/--month must be provided")
        features_path = default_features_path(data_root, season, month)
    return _read_parquet_tree(features_path)


@dataclass(frozen=True)
class TimeSplit:
    """Configuration for blocked train/validation splits."""

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp

    @classmethod
    def from_args(
        cls,
        *,
        train_start: datetime,
        train_end: datetime,
        val_end: datetime,
        val_start: datetime | None = None,
    ) -> "TimeSplit":
        start = _normalize_date(train_start)
        end = _normalize_date(train_end)
        if start > end:
            raise ValueError("train_start must be on/before train_end")
        val_end_norm = _normalize_date(val_end)
        if val_start is None:
            val_start_norm = end + timedelta(days=1)
        else:
            val_start_norm = _normalize_date(val_start)
        if val_start_norm > val_end_norm:
            raise ValueError("val_start must be on/before val_end")
        if val_start_norm <= end:
            raise ValueError("Validation window must follow the training window")
        return cls(start, end, val_start_norm, val_end_norm)


def apply_time_split(df: pd.DataFrame, split: TimeSplit, *, date_column: str = "game_date") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Slice a dataframe into train/validation windows based on game_date."""

    if date_column not in df:
        raise ValueError(f"Dataframe missing '{date_column}' column for time split")
    working = df.copy()
    working[date_column] = pd.to_datetime(working[date_column]).dt.normalize()
    train_mask = (working[date_column] >= split.train_start) & (working[date_column] <= split.train_end)
    val_mask = (working[date_column] >= split.val_start) & (working[date_column] <= split.val_end)
    train_df = working.loc[train_mask].copy()
    val_df = working.loc[val_mask].copy()
    if train_df.empty:
        raise ValueError("Training window produced zero rows — check date bounds")
    if val_df.empty:
        raise ValueError("Validation window produced zero rows — check date bounds")
    return train_df, val_df


def ensure_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")


def deduplicate_latest(
    df: pd.DataFrame,
    *,
    key_cols: Sequence[str] = KEY_COLUMNS,
    order_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return a dataframe with the latest row per key based on ``order_cols``."""

    if not set(key_cols).issubset(df.columns):
        raise ValueError(f"Cannot deduplicate without key columns: {', '.join(key_cols)}")
    sort_cols: list[str] = list(key_cols)
    if order_cols:
        sort_cols.extend([col for col in order_cols if col in df.columns])
    sorted_df = df.sort_values(sort_cols, kind="mergesort")
    deduped = sorted_df.drop_duplicates(subset=list(key_cols), keep="last")
    if deduped.duplicated(subset=list(key_cols)).any():
        raise AssertionError("Failed to enforce unique keys after deduplication")
    return deduped.reset_index(drop=True)


def write_ids_csv(df: pd.DataFrame, path: Path, *, key_cols: Sequence[str] = KEY_COLUMNS) -> None:
    ids = df.loc[:, list(key_cols)].drop_duplicates().copy()
    if "game_id" in ids.columns:
        ids["game_id"] = ids["game_id"].astype(str)
    path.parent.mkdir(parents=True, exist_ok=True)
    ids.to_csv(path, index=False)


def left_anti_keys(
    expected: pd.DataFrame,
    actual: pd.DataFrame,
    *,
    key_cols: Sequence[str] = KEY_COLUMNS,
) -> pd.DataFrame:
    """Return keys present in ``expected`` but missing from ``actual``."""

    expected_keys = expected.loc[:, list(key_cols)].drop_duplicates().copy()
    actual_keys = actual.loc[:, list(key_cols)].drop_duplicates().copy()
    if "game_id" in expected_keys.columns:
        expected_keys["game_id"] = expected_keys["game_id"].astype(str)
    if "game_id" in actual_keys.columns:
        actual_keys["game_id"] = actual_keys["game_id"].astype(str)
    merged = expected_keys.merge(actual_keys, on=list(key_cols), how="left", indicator=True)
    return merged[merged["_merge"] == "left_only"].drop(columns="_merge")


def schedule_game_ids_in_range(
    schedule: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    tip_col: str = "tip_ts",
) -> pd.Series:
    """Return schedule game_ids whose tip date (UTC) falls within [start, end]."""

    ensure_columns(schedule, {"game_id", tip_col})
    tip_dates = pd.to_datetime(schedule[tip_col], utc=True).dt.tz_convert(None).dt.normalize()
    mask = (tip_dates >= start) & (tip_dates <= end)
    return schedule.loc[mask, "game_id"].drop_duplicates()
