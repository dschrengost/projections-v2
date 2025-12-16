"""Helpers for building snapshot-rich minutes training datasets.

This module exists to convert live feature snapshots (many `feature_as_of_ts`
values per game/player) into a labeled training table by joining gold minutes
labels onto every snapshot row.

The resulting parquet is the expected input to `projections/cli/train_minutes_dual.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd

LIVE_FEATURE_FILENAME = "features.parquet"
LABEL_FILENAME = "labels.parquet"


def season_start_year(day: date) -> int:
    """Return NBA season start year (Aug–Jul) for a calendar day."""

    return int(day.year if day.month >= 8 else day.year - 1)


def iter_days(start: date, end: date) -> Iterable[date]:
    cursor = start
    while cursor <= end:
        yield cursor
        cursor = cursor.fromordinal(cursor.toordinal() + 1)


def discover_live_feature_paths(
    live_root: Path,
    *,
    start_date: date,
    end_date: date,
    max_runs_per_day: int | None = None,
) -> list[Path]:
    """Discover `features.parquet` files under `live/features_minutes_v1/<day>/run=*/`."""

    if max_runs_per_day is not None and max_runs_per_day <= 0:
        raise ValueError("max_runs_per_day must be positive when provided")

    paths: list[Path] = []
    for day in iter_days(start_date, end_date):
        day_dir = live_root / day.isoformat()
        if not day_dir.exists():
            continue

        direct = day_dir / LIVE_FEATURE_FILENAME
        if direct.exists():
            paths.append(direct)

        run_paths = sorted(day_dir.glob(f"run=*/{LIVE_FEATURE_FILENAME}"))
        if max_runs_per_day is not None and len(run_paths) > max_runs_per_day:
            run_paths = run_paths[-max_runs_per_day:]
        paths.extend(run_paths)

    return paths


@dataclass(frozen=True)
class LabelDiscovery:
    paths: list[Path]
    missing_days: list[str]


def discover_label_paths(
    labels_root: Path,
    *,
    start_date: date,
    end_date: date,
) -> LabelDiscovery:
    """Discover gold label partitions under `gold/labels_minutes_v1/season=*/game_date=*/labels.parquet`."""

    paths: list[Path] = []
    missing: list[str] = []
    for day in iter_days(start_date, end_date):
        season = season_start_year(day)
        path = labels_root / f"season={season}" / f"game_date={day.isoformat()}" / LABEL_FILENAME
        if path.exists():
            paths.append(path)
        else:
            missing.append(day.isoformat())
    return LabelDiscovery(paths=paths, missing_days=missing)


def _parse_run_id_token(token: str) -> pd.Timestamp | None:
    raw = token.strip()
    if not raw:
        return None
    # Expected format: 20251203T171013Z
    ts = pd.to_datetime(raw, utc=True, errors="coerce", format="%Y%m%dT%H%M%SZ")
    if pd.isna(ts):
        ts = pd.to_datetime(raw, utc=True, errors="coerce")
    return None if pd.isna(ts) else ts


def ingested_ts_from_path(path: Path) -> pd.Timestamp | None:
    """Extract an ingested timestamp from a `.../run=<ts>/...` path when available."""

    for part in path.parts:
        if part.startswith("run="):
            return _parse_run_id_token(part.split("=", 1)[1])
        if part.startswith("run_ts="):
            return _parse_run_id_token(part.split("=", 1)[1])
    return None


def load_parquet_files(paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        frames.append(pd.read_parquet(path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def load_live_feature_snapshots(
    paths: list[Path],
    *,
    ingested_ts_col: str = "ingested_ts",
) -> pd.DataFrame:
    """Load live feature parquet files and add `ingested_ts` based on run directory."""

    frames: list[pd.DataFrame] = []
    for path in paths:
        df = pd.read_parquet(path)
        ts = ingested_ts_from_path(path)
        if ts is not None and ingested_ts_col not in df.columns:
            df = df.copy()
            df[ingested_ts_col] = ts
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    if ingested_ts_col in combined.columns:
        combined[ingested_ts_col] = pd.to_datetime(combined[ingested_ts_col], utc=True, errors="coerce")
    return combined


def merge_features_with_labels(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    key_cols: Iterable[str] = ("game_id", "player_id", "team_id"),
    minutes_col: str = "minutes",
    starter_label_col: str = "starter_flag_label",
) -> pd.DataFrame:
    """Left-join gold labels onto snapshot features and use label minutes as target."""

    keys = list(key_cols)
    missing_features = set(keys) - set(features.columns)
    if missing_features:
        raise ValueError(f"Snapshot features missing join keys: {', '.join(sorted(missing_features))}")
    missing_labels = set(keys) - set(labels.columns)
    if missing_labels:
        raise ValueError(f"Labels missing join keys: {', '.join(sorted(missing_labels))}")

    feat = features.copy()
    for col in keys:
        feat[col] = pd.to_numeric(feat[col], errors="coerce").astype("Int64")
    feat = feat.dropna(subset=keys).copy()

    lab = labels.copy()
    for col in keys:
        lab[col] = pd.to_numeric(lab[col], errors="coerce").astype("Int64")
    lab = lab.dropna(subset=keys).copy()
    if "label_frozen_ts" in lab.columns:
        lab["label_frozen_ts"] = pd.to_datetime(lab["label_frozen_ts"], utc=True, errors="coerce")
        lab = lab.sort_values(keys + ["label_frozen_ts"], kind="mergesort")
    lab = lab.drop_duplicates(subset=keys, keep="last")

    keep_cols = []
    for col in (minutes_col, starter_label_col, "starter_flag", "label_frozen_ts"):
        if col in lab.columns:
            keep_cols.append(col)
    if minutes_col not in keep_cols:
        raise ValueError(f"Labels missing required target column '{minutes_col}'.")

    lab = lab[keys + keep_cols]

    # Avoid name clashes if live features already include the target column.
    if minutes_col in feat.columns:
        feat = feat.drop(columns=[minutes_col])

    merged = feat.merge(lab, on=keys, how="left", validate="many_to_one")
    merged[minutes_col] = pd.to_numeric(merged[minutes_col], errors="coerce")
    if starter_label_col in merged.columns:
        merged[starter_label_col] = pd.to_numeric(merged[starter_label_col], errors="coerce").astype("Int64")
    if "label_frozen_ts" in merged.columns:
        merged["label_frozen_ts"] = pd.to_datetime(merged["label_frozen_ts"], utc=True, errors="coerce")
    return merged


def normalize_snapshot_timestamps(
    df: pd.DataFrame,
    *,
    tip_ts_col: str = "tip_ts",
    feature_as_of_col: str = "feature_as_of_ts",
) -> pd.DataFrame:
    """Coerce tip + as-of timestamps to UTC pandas Timestamps."""

    working = df.copy()
    if tip_ts_col in working.columns:
        working[tip_ts_col] = pd.to_datetime(working[tip_ts_col], utc=True, errors="coerce")
    if feature_as_of_col in working.columns:
        working[feature_as_of_col] = pd.to_datetime(working[feature_as_of_col], utc=True, errors="coerce")
    return working


def filter_to_labeled_rows(
    df: pd.DataFrame,
    *,
    minutes_col: str = "minutes",
) -> pd.DataFrame:
    """Drop rows where minutes labels are missing."""

    if df.empty:
        return df.copy()
    if minutes_col not in df.columns:
        raise ValueError(f"Expected '{minutes_col}' column for labeled filtering.")
    return df.loc[df[minutes_col].notna()].copy()
