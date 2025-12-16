"""Shared helpers for bronze storage layout.

Legacy bronze path contract ("latest view"):

    <data_root>/bronze/<dataset>/season=<season>/date=<YYYY-MM-DD>/<filename>.parquet

This repo is migrating to append-only bronze history partitions while temporarily
keeping the legacy flat daily parquet as a backwards-compatible "latest view":

    <data_root>/bronze/<dataset>/season=<season>/date=<YYYY-MM-DD>/
      <filename>.parquet                            # legacy latest view (overwritten)
      hour=<HH>/<filename>.parquet                  # canonical history (injuries_raw)
      run_ts=<YYYYMMDDTHHMMSSZ>/<filename>.parquet  # canonical history (odds_raw)

Partition keys (date/hour) are interpreted in America/New_York for domain alignment.
Timestamps stored in parquet remain UTC.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import os
from pathlib import Path
from typing import Iterable
from uuid import uuid4

import pandas as pd

DEFAULT_BRONZE_FILENAMES: dict[str, str] = {
    "injuries_raw": "injuries.parquet",
    "odds_raw": "odds.parquet",
    "roster_nightly_raw": "roster.parquet",
    "daily_lineups": "daily_lineups_raw.parquet",
    "boxscores_raw": "boxscores_raw.parquet",
}


def iter_days(start: pd.Timestamp, end: pd.Timestamp) -> Iterable[pd.Timestamp]:
    """Yield normalized days from start to end, inclusive."""
    cursor = start.normalize()
    end_norm = end.normalize()
    while cursor <= end_norm:
        yield cursor
        cursor += pd.Timedelta(days=1)


def default_bronze_root(dataset: str, data_root: Path) -> Path:
    """Return the default bronze root for ``dataset`` under ``data_root``."""
    return (data_root / "bronze" / dataset).resolve()


def bronze_partition_dir(
    dataset: str,
    *,
    data_root: Path,
    season: int,
    target_date: date,
    bronze_root: Path | None = None,
) -> Path:
    """Return the partition directory for ``dataset`` on ``target_date``."""
    root = (bronze_root or default_bronze_root(dataset, data_root)).resolve()
    return root / f"season={season}" / f"date={target_date.isoformat()}"


def bronze_partition_path(
    dataset: str,
    *,
    data_root: Path,
    season: int,
    target_date: date,
    bronze_root: Path | None = None,
    filename: str | None = None,
) -> Path:
    """Return the parquet path for ``dataset`` on ``target_date``."""
    partition_dir = bronze_partition_dir(
        dataset,
        data_root=data_root,
        season=season,
        target_date=target_date,
        bronze_root=bronze_root,
    )
    output_name = filename or DEFAULT_BRONZE_FILENAMES.get(dataset, "data.parquet")
    return partition_dir / output_name


@dataclass
class BronzeWriteResult:
    dataset: str
    target_date: date
    path: Path
    rows: int


def _write_parquet_atomic(frame: pd.DataFrame, destination: Path) -> None:
    """Write a parquet file atomically (tmp + os.replace)."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        frame.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, destination)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def write_bronze_partition(
    frame: pd.DataFrame,
    *,
    dataset: str,
    data_root: Path,
    season: int,
    target_date: date,
    bronze_root: Path | None = None,
    filename: str | None = None,
) -> BronzeWriteResult:
    """Write ``frame`` to the bronze partition and return metadata."""
    destination = bronze_partition_path(
        dataset,
        data_root=data_root,
        season=season,
        target_date=target_date,
        bronze_root=bronze_root,
        filename=filename,
    )
    _write_parquet_atomic(frame, destination)
    return BronzeWriteResult(
        dataset=dataset,
        target_date=target_date,
        path=destination,
        rows=len(frame),
    )


def ensure_datetime(value: datetime | pd.Timestamp | str) -> pd.Timestamp:
    """Coerce the provided value into a timezone-aware UTC timestamp."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def write_bronze_partition_hourly(
    frame: pd.DataFrame,
    *,
    dataset: str,
    data_root: Path,
    season: int,
    target_date: date,
    hour: int,
    bronze_root: Path | None = None,
    filename: str | None = None,
) -> BronzeWriteResult:
    """Write an hourly bronze subpartition (append-only semantics)."""
    if hour < 0 or hour > 23:
        raise ValueError(f"hour must be in [0, 23], got {hour}")
    day_dir = bronze_partition_dir(
        dataset,
        data_root=data_root,
        season=season,
        target_date=target_date,
        bronze_root=bronze_root,
    )
    output_name = filename or DEFAULT_BRONZE_FILENAMES.get(dataset, "data.parquet")
    destination = day_dir / f"hour={hour:02d}" / output_name
    _write_parquet_atomic(frame, destination)
    return BronzeWriteResult(dataset=dataset, target_date=target_date, path=destination, rows=len(frame))


def write_bronze_partition_run(
    frame: pd.DataFrame,
    *,
    dataset: str,
    data_root: Path,
    season: int,
    target_date: date,
    run_ts: datetime | pd.Timestamp | str,
    bronze_root: Path | None = None,
    filename: str | None = None,
) -> BronzeWriteResult:
    """Write a per-run bronze subpartition keyed by run_ts (append-only semantics)."""
    run_slug = ensure_datetime(run_ts).strftime("%Y%m%dT%H%M%SZ")
    day_dir = bronze_partition_dir(
        dataset,
        data_root=data_root,
        season=season,
        target_date=target_date,
        bronze_root=bronze_root,
    )
    output_name = filename or DEFAULT_BRONZE_FILENAMES.get(dataset, "data.parquet")
    destination = day_dir / f"run_ts={run_slug}" / output_name
    _write_parquet_atomic(frame, destination)
    return BronzeWriteResult(dataset=dataset, target_date=target_date, path=destination, rows=len(frame))


def read_bronze_day(
    dataset: str,
    data_root: Path,
    season: int,
    target_date: date,
    *,
    include_runs: bool = True,
    prefer_history: bool = True,
    bronze_root: Path | None = None,
    filename: str | None = None,
) -> pd.DataFrame:
    """Read a day's bronze rows, preferring history partitions over the legacy flat file."""
    day_dir = bronze_partition_dir(
        dataset,
        data_root=data_root,
        season=season,
        target_date=target_date,
        bronze_root=bronze_root,
    )
    output_name = filename or DEFAULT_BRONZE_FILENAMES.get(dataset, "data.parquet")

    hourly_paths = sorted(day_dir.glob(f"hour=*/{output_name}"))
    run_paths = sorted(day_dir.glob(f"run_ts=*/{output_name}")) if include_runs else []
    history_paths = hourly_paths + run_paths

    paths: list[Path] = []
    if prefer_history and history_paths:
        paths = history_paths
    else:
        flat_file = day_dir / output_name
        if flat_file.exists():
            paths = [flat_file]

    if not paths:
        return pd.DataFrame()
    frames = [pd.read_parquet(path) for path in paths]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
