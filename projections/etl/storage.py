"""Shared helpers for bronze storage layout.

Bronze path contract:

    <data_root>/bronze/<dataset>/season=<season>/date=<YYYY-MM-DD>/<filename>.parquet

Each partition stores a single day's worth of rows for the given dataset. The default
filename per dataset is defined in ``DEFAULT_BRONZE_FILENAMES``. Callers can override
the bronze root (the directory that replaces ``<data_root>/bronze/<dataset>``) when
debugging or writing to temporary locations.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

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
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(destination, index=False)
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
