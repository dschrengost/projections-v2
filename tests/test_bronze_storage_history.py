from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from projections.etl import storage


def test_read_bronze_day_prefers_history_partitions_over_flat_file(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    season = 2025
    target_date = date(2025, 12, 7)

    flat = pd.DataFrame({"source": ["flat"]})
    hourly = pd.DataFrame({"source": ["hour"]})

    storage.write_bronze_partition(
        flat,
        dataset="injuries_raw",
        data_root=data_root,
        season=season,
        target_date=target_date,
    )
    storage.write_bronze_partition_hourly(
        hourly,
        dataset="injuries_raw",
        data_root=data_root,
        season=season,
        target_date=target_date,
        hour=18,
    )

    out = storage.read_bronze_day(
        "injuries_raw",
        data_root,
        season,
        target_date,
        include_runs=False,
        prefer_history=True,
    )
    assert out["source"].tolist() == ["hour"]


def test_read_bronze_day_falls_back_to_flat_file_when_no_history(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    season = 2025
    target_date = date(2025, 12, 7)

    flat = pd.DataFrame({"source": ["flat"]})
    storage.write_bronze_partition(
        flat,
        dataset="injuries_raw",
        data_root=data_root,
        season=season,
        target_date=target_date,
    )

    out = storage.read_bronze_day(
        "injuries_raw",
        data_root,
        season,
        target_date,
        include_runs=False,
        prefer_history=True,
    )
    assert out["source"].tolist() == ["flat"]


def test_read_bronze_day_include_runs_controls_run_ts_partitions(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    season = 2025
    target_date = date(2025, 12, 7)

    run_df = pd.DataFrame({"source": ["run"]})
    flat_df = pd.DataFrame({"source": ["flat"]})

    storage.write_bronze_partition_run(
        run_df,
        dataset="odds_raw",
        data_root=data_root,
        season=season,
        target_date=target_date,
        run_ts=datetime(2025, 12, 7, 10, 0, tzinfo=timezone.utc),
    )
    storage.write_bronze_partition(
        flat_df,
        dataset="odds_raw",
        data_root=data_root,
        season=season,
        target_date=target_date,
    )

    with_runs = storage.read_bronze_day(
        "odds_raw",
        data_root,
        season,
        target_date,
        include_runs=True,
        prefer_history=True,
    )
    assert with_runs["source"].tolist() == ["run"]

    without_runs = storage.read_bronze_day(
        "odds_raw",
        data_root,
        season,
        target_date,
        include_runs=False,
        prefer_history=True,
    )
    assert without_runs["source"].tolist() == ["flat"]
