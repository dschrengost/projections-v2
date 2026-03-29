from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def _write_boxscore_day(root: Path, season_year: int, day: str) -> None:
    _touch(
        root
        / "bronze"
        / "boxscores_raw"
        / f"season={season_year}"
        / f"date={day}"
        / "boxscores_raw.parquet"
    )


def _write_tracking_day(root: Path, season_id: str, day: str, measures: list[str]) -> None:
    for measure in measures:
        _touch(
            root
            / "bronze"
            / "nba"
            / "tracking"
            / f"season={season_id}"
            / "season_type=Regular+Season"
            / f"game_date={day}"
            / f"pt_measure_type={measure}"
            / "part-00000.parquet"
        )


def test_tracking_coverage_guard_ok(tmp_path: Path) -> None:
    import prefect_flows.rotation_priors_update as flow_mod

    _write_boxscore_day(tmp_path, 2025, "2025-10-21")
    _write_boxscore_day(tmp_path, 2026, "2026-03-24")
    _write_tracking_day(
        tmp_path,
        "2025-26",
        "2026-03-23",
        ["Possessions", "Passing", "Drives", "CatchShoot", "PullUpShot"],
    )

    out = flow_mod.tracking_coverage_guard_task.fn(
        data_root=tmp_path,
        season_id="2025-26",
        as_of_date=date(2026, 3, 24),
        strict=True,
        max_lag_days=3,
        recent_window_days=14,
        max_recent_missing_dates=2,
        required_measures=None,
        max_recent_measure_gap_dates=2,
    )

    assert out["status"] == "ok"
    assert out["latest_tracking_date"] == "2026-03-23"


def test_tracking_coverage_guard_raises_when_stale(tmp_path: Path) -> None:
    import prefect_flows.rotation_priors_update as flow_mod

    _write_boxscore_day(tmp_path, 2026, "2026-03-24")
    _write_tracking_day(
        tmp_path,
        "2025-26",
        "2026-03-10",
        ["Possessions", "Passing", "Drives", "CatchShoot", "PullUpShot"],
    )

    with pytest.raises(RuntimeError, match="lag_days"):
        flow_mod.tracking_coverage_guard_task.fn(
            data_root=tmp_path,
            season_id="2025-26",
            as_of_date=date(2026, 3, 24),
            strict=True,
            max_lag_days=3,
            recent_window_days=14,
            max_recent_missing_dates=2,
            required_measures=None,
            max_recent_measure_gap_dates=2,
        )
