from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from projections.etl import common


def _write_schedule(path: Path) -> None:
    frame = pd.DataFrame(
        {
            "game_id": [22500001],
            "game_code": ["20260101/ATLBOS"],
            "season": ["2025-26"],
            "game_date": [pd.Timestamp("2026-01-01")],
            "tip_ts": [pd.Timestamp("2026-01-01T00:30:00Z")],
            "home_team_id": [1610612738],
            "home_team_name": ["Celtics"],
            "home_team_city": ["Boston"],
            "home_team_tricode": ["BOS"],
            "away_team_id": [1610612737],
            "away_team_name": ["Hawks"],
            "away_team_city": ["Atlanta"],
            "away_team_tricode": ["ATL"],
            "arena_id": [pd.NA],
            "arena_name": [pd.NA],
            "arena_city": [pd.NA],
            "arena_state": [pd.NA],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def test_load_schedule_data_allow_empty_returns_empty_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    schedule_path = tmp_path / "schedule.parquet"
    _write_schedule(schedule_path)

    def _fake_schedule_from_api(start: pd.Timestamp, end: pd.Timestamp, timeout: float) -> pd.DataFrame:
        raise RuntimeError("NBA schedule API did not return any games for requested window.")

    monkeypatch.setattr(common, "schedule_from_api", _fake_schedule_from_api)

    out = common.load_schedule_data(
        [str(schedule_path)],
        pd.Timestamp("2026-01-02"),
        pd.Timestamp("2026-01-02"),
        timeout=1.0,
        allow_empty=True,
    )
    assert out.empty
    assert "game_id" in out.columns
    assert "tip_ts" in out.columns


def test_load_schedule_data_without_allow_empty_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    schedule_path = tmp_path / "schedule.parquet"
    _write_schedule(schedule_path)

    def _fake_schedule_from_api(start: pd.Timestamp, end: pd.Timestamp, timeout: float) -> pd.DataFrame:
        raise RuntimeError("NBA schedule API did not return any games for requested window.")

    monkeypatch.setattr(common, "schedule_from_api", _fake_schedule_from_api)

    with pytest.raises(RuntimeError, match="did not return any games"):
        common.load_schedule_data(
            [str(schedule_path)],
            pd.Timestamp("2026-01-02"),
            pd.Timestamp("2026-01-02"),
            timeout=1.0,
            allow_empty=False,
        )
