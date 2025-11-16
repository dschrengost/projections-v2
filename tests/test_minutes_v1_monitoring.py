"""Tests for Minutes V1 monitoring calculations."""

from __future__ import annotations

import pandas as pd
import pytest

from projections.minutes_v1.monitoring import compute_monitoring_snapshot


def test_compute_monitoring_snapshot_returns_metrics():
    df = pd.DataFrame(
        {
            "game_id": [1, 1, 2, 2],
            "player_id": [10, 11, 10, 11],
            "game_date": ["2024-10-21", "2024-10-21", "2024-10-23", "2024-10-23"],
            "minutes": [30.0, 24.0, 32.0, 20.0],
            "p50": [31.0, 22.0, 29.0, 21.0],
            "p10_calibrated": [26.0, 18.0, 24.0, 17.0],
            "p90_calibrated": [36.0, 28.0, 34.0, 25.0],
            "tip_ts": ["2024-10-21T23:00:00Z", "2024-10-21T23:00:00Z", "2024-10-23T01:00:00Z", "2024-10-23T01:00:00Z"],
            "feature_as_of_ts": [
                "2024-10-21T20:00:00Z",
                "2024-10-21T20:00:00Z",
                "2024-10-22T21:30:00Z",
                "2024-10-22T21:30:00Z",
            ],
        }
    )
    snapshot = compute_monitoring_snapshot(df)
    assert "mae" in snapshot.overall
    assert "p_gt_err_threshold" in snapshot.overall
    assert "freshness_minutes_mean" in snapshot.overall
    assert not snapshot.rolling.empty
    assert {"game_date", "mae", "rolling_mae"}.issubset(snapshot.rolling.columns)


def test_monitoring_snapshot_handles_full_month():
    dates = pd.date_range("2024-12-01", periods=31, freq="D")
    df = pd.DataFrame(
        {
            "game_id": range(31),
            "player_id": range(31),
            "game_date": dates.date,
            "minutes": 25 + (dates.day % 5),
            "p50": 24 + (dates.day % 6),
            "p10_calibrated": 20 + (dates.day % 3),
            "p90_calibrated": 30 + (dates.day % 4),
            "tip_ts": dates.tz_localize("UTC"),
            "feature_as_of_ts": dates.tz_localize("UTC") - pd.Timedelta(hours=2),
        }
    )
    snapshot = compute_monitoring_snapshot(df, rolling_window=7)
    assert len(snapshot.rolling) == 31
    assert snapshot.rolling["rolling_mae"].iloc[-1] == pytest.approx(snapshot.rolling["mae"].tail(7).mean())
