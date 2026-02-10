from __future__ import annotations

import pandas as pd

from projections.pipelines.build_features_minutes_v1 import (
    _is_full_calendar_month,
    _merge_features_with_existing,
)


def test_merge_features_with_existing_preserves_outside_range() -> None:
    existing = pd.DataFrame(
        {
            "game_id": [1001, 1002],
            "player_id": [2001, 2002],
            "team_id": [3001, 3002],
            "game_date": ["2026-01-01", "2026-01-02"],
            "feature_as_of_ts": ["2026-01-01T18:00:00Z", "2026-01-02T18:00:00Z"],
            "status": ["available", "available"],
            "starter_flag": [1, 1],
        }
    )
    incoming = pd.DataFrame(
        {
            "game_id": [1002],
            "player_id": [2002],
            "team_id": [3002],
            "game_date": ["2026-01-02"],
            "feature_as_of_ts": ["2026-01-02T20:00:00Z"],
            "status": ["out"],
            "starter_flag": [0],
        }
    )

    merged = _merge_features_with_existing(
        existing,
        incoming,
        start=pd.Timestamp("2026-01-02"),
        end=pd.Timestamp("2026-01-02"),
    )

    assert len(merged) == 2
    jan1 = merged[merged["game_id"] == 1001].iloc[0]
    jan2 = merged[merged["game_id"] == 1002].iloc[0]
    assert pd.Timestamp(jan1["game_date"]) == pd.Timestamp("2026-01-01")
    assert jan2["status"] == "out"
    assert pd.Timestamp(jan2["feature_as_of_ts"], tz="UTC") == pd.Timestamp(
        "2026-01-02T20:00:00Z"
    )


def test_is_full_calendar_month() -> None:
    assert _is_full_calendar_month(
        pd.Timestamp("2026-01-01"),
        pd.Timestamp("2026-01-31"),
    )
    assert not _is_full_calendar_month(
        pd.Timestamp("2026-01-01"),
        pd.Timestamp("2026-01-30"),
    )
