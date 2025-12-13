from __future__ import annotations

import pandas as pd

from projections.minutes_v1.snapshots import select_latest_before


def test_select_latest_before_uses_asof_then_ingested_for_ties() -> None:
    df = pd.DataFrame(
        {
            "game_id": [1, 1, 1],
            "player_id": [10, 10, 10],
            "as_of_ts": [
                "2025-01-01T10:00:00Z",
                "2025-01-01T11:00:00Z",
                "2025-01-01T11:00:00Z",
            ],
            "ingested_ts": [
                "2025-01-01T10:01:00Z",
                "2025-01-01T11:01:00Z",
                "2025-01-01T11:02:00Z",
            ],
            "value": [1, 2, 3],
        }
    )

    out = select_latest_before(
        df,
        "2025-01-01T11:00:00Z",
        group_cols=["game_id", "player_id"],
        as_of_col="as_of_ts",
        ingested_col="ingested_ts",
    )
    assert out["value"].tolist() == [3]

    out_early = select_latest_before(
        df,
        "2025-01-01T10:30:00Z",
        group_cols=["game_id", "player_id"],
        as_of_col="as_of_ts",
        ingested_col="ingested_ts",
    )
    assert out_early["value"].tolist() == [1]


def test_select_latest_before_falls_back_to_ingested_when_asof_missing() -> None:
    df = pd.DataFrame(
        {
            "game_id": [1, 1],
            "player_id": [10, 10],
            "as_of_ts": [pd.NaT, pd.NaT],
            "ingested_ts": ["2025-01-01T10:01:00Z", "2025-01-01T10:02:00Z"],
            "value": [1, 2],
        }
    )

    out = select_latest_before(
        df,
        "2025-01-01T10:05:00Z",
        group_cols=["game_id", "player_id"],
        as_of_col="as_of_ts",
        ingested_col="ingested_ts",
    )
    assert out["value"].tolist() == [2]
