from __future__ import annotations

import pandas as pd

from projections.minutes_v1.datasets import deduplicate_latest


def test_deduplicate_latest_keeps_last_ordered_row_per_key() -> None:
    df = pd.DataFrame(
        {
            "game_id": [1, 1, 1],
            "player_id": [10, 10, 10],
            "team_id": [100, 100, 100],
            "as_of_ts": pd.to_datetime(
                ["2025-01-01T17:00:00Z", "2025-01-01T17:30:00Z", "2025-01-01T18:00:00Z"],
                utc=True,
            ),
            "minutes": [20.0, 22.0, 24.0],
        }
    )

    out = deduplicate_latest(df, order_cols=["as_of_ts"])

    assert len(out) == 1
    row = out.iloc[0]
    assert row["game_id"] == 1
    assert row["player_id"] == 10
    assert row["team_id"] == 100
    assert float(row["minutes"]) == 24.0
