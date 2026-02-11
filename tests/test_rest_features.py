from __future__ import annotations

import pandas as pd

from projections.features.rest import attach_rest_features


def test_attach_rest_features_uses_chronological_order_per_player() -> None:
    df = pd.DataFrame(
        [
            {"player_id": 1, "game_date": "2025-11-15", "tip_ts": "2025-11-15T01:00:00Z", "game_id": "g3"},
            {"player_id": 1, "game_date": "2025-10-23", "tip_ts": "2025-10-23T01:00:00Z", "game_id": "g1"},
            {"player_id": 2, "game_date": "2025-10-30", "tip_ts": "2025-10-30T01:00:00Z", "game_id": "g4"},
            {"player_id": 1, "game_date": "2025-10-25", "tip_ts": "2025-10-25T01:00:00Z", "game_id": "g2"},
        ],
        index=[100, 10, 200, 50],
    )

    out = attach_rest_features(df)

    assert out.index.tolist() == [100, 10, 200, 50]
    # player 1 chronology is 10/23 -> 10/25 -> 11/15
    assert pd.isna(out.loc[10, "days_since_last"])
    assert int(out.loc[50, "days_since_last"]) == 2
    assert int(out.loc[100, "days_since_last"]) == 21
    assert int(out.loc[50, "is_b2b"]) == 0
    assert int(out.loc[100, "is_b2b"]) == 0
