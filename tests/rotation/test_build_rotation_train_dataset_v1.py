from __future__ import annotations

import pandas as pd

from scripts.rotation.build_rotation_train_dataset_v1 import (
    PLAYER_MINUTES_FROM_STINTS_COL,
    TEAM_TOTAL_MINUTES_COL,
    _align_labels_to_features,
    _filter_invalid_team_games,
)


def test_builder_drops_incomplete_team_games_and_fills_dnp_labels() -> None:
    # Team-game A: complete coverage; one DNP row missing label -> should be kept and filled to 0.
    features = pd.DataFrame(
        {
            "game_id": [1, 1, 1, 1, 2, 2],
            "team_id": [10, 10, 10, 10, 20, 20],
            "player_id": [101, 102, 103, 104, 201, 202],
            TEAM_TOTAL_MINUTES_COL: [240.0, 240.0, 240.0, 240.0, 240.0, 240.0],
            PLAYER_MINUTES_FROM_STINTS_COL: [30.0, 20.0, 190.0, 0.0, 40.0, 60.0],
            "rotation_team_missing": [0, 0, 0, 0, 0, 0],
        }
    )
    labels = pd.DataFrame(
        {
            "game_id": [1, 1, 1, 2, 2],
            "team_id": [10, 10, 10, 20, 20],
            "player_id": [101, 102, 103, 201, 202],
            "minutes": [30.0, 20.0, 190.0, 40.0, 60.0],
        }
    )

    aligned = _align_labels_to_features(features, labels)
    filtered_features, filtered_labels, meta = _filter_invalid_team_games(
        features,
        aligned,
        label_col="minutes",
        min_team_minutes_from_stints=200.0,
        max_team_minutes_gap=2.0,
    )

    # Team-game B should be dropped due to missing player coverage (only 100 minutes represented).
    assert meta["team_games_total"] == 2
    assert meta["team_games_kept"] == 1
    assert meta["team_games_dropped_by_reason"]["missing_player_coverage"] == 1

    assert filtered_features["team_id"].nunique() == 1
    assert int(filtered_features["team_id"].iloc[0]) == 10
    assert len(filtered_features) == 4

    # DNP row (player_id=104) should have minutes filled to 0.
    dnp_row = filtered_labels.loc[filtered_labels["player_id"] == 104, "minutes"]
    assert len(dnp_row) == 1
    assert float(dnp_row.iloc[0]) == 0.0

    # Kept team-game should have full team total minutes.
    assert abs(float(filtered_labels["minutes"].sum()) - 240.0) <= 1e-6

