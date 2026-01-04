from __future__ import annotations

import pandas as pd

from scripts.rotation.train_rotation_set_model_v1 import (
    TEAM_TOTAL_MINUTES_COL,
    _filter_invalid_team_games_for_training,
    _rescale_labels_to_240,
)


def test_ot_rate_uses_team_total_minutes_from_stints_not_label_sum() -> None:
    df = pd.DataFrame(
        {
            "game_id_norm": ["0000000001", "0000000001", "0000000002", "0000000002"],
            "team_id": [10, 10, 20, 20],
            "player_id": [1, 2, 3, 4],
            # Incomplete label coverage (would look like OT if you used label sums).
            "target_minutes": [50.0, 60.0, 80.0, 90.0],
            # Correct team totals from stints (regulation).
            TEAM_TOTAL_MINUTES_COL: [240.0, 240.0, 240.0, 240.0],
        }
    )

    _, meta = _rescale_labels_to_240(df, label_col="target_minutes")
    assert meta["team_minutes_source_for_ot_rate"] == TEAM_TOTAL_MINUTES_COL
    assert meta["team_games_ot_rate"] < 0.5


def test_trainer_drops_team_games_with_missing_player_coverage() -> None:
    df = pd.DataFrame(
        {
            "game_id_norm": ["0000000001"] * 4 + ["0000000002"] * 2,
            "team_id": [10] * 4 + [20] * 2,
            "player_id": [101, 102, 103, 104, 201, 202],
            TEAM_TOTAL_MINUTES_COL: [240.0] * 6,
            "minutes_from_stints": [30.0, 20.0, 190.0, 0.0, 40.0, 60.0],
            # Complete labels for team 10, incomplete coverage for team 20 (only 100 minutes represented).
            "target_minutes": [30.0, 20.0, 190.0, 0.0, 40.0, 60.0],
        }
    )

    filtered, meta = _filter_invalid_team_games_for_training(
        df, label_col="target_minutes", min_team_minutes_from_stints=200.0, max_team_minutes_gap=2.0
    )
    assert meta["team_games_total"] == 2
    assert meta["team_games_kept"] == 1
    assert filtered["team_id"].nunique() == 1
    assert int(filtered["team_id"].iloc[0]) == 10
