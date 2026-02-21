from __future__ import annotations

import pandas as pd

from projections.builders.features_builder import SharedFeaturesBuilder


def test_all_star_games_are_filtered_from_history() -> None:
    labels = pd.DataFrame(
        {
            "game_id": [int("0060000001"), int("0022600001")],
            "player_id": [1, 2],
            "team_id": [999001, 1610612737],
        }
    )
    schedule = pd.DataFrame({"game_id": labels["game_id"]})

    filtered_labels, filtered_schedule, game_ids, dropped = SharedFeaturesBuilder._filter_all_star_games(
        labels=labels,
        schedule=schedule,
        game_ids=labels["game_id"].tolist(),
    )

    # All-Star game removed
    assert set(dropped) == {int("0060000001")}
    assert filtered_labels["game_id"].tolist() == [int("0022600001")]
    assert filtered_schedule["game_id"].tolist() == [int("0022600001")]
    assert game_ids == [int("0022600001")]
