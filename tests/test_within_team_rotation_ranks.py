"""Unit tests for within-team rotation rank features."""

from __future__ import annotations

import pandas as pd

from projections.minutes_v1.features import MinutesFeatureBuilder


def test_within_team_roll_mean_10_ranks_are_deterministic() -> None:
    schedule = pd.DataFrame(
        {
            "game_id": [101, 102],
            "season": ["2023-24", "2023-24"],
            "game_date": ["2023-10-25", "2023-10-27"],
            "tip_ts": ["2023-10-25T23:00:00Z", "2023-10-27T23:00:00Z"],
            "home_team_id": [100, 101],
            "away_team_id": [101, 100],
        }
    )
    odds = pd.DataFrame(
        {
            "game_id": [101, 102],
            "home_line": [-3.5, 2.0],
            "total": [224.5, 219.0],
            "as_of_ts": ["2023-10-25T20:00:00Z", "2023-10-27T20:00:00Z"],
        }
    )
    roster_rows = []
    for game_date, as_of in [
        ("2023-10-25", "2023-10-24T12:00:00Z"),
        ("2023-10-27", "2023-10-26T12:00:00Z"),
    ]:
        for player_id in [1, 2, 3, 4, 5]:
            roster_rows.append(
                {
                    "team_id": 100,
                    "game_date": game_date,
                    "player_id": player_id,
                    "active_flag": True,
                    "listed_pos": "G",
                    "as_of_ts": as_of,
                }
            )
    roster = pd.DataFrame(roster_rows)
    labels = pd.DataFrame(
        {
            "game_id": [101] * 5 + [102] * 5,
            "player_id": [1, 2, 3, 4, 5] * 2,
            "team_id": [100] * 10,
            "season": ["2023-24"] * 10,
            "game_date": ["2023-10-25"] * 5 + ["2023-10-27"] * 5,
            "minutes": [30.0, 28.0, 20.0, 5.0, 0.0] + [0.0] * 5,
            "starter_flag": [1, 1, 1, 0, 0] + [1, 1, 1, 0, 0],
        }
    )

    builder = MinutesFeatureBuilder(
        schedule=schedule,
        injuries_snapshot=pd.DataFrame(),
        odds_snapshot=odds,
        roster_nightly=roster,
        coach_tenure=pd.DataFrame(),
    )
    features = builder.build(labels)
    game2 = features[features["game_id"] == 102].copy()

    ranks = (
        game2.loc[:, ["player_id", "roll_mean_10", "team_roll_mean_10_rank", "team_roll_mean_10_rank_pct"]]
        .sort_values("team_roll_mean_10_rank")
        .reset_index(drop=True)
    )

    # roll_mean_10 for game 102 is based on game 101 minutes (shifted by 1).
    expected_order = [1, 2, 3, 4, 5]
    assert ranks["player_id"].tolist() == expected_order
    assert ranks["team_roll_mean_10_rank"].tolist() == [1, 2, 3, 4, 5]
    # Best player has pct=0.0, worst has pct=1.0 when team size is 5.
    assert ranks["team_roll_mean_10_rank_pct"].tolist() == [0.0, 0.25, 0.5, 0.75, 1.0]

