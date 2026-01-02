import pandas as pd

from scripts.rates.build_training_base import (
    _compute_vacated_team_features_from_minutes_preds,
    _position_flags_from_string,
)


def test_position_flags_from_string_basic():
    pos = pd.Series(["PG", "G", "F", "F-C", "BIG", "W", "UNK", None])
    flags = _position_flags_from_string(pos)

    assert flags.loc[0, "position_flags_PG"] == 1
    assert flags.loc[0, "position_flags_SG"] == 0
    assert flags.loc[0, "position_flags_C"] == 0

    # Guard bucket => coarse PG/SG flags
    assert flags.loc[1, "position_flags_PG"] == 1
    assert flags.loc[1, "position_flags_SG"] == 1

    # Forward bucket => coarse SF/PF flags
    assert flags.loc[2, "position_flags_SF"] == 1
    assert flags.loc[2, "position_flags_PF"] == 1

    # Hybrid forward/center => include PF + C
    assert flags.loc[3, "position_flags_PF"] == 1
    assert flags.loc[3, "position_flags_C"] == 1

    # BIG => center
    assert flags.loc[4, "position_flags_C"] == 1


def test_vacancy_from_minutes_preds_respects_out_status():
    stats = pd.DataFrame(
        [
            # game 1 (both players play)
            {
                "season": 2025,
                "game_id": 1,
                "team_id": 100,
                "opponent_id": 200,
                "player_id": 10,
                "game_date": pd.Timestamp("2025-10-21"),
                "tip_ts": pd.Timestamp("2025-10-21T23:00:00Z"),
                "minutes_played": 30.0,
                "fga": 12.0,
                "three_pa": 4.0,
                "fta": 2.0,
                "assists": 6.0,
                "turnovers": 2.0,
                "oreb": 1.0,
                "dreb": 4.0,
                "steals": 1.0,
                "blocks": 0.0,
            },
            {
                "season": 2025,
                "game_id": 1,
                "team_id": 100,
                "opponent_id": 200,
                "player_id": 11,
                "game_date": pd.Timestamp("2025-10-21"),
                "tip_ts": pd.Timestamp("2025-10-21T23:00:00Z"),
                "minutes_played": 28.0,
                "fga": 10.0,
                "three_pa": 3.0,
                "fta": 1.0,
                "assists": 4.0,
                "turnovers": 1.0,
                "oreb": 0.0,
                "dreb": 3.0,
                "steals": 0.0,
                "blocks": 1.0,
            },
            # game 2 (player 10 is OUT, player 11 plays)
            {
                "season": 2025,
                "game_id": 2,
                "team_id": 100,
                "opponent_id": 201,
                "player_id": 11,
                "game_date": pd.Timestamp("2025-10-23"),
                "tip_ts": pd.Timestamp("2025-10-23T23:00:00Z"),
                "minutes_played": 29.0,
                "fga": 11.0,
                "three_pa": 2.0,
                "fta": 2.0,
                "assists": 5.0,
                "turnovers": 2.0,
                "oreb": 1.0,
                "dreb": 2.0,
                "steals": 1.0,
                "blocks": 0.0,
            },
        ]
    )

    # Labels are optional; pass an empty table with expected columns.
    labels = pd.DataFrame(columns=["game_id", "player_id", "minutes_actual"])

    minutes_preds = pd.DataFrame(
        [
            {
                "season": 2025,
                "game_id": 2,
                "team_id": 100,
                "player_id": 10,
                "minutes_pred_play_prob": 0.0,
                "status_min": "OUT",
                "pos_bucket_min": "G",
            }
        ]
    )

    vac = _compute_vacated_team_features_from_minutes_preds(stats, labels, minutes_preds)
    assert len(vac) == 1
    row = vac.iloc[0].to_dict()
    assert row["season"] == 2025
    assert row["game_id"] == 2
    assert row["team_id"] == 100

    # Player 10 had 30 minutes in game 1 -> hist_minutes_szn at game 2 tip is 30.
    assert row["vac_min_szn"] == 30.0
