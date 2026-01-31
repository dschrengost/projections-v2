from __future__ import annotations

import pandas as pd

from projections.rotations.dataset import build_rotation_dataset
from projections.rotations.schemas import LINEUP_COLS, ROTATION_EVENTS_COLS, ROTATION_LABELS_COLS
from projections.rotations.qa import infer_game_seconds


def test_rotation_dataset_builder_smoke() -> None:
    schedule = pd.DataFrame(
        [
            {
                "game_id": 1,
                "season": "2024-25",
                "home_team_id": 10,
                "away_team_id": 20,
            }
        ]
    )

    stints = pd.DataFrame(
        [
            {
                "game_id": "0000000001",
                "stint_id": 1,
                "period": 1,
                "start_clock_sec": 720,
                "end_clock_sec": 420,
                "duration_sec": 300,
                "home_p1": 1,
                "home_p2": 2,
                "home_p3": 3,
                "home_p4": 4,
                "home_p5": 5,
                "away_p1": 101,
                "away_p2": 102,
                "away_p3": 103,
                "away_p4": 104,
                "away_p5": 105,
            },
            {
                "game_id": "0000000001",
                "stint_id": 2,
                "period": 1,
                "start_clock_sec": 420,
                "end_clock_sec": 0,
                "duration_sec": 420,
                "home_p1": 1,
                "home_p2": 2,
                "home_p3": 3,
                "home_p4": 4,
                "home_p5": 6,
                "away_p1": 101,
                "away_p2": 102,
                "away_p3": 103,
                "away_p4": 104,
                "away_p5": 106,
            },
        ]
    )

    player_stints_rows = []
    for stint in stints.to_dict(orient="records"):
        for side in ["home", "away"]:
            for i in range(1, 6):
                player_stints_rows.append(
                    {
                        "game_id": stint["game_id"],
                        "team_side": side,
                        "player_id": stint[f"{side}_p{i}"],
                        "duration_sec": stint["duration_sec"],
                    }
                )
    player_stints = pd.DataFrame(player_stints_rows)

    ds = build_rotation_dataset(stints=stints, player_stints=player_stints, schedule=schedule)
    events = ds.rotation_events
    labels = ds.rotation_labels

    assert list(events.columns) == list(ROTATION_EVENTS_COLS)
    assert list(labels.columns) == list(ROTATION_LABELS_COLS)

    assert events[list(LINEUP_COLS)].isna().sum().sum() == 0
    assert (events["duration_sec"] >= 0).all()

    # Exactly 5 unique players per segment.
    assert (events[list(LINEUP_COLS)].nunique(axis=1) == 5).all()

    # Coverage: sums to period seconds per team-game.
    for (team_id, game_id), g in events.groupby(["team_id", "game_id"], sort=True):
        expected = infer_game_seconds(int(g["period"].max()))
        assert int(g["duration_sec"].sum()) == expected

    # Starter inference: first segment lineup.
    home_starters = set(
        events[(events["team_id"] == 10) & (events["segment_idx"] == 0)][list(LINEUP_COLS)].iloc[0].tolist()
    )
    labeled_home_starters = set(labels[(labels["team_id"] == 10) & (labels["starter_actual"])].player_id.tolist())
    assert home_starters == labeled_home_starters
