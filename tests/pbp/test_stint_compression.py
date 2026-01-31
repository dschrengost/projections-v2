from __future__ import annotations

import pandas as pd

from projections.pbp.constants import PBP_V1_SCHEMA_VERSION
from projections.pbp.stints import build_stints_from_pbp_events


def test_stint_compression_same_lineup_then_change() -> None:
    pbp = pd.DataFrame(
        [
            {
                "schema_version": PBP_V1_SCHEMA_VERSION,
                "season_id": "2024-25",
                "game_id": "0022400001",
                "period": 1,
                "play_id": 1,
                "period_elapsed_sec": 0,
                "clock_sec": 720,
                "away_lineup_key": "1|2|3|4|5",
                "home_lineup_key": "6|7|8|9|10",
                "away_p1": 1,
                "away_p2": 2,
                "away_p3": 3,
                "away_p4": 4,
                "away_p5": 5,
                "home_p1": 6,
                "home_p2": 7,
                "home_p3": 8,
                "home_p4": 9,
                "home_p5": 10,
            },
            {
                "schema_version": PBP_V1_SCHEMA_VERSION,
                "season_id": "2024-25",
                "game_id": "0022400001",
                "period": 1,
                "play_id": 2,
                "period_elapsed_sec": 10,
                "clock_sec": 710,
                "away_lineup_key": "1|2|3|4|5",
                "home_lineup_key": "6|7|8|9|10",
                "away_p1": 1,
                "away_p2": 2,
                "away_p3": 3,
                "away_p4": 4,
                "away_p5": 5,
                "home_p1": 6,
                "home_p2": 7,
                "home_p3": 8,
                "home_p4": 9,
                "home_p5": 10,
            },
            {
                "schema_version": PBP_V1_SCHEMA_VERSION,
                "season_id": "2024-25",
                "game_id": "0022400001",
                "period": 1,
                "play_id": 3,
                "period_elapsed_sec": 20,
                "clock_sec": 700,
                "away_lineup_key": "11|2|3|4|5",
                "home_lineup_key": "6|7|8|9|10",
                "away_p1": 2,
                "away_p2": 3,
                "away_p3": 4,
                "away_p4": 5,
                "away_p5": 11,
                "home_p1": 6,
                "home_p2": 7,
                "home_p3": 8,
                "home_p4": 9,
                "home_p5": 10,
            },
        ]
    )

    result = build_stints_from_pbp_events(pbp, schema_version=PBP_V1_SCHEMA_VERSION)
    stints = result.stints

    assert len(stints) == 2
    assert stints["duration_sec"].tolist() == [20, 700]
    assert stints["start_period_elapsed_sec"].tolist() == [0, 20]
    assert stints["end_period_elapsed_sec"].tolist() == [20, 720]

