from __future__ import annotations

import pandas as pd

from projections.pbp.constants import PBP_V1_SCHEMA_VERSION
from projections.pbp.qa import run_qa_gates
from projections.pbp.stints import build_stints_from_pbp_events


def test_stint_semantics_ordering() -> None:
    # Two events at the same clock with different lineups, intentionally out of play_id order.
    # Deterministic ordering must use play_id as the tiebreaker so stint segmentation is stable.
    pbp = pd.DataFrame(
        [
            {
                "schema_version": PBP_V1_SCHEMA_VERSION,
                "season_id": "2024-25",
                "game_id": "0022400001",
                "period": 1,
                "play_id": 3,
                "period_elapsed_sec": 10,
                "clock_sec": 710,
                "away_lineup_key": "11|2|3|4|5",
                "home_lineup_key": "101|102|103|104|105",
                "away_p1": 11,
                "away_p2": 2,
                "away_p3": 3,
                "away_p4": 4,
                "away_p5": 5,
                "home_p1": 101,
                "home_p2": 102,
                "home_p3": 103,
                "home_p4": 104,
                "home_p5": 105,
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
                "home_lineup_key": "101|102|103|104|105",
                "away_p1": 1,
                "away_p2": 2,
                "away_p3": 3,
                "away_p4": 4,
                "away_p5": 5,
                "home_p1": 101,
                "home_p2": 102,
                "home_p3": 103,
                "home_p4": 104,
                "home_p5": 105,
            },
            {
                "schema_version": PBP_V1_SCHEMA_VERSION,
                "season_id": "2024-25",
                "game_id": "0022400001",
                "period": 1,
                "play_id": 4,
                "period_elapsed_sec": 20,
                "clock_sec": 700,
                "away_lineup_key": "11|2|3|4|5",
                "home_lineup_key": "101|102|103|104|105",
                "away_p1": 11,
                "away_p2": 2,
                "away_p3": 3,
                "away_p4": 4,
                "away_p5": 5,
                "home_p1": 101,
                "home_p2": 102,
                "home_p3": 103,
                "home_p4": 104,
                "home_p5": 105,
            },
        ]
    )

    stints = build_stints_from_pbp_events(pbp, schema_version=PBP_V1_SCHEMA_VERSION).stints

    assert stints["away_lineup_key"].tolist() == ["1|2|3|4|5", "11|2|3|4|5"]
    assert stints["start_period_elapsed_sec"].tolist() == [10, 10]
    assert stints["duration_sec"].tolist() == [0, 710]


def test_zero_duration_stints_policy() -> None:
    # Multiple lineup changes at the same clock can create zero-duration stints. Policy:
    # keep them (not filtered) and surface counts in QA trust metrics.
    pbp = pd.DataFrame(
        [
            {
                "schema_version": PBP_V1_SCHEMA_VERSION,
                "season_id": "2024-25",
                "game_id": "0022400002",
                "period": 1,
                "play_id": 1,
                "period_elapsed_sec": 0,
                "clock_sec": 720,
                "away_lineup_key": "1|2|3|4|5",
                "home_lineup_key": "101|102|103|104|105",
                "away_p1": 1,
                "away_p2": 2,
                "away_p3": 3,
                "away_p4": 4,
                "away_p5": 5,
                "home_p1": 101,
                "home_p2": 102,
                "home_p3": 103,
                "home_p4": 104,
                "home_p5": 105,
            },
            {
                "schema_version": PBP_V1_SCHEMA_VERSION,
                "season_id": "2024-25",
                "game_id": "0022400002",
                "period": 1,
                "play_id": 2,
                "period_elapsed_sec": 10,
                "clock_sec": 710,
                "away_lineup_key": "1|2|3|4|5",
                "home_lineup_key": "101|102|103|104|105",
                "away_p1": 1,
                "away_p2": 2,
                "away_p3": 3,
                "away_p4": 4,
                "away_p5": 5,
                "home_p1": 101,
                "home_p2": 102,
                "home_p3": 103,
                "home_p4": 104,
                "home_p5": 105,
            },
            {
                "schema_version": PBP_V1_SCHEMA_VERSION,
                "season_id": "2024-25",
                "game_id": "0022400002",
                "period": 1,
                "play_id": 3,
                "period_elapsed_sec": 10,
                "clock_sec": 710,
                "away_lineup_key": "11|2|3|4|5",
                "home_lineup_key": "101|102|103|104|105",
                "away_p1": 11,
                "away_p2": 2,
                "away_p3": 3,
                "away_p4": 4,
                "away_p5": 5,
                "home_p1": 101,
                "home_p2": 102,
                "home_p3": 103,
                "home_p4": 104,
                "home_p5": 105,
            },
            {
                "schema_version": PBP_V1_SCHEMA_VERSION,
                "season_id": "2024-25",
                "game_id": "0022400002",
                "period": 1,
                "play_id": 4,
                "period_elapsed_sec": 10,
                "clock_sec": 710,
                "away_lineup_key": "12|2|3|4|5",
                "home_lineup_key": "101|102|103|104|105",
                "away_p1": 12,
                "away_p2": 2,
                "away_p3": 3,
                "away_p4": 4,
                "away_p5": 5,
                "home_p1": 101,
                "home_p2": 102,
                "home_p3": 103,
                "home_p4": 104,
                "home_p5": 105,
            },
            {
                "schema_version": PBP_V1_SCHEMA_VERSION,
                "season_id": "2024-25",
                "game_id": "0022400002",
                "period": 1,
                "play_id": 5,
                "period_elapsed_sec": 10,
                "clock_sec": 710,
                "away_lineup_key": "13|2|3|4|5",
                "home_lineup_key": "101|102|103|104|105",
                "away_p1": 13,
                "away_p2": 2,
                "away_p3": 3,
                "away_p4": 4,
                "away_p5": 5,
                "home_p1": 101,
                "home_p2": 102,
                "home_p3": 103,
                "home_p4": 104,
                "home_p5": 105,
            },
            {
                "schema_version": PBP_V1_SCHEMA_VERSION,
                "season_id": "2024-25",
                "game_id": "0022400002",
                "period": 1,
                "play_id": 6,
                "period_elapsed_sec": 20,
                "clock_sec": 700,
                "away_lineup_key": "13|2|3|4|5",
                "home_lineup_key": "101|102|103|104|105",
                "away_p1": 13,
                "away_p2": 2,
                "away_p3": 3,
                "away_p4": 4,
                "away_p5": 5,
                "home_p1": 101,
                "home_p2": 102,
                "home_p3": 103,
                "home_p4": 104,
                "home_p5": 105,
            },
        ]
    )

    stints = build_stints_from_pbp_events(pbp, schema_version=PBP_V1_SCHEMA_VERSION).stints

    assert int((stints["duration_sec"] == 0).sum()) == 2

    outputs = run_qa_gates(
        stints,
        season_id="2024-25",
        run_id="test",
        schema_version=PBP_V1_SCHEMA_VERSION,
    )
    assert len(outputs.failures) == 0
    assert outputs.report["trust_metrics"]["zero_duration_stints_total"] == 2
    assert outputs.report["trust_metrics"]["max_zero_duration_stints_in_game"] == 2
    assert outputs.report["trust_metrics"]["stint_duration_summary"]["min"] == 0
    assert outputs.report["trust_metrics"]["stint_duration_summary"]["max"] == 710

