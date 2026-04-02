from __future__ import annotations

import pandas as pd

from projections.cli.build_minutes_live import (
    _remap_roster_game_ids_to_schedule_day,
    _restrict_roster_slice_to_slate_game_ids,
)


def test_remap_roster_game_ids_to_schedule_day_updates_stale_ids() -> None:
    roster_slice = pd.DataFrame(
        {
            "game_id": [22501111, 22501111],
            "team_id": [1610612763, 1610612752],  # MEM, NYK
            "player_id": [1, 2],
        }
    )
    schedule_df = pd.DataFrame(
        {
            "game_id": [22501003],
            "game_date": [pd.Timestamp("2026-04-01")],
            "home_team_id": [1610612763],
            "away_team_id": [1610612752],
        }
    )
    warnings: list[str] = []

    out = _remap_roster_game_ids_to_schedule_day(
        roster_slice,
        schedule_df=schedule_df,
        target_day=pd.Timestamp("2026-04-01"),
        warnings=warnings,
    )

    assert out["game_id"].tolist() == [22501003, 22501003]
    assert warnings
    assert "Remapped roster game_id" in warnings[0]


def test_restrict_roster_slice_to_slate_game_ids_drops_off_slate_rows() -> None:
    roster_slice = pd.DataFrame(
        {
            "game_id": [22501003, 22501003, 22501111],
            "team_id": [1610612763, 1610612752, 1610612742],
            "player_id": [1, 2, 3],
        }
    )
    warnings: list[str] = []

    out = _restrict_roster_slice_to_slate_game_ids(
        roster_slice,
        slate_game_ids={22501003},
        warnings=warnings,
    )

    assert out["game_id"].tolist() == [22501003, 22501003]
    assert warnings
    assert "Dropped 1 roster rows" in warnings[0]
