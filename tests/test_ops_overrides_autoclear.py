from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from projections.ops.overrides import (
    auto_clear_nonsticky_confirmed_starters_from_rotowire,
    apply_overrides_to_minutes_df,
    list_overrides,
    upsert_overrides,
)


def test_autoclear_nonsticky_confirmed_starter_override(tmp_path: Path) -> None:
    slate_day = date(2025, 12, 27)

    # Operator tags Player A as confirmed starter for ORL game.
    upsert_overrides(
        slate_day,
        [
            {
                "game_id": "22500433",
                "player_id": "1641710",
                "fields": {"is_confirmed_starter": True},
            }
        ],
        data_root=tmp_path,
    )

    roster_df = pd.DataFrame(
        [
            {
                "game_id": 22500433,
                "player_id": 1641710,
                "player_name": "Anthony Black",
                "team_tricode": "ORL",
            },
            {
                "game_id": 22500433,
                "player_id": 1001,
                "player_name": "Paolo Banchero",
                "team_tricode": "ORL",
            },
        ]
    )

    # Rotowire confirms ORL lineup, but Anthony Black is NOT among confirmed starters.
    rotowire_df = pd.DataFrame(
        [
            {
                "team_abbreviation": "ORL",
                "player_name": "Paolo Banchero",
                "lineup_role": "confirmed_starter",
            }
        ]
    )

    cleared = auto_clear_nonsticky_confirmed_starters_from_rotowire(
        game_date=slate_day,
        roster_df=roster_df,
        rotowire_df=rotowire_df,
        data_root=tmp_path,
    )
    assert cleared == 1

    remaining = list_overrides(slate_day, data_root=tmp_path)
    assert remaining == []


def test_minutes_override_reconciles_team_without_changing_locked_player(tmp_path: Path) -> None:
    slate_day = date(2025, 12, 27)

    # Jokic minutes override (locked) should remain, and other players should adjust to keep 240.
    upsert_overrides(
        slate_day,
        [
            {
                "game_id": "22500433",
                "player_id": "203999",
                "fields": {"minutes_p50": 38.5},
            }
        ],
        data_root=tmp_path,
    )

    minutes_df = pd.DataFrame(
        [
            {"game_id": 22500433, "team_id": 1610612743, "player_id": 203999, "status": "ava", "minutes_p50": 35.0},
            {"game_id": 22500433, "team_id": 1610612743, "player_id": 1, "status": "ava", "minutes_p50": 30.0},
            {"game_id": 22500433, "team_id": 1610612743, "player_id": 2, "status": "ava", "minutes_p50": 30.0},
            {"game_id": 22500433, "team_id": 1610612743, "player_id": 3, "status": "ava", "minutes_p50": 30.0},
            {"game_id": 22500433, "team_id": 1610612743, "player_id": 4, "status": "ava", "minutes_p50": 30.0},
            {"game_id": 22500433, "team_id": 1610612743, "player_id": 5, "status": "ava", "minutes_p50": 30.0},
            {"game_id": 22500433, "team_id": 1610612743, "player_id": 6, "status": "ava", "minutes_p50": 25.0},
        ]
    )

    out = apply_overrides_to_minutes_df(minutes_df, game_date=slate_day, data_root=tmp_path)
    g = out[(out["game_id"] == 22500433) & (out["team_id"] == 1610612743)].copy()
    g["minutes_p50"] = pd.to_numeric(g["minutes_p50"], errors="coerce").fillna(0.0)
    assert abs(float(g["minutes_p50"].sum()) - 240.0) < 1e-3

    jokic = g[g["player_id"].astype(str) == "203999"]
    assert abs(float(jokic["minutes_p50"].iloc[0]) - 38.5) < 1e-6
