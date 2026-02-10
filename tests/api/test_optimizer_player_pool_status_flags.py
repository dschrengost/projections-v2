from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from projections.api import optimizer_service


def test_build_player_pool_derives_is_out_and_is_active_without_overrides(
    monkeypatch,
) -> None:
    proj_df = pd.DataFrame(
        [
            {
                "player_id": "p1",
                "player_name": "Player One",
                "team_tricode": "AAA",
                "proj_fpts": 30.0,
            },
            {
                "player_id": "p2",
                "player_name": "Player Two",
                "team_tricode": "BBB",
                "proj_fpts": 28.0,
            },
            {
                "player_id": "p3",
                "player_name": "Player Three",
                "team_tricode": "CCC",
                "proj_fpts": 24.0,
            },
        ]
    )

    sal_df = pd.DataFrame(
        [
            {
                "dk_player_id": 101,
                "display_name": "Player One",
                "positions": ["PG"],
                "salary": 7000,
                "team_abbrev": "AAA",
                "status": "OUT",
                "is_disabled": False,
                "game_matchup": "AAA@BBB",
                "game_start_utc": datetime(2026, 1, 10, 0, 0, tzinfo=timezone.utc),
            },
            {
                "dk_player_id": 102,
                "display_name": "Player Two",
                "positions": ["SG"],
                "salary": 6900,
                "team_abbrev": "BBB",
                "status": "Q",
                "is_disabled": False,
                "game_matchup": "AAA@BBB",
                "game_start_utc": datetime(2026, 1, 10, 0, 0, tzinfo=timezone.utc),
            },
            {
                "dk_player_id": 103,
                "display_name": "Player Three",
                "positions": ["SF"],
                "salary": 6600,
                "team_abbrev": "CCC",
                "status": None,
                "is_disabled": True,
                "game_matchup": "CCC@DDD",
                "game_start_utc": datetime(2026, 1, 10, 2, 30, tzinfo=timezone.utc),
            },
        ]
    )

    monkeypatch.setattr(
        optimizer_service,
        "load_projections_for_date",
        lambda *args, **kwargs: proj_df,
    )
    monkeypatch.setattr(
        optimizer_service,
        "load_salaries_for_date",
        lambda *args, **kwargs: sal_df,
    )

    pool = optimizer_service.build_player_pool(
        game_date="2026-01-10",
        draft_group_id=12345,
        site="dk",
        use_user_overrides=False,
    )

    by_id = {p["player_id"]: p for p in pool}
    assert by_id["p1"]["is_out"] is True
    assert by_id["p1"]["is_active"] is False

    assert by_id["p2"]["is_out"] is False
    assert by_id["p2"]["is_active"] is True

    assert by_id["p3"]["is_out"] is False
    assert by_id["p3"]["is_active"] is False
