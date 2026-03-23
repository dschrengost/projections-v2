from __future__ import annotations

from projections.api import contest_sim_api


def test_normalize_lineups_for_site_canonicalizes_floatish_player_ids(monkeypatch) -> None:
    def fake_build_player_pool(**kwargs):
        return [
            {"player_id": "1.0", "positions": ["PG"]},
            {"player_id": "2.0", "positions": ["SG"]},
            {"player_id": "3.0", "positions": ["SF"]},
            {"player_id": "4.0", "positions": ["PF"]},
            {"player_id": "5.0", "positions": ["C"]},
            {"player_id": "6.0", "positions": ["PG", "SG"]},
            {"player_id": "7.0", "positions": ["SF", "PF"]},
            {"player_id": "8.0", "positions": ["SG", "SF"]},
        ]

    monkeypatch.setattr(contest_sim_api, "build_player_pool", fake_build_player_pool)

    normalized = contest_sim_api._normalize_lineups_for_site(
        [["1", "2", "3", "4", "5", "6", "7", "8"]],
        game_date="2026-03-23",
        draft_group_id=144063,
        site="dk",
        run_id=None,
        context="lineups",
    )

    assert len(normalized) == 1
    assert set(normalized[0]) == {"1", "2", "3", "4", "5", "6", "7", "8"}
    assert all(not pid.endswith(".0") for pid in normalized[0])

