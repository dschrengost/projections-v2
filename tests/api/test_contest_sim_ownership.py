from __future__ import annotations

import pytest

from projections.api import contest_sim_api


def test_load_player_ownership_uses_slate_player_pool(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_build_player_pool(**kwargs):
        captured.update(kwargs)
        return [
            {"player_id": "2", "own_proj": 42.5},
            {"player_id": "3"},
        ]

    monkeypatch.setattr(contest_sim_api, "build_player_pool", fake_build_player_pool)
    monkeypatch.setattr(
        contest_sim_api,
        "load_projections_for_date",
        lambda *args, **kwargs: pytest.fail("contest sim should use slate player pool first"),
    )

    ownership = contest_sim_api._load_player_ownership(
        "2026-02-28",
        run_id="20260228T180002Z",
        draft_group_id=222222,
    )

    assert captured["game_date"] == "2026-02-28"
    assert captured["draft_group_id"] == 222222
    assert captured["run_id"] == "20260228T180002Z"
    assert ownership == {"2": 42.5}


def test_contest_sim_request_defaults_disable_dupe_penalty() -> None:
    request = contest_sim_api.ContestSimRequest(game_date="2026-02-28", lineups=[["1"]])

    assert request.ownership_mode == "field_only"


def test_build_field_library_request_defaults_disable_dupe_penalty() -> None:
    request = contest_sim_api.BuildFieldLibraryRequest(game_date="2026-02-28", draft_group_id=222222)

    assert request.ownership_mode == "field_only"
