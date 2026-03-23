from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from projections.api import contest_sim_api
from projections.api.minutes_api import create_app
from projections.contest_sim.scoring_models import (
    ContestConfig,
    ContestSimResult,
    LineupEVResult,
    SummaryStats,
)


def _client(tmp_path: Path, monkeypatch) -> TestClient:
    data_root = tmp_path / "data"
    monkeypatch.setattr(contest_sim_api.paths, "data_path", lambda *args, **kwargs: data_root)
    return TestClient(
        create_app(
            daily_root=tmp_path,
            dashboard_dist=tmp_path,
            fpts_root=tmp_path,
            sim_root=tmp_path,
        )
    )


def _fake_sim_result(lineup: list[str]) -> ContestSimResult:
    return ContestSimResult(
        results=[
            LineupEVResult(
                lineup_id=0,
                player_ids=list(lineup),
                mean=300.0,
                std=15.0,
                p90=325.0,
                p95=332.0,
                expected_payout=8.5,
                expected_value=5.5,
                roi=1.8333,
                win_rate=0.02,
                top_1pct_rate=0.06,
                top_5pct_rate=0.18,
                top_10pct_rate=0.33,
                cash_rate=0.56,
            )
        ],
        config=ContestConfig(
            field_size=5000,
            entry_fee=3.0,
            archetype="GPP Standard (20% paid)",
            rake=0.12,
        ),
        stats=SummaryStats(
            lineup_count=1,
            worlds_count=100,
            avg_ev=5.5,
            avg_roi=1.8333,
            positive_ev_count=1,
            best_ev_lineup_id=0,
            best_win_rate_lineup_id=0,
            best_top1pct_lineup_id=0,
            debug={},
        ),
    )


def test_contest_sim_run_normalizes_fd_lineups(tmp_path: Path, monkeypatch) -> None:
    player_pool = [
        {"player_id": "pg1", "positions": ["PG"]},
        {"player_id": "pg2", "positions": ["PG"]},
        {"player_id": "sg1", "positions": ["SG"]},
        {"player_id": "sg2", "positions": ["SG"]},
        {"player_id": "sf1", "positions": ["SF"]},
        {"player_id": "sf2", "positions": ["SF"]},
        {"player_id": "pf1", "positions": ["PF"]},
        {"player_id": "pf2", "positions": ["PF"]},
        {"player_id": "c1", "positions": ["C"]},
    ]
    monkeypatch.setattr(contest_sim_api, "build_player_pool", lambda **_: player_pool)

    captured: dict[str, object] = {}

    def _fake_run_contest_simulation(**kwargs):
        captured["site"] = kwargs.get("site")
        captured["user_lineups"] = kwargs.get("user_lineups")
        return _fake_sim_result(kwargs["user_lineups"][0])

    monkeypatch.setattr(contest_sim_api, "run_contest_simulation", _fake_run_contest_simulation)

    client = _client(tmp_path, monkeypatch)
    response = client.post(
        "/api/contest-sim/run",
        json={
            "game_date": "2026-03-15",
            "site": "fd",
            "draft_group_id": 127613,
            "lineups": [["c1", "pf2", "sg2", "pg2", "sf1", "pg1", "pf1", "sf2", "sg1"]],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    expected = ["pg1", "pg2", "sg1", "sg2", "sf1", "sf2", "pf1", "pf2", "c1"]

    assert captured["site"] == "fd"
    assert captured["user_lineups"] == [expected]
    assert payload["results"][0]["player_ids"] == expected

    build_id = payload["build_id"]
    build_path = (
        tmp_path
        / "data"
        / "builds"
        / "contest_sim"
        / "2026-03-15"
        / f"{build_id}.json"
    )
    saved = json.loads(build_path.read_text(encoding="utf-8"))
    assert saved["site"] == "fd"
    assert saved["lineups"] == [expected]


def test_contest_sim_run_rejects_invalid_fd_lineups(tmp_path: Path, monkeypatch) -> None:
    player_pool = [
        {"player_id": "pg1", "positions": ["PG"]},
        {"player_id": "sg1", "positions": ["SG"]},
        {"player_id": "sg2", "positions": ["SG"]},
        {"player_id": "sf1", "positions": ["SF"]},
        {"player_id": "sf2", "positions": ["SF"]},
        {"player_id": "pf1", "positions": ["PF"]},
        {"player_id": "pf2", "positions": ["PF"]},
        {"player_id": "c1", "positions": ["C"]},
        {"player_id": "c2", "positions": ["C"]},
    ]
    monkeypatch.setattr(contest_sim_api, "build_player_pool", lambda **_: player_pool)
    monkeypatch.setattr(
        contest_sim_api,
        "run_contest_simulation",
        lambda **kwargs: _fake_sim_result(kwargs["user_lineups"][0]),
    )

    client = _client(tmp_path, monkeypatch)
    response = client.post(
        "/api/contest-sim/run",
        json={
            "game_date": "2026-03-15",
            "site": "fd",
            "draft_group_id": 127613,
            "lineups": [["pg1", "sg1", "sg2", "sf1", "sf2", "pf1", "pf2", "c1", "c2"]],
        },
    )

    assert response.status_code == 400
    assert "cannot be assigned to a valid FD roster" in response.json()["detail"]


def test_contest_sim_run_infers_dk_draft_group_from_lineups(tmp_path: Path, monkeypatch) -> None:
    requested_dg = 111111
    actual_dg = 222222
    lineup = ["pg1", "sg1", "sf1", "pf1", "c1", "pg2", "sf2", "pf2"]

    by_dg = {
        requested_dg: [
            {"player_id": "alt1", "positions": ["PG"]},
            {"player_id": "alt2", "positions": ["SG"]},
            {"player_id": "alt3", "positions": ["SF"]},
            {"player_id": "alt4", "positions": ["PF"]},
            {"player_id": "alt5", "positions": ["C"]},
            {"player_id": "alt6", "positions": ["PG"]},
            {"player_id": "alt7", "positions": ["SF"]},
            {"player_id": "alt8", "positions": ["PF"]},
        ],
        actual_dg: [
            {"player_id": "pg1", "positions": ["PG"]},
            {"player_id": "sg1", "positions": ["SG"]},
            {"player_id": "sf1", "positions": ["SF"]},
            {"player_id": "pf1", "positions": ["PF"]},
            {"player_id": "c1", "positions": ["C"]},
            {"player_id": "pg2", "positions": ["PG"]},
            {"player_id": "sf2", "positions": ["SF"]},
            {"player_id": "pf2", "positions": ["PF"]},
        ],
    }

    monkeypatch.setattr(
        contest_sim_api,
        "build_player_pool",
        lambda **kwargs: by_dg.get(int(kwargs["draft_group_id"]), []),
    )
    monkeypatch.setattr(
        contest_sim_api,
        "get_slates_for_date",
        lambda game_date, slate_type="all", site="dk": [
            {"draft_group_id": requested_dg, "slate_type": "night", "n_contests": 20, "games": [{}]},
            {"draft_group_id": actual_dg, "slate_type": "main", "n_contests": 200, "games": [{}, {}]},
        ],
    )

    captured: dict[str, object] = {}

    def _fake_run_contest_simulation(**kwargs):
        captured["draft_group_id"] = kwargs.get("draft_group_id")
        captured["user_lineups"] = kwargs.get("user_lineups")
        return _fake_sim_result(kwargs["user_lineups"][0])

    monkeypatch.setattr(contest_sim_api, "run_contest_simulation", _fake_run_contest_simulation)

    client = _client(tmp_path, monkeypatch)
    response = client.post(
        "/api/contest-sim/run",
        json={
            "game_date": "2026-03-15",
            "site": "dk",
            "draft_group_id": requested_dg,
            "lineups": [lineup],
            "ownership_mode": "off",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert captured["draft_group_id"] == actual_dg
    assert len(captured["user_lineups"]) == 1
    assert sorted(captured["user_lineups"][0]) == sorted(lineup)
    assert payload["stats"]["debug"]["draft_group_resolution"]["requested_draft_group_id"] == requested_dg
    assert payload["stats"]["debug"]["draft_group_resolution"]["effective_draft_group_id"] == actual_dg
    assert payload["stats"]["debug"]["draft_group_resolution"]["inferred_from_lineups"] is True

    build_id = payload["build_id"]
    build_path = (
        tmp_path
        / "data"
        / "builds"
        / "contest_sim"
        / "2026-03-15"
        / f"{build_id}.json"
    )
    saved = json.loads(build_path.read_text(encoding="utf-8"))
    assert saved["draft_group_id"] == actual_dg


def test_resolve_dk_draft_group_uses_high_confidence_partial_match(monkeypatch) -> None:
    requested_dg = 111111
    actual_dg = 222222
    lineup = ["pg1", "sg1", "sf1", "pf1", "c1", "pg2", "sf2", "pf2"]

    by_dg = {
        requested_dg: [
            {"player_id": "pg1", "positions": ["PG"]},
            {"player_id": "sg1", "positions": ["SG"]},
        ],
        actual_dg: [
            {"player_id": "pg1", "positions": ["PG"]},
            {"player_id": "sg1", "positions": ["SG"]},
            {"player_id": "sf1", "positions": ["SF"]},
            {"player_id": "pf1", "positions": ["PF"]},
            {"player_id": "c1", "positions": ["C"]},
            {"player_id": "pg2", "positions": ["PG"]},
            {"player_id": "sf2", "positions": ["SF"]},
        ],
    }

    monkeypatch.setattr(
        contest_sim_api,
        "build_player_pool",
        lambda **kwargs: by_dg.get(int(kwargs["draft_group_id"]), []),
    )
    monkeypatch.setattr(
        contest_sim_api,
        "get_slates_for_date",
        lambda game_date, slate_type="all", site="dk": [
            {"draft_group_id": requested_dg, "slate_type": "night", "n_contests": 20, "games": [{}]},
            {"draft_group_id": actual_dg, "slate_type": "main", "n_contests": 200, "games": [{}, {}]},
        ],
    )

    effective_dg, debug = contest_sim_api._resolve_draft_group_id_for_lineups(
        game_date="2026-03-15",
        lineups=[lineup],
        site="dk",
        run_id=None,
        requested_draft_group_id=requested_dg,
    )

    assert effective_dg == actual_dg
    assert debug["requested_draft_group_id"] == requested_dg
    assert debug["effective_draft_group_id"] == actual_dg
    assert debug["requested_match_count"] == 2
    assert debug["best_match_count"] == 7
    assert debug["inference_reason"] == "coverage_improvement"
    assert debug["inferred_from_lineups"] is True


def test_resolve_dk_draft_group_keeps_requested_when_signal_is_weak(monkeypatch) -> None:
    requested_dg = 111111
    alt_dg = 222222
    lineup = ["pg1", "sg1", "sf1", "pf1", "c1", "pg2", "sf2", "pf2"]

    by_dg = {
        requested_dg: [
            {"player_id": "pg1", "positions": ["PG"]},
            {"player_id": "sg1", "positions": ["SG"]},
            {"player_id": "sf1", "positions": ["SF"]},
            {"player_id": "pf1", "positions": ["PF"]},
        ],
        alt_dg: [
            {"player_id": "pg1", "positions": ["PG"]},
            {"player_id": "sg1", "positions": ["SG"]},
            {"player_id": "sf1", "positions": ["SF"]},
            {"player_id": "pf1", "positions": ["PF"]},
            {"player_id": "c1", "positions": ["C"]},
        ],
    }

    monkeypatch.setattr(
        contest_sim_api,
        "build_player_pool",
        lambda **kwargs: by_dg.get(int(kwargs["draft_group_id"]), []),
    )
    monkeypatch.setattr(
        contest_sim_api,
        "get_slates_for_date",
        lambda game_date, slate_type="all", site="dk": [
            {"draft_group_id": requested_dg, "slate_type": "main", "n_contests": 150, "games": [{}, {}]},
            {"draft_group_id": alt_dg, "slate_type": "night", "n_contests": 20, "games": [{}]},
        ],
    )

    effective_dg, debug = contest_sim_api._resolve_draft_group_id_for_lineups(
        game_date="2026-03-15",
        lineups=[lineup],
        site="dk",
        run_id=None,
        requested_draft_group_id=requested_dg,
    )

    assert effective_dg == requested_dg
    assert debug["requested_match_count"] == 4
    assert debug["best_match_count"] == 5
    assert debug["inference_reason"] == "no_high_confidence_match"
    assert debug["inferred_from_lineups"] is False


def test_resolve_dk_draft_group_considers_players_beyond_first_three_lineups(monkeypatch) -> None:
    requested_dg = 111111
    actual_dg = 222222
    lineups = [
        [f"a{i}" for i in range(1, 9)],
        [f"b{i}" for i in range(1, 9)],
        [f"c{i}" for i in range(1, 9)],
        [f"d{i}" for i in range(1, 9)],
    ]

    requested_players = [pid for lineup in lineups[:3] for pid in lineup]
    actual_players = [pid for lineup in lineups for pid in lineup]

    by_dg = {
        requested_dg: [{"player_id": pid, "positions": ["PG"]} for pid in requested_players],
        actual_dg: [{"player_id": pid, "positions": ["PG"]} for pid in actual_players],
    }

    monkeypatch.setattr(
        contest_sim_api,
        "build_player_pool",
        lambda **kwargs: by_dg.get(int(kwargs["draft_group_id"]), []),
    )
    monkeypatch.setattr(
        contest_sim_api,
        "get_slates_for_date",
        lambda game_date, slate_type="all", site="dk": [
            {"draft_group_id": requested_dg, "slate_type": "main", "n_contests": 100, "games": [{}, {}]},
            {"draft_group_id": actual_dg, "slate_type": "main", "n_contests": 90, "games": [{}, {}]},
        ],
    )

    effective_dg, debug = contest_sim_api._resolve_draft_group_id_for_lineups(
        game_date="2026-03-15",
        lineups=lineups,
        site="dk",
        run_id=None,
        requested_draft_group_id=requested_dg,
    )

    assert effective_dg == actual_dg
    assert debug["requested_match_count"] == 24
    assert debug["best_match_count"] == 32
    assert debug["inferred_from_lineups"] is True


def test_contest_sim_run_retries_alternate_dk_slate_when_validation_fails(tmp_path: Path, monkeypatch) -> None:
    requested_dg = 111111
    actual_dg = 222222
    lineup = ["pg1", "sg1", "sf1", "pf1", "c1", "pg2", "sf2", "pf2"]

    by_dg = {
        requested_dg: [
            {"player_id": "pg1", "positions": ["PG"]},
            {"player_id": "sg1", "positions": ["SG"]},
        ],
        actual_dg: [
            {"player_id": "pg1", "positions": ["PG"]},
            {"player_id": "sg1", "positions": ["SG"]},
            {"player_id": "sf1", "positions": ["SF"]},
            {"player_id": "pf1", "positions": ["PF"]},
            {"player_id": "c1", "positions": ["C"]},
            {"player_id": "pg2", "positions": ["PG"]},
            {"player_id": "sf2", "positions": ["SF"]},
            {"player_id": "pf2", "positions": ["PF"]},
        ],
    }

    monkeypatch.setattr(
        contest_sim_api,
        "build_player_pool",
        lambda **kwargs: by_dg.get(int(kwargs["draft_group_id"]), []),
    )
    monkeypatch.setattr(
        contest_sim_api,
        "_resolve_draft_group_id_for_lineups",
        lambda **kwargs: (requested_dg, {"requested_draft_group_id": requested_dg, "effective_draft_group_id": requested_dg}),
    )
    monkeypatch.setattr(
        contest_sim_api,
        "get_slates_for_date",
        lambda game_date, slate_type="all", site="dk": [
            {"draft_group_id": requested_dg, "slate_type": "main", "n_contests": 120, "games": [{}, {}]},
            {"draft_group_id": actual_dg, "slate_type": "main", "n_contests": 100, "games": [{}, {}]},
        ],
    )

    captured: dict[str, object] = {}

    def _fake_run_contest_simulation(**kwargs):
        captured["draft_group_id"] = kwargs.get("draft_group_id")
        captured["user_lineups"] = kwargs.get("user_lineups")
        return _fake_sim_result(kwargs["user_lineups"][0])

    monkeypatch.setattr(contest_sim_api, "run_contest_simulation", _fake_run_contest_simulation)

    client = _client(tmp_path, monkeypatch)
    response = client.post(
        "/api/contest-sim/run",
        json={
            "game_date": "2026-03-15",
            "site": "dk",
            "draft_group_id": requested_dg,
            "lineups": [lineup],
            "ownership_mode": "off",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert captured["draft_group_id"] == actual_dg
    assert payload["stats"]["debug"]["draft_group_resolution"]["effective_draft_group_id"] == actual_dg
    assert payload["stats"]["debug"]["draft_group_resolution"]["inference_reason"] == "validation_retry_success"
