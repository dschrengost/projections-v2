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
