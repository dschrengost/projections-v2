from __future__ import annotations

from fastapi.testclient import TestClient

from projections.api import contest_sim_api
from projections.api.minutes_api import create_app


def _client(tmp_path, monkeypatch) -> TestClient:
    data_root = tmp_path / "data"
    monkeypatch.setattr(contest_sim_api.paths, "data_path", lambda *args, **kwargs: data_root)
    return TestClient(create_app(daily_root=tmp_path, dashboard_dist=tmp_path, fpts_root=tmp_path, sim_root=tmp_path))


def test_portfolio_endpoint_propagates_worlds_source_and_diagnostics(tmp_path, monkeypatch) -> None:
    build = {
        "build_id": "build-1",
        "game_date": "2026-03-01",
        "draft_group_id": 123,
        "results": [
            {
                "lineup_id": 0,
                "player_ids": ["1", "2"],
                "mean": 100.0,
                "std": 10.0,
                "p90": 120.0,
                "p95": 125.0,
                "expected_payout": 5.0,
                "expected_value": 2.0,
                "roi": 0.4,
                "win_rate": 0.1,
                "top_1pct_rate": 0.05,
                "top_5pct_rate": 0.10,
                "top_10pct_rate": 0.15,
                "cash_rate": 0.4,
                "tail_score": 118.0,
                "select_score": 117.0,
                "robust_floor": 80.0,
            },
            {
                "lineup_id": 1,
                "player_ids": ["1", "3"],
                "mean": 99.0,
                "std": 11.0,
                "p90": 119.0,
                "p95": 124.0,
                "expected_payout": 4.8,
                "expected_value": 1.8,
                "roi": 0.35,
                "win_rate": 0.09,
                "top_1pct_rate": 0.045,
                "top_5pct_rate": 0.09,
                "top_10pct_rate": 0.14,
                "cash_rate": 0.39,
                "tail_score": 117.0,
                "select_score": 116.0,
                "robust_floor": 79.0,
            },
            {
                "lineup_id": 2,
                "player_ids": ["4", "5"],
                "mean": 96.0,
                "std": 9.0,
                "p90": 112.0,
                "p95": 118.0,
                "expected_payout": 4.1,
                "expected_value": 1.7,
                "roi": 0.3,
                "win_rate": 0.07,
                "top_1pct_rate": 0.04,
                "top_5pct_rate": 0.085,
                "top_10pct_rate": 0.13,
                "cash_rate": 0.35,
                "tail_score": 112.0,
                "select_score": 111.0,
                "robust_floor": 78.0,
            },
        ],
        "request": {"run_id": "run-123", "worlds_source": "gtv2"},
    }
    captured: dict[str, object] = {}

    monkeypatch.setattr(contest_sim_api, "_load_sim_build", lambda game_date, build_id: build)
    monkeypatch.setattr(
        contest_sim_api,
        "_load_player_ownership",
        lambda *args, **kwargs: {"1": 25.0, "2": 20.0, "3": 15.0, "4": 10.0, "5": 8.0},
    )

    def fake_load_worlds_matrix(game_date: str, data_root=None, run_id=None, worlds_source=None):  # type: ignore[no-untyped-def]
        captured["game_date"] = game_date
        captured["run_id"] = run_id
        captured["worlds_source"] = worlds_source
        return (
            [
                [10.0, 10.0, 0.0, 1.0, 1.0],
                [20.0, 20.0, 0.0, 0.5, 1.0],
                [30.0, 30.0, 0.0, 1.5, 1.0],
                [40.0, 40.0, 0.0, 2.0, 1.0],
            ],
            {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4},
        )

    monkeypatch.setattr(contest_sim_api, "load_worlds_matrix", fake_load_worlds_matrix)

    client = _client(tmp_path, monkeypatch)
    response = client.post(
        "/api/contest-sim/portfolio",
        json={
            "game_date": "2026-03-01",
            "draft_group_id": 123,
            "source_build_id": "build-1",
            "mode": "decorrelated_ev",
            "worlds_source": "gtv2",
            "portfolio_size": 2,
            "sort_key": "expected_value",
            "sort_dir": "desc",
            "worlds_train_frac": 0.5,
            "worlds_sample": 4,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert captured["worlds_source"] == "gtv2"
    assert captured["run_id"] == "run-123"
    assert payload["mode"] == "decorrelated_ev"
    assert payload["diagnostics"]["worlds_source"] == "gtv2"
    assert payload["diagnostics"]["world_selection_policy"].startswith("train_frac:")
    assert payload["diagnostics"]["holdout_worlds_count"] == 2
    assert len(payload["selected_lineup_ids"]) == 2


def test_portfolio_endpoint_rejects_min_exposure_bounds(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        contest_sim_api,
        "_load_sim_build",
        lambda game_date, build_id: {
            "build_id": "build-1",
            "game_date": game_date,
            "results": [
                {
                    "lineup_id": 0,
                    "player_ids": ["1", "2"],
                    "mean": 100.0,
                    "std": 10.0,
                    "p90": 120.0,
                    "p95": 125.0,
                    "expected_payout": 5.0,
                    "expected_value": 2.0,
                    "roi": 0.4,
                    "win_rate": 0.1,
                    "top_1pct_rate": 0.05,
                    "top_5pct_rate": 0.10,
                    "top_10pct_rate": 0.15,
                    "cash_rate": 0.4,
                }
            ],
            "request": {},
        },
    )
    monkeypatch.setattr(contest_sim_api, "_load_player_ownership", lambda *args, **kwargs: {"1": 25.0, "2": 20.0})

    client = _client(tmp_path, monkeypatch)
    response = client.post(
        "/api/contest-sim/portfolio",
        json={
            "game_date": "2026-03-01",
            "source_build_id": "build-1",
            "mode": "greedy_constraints",
            "portfolio_size": 1,
            "exposure_bounds": {"1": {"min": 25}},
        },
    )

    assert response.status_code == 400
    assert "Minimum exposure is not supported yet" in response.json()["detail"]


def test_save_sim_lineups_persists_portfolio_metadata(tmp_path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)
    response = client.post(
        "/api/contest-sim/saved-lineups",
        json={
            "game_date": "2026-03-01",
            "draft_group_id": 123,
            "name": "Portfolio 20",
            "kind": "portfolio",
            "lineups": [["1", "2"], ["3", "4"]],
            "results": [
                {
                    "lineup_id": 0,
                    "player_ids": ["1", "2"],
                    "mean": 100.0,
                    "std": 10.0,
                    "p90": 120.0,
                    "p95": 125.0,
                    "expected_payout": 5.0,
                    "expected_value": 2.0,
                    "roi": 0.4,
                    "win_rate": 0.1,
                    "top_1pct_rate": 0.05,
                    "top_5pct_rate": 0.10,
                    "top_10pct_rate": 0.15,
                    "cash_rate": 0.4,
                },
                {
                    "lineup_id": 1,
                    "player_ids": ["3", "4"],
                    "mean": 98.0,
                    "std": 9.0,
                    "p90": 118.0,
                    "p95": 123.0,
                    "expected_payout": 4.7,
                    "expected_value": 1.8,
                    "roi": 0.35,
                    "win_rate": 0.09,
                    "top_1pct_rate": 0.045,
                    "top_5pct_rate": 0.09,
                    "top_10pct_rate": 0.14,
                    "cash_rate": 0.38,
                },
            ],
            "config": {
                "field_size": 5000,
                "entry_fee": 3.0,
                "archetype": "GPP",
                "rake": 0.15,
                "prize_pool": 15000.0,
            },
            "stats": {
                "lineup_count": 2,
                "worlds_count": 4000,
                "avg_ev": 1.9,
                "avg_roi": 0.375,
                "positive_ev_count": 2,
                "best_ev_lineup_id": 0,
                "best_win_rate_lineup_id": 0,
                "best_top1pct_lineup_id": 0,
                "debug": {},
            },
            "source_build_id": "build-1",
            "selection_mode": "decorrelated_ev",
            "selection_config": {"portfolio_size": 2, "worlds_source": "gtv2"},
            "selection_diagnostics": {"ev_selected": 3.8, "risk_var_total_reduction_pct": 22.5},
            "warnings": ["example warning"],
        },
    )

    assert response.status_code == 200
    build_id = response.json()["build_id"]

    saved = contest_sim_api._load_sim_build("2026-03-01", build_id)
    assert saved is not None
    assert saved["kind"] == "portfolio"
    assert saved["request"]["selection_mode"] == "decorrelated_ev"
    assert saved["request"]["selection_diagnostics"]["ev_selected"] == 3.8
    assert saved["stats"]["debug"]["selection"]["risk_var_total_reduction_pct"] == 22.5
