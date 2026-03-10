from __future__ import annotations

import sys
import types
from types import SimpleNamespace

from fastapi.testclient import TestClient

from projections.api import live_status_api
from projections.api.minutes_api import create_app


def _install_fake_prefect_run_deployment(monkeypatch, recorder: dict) -> None:
    def _fake_run_deployment(name: str, *, parameters: dict, tags: list[str], as_subflow: bool):
        recorder["name"] = name
        recorder["parameters"] = parameters
        recorder["tags"] = tags
        recorder["as_subflow"] = as_subflow
        return SimpleNamespace(id="flow-run-123")

    prefect_module = types.ModuleType("prefect")
    deployments_module = types.ModuleType("prefect.deployments")
    deployments_module.run_deployment = _fake_run_deployment  # type: ignore[attr-defined]
    prefect_module.deployments = deployments_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "prefect", prefect_module)
    monkeypatch.setitem(sys.modules, "prefect.deployments", deployments_module)


def test_trigger_game_rerun_success(monkeypatch, tmp_path) -> None:
    recorder: dict = {}
    _install_fake_prefect_run_deployment(monkeypatch, recorder)
    monkeypatch.setattr(
        live_status_api,
        "get_live_status",
        lambda date=None: {"games": [{"game_id": "1001"}, {"game_id": "1002"}]},
    )

    app = create_app(daily_root=tmp_path, dashboard_dist=tmp_path, fpts_root=tmp_path, sim_root=tmp_path)
    client = TestClient(app)

    res = client.post("/api/trigger/game", params={"date": "2026-03-10", "game_id": 1002})
    assert res.status_code == 200
    payload = res.json()
    assert payload["status"] == "triggered"
    assert payload["target_game_ids"] == [1002]
    assert payload["flow_run_id"] == "flow-run-123"

    assert recorder["name"] == "nba-live-pipeline/nba-live-pipeline"
    assert recorder["parameters"] == {
        "game_date": "2026-03-10",
        "manual_target_game_ids": [1002],
    }
    assert "targeted-rerun" in recorder["tags"]
    assert "game-1002" in recorder["tags"]
    assert recorder["as_subflow"] is False


def test_trigger_game_rerun_rejects_unknown_game_id(monkeypatch, tmp_path) -> None:
    recorder: dict = {}
    _install_fake_prefect_run_deployment(monkeypatch, recorder)
    monkeypatch.setattr(
        live_status_api,
        "get_live_status",
        lambda date=None: {"games": [{"game_id": "2001"}]},
    )

    app = create_app(daily_root=tmp_path, dashboard_dist=tmp_path, fpts_root=tmp_path, sim_root=tmp_path)
    client = TestClient(app)

    res = client.post("/api/trigger/game", params={"date": "2026-03-10", "game_id": 9999})
    assert res.status_code == 404
    assert "not found on slate" in res.json()["detail"]
    assert recorder == {}
