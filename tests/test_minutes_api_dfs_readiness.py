from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from projections.api.minutes_api import create_app


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


@pytest.mark.usefixtures("monkeypatch")
def test_minutes_api_dfs_readiness_happy_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PROJECTIONS_DATA_ROOT", str(tmp_path))

    slate_day = date(2025, 1, 3)
    run_id = "20250103T000000Z"

    projections_day = tmp_path / "artifacts" / "projections" / slate_day.isoformat()
    projections_run = projections_day / f"run={run_id}"
    projections_run.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"minutes_p50": 30.0, "play_prob": 1.0}]).to_parquet(
        projections_run / "projections.parquet", index=False
    )
    _write_json(projections_day / "latest_run.json", {"run_id": run_id})

    salaries_dir = tmp_path / "gold" / "dk_salaries" / "site=dk" / f"game_date={slate_day.isoformat()}" / "draft_group_id=1"
    salaries_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"player_id": 1, "salary": 5000}]).to_parquet(salaries_dir / "salaries.parquet", index=False)

    app = create_app(daily_root=tmp_path, dashboard_dist=tmp_path, fpts_root=tmp_path, sim_root=tmp_path)
    client = TestClient(app)

    resp = client.get("/api/dfs-readiness", params={"date": slate_day.isoformat()})
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["run_id"] == run_id
    assert payload["game_date"] == slate_day.isoformat()
    assert payload["passed"] is True


@pytest.mark.usefixtures("monkeypatch")
def test_minutes_api_dfs_readiness_flags_missing_salaries(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PROJECTIONS_DATA_ROOT", str(tmp_path))

    slate_day = date(2025, 1, 3)
    run_id = "20250103T000000Z"

    projections_day = tmp_path / "artifacts" / "projections" / slate_day.isoformat()
    projections_run = projections_day / f"run={run_id}"
    projections_run.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"minutes_p50": 30.0, "play_prob": 1.0}]).to_parquet(
        projections_run / "projections.parquet", index=False
    )
    _write_json(projections_day / "latest_run.json", {"run_id": run_id})

    app = create_app(daily_root=tmp_path, dashboard_dist=tmp_path, fpts_root=tmp_path, sim_root=tmp_path)
    client = TestClient(app)

    resp = client.get("/api/dfs-readiness", params={"date": slate_day.isoformat()})
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["passed"] is False
    assert any("dk_salaries:" in msg for msg in payload["errors"])

