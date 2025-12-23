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
def test_minutes_api_prefers_pinned_unified_run_and_meta(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PROJECTIONS_DATA_ROOT", str(tmp_path))

    slate_day = date(2025, 1, 1)
    day_dir = tmp_path / "artifacts" / "projections" / slate_day.isoformat()
    run_a = "PROJ_A"
    run_b = "PROJ_B"

    def _write_run(run_id: str, player_id: int) -> None:
        run_dir = day_dir / f"run={run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            [
                {
                    "game_date": slate_day.isoformat(),
                    "game_id": 1,
                    "player_id": player_id,
                    "player_name": f"Player {player_id}",
                    "team_id": 10,
                    "team_tricode": "AAA",
                    "status": "ACTIVE",
                    "minutes_p50": 30.0,
                    "minutes_p90": 36.0,
                    "minutes_sim_mean": 30.5,
                    "dk_fpts_mean": 40.0,
                    "sim_profile": "test_profile",
                    "n_worlds": 123,
                }
            ]
        )
        df.to_parquet(run_dir / "projections.parquet", index=False)
        _write_json(
            run_dir / "summary.json",
            {
                "game_date": slate_day.isoformat(),
                "generated_at": "2025-01-01T00:00:00Z",
                "projections_run_id": run_id,
                "minutes_run_id": "MIN123",
                "rates_run_id": "RATES123",
                "sim_run_id": "SIM123",
                "sim_profile": "test_profile",
                "n_worlds": 123,
                "rows": 1,
            },
        )

    _write_run(run_a, player_id=101)
    _write_run(run_b, player_id=202)
    _write_json(day_dir / "latest_run.json", {"run_id": run_b})
    _write_json(day_dir / "pinned_run.json", {"run_id": run_a, "updated_at": "2025-01-01T01:23:45Z"})

    app = create_app(daily_root=tmp_path, dashboard_dist=tmp_path, fpts_root=tmp_path, sim_root=tmp_path)
    client = TestClient(app)

    runs = client.get("/api/minutes/runs", params={"date": slate_day.isoformat()})
    assert runs.status_code == 200
    runs_payload = runs.json()
    assert runs_payload["latest"] == run_b
    assert runs_payload["pinned"] == run_a

    minutes = client.get("/api/minutes", params={"date": slate_day.isoformat()})
    assert minutes.status_code == 200
    minutes_payload = minutes.json()
    assert minutes_payload["run_id"] == run_a
    assert minutes_payload["pinned_run_id"] == run_a
    assert minutes_payload["latest_run_id"] == run_b
    assert len(minutes_payload["players"]) == 1
    assert str(minutes_payload["players"][0]["player_id"]) == "101"

    meta = client.get("/api/minutes/meta", params={"date": slate_day.isoformat()})
    assert meta.status_code == 200
    meta_payload = meta.json()
    assert meta_payload["run_id"] == run_a
    assert meta_payload["pinned_run_id"] == run_a
    assert meta_payload["latest_run_id"] == run_b
    assert meta_payload["minutes_run_id"] == "MIN123"
    assert meta_payload["sim_run_id"] == "SIM123"
    assert meta_payload["n_worlds"] == 123

