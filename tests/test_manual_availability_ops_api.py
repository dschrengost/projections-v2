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
def test_manual_availability_override_endpoints_and_game_view(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("PROJECTIONS_DATA_ROOT", str(tmp_path))

    slate_day = date(2026, 3, 1)
    gid = 111
    pid = 999
    projections_run_id = "PROJ_RUN"
    minutes_run_id = "MIN_RUN"
    rates_run_id = "RATES_RUN"

    unified_day = tmp_path / "artifacts" / "projections" / slate_day.isoformat()
    unified_run = unified_day / f"run={projections_run_id}"
    unified_run.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "game_date": slate_day.isoformat(),
                "game_id": gid,
                "player_id": pid,
                "player_name": "Test Player",
                "team_id": 10,
                "team_tricode": "AAA",
                "status": "active",
                "is_projected_starter": True,
                "is_confirmed_starter": False,
            }
        ]
    ).to_parquet(unified_run / "projections.parquet", index=False)
    _write_json(
        unified_run / "summary.json",
        {
            "game_date": slate_day.isoformat(),
            "generated_at": "2026-03-01T00:00:00Z",
            "projections_run_id": projections_run_id,
            "minutes_run_id": minutes_run_id,
            "rates_run_id": rates_run_id,
            "sim_run_id": projections_run_id,
            "draft_group_id": "123",
        },
    )
    _write_json(unified_day / "latest_run.json", {"run_id": projections_run_id})

    minutes_day = tmp_path / "artifacts" / "minutes_v1" / "daily" / slate_day.isoformat()
    minutes_run = minutes_day / f"run={minutes_run_id}"
    minutes_run.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "game_date": slate_day.isoformat(),
                "game_id": gid,
                "player_id": pid,
                "status": "active",
                "play_prob": 1.0,
                "minutes_p10": 20.0,
                "minutes_p50": 30.0,
                "minutes_p90": 36.0,
                "minutes_final": 30.0,
                "rotation_prob": 0.8,
            }
        ]
    ).to_parquet(minutes_run / "minutes.parquet", index=False)

    rates_day = tmp_path / "gold" / "rates_v1_live" / slate_day.isoformat()
    rates_run_dir = rates_day / f"run={rates_run_id}"
    rates_run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "game_date": pd.to_datetime(slate_day.isoformat()),
                "game_id": gid,
                "team_id": 10,
                "player_id": pid,
                "pred_fga2_per_min": 0.6,
            }
        ]
    ).to_parquet(rates_run_dir / "rates.parquet", index=False)

    app = create_app(daily_root=tmp_path, dashboard_dist=tmp_path, fpts_root=tmp_path, sim_root=tmp_path)
    client = TestClient(app)

    create_resp = client.post(
        "/api/ops/manual-availability-overrides",
        json={
            "date": slate_day.isoformat(),
            "game_id": str(gid),
            "player_id": str(pid),
            "override_type": "force_out",
            "entered_by": "daniel",
            "reason_code": "operator_report",
            "reason_text": "Beat writer late scratch",
            "source_label": "twitter",
        },
    )
    assert create_resp.status_code == 200
    created = create_resp.json()["override"]
    assert created["override_type"] == "force_out"

    list_resp = client.get(
        "/api/ops/manual-availability-overrides",
        params={"date": slate_day.isoformat()},
    )
    assert list_resp.status_code == 200
    assert len(list_resp.json()["overrides"]) == 1

    game_resp = client.get(
        "/api/ops/game",
        params={"date": slate_day.isoformat(), "game_id": str(gid)},
    )
    assert game_resp.status_code == 200
    player = game_resp.json()["players"][0]
    assert player["manual_override"]["override_type"] == "force_out"
    assert player["minutes_effective"]["status"] == "OUT"
    assert player["minutes_effective"]["minutes_p50"] == 0.0
    assert player["minutes_effective"]["manual_override_type"] == "force_out"

    clear_resp = client.delete(
        f"/api/ops/manual-availability-overrides/{created['override_id']}",
        params={"date": slate_day.isoformat(), "cleared_by": "daniel"},
    )
    assert clear_resp.status_code == 200
    assert clear_resp.json()["active_overrides"] == []
