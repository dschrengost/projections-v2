from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from projections.api.minutes_api import create_app


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


@pytest.mark.usefixtures("monkeypatch")
def test_ops_game_includes_minutes_and_rates_and_applies_manual_availability_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("PROJECTIONS_DATA_ROOT", str(tmp_path))

    slate_day = date(2025, 1, 2)
    gid = 111
    pid = 999

    projections_run_id = "PROJ_RUN"
    minutes_run_id = "MIN_RUN"
    rates_run_id = "RATES_RUN"

    # Unified projections (source of truth for player identity + selection)
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
            "generated_at": "2025-01-02T00:00:00Z",
            "projections_run_id": projections_run_id,
            "minutes_run_id": minutes_run_id,
            "rates_run_id": rates_run_id,
            "sim_run_id": projections_run_id,
            "draft_group_id": "123",
        },
    )
    _write_json(unified_day / "latest_run.json", {"run_id": projections_run_id})

    # Minutes artifact (raw minutes model output)
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
                "rotation_prob": 0.8,
            }
        ]
    ).to_parquet(minutes_run / "minutes.parquet", index=False)

    # Rates artifact (raw rates model output)
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
                "pred_fga3_per_min": 0.2,
                "pred_fta_per_min": 0.15,
                "pred_ast_per_min": 0.12,
                "pred_tov_per_min": 0.06,
                "pred_oreb_per_min": 0.05,
                "pred_dreb_per_min": 0.12,
                "pred_stl_per_min": 0.02,
                "pred_blk_per_min": 0.01,
                "pred_fg2_pct": 0.55,
                "pred_fg3_pct": 0.38,
                "pred_ft_pct": 0.8,
            }
        ]
    ).to_parquet(rates_run_dir / "rates.parquet", index=False)

    app = create_app(daily_root=tmp_path, dashboard_dist=tmp_path, fpts_root=tmp_path, sim_root=tmp_path)
    client = TestClient(app)

    # Baseline ops view
    resp = client.get("/api/ops/game", params={"date": slate_day.isoformat(), "game_id": str(gid)})
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["date"] == slate_day.isoformat()
    assert payload["game_id"] == str(gid)
    assert len(payload["players"]) == 1
    player = payload["players"][0]
    assert player["player_id"] == str(pid)
    assert player["minutes_baseline"]["minutes_p50"] == 30.0
    assert player["rates_baseline"]["pred_fga2_per_min"] == 0.6

    legacy_upsert = client.post(
        "/api/ops/overrides",
        json={
            "date": slate_day.isoformat(),
            "updates": [
                {
                    "game_id": str(gid),
                    "player_id": str(pid),
                    "minutes_p50": 34.0,
                    "pred_fga2_per_min": 0.75,
                }
            ],
        },
    )
    assert legacy_upsert.status_code == 200

    manual_upsert = client.post(
        "/api/ops/manual-availability-overrides",
        json={
            "date": slate_day.isoformat(),
            "game_id": str(gid),
            "player_id": str(pid),
            "override_type": "force_out",
            "entered_by": "daniel",
            "reason_code": "operator_report",
        },
    )
    assert manual_upsert.status_code == 200

    resp2 = client.get("/api/ops/game", params={"date": slate_day.isoformat(), "game_id": str(gid)})
    assert resp2.status_code == 200
    player2 = resp2.json()["players"][0]
    assert player2["minutes_effective"]["minutes_p50"] == 0.0
    assert player2["minutes_effective"]["play_prob"] == 0.0
    assert player2["rates_effective"]["pred_fga2_per_min"] == 0.6
    assert player2["override"] is None
    assert player2["manual_override"]["override_type"] == "force_out"


@pytest.mark.usefixtures("monkeypatch")
def test_ops_v2_apply_serializes_numpy_scalars(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PROJECTIONS_DATA_ROOT", str(tmp_path))

    slate_day = date(2025, 1, 3)
    gid = 222
    pid = 1234
    team_id = 10
    projections_run_id = "PROJ_RUN_V2"

    unified_day = tmp_path / "artifacts" / "projections" / slate_day.isoformat()
    unified_run = unified_day / f"run={projections_run_id}"
    unified_run.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "game_date": slate_day.isoformat(),
                "game_id": gid,
                "player_id": pid,
                "team_id": team_id,
                "player_name": "V2 Player",
                "minutes_final": 30.0,
            }
        ]
    ).to_parquet(unified_run / "projections.parquet", index=False)
    _write_json(unified_day / "latest_run.json", {"run_id": projections_run_id})

    def _fake_apply_minutes_overrides_v2(game_df: pd.DataFrame, _payload: dict, **_kwargs: object):
        resolved = game_df.copy()
        resolved["b_minutes"] = np.float64(30.0)
        resolved["mu_minutes"] = np.float64(31.0)
        resolved["lb_minutes"] = np.float64(0.0)
        resolved["ub_minutes"] = np.float64(48.0)
        resolved["eligible"] = np.bool_(True)
        resolved["force_active"] = np.bool_(False)
        resolved["force_inactive"] = np.bool_(False)
        resolved["weight"] = np.float64(1.0)
        resolved["constraint_kind"] = "none"
        resolved["override_present"] = np.bool_(False)
        diag = {
            "team_diagnostics": [
                {
                    "game_id": str(gid),
                    "team_id": np.int64(team_id),
                    "sum_lb": np.float64(0.0),
                    "sum_ub": np.float64(48.0),
                    "sum_mu": np.float64(31.0),
                    "locked_minutes_total": np.float64(0.0),
                    "n_players": np.int64(1),
                    "n_overrides": np.int64(0),
                }
            ]
        }
        return resolved, diag

    monkeypatch.setattr("projections.api.ops_api.apply_minutes_overrides_v2", _fake_apply_minutes_overrides_v2)

    app = create_app(daily_root=tmp_path, dashboard_dist=tmp_path, fpts_root=tmp_path, sim_root=tmp_path)
    client = TestClient(app)

    resp = client.post(
        "/api/ops/overrides-v2/apply",
        json={
            "date": slate_day.isoformat(),
            "game_id": str(gid),
            "run_id": None,
            "override_infeasible": "error",
            "overrides": [],
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["game_id"] == str(gid)
    assert len(payload["resolved_players"]) == 1
    assert payload["resolved_players"][0]["player_id"] == str(pid)
    assert payload["team_diagnostics"][0]["team_id"] == team_id
    assert payload["team_diagnostics"][0]["n_players"] == 1
