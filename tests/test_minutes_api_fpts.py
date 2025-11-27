from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from projections.api import minutes_api


def _write_minutes(day_dir: Path, run_id: str) -> None:
    run_dir = day_dir / f"run={run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        [
            {
                "game_date": pd.Timestamp("2024-10-01"),
                "tip_ts": pd.Timestamp("2024-10-01T23:00:00Z"),
                "game_id": 1001,
                "player_id": 10,
                "player_name": "Alpha Guard",
                "team_id": 200,
                "team_name": "Home",
                "team_tricode": "HOM",
                "opponent_team_id": 300,
                "opponent_team_name": "Away",
                "opponent_team_tricode": "AWY",
                "starter_flag": 1,
                "pos_bucket": "G",
                "play_prob": 0.9,
                "minutes_p10": 30.0,
                "minutes_p50": 33.0,
                "minutes_p90": 37.0,
                "minutes_p10_cond": 30.0,
                "minutes_p50_cond": 33.0,
                "minutes_p90_cond": 37.0,
            }
        ]
    )
    df.to_parquet(run_dir / "minutes.parquet", index=False)
    (run_dir / "summary.json").write_text(json.dumps({"run_id": run_id}), encoding="utf-8")


def _write_fpts(day_dir: Path, run_id: str) -> None:
    run_dir = day_dir / f"run={run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        [
            {
                "game_id": 1001,
                "player_id": 10,
                "fpts_per_min_pred": 1.1,
                "proj_fpts": 36.63,
                "scoring_system": "dk",
            }
        ]
    )
    df.to_parquet(run_dir / "fpts.parquet", index=False)
    (run_dir / "summary.json").write_text(
        json.dumps({"fpts_model_run_id": "testrun"}), encoding="utf-8"
    )


def test_minutes_api_includes_fpts(tmp_path: Path) -> None:
    run_id = "live123"
    date_text = "2024-10-01"
    minutes_root = tmp_path / "minutes_daily"
    minutes_day_dir = minutes_root / date_text
    _write_minutes(minutes_day_dir, run_id)
    dashboard_dist = tmp_path / "dist"
    dashboard_dist.mkdir()

    fpts_root = tmp_path / "fpts_gold"
    fpts_day_dir = fpts_root / date_text
    _write_fpts(fpts_day_dir, run_id)

    app = minutes_api.create_app(
        daily_root=minutes_root,
        dashboard_dist=dashboard_dist,
        fpts_root=fpts_root,
    )
    client = TestClient(app)
    resp = client.get("/api/minutes", params={"date": date_text, "run_id": run_id})
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["count"] == 1
    player = payload["players"][0]
    assert player["fpts_per_min_pred"] == 1.1
    assert player["proj_fpts"] == 36.63

    meta = client.get("/api/minutes/meta", params={"date": date_text, "run_id": run_id}).json()
    assert meta["fpts_available"] is True
    assert meta["fpts_meta"]["fpts_model_run_id"] == "testrun"
