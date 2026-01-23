import json
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from projections.api.minutes_api import create_app


def _write_parquet(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def test_minutes_model_id_routing_overlays_shadow_minutes(tmp_path, monkeypatch) -> None:
    data_root = tmp_path / "projections-data"
    monkeypatch.setenv("PROJECTIONS_DATA_ROOT", str(data_root))

    day = "2026-01-01"
    run_id = "20260101T000000Z"

    # Base (production) frame: unified projections artifact.
    base_run_dir = data_root / "artifacts" / "projections" / day / f"run={run_id}"
    base_df = pd.DataFrame(
        [
            {
                "game_date": day,
                "game_id": 1001,
                "team_id": 10,
                "player_id": 1,
                "player_name": "Test Player",
                "team_tricode": "AAA",
                "opponent_team_tricode": "BBB",
                "minutes_p10": 12.0,
                "minutes_p50": 20.0,
                "minutes_p90": 30.0,
                "play_prob": 0.8,
            }
        ]
    )
    _write_parquet(base_run_dir / "projections.parquet", base_df)

    # Shadow (RMH) minutes frame: stored under model-partitioned root.
    shadow_run_dir = (
        data_root
        / "artifacts"
        / "minutes_models"
        / "daily"
        / "model_id=rmh_v1_1"
        / day
        / f"run={run_id}"
    )
    shadow_df = pd.DataFrame(
        [
            {
                "game_id": 1001,
                "team_id": 10,
                "player_id": 1,
                "snapshot_ts": "2026-01-01T00:00:00Z",
                "p_in_rotation": 0.25,
                "minutes_p10": 5.0,
                "minutes_p50": 33.0,
                "minutes_p90": 40.0,
                "minutes_mean_uncond": 8.25,
                "minutes_q10_uncond": 5.0,
                "minutes_q50_uncond": 33.0,
                "minutes_q90_uncond": 40.0,
            }
        ]
    )
    _write_parquet(shadow_run_dir / "minutes.parquet", shadow_df)
    (shadow_run_dir / "summary.json").write_text(
        json.dumps(
            {
                "date": day,
                "run_id": run_id,
                "model_id": "rmh_v1_1",
                "model_label": "RMH v1.1",
                "model_meta": {"play_threshold": 5.0},
            }
        ),
        encoding="utf-8",
    )

    app = create_app(
        daily_root=data_root / "artifacts" / "minutes_v1" / "daily",
        minutes_models_root=data_root / "artifacts" / "minutes_models" / "daily",
        dashboard_dist=tmp_path / "does-not-exist",
        fpts_root=data_root / "gold" / "projections_fpts_v1",
        sim_root=data_root / "artifacts" / "sim_v2" / "worlds_fpts_v2",
    )
    client = TestClient(app)

    models = client.get("/api/minutes/models", params={"date": day})
    assert models.status_code == 200
    assert any(m.get("model_id") == "prod" for m in models.json())
    assert any(m.get("model_id") == "rmh_v1_1" for m in models.json())

    prod = client.get("/api/minutes", params={"date": day, "run_id": run_id})
    assert prod.status_code == 200
    prod_row = prod.json()["players"][0]
    assert prod_row["minutes_p50"] == 20.0
    assert prod_row["play_prob"] == 0.8

    rmh = client.get("/api/minutes", params={"date": day, "run_id": run_id, "model_id": "rmh_v1_1"})
    assert rmh.status_code == 200
    rmh_row = rmh.json()["players"][0]
    assert rmh_row["minutes_p50"] == 33.0
    assert rmh_row["p_in_rotation"] == 0.25
    assert rmh_row["play_prob"] == 0.25

    unknown = client.get("/api/minutes", params={"date": day, "run_id": run_id, "model_id": "nope"})
    assert unknown.status_code == 400

