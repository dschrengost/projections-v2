from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from projections.api.minutes_api import create_app


def _write_parquet(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def test_minutes_api_overlays_run_scoped_ownership_when_unified_artifact_lacks_it(
    tmp_path: Path, monkeypatch
) -> None:
    data_root = tmp_path / "projections-data"
    monkeypatch.setenv("PROJECTIONS_DATA_ROOT", str(data_root))

    day = "2026-02-28"
    run_id = "20260228T174500Z"

    unified_dir = data_root / "artifacts" / "projections" / day / f"run={run_id}"
    unified_df = pd.DataFrame(
        [
            {
                "game_date": day,
                "game_id": 1001,
                "team_id": 10,
                "player_id": 23,
                "player_name": "LeBron James",
                "team_tricode": "LAL",
                "opponent_team_tricode": "GSW",
                "dk_fpts_mean": 47.8,
                "minutes_sim_mean": 35.2,
            }
        ]
    )
    _write_parquet(unified_dir / "projections.parquet", unified_df)

    own_dir = (
        data_root
        / "silver"
        / "ownership_predictions"
        / day
        / f"run={run_id}"
    )
    own_df = pd.DataFrame(
        [
            {
                "player_name": "LeBron James",
                "team": "LAL",
                "salary": 8300,
                "pred_own_pct": 12.5,
                "draft_group_id": "142837",
            }
        ]
    )
    _write_parquet(own_dir / "142837.parquet", own_df)

    app = create_app(
        daily_root=data_root / "artifacts" / "minutes_v1" / "daily",
        dashboard_dist=tmp_path / "does-not-exist",
        fpts_root=data_root / "gold" / "projections_fpts_v1",
        sim_root=data_root / "artifacts" / "sim_v2" / "worlds_fpts_v2",
    )
    client = TestClient(app)

    resp = client.get("/api/minutes", params={"date": day, "run_id": run_id})
    assert resp.status_code == 200

    payload = resp.json()
    assert payload["run_id"] == run_id
    assert len(payload["players"]) == 1

    player = payload["players"][0]
    assert player["player_name"] == "LeBron James"
    assert player["salary"] == 8300
    assert player["pred_own_pct"] == 12.5
    assert player["value"] == 5.76


def test_minutes_api_filters_to_selected_slate_and_uses_matching_ownership(
    tmp_path: Path, monkeypatch
) -> None:
    data_root = tmp_path / "projections-data"
    monkeypatch.setenv("PROJECTIONS_DATA_ROOT", str(data_root))

    day = "2026-02-28"
    run_id = "20260228T180002Z"

    unified_dir = data_root / "artifacts" / "projections" / day / f"run={run_id}"
    unified_df = pd.DataFrame(
        [
            {
                "game_date": day,
                "game_id": 2001,
                "team_id": 10,
                "player_id": 1,
                "player_name": "Alpha Guard",
                "team_tricode": "AAA",
                "opponent_team_tricode": "BBB",
                "salary": 9100,
                "pred_own_pct": 25.0,
                "dk_fpts_mean": 45.0,
            },
            {
                "game_date": day,
                "game_id": 2002,
                "team_id": 20,
                "player_id": 2,
                "player_name": "Beta Wing",
                "team_tricode": "CCC",
                "opponent_team_tricode": "DDD",
                "salary": 7600,
                "pred_own_pct": 18.0,
                "dk_fpts_mean": 36.0,
            },
        ]
    )
    _write_parquet(unified_dir / "projections.parquet", unified_df)

    own_dir = (
        data_root
        / "silver"
        / "ownership_predictions"
        / day
        / f"run={run_id}"
    )
    alt_own_df = pd.DataFrame(
        [
            {
                "player_name": "Beta Wing",
                "team": "CCC",
                "salary": 6400,
                "pred_own_pct": 42.5,
                "draft_group_id": "222222",
            }
        ]
    )
    _write_parquet(own_dir / "222222.parquet", alt_own_df)

    salaries_dir = (
        data_root
        / "gold"
        / "dk_salaries"
        / "site=dk"
        / f"game_date={day}"
        / "draft_group_id=222222"
    )
    salaries_df = pd.DataFrame(
        [
            {
                "display_name": "Beta Wing",
                "team_abbrev": "CCC",
                "salary": 6400,
                "draft_group_id": 222222,
            }
        ]
    )
    _write_parquet(salaries_dir / "salaries.parquet", salaries_df)

    app = create_app(
        daily_root=data_root / "artifacts" / "minutes_v1" / "daily",
        dashboard_dist=tmp_path / "does-not-exist",
        fpts_root=data_root / "gold" / "projections_fpts_v1",
        sim_root=data_root / "artifacts" / "sim_v2" / "worlds_fpts_v2",
    )
    client = TestClient(app)

    resp = client.get(
        "/api/minutes",
        params={"date": day, "run_id": run_id, "draft_group_id": "222222"},
    )
    assert resp.status_code == 200

    payload = resp.json()
    assert payload["count"] == 1
    assert payload["draft_group_id"] == "222222"
    player = payload["players"][0]
    assert player["player_name"] == "Beta Wing"
    assert player["salary"] == 6400
    assert player["pred_own_pct"] == 42.5
    assert player["value"] == 5.62

    meta = client.get(
        "/api/minutes/meta",
        params={"date": day, "run_id": run_id, "draft_group_id": "222222"},
    )
    assert meta.status_code == 200
    meta_payload = meta.json()
    assert meta_payload["counts"] == {"rows": 1, "players": 1, "teams": 1}
