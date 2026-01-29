from __future__ import annotations

from pathlib import Path

import pandas as pd

from fastapi.testclient import TestClient

from projections.api.minutes_api import create_app
from projections.api.optimizer_service import load_projections_for_date


def _write_unified_projections_fixture(
    *,
    root: Path,
    game_date: str,
    run_id: str,
) -> None:
    run_dir = root / "artifacts" / "projections" / game_date / f"run={run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        [
            {
                "player_id": 101,
                "player_name": "Player A",
                "team_tricode": "AAA",
                "opponent_team_tricode": "BBB",
                "game_id": 1,
                "game_date": game_date,
                "tip_ts": f"{game_date}T00:00:00Z",
                "status": "active",
                "play_prob": 0.4,
                # sim summaries (legacy naming from sim_v2)
                "sim_p_active": 0.2,
                "minutes_sim_mean_uncond": 4.0,
                "minutes_sim_p50_uncond": 0.0,
                "dk_fpts_mean_uncond": 5.0,
            },
            {
                "player_id": 202,
                "player_name": "Player B",
                "team_tricode": "BBB",
                "opponent_team_tricode": "AAA",
                "game_id": 1,
                "game_date": game_date,
                "tip_ts": f"{game_date}T00:00:00Z",
                "status": "active",
                "play_prob": 0.9,
                "sim_p_active": 0.95,
                "minutes_sim_mean_uncond": 30.0,
                "minutes_sim_p50_uncond": 30.0,
                "dk_fpts_mean_uncond": 40.0,
            },
        ]
    )
    df.to_parquet(run_dir / "projections.parquet", index=False)


def test_minutes_api_matches_optimizer_inputs_for_canonical_fields(monkeypatch, tmp_path: Path) -> None:
    game_date = "2099-01-01"
    run_id = "20990101T000000Z"

    _write_unified_projections_fixture(root=tmp_path, game_date=game_date, run_id=run_id)

    monkeypatch.setenv("PROJECTIONS_DATA_ROOT", str(tmp_path))

    client = TestClient(create_app())
    resp = client.get("/api/minutes", params={"date": game_date, "run_id": run_id})
    assert resp.status_code == 200
    payload = resp.json()
    api_players = pd.DataFrame(payload["players"])
    api_players["player_id"] = api_players["player_id"].astype(str)

    opt_df = load_projections_for_date(game_date, run_id=run_id, data_root=tmp_path)
    opt_df = opt_df.copy()
    opt_df["player_id"] = opt_df["player_id"].astype(str)

    fields = [
        "minutes_sim_p_active",
        "minutes_sim_uncond_mean",
        "minutes_sim_uncond_p50",
        "fpts_sim_uncond_mean",
    ]

    merged = api_players[["player_id", *fields]].merge(
        opt_df[["player_id", *fields]],
        on="player_id",
        how="inner",
        suffixes=("__api", "__opt"),
    )

    assert len(merged) == 2

    for f in fields:
        a = pd.to_numeric(merged[f"{f}__api"], errors="coerce").fillna(0.0)
        b = pd.to_numeric(merged[f"{f}__opt"], errors="coerce").fillna(0.0)
        assert (a == b).all(), f"Field {f} mismatch"
