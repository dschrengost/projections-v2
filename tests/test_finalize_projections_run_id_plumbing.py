from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from projections.cli.finalize_projections import finalize_projections
from projections.cli.check_health import check_artifact_pointers


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def test_finalize_projections_uses_explicit_run_ids_and_preserves_sim_values(tmp_path: Path) -> None:
    data_root = tmp_path
    game_date = date(2025, 1, 1)
    minutes_run_id = "MIN123"
    sim_run_id = "SIM123"
    projections_run_id = "PROJ123"
    draft_group_id = "999"

    # Minutes artifact: only exists for minutes_run_id (not projections_run_id).
    minutes_day = data_root / "artifacts" / "minutes_v1" / "daily" / game_date.isoformat()
    minutes_dir = minutes_day / f"run={minutes_run_id}"
    minutes_dir.mkdir(parents=True, exist_ok=True)
    minutes_df = pd.DataFrame(
        [
            {
                "game_date": game_date.isoformat(),
                "game_id": 1,
                "player_id": 101,
                "player_name": "Player One",
                "team_id": 10,
                "team_name": "Team A",
                "team_tricode": "AAA",
                "opponent_team_id": 20,
                "opponent_team_name": "Team B",
                "opponent_team_tricode": "BBB",
                "starter_flag": "STARTER",
                "is_projected_starter": True,
                "is_confirmed_starter": False,
                "status": "ACTIVE",
                "play_prob": 1.0,
                "minutes_p10": 24.0,
                "minutes_p50": 30.0,
                "minutes_p90": 36.0,
                "minutes_p10_cond": 24.0,
                "minutes_p50_cond": 30.0,
                "minutes_p90_cond": 36.0,
            },
            {
                "game_date": game_date.isoformat(),
                "game_id": 1,
                "player_id": 202,
                "player_name": "Player Two",
                "team_id": 20,
                "team_name": "Team B",
                "team_tricode": "BBB",
                "opponent_team_id": 10,
                "opponent_team_name": "Team A",
                "opponent_team_tricode": "AAA",
                "starter_flag": "BENCH",
                "is_projected_starter": False,
                "is_confirmed_starter": False,
                "status": "ACTIVE",
                "play_prob": 1.0,
                "minutes_p10": 10.0,
                "minutes_p50": 16.0,
                "minutes_p90": 22.0,
                "minutes_p10_cond": 10.0,
                "minutes_p50_cond": 16.0,
                "minutes_p90_cond": 22.0,
            },
        ]
    )
    minutes_df.to_parquet(minutes_dir / "minutes.parquet", index=False)
    _write_json(minutes_day / "latest_run.json", {"run_id": minutes_run_id})
    _write_json(minutes_dir / "summary.json", {"run_id": minutes_run_id})

    # Sim projections (source of truth): worlds_fpts_v2.
    sim_day = data_root / "artifacts" / "sim_v2" / "worlds_fpts_v2" / f"game_date={game_date.isoformat()}"
    sim_dir = sim_day / f"run={sim_run_id}"
    sim_dir.mkdir(parents=True, exist_ok=True)
    sim_df = pd.DataFrame(
        [
            {
                "game_date": game_date.isoformat(),
                "game_id": 1,
                "team_id": 10,
                "player_id": 101,
                "minutes_mean": 30.0,
                "minutes_sim_mean": 30.25,
                "minutes_sim_std": 4.0,
                "minutes_sim_p10": 24.0,
                "minutes_sim_p50": 30.0,
                "minutes_sim_p90": 36.0,
                "dk_fpts_mean": 40.0,
                "dk_fpts_std": 8.0,
                "dk_fpts_p05": 25.0,
                "dk_fpts_p10": 28.0,
                "dk_fpts_p25": 35.0,
                "dk_fpts_p50": 40.0,
                "dk_fpts_p75": 45.0,
                "dk_fpts_p90": 52.0,
                "dk_fpts_p95": 58.0,
                "sim_profile": "test_profile",
                "n_worlds": 123,
                "minutes_run_id": minutes_run_id,
                "rates_run_id": "RATES456",
                "is_starter": True,
            },
            {
                "game_date": game_date.isoformat(),
                "game_id": 1,
                "team_id": 20,
                "player_id": 202,
                "minutes_mean": 16.0,
                "minutes_sim_mean": 15.75,
                "minutes_sim_std": 3.0,
                "minutes_sim_p10": 10.0,
                "minutes_sim_p50": 16.0,
                "minutes_sim_p90": 22.0,
                "dk_fpts_mean": 18.0,
                "dk_fpts_std": 6.0,
                "dk_fpts_p05": 5.0,
                "dk_fpts_p10": 8.0,
                "dk_fpts_p25": 12.0,
                "dk_fpts_p50": 18.0,
                "dk_fpts_p75": 22.0,
                "dk_fpts_p90": 28.0,
                "dk_fpts_p95": 32.0,
                "sim_profile": "test_profile",
                "n_worlds": 123,
                "minutes_run_id": minutes_run_id,
                "rates_run_id": "RATES456",
                "is_starter": False,
            },
        ]
    )
    sim_df.to_parquet(sim_dir / "projections.parquet", index=False)
    _write_json(sim_day / "latest_run.json", {"run_id": sim_run_id})

    # Legacy sim_v2/projections exists but must not override worlds_fpts_v2.
    legacy_sim_dir = data_root / "artifacts" / "sim_v2" / "projections" / f"game_date={game_date.isoformat()}"
    legacy_sim_dir.mkdir(parents=True, exist_ok=True)
    legacy_df = sim_df.copy()
    legacy_df["dk_fpts_mean"] = legacy_df["dk_fpts_mean"] + 100.0
    legacy_df["n_worlds"] = 999
    legacy_df.to_parquet(legacy_sim_dir / "projections.parquet", index=False)

    # Rates artifact (needed for pointer consistency check).
    rates_day = data_root / "gold" / "rates_v1_live" / game_date.isoformat()
    rates_dir = rates_day / "run=RATES456"
    rates_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "game_date": game_date.isoformat(),
                "game_id": 1,
                "team_id": 10,
                "player_id": 101,
                "pred_pts_per_min": 1.0,
            }
        ]
    ).to_parquet(rates_dir / "rates.parquet", index=False)
    _write_json(rates_dir / "summary.json", {"run_id": "RATES456"})
    _write_json(rates_day / "latest_run.json", {"run_id": "RATES456"})

    out_path = finalize_projections(
        game_date,
        projections_run_id,
        draft_group_id,
        data_root,
        minutes_run_id=minutes_run_id,
        sim_run_id=sim_run_id,
    )
    assert out_path is not None
    assert out_path.exists()

    unified = pd.read_parquet(out_path)
    assert set(unified["projections_run_id"].unique()) == {projections_run_id}
    assert set(unified["minutes_run_id"].unique()) == {minutes_run_id}
    assert set(unified["sim_run_id"].dropna().unique()) == {sim_run_id}

    merged = unified.merge(
        sim_df[["player_id", "game_id", "dk_fpts_mean", "minutes_sim_mean", "minutes_sim_p50", "n_worlds"]],
        on=["player_id", "game_id"],
        how="inner",
        suffixes=("_unified", "_sim"),
    )
    assert (merged["dk_fpts_mean_unified"].values == merged["dk_fpts_mean_sim"].values).all()
    assert (merged["minutes_sim_mean_unified"].values == merged["minutes_sim_mean_sim"].values).all()
    assert (merged["minutes_sim_p50_unified"].values == merged["minutes_sim_p50_sim"].values).all()
    assert (merged["n_worlds_unified"].values == merged["n_worlds_sim"].values).all()

    assert 999 not in set(pd.to_numeric(unified["n_worlds"], errors="coerce").dropna().astype(int).unique())

    summary = json.loads((out_path.parent / "summary.json").read_text(encoding="utf-8"))
    assert summary["projections_run_id"] == projections_run_id
    assert summary["minutes_run_id"] == minutes_run_id
    assert summary["sim_run_id"] == sim_run_id
    assert summary["n_worlds"] == 123

    # Validate pointer consistency against the tiny fixture tree.
    check_artifact_pointers(date_str=game_date.isoformat(), data_root=data_root, strict=True)
