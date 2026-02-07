from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.sim_v2.generate_worlds_fpts_v2 import main as generate_worlds_main


def _write_profiles(path: Path) -> None:
    payload = {
        "profiles": {
            "test_forced_inactive": {
                "mean_source": "rates",
                "minutes_source": "minutes_v1",
                "rates_source": "rates_v1_live",
                "worlds": {"n_worlds": 120, "batch_size": 120},
                "noise": {"epsilon_dist": "normal", "nu": 5, "k_default": 0.0},
                "rates_noise": {"enabled": False},
                "minutes_noise": {"enabled": False, "sigma_min": 0.0},
                "minutes_worlds": {"mode": "model_space_v1", "gate_temperature": 1.0},
                "team_factor_sigma": 0.0,
                "team_factor_gamma": 1.0,
                "enforce_team_240": False,
                "use_play_prob_masking": True,
                "seed": 1337,
            }
        }
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_minutes_projection(root: Path, game_date: str, forced_player_id: int) -> None:
    base_dir = root / "gold" / "projections_minutes_v1" / f"game_date={game_date}"
    base_dir.mkdir(parents=True, exist_ok=True)
    run_id = "test"
    (base_dir / "latest_run.json").write_text(json.dumps({"run_id": run_id}), encoding="utf-8")
    out_dir = base_dir / f"run={run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    team_player_ids = {
        10: [1001, 1002, 1003, forced_player_id, 1005],
        20: [2001, 2002, 2003, 2004, 2005],
    }
    for team_id, player_ids in team_player_ids.items():
        for idx, pid in enumerate(player_ids):
            minutes = 31.0 - idx * 2.0
            row: dict[str, object] = {
                "game_date": game_date,
                "tip_ts": f"{game_date}T00:00:00Z",
                "game_id": 1,
                "team_id": team_id,
                "player_id": pid,
                "player_name": f"P{pid}",
                "status": "available",
                "starter_flag": 1 if idx < 3 else 0,
                "is_projected_starter": 1 if idx < 3 else 0,
                "play_prob": 0.9 if idx < 4 else 0.7,
                "minutes_p10": max(minutes - 4.0, 0.0),
                "minutes_p50": minutes,
                "minutes_p90": min(minutes + 4.0, 48.0),
                "minutes_lock_eff": False,
                "minutes_target_eff": minutes,
                "ops_override_applied": False,
            }
            if pid == forced_player_id:
                row["play_prob"] = 0.8
                row["minutes_p10"] = 0.0
                row["minutes_p50"] = 0.2
                row["minutes_p90"] = 1.0
                row["minutes_lock_eff"] = True
                row["minutes_target_eff"] = 0.2
                row["ops_override_applied"] = True
            rows.append(row)

    pd.DataFrame(rows).to_parquet(out_dir / "minutes.parquet", index=False)


def _write_rates_live(root: Path, game_date: str, player_ids: list[int]) -> None:
    out_dir = root / "gold" / "rates_v1_live" / game_date
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for pid in player_ids:
        team_id = 10 if str(pid).startswith("1") else 20
        rows.append(
            {
                "game_date": game_date,
                "game_id": 1,
                "team_id": team_id,
                "player_id": pid,
                "fga2_per_min": 0.5,
                "fga3_per_min": 0.25,
                "fta_per_min": 0.18,
                "ast_per_min": 0.10,
                "tov_per_min": 0.06,
                "oreb_per_min": 0.05,
                "dreb_per_min": 0.13,
                "stl_per_min": 0.03,
                "blk_per_min": 0.02,
                "fg2_pct": 0.52,
                "fg3_pct": 0.36,
                "ft_pct": 0.79,
            }
        )
    pd.DataFrame(rows).to_parquet(out_dir / "rates.parquet", index=False)


def test_sim_v2_forces_manual_near_zero_lock_overrides_inactive(tmp_path: Path) -> None:
    game_date = "2025-01-06"
    forced_player_id = 1004

    _write_minutes_projection(tmp_path, game_date, forced_player_id)
    all_player_ids = [1001, 1002, 1003, 1004, 1005, 2001, 2002, 2003, 2004, 2005]
    _write_rates_live(tmp_path, game_date, all_player_ids)
    profiles_path = tmp_path / "sim_profiles.json"
    _write_profiles(profiles_path)

    output_root = tmp_path / "out"
    generate_worlds_main(
        start_date=game_date,
        end_date=game_date,
        n_worlds=120,
        profile="test_forced_inactive",
        data_root=tmp_path,
        profiles_path=profiles_path,
        output_root=output_root,
        sim_run_id=None,
        use_rates_noise=False,
        rates_noise_split=None,
        team_sigma_scale=None,
        player_sigma_scale=None,
        rates_run_id=None,
        minutes_run_id=None,
        use_minutes_noise=False,
        minutes_noise_run_id=None,
        minutes_sigma_min=None,
        seed=1337,
        min_play_prob=None,
        team_factor_sigma=None,
        team_factor_gamma=None,
        use_efficiency_scoring=True,
    )

    run_dir = output_root / f"game_date={game_date}"
    proj_df = pd.read_parquet(run_dir / "projections.parquet")
    worlds_df = pd.read_parquet(run_dir / "worlds_matrix.parquet")

    forced_row = proj_df.loc[proj_df["player_id"] == forced_player_id].iloc[0]
    assert float(forced_row["play_prob"]) > 0.0
    assert float(forced_row["play_prob_eff"]) == 0.0
    assert float(forced_row["sim_p_available"]) == 0.0
    assert float(forced_row["sim_p_active"]) == 0.0
    assert float(forced_row["sim_p_rotation"]) == 0.0

    forced_worlds = worlds_df[str(forced_player_id)].to_numpy(dtype=float)
    assert np.allclose(forced_worlds, 0.0, atol=1e-9)

    # Sanity check: simulation still produced non-zero worlds for the slate.
    assert float(worlds_df.drop(columns=[str(forced_player_id)]).to_numpy(dtype=float).sum()) > 0.0
