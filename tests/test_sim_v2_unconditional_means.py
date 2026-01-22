from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.sim_v2.generate_worlds_fpts_v2 import main as generate_worlds_main


def _write_profiles(path: Path) -> None:
    payload = {
        "profiles": {
            "test_uncond": {
                "mean_source": "rates",
                "minutes_source": "minutes_v1",
                "rates_source": "rates_v1_live",
                "worlds": {"n_worlds": 200, "batch_size": 200},
                "noise": {"epsilon_dist": "normal", "nu": 5, "k_default": 0.0},
                "rates_noise": {"enabled": False},
                "minutes_noise": {"enabled": False, "sigma_min": 0.0},
                "team_factor_sigma": 0.0,
                "team_factor_gamma": 1.0,
                "enforce_team_240": False,
                "use_play_prob_masking": False,
                "seed": 1337,
            }
        }
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_minutes_projection(root: Path, game_date: str) -> None:
    base_dir = root / "gold" / "projections_minutes_v1" / f"game_date={game_date}"
    base_dir.mkdir(parents=True, exist_ok=True)
    # generate_worlds_fpts_v2 resolves minutes_v1 via latest_run.json when run_id is not provided.
    run_id = "test"
    (base_dir / "latest_run.json").write_text(json.dumps({"run_id": run_id}), encoding="utf-8")
    out_dir = base_dir / f"run={run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "game_date": game_date,
            "tip_ts": f"{game_date}T00:00:00Z",
            "game_id": 1,
            "team_id": 10,
            "player_id": 100,
            "player_name": "A",
            "status": "available",
            "starter_flag": 1,
            "is_projected_starter": 1,
            "play_prob": 1.0,
            "minutes_p10": 28.0,
            "minutes_p50": 32.0,
            "minutes_p90": 36.0,
        },
        {
            "game_date": game_date,
            "tip_ts": f"{game_date}T00:00:00Z",
            "game_id": 1,
            "team_id": 20,
            "player_id": 200,
            "player_name": "B",
            "status": "questionable",
            "starter_flag": 0,
            "is_projected_starter": 0,
            "play_prob": 0.4,
            "minutes_p10": 14.0,
            "minutes_p50": 18.0,
            "minutes_p90": 22.0,
        },
    ]
    pd.DataFrame(rows).to_parquet(out_dir / "minutes.parquet", index=False)


def _write_rates_live(root: Path, game_date: str) -> None:
    out_dir = root / "gold" / "rates_v1_live" / game_date
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for team_id, player_id in ((10, 100), (20, 200)):
        rows.append(
            {
                "game_date": game_date,
                "game_id": 1,
                "team_id": team_id,
                "player_id": player_id,
                "fga2_per_min": 0.6,
                "fga3_per_min": 0.3,
                "fta_per_min": 0.2,
                "ast_per_min": 0.1,
                "tov_per_min": 0.05,
                "oreb_per_min": 0.05,
                "dreb_per_min": 0.15,
                "stl_per_min": 0.03,
                "blk_per_min": 0.02,
                "fg2_pct": 0.52,
                "fg3_pct": 0.36,
                "ft_pct": 0.78,
            }
        )
    pd.DataFrame(rows).to_parquet(out_dir / "rates.parquet", index=False)


def test_sim_v2_outputs_availability_weighted_means_when_masking_disabled(tmp_path: Path) -> None:
    game_date = "2025-01-03"
    _write_minutes_projection(tmp_path, game_date)
    _write_rates_live(tmp_path, game_date)
    profiles_path = tmp_path / "sim_profiles.json"
    _write_profiles(profiles_path)

    output_root = tmp_path / "out"
    generate_worlds_main(
        start_date=game_date,
        end_date=game_date,
        n_worlds=200,
        profile="test_uncond",
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

    proj_path = output_root / f"game_date={game_date}" / "projections.parquet"
    df = pd.read_parquet(proj_path)
    assert {"dk_fpts_mean", "dk_fpts_mean_uncond", "play_prob"}.issubset(df.columns)

    df = df.sort_values("player_id").reset_index(drop=True)
    cond = df["dk_fpts_mean"].to_numpy(dtype=float)
    uncond = df["dk_fpts_mean_uncond"].to_numpy(dtype=float)
    p = df["play_prob"].to_numpy(dtype=float)

    # When use_play_prob_masking=False, uncond mean should be conditional * play_prob (analytic).
    assert np.isclose(uncond[0], cond[0] * p[0], rtol=1e-6, atol=1e-6)
    assert np.isclose(uncond[1], cond[1] * p[1], rtol=1e-6, atol=1e-6)
