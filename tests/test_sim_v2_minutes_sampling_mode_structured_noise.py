from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.sim_v2.generate_worlds_fpts_v2 import main as generate_worlds_main


def _write_profiles(path: Path) -> None:
    payload = {
        "profiles": {
            "structured_noise": {
                "mean_source": "rates",
                "minutes_source": "minutes_v1",
                "rates_source": "rates_v1_live",
                "worlds": {"n_worlds": 20, "batch_size": 20},
                "noise": {"epsilon_dist": "normal", "nu": 5, "k_default": 0.0},
                "rates_noise": {"enabled": False},
                "minutes_noise": {"enabled": False, "sigma_min": 0.0},
                "minutes_noise_config": {
                    "enabled": True,
                    "sigma_starter": 0.0,
                    "sigma_bench": 0.0,
                    "min_minutes_for_noise": 0.0,
                    "cap_abs": 0.0,
                    "use_student_t": False,
                    "t_df": 8,
                    "lo_source": "zero",
                    "hi_source": "p90",
                    "lo_pad": 0.0,
                    "hi_pad": 0.0,
                },
                "minutes_sampling_mode": "structured_noise",
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
    out_dir = root / "gold" / "projections_minutes_v1" / f"game_date={game_date}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    # One team with minutes that do NOT sum to 240 (200 total).
    for pid, m50 in enumerate([40.0, 40.0, 40.0, 40.0, 40.0], start=100):
        rows.append(
            {
                "game_date": game_date,
                "tip_ts": f"{game_date}T00:00:00Z",
                "game_id": 1,
                "team_id": 10,
                "player_id": pid,
                "player_name": f"P{pid}",
                "status": "available",
                "starter_flag": 1,
                "is_projected_starter": 1,
                "play_prob": 1.0,
                "minutes_p10": 0.0,
                "minutes_p50": m50,
                "minutes_p90": 48.0,
            }
        )
    pd.DataFrame(rows).to_parquet(out_dir / "minutes.parquet", index=False)


def _write_rates_live(root: Path, game_date: str) -> None:
    out_dir = root / "gold" / "rates_v1_live" / game_date
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for pid in range(100, 105):
        rows.append(
            {
                "game_date": game_date,
                "game_id": 1,
                "team_id": 10,
                "player_id": pid,
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


def test_structured_noise_sampling_projects_team_to_240(tmp_path: Path) -> None:
    game_date = "2025-01-04"
    _write_minutes_projection(tmp_path, game_date)
    _write_rates_live(tmp_path, game_date)
    profiles_path = tmp_path / "sim_profiles.json"
    _write_profiles(profiles_path)

    output_root = tmp_path / "out"
    generate_worlds_main(
        start_date=game_date,
        end_date=game_date,
        n_worlds=20,
        profile="structured_noise",
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

    team_sum = float(df[df["team_id"] == 10]["minutes_sim_mean"].sum())
    assert 239.0 <= team_sum <= 241.0, f"team minutes_sim_mean sum not ~240: {team_sum}"

