from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.sim_v2.generate_worlds_fpts_v2 import main as generate_worlds_main


def _write_profiles(path: Path) -> None:
    payload = {
        "profiles": {
            "vegas_env_deterministic": {
                "mean_source": "rates",
                "minutes_source": "minutes_v1",
                "rates_source": "rates_v1_live",
                "worlds": {"n_worlds": 2000, "batch_size": 2000},
                "noise": {
                    "epsilon_dist": "normal",
                    "nu": 5,
                    "k_default": 0.0,
                    "vegas_env": {
                        "enabled": True,
                        "dist": "normal",
                        "total_sigma": 0.0,
                        "spread_sigma": 0.0,
                        "pace_weight": 0.0,
                        "pace_sigma": 0.0,
                        "delta_iters": 20,
                    },
                },
                "rates_noise": {"enabled": False},
                "minutes_noise": {"enabled": False, "sigma_min": 0.0},
                "team_factor_sigma": 0.0,
                "team_factor_gamma": 1.0,
                "enforce_team_240": False,
                "efficiency_scoring": True,
                "use_play_prob_masking": False,
                "seed": 1337,
            }
        }
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_schedule(root: Path, game_date: str) -> None:
    # _load_schedule_for_date: season is (year-1) for months < 8.
    season = 2024
    month = 1
    out_dir = root / "silver" / "schedule" / f"season={season}" / f"month={month:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "game_date": game_date,
                "game_id": 1,
                "home_team_id": 10,
                "away_team_id": 20,
            }
        ]
    ).to_parquet(out_dir / "schedule.parquet", index=False)


def _write_minutes_projection(root: Path, game_date: str, *, total: float, spread_home: float) -> None:
    out_dir = root / "gold" / "projections_minutes_v1" / f"game_date={game_date}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for team_id, player_id in ((10, 100), (20, 200)):
        rows.append(
            {
                "game_date": game_date,
                "tip_ts": f"{game_date}T00:00:00Z",
                "game_id": 1,
                "team_id": team_id,
                "player_id": player_id,
                "player_name": f"P{player_id}",
                "status": "available",
                "starter_flag": 1,
                "is_projected_starter": 1,
                "is_starter": 1,
                "play_prob": 1.0,
                "minutes_p10": 40.0,
                "minutes_p50": 40.0,
                "minutes_p90": 40.0,
                "total": total,
                "spread_home": spread_home,
            }
        )
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
                "fga2_per_min": 1.0,
                "fga3_per_min": 1.0,
                "fta_per_min": 0.5,
                "ast_per_min": 0.0,
                "tov_per_min": 0.0,
                "oreb_per_min": 0.0,
                "dreb_per_min": 0.0,
                "stl_per_min": 0.0,
                "blk_per_min": 0.0,
                "fg2_pct": 0.5,
                "fg3_pct": 0.35,
                "ft_pct": 0.8,
            }
        )
    pd.DataFrame(rows).to_parquet(out_dir / "rates.parquet", index=False)


def test_vegas_env_matches_implied_team_points_when_sigmas_zero(tmp_path: Path) -> None:
    game_date = "2025-01-06"
    total = 220.0
    spread_home = -4.0  # home favored by 4

    _write_schedule(tmp_path, game_date)
    _write_minutes_projection(tmp_path, game_date, total=total, spread_home=spread_home)
    _write_rates_live(tmp_path, game_date)
    profiles_path = tmp_path / "sim_profiles.json"
    _write_profiles(profiles_path)

    output_root = tmp_path / "out"
    generate_worlds_main(
        start_date=game_date,
        end_date=game_date,
        n_worlds=2000,
        profile="vegas_env_deterministic",
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
    assert {"team_id", "pts_mean"}.issubset(df.columns)

    pts_by_team = df.groupby("team_id")["pts_mean"].sum().to_dict()
    assert 10 in pts_by_team and 20 in pts_by_team

    implied_home = total / 2.0 - spread_home / 2.0
    implied_away = total - implied_home

    assert np.isclose(float(pts_by_team[10]), implied_home, rtol=0.0, atol=0.75)
    assert np.isclose(float(pts_by_team[20]), implied_away, rtol=0.0, atol=0.75)

