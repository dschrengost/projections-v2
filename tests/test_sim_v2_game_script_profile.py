from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.sim_v2.generate_worlds_fpts_v2 import main as generate_worlds_main


def _write_minutes_projection(root: Path, game_date: str) -> None:
    out_dir = root / "gold" / "projections_minutes_v1" / f"game_date={game_date}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "game_date": game_date,
            "tip_ts": f"{game_date}T00:00:00Z",
            "game_id": 1,
            "team_id": 10,
            "player_id": 100,
            "player_name": "Starter",
            "status": "available",
            "starter_flag": 1,
            "is_projected_starter": 1,
            "play_prob": 1.0,
            "spread_home": 0.0,
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
            "player_name": "Bench",
            "status": "available",
            "starter_flag": 0,
            "is_projected_starter": 0,
            "play_prob": 1.0,
            "spread_home": 0.0,
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


def test_sim_v2_game_script_quantile_targets_from_profile(tmp_path: Path) -> None:
    game_date = "2025-01-03"
    _write_minutes_projection(tmp_path, game_date)
    _write_rates_live(tmp_path, game_date)

    profiles_path = tmp_path / "profiles.json"
    profiles_path.write_text(
        json.dumps(
            {
                "profiles": {
                    "script_test": {
                        "mean_source": "rates",
                        "minutes_source": "minutes_v1",
                        "rates_source": "rates_v1_live",
                        "worlds": {"n_worlds": 50, "batch_size": 50},
                        "rates_noise": {"enabled": False, "split": "val"},
                        "minutes_noise": {"enabled": False, "sigma_min": 1.0},
                        "enforce_team_240": False,
                        "efficiency_scoring": True,
                        "game_script": {
                            "enabled": True,
                            "margin_std": 0.0,
                            "spread_coef": 0.0,
                            "quantile_noise_std": 0.0,
                            "quantile_targets": {
                                "close": {"starter": 0.8, "bench": 0.5},
                                "comfortable_win": {"starter": 0.5, "bench": 0.5},
                                "comfortable_loss": {"starter": 0.5, "bench": 0.5},
                                "blowout_win": {"starter": 0.5, "bench": 0.5},
                                "blowout_loss": {"starter": 0.5, "bench": 0.5},
                            },
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    output_root = tmp_path / "out"
    generate_worlds_main(
        start_date=game_date,
        end_date=game_date,
        n_worlds=50,
        profile="script_test",
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
    starter = df[df["player_id"] == 100].iloc[0]
    bench = df[df["player_id"] == 200].iloc[0]

    z90 = 1.2815515655446004
    z = 0.8416212335729143  # stats.norm.ppf(0.8)
    sigma_high = (36.0 - 32.0) / z90
    expected_starter = 32.0 + sigma_high * z

    assert starter["minutes_sim_p50"] > starter["minutes_mean"]
    np.testing.assert_allclose(bench["minutes_sim_p50"], bench["minutes_mean"], rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(starter["minutes_sim_p50"], expected_starter, rtol=0.0, atol=1e-6)
