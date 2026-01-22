from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.sim_v2.generate_worlds_fpts_v2 import main as generate_worlds_main


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
            "is_starter": 1,
            "play_prob": 1.0,
            "minutes_p10": 20.0,
            "minutes_p50": 20.0,
            "minutes_p90": 20.0,
        }
    ]
    pd.DataFrame(rows).to_parquet(out_dir / "minutes.parquet", index=False)


def _write_rates_live(root: Path, game_date: str) -> None:
    out_dir = root / "gold" / "rates_v1_live" / game_date
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "game_date": game_date,
            "game_id": 1,
            "team_id": 10,
            "player_id": 100,
            "fga2_per_min": 0.0,
            "fga3_per_min": 0.0,
            "fta_per_min": 0.0,
            "ast_per_min": 0.0,
            "tov_per_min": 0.0,
            "oreb_per_min": 0.0,
            "dreb_per_min": 0.0,
            "stl_per_min": 0.02,  # mu ~= 0.4 steals at 20 minutes
            "blk_per_min": 0.0,
            "fg2_pct": 0.5,
            "fg3_pct": 0.35,
            "ft_pct": 0.8,
        }
    ]
    pd.DataFrame(rows).to_parquet(out_dir / "rates.parquet", index=False)


def test_stat_sampling_produces_nontrivial_upper_tail_for_low_mean_counts(tmp_path: Path) -> None:
    game_date = "2025-01-05"
    _write_minutes_projection(tmp_path, game_date)
    _write_rates_live(tmp_path, game_date)

    profiles_path = tmp_path / "sim_profiles.json"
    profiles_path.write_text(
        json.dumps(
            {
                "profiles": {
                    "stat_sampling_test": {
                        "mean_source": "rates",
                        "minutes_source": "minutes_v1",
                        "rates_source": "rates_v1_live",
                        "worlds": {"n_worlds": 5000, "batch_size": 5000},
                        "rates_noise": {"enabled": False},
                        "minutes_noise": {"enabled": False, "sigma_min": 0.0},
                        "enforce_team_240": False,
                        "efficiency_scoring": True,
                        "use_play_prob_masking": False,
                        "seed": 1337,
                        "noise": {
                            "epsilon_dist": "normal",
                            "nu": 5,
                            "k_default": 0.0,
                            "stat_sampling": {"enabled": True},
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
        n_worlds=5000,
        profile="stat_sampling_test",
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
    assert len(df) == 1

    # For mu~0.4 steals, Poisson p95 is 2 steals => 4 DK points from steals alone.
    assert float(df["dk_fpts_p95"].iloc[0]) >= 4.0
