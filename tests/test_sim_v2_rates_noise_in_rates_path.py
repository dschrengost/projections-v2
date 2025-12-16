from __future__ import annotations

import json
from pathlib import Path

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
            "player_name": "A",
            "status": "available",
            "starter_flag": 1,
            "is_projected_starter": 1,
            "play_prob": 1.0,
            "minutes_p10": 30.0,
            "minutes_p50": 32.0,
            "minutes_p90": 34.0,
        },
        {
            "game_date": game_date,
            "tip_ts": f"{game_date}T00:00:00Z",
            "game_id": 1,
            "team_id": 10,
            "player_id": 101,
            "player_name": "B",
            "status": "available",
            "starter_flag": 1,
            "is_projected_starter": 1,
            "play_prob": 1.0,
            "minutes_p10": 30.0,
            "minutes_p50": 32.0,
            "minutes_p90": 34.0,
        },
    ]
    pd.DataFrame(rows).to_parquet(out_dir / "minutes.parquet", index=False)


def _write_rates_live(root: Path, game_date: str) -> None:
    out_dir = root / "gold" / "rates_v1_live" / game_date
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for player_id in (100, 101):
        rows.append(
            {
                "game_date": game_date,
                "game_id": 1,
                "team_id": 10,
                "player_id": player_id,
                "fga2_per_min": 0.5,
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


def _write_rates_noise(root: Path, run_id: str) -> None:
    out_dir = root / "artifacts" / "sim_v2" / "rates_noise"
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = {}
    for name in (
        "fga2_per_min",
        "fga3_per_min",
        "fta_per_min",
        "ast_per_min",
        "tov_per_min",
        "oreb_per_min",
        "dreb_per_min",
        "stl_per_min",
        "blk_per_min",
    ):
        targets[name] = {"sigma_team": 0.0, "sigma_player": 1.0}
    payload = {"run_id": run_id, "split": "val", "targets": targets}
    (out_dir / f"{run_id}_val_noise.json").write_text(json.dumps(payload), encoding="utf-8")


def test_rates_noise_applies_in_rates_mean_source(tmp_path: Path) -> None:
    game_date = "2025-01-04"
    _write_minutes_projection(tmp_path, game_date)
    _write_rates_live(tmp_path, game_date)
    _write_rates_noise(tmp_path, run_id="dummy_rates_noise")

    profiles_path = tmp_path / "profiles.json"
    profiles_path.write_text(
        json.dumps(
            {
                "profiles": {
                    "rates_noise_test": {
                        "mean_source": "rates",
                        "minutes_source": "minutes_v1",
                        "rates_source": "rates_v1_live",
                        "worlds": {"n_worlds": 200, "batch_size": 200},
                        "rates_noise": {"enabled": True, "split": "val", "run_id": "dummy_rates_noise"},
                        "minutes_noise": {"enabled": False, "sigma_min": 1.0},
                        "enforce_team_240": False,
                        "efficiency_scoring": True,
                        "noise": {"epsilon_dist": "student_t", "nu": 5, "k_default": 0.0},
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
        n_worlds=200,
        profile="rates_noise_test",
        data_root=tmp_path,
        profiles_path=profiles_path,
        output_root=output_root,
        fpts_run_id=None,
        use_rates_noise=None,
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
    assert "dk_fpts_std" in df.columns
    # If rates_noise isn't applied, k_default=0 would produce zero variance.
    assert (df["dk_fpts_std"] > 0).any()

