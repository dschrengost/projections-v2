from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.sim_v2.generate_worlds_fpts_v2 import main as generate_worlds_main


def _write_minutes_projection(root: Path, game_date: str, *, include_spread: bool) -> None:
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
            "status": "available",
            "starter_flag": 0,
            "is_projected_starter": 0,
            "play_prob": 1.0,
            "minutes_p10": 14.0,
            "minutes_p50": 18.0,
            "minutes_p90": 22.0,
        },
    ]
    if include_spread:
        for row in rows:
            row["spread_home"] = -3.5
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


def test_sim_v2_rates_outputs_minutes_stats(tmp_path: Path) -> None:
    game_date = "2025-01-01"
    _write_minutes_projection(tmp_path, game_date, include_spread=True)
    _write_rates_live(tmp_path, game_date)

    output_root = tmp_path / "out"
    generate_worlds_main(
        start_date=game_date,
        end_date=game_date,
        n_worlds=250,
        profile="debug",
        data_root=tmp_path,
        profiles_path=None,
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
    assert "minutes_sim_mean" in df.columns
    assert "minutes_sim_p50" in df.columns
    assert "minutes_sim_std" in df.columns
    assert (df["minutes_sim_std"] > 0).any()


def test_sim_v2_baseline_missing_spread_still_varies(tmp_path: Path) -> None:
    game_date = "2025-01-02"
    _write_minutes_projection(tmp_path, game_date, include_spread=False)
    _write_rates_live(tmp_path, game_date)

    output_root = tmp_path / "out"
    generate_worlds_main(
        start_date=game_date,
        end_date=game_date,
        n_worlds=250,
        profile="baseline",
        data_root=tmp_path,
        profiles_path=None,
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
    assert "minutes_sim_std" in df.columns
    assert (df["minutes_sim_std"] > 0).any()
