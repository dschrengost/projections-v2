from __future__ import annotations

import numpy as np
import pandas as pd

from projections.sim_v2.minutes_noise import enforce_team_240_minutes
from scripts.sim_v2.generate_worlds_fpts_v2 import build_rates_mean_fpts


def test_rotalloc_promote_smoke_respects_eligible_and_sums() -> None:
    game_date = pd.Timestamp("2025-01-01")
    minutes_df = pd.DataFrame(
        {
            "game_date": [game_date] * 6,
            "game_id": [1] * 3 + [2] * 3,
            "team_id": [10] * 3 + [20] * 3,
            "player_id": [101, 102, 103, 201, 202, 203],
            # Eligible players sum to 240 per team-game.
            "minutes_mean": [200.0, 40.0, 0.0, 180.0, 60.0, 0.0],
            "eligible_flag": [1, 1, 0, 1, 1, 0],
            "minutes_alloc_mode": ["rotalloc_expk"] * 6,
            "play_prob": [1.0] * 6,
            "is_starter": [1, 0, 0, 1, 0, 0],
            "rotation_prob": [1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
        }
    )

    rates_df = minutes_df.loc[:, ["game_date", "game_id", "team_id", "player_id"]].copy()
    for col in (
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
        rates_df[col] = 0.0

    mu_df = build_rates_mean_fpts(minutes_df, rates_df)
    assert "eligible_flag" in mu_df.columns
    assert "minutes_alloc_mode" in mu_df.columns

    team_codes = mu_df["team_id"].astype("category")
    team_indices = team_codes.cat.codes.to_numpy(dtype=int)
    n_teams = int(team_indices.max()) + 1
    minutes_mean = mu_df["minutes_mean"].to_numpy(dtype=float)
    eligible = pd.to_numeric(mu_df["eligible_flag"], errors="coerce").fillna(0.0).to_numpy(dtype=float) > 0.5

    # Simulate a no-noise minutes_world and apply eligibility gating.
    n_worlds = 3
    minutes_world = np.broadcast_to(minutes_mean[None, :], (n_worlds, len(minutes_mean))).copy()
    active_mask = np.broadcast_to(eligible[None, :], minutes_world.shape)
    minutes_world = minutes_world * active_mask.astype(float)

    # Enforce team 240 with a cap size large enough to retain the eligible set.
    eligible_size_max = int(
        mu_df.assign(_eligible=eligible.astype(int)).groupby(["game_id", "team_id"])["_eligible"].sum().max()
    )
    rotation_mask = eligible.copy()
    bench_mask = (~rotation_mask) & (minutes_mean > 0.0)
    out = enforce_team_240_minutes(
        minutes_world=minutes_world,
        team_indices=team_indices,
        rotation_mask=rotation_mask,
        bench_mask=bench_mask,
        baseline_minutes=minutes_mean,
        active_mask=active_mask,
        starter_mask=pd.to_numeric(mu_df.get("is_starter", 0), errors="coerce").fillna(0).to_numpy() > 0,
        max_rotation_size=eligible_size_max,
    )

    assert np.all(out[:, ~eligible] == 0.0)
    for t in range(n_teams):
        idxs = team_indices == t
        assert np.max(np.abs(out[:, idxs].sum(axis=1) - 240.0)) < 1e-6

