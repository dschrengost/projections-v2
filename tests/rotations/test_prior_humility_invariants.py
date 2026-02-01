from __future__ import annotations

import pandas as pd

from projections.rotations.priors_humility import HumilityConfig, apply_prior_humility


def test_prior_humility_invariants() -> None:
    df = pd.DataFrame(
        {
            "game_id": ["g1"] * 6,
            "team_id": [1] * 6,
            "player_id": [201, 202, 203, 204, 205, 206],
            "minutes_prior": [36, 18, 12, 9, 6, 2],
            "minutes_p10": [30, 10, 7, 2, 0, 0],
            "minutes_p90": [40, 24, 18, 14, 10, 4],
            "play_prob": [1.0, 1.0, 1.0, 0.9, 0.8, 0.7],
            "starter_candidate": [True, True, True, True, True, False],
        }
    )
    cfg = HumilityConfig(enabled=True, top_n_lock=2, protect_starters=True, protect_top_n=True)
    out = apply_prior_humility(df, cfg)

    assert (out["minutes_prior_adj"] >= 0.0).all()
    assert (out["play_prob_adj"] >= 0.0).all()
    assert (out["play_prob_adj"] <= 1.0).all()

    assert (out["minutes_p10_adj"] <= out["minutes_prior_adj"]).all()
    assert (out["minutes_prior_adj"] <= out["minutes_p90_adj"]).all()

    fringe = out[out["humility_tier"] == "fringe"]
    if not fringe.empty:
        assert (fringe["minutes_prior_adj"] <= cfg.minutes_p50_fringe_max + 1e-9).all()
        assert (fringe["play_prob_adj"] <= cfg.cap_play_prob_fringe + 1e-9).all()
        assert (fringe["play_prob_adj"] <= (1.0 - cfg.min_p_eq0_fringe) + 1e-9).all()

