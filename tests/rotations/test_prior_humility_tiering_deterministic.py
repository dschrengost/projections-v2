from __future__ import annotations

import pandas as pd

from projections.rotations.priors_humility import HumilityConfig, apply_prior_humility


def test_prior_humility_tiering_deterministic() -> None:
    # One team-game, 12 players with a minutes_p50 spread + a tie at the top_n boundary.
    df = pd.DataFrame(
        {
            "game_id": ["g1"] * 12,
            "team_id": [1] * 12,
            "player_id": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112],
            "minutes_prior": [32, 30, 28, 26, 24, 18, 16, 14, 14, 10, 7, 2],
            # Choose p10/p90 so implied P(minutes>=5) is already under caps for bench/fringe.
            "minutes_p10": [28, 26, 24, 22, 20, 10, 9, 8, 8, 2, 0, 0],
            "minutes_p90": [36, 34, 32, 30, 28, 22, 20, 18, 18, 12, 4, 3],
            "play_prob": [1.0] * 12,
            "starter_candidate": [True] * 5 + [False] * 7,
        }
    )

    cfg = HumilityConfig(top_n_lock=8, protect_starters=True, protect_top_n=True, enabled=True)

    out1 = apply_prior_humility(df, cfg)
    out2 = apply_prior_humility(df, cfg)
    pd.testing.assert_frame_equal(out1, out2, check_like=False)

    tiers = dict(zip(out1["player_id"].tolist(), out1["humility_tier"].tolist()))

    # Starters protected.
    for pid in [101, 102, 103, 104, 105]:
        assert tiers[pid] == "starter"

    # Top-N (by minutes_p50, tie-broken by player_id asc) protected among non-starters.
    assert tiers[106] == "top_n"
    assert tiers[107] == "top_n"
    assert tiers[108] == "top_n"
    assert tiers[109] != "top_n"  # tie boundary falls to lower player_id

    # Bench / fringe tiering by minutes_p50.
    assert tiers[110] == "bench"
    assert tiers[111] == "fringe"
    assert tiers[112] == "fringe"

    # Fringe players get humbled.
    fringe = out1[out1["humility_tier"] == "fringe"].set_index("player_id", drop=False)
    assert float(fringe.loc[111, "minutes_prior_adj"]) < float(fringe.loc[111, "minutes_prior"])
    assert float(fringe.loc[112, "minutes_prior_adj"]) < float(fringe.loc[112, "minutes_prior"])
    assert float(fringe.loc[111, "play_prob_adj"]) <= cfg.cap_play_prob_fringe

