from __future__ import annotations

import pandas as pd

from projections.rotations.priors_humility import HumilityConfig, apply_prior_humility


def test_prior_humility_heuristics_cap_fringe_ge5_and_do_not_promote() -> None:
    df = pd.DataFrame(
        {
            "game_id": ["g1", "g1", "g1"],
            "team_id": [1, 1, 1],
            "player_id": [101, 102, 103],
            "starter_candidate": [False, False, False],
            # Fringe, dead-ish: p90 < 5 but upstream model spuriously thinks ge5 is high.
            "minutes_prior": [2.0, 28.0, 2.0],
            "minutes_p10": [0.0, 24.0, 0.0],
            "minutes_p90": [4.0, 30.0, 4.0],
            "play_prob": [1.0, 1.0, 1.0],
            "p_played_ge_5_pred": [0.90, 0.90, 0.02],
        }
    )
    cfg = HumilityConfig(
        enabled=True,
        protect_starters=False,
        protect_top_n=False,
        use_rotation_prior_heuristics=True,
    )
    out = apply_prior_humility(df, cfg).set_index("player_id", drop=False)

    # Fringe row gets capped down hard by heuristics (suppression).
    assert float(out.loc[101, "p_played_ge_5_pred_adj"]) <= 0.05 + 1e-12
    assert float(out.loc[101, "p_played_ge_5_pred_adj"]) <= float(out.loc[101, "p_played_ge_5_pred"]) + 1e-12

    # Rotation-ish row should not be suppressed (core tier).
    assert out.loc[102, "humility_tier"] == "core"
    assert float(out.loc[102, "p_played_ge_5_pred_adj"]) == float(out.loc[102, "p_played_ge_5_pred"])

    # If baseline is already low, heuristics never promote upward.
    assert float(out.loc[103, "p_played_ge_5_pred_adj"]) == float(out.loc[103, "p_played_ge_5_pred"])


def test_prior_humility_heuristics_floor_p0_suppression_only() -> None:
    df = pd.DataFrame(
        {
            "game_id": ["g1"],
            "team_id": [1],
            "player_id": [201],
            "starter_candidate": [False],
            "minutes_prior": [2.0],
            "minutes_p10": [0.0],
            "minutes_p90": [4.0],
            "play_prob": [1.0],
            "p_minutes_eq0_pred": [0.05],
        }
    )
    cfg = HumilityConfig(
        enabled=True,
        protect_starters=False,
        protect_top_n=False,
        use_rotation_prior_heuristics=True,
    )
    out = apply_prior_humility(df, cfg)
    assert float(out.loc[0, "p_minutes_eq0_pred_adj"]) >= float(out.loc[0, "p_minutes_eq0_pred"]) - 1e-12

