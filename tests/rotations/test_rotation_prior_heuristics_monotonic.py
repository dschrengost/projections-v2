from __future__ import annotations

import pandas as pd

from projections.rotations.rotation_prior_heuristics import derive_rotation_priors


def test_rotation_prior_heuristics_monotonic_minutes_prior_same_p90_bucket() -> None:
    df = pd.DataFrame(
        {
            "minutes_prior": [4.0, 6.0, 9.0],
            "minutes_p10": [0.0, 0.0, 0.0],
            "minutes_p90": [10.0, 10.0, 10.0],
        }
    )
    out = derive_rotation_priors(df)
    vals = out["p_ge5_prior_heur"].tolist()
    assert vals == sorted(vals)


def test_rotation_prior_heuristics_p90_below_5_forces_low_ge5_and_high_eq0() -> None:
    df = pd.DataFrame(
        {
            "minutes_prior": [20.0],
            "minutes_p10": [0.0],
            "minutes_p90": [4.0],
        }
    )
    out = derive_rotation_priors(df)
    assert float(out.loc[0, "p_ge5_prior_heur"]) <= 0.05 + 1e-12
    assert float(out.loc[0, "p_eq0_prior_heur"]) >= 0.70 - 1e-12


def test_rotation_prior_heuristics_rotationish_thresholds() -> None:
    df = pd.DataFrame(
        {
            "minutes_prior": [24.0, 6.0],
            "minutes_p10": [0.0, 0.0],
            "minutes_p90": [8.0, 28.0],
        }
    )
    out = derive_rotation_priors(df)
    assert float(out.loc[0, "p_ge5_prior_heur"]) >= 0.97 - 1e-12
    assert float(out.loc[0, "p_eq0_prior_heur"]) <= 0.05 + 1e-12
    assert float(out.loc[1, "p_ge5_prior_heur"]) >= 0.97 - 1e-12
    assert float(out.loc[1, "p_eq0_prior_heur"]) <= 0.05 + 1e-12

