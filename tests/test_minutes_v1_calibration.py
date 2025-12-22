from __future__ import annotations

import pandas as pd

from projections.minutes_v1 import calibration


def test_fit_k_params_and_apply_k_params_produces_monotone_quantiles() -> None:
    df = pd.DataFrame(
        {
            "starter_flag": [1, 1, 0, 0],
            "status": ["OK", "OK", "OK", "OK"],
            "minutes": [32.0, 28.0, 18.0, 22.0],
            "minutes_p10": [22.0, 18.0, 8.0, 12.0],
            "minutes_p50": [32.0, 28.0, 18.0, 22.0],
            "minutes_p90": [40.0, 38.0, 28.0, 32.0],
        }
    )

    params = calibration.fit_k_params(
        df,
        group_keys=["starter_flag"],
        min_count=1,
        playable_minutes_threshold=0.0,
    )
    adjusted = calibration.apply_k_params(
        df,
        params,
        out_p10_col="minutes_p10_cal",
        out_p50_col="minutes_p50_cal",
        out_p90_col="minutes_p90_cal",
    )

    assert {"minutes_p10_cal", "minutes_p50_cal", "minutes_p90_cal"}.issubset(adjusted.columns)
    assert (adjusted["minutes_p10_cal"] <= adjusted["minutes_p50_cal"]).all()
    assert (adjusted["minutes_p50_cal"] <= adjusted["minutes_p90_cal"]).all()


def test_star_k_high_params_round_trip() -> None:
    payload = {"p50_threshold": 30.0, "k_high_star": 1.2, "group_keys": ["starter_flag"]}
    params = calibration.StarKHighParams.from_dict(payload)
    assert params.to_dict() == payload

