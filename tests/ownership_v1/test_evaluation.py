"""Tests for ownership_v1 evaluation utilities."""

import numpy as np
import pandas as pd

from projections.ownership_v1.evaluation import (
    OwnershipEvalSlice,
    compute_calibration_table,
    evaluate_predictions,
)


def test_compute_calibration_table_ece_zero_for_perfect_calibration():
    df = pd.DataFrame(
        {
            "actual_own_pct": [0.0, 1.0, 5.0, 20.0, 50.0],
            "pred_own_pct": [0.0, 1.0, 5.0, 20.0, 50.0],
        }
    )
    ece, table = compute_calibration_table(df, actual_col="actual_own_pct", pred_col="pred_own_pct")
    assert np.isclose(ece, 0.0)
    assert not table.empty


def test_evaluate_predictions_scale_to_sum_enforces_target_per_slate():
    df = pd.DataFrame(
        {
            "slate_id": ["A"] * 3 + ["B"] * 2,
            "player_id": ["1", "2", "3", "1", "2"],
            "game_date": ["2025-01-01"] * 5,
            "actual_own_pct": [50.0, 30.0, 20.0, 60.0, 40.0],
            "pred_own_pct": [10.0, 10.0, 10.0, 5.0, 5.0],
        }
    )
    res = evaluate_predictions(
        df,
        slice_name="unit",
        target_sum_pct=800.0,
        normalization="scale_to_sum",
    )
    # Under scale_to_sum, sums must be exactly the target (up to float error)
    # We re-check via the internal contract: mean_abs_sum_error_to_target should be ~0.
    assert res.sums.mean_abs_sum_error_to_target < 1e-6


def test_scale_to_sum_clips_negative_before_normalizing():
    df = pd.DataFrame(
        {
            "slate_id": ["A"] * 3,
            "player_id": ["1", "2", "3"],
            "game_date": ["2025-01-01"] * 3,
            "actual_own_pct": [50.0, 30.0, 20.0],
            # One negative model output (possible for unconstrained regressors)
            "pred_own_pct": [10.0, -5.0, 15.0],
        }
    )
    res = evaluate_predictions(
        df,
        slice_name="unit",
        target_sum_pct=800.0,
        normalization="scale_to_sum",
    )
    assert res.sums.mean_abs_sum_error_to_target < 1e-6


def test_eval_slice_filters_slate_ids_and_data_source():
    df = pd.DataFrame(
        {
            "slate_id": ["A", "A", "B", "B"],
            "data_source": ["dk", "linestar", "dk", "dk"],
            "actual_own_pct": [1.0, 1.0, 1.0, 1.0],
            "pred_own_pct": [1.0, 1.0, 1.0, 1.0],
        }
    )
    sl = OwnershipEvalSlice(name="s", slate_ids=["A", "B"], data_source="dk")
    out = sl.filter_df(df)
    assert out["slate_id"].nunique() == 2
    assert (out["data_source"] == "dk").all()
