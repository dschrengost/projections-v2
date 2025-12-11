from __future__ import annotations

import numpy as np
import pandas as pd

from projections.fpts_v1.eval import evaluate_fpts_run


def test_evaluate_fpts_run_produces_expected_slices() -> None:
    df = pd.DataFrame(
        {
            "game_id": [101, 101, 102],
            "player_id": [10, 11, 12],
            "actual_minutes": [32.0, 22.0, 36.0],
            "actual_fpts": [34.0, 18.0, 48.0],
            "minutes_p10_pred": [25.0, 15.0, 30.0],
            "minutes_p50_pred": [32.0, 24.0, 34.0],
            "minutes_p90_pred": [38.0, 32.0, 40.0],
            "minutes_volatility_pred": [3.0, 7.0, 9.0],
            "teammate_out_count": [0, 2, 4],
            "dk_salary": [3500, 5500, 9200],
            "proj_fpts": [18.0, 32.0, 55.0],
        }
    )
    df["fpts_per_min_actual"] = df["actual_fpts"] / df["actual_minutes"]

    model_preds = np.array([1.05, 0.75, 1.3])
    baseline_preds = np.array([0.9, 0.65, 1.1])
    metrics = evaluate_fpts_run(df, model_preds=model_preds, baseline_preds=baseline_preds)

    assert "overall" in metrics
    overall = metrics["overall"]
    assert overall["rows"] == 3
    assert overall["model"]["mae_fpts"] != overall["baseline"]["mae_fpts"]

    proj_buckets = metrics.get("by_proj_fpts_bucket", {})
    assert "<20 FPTS" in proj_buckets
    assert proj_buckets["<20 FPTS"]["rows"] == 1

    minute_buckets = metrics.get("by_minutes_bucket", {})
    assert ">=34 minutes" in minute_buckets

    injury_buckets = metrics.get("by_injury_context", {})
    assert "3+_out" in injury_buckets

    volatility_buckets = metrics.get("by_volatility_bucket", {})
    assert "4-8 volatility" in volatility_buckets

    salary_buckets = metrics.get("by_salary_bucket", {})
    assert ">=8k salary" in salary_buckets


def test_evaluate_fpts_run_handles_nan_starter_flags() -> None:
    df = pd.DataFrame(
        {
            "game_id": [201, 201],
            "player_id": [21, 22],
            "actual_minutes": [30.0, 28.0],
            "actual_fpts": [36.0, 24.0],
            "starter_flag": [1, pd.NA],
            "minutes_p10_pred": [20.0, 18.0],
            "minutes_p50_pred": [30.0, 28.0],
            "minutes_p90_pred": [34.0, 32.0],
            "proj_fpts": [40.0, 28.0],
        }
    )
    df["fpts_per_min_actual"] = df["actual_fpts"] / df["actual_minutes"]
    model_preds = np.array([1.2, 0.9])
    baseline_preds = np.array([1.1, 0.8])

    metrics = evaluate_fpts_run(df, model_preds=model_preds, baseline_preds=baseline_preds)
    assert metrics["overall"]["rows"] == 2
    # Presence of starter flag NaN should not break the role buckets.
    role_buckets = metrics.get("by_role_context", {})
    assert "starters" in role_buckets
