from __future__ import annotations

import math

import pandas as pd

from scripts.rotation.eval_game_transformer_v2 import _bench_riser_metrics, _sparse_rotation_metrics


def test_sparse_rotation_metrics_captures_next_up_underprediction() -> None:
    df = pd.DataFrame(
        [
            {
                "minutes_pred": 5.0,
                "minutes_actual": 25.0,
                "active_pred": 0,
                "lineup_starter_announced": 1,
                "prior_play_prob": 0.10,
                "minutes_from_stints_prior_20": 2.0,
                "recent_start_pct_10": 0.0,
                "started_proxy_rate_prior_10": 0.0,
                "started_proxy_rate_prior_20": 0.0,
            },
            {
                "minutes_pred": 28.0,
                "minutes_actual": 30.0,
                "active_pred": 1,
                "lineup_starter_announced": 1,
                "prior_play_prob": 0.90,
                "minutes_from_stints_prior_20": 32.0,
                "recent_start_pct_10": 1.0,
                "started_proxy_rate_prior_10": 1.0,
                "started_proxy_rate_prior_20": 1.0,
            },
            {
                "minutes_pred": 9.0,
                "minutes_actual": 22.0,
                "active_pred": 1,
                "lineup_starter_announced": 0,
                "prior_play_prob": 0.08,
                "minutes_from_stints_prior_20": 3.0,
                "recent_start_pct_10": 0.0,
                "started_proxy_rate_prior_10": 0.0,
                "started_proxy_rate_prior_20": 0.0,
            },
        ]
    )

    out = _sparse_rotation_metrics(
        df,
        active_threshold=4.0,
        low_minutes_threshold=8.0,
        sparse_prior_play_prob_max=0.20,
        sparse_prior_minutes_max=6.0,
        starter_promotion_prior_minutes_max=12.0,
        starter_promotion_hist_start_rate_max=0.20,
        next_up_actual_min=20.0,
        next_up_pred_min=10.0,
    )

    assert out["slices"]["starters"]["n"] == 2
    assert out["slices"]["starters"]["pred_low_minutes_rate"] == 0.5
    assert out["slices"]["starters"]["active_recall"] == 0.5
    assert out["slices"]["starter_promotion_candidate"]["n"] == 1
    assert out["failure_rates"]["starter_sparse_next_up_underprediction_rate"] == 1.0
    assert out["failure_rates"]["starter_promotion_next_up_underprediction_rate"] == 1.0
    assert out["failure_rates"]["sparse_next_up_underprediction_rate"] == 1.0


def test_sparse_rotation_metrics_handles_missing_context_columns() -> None:
    df = pd.DataFrame(
        [
            {"minutes_pred": 12.0, "minutes_actual": 15.0, "active_pred": 1},
            {"minutes_pred": 4.0, "minutes_actual": 18.0, "active_pred": 0},
        ]
    )

    out = _sparse_rotation_metrics(
        df,
        active_threshold=4.0,
        low_minutes_threshold=8.0,
        sparse_prior_play_prob_max=0.20,
        sparse_prior_minutes_max=6.0,
        starter_promotion_prior_minutes_max=12.0,
        starter_promotion_hist_start_rate_max=0.20,
        next_up_actual_min=20.0,
        next_up_pred_min=10.0,
    )

    assert out["fields_available"]["starter_fields"] == []
    assert not out["fields_available"]["has_prior_play_prob"]
    assert not out["fields_available"]["has_minutes_from_stints_prior_20"]
    assert out["fields_available"]["historical_start_fields"] == []
    assert out["slices"]["starters"]["n"] == 0
    assert math.isnan(out["failure_rates"]["sparse_next_up_underprediction_rate"])
    assert math.isnan(out["failure_rates"]["starter_promotion_next_up_underprediction_rate"])


def test_bench_riser_metrics_capture_high_minute_nonstarter_underprediction() -> None:
    df = pd.DataFrame(
        [
            {
                "minutes_pred": 12.0,
                "minutes_actual": 28.0,
                "active_pred": 1,
                "lineup_starter_announced": 0,
                "recent_start_pct_10": 0.0,
                "started_proxy_rate_prior_10": 0.0,
                "started_proxy_rate_prior_20": 0.0,
            },
            {
                "minutes_pred": 22.0,
                "minutes_actual": 34.0,
                "active_pred": 1,
                "lineup_starter_announced": 0,
                "recent_start_pct_10": 0.1,
                "started_proxy_rate_prior_10": 0.1,
                "started_proxy_rate_prior_20": 0.1,
            },
            {
                "minutes_pred": 31.0,
                "minutes_actual": 31.0,
                "active_pred": 1,
                "lineup_starter_announced": 1,
                "recent_start_pct_10": 1.0,
                "started_proxy_rate_prior_10": 1.0,
                "started_proxy_rate_prior_20": 1.0,
            },
        ]
    )

    out = _bench_riser_metrics(
        df,
        active_threshold=4.0,
        low_minutes_threshold=8.0,
        bench_hist_start_rate_max=0.5,
        bench_riser_actual_min=20.0,
        bench_riser_pred_min=16.0,
        bench_core_actual_min=32.0,
    )

    assert out["slices"]["bench_riser_candidate"]["n"] == 2
    assert out["slices"]["bench_riser_next_up"]["n"] == 2
    assert out["slices"]["bench_core_next_up"]["n"] == 1
    assert out["failure_rates"]["bench_riser_underprediction_rate"] == 0.5
    assert out["failure_rates"]["bench_core_underprediction_rate"] == 0.0
