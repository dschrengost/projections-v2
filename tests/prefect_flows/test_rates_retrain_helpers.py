from __future__ import annotations

import pandas as pd

from prefect_flows.rates_retrain import (
    _resolve_training_window,
    _should_run_biweekly,
    assess_eval_guardrails,
)


def test_resolve_training_window_clamps_to_labels_min() -> None:
    window = _resolve_training_window(
        labels_max_date=pd.Timestamp("2026-02-05"),
        labels_min_date=pd.Timestamp("2025-04-01"),
        train_window_days=365,
        cal_window_days=14,
        val_window_days=14,
    )
    assert window["end_date"] == "2026-02-05"
    assert window["train_end_date"] == "2026-01-09"
    assert window["cal_end_date"] == "2026-01-23"
    assert window["start_date"] == "2025-04-01"


def test_should_run_biweekly_anchor_week() -> None:
    assert _should_run_biweekly(run_day=pd.Timestamp("2026-02-03").date(), anchor_day=pd.Timestamp("2026-02-03").date())
    assert not _should_run_biweekly(run_day=pd.Timestamp("2026-02-10").date(), anchor_day=pd.Timestamp("2026-02-03").date())
    assert _should_run_biweekly(run_day=pd.Timestamp("2026-02-17").date(), anchor_day=pd.Timestamp("2026-02-03").date())


def test_assess_eval_guardrails_detects_slice_and_head_failures() -> None:
    eval_summary = {
        "slices": {
            "normal_pre_deadline": {
                "summary": {"avg_mae_delta_retrain_minus_current": 0.0003},
                "per_target": {"fga2_per_min": {"delta_mae_retrain_minus_current": 0.0002}},
            },
            "chaos_deadline": {
                "summary": {"avg_mae_delta_retrain_minus_current": 0.0012},
                "per_target": {"fg2_pct": {"delta_mae_retrain_minus_current": 0.0018}},
            },
        }
    }
    guard = assess_eval_guardrails(
        eval_summary,
        max_avg_mae_delta=0.001,
        max_head_mae_regression=0.0015,
    )
    assert not guard["passed"]
    assert len(guard["failing_slices"]) == 1
    assert guard["worst_head_regression"]["target"] == "fg2_pct"
