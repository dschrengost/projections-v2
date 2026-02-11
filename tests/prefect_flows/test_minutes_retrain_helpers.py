from __future__ import annotations

import pandas as pd
import pytest

from prefect_flows.minutes_retrain import (
    assess_minutes_eval_guardrails,
    classify_minutes_quality_outcome,
    _resolve_requested_windows,
    _resolve_training_window,
)


def test_resolve_training_window_clamps_to_labels_min() -> None:
    window = _resolve_training_window(
        labels_max_date=pd.Timestamp("2026-02-05"),
        labels_min_date=pd.Timestamp("2025-10-21"),
        train_window_days=120,
        cal_window_days=14,
    )
    assert window["cal_end_date"] == "2026-02-05"
    assert window["cal_start_date"] == "2026-01-23"
    assert window["train_end_date"] == "2026-01-22"
    assert window["train_start_date"] == "2025-10-21"


def test_resolve_requested_windows_requires_all_explicit_dates() -> None:
    with pytest.raises(ValueError, match="Provide either all explicit train/cal dates or none"):
        _resolve_requested_windows(
            labels_max_date=pd.Timestamp("2026-02-05"),
            labels_min_date=pd.Timestamp("2025-10-21"),
            train_start_date="2025-11-01",
            train_end_date=None,
            cal_start_date="2026-02-01",
            cal_end_date="2026-02-05",
            train_window_days=120,
            cal_window_days=14,
        )


def test_resolve_requested_windows_uses_explicit_dates_when_provided() -> None:
    window = _resolve_requested_windows(
        labels_max_date=pd.Timestamp("2026-02-05"),
        labels_min_date=pd.Timestamp("2025-10-21"),
        train_start_date="2025-11-01T00:00:00Z",
        train_end_date="2026-01-31",
        cal_start_date="2026-02-01",
        cal_end_date="2026-02-05",
        train_window_days=120,
        cal_window_days=14,
    )
    assert window == {
        "train_start_date": "2025-11-01",
        "train_end_date": "2026-01-31",
        "cal_start_date": "2026-02-01",
        "cal_end_date": "2026-02-05",
    }


def test_resolve_training_window_raises_when_no_train_data_after_clamp() -> None:
    with pytest.raises(RuntimeError, match="invalid rolling window after clamp"):
        _resolve_training_window(
            labels_max_date=pd.Timestamp("2026-02-05"),
            labels_min_date=pd.Timestamp("2026-02-01"),
            train_window_days=120,
            cal_window_days=14,
        )


def test_assess_minutes_eval_guardrails_detects_regression() -> None:
    summary = {
        "slices": {
            "deadline_chaos": {
                "metrics_current": {
                    "rows": 1500,
                    "brier_play_prob": 0.130,
                    "false_active_rate_p_ge_0_5": 0.100,
                    "mae_p50_conditional": 6.1,
                    "p10_coverage_leq": 0.45,
                    "p90_coverage_leq": 0.98,
                },
                "metrics_retrain": {
                    "rows": 1500,
                    "brier_play_prob": 0.150,
                    "false_active_rate_p_ge_0_5": 0.112,
                    "mae_p50_conditional": 6.3,
                    "p10_coverage_leq": 0.52,
                    "p90_coverage_leq": 0.995,
                },
                "delta_retrain_minus_current": {
                    "brier_play_prob": 0.020,
                    "false_active_rate_p_ge_0_5": 0.012,
                    "mae_p50_conditional": 0.20,
                    "p10_coverage_leq": 0.07,
                    "p90_coverage_leq": 0.015,
                },
            },
            "pre_deadline_stability": {
                "metrics_current": {
                    "rows": 1500,
                    "brier_play_prob": 0.140,
                    "false_active_rate_p_ge_0_5": 0.110,
                    "mae_p50_conditional": 6.0,
                    "p10_coverage_leq": 0.47,
                    "p90_coverage_leq": 0.99,
                },
                "metrics_retrain": {
                    "rows": 1500,
                    "brier_play_prob": 0.145,
                    "false_active_rate_p_ge_0_5": 0.112,
                    "mae_p50_conditional": 6.05,
                    "p10_coverage_leq": 0.50,
                    "p90_coverage_leq": 0.999,
                },
                "delta_retrain_minus_current": {
                    "brier_play_prob": 0.005,
                    "false_active_rate_p_ge_0_5": 0.002,
                    "mae_p50_conditional": 0.05,
                    "p10_coverage_leq": 0.03,
                    "p90_coverage_leq": 0.009,
                },
            },
        }
    }
    gates = assess_minutes_eval_guardrails(summary)

    assert gates["passed"] is False
    metrics = {(f["scope"], f["metric"]) for f in gates["failures"]}
    assert ("slice", "brier_play_prob") in metrics
    assert ("weighted", "brier_play_prob") in metrics


def test_assess_minutes_eval_guardrails_can_use_occupancy_candidate_variant() -> None:
    summary = {
        "slices": {
            "deadline_chaos": {
                "metrics_current": {
                    "rows": 1500,
                    "brier_play_prob": 0.140,
                    "false_active_rate_p_ge_0_5": 0.110,
                    "mae_p50_conditional": 6.0,
                    "p10_coverage_leq": 0.46,
                    "p90_coverage_leq": 0.99,
                },
                "metrics_retrain": {
                    "rows": 1500,
                    "brier_play_prob": 0.160,
                    "false_active_rate_p_ge_0_5": 0.124,
                    "mae_p50_conditional": 6.3,
                    "p10_coverage_leq": 0.50,
                    "p90_coverage_leq": 0.999,
                },
                "delta_retrain_minus_current": {
                    "brier_play_prob": 0.020,
                    "false_active_rate_p_ge_0_5": 0.014,
                    "mae_p50_conditional": 0.30,
                    "p10_coverage_leq": 0.04,
                    "p90_coverage_leq": 0.009,
                },
                "metrics_retrain_occupancy_v0": {
                    "rows": 1500,
                    "brier_play_prob": 0.138,
                    "false_active_rate_p_ge_0_5": 0.106,
                    "mae_p50_conditional": 6.2,
                    "p10_coverage_leq": 0.47,
                    "p90_coverage_leq": 0.991,
                },
                "delta_retrain_occupancy_v0_minus_current": {
                    "brier_play_prob": -0.002,
                    "false_active_rate_p_ge_0_5": -0.004,
                    "mae_p50_conditional": 0.20,
                    "p10_coverage_leq": 0.01,
                    "p90_coverage_leq": 0.001,
                },
            }
        }
    }

    base_gates = assess_minutes_eval_guardrails(
        summary,
        eval_candidate_variant="retrain",
        adaptive_thresholds_enabled=False,
        drift_override_enabled=False,
    )
    assert base_gates["passed"] is False

    occ_gates = assess_minutes_eval_guardrails(
        summary,
        eval_candidate_variant="retrain_occupancy_v0",
        adaptive_thresholds_enabled=False,
        drift_override_enabled=False,
    )
    assert occ_gates["passed"] is True
    assert occ_gates["candidate_metric_key"] == "metrics_retrain_occupancy_v0"


def test_assess_minutes_eval_guardrails_mae_soft_threshold_records_advisory() -> None:
    summary = {
        "slices": {
            "deadline_chaos": {
                "metrics_current": {
                    "rows": 1500,
                    "brier_play_prob": 0.140,
                    "false_active_rate_p_ge_0_5": 0.110,
                    "mae_p50_conditional": 6.0,
                    "p10_coverage_leq": 0.46,
                    "p90_coverage_leq": 0.99,
                },
                "metrics_retrain": {
                    "rows": 1500,
                    "brier_play_prob": 0.141,
                    "false_active_rate_p_ge_0_5": 0.109,
                    "mae_p50_conditional": 6.25,
                    "p10_coverage_leq": 0.46,
                    "p90_coverage_leq": 0.99,
                },
                "delta_retrain_minus_current": {
                    "brier_play_prob": 0.001,
                    "false_active_rate_p_ge_0_5": -0.001,
                    "mae_p50_conditional": 0.25,
                    "p10_coverage_leq": 0.0,
                    "p90_coverage_leq": 0.0,
                },
            }
        }
    }

    gates = assess_minutes_eval_guardrails(
        summary,
        adaptive_thresholds_enabled=False,
        drift_override_enabled=False,
        max_weighted_mae_p50_cond_delta=0.1,
        max_slice_mae_p50_cond_delta=0.2,
        max_weighted_mae_p50_cond_delta_soft=0.3,
        max_slice_mae_p50_cond_delta_soft=0.3,
    )

    assert gates["passed"] is True
    assert gates["soft_failures"]
    assert gates["thresholds"]["max_weighted_mae_p50_cond_delta_soft"] == pytest.approx(0.3)
    assert gates["thresholds"]["max_slice_mae_p50_cond_delta_soft"] == pytest.approx(0.3)


def test_classify_minutes_quality_outcome_prefers_data_issue() -> None:
    classification = classify_minutes_quality_outcome(
        guardrails={"passed": False, "failures": [], "thresholds": {}, "slice_diagnostics": {}},
        data_quality={"violations": ["days_since_last: 10 values below 0"]},
    )
    assert classification["classification"] == "data_issue"


def test_classify_minutes_quality_outcome_marks_hard_slice() -> None:
    guardrails = {
        "passed": False,
        "failures": [{"scope": "slice", "metric": "p10_coverage_leq", "delta": 0.05, "threshold": 0.04}],
        "thresholds": {
            "max_slice_brier_delta": 0.01,
            "max_slice_false_active_delta": 0.015,
            "max_slice_mae_p50_cond_delta": 0.2,
        },
        "slice_diagnostics": {
            "deadline_chaos": {
                "metrics_current": {
                    "brier_play_prob": 0.19,
                    "false_active_rate_p_ge_0_5": 0.15,
                    "mae_p50_conditional": 7.0,
                },
                "delta_retrain_minus_current": {
                    "brier_play_prob": 0.008,
                    "false_active_rate_p_ge_0_5": 0.01,
                    "mae_p50_conditional": 0.1,
                },
            }
        },
    }
    classification = classify_minutes_quality_outcome(
        guardrails=guardrails,
        data_quality={"violations": []},
    )
    assert classification["classification"] == "hard_slice"


def test_classify_minutes_quality_outcome_marks_model_regression() -> None:
    guardrails = {
        "passed": False,
        "failures": [{"scope": "weighted", "metric": "brier_play_prob", "delta": 0.02, "threshold": 0.005}],
        "thresholds": {
            "max_slice_brier_delta": 0.01,
            "max_slice_false_active_delta": 0.015,
            "max_slice_mae_p50_cond_delta": 0.2,
        },
        "slice_diagnostics": {
            "deadline_chaos": {
                "metrics_current": {
                    "brier_play_prob": 0.13,
                    "false_active_rate_p_ge_0_5": 0.10,
                    "mae_p50_conditional": 6.1,
                },
                "delta_retrain_minus_current": {
                    "brier_play_prob": 0.02,
                    "false_active_rate_p_ge_0_5": 0.02,
                    "mae_p50_conditional": 0.3,
                },
            }
        },
    }
    classification = classify_minutes_quality_outcome(
        guardrails=guardrails,
        data_quality={"violations": []},
    )
    assert classification["classification"] == "model_regression"


def test_assess_minutes_eval_guardrails_adapts_weighted_thresholds() -> None:
    summary = {
        "slices": {
            "deadline_chaos": {
                "metrics_current": {
                    "rows": 1500,
                    "brier_play_prob": 0.130,
                    "false_active_rate_p_ge_0_5": 0.100,
                    "mae_p50_conditional": 6.1,
                    "p10_coverage_leq": 0.45,
                    "p90_coverage_leq": 0.98,
                },
                "metrics_retrain": {
                    "rows": 1500,
                    "brier_play_prob": 0.137,
                    "false_active_rate_p_ge_0_5": 0.095,
                    "mae_p50_conditional": 6.0,
                    "p10_coverage_leq": 0.46,
                    "p90_coverage_leq": 0.98,
                },
                "delta_retrain_minus_current": {
                    "brier_play_prob": 0.007,
                    "false_active_rate_p_ge_0_5": -0.005,
                    "mae_p50_conditional": -0.1,
                    "p10_coverage_leq": 0.01,
                    "p90_coverage_leq": 0.0,
                },
            }
        }
    }
    history_runs = []
    for delta in (0.010, 0.011, 0.012, 0.013, 0.014, 0.015):
        history_runs.append(
            {
                "data_quality": {"passed": True},
                "guardrails": {
                    "weighted_metrics": {"brier_play_prob": {"delta": delta}},
                    "slice_diagnostics": {},
                },
            }
        )

    gates = assess_minutes_eval_guardrails(
        summary,
        history_runs=history_runs,
        adaptive_thresholds_enabled=True,
        adaptive_min_history_runs=3,
        adaptive_iqr_multiplier=1.5,
        adaptive_threshold_cap_multiplier=3.0,
        drift_override_enabled=False,
    )

    assert gates["passed"] is True
    assert gates["thresholds"]["max_weighted_brier_delta"] > 0.005
    assert gates["adaptive"]["details"]["max_weighted_brier_delta"]["used_history"] is True


def test_assess_minutes_eval_guardrails_drift_override_relaxes_thresholds() -> None:
    summary = {
        "slices": {
            "deadline_chaos": {
                "metrics_current": {
                    "rows": 1500,
                    "brier_play_prob": 0.200,
                    "false_active_rate_p_ge_0_5": 0.180,
                    "mae_p50_conditional": 8.0,
                    "p10_coverage_leq": 0.45,
                    "p90_coverage_leq": 0.98,
                },
                "metrics_retrain": {
                    "rows": 1500,
                    "brier_play_prob": 0.206,
                    "false_active_rate_p_ge_0_5": 0.182,
                    "mae_p50_conditional": 8.05,
                    "p10_coverage_leq": 0.45,
                    "p90_coverage_leq": 0.98,
                },
                "delta_retrain_minus_current": {
                    "brier_play_prob": 0.006,
                    "false_active_rate_p_ge_0_5": 0.002,
                    "mae_p50_conditional": 0.05,
                    "p10_coverage_leq": 0.0,
                    "p90_coverage_leq": 0.0,
                },
            }
        }
    }
    history_runs = []
    for brier, false_active, mae in (
        (0.10, 0.08, 5.0),
        (0.11, 0.09, 5.2),
        (0.09, 0.07, 4.8),
    ):
        history_runs.append(
            {
                "data_quality": {"passed": True},
                "guardrails": {
                    "weighted_metrics": {
                        "brier_play_prob": {"current": brier, "delta": 0.0},
                        "false_active_rate_p_ge_0_5": {"current": false_active, "delta": 0.0},
                        "mae_p50_conditional": {"current": mae, "delta": 0.0},
                    },
                    "slice_diagnostics": {},
                },
            }
        )

    gates = assess_minutes_eval_guardrails(
        summary,
        history_runs=history_runs,
        adaptive_thresholds_enabled=False,
        adaptive_min_history_runs=3,
        drift_override_enabled=True,
        drift_override_relax_multiplier=1.5,
        drift_override_min_trigger_metrics=1,
        drift_override_iqr_multiplier=1.0,
    )

    assert gates["passed"] is True
    assert gates["drift_override"]["applied"] is True
    assert gates["thresholds"]["max_weighted_brier_delta"] == pytest.approx(0.0075)
