from __future__ import annotations

import json

import pytest

from scripts.rotation.sweep_game_transformer_v2_minutes_core import (
    EvalMetrics,
    _composite_score,
    _diff_metrics,
    _is_finite_eval,
    _load_eval_metrics,
)


def test_load_eval_metrics_reads_sparse_fields(tmp_path) -> None:
    payload = {
        "lineup_state_parity": {
            "lineup_available_0": {"minutes_mae": 3.2},
            "lineup_available_1": {"minutes_mae": 3.5, "active_acc": 0.91},
        },
        "game_volume_calibration": {
            "active_count": {"mae": 0.7},
        },
        "sparse_rotation_diagnostics": {
            "failure_rates": {"sparse_next_up_underprediction_rate": 0.4},
            "slices": {"starter_sparse_prior": {"pred_minutes_mean": 12.0}},
        },
    }
    p = tmp_path / "eval.json"
    p.write_text(json.dumps(payload), encoding="utf-8")

    metrics = _load_eval_metrics(p)
    assert metrics == EvalMetrics(
        minutes_mae_lineup0=3.2,
        minutes_mae_lineup1=3.5,
        active_acc_lineup1=0.91,
        active_count_mae=0.7,
        sparse_next_up_underpred_rate=0.4,
        starter_sparse_pred_minutes_mean=12.0,
    )


def test_diff_metrics_tracks_regressions_and_improvements() -> None:
    baseline = EvalMetrics(3.2, 3.3, 0.91, 0.7, 0.4, 12.0)
    candidate = EvalMetrics(3.1, 3.4, 0.89, 0.8, 0.5, 10.0)
    diff = _diff_metrics(candidate, baseline)
    assert diff == {
        "delta_minutes_mae_lineup0": pytest.approx(-0.1),
        "delta_minutes_mae_lineup1": pytest.approx(0.1),
        "delta_active_acc_lineup1": pytest.approx(-0.02),
        "delta_active_count_mae": pytest.approx(0.1),
        "delta_sparse_next_up_underpred_rate": pytest.approx(0.1),
        "delta_starter_sparse_pred_minutes_mean": pytest.approx(-2.0),
    }


def test_composite_score_penalizes_sparse_regressions_heavily() -> None:
    better_14 = {
        "delta_minutes_mae_lineup0": -0.1,
        "delta_minutes_mae_lineup1": 0.0,
        "delta_active_acc_lineup1": 0.0,
        "delta_active_count_mae": 0.0,
        "delta_sparse_next_up_underpred_rate": 0.0,
        "delta_starter_sparse_pred_minutes_mean": 1.0,
    }
    better_60 = dict(better_14)
    worse_sparse_14 = dict(better_14)
    worse_sparse_14["delta_sparse_next_up_underpred_rate"] = 0.1
    worse_sparse_60 = dict(better_60)
    worse_sparse_60["delta_sparse_next_up_underpred_rate"] = 0.1

    better_score = _composite_score(deltas_14d=better_14, deltas_60d=better_60)
    worse_score = _composite_score(deltas_14d=worse_sparse_14, deltas_60d=worse_sparse_60)

    assert better_score == pytest.approx(0.0)
    assert worse_score > better_score


def test_is_finite_eval_allows_missing_lineup1_for_14d() -> None:
    metrics = EvalMetrics(
        minutes_mae_lineup0=4.8,
        minutes_mae_lineup1=float("nan"),
        active_acc_lineup1=float("nan"),
        active_count_mae=1.4,
        sparse_next_up_underpred_rate=0.63,
        starter_sparse_pred_minutes_mean=8.1,
    )

    assert _is_finite_eval(metrics, require_lineup1=False) is True
    assert _is_finite_eval(metrics, require_lineup1=True) is False
