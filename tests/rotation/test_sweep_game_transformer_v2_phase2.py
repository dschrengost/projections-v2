from __future__ import annotations

import json

from scripts.rotation.sweep_game_transformer_v2_phase2 import (
    EvalMetrics,
    _composite_score,
    _diff_metrics,
    _load_eval_metrics,
    _meets_multi_seed_promotion_gate,
    _meets_promotion_gate,
    _parse_seed_list,
)


def test_load_eval_metrics_reads_required_slices(tmp_path) -> None:
    payload = {
        "lineup_state_parity": {
            "lineup_available_0": {"minutes_mae": 3.2},
            "lineup_available_1": {"minutes_mae": 3.3},
            "minutes_mae_gap_abs": 0.1,
        },
        "game_volume_calibration": {
            "active_count": {"mae": 0.7},
            "possessions_proxy": {"mae": 4.1},
        },
    }
    p = tmp_path / "eval.json"
    p.write_text(json.dumps(payload), encoding="utf-8")

    m = _load_eval_metrics(p)
    assert m == EvalMetrics(
        minutes_mae_lineup0=3.2,
        minutes_mae_lineup1=3.3,
        minutes_mae_gap_abs=0.1,
        active_count_mae=0.7,
        possessions_proxy_mae=4.1,
    )


def test_composite_score_penalizes_only_positive_regressions() -> None:
    baseline = EvalMetrics(3.2, 3.2, 0.01, 0.65, 4.08)
    better = EvalMetrics(3.1, 3.15, 0.01, 0.64, 4.05)
    worse = EvalMetrics(3.35, 3.4, 0.09, 0.8, 4.3)

    better_score = _composite_score(_diff_metrics(better, baseline))
    worse_score = _composite_score(_diff_metrics(worse, baseline))

    assert better_score == 0.0
    assert worse_score > better_score


def test_promotion_gate_rejects_rollbacks_and_large_deltas() -> None:
    deltas_ok = {
        "delta_minutes_mae_lineup0": 0.05,
        "delta_minutes_mae_lineup1": 0.10,
        "delta_minutes_mae_gap_abs": 0.02,
        "delta_active_count_mae": 0.06,
        "delta_possessions_proxy_mae": 0.0,
    }
    assert _meets_promotion_gate(
        deltas=deltas_ok,
        rollback_triggered=False,
        max_delta_minutes_mae_lineup0=0.12,
        max_delta_minutes_mae_lineup1=0.15,
        max_delta_minutes_gap_abs=0.05,
        max_delta_active_count_mae=0.10,
    )

    assert not _meets_promotion_gate(
        deltas=deltas_ok,
        rollback_triggered=True,
        max_delta_minutes_mae_lineup0=0.12,
        max_delta_minutes_mae_lineup1=0.15,
        max_delta_minutes_gap_abs=0.05,
        max_delta_active_count_mae=0.10,
    )

    deltas_bad = dict(deltas_ok)
    deltas_bad["delta_minutes_mae_gap_abs"] = 0.08
    assert not _meets_promotion_gate(
        deltas=deltas_bad,
        rollback_triggered=False,
        max_delta_minutes_mae_lineup0=0.12,
        max_delta_minutes_mae_lineup1=0.15,
        max_delta_minutes_gap_abs=0.05,
        max_delta_active_count_mae=0.10,
    )


def test_parse_seed_list_includes_base_seed_and_minimum_count() -> None:
    seeds = _parse_seed_list("17,42,17", base_seed=42, min_seeds=3)
    assert seeds == [42, 17, 34]


def test_multi_seed_gate_requires_consistent_pass_and_mean_gains() -> None:
    rows = [
        {
            "status": "ok",
            "promotion_gate_pass": True,
            "deltas_vs_baseline": {
                "delta_minutes_mae_lineup0": -0.03,
                "delta_minutes_mae_lineup1": 0.01,
                "delta_minutes_mae_gap_abs": 0.02,
                "delta_active_count_mae": -0.02,
                "delta_possessions_proxy_mae": 0.0,
            },
        },
        {
            "status": "ok",
            "promotion_gate_pass": True,
            "deltas_vs_baseline": {
                "delta_minutes_mae_lineup0": -0.01,
                "delta_minutes_mae_lineup1": 0.02,
                "delta_minutes_mae_gap_abs": 0.01,
                "delta_active_count_mae": -0.01,
                "delta_possessions_proxy_mae": 0.0,
            },
        },
        {
            "status": "ok",
            "promotion_gate_pass": True,
            "deltas_vs_baseline": {
                "delta_minutes_mae_lineup0": 0.0,
                "delta_minutes_mae_lineup1": 0.03,
                "delta_minutes_mae_gap_abs": 0.02,
                "delta_active_count_mae": 0.0,
                "delta_possessions_proxy_mae": 0.0,
            },
        },
    ]

    assert _meets_multi_seed_promotion_gate(
        seed_rows=rows,
        min_required=3,
        require_all_pass=True,
        require_mean_gains=True,
        max_mean_delta_minutes_mae_lineup1=0.05,
        max_mean_delta_minutes_gap_abs=0.05,
    )


def test_multi_seed_gate_rejects_if_any_seed_fails_when_require_all() -> None:
    rows = [
        {
            "status": "ok",
            "promotion_gate_pass": True,
            "deltas_vs_baseline": {
                "delta_minutes_mae_lineup0": -0.02,
                "delta_minutes_mae_lineup1": 0.01,
                "delta_minutes_mae_gap_abs": 0.02,
                "delta_active_count_mae": -0.01,
                "delta_possessions_proxy_mae": 0.0,
            },
        },
        {
            "status": "ok",
            "promotion_gate_pass": False,
            "deltas_vs_baseline": {
                "delta_minutes_mae_lineup0": 0.01,
                "delta_minutes_mae_lineup1": 0.07,
                "delta_minutes_mae_gap_abs": 0.09,
                "delta_active_count_mae": 0.03,
                "delta_possessions_proxy_mae": 0.0,
            },
        },
        {
            "status": "ok",
            "promotion_gate_pass": True,
            "deltas_vs_baseline": {
                "delta_minutes_mae_lineup0": -0.01,
                "delta_minutes_mae_lineup1": 0.02,
                "delta_minutes_mae_gap_abs": 0.02,
                "delta_active_count_mae": -0.01,
                "delta_possessions_proxy_mae": 0.0,
            },
        },
    ]

    assert not _meets_multi_seed_promotion_gate(
        seed_rows=rows,
        min_required=3,
        require_all_pass=True,
        require_mean_gains=True,
        max_mean_delta_minutes_mae_lineup1=0.05,
        max_mean_delta_minutes_gap_abs=0.05,
    )
