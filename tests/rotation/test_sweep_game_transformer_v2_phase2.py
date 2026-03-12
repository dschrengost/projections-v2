from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from scripts.rotation.sweep_game_transformer_v2_phase2 import (
    EvalMetrics,
    _build_eval_cmd,
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
            "lineup_available_0": {"minutes_mae": 3.2, "active_acc": 0.90},
            "lineup_available_1": {"minutes_mae": 3.3, "active_acc": 0.92},
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
        active_acc_lineup0=0.90,
        active_acc_lineup1=0.92,
        active_count_mae=0.7,
        possessions_proxy_mae=4.1,
    )


def test_composite_score_penalizes_only_positive_regressions() -> None:
    baseline = EvalMetrics(3.2, 3.2, 0.01, 0.90, 0.92, 0.65, 4.08)
    better = EvalMetrics(3.1, 3.15, 0.01, 0.91, 0.93, 0.64, 4.05)
    worse = EvalMetrics(3.35, 3.4, 0.09, 0.88, 0.90, 0.8, 4.3)

    better_score = _composite_score(_diff_metrics(better, baseline), promotion_gate_mode="prod_like")
    worse_score = _composite_score(_diff_metrics(worse, baseline), promotion_gate_mode="prod_like")

    assert better_score == 0.0
    assert worse_score > better_score


def test_promotion_gate_rejects_rollbacks_and_large_deltas() -> None:
    deltas_ok = {
        "delta_minutes_mae_lineup0": 0.05,
        "delta_minutes_mae_lineup1": 0.10,
        "delta_minutes_mae_gap_abs": 0.02,
        "delta_active_acc_lineup0": -0.005,
        "delta_active_acc_lineup1": 0.01,
        "delta_active_count_mae": 0.06,
        "delta_possessions_proxy_mae": 0.0,
    }
    assert _meets_promotion_gate(
        deltas=deltas_ok,
        rollback_triggered=False,
        promotion_gate_mode="parity_gap",
        max_delta_minutes_mae_lineup0=0.12,
        max_delta_minutes_mae_lineup1=0.15,
        max_delta_minutes_gap_abs=0.05,
        min_delta_active_acc_lineup1=-0.01,
        max_delta_active_count_mae=0.10,
    )

    assert not _meets_promotion_gate(
        deltas=deltas_ok,
        rollback_triggered=True,
        promotion_gate_mode="parity_gap",
        max_delta_minutes_mae_lineup0=0.12,
        max_delta_minutes_mae_lineup1=0.15,
        max_delta_minutes_gap_abs=0.05,
        min_delta_active_acc_lineup1=-0.01,
        max_delta_active_count_mae=0.10,
    )

    deltas_bad = dict(deltas_ok)
    deltas_bad["delta_minutes_mae_gap_abs"] = 0.08
    assert not _meets_promotion_gate(
        deltas=deltas_bad,
        rollback_triggered=False,
        promotion_gate_mode="parity_gap",
        max_delta_minutes_mae_lineup0=0.12,
        max_delta_minutes_mae_lineup1=0.15,
        max_delta_minutes_gap_abs=0.05,
        min_delta_active_acc_lineup1=-0.01,
        max_delta_active_count_mae=0.10,
    )


def test_prod_like_promotion_gate_ignores_parity_gap_if_lineup1_slice_is_strong() -> None:
    deltas = {
        "delta_minutes_mae_lineup0": 0.18,
        "delta_minutes_mae_lineup1": -0.05,
        "delta_minutes_mae_gap_abs": 0.20,
        "delta_active_acc_lineup0": 0.0,
        "delta_active_acc_lineup1": 0.008,
        "delta_active_count_mae": 0.03,
        "delta_possessions_proxy_mae": 0.0,
    }

    assert _meets_promotion_gate(
        deltas=deltas,
        rollback_triggered=False,
        promotion_gate_mode="prod_like",
        max_delta_minutes_mae_lineup0=0.12,
        max_delta_minutes_mae_lineup1=0.15,
        max_delta_minutes_gap_abs=0.05,
        min_delta_active_acc_lineup1=-0.01,
        max_delta_active_count_mae=0.10,
    )

    bad_acc = dict(deltas)
    bad_acc["delta_active_acc_lineup1"] = -0.03
    assert not _meets_promotion_gate(
        deltas=bad_acc,
        rollback_triggered=False,
        promotion_gate_mode="prod_like",
        max_delta_minutes_mae_lineup0=0.12,
        max_delta_minutes_mae_lineup1=0.15,
        max_delta_minutes_gap_abs=0.05,
        min_delta_active_acc_lineup1=-0.01,
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
        promotion_gate_mode="parity_gap",
        max_mean_delta_minutes_mae_lineup1=0.05,
        max_mean_delta_minutes_gap_abs=0.05,
        min_mean_delta_active_acc_lineup1=-0.005,
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
        promotion_gate_mode="parity_gap",
        max_mean_delta_minutes_mae_lineup1=0.05,
        max_mean_delta_minutes_gap_abs=0.05,
        min_mean_delta_active_acc_lineup1=-0.005,
    )


def test_multi_seed_prod_like_gate_uses_lineup1_active_accuracy() -> None:
    rows = [
        {
            "status": "ok",
            "promotion_gate_pass": True,
            "deltas_vs_baseline": {
                "delta_minutes_mae_lineup0": 0.12,
                "delta_minutes_mae_lineup1": -0.02,
                "delta_minutes_mae_gap_abs": 0.18,
                "delta_active_acc_lineup0": 0.0,
                "delta_active_acc_lineup1": 0.010,
                "delta_active_count_mae": -0.01,
                "delta_possessions_proxy_mae": 0.0,
            },
        },
        {
            "status": "ok",
            "promotion_gate_pass": True,
            "deltas_vs_baseline": {
                "delta_minutes_mae_lineup0": 0.09,
                "delta_minutes_mae_lineup1": 0.01,
                "delta_minutes_mae_gap_abs": 0.16,
                "delta_active_acc_lineup0": 0.0,
                "delta_active_acc_lineup1": 0.004,
                "delta_active_count_mae": 0.0,
                "delta_possessions_proxy_mae": 0.0,
            },
        },
        {
            "status": "ok",
            "promotion_gate_pass": True,
            "deltas_vs_baseline": {
                "delta_minutes_mae_lineup0": 0.11,
                "delta_minutes_mae_lineup1": 0.03,
                "delta_minutes_mae_gap_abs": 0.14,
                "delta_active_acc_lineup0": 0.0,
                "delta_active_acc_lineup1": 0.002,
                "delta_active_count_mae": 0.0,
                "delta_possessions_proxy_mae": 0.0,
            },
        },
    ]

    assert _meets_multi_seed_promotion_gate(
        seed_rows=rows,
        min_required=3,
        require_all_pass=True,
        require_mean_gains=False,
        promotion_gate_mode="prod_like",
        max_mean_delta_minutes_mae_lineup1=0.05,
        max_mean_delta_minutes_gap_abs=0.05,
        min_mean_delta_active_acc_lineup1=-0.005,
    )

    rows_bad = list(rows)
    rows_bad[2] = {
        **rows_bad[2],
        "deltas_vs_baseline": {
            **rows_bad[2]["deltas_vs_baseline"],
            "delta_active_acc_lineup1": -0.03,
        },
    }
    assert not _meets_multi_seed_promotion_gate(
        seed_rows=rows_bad,
        min_required=3,
        require_all_pass=True,
        require_mean_gains=False,
        promotion_gate_mode="prod_like",
        max_mean_delta_minutes_mae_lineup1=0.05,
        max_mean_delta_minutes_gap_abs=0.05,
        min_mean_delta_active_acc_lineup1=-0.005,
    )


def test_build_eval_cmd_uses_eval_device_override() -> None:
    args = Namespace(
        eval_val_days=60,
        batch_size=32,
        num_workers=0,
        device="cuda",
        eval_device="cpu",
    )

    cmd = _build_eval_cmd(
        args=args,
        dataset_dir=Path("/tmp/dataset"),
        run_dir=Path("/tmp/run"),
        eval_json=Path("/tmp/eval.json"),
        params={},
    )

    device_idx = cmd.index("--device")
    assert cmd[device_idx + 1] == "cpu"
