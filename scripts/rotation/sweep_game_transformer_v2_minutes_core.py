#!/usr/bin/env python3
"""Focused core-minutes sweep for GameTransformerV2.

This sweep is intentionally narrow:
- minutes/active-set only (no phase2 flow, no downstream heads)
- early stopping and checkpoint selection use val_minutes_mae
- candidate ranking includes sparse next-up diagnostics alongside MAE
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from projections import paths

PYTHON_EXE = sys.executable


@dataclass(frozen=True)
class Trial:
    name: str
    params: dict[str, Any]


@dataclass(frozen=True)
class EvalMetrics:
    minutes_mae_lineup0: float
    minutes_mae_lineup1: float
    active_acc_lineup1: float
    active_count_mae: float
    sparse_next_up_underpred_rate: float
    starter_sparse_pred_minutes_mean: float


def _utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _slugify(text: str) -> str:
    out: list[str] = []
    for ch in text.lower():
        if ch.isalnum() or ch in {"-", "_"}:
            out.append(ch)
        elif ch in {" ", "."}:
            out.append("_")
    return "".join(out).strip("_") or "trial"


def _default_trials() -> list[Trial]:
    rows = [
        ("baseline_lr3e4", {}),
        ("baseline_lr1e4", {"lr": 1e-4}),
        ("lineup_w2", {"lineup_available_sample_weight": 2.0}),
        ("lineup_w3", {"lineup_available_sample_weight": 3.0}),
        ("count_member_065", {"w_count": 0.65, "w_member": 0.65}),
        (
            "count_member_065_lineup_w2",
            {"w_count": 0.65, "w_member": 0.65, "lineup_available_sample_weight": 2.0},
        ),
    ]
    return [Trial(name=_slugify(name), params=params) for name, params in rows]


def _load_eval_metrics(path: Path) -> EvalMetrics:
    payload = json.loads(path.read_text(encoding="utf-8"))
    parity = payload.get("lineup_state_parity", {}) or {}
    volume = payload.get("game_volume_calibration", {}) or {}
    active_count = (volume.get("active_count", {}) or {})
    sparse = payload.get("sparse_rotation_diagnostics", {}) or {}
    failure = sparse.get("failure_rates", {}) or {}
    slices = sparse.get("slices", {}) or {}
    starter_sparse = (slices.get("starter_sparse_prior", {}) or {})
    return EvalMetrics(
        minutes_mae_lineup0=_float_or_nan((parity.get("lineup_available_0", {}) or {}).get("minutes_mae")),
        minutes_mae_lineup1=_float_or_nan((parity.get("lineup_available_1", {}) or {}).get("minutes_mae")),
        active_acc_lineup1=_float_or_nan((parity.get("lineup_available_1", {}) or {}).get("active_acc")),
        active_count_mae=_float_or_nan(active_count.get("mae")),
        sparse_next_up_underpred_rate=_float_or_nan(failure.get("sparse_next_up_underprediction_rate")),
        starter_sparse_pred_minutes_mean=_float_or_nan(starter_sparse.get("pred_minutes_mean")),
    )


def _diff_metrics(candidate: EvalMetrics, baseline: EvalMetrics) -> dict[str, float]:
    return {
        "delta_minutes_mae_lineup0": float(candidate.minutes_mae_lineup0 - baseline.minutes_mae_lineup0),
        "delta_minutes_mae_lineup1": float(candidate.minutes_mae_lineup1 - baseline.minutes_mae_lineup1),
        "delta_active_acc_lineup1": float(candidate.active_acc_lineup1 - baseline.active_acc_lineup1),
        "delta_active_count_mae": float(candidate.active_count_mae - baseline.active_count_mae),
        "delta_sparse_next_up_underpred_rate": float(
            candidate.sparse_next_up_underpred_rate - baseline.sparse_next_up_underpred_rate
        ),
        "delta_starter_sparse_pred_minutes_mean": float(
            candidate.starter_sparse_pred_minutes_mean - baseline.starter_sparse_pred_minutes_mean
        ),
    }


def _positive_regression(x: float) -> float:
    return max(0.0, float(x))


def _composite_score(
    *,
    deltas_14d: dict[str, float],
    deltas_60d: dict[str, float],
) -> float:
    return float(
        1.0 * _positive_regression(deltas_14d["delta_minutes_mae_lineup0"])
        + 1.5 * _positive_regression(deltas_60d["delta_minutes_mae_lineup0"])
        + 1.25 * _positive_regression(deltas_60d["delta_minutes_mae_lineup1"])
        + 1.0 * _positive_regression(deltas_60d["delta_active_count_mae"])
        + 6.0 * _positive_regression(deltas_14d["delta_sparse_next_up_underpred_rate"])
        + 6.0 * _positive_regression(deltas_60d["delta_sparse_next_up_underpred_rate"])
        + 2.0 * _positive_regression(-deltas_60d["delta_active_acc_lineup1"])
        + 0.25 * _positive_regression(-deltas_14d["delta_starter_sparse_pred_minutes_mean"])
        + 0.50 * _positive_regression(-deltas_60d["delta_starter_sparse_pred_minutes_mean"])
    )


def _is_finite_eval(metrics: EvalMetrics, *, require_lineup1: bool) -> bool:
    required = [
        metrics.minutes_mae_lineup0,
        metrics.active_count_mae,
        metrics.sparse_next_up_underpred_rate,
        metrics.starter_sparse_pred_minutes_mean,
    ]
    if require_lineup1:
        required.extend([metrics.minutes_mae_lineup1, metrics.active_acc_lineup1])
    return all(math.isfinite(v) for v in required)


def _resolve_dataset_dir(value: str | None) -> Path:
    root = paths.get_data_root() / "training" / "datasets"
    if value:
        p = Path(value).expanduser()
        if p.exists():
            return p.resolve()
        p2 = root / value
        if p2.exists():
            return p2.resolve()
        raise FileNotFoundError(f"Dataset directory not found: {value}")
    candidates = sorted(root.glob("joint_rotation_rates_v1*"))
    if not candidates:
        raise FileNotFoundError(f"No joint_rotation_rates_v1* datasets found under {root}")
    return candidates[-1].resolve()


def _to_cli_args(params: dict[str, Any]) -> list[str]:
    args: list[str] = []
    for key, value in params.items():
        flag = f"--{str(key).replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                args.append(flag)
            continue
        args.extend([flag, str(value)])
    return args


def _run(cmd: list[str], *, log_path: Path) -> subprocess.CompletedProcess[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    proc = subprocess.run(
        cmd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.write_text(proc.stdout or "", encoding="utf-8")
    return proc


def _print_cmd(prefix: str, cmd: list[str]) -> None:
    print(f"{prefix}: {' '.join(shlex.quote(c) for c in cmd)}", flush=True)


def _build_train_cmd(
    *,
    args: argparse.Namespace,
    dataset_dir: Path,
    run_dir: Path,
    seed: int,
    params: dict[str, Any],
) -> list[str]:
    cmd = [
        PYTHON_EXE,
        "scripts/rotation/train_game_transformer_v2.py",
        "--dataset-dir",
        str(dataset_dir),
        "--out-dir",
        str(run_dir),
        "--epochs",
        str(int(args.epochs)),
        "--val-days",
        str(int(args.train_val_days)),
        "--batch-size",
        str(int(args.batch_size)),
        "--num-workers",
        str(int(args.num_workers)),
        "--device",
        str(args.device),
        "--seed",
        str(int(seed)),
        "--lr",
        str(float(args.lr)),
        "--weight-decay",
        str(float(args.weight_decay)),
        "--early-stop-patience",
        str(int(args.early_stop_patience)),
        "--early-stop-min-epochs",
        str(int(args.early_stop_min_epochs)),
        "--early-stop-metric",
        "val_minutes_mae",
        "--best-checkpoint-metric",
        "val_minutes_mae",
    ]
    cmd.extend(_to_cli_args(params))
    return cmd


def _build_eval_cmd(
    *,
    args: argparse.Namespace,
    dataset_dir: Path,
    run_dir: Path,
    val_days: int,
    out_json: Path,
) -> list[str]:
    return [
        PYTHON_EXE,
        "scripts/rotation/eval_game_transformer_v2.py",
        "--run-dir",
        str(run_dir),
        "--dataset-dir",
        str(dataset_dir),
        "--val-days",
        str(int(val_days)),
        "--batch-size",
        str(int(args.batch_size)),
        "--num-workers",
        str(int(args.num_workers)),
        "--device",
        str(args.eval_device or args.device),
        "--out-json",
        str(out_json),
    ]


def _run_trial(
    *,
    args: argparse.Namespace,
    dataset_dir: Path,
    baseline_14d: EvalMetrics,
    baseline_60d: EvalMetrics,
    trial: Trial,
) -> dict[str, Any]:
    run_root = Path(args.sweep_root).expanduser().resolve() / "trials" / trial.name
    run_root.mkdir(parents=True, exist_ok=True)
    run_dir = run_root / "run"
    train_cmd = _build_train_cmd(
        args=args,
        dataset_dir=dataset_dir,
        run_dir=run_dir,
        seed=int(args.seed),
        params=trial.params,
    )
    eval14_json = run_root / "eval_slices_14d.json"
    eval60_json = run_root / "eval_slices_60d.json"
    eval14_cmd = _build_eval_cmd(
        args=args,
        dataset_dir=dataset_dir,
        run_dir=run_dir,
        val_days=14,
        out_json=eval14_json,
    )
    eval60_cmd = _build_eval_cmd(
        args=args,
        dataset_dir=dataset_dir,
        run_dir=run_dir,
        val_days=60,
        out_json=eval60_json,
    )

    result: dict[str, Any] = {
        "trial_name": trial.name,
        "params": dict(trial.params),
        "run_dir": str(run_dir),
        "status": "planned",
    }

    _print_cmd(f"[minutes_core_sweep] train {trial.name}", train_cmd)
    if bool(args.dry_run):
        result["status"] = "dry_run"
        return result

    train_proc = _run(train_cmd, log_path=run_root / "train.log")
    result["train_rc"] = int(train_proc.returncode)
    if train_proc.returncode != 0:
        result["status"] = "train_failed"
        return result

    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        result["summary"] = json.loads(summary_path.read_text(encoding="utf-8"))

    _print_cmd(f"[minutes_core_sweep] eval14 {trial.name}", eval14_cmd)
    eval14_proc = _run(eval14_cmd, log_path=run_root / "eval_14d.log")
    result["eval14_rc"] = int(eval14_proc.returncode)
    if eval14_proc.returncode != 0 or not eval14_json.exists():
        result["status"] = "eval14_failed"
        return result

    _print_cmd(f"[minutes_core_sweep] eval60 {trial.name}", eval60_cmd)
    eval60_proc = _run(eval60_cmd, log_path=run_root / "eval_60d.log")
    result["eval60_rc"] = int(eval60_proc.returncode)
    if eval60_proc.returncode != 0 or not eval60_json.exists():
        result["status"] = "eval60_failed"
        return result

    metrics_14d = _load_eval_metrics(eval14_json)
    metrics_60d = _load_eval_metrics(eval60_json)
    if not _is_finite_eval(metrics_14d, require_lineup1=False) or not _is_finite_eval(
        metrics_60d, require_lineup1=True
    ):
        result["status"] = "eval_nonfinite"
        result["metrics_14d"] = asdict(metrics_14d)
        result["metrics_60d"] = asdict(metrics_60d)
        return result

    deltas_14d = _diff_metrics(metrics_14d, baseline_14d)
    deltas_60d = _diff_metrics(metrics_60d, baseline_60d)
    composite = _composite_score(deltas_14d=deltas_14d, deltas_60d=deltas_60d)

    result.update(
        {
            "status": "ok",
            "metrics_14d": asdict(metrics_14d),
            "metrics_60d": asdict(metrics_60d),
            "deltas_14d_vs_baseline": deltas_14d,
            "deltas_60d_vs_baseline": deltas_60d,
            "composite_score": float(composite),
        }
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=str, default=None)
    parser.add_argument("--sweep-root", type=str, default=None)
    parser.add_argument("--baseline-eval-json-14d", type=str, required=True)
    parser.add_argument("--baseline-eval-json-60d", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=36)
    parser.add_argument("--train-val-days", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--eval-device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--early-stop-min-epochs", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = _resolve_dataset_dir(args.dataset_dir)
    baseline_14d_path = Path(args.baseline_eval_json_14d).expanduser().resolve()
    baseline_60d_path = Path(args.baseline_eval_json_60d).expanduser().resolve()
    if not baseline_14d_path.exists():
        raise FileNotFoundError(f"baseline 14d eval json not found: {baseline_14d_path}")
    if not baseline_60d_path.exists():
        raise FileNotFoundError(f"baseline 60d eval json not found: {baseline_60d_path}")
    baseline_14d = _load_eval_metrics(baseline_14d_path)
    baseline_60d = _load_eval_metrics(baseline_60d_path)

    root_default = paths.get_data_root() / "training" / "runs" / f"gtv2_minutes_core_sweep_{_utc_now_compact()}"
    sweep_root = Path(args.sweep_root).expanduser().resolve() if args.sweep_root else root_default
    sweep_root.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for trial in _default_trials():
        results.append(
            _run_trial(
                args=argparse.Namespace(**{**vars(args), "sweep_root": str(sweep_root)}),
                dataset_dir=dataset_dir,
                baseline_14d=baseline_14d,
                baseline_60d=baseline_60d,
                trial=trial,
            )
        )

    ok_results = [row for row in results if str(row.get("status")) == "ok"]
    ok_results_sorted = sorted(ok_results, key=lambda row: float(row.get("composite_score", float("inf"))))
    best_result = ok_results_sorted[0] if ok_results_sorted else None

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(dataset_dir),
        "baseline_eval_json_14d": str(baseline_14d_path),
        "baseline_eval_json_60d": str(baseline_60d_path),
        "baseline_metrics_14d": asdict(baseline_14d),
        "baseline_metrics_60d": asdict(baseline_60d),
        "sweep_root": str(sweep_root),
        "results": results,
        "best_result": best_result,
    }
    (sweep_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    flat_rows: list[dict[str, Any]] = []
    for row in results:
        flat = {
            "trial_name": row.get("trial_name"),
            "status": row.get("status"),
            "composite_score": row.get("composite_score"),
            "run_dir": row.get("run_dir"),
        }
        if "metrics_14d" in row:
            flat.update({f"m14_{k}": v for k, v in row["metrics_14d"].items()})
        if "metrics_60d" in row:
            flat.update({f"m60_{k}": v for k, v in row["metrics_60d"].items()})
        if "deltas_14d_vs_baseline" in row:
            flat.update({f"d14_{k}": v for k, v in row["deltas_14d_vs_baseline"].items()})
        if "deltas_60d_vs_baseline" in row:
            flat.update({f"d60_{k}": v for k, v in row["deltas_60d_vs_baseline"].items()})
        flat_rows.append(flat)
    if flat_rows:
        df = pd.DataFrame(flat_rows)
        df.to_csv(sweep_root / "results.csv", index=False)
        (sweep_root / "results.txt").write_text(df.to_string(index=False) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
