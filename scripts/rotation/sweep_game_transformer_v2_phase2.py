#!/usr/bin/env python3
"""Targeted Phase 2 sweep for GameTransformerV2 anchor-parity recovery."""

from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from projections import paths


@dataclass(frozen=True)
class Trial:
    name: str
    params: dict[str, Any]


@dataclass(frozen=True)
class EvalMetrics:
    minutes_mae_lineup0: float
    minutes_mae_lineup1: float
    minutes_mae_gap_abs: float
    active_count_mae: float
    possessions_proxy_mae: float


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
        (
            "anchor90_warm8_flow025_count060",
            {
                "phase2_anchor_end_weight": 0.90,
                "phase2_flow_warmup_epochs": 8,
                "w_count": 0.60,
                "w_member": 0.60,
                "w_minutes_nll": 0.75,
                "w_flow_nll": 0.25,
            },
        ),
        (
            "anchor90_warm10_flow020_count065",
            {
                "phase2_anchor_end_weight": 0.90,
                "phase2_flow_warmup_epochs": 10,
                "w_count": 0.65,
                "w_member": 0.65,
                "w_minutes_nll": 0.75,
                "w_flow_nll": 0.20,
            },
        ),
        (
            "anchor85_warm8_flow025_count060",
            {
                "phase2_anchor_end_weight": 0.85,
                "phase2_flow_warmup_epochs": 8,
                "w_count": 0.60,
                "w_member": 0.60,
                "w_minutes_nll": 1.00,
                "w_flow_nll": 0.25,
            },
        ),
        (
            "anchor85_warm6_flow030_count055",
            {
                "phase2_anchor_end_weight": 0.85,
                "phase2_flow_warmup_epochs": 6,
                "w_count": 0.55,
                "w_member": 0.55,
                "w_minutes_nll": 1.00,
                "w_flow_nll": 0.30,
            },
        ),
        (
            "anchor80_warm8_flow020_count050",
            {
                "phase2_anchor_end_weight": 0.80,
                "phase2_flow_warmup_epochs": 8,
                "w_count": 0.50,
                "w_member": 0.50,
                "w_minutes_nll": 1.00,
                "w_flow_nll": 0.20,
            },
        ),
        (
            "anchor80_warm6_flow035_count050",
            {
                "phase2_anchor_end_weight": 0.80,
                "phase2_flow_warmup_epochs": 6,
                "w_count": 0.50,
                "w_member": 0.50,
                "w_minutes_nll": 1.00,
                "w_flow_nll": 0.35,
            },
        ),
    ]
    return [Trial(name=_slugify(name), params=params) for name, params in rows]


def _read_trials_file(path: Path) -> list[Trial]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("trials JSON must be a list")

    out: list[Trial] = []
    for idx, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"trial[{idx}] must be an object")
        params = item.get("params", {k: v for k, v in item.items() if k != "name"})
        if not isinstance(params, dict):
            raise ValueError(f"trial[{idx}] params must be object")
        name = str(item.get("name", f"trial_{idx:03d}"))
        out.append(Trial(name=_slugify(name), params={str(k): v for k, v in params.items()}))
    return out


def _to_cli_args(params: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for key, value in params.items():
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                out.append(flag)
            continue
        out.extend([flag, str(value)])
    return out


def _resolve_dataset_dir(value: str | None) -> Path:
    root = paths.get_data_root() / "training" / "datasets"
    if value:
        p = Path(value).expanduser()
        if p.exists():
            return p.resolve()
        p2 = root / value
        if p2.exists():
            return p2.resolve()
        raise FileNotFoundError(f"dataset directory not found: {value}")
    candidates = sorted(root.glob("joint_rotation_rates_v1*"))
    if not candidates:
        raise FileNotFoundError(f"no joint_rotation_rates_v1* datasets found under {root}")
    return candidates[-1].resolve()


def _load_eval_metrics(path: Path) -> EvalMetrics:
    payload = json.loads(path.read_text(encoding="utf-8"))
    parity = payload.get("lineup_state_parity", {}) or {}
    volume = payload.get("game_volume_calibration", {}) or {}
    active_count = (volume.get("active_count", {}) or {})
    poss = (volume.get("possessions_proxy", {}) or {})
    return EvalMetrics(
        minutes_mae_lineup0=_float_or_nan((parity.get("lineup_available_0", {}) or {}).get("minutes_mae")),
        minutes_mae_lineup1=_float_or_nan((parity.get("lineup_available_1", {}) or {}).get("minutes_mae")),
        minutes_mae_gap_abs=_float_or_nan(parity.get("minutes_mae_gap_abs")),
        active_count_mae=_float_or_nan(active_count.get("mae")),
        possessions_proxy_mae=_float_or_nan(poss.get("mae")),
    )


def _diff_metrics(candidate: EvalMetrics, baseline: EvalMetrics) -> dict[str, float]:
    return {
        "delta_minutes_mae_lineup0": float(candidate.minutes_mae_lineup0 - baseline.minutes_mae_lineup0),
        "delta_minutes_mae_lineup1": float(candidate.minutes_mae_lineup1 - baseline.minutes_mae_lineup1),
        "delta_minutes_mae_gap_abs": float(candidate.minutes_mae_gap_abs - baseline.minutes_mae_gap_abs),
        "delta_active_count_mae": float(candidate.active_count_mae - baseline.active_count_mae),
        "delta_possessions_proxy_mae": float(candidate.possessions_proxy_mae - baseline.possessions_proxy_mae),
    }


def _composite_score(deltas: dict[str, float]) -> float:
    d0 = max(0.0, float(deltas["delta_minutes_mae_lineup0"]))
    d1 = max(0.0, float(deltas["delta_minutes_mae_lineup1"]))
    dgap = max(0.0, float(deltas["delta_minutes_mae_gap_abs"]))
    dactive = max(0.0, float(deltas["delta_active_count_mae"]))
    dposs = max(0.0, float(deltas["delta_possessions_proxy_mae"]))
    return float(1.0 * d0 + 1.0 * d1 + 2.0 * dgap + 1.5 * dactive + 0.25 * dposs)


def _is_finite_eval(m: EvalMetrics) -> bool:
    return all(
        math.isfinite(v)
        for v in (
            m.minutes_mae_lineup0,
            m.minutes_mae_lineup1,
            m.minutes_mae_gap_abs,
            m.active_count_mae,
            m.possessions_proxy_mae,
        )
    )


def _meets_promotion_gate(
    *,
    deltas: dict[str, float],
    rollback_triggered: bool,
    max_delta_minutes_mae_lineup0: float,
    max_delta_minutes_mae_lineup1: float,
    max_delta_minutes_gap_abs: float,
    max_delta_active_count_mae: float,
) -> bool:
    if rollback_triggered:
        return False
    return bool(
        float(deltas["delta_minutes_mae_lineup0"]) <= float(max_delta_minutes_mae_lineup0)
        and float(deltas["delta_minutes_mae_lineup1"]) <= float(max_delta_minutes_mae_lineup1)
        and float(deltas["delta_minutes_mae_gap_abs"]) <= float(max_delta_minutes_gap_abs)
        and float(deltas["delta_active_count_mae"]) <= float(max_delta_active_count_mae)
    )


def _run(cmd: list[str], *, log_path: Path) -> subprocess.CompletedProcess[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    output = proc.stdout or ""
    log_path.write_text(output, encoding="utf-8")
    return proc


def _print_cmd(prefix: str, cmd: list[str]) -> None:
    print(f"{prefix}: {' '.join(shlex.quote(c) for c in cmd)}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=str, default=None)
    parser.add_argument("--baseline-eval-json", type=str, required=True)
    parser.add_argument("--init-model-pt", type=str, default=None)
    parser.add_argument("--trials-json", type=str, default=None)
    parser.add_argument("--sweep-root", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train-val-days", type=int, default=60)
    parser.add_argument("--eval-val-days", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--phase2-anchor-start-weight", type=float, default=1.0)
    parser.add_argument("--w-minutes", type=float, default=1.0)
    parser.add_argument("--phase2-nll-guard-ratio", type=float, default=3.0)
    parser.add_argument("--phase2-nll-guard-abs", type=float, default=25.0)
    parser.add_argument("--phase2-nll-guard-ema-alpha", type=float, default=0.1)
    parser.add_argument("--phase2-nll-guard-consecutive-batches", type=int, default=2)
    parser.add_argument("--phase2-max-backoffs-before-rollback", type=int, default=3)
    parser.add_argument("--phase2-min-a2-scale", type=float, default=0.125)
    parser.add_argument("--max-delta-minutes-mae-lineup0", type=float, default=0.12)
    parser.add_argument("--max-delta-minutes-mae-lineup1", type=float, default=0.15)
    parser.add_argument("--max-delta-minutes-gap-abs", type=float, default=0.05)
    parser.add_argument("--max-delta-active-count-mae", type=float, default=0.10)
    parser.add_argument("--skip-world-contract-check", action="store_true")
    parser.add_argument("--world-num-games", type=int, default=1)
    parser.add_argument("--world-num-worlds", type=int, default=64)
    parser.add_argument("--auto-promote", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_dir = _resolve_dataset_dir(args.dataset_dir)
    baseline_eval_path = Path(args.baseline_eval_json).expanduser().resolve()
    if not baseline_eval_path.exists():
        raise FileNotFoundError(f"baseline eval json not found: {baseline_eval_path}")
    baseline = _load_eval_metrics(baseline_eval_path)
    if not _is_finite_eval(baseline):
        raise ValueError("baseline eval has non-finite metrics")

    root_default = paths.get_data_root() / "training" / "runs" / f"game_transformer_v2_phase2_sweep_{_utc_now_compact()}"
    sweep_root = Path(args.sweep_root).expanduser().resolve() if args.sweep_root else root_default
    sweep_root.mkdir(parents=True, exist_ok=True)
    trials_dir = sweep_root / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)

    if args.trials_json:
        trials = _read_trials_file(Path(args.trials_json).expanduser().resolve())
    else:
        trials = _default_trials()
    if not trials:
        raise ValueError("no trials resolved")

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(dataset_dir),
        "baseline_eval_json": str(baseline_eval_path),
        "baseline_metrics": baseline.__dict__,
        "sweep_root": str(sweep_root),
        "epochs": int(args.epochs),
        "train_val_days": int(args.train_val_days),
        "eval_val_days": int(args.eval_val_days),
        "batch_size": int(args.batch_size),
        "device": str(args.device),
        "init_model_pt": str(Path(args.init_model_pt).expanduser().resolve()) if args.init_model_pt else None,
        "promotion_gate": {
            "max_delta_minutes_mae_lineup0": float(args.max_delta_minutes_mae_lineup0),
            "max_delta_minutes_mae_lineup1": float(args.max_delta_minutes_mae_lineup1),
            "max_delta_minutes_gap_abs": float(args.max_delta_minutes_gap_abs),
            "max_delta_active_count_mae": float(args.max_delta_active_count_mae),
        },
        "trials": [{"name": t.name, "params": t.params} for t in trials],
    }
    (sweep_root / "sweep_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    results: list[dict[str, Any]] = []
    for idx, trial in enumerate(trials, start=1):
        trial_root = trials_dir / trial.name
        run_dir = trial_root / "run"
        eval_json = trial_root / f"eval_slices_{int(args.eval_val_days)}d.json"
        train_log = trial_root / "train.log"
        eval_log = trial_root / "eval.log"

        trial_result: dict[str, Any] = {
            "trial_name": trial.name,
            "trial_index": idx,
            "params": trial.params,
            "run_dir": str(run_dir),
            "eval_json": str(eval_json),
            "status": "planned",
        }

        train_cmd = [
            "uv",
            "run",
            "python",
            "-m",
            "scripts.rotation.train_game_transformer_v2",
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
            str(int(args.seed)),
            "--w-minutes",
            str(float(args.w_minutes)),
            "--phase2-anchor-start-weight",
            str(float(args.phase2_anchor_start_weight)),
            "--enable-phase2-flow",
            "--phase2-nll-guard-ratio",
            str(float(args.phase2_nll_guard_ratio)),
            "--phase2-nll-guard-abs",
            str(float(args.phase2_nll_guard_abs)),
            "--phase2-nll-guard-ema-alpha",
            str(float(args.phase2_nll_guard_ema_alpha)),
            "--phase2-nll-guard-consecutive-batches",
            str(int(args.phase2_nll_guard_consecutive_batches)),
            "--phase2-max-backoffs-before-rollback",
            str(int(args.phase2_max_backoffs_before_rollback)),
            "--phase2-min-a2-scale",
            str(float(args.phase2_min_a2_scale)),
        ]
        if args.init_model_pt:
            train_cmd.extend(["--init-model-pt", str(Path(args.init_model_pt).expanduser().resolve())])
        train_cmd.extend(_to_cli_args(trial.params))

        eval_cmd = [
            "uv",
            "run",
            "python",
            "-m",
            "scripts.rotation.eval_game_transformer_v2",
            "--run-dir",
            str(run_dir),
            "--dataset-dir",
            str(dataset_dir),
            "--val-days",
            str(int(args.eval_val_days)),
            "--batch-size",
            str(int(args.batch_size)),
            "--num-workers",
            str(int(args.num_workers)),
            "--device",
            str(args.device),
            "--active-threshold",
            "4.0",
            "--out-json",
            str(eval_json),
        ]

        _print_cmd(f"[phase2_sweep] train {idx}/{len(trials)} {trial.name}", train_cmd)
        if args.dry_run:
            trial_result["status"] = "dry_run"
            results.append(trial_result)
            continue
        train_proc = _run(train_cmd, log_path=train_log)
        trial_result["train_rc"] = int(train_proc.returncode)
        if train_proc.returncode != 0:
            trial_result["status"] = "train_failed"
            results.append(trial_result)
            continue

        if not (run_dir / "summary.json").exists():
            trial_result["status"] = "missing_summary"
            results.append(trial_result)
            continue

        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        stability = summary.get("phase2_stability", {}) or {}
        rollback = bool(stability.get("rollback_triggered", False))
        trial_result["rollback_triggered"] = rollback
        trial_result["phase2_backoff_count"] = int(stability.get("backoff_count", 0))
        trial_result["phase2_final_a2_scale"] = _float_or_nan(stability.get("final_a2_scale"))

        _print_cmd(f"[phase2_sweep] eval {idx}/{len(trials)} {trial.name}", eval_cmd)
        eval_proc = _run(eval_cmd, log_path=eval_log)
        trial_result["eval_rc"] = int(eval_proc.returncode)
        if eval_proc.returncode != 0 or not eval_json.exists():
            trial_result["status"] = "eval_failed"
            results.append(trial_result)
            continue

        metrics = _load_eval_metrics(eval_json)
        if not _is_finite_eval(metrics):
            trial_result["status"] = "eval_nonfinite"
            trial_result["metrics"] = metrics.__dict__
            results.append(trial_result)
            continue

        deltas = _diff_metrics(metrics, baseline)
        score = _composite_score(deltas)
        promotion_ok = _meets_promotion_gate(
            deltas=deltas,
            rollback_triggered=rollback,
            max_delta_minutes_mae_lineup0=float(args.max_delta_minutes_mae_lineup0),
            max_delta_minutes_mae_lineup1=float(args.max_delta_minutes_mae_lineup1),
            max_delta_minutes_gap_abs=float(args.max_delta_minutes_gap_abs),
            max_delta_active_count_mae=float(args.max_delta_active_count_mae),
        )

        trial_result["metrics"] = metrics.__dict__
        trial_result["deltas_vs_baseline"] = deltas
        trial_result["composite_score"] = float(score)
        trial_result["promotion_gate_pass"] = bool(promotion_ok)
        trial_result["status"] = "ok"
        results.append(trial_result)

    (sweep_root / "trial_results.json").write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")

    ok_rows = [r for r in results if r.get("status") == "ok"]
    promoted: dict[str, Any] | None = None
    if ok_rows:
        ranked = sorted(ok_rows, key=lambda r: (float(r["composite_score"]), str(r["trial_name"])))
        leaderboard = pd.DataFrame(
            [
                {
                    "trial_name": r["trial_name"],
                    "composite_score": float(r["composite_score"]),
                    "promotion_gate_pass": bool(r["promotion_gate_pass"]),
                    "minutes_mae_lineup0": _float_or_nan((r.get("metrics", {}) or {}).get("minutes_mae_lineup0")),
                    "minutes_mae_lineup1": _float_or_nan((r.get("metrics", {}) or {}).get("minutes_mae_lineup1")),
                    "minutes_mae_gap_abs": _float_or_nan((r.get("metrics", {}) or {}).get("minutes_mae_gap_abs")),
                    "active_count_mae": _float_or_nan((r.get("metrics", {}) or {}).get("active_count_mae")),
                    "delta_minutes_mae_lineup0": _float_or_nan((r.get("deltas_vs_baseline", {}) or {}).get("delta_minutes_mae_lineup0")),
                    "delta_minutes_mae_lineup1": _float_or_nan((r.get("deltas_vs_baseline", {}) or {}).get("delta_minutes_mae_lineup1")),
                    "delta_minutes_mae_gap_abs": _float_or_nan((r.get("deltas_vs_baseline", {}) or {}).get("delta_minutes_mae_gap_abs")),
                    "delta_active_count_mae": _float_or_nan((r.get("deltas_vs_baseline", {}) or {}).get("delta_active_count_mae")),
                    "rollback_triggered": bool(r.get("rollback_triggered", False)),
                    "run_dir": str(r.get("run_dir", "")),
                    "eval_json": str(r.get("eval_json", "")),
                }
                for r in ranked
            ]
        )
        leaderboard.to_csv(sweep_root / "leaderboard.csv", index=False)

        md_cols = list(leaderboard.columns)
        md_header = "| " + " | ".join(md_cols) + " |"
        md_sep = "| " + " | ".join(["---"] * len(md_cols)) + " |"
        md_rows = [
            "| " + " | ".join(str(row[c]) for c in md_cols) + " |"
            for row in leaderboard.to_dict(orient="records")
        ]
        md_lines = ["# GameTransformerV2 Phase 2 Sweep Leaderboard", "", f"Generated at: {datetime.now(timezone.utc).isoformat()}", "", md_header, md_sep, *md_rows, ""]
        (sweep_root / "leaderboard.md").write_text("\n".join(md_lines), encoding="utf-8")

        passing = [r for r in ranked if bool(r.get("promotion_gate_pass"))]
        if passing and args.auto_promote:
            best = passing[0]
            promoted = {
                "trial_name": str(best["trial_name"]),
                "run_dir": str(best["run_dir"]),
                "eval_json": str(best["eval_json"]),
                "metrics": best["metrics"],
                "deltas_vs_baseline": best["deltas_vs_baseline"],
                "composite_score": float(best["composite_score"]),
            }
            if not args.skip_world_contract_check:
                world_summary = sweep_root / "promoted_world_summary.json"
                world_cmd = [
                    "uv",
                    "run",
                    "python",
                    "-m",
                    "scripts.rotation.generate_worlds_game_transformer_v2",
                    "--run-dir",
                    str(best["run_dir"]),
                    "--dataset-dir",
                    str(dataset_dir),
                    "--val-days",
                    str(int(args.eval_val_days)),
                    "--num-games",
                    str(int(args.world_num_games)),
                    "--num-worlds",
                    str(int(args.world_num_worlds)),
                    "--batch-size",
                    "1",
                    "--num-workers",
                    "0",
                    "--device",
                    str(args.device),
                    "--strict-contracts",
                    "--out-summary-json",
                    str(world_summary),
                ]
                _print_cmd("[phase2_sweep] promoted world check", world_cmd)
                world_proc = _run(world_cmd, log_path=sweep_root / "promoted_world_check.log")
                promoted["world_check_rc"] = int(world_proc.returncode)
                promoted["world_check_summary_json"] = str(world_summary)
                if world_proc.returncode != 0:
                    promoted["world_check_failed"] = True
            (sweep_root / "promoted_phase2.json").write_text(
                json.dumps(promoted, indent=2, sort_keys=True),
                encoding="utf-8",
            )
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(dataset_dir),
        "baseline_eval_json": str(baseline_eval_path),
        "baseline_metrics": baseline.__dict__,
        "sweep_root": str(sweep_root),
        "num_trials": int(len(trials)),
        "num_completed": int(len([r for r in results if r.get("status") == "ok"])),
        "num_promotion_pass": int(len([r for r in results if bool(r.get("promotion_gate_pass"))])),
        "promoted": promoted,
    }
    (sweep_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
