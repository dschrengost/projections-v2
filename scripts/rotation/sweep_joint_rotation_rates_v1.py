#!/usr/bin/env python3
"""Automated hyperparameter sweep for joint rotation+minutes+rates training.

Runs multiple training trials via:
  scripts/rotation/train_joint_rotation_rates_model_v1.py

Outputs:
  - <sweep_root>/sweep_manifest.json
  - <sweep_root>/trial_results.json
  - <sweep_root>/leaderboard.csv
  - <sweep_root>/leaderboard.md
  - <sweep_root>/trials/<trial_name>/train.log
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd

from projections import paths

MANIFEST_LINE_RE = re.compile(r"wrote manifest\s*->\s*(?P<path>.+manifest\.json)\s*$")


@dataclass(frozen=True)
class Trial:
    name: str
    params: dict[str, Any]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slugify(text: str) -> str:
    allowed = []
    for ch in text.lower():
        if ch.isalnum() or ch in {"-", "_"}:
            allowed.append(ch)
        elif ch in {".", " "}:
            allowed.append("_")
    out = "".join(allowed).strip("_")
    return out or "trial"


def _format_value(v: Any) -> str:
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if v == 0:
            return "0"
        if abs(v) >= 1:
            txt = f"{v:.4f}".rstrip("0").rstrip(".")
            return txt.replace("-", "m")
        txt = f"{v:.2e}".replace("+", "")
        txt = txt.replace("-", "m").replace(".", "p")
        return txt
    return str(v)


def _default_trials(preset: str) -> list[Trial]:
    if preset == "quick":
        grid = {
            "lr": [3e-4, 6e-4, 1e-3],
            "weight_decay": [1e-5, 1e-4],
            "dropout": [0.05, 0.1],
            "k_reg_weight": [0.05],
            "anti_smear_weight": [0.05],
        }
    elif preset == "standard":
        grid = {
            "lr": [3e-4, 6e-4, 1e-3],
            "weight_decay": [1e-5, 1e-4],
            "dropout": [0.05, 0.1, 0.15],
            "embed_dim": [96, 128],
            "hidden_dim": [192, 256],
            "share_temperature": [0.9, 1.0],
            "k_reg_weight": [0.05, 0.1],
            "anti_smear_weight": [0.05, 0.1],
        }
    else:
        raise ValueError(f"Unknown preset: {preset}")

    keys = list(grid.keys())
    out: list[Trial] = []
    for idx, vals in enumerate(product(*[grid[k] for k in keys]), start=1):
        params = dict(zip(keys, vals, strict=False))
        parts = [f"{k}-{_format_value(params[k])}" for k in keys if k in {"lr", "weight_decay", "dropout"}]
        name = f"t{idx:03d}_{'_'.join(parts)}"
        out.append(Trial(name=_slugify(name), params=params))
    return out


def _read_trials_file(path: Path) -> list[Trial]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("trials file must be a JSON list")
    trials: list[Trial] = []
    for idx, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"trial[{idx}] must be an object")
        if "params" in item:
            params = item["params"]
            name = item.get("name")
        else:
            params = {k: v for k, v in item.items() if k != "name"}
            name = item.get("name")
        if not isinstance(params, dict):
            raise ValueError(f"trial[{idx}] params must be an object")
        clean = {str(k): v for k, v in params.items()}
        trial_name = _slugify(str(name)) if name else _slugify(f"trial_{idx:03d}")
        trials.append(Trial(name=trial_name, params=clean))
    return trials


def _to_cli_args(params: dict[str, Any]) -> list[str]:
    args: list[str] = []
    for key, value in params.items():
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                args.append(flag)
            continue
        args.extend([flag, str(value)])
    return args


def _split_extra_args(extra_args: list[str]) -> list[str]:
    parts: list[str] = []
    for item in extra_args:
        parts.extend(shlex.split(item))
    return parts


def _parse_manifest_path(stdout: str, stderr: str) -> Path | None:
    for block in (stdout, stderr):
        for line in block.splitlines():
            m = MANIFEST_LINE_RE.search(line.strip())
            if m:
                return Path(m.group("path")).expanduser().resolve()
    return None


def _latest_run_dir(out_dir: Path, run_tag_prefix: str) -> Path | None:
    matches = sorted(
        [p for p in out_dir.glob(f"{run_tag_prefix}_*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None


def _score_rows(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        ("val_loss", 0.10),
        ("val_minutes_mae", 0.40),
        ("val_rates_mae", 0.25),
        ("val_eff_mae", 0.15),
        ("val_anti_smear", 0.10),
    ]
    work = df.copy()
    for metric, _weight in metrics:
        rank_col = f"rank_{metric}"
        work[rank_col] = work[metric].rank(method="min", ascending=True)
        denom = max(len(work) - 1, 1)
        work[f"rank01_{metric}"] = (work[rank_col] - 1.0) / float(denom)
    work["composite_rank_score"] = 0.0
    for metric, weight in metrics:
        work["composite_rank_score"] += float(weight) * work[f"rank01_{metric}"]
    work = work.sort_values(
        by=["composite_rank_score", "val_minutes_mae", "val_rates_mae", "val_loss"],
        ascending=[True, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    work["place"] = range(1, len(work) + 1)
    return work


def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _load_best_metrics(manifest_path: Path) -> dict[str, float]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    best = manifest.get("best", {}) or {}
    return {
        "val_loss": _float_or_nan(best.get("val_loss")),
        "val_minutes_mae": _float_or_nan(best.get("val_minutes_mae")),
        "val_rates_mae": _float_or_nan(best.get("val_rates_mae")),
        "val_eff_mae": _float_or_nan(best.get("val_eff_mae")),
        "val_anti_smear": _float_or_nan(best.get("val_anti_smear")),
    }


def _write_leaderboard_md(path: Path, scored: pd.DataFrame) -> None:
    cols = [
        "place",
        "trial_name",
        "val_minutes_mae",
        "val_rates_mae",
        "val_eff_mae",
        "val_anti_smear",
        "val_loss",
        "composite_rank_score",
        "run_id",
    ]
    top = scored.loc[:, cols].copy()
    for c in ["val_minutes_mae", "val_rates_mae", "val_eff_mae", "val_anti_smear", "val_loss", "composite_rank_score"]:
        if c in top.columns:
            top[c] = top[c].map(lambda v: f"{float(v):.6f}")

    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows: list[str] = []
    for row in top.itertuples(index=False):
        rows.append("| " + " | ".join(str(v) for v in row) + " |")

    lines = [
        "# Joint Sweep Leaderboard",
        "",
        f"Generated at: {_utc_now_iso()}",
        "",
        header,
        sep,
        *rows,
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _load_existing_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    return []


def _result_index(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in results:
        name = str(row.get("trial_name", "")).strip()
        if name:
            out[name] = row
    return out


def _resolve_device(device_arg: str) -> str:
    val = str(device_arg).strip().lower()
    if val in {"cpu", "cuda", "mps"}:
        return val
    if val != "auto":
        raise ValueError("--device must be one of: auto, cpu, cuda, mps")
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _build_subprocess_env(*, stability_env: bool, omp_num_threads: int) -> dict[str, str]:
    env = dict(os.environ)
    if not stability_env:
        return env
    threads = str(max(int(omp_num_threads), 1))
    defaults = {
        "OMP_NUM_THREADS": threads,
        "MKL_NUM_THREADS": threads,
        "OPENBLAS_NUM_THREADS": threads,
        "NUMEXPR_NUM_THREADS": threads,
        "PYTHONFAULTHANDLER": "1",
        "TORCH_SHOW_CPP_STACKTRACES": "1",
    }
    for k, v in defaults.items():
        env.setdefault(k, v)
    return env


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Joint dataset directory.")
    parser.add_argument(
        "--train-script",
        type=Path,
        default=Path("scripts/rotation/train_joint_rotation_rates_model_v1.py"),
        help="Trainer entrypoint.",
    )
    parser.add_argument(
        "--sweep-root",
        type=Path,
        default=paths.get_data_root() / "artifacts" / "joint_rotation_rates_v1" / "sweeps" / "latest",
        help="Output directory for this sweep run.",
    )
    parser.add_argument(
        "--runs-out-dir",
        type=Path,
        default=paths.get_data_root() / "artifacts" / "joint_rotation_rates_v1" / "runs",
        help="Trainer out-dir where model runs are created.",
    )
    parser.add_argument("--run-tag-prefix", type=str, default="joint_sweep")
    parser.add_argument("--preset", type=str, choices=["quick", "standard"], default="quick")
    parser.add_argument("--trials-json", type=Path, default=None, help="Optional JSON list of trial specs.")
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--skip-completed", action="store_true", help="Skip trials already in trial_results.json.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop sweep on first failed trial.")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-days", type=int, default=14)
    parser.add_argument("--val-start-date", type=str, default=None)
    parser.add_argument("--val-end-date", type=str, default=None)
    parser.add_argument("--max-team-games", type=int, default=None)

    parser.add_argument("--lambda-minutes", type=float, default=1.0)
    parser.add_argument("--lambda-rates", type=float, default=0.6)
    parser.add_argument("--lambda-eff", type=float, default=0.2)
    parser.add_argument("--lambda-rot", type=float, default=0.4)
    parser.add_argument("--gate-bce-weight", type=float, default=1.0)
    parser.add_argument("--minutes-out-weight", type=float, default=0.25)
    parser.add_argument("--k-target", type=float, default=9.5)
    parser.add_argument("--k-target-source", type=str, choices=["fixed", "label", "blend"], default="fixed")
    parser.add_argument("--k-target-blend", type=float, default=0.5)
    parser.add_argument("--anti-smear-floor", type=float, default=4.0)
    parser.add_argument("--freeze-minutes-epochs", type=int, default=0)

    parser.add_argument("--use-prior-head", action="store_true")
    parser.add_argument("--use-team-embeddings", action="store_true")
    parser.add_argument("--use-player-embeddings", action="store_true")
    parser.add_argument("--use-player-team-embeddings", action="store_true")
    parser.add_argument("--prior-weight-col", type=str, default="minutes_from_stints_prior_20")
    parser.add_argument("--prior-weight-floor", type=float, default=1.0)
    parser.add_argument("--team-embedding-dim", type=int, default=8)
    parser.add_argument("--player-embedding-dim", type=int, default=16)
    parser.add_argument("--player-team-hash-buckets", type=int, default=16384)
    parser.add_argument("--player-team-embedding-dim", type=int, default=8)
    parser.add_argument("--num-transformer-layers", type=int, default=2)
    parser.add_argument("--num-attention-heads", type=int, default=4)
    parser.add_argument("--rates-hidden-dim", type=int, default=128)
    parser.add_argument("--alloc-activation", type=str, default="entmax")
    parser.add_argument("--entmax-alpha", type=float, default=1.5)
    parser.add_argument(
        "--omp-num-threads",
        type=int,
        default=1,
        help="Thread cap used by stability env defaults for trainer subprocesses.",
    )
    stability_group = parser.add_mutually_exclusive_group()
    stability_group.add_argument(
        "--stability-env",
        dest="stability_env",
        action="store_true",
        help="Enable conservative threading/debug env defaults for trainer subprocesses.",
    )
    stability_group.add_argument(
        "--no-stability-env",
        dest="stability_env",
        action="store_false",
        help="Disable conservative threading/debug env defaults for trainer subprocesses.",
    )
    parser.set_defaults(stability_env=True)

    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Additional argument(s) passed to trainer. Can be repeated.",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.expanduser().resolve()
    train_script = args.train_script.expanduser().resolve()
    sweep_root = args.sweep_root.expanduser().resolve()
    runs_out_dir = args.runs_out_dir.expanduser().resolve()
    trials_dir = sweep_root / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)
    runs_out_dir.mkdir(parents=True, exist_ok=True)

    if not train_script.exists():
        raise FileNotFoundError(f"Missing train script: {train_script}")
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Missing dataset dir: {dataset_dir}")

    if args.trials_json is not None:
        trials = _read_trials_file(args.trials_json.expanduser().resolve())
    else:
        trials = _default_trials(args.preset)
    if args.max_trials is not None and args.max_trials > 0:
        trials = trials[: int(args.max_trials)]
    if not trials:
        raise ValueError("No trials resolved for sweep")

    sweep_manifest = {
        "created_at": _utc_now_iso(),
        "dataset_dir": str(dataset_dir),
        "train_script": str(train_script),
        "sweep_root": str(sweep_root),
        "runs_out_dir": str(runs_out_dir),
        "preset": args.preset,
        "trial_count": len(trials),
        "trials": [{"name": t.name, "params": t.params} for t in trials],
    }
    (sweep_root / "sweep_manifest.json").write_text(json.dumps(sweep_manifest, indent=2, sort_keys=True), encoding="utf-8")

    results_path = sweep_root / "trial_results.json"
    results = _load_existing_results(results_path)
    existing = _result_index(results)

    resolved_device = _resolve_device(str(args.device))
    print(f"[joint_sweep] resolved device: {resolved_device} (requested={args.device})")
    child_env = _build_subprocess_env(
        stability_env=bool(args.stability_env),
        omp_num_threads=int(args.omp_num_threads),
    )
    if bool(args.stability_env):
        print(
            "[joint_sweep] stability env:",
            f"OMP_NUM_THREADS={child_env.get('OMP_NUM_THREADS')}",
            f"MKL_NUM_THREADS={child_env.get('MKL_NUM_THREADS')}",
            f"OPENBLAS_NUM_THREADS={child_env.get('OPENBLAS_NUM_THREADS')}",
        )

    base_args = [
        "--dataset-dir",
        str(dataset_dir),
        "--out-dir",
        str(runs_out_dir),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--device",
        str(resolved_device),
        "--num-workers",
        str(args.num_workers),
        "--seed",
        str(args.seed),
        "--val-days",
        str(args.val_days),
        "--lambda-minutes",
        str(args.lambda_minutes),
        "--lambda-rates",
        str(args.lambda_rates),
        "--lambda-eff",
        str(args.lambda_eff),
        "--lambda-rot",
        str(args.lambda_rot),
        "--gate-bce-weight",
        str(args.gate_bce_weight),
        "--minutes-out-weight",
        str(args.minutes_out_weight),
        "--k-target",
        str(args.k_target),
        "--k-target-source",
        str(args.k_target_source),
        "--k-target-blend",
        str(args.k_target_blend),
        "--anti-smear-floor",
        str(args.anti_smear_floor),
        "--freeze-minutes-epochs",
        str(args.freeze_minutes_epochs),
        "--prior-weight-col",
        str(args.prior_weight_col),
        "--prior-weight-floor",
        str(args.prior_weight_floor),
        "--team-embedding-dim",
        str(args.team_embedding_dim),
        "--player-embedding-dim",
        str(args.player_embedding_dim),
        "--player-team-hash-buckets",
        str(args.player_team_hash_buckets),
        "--player-team-embedding-dim",
        str(args.player_team_embedding_dim),
        "--num-transformer-layers",
        str(args.num_transformer_layers),
        "--num-attention-heads",
        str(args.num_attention_heads),
        "--rates-hidden-dim",
        str(args.rates_hidden_dim),
        "--alloc-activation",
        str(args.alloc_activation),
        "--entmax-alpha",
        str(args.entmax_alpha),
    ]
    if args.use_prior_head:
        base_args.append("--use-prior-head")
    if args.use_team_embeddings:
        base_args.append("--use-team-embeddings")
    if args.use_player_embeddings:
        base_args.append("--use-player-embeddings")
    if args.use_player_team_embeddings:
        base_args.append("--use-player-team-embeddings")
    if args.val_start_date:
        base_args.extend(["--val-start-date", str(args.val_start_date)])
    if args.val_end_date:
        base_args.extend(["--val-end-date", str(args.val_end_date)])
    if args.max_team_games is not None:
        base_args.extend(["--max-team-games", str(args.max_team_games)])

    base_args.extend(_split_extra_args(list(args.extra_arg)))

    for idx, trial in enumerate(trials, start=1):
        if args.skip_completed and trial.name in existing and existing[trial.name].get("status") == "ok":
            print(f"[joint_sweep] skip {idx}/{len(trials)} {trial.name} (already completed)")
            continue

        trial_dir = trials_dir / trial.name
        trial_dir.mkdir(parents=True, exist_ok=True)
        log_path = trial_dir / "train.log"
        run_tag = _slugify(f"{args.run_tag_prefix}_{idx:03d}_{trial.name}")[:80]

        cmd = [sys.executable, str(train_script), "--run-tag", run_tag, *base_args, *_to_cli_args(trial.params)]
        print(f"[joint_sweep] run {idx}/{len(trials)} {trial.name}")
        print(f"[joint_sweep] cmd: {' '.join(shlex.quote(c) for c in cmd)}")

        start = time.time()
        if args.dry_run:
            duration = 0.0
            row = {
                "trial_name": trial.name,
                "trial_index": idx,
                "status": "dry_run",
                "return_code": 0,
                "duration_sec": duration,
                "run_tag": run_tag,
                "run_id": None,
                "run_dir": None,
                "manifest_path": None,
                "params": trial.params,
                "metrics": {},
                "log_path": str(log_path),
            }
            existing[trial.name] = row
            continue

        proc = subprocess.run(cmd, check=False, capture_output=True, text=True, env=child_env)
        duration = time.time() - start
        log_blob = (
            f"# CMD\n{' '.join(shlex.quote(c) for c in cmd)}\n\n"
            f"# STDOUT\n{proc.stdout}\n\n"
            f"# STDERR\n{proc.stderr}\n"
        )
        log_path.write_text(log_blob, encoding="utf-8")

        manifest_path = _parse_manifest_path(proc.stdout or "", proc.stderr or "")
        run_dir: Path | None = None
        run_id: str | None = None
        metrics: dict[str, float] = {}
        status = "ok" if proc.returncode == 0 else "failed"
        error: str | None = None

        if proc.returncode == 0:
            if manifest_path is None:
                latest = _latest_run_dir(runs_out_dir, run_tag)
                if latest is not None:
                    manifest_path = latest / "manifest.json"
            if manifest_path is not None and manifest_path.exists():
                run_dir = manifest_path.parent
                run_id = run_dir.name
                try:
                    metrics = _load_best_metrics(manifest_path)
                except Exception as exc:
                    status = "failed"
                    error = f"Failed loading manifest metrics: {exc}"
            else:
                status = "failed"
                error = "Trainer succeeded but manifest path could not be resolved"
        else:
            error = f"Trainer exited with code {proc.returncode}"

        row = {
            "trial_name": trial.name,
            "trial_index": idx,
            "status": status,
            "return_code": int(proc.returncode),
            "duration_sec": float(duration),
            "run_tag": run_tag,
            "run_id": run_id,
            "run_dir": str(run_dir) if run_dir else None,
            "manifest_path": str(manifest_path) if manifest_path else None,
            "params": trial.params,
            "metrics": metrics,
            "log_path": str(log_path),
            "error": error,
        }
        existing[trial.name] = row
        print(
            f"[joint_sweep] result {trial.name}: status={status} rc={proc.returncode} "
            f"mins={duration/60.0:.1f} run_id={run_id}"
        )
        if error:
            print(f"[joint_sweep] error: {error}")
        if status != "ok" and args.fail_fast:
            break

        results = [existing[t.name] for t in trials if t.name in existing]
        results_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")

    results = [existing[t.name] for t in trials if t.name in existing]
    results_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")

    ok_rows: list[dict[str, Any]] = []
    for row in results:
        if row.get("status") != "ok":
            continue
        metrics = row.get("metrics") or {}
        ok_rows.append(
            {
                "trial_name": row.get("trial_name"),
                "trial_index": row.get("trial_index"),
                "run_id": row.get("run_id"),
                "run_dir": row.get("run_dir"),
                "duration_sec": row.get("duration_sec"),
                "val_loss": _float_or_nan(metrics.get("val_loss")),
                "val_minutes_mae": _float_or_nan(metrics.get("val_minutes_mae")),
                "val_rates_mae": _float_or_nan(metrics.get("val_rates_mae")),
                "val_eff_mae": _float_or_nan(metrics.get("val_eff_mae")),
                "val_anti_smear": _float_or_nan(metrics.get("val_anti_smear")),
            }
        )

    leaderboard_csv = sweep_root / "leaderboard.csv"
    leaderboard_md = sweep_root / "leaderboard.md"

    if ok_rows:
        df = pd.DataFrame(ok_rows)
        for col in ["val_loss", "val_minutes_mae", "val_rates_mae", "val_eff_mae", "val_anti_smear"]:
            df = df[df[col].map(lambda x: not (isinstance(x, float) and math.isnan(x)))]
        if df.empty:
            print("[joint_sweep] no successful rows with finite metrics; leaderboard not generated")
            return 1
        scored = _score_rows(df)
        scored.to_csv(leaderboard_csv, index=False)
        _write_leaderboard_md(leaderboard_md, scored)
        top = scored.iloc[0].to_dict()
        print("[joint_sweep] best trial:")
        print(
            f"  place={int(top['place'])} trial={top['trial_name']} run_id={top['run_id']} "
            f"val_minutes_mae={top['val_minutes_mae']:.4f} val_rates_mae={top['val_rates_mae']:.4f} "
            f"val_eff_mae={top['val_eff_mae']:.4f} val_smear={top['val_anti_smear']:.4f} "
            f"val_loss={top['val_loss']:.4f}"
        )
        print(f"[joint_sweep] wrote leaderboard -> {leaderboard_csv}")
        print(f"[joint_sweep] wrote leaderboard -> {leaderboard_md}")
        return 0

    if args.dry_run:
        print(f"[joint_sweep] dry-run complete ({len(results)} planned trials, no training executed)")
        return 0

    print("[joint_sweep] no successful trials")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
