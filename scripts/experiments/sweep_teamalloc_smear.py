"""CPU-friendly sweep runner for DeepSets team-allocation smear tuning.

Example:
    uv run python -m scripts.experiments.sweep_teamalloc_smear \\
      --train-parquet /path/to/train.parquet \\
      --out-root-dir artifacts/experiments/team_alloc_sweeps/smear_v1 \\
      --split-mode time --val-frac 0.2 --epochs 8 --seed 7 \\
      --hidden 128 --max-players 21 --filter-horizon-min 180

Notes:
    - This script runs multiple short trainings via
      `scripts/experiments/minutes_teamalloc_deepsets.py`.
    - Metrics are computed from each run's `val_predictions.parquet`.
    - `gate_threshold` affects only gate metrics; it does not change allocation.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class SweepConfig:
    run_name: str
    alloc_eps: float
    gamma_entropy: float
    delta_sparsity: float
    lambda_fp_dnp: float
    fp_threshold: float
    fp_power: int


def _parse_time_split_ranges(stdout: str) -> tuple[str | None, str | None, str | None, str | None]:
    """Parse the time split log line to capture train/val date ranges."""
    # Example:
    # Time split: train=1234 (2024-10-01..2025-02-10) | val=345 (2025-02-11..2025-03-10)
    pattern = re.compile(
        r"Time split:\s+train=\d+\s+\((?P<train_start>\d{4}-\d{2}-\d{2})\.\.(?P<train_end>\d{4}-\d{2}-\d{2})\)\s+\|\s+val=\d+\s+\((?P<val_start>\d{4}-\d{2}-\d{2})\.\.(?P<val_end>\d{4}-\d{2}-\d{2})\)"
    )
    match = pattern.search(stdout)
    if not match:
        return None, None, None, None
    return (
        match.group("train_start"),
        match.group("train_end"),
        match.group("val_start"),
        match.group("val_end"),
    )


def _safe_mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(values))


def _top_k_overlap(
    df: pd.DataFrame,
    *,
    top_k: int,
    group_cols: tuple[str, str] = ("game_id", "team_id"),
) -> float:
    overlaps: list[float] = []
    for _, g in df.groupby(list(group_cols), sort=False):
        if g.empty:
            continue
        n = len(g)
        k = min(int(top_k), n)
        if k <= 0:
            continue
        top_actual = set(g.nlargest(k, "minutes_actual")["player_id"].tolist())
        top_pred = set(g.nlargest(k, "minutes_pred")["player_id"].tolist())
        overlaps.append(len(top_actual & top_pred) / k)
    return _safe_mean(overlaps)


def _compute_gate_metrics_from_parquet(
    df: pd.DataFrame,
    *,
    gate_threshold: float,
) -> tuple[float, float, float]:
    if "gate_prob" not in df.columns:
        return float("nan"), float("nan"), float("nan")

    y_true = (df["minutes_actual"].to_numpy() > 0).astype(bool)
    y_pred = (df["gate_prob"].to_numpy() > gate_threshold).astype(bool)

    tp = float(np.sum(y_true & y_pred))
    fp = float(np.sum(~y_true & y_pred))
    fn = float(np.sum(y_true & ~y_pred))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def compute_parquet_metrics(
    df: pd.DataFrame,
    *,
    top_k: int,
    gate_threshold: float,
) -> dict[str, float]:
    required = {"game_id", "team_id", "player_id", "minutes_pred", "minutes_actual"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"val_predictions parquet missing required columns: {sorted(missing)}")

    abs_err = (df["minutes_pred"] - df["minutes_actual"]).abs()
    mae = float(abs_err.mean()) if len(df) else float("nan")

    ge10 = df["minutes_actual"] >= 10
    mae_ge_10 = float(abs_err[ge10].mean()) if int(ge10.sum()) else float("nan")

    top_overlap = _top_k_overlap(df, top_k=top_k)

    smear = (df["minutes_pred"] > 0) & (df["minutes_pred"] < 3)
    smear_per_team = df.assign(_smear=smear.astype(int)).groupby(["game_id", "team_id"])["_smear"].sum()
    smear_avg_per_team = float(smear_per_team.mean()) if len(smear_per_team) else float("nan")

    smear_dnp = smear & (df["minutes_actual"] == 0)
    smear_on_dnp_frac = float(smear_dnp.mean()) if len(df) else float("nan")

    dnp = df["minutes_actual"] == 0
    if dnp.any():
        mean_pred_on_dnp = float(df.loc[dnp, "minutes_pred"].mean())
        p90_pred_on_dnp = float(df.loc[dnp, "minutes_pred"].quantile(0.9))
    else:
        mean_pred_on_dnp = float("nan")
        p90_pred_on_dnp = float("nan")

    pred_sums = df.groupby(["game_id", "team_id"])["minutes_pred"].sum()
    max_sum_dev = float((pred_sums - 240.0).abs().max()) if len(pred_sums) else float("nan")

    gate_p, gate_r, gate_f1 = _compute_gate_metrics_from_parquet(df, gate_threshold=gate_threshold)

    return {
        "mae": mae,
        "mae_ge_10": mae_ge_10,
        "top8_overlap": top_overlap,
        "smear_avg_per_team": smear_avg_per_team,
        "smear_on_dnp_frac": smear_on_dnp_frac,
        "mean_pred_on_dnp": mean_pred_on_dnp,
        "p90_pred_on_dnp": p90_pred_on_dnp,
        "gate_precision": gate_p,
        "gate_recall": gate_r,
        "gate_f1": gate_f1,
        "max_sum_dev": max_sum_dev,
    }


def _run_one(
    *,
    cfg: SweepConfig,
    base_args: list[str],
    out_dir: Path,
    dry_run: bool,
    skip_existing: bool,
) -> tuple[int, str, Path]:
    run_dir = out_dir / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    val_path = run_dir / "val_predictions.parquet"
    log_path = run_dir / "train.log"

    if skip_existing and val_path.exists():
        return 0, f"[skip] {cfg.run_name} (found {val_path})", val_path

    cmd = [
        sys.executable,
        "-m",
        "scripts.experiments.minutes_teamalloc_deepsets",
        *base_args,
        "--out-dir",
        str(run_dir),
        "--alloc-eps",
        str(cfg.alloc_eps),
        "--gamma-entropy",
        str(cfg.gamma_entropy),
        "--delta-sparsity",
        str(cfg.delta_sparsity),
        "--lambda-fp-dnp",
        str(cfg.lambda_fp_dnp),
        "--debug-batch",
        "none",
    ]

    if dry_run:
        return 0, f"[dry-run] {cfg.run_name}: {' '.join(cmd)}", val_path

    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    log_path.write_text((proc.stdout or "") + "\n" + (proc.stderr or ""), encoding="utf-8")
    return proc.returncode, f"[run] {cfg.run_name} exit={proc.returncode} log={log_path}", val_path


def _pick_best(
    summary: pd.DataFrame,
    *,
    baseline_name: str,
    mae_ge_10_tolerance: float,
    top8_overlap_tolerance: float,
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    if summary.empty or baseline_name not in set(summary["run_name"].astype(str)):
        return summary, None

    baseline = summary[summary["run_name"] == baseline_name].iloc[0].to_dict()
    base_mae_ge_10 = float(baseline["mae_ge_10"])
    base_top8 = float(baseline["top8_overlap"])

    eligible = summary.copy()
    eligible = eligible[eligible["mae_ge_10"] <= base_mae_ge_10 + mae_ge_10_tolerance]
    eligible = eligible[eligible["top8_overlap"] >= base_top8 - top8_overlap_tolerance]

    if eligible.empty:
        return summary, None

    eligible = eligible.sort_values(
        by=["smear_on_dnp_frac", "smear_avg_per_team", "mae_ge_10", "mae"],
        ascending=[True, True, True, True],
        kind="mergesort",
    )
    best = eligible.iloc[0].to_dict()
    return summary, best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-parquet", type=Path, required=True)
    parser.add_argument("--out-root-dir", type=Path, required=True)
    parser.add_argument("--split-mode", type=str, choices=["random", "time"], default="time")
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--max-players", type=int, default=21)
    parser.add_argument("--filter-horizon-min", type=int, default=None)
    parser.add_argument("--drop-features", type=str, default=None)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=128)

    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--gate-threshold", type=float, default=0.5)

    parser.add_argument("--alloc-eps-base", type=float, default=1e-4)
    parser.add_argument("--gamma-entropy-base", type=float, default=0.01)
    parser.add_argument("--delta-sparsity-base", type=float, default=0.0)
    parser.add_argument("--lambda-fp-dnp-base", type=float, default=0.0)
    parser.add_argument("--fp-threshold", type=float, default=0.25)
    parser.add_argument("--fp-power", type=int, choices=[1, 2], default=2)

    parser.add_argument("--alloc-eps-sweep", type=str, default="1e-3,1e-4,1e-5")
    parser.add_argument("--gamma-entropy-mults", type=str, default="1,2")
    parser.add_argument("--delta-sparsity-values", type=str, default="0.0,0.01,0.02")
    parser.add_argument(
        "--lambda-fp-dnp-sweep",
        type=str,
        default="",
        help="Optional comma-separated lambda_fp_dnp values to sweep (e.g. 0.0,0.1,0.3,1.0)",
    )

    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")

    parser.add_argument("--baseline-name", type=str, default="baseline")
    parser.add_argument("--mae-ge-10-tol", type=float, default=0.15)
    parser.add_argument("--top8-overlap-tol", type=float, default=0.02)

    args = parser.parse_args()

    args.out_root_dir.mkdir(parents=True, exist_ok=True)

    base_args: list[str] = [
        "--train-parquet",
        str(args.train_parquet),
        "--split-mode",
        str(args.split_mode),
        "--val-frac",
        str(args.val_frac),
        "--epochs",
        str(args.epochs),
        "--seed",
        str(args.seed),
        "--hidden",
        str(args.hidden),
        "--max-players",
        str(args.max_players),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--grad-clip-norm",
        str(args.grad_clip_norm),
        "--batch-size",
        str(args.batch_size),
        "--top-k",
        str(args.top_k),
        "--gate-threshold",
        str(args.gate_threshold),
        "--fp-threshold",
        str(args.fp_threshold),
        "--fp-power",
        str(args.fp_power),
    ]
    if args.filter_horizon_min is not None:
        base_args.extend(["--filter-horizon-min", str(args.filter_horizon_min)])
    if args.drop_features:
        base_args.extend(["--drop-features", str(args.drop_features)])

    alloc_eps_values = [float(x.strip()) for x in args.alloc_eps_sweep.split(",") if x.strip()]
    gamma_mults = [float(x.strip()) for x in args.gamma_entropy_mults.split(",") if x.strip()]
    delta_values = [float(x.strip()) for x in args.delta_sparsity_values.split(",") if x.strip()]
    fp_lambda_values = [
        float(x.strip())
        for x in str(args.lambda_fp_dnp_sweep).split(",")
        if str(x).strip()
    ]

    configs: list[SweepConfig] = []
    configs.append(
        SweepConfig(
            run_name=args.baseline_name,
            alloc_eps=float(args.alloc_eps_base),
            gamma_entropy=float(args.gamma_entropy_base),
            delta_sparsity=float(args.delta_sparsity_base),
            lambda_fp_dnp=float(args.lambda_fp_dnp_base),
            fp_threshold=float(args.fp_threshold),
            fp_power=int(args.fp_power),
        )
    )

    for eps in alloc_eps_values:
        name = f"eps_{eps:g}"
        configs.append(
            SweepConfig(
                run_name=name,
                alloc_eps=float(eps),
                gamma_entropy=float(args.gamma_entropy_base),
                delta_sparsity=float(args.delta_sparsity_base),
                lambda_fp_dnp=float(args.lambda_fp_dnp_base),
                fp_threshold=float(args.fp_threshold),
                fp_power=int(args.fp_power),
            )
        )

    for mult in gamma_mults:
        gamma = float(args.gamma_entropy_base) * float(mult)
        name = f"gamma_{gamma:g}"
        configs.append(
            SweepConfig(
                run_name=name,
                alloc_eps=float(args.alloc_eps_base),
                gamma_entropy=float(gamma),
                delta_sparsity=float(args.delta_sparsity_base),
                lambda_fp_dnp=float(args.lambda_fp_dnp_base),
                fp_threshold=float(args.fp_threshold),
                fp_power=int(args.fp_power),
            )
        )

    for delta in delta_values:
        name = f"delta_{delta:g}"
        configs.append(
            SweepConfig(
                run_name=name,
                alloc_eps=float(args.alloc_eps_base),
                gamma_entropy=float(args.gamma_entropy_base),
                delta_sparsity=float(delta),
                lambda_fp_dnp=float(args.lambda_fp_dnp_base),
                fp_threshold=float(args.fp_threshold),
                fp_power=int(args.fp_power),
            )
        )

    for lam in fp_lambda_values:
        name = f"fp_{lam:g}"
        configs.append(
            SweepConfig(
                run_name=name,
                alloc_eps=float(args.alloc_eps_base),
                gamma_entropy=float(args.gamma_entropy_base),
                delta_sparsity=float(args.delta_sparsity_base),
                lambda_fp_dnp=float(lam),
                fp_threshold=float(args.fp_threshold),
                fp_power=int(args.fp_power),
            )
        )

    # Deduplicate identical configs by hyperparams (keep first run_name encountered).
    unique: dict[tuple[float, float, float, float, float, int], SweepConfig] = {}
    for cfg in configs:
        key = (
            cfg.alloc_eps,
            cfg.gamma_entropy,
            cfg.delta_sparsity,
            cfg.lambda_fp_dnp,
            cfg.fp_threshold,
            cfg.fp_power,
        )
        unique.setdefault(key, cfg)
    configs = list(unique.values())

    rows: list[dict[str, Any]] = []
    for cfg in configs:
        rc, msg, val_path = _run_one(
            cfg=cfg,
            base_args=base_args,
            out_dir=args.out_root_dir,
            dry_run=bool(args.dry_run),
            skip_existing=bool(args.skip_existing),
        )
        print(msg, file=sys.stderr)
        if args.dry_run:
            continue
        if rc != 0 or not val_path.exists():
            rows.append(
                {
                    "run_name": cfg.run_name,
                    "alloc_eps": cfg.alloc_eps,
                    "gamma_entropy": cfg.gamma_entropy,
                    "delta_sparsity": cfg.delta_sparsity,
                    "lambda_fp_dnp": cfg.lambda_fp_dnp,
                    "fp_threshold": cfg.fp_threshold,
                    "fp_power": cfg.fp_power,
                    "status": "failed",
                }
            )
            continue

        df = pd.read_parquet(val_path)
        metrics = compute_parquet_metrics(df, top_k=args.top_k, gate_threshold=args.gate_threshold)
        log_path = val_path.parent / "train.log"
        log_text = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
        train_start, train_end, val_start, val_end = _parse_time_split_ranges(log_text)

        val_dt_min = None
        val_dt_max = None
        if "game_date" in df.columns and len(df):
            dt = pd.to_datetime(df["game_date"], errors="coerce")
            if dt.notna().any():
                val_dt_min = str(dt.min().date())
                val_dt_max = str(dt.max().date())

        rows.append(
            {
                "run_name": cfg.run_name,
                "alloc_eps": cfg.alloc_eps,
                "gamma_entropy": cfg.gamma_entropy,
                "delta_sparsity": cfg.delta_sparsity,
                "lambda_fp_dnp": cfg.lambda_fp_dnp,
                "fp_threshold": cfg.fp_threshold,
                "fp_power": cfg.fp_power,
                "status": "ok",
                "train_start": train_start,
                "train_end": train_end,
                "val_start": val_start,
                "val_end": val_end,
                "val_min_game_date": val_dt_min,
                "val_max_game_date": val_dt_max,
                **metrics,
            }
        )

    if args.dry_run:
        return 0

    summary = pd.DataFrame(rows)
    summary_path = args.out_root_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote sweep summary: {summary_path}", file=sys.stderr)

    _, best = _pick_best(
        summary[summary["status"] == "ok"].copy(),
        baseline_name=args.baseline_name,
        mae_ge_10_tolerance=float(args.mae_ge_10_tol),
        top8_overlap_tolerance=float(args.top8_overlap_tol),
    )
    if best:
        print(
            "Best (constraints applied): "
            f"run={best['run_name']} smear_on_dnp_frac={best['smear_on_dnp_frac']:.4f} "
            f"smear_avg/team={best['smear_avg_per_team']:.2f} "
            f"mae_ge_10={best['mae_ge_10']:.3f} top8={best['top8_overlap']:.3f}",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
