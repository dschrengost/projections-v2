#!/usr/bin/env python3
"""Finalize a RotAlloc candidate with optional sweep, training, eval, and promotion config.

Example (A: sweep + finalize):
  uv run python -m scripts.experiments.finalize_rotalloc_candidate \
    --train-parquet artifacts/experiments/team_alloc_dataset/minutes_teamalloc_train_no_ot.parquet \
    --val-parquet artifacts/experiments/team_alloc_v3_time/val_predictions.parquet \
    --projections-data-root /home/daniel/projections-data \
    --out-dir artifacts/experiments/lgbm_rotalloc_final_v1 \
    --split-mode time --val-frac 0.2 --rot-threshold 8 --seed 7 \
    --use-expected-k --k-min 8 --cap-max 48 \
    --sweep

Example (B: explicit knobs, no sweep):
  uv run python -m scripts.experiments.finalize_rotalloc_candidate \
    --train-parquet artifacts/experiments/team_alloc_dataset/minutes_teamalloc_train_no_ot.parquet \
    --val-parquet artifacts/experiments/team_alloc_v3_time/val_predictions.parquet \
    --projections-data-root /home/daniel/projections-data \
    --out-dir artifacts/experiments/lgbm_rotalloc_final_v1 \
    --split-mode time --val-frac 0.2 --rot-threshold 8 --seed 7 \
    --use-expected-k --k-min 8 --cap-max 48 \
    --a 2.5 --k-max 12 --p-cutoff 0.2
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from projections.models.rotalloc import build_eligible_mask
from scripts.experiments.train_rotation_minutes_lgbm import _split_train_val_team_games


GROUP_COLS = ("game_id", "team_id")


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}")


def _compute_date_ranges(
    df: pd.DataFrame,
    *,
    split_mode: str,
    val_frac: float,
    seed: int,
    exclude_out: bool,
) -> dict[str, dict[str, str]]:
    work = df.copy()
    if exclude_out and "status" in work.columns:
        status = work["status"].astype(str).str.upper()
        work = work.loc[status != "OUT"].copy()

    train_df, val_df = _split_train_val_team_games(work, val_frac=val_frac, seed=seed, mode=split_mode)
    train_dates = pd.to_datetime(train_df["game_date"], errors="coerce")
    val_dates = pd.to_datetime(val_df["game_date"], errors="coerce")
    ranges = {
        "train": {
            "min": str(train_dates.min().date()) if not train_dates.empty else "",
            "max": str(train_dates.max().date()) if not train_dates.empty else "",
        },
        "val": {
            "min": str(val_dates.min().date()) if not val_dates.empty else "",
            "max": str(val_dates.max().date()) if not val_dates.empty else "",
        },
    }
    return ranges


def _select_best_config(summary: pd.DataFrame, *, guardrail: float) -> dict[str, float | int | None]:
    df = summary.copy()

    def _to_bool(val: object) -> bool:
        if isinstance(val, (bool, np.bool_)):
            return bool(val)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return False
        if isinstance(val, (int, np.integer)):
            return bool(int(val))
        if isinstance(val, str):
            return val.strip().lower() in {"1", "true", "yes", "y"}
        return False

    df["p_cutoff"] = df["p_cutoff"].where(~df["p_cutoff"].isna(), None)
    df.loc[df["p_cutoff"].astype(str).str.lower().isin({"none", "nan"}), "p_cutoff"] = None
    df["use_expected_k"] = df["use_expected_k"].apply(_to_bool)
    df["k_max"] = df["k_max"].astype(int)

    baseline_mask = (
        (df["a"] == 1.5)
        & (df["use_expected_k"])
        & (df["k_max"] == 11)
        & (df["p_cutoff"].isna())
    )
    if baseline_mask.any():
        baseline_mae_ge_10 = float(df.loc[baseline_mask, "mae_ge_10"].iloc[0])
    else:
        baseline_mae_ge_10 = float(df.iloc[0]["mae_ge_10"])

    eligible = df[df["mae_ge_10"] <= baseline_mae_ge_10 + float(guardrail)]
    if eligible.empty:
        eligible = df

    eligible = eligible.sort_values(["leak_p50", "leak_p90", "leak_p99", "mae_ge_10", "mae"])
    best = eligible.iloc[0].to_dict()
    best["baseline_mae_ge_10"] = baseline_mae_ge_10
    return best


def _compute_metrics(
    rows_scored: pd.DataFrame,
    teamgame: pd.DataFrame,
    *,
    allocator: dict[str, float | int | None],
) -> dict[str, float]:
    leak_col = "leak_rotalloc"
    if leak_col not in teamgame.columns:
        raise ValueError(f"Missing {leak_col} in teamgame_leak.parquet")

    leak = teamgame[leak_col]
    metrics = {
        "mae": float((rows_scored["rotalloc_pred_240"] - rows_scored["minutes_actual"]).abs().mean()),
        "mae_ge_10": float(
            (
                rows_scored.loc[rows_scored["minutes_actual"] >= 10, "rotalloc_pred_240"]
                - rows_scored.loc[rows_scored["minutes_actual"] >= 10, "minutes_actual"]
            )
            .abs()
            .mean()
        ),
        "leak_p50": float(leak.quantile(0.5)),
        "leak_p90": float(leak.quantile(0.9)),
        "leak_p99": float(leak.quantile(0.99)),
    }

    eligible_sizes = []
    for _, g in rows_scored.groupby(list(GROUP_COLS), sort=False):
        eligible = build_eligible_mask(
            g["p_rot"].to_numpy(dtype=float),
            g["mu"].to_numpy(dtype=float),
            np.ones(len(g), dtype=bool),
            a=float(allocator["a"]),
            p_cutoff=allocator.get("p_cutoff"),
            k_min=int(allocator["k_min"]),
            k_max=int(allocator["k_max"]),
            use_expected_k=bool(allocator["use_expected_k"]),
        )
        eligible_sizes.append(int(np.sum(eligible)))

    eligible_series = pd.Series(eligible_sizes)
    metrics["eligible_size_p50"] = float(eligible_series.quantile(0.5))
    metrics["eligible_size_p90"] = float(eligible_series.quantile(0.9))
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-parquet", type=Path, required=True)
    parser.add_argument("--val-parquet", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--projections-data-root", type=Path, required=True)
    parser.add_argument("--split-mode", type=str, choices=["random", "time"], default="time")
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--label-mode", type=str, choices=["threshold", "topk"], default="threshold")
    parser.add_argument("--rot-threshold", type=float, default=8.0)
    parser.add_argument("--rot-topk", type=int, default=9)
    parser.add_argument(
        "--exclude-out",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude status=='OUT' rows before splitting/training (default: True)",
    )
    parser.add_argument("--calibration", type=str, choices=["isotonic", "sigmoid", "none"], default="isotonic")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--a", type=float, default=1.5)
    parser.add_argument("--p-cutoff", type=float, default=None)
    parser.add_argument("--use-expected-k", action="store_true")
    parser.add_argument("--k-min", type=int, default=8)
    parser.add_argument("--k-max", type=int, default=11)
    parser.add_argument("--cap-max", type=float, default=48.0)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--mae-ge-10-guardrail", type=float, default=0.5)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = args.out_dir / "models"
    eval_dir = args.out_dir / "eval"
    sweep_dir = args.out_dir / "sweep"

    allocator = {
        "a": float(args.a),
        "p_cutoff": args.p_cutoff,
        "use_expected_k": bool(args.use_expected_k),
        "k_min": int(args.k_min),
        "k_max": int(args.k_max),
        "cap_max": float(args.cap_max),
    }

    train_cmd = [
        sys.executable,
        "-m",
        "scripts.experiments.train_rotation_minutes_lgbm",
        "--train-parquet",
        str(args.train_parquet),
        "--out-dir",
        str(models_dir),
        "--split-mode",
        str(args.split_mode),
        "--val-frac",
        str(args.val_frac),
        "--label-mode",
        str(args.label_mode),
        "--rot-threshold",
        str(args.rot_threshold),
        "--rot-topk",
        str(args.rot_topk),
        "--calibration",
        str(args.calibration),
        "--seed",
        str(args.seed),
    ]
    if args.exclude_out:
        train_cmd.append("--exclude-out")
    else:
        train_cmd.append("--no-exclude-out")
    _run(train_cmd)

    if args.sweep:
        sweep_cmd = [
            sys.executable,
            "-m",
            "scripts.experiments.sweep_rotalloc_leak",
            "--val-parquet",
            str(args.val_parquet),
            "--rotalloc-dir",
            str(models_dir),
            "--projections-data-root",
            str(args.projections_data_root),
            "--out-dir",
            str(sweep_dir),
            "--k-min",
            str(args.k_min),
            "--cap-max",
            str(args.cap_max),
            "--mae-ge-10-guardrail",
            str(args.mae_ge_10_guardrail),
        ]
        _run(sweep_cmd)
        summary = pd.read_csv(sweep_dir / "summary.csv")
        best = _select_best_config(summary, guardrail=float(args.mae_ge_10_guardrail))
        allocator.update(
            {
                "a": float(best["a"]),
                "p_cutoff": best.get("p_cutoff", None),
                "use_expected_k": bool(best["use_expected_k"]),
                "k_min": int(best["k_min"]),
                "k_max": int(best["k_max"]),
            }
        )
        print(f"Selected sweep config: {allocator}")

    eval_cmd = [
        sys.executable,
        "-m",
        "scripts.experiments.eval_rotalloc_vs_teamalloc",
        "--val-parquet",
        str(args.val_parquet),
        "--rotalloc-dir",
        str(models_dir),
        "--projections-data-root",
        str(args.projections_data_root),
        "--out-dir",
        str(eval_dir),
        "--a",
        str(allocator["a"]),
        "--cap-max",
        str(allocator["cap_max"]),
        "--k-min",
        str(allocator["k_min"]),
        "--k-max",
        str(allocator["k_max"]),
    ]
    if allocator.get("p_cutoff") is not None:
        eval_cmd.extend(["--p-cutoff", str(allocator["p_cutoff"])])
    if allocator.get("use_expected_k"):
        eval_cmd.append("--use-expected-k")
    _run(eval_cmd)

    rows_scored = pd.read_parquet(eval_dir / "rows_scored.parquet")
    teamgame = pd.read_parquet(eval_dir / "teamgame_leak.parquet")
    metrics = _compute_metrics(rows_scored, teamgame, allocator=allocator)

    train_df = pd.read_parquet(args.train_parquet)
    date_ranges = _compute_date_ranges(
        train_df,
        split_mode=str(args.split_mode),
        val_frac=float(args.val_frac),
        seed=int(args.seed),
        exclude_out=bool(args.exclude_out),
    )

    promote = {
        "rotalloc_dir": str(args.out_dir),
        "allocator": allocator,
        "train": {
            "split_mode": str(args.split_mode),
            "val_frac": float(args.val_frac),
            "label_mode": str(args.label_mode),
            "date_ranges": date_ranges,
            "rot_threshold": float(args.rot_threshold),
            "rot_topk": int(args.rot_topk),
            "calibration": str(args.calibration),
            "seed": int(args.seed),
        },
        "metrics": metrics,
    }
    promote_path = args.out_dir / "promote_config.json"
    promote_path.write_text(json.dumps(promote, indent=2), encoding="utf-8")
    print(f"Wrote: {promote_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
