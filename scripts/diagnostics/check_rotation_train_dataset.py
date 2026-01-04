#!/usr/bin/env python3
"""Diagnostics for rotation_train TEAM-SET minutes datasets.

This script summarizes per-(game_id, team_id) coverage and label completeness:
  - n_players (rows in features)
  - label_sum / n_pos (from labels minutes)
  - stints_team_total (team_total_minutes_from_stints)
  - minutes_from_stints_sum (sum over player rows)
  - rotation_missing

It also buckets likely failure modes:
  (1) label_sum low AND stints_team_total near regulation -> label/join likely
  (2) label_sum low AND stints_team_total low -> scrape/incomplete stints likely
  (3) n_players small -> upstream roster/features coverage issue

Example:
  uv run python scripts/diagnostics/check_rotation_train_dataset.py \
    --dataset-dir /home/daniel/projections-data/training/datasets/rotation_train_v1_20260104
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from projections.rotation.set_model import zfill_game_id_series


def _quantiles(series: pd.Series) -> dict[str, float | None]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {"min": None, "p10": None, "p50": None, "mean": None, "p90": None, "max": None}
    return {
        "min": float(clean.min()),
        "p10": float(clean.quantile(0.1)),
        "p50": float(clean.quantile(0.5)),
        "mean": float(clean.mean()),
        "p90": float(clean.quantile(0.9)),
        "max": float(clean.max()),
    }


def _print_k(df: pd.DataFrame, *, title: str, k: int) -> None:
    if df.empty:
        print(f"\n[{title}] (empty)")
        return
    print(f"\n[{title}] (k={k})")
    cols = [
        "game_id_norm",
        "team_id",
        "n_players",
        "n_pos",
        "label_sum",
        "stints_team_total",
        "minutes_from_stints_sum",
        "rotation_missing",
        "stints_gap",
        "label_gap",
    ]
    present = [c for c in cols if c in df.columns]
    print(df.loc[:, present].head(k).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--label-col", type=str, default="minutes")
    parser.add_argument("--low-label-threshold", type=float, default=200.0)
    parser.add_argument("--reg-lo", type=float, default=238.0)
    parser.add_argument("--reg-hi", type=float, default=242.0)
    parser.add_argument("--min-stints-team-total", type=float, default=200.0)
    parser.add_argument("--gap-tol", type=float, default=2.0)
    parser.add_argument("--small-n-players", type=int, default=8)
    parser.add_argument("--bottom-k", type=int, default=20)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    features_path = dataset_dir / "features.parquet"
    labels_path = dataset_dir / "labels.parquet"
    if not features_path.exists():
        raise FileNotFoundError(f"Missing features.parquet at {features_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels.parquet at {labels_path}")

    features = pd.read_parquet(features_path)
    labels = pd.read_parquet(labels_path)

    for col in ["game_id", "team_id", "player_id"]:
        if col not in features.columns:
            raise ValueError(f"features.parquet missing required column: {col}")
        if col not in labels.columns:
            raise ValueError(f"labels.parquet missing required column: {col}")

    label_col = args.label_col
    if label_col not in labels.columns:
        raise ValueError(f"labels.parquet missing label column: {label_col}")

    features = features.copy()
    labels = labels.copy()
    features["game_id_norm"] = zfill_game_id_series(features["game_id"])
    labels["game_id_norm"] = zfill_game_id_series(labels["game_id"])
    features["team_id"] = pd.to_numeric(features["team_id"], errors="coerce").astype("Int64")
    labels["team_id"] = pd.to_numeric(labels["team_id"], errors="coerce").astype("Int64")

    group_key = ["game_id_norm", "team_id"]

    n_players = features.groupby(group_key, sort=False).size().rename("n_players")
    minutes_from_stints_sum = (
        pd.to_numeric(features.get("minutes_from_stints", pd.Series([np.nan] * len(features))), errors="coerce")
        .fillna(0.0)
        .groupby([features["game_id_norm"], features["team_id"]], sort=False)
        .sum()
        .rename("minutes_from_stints_sum")
    )
    stints_team_total = (
        pd.to_numeric(features.get("team_total_minutes_from_stints", pd.Series([np.nan] * len(features))), errors="coerce")
        .groupby([features["game_id_norm"], features["team_id"]], sort=False)
        .mean()
        .rename("stints_team_total")
    )
    rotation_missing = (
        pd.to_numeric(features.get("rotation_missing", pd.Series([np.nan] * len(features))), errors="coerce")
        .groupby([features["game_id_norm"], features["team_id"]], sort=False)
        .max()
        .rename("rotation_missing")
    )

    label_minutes = pd.to_numeric(labels[label_col], errors="coerce").fillna(0.0)
    label_sum = label_minutes.groupby([labels["game_id_norm"], labels["team_id"]], sort=False).sum().rename("label_sum")
    n_pos = (label_minutes > 0).groupby([labels["game_id_norm"], labels["team_id"]], sort=False).sum().rename("n_pos")

    summary = pd.concat([n_players, n_pos, label_sum, stints_team_total, minutes_from_stints_sum, rotation_missing], axis=1).reset_index()
    summary["stints_gap"] = (summary["stints_team_total"] - summary["minutes_from_stints_sum"]).abs()
    summary["label_gap"] = (summary["stints_team_total"] - summary["label_sum"]).abs()

    print("[rotation_train] Dataset summary")
    print(f"  dataset_dir: {dataset_dir}")
    print(f"  rows: features={len(features):,} labels={len(labels):,}")
    print(f"  team_games: {len(summary):,}")
    print("  n_players:", _quantiles(summary["n_players"]))
    print("  stints_team_total:", _quantiles(summary["stints_team_total"]))
    print("  minutes_from_stints_sum:", _quantiles(summary["minutes_from_stints_sum"]))
    print("  label_sum:", _quantiles(summary["label_sum"]))

    low_label = summary["label_sum"] < float(args.low_label_threshold)
    near_reg = summary["stints_team_total"].between(float(args.reg_lo), float(args.reg_hi), inclusive="both")
    stints_low = summary["stints_team_total"] < float(args.min_stints_team_total)
    small_n = summary["n_players"] < int(args.small_n_players)

    print("\n[rotation_train] Case counts")
    print(f"  (1) label_sum<{args.low_label_threshold:g} & stints_team_total in [{args.reg_lo:g},{args.reg_hi:g}]: {int((low_label & near_reg).sum()):,}")
    print(f"  (2) label_sum<{args.low_label_threshold:g} & stints_team_total<{args.min_stints_team_total:g}: {int((low_label & stints_low).sum()):,}")
    print(f"  (3) n_players<{args.small_n_players:d}: {int(small_n.sum()):,}")
    if "stints_gap" in summary.columns:
        print(f"  coverage_gap>{args.gap_tol:g}: {int((summary['stints_gap'] > float(args.gap_tol)).sum()):,}")

    worst_label = summary.sort_values(["label_sum", "n_players"], ascending=[True, True])
    worst_coverage = summary.sort_values(["stints_gap", "label_sum"], ascending=[False, True])
    worst_label_gap = summary.sort_values(["label_gap", "label_sum"], ascending=[False, True])

    k = int(args.bottom_k)
    _print_k(worst_label, title="Bottom label_sum", k=k)
    _print_k(worst_coverage, title="Top stints_gap (missing coverage)", k=k)
    _print_k(worst_label_gap, title="Top label_gap vs stints", k=k)


if __name__ == "__main__":
    main()

