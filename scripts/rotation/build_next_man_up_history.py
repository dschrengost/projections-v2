#!/usr/bin/env python3
"""Build a full-history next-man-up labeling artifact from a GTv2 dataset.

Outputs:
  <out_dir>/
    - labeled_rows.parquet
    - archetype_summary.csv
    - season_archetype_summary.csv
    - top_examples.csv
    - summary.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from projections import paths


def _utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


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


def _hist_start_rate(df: pd.DataFrame) -> pd.Series:
    cols = [
        c
        for c in (
            "recent_start_pct_10",
            "started_proxy_rate_prior_10",
            "started_proxy_rate_prior_20",
        )
        if c in df.columns
    ]
    if not cols:
        return pd.Series(np.zeros(len(df), dtype=float), index=df.index, dtype=float)
    vals = [pd.to_numeric(df[c], errors="coerce").fillna(0.0) for c in cols]
    out = vals[0].copy()
    for v in vals[1:]:
        out = np.maximum(out, v)
    return pd.Series(out, index=df.index, dtype=float)


def build_next_man_up_labels(
    features_df: pd.DataFrame,
    labels_minutes_df: pd.DataFrame,
    *,
    sparse_prior_play_prob_max: float,
    sparse_prior_minutes_max: float,
    surprise_actual_min: float,
    entrant_actual_min: float,
    core_actual_min: float,
    starter_actual_min: float,
    starter_hist_start_rate_max: float,
) -> pd.DataFrame:
    key_cols = ["game_id", "game_date", "team_id", "player_id"]
    merged = features_df.merge(
        labels_minutes_df.loc[:, [*key_cols, "minutes_label", "starter_flag_label"]],
        on=key_cols,
        how="inner",
        validate="one_to_one",
    ).copy()
    merged["game_date"] = pd.to_datetime(merged["game_date"], errors="coerce")
    merged["actual_minutes"] = pd.to_numeric(merged["minutes_label"], errors="coerce").fillna(0.0)
    merged["prior_minutes"] = pd.to_numeric(merged["minutes_from_stints_prior_20"], errors="coerce").fillna(0.0)
    merged["prior_play_prob"] = pd.to_numeric(merged["prior_play_prob"], errors="coerce").fillna(0.0)
    merged["hist_start_rate"] = _hist_start_rate(merged)
    merged["actual_starter"] = pd.to_numeric(merged["starter_flag_label"], errors="coerce").fillna(0.0).ge(0.5)
    merged["announced_starter"] = (
        pd.to_numeric(merged.get("lineup_starter_announced", 0.0), errors="coerce").fillna(0.0).ge(0.5)
    )
    merged["propless"] = pd.to_numeric(merged.get("an_has_any_props", 0.0), errors="coerce").fillna(0.0).lt(0.5)
    merged["implied_minutes"] = pd.to_numeric(merged.get("an_implied_minutes", 0.0), errors="coerce").fillna(0.0)
    merged["sparse_prior_signal"] = (
        merged["prior_play_prob"].le(float(sparse_prior_play_prob_max))
        | merged["prior_minutes"].le(float(sparse_prior_minutes_max))
    )
    merged["starter_promotion_signal"] = (
        merged["actual_starter"]
        & merged["prior_minutes"].le(float(sparse_prior_minutes_max))
        & merged["hist_start_rate"].le(float(starter_hist_start_rate_max))
    )
    merged["minutes_delta"] = merged["actual_minutes"] - merged["prior_minutes"]

    merged["emergency_starter"] = (
        merged["sparse_prior_signal"]
        & merged["actual_starter"]
        & merged["actual_minutes"].ge(float(starter_actual_min))
    )
    merged["bench_rotation_entrant"] = (
        merged["sparse_prior_signal"]
        & (~merged["actual_starter"])
        & merged["actual_minutes"].ge(float(entrant_actual_min))
        & merged["actual_minutes"].lt(float(core_actual_min))
    )
    merged["bench_core_riser"] = (
        merged["sparse_prior_signal"]
        & (~merged["actual_starter"])
        & merged["actual_minutes"].ge(float(core_actual_min))
    )
    merged["sparse_active_surprise"] = (
        merged["sparse_prior_signal"]
        & merged["actual_minutes"].ge(float(surprise_actual_min))
        & merged["actual_minutes"].lt(float(entrant_actual_min))
    )

    primary = np.full(len(merged), "none", dtype=object)
    primary = np.where(merged["emergency_starter"], "emergency_starter", primary)
    primary = np.where(
        (~merged["emergency_starter"]) & merged["bench_core_riser"],
        "bench_core_riser",
        primary,
    )
    primary = np.where(
        (~merged["emergency_starter"]) & (~merged["bench_core_riser"]) & merged["bench_rotation_entrant"],
        "bench_rotation_entrant",
        primary,
    )
    primary = np.where(
        (primary == "none") & merged["sparse_active_surprise"],
        "sparse_active_surprise",
        primary,
    )
    merged["primary_archetype"] = pd.Series(primary, index=merged.index, dtype="string")
    return merged


def _summary_table(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["propless_bucket"] = np.where(work["propless"], "propless", "with_props")
    grouped = (
        work.loc[work["primary_archetype"].ne("none")]
        .groupby(["primary_archetype", "propless_bucket"], dropna=False)
        .agg(
            n=("player_id", "size"),
            seasons=("season", "nunique"),
            actual_minutes_mean=("actual_minutes", "mean"),
            prior_minutes_mean=("prior_minutes", "mean"),
            minutes_delta_mean=("minutes_delta", "mean"),
            actual_starter_rate=("actual_starter", "mean"),
            announced_starter_rate=("announced_starter", "mean"),
            implied_minutes_mean=("implied_minutes", "mean"),
        )
        .reset_index()
        .sort_values(["primary_archetype", "propless_bucket"])
    )
    return grouped


def _season_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    work = df.loc[df["primary_archetype"].ne("none")].copy()
    grouped = (
        work.groupby(["season", "primary_archetype"], dropna=False)
        .agg(
            n=("player_id", "size"),
            propless_rate=("propless", "mean"),
            actual_minutes_mean=("actual_minutes", "mean"),
            prior_minutes_mean=("prior_minutes", "mean"),
            minutes_delta_mean=("minutes_delta", "mean"),
        )
        .reset_index()
        .sort_values(["season", "primary_archetype"])
    )
    return grouped


def _top_examples_table(df: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
    cols = [
        "season",
        "game_date",
        "game_id",
        "team_id",
        "team_tricode",
        "player_id",
        "player_name",
        "pos_bucket",
        "archetype",
        "primary_archetype",
        "propless",
        "actual_starter",
        "announced_starter",
        "prior_play_prob",
        "prior_minutes",
        "hist_start_rate",
        "implied_minutes",
        "actual_minutes",
        "minutes_delta",
    ]
    available = [c for c in cols if c in df.columns]
    top = (
        df.loc[df["primary_archetype"].ne("none"), available]
        .sort_values(["minutes_delta", "actual_minutes"], ascending=[False, False])
        .head(int(top_n))
        .reset_index(drop=True)
    )
    return top


@dataclass(frozen=True)
class BuildArgs:
    dataset_dir: Path
    out_dir: Path
    sparse_prior_play_prob_max: float
    sparse_prior_minutes_max: float
    surprise_actual_min: float
    entrant_actual_min: float
    core_actual_min: float
    starter_actual_min: float
    starter_hist_start_rate_max: float
    top_n: int


def parse_args() -> BuildArgs:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--sparse-prior-play-prob-max", type=float, default=0.50)
    parser.add_argument("--sparse-prior-minutes-max", type=float, default=12.0)
    parser.add_argument("--surprise-actual-min", type=float, default=8.0)
    parser.add_argument("--entrant-actual-min", type=float, default=16.0)
    parser.add_argument("--core-actual-min", type=float, default=24.0)
    parser.add_argument("--starter-actual-min", type=float, default=20.0)
    parser.add_argument("--starter-hist-start-rate-max", type=float, default=0.20)
    parser.add_argument("--top-n", type=int, default=100)
    ns = parser.parse_args()

    dataset_dir = _resolve_dataset_dir(ns.dataset_dir)
    out_dir = (
        Path(ns.out_dir).expanduser().resolve()
        if ns.out_dir
        else (paths.get_data_root() / "training" / "runs" / f"next_man_up_history_{_utc_now_compact()}").resolve()
    )
    return BuildArgs(
        dataset_dir=dataset_dir,
        out_dir=out_dir,
        sparse_prior_play_prob_max=float(ns.sparse_prior_play_prob_max),
        sparse_prior_minutes_max=float(ns.sparse_prior_minutes_max),
        surprise_actual_min=float(ns.surprise_actual_min),
        entrant_actual_min=float(ns.entrant_actual_min),
        core_actual_min=float(ns.core_actual_min),
        starter_actual_min=float(ns.starter_actual_min),
        starter_hist_start_rate_max=float(ns.starter_hist_start_rate_max),
        top_n=int(ns.top_n),
    )


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = [
        "game_id",
        "game_date",
        "season",
        "team_id",
        "player_id",
        "player_name",
        "team_tricode",
        "archetype",
        "pos_bucket",
        "prior_play_prob",
        "minutes_from_stints_prior_20",
        "an_has_any_props",
        "an_implied_minutes",
        "recent_start_pct_10",
        "started_proxy_rate_prior_10",
        "started_proxy_rate_prior_20",
        "lineup_starter_announced",
    ]
    features = pd.read_parquet(args.dataset_dir / "features.parquet", columns=feature_cols)
    labels_minutes = pd.read_parquet(args.dataset_dir / "labels_minutes.parquet")

    labeled = build_next_man_up_labels(
        features,
        labels_minutes,
        sparse_prior_play_prob_max=args.sparse_prior_play_prob_max,
        sparse_prior_minutes_max=args.sparse_prior_minutes_max,
        surprise_actual_min=args.surprise_actual_min,
        entrant_actual_min=args.entrant_actual_min,
        core_actual_min=args.core_actual_min,
        starter_actual_min=args.starter_actual_min,
        starter_hist_start_rate_max=args.starter_hist_start_rate_max,
    )

    archetype_summary = _summary_table(labeled)
    season_summary = _season_summary_table(labeled)
    top_examples = _top_examples_table(labeled, top_n=args.top_n)

    labeled.to_parquet(args.out_dir / "labeled_rows.parquet", index=False)
    archetype_summary.to_csv(args.out_dir / "archetype_summary.csv", index=False)
    season_summary.to_csv(args.out_dir / "season_archetype_summary.csv", index=False)
    top_examples.to_csv(args.out_dir / "top_examples.csv", index=False)

    next_man_up_rows = labeled.loc[labeled["primary_archetype"].ne("none")]
    summary = {
        "dataset_dir": str(args.dataset_dir),
        "out_dir": str(args.out_dir),
        "thresholds": {
            "sparse_prior_play_prob_max": args.sparse_prior_play_prob_max,
            "sparse_prior_minutes_max": args.sparse_prior_minutes_max,
            "surprise_actual_min": args.surprise_actual_min,
            "entrant_actual_min": args.entrant_actual_min,
            "core_actual_min": args.core_actual_min,
            "starter_actual_min": args.starter_actual_min,
            "starter_hist_start_rate_max": args.starter_hist_start_rate_max,
        },
        "counts": {
            "rows_total": int(len(labeled)),
            "sparse_prior_rows": int(labeled["sparse_prior_signal"].sum()),
            "next_man_up_rows": int(len(next_man_up_rows)),
            "propless_next_man_up_rows": int((next_man_up_rows["propless"]).sum()),
        },
        "primary_archetype_counts": {
            str(k): int(v)
            for k, v in labeled["primary_archetype"].value_counts(dropna=False).sort_index().items()
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"saved_labeled={args.out_dir / 'labeled_rows.parquet'}")
    print(f"saved_summary={args.out_dir / 'archetype_summary.csv'}")
    print(f"saved_season_summary={args.out_dir / 'season_archetype_summary.csv'}")
    print(f"saved_top_examples={args.out_dir / 'top_examples.csv'}")


if __name__ == "__main__":
    main()
