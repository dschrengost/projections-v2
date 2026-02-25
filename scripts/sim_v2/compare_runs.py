#!/usr/bin/env python
"""Compare two sim runs to identify differences in per-player projections.

Usage:
    uv run python -m scripts.sim_v2.compare_runs \
        --date 2026-01-08 \
        --run-a 20260108T182500Z \
        --run-b ablation_rates_noise_20260108
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from projections import paths


def _pick_shared_column(df_a: pd.DataFrame, df_b: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return first candidate present in both DataFrames."""
    cols_a = set(df_a.columns)
    cols_b = set(df_b.columns)
    for col in candidates:
        if col in cols_a and col in cols_b:
            return col
    return None


def load_projections(data_root: Path, game_date: str, run_id: str) -> pd.DataFrame:
    """Load projections.parquet for a sim run."""
    path = (
        data_root
        / "artifacts"
        / "sim_v2"
        / "worlds_fpts_v2"
        / f"game_date={game_date}"
        / f"run={run_id}"
        / "projections.parquet"
    )
    if not path.exists():
        raise FileNotFoundError(f"Projections not found: {path}")
    return pd.read_parquet(path)


def load_worlds_matrix(data_root: Path, game_date: str, run_id: str) -> pd.DataFrame:
    """Load worlds_matrix.parquet for a sim run."""
    path = (
        data_root
        / "artifacts"
        / "sim_v2"
        / "worlds_fpts_v2"
        / f"game_date={game_date}"
        / f"run={run_id}"
        / "worlds_matrix.parquet"
    )
    if not path.exists():
        raise FileNotFoundError(f"Worlds matrix not found: {path}")
    return pd.read_parquet(path)


def compute_per_player_stats(worlds: pd.DataFrame) -> pd.DataFrame:
    """Compute per-player stats from worlds matrix.

    Expects columns: player_id, world_id, dk_fpts (or similar).
    """
    # Identify the fpts column
    fpts_col = None
    for col in ["dk_fpts", "fpts", "dk_fpts_total"]:
        if col in worlds.columns:
            fpts_col = col
            break
    if fpts_col is None:
        raise ValueError(f"No fpts column found. Columns: {list(worlds.columns)}")

    stats = worlds.groupby("player_id")[fpts_col].agg(
        mean="mean",
        std="std",
        p50=lambda x: np.percentile(x, 50),
        p95=lambda x: np.percentile(x, 95),
        p99=lambda x: np.percentile(x, 99),
    ).reset_index()
    return stats


def main():
    parser = argparse.ArgumentParser(description="Compare two sim runs")
    parser.add_argument("--date", required=True, help="Game date YYYY-MM-DD")
    parser.add_argument("--run-a", required=True, help="First run_id (baseline)")
    parser.add_argument("--run-b", required=True, help="Second run_id (ablation)")
    parser.add_argument("--data-root", default=None, help="Data root path")
    args = parser.parse_args()

    data_root = Path(args.data_root) if args.data_root else paths.get_data_root()

    print("=" * 70)
    print(f"COMPARING SIM RUNS FOR {args.date}")
    print(f"  Run A (baseline): {args.run_a}")
    print(f"  Run B (ablation): {args.run_b}")
    print("=" * 70)

    # Load projections
    proj_a = load_projections(data_root, args.date, args.run_a)
    proj_b = load_projections(data_root, args.date, args.run_b)

    # Development default: compare unconditional (DNP=0) summaries when available.
    mean_col = _pick_shared_column(
        proj_a,
        proj_b,
        ["dk_fpts_mean_uncond", "sim_dk_fpts_mean_uncond", "fpts_sim_uncond_mean", "dk_fpts_mean", "sim_dk_fpts_mean", "fpts_sim_cond_mean"],
    )
    p50_col = _pick_shared_column(
        proj_a,
        proj_b,
        ["dk_fpts_p50_uncond", "sim_dk_fpts_p50_uncond", "fpts_sim_uncond_p50", "dk_fpts_p50", "sim_dk_fpts_p50", "fpts_sim_cond_p50"],
    )
    p95_col = _pick_shared_column(
        proj_a,
        proj_b,
        ["dk_fpts_p95_uncond", "sim_dk_fpts_p95_uncond", "fpts_sim_uncond_p95", "dk_fpts_p95", "sim_dk_fpts_p95", "fpts_sim_cond_p95"],
    )
    std_col = _pick_shared_column(
        proj_a,
        proj_b,
        ["dk_fpts_std_uncond", "sim_dk_fpts_std_uncond", "fpts_sim_uncond_std", "dk_fpts_std", "sim_dk_fpts_std", "fpts_sim_cond_std"],
    )
    missing_metric_cols = [name for name, col in {"mean": mean_col, "p50": p50_col, "p95": p95_col, "std": std_col}.items() if col is None]
    if missing_metric_cols:
        raise ValueError(f"Missing comparable projection columns for: {', '.join(missing_metric_cols)}")

    proj_a_eval = (
        proj_a[["player_id", mean_col, p50_col, p95_col, std_col]]
        .rename(columns={mean_col: "fpts_mean_eval", p50_col: "fpts_p50_eval", p95_col: "fpts_p95_eval", std_col: "fpts_std_eval"})
        .copy()
    )
    proj_b_eval = (
        proj_b[["player_id", mean_col, p50_col, p95_col, std_col]]
        .rename(columns={mean_col: "fpts_mean_eval", p50_col: "fpts_p50_eval", p95_col: "fpts_p95_eval", std_col: "fpts_std_eval"})
        .copy()
    )

    # Verify input run_ids match
    print("\n--- INPUT VERIFICATION ---")
    a_minutes = proj_a["minutes_run_id"].iloc[0] if "minutes_run_id" in proj_a.columns else "N/A"
    a_rates = proj_a["rates_run_id"].iloc[0] if "rates_run_id" in proj_a.columns else "N/A"
    b_minutes = proj_b["minutes_run_id"].iloc[0] if "minutes_run_id" in proj_b.columns else "N/A"
    b_rates = proj_b["rates_run_id"].iloc[0] if "rates_run_id" in proj_b.columns else "N/A"
    a_profile = proj_a["sim_profile"].iloc[0] if "sim_profile" in proj_a.columns else "N/A"
    b_profile = proj_b["sim_profile"].iloc[0] if "sim_profile" in proj_b.columns else "N/A"

    print(f"Run A: profile={a_profile} minutes_run_id={a_minutes} rates_run_id={a_rates}")
    print(f"Run B: profile={b_profile} minutes_run_id={b_minutes} rates_run_id={b_rates}")
    print(
        "Eval columns (uncond-preferred): "
        f"mean={mean_col} p50={p50_col} p95={p95_col} std={std_col}"
    )

    inputs_match = (a_minutes == b_minutes) and (a_rates == b_rates)
    if not inputs_match:
        print("WARNING: Input run_ids DO NOT MATCH - comparison may be invalid!")
    else:
        print("OK: Input run_ids match")

    # Task 1: Compare MEANS
    print("\n--- TASK 1: MEAN COMPARISON ---")
    print(f"\nMax fpts mean ({mean_col}):")
    print(f"  Run A: {proj_a_eval['fpts_mean_eval'].max():.4f}")
    print(f"  Run B: {proj_b_eval['fpts_mean_eval'].max():.4f}")

    print(f"\nTop-10 players by fpts mean ({mean_col}) (Run A):")
    top_a = proj_a_eval.nlargest(10, "fpts_mean_eval")[["player_id", "fpts_mean_eval", "fpts_p50_eval", "fpts_p95_eval", "fpts_std_eval"]]
    print(top_a.to_string(index=False))

    print(f"\nTop-10 players by fpts mean ({mean_col}) (Run B):")
    top_b = proj_b_eval.nlargest(10, "fpts_mean_eval")[["player_id", "fpts_mean_eval", "fpts_p50_eval", "fpts_p95_eval", "fpts_std_eval"]]
    print(top_b.to_string(index=False))

    # Join and compute differences
    merged = proj_a_eval[["player_id", "fpts_mean_eval", "fpts_p50_eval", "fpts_p95_eval", "fpts_std_eval"]].merge(
        proj_b_eval[["player_id", "fpts_mean_eval", "fpts_p50_eval", "fpts_p95_eval", "fpts_std_eval"]],
        on="player_id",
        suffixes=("_a", "_b"),
        how="inner",
    )
    merged["mean_diff"] = merged["fpts_mean_eval_b"] - merged["fpts_mean_eval_a"]
    merged["p95_diff"] = merged["fpts_p95_eval_b"] - merged["fpts_p95_eval_a"]
    merged["std_diff"] = merged["fpts_std_eval_b"] - merged["fpts_std_eval_a"]
    merged["mean_abs_diff"] = merged["mean_diff"].abs()

    print(f"\nJoined {len(merged)} players")
    print(f"Max absolute difference in mean: {merged['mean_abs_diff'].max():.4f}")
    print(f"Mean of mean differences: {merged['mean_diff'].mean():.4f}")
    print(f"Players with mean diff > 0.1: {(merged['mean_abs_diff'] > 0.1).sum()}")
    print(f"Players with mean diff > 0.5: {(merged['mean_abs_diff'] > 0.5).sum()}")
    print(f"Players with mean diff > 1.0: {(merged['mean_abs_diff'] > 1.0).sum()}")

    # Task 2: P95 comparison
    print("\n--- TASK 2: P95 COMPARISON ---")
    print(f"\nTop-10 by p95 ({p95_col}) (Run A):")
    top_p95_a = proj_a_eval.nlargest(10, "fpts_p95_eval")[["player_id", "fpts_mean_eval", "fpts_p95_eval"]]
    print(top_p95_a.to_string(index=False))

    print(f"\nTop-10 by p95 ({p95_col}) (Run B):")
    top_p95_b = proj_b_eval.nlargest(10, "fpts_p95_eval")[["player_id", "fpts_mean_eval", "fpts_p95_eval"]]
    print(top_p95_b.to_string(index=False))

    # Top-10 by upside (p95 - mean)
    proj_a_eval["upside"] = proj_a_eval["fpts_p95_eval"] - proj_a_eval["fpts_mean_eval"]
    proj_b_eval["upside"] = proj_b_eval["fpts_p95_eval"] - proj_b_eval["fpts_mean_eval"]

    print("\nTop-10 by upside (p95 - mean) (Run A):")
    top_up_a = proj_a_eval.nlargest(10, "upside")[["player_id", "fpts_mean_eval", "fpts_p95_eval", "upside"]]
    print(top_up_a.to_string(index=False))

    print("\nTop-10 by upside (p95 - mean) (Run B):")
    top_up_b = proj_b_eval.nlargest(10, "upside")[["player_id", "fpts_mean_eval", "fpts_p95_eval", "upside"]]
    print(top_up_b.to_string(index=False))

    # Single top-mean player detailed stats
    top_player_a = proj_a_eval.loc[proj_a_eval["fpts_mean_eval"].idxmax(), "player_id"]
    print(f"\nTop-mean player from Run A (player_id={top_player_a}):")
    row_a = proj_a_eval[proj_a_eval["player_id"] == top_player_a].iloc[0]
    row_b = proj_b_eval[proj_b_eval["player_id"] == top_player_a].iloc[0] if top_player_a in proj_b_eval["player_id"].values else None
    print(f"  Run A: mean={row_a['fpts_mean_eval']:.2f} p50={row_a['fpts_p50_eval']:.2f} p95={row_a['fpts_p95_eval']:.2f} std={row_a['fpts_std_eval']:.2f}")
    if row_b is not None:
        print(f"  Run B: mean={row_b['fpts_mean_eval']:.2f} p50={row_b['fpts_p50_eval']:.2f} p95={row_b['fpts_p95_eval']:.2f} std={row_b['fpts_std_eval']:.2f}")

    # Summary stats
    print("\n--- SUMMARY STATISTICS ---")
    print(f"Mean of means (A): {proj_a_eval['fpts_mean_eval'].mean():.4f}")
    print(f"Mean of means (B): {proj_b_eval['fpts_mean_eval'].mean():.4f}")
    print(f"Mean of stds (A): {proj_a_eval['fpts_std_eval'].mean():.4f}")
    print(f"Mean of stds (B): {proj_b_eval['fpts_std_eval'].mean():.4f}")
    print(f"Mean of p95-mean (A): {proj_a_eval['upside'].mean():.4f}")
    print(f"Mean of p95-mean (B): {proj_b_eval['upside'].mean():.4f}")

    # Detailed diff table for top players
    print("\n--- TOP-10 PLAYERS DETAILED DIFF ---")
    top_ids = proj_a_eval.nlargest(10, "fpts_mean_eval")["player_id"].tolist()
    diff_table = merged[merged["player_id"].isin(top_ids)].copy()
    diff_table = diff_table.sort_values("fpts_mean_eval_a", ascending=False)
    print(diff_table[["player_id", "fpts_mean_eval_a", "fpts_mean_eval_b", "mean_diff", "fpts_std_eval_a", "fpts_std_eval_b", "std_diff"]].to_string(index=False))

    # Stat-level breakdown (if columns available)
    stat_cols = ["pts_mean", "reb_mean", "ast_mean", "stl_mean", "blk_mean", "tov_mean"]
    available_stats = [c for c in stat_cols if c in proj_a.columns and c in proj_b.columns]
    if available_stats:
        print("\n--- STAT-LEVEL MEAN COMPARISON (Top player) ---")
        top_id = proj_a_eval.loc[proj_a_eval["fpts_mean_eval"].idxmax(), "player_id"]
        row_a = proj_a[proj_a["player_id"] == top_id].iloc[0]
        row_b = proj_b[proj_b["player_id"] == top_id].iloc[0]
        for stat in available_stats:
            diff = row_b[stat] - row_a[stat]
            print(f"  {stat}: A={row_a[stat]:.2f} B={row_b[stat]:.2f} diff={diff:.2f}")

    print("\n" + "=" * 70)
    if merged["mean_abs_diff"].max() < 0.01:
        print("CONCLUSION: Per-player means are IDENTICAL (max diff < 0.01)")
        print("The observed difference is in VARIANCE/TAILS only.")
    elif merged["mean_abs_diff"].max() < 1.0:
        print("CONCLUSION: Per-player means have SMALL differences (max diff < 1.0)")
        print("This could be due to different RNG seeds or minor floating point differences.")
    else:
        print("CONCLUSION: Per-player means DIFFER SIGNIFICANTLY (max diff >= 1.0)")
        print("Root cause: Legacy noise is MULTIPLICATIVE (scale = k * mu_stat) with heavy-tailed")
        print("student-t distribution, while rates_noise is ADDITIVE with fixed sigma.")
        print("")
        print("The multiplicative noise + heavy tails + clip-at-0 creates upward mean bias for studs.")
        print("This is by design in legacy - studs have genuinely positive-skewed fantasy outcomes.")
        print("")
        print("RECOMMENDATIONS:")
        print("  1. If you want rates_noise to match legacy means: multiply sigma by (mu_stat / baseline)")
        print("     to restore proportional scaling, OR add a mean adjustment term.")
        print("  2. If the lower means are acceptable: keep rates_noise but accept tighter distributions.")
        print("  3. Hybrid: use rates_noise for correlation structure but add student-t tails on TOTAL fpts.")
    print("=" * 70)


if __name__ == "__main__":
    main()
