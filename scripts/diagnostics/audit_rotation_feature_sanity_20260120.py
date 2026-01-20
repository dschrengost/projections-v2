#!/usr/bin/env python3
"""Audit rotation feature sanity for 2026-01-20.

This script diagnoses why the rotation SetTransformer minutes overlay is flattening
star minutes by examining starter-related features across all team-games.

Key checks:
- recent_start_pct_10: Should NOT be all zeros for team-games
- is_confirmed_starter: Should have some 1s once lineups are confirmed
- is_projected_starter: Should sum to ~5 per team-game
- prior_play_prob: Should have variance across players
- available_G/W/B: Should be team-level constants (1 unique value per team-game)

Usage:
    uv run python scripts/diagnostics/audit_rotation_feature_sanity_20260120.py

    # With custom paths:
    uv run python scripts/diagnostics/audit_rotation_feature_sanity_20260120.py \
        --rotation-features /path/to/features.parquet \
        --baseline-minutes /path/to/baseline/minutes.parquet \
        --final-minutes /path/to/final/minutes.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import numpy as np


def audit_team_game(df: pd.DataFrame, team_id: int) -> dict:
    """Compute sanity metrics for a single team-game."""
    grp = df[df["team_id"] == team_id].copy()
    n = len(grp)
    
    # is_projected_starter
    proj_starter_sum = grp["is_projected_starter"].sum() if "is_projected_starter" in grp.columns else 0
    proj_starter_mean = grp["is_projected_starter"].mean() if "is_projected_starter" in grp.columns else 0
    
    # is_confirmed_starter
    conf_starter_sum = grp["is_confirmed_starter"].sum() if "is_confirmed_starter" in grp.columns else 0
    conf_starter_mean = grp["is_confirmed_starter"].mean() if "is_confirmed_starter" in grp.columns else 0
    
    # recent_start_pct_10
    if "recent_start_pct_10" in grp.columns:
        rsp = grp["recent_start_pct_10"].fillna(0)
        rsp_zeros = int((rsp == 0).sum())
        rsp_zero_pct = rsp_zeros / n * 100 if n > 0 else 100
        rsp_min = float(rsp.min())
        rsp_median = float(rsp.median())
        rsp_max = float(rsp.max())
    else:
        rsp_zeros, rsp_zero_pct, rsp_min, rsp_median, rsp_max = n, 100, 0, 0, 0
    
    # prior_play_prob
    if "prior_play_prob" in grp.columns:
        ppp = grp["prior_play_prob"].fillna(0)
        ppp_min = float(ppp.min())
        ppp_median = float(ppp.median())
        ppp_max = float(ppp.max())
        ppp_std = float(ppp.std())
    else:
        ppp_min, ppp_median, ppp_max, ppp_std = 0, 0, 0, 0
    
    # roster_active_pre_tip
    if "roster_active_pre_tip" in grp.columns:
        rapt = grp["roster_active_pre_tip"].fillna(0)
        rapt_frac_1 = float((rapt == 1).sum() / n) if n > 0 else 0
    else:
        rapt_frac_1 = 0
    
    # available_* columns
    avail_unique = {}
    for col in ["available_G", "available_W", "available_B"]:
        if col in grp.columns:
            avail_unique[col] = len(grp[col].unique())
    
    # Flags
    flags = []
    if rsp_zero_pct >= 90:
        flags.append("recent_start_pct_10_all_zeros")
    if proj_starter_sum < 3 or proj_starter_sum > 7:
        flags.append(f"proj_starter_implausible({proj_starter_sum:.0f})")
    if ppp_std < 0.02:
        flags.append(f"prior_play_prob_near_constant(std={ppp_std:.3f})")
    
    return {
        "team_id": team_id,
        "n_players": n,
        "proj_starter_sum": proj_starter_sum,
        "proj_starter_mean": proj_starter_mean,
        "conf_starter_sum": conf_starter_sum,
        "conf_starter_mean": conf_starter_mean,
        "rsp_zeros": rsp_zeros,
        "rsp_zero_pct": rsp_zero_pct,
        "rsp_min": rsp_min,
        "rsp_median": rsp_median,
        "rsp_max": rsp_max,
        "ppp_min": ppp_min,
        "ppp_median": ppp_median,
        "ppp_max": ppp_max,
        "ppp_std": ppp_std,
        "rapt_frac_1": rapt_frac_1,
        "avail_unique": avail_unique,
        "flags": flags,
    }


def print_flagged_team_detail(rot_feat: pd.DataFrame, baseline_min: pd.DataFrame, team_id: int, n_top: int = 10):
    """Print detailed info for flagged team-game."""
    grp_rot = rot_feat[rot_feat["team_id"] == team_id].copy()
    grp_base = baseline_min[baseline_min["team_id"] == team_id].copy() if baseline_min is not None else pd.DataFrame()
    
    # Get minutes column
    mins_col = "minutes_p50" if "minutes_p50" in grp_base.columns else "p50" if "p50" in grp_base.columns else None
    
    # Get available starter columns
    starter_cols = ["is_projected_starter", "is_confirmed_starter", "recent_start_pct_10", "prior_play_prob"]
    available_cols = ["player_id"] + [c for c in starter_cols if c in grp_rot.columns]
    if "player_name" in grp_rot.columns:
        available_cols.insert(1, "player_name")
    
    if not grp_base.empty and mins_col:
        merged = grp_rot[available_cols].copy()
        # Add player_name from baseline if missing
        if "player_name" not in merged.columns and "player_name" in grp_base.columns:
            merged = merged.merge(grp_base[["player_id", "player_name", mins_col]], on="player_id", how="left")
        else:
            merged = merged.merge(grp_base[["player_id", mins_col]], on="player_id", how="left")
        merged = merged.rename(columns={mins_col: "baseline_p50"})
        if "baseline_p50" in merged.columns:
            merged = merged.sort_values("baseline_p50", ascending=False).head(n_top)
        else:
            merged = merged.head(n_top)
    else:
        merged = grp_rot[available_cols].head(n_top)
    
    print(f"\n  Top {n_top} players by baseline minutes for team_id={team_id}:")
    print(merged.to_string(index=False))



def main():
    parser = argparse.ArgumentParser(description="Audit rotation feature sanity for 2026-01-20")
    parser.add_argument(
        "--rotation-features",
        type=Path,
        default=Path("/home/daniel/projections-data/live/features_rotation_set_minutes_v1/2026-01-20/run=20260120T192500Z/features.parquet"),
        help="Path to rotation features parquet",
    )
    parser.add_argument(
        "--baseline-minutes",
        type=Path,
        default=Path("/home/daniel/projections-data/artifacts/minutes_v1/daily/2026-01-20/run=dev-baseline-1511/minutes.parquet"),
        help="Path to baseline-only minutes parquet",
    )
    parser.add_argument(
        "--final-minutes",
        type=Path,
        default=Path("/home/daniel/projections-data/artifacts/minutes_v1/daily/2026-01-20/run=20260120T192500Z/minutes.parquet"),
        help="Path to final minutes (with rotation overlay) parquet",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("/home/daniel/projections-data/labels/season=2025/boxscore_labels.parquet"),
        help="Path to historical labels parquet",
    )
    args = parser.parse_args()
    
    print("=" * 80)
    print("ROTATION FEATURE SANITY AUDIT — 2026-01-20")
    print("=" * 80)
    
    # Load data
    rot_feat = pd.read_parquet(args.rotation_features) if args.rotation_features.exists() else pd.DataFrame()
    baseline_min = pd.read_parquet(args.baseline_minutes) if args.baseline_minutes.exists() else pd.DataFrame()
    final_min = pd.read_parquet(args.final_minutes) if args.final_minutes.exists() else pd.DataFrame()
    labels = pd.read_parquet(args.labels) if args.labels.exists() else pd.DataFrame()
    
    print(f"\nData loaded:")
    print(f"  Rotation features: {len(rot_feat)} rows")
    print(f"  Baseline minutes:  {len(baseline_min)} rows")
    print(f"  Final minutes:     {len(final_min)} rows")
    print(f"  Historical labels: {len(labels)} rows")
    
    # Check historical labels for starter_flag issue
    if not labels.empty:
        print("\n" + "-" * 80)
        print("HISTORICAL LABELS — starter_flag CHECK")
        print("-" * 80)
        if "starter_flag" in labels.columns:
            sf = labels["starter_flag"].fillna(0)
            print(f"  starter_flag: zeros={int((sf==0).sum())}, ones={int((sf==1).sum())}")
        if "starter_flag_label" in labels.columns:
            sfl = labels["starter_flag_label"].fillna(0)
            print(f"  starter_flag_label: zeros={int((sfl==0).sum())}, ones={int((sfl==1).sum())}")
        # Check if starter_flag is all 1s (the bug)
        if "starter_flag" in labels.columns and (labels["starter_flag"] == 1).all():
            print("\n  *** BUG DETECTED: starter_flag is ALL 1s — this is wrong! ***")
            print("  *** recent_start_pct_10 recompute uses starter_flag instead of starter_flag_label ***")
    
    if rot_feat.empty:
        print("\nNo rotation features found. Exiting.")
        return
    
    # Audit each team-game
    print("\n" + "-" * 80)
    print("ALL TEAM-GAMES — Starter Feature Sanity")
    print("-" * 80)
    
    team_ids = sorted(rot_feat["team_id"].dropna().astype(int).unique())
    results = []
    flagged_teams = []
    
    for tid in team_ids:
        metrics = audit_team_game(rot_feat, tid)
        results.append(metrics)
        flag_str = ", ".join(metrics["flags"]) if metrics["flags"] else "OK"
        print(f"  team_id={tid}: n={metrics['n_players']}, proj_starter={metrics['proj_starter_sum']:.0f}, conf_starter={metrics['conf_starter_sum']:.0f}, rsp_zeros={metrics['rsp_zeros']}/{metrics['n_players']} ({metrics['rsp_zero_pct']:.0f}%), ppp_std={metrics['ppp_std']:.3f} — {flag_str}")
        if metrics["flags"]:
            flagged_teams.append(tid)
    
    # Summary
    print(f"\n  SUMMARY: {len(flagged_teams)}/{len(team_ids)} team-games FLAGGED")
    
    # Show flagged team details
    if flagged_teams:
        print("\n" + "-" * 80)
        print("FLAGGED TEAM-GAMES — Detailed Player Analysis")
        print("-" * 80)
        
        for tid in flagged_teams[:3]:  # Show first 3 flagged teams
            print_flagged_team_detail(rot_feat, baseline_min, tid)
    
    # Compare baseline vs final for top players
    print("\n" + "-" * 80)
    print("MINUTES IMPACT — Top 10 Baseline Players")
    print("-" * 80)
    
    if not baseline_min.empty and not final_min.empty:
        mins_col = "minutes_p50" if "minutes_p50" in baseline_min.columns else "p50"
        baseline_sorted = baseline_min.sort_values(mins_col, ascending=False).head(10)
        
        comparison = baseline_sorted[["player_id", "player_name", "team_id", mins_col]].copy()
        comparison = comparison.rename(columns={mins_col: "baseline_p50"})
        
        final_mins_col = "minutes_p50" if "minutes_p50" in final_min.columns else "p50"
        comparison = comparison.merge(
            final_min[["player_id", final_mins_col]].rename(columns={final_mins_col: "final_p50"}),
            on="player_id",
            how="left"
        )
        comparison["delta"] = comparison["final_p50"] - comparison["baseline_p50"]
        
        # Add rotation features
        if not rot_feat.empty:
            comparison = comparison.merge(
                rot_feat[["player_id", "is_projected_starter", "recent_start_pct_10"]],
                on="player_id",
                how="left"
            )
        
        print(comparison.to_string(index=False))
        
        avg_delta = comparison["delta"].mean()
        print(f"\n  Average delta for top 10: {avg_delta:.2f} minutes (negative = flattening)")
    
    print("\n" + "=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
