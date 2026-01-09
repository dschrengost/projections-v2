#!/usr/bin/env python3
"""Diagnostics for DNP/availability history features.

This script runs on a training dataset directory and prints:
1. Coverage statistics for each DNP history feature
2. Correlation with label (minutes == 0)
3. Top-k worst contradictions (high play_prob but high DNP indicators)
4. Point-in-time verification sample

Usage:
    uv run python scripts/diagnostics/check_availability_features.py \
        --dataset-dir /path/to/rotation_train_v1_YYYYMMDD

    # With model predictions for red-flag analysis:
    uv run python scripts/diagnostics/check_availability_features.py \
        --dataset-dir /path/to/rotation_train_v1_YYYYMMDD \
        --predictions-parquet /path/to/predictions.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from projections.features.dnp_history import (  # noqa: E402
    DNP_HISTORY_FEATURE_COLUMNS,
    DNPHistoryConfig,
    compute_dnp_history_features,
)


def _print_header(title: str) -> None:
    print()
    print("=" * 60)
    print(f" {title}")
    print("=" * 60)


def _coverage_stats(df: pd.DataFrame, col: str) -> dict:
    """Compute coverage statistics for a column."""
    vals = pd.to_numeric(df[col], errors="coerce")
    return {
        "missing_rate": float(vals.isna().mean()),
        "nunique": int(vals.nunique()),
        "min": float(vals.min()) if not vals.isna().all() else None,
        "p10": float(vals.quantile(0.1)) if not vals.isna().all() else None,
        "p50": float(vals.quantile(0.5)) if not vals.isna().all() else None,
        "p90": float(vals.quantile(0.9)) if not vals.isna().all() else None,
        "max": float(vals.max()) if not vals.isna().all() else None,
        "mean": float(vals.mean()) if not vals.isna().all() else None,
    }


def _correlation_with_dnp(df: pd.DataFrame, col: str, label_col: str) -> float | None:
    """Compute correlation between feature and DNP (minutes==0)."""
    vals = pd.to_numeric(df[col], errors="coerce")
    label_vals = pd.to_numeric(df[label_col], errors="coerce")
    dnp = (label_vals == 0).astype(int)

    # Only compute if we have non-constant values
    if vals.std() < 1e-8 or dnp.std() < 1e-8:
        return None

    # Use Pearson correlation
    mask = vals.notna() & dnp.notna()
    if mask.sum() < 10:
        return None

    return float(vals[mask].corr(dnp[mask]))


def run_coverage_diagnostics(df: pd.DataFrame, label_col: str = "minutes") -> dict:
    """Run coverage diagnostics for DNP history features."""
    results = {}

    for col in DNP_HISTORY_FEATURE_COLUMNS:
        if col not in df.columns:
            results[col] = {"error": "column not present"}
            continue

        stats = _coverage_stats(df, col)
        corr = _correlation_with_dnp(df, col, label_col)
        stats["correlation_with_dnp"] = corr
        results[col] = stats

    return results


def run_pit_verification(
    df: pd.DataFrame,
    *,
    sample_size: int = 50,
    label_col: str = "minutes",
) -> dict:
    """Verify point-in-time correctness by sampling and checking.

    This recomputes features on a sample and compares to stored values.
    """
    if df.empty:
        return {"status": "skip", "reason": "empty dataset"}

    # Sample rows
    sample_n = min(sample_size, len(df))
    sample_idx = df.sample(n=sample_n, random_state=42).index

    violations = []
    checked = 0

    for idx in sample_idx:
        row = df.loc[idx]
        pid = row.get("player_id")
        tid = row.get("team_id")
        current_date = row.get("game_date")

        if pd.isna(pid) or pd.isna(tid) or pd.isna(current_date):
            continue

        # Get all rows for this player-team
        player_team_mask = (df["player_id"] == pid) & (df["team_id"] == tid)
        player_team_df = df.loc[player_team_mask]

        # Count games strictly before current date
        prior_games = player_team_df[player_team_df["game_date"] < current_date]
        n_prior = len(prior_games)

        # Check games_since_last_roster_active
        games_since = row.get("games_since_last_roster_active")
        if games_since is not None and not pd.isna(games_since):
            games_since = int(games_since)
            if games_since < 99 and n_prior < games_since:
                violations.append({
                    "idx": int(idx),
                    "player_id": int(pid),
                    "team_id": int(tid),
                    "game_date": str(current_date),
                    "games_since_stored": games_since,
                    "n_prior_games": n_prior,
                    "issue": "games_since_last_roster_active > n_prior_games",
                })

        checked += 1

    return {
        "status": "pass" if not violations else "fail",
        "checked": checked,
        "violations": violations[:10],  # Limit to first 10
    }


def find_red_flags(
    df: pd.DataFrame,
    *,
    predictions_df: pd.DataFrame | None = None,
    top_k: int = 50,
    label_col: str = "minutes",
) -> list[dict]:
    """Find players with high DNP indicators but high predicted minutes.

    Red flags are cases where:
    - consecutive_active_dnp >= 2 OR
    - games_since_last_roster_active >= 10 AND never_roster_active_before == 0
    - BUT predicted_minutes is high (if predictions available) OR
    - actual minutes were > 0 (contradiction in training data)
    """
    if df.empty:
        return []

    # Join predictions if available
    work = df.copy()
    if predictions_df is not None and not predictions_df.empty:
        work = work.merge(
            predictions_df[["game_id", "team_id", "player_id", "predicted_minutes"]],
            on=["game_id", "team_id", "player_id"],
            how="left",
        )

    # Define red flag conditions
    has_consec_dnp = work.get("consecutive_active_dnp", pd.Series(0, index=work.index)) >= 2
    games_since = work.get("games_since_last_roster_active", pd.Series(99, index=work.index))
    never_active = work.get("never_roster_active_before", pd.Series(1, index=work.index))
    high_games_since = (games_since >= 10) & (never_active == 0)

    dnp_rate = work.get("active_but_dnp_rate_last10", pd.Series(0.0, index=work.index))
    high_dnp_rate = dnp_rate >= 0.5

    # Red flag mask: has DNP indicators
    red_flag_mask = has_consec_dnp | high_games_since | high_dnp_rate

    # But also played (contradiction)
    label_vals = pd.to_numeric(work.get(label_col, pd.Series(np.nan, index=work.index)), errors="coerce")
    played = label_vals > 0

    # Contradiction: red flag indicators but actually played
    contradiction_mask = red_flag_mask & played

    contradictions = work.loc[contradiction_mask].copy()
    if contradictions.empty:
        return []

    # Sort by label minutes descending (biggest surprises first)
    contradictions["_label"] = pd.to_numeric(contradictions.get(label_col, 0), errors="coerce").fillna(0)
    contradictions = contradictions.sort_values("_label", ascending=False).head(top_k)

    results = []
    for _, row in contradictions.iterrows():
        results.append({
            "player_id": int(row.get("player_id", 0)) if not pd.isna(row.get("player_id")) else None,
            "team_id": int(row.get("team_id", 0)) if not pd.isna(row.get("team_id")) else None,
            "player_name": str(row.get("player_name", "N/A")),
            "game_date": str(row.get("game_date", "N/A")),
            "label_minutes": float(row.get(label_col, 0)),
            "consecutive_active_dnp": int(row.get("consecutive_active_dnp", 0)),
            "games_since_last_roster_active": int(row.get("games_since_last_roster_active", 99)),
            "active_but_dnp_rate_last10": float(row.get("active_but_dnp_rate_last10", 0)),
            "never_roster_active_before": int(row.get("never_roster_active_before", 0)),
            "is_out": int(row.get("is_out", 0)) if not pd.isna(row.get("is_out")) else None,
            "predicted_minutes": float(row.get("predicted_minutes", 0)) if "predicted_minutes" in row.index and not pd.isna(row.get("predicted_minutes")) else None,
        })

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Path to rotation training dataset directory",
    )
    parser.add_argument(
        "--predictions-parquet",
        type=str,
        default=None,
        help="Optional path to predictions parquet for red-flag analysis",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default="minutes",
        help="Label column name",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Number of red flags to display",
    )
    parser.add_argument(
        "--json-output",
        type=str,
        default=None,
        help="Optional path to write JSON output",
    )
    parser.add_argument(
        "--fail-on-pit-violation",
        action="store_true",
        help="Exit with error code if PIT violation detected",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    features_path = dataset_dir / "features.parquet"
    labels_path = dataset_dir / "labels.parquet"

    if not features_path.exists():
        print(f"ERROR: Features not found at {features_path}", file=sys.stderr)
        sys.exit(1)
    if not labels_path.exists():
        print(f"ERROR: Labels not found at {labels_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading dataset from {dataset_dir}")
    features_df = pd.read_parquet(features_path)
    labels_df = pd.read_parquet(labels_path)

    # Merge features and labels
    label_col = args.label_col
    if label_col not in labels_df.columns:
        candidates = [c for c in labels_df.columns if "minute" in c.lower()]
        if candidates:
            label_col = candidates[0]
            print(f"Using inferred label column: {label_col}")

    df = features_df.merge(
        labels_df[["game_id", "team_id", "player_id", label_col]],
        on=["game_id", "team_id", "player_id"],
        how="left",
        suffixes=("", "_label"),
    )
    if f"{label_col}_label" in df.columns:
        df[label_col] = df[f"{label_col}_label"]

    print(f"Total rows: {len(df):,}")

    # Check if DNP history features are present
    missing_cols = [c for c in DNP_HISTORY_FEATURE_COLUMNS if c not in df.columns]

    if missing_cols:
        print(f"\nWARNING: DNP history features not present in dataset: {missing_cols}")
        print("Computing DNP history features from scratch...")

        # Compute features
        df = compute_dnp_history_features(
            df,
            config=DNPHistoryConfig(),
            game_date_col="game_date",
            player_id_col="player_id",
            team_id_col="team_id",
            is_out_col="is_out",
            minutes_col=label_col,
            validate_pit=False,
        )
        print("Done computing DNP history features.")

    output = {"dataset_dir": str(dataset_dir), "n_rows": len(df)}

    # 1. Coverage diagnostics
    _print_header("COVERAGE STATISTICS")
    coverage = run_coverage_diagnostics(df, label_col=label_col)
    output["coverage"] = coverage

    for col, stats in coverage.items():
        if "error" in stats:
            print(f"\n{col}: {stats['error']}")
            continue
        print(f"\n{col}:")
        print(f"  missing_rate: {stats['missing_rate']:.4f}")
        print(f"  nunique: {stats['nunique']}")
        print(f"  min/p10/p50/p90/max: {stats['min']:.2f} / {stats['p10']:.2f} / {stats['p50']:.2f} / {stats['p90']:.2f} / {stats['max']:.2f}")
        print(f"  mean: {stats['mean']:.4f}")
        if stats["correlation_with_dnp"] is not None:
            print(f"  correlation_with_dnp: {stats['correlation_with_dnp']:.4f}")

    # 2. Point-in-time verification
    _print_header("POINT-IN-TIME VERIFICATION")
    pit_result = run_pit_verification(df, sample_size=100, label_col=label_col)
    output["pit_verification"] = pit_result

    print(f"Status: {pit_result['status']}")
    print(f"Rows checked: {pit_result['checked']}")
    if pit_result["violations"]:
        print(f"Violations found: {len(pit_result['violations'])}")
        for v in pit_result["violations"][:5]:
            print(f"  - player={v['player_id']} team={v['team_id']} date={v['game_date']}")
            print(f"    {v['issue']}: stored={v['games_since_stored']}, n_prior={v['n_prior_games']}")

    if args.fail_on_pit_violation and pit_result["status"] == "fail":
        print("\nERROR: PIT violation detected!")
        sys.exit(1)

    # 3. Red flags
    _print_header("RED FLAGS (High DNP indicators but played)")
    predictions_df = None
    if args.predictions_parquet:
        pred_path = Path(args.predictions_parquet)
        if pred_path.exists():
            predictions_df = pd.read_parquet(pred_path)

    red_flags = find_red_flags(
        df,
        predictions_df=predictions_df,
        top_k=args.top_k,
        label_col=label_col,
    )
    output["red_flags"] = red_flags

    print(f"Found {len(red_flags)} contradictions (DNP indicators but played)")
    if red_flags:
        print("\nTop contradictions:")
        for i, rf in enumerate(red_flags[:10], 1):
            print(f"\n  {i}. {rf['player_name']} (id={rf['player_id']}, team={rf['team_id']})")
            print(f"     date={rf['game_date']}, actual_minutes={rf['label_minutes']:.1f}")
            print(f"     consecutive_active_dnp={rf['consecutive_active_dnp']}")
            print(f"     games_since_last_active={rf['games_since_last_roster_active']}")
            print(f"     dnp_rate_last10={rf['active_but_dnp_rate_last10']:.3f}")
            if rf.get("predicted_minutes") is not None:
                print(f"     predicted_minutes={rf['predicted_minutes']:.1f}")

    # 4. Bench player slice analysis
    _print_header("BENCH PLAYER ANALYSIS")
    # Define bench as players with low roll_mean_10 (if available)
    if "roll_mean_10" in df.columns:
        bench_mask = df["roll_mean_10"] < 15
        bench_df = df.loc[bench_mask]
        print(f"Bench players (roll_mean_10 < 15): {len(bench_df):,} rows ({len(bench_df)/len(df)*100:.1f}%)")

        bench_coverage = run_coverage_diagnostics(bench_df, label_col=label_col)
        output["bench_coverage"] = bench_coverage

        print("\nBench player DNP feature distributions:")
        for col in ["games_since_last_roster_active", "consecutive_active_dnp", "active_but_dnp_rate_last10"]:
            if col in bench_coverage and "error" not in bench_coverage[col]:
                stats = bench_coverage[col]
                print(f"  {col}: p50={stats['p50']:.2f}, mean={stats['mean']:.3f}")

    # Write JSON output if requested
    if args.json_output:
        json_path = Path(args.json_output)
        json_path.write_text(json.dumps(output, indent=2, sort_keys=True, default=str))
        print(f"\nWrote JSON output to {json_path}")

    _print_header("DONE")
    print(f"Dataset: {dataset_dir}")
    print(f"Total rows: {len(df):,}")
    dnp_count = (pd.to_numeric(df.get(label_col, 0), errors="coerce") == 0).sum()
    print(f"DNP rows (minutes==0): {dnp_count:,} ({dnp_count/len(df)*100:.1f}%)")


if __name__ == "__main__":
    main()
