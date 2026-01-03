#!/usr/bin/env python3
"""Diagnostic script to analyze minutes triplet duplication.

Reports:
- Top (minutes_p10, minutes_p50, minutes_p90) triplet and its count
- Same restricted to non-OUT players
- Feature analysis for templated rows

Usage:
    uv run python scripts/debug_templated_minutes.py <path_to_minutes.parquet>
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def analyze_triplets(df: pd.DataFrame) -> dict:
    """Analyze triplet duplication in a minutes DataFrame."""
    
    # Create triplet column
    triplet_cols = ["minutes_p10", "minutes_p50", "minutes_p90"]
    for col in triplet_cols:
        if col not in df.columns:
            return {"error": f"Missing column: {col}"}
    
    df = df.copy()
    df["triplet"] = df[triplet_cols].apply(lambda r: tuple(round(v, 4) for v in r), axis=1)
    
    # All players
    all_counts = df["triplet"].value_counts()
    top_triplet_all = all_counts.index[0]
    top_count_all = int(all_counts.iloc[0])
    
    # Non-OUT players
    status_col = "status" if "status" in df.columns else None
    if status_col:
        non_out = df[~df[status_col].astype(str).str.upper().isin({"OUT", "O", "INACTIVE"})]
    else:
        non_out = df
    
    non_out_counts = non_out["triplet"].value_counts()
    top_triplet_non_out = non_out_counts.index[0] if len(non_out_counts) > 0 else None
    top_count_non_out = int(non_out_counts.iloc[0]) if len(non_out_counts) > 0 else 0
    
    result = {
        "total_rows": len(df),
        "non_out_rows": len(non_out),
        "unique_triplets_all": len(all_counts),
        "unique_triplets_non_out": len(non_out_counts),
        "top_triplet_all": top_triplet_all,
        "top_count_all": top_count_all,
        "top_triplet_non_out": top_triplet_non_out,
        "top_count_non_out": top_count_non_out,
        "pass": top_count_non_out <= 10,
    }
    
    # Top 5 triplets for non-OUT
    result["top5_non_out"] = [(t, int(c)) for t, c in non_out_counts.head(5).items()]
    
    return result


def analyze_templated_rows(df: pd.DataFrame, triplet: tuple) -> dict:
    """Analyze rows matching the templated triplet to find root cause."""
    
    triplet_cols = ["minutes_p10", "minutes_p50", "minutes_p90"]
    df = df.copy()
    df["triplet"] = df[triplet_cols].apply(lambda r: tuple(round(v, 4) for v in r), axis=1)
    
    templated = df[df["triplet"] == triplet]
    if len(templated) == 0:
        return {"error": "No rows match triplet"}
    
    # Analyze key feature distributions
    feature_cols = [
        "starter_flag", "is_starter", "is_out", "status", "play_prob", 
        "rotation_prob", "p_rot", "mu_cond",
        "team_roll_mean_10", "team_minutes_rolling_mean",
        "minutes_p50_raw",  # If raw outputs exist
    ]
    
    analysis = {
        "templated_row_count": len(templated),
        "features": {},
    }
    
    for col in feature_cols:
        if col in templated.columns:
            unique_vals = templated[col].nunique()
            sample_vals = templated[col].dropna().head(5).tolist()
            analysis["features"][col] = {
                "unique_count": int(unique_vals),
                "sample_values": sample_vals,
            }
    
    # Check for constant feature patterns
    constant_features = [
        col for col, info in analysis["features"].items() 
        if info["unique_count"] == 1
    ]
    analysis["constant_features"] = constant_features
    
    return analysis


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/debug_templated_minutes.py <path_to_minutes.parquet>")
        sys.exit(1)
    
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)
    
    print(f"Loading {path}...")
    df = pd.read_parquet(path)
    
    print("\n=== TRIPLET ANALYSIS ===")
    result = analyze_triplets(df)
    for key, value in result.items():
        if key == "top5_non_out":
            print(f"\n{key}:")
            for triplet, count in value:
                print(f"  {triplet}: {count}")
        else:
            print(f"{key}: {value}")
    
    if not result.get("pass", True) and result.get("top_triplet_non_out"):
        print("\n=== TEMPLATED ROW ANALYSIS ===")
        templated_analysis = analyze_templated_rows(df, result["top_triplet_non_out"])
        for key, value in templated_analysis.items():
            if key == "features":
                print(f"\n{key}:")
                for feat, info in value.items():
                    print(f"  {feat}: unique={info['unique_count']}, samples={info['sample_values'][:3]}")
            else:
                print(f"{key}: {value}")
    
    print(f"\n{'PASS' if result.get('pass', False) else 'FAIL'}: top_count_non_out={result.get('top_count_non_out', 'N/A')} (threshold=10)")


if __name__ == "__main__":
    main()
