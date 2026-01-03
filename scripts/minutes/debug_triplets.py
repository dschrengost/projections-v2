#!/usr/bin/env python3
"""Diagnostic script to analyze minutes triplet duplication.

Reports:
- Top 10 triplet counts for raw and final minutes
- Same restricted to status != OUT
- Counts of has_any_history=False (when available)

Usage:
    uv run python scripts/minutes/debug_triplets.py <path_to_minutes.parquet>
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def _triplet_series(df: pd.DataFrame, p10_col: str, p50_col: str, p90_col: str) -> pd.Series:
    """Create a triplet Series from three column names, rounding to 2 decimals."""
    if not {p10_col, p50_col, p90_col}.issubset(df.columns):
        return pd.Series(dtype=object)
    return df[[p10_col, p50_col, p90_col]].apply(
        lambda r: (round(r.iloc[0], 2), round(r.iloc[1], 2), round(r.iloc[2], 2)), axis=1
    )


def _non_out_mask(df: pd.DataFrame) -> pd.Series:
    """Return mask for non-OUT players."""
    if "status" not in df.columns:
        return pd.Series(True, index=df.index)
    return ~df["status"].astype(str).str.upper().isin({"OUT", "O", "INACTIVE"})


def analyze_triplets(df: pd.DataFrame, prefix: str, p10: str, p50: str, p90: str) -> None:
    """Print triplet analysis for given columns."""
    triplets = _triplet_series(df, p10, p50, p90)
    if triplets.empty:
        print(f"\n=== {prefix} (columns missing) ===")
        return
    
    non_out = _non_out_mask(df)
    
    all_counts = triplets.value_counts()
    non_out_counts = triplets[non_out].value_counts()
    
    print(f"\n=== {prefix} ===")
    print(f"Total rows: {len(df)}")
    print(f"Non-OUT rows: {non_out.sum()}")
    print(f"Unique triplets (all): {len(all_counts)}")
    print(f"Unique triplets (non-OUT): {len(non_out_counts)}")
    
    print(f"\nTop 10 triplets (all):")
    for triplet, count in all_counts.head(10).items():
        print(f"  {triplet}: {count}")
    
    print(f"\nTop 10 triplets (non-OUT):")
    for triplet, count in non_out_counts.head(10).items():
        print(f"  {triplet}: {count}")
    
    top_non_out_count = int(non_out_counts.iloc[0]) if len(non_out_counts) > 0 else 0
    status = "PASS" if top_non_out_count <= 10 else "FAIL"
    print(f"\n{status}: top triplet count (non-OUT) = {top_non_out_count} (threshold=10)")


def analyze_history(df: pd.DataFrame) -> None:
    """Print has_any_history analysis."""
    if "has_any_history" not in df.columns:
        print("\n=== has_any_history (column missing) ===")
        return
    
    non_out = _non_out_mask(df)
    has_history = df["has_any_history"].astype(int)
    
    print("\n=== has_any_history ===")
    print(f"All rows: {len(df)}, with history={has_history.sum()}, without={(has_history == 0).sum()}")
    print(f"Non-OUT:  {non_out.sum()}, with history={has_history[non_out].sum()}, without={(has_history[non_out] == 0).sum()}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/minutes/debug_triplets.py <path_to_minutes.parquet>")
        sys.exit(1)
    
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)
    
    print(f"Loading {path}...")
    df = pd.read_parquet(path)
    
    # Analyze raw triplets (if debug columns present)
    analyze_triplets(df, "RAW TRIPLETS", "minutes_p10_raw", "minutes_p50_raw", "minutes_p90_raw")
    
    # Analyze final triplets
    analyze_triplets(df, "FINAL TRIPLETS", "minutes_p10", "minutes_p50", "minutes_p90")
    
    # Analyze history availability
    analyze_history(df)
    
    # Summary
    print("\n=== SUMMARY ===")
    has_raw = {"minutes_p10_raw", "minutes_p50_raw", "minutes_p90_raw"}.issubset(df.columns)
    if has_raw:
        raw_triplets = _triplet_series(df, "minutes_p10_raw", "minutes_p50_raw", "minutes_p90_raw")
        raw_non_out = raw_triplets[_non_out_mask(df)].value_counts()
        raw_top = int(raw_non_out.iloc[0]) if len(raw_non_out) > 0 else 0
        print(f"Raw triplets top count (non-OUT): {raw_top}")
    else:
        print("Raw columns not present - run with PROJECTIONS_DEBUG_MINUTES_RAW=1")
    
    final_triplets = _triplet_series(df, "minutes_p10", "minutes_p50", "minutes_p90")
    final_non_out = final_triplets[_non_out_mask(df)].value_counts()
    final_top = int(final_non_out.iloc[0]) if len(final_non_out) > 0 else 0
    print(f"Final triplets top count (non-OUT): {final_top}")


if __name__ == "__main__":
    main()
