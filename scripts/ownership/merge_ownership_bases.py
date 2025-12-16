"""
Merge Linestar and DK ownership training bases.

Creates a combined training dataset with source weighting.

Inputs:
    gold/ownership_training_base/ownership_training_base.parquet  (Linestar)
    gold/ownership_dk_base/ownership_dk_base.parquet              (DK)

Output:
    gold/ownership_merged_base/ownership_merged_base.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from projections.paths import data_path


def merge_bases(
    linestar_path: Path,
    dk_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    """Merge Linestar and DK ownership bases.

    DK data is higher quality (actual main slate GPP ownership),
    so it gets priority for overlapping dates.
    """
    # Load datasets
    print("Loading Linestar base...")
    linestar = pd.read_parquet(linestar_path)
    print(f"  Loaded {len(linestar):,} rows")

    # Add data_source if missing (backwards compatibility)
    if "data_source" not in linestar.columns:
        linestar["data_source"] = "linestar"

    print("Loading DK base...")
    dk = pd.read_parquet(dk_path)
    print(f"  Loaded {len(dk):,} rows")

    # Standardize column names (DK uses dk_slate_id, Linestar uses slate_id)
    if "dk_slate_id" in dk.columns and "slate_id" not in dk.columns:
        dk = dk.rename(columns={"dk_slate_id": "slate_id"})

    # For overlapping dates, prefer DK data (remove Linestar rows)
    dk_dates = set(dk["game_date"].unique())
    linestar_dates_overlap = linestar["game_date"].isin(dk_dates)

    print(f"\nData overlap:")
    print(f"  DK dates: {len(dk_dates)}")
    print(f"  Linestar rows in overlapping dates: {linestar_dates_overlap.sum():,}")

    # Option 1: Remove Linestar for overlapping dates (cleanest)
    # Option 2: Keep both but down-weight Linestar (more data)
    # Using Option 1 for cleaner data
    linestar_clean = linestar[~linestar_dates_overlap].copy()
    print(f"  Linestar rows after removing overlap: {len(linestar_clean):,}")

    # Align columns (keep intersection)
    common_cols = list(set(linestar_clean.columns) & set(dk.columns))
    print(f"\nCommon columns: {len(common_cols)}")

    # Concatenate
    merged = pd.concat([
        linestar_clean[common_cols],
        dk[common_cols],
    ], ignore_index=True)

    # Sort by date
    merged = merged.sort_values(["game_date", "slate_id"]).reset_index(drop=True)

    # Summary
    print(f"\n--- Merged Base Summary ---")
    print(f"Total rows: {len(merged):,}")
    print(f"By source:")
    print(merged["data_source"].value_counts().to_string())
    print(f"\nDate range: {merged['game_date'].min()} to {merged['game_date'].max()}")
    print(f"Unique slates: {merged['slate_id'].nunique():,}")
    print(f"Mean ownership (Linestar): {merged[merged['data_source']=='linestar']['actual_own_pct'].mean():.2f}%")
    print(f"Mean ownership (DK): {merged[merged['data_source']=='dk']['actual_own_pct'].mean():.2f}%")

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_path, index=False)
    print(f"\nWrote {len(merged):,} rows to {output_path}")

    return merged


def main():
    parser = argparse.ArgumentParser(description="Merge ownership training bases")
    parser.add_argument(
        "--linestar-path",
        type=Path,
        default=None,
        help="Path to Linestar training base",
    )
    parser.add_argument(
        "--dk-path",
        type=Path,
        default=None,
        help="Path to DK training base",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output parquet path",
    )

    args = parser.parse_args()

    # Defaults
    if args.linestar_path is None:
        args.linestar_path = data_path() / "gold" / "ownership_training_base" / "ownership_training_base.parquet"

    if args.dk_path is None:
        args.dk_path = data_path() / "gold" / "ownership_dk_base" / "ownership_dk_base.parquet"

    if args.output is None:
        args.output = data_path() / "gold" / "ownership_merged_base" / "ownership_merged_base.parquet"

    merge_bases(
        linestar_path=args.linestar_path,
        dk_path=args.dk_path,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
