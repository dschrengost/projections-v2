"""
Build ownership training base from DK actual ownership data.

Joins DK contest ownership with Linestar projected data for features.
Uses player name matching to combine the datasets.

Inputs:
    bronze/dk_contests/ownership_by_slate/*.parquet  (DK actual ownership)
    gold/ownership_training_base/ownership_training_base.parquet (Linestar features)

Output:
    gold/ownership_dk_base/ownership_dk_base.parquet
"""

from __future__ import annotations

import argparse
import unicodedata
from pathlib import Path

import pandas as pd

from projections.paths import data_path


def normalize_name(val: object) -> str:
    """Normalize player name for matching: strip accents, lowercase, trim."""
    if val is None or pd.isna(val):
        return ""
    normalized = unicodedata.normalize("NFKD", str(val))
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    return ascii_only.strip().lower()


def load_dk_ownership(dk_ownership_path: Path) -> pd.DataFrame:
    """Load all DK ownership parquet files."""
    all_files = sorted(dk_ownership_path.glob("*.parquet"))

    # Exclude the combined file if present
    all_files = [f for f in all_files if not f.name.startswith("all_")]

    if not all_files:
        raise FileNotFoundError(f"No ownership files found in {dk_ownership_path}")

    print(f"Loading {len(all_files)} DK ownership files...")

    dfs = []
    for f in all_files:
        try:
            df = pd.read_parquet(f)
            dfs.append(df)
        except Exception as e:
            print(f"  Warning: failed to read {f.name}: {e}")

    result = pd.concat(dfs, ignore_index=True)
    print(f"  Loaded {len(result):,} rows from DK ownership")

    return result


def load_linestar_features(training_base_path: Path) -> pd.DataFrame:
    """Load Linestar training base for feature matching."""
    if not training_base_path.exists():
        raise FileNotFoundError(f"Training base not found: {training_base_path}")

    df = pd.read_parquet(training_base_path)
    print(f"Loaded {len(df):,} rows from Linestar training base")

    return df


def build_dk_base(
    dk_ownership_path: Path,
    linestar_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    """Build training base from DK ownership + Linestar features.

    Strategy:
    1. Load DK ownership (actual ownership from DK contests)
    2. Load Linestar features (salary, proj_fpts, etc.)
    3. Match by (game_date, normalized_player_name)
    4. Use DK actual ownership as label, Linestar for features
    """
    # Load data
    dk_own = load_dk_ownership(dk_ownership_path)
    linestar = load_linestar_features(linestar_path)

    # Normalize names
    dk_own["player_name_norm"] = dk_own["Player"].apply(normalize_name)

    # Ensure linestar has normalized names
    if "player_name_norm" not in linestar.columns:
        linestar["player_name_norm"] = linestar["player_name"].apply(normalize_name)

    # Create join keys
    dk_own["_join_key"] = dk_own["game_date"] + "_" + dk_own["player_name_norm"]
    linestar["_join_key"] = linestar["game_date"] + "_" + linestar["player_name_norm"]

    # Get unique Linestar rows per join key (may have duplicates from different slates)
    # Prefer main slate (largest slate_size or first occurrence)
    linestar_dedup = (
        linestar
        .sort_values("slate_id")  # Consistent ordering
        .groupby("_join_key")
        .first()
        .reset_index()
    )

    print(f"\nMatching data:")
    print(f"  DK ownership rows: {len(dk_own):,}")
    print(f"  DK unique join keys: {dk_own['_join_key'].nunique():,}")
    print(f"  Linestar unique join keys: {linestar_dedup['_join_key'].nunique():,}")

    # Feature columns to bring from Linestar
    feature_cols = [
        "_join_key",
        "season",
        "player_id",
        "player_name",
        "team",
        "pos",
        "salary",
        "proj_fpts",
        "floor_fpts",
        "ceil_fpts",
        "conf",
        "value_per_k",
        "ppg",
        "matchup",
        "home_team",
        "away_team",
        "opp_rank",
        "opp_total",
        "player_is_out",
        "player_is_questionable",
        "team_outs_count",
        "proj_own_pct",  # Linestar projection for comparison
    ]

    # Only include columns that exist
    feature_cols = [c for c in feature_cols if c in linestar_dedup.columns]

    # Join
    merged = dk_own.merge(
        linestar_dedup[feature_cols],
        on="_join_key",
        how="inner",  # Only keep matches
    )

    print(f"  Matched rows: {len(merged):,} ({len(merged)/len(dk_own)*100:.1f}% of DK data)")

    # Rename DK ownership columns (keep slate_id for compatibility)
    merged = merged.rename(columns={
        "own_pct": "actual_own_pct",
        "slate_size": "dk_slate_size",
        "FPTS": "scored_fpts_dk",
    })

    # Add data source marker
    merged["data_source"] = "dk"

    # Select output columns
    output_cols = [
        # Identifiers
        "season",
        "slate_id",
        "game_date",
        "player_id",
        "player_name",
        "player_name_norm",
        "team",
        "pos",
        # Salaries
        "salary",
        # Projections (from Linestar)
        "proj_fpts",
        "floor_fpts",
        "ceil_fpts",
        "conf",
        "value_per_k",
        "ppg",
        # Vegas context
        "matchup",
        "home_team",
        "away_team",
        "opp_rank",
        "opp_total",
        # Injury context
        "player_is_out",
        "player_is_questionable",
        "team_outs_count",
        # Ownership
        "proj_own_pct",
        "actual_own_pct",
        # DK-specific
        "dk_slate_size",
        "scored_fpts_dk",
        "entries",
        "num_contests",
        # Source
        "data_source",
    ]

    output_cols = [c for c in output_cols if c in merged.columns]
    result = merged[output_cols].copy()

    # Filter out broken data
    # Remove rows with extreme ownership (>98% usually indicates data issues)
    before_filter = len(result)
    result = result[result["actual_own_pct"] <= 98.0].copy()
    print(f"\nFiltered: {before_filter - len(result):,} rows with ownership > 98%")

    # Summary
    print(f"\n--- DK Base Summary ---")
    print(f"Total rows: {len(result):,}")
    print(f"Unique slates: {result['slate_id'].nunique():,}")
    print(f"Unique players: {result['player_id'].nunique():,}")
    print(f"Date range: {result['game_date'].min()} to {result['game_date'].max()}")
    print(f"Mean ownership: {result['actual_own_pct'].mean():.2f}%")
    if "dk_slate_size" in result.columns:
        print(f"Slate size: mean={result['dk_slate_size'].mean():.0f}, median={result['dk_slate_size'].median():.0f}")

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    print(f"\nWrote {len(result):,} rows to {output_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Build DK ownership training base")
    parser.add_argument(
        "--dk-ownership-path",
        type=Path,
        default=None,
        help="Path to DK ownership parquet files",
    )
    parser.add_argument(
        "--linestar-path",
        type=Path,
        default=None,
        help="Path to Linestar training base parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output parquet path",
    )

    args = parser.parse_args()

    # Defaults
    if args.dk_ownership_path is None:
        args.dk_ownership_path = data_path() / "bronze" / "dk_contests" / "ownership_by_slate"

    if args.linestar_path is None:
        args.linestar_path = data_path() / "gold" / "ownership_training_base" / "ownership_training_base.parquet"

    if args.output is None:
        args.output = data_path() / "gold" / "ownership_dk_base" / "ownership_dk_base.parquet"

    build_dk_base(
        dk_ownership_path=args.dk_ownership_path,
        linestar_path=args.linestar_path,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
