#!/usr/bin/env python3
"""
Build training dataset for dupe penalty model.

Creates entry-level training data from historical contest results,
computing ownership features for each lineup to predict dupe counts.

Input:  bronze/dk_contests/nba_gpp_data/{date}/results/contest_*_results.csv
        bronze/dk_contests/nba_gpp_data/{date}/nba_gpp_{date}.csv (contest metadata)
Output: gold/dupe_training_data.parquet
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from projections.api.contest_service import (
    get_contest_data_dir,
    get_data_root,
    parse_lineup,
)


def load_contest_metadata(date_dir: Path) -> dict[str, dict]:
    """Load contest metadata (field_size, entry_max, entry_fee) for a date."""
    # Try both naming conventions
    meta_file = date_dir / f"nba_gpp_{date_dir.name}.csv"
    if not meta_file.exists():
        meta_file = date_dir / "contests.csv"
    if not meta_file.exists():
        return {}

    try:
        df = pd.read_csv(meta_file)
        metadata = {}
        for _, row in df.iterrows():
            cid = str(row.get("contest_id", ""))
            if cid:
                metadata[cid] = {
                    "field_size": int(row.get("current_entries", 0)),
                    "entry_max": int(row.get("max_entries", 1)),
                    "entry_fee": float(row.get("entry_fee", 0)),
                    "contest_name": str(row.get("contest_name", "")),
                }
        return metadata
    except Exception:
        return {}


def parse_contest_results(csv_path: Path) -> pd.DataFrame | None:
    """Parse a contest results CSV and extract lineup-level data."""
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False)
        df.columns = df.columns.str.strip().str.replace("ï»¿", "")

        # Need Lineup column
        if "Lineup" not in df.columns:
            return None

        # Parse ownership column
        if "%Drafted" in df.columns:
            own_col = "%Drafted"
        elif "X.Drafted" in df.columns:
            own_col = "X.Drafted"
        else:
            return None

        # Build player -> ownership lookup
        player_own = {}
        for _, row in df.iterrows():
            player = row.get("Player", "")
            own_str = str(row.get(own_col, "0")).replace("%", "")
            try:
                player_own[player] = float(own_str)
            except ValueError:
                pass

        # Get unique entries (one row per entry)
        if "EntryId" in df.columns:
            entries_df = df.drop_duplicates(subset=["EntryId"])
        else:
            entries_df = df.drop_duplicates(subset=["Lineup"])

        return entries_df, player_own

    except Exception:
        return None


def compute_lineup_features(
    lineup_players: list[str], player_own: dict[str, float]
) -> dict:
    """Compute ownership-based features for a lineup."""
    ownerships = [player_own.get(p, 0) for p in lineup_players]

    if not ownerships:
        return None

    return {
        "sum_own": sum(ownerships),
        "max_own": max(ownerships),
        "min_own": min(ownerships),
        "num_under_10": sum(1 for o in ownerships if o < 10),
        "num_under_5": sum(1 for o in ownerships if o < 5),
        "num_over_30": sum(1 for o in ownerships if o > 30),
    }


def process_contest(
    csv_path: Path, contest_meta: dict
) -> list[dict]:
    """Process a single contest file and return entry-level records."""
    result = parse_contest_results(csv_path)
    if result is None:
        return []

    entries_df, player_own = result
    contest_id = csv_path.stem.replace("contest_", "").replace("_results", "")

    # Get metadata
    meta = contest_meta.get(contest_id, {})
    field_size = meta.get("field_size", len(entries_df))
    entry_max = meta.get("entry_max", 1)
    entry_fee = meta.get("entry_fee", 0)

    # Build lineup_key -> list of entries
    lineup_map: dict[str, list[dict]] = defaultdict(list)

    for _, entry in entries_df.iterrows():
        lineup_str = entry.get("Lineup", "")
        lineup_players = parse_lineup(lineup_str)

        if len(lineup_players) < 7:  # Invalid lineup
            continue

        # Canonical key (sorted player names)
        lineup_key = "|".join(sorted(lineup_players))

        features = compute_lineup_features(lineup_players, player_own)
        if features is None:
            continue

        lineup_map[lineup_key].append({
            "lineup_players": lineup_players,
            **features,
        })

    # Now create entry-level records with dupe counts
    records = []
    for lineup_key, entries in lineup_map.items():
        dupe_count = len(entries)  # K
        first_entry = entries[0]

        # Create one record per entry (entry-level weighting)
        for _ in range(dupe_count):
            records.append({
                "contest_id": contest_id,
                "lineup_key": lineup_key,
                "dupe_count": dupe_count,
                "sum_own": first_entry["sum_own"],
                "max_own": first_entry["max_own"],
                "min_own": first_entry["min_own"],
                "num_under_10": first_entry["num_under_10"],
                "num_under_5": first_entry["num_under_5"],
                "num_over_30": first_entry["num_over_30"],
                "field_size": field_size,
                "entry_max": entry_max,
                "entry_fee": entry_fee,
            })

    return records


def process_date(date_dir: Path) -> list[dict]:
    """Process all contests for a single date."""
    results_dir = date_dir / "results"
    if not results_dir.exists():
        return []

    contest_files = list(results_dir.glob("contest_*_results.csv"))
    if not contest_files:
        return []

    # Load contest metadata
    contest_meta = load_contest_metadata(date_dir)

    all_records = []
    for csv_path in contest_files:
        records = process_contest(csv_path, contest_meta)
        for r in records:
            r["date"] = date_dir.name
        all_records.extend(records)

    return all_records


def main():
    parser = argparse.ArgumentParser(description="Build dupe training data")
    parser.add_argument(
        "--dates",
        nargs="*",
        help="Specific dates to process (default: all)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (default: gold/dupe_training_data.parquet)",
    )
    args = parser.parse_args()

    data_root = get_data_root()
    input_dir = get_contest_data_dir()
    output_path = args.output or (data_root / "gold" / "dupe_training_data.parquet")

    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
        sys.exit(1)

    # Get dates to process
    if args.dates:
        dates = [input_dir / d for d in args.dates if (input_dir / d).exists()]
    else:
        dates = sorted([d for d in input_dir.iterdir() if d.is_dir() and d.name.startswith("20")])

    print(f"Processing {len(dates)} dates...")

    all_records = []
    for date_dir in dates:
        records = process_date(date_dir)
        if records:
            print(f"  {date_dir.name}: {len(records):,} entries")
            all_records.extend(records)

    if not all_records:
        print("No data found!")
        sys.exit(1)

    # Create output DataFrame
    df = pd.DataFrame(all_records)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save
    df.to_parquet(output_path, index=False)

    print()
    print("=== SUMMARY ===")
    print(f"Total entries: {len(df):,}")
    print(f"Unique lineups: {df['lineup_key'].nunique():,}")
    print(f"Unique contests: {df['contest_id'].nunique():,}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print()
    print(f"Dupe count distribution:")
    print(df["dupe_count"].value_counts().head(10).to_string())
    print()
    print(f"Mean dupe count (K): {df['dupe_count'].mean():.2f}")
    print(f"Mean sum_own: {df['sum_own'].mean():.1f}%")
    print()
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
