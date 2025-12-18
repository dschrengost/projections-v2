#!/usr/bin/env python3
"""
Build ownership data from DK contest results.

Aggregates lineup-level contest data to player-level ownership per SLATE.
Clusters contests by player pool overlap to identify distinct slates.

Input:  bronze/dk_contests/nba_gpp_data/{date}/results/contest_*_results.csv
Output: bronze/dk_contests/ownership_by_slate/{date}_{slate_id}.parquet
        bronze/dk_contests/ownership_by_slate/all_ownership.parquet
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from typing import Optional, List, Dict, Tuple
from collections import defaultdict

DATA_ROOT = Path.home() / "projections-data" / "bronze" / "dk_contests"
INPUT_DIR = DATA_ROOT / "nba_gpp_data"
OUTPUT_DIR = DATA_ROOT / "ownership_by_slate"


def parse_contest_file(csv_path: Path) -> Optional[pd.DataFrame]:
    """Parse a single contest results CSV and extract player ownership."""
    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig', low_memory=False)
        df.columns = df.columns.str.strip().str.replace('ï»¿', '')
        
        contest_id = csv_path.stem.replace('contest_', '').replace('_results', '')
        
        # Parse %Drafted column
        if '%Drafted' in df.columns:
            own_col = '%Drafted'
        elif 'X.Drafted' in df.columns:
            own_col = 'X.Drafted'
        else:
            return None
        
        df['own_pct'] = df[own_col].astype(str).str.replace('%', '').astype(float)
        
        # Get unique players with their ownership
        player_own = df.groupby('Player', as_index=False).agg({
            'own_pct': 'first',
            'FPTS': 'first',
            'Roster Position': 'first',
        })
        
        player_own['contest_id'] = int(contest_id)
        
        # Count entries
        if 'EntryId' in df.columns:
            player_own['entries'] = df['EntryId'].nunique()
        else:
            player_own['entries'] = len(df) // 8
        
        return player_own
        
    except Exception as e:
        return None


def cluster_contests_by_slate(contest_data: Dict[str, pd.DataFrame]) -> List[List[str]]:
    """
    Cluster contests into slates based on player pool overlap.
    Contests with >80% player overlap are considered the same slate.
    """
    # Get player sets for each contest
    contest_players = {}
    for cid, df in contest_data.items():
        contest_players[cid] = frozenset(df['Player'].unique())
    
    def overlap_ratio(set1, set2):
        if not set1 or not set2:
            return 0
        intersection = len(set1 & set2)
        smaller = min(len(set1), len(set2))
        return intersection / smaller if smaller > 0 else 0
    
    # Greedy clustering
    groups = []
    assigned = set()
    
    # Sort by contest size (larger contests first - likely main slate)
    sorted_contests = sorted(contest_players.items(), key=lambda x: len(x[1]), reverse=True)
    
    for cid1, players1 in sorted_contests:
        if cid1 in assigned:
            continue
        
        group = [cid1]
        assigned.add(cid1)
        
        for cid2, players2 in sorted_contests:
            if cid2 in assigned:
                continue
            if overlap_ratio(players1, players2) > 0.8:
                group.append(cid2)
                assigned.add(cid2)
        
        groups.append(group)
    
    return groups


def aggregate_slate_ownership(
    slate_contests: List[str],
    contest_data: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Aggregate ownership for a single slate (group of contests).
    Computes entry-weighted average ownership.

    Important: players missing from a contest are treated as 0% in that contest.
    This prevents inflated ownership and ensures slate-level sums remain stable
    (≈ 800% for DK NBA classic) when contests within a cluster have different
    player pools.
    """
    all_data = []
    for cid in slate_contests:
        if cid in contest_data:
            all_data.append(contest_data[cid])
    
    if not all_data:
        return pd.DataFrame()
    
    combined = pd.concat(all_data, ignore_index=True)

    # Total contest weights (entries) for the slate.
    contest_weights = combined[["contest_id", "entries"]].drop_duplicates("contest_id")
    total_entries = float(contest_weights["entries"].sum())
    total_contests = int(contest_weights["contest_id"].nunique())

    def weighted_avg_with_zeros(group: pd.DataFrame) -> float:
        """Entry-weighted mean ownership, treating missing contests as 0%."""
        if total_entries <= 0:
            return float(group["own_pct"].mean())
        return float((group["own_pct"] * group["entries"]).sum() / total_entries)

    def mean_with_zeros(group: pd.DataFrame) -> float:
        """Unweighted mean across contests, treating missing contests as 0%."""
        if total_contests <= 0:
            return float(group["own_pct"].mean())
        return float(group["own_pct"].sum() / total_contests)

    agg = (
        combined.groupby("Player", sort=False)
        .apply(
            lambda g: pd.Series(
                {
                    "own_pct": weighted_avg_with_zeros(g),
                    "own_pct_simple": mean_with_zeros(g),
                    "FPTS": g["FPTS"].iloc[0],
                    # Backwards-compatible per-player metadata
                    "entries": g["entries"].sum(),
                    "num_contests": g["contest_id"].nunique(),
                    # Slate-level metadata (same for all players)
                    "slate_entries": total_entries,
                    "slate_num_contests": total_contests,
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )

    return agg


def process_date(date_dir: Path) -> List[pd.DataFrame]:
    """Process all contests for a single date, separated by slate."""
    results_dir = date_dir / "results"
    if not results_dir.exists():
        return []
    
    contest_files = sorted(results_dir.glob("contest_*_results.csv"))
    if not contest_files:
        return []
    
    # Parse all contests
    contest_data = {}
    for csv_path in contest_files:
        df = parse_contest_file(csv_path)
        if df is not None and not df.empty:
            cid = csv_path.stem.replace('contest_', '').replace('_results', '')
            contest_data[cid] = df
    
    if not contest_data:
        return []
    
    # Cluster into slates
    slates = cluster_contests_by_slate(contest_data)
    
    # Aggregate each slate
    results = []
    for i, slate_contests in enumerate(slates):
        agg = aggregate_slate_ownership(slate_contests, contest_data)
        if not agg.empty:
            agg['game_date'] = date_dir.name
            agg['slate_id'] = f"{date_dir.name}_{i}"
            agg['slate_size'] = len(agg)
            results.append(agg)
    
    return results


def main():
    """Build ownership data for all dates."""
    if not INPUT_DIR.exists():
        print(f"Input directory not found: {INPUT_DIR}")
        sys.exit(1)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    dates = sorted([d for d in INPUT_DIR.iterdir() if d.is_dir()])
    print(f"Processing {len(dates)} dates...")
    print()
    
    all_slates = []
    total_slates = 0
    
    for date_dir in dates:
        slates = process_date(date_dir)
        if slates:
            total_slates += len(slates)
            for slate_df in slates:
                # Save individual slate file
                slate_id = slate_df['slate_id'].iloc[0]
                out_path = OUTPUT_DIR / f"{slate_id}.parquet"
                slate_df.to_parquet(out_path)
                all_slates.append(slate_df)
            
            print(f"{date_dir.name}: {len(slates)} slates, "
                  f"{sum(len(s) for s in slates)} total player rows")
    
    # Save combined file
    if all_slates:
        combined = pd.concat(all_slates, ignore_index=True)
        combined_path = OUTPUT_DIR / "all_ownership.parquet"
        combined.to_parquet(combined_path)
        
        print()
        print("=== SUMMARY ===")
        print(f"Total dates: {len(dates)}")
        print(f"Total slates: {total_slates}")
        print(f"Total player-slate rows: {len(combined)}")
        print(f"Unique players: {combined['Player'].nunique()}")
        print(f"Date range: {combined['game_date'].min()} to {combined['game_date'].max()}")
        print(f"Avg ownership: {combined['own_pct'].mean():.1f}%")
        print(f"Max ownership: {combined['own_pct'].max():.1f}%")
        print()
        print(f"Output: {combined_path}")
        
        # Show sample
        print()
        print("=== SAMPLE (high ownership players) ===")
        sample = combined.nlargest(10, 'own_pct')[
            ['game_date', 'slate_id', 'Player', 'own_pct', 'entries', 'num_contests', 'slate_size']
        ]
        print(sample.to_string(index=False))


if __name__ == "__main__":
    main()
