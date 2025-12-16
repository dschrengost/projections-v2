#!/usr/bin/env python3
"""
Parse DK contest results CSVs and aggregate to player-level ownership.

Input: dk_contests/nba_gpp_data/{date}/results/contest_*_results.csv
Output: Summary of player ownership by date/contest
"""
import pandas as pd
from pathlib import Path
import sys

DATA_ROOT = Path.home() / "projections-data" / "bronze" / "dk_contests" / "nba_gpp_data"

def parse_contest_file(csv_path: Path) -> pd.DataFrame:
    """Parse a single contest results CSV."""
    try:
        # Handle BOM in some files
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        
        # Normalize column names
        df.columns = df.columns.str.strip().str.replace('ï»¿', '')
        
        # Extract contest ID from filename
        contest_id = csv_path.stem.replace('contest_', '').replace('_results', '')
        
        # Parse %Drafted column (remove % sign)
        if '%Drafted' in df.columns:
            df['own_pct'] = df['%Drafted'].str.replace('%', '').astype(float)
        elif 'X.Drafted' in df.columns:
            df['own_pct'] = df['X.Drafted'].str.replace('%', '').astype(float)
        else:
            return pd.DataFrame()  # Skip if no ownership column
        
        # Extract player-level ownership (dedupe by player within this file)
        # Each row is a player in a lineup, but %Drafted is contest-wide
        player_own = df.groupby('Player').agg({
            'own_pct': 'first',  # Same for all rows of same player
            'FPTS': 'first',
            'Roster Position': 'first',
        }).reset_index()
        
        player_own['contest_id'] = contest_id
        player_own['file_path'] = str(csv_path)
        
        return player_own
        
    except Exception as e:
        print(f"  Error parsing {csv_path.name}: {e}")
        return pd.DataFrame()

def analyze_date(date_dir: Path) -> dict:
    """Analyze all contests for a single date."""
    results_dir = date_dir / "results"
    if not results_dir.exists():
        return {}
    
    contest_files = list(results_dir.glob("contest_*_results.csv"))
    
    all_players = []
    for csv_path in contest_files:
        df = parse_contest_file(csv_path)
        if not df.empty:
            all_players.append(df)
    
    if not all_players:
        return {}
    
    combined = pd.concat(all_players, ignore_index=True)
    
    return {
        'date': date_dir.name,
        'num_contests': len(contest_files),
        'num_players': combined['Player'].nunique(),
        'total_rows': len(combined),
        'avg_ownership': combined['own_pct'].mean(),
        'max_ownership': combined['own_pct'].max(),
        'sample_df': combined.head(10),
    }

def main():
    """Run analysis on all dates."""
    if not DATA_ROOT.exists():
        print(f"Data root not found: {DATA_ROOT}")
        sys.exit(1)
    
    dates = sorted([d for d in DATA_ROOT.iterdir() if d.is_dir()])
    print(f"Found {len(dates)} dates to analyze")
    print()
    
    summaries = []
    for date_dir in dates:
        result = analyze_date(date_dir)
        if result:
            summaries.append(result)
            print(f"{result['date']}: {result['num_contests']} contests, "
                  f"{result['num_players']} players, "
                  f"avg own: {result['avg_ownership']:.1f}%, "
                  f"max: {result['max_ownership']:.1f}%")
    
    print()
    print("=== SUMMARY ===")
    print(f"Total dates: {len(summaries)}")
    print(f"Total contests: {sum(s['num_contests'] for s in summaries)}")
    
    # Show sample data
    if summaries:
        print()
        print("=== SAMPLE DATA (first date) ===")
        print(summaries[0]['sample_df'].to_string())

if __name__ == "__main__":
    main()
