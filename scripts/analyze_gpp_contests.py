#!/usr/bin/env python3
"""
Analyze DraftKings NBA GPP contest results to extract insights for lineup building.

Insights produced:
1. Winning lineup ownership characteristics (total ownership, leverage)
2. Ownership band performance by slate size
3. What differentiates 1st place finishes from cash line
4. Contest size/type impact on strategy
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import re

DATA_ROOT = Path.home() / "projections-data" / "bronze" / "dk_contests" / "nba_gpp_data"


def parse_lineup(lineup_str: str) -> list[str]:
    """Parse a lineup string into player names."""
    if pd.isna(lineup_str):
        return []
    # Format: "C Player1 F Player2 G Player3 PF Player4 PG Player5 SF Player6 SG Player7 UTIL Player8"
    positions = ["C ", "F ", "G ", "PF ", "PG ", "SF ", "SG ", "UTIL "]
    players = []
    for pos in positions:
        if pos in lineup_str:
            start = lineup_str.find(pos) + len(pos)
            # Find the next position marker or end of string
            end = len(lineup_str)
            for next_pos in positions:
                idx = lineup_str.find(next_pos, start)
                if idx > 0 and idx < end:
                    end = idx
            player = lineup_str[start:end].strip()
            if player:
                players.append(player)
    return players


def parse_contest_file(csv_path: Path) -> tuple[pd.DataFrame, dict]:
    """
    Parse a single contest results CSV.
    Returns (entries_df, player_ownership_dict)
    """
    try:
        # Read with BOM handling
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        df.columns = df.columns.str.strip().str.replace('\ufeff', '').str.replace('ï»¿', '')
        
        # Parse ownership column
        own_col = None
        for col in ['%Drafted', 'X.Drafted']:
            if col in df.columns:
                own_col = col
                break
        
        if own_col is None:
            return pd.DataFrame(), {}
        
        # Build player ownership lookup
        player_own = {}
        for _, row in df.iterrows():
            if pd.notna(row.get('Player')) and pd.notna(row.get(own_col)):
                try:
                    own_str = str(row[own_col]).replace('%', '')
                    player_own[row['Player']] = float(own_str)
                except:
                    pass
        
        # Get unique entries (first occurrence of each EntryId)
        entries = df[df['EntryId'].notna()].drop_duplicates(subset=['EntryId'])
        
        if entries.empty:
            return pd.DataFrame(), player_own
        
        # Parse each entry's lineup and calculate ownership metrics
        results = []
        for _, entry in entries.iterrows():
            lineup_players = parse_lineup(entry.get('Lineup', ''))
            if not lineup_players:
                continue
            
            # Calculate ownership metrics for this lineup
            player_owns = [player_own.get(p, 0) for p in lineup_players]
            valid_owns = [o for o in player_owns if o > 0]
            
            if not valid_owns:
                continue
                
            results.append({
                'rank': int(entry['Rank']) if pd.notna(entry.get('Rank')) else None,
                'points': float(entry['Points']) if pd.notna(entry.get('Points')) else None,
                'total_own': sum(valid_owns),
                'avg_own': np.mean(valid_owns),
                'min_own': min(valid_owns),
                'max_own': max(valid_owns),
                'num_players': len(valid_owns),
                'num_under_10': sum(1 for o in valid_owns if o < 10),
                'num_under_5': sum(1 for o in valid_owns if o < 5),
                'num_over_50': sum(1 for o in valid_owns if o > 50),
            })
        
        return pd.DataFrame(results), player_own
        
    except Exception as e:
        print(f"Error parsing {csv_path.name}: {e}")
        return pd.DataFrame(), {}


def analyze_date(date_dir: Path) -> list[dict]:
    """Analyze all contests for a single date, returning contest-level summaries."""
    results_dir = date_dir / "results"
    if not results_dir.exists():
        return []
    
    # Try to get contest metadata from the daily CSV
    csv_files = list(date_dir.glob("*.csv"))
    contest_meta = {}
    for csv_file in csv_files:
        if "results" not in csv_file.name:
            try:
                meta_df = pd.read_csv(csv_file, encoding='utf-8-sig')
                meta_df.columns = meta_df.columns.str.strip()
                for _, row in meta_df.iterrows():
                    cid = str(row.get('ContestId', row.get('contest_id', '')))
                    contest_meta[cid] = {
                        'contest_name': row.get('ContestName', row.get('contest_name', '')),
                        'entry_fee': row.get('EntryFee', row.get('entry_fee', '')),
                        'total_entries': row.get('CurrentEntries', row.get('current_entries', '')),
                        'max_entries': row.get('MaxEntries', row.get('max_entries', '')),
                    }
            except:
                pass
    
    # Count games by parsing player team info (rough estimate)
    all_contests = []
    
    for results_file in results_dir.glob("contest_*_results.csv"):
        contest_id = results_file.stem.replace('contest_', '').replace('_results', '')
        entries_df, player_own = parse_contest_file(results_file)
        
        if entries_df.empty:
            continue
        
        # Get metadata
        meta = contest_meta.get(contest_id, {})
        
        # Calculate winning lineup characteristics
        total_entries = len(entries_df)
        if total_entries < 10:
            continue  # Skip very small contests
        
        winner = entries_df[entries_df['rank'] == 1].iloc[0] if len(entries_df[entries_df['rank'] == 1]) > 0 else None
        top10 = entries_df[entries_df['rank'] <= 10]
        top1pct = entries_df[entries_df['rank'] <= max(1, int(total_entries * 0.01))]
        cash = entries_df[entries_df['rank'] <= int(total_entries * 0.2)]  # Rough 20% cash line
        
        contest_data = {
            'date': date_dir.name,
            'contest_id': contest_id,
            'contest_name': meta.get('contest_name', ''),
            'entry_fee': meta.get('entry_fee', ''),
            'total_entries': total_entries,
            # Winner metrics
            'winner_points': winner['points'] if winner is not None else None,
            'winner_total_own': winner['total_own'] if winner is not None else None,
            'winner_avg_own': winner['avg_own'] if winner is not None else None,
            'winner_min_own': winner['min_own'] if winner is not None else None,
            'winner_num_under_10': winner['num_under_10'] if winner is not None else None,
            'winner_num_under_5': winner['num_under_5'] if winner is not None else None,
            # Top 10 metrics (aggregated)
            'top10_avg_total_own': top10['total_own'].mean() if not top10.empty else None,
            'top10_avg_points': top10['points'].mean() if not top10.empty else None,
            # Top 1% metrics
            'top1pct_avg_total_own': top1pct['total_own'].mean() if not top1pct.empty else None,
            'top1pct_avg_points': top1pct['points'].mean() if not top1pct.empty else None,
            # Cash line metrics
            'cash_avg_total_own': cash['total_own'].mean() if not cash.empty else None,
            'cash_avg_points': cash['points'].mean() if not cash.empty else None,
            # Field average
            'field_avg_total_own': entries_df['total_own'].mean(),
            'field_avg_points': entries_df['points'].mean(),
        }
        
        all_contests.append(contest_data)
    
    return all_contests


def categorize_contest(name: str) -> str:
    """Categorize contest by type based on name."""
    name_lower = name.lower()
    if 'high five' in name_lower:
        return 'High Five'
    elif 'hot shot' in name_lower:
        return 'Hot Shot'
    elif 'pick and roll' in name_lower:
        return 'Pick and Roll'
    elif 'elbow shot' in name_lower or 'elbow' in name_lower:
        return 'Elbow Shot'
    elif 'milly' in name_lower or 'million' in name_lower:
        return 'Millionaire'
    elif 'single entry' in name_lower:
        return 'Single Entry'
    elif 'max' in name_lower:
        return 'Multi-Entry'
    else:
        return 'Other'


def main():
    print("=" * 80)
    print("NBA GPP CONTEST ANALYSIS")
    print("=" * 80)
    
    if not DATA_ROOT.exists():
        print(f"Data root not found: {DATA_ROOT}")
        return
    
    # Collect all contest data
    dates = sorted([d for d in DATA_ROOT.iterdir() if d.is_dir()])
    print(f"\nFound {len(dates)} dates of data")
    
    all_contests = []
    for date_dir in dates:
        print(f"Processing {date_dir.name}...")
        contests = analyze_date(date_dir)
        all_contests.extend(contests)
    
    df = pd.DataFrame(all_contests)
    print(f"\nTotal contests analyzed: {len(df)}")
    
    # Add contest type categorization
    df['contest_type'] = df['contest_name'].apply(categorize_contest)
    
    # ===== ANALYSIS 1: Overall Winning Lineup Characteristics =====
    print("\n" + "=" * 80)
    print("1. WINNING LINEUP OWNERSHIP CHARACTERISTICS")
    print("=" * 80)
    
    print(f"\nWinner Total Ownership (sum of 8 players):")
    print(f"  Mean: {df['winner_total_own'].mean():.1f}%")
    print(f"  Median: {df['winner_total_own'].median():.1f}%")
    print(f"  Std Dev: {df['winner_total_own'].std():.1f}%")
    print(f"  25th percentile: {df['winner_total_own'].quantile(0.25):.1f}%")
    print(f"  75th percentile: {df['winner_total_own'].quantile(0.75):.1f}%")
    
    print(f"\nWinner Average Player Ownership:")
    print(f"  Mean: {df['winner_avg_own'].mean():.1f}%")
    print(f"  Median: {df['winner_avg_own'].median():.1f}%")
    
    print(f"\nWinner Min Player Ownership (lowest owned player):")
    print(f"  Mean: {df['winner_min_own'].mean():.1f}%")
    print(f"  Median: {df['winner_min_own'].median():.1f}%")
    
    print(f"\nWinner Leverage Players (under 10% owned):")
    print(f"  Mean count: {df['winner_num_under_10'].mean():.1f}")
    print(f"  Mean count under 5%: {df['winner_num_under_5'].mean():.1f}")
    
    # ===== ANALYSIS 2: Winner vs Field Comparison =====
    print("\n" + "=" * 80)
    print("2. WINNER VS FIELD COMPARISON (OWNERSHIP DIFFERENTIAL)")
    print("=" * 80)
    
    df['own_diff_vs_field'] = df['winner_total_own'] - df['field_avg_total_own']
    df['own_diff_vs_cash'] = df['winner_total_own'] - df['cash_avg_total_own']
    
    print(f"\nWinner Total Own vs Field Average:")
    print(f"  Mean difference: {df['own_diff_vs_field'].mean():+.1f}%")
    print(f"  Winners with LOWER ownership than field: {(df['own_diff_vs_field'] < 0).sum()} / {len(df)} ({100*(df['own_diff_vs_field'] < 0).mean():.1f}%)")
    
    print(f"\nWinner Total Own vs Cash Line:")
    print(f"  Mean difference: {df['own_diff_vs_cash'].mean():+.1f}%")
    
    # ===== ANALYSIS 3: By Contest Type =====
    print("\n" + "=" * 80)
    print("3. ANALYSIS BY CONTEST TYPE")
    print("=" * 80)
    
    for ctype in ['High Five', 'Hot Shot', 'Pick and Roll', 'Elbow Shot', 'Other']:
        subset = df[df['contest_type'] == ctype]
        if len(subset) < 3:
            continue
        print(f"\n{ctype} ({len(subset)} contests):")
        print(f"  Avg entries: {subset['total_entries'].mean():.0f}")
        print(f"  Winner avg ownership: {subset['winner_avg_own'].mean():.1f}%")
        print(f"  Winner total ownership: {subset['winner_total_own'].mean():.1f}%")
        print(f"  Winners below field avg: {(subset['own_diff_vs_field'] < 0).mean()*100:.1f}%")
        print(f"  Avg low-owned players (<10%): {subset['winner_num_under_10'].mean():.1f}")
    
    # ===== ANALYSIS 4: By Contest Size (Entry Count) =====
    print("\n" + "=" * 80)
    print("4. ANALYSIS BY CONTEST SIZE")
    print("=" * 80)
    
    size_bins = [(0, 500), (500, 2000), (2000, 10000), (10000, float('inf'))]
    size_labels = ['Small (<500)', 'Medium (500-2K)', 'Large (2K-10K)', 'Massive (10K+)']
    
    for (lo, hi), label in zip(size_bins, size_labels):
        subset = df[(df['total_entries'] >= lo) & (df['total_entries'] < hi)]
        if len(subset) < 3:
            continue
        print(f"\n{label} ({len(subset)} contests):")
        print(f"  Winner avg ownership: {subset['winner_avg_own'].mean():.1f}%")
        print(f"  Winner total ownership: {subset['winner_total_own'].mean():.1f}%")
        print(f"  Avg low-owned players (<10%): {subset['winner_num_under_10'].mean():.1f}")
        print(f"  Avg very low-owned (<5%): {subset['winner_num_under_5'].mean():.1f}")
        print(f"  Winning score avg: {subset['winner_points'].mean():.1f}")
    
    # ===== ANALYSIS 5: Optimal Ownership Band Distribution =====
    print("\n" + "=" * 80)
    print("5. OPTIMAL OWNERSHIP COMPOSITION (Winners)")
    print("=" * 80)
    
    # Distribution of leverage plays
    print("\nNumber of players under 10% owned in winning lineups:")
    for n in range(9):
        pct = (df['winner_num_under_10'] == n).mean() * 100
        print(f"  {n} players: {pct:.1f}%")
    
    print("\nNumber of players under 5% owned in winning lineups:")
    for n in range(9):
        pct = (df['winner_num_under_5'] == n).mean() * 100
        if pct > 0:
            print(f"  {n} players: {pct:.1f}%")
    
    # ===== ANALYSIS 6: Scoring Benchmarks =====
    print("\n" + "=" * 80)
    print("6. SCORING BENCHMARKS")
    print("=" * 80)
    
    print(f"\nWinning Scores:")
    print(f"  Mean: {df['winner_points'].mean():.1f}")
    print(f"  Median: {df['winner_points'].median():.1f}")
    print(f"  Min: {df['winner_points'].min():.1f}")
    print(f"  Max: {df['winner_points'].max():.1f}")
    
    print(f"\nCash Line Scores (top 20%):")
    print(f"  Mean: {df['cash_avg_points'].mean():.1f}")
    
    print(f"\nTop 1% Scores:")
    print(f"  Mean: {df['top1pct_avg_points'].mean():.1f}")
    
    # ===== KEY TAKEAWAYS =====
    print("\n" + "=" * 80)
    print("KEY TAKEAWAYS FOR GPP CONSTRUCTION")
    print("=" * 80)
    
    print("""
Based on the analysis above, here are the key insights:

1. WINNING OWNERSHIP PROFILE:
   - Winners tend to have {:.0f}-{:.0f}% total lineup ownership (sum of 8 players)
   - That's an average of {:.1f}% per player
   
2. LEVERAGE IS CRITICAL:
   - Winning lineups typically have {:.1f} players under 10% owned
   - About {:.1f}% of winners have LOWER total ownership than the field average
   
3. CONTRARIAN APPROACH:
   - Look for 2-3 true differentiators (under 10% owned)
   - 1-2 players under 5% owned is common in winning lineups
   
4. BALANCE IS KEY:
   - Don't go all chalk or all contrarian
   - Mix solid plays (30-50% owned) with leverage (under 10%)
    """.format(
        df['winner_total_own'].quantile(0.25),
        df['winner_total_own'].quantile(0.75),
        df['winner_avg_own'].mean(),
        df['winner_num_under_10'].mean(),
        (df['own_diff_vs_field'] < 0).mean() * 100,
    ))
    
    # Save results to parquet for further analysis
    output_path = Path.home() / "projections-data" / "bronze" / "dk_contests" / "contest_analysis.parquet"
    df.to_parquet(output_path, index=False)
    print(f"\nFull results saved to: {output_path}")


if __name__ == "__main__":
    main()
