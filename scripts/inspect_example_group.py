
import pandas as pd
from pathlib import Path

def main():
    # Pick a specific file we know exists
    path = Path("/home/daniel/projections-data/gold/training/player_game_features/season=2023/game_date=2024-02-08/asof=lock/features.parquet")
    
    if not path.exists():
        print(f"Path not found: {path}")
        return

    df = pd.read_parquet(path)
    
    # Pick first game/home team
    if df.empty:
        print("Empty DataFrame")
        return
        
    game_id = df["game_id"].iloc[0]
    team_id = df["team_id"].iloc[0]
    
    # Filter
    mask = (df["game_id"] == game_id) & (df["team_id"] == team_id)
    group = df[mask]
    
    print(f"Row Count: {len(group)}")
    print(f"Unique Player Count: {group['player_id'].nunique()}")
    
    # Find duplicates
    dupes = group[group.duplicated(subset=["player_id"], keep=False)]
    if not dupes.empty:
        print("\nDuplicates found! Examples:")
        sample_pid = dupes["player_id"].iloc[0]
        sample_rows = dupes[dupes["player_id"] == sample_pid]
        print(f"Player ID: {sample_pid}")
        # Transpose to see diffs
        print(sample_rows.T)
        
        # Identify columns that differ
        nunique = sample_rows.nunique()
        diff_cols = nunique[nunique > 1].index.tolist()
        print(f"\nDiffering Columns: {diff_cols}")
    else:
        print("No duplicates found.")

if __name__ == "__main__":
    main()
