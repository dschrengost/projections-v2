
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

def main():
    path = Path("/home/daniel/projections-data/gold/training/player_game_features")
    files = sorted(path.rglob("*.parquet"))
    print(f"Found {len(files)} parquet files.")
    
    if not files:
        return

    # Read all files, game_date only
    frames = []
    print("Reading all files...")
    for f in files:
        try:
            frames.append(pd.read_parquet(f, columns=["game_date", "minutes"]))
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    if not frames:
        print("No frames loaded.")
        return

    df = pd.concat(frames)
    print("Total rows:", len(df))
    print("Min date:", df["game_date"].min())
    print("Max date:", df["game_date"].max())
    
    # Check distribution
    print("Dates per year:")
    print(df["game_date"].dt.year.value_counts())
    
    # Check specific range
    start_date = pd.Timestamp("2023-10-24", tz="UTC")
    end_date = pd.Timestamp("2024-02-18", tz="UTC")
    
    dt_utc = pd.to_datetime(df["game_date"], utc=True)
    mask = (dt_utc >= start_date) & (dt_utc <= end_date)
    print(f"Rows in Fold 1 Range ({start_date.date()} to {end_date.date()}): {mask.sum()}")
    
    if "minutes" in df.columns:
        train_rows = df[mask]
        print("Minutes in Fold 1 stats:")
        print(train_rows["minutes"].describe())
        print(f"Non-zero minutes: {(train_rows['minutes'] > 0).sum()}")
        
if __name__ == "__main__":
    main()
