import pandas as pd
from pathlib import Path
from projections import paths

def check_labels():
    features_root = paths.data_path("gold", "features_minutes_v1")
    files = sorted(features_root.rglob("*.parquet"))
    
    print(f"Checking {len(files)} files in {features_root}")
    
    total_rows = 0
    nan_rows = 0
    
    for f in files:
        df = pd.read_parquet(f)
        total_rows += len(df)
        
        if "minutes" not in df.columns:
            print(f"[WARN] 'minutes' column missing in {f}")
            continue
            
        nans = df["minutes"].isna().sum()
        if nans > 0:
            print(f"[FAIL] {f} has {nans} NaN labels!")
            print(df[df["minutes"].isna()][["season", "game_date", "player_name", "minutes"]].head())
            nan_rows += nans
            
    print(f"Total rows: {total_rows}")
    print(f"Total NaN labels: {nan_rows}")

if __name__ == "__main__":
    check_labels()
