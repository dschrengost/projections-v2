
import os
from pathlib import Path

path = Path("/home/daniel/projections-data/silver")
print(f"Checking {path}")

if path.exists():
    sched_dir = path / "schedule"
    if sched_dir.exists():
        print(f"\nListing {sched_dir}:")
        for x in sched_dir.iterdir():
            print(f" - {x.name}")
        print(f"Found {len(files)} parquet files")
        if files:
            sample = files[0]
            print(f"Reading {sample.name}:")
            import pandas as pd
            try:
                df = pd.read_parquet(sample)
                print(df.columns.tolist())
                print(df.head(2))
                
                # Check distinct games per date
                if "game_date" in df.columns:
                    print("\nGames per date:")
                    print(df.groupby("game_date").size().head())
            except Exception as e:
                print(e)
else:
    print("Path does not exist")
