
import pandas as pd
import numpy as np
from pathlib import Path
import os

# Path for 2025-12-14
date = "2025-12-14"
base_dir = Path(f"/home/daniel/projections-data/artifacts/sim_v2/worlds_fpts_v2/game_date={date}")

print(f"Checking directory: {base_dir}")
if not base_dir.exists():
    print("Directory does not exist!")
    exit()

print("Files in directory:")
for f in base_dir.iterdir():
    print(f" - {f.name}")

# Check projections.parquet if it exists (source of synthetic worlds)
proj_path = base_dir / "projections.parquet"
if proj_path.exists():
    print(f"\nLoading {proj_path}...")
    df = pd.read_parquet(proj_path)
    print("Columns:", df.columns)
    
    # Check mean and std
    if "dk_fpts_mean" in df.columns:
        print("\nChecking dk_fpts_mean:")
        print(df["dk_fpts_mean"].describe())
        
        # Check for huge means
        bad_means = df[df["dk_fpts_mean"] > 1000]
        if not bad_means.empty:
            print("ALERT: Huge means found!")
            print(bad_means[["player_name", "dk_fpts_mean"]])
            
    if "dk_fpts_std" in df.columns:
        print("\nChecking dk_fpts_std:")
        print(df["dk_fpts_std"].describe())

# Check for individual world files
world_files = list(base_dir.glob("world=*.parquet"))
if world_files:
    print(f"\nFound {len(world_files)} world files. Inspecting first one...")
    w_df = pd.read_parquet(world_files[0])
    print(w_df.describe())
