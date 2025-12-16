
import pandas as pd
import numpy as np
from pathlib import Path

# Path to worlds matrix for 2025-12-14
date = "2025-12-14"
path = Path(f"/home/daniel/projections-data/artifacts/sim_v2/worlds_fpts_v2/game_date={date}/worlds_matrix.parquet")

if not path.exists():
    print(f"File not found: {path}")
    # Try finding where it might be
    base = Path("/home/daniel/projects/projections-v2/artifacts/sim_v2/worlds_fpts_v2")
    print(f"Searching in {base}...")
    for p in base.rglob("worlds_matrix.parquet"):
        print(f"Found: {p}")
else:
    print(f"Loading {path}...")
    df = pd.read_parquet(path)
    print(f"Shape: {df.shape}")
    
    # Check for infinity or huge values
    vals = df.values
    max_val = np.max(vals)
    print(f"Max value in matrix: {max_val}")
    
    if max_val > 1000:
        print("ALERT: Huge values found!")
        # Find columns with huge values
        max_per_col = df.max()
        bad_cols = max_per_col[max_per_col > 1000]
        print("Columns with values > 1000:")
        print(bad_cols)
