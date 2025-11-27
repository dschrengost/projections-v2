import pandas as pd
from pathlib import Path

# Use the discovered data root
data_root = Path("/home/daniel/projections-data")
fpts_root = data_root / "gold/projections_fpts_v1"

# Find latest run for today
daily_root = fpts_root / "2025-11-24"
if not daily_root.exists():
    print(f"No data for today in {daily_root}")
    exit()

# Just pick the first run dir found
run_dirs = sorted([d for d in daily_root.iterdir() if d.is_dir() and d.name.startswith("run=")])
if not run_dirs:
    print("No runs found")
    exit()

latest_run = run_dirs[-1]
parquet_path = latest_run / "fpts.parquet"

if not parquet_path.exists():
    print("No fpts parquet found")
    exit()

df = pd.read_parquet(parquet_path)
print("Columns:")
for col in sorted(df.columns):
    print(f"- {col}")

# Check for vegas specific keywords
keywords = ["vegas", "spread", "total", "line", "implied"]
print("\nPotential Vegas columns:")
for col in df.columns:
    if any(k in col.lower() for k in keywords):
        print(f"- {col}: {df[col].iloc[0] if not df.empty else 'Empty'}")
