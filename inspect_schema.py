import sys
import pandas as pd
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python inspect_schema.py <parquet_path>")
    sys.exit(1)

path = Path(sys.argv[1])
if not path.exists():
    print(f"File not found: {path}")
    sys.exit(1)

try:
    df = pd.read_parquet(path)
    print(f"Scanning {path.name}...")
    print(f"Shape: {df.shape}")
    print("Columns:")
    for col in sorted(df.columns):
        dtype = df[col].dtype
        print(f"- {col} ({dtype})")
except Exception as e:
    print(f"Error reading {path}: {e}")
