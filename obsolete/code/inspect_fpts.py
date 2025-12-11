import pandas as pd
from pathlib import Path

# Find latest run for today
daily_root = Path("artifacts/minutes_v1/daily/2025-11-24")
if not daily_root.exists():
    print("No data for today")
    exit()

# Just pick the first run dir found
run_dirs = sorted([d for d in daily_root.iterdir() if d.is_dir() and d.name.startswith("run=")])
if not run_dirs:
    print("No runs found")
    exit()

latest_run = run_dirs[-1]
# Check for fpts parquet in the fpts root?
# minutes_api.py says: DEFAULT_FPTS_ROOT = paths.data_path("gold", "projections_fpts_v1")
# But it also looks for fpts in the run dir?
# Line 138: parquet_path = run_dir / FPTS_FILENAME
# Wait, _load_fpts uses `root` which defaults to DEFAULT_FPTS_ROOT.
# It constructs path: root / day / run=... / fpts.parquet

# Let's try to find where fpts are.
# I'll check the minutes_api logic again.
# It seems it looks in a different root.

# I'll search for fpts.parquet in the whole project to find a sample.
pass
