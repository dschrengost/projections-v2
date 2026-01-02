
import os
from pathlib import Path

path = Path("/home/daniel/projections-data/bronze")
print(f"Checking {path}")

if path.exists():
    # Look for contest results
    print("\nsubdirectories:")
    for p in path.iterdir():
        if p.is_dir():
            print(f"DIR: {p.name}")
        else:
            if "contest" in p.name.lower() or "result" in p.name.lower():
                print(f"FILE: {p.name}")

    # Inspect dk_contests recursively
    contest_dir = path / "dk_contests"
    if contest_dir.exists():
        print(f"\nSearching {contest_dir} recursively...")
        # Get first 3 subdirs if any
        subdirs = [x for x in contest_dir.iterdir() if x.is_dir()]
        print(f"Subdirectories: {[d.name for d in subdirs[:5]]}")
        
        # Find any csv
        files = list(contest_dir.rglob("*.csv"))
        print(f"Found {len(files)} CSV files")
        
        if files:
            sample = files[0]
            print(f"\nReading header of {sample.relative_to(path)}:")
            import pandas as pd
            try:
                df = pd.read_csv(sample, nrows=5)
                print(df.columns.tolist())
                print(df.head(2))
            except Exception as e:
                print(f"Error reading CSV: {e}")

    # Inspect Standings SCV
    standings_path = path / "dk_contests/nba_gpp_data/2025-11-17/results/contest_185010975_standings.csv"
    if standings_path.exists():
        print(f"\nReading Standings: {standings_path.relative_to(path)}")
        try:
            df = pd.read_csv(standings_path, nrows=5)
            print(f"Columns: {df.columns.tolist()}")
            print(df[["Rank", "EntryName", "Points", "Lineup"]].head(2))
        except Exception as e:
            print(f"Error: {e}")
        print(f"\nSearching {dl_dir} recursively...")
        files = list(dl_dir.rglob("*.csv"))
        print(f"Found {len(files)} CSV files in daily_lineups")
        if files:
            sample = files[0]
            print(f"Reading header of {sample.relative_to(path)}:")
            try:
                df = pd.read_csv(sample, nrows=5)
                print(df.columns.tolist())
            except: pass

    # General search for "result" or "standings"
    print("\nSearching for 'standings' or 'result' in filenames...")
    for p in path.rglob("*standings*.csv"):
        print(f"Found Standings: {p.relative_to(path)}")
        break # just show one
    for p in path.rglob("*result*.csv"):
        print(f"Found Result: {p.relative_to(path)}")
        if "contest" in p.name:
             # Peak this one
             try:
                 df = pd.read_csv(p, nrows=2)
                 print(f"Columns: {df.columns.tolist()}")
             except: pass
        break

else:
    print("Path does not exist")
