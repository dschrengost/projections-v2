#!/usr/bin/env python3
"""
Backfill contest metadata for season_2025 slates.
The GetSalariesV5 API returns ContestResults which contain full contest metadata.
"""
import json, pathlib, sys, time, csv
from playwright.sync_api import sync_playwright

BACKFILL_DIR = pathlib.Path(__file__).parent / "backfill" / "season_2025" / "dk"
STATE_PATH = pathlib.Path(__file__).parent / "storage_state.json"

API = "https://www.linestarapp.com/DesktopModules/DailyFantasyApi/API/Fantasy/GetSalariesV5"
SITE_ID = 1  # DraftKings
SPORT_ID = 2  # NBA

def fetch_contests(rc, pid):
    """Fetch ownership data and extract contest metadata from ContestResults."""
    r = rc.get(API, params={"periodId": pid, "site": SITE_ID, "sport": SPORT_ID}, timeout=30000)
    if r.status != 200:
        return []
    
    try:
        data = r.json()
    except:
        return []
    
    if not isinstance(data, dict):
        return []
    
    # Get ownership data
    own = data.get("Ownership") or {}
    contest_results = own.get("ContestResults") or []
    
    contests = []
    seen_ids = set()
    
    for cr in contest_results:
        c = cr.get("Contest") or {}
        if not c:
            continue
            
        cid = c.get("Id") or c.get("ContestId")
        if cid and cid not in seen_ids:
            contests.append({
                "period_id": pid,
                "contest_id": cid,
                "contest_name": c.get("ContestName"),
                "contest_type": c.get("ContestType"),
                "entries": c.get("EntryCount"),
                "buyin": c.get("EntryFee"),
                "purse": c.get("Purse"),
            })
            seen_ids.add(cid)
    
    return contests

def main():
    # Find all PIDs in season_2025
    pids = set()
    for f in BACKFILL_DIR.glob("pid_*_actual.csv"):
        pid = f.stem.replace("pid_", "").replace("_actual", "")
        pids.add(int(pid))
    pids = sorted(pids)
    
    # Filter to those missing contest files
    missing = [p for p in pids if not (BACKFILL_DIR / f"pid_{p}_contests.csv").exists()]
    
    print(f"Found {len(pids)} total 2025 slates, {len(missing)} missing contest metadata")
    
    if not missing:
        print("All contest files already exist!")
        return
    
    with sync_playwright() as p:
        rc = p.request.new_context(storage_state=str(STATE_PATH) if STATE_PATH.exists() else None)
        
        for i, pid in enumerate(missing):
            print(f"[{i+1}/{len(missing)}] Fetching contests for PID {pid}...", end=" ", flush=True)
            
            try:
                contests = fetch_contests(rc, pid)
                
                if contests:
                    out_path = BACKFILL_DIR / f"pid_{pid}_contests.csv"
                    cols = ["period_id", "contest_id", "contest_name", "contest_type", "entries", "buyin", "purse"]
                    with open(out_path, "w", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=cols)
                        w.writeheader()
                        w.writerows(contests)
                    print(f"OK ({len(contests)} contests)")
                else:
                    print("No contests found")
                    
            except Exception as e:
                print(f"Error: {e}")
            
            time.sleep(0.5)  # Be polite
    
    print("\nDone!")

if __name__ == "__main__":
    main()
