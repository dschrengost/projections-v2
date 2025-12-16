# scrapers/linestar/list_pids.py
import argparse, json, pathlib
from datetime import datetime
from playwright.sync_api import sync_playwright

BASE = pathlib.Path(__file__).parent
STATE = BASE / "storage_state.json"
if not STATE.exists():
    alt = BASE.parent / "storage_state.json"
    if alt.exists(): STATE = alt

API = "https://www.linestarapp.com/DesktopModules/DailyFantasyApi/API/Fantasy/GetSalariesV5"
SITE_MAP = {"dk": 1, "draftkings": 1, "fd": 2, "fanduel": 2}
SPORT_MAP = {"nba": 2}

def main(site: str, sport: str, show: int):
    site_id = SITE_MAP[site.lower()]
    sport_id = SPORT_MAP[sport.lower()]

    with sync_playwright() as p:
        rc = p.request.new_context(storage_state=str(STATE) if STATE.exists() else None)
        # Intentionally omit periodId so LS returns the default slate PLUS a 'Periods' list
        r = rc.get(API, params={"site": site_id, "sport": sport_id})
        if r.status != 200:
            raise SystemExit(f"HTTP {r.status}: {r.text()[:200]}")
        try:
            data = r.json()
        except Exception:
            data = json.loads(r.text())

    periods = data.get("Periods") or []
    print(f"Total periods returned: {len(periods)}")

    # Show a quick schema peek
    if periods:
        print("Period keys example:", list(periods[0].keys()))

    # Normalize a few common fields if present
    def norm_period(p):
        pid = p.get("Id") or p.get("PeriodId") or p.get("PID")
        label = p.get("Name") or p.get("Label") or p.get("Text") or ""
        # try to find a date-like field
        dt_raw = p.get("Date") or p.get("Start") or p.get("GameDate") or ""
        # parse /Date(…)/ style if present
        iso = ""
        if isinstance(dt_raw, str) and dt_raw.startswith("/Date("):
            # /Date(1750626000000-0400)/
            import re, datetime as dt
            m = re.search(r"/Date\((\d+)(?:[+-]\d+)?\)/", dt_raw)
            if m:
                ts_ms = int(m.group(1))
                iso = dt.datetime.utcfromtimestamp(ts_ms/1000.0).isoformat() + "Z"
        elif isinstance(dt_raw, str):
            iso = dt_raw
        return {
            "pid": pid,
            "label": label,
            "date": iso,
            "raw": p,
        }

    rows = [norm_period(p) for p in periods]
    # Show the latest N (as they arrive from server order)
    for r in rows[:show]:
        print(f"PID={r['pid']}  date={r['date']}  label={r['label']}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="dk", help="dk or fd (default: dk)")
    ap.add_argument("--sport", default="nba", help="default: nba")
    ap.add_argument("--show", type=int, default=20, help="how many to print (default 20)")
    args = ap.parse_args()
    if args.site.lower() not in SITE_MAP: raise SystemExit("site must be dk or fd")
    if args.sport.lower() not in SPORT_MAP: raise SystemExit("sport must be nba")
    main(args.site, args.sport, args.show)