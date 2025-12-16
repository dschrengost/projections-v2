# scrapers/linestar/get_contests.py
import json, pathlib, sys, re
from playwright.sync_api import sync_playwright

PID  = int(sys.argv[1]) if len(sys.argv) > 1 else 2395  # periodId (slate)
SITE = int(sys.argv[2]) if len(sys.argv) > 2 else 1     # DK=1
SPORT= int(sys.argv[3]) if len(sys.argv) > 3 else 2     # NBA=2

HOST = "https://www.linestarapp.com"
BASE_API = f"{HOST}/DesktopModules/DailyFantasyApi/API/Fantasy"

STATE_PATH = pathlib.Path(__file__).parent / "storage_state.json"
if not STATE_PATH.exists():
    alt = pathlib.Path(__file__).parent.parent / "storage_state.json"
    if alt.exists(): STATE_PATH = alt

def pick(d, *keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default

def parse_money(x):
    if x is None: return None
    if isinstance(x, (int, float)): return float(x)
    s = str(x).strip()
    s = s.replace("$","").replace(",","")
    try: return float(s)
    except: return None

def cents_to_dollars(x):
    try: return round(float(x)/100.0, 2)
    except: return None

def normalize_info(info, period_id, site, sport):
    # Map many possible key names into our schema
    cid   = pick(info, "ContestId","Id","DailyContestId")
    name  = pick(info, "ContestName","Name","Title")
    entries = pick(info, "EntryCount","Entries","TotalEntries")
    maxent  = pick(info, "EntryCap","MaxEntries","MaxEntry")
    fee     = pick(info, "EntryFee","EntryFeeDollars")
    feec    = pick(info, "EntryFeeCents","EntryFeeInCents")
    rake    = pick(info, "Rake","RakePct","RakePercent")
    prize   = pick(info, "PrizePool","TotalPayout","PayoutsTotal")
    start   = pick(info, "StartTime","StartDate","ContestStart")

    # Normalize units
    if fee is None and feec is not None:
        fee = cents_to_dollars(feec)
    else:
        fee = parse_money(fee)
    if rake is not None and isinstance(rake, str) and rake.endswith("%"):
        try: rake = float(rake[:-1])
        except: pass
    if isinstance(rake, (int,float)): rake = float(rake)

    return {
        "site": site, "sport": sport, "period_id": period_id,
        "contest_id": cid, "contest_name": name,
        "entry_count": entries, "entry_cap": maxent,
        "entry_fee": fee, "rake_pct": rake, "prize_pool": prize,
        "start_time": start
    }

def get_contests_via_candidates(rc, period_id, site, sport):
    params = {"periodId": period_id, "site": site, "sport": sport}
    candidates = [
        "GetPeriodInformation",
        "GetContestOwnership",          # some variants don’t need contestId to list
        "GetContestOwnershipV2",
        "GetSiteContests",              # if present
        "GetOwnershipPageData"          # catch-all some builds use
    ]
    found = []
    for ep in candidates:
        url = f"{BASE_API}/{ep}"
        r = rc.get(url, params=params)
        if r.status != 200:
            continue
        try:
            js = r.json()
        except:
            try: js = json.loads(r.text())
            except: continue

        # Case 1: explicit list of contests
        for key in ("Contests","ContestList","ContestSummaries","ContestData"):
            if isinstance(js, dict) and key in js and isinstance(js[key], list):
                for c in js[key]:
                    info = c if isinstance(c, dict) else {}
                    norm = normalize_info(info, period_id, site, sport)
                    if norm["contest_id"] is not None:
                        found.append(norm)

        # Case 2: Ownership bundle with Info for a single contest
        if isinstance(js, dict) and "Info" in js and isinstance(js["Info"], dict):
            norm = normalize_info(js["Info"], period_id, site, sport)
            if norm["contest_id"] is not None:
                found.append(norm)

    # Deduplicate by contest_id
    dedup = {}
    for row in found:
        cid = row["contest_id"]
        if cid is not None:
            dedup[cid] = {**dedup.get(cid, {}), **row}
    return list(dedup.values())

def fill_info_via_ownership(rc, contests, period_id, site, sport):
    """If we only got contest IDs/names without counts/fees, hit ownership endpoint per contest to enrich Info."""
    enriched = []
    for c in contests:
        cid = c.get("contest_id")
        if not cid:
            continue
        # Try common ownership endpoints
        for ep in ("GetContestOwnership","GetContestOwnershipV2","GetOwnership"):
            url = f"{BASE_API}/{ep}"
            r = rc.get(url, params={"periodId": period_id, "site": site, "sport": sport, "contestId": cid})
            if r.status != 200:
                continue
            try:
                js = r.json()
            except:
                continue
            if isinstance(js, dict) and "Info" in js and isinstance(js["Info"], dict):
                info = normalize_info(js["Info"], period_id, site, sport)
                # merge
                merged = {**c, **{k:v for k,v in info.items() if v is not None}}
                enriched.append(merged)
                break
        else:
            enriched.append(c)
    return enriched

def main():
    with sync_playwright() as p:
        rc = p.request.new_context(storage_state=str(STATE_PATH) if STATE_PATH.exists() else None)

        contests = get_contests_via_candidates(rc, PID, SITE, SPORT)
        if not contests:
            print("No contests found via candidate endpoints. If you’re logged in, try scraping the dropdown (Approach B).")
            return

        # Enrich with per-contest Info if needed
        have_counts = any(c.get("entry_count") is not None for c in contests)
        if not have_counts:
            contests = fill_info_via_ownership(rc, contests, PID, SITE, SPORT)

        # Print + save CSV
        import csv
        cols = ["site","sport","period_id","contest_id","contest_name",
                "entry_count","entry_cap","entry_fee","rake_pct","prize_pool","start_time"]
        contests = [c for c in contests if c.get("contest_id") is not None]
        contests.sort(key=lambda x: (x.get("start_time") or "", str(x["contest_id"])))
        with open("contests.csv","w",newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader(); w.writerows(contests)

        print(f"Wrote contests.csv with {len(contests)} contests.")
        for c in contests[:5]:
            print("-", c["contest_id"], "|", c.get("contest_name"), "| entries:", c.get("entry_count"), "| fee:", c.get("entry_fee"))

if __name__ == "__main__":
    main()