# scripts/probe_contests.py (Playwright-based endpoint probe)
import os, sys, json, argparse, pprint, urllib.parse, datetime as dt, csv
from playwright.sync_api import sync_playwright

# Defaults can still be provided via env vars
BASE   = os.environ.get("LS_BASE", "https://<linestar-base-domain>")
PATH   = os.environ.get("LS_CONTESTS_PATH", "/DesktopModules/DailyFantasyApi/API/Fantasy/GetPeriodInformation")
SITE   = os.environ.get("LS_SITE", "dk")   # dk or fd
DATE   = os.environ.get("LS_DATE", dt.date.today().isoformat())
STATE  = os.environ.get("LS_STATE", "storage_state.json")  # Playwright storage

# ---------------- helpers ----------------

def parse_params(s: str):
    """Parse query string params passed as 'a=1&b=2' or 'a=1,b=2'."""
    if not s:
        return {}
    if "," in s and "&" not in s:
        s = s.replace(",", "&")
    parsed = urllib.parse.parse_qs(s, keep_blank_values=True)
    return {k: v[0] if isinstance(v, list) else v for k, v in parsed.items()}


def maybe_json(text: str):
    try:
        return json.loads(text)
    except Exception:
        return None


def normalize_info(info: dict, period_id, site, sport=None):
    # Map many possible key names into our schema
    def pick(d, *keys, default=None):
        for k in keys:
            if k in d and d[k] is not None:
                return d[k]
        return default
    def parse_money(x):
        if x is None: return None
        if isinstance(x, (int, float)): return float(x)
        s = str(x).strip().replace("$","{}").replace(",","{}").format("","")
        try: return float(s)
        except: return None
    def cents_to_dollars(x):
        try: return round(float(x)/100.0, 2)
        except: return None

    cid   = pick(info, "ContestId","Id","DailyContestId")
    name  = pick(info, "ContestName","Name","Title")
    entries = pick(info, "EntryCount","Entries","TotalEntries")
    maxent  = pick(info, "EntryCap","MaxEntries","MaxEntry")
    fee     = pick(info, "EntryFee","EntryFeeDollars")
    feec    = pick(info, "EntryFeeCents","EntryFeeInCents")
    rake    = pick(info, "Rake","RakePct","RakePercent")
    prize   = pick(info, "PrizePool","TotalPayout","PayoutsTotal")
    start   = pick(info, "StartTime","StartDate","ContestStart")

    if fee is None and feec is not None:
        fee = cents_to_dollars(feec)
    else:
        fee = parse_money(fee)
    if isinstance(rake, str) and rake.endswith("%"):
        try: rake = float(rake[:-1])
        except: pass
    if isinstance(rake, (int,float)):
        rake = float(rake)

    return {
        "site": site, "sport": sport, "period_id": period_id,
        "contest_id": cid, "contest_name": name,
        "entry_count": entries, "entry_cap": maxent,
        "entry_fee": fee, "rake_pct": rake, "prize_pool": prize,
        "start_time": start,
    }

# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser(description="Probe a JSON endpoint with Playwright's request context (uses storage_state cookies).")
    ap.add_argument("--url", help="Full URL to hit. If absent, BASE+PATH are used.")
    ap.add_argument("--base", default=BASE, help="Base URL (if --url not set)")
    ap.add_argument("--path", default=PATH, help="Path (if --url not set)")
    ap.add_argument("--params", default="", help="Query params 'a=1&b=2' or 'a=1,b=2'.")
    ap.add_argument("--method", default="GET", choices=["GET","POST"], help="HTTP method")
    ap.add_argument("--data", default="", help="POST body as JSON string or 'a=1&b=2'.")
    ap.add_argument("--site", default=SITE, help="Shortcut param to include (dk/fd)")
    ap.add_argument("--date", default=DATE, help="Shortcut param to include (YYYY-MM-DD)")
    ap.add_argument("--state", default=STATE, help="Path to Playwright storage_state.json for auth cookies")
    ap.add_argument("--no-auth", action="store_true", help="Do not attach storage_state (anonymous request)")
    ap.add_argument("--verbose", "-v", action="count", default=0, help="-v show sample; -vv dump full JSON")
    ap.add_argument("--dump", help="Write full response body to this file")
    ap.add_argument("--as-contests", help="If set, map common 'Info' fields to contests.csv at this path.")
    args = ap.parse_args()

    # Build URL and params
    url = args.url or (args.base.rstrip("/") + "/" + args.path.lstrip("/"))
    q = parse_params(args.params)
    q.setdefault("site", args.site)
    q.setdefault("date", args.date)
    q.setdefault("slateDate", args.date)

    with sync_playwright() as p:
        rc = p.request.new_context(
            storage_state=(None if args.no_auth else args.state),
            extra_http_headers={
                "Accept": "application/json, text/plain, */*",
                "User-Agent": "Mozilla/5.0"
            }
        )
        # Prepare body if POST
        data = None
        json_body = None
        if args.method == "POST" and args.data:
            try:
                json_body = json.loads(args.data)
            except Exception:
                # form-style fallback
                data = parse_params(args.data)

        # Execute
        if args.method == "GET":
            r = rc.get(url, params=q, timeout=30_000)
        else:
            r = rc.post(url, params=q, data=data, json=json_body, timeout=30_000)

        status = r.status
        ctype = r.headers.get("content-type", "")
        text = r.text()
        print("status:", status)
        print("final url:", r.url)
        print("content-type:", ctype)

        if args.dump:
            with open(args.dump, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"wrote body to {args.dump}")

        js = None
        if "json" in ctype.lower():
            try:
                js = r.json()
            except Exception:
                js = maybe_json(text)

        if js is not None:
            if isinstance(js, list):
                print(f"json: list[{len(js)}]")
                if args.verbose:
                    pprint.pp(js[:3], width=120, compact=True)
            elif isinstance(js, dict):
                keys = list(js.keys())
                print(f"json: dict keys={keys[:20]}")
                for k in ("Contests","ContestList","ContestSummaries","ContestData","OwnershipData","Salaries","items","data","rows"):
                    if k in js and isinstance(js[k], list):
                        print(f"{k}: list[{len(js[k])}] (showing up to 2)")
                        pprint.pp(js[k][:2], width=120, compact=True)
                if args.verbose >= 2:
                    print("\nFULL JSON:\n"); pprint.pp(js, width=120, compact=False)

                # Optional contests CSV mapping
                if args.as_contests:
                    period_id = q.get("periodId") or q.get("period_id") or q.get("slateId") or q.get("slate")
                    site = q.get("site")
                    sport = q.get("sport")
                    rows = []
                    # Case A: bundle with Info (single contest)
                    if isinstance(js.get("Info"), dict):
                        rows.append(normalize_info(js["Info"], period_id, site, sport))
                    # Case B: explicit list of contests
                    for key in ("Contests","ContestList","ContestSummaries","ContestData"):
                        if key in js and isinstance(js[key], list):
                            for c in js[key]:
                                if isinstance(c, dict):
                                    rows.append(normalize_info(c, period_id, site, sport))
                    if rows:
                        cols = ["site","sport","period_id","contest_id","contest_name","entry_count","entry_cap","entry_fee","rake_pct","prize_pool","start_time"]
                        with open(args.as_contests, "w", newline="") as f:
                            w = csv.DictWriter(f, fieldnames=cols)
                            w.writeheader(); w.writerows(rows)
                        print(f"wrote contests csv: {args.as_contests} ({len(rows)} rows)")
                    else:
                        print("[map] no recognizable contest structures found to save.")
            else:
                print("json: (parsed) but not list/dict, type=", type(js))
        else:
            print("\nTEXT SNIPPET:\n", text[:500])

if __name__ == "__main__":
    main()