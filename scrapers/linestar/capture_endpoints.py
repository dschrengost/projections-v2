import json, time, hashlib, pathlib, csv
from datetime import datetime
from urllib.parse import urlparse
from playwright.sync_api import sync_playwright

SPORT = "NBA"
SITE  = "DraftKings"   # or "FanDuel"; this is written to contests.csv
PID   = "1981"         # Period/Slate Id (edit this per slate)
BASE_DIR = pathlib.Path(__file__).parent
PROFILE_DIR = BASE_DIR / ".pw_linestar_profile"  # same dir used during login
OWNERSHIP_URL = f"https://linestarapp.com/Ownership/Sport/{SPORT}/Site/{SITE}/PID/{PID}"

# ---------- helpers ----------

def fname(url: str) -> str:
    return hashlib.sha1(url.encode()).hexdigest()[:16] + ".json"

def _pick(d, *keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default

def _parse_money(x):
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().replace("$", "").replace(",", "")
    try:
        return float(s)
    except Exception:
        return None

def _cents_to_dollars(x):
    try:
        return round(float(x) / 100.0, 2)
    except Exception:
        return None

def normalize_contest_info(info: dict, period_id: str, site_label: str, sport_label: str):
    """Map many possible key names returned by different LS endpoints into our standard schema."""
    cid   = _pick(info, "ContestId", "Id", "DailyContestId")
    name  = _pick(info, "ContestName", "Name", "Title")
    entries = _pick(info, "EntryCount", "Entries", "TotalEntries")
    maxent  = _pick(info, "EntryCap", "MaxEntries", "MaxEntry")
    fee     = _pick(info, "EntryFee", "EntryFeeDollars")
    feec    = _pick(info, "EntryFeeCents", "EntryFeeInCents")
    rake    = _pick(info, "Rake", "RakePct", "RakePercent")
    prize   = _pick(info, "PrizePool", "TotalPayout", "PayoutsTotal")
    start   = _pick(info, "StartTime", "StartDate", "ContestStart")

    if fee is None and feec is not None:
        fee = _cents_to_dollars(feec)
    else:
        fee = _parse_money(fee)
    if isinstance(rake, str) and rake.endswith("%"):
        try:
            rake = float(rake[:-1])
        except Exception:
            pass
    if isinstance(rake, (int, float)):
        rake = float(rake)

    return {
        "site": site_label,
        "sport": sport_label,
        "period_id": period_id,
        "contest_id": cid,
        "contest_name": name,
        "entry_count": entries,
        "entry_cap": maxent,
        "entry_fee": fee,
        "rake_pct": rake,
        "prize_pool": prize,
        "start_time": start,
    }

# ---------- main ----------

def main():
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    outdir = BASE_DIR / "captures" / ts
    outdir.mkdir(parents=True, exist_ok=True)
    index_path = outdir / "index.jsonl"
    endpoints = {}
    contests = {}  # contest_id -> merged row

    with sync_playwright() as p:
        # Use the persisted Chrome profile so your LS login is present
        ctx = p.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE_DIR),
            channel="chrome",
            headless=False,
            record_har_path=str(outdir / "linestar.har"),
            record_har_omit_content=False,
            args=["--disable-blink-features=AutomationControlled"],
        )

        # Be polite: skip heavy assets
        def blocker(route):
            if route.request.resource_type in {"image", "font", "media"}:
                return route.abort()
            return route.continue_()
        ctx.route("**/*", blocker)

        page = ctx.new_page()

        # Capture JSON/XHR and extract contests when we see them
        def on_response(resp):
            try:
                url = resp.url or ""
                if not url:
                    return
                ct = (resp.headers.get("content-type") or "").lower()
                looks_api = ("linestarapp.com/DesktopModules/DailyFantasyApi/API" in url) or ("json" in ct) or url.endswith(".json")
                if not looks_api:
                    return

                # Try raw body first
                try:
                    body_bytes = resp.body()
                except Exception:
                    body_bytes = None

                data = None
                if body_bytes:
                    txt = body_bytes.decode("utf-8", errors="ignore")
                    try:
                        data = json.loads(txt)
                    except Exception:
                        data = {"text": txt}
                else:
                    try:
                        data = resp.json()
                    except Exception:
                        try:
                            data = {"text": resp.text()}
                        except Exception:
                            data = {"text": ""}

                base = urlparse(url)
                key = f"{base.scheme}://{base.netloc}{base.path}"
                endpoints[key] = endpoints.get(key, 0) + 1

                # Save every API-ish response
                with open(outdir / fname(url), "w", encoding="utf-8") as f:
                    json.dump({
                        "url": url,
                        "status": resp.status,
                        "method": resp.request.method if resp.request else "GET",
                        "headers": resp.headers,
                        "data": data
                    }, f, ensure_ascii=False, indent=2)
                with open(index_path, "a", encoding="utf-8") as idx:
                    idx.write(json.dumps({"url": url, "status": resp.status}) + "\n")

                # ---- Extract contest metadata on the fly ----
                if isinstance(data, dict):
                    # Case 1: ownership-like bundle with Info
                    if isinstance(data.get("Info"), dict):
                        row = normalize_contest_info(data["Info"], PID, SITE, SPORT)
                        cid = row.get("contest_id")
                        if cid is not None:
                            contests[cid] = {**contests.get(cid, {}), **{k: v for k, v in row.items() if v is not None}}
                    # Case 2: explicit contest lists
                    for key_list in ("Contests", "ContestList", "ContestSummaries", "ContestData"):
                        if isinstance(data.get(key_list), list):
                            for it in data[key_list]:
                                if isinstance(it, dict):
                                    row = normalize_contest_info(it, PID, SITE, SPORT)
                                    cid = row.get("contest_id")
                                    if cid is not None:
                                        contests[cid] = {**contests.get(cid, {}), **{k: v for k, v in row.items() if v is not None}}
            except Exception as e:
                try:
                    with open(outdir / "capture_errors.log", "a", encoding="utf-8") as ef:
                        ef.write(f"{datetime.utcnow().isoformat()} {e}\n")
                except Exception:
                    pass

        page.on("response", on_response)

        print("Opening Ownership page…")
        page.goto(OWNERSHIP_URL, wait_until="networkidle")
        print(f"Loaded: {page.url}")

        def _is_vis(locator):
            try:
                return locator.is_visible()
            except Exception:
                return False

        if _is_vis(page.get_by_text("Become a LineStar Member", exact=False)) or _is_vis(page.get_by_text("Sign In", exact=False)):
            print("⚠️ Logged out. In the Chrome window, click 'Sign In' → Continue with Google, complete login, and wait for the table.")
            input("Press ENTER here after you see the table… ")

        try:
            print("Title:", page.title())
        except Exception:
            pass

        page.wait_for_selector("table thead", timeout=15000)
        screenshot_path = outdir / "after_goto.png"
        page.screenshot(path=str(screenshot_path), full_page=True)
        print(f"Saved screenshot: {screenshot_path}")

        print("\nTip: change the contest dropdown to force ownership calls for each contest you care about.")
        print("When done, press ENTER here to finalize and save contests.csv …")
        input()

        ctx.close()

    # ---- Write contest outputs ----
    if contests:
        rows = list(contests.values())
        # Sort for readability (by start_time if present, else by contest_id)
        rows.sort(key=lambda r: (str(r.get("start_time") or ""), str(r.get("contest_id") or "")))
        cols = [
            "site", "sport", "period_id", "contest_id", "contest_name",
            "entry_count", "entry_cap", "entry_fee", "rake_pct", "prize_pool", "start_time"
        ]
        with open(outdir / "contests.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader(); w.writerows(rows)
        with open(outdir / "contests.json", "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        print(f"\n✅ Wrote {outdir / 'contests.csv'} with {len(rows)} contests.")
    else:
        print("\nℹ️ No recognizable contest metadata found in captured responses.")

    print("\nEndpoints seen:")
    for k, v in sorted(endpoints.items(), key=lambda x: -x[1]):
        print(f"{v:3}  {k}")
    print(f"\nSaved capture files in: {outdir}")

if __name__ == "__main__":
    main()