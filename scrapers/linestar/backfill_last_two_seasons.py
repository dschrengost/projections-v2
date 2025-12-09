# scrapers/linestar/backfill_last_two_seasons.py
import argparse, json, pathlib, random, time, re
from datetime import datetime, timezone
import pandas as pd
from playwright.sync_api import sync_playwright
from datetime import datetime, timezone, timedelta
import re

def parse_period_dt(pinfo):
    """Return a timezone-aware UTC datetime for the period start."""
    val = (pinfo.get("StartDateTicks") or pinfo.get("StartTicks") or
           pinfo.get("StartDate") or pinfo.get("Date") or pinfo.get("Start"))

    dt = None
    if isinstance(val, int):
        # .NET ticks? (100-ns since 0001-01-01)
        if val > 1e14:
            dt = datetime(1, 1, 1, tzinfo=timezone.utc) + timedelta(microseconds=val / 10)
        # UNIX ms
        elif val > 1e12:
            dt = datetime.fromtimestamp(val / 1000, tz=timezone.utc)
        # UNIX s
        elif val > 1e9:
            dt = datetime.fromtimestamp(val, tz=timezone.utc)
    elif isinstance(val, str):
        if val.startswith("/Date("):
            m = re.search(r"/Date\((\d+)", val)
            if m:
                dt = datetime.fromtimestamp(int(m.group(1)) / 1000, tz=timezone.utc)
        else:
            # Try a few friendly formats (label or ISO)
            for fmt in ("%b %d, %Y", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S%z"):
                try:
                    d = datetime.strptime(val, fmt)
                    dt = d.astimezone(timezone.utc) if d.tzinfo else d.replace(tzinfo=timezone.utc)
                    break
                except Exception:
                    pass

    # final fallback from label "Name"/"Label" like "Jun 22, 2025"
    if dt is None:
        label = (pinfo.get("Name") or pinfo.get("Label") or "").strip()
        for fmt in ("%b %d, %Y", "%m/%d/%Y"):
            try:
                d = datetime.strptime(label, fmt)
                dt = d.replace(tzinfo=timezone.utc)
                break
            except Exception:
                pass
    return dt

BASE = pathlib.Path(__file__).parent
STATE_PATH = BASE / "storage_state.json"
if not STATE_PATH.exists():
    STATE_PATH = BASE.parent / "storage_state.json"

API = "https://www.linestarapp.com/DesktopModules/DailyFantasyApi/API/Fantasy/GetSalariesV5"
SITE_ID = 1        # DraftKings
SPORT_ID = 2       # NBA

# ---------- helpers ----------
def season_start_year(dt):
    # NBA season rolls over around Aug/Sep; treat Aug+ as next season start
    return dt.year if dt.month >= 8 else dt.year - 1

def pct_to_float(v):
    if v is None: return None
    try:
        f = float(v)
    except Exception:
        m = re.search(r"-?\d+(?:\.\d+)?", str(v)); f = float(m.group(0)) if m else None
    return f*100.0 if (f is not None and f <= 1.0) else f

def money_to_int(v):
    if v is None: return None
    m = re.search(r"(\d[\d,]*)", str(v));  return int(m.group(1).replace(",", "")) if m else None

def _as_float(x):
    try:
        return float(x)
    except Exception:
        return None

def pick_proj_fpts(r: dict):
    # prefer any positive projection-like field LS might use
    for k in ("AggProj", "Proj", "PP", "Projection", "ProjPoints", "FPTS", "FP", "PS"):
        v = _as_float(r.get(k))
        if v is not None and v > 0:
            return v
    # fallback: average of floor & ceiling if present
    f = _as_float(r.get("Floor")); c = _as_float(r.get("Ceil"))
    if f is not None and c is not None:
        return (f + c) / 2.0
    # last resort: PPG
    return _as_float(r.get("PPG"))

def pick_scored_fpts(r: dict):
    # actual fantasy points once the slate is complete
    for k in ("PS", "Scored", "Score", "Actual", "ActualFP"):
        v = _as_float(r.get(k))
        if v is not None and v >= 0:
            return v
    return None

def normalize_salary_row(r: dict) -> dict:
    """Map LineStar salary row to common fields + projection feature columns."""
    out = {}
    # ids / basics
    out["salary_id"]   = r.get("SalaryId") or r.get("Id")
    out["player_id"]   = r.get("PlayerId") or r.get("PID")
    out["player_name"] = r.get("Name") or r.get("PlayerName") or r.get("Player") or r.get("FullName")
    out["pos"]         = r.get("POS") or r.get("Position") or r.get("Positions")
    out["team"]        = r.get("PTEAM") or r.get("Team") or r.get("TeamAbbrev")
    sal                = r.get("SAL") or r.get("Salary") or r.get("DKSalary") or r.get("S")
    out["salary"]      = money_to_int(sal)

    # projections & context
    out["proj_fpts"]  = pick_proj_fpts(r)
    out["scored_fpts"] = pick_scored_fpts(r)
    out["floor_fpts"] = r.get("Floor")
    out["ceil_fpts"]  = r.get("Ceil")
    out["conf"]       = r.get("Conf")
    out["ppg"]        = r.get("PPG")
    out["opp_rank"]   = r.get("OppRank")
    out["opp_total"]  = r.get("OppTotal")
    out["game_id"]    = r.get("GID")
    out["home_team"]  = r.get("HTEAM")
    out["away_team"]  = r.get("OTEAM")
    out["matchup"]    = r.get("GI")
    out["game_ticks"] = r.get("GT")

    # fallbacks via names
    for k,v in r.items():
        lk = k.lower()
        if out["player_name"] is None and "name" in lk: out["player_name"] = v
        if out["team"] is None and (lk in ("pteam","team","teamabbrev")): out["team"] = v
        if out["pos"] is None and lk.startswith("pos"): out["pos"] = v
        if out["salary"] is None and "salary" in lk: out["salary"] = money_to_int(v)

    # numeric coercion
    for c in ["proj_fpts","scored_fpts","floor_fpts","ceil_fpts","conf","ppg","opp_rank","opp_total"]:
        try:
            out[c] = float(out[c]) if out[c] is not None else None
        except Exception:
            out[c] = None

    # value per $1k
    if out.get("proj_fpts") is not None and out.get("salary"):
        out["value_per_k"] = out["proj_fpts"] / (out["salary"] / 1000.0)
    else:
        out["value_per_k"] = None

    # strip strings
    for k in ("player_name","team","pos","matchup","home_team","away_team"):
        if out.get(k) is not None: out[k] = str(out[k]).strip()
    return out

def fetch_payload(rc, pid: int):
    r = rc.get(API, params={"periodId": pid, "site": SITE_ID, "sport": SPORT_ID})
    return r

def parse_payload(text: str):
    try:
        data = json.loads(text)
    except Exception:
        return [], [], {}  # proj, actual, salaries_index
    if not isinstance(data, dict):
        return [], [], {}
    # salaries container
    salaries = []
    s_index = {}
    scj = data.get("SalaryContainerJson")
    if isinstance(scj, str):
        try:
            container = json.loads(scj)
            salaries = container.get("Salaries") or []
        except Exception:
            salaries = []
    for row in salaries:
        norm = normalize_salary_row(row)
        sid = str(norm.get("salary_id") or "")
        if sid:
            s_index[sid] = norm
    # projected
    proj_rows = []
    own = data.get("Ownership") or {}
    projected = own.get("Projected") if isinstance(own, dict) else None
    if isinstance(projected, dict) and projected:
        slate_key = next(iter(projected.keys()))
        for item in projected.get(slate_key) or []:
            sid = str(item.get("SalaryId") or "")
            srow = s_index.get(sid, {})
            proj_rows.append({
                "player_name": srow.get("player_name"),
                "team": srow.get("team"),
                "pos": srow.get("pos"),
                "salary": srow.get("salary"),
                "proj_own_pct": pct_to_float(item.get("Owned")),
                # feature columns from salaries
                "proj_fpts": srow.get("proj_fpts"),
                "scored_fpts": srow.get("scored_fpts"),
                "floor_fpts": srow.get("floor_fpts"),
                "ceil_fpts": srow.get("ceil_fpts"),
                "conf": srow.get("conf"),
                "value_per_k": srow.get("value_per_k"),
                "game_id": srow.get("game_id"),
                "home_team": srow.get("home_team"),
                "away_team": srow.get("away_team"),
                "matchup": srow.get("matchup"),
                "game_ticks": srow.get("game_ticks"),
                "ppg": srow.get("ppg"),
                "opp_rank": srow.get("opp_rank"),
                "opp_total": srow.get("opp_total"),
                # ids
                "player_id": item.get("PlayerId"),
                "salary_id": item.get("SalaryId"),
                "daily_contest_id": item.get("DailyContestId"),
            })
    # actual (ContestResults list)
    act_rows = []
    for contest in (own.get("ContestResults") or []):
        contest_id = (contest.get("Contest") or {}).get("DailyContestId")
        for item in contest.get("OwnershipData") or []:
            sid = str(item.get("SalaryId") or "")
            srow = s_index.get(sid, {})
            act_rows.append({
                "player_name": srow.get("player_name"),
                "team": srow.get("team"),
                "pos": srow.get("pos"),
                "salary": srow.get("salary"),
                "actual_own_pct": pct_to_float(item.get("Owned")),
                "contest_id": contest_id or item.get("DailyContestId"),
                "player_id": item.get("PlayerId"),
                "salary_id": item.get("SalaryId"),
            })
    return proj_rows, act_rows, s_index

def write_csvs(outroot: pathlib.Path, season: int, pid: int, proj_rows, act_rows, now_iso: str):
    season_dir = outroot / f"season_{season}" / "dk"
    season_dir.mkdir(parents=True, exist_ok=True)
    # enrich + write
    def df_with_context(rows, is_proj):
        df = pd.DataFrame(rows)
        if is_proj and "proj_own_pct" not in df.columns: df["proj_own_pct"] = pd.NA
        if not is_proj and "actual_own_pct" not in df.columns: df["actual_own_pct"] = pd.NA
        for col, val in {
            "sport": "NBA", "site": "DraftKings", "slate_id": str(pid), "scraped_at": now_iso,
        }.items():
            if col not in df.columns: df[col] = val
        # stable column order
        if is_proj:
            cols = ["player_name","team","pos","salary","proj_own_pct",
                    "proj_fpts","scored_fpts","floor_fpts","ceil_fpts","conf","value_per_k",
                    "game_id","home_team","away_team","matchup","game_ticks",
                    "ppg","opp_rank","opp_total",
                    "sport","site","slate_id","scraped_at","player_id","salary_id","daily_contest_id"]
        else:
            cols = ["player_name","team","pos","salary","actual_own_pct","contest_id",
                    "sport","site","slate_id","scraped_at","player_id","salary_id"]
        for c in cols:
            if c not in df.columns: df[c] = pd.NA
        return df[cols]
    proj_path = season_dir / f"pid_{pid}_proj.csv"
    act_path  = season_dir / f"pid_{pid}_actual.csv"
    df_with_context(proj_rows, True).to_csv(proj_path, index=False)
    df_with_context(act_rows, False).to_csv(act_path, index=False)
    return proj_path, act_path

# ---------- main ----------
def main(seasons:int, sleep_min:float, sleep_max:float, limit:int, dry_run:bool, resume:bool):
    # Pick target season start years
    now = datetime.now(timezone.utc)
    years = [season_start_year(now) - i for i in range(seasons)]
    years_set = set(years)

    outroot = BASE / "backfill"
    log_path = outroot / "backfill.log.jsonl"
    outroot.mkdir(exist_ok=True)

    with sync_playwright() as p:
        rc = p.request.new_context(storage_state=str(STATE_PATH) if STATE_PATH.exists() else None)
        # Grab periods list once
        r = rc.get(API, params={"site": SITE_ID, "sport": SPORT_ID})
        if r.status != 200:
            raise SystemExit(f"Periods HTTP {r.status}: {r.text()[:200]}")
        data = r.json()
        periods = data.get("Periods") or []
        # Normalize to (pid, dt)
        tasks = []
        for pinfo in periods:
            pid = pinfo.get("Id") or pinfo.get("PeriodId")
            dt = parse_period_dt(pinfo)
            # fallback: parse label date (e.g., "Jun 22, 2025")
            if dt is None:
                label = (pinfo.get("Name") or pinfo.get("Label") or "")
                try: dt = datetime.strptime(label.strip(), "%b %d, %Y").replace(tzinfo=timezone.utc)
                except Exception: pass
            if not (pid and dt): continue
            if season_start_year(dt) in years_set:
                tasks.append((pid, dt))
        # Work oldest → newest to be gentle
        tasks.sort(key=lambda x: x[1])

        if limit > 0:
            tasks = tasks[:limit]

        print(f"Backfilling DK NBA for seasons {sorted(years_set)} | total PIDs: {len(tasks)}"
              f" | dry_run={dry_run} | resume={resume}")

        # resume: skip if both files exist
        fetched = 0
        for i,(pid, dt) in enumerate(tasks, 1):
            season = season_start_year(dt)
            season_dir = outroot / f"season_{season}" / "dk"
            proj_path = season_dir / f"pid_{pid}_proj.csv"
            act_path  = season_dir / f"pid_{pid}_actual.csv"
            if resume and proj_path.exists() and act_path.exists():
                print(f"[{i}/{len(tasks)}] PID {pid} ({dt.date()}) — skip (already exists)")
                continue

            print(f"[{i}/{len(tasks)}] PID {pid} ({dt.date()}) — fetching…")
            if dry_run:
                continue

            # polite jitter
            time.sleep(random.uniform(sleep_min, sleep_max))

            # fetch with backoff on 429/403
            backoff = 30.0
            for attempt in range(1, 5):
                resp = fetch_payload(rc, pid)
                status = resp.status
                if status == 200:
                    txt = resp.text()
                    proj_rows, act_rows, _ = parse_payload(txt)
                    now_iso = datetime.now(timezone.utc).isoformat()
                    write_csvs(outroot, season, pid, proj_rows, act_rows, now_iso)
                    fetched += 1
                    # log
                    with open(log_path, "a", encoding="utf-8") as lf:
                        lf.write(json.dumps({
                            "pid": pid, "date": dt.isoformat(),
                            "proj_rows": len(proj_rows), "act_rows": len(act_rows),
                            "status": status, "time": now_iso
                        }) + "\n")
                    print(f"   → ok (proj={len(proj_rows)}, act={len(act_rows)})")
                    break
                elif status in (403, 429, 503):
                    print(f"   → HTTP {status}; backing off {int(backoff)}s…")
                    time.sleep(backoff); backoff = min(backoff*2, 300)
                else:
                    print(f"   → HTTP {status}: {resp.text()[:200]}")
                    # log and move on
                    with open(log_path, "a", encoding="utf-8") as lf:
                        lf.write(json.dumps({
                            "pid": pid, "date": dt.isoformat(),
                            "error": f"HTTP {status}", "time": datetime.now(timezone.utc).isoformat()
                        }) + "\n")
                    break

        print(f"Done. Fetched {fetched} PIDs. Output: {outroot}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, default=2, help="how many most-recent NBA seasons (default 2)")
    ap.add_argument("--sleep-min", type=float, default=8.0, help="min seconds between calls (default 8)")
    ap.add_argument("--sleep-max", type=float, default=14.0, help="max seconds between calls (default 14)")
    ap.add_argument("--limit", type=int, default=0, help="optional cap on number of PIDs (0 = no cap)")
    ap.add_argument("--dry-run", action="store_true", help="list what would run, but do not fetch")
    ap.add_argument("--no-resume", dest="resume", action="store_false", help="ignore existing files")
    ap.set_defaults(resume=True)
    args = ap.parse_args()
    main(args.seasons, args.sleep_min, args.sleep_max, args.limit, args.dry_run, args.resume)