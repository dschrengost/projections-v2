# scrapers/linestar/fetch_ownership_by_pid.py
import argparse, json, pathlib, re
import pandas as pd
from datetime import datetime, timezone
from playwright.sync_api import sync_playwright

BASE = pathlib.Path(__file__).parent

# Reuse cookies saved by login_password.py
STATE_PATH = BASE / "storage_state.json"
if not STATE_PATH.exists():
    STATE_PATH = BASE.parent / "storage_state.json"

API = "https://www.linestarapp.com/DesktopModules/DailyFantasyApi/API/Fantasy/GetSalariesV5"
SITE_MAP  = {"dk": 1, "draftkings": 1, "fd": 2, "fanduel": 2}
SPORT_MAP = {"nba": 2}

def _nkey(k: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (k or "").lower())

def pct_to_float(v):
    if v is None: return None
    try:
        f = float(v)
    except Exception:
        m = re.search(r"-?\d+(?:\.\d+)?", str(v))
        f = float(m.group(0)) if m else None
    if f is None: return None
    # Heuristic: fractions → %
    return f*100.0 if f <= 1.0 else f

def money_to_int(v):
    if v is None: return None
    m = re.search(r"(\d[\d,]*)", str(v))
    return int(m.group(1).replace(",", "")) if m else None

def first_match(d: dict, *cands):
    for c in cands:
        if c in d: return d[c]
    return None

def _as_float(x):
    try:
        return float(x)
    except Exception:
        return None

def pick_proj_fpts(r: dict):
    for k in ("AggProj", "Proj", "PP", "Projection", "ProjPoints", "FPTS", "FP", "PS"):
        v = _as_float(r.get(k))
        if v is not None and v > 0:
            return v
    f = _as_float(r.get("Floor")); c = _as_float(r.get("Ceil"))
    if f is not None and c is not None:
        return (f + c) / 2.0
    return _as_float(r.get("PPG"))

def pick_scored_fpts(r: dict):
    # actual fantasy points once slate completes (simple, not exhaustive)
    for k in ("PS", "Scored", "Score", "Actual", "ActualFP"):
        v = _as_float(r.get(k))
        if v is not None and v >= 0:
            return v
    return None

def normalize_salary_row(r: dict) -> dict:
    """Map LineStar salary row to common fields + feature columns."""
    out = {}
    out["salary_id"]   = r.get("SalaryId") or r.get("Id")
    out["player_id"]   = r.get("PlayerId") or r.get("PID")
    out["player_name"] = r.get("Name") or r.get("PlayerName") or r.get("Player") or r.get("FullName")
    out["pos"]         = r.get("POS") or r.get("Position") or r.get("Positions")
    out["team"]        = r.get("PTEAM") or r.get("Team") or r.get("TeamAbbrev")
    sal                = r.get("SAL") or r.get("Salary") or r.get("DKSalary") or r.get("FDSalary") or r.get("S")
    out["salary"]      = money_to_int(sal)

    # projections & context
    out["proj_fpts"]   = pick_proj_fpts(r)
    out["scored_fpts"] = pick_scored_fpts(r)
    out["floor_fpts"]  = r.get("Floor")
    out["ceil_fpts"]   = r.get("Ceil")
    out["conf"]        = r.get("Conf")
    out["ppg"]         = r.get("PPG")
    out["opp_rank"]    = r.get("OppRank")
    out["opp_total"]   = r.get("OppTotal")
    out["game_id"]     = r.get("GID")
    out["home_team"]   = r.get("HTEAM")
    out["away_team"]   = r.get("OTEAM")
    out["matchup"]     = r.get("GI")
    out["game_ticks"]  = r.get("GT")

    # fallbacks via key names
    for k,v in r.items():
        lk = k.lower()
        if out["player_name"] is None and "name" in lk: out["player_name"] = v
        if out["team"] is None and (lk in ("pteam","team","teamabbrev")): out["team"] = v
        if out["pos"] is None and lk.startswith("pos"): out["pos"] = v
        if out["salary"] is None and "salary" in lk: out["salary"] = money_to_int(v)

    # tidy types
    num_fields = ["proj_fpts","scored_fpts","floor_fpts","ceil_fpts","conf","ppg","opp_rank","opp_total"]
    for c in num_fields:
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

def fetch_payload(pid:int, site:str, sport:str):
    with sync_playwright() as p:
        rc = p.request.new_context(storage_state=str(STATE_PATH) if STATE_PATH.exists() else None)
        params = {"periodId": pid, "site": SITE_MAP[site.lower()], "sport": SPORT_MAP[sport.lower()]}
        r = rc.get(API, params=params)
        if r.status != 200:
            raise SystemExit(f"HTTP {r.status}: {r.text()[:200]}")
        try:
            return r.json()
        except Exception:
            return json.loads(r.text())

def main(pid: int, site: str, sport: str):
    if site.lower() not in SITE_MAP:  raise SystemExit(f"Unknown site '{site}'. Use dk or fd.")
    if sport.lower() not in SPORT_MAP: raise SystemExit(f"Unknown sport '{sport}'. Use nba.")

    data = fetch_payload(pid, site, sport)
    if not isinstance(data, dict):
        raise SystemExit(f"Unexpected payload (type={type(data)})")

    # ---------------- Salaries (for join) ----------------
    salaries = []
    salaries_map = {}
    container_raw = data.get("SalaryContainerJson")
    if isinstance(container_raw, str):
        try:
            container = json.loads(container_raw)
            salaries = container.get("Salaries") or []
            salaries_map = (container.get("SalariesMap") or {}).get("SlateSalaryMap", {})
        except Exception:
            salaries, salaries_map = [], {}
    # Build index by SalaryId
    s_by_id = {}
    for row in salaries:
        norm = normalize_salary_row(row)
        sid = norm.get("salary_id") or row.get("SalaryId") or row.get("Id")
        if sid is not None:
            s_by_id[str(sid)] = norm

    # ---------------- Projected ownership ----------------
    own = data.get("Ownership") or {}
    projected = own.get("Projected") if isinstance(own, dict) else None
    proj_rows = []
    if isinstance(projected, dict) and projected:
        slate_key = next(iter(projected.keys()))
        for item in projected.get(slate_key) or []:
            sid = str(item.get("SalaryId") or "")
            srow = s_by_id.get(sid, {})
            proj_rows.append({
                "player_name": srow.get("player_name"),
                "team": srow.get("team"),
                "pos": srow.get("pos"),
                "salary": srow.get("salary"),
                "proj_own_pct": pct_to_float(item.get("Owned")),
                # --- added feature columns from salaries ---
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
                # --- context ---
                "sport": sport.upper(),
                "site": "DraftKings" if SITE_MAP[site.lower()] == 1 else "FanDuel",
                "slate_id": str(pid),
                "scraped_at": datetime.now(timezone.utc).isoformat(),
                "player_id": item.get("PlayerId"),
                "salary_id": item.get("SalaryId"),
                "daily_contest_id": item.get("DailyContestId"),
            })

    df_proj = pd.DataFrame(proj_rows)

    # ---------------- Actual ownership ----------------
    actual_rows = []
    cr_list = own.get("ContestResults") or []
    for contest in cr_list:
        contest_meta = contest.get("Contest", {})
        contest_id = contest_meta.get("DailyContestId") or None
        for item in contest.get("OwnershipData") or []:
            sid = str(item.get("SalaryId") or "")
            srow = s_by_id.get(sid, {})
            actual_rows.append({
                "player_name": srow.get("player_name"),
                "team": srow.get("team"),
                "pos": srow.get("pos"),
                "salary": srow.get("salary"),
                "actual_own_pct": pct_to_float(item.get("Owned")),
                "contest_id": contest_id or item.get("DailyContestId"),
                "sport": sport.upper(),
                "site": "DraftKings" if SITE_MAP[site.lower()] == 1 else "FanDuel",
                "slate_id": str(pid),
                "scraped_at": datetime.now(timezone.utc).isoformat(),
                "player_id": item.get("PlayerId"),
                "salary_id": item.get("SalaryId"),
            })

    df_act = pd.DataFrame(actual_rows)

    # Ensure expected columns exist
    for df, cols in (
        (df_proj, ["player_name","team","pos","salary","proj_own_pct",
                   "proj_fpts","scored_fpts","floor_fpts","ceil_fpts","conf","value_per_k",
                   "game_id","home_team","away_team","matchup","game_ticks",
                   "ppg","opp_rank","opp_total",
                   "sport","site","slate_id","scraped_at","player_id","salary_id","daily_contest_id"]),
        (df_act,  ["player_name","team","pos","salary","actual_own_pct","contest_id","sport","site","slate_id","scraped_at"]),
    ):
        for c in cols:
            if c not in df.columns:
                df[c] = pd.NA

    outdir = BASE / "out"
    outdir.mkdir(exist_ok=True)

    # Write CSVs (header-only if empty)
    proj_path = outdir / "ownership_proj.csv"
    act_path  = outdir / "ownership_actual.csv"

    if not df_proj.empty:
        df_proj.to_csv(proj_path, index=False)
    else:
        pd.DataFrame(columns=[
            "player_name","team","pos","salary","proj_own_pct",
            "proj_fpts","scored_fpts","floor_fpts","ceil_fpts","conf","value_per_k",
            "game_id","home_team","away_team","matchup","game_ticks",
            "ppg","opp_rank","opp_total",
            "sport","site","slate_id","scraped_at",
            "player_id","salary_id","daily_contest_id"
        ]).to_csv(proj_path, index=False)

    if not df_act.empty:
        df_act.to_csv(act_path, index=False)
    else:
        pd.DataFrame(columns=["player_name","team","pos","salary","actual_own_pct","contest_id","sport","site","slate_id","scraped_at"]).to_csv(act_path, index=False)

    print({
        "proj_rows": int(len(df_proj)),
        "act_rows": int(len(df_act)),
        "out": str(outdir)
    })
    if not df_proj.empty:
        print("Projected sample:", df_proj.head(3).to_dict(orient="records"))
    if not df_act.empty:
        print("Actual sample:", df_act.head(3).to_dict(orient="records"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, required=True, help="LineStar periodId (PID), e.g. 2395")
    ap.add_argument("--site", default="dk", help="dk or fd (default: dk)")
    ap.add_argument("--sport", default="nba", help="default: nba")
    args = ap.parse_args()
    main(args.pid, args.site, args.sport)