import json, pathlib, re
from datetime import datetime, timezone
import pandas as pd

BASE = pathlib.Path(__file__).parent
# Search both scrapers/linestar/captures and scrapers/captures (top-level)
CANDIDATE_ROOTS = [BASE / "captures", BASE.parent / "captures"]
ALL_CAP_DIRS = []
for root in CANDIDATE_ROOTS:
    if root.exists():
        ALL_CAP_DIRS.extend([d for d in sorted(root.glob("*")) if d.is_dir()])

if not ALL_CAP_DIRS:
    raise SystemExit("No capture folders found under scrapers/linestar/captures/ or scrapers/captures/")

# Pick the newest capture that actually contains a GetSalariesV5 JSON
target_meta = None
target_path = None
cap = None
for capdir in reversed(ALL_CAP_DIRS):  # newest first
    for p in capdir.glob("*.json"):
        try:
            meta = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "GetSalariesV5" in str(meta.get("url", "")):
            target_meta = meta
            target_path = p
            cap = capdir
            break
    if target_meta:
        break

if not target_meta:
    # Friendly error with hints on where we looked
    searched = "\n  - " + "\n  - ".join(str(d) for d in ALL_CAP_DIRS[-5:])
    raise SystemExit("No GetSalariesV5 JSON found in the latest captures.\nSearched (newest last):" + searched)

# payload can be a list, dict, or text-wrapped JSON
payload = target_meta.get("data")
if isinstance(payload, dict) and "text" in payload and isinstance(payload["text"], str):
    try:
        payload = json.loads(payload["text"])
    except Exception:
        payload = []
if isinstance(payload, dict):
    rows = (
        payload.get("Data") or payload.get("data") or
        payload.get("Result") or payload.get("rows") or
        payload.get("Items") or []
    )
else:
    rows = payload if isinstance(payload, list) else []

def norm_key(k: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (k or "").lower())

def pct_to_float(v):
    if v is None: return None
    m = re.search(r"-?\d+(?:\.\d+)?", str(v));  return float(m.group(0)) if m else None

def money_to_int(v):
    if v is None: return None
    m = re.search(r"(\d[\d,]*)", str(v));  return int(m.group(1).replace(",", "")) if m else None

def to_float(v):
    try: return float(str(v).replace(",",""))
    except: return None

def normalize_row(r: dict) -> dict:
    out = {}
    for k,v in (r or {}).items():
        nk = norm_key(k)
        if "projown" in nk:            out["proj_own_pct"]   = pct_to_float(v)
        elif "actualown" in nk:        out["actual_own_pct"] = pct_to_float(v)
        elif nk in ("player","name","playername"):
                                       out["player_name"] = (v or "").strip()
        elif nk.startswith("pos"):     out["pos"] = (v or "").strip()
        elif nk.startswith("team"):    out["team"] = (v or "").strip()
        elif "salary" in nk:           out["salary"] = money_to_int(v)
        elif nk in ("proj","projection","fpts","projfpts","points","fp"):
                                       out["proj_fpts"] = to_float(v)
        elif nk in ("scored","actualfpts","actualpoints","fptsactual"):
                                       out["actual_fpts"] = to_float(v)
    return out

normed = [normalize_row(r) for r in rows if isinstance(r, dict)]
# add context columns
# try to extract periodId/site/sport from the captured URL (if present)
pid = None
site_id = None
sport_id = None
m_pid = re.search(r"[?&]periodId=(\d+)", target_meta["url"])
m_site = re.search(r"[?&]site=(\d+)", target_meta["url"])
m_sport = re.search(r"[?&]sport=(\d+)", target_meta["url"])
if m_pid: pid = m_pid.group(1)
if m_site: site_id = int(m_site.group(1))
if m_sport: sport_id = int(m_sport.group(1))

site_name = {1:"DraftKings", 2:"FanDuel"}.get(site_id, None)
sport_name = {2:"NBA"}.get(sport_id, None)

for o in normed:
    if site_name:  o.setdefault("site", site_name)
    if sport_name: o.setdefault("sport", sport_name)
    if pid:        o.setdefault("slate_id", pid)
    o.setdefault("scraped_at", datetime.now(timezone.utc).isoformat())

df = pd.DataFrame(normed)

outdir = BASE / "out"
outdir.mkdir(exist_ok=True)

proj = df.dropna(subset=["proj_own_pct"], how="all")
act  = df.dropna(subset=["actual_own_pct"], how="all")

if not proj.empty: proj.to_csv(outdir / "ownership_proj.csv", index=False)
if not act.empty:  act.to_csv(outdir / "ownership_actual.csv", index=False)

print({
    "capture_dir": str(cap),
    "json_file": str(target_path.name),
    "rows": len(df),
    "proj_rows": int(len(proj)),
    "act_rows": int(len(act)),
    "outdir": str(outdir),
    "url": target_meta["url"]
})
if not df.empty:
    print("sample keys:", list(df.columns)[:15])
else:
    print("No rows parsed — try the direct fetcher next.")