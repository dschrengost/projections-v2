# scripts/scrape_linestar_contest_meta.py
import argparse, concurrent.futures, pandas as pd, re, time
from pathlib import Path
from typing import Dict, Any, List

# ---------- LineStar client (swap these two stubs for your real calls) ----------
class LineStarClient:
    def __init__(self, base_url: str, auth_cookie: str):
        self.base_url = base_url
        self.auth_cookie = auth_cookie
        # TODO: requests.Session() w/ cookie headers, retries, etc.

    def get_contests_for_date(self, site: str, date_str: str) -> List[Dict[str, Any]]:
        """
        Return a list of contests for (site, date). Each item should include:
        contest_id, contest_name, entry_fee, field_size, max_entries_per_user,
        prize_pool, first_prize, start_time, game_style ('classic'/'showdown'/'tiers'),
        payout_table (list of {place, prize}), paid_out_count (if available).
        """
        # IMPLEMENT: call your LineStar endpoint(s) and return normalized dicts
        raise NotImplementedError

    def get_contest_detail(self, site: str, contest_id: str) -> Dict[str, Any]:
        """Optional: detail endpoint if list endpoint is sparse."""
        raise NotImplementedError

# ---------- Tagging logic (regex + numeric heuristics) ----------
R = lambda s: re.compile(s, re.I)
PAT = {
    "CASH_50_50": [R(r"\b50\s*[/ ]?\s*50\b")],
    "CASH_DU":    [R(r"double\s*up")],
    "CASH_H2H":   [R(r"head\s*[- ]?\s*to\s*[- ]?\s*head|\bh2h\b")],
    "CASH_MULTIPLIER": [R(r"\b(multiplier|3x|4x|5x|10x)\b")],
    "OTHER_WTA":  [R(r"winner\s*[- ]?\s*take\s*[- ]?\s*all|\bwta\b")],
    "OTHER_SAT":  [R(r"satellite|qualifier|ticket")],
    "SHOWDOWN":   [R(r"showdown|single[- ]?game|captain")],
    "PICKEM":     [R(r"pick[’' ]?em|tiers")],
    "GPP_SE":     [R(r"single\s*entry|\bSE\b(?![a-z])")],
    "GPP_3MAX":   [R(r"\b3\s*[- ]?\s*max\b|\bthree\s*[- ]?\s*max\b|\b3\s*entry\s*max\b")],
}
def _norm(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x)).strip().lower() if pd.notna(x) else ""

def _derive_payout_stats(payouts: List[Dict[str, Any]], field_size: int) -> Dict[str, Any]:
    if not payouts or not field_size:
        return {"payout_paid_pct": None, "min_cash_mult": None, "top_heavy_pct": None}
    paid = sum(1 for p in payouts if (p.get("prize") or 0) > 0)
    payout_paid_pct = round(100 * paid / max(field_size, 1), 2)
    total_pool = sum(p.get("prize") or 0 for p in payouts)
    first = payouts[0]["prize"] if payouts else None
    top_heavy_pct = round(100 * (first or 0) / max(total_pool, 1), 2) if total_pool else None
    # crude min-cash ≈ last paid prize / entry_fee handled later when entry_fee known
    return {"payout_paid_pct": payout_paid_pct, "min_cash_mult": None, "top_heavy_pct": top_heavy_pct}

def tag_contest_row(row: pd.Series) -> str:
    name = _norm(row.get("contest_name", ""))
    game_style = _norm(row.get("game_style", ""))  # 'classic'/'showdown'/'tiers'
    size = row.get("field_size", None)
    max_per_user = row.get("max_entries_per_user", None)
    payout_paid_pct = row.get("payout_paid_pct", None)
    entry_fee = row.get("entry_fee", None)
    payouts = row.get("_payouts", None)  # optional passthrough to compute min-cash

    # Hard style exclusion
    if "showdown" in game_style or "captain" in game_style:
        return "SHOWDOWN"
    if "tiers" in game_style or "pickem" in name:
        return "PICKEM"

    # Name regex
    for label, patterns in PAT.items():
        if label.startswith("GPP_"):  # defer these if numeric hints exist
            continue
        if any(rx.search(name) for rx in patterns):
            return label

    # Numeric hints (cash payout structures)
    if isinstance(payout_paid_pct, (int, float)):
        if 45 <= payout_paid_pct <= 55:
            # Try to disambiguate 50/50 vs DU using min-cash multiple
            if payouts and entry_fee:
                last_paid = next((p.get("prize") for p in reversed(payouts) if (p.get("prize") or 0) > 0), None)
                if last_paid:
                    mult = (last_paid or 0) / entry_fee
                    if 1.8 <= mult <= 2.2:
                        return "CASH_DU"
            return "CASH_50_50"

    # Entry limit → GPP archetype
    if isinstance(max_per_user, (int, float)):
        m = int(max_per_user)
        if m == 1:  return "GPP_SE"
        if m == 3:  return "GPP_3MAX"
        if m >= 20: return "GPP_MME"

    # Name cues for GPP if entry limit missing
    if any(k in name for k in ["milly", "mega", "minimax", "mini-max", "fadeaway", "sharpshooter", "and-one", "four point", "bank shot"]):
        return "GPP_MME"

    # Small private leagues
    if isinstance(size, (int, float)) and size and size <= 20:
        return "OTHER_LEAGUE_SMALL"

    return "GPP_MME"  # conservative default for NBA Classic

def tag_contests(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["contest_archetype"] = out.apply(tag_contest_row, axis=1)
    out["is_gpp"] = out["contest_archetype"].str.startswith("GPP_")
    return out

# ---------- Runner ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ownership", required=True, help="Path to ownership_actual.csv with at least [site, slate_date, contest_id]")
    ap.add_argument("--out", required=True, help="contest_meta_tagged.csv")
    ap.add_argument("--base-url", default="https://api.linestar.*")  # fill
    ap.add_argument("--auth-cookie", default="")  # fill from env/secret
    ap.add_argument("--max-workers", type=int, default=8)
    args = ap.parse_args()

    own = pd.read_csv(args.ownership)
    # Normalize headers the way we do elsewhere
    rename = {"date":"slate_date","contestId":"contest_id","site_name":"site"}
    for k,v in rename.items():
        if k in own.columns and v not in own.columns: own = own.rename(columns={k:v})
    needed = {"site","slate_date","contest_id"}
    missing = needed - set(own.columns)
    if missing:
        raise SystemExit(f"ownership file lacks columns: {missing}")

    # Unique lookup keys to fetch
    keys = own[["site","slate_date","contest_id"]].drop_duplicates().reset_index(drop=True)

    client = LineStarClient(args.base_url, args.auth_cookie)

    def fetch_row(row):
        site, date_str, cid = row["site"], row["slate_date"], str(row["contest_id"])
        # Prefer list-by-date, then detail if needed
        try:
            contests = client.get_contests_for_date(site, date_str)  # list of dicts
            meta = next((c for c in contests if str(c.get("contest_id")) == cid), None)
            if not meta and hasattr(client, "get_contest_detail"):
                meta = client.get_contest_detail(site, cid)
            if not meta:
                return None
            # compute payout stats & attach raw payouts for min-cash check
            payout_stats = _derive_payout_stats(meta.get("payout_table"), meta.get("field_size") or 0)
            meta = {**meta, **payout_stats, "_payouts": meta.get("payout_table")}
            meta["site"] = site
            meta["slate_date"] = date_str
            meta["contest_id"] = cid
            return meta
        except Exception as e:
            return {"site":site,"slate_date":date_str,"contest_id":cid,"_error":str(e)}

    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futs = [ex.submit(fetch_row, r) for _, r in keys.iterrows()]
        for f in concurrent.futures.as_completed(futs):
            res = f.result()
            if res: rows.append(res)

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No contest metadata fetched. Check auth/endpoint stubs.")

    # Drop helper columns & tag
    # Ensure the key fields exist even on error rows
    base_cols = ["site","slate_date","contest_id","contest_name","entry_fee","field_size",
                 "max_entries_per_user","prize_pool","first_prize","start_time","game_style",
                 "payout_paid_pct","top_heavy_pct"]
    for c in base_cols:
        if c not in df.columns: df[c] = None
    tagged = tag_contests(df)

    # Persist (keep raw payouts only if you want a sidecar JSON)
    tagged.drop(columns=[c for c in tagged.columns if c.startswith("_")], inplace=True, errors="ignore")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    tagged.to_csv(args.out, index=False)

    # Quick sanity prints
    print("rows:", len(tagged),
          "| gpp:", int(tagged["is_gpp"].sum()),
          "| cash/showdown:", int((~tagged["is_gpp"]).sum()))
    print(tagged["contest_archetype"].value_counts().to_string())

if __name__ == "__main__":
    main()