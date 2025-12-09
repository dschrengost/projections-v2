#!/usr/bin/env python3
"""
Scrape payout curves for DK contests via contests API.

Endpoint: https://api.draftkings.com/contests/v1/contests/{contest_id}
We parse `contestDetail.payoutSummary` which includes min/max positions and cash values.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import duckdb  # type: ignore
import requests


DATA_ROOT = Path("nba_gpp_data")
DEFAULT_DB = Path("analytics/contests.duckdb")
API_URL_TMPL = "https://api.draftkings.com/contests/v1/contests/{contest_id}"


def _session(cookie: Optional[str] = None) -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json, text/plain, */*",
            "Referer": "https://www.draftkings.com/",
        }
    )
    if cookie:
        s.headers["Cookie"] = cookie
    return s


def read_contest_ids(csv_path: Path) -> List[str]:
    ids: List[str] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as h:
        rdr = csv.DictReader(h)
        for r in rdr:
            cid = (r.get("contest_id") or "").strip()
            gt = (r.get("game_type") or "").strip()
            if cid and gt == "Classic":
                ids.append(cid)
    return sorted(set(ids))


def fetch_payouts(sess: requests.Session, contest_id: str) -> Dict:
    url = API_URL_TMPL.format(contest_id=contest_id)
    r = sess.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def parse_payout_summary(payload: Dict) -> List[Dict]:
    cur = payload.get("contestDetail", {})
    summary = cur.get("payoutSummary") or []
    out: List[Dict] = []
    for tier in summary:
        minpos = int(tier.get("minPosition", 0) or 0)
        maxpos = int(tier.get("maxPosition", minpos) or minpos)
        amount = None
        ptype = "cash"
        # prefer numeric value
        if tier.get("payoutDescriptions"):
            for desc in tier["payoutDescriptions"]:
                val = desc.get("value")
                if val is not None:
                    try:
                        amount = float(val)
                        break
                    except Exception:
                        pass
        if amount is None and tier.get("tierPayoutDescriptions"):
            # e.g., {"Cash":"$1,000.00"}
            d = tier["tierPayoutDescriptions"]
            for k, v in d.items():
                ptype = k.lower()
                # strip currency
                s = str(v).replace("$", "").replace(",", "")
                try:
                    amount = float(s)
                except Exception:
                    amount = None
        if minpos and amount is not None:
            out.append(
                {
                    "min_rank": minpos,
                    "max_rank": maxpos,
                    "amount": amount,
                    "payout_type": ptype,
                }
            )
    return out


def ensure_schema(con: duckdb.DuckDBPyConnection) -> None:
    schema_sql = (Path(__file__).parent.parent.parent / "sql" / "schema.sql").read_text(encoding="utf-8")
    con.execute(schema_sql)


def insert_payouts(con: duckdb.DuckDBPyConnection, contest_id: str, rows: List[Dict]) -> None:
    con.execute("delete from payouts where contest_id = ?", [contest_id])
    if not rows:
        return
    con.executemany(
        """
        insert into payouts (contest_id, min_rank, max_rank, amount, payout_type)
        values (?, ?, ?, ?, ?)
        """,
        [
            (
                contest_id,
                r.get("min_rank"),
                r.get("max_rank"),
                r.get("amount"),
                r.get("payout_type", "cash"),
            )
            for r in rows
        ],
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scrape DK payout curves for contests in a daily CSV")
    p.add_argument("--date", required=True, help="Contest date (YYYY-MM-DD)")
    p.add_argument("--data-root", default=str(DATA_ROOT))
    p.add_argument("--db", default=str(DEFAULT_DB))
    p.add_argument("--cookie", help="Optional Cookie header value (falls back to DK_RESULTS_COOKIE)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    day_dir = Path(args.data_root) / args.date
    csv_path = day_dir / f"nba_gpp_{args.date}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Contest CSV not found: {csv_path}")

    ids = read_contest_ids(csv_path)
    if not ids:
        print("No Classic contests found in CSV")
        return

    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    ensure_schema(con)

    results_dir = day_dir / "payouts"
    results_dir.mkdir(parents=True, exist_ok=True)

    cookie = args.cookie or os.getenv("DK_RESULTS_COOKIE") or ""
    sess = _session(cookie=cookie)

    for cid in ids:
        try:
            payload = fetch_payouts(sess, cid)
            rows = parse_payout_summary(payload)
            insert_payouts(con, cid, rows)
            # also write a CSV artifact
            out_csv = results_dir / f"contest_{cid}_payouts.csv"
            with out_csv.open("w", newline="", encoding="utf-8") as h:
                w = csv.writer(h)
                w.writerow(["min_rank", "max_rank", "amount", "payout_type"])
                for r in rows:
                    w.writerow([r["min_rank"], r["max_rank"], r["amount"], r["payout_type"]])
            print(f"Saved payouts for {cid}: {len(rows)} tiers")
        except Exception as exc:
            print(f"Failed payouts for {cid}: {exc}")


if __name__ == "__main__":
    main()

