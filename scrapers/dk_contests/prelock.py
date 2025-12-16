#!/usr/bin/env python3
"""
Pre-lock player ingestion by DraftKings draft_group_id.

Fetches player pools for Classic slates and inserts into DuckDB (draft_group_players).
If network access is unavailable, will look for cached JSON under
  nba_gpp_data/{date}/players_dg_{draft_group_id}.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import duckdb  # type: ignore
import requests
from dotenv import load_dotenv


DATA_ROOT = Path("nba_gpp_data")
DEFAULT_DB = Path("analytics/contests.duckdb")
COOKIE_ENV_VAR = "DK_RESULTS_COOKIE"


def _format_cookie(raw_cookie: Optional[str]) -> Optional[str]:
    if not raw_cookie:
        return None
    segs = [s.strip() for s in raw_cookie.split(";") if s.strip()]
    return "; ".join(segs) if segs else None


def _session(cookie: Optional[str]) -> requests.Session:
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


def fetch_draft_group(session: requests.Session, draft_group_id: str) -> Dict:
    """Fetch DK draft group player pool, trying multiple endpoints."""
    urls = [
        f"https://www.draftkings.com/lineup/getavailableplayers?draftGroupId={draft_group_id}",
        f"https://www.draftkings.com/api/draftables/v1/{draft_group_id}",
        f"https://api.draftkings.com/draftgroups/v1/draftgroups/{draft_group_id}/draftables",
    ]
    last_err: Optional[Exception] = None
    for url in urls:
        try:
            r = session.get(url, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            last_err = exc
            continue
    if last_err:
        raise last_err
    return {}


def parse_draft_group_payload(payload: Dict) -> List[Dict]:
    """Extract minimal player rows from a DK draft group payload.

    We try to handle multiple shapes: `players` or `playerList` or nested `draftables`.
    """
    candidates = []
    if isinstance(payload, dict):
        for key in ("players", "playerList", "draftables", "draftablePlayers"):
            v = payload.get(key)
            if isinstance(v, list):
                candidates = v
                break
        # Some responses are wrapped
        if not candidates and "Draftables" in payload and isinstance(payload["Draftables"], list):
            candidates = payload["Draftables"]

    out: List[Dict] = []
    for p in candidates:
        # Try multiple key names defensively
        pid = str(p.get("id") or p.get("PlayerId") or p.get("playerId") or p.get("dkId") or "").strip()
        name = (p.get("Name") or p.get("name") or p.get("displayName") or p.get("FirstName", "") + " " + p.get("LastName", "")).strip()
        team = (p.get("Team") or p.get("team") or p.get("proTeamAbbreviation") or "").strip()
        pos = p.get("Position") or p.get("position") or p.get("positions") or p.get("rosterSlots") or ""
        if isinstance(pos, list):
            positions = "/".join(str(x) for x in pos)
        else:
            positions = str(pos)
        salary_raw = p.get("Salary") or p.get("salary") or p.get("displaySalary")
        try:
            salary = int(str(salary_raw).replace("$", "").replace(",", "")) if salary_raw is not None else None
        except Exception:
            salary = None
        game_id = str(p.get("GameId") or p.get("gameId") or "").strip()
        game_time = str(p.get("GameStartTime") or p.get("gameStartTime") or p.get("startTime") or "").strip()

        if name:
            out.append(
                {
                    "player_id": pid,
                    "player_name": name,
                    "team": team,
                    "positions": positions,
                    "salary": salary,
                    "game_id": game_id,
                    "game_time": game_time,
                }
            )
    return out


def insert_draft_group_players(
    con: duckdb.DuckDBPyConnection, draft_group_id: str, rows: List[Dict]
) -> None:
    con.execute(
        """
        insert into draft_group_players (
          draft_group_id, player_id, player_name, team, positions, salary, game_id, game_time
        ) values (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                draft_group_id,
                r.get("player_id", ""),
                r.get("player_name", ""),
                r.get("team", ""),
                r.get("positions", ""),
                r.get("salary"),
                r.get("game_id", ""),
                r.get("game_time", ""),
            )
            for r in rows
        ],
    )


def ingest_for_date(date_str: str, data_root: Path, con: duckdb.DuckDBPyConnection, session: Optional[requests.Session]) -> None:
    day_dir = data_root / date_str
    csv_path = day_dir / f"nba_gpp_{date_str}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Contest CSV not found: {csv_path}")

    # collect draft groups
    import csv as _csv
    dgs: List[str] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as h:
        rdr = _csv.DictReader(h)
        for r in rdr:
            dg = (r.get("draft_group_id") or "").strip()
            if dg:
                dgs.append(dg)
    dgs = sorted(set(dgs))

    for dg in dgs:
        cache_path = day_dir / f"players_dg_{dg}.json"
        payload = None
        if cache_path.exists():
            try:
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:
                payload = None
        if payload is None and session is not None:
            try:
                payload = fetch_draft_group(session, dg)
                cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            except Exception as exc:
                print(f"Failed to fetch draft group {dg}: {exc}")
                continue

        if payload is None:
            print(f"No payload for draft group {dg} (offline and no cache)")
            continue

        rows = parse_draft_group_payload(payload)
        if not rows:
            print(f"Draft group {dg} payload parsed 0 players")
            continue
        insert_draft_group_players(con, dg, rows)
        print(f"Ingested {len(rows)} players for draft group {dg}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest DK pre-lock player lists by draft group")
    p.add_argument("--date", required=True, help="YYYY-MM-DD date")
    p.add_argument("--data-root", default=str(DATA_ROOT))
    p.add_argument("--db", default=str(DEFAULT_DB))
    p.add_argument("--no-network", action="store_true", help="Disable network fetch; use cache only")
    return p.parse_args()


def main() -> None:
    # Load .env if present for DK_RESULTS_COOKIE
    try:
        load_dotenv()
    except Exception:
        pass
    args = _parse_args()
    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))

    # Ensure schema (tables)
    schema_sql = (Path(__file__).parent.parent.parent / "sql" / "schema.sql").read_text(encoding="utf-8")
    con.execute(schema_sql)

    sess = None
    if not args.no_network:
        cookie = _format_cookie(os.getenv(COOKIE_ENV_VAR))
        if not cookie:
            print("Warning: no DK_RESULTS_COOKIE set; endpoint may return empty/unauthorized.")
        sess = _session(cookie)

    ingest_for_date(args.date, Path(args.data_root), con, sess)


if __name__ == "__main__":
    main()
