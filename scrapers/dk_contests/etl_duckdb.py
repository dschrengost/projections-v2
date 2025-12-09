#!/usr/bin/env python3
"""
ETL loader for DraftKings NBA Classic GPP results into DuckDB.

Loads contest metadata (daily CSV), contest results (mixed CSV/ZIP payloads),
and builds normalized tables: contests, entries, lineup_players, player_ownership.
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import duckdb  # type: ignore


DATA_ROOT = Path("nba_gpp_data")
DEFAULT_DB = Path("analytics/contests.duckdb")


SLOT_TOKENS = {
    "PG",
    "SG",
    "SF",
    "PF",
    "C",
    "G",
    "F",
    "UTIL",
}


@dataclass
class ContestRow:
    contest_id: str
    contest_name: str
    start_time: str
    start_time_readable: str
    prize_pool: Optional[float]
    first_place_prize: str
    entry_fee: Optional[float]
    max_entries: Optional[int]
    current_entries: Optional[int]
    game_type: str
    is_guaranteed: bool
    is_starred: bool
    template_id: str
    draft_group_id: str
    scrape_time: str


def _to_float(x: str | None) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip().replace(",", "")
    if s == "" or s.lower() == "not specified":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _to_int(x: str | None) -> Optional[int]:
    if x is None:
        return None
    s = str(x).strip()
    if not s or s.lower() == "unlimited":
        return None
    try:
        # Some values like "0.5" exist; cast safely
        return int(float(s))
    except ValueError:
        return None


def ensure_db(con: duckdb.DuckDBPyConnection) -> None:
    schema_sql = (Path(__file__).parent.parent.parent / "sql" / "schema.sql").read_text(
        encoding="utf-8"
    )
    con.execute(schema_sql)


def create_views(con: duckdb.DuckDBPyConnection) -> None:
    views_sql = (Path(__file__).parent.parent.parent / "sql" / "views_metrics.sql").read_text(
        encoding="utf-8"
    )
    con.execute(views_sql)


def read_contests_csv(csv_path: Path) -> List[ContestRow]:
    rows: List[ContestRow] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for r in reader:
            rows.append(
                ContestRow(
                    contest_id=str(r.get("contest_id", "")).strip(),
                    contest_name=(r.get("contest_name") or "").strip(),
                    start_time=(r.get("start_time") or "").strip(),
                    start_time_readable=(r.get("start_time_readable") or "").strip(),
                    prize_pool=_to_float(str(r.get("prize_pool", "")).replace("$", "")),
                    first_place_prize=(r.get("first_place_prize") or "").strip(),
                    entry_fee=_to_float(str(r.get("entry_fee", "")).replace("$", "")),
                    max_entries=_to_int(r.get("max_entries")),
                    current_entries=_to_int(r.get("current_entries")),
                    game_type=(r.get("game_type") or "").strip(),
                    is_guaranteed=str(r.get("is_guaranteed", "false")).lower() == "true",
                    is_starred=str(r.get("is_starred", "false")).lower() == "true",
                    template_id=str(r.get("template_id", "")).strip(),
                    draft_group_id=str(r.get("draft_group_id", "")).strip(),
                    scrape_time=str(r.get("scrape_time", "")).strip(),
                )
            )
    return rows


def _read_zip_csvs(path: Path) -> Dict[str, str]:
    """Return all CSV file texts from a ZIP keyed by filename."""
    out: Dict[str, str] = {}
    b = path.read_bytes()
    with zipfile.ZipFile(io.BytesIO(b)) as zf:
        for info in zf.infolist():
            if info.filename.lower().endswith(".csv"):
                out[info.filename] = zf.read(info.filename).decode("utf-8", errors="ignore")
    return out


def _read_text_auto(path: Path) -> str:
    """Return decoded CSV text for single-CSV payloads only."""
    b = path.read_bytes()
    if b[:2] == b"PK":
        # Multi-file ZIP: caller should use _read_zip_csvs instead
        raise RuntimeError("zip payload has multiple CSVs; use _read_zip_csvs")
    return b.decode("utf-8", errors="ignore")


def _sanitize_header(names: List[str]) -> List[str]:
    out: List[str] = []
    used: Dict[str, int] = {}
    for i, n in enumerate(names):
        name = (n or "").strip()
        if name == "":
            name = f"_sep_{i}"
        # Deduplicate
        base = name
        k = used.get(base, 0)
        if k:
            name = f"{base}__{k}"
        used[base] = k + 1
        out.append(name)
    return out


def _read_csv_dicts_from_text(text: str) -> List[Dict[str, str]]:
    rdr = csv.reader(io.StringIO(text))
    try:
        header = next(rdr)
    except StopIteration:
        return []
    header = _sanitize_header(header)
    rows: List[Dict[str, str]] = []
    for row in rdr:
        d = {header[i]: (row[i] if i < len(row) else "") for i in range(len(header))}
        rows.append(d)
    return rows


def parse_results_rows(rows: List[Dict[str, str]]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split full-standings CSV into (entries, lineup_players, player_ownership).

    Heuristic:
      - Any row with non-empty EntryId is a standings row.
      - Any row with non-empty Player is an ownership row.
      - Lineup is parsed into slot/name pairs.
    """
    entries: List[Dict] = []
    lineup_players: List[Dict] = []
    ownership: List[Dict] = []

    for r in rows:
        entry_id = (r.get("EntryId") or "").strip()
        player = (r.get("Player") or "").strip()

        if entry_id:
            rank_str = (r.get("Rank") or "").strip()
            points_str = (r.get("Points") or "").strip()
            entry_name = (r.get("EntryName") or "").strip()
            # DK often embeds handle in EntryName e.g., "user (96/150)"; keep raw as name and try to split handle
            user_handle = entry_name.split(" (")[0] if entry_name else ""
            lineup_raw = (r.get("Lineup") or "").strip()

            try:
                rank = int(float(rank_str)) if rank_str else None
            except ValueError:
                rank = None
            try:
                points = float(points_str) if points_str else None
            except ValueError:
                points = None

            entries.append(
                {
                    "entry_id": entry_id,
                    "entry_name": entry_name,
                    "user_handle": user_handle,
                    "rank": rank,
                    "points": points,
                    "winnings": None,
                    "lineup_raw": lineup_raw,
                }
            )

            # Parse lineup into slot/name pairs
            if lineup_raw:
                lineup_players.extend(_explode_lineup(entry_id, lineup_raw))

        if player:
            pos = (r.get("Roster Position") or "").strip()
            pct = (r.get("%Drafted") or r.get("%Drafted__1") or "").strip()
            fpts = (r.get("FPTS") or r.get("FPTS__1") or "").strip()
            ownership.append(
                {
                    "player_name": player,
                    "roster_position": pos,
                    "pct_drafted": _pct_to_float(pct),
                    "fpts": _to_float(fpts),
                }
            )

    return entries, lineup_players, ownership


def parse_contest_players_rows(rows: List[Dict[str, str]]) -> List[Dict]:
    """Parse contest-players CSV to normalized player pool rows.

    Expected columns include some of: Position, Roster Position, Player, Name,
    TeamAbbrev, Team, Salary, AvgPointsPerGame, GameInfo, Opponent.
    """
    out: List[Dict] = []
    for r in rows:
        name = (r.get("Player") or r.get("Name") or "").strip()
        if not name:
            continue
        pos = (r.get("Roster Position") or r.get("Position") or r.get("Positions") or "").strip()
        team = (r.get("TeamAbbrev") or r.get("Team") or "").strip()
        opp = (r.get("Opponent") or "").strip()
        game_info = (r.get("GameInfo") or r.get("Game Info") or r.get("Game") or "").strip()
        avg_fpts = _to_float(r.get("AvgPointsPerGame") or r.get("Avg Fpts") or r.get("AvgFp"))
        salary_val: Optional[int] = None
        sal_raw = r.get("Salary") or r.get("Salary($)") or r.get("Salary (DK)")
        if sal_raw is not None:
            try:
                salary_val = int(str(sal_raw).replace("$", "").replace(",", "").strip())
            except Exception:
                salary_val = None
        out.append(
            {
                "player_name": name,
                "roster_position": pos,
                "team": team,
                "opponent": opp,
                "game_info": game_info,
                "salary": salary_val,
                "avg_fpts": avg_fpts,
            }
        )
    return out


def _pct_to_float(s: str | None) -> Optional[float]:
    if not s:
        return None
    t = s.strip().replace("%", "")
    return _to_float(t)


def _explode_lineup(entry_id: str, lineup_raw: str) -> List[Dict[str, str]]:
    tokens = lineup_raw.strip().split()
    items: List[Dict[str, str]] = []
    cur_slot: Optional[str] = None
    cur_name: List[str] = []

    def flush():
        nonlocal items, cur_slot, cur_name
        if cur_slot and cur_name:
            name = " ".join(cur_name).strip()
            items.append({"entry_id": entry_id, "slot": cur_slot, "player_name": name})
        cur_slot, cur_name = None, []

    for tok in tokens:
        if tok in SLOT_TOKENS:
            flush()
            cur_slot = tok
        else:
            cur_name.append(tok)
    flush()
    return items


def _insert_contests(con: duckdb.DuckDBPyConnection, rows: List[ContestRow]) -> None:
    con.executemany(
        """
        insert or ignore into contests (
          contest_id, contest_name, start_time, start_time_readable,
          prize_pool, first_place_prize, entry_fee, max_entries,
          current_entries, game_type, is_guaranteed, is_starred,
          template_id, draft_group_id, scrape_time
        ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                r.contest_id,
                r.contest_name,
                r.start_time,
                r.start_time_readable,
                r.prize_pool,
                r.first_place_prize,
                r.entry_fee,
                r.max_entries,
                r.current_entries,
                r.game_type,
                r.is_guaranteed,
                r.is_starred,
                r.template_id,
                r.draft_group_id,
                r.scrape_time,
            )
            for r in rows
            if r.contest_id
        ],
    )


def _insert_results(
    con: duckdb.DuckDBPyConnection,
    contest_id: str,
    entries: List[Dict],
    lineup_players: List[Dict],
    ownership: List[Dict],
) -> None:
    if not contest_id:
        return

    # entries
    con.executemany(
        """
        insert or replace into entries (
          contest_id, entry_id, user_handle, entry_name, rank, points, winnings, lineup_raw
        ) values (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                contest_id,
                e.get("entry_id", ""),
                e.get("user_handle", ""),
                e.get("entry_name", ""),
                e.get("rank"),
                e.get("points"),
                e.get("winnings"),
                e.get("lineup_raw", ""),
            )
            for e in entries
            if e.get("entry_id")
        ],
    )

    # lineup_players
    con.executemany(
        """
        insert into lineup_players (contest_id, entry_id, slot, player_name)
        values (?, ?, ?, ?)
        """,
        [
            (
                contest_id,
                lp.get("entry_id", ""),
                lp.get("slot", ""),
                lp.get("player_name", ""),
            )
            for lp in lineup_players
            if lp.get("entry_id") and lp.get("player_name")
        ],
    )

    # player_ownership
    con.executemany(
        """
        insert into player_ownership (contest_id, player_name, roster_position, pct_drafted, fpts)
        values (?, ?, ?, ?, ?)
        """,
        [
            (
                contest_id,
                o.get("player_name", ""),
                o.get("roster_position", ""),
                o.get("pct_drafted"),
                o.get("fpts"),
            )
            for o in ownership
            if o.get("player_name")
        ],
    )


def _insert_contest_players(
    con: duckdb.DuckDBPyConnection, contest_id: str, rows: List[Dict]
) -> None:
    if not rows:
        return
    # Replace existing rows for this contest to avoid duplicates on re-run
    con.execute("delete from contest_players where contest_id = ?", [contest_id])
    con.executemany(
        """
        insert into contest_players (contest_id, player_name, roster_position, team, opponent, game_info, salary, avg_fpts)
        values (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                contest_id,
                r.get("player_name", ""),
                r.get("roster_position", ""),
                r.get("team", ""),
                r.get("opponent", ""),
                r.get("game_info", ""),
                r.get("salary"),
                r.get("avg_fpts"),
            )
            for r in rows
        ],
    )


def load_date(date_str: str, data_root: Path, con: duckdb.DuckDBPyConnection) -> None:
    day_dir = data_root / date_str
    csv_path = day_dir / f"nba_gpp_{date_str}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Contest CSV not found: {csv_path}")

    contests = read_contests_csv(csv_path)
    _insert_contests(con, contests)

    results_dir = day_dir / "results"
    if not results_dir.exists():
        print(f"No results directory found: {results_dir}")
        return

    # Prefer explicit files if present; otherwise fallback to legacy results.csv
    for standings_file in sorted(results_dir.glob("contest_*_standings.csv")):
        contest_id = standings_file.name.split("_")[1]
        try:
            rows = _read_csv_dicts_from_text(standings_file.read_text(encoding="utf-8", errors="ignore"))
            e, lp, po = parse_results_rows(rows)
            _insert_results(con, contest_id, e, lp, po)
            print(f"Loaded contest {contest_id} standings: entries={len(e)} own={len(po)}")
        except Exception as exc:
            print(f"Failed to load standings {standings_file}: {exc}")

    for players_file in sorted(results_dir.glob("contest_*_players.csv")):
        contest_id = players_file.name.split("_")[1]
        try:
            prows = _read_csv_dicts_from_text(players_file.read_text(encoding="utf-8", errors="ignore"))
            parsed = parse_contest_players_rows(prows)
            _insert_contest_players(con, contest_id, parsed)
            print(f"Loaded contest {contest_id} players: n={len(parsed)}")
        except Exception as exc:
            print(f"Failed to load players {players_file}: {exc}")

    # Fallback to legacy combined file if explicit not present
    for f in sorted(results_dir.glob("contest_*_results.csv")):
        contest_id = f.name.split("_")[1]
        standings_file = results_dir / f"contest_{contest_id}_standings.csv"
        if standings_file.exists():
            continue  # already loaded
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
            rows = _read_csv_dicts_from_text(text)
            e, lp, po = parse_results_rows(rows)
            _insert_results(con, contest_id, e, lp, po)
            print(f"Loaded contest {contest_id} (legacy): entries={len(e)} own={len(po)}")
        except Exception as exc:
            print(f"Failed to load legacy {f}: {exc}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Load DK NBA GPP results into DuckDB")
    p.add_argument("--date", help="YYYY-MM-DD date to load")
    p.add_argument("--data-root", default=str(DATA_ROOT))
    p.add_argument("--db", default=str(DEFAULT_DB))
    p.add_argument("--init", action="store_true", help="(Re)create schema and views")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))

    ensure_db(con)
    if args.init:
        create_views(con)

    if args.date:
        load_date(args.date, Path(args.data_root), con)
        # Refresh views after load
        create_views(con)
    else:
        print("No --date provided; schema ensured only.")


if __name__ == "__main__":
    main()
