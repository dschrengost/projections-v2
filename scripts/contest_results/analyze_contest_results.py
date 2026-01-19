#!/usr/bin/env python3
"""
Season-long analysis of DraftKings NBA contest results stored under PROJECTIONS_DATA_ROOT.

Primary inputs (default):
- PROJECTIONS_DATA_ROOT/bronze/dk_contests/nba_gpp_data/<date>/nba_gpp_<date>.csv
- PROJECTIONS_DATA_ROOT/bronze/dk_contests/nba_gpp_data/<date>/results/contest_<id>_results.csv

Optional inputs for richer lineup features (salary/team/game stacks):
- PROJECTIONS_DATA_ROOT/bronze/dk/draftables/draftables_raw_<draft_group_id>.json

Outputs (default):
- PROJECTIONS_DATA_ROOT/analytics/contest_results/contest_inventory.parquet
- PROJECTIONS_DATA_ROOT/analytics/contest_results/contest_entries_tidy.parquet
- PROJECTIONS_DATA_ROOT/analytics/contest_results/contest_cohort_summary.parquet
- PROJECTIONS_DATA_ROOT/analytics/contest_results/user_entries.parquet
- reports/contest_results/season_analysis.md (when --write-report)

The web dashboard's "Contest" tab is treated as a hypothesis only. This script re-derives
conclusions directly from the raw CSVs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


LINEUP_RE = re.compile(
    r"(?:^|\s)(PG|SG|SF|PF|UTIL|C|F|G)\s+(.+?)(?=(?:\s(?:PG|SG|SF|PF|UTIL|C|F|G)\s)|$)"
)

TIDY_COLS: list[str] = [
    "date",
    "contest_id",
    "draft_group_id",
    "contest_name",
    "contest_class",
    "entry_limit_bucket",
    "entry_fee",
    "prize_pool",
    "first_place_prize",
    "is_low_stakes",
    "is_flagship",
    "entry_id",
    "entry_name",
    "user_name",
    "is_user",
    "rank",
    "points",
    "lineup_key",
    "dupe_count",
    "is_dupe",
    "own_total",
    "own_avg",
    "own_min",
    "own_max",
    "own_gini",
    "own_num_lt10",
    "own_num_lt5",
    "own_num_gt50",
    "salary_total",
    "salary_left",
    "salary_gini",
    "salary_min",
    "salary_max",
    "num_teams",
    "max_from_team",
    "num_games",
    "max_from_game",
    "has_bring_back",
    "sum_player_fpts",
]

TIDY_SCHEMA = pa.schema(
    [
        pa.field("date", pa.string()),
        pa.field("contest_id", pa.string()),
        pa.field("draft_group_id", pa.int64()),
        pa.field("contest_name", pa.string()),
        pa.field("contest_class", pa.string()),
        pa.field("entry_limit_bucket", pa.string()),
        pa.field("entry_fee", pa.float64()),
        pa.field("prize_pool", pa.float64()),
        pa.field("first_place_prize", pa.float64()),
        pa.field("is_low_stakes", pa.bool_()),
        pa.field("is_flagship", pa.bool_()),
        pa.field("entry_id", pa.int64()),
        pa.field("entry_name", pa.string()),
        pa.field("user_name", pa.string()),
        pa.field("is_user", pa.bool_()),
        pa.field("rank", pa.int32()),
        pa.field("points", pa.float64()),
        pa.field("lineup_key", pa.string()),
        pa.field("dupe_count", pa.int32()),
        pa.field("is_dupe", pa.bool_()),
        pa.field("own_total", pa.float64()),
        pa.field("own_avg", pa.float64()),
        pa.field("own_min", pa.float64()),
        pa.field("own_max", pa.float64()),
        pa.field("own_gini", pa.float64()),
        pa.field("own_num_lt10", pa.int32()),
        pa.field("own_num_lt5", pa.int32()),
        pa.field("own_num_gt50", pa.int32()),
        pa.field("salary_total", pa.int32()),
        pa.field("salary_left", pa.int32()),
        pa.field("salary_gini", pa.float64()),
        pa.field("salary_min", pa.int32()),
        pa.field("salary_max", pa.int32()),
        pa.field("num_teams", pa.int32()),
        pa.field("max_from_team", pa.int32()),
        pa.field("num_games", pa.int32()),
        pa.field("max_from_game", pa.int32()),
        pa.field("has_bring_back", pa.bool_()),
        pa.field("sum_player_fpts", pa.float64()),
    ]
)


def get_data_root(data_root: str | None) -> Path:
    if data_root:
        return Path(data_root)
    env_root = os.environ.get("PROJECTIONS_DATA_ROOT")
    if env_root:
        return Path(env_root)
    return Path.home() / "projections-data"


def contest_data_dir(data_root: Path) -> Path:
    return data_root / "bronze" / "dk_contests" / "nba_gpp_data"


def draftables_dir(data_root: Path) -> Path:
    return data_root / "bronze" / "dk" / "draftables"


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace("\ufeff", "", regex=False)
        .str.replace("ï»¿", "", regex=False)
    )
    return df


def looks_like_zip(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            return f.read(2) == b"PK"
    except Exception:
        return False


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if isinstance(value, str) and value.strip().lower() in {"nan", ""}:
            return None
        f = float(value)
        return None if math.isnan(f) else f
    except Exception:
        return None


def safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        if isinstance(value, str) and value.strip().lower() in {"nan", ""}:
            return None
        i = int(float(value))
        return i
    except Exception:
        return None


def parse_first_place_prize(value: Any) -> float | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in {"not specified", "nan"}:
        return None
    s = s.replace("$", "").replace(",", "").strip()
    m = re.match(r"^(\d+(?:\.\d+)?)([KMB])?$", s, flags=re.IGNORECASE)
    if not m:
        return None
    num = float(m.group(1))
    suffix = (m.group(2) or "").upper()
    mult = {"": 1.0, "K": 1_000.0, "M": 1_000_000.0, "B": 1_000_000_000.0}.get(
        suffix
    )
    return None if mult is None else num * mult


def parse_username(entry_name: str) -> str | None:
    name = (entry_name or "").strip()
    if not name:
        return None
    if " (" in name:
        name = name.split(" (", 1)[0].strip()
    return name or None


def parse_lineup_players(lineup_str: str) -> list[str]:
    if not isinstance(lineup_str, str) or not lineup_str.strip():
        return []
    return [m.group(2).strip() for m in LINEUP_RE.finditer(lineup_str) if m.group(2)]


def build_user_mask(
    user_series: pd.Series,
    user_pattern: str,
    *,
    match: str,
) -> pd.Series:
    s = user_series.astype("string")
    if not user_pattern:
        return pd.Series(False, index=s.index)

    match = (match or "").strip().lower()
    if match == "exact":
        return s.str.lower() == user_pattern.lower()

    pattern = re.escape(user_pattern)
    return s.str.contains(pattern, case=False, na=False)


def gini(values: list[float]) -> float | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    if np.all(arr == 0):
        return 0.0
    arr = np.sort(arr)
    n = arr.size
    idx = np.arange(1, n + 1)
    return float((np.sum((2 * idx - n - 1) * arr)) / (n * np.sum(arr)))


def infer_contest_class(contest_name: str) -> str:
    n = (contest_name or "").lower()
    if any(x in n for x in ["double up", "50/50", "head-to-head", "h2h"]):
        return "cash"
    if any(x in n for x in ["satellite", "qualifier", "ticket", "step"]):
        return "satellite"
    if "multiplier" in n:
        return "multiplier"
    return "gpp"


def infer_entry_limit_bucket(max_entries: int | None, contest_name: str) -> str:
    n = (contest_name or "").lower()
    if max_entries == 1 or "single entry" in n:
        return "se"
    if max_entries in {3, 5, 10, 20, 50, 150}:
        return f"{max_entries}-max"
    if max_entries is None:
        return "unknown"
    if max_entries > 150:
        return "large-max"
    return f"{max_entries}-max"


@dataclass(frozen=True)
class DraftablesMaps:
    player_to_salary: dict[str, int]
    player_to_team: dict[str, str]
    player_to_game: dict[str, int]


def load_draftables_maps(path: Path) -> DraftablesMaps | None:
    if not path.exists():
        return None
    try:
        obj = json.loads(path.read_text())
    except Exception:
        return None
    draftables = obj.get("draftables")
    if not isinstance(draftables, list) or not draftables:
        return None

    player_to_salary: dict[str, int] = {}
    player_to_team: dict[str, str] = {}
    player_to_game: dict[str, int] = {}

    for d in draftables:
        if not isinstance(d, dict):
            continue
        name = str(d.get("displayName") or "").strip()
        if not name:
            continue
        salary = safe_int(d.get("salary"))
        team = str(d.get("teamAbbreviation") or "").strip()
        comp = d.get("competition") or {}
        game_id = safe_int(comp.get("competitionId"))

        if salary is not None:
            player_to_salary[name] = max(salary, player_to_salary.get(name, -1))
        if team:
            player_to_team[name] = team
        if game_id is not None:
            player_to_game[name] = game_id

    return DraftablesMaps(
        player_to_salary=player_to_salary,
        player_to_team=player_to_team,
        player_to_game=player_to_game,
    )


def build_player_lookups(df_raw: pd.DataFrame) -> tuple[dict[str, float], dict[str, float]]:
    if "Player" not in df_raw.columns:
        return {}, {}

    player = df_raw["Player"].astype(str)

    own: dict[str, float] = {}
    if "%Drafted" in df_raw.columns:
        s = df_raw["%Drafted"].astype(str).str.replace("%", "", regex=False)
        own_series = pd.to_numeric(s, errors="coerce")
        own = (
            pd.DataFrame({"Player": player, "own": own_series})
            .dropna(subset=["Player", "own"])
            .groupby("Player")["own"]
            .max()
            .to_dict()
        )

    fpts: dict[str, float] = {}
    if "FPTS" in df_raw.columns:
        fpts_series = pd.to_numeric(df_raw["FPTS"], errors="coerce")
        fpts = (
            pd.DataFrame({"Player": player, "fpts": fpts_series})
            .dropna(subset=["Player", "fpts"])
            .groupby("Player")["fpts"]
            .max()
            .to_dict()
        )

    return own, fpts


def compute_lineup_features(
    lineup_players: list[str],
    player_own: dict[str, float],
    player_fpts: dict[str, float],
    draftables: DraftablesMaps | None,
    salary_cap: int,
) -> dict[str, Any]:
    if not lineup_players:
        return {}

    own_vals = [player_own.get(p) for p in lineup_players]
    own_vals_num = [float(v) for v in own_vals if v is not None and math.isfinite(v)]

    fpts_vals = [player_fpts.get(p) for p in lineup_players]
    fpts_vals_num = [float(v) for v in fpts_vals if v is not None and math.isfinite(v)]

    out: dict[str, Any] = {
        "own_total": float(np.sum(own_vals_num)) if own_vals_num else None,
        "own_avg": float(np.mean(own_vals_num)) if own_vals_num else None,
        "own_min": float(np.min(own_vals_num)) if own_vals_num else None,
        "own_max": float(np.max(own_vals_num)) if own_vals_num else None,
        "own_gini": gini(own_vals_num),
        "own_num_lt10": int(sum(1 for v in own_vals_num if v < 10)) if own_vals_num else None,
        "own_num_lt5": int(sum(1 for v in own_vals_num if v < 5)) if own_vals_num else None,
        "own_num_gt50": int(sum(1 for v in own_vals_num if v > 50)) if own_vals_num else None,
        "sum_player_fpts": float(np.sum(fpts_vals_num)) if fpts_vals_num else None,
    }

    if draftables is None:
        out.update(
            {
                "salary_total": None,
                "salary_left": None,
                "salary_gini": None,
                "salary_min": None,
                "salary_max": None,
                "num_teams": None,
                "max_from_team": None,
                "num_games": None,
                "max_from_game": None,
                "has_bring_back": None,
            }
        )
        return out

    salaries = [draftables.player_to_salary.get(p) for p in lineup_players]
    teams = [draftables.player_to_team.get(p) for p in lineup_players]
    games = [draftables.player_to_game.get(p) for p in lineup_players]

    salary_total = None
    salary_left = None
    salary_vals = [float(s) for s in salaries if s is not None]
    if len(salary_vals) == len(lineup_players):
        salary_total = int(sum(salary_vals))
        salary_left = int(salary_cap - salary_total)

    team_vals = [t for t in teams if t]
    game_vals = [g for g in games if g is not None]

    team_counts = pd.Series(team_vals).value_counts() if team_vals else pd.Series(dtype=int)
    game_counts = pd.Series(game_vals).value_counts() if game_vals else pd.Series(dtype=int)

    has_bring_back = None
    if len(game_vals) == len(lineup_players) and len(team_vals) == len(lineup_players):
        bring_back = False
        for game_id, cnt in game_counts.items():
            if cnt < 2:
                continue
            teams_in_game = [t for t, g in zip(team_vals, game_vals) if g == game_id and t]
            if len(set(teams_in_game)) >= 2:
                bring_back = True
                break
        has_bring_back = bring_back

    out.update(
        {
            "salary_total": salary_total,
            "salary_left": salary_left,
            "salary_gini": gini(salary_vals) if salary_vals else None,
            "salary_min": int(min(salary_vals)) if salary_vals else None,
            "salary_max": int(max(salary_vals)) if salary_vals else None,
            "num_teams": int(team_counts.size) if not team_counts.empty else 0,
            "max_from_team": int(team_counts.max()) if not team_counts.empty else 0,
            "num_games": int(game_counts.size) if not game_counts.empty else 0,
            "max_from_game": int(game_counts.max()) if not game_counts.empty else 0,
            "has_bring_back": has_bring_back,
        }
    )
    return out


def discover_contests(contest_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for date_dir in sorted(contest_dir.iterdir()):
        if not date_dir.is_dir() or not re.match(r"^\d{4}-\d{2}-\d{2}$", date_dir.name):
            continue
        date = date_dir.name
        meta_path = date_dir / f"nba_gpp_{date}.csv"
        if not meta_path.exists():
            continue
        meta = normalize_columns(pd.read_csv(meta_path, encoding="utf-8-sig"))

        cid_col = "contest_id" if "contest_id" in meta.columns else "ContestId"
        name_col = "contest_name" if "contest_name" in meta.columns else "ContestName"

        for _, r in meta.iterrows():
            contest_id = str(r.get(cid_col, "")).strip()
            if not contest_id or contest_id == "nan":
                continue

            results_path = date_dir / "results" / f"contest_{contest_id}_results.csv"
            if not results_path.exists():
                continue

            contest_name = str(r.get(name_col) or "").strip()
            entry_fee = safe_float(r.get("entry_fee", r.get("EntryFee")))
            prize_pool = safe_float(r.get("prize_pool", r.get("PrizePool")))
            first_place_prize = parse_first_place_prize(
                r.get("first_place_prize", r.get("FirstPlacePrize"))
            )
            current_entries = safe_int(r.get("current_entries", r.get("CurrentEntries")))
            max_entries = safe_int(r.get("max_entries", r.get("MaxEntries")))
            draft_group_id = safe_int(r.get("draft_group_id", r.get("DraftGroupId")))
            start_time = str(r.get("start_time_readable", r.get("StartTimeReadable")) or "")

            rows.append(
                {
                    "date": date,
                    "contest_id": contest_id,
                    "contest_name": contest_name,
                    "entry_fee": entry_fee,
                    "prize_pool": prize_pool,
                    "first_place_prize": first_place_prize,
                    "current_entries_meta": current_entries,
                    "max_entries_per_user": max_entries,
                    "draft_group_id": draft_group_id,
                    "start_time": start_time,
                    "results_path": str(results_path),
                    "results_is_zip": looks_like_zip(results_path),
                    "contest_class": infer_contest_class(contest_name),
                    "entry_limit_bucket": infer_entry_limit_bucket(max_entries, contest_name),
                }
            )

    inv = pd.DataFrame(rows)
    if inv.empty:
        return inv

    inv["entry_fee"] = pd.to_numeric(inv["entry_fee"], errors="coerce")
    inv["is_low_stakes"] = inv["entry_fee"].fillna(np.inf) <= 5.0

    # Flagship contest per slate (date + draft_group_id): max prize pool, tie-break max entries.
    inv["is_flagship"] = False
    eligible = inv[inv["draft_group_id"].notna()].copy()
    if not eligible.empty:
        eligible["prize_pool_fill"] = eligible["prize_pool"].fillna(-1.0)
        eligible["entries_fill"] = eligible["current_entries_meta"].fillna(-1).astype(float)
        idx = (
            eligible.sort_values(["prize_pool_fill", "entries_fill"], ascending=False)
            .groupby(["date", "draft_group_id"], dropna=False)
            .head(1)
            .index
        )
        inv.loc[idx, "is_flagship"] = True

    return inv


def summarize_cohorts(entries: pd.DataFrame, cash_pct: float) -> pd.DataFrame:
    if entries.empty:
        return pd.DataFrame()
    n = len(entries)
    top0_1 = max(1, int(math.ceil(0.001 * n)))
    top1 = max(1, int(math.ceil(0.01 * n)))
    top5 = max(1, int(math.ceil(0.05 * n)))
    cash_n = max(1, int(math.ceil(cash_pct * n)))

    cohorts: list[tuple[str, pd.DataFrame]] = [
        ("winner", entries[entries["rank"] == 1]),
        ("top_0_1_pct", entries[entries["rank"] <= top0_1]),
        ("top_1_pct", entries[entries["rank"] <= top1]),
        ("top_5_pct", entries[entries["rank"] <= top5]),
        ("min_cash", entries[entries["rank"] <= cash_n]),
        ("field", entries),
    ]

    numeric_cols = [
        c
        for c in entries.columns
        if pd.api.types.is_numeric_dtype(entries[c]) and c not in {"entry_id"}
    ]

    rows: list[dict[str, Any]] = []
    for name, df in cohorts:
        if df.empty:
            continue
        row: dict[str, Any] = {"cohort": name, "n": int(len(df))}
        for col in numeric_cols:
            row[f"{col}__mean"] = float(df[col].mean()) if df[col].notna().any() else None
            row[f"{col}__median"] = (
                float(df[col].median()) if df[col].notna().any() else None
            )
        rows.append(row)

    # Field median as its own cohort row (explicit requirement)
    median_row: dict[str, Any] = {"cohort": "field_median", "n": int(len(entries))}
    for col in numeric_cols:
        median_row[f"{col}__mean"] = None
        median_row[f"{col}__median"] = (
            float(entries[col].median()) if entries[col].notna().any() else None
        )
    rows.append(median_row)

    return pd.DataFrame(rows)


def process_contest(
    inv_row: dict[str, Any],
    draftables_cache: dict[int, DraftablesMaps | None],
    draftables_base: Path,
    salary_cap: int,
    cash_pct: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    results_path = Path(inv_row["results_path"])
    if inv_row.get("results_is_zip") or looks_like_zip(results_path):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {"skipped": "zip_like"}

    try:
        wanted_cols = {"Rank", "EntryId", "EntryName", "Points", "Lineup", "Player", "%Drafted", "FPTS"}
        df_raw = pd.read_csv(
            results_path,
            encoding="utf-8-sig",
            low_memory=False,
            usecols=lambda c: c in wanted_cols,
        )
        df_raw = normalize_columns(df_raw)
    except Exception as e:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {"skipped": f"read_error: {e}"}

    required = {"Rank", "EntryId", "EntryName", "Points", "Lineup"}
    if not required.issubset(set(df_raw.columns)):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {"skipped": "missing_columns"}

    player_own, player_fpts = build_player_lookups(df_raw)

    entries = (
        df_raw[df_raw["EntryId"].notna()]
        .drop_duplicates(subset=["EntryId"])
        .rename(
            columns={
                "Rank": "rank",
                "EntryId": "entry_id",
                "EntryName": "entry_name",
                "Points": "points",
                "Lineup": "lineup_str",
            }
        )
        .copy()
    )

    entries["rank"] = pd.to_numeric(entries["rank"], errors="coerce").astype("Int32")
    entries["entry_id"] = pd.to_numeric(entries["entry_id"], errors="coerce").astype("Int64")
    entries["points"] = pd.to_numeric(entries["points"], errors="coerce")

    entries_total = int(len(entries))
    entries["has_lineup"] = entries["lineup_str"].notna() & (
        entries["lineup_str"].astype(str).str.len() > 0
    )
    entries_with_lineup = int(entries["has_lineup"].sum())

    entries = entries[entries["has_lineup"]].copy()
    if entries.empty:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            {
                "entries_total": entries_total,
                "entries_with_lineup": entries_with_lineup,
                "pct_with_lineup": entries_with_lineup / entries_total if entries_total else None,
                "skipped": "no_lineups",
            },
        )

    entries["user_name"] = entries["entry_name"].astype(str).map(parse_username)

    lineup_players = entries["lineup_str"].astype(str).map(parse_lineup_players)
    entries["lineup_size"] = lineup_players.map(len).astype("int16")
    entries = entries[entries["lineup_size"] == 8].copy()
    if entries.empty:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            {
                "entries_total": entries_total,
                "entries_with_lineup": entries_with_lineup,
                "pct_with_lineup": entries_with_lineup / entries_total if entries_total else None,
                "skipped": "no_valid_lineups",
            },
        )

    entries["lineup_key"] = lineup_players.loc[entries.index].map(
        lambda ps: "|".join(sorted(ps))
    )
    dupe_counts = entries["lineup_key"].value_counts()
    entries["dupe_count"] = entries["lineup_key"].map(dupe_counts).astype("int32")
    entries["is_dupe"] = entries["dupe_count"] > 1

    # Load draftables maps if available
    draft_group_id = inv_row.get("draft_group_id")
    draftables = None
    if isinstance(draft_group_id, int):
        if draft_group_id not in draftables_cache:
            draftables_cache[draft_group_id] = load_draftables_maps(
                draftables_base / f"draftables_raw_{draft_group_id}.json"
            )
        draftables = draftables_cache[draft_group_id]

    feat_rows: list[dict[str, Any]] = []
    for ps in lineup_players.loc[entries.index]:
        feat_rows.append(
            compute_lineup_features(
                ps,
                player_own=player_own,
                player_fpts=player_fpts,
                draftables=draftables,
                salary_cap=salary_cap,
            )
        )
    feat_df = pd.DataFrame(feat_rows, index=entries.index)
    entries = pd.concat([entries, feat_df], axis=1)

    # Cohort summaries and user rows
    cohort_summary = summarize_cohorts(entries, cash_pct=cash_pct)
    if not cohort_summary.empty:
        cohort_summary.insert(0, "contest_id", str(inv_row["contest_id"]))
        cohort_summary.insert(0, "date", str(inv_row["date"]))

    user_pattern = str(inv_row.get("_user", "")).strip()
    user_match = str(inv_row.get("_user_match", "contains")).strip().lower()
    user_mask = build_user_mask(entries["user_name"], user_pattern, match=user_match)
    user_entries = entries[user_mask].copy()

    stats = {
        "entries_total": entries_total,
        "entries_with_lineup": entries_with_lineup,
        "pct_with_lineup": entries_with_lineup / entries_total if entries_total else None,
        "entries_analyzed": int(len(entries)),
        "draftables_available": bool(draftables is not None),
        "dupe_rate": float(entries["is_dupe"].mean()) if len(entries) else None,
        "avg_dupe_count": float(entries["dupe_count"].mean()) if len(entries) else None,
        "skipped": None,
    }
    return entries, cohort_summary, user_entries, stats


def rebuild_user_entries_from_tidy(
    *,
    entries_path: Path,
    out_path: Path,
    user_pattern: str,
    user_match: str,
) -> None:
    if not entries_path.exists():
        raise SystemExit(f"Tidy dataset not found: {entries_path}")
    if not user_pattern:
        raise SystemExit("--user is required for --rebuild-user-entries")

    parquet_file = pq.ParquetFile(entries_path)
    batches: list[pd.DataFrame] = []

    for batch in parquet_file.iter_batches(batch_size=250_000, columns=TIDY_COLS):
        df = batch.to_pandas()
        mask = build_user_mask(df["user_name"], user_pattern, match=user_match)
        if bool(mask.any()):
            batches.append(df.loc[mask])

    user_df = (
        pd.concat(batches, ignore_index=True)
        if batches
        else pd.DataFrame(columns=TIDY_COLS)
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    user_df.to_parquet(out_path, index=False)
    print(f"Rebuilt user entries: {len(user_df)} rows -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--user", type=str, default="angrydingo")
    parser.add_argument("--user-match", type=str, default="contains", choices=["contains", "exact"])
    parser.add_argument("--salary-cap", type=int, default=50000)
    parser.add_argument("--cash-pct", type=float, default=0.2)
    parser.add_argument("--max-contests", type=int, default=None)
    parser.add_argument("--write-dataset", action="store_true")
    parser.add_argument("--write-report", action="store_true")
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--rebuild-user-entries", action="store_true")
    args = parser.parse_args()

    data_root = get_data_root(args.data_root)
    contest_dir = contest_data_dir(data_root)
    if not contest_dir.exists():
        raise SystemExit(f"Contest dir not found: {contest_dir}")

    out_dir = Path(args.out_dir) if args.out_dir else data_root / "analytics" / "contest_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    entries_path = out_dir / "contest_entries_tidy.parquet"
    user_path = out_dir / "user_entries.parquet"

    if args.rebuild_user_entries:
        rebuild_user_entries_from_tidy(
            entries_path=entries_path,
            out_path=user_path,
            user_pattern=args.user,
            user_match=args.user_match,
        )
        return

    inventory = discover_contests(contest_dir)
    if inventory.empty:
        raise SystemExit(f"No contests found under {contest_dir}")

    # Required: brief inventory summary before proceeding
    inv_nonzip = inventory[~inventory["results_is_zip"]].copy()
    print("== Contest Results Inventory ==")
    print(f"dates: {inventory['date'].nunique()} ({inventory['date'].min()} → {inventory['date'].max()})")
    print(f"contests_with_results: {len(inventory)}")
    print(f"zip_like_results: {int(inventory['results_is_zip'].sum())}")
    print(f"low_stakes_contests: {int(inventory['is_low_stakes'].sum())}")
    print(f"flagship_contests: {int(inventory['is_flagship'].sum())}")
    if not inv_nonzip.empty and inv_nonzip["entry_fee"].notna().any():
        print("entry_fee percentiles:", inv_nonzip["entry_fee"].quantile([0.1, 0.5, 0.9]).to_dict())

    inv_path = out_dir / "contest_inventory.parquet"
    inventory.to_parquet(inv_path, index=False)

    if args.inventory_only:
        print("Inventory written:", inv_path)
        return

    # Process contests
    draftables_cache: dict[int, DraftablesMaps | None] = {}
    writer: pq.ParquetWriter | None = None

    cohort_rows: list[pd.DataFrame] = []
    user_rows: list[pd.DataFrame] = []
    stats_rows: list[dict[str, Any]] = []

    rows = inventory.to_dict("records")
    if args.max_contests is not None:
        rows = rows[: args.max_contests]

    if args.write_dataset:
        if entries_path.exists():
            entries_path.unlink()
        writer = pq.ParquetWriter(entries_path, TIDY_SCHEMA, compression="zstd")

    for i, r in enumerate(rows, start=1):
        r = {**r, "_user": args.user, "_user_match": args.user_match}
        entries, cohort_summary, user_entries, stats = process_contest(
            r,
            draftables_cache=draftables_cache,
            draftables_base=draftables_dir(data_root),
            salary_cap=args.salary_cap,
            cash_pct=args.cash_pct,
        )

        stats_rows.append({"date": r["date"], "contest_id": r["contest_id"], **stats})
        if not cohort_summary.empty:
            cohort_rows.append(cohort_summary)
        if not user_entries.empty:
            user_rows.append(user_entries)

        if writer is not None and not entries.empty:
            # Attach contest-level metadata onto each entry row.
            meta_cols = [
                "date",
                "contest_id",
                "draft_group_id",
                "contest_name",
                "contest_class",
                "entry_limit_bucket",
                "entry_fee",
                "prize_pool",
                "first_place_prize",
                "is_low_stakes",
                "is_flagship",
            ]
            for c in meta_cols:
                entries[c] = r.get(c)

            entries["is_user"] = build_user_mask(
                entries["user_name"],
                args.user,
                match=args.user_match,
            )

            for col in TIDY_COLS:
                if col not in entries.columns:
                    entries[col] = pd.NA
            tidy = entries[TIDY_COLS].copy()
            table = pa.Table.from_pandas(tidy, schema=TIDY_SCHEMA, preserve_index=False)
            writer.write_table(table)

        if i % 50 == 0 or i == len(rows):
            suffix = ""
            if stats.get("skipped"):
                suffix = f" skipped={stats['skipped']}"
            else:
                suffix = f" entries={stats.get('entries_analyzed')}, draftables={stats.get('draftables_available')}"
            print(f"[{i}/{len(rows)}] {r['date']} contest={r['contest_id']}{suffix}")

    if writer is not None:
        writer.close()

    stats_df = pd.DataFrame(stats_rows)
    stats_path = out_dir / "contest_processing_stats.parquet"
    stats_df.to_parquet(stats_path, index=False)

    cohort_df = pd.concat(cohort_rows, ignore_index=True) if cohort_rows else pd.DataFrame()
    cohort_path = out_dir / "contest_cohort_summary.parquet"
    cohort_df.to_parquet(cohort_path, index=False)

    inventory_plus = inventory.merge(stats_df, on=["date", "contest_id"], how="left")
    inv_plus_path = out_dir / "contest_inventory_with_field_sizes.parquet"
    inventory_plus.to_parquet(inv_plus_path, index=False)

    user_df = pd.concat(user_rows, ignore_index=True) if user_rows else pd.DataFrame()
    user_df.to_parquet(user_path, index=False)

    if args.write_report:
        report_path = Path("reports/contest_results/season_analysis.md")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            "\n".join(
                [
                    "# Season Contest Results Analysis",
                    "",
                    "_Generated scaffold. See script outputs in_",
                    f"`{out_dir}`",
                    "",
                    "## Inventory (brief)",
                    f"- Date range: `{inventory['date'].min()}` → `{inventory['date'].max()}`",
                    f"- Contests with result files: {len(inventory)}",
                    f"- Zip-like/corrupt result files: {int(inventory['results_is_zip'].sum())}",
                    f"- Tidy dataset: `{entries_path}`",
                    f"- Cohort summary: `{cohort_path}`",
                    f"- User entries: `{user_path}`",
                    "",
                    "## Next steps",
                    "- Run the script without `--max-contests` to process the full dataset (can take a while).",
                    "- Use the saved parquet outputs to build the full narrative report with segmented comparisons.",
                    "",
                ]
            )
        )

    print("Inventory written:", inv_path)
    print("Inventory + field sizes written:", inv_plus_path)
    print("Cohort summary written:", cohort_path)
    print("User entries written:", user_path)
    print("Processing stats written:", stats_path)
    if args.write_dataset:
        print("Tidy dataset written:", entries_path)


if __name__ == "__main__":
    main()
