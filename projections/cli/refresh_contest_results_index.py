from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import typer

from projections.api.contest_service import parse_contest_csv, parse_lineup
from projections.paths import get_data_root

app = typer.Typer(add_completion=False, help="Refresh normalized contest result indexes from raw DK CSVs.")


def _coerce_float(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip().replace("$", "").replace(",", "").replace("%", "")
    if not text or text.lower() == "nan":
        return None
    multiplier = 1.0
    if text.endswith("K"):
        text = text[:-1]
        multiplier = 1000.0
    try:
        return float(text) * multiplier
    except (TypeError, ValueError):
        return None


def _coerce_int(value: object) -> Optional[int]:
    num = _coerce_float(value)
    return int(num) if num is not None else None


def _entry_limit_bucket(max_entries: Optional[int]) -> str:
    if max_entries is None:
        return "unknown"
    if max_entries <= 1:
        return "single-entry"
    if max_entries <= 3:
        return "3-max"
    if max_entries <= 5:
        return "5-max"
    if max_entries <= 9:
        return "9-max"
    if max_entries <= 20:
        return "20-max"
    if max_entries <= 150:
        return "150-max"
    return "mass-multi-entry"


def _user_name(entry_name: str) -> str:
    text = str(entry_name or "").strip()
    return re.sub(r"\s+\(\d+/\d+\)\s*$", "", text).strip()


def _lineup_key(lineup_players: List[str]) -> str:
    cleaned = [str(name).strip() for name in lineup_players if str(name).strip()]
    return "|".join(sorted(cleaned))


def _ownership_lookup(df: pd.DataFrame) -> Dict[str, float]:
    if "Player" not in df.columns or "%Drafted" not in df.columns:
        return {}
    own: Dict[str, float] = {}
    subset = df[["Player", "%Drafted"]].dropna().drop_duplicates(subset=["Player"])
    for _, row in subset.iterrows():
        player = str(row["Player"]).strip()
        pct = _coerce_float(row["%Drafted"])
        if player and pct is not None:
            own[player] = float(pct)
    return own


def _results_paths_for_date(date_dir: Path) -> List[Path]:
    results_dir = date_dir / "results"
    if not results_dir.exists():
        return []
    return sorted(
        {
            *results_dir.glob("contest_*_results.csv"),
            *results_dir.glob("contest_*_standings.csv"),
        }
    )


def _normalize_date(
    *,
    date_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    game_date = date_dir.name
    meta_path = date_dir / f"nba_gpp_{game_date}.csv"
    if not meta_path.exists():
        return pd.DataFrame(), pd.DataFrame()

    meta_df = pd.read_csv(meta_path)
    meta_df["contest_id"] = meta_df["contest_id"].astype(str)
    meta_by_contest = {str(row["contest_id"]): row for _, row in meta_df.iterrows()}

    inventory_rows: List[Dict[str, object]] = []
    user_entry_rows: List[Dict[str, object]] = []

    for results_path in _results_paths_for_date(date_dir):
        contest_id = results_path.stem.replace("contest_", "").replace("_results", "").replace("_standings", "")
        meta = meta_by_contest.get(str(contest_id))
        if meta is None:
            continue

        try:
            df = parse_contest_csv(results_path)
        except Exception:
            continue
        if df.empty or "EntryId" not in df.columns or "Lineup" not in df.columns:
            continue

        prize_values = [
            value
            for value in (_coerce_float(item) for item in df.get("Prize", pd.Series(dtype=object)).tolist())
            if value is not None
        ]
        max_entries = _coerce_int(meta.get("max_entries"))
        inventory_rows.append(
            {
                "date": game_date,
                "contest_id": str(contest_id),
                "contest_name": meta.get("contest_name"),
                "entry_fee": _coerce_float(meta.get("entry_fee")),
                "prize_pool": _coerce_float(meta.get("prize_pool")),
                "first_place_prize": _coerce_float(meta.get("first_place_prize")) or (max(prize_values) if prize_values else None),
                "current_entries_meta": _coerce_int(meta.get("current_entries")),
                "max_entries_per_user": max_entries,
                "draft_group_id": _coerce_int(meta.get("draft_group_id")),
                "start_time": meta.get("start_time_readable"),
                "results_path": str(results_path),
                "results_is_zip": bool((results_path.parent / f"contest_{contest_id}_players.csv").exists()),
                "contest_class": "gpp",
                "entry_limit_bucket": _entry_limit_bucket(max_entries),
                "is_low_stakes": (_coerce_float(meta.get("entry_fee")) or 0.0) <= 5.0,
                "is_flagship": bool(meta.get("is_starred")),
            }
        )

        deduped = df.drop_duplicates(subset=["EntryId"]).copy()
        ownership = _ownership_lookup(df)
        lineup_keys = []
        lineup_ownership_totals = []
        for _, row in deduped.iterrows():
            lineup_players = parse_lineup(str(row.get("Lineup") or ""))
            lineup_key = _lineup_key(lineup_players)
            own_values = [ownership[player] for player in lineup_players if player in ownership]
            lineup_keys.append(lineup_key)
            lineup_ownership_totals.append(own_values)

        dupe_counts = pd.Series(lineup_keys).value_counts().to_dict()

        for idx, (_, row) in enumerate(deduped.iterrows()):
            entry_name = str(row.get("EntryName") or "").strip()
            user_name = _user_name(entry_name)
            own_values = lineup_ownership_totals[idx]
            lineup_key = lineup_keys[idx]
            user_entry_rows.append(
                {
                    "date": game_date,
                    "contest_id": str(contest_id),
                    "draft_group_id": _coerce_int(meta.get("draft_group_id")),
                    "contest_name": meta.get("contest_name"),
                    "contest_class": "gpp",
                    "entry_limit_bucket": _entry_limit_bucket(max_entries),
                    "entry_fee": _coerce_float(meta.get("entry_fee")),
                    "prize_pool": _coerce_float(meta.get("prize_pool")),
                    "first_place_prize": _coerce_float(meta.get("first_place_prize")),
                    "is_low_stakes": (_coerce_float(meta.get("entry_fee")) or 0.0) <= 5.0,
                    "is_flagship": bool(meta.get("is_starred")),
                    "entry_id": _coerce_int(row.get("EntryId")),
                    "entry_name": entry_name,
                    "user_name": user_name,
                    "is_user": False,
                    "rank": _coerce_int(row.get("Rank")),
                    "points": _coerce_float(row.get("Points")),
                    "lineup_key": lineup_key,
                    "dupe_count": int(dupe_counts.get(lineup_key, 1)),
                    "is_dupe": int(dupe_counts.get(lineup_key, 1)) > 1,
                    "own_total": float(sum(own_values)) if own_values else None,
                    "own_avg": float(sum(own_values) / len(own_values)) if own_values else None,
                    "own_min": float(min(own_values)) if own_values else None,
                    "own_max": float(max(own_values)) if own_values else None,
                    "own_gini": None,
                    "own_num_lt10": sum(value < 10 for value in own_values) if own_values else None,
                    "own_num_lt5": sum(value < 5 for value in own_values) if own_values else None,
                    "own_num_gt50": sum(value > 50 for value in own_values) if own_values else None,
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
                    "sum_player_fpts": _coerce_float(row.get("Points")),
                }
            )

    return pd.DataFrame(inventory_rows), pd.DataFrame(user_entry_rows)


def _merge_by_dates(existing_path: Path, incoming: pd.DataFrame, dates: List[str]) -> pd.DataFrame:
    if existing_path.exists():
        existing = pd.read_parquet(existing_path)
        if "date" in existing.columns:
            existing = existing[~existing["date"].astype(str).isin(dates)].copy()
        merged = pd.concat([existing, incoming], ignore_index=True, sort=False)
    else:
        merged = incoming.copy()
    if "date" in merged.columns:
        merged = merged.sort_values(["date"], kind="stable").reset_index(drop=True)
    return merged


@app.command()
def main(
    start_date: str = typer.Option(..., help="Inclusive YYYY-MM-DD start date"),
    end_date: str = typer.Option(..., help="Inclusive YYYY-MM-DD end date"),
) -> None:
    data_root = get_data_root()
    raw_root = data_root / "bronze" / "dk_contests" / "nba_gpp_data"
    target_dates = [
        path.name
        for path in sorted(raw_root.iterdir())
        if path.is_dir() and start_date <= path.name <= end_date
    ]

    inventory_frames: List[pd.DataFrame] = []
    user_entry_frames: List[pd.DataFrame] = []
    for date in target_dates:
        inventory_df, user_entries_df = _normalize_date(date_dir=raw_root / date)
        if not inventory_df.empty:
            inventory_frames.append(inventory_df)
        if not user_entries_df.empty:
            user_entry_frames.append(user_entries_df)

    inventory_all = pd.concat(inventory_frames, ignore_index=True, sort=False) if inventory_frames else pd.DataFrame()
    user_entries_all = pd.concat(user_entry_frames, ignore_index=True, sort=False) if user_entry_frames else pd.DataFrame()

    out_dir = data_root / "analytics" / "contest_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    inventory_path = out_dir / "contest_inventory.parquet"
    user_entries_path = out_dir / "user_entries.parquet"

    merged_inventory = _merge_by_dates(inventory_path, inventory_all, target_dates)
    merged_user_entries = _merge_by_dates(user_entries_path, user_entries_all, target_dates)

    merged_inventory.to_parquet(inventory_path, index=False)
    merged_user_entries.to_parquet(user_entries_path, index=False)

    typer.echo(
        f"Refreshed contest indexes for {len(target_dates)} dates: "
        f"{len(inventory_all)} inventory rows, {len(user_entries_all)} user-entry rows"
    )


if __name__ == "__main__":
    app()
