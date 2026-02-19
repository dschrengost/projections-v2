#!/usr/bin/env python3
"""
Action Network NBA Player Props Backfill.

Strategy: Use our NBA schedule to know which games exist, then search
Action Network for matching games by team/date. This is much faster than
scanning all ~40k game IDs.

Usage:
    # Full backfill for 2024-25 and 2025-26 seasons
    PROJECTIONS_DATA_ROOT=/home/daniel/projections-data uv run python scrapers/action_network/props_backfill.py

    # Specific date range
    uv run python scrapers/action_network/props_backfill.py --start-date 2024-10-22 --end-date 2024-11-01
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import requests
from tqdm import tqdm

# Configuration
DATA_ROOT = Path(os.environ.get("PROJECTIONS_DATA_ROOT", "data"))
BRONZE_DIR = DATA_ROOT / "bronze" / "action_network" / "props"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Origin": "https://www.actionnetwork.com",
    "Referer": "https://www.actionnetwork.com/",
}

API_BASE = "https://api.actionnetwork.com/web/v2"

# Team abbreviation mapping (NBA API -> Action Network)
TEAM_ABBR_MAP = {
    "PHX": "PHO",  # Phoenix
    "GSW": "GS",   # Golden State
    "NOP": "NO",   # New Orleans
    "SAS": "SA",   # San Antonio
    "NYK": "NY",   # New York
}

# Known anchor points for ID range estimation
ANCHORS = [
    (232573, "2024-10-22"),  # 2024-25 opener
    (247000, "2025-04-06"),
    (261653, "2025-10-24"),  # 2025-26 early
    (273525, "2026-02-20"),  # Recent
]


def estimate_game_id_range(target_date: str) -> tuple[int, int]:
    """Estimate game ID range for a date based on known anchors."""
    from datetime import datetime

    target = datetime.strptime(target_date, "%Y-%m-%d")

    # Find surrounding anchors
    before = None
    after = None
    for gid, date_str in ANCHORS:
        anchor_date = datetime.strptime(date_str, "%Y-%m-%d")
        if anchor_date <= target:
            before = (gid, anchor_date)
        if anchor_date >= target and after is None:
            after = (gid, anchor_date)

    if before is None:
        before = ANCHORS[0]
        before = (before[0], datetime.strptime(before[1], "%Y-%m-%d"))
    if after is None:
        after = ANCHORS[-1]
        after = (after[0], datetime.strptime(after[1], "%Y-%m-%d"))

    # Interpolate between surrounding anchors when both exist.
    # Previous bug: only used `before` anchor with a fixed 75-IDs/day rate,
    # which drifted ~2,500 IDs off for dates far from the nearest anchor.
    days_from_before = (target - before[1]).days
    total_span_days = (after[1] - before[1]).days
    if total_span_days > 0 and before[1] != after[1]:
        # Linear interpolation between the two surrounding anchors
        id_span = after[0] - before[0]
        estimated_id = before[0] + int(id_span * days_from_before / total_span_days)
    else:
        # Exact anchor match or extrapolation beyond all anchors
        ids_per_day = 75
        estimated_id = before[0] + (days_from_before * ids_per_day)

    # Search window of +/- 500 IDs (about a week of all sports)
    return (max(230000, estimated_id - 500), estimated_id + 500)


def get_game_info(game_id: int) -> dict | None:
    """Get basic game info."""
    try:
        url = f"{API_BASE}/games/{game_id}/polling"
        resp = requests.get(
            url, params={"bookIds": "15", "periods": "event"}, headers=HEADERS, timeout=10
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def get_game_props(game_id: int) -> dict | None:
    """Get player props for a game."""
    try:
        url = f"{API_BASE}/games/{game_id}/props"
        resp = requests.get(url, headers=HEADERS, timeout=30)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def normalize_team(abbr: str) -> str:
    """Normalize team abbreviation for matching."""
    return TEAM_ABBR_MAP.get(abbr.upper(), abbr.upper())


def find_action_network_game(
    date: str, home_team: str, away_team: str, id_range: tuple[int, int]
) -> dict | None:
    """Find Action Network game ID for a specific matchup."""
    home = normalize_team(home_team)
    away = normalize_team(away_team)

    start_id, end_id = id_range

    for game_id in range(start_id, end_id + 1):
        info = get_game_info(game_id)
        if not info:
            continue

        if info.get("league_name") != "nba":
            continue

        game_date = info.get("start_time", "")[:10]
        if game_date != date:
            continue

        teams = [normalize_team(t.get("abbr", "")) for t in info.get("teams", [])]
        if home in teams and away in teams:
            return {
                "game_id": game_id,
                "start_time": info.get("start_time"),
                "status": info.get("status"),
                "type": info.get("type"),
                "home_team_id": info.get("home_team_id"),
                "away_team_id": info.get("away_team_id"),
                "teams": teams,
            }

    return None


def scan_date_range_parallel(start_id: int, end_id: int, workers: int = 100) -> dict[str, list]:
    """Parallel scan of ID range, grouped by date."""
    games_by_date = {}

    def check_game(game_id):
        info = get_game_info(game_id)
        if info and info.get("league_name") == "nba":
            return {
                "game_id": game_id,
                "date": info.get("start_time", "")[:10],
                "start_time": info.get("start_time"),
                "status": info.get("status"),
                "type": info.get("type"),
                "home_team_id": info.get("home_team_id"),
                "away_team_id": info.get("away_team_id"),
                "teams": [t.get("abbr") for t in info.get("teams", [])],
            }
        return None

    game_ids = list(range(start_id, end_id + 1))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(check_game, gid): gid for gid in game_ids}

        with tqdm(total=len(game_ids), desc="Scanning IDs", unit="ids") as pbar:
            for future in as_completed(futures):
                pbar.update(1)
                result = future.result()
                if result:
                    date = result["date"]
                    if date not in games_by_date:
                        games_by_date[date] = []
                    games_by_date[date].append(result)
                    pbar.set_postfix({"games": sum(len(v) for v in games_by_date.values())})

    return games_by_date


def fetch_props_for_games(games: list[dict], output_dir: Path, workers: int = 30) -> dict:
    """Fetch props for games and save to bronze."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = {"success": 0, "no_props": 0, "failed": 0, "skipped": 0}

    def process_game(game: dict) -> tuple[str, dict | None]:
        game_id = game["game_id"]
        date = game["date"]
        teams = game["teams"]

        # Check if already fetched
        filename = f"{date}_{game_id}_{'_'.join(teams)}.json"
        output_path = output_dir / filename
        if output_path.exists():
            return ("skipped", None)

        # Fetch props
        props = get_game_props(game_id)
        if not props:
            return ("failed", None)

        if not props.get("player_props"):
            return ("no_props", None)

        # Save
        output = {
            "game_id": game_id,
            "date": date,
            "teams": teams,
            "status": game["status"],
            "type": game["type"],
            "home_team_id": game.get("home_team_id"),
            "away_team_id": game.get("away_team_id"),
            "fetched_at": datetime.utcnow().isoformat(),
            "props": props,
        }

        with open(output_path, "w") as f:
            json.dump(output, f)

        prop_count = sum(len(v) for v in props.get("player_props", {}).values())
        return ("success", {"filename": filename, "prop_count": prop_count})

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_game, g): g for g in games}

        with tqdm(total=len(games), desc="Fetching props", unit="games") as pbar:
            for future in as_completed(futures):
                pbar.update(1)
                try:
                    status, result = future.result()
                    stats[status] += 1
                    pbar.set_postfix(stats)
                except Exception:
                    stats["failed"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(description="Backfill Action Network NBA props")
    parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--scan-only", action="store_true", help="Only scan for games")
    parser.add_argument("--workers", type=int, default=100, help="Parallel workers for scanning")
    args = parser.parse_args()

    BRONZE_DIR.mkdir(parents=True, exist_ok=True)

    # Default to 2024-25 and 2025-26 seasons
    if args.start_date and args.end_date:
        start_date = args.start_date
        end_date = args.end_date
    else:
        start_date = "2024-10-01"  # Include preseason
        end_date = datetime.now().strftime("%Y-%m-%d")

    print(f"Action Network NBA Props Backfill")
    print(f"{'='*60}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Output: {BRONZE_DIR}")

    # Estimate ID range
    start_range = estimate_game_id_range(start_date)
    end_range = estimate_game_id_range(end_date)

    full_start = start_range[0]
    full_end = end_range[1]

    print(f"Estimated ID range: {full_start:,} to {full_end:,} ({full_end - full_start:,} IDs)")

    # Check for cached game list
    cache_file = BRONZE_DIR / f"games_cache_{start_date}_{end_date}.json".replace("-", "")

    use_cache = False
    if cache_file.exists():
        print(f"\nLoading cached game list from {cache_file}")
        with open(cache_file) as f:
            games_by_date = json.load(f)
        if games_by_date:
            use_cache = True
        else:
            print(f"WARNING: cached game list is empty — deleting stale cache and rescanning")
            cache_file.unlink()

    if use_cache:
        all_games = []
        for date_games in games_by_date.values():
            all_games.extend(date_games)
    else:
        print(f"\nScanning for NBA games...")
        games_by_date = scan_date_range_parallel(full_start, full_end, workers=args.workers)

        # Filter to date range
        filtered = {}
        for date, games in games_by_date.items():
            if start_date <= date <= end_date:
                filtered[date] = games
        games_by_date = filtered

        # Only cache non-empty results to avoid poisoning future runs
        if games_by_date:
            with open(cache_file, "w") as f:
                json.dump(games_by_date, f, indent=2)
            print(f"Cached game list to {cache_file}")
        else:
            print(
                f"WARNING: scan found 0 NBA games for {start_date}..{end_date} "
                f"in ID range {full_start:,}–{full_end:,}; NOT caching empty result. "
                f"Anchor staleness or ID drift may need attention."
            )

        all_games = []
        for date_games in games_by_date.values():
            all_games.extend(date_games)

    all_games.sort(key=lambda x: (x["date"], x["game_id"]))

    print(f"\nFound {len(all_games)} NBA games")
    if all_games:
        dates = sorted(set(g["date"] for g in all_games))
        print(f"Date range: {dates[0]} to {dates[-1]}")
        print(f"Unique dates: {len(dates)}")

    if args.scan_only:
        print("\nScan complete (--scan-only mode)")
        return

    # Fetch props
    print(f"\nFetching props...")
    stats = fetch_props_for_games(all_games, BRONZE_DIR, workers=30)

    print(f"\n{'='*60}")
    print("Backfill Complete")
    print(f"{'='*60}")
    print(f"  Success: {stats['success']}")
    print(f"  No props: {stats['no_props']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Skipped (exists): {stats['skipped']}")
    print(f"\nData saved to: {BRONZE_DIR}")


if __name__ == "__main__":
    main()
