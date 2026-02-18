#!/usr/bin/env python3
"""
Action Network NBA Player Props Scraper.

Scrapes historical player prop odds from Action Network's public API.
No authentication required - just needs proper headers.

Usage:
    # Scan for NBA games in a game ID range
    uv run python tools/action_network_props_scraper.py scan --start 232000 --end 233000

    # Fetch props for specific game IDs
    uv run python tools/action_network_props_scraper.py fetch --game-ids 232573,233089

    # Fetch all NBA games for a date range (uses internal schedule)
    uv run python tools/action_network_props_scraper.py backfill --start-date 2024-10-22 --end-date 2024-10-30

API Endpoints discovered:
    - GET /web/v2/games/{game_id}/polling - Game info (teams, date, status)
    - GET /web/v2/games/{game_id}/props - Player props with odds
    - GET /web/v2/scoreboard/nba - Today's games only
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

# Headers to mimic browser requests
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Origin": "https://www.actionnetwork.com",
    "Referer": "https://www.actionnetwork.com/",
}

API_BASE = "https://api.actionnetwork.com/web/v2"


def get_game_info(game_id: int) -> dict[str, Any] | None:
    """Get basic game info (teams, date, league, status)."""
    url = f"{API_BASE}/games/{game_id}/polling"
    params = {"bookIds": "15", "periods": "event"}

    resp = requests.get(url, params=params, headers=HEADERS, timeout=10)
    if resp.status_code == 200:
        return resp.json()
    return None


def get_game_props(game_id: int, state_code: str = "OH") -> dict[str, Any] | None:
    """Get player props for a game."""
    url = f"{API_BASE}/games/{game_id}/props"
    params = {"stateCode": state_code}

    resp = requests.get(url, params=params, headers=HEADERS, timeout=30)
    if resp.status_code == 200:
        return resp.json()
    return None


def scan_for_nba_games(start_id: int, end_id: int, delay: float = 0.1) -> list[dict]:
    """Scan a range of game IDs to find NBA games."""
    nba_games = []

    print(f"Scanning game IDs {start_id} to {end_id}...")

    for game_id in range(start_id, end_id + 1):
        info = get_game_info(game_id)

        if info and info.get("league_name") == "nba":
            teams = [t.get("abbr", "?") for t in info.get("teams", [])]
            game_data = {
                "game_id": game_id,
                "start_time": info.get("start_time"),
                "status": info.get("status"),
                "teams": teams,
            }
            nba_games.append(game_data)
            print(f"  Found: {game_id} - {info.get('start_time', 'N/A')[:10]} {' @ '.join(teams)}")

        time.sleep(delay)

    return nba_games


def fetch_props_for_games(game_ids: list[int], output_dir: Path, delay: float = 0.2) -> dict[str, int]:
    """Fetch and save props for multiple games."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = {"success": 0, "no_props": 0, "failed": 0}

    for game_id in game_ids:
        print(f"Fetching props for game {game_id}...")

        # First get game info
        info = get_game_info(game_id)
        if not info:
            print(f"  Failed to get game info")
            stats["failed"] += 1
            continue

        # Then get props
        props = get_game_props(game_id)
        if not props or not props.get("player_props"):
            print(f"  No props available")
            stats["no_props"] += 1
            continue

        # Combine and save
        teams = [t.get("abbr", "?") for t in info.get("teams", [])]
        date_str = info.get("start_time", "")[:10]

        output = {
            "game_id": game_id,
            "date": date_str,
            "teams": teams,
            "status": info.get("status"),
            "fetched_at": datetime.utcnow().isoformat(),
            "game_info": info,
            "props": props,
        }

        filename = f"{date_str}_{game_id}_{'_'.join(teams)}.json"
        output_path = output_dir / filename

        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        prop_count = sum(len(v) for v in props.get("player_props", {}).values())
        print(f"  Saved {prop_count} props to {filename}")
        stats["success"] += 1

        time.sleep(delay)

    return stats


def summarize_props(props_data: dict) -> dict:
    """Summarize props data into a more compact format."""
    player_props = props_data.get("player_props", {})

    summary = {
        "prop_types": list(player_props.keys()),
        "total_props": sum(len(v) for v in player_props.values()),
        "players": set(),
        "markets": [],
    }

    # Extract unique players and market info
    for prop_type, props_list in player_props.items():
        for prop in props_list:
            player_name = prop.get("player_abbr", "Unknown")
            summary["players"].add(player_name)

            # Get the best available odds from all books
            lines = prop.get("lines", {})
            for book_id, book_lines in lines.items():
                for line in book_lines:
                    if line.get("period") == "event":  # Full game props
                        summary["markets"].append(
                            {
                                "player": player_name,
                                "type": prop.get("custom_pick_type_name", prop_type),
                                "line": line.get("value"),
                                "over_odds": line.get("odds") if line.get("side") == "over" else None,
                                "under_odds": line.get("odds") if line.get("side") == "under" else None,
                                "book_id": book_id,
                            }
                        )

    summary["players"] = list(summary["players"])
    return summary


def main():
    parser = argparse.ArgumentParser(description="Scrape NBA player props from Action Network")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Scan command
    scan_parser = subparsers.add_parser("scan", help="Scan for NBA games in a game ID range")
    scan_parser.add_argument("--start", type=int, required=True, help="Start game ID")
    scan_parser.add_argument("--end", type=int, required=True, help="End game ID")
    scan_parser.add_argument("--output", type=str, default="nba_games_scan.json", help="Output file")

    # Fetch command
    fetch_parser = subparsers.add_parser("fetch", help="Fetch props for specific game IDs")
    fetch_parser.add_argument("--game-ids", type=str, required=True, help="Comma-separated game IDs")
    fetch_parser.add_argument("--output-dir", type=str, default="props_data", help="Output directory")

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Demo: fetch props for a known game")

    args = parser.parse_args()

    if args.command == "scan":
        games = scan_for_nba_games(args.start, args.end)
        with open(args.output, "w") as f:
            json.dump(games, f, indent=2)
        print(f"\nFound {len(games)} NBA games, saved to {args.output}")

    elif args.command == "fetch":
        game_ids = [int(x.strip()) for x in args.game_ids.split(",")]
        output_dir = Path(args.output_dir)
        stats = fetch_props_for_games(game_ids, output_dir)
        print(f"\nComplete: {stats}")

    elif args.command == "demo":
        # Demo with a known game from each season
        demo_games = [
            (261653, "Oct 24, 2025 - Bucks @ Raptors (2025-26)"),
            (232573, "Oct 22, 2024 - Knicks @ Celtics (2024-25)"),
            (205185, "Oct 25, 2023 - Celtics @ Knicks (2023-24)"),
        ]

        print("Action Network Props API Demo")
        print("=" * 60)

        for game_id, desc in demo_games:
            print(f"\n{desc}")
            print("-" * 40)

            props = get_game_props(game_id)
            if props and props.get("player_props"):
                summary = summarize_props(props)
                print(f"  Prop types: {len(summary['prop_types'])}")
                print(f"  Total props: {summary['total_props']}")
                print(f"  Players: {len(summary['players'])}")
                print(f"  Sample types: {summary['prop_types'][:3]}")
            else:
                print("  No props available")


if __name__ == "__main__":
    main()
