#!/usr/bin/env python3
"""Backfill NBA Stats GameRotation raw JSON responses to bronze storage.

This script fetches GameRotation data from stats.nba.com for all games
in our historical dataset (2022-23 season onward) and stores the raw
JSON responses with fetch metadata for later processing.

Usage examples:
    # Smoke test with 3 games
    uv run python scripts/backfill_gamerotation_bronze.py --max-games 3

    # Full backfill (skips existing files by default)
    uv run python scripts/backfill_gamerotation_bronze.py

    # Backfill specific date range
    uv run python scripts/backfill_gamerotation_bronze.py --start-date 2024-01-01 --end-date 2024-01-31

    # Force re-fetch existing files
    uv run python scripts/backfill_gamerotation_bronze.py --overwrite --max-games 10

Output structure:
    ${PROJECTIONS_DATA_ROOT}/bronze/nba_stats/gamerotation_v1/
        season=YYYY/
            game_id=XXXXXXXXXX.json   # Raw response wrapper
        _failures.jsonl               # Failed fetches log
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from projections import paths

# NBA Stats API endpoint
GAMEROTATION_URL = "https://stats.nba.com/stats/gamerotation"

# Browser-like headers required by stats.nba.com
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://stats.nba.com/",
    "Origin": "https://stats.nba.com",
    "Connection": "keep-alive",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
}

# Retry configuration
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2.0
RETRY_STATUS_CODES = {429, 500, 502, 503, 504}


@dataclass
class GameRecord:
    """A game to fetch rotation data for."""

    game_id: int
    game_date: date
    season: int


@dataclass
class FetchResult:
    """Result of a single fetch attempt."""

    game_id: int
    season: int
    success: bool
    http_status: int | None = None
    error: str | None = None
    skipped: bool = False


@dataclass
class BackfillStats:
    """Aggregate statistics for the backfill run."""

    total: int = 0
    fetched: int = 0
    skipped: int = 0
    failed: int = 0
    failures: list[dict[str, Any]] = field(default_factory=list)


def _season_for_date(d: date) -> int:
    """Return the NBA season start year for a given date.

    NBA seasons span Oct-Jun, so dates before August belong to the
    prior calendar year's season.
    """
    return d.year if d.month >= 8 else d.year - 1


def _load_game_ids_from_features(data_root: Path) -> list[GameRecord]:
    """Load game IDs from the tmp_gold_features_minutes_v1_ranks directory.

    This is our primary source of historical game IDs for seasons 2023+.
    """
    features_root = data_root / "tmp_gold_features_minutes_v1_ranks"
    if not features_root.exists():
        return []

    games: dict[int, GameRecord] = {}

    for season_dir in sorted(features_root.glob("season=*")):
        season_str = season_dir.name.split("=")[1]
        try:
            season = int(season_str)
        except ValueError:
            continue

        for month_dir in season_dir.glob("month=*"):
            parquet_file = month_dir / "features.parquet"
            if not parquet_file.exists():
                continue

            try:
                df = pd.read_parquet(parquet_file, columns=["game_id", "game_date"])
                unique = df.drop_duplicates("game_id")
                for _, row in unique.iterrows():
                    gid = int(row["game_id"])
                    gdate = pd.Timestamp(row["game_date"]).date()
                    if gid not in games:
                        games[gid] = GameRecord(
                            game_id=gid,
                            game_date=gdate,
                            season=season,
                        )
            except Exception:
                continue

    return sorted(games.values(), key=lambda g: (g.game_date, g.game_id))


def _load_game_ids_from_labels(data_root: Path) -> list[GameRecord]:
    """Load game IDs from the labels directory as a fallback."""
    labels_root = data_root / "labels"
    if not labels_root.exists():
        return []

    games: dict[int, GameRecord] = {}

    for season_dir in sorted(labels_root.glob("season=*")):
        season_str = season_dir.name.split("=")[1]
        try:
            season = int(season_str)
        except ValueError:
            continue

        for parquet_file in season_dir.glob("*.parquet"):
            try:
                df = pd.read_parquet(parquet_file, columns=["game_id", "game_date"])
                unique = df.drop_duplicates("game_id")
                for _, row in unique.iterrows():
                    gid = int(row["game_id"])
                    gdate = pd.Timestamp(row["game_date"]).date()
                    if gid not in games:
                        games[gid] = GameRecord(
                            game_id=gid,
                            game_date=gdate,
                            season=season,
                        )
            except Exception:
                continue

    return sorted(games.values(), key=lambda g: (g.game_date, g.game_id))


def _output_path(bronze_root: Path, game: GameRecord) -> Path:
    """Return the output path for a game's rotation JSON."""
    return bronze_root / f"season={game.season}" / f"game_id={game.game_id}.json"


def _failures_path(bronze_root: Path) -> Path:
    """Return the path to the failures log."""
    return bronze_root / "_failures.jsonl"


def _fetch_gamerotation(
    game_id: int,
    timeout: float = 30.0,
    verbose: bool = False,
) -> tuple[int, dict[str, Any] | None, str | None]:
    """Fetch GameRotation data for a single game with retries.

    Returns:
        Tuple of (http_status, response_json, error_message)
    """
    params = {"GameID": str(game_id).zfill(10)}
    url = GAMEROTATION_URL

    for attempt in range(MAX_RETRIES):
        try:
            if verbose:
                print(f"  [request] attempt {attempt + 1}/{MAX_RETRIES}...", flush=True)
            with httpx.Client(timeout=timeout, follow_redirects=True) as client:
                response = client.get(url, params=params, headers=DEFAULT_HEADERS)
                status = response.status_code

                if status == 200:
                    return status, response.json(), None

                if status in RETRY_STATUS_CODES and attempt < MAX_RETRIES - 1:
                    wait = RETRY_BACKOFF_BASE ** attempt + random.uniform(0.1, 0.5)
                    if verbose:
                        print(f"  [request] Got {status}, retrying in {wait:.1f}s...", flush=True)
                    time.sleep(wait)
                    continue

                return status, None, f"HTTP {status}"

        except httpx.TimeoutException:
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_BACKOFF_BASE ** attempt + random.uniform(0.1, 0.5)
                if verbose:
                    print(f"  [request] Timeout, retrying in {wait:.1f}s...", flush=True)
                time.sleep(wait)
                continue
            return 0, None, "Timeout"

        except httpx.RequestError as exc:
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_BACKOFF_BASE ** attempt + random.uniform(0.1, 0.5)
                if verbose:
                    print(f"  [request] Error: {exc}, retrying in {wait:.1f}s...", flush=True)
                time.sleep(wait)
                continue
            return 0, None, f"Request error: {exc}"

    return 0, None, "Max retries exceeded"


def _write_response(
    output_path: Path,
    game: GameRecord,
    http_status: int,
    response_json: dict[str, Any] | None,
    error: str | None,
) -> None:
    """Write the response wrapper JSON to disk."""
    wrapper = {
        "game_id": game.game_id,
        "season": game.season,
        "game_date": game.game_date.isoformat(),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "url": GAMEROTATION_URL,
        "params": {"GameID": str(game.game_id).zfill(10)},
        "http_status": http_status,
        "response_json": response_json,
        "error": error,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(wrapper, indent=2), encoding="utf-8")


def _append_failure(failures_path: Path, failure: dict[str, Any]) -> None:
    """Append a failure record to the failures log."""
    failures_path.parent.mkdir(parents=True, exist_ok=True)
    with failures_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(failure) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill NBA Stats GameRotation data to bronze storage.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD). Defaults to earliest available.",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD). Defaults to latest available.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-fetch games that already have output files.",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="Maximum number of games to process (for testing).",
    )
    parser.add_argument(
        "--sleep-ms",
        type=int,
        default=500,
        help="Base sleep time in milliseconds between requests (default: 500).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="Request timeout in seconds (default: 15).",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Data root directory (defaults to PROJECTIONS_DATA_ROOT).",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=50,
        help="Log progress every N games (default: 50).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed request logging.",
    )

    args = parser.parse_args()

    # Resolve data root
    data_root = Path(args.data_root) if args.data_root else paths.get_data_root()
    bronze_root = data_root / "bronze" / "nba_stats" / "gamerotation_v1"

    print(f"[gamerotation] Data root: {data_root}")
    print(f"[gamerotation] Bronze output: {bronze_root}")

    # Load game IDs from features (primary) and labels (fallback)
    games = _load_game_ids_from_features(data_root)
    if not games:
        games = _load_game_ids_from_labels(data_root)

    if not games:
        print("[gamerotation] ERROR: No game IDs found in features or labels directories.")
        print("[gamerotation] Ensure tmp_gold_features_minutes_v1_ranks/ or labels/ exist with data.")
        return

    print(f"[gamerotation] Found {len(games)} total games in source data.")

    # Apply date filters
    start_date = date.fromisoformat(args.start_date) if args.start_date else None
    end_date = date.fromisoformat(args.end_date) if args.end_date else None

    if start_date:
        games = [g for g in games if g.game_date >= start_date]
    if end_date:
        games = [g for g in games if g.game_date <= end_date]

    if start_date or end_date:
        print(f"[gamerotation] After date filter: {len(games)} games")

    if args.max_games:
        games = games[: args.max_games]
        print(f"[gamerotation] Limited to {len(games)} games (--max-games)")

    if not games:
        print("[gamerotation] No games to process after filters.")
        return

    # Initialize stats
    stats = BackfillStats(total=len(games))
    failures_path = _failures_path(bronze_root)

    # Process games
    sleep_base = args.sleep_ms / 1000.0

    print(f"[gamerotation] Starting backfill of {stats.total} games...")

    for i, game in enumerate(games):
        output_path = _output_path(bronze_root, game)

        # Skip if exists and not overwriting
        if output_path.exists() and not args.overwrite:
            stats.skipped += 1
            # Progress logging for skips too
            total_processed = stats.fetched + stats.failed + stats.skipped
            if total_processed % args.progress_interval == 0:
                print(
                    f"[gamerotation] Progress: {total_processed}/{stats.total} "
                    f"(fetched={stats.fetched}, skipped={stats.skipped}, failed={stats.failed})"
                )
            continue

        # Log the fetch attempt (every game now, not just first)
        print(f"[gamerotation] Fetching {game.game_id}...", end=" ", flush=True)

        # Fetch the data
        http_status, response_json, error = _fetch_gamerotation(
            game.game_id, timeout=args.timeout, verbose=args.verbose
        )

        if error is None and response_json is not None:
            _write_response(output_path, game, http_status, response_json, None)
            stats.fetched += 1
            print(f"✓", flush=True)
        else:
            # Write error response too (for debugging)
            _write_response(output_path, game, http_status or 0, None, error)
            stats.failed += 1
            print(f"[gamerotation] FAILED game {game.game_id}: {error}")

            failure_record = {
                "game_id": game.game_id,
                "season": game.season,
                "game_date": game.game_date.isoformat(),
                "http_status": http_status,
                "error": error,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            stats.failures.append(failure_record)
            _append_failure(failures_path, failure_record)

        # Progress logging
        total_processed = stats.fetched + stats.failed + stats.skipped
        if total_processed % args.progress_interval == 0:
            print(
                f"[gamerotation] Progress: {total_processed}/{stats.total} "
                f"(fetched={stats.fetched}, skipped={stats.skipped}, failed={stats.failed})"
            )

        # Sleep between requests (with jitter)
        if i < len(games) - 1:
            jitter = random.uniform(0.0, sleep_base * 0.5)
            time.sleep(sleep_base + jitter)

    # Final summary
    print()
    print("=" * 60)
    print("[gamerotation] BACKFILL COMPLETE")
    print(f"  Total games:   {stats.total}")
    print(f"  Fetched:       {stats.fetched}")
    print(f"  Skipped:       {stats.skipped}")
    print(f"  Failed:        {stats.failed}")
    print(f"  Output dir:    {bronze_root}")
    if stats.failed > 0:
        print(f"  Failures log:  {failures_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
