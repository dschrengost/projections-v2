#!/usr/bin/env python3
"""Download DraftKings NBA GPP contest results for a given day."""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

from auth import authenticate_with_browser


load_dotenv()

DEFAULT_DATA_ROOT = Path("nba_gpp_data")
RESULTS_ENDPOINT = "https://www.draftkings.com/contest/exportfullstandingscsv/{contest_id}"
REQUEST_TIMEOUT = 30
COOKIE_ENV_VAR = "DK_RESULTS_COOKIE"


def _format_cookie(raw_cookie: Optional[str]) -> Optional[str]:
    """Normalize a raw cookie string for use in headers."""
    if not raw_cookie:
        return None

    segments = [segment.strip() for segment in raw_cookie.split(";") if segment.strip()]
    if not segments:
        return None

    return "; ".join(segments)


def _default_session(cookie: Optional[str] = None) -> requests.Session:
    """Create a session with headers that mimic a standard browser."""
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/csv,application/csv,text/plain,*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.draftkings.com/",
        }
    )
    formatted_cookie = _format_cookie(cookie)
    if formatted_cookie:
        session.headers["Cookie"] = formatted_cookie
    return session


@dataclass
class Contest:
    """Data captured for each contest row in the daily CSV."""

    contest_id: str
    contest_name: str
    prize_pool: str
    entry_fee: str


class ContestResultsDownloader:
    """Fetch and persist contest result CSVs for a given day."""

    def __init__(
        self,
        csv_path: Path,
        session: Optional[requests.Session] = None,
        cookie: Optional[str] = None,
    ) -> None:
        self.csv_path = csv_path
        self.session = session or _default_session(cookie=cookie)

    def read_contests(self) -> List[Contest]:
        """Read contest metadata from the scraped GPP CSV file."""
        contests: List[Contest] = []

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        with self.csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                contest_id = (row.get("contest_id") or "").strip()
                if not contest_id:
                    continue
                contests.append(
                    Contest(
                        contest_id=contest_id,
                        contest_name=(row.get("contest_name") or "").strip(),
                        prize_pool=(row.get("prize_pool") or "").strip(),
                        entry_fee=(row.get("entry_fee") or "").strip(),
                    )
                )

        return contests

    def ensure_results_dir(self) -> Path:
        """Ensure the `results/` directory exists next to the source CSV."""
        results_dir = self.csv_path.parent / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        return results_dir

    def download_contest(
        self,
        contest: Contest,
        results_dir: Path,
        overwrite: bool = False,
    ) -> Tuple[bool, Optional[str]]:
        """Download a single contest results CSV.

        Returns a tuple indicating success and an optional error message.
        """

        legacy_path = results_dir / f"contest_{contest.contest_id}_results.csv"
        standings_path = results_dir / f"contest_{contest.contest_id}_standings.csv"
        players_path = results_dir / f"contest_{contest.contest_id}_players.csv"

        if any(p.exists() for p in (standings_path, legacy_path)) and not overwrite:
            # If an existing non-HTML file is present, skip
            for p in (standings_path, legacy_path):
                if p.exists() and not self._looks_like_html(p):
                    return True, None

        url = RESULTS_ENDPOINT.format(contest_id=contest.contest_id)

        try:
            response = self.session.get(url, timeout=REQUEST_TIMEOUT)
        except requests.RequestException as exc:
            return False, f"request error: {exc}"

        if response.status_code != 200:
            return False, f"HTTP {response.status_code}"

        content_type = (response.headers.get("content-type") or "").lower()
        content_encoding = (response.headers.get("content-encoding") or "").lower()

        # Handle gzip compressed responses
        if content_encoding == "gzip":
            try:
                import gzip
                response_text = gzip.decompress(response.content).decode('utf-8')
            except Exception as e:
                return False, f"failed to decompress gzip response: {e}"
        # Handle zip archive responses (may contain multiple CSVs)
        elif response.content.startswith(b'PK'):
            try:
                import zipfile
                import io
                with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
                    standings_text = None
                    players_text = None
                    for file_info in zip_file.infolist():
                        if not file_info.filename.lower().endswith('.csv'):
                            continue
                        data = zip_file.read(file_info.filename).decode('utf-8')
                        fname = file_info.filename.lower()
                        if 'standings' in fname:
                            standings_text = data
                        elif 'players' in fname:
                            players_text = data
                        else:
                            # Fallback: if unknown, treat as standings if not set
                            if standings_text is None:
                                standings_text = data

                if standings_text is None and players_text is None:
                    return False, "no CSV file found in zip archive"

                # Write outputs: legacy results (standings) + explicit files
                if standings_text:
                    legacy_path.write_text(standings_text, encoding='utf-8')
                    standings_path.write_text(standings_text, encoding='utf-8')
                if players_text:
                    players_path.write_text(players_text, encoding='utf-8')
                return True, None
            except Exception as e:
                return False, f"failed to extract from zip archive: {e}"
        else:
            response_text = response.text

        if "text/html" in content_type or response.url.startswith("https://myaccount.draftkings.com"):
            return False, "authentication required (received HTML login page)"

        if not response_text.strip():
            return False, "empty response"

        if "," not in response_text:
            return False, "invalid csv payload"

        # Non-zip payload: write both legacy and explicit standings files
        legacy_path.write_text(response_text, encoding="utf-8")
        standings_path.write_text(response_text, encoding="utf-8")
        return True, None

    @staticmethod
    def _looks_like_html(path: Path) -> bool:
        try:
            snippet = path.read_text(encoding="utf-8", errors="ignore")[:200].lower()
        except OSError:
            return False
        return snippet.startswith("<!doctype html") or "<html" in snippet

    def download_all(self, overwrite: bool = False) -> Dict[str, int]:
        """Download results for every contest listed in the CSV."""
        contests = self.read_contests()
        results_dir = self.ensure_results_dir()

        stats = {"total": len(contests), "success": 0, "failed": 0, "skipped": 0}

        for contest in contests:
            target_path = results_dir / f"contest_{contest.contest_id}_results.csv"
            if target_path.exists() and not overwrite:
                if not self._looks_like_html(target_path):
                    stats["skipped"] += 1
                    continue

            success, error = self.download_contest(contest, results_dir, overwrite=overwrite)

            if success:
                stats["success"] += 1
            else:
                stats["failed"] += 1
                print(f"✗ {contest.contest_id} ({contest.contest_name}): {error}")

        return stats

    def write_summary(self, stats: Dict[str, int]) -> Path:
        """Persist a simple text summary alongside the downloaded CSVs."""
        results_dir = self.ensure_results_dir()
        summary_path = results_dir / "download_summary.txt"

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        total = stats.get("total", 0)
        if total:
            success_rate = (stats.get("success", 0) / total) * 100
        else:
            success_rate = 0.0

        lines = [
            "Contest Results Download Summary",
            "=" * 40,
            "",
            f"Download Time: {timestamp}",
            f"Source CSV: {self.csv_path}",
            f"Total Contests: {stats.get('total', 0)}",
            f"Successfully Downloaded: {stats.get('success', 0)}",
            f"Skipped (already present): {stats.get('skipped', 0)}",
            f"Failed: {stats.get('failed', 0)}",
            f"Success Rate: {success_rate:.1f}%",
            "",
            f"Results Directory: {results_dir}",
        ]

        summary_path.write_text("\n".join(lines), encoding="utf-8")
        return summary_path


def _resolve_csv_path(data_root: Path, date_str: Optional[str]) -> Path:
    """Infer the location of the daily contest CSV."""
    if date_str is None:
        date_str = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    daily_dir = data_root / date_str
    return daily_dir / f"nba_gpp_{date_str}.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download DraftKings contest results CSVs")
    parser.add_argument(
        "--date",
        help="Target contest date in YYYY-MM-DD format (defaults to yesterday)",
    )
    parser.add_argument(
        "--data-root",
        default=str(DEFAULT_DATA_ROOT),
        help="Root directory where daily NBA GPP data is stored",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download contests even if a results CSV already exists",
    )
    parser.add_argument(
        "--csv-path",
        help="Direct path to a scraped NBA GPP CSV (overrides --date)",
    )
    parser.add_argument(
        "--cookie",
        help=(
            "DraftKings cookie string to include in requests. If omitted, "
            f"the value of ${COOKIE_ENV_VAR} will be used when available."
        ),
    )
    parser.add_argument(
        "--cookie-file",
        type=Path,
        help="Optional path to a file containing the Cookie header value",
    )
    parser.add_argument(
        "--auth-browser",
        action="store_true",
        help="Use browser automation to authenticate and get fresh cookies",
    )
    parser.add_argument(
        "--auth-interactive",
        action="store_true",
        help="Open browser for interactive login (overrides --auth-browser)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=True,
        help="Run browser in headless mode (default: True)",
    )
    parser.add_argument(
        "--no-headless",
        action="store_true",
        help="Run browser with GUI (useful for debugging)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_root = Path(args.data_root)

    if args.csv_path:
        csv_path = Path(args.csv_path)
    else:
        csv_path = _resolve_csv_path(data_root, args.date)

    cookie_value: Optional[str] = None

    # Handle browser authentication
    if args.auth_browser or args.auth_interactive:
        print("Initiating browser authentication...")
        try:
            headless_mode = args.headless and not args.no_headless and not args.auth_interactive
            cookie_value = authenticate_with_browser(
                headless=headless_mode,
                interactive=args.auth_interactive
            )
            print("Browser authentication successful!")
        except Exception as e:
            print(f"Browser authentication failed: {e}")
            return
    else:
        # Use existing cookie methods
        if args.cookie:
            cookie_value = args.cookie
        elif args.cookie_file:
            cookie_value = args.cookie_file.read_text(encoding="utf-8").strip()
        else:
            cookie_value = os.getenv(COOKIE_ENV_VAR)

    cookie_value = _format_cookie(cookie_value)

    if cookie_value:
        print("Using DraftKings cookie from provided configuration")
    else:
        print(
            "No DraftKings cookie provided. Unauthenticated downloads may return HTML login pages."
        )

    downloader = ContestResultsDownloader(csv_path, cookie=cookie_value)

    print(f"Loading contests from {csv_path}")
    stats = downloader.download_all(overwrite=args.overwrite)

    print(
        "Downloaded results for "
        f"{stats['success']} contests (skipped: {stats['skipped']}, failed: {stats['failed']})"
    )

    summary_path = downloader.write_summary(stats)
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
