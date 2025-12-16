#!/usr/bin/env python3
"""
NBA GPP Tournament Scraper for DraftKings.
Fetches Classic format NBA GPP tournaments daily.
"""

import csv
import datetime
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv


load_dotenv()

COOKIE_ENV_VAR = "DK_RESULTS_COOKIE"
REQUEST_TIMEOUT = 30
DEFAULT_DATA_ROOT = Path("nba_gpp_data")


def _format_cookie(raw_cookie: Optional[str]) -> Optional[str]:
    """Normalize a raw cookie string for use in headers."""
    if not raw_cookie:
        return None

    segments = [segment.strip() for segment in raw_cookie.split(";") if segment.strip()]
    if not segments:
        return None

    # Preserve quoted values while ensuring consistent delimiter spacing.
    return "; ".join(segments)


class NBAGPPScraper:
    def __init__(
        self,
        output_dir: str | Path = DEFAULT_DATA_ROOT,
        append_mode: bool = True,
        session: Optional[requests.Session] = None,
        cookie: Optional[str] = None,
    ):
        self.base_url = "https://www.draftkings.com/lobby/getcontests?sport=NBA"
        self.output_dir = Path(output_dir).expanduser()
        self.append_mode = append_mode  # If True, amend existing files; if False, overwrite
        raw_cookie = cookie if cookie is not None else os.getenv(COOKIE_ENV_VAR)
        self.cookie = _format_cookie(raw_cookie)
        self.request_timeout = REQUEST_TIMEOUT
        self.session = session or self._create_session()
        self.ensure_output_dir()

    def _create_session(self) -> requests.Session:
        """Create a requests session with headers that mimic a browser."""
        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept": "application/json, text/plain, */*",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://www.draftkings.com/",
            }
        )

        if self.cookie:
            session.headers["Cookie"] = self.cookie

        return session

    def ensure_output_dir(self) -> None:
        """Create main output directory if it doesn't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_date_directory(self, target_date: Optional[str] = None) -> Path:
        """Get the date-specific subdirectory path."""
        if target_date is None:
            target_date = datetime.datetime.now().strftime("%Y-%m-%d")

        date_dir = self.output_dir / target_date
        date_dir.mkdir(parents=True, exist_ok=True)
        return date_dir

    def fetch_contests(self) -> Dict[str, Any]:
        """Fetch NBA contests from DraftKings API"""
        try:
            response = self.session.get(self.base_url, timeout=self.request_timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error fetching contests: {e}")
            sys.exit(1)

    def convert_timestamp(self, ts: str) -> str:
        """Convert DraftKings timestamp to readable format"""
        timestamp = int(ts.split('(')[1].split(')')[0])
        dt = datetime.datetime.fromtimestamp(timestamp/1000)
        return dt.strftime('%Y-%m-%d %H:%M:%S')

    def is_gpp_tournament(self, contest: Dict[str, Any]) -> bool:
        """
        Determine if a contest is a GPP (Guaranteed Prize Pool) tournament
        using multi-layered filtering approach
        """
        # LAYER 1: Prize pool threshold - eliminate micro-stakes immediately
        if contest['po'] < 1000:  # $1,000 minimum prize pool
            return False

        # LAYER 2: Must be Classic game type
        if contest.get('gameType') != 'Classic':
            return False

        # LAYER 3: Check contest name for indicators of non-GPP formats
        name_lower = contest['n'].lower()
        exclude_terms = [
            # Cash games
            'double up', '50/50', 'fifty fifty', 'head to head', 'h2h',
            # Satellites and qualifiers
            'satellite', 'supersat', 'supersatellite', 'qualifier', 'ticket',
            # Winner take all formats
            'winner take all', 'wta', 'winner takes all',
            # Small player pool formats
            'heavy hitter', '3-player', '4-player', '2-player', 'must fill',
            # Multiplier contests (NEW)
            'triple up', 'triple-up', 'quintuple up', 'quintuple-up',
            'booster', '10x', '5x', '3x',
            # Free and micro contests (NEW)
            'free', 'micro', 'nano'
        ]

        for term in exclude_terms:
            if term in name_lower:
                return False

        # LAYER 4: Additional attribute filters
        if 'attr' in contest:
            # Exclude Double Ups, 50/50s, and other cash formats
            if contest['attr'].get('IsDoubleUp') == 'true':
                return False
            if contest['attr'].get('IsFiftyfifty') == 'true':
                return False

        # LAYER 5: Must be a guaranteed tournament to be considered a GPP
        if 'attr' in contest and 'IsGuaranteed' in contest['attr']:
            if contest['attr']['IsGuaranteed'] == 'true':
                return True

        return False

    def filter_gpp_tournaments(self, contests: List[Dict[str, Any]], target_date: str = None) -> List[Dict[str, Any]]:
        """Filter contests to only include Classic GPP tournaments"""
        if target_date is None:
            target_date = datetime.datetime.now().strftime('%Y-%m-%d')

        gpp_tournaments = []
        for contest in contests:
            # Check if contest is on target date
            contest_date = self.convert_timestamp(contest['sd'])
            if not contest_date.startswith(target_date):
                continue

            # Check if it's a GPP tournament
            if self.is_gpp_tournament(contest):
                # Add human-readable date to contest data
                contest['start_time_readable'] = contest_date
                gpp_tournaments.append(contest)

        # Sort by prize pool (descending)
        gpp_tournaments.sort(key=lambda x: x['po'], reverse=True)
        return gpp_tournaments

    def save_data(
        self,
        gpp_tournaments: List[Dict[str, Any]],
        target_date: Optional[str] = None,
        scrape_time: Optional[str] = None,
    ) -> tuple[Path, Path]:
        """Save GPP tournament data to JSON and CSV files"""
        if target_date is None:
            target_date = datetime.datetime.now().strftime('%Y-%m-%d')

        if scrape_time is None:
            scrape_time = datetime.datetime.now().strftime('%H:%M:%S')

        # Get date-specific directory
        date_dir = self.get_date_directory(target_date)

        # Save JSON file
        json_filename = date_dir / f"nba_gpp_{target_date}.json"
        if self.append_mode and json_filename.exists():
            self.append_json_data(json_filename, gpp_tournaments, scrape_time)
        else:
            self.save_new_json_data(json_filename, gpp_tournaments, target_date, scrape_time)

        # Save CSV file
        csv_filename = date_dir / f"nba_gpp_{target_date}.csv"
        if self.append_mode and csv_filename.exists():
            self.append_csv_data(csv_filename, gpp_tournaments, scrape_time)
        else:
            self.save_to_csv(gpp_tournaments, csv_filename, scrape_time)

        return json_filename, csv_filename

    def save_new_json_data(
        self,
        filename: Path,
        gpp_tournaments: List[Dict[str, Any]],
        target_date: str,
        scrape_time: str,
    ) -> None:
        """Save new JSON data with first scrape"""
        data = {
            'date': target_date,
            'scrape_times': [scrape_time],
            'total_tournaments': len(gpp_tournaments),
            'total_prize_pool': sum(contest['po'] for contest in gpp_tournaments),
            'tournaments': gpp_tournaments
        }
        with filename.open('w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def append_json_data(
        self,
        filename: Path,
        gpp_tournaments: List[Dict[str, Any]],
        scrape_time: str,
    ) -> None:
        """Append new data to existing JSON file"""
        with filename.open('r', encoding='utf-8') as f:
            existing_data = json.load(f)

        # Handle old format files that don't have scrape_times
        if 'scrape_times' not in existing_data:
            existing_data['scrape_times'] = [scrape_time]
        else:
            existing_data['scrape_times'].append(scrape_time)

        # Update tournament data (replace with latest)
        existing_data['tournaments'] = gpp_tournaments
        existing_data['total_tournaments'] = len(gpp_tournaments)
        existing_data['total_prize_pool'] = sum(contest['po'] for contest in gpp_tournaments)
        existing_data['last_updated'] = scrape_time

        with filename.open('w', encoding='utf-8') as f:
            json.dump(existing_data, f, indent=2)

    def append_csv_data(self, filename: Path, gpp_tournaments: List[Dict[str, Any]], scrape_time: str) -> None:
        """Append new data to existing CSV file with scrape_time tracking"""
        # Read existing contests to avoid duplicates
        existing_contests = set()
        if filename.exists():
            with filename.open('r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    existing_contests.add(row['contest_id'])

        # Append only new contests
        new_contests = [contest for contest in gpp_tournaments if str(contest['id']) not in existing_contests]

        if new_contests:
            with filename.open('a', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=[
                    'contest_id', 'contest_name', 'start_time', 'start_time_readable',
                    'prize_pool', 'first_place_prize', 'entry_fee', 'max_entries',
                    'current_entries', 'game_type', 'is_guaranteed', 'is_starred',
                    'template_id', 'draft_group_id', 'scrape_time'
                ])

                # Add scrape_time column if this is first append
                if existing_contests and 'scrape_time' not in csvfile.name:
                    # Need to add header for scrape_time column
                    pass  # We'll handle this in a more robust way

                for contest in new_contests:
                    row = {
                        'contest_id': contest.get('id', ''),
                        'contest_name': contest['n'],
                        'start_time': contest.get('sd', ''),
                        'start_time_readable': contest.get('start_time_readable', ''),
                        'prize_pool': contest['po'],
                        'first_place_prize': self.extract_first_place_prize(contest['n']),
                        'entry_fee': contest['a'],
                        'max_entries': contest.get('mec', 'Unlimited'),
                        'current_entries': contest.get('m', 0),
                        'game_type': contest.get('gameType', ''),
                        'is_guaranteed': contest['attr'].get('IsGuaranteed', 'false') if 'attr' in contest else 'false',
                        'is_starred': contest['attr'].get('IsStarred', 'false') if 'attr' in contest else 'false',
                        'template_id': contest.get('tmpl', ''),
                        'draft_group_id': contest.get('dg', ''),
                        'scrape_time': scrape_time
                    }
                    writer.writerow(row)

        print(f"Added {len(new_contests)} new contests (out of {len(gpp_tournaments)} total)")

    def save_to_csv(self, gpp_tournaments: List[Dict[str, Any]], filename: Path, scrape_time: str) -> None:
        """Save GPP tournament data to CSV file"""
        fieldnames = [
            'contest_id',
            'contest_name',
            'start_time',
            'start_time_readable',
            'prize_pool',
            'first_place_prize',
            'entry_fee',
            'max_entries',
            'current_entries',
            'game_type',
            'is_guaranteed',
            'is_starred',
            'template_id',
            'draft_group_id',
            'scrape_time'
        ]

        with filename.open('w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for contest in gpp_tournaments:
                # Extract first place prize from name if available
                first_place = self.extract_first_place_prize(contest['n'])

                row = {
                    'contest_id': contest.get('id', ''),
                    'contest_name': contest['n'],
                    'start_time': contest.get('sd', ''),
                    'start_time_readable': contest.get('start_time_readable', ''),
                    'prize_pool': contest['po'],
                    'first_place_prize': first_place,
                    'entry_fee': contest['a'],
                    'max_entries': contest.get('mec', 'Unlimited'),
                    'current_entries': contest.get('m', 0),
                    'game_type': contest.get('gameType', ''),
                    'is_guaranteed': contest['attr'].get('IsGuaranteed', 'false') if 'attr' in contest else 'false',
                    'is_starred': contest['attr'].get('IsStarred', 'false') if 'attr' in contest else 'false',
                    'template_id': contest.get('tmpl', ''),
                    'draft_group_id': contest.get('dg', ''),
                    'scrape_time': scrape_time
                }
                writer.writerow(row)

    def extract_first_place_prize(self, contest_name: str) -> str:
        """Extract first place prize from contest name"""
        import re
        # Look for patterns like "$200K to 1st" or "[$100K to 1st]"
        match = re.search(r'\$?([\d\.]+[KMB]?)\s*to\s*1st', contest_name, re.IGNORECASE)
        if match:
            return f"${match.group(1)}"
        return "Not specified"

    def print_summary(self, gpp_tournaments: List[Dict[str, Any]], target_date: str = None):
        """Print a summary of the GPP tournaments"""
        if target_date is None:
            target_date = datetime.datetime.now().strftime('%Y-%m-%d')

        print(f"\n=== NBA GPP Tournaments for {target_date} ===")
        print(f"Total GPP tournaments: {len(gpp_tournaments)}")

        if gpp_tournaments:
            total_prize_pool = sum(contest['po'] for contest in gpp_tournaments)
            avg_entry_fee = sum(contest['a'] for contest in gpp_tournaments) / len(gpp_tournaments)

            print(f"Total prize pool: ${total_prize_pool:,.0f}")
            print(f"Average entry fee: ${avg_entry_fee:.2f}")
            print()

            # Show top 10 tournaments
            print("Top 10 GPP Tournaments:")
            for i, contest in enumerate(gpp_tournaments[:10]):
                print(f"{i+1:2d}. {contest['n']}")
                print(f"     Start: {contest['start_time_readable']}")
                print(f"     Prize: ${contest['po']:,.0f} | Entry: ${contest['a']} | Max: {contest.get('mec', 'Unlimited')}")
                print(f"     Entries: {contest.get('m', 0)}")
                print()
        else:
            print("No GPP tournaments found for this date.")

    def run_scraper(self, target_date: str = None) -> tuple:
        """Run the complete scraping process"""
        if target_date is None:
            target_date = datetime.datetime.now().strftime('%Y-%m-%d')

        print(f"Fetching NBA GPP tournaments for {target_date}...")

        # Fetch contests
        data = self.fetch_contests()
        print(f"Fetched {len(data['Contests'])} total NBA contests")

        # Filter for GPP tournaments
        gpp_tournaments = self.filter_gpp_tournaments(data['Contests'], target_date)

        # Save data
        json_filename, csv_filename = self.save_data(gpp_tournaments, target_date)
        print(f"JSON data saved to: {json_filename}")
        print(f"CSV data saved to: {csv_filename}")

        # Print summary
        self.print_summary(gpp_tournaments, target_date)

        return json_filename, csv_filename


def main():
    """Main function to run the scraper"""
    import argparse

    parser = argparse.ArgumentParser(description='NBA GPP Tournament Scraper')
    parser.add_argument('--date', type=str, help='Target date (YYYY-MM-DD). Defaults to today.')
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(DEFAULT_DATA_ROOT),
        help='Output directory for saved data.',
    )

    args = parser.parse_args()

    # Validate date format if provided
    if args.date:
        try:
            datetime.datetime.strptime(args.date, '%Y-%m-%d')
        except ValueError:
            print("Error: Date must be in YYYY-MM-DD format")
            sys.exit(1)

    # Create and run scraper
    scraper = NBAGPPScraper(output_dir=args.output_dir)
    scraper.run_scraper(target_date=args.date)


if __name__ == "__main__":
    main()
