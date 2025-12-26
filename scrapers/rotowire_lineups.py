"""Scraper for Rotowire NBA daily starting lineups.

This scraper fetches the public Rotowire NBA lineups page and extracts
confirmed/expected starting lineups for each game. It serves as a faster
alternative to stats.nba.com for lineup confirmations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import httpx
import pandas as pd

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None  # type: ignore

ROTOWIRE_LINEUPS_URL = "https://www.rotowire.com/basketball/nba-lineups.php"

DEFAULT_HEADERS: Dict[str, str] = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}

# Map Rotowire status indicators to our standard lineup roles
LINEUP_ROLE_PROJECTED = "projected_starter"
LINEUP_ROLE_CONFIRMED = "confirmed_starter"
LINEUP_ROLE_BENCH = "bench"
LINEUP_ROLE_OUT = "out"


@dataclass(frozen=True)
class RotowireLineupRecord:
    """A single player's lineup entry from Rotowire."""

    team_abbreviation: str
    opponent_abbreviation: str
    player_name: str
    position: str
    lineup_role: str
    is_confirmed: bool
    injury_status: Optional[str]
    ingested_ts: pd.Timestamp


@dataclass
class RotowireGameLineup:
    """Container for a single game's lineup data."""

    home_team: str
    away_team: str
    game_time: Optional[str]
    is_confirmed: bool
    home_players: List[RotowireLineupRecord]
    away_players: List[RotowireLineupRecord]


class RotowireLineupsScraper:
    """Scrape NBA starting lineups from Rotowire."""

    def __init__(
        self,
        *,
        timeout: float = 15.0,
        user_agent: str | None = None,
        url: str = ROTOWIRE_LINEUPS_URL,
        client: httpx.Client | None = None,
    ) -> None:
        if BeautifulSoup is None:
            raise ImportError(
                "BeautifulSoup4 is required for Rotowire scraping. "
                "Install with: pip install beautifulsoup4 lxml"
            )
        self.timeout = timeout
        self.url = url
        self.user_agent = user_agent or (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
        self.client = client

    def fetch_html(self) -> str:
        """Fetch the raw HTML from Rotowire lineups page."""
        headers = dict(DEFAULT_HEADERS)
        headers["User-Agent"] = self.user_agent

        session = self.client
        close_session = False
        if session is None:
            session = httpx.Client(timeout=self.timeout, follow_redirects=True)
            close_session = True

        try:
            response = session.get(
                self.url,
                headers=headers,
                timeout=self.timeout,
                follow_redirects=True,
            )
            response.raise_for_status()
            return response.text
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"Rotowire returned {exc.response.status_code} for {self.url}"
            ) from exc
        except httpx.RequestError as exc:
            raise RuntimeError(f"Failed to fetch {self.url}") from exc
        finally:
            if close_session:
                session.close()

    def parse_lineups(self, html: str) -> List[RotowireGameLineup]:
        """Parse the HTML and extract lineup data for each game."""
        soup = BeautifulSoup(html, "lxml")
        games: List[RotowireGameLineup] = []
        ingested_ts = pd.Timestamp.now(tz="UTC")

        # Find all lineup cards/boxes
        # Rotowire uses divs with class "lineup" or similar
        lineup_cards = soup.find_all("div", class_=re.compile(r"lineup\s*$|lineup__main"))

        if not lineup_cards:
            # Try alternative selectors
            lineup_cards = soup.find_all("div", class_="lineup is-nba")

        for card in lineup_cards:
            game = self._parse_game_card(card, ingested_ts)
            if game:
                games.append(game)

        return games

    def _parse_game_card(
        self, card, ingested_ts: pd.Timestamp
    ) -> Optional[RotowireGameLineup]:
        """Parse a single game's lineup card."""
        try:
            # Find teams - look for lineup__abbr or team abbreviation elements
            team_elements = card.find_all(class_=re.compile(r"lineup__abbr|lineup__team"))
            if len(team_elements) < 2:
                # Try finding in the header
                header = card.find(class_=re.compile(r"lineup__header|lineup__teams"))
                if header:
                    team_elements = header.find_all("a") or header.find_all("span")

            if len(team_elements) < 2:
                return None

            # Extract team abbreviations
            away_team = self._extract_team_abbr(team_elements[0])
            home_team = self._extract_team_abbr(team_elements[1])

            if not away_team or not home_team:
                return None

            # Check if lineup is confirmed
            # Rotowire typically marks confirmed lineups with specific CSS classes
            is_confirmed = self._is_game_confirmed(card)

            # Extract game time if available
            game_time = None
            time_elem = card.find(class_=re.compile(r"lineup__time|lineup__status"))
            if time_elem:
                game_time = time_elem.get_text(strip=True)

            # Find player sections for each team
            player_sections = card.find_all(class_=re.compile(r"lineup__list|lineup__players"))

            away_players: List[RotowireLineupRecord] = []
            home_players: List[RotowireLineupRecord] = []

            if len(player_sections) >= 2:
                away_players = self._parse_player_list(
                    player_sections[0],
                    away_team,
                    home_team,
                    is_confirmed,
                    ingested_ts,
                )
                home_players = self._parse_player_list(
                    player_sections[1],
                    home_team,
                    away_team,
                    is_confirmed,
                    ingested_ts,
                )
            else:
                # Try parsing players from the entire card
                all_players = card.find_all(class_=re.compile(r"lineup__player"))
                mid = len(all_players) // 2
                for i, player_elem in enumerate(all_players[:mid]):
                    record = self._parse_player(
                        player_elem, away_team, home_team, is_confirmed, ingested_ts
                    )
                    if record:
                        away_players.append(record)
                for player_elem in all_players[mid:]:
                    record = self._parse_player(
                        player_elem, home_team, away_team, is_confirmed, ingested_ts
                    )
                    if record:
                        home_players.append(record)

            return RotowireGameLineup(
                home_team=home_team,
                away_team=away_team,
                game_time=game_time,
                is_confirmed=is_confirmed,
                home_players=home_players,
                away_players=away_players,
            )

        except Exception:
            # Log but don't fail on individual card parsing errors
            return None

    def _extract_team_abbr(self, element) -> Optional[str]:
        """Extract team abbreviation from an element."""
        # Try text content first
        text = element.get_text(strip=True)
        if text and len(text) <= 4:
            return text.upper()

        # Try href attribute for links
        href = element.get("href", "")
        if "/basketball/team/" in href:
            # Extract from URL like /basketball/team/PHI
            match = re.search(r"/team/([A-Z]{2,3})", href, re.IGNORECASE)
            if match:
                return match.group(1).upper()

        # Try data attributes
        for attr in ["data-team", "data-abbr"]:
            if element.get(attr):
                return element.get(attr).upper()

        return None

    def _is_game_confirmed(self, card) -> bool:
        """Check if the game's lineup is confirmed."""
        # Look for "Confirmed" text or status indicators
        status_elem = card.find(class_=re.compile(r"lineup__status"))
        if status_elem:
            status_text = status_elem.get_text(strip=True).lower()
            if "confirmed" in status_text:
                return True
            if "expected" in status_text:
                return False

        # Check for CSS class indicators
        card_classes = " ".join(card.get("class", []))
        if "is-confirmed" in card_classes:
            return True
        if "is-expected" in card_classes:
            return False

        # Check main container for confirmation indicator
        confirmed_indicator = card.find(class_=re.compile(r"confirmed|is-confirmed"))
        return confirmed_indicator is not None

    def _parse_player_list(
        self,
        section,
        team: str,
        opponent: str,
        is_confirmed: bool,
        ingested_ts: pd.Timestamp,
    ) -> List[RotowireLineupRecord]:
        """Parse all players from a team's section."""
        players: List[RotowireLineupRecord] = []
        player_elements = section.find_all(class_=re.compile(r"lineup__player"))
        
        if not player_elements:
            # Try finding list items or other player containers
            player_elements = section.find_all("li") or section.find_all("a")

        for elem in player_elements:
            record = self._parse_player(elem, team, opponent, is_confirmed, ingested_ts)
            if record:
                players.append(record)

        return players

    def _parse_player(
        self,
        element,
        team: str,
        opponent: str,
        is_confirmed: bool,
        ingested_ts: pd.Timestamp,
    ) -> Optional[RotowireLineupRecord]:
        """Parse a single player element."""
        try:
            # Get player name - try link first, then text
            name_elem = element.find("a") or element
            player_name = None

            if name_elem.get("title"):
                player_name = name_elem.get("title")
            else:
                player_name = name_elem.get_text(strip=True)

            if not player_name or len(player_name) < 2:
                return None

            # Clean up name - remove position prefixes like "PG "
            player_name = re.sub(r"^(PG|SG|SF|PF|C)\s+", "", player_name).strip()

            # Get position
            pos_elem = element.find(class_=re.compile(r"lineup__pos|pos"))
            position = ""
            if pos_elem:
                position = pos_elem.get_text(strip=True)
            else:
                # Try to extract from parent or preceding element
                pos_match = re.match(r"^(PG|SG|SF|PF|C)\b", name_elem.get_text(strip=True))
                if pos_match:
                    position = pos_match.group(1)

            # Check for injury status
            injury_status = None
            injury_elem = element.find(class_=re.compile(r"injury|status"))
            if injury_elem:
                injury_status = injury_elem.get_text(strip=True)

            # Check if player is OUT
            element_classes = " ".join(element.get("class", []))
            element_text = element.get_text(strip=True).lower()
            is_out = (
                "is-out" in element_classes
                or "out" in element_classes
                or injury_status and "out" in injury_status.lower()
            )

            if is_out:
                lineup_role = LINEUP_ROLE_OUT
            elif is_confirmed:
                lineup_role = LINEUP_ROLE_CONFIRMED
            else:
                lineup_role = LINEUP_ROLE_PROJECTED

            return RotowireLineupRecord(
                team_abbreviation=team,
                opponent_abbreviation=opponent,
                player_name=player_name,
                position=position,
                lineup_role=lineup_role,
                is_confirmed=is_confirmed,
                injury_status=injury_status,
                ingested_ts=ingested_ts,
            )

        except Exception:
            return None

    def scrape(self) -> List[RotowireGameLineup]:
        """Fetch and parse Rotowire lineups."""
        html = self.fetch_html()
        return self.parse_lineups(html)


def scrape_rotowire_lineups() -> pd.DataFrame:
    """
    Fetch and normalize Rotowire NBA lineups into a DataFrame.

    Returns a DataFrame with columns:
        - team_abbreviation: str
        - opponent_abbreviation: str
        - player_name: str
        - position: str
        - lineup_role: str (projected_starter, confirmed_starter, out)
        - is_confirmed: bool
        - injury_status: str | None
        - ingested_ts: Timestamp
    """
    scraper = RotowireLineupsScraper()
    games = scraper.scrape()

    records: List[Dict] = []
    for game in games:
        for player in game.away_players + game.home_players:
            records.append({
                "team_abbreviation": player.team_abbreviation,
                "opponent_abbreviation": player.opponent_abbreviation,
                "player_name": player.player_name,
                "position": player.position,
                "lineup_role": player.lineup_role,
                "is_confirmed": player.is_confirmed,
                "injury_status": player.injury_status,
                "ingested_ts": player.ingested_ts,
            })

    if not records:
        return pd.DataFrame(columns=[
            "team_abbreviation",
            "opponent_abbreviation",
            "player_name",
            "position",
            "lineup_role",
            "is_confirmed",
            "injury_status",
            "ingested_ts",
        ])

    return pd.DataFrame.from_records(records)


def normalize_rotowire_to_nba_format(
    rotowire_df: pd.DataFrame,
    *,
    target_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Convert Rotowire lineup data to match nba_daily_lineups.py output format.

    This allows Rotowire data to be merged with stats.nba.com lineups.
    Note: player_id is NOT available from Rotowire - must be joined separately.
    """
    if rotowire_df.empty:
        return rotowire_df

    df = rotowire_df.copy()
    df["source"] = "rotowire"
    df["date"] = target_date or pd.Timestamp.now(tz="UTC").normalize()

    # Rename to match nba_daily_lineups schema where possible
    df = df.rename(columns={
        "injury_status": "roster_status",
    })

    # Add missing columns with None/defaults
    df["game_id"] = None  # Not available from Rotowire
    df["player_id"] = None  # Would need name matching
    df["team_id"] = None  # Would need team lookup
    df["game_status"] = None
    df["game_status_text"] = None
    df["lineup_timestamp"] = df["ingested_ts"]

    return df


__all__ = [
    "RotowireLineupRecord",
    "RotowireGameLineup",
    "RotowireLineupsScraper",
    "scrape_rotowire_lineups",
    "normalize_rotowire_to_nba_format",
    "LINEUP_ROLE_CONFIRMED",
    "LINEUP_ROLE_PROJECTED",
    "LINEUP_ROLE_OUT",
]


if __name__ == "__main__":
    # Quick test
    print("Fetching Rotowire lineups...")
    df = scrape_rotowire_lineups()
    print(f"Found {len(df)} players across games")
    if not df.empty:
        print("\nSample data:")
        print(df.head(10).to_string(index=False))
        print(f"\nConfirmed games: {df['is_confirmed'].sum()} / {len(df)} players")
        print(f"\nTeams: {df['team_abbreviation'].unique().tolist()}")
