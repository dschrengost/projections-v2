"""Scraper for RotoWire NBA player props.

This scraper fetches player prop betting lines from RotoWire's props page.
The data is embedded as JSON in the page HTML, making extraction straightforward.

Sportsbooks available: DraftKings, FanDuel, BetMGM, Caesars, BetRivers, Hard Rock, ESPN Bet
Prop types: pts, reb, ast, threes, blk, stl, turnovers, and combos (ptsrebast, ptsreb, etc.)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

ROTOWIRE_PROPS_URL = "https://www.rotowire.com/betting/nba/player-props.php"

DEFAULT_HEADERS: dict[str, str] = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}

# Prop types available on RotoWire
PROP_TYPES = [
    "pts",
    "reb",
    "ast",
    "threes",
    "blk",
    "stl",
    "turnovers",
    "ptsrebast",
    "ptsreb",
    "ptsast",
    "rebast",
    "stlblk",
]

# Sportsbooks available
SPORTSBOOKS = [
    "draftkings",
    "fanduel",
    "mgm",
    "caesars",
    "betrivers",
    "hardrock",
    "espnbet",
]


@dataclass
class PropLine:
    """A single prop line from a sportsbook."""

    book: str
    prop_type: str
    line: float | None
    over_odds: int | None
    under_odds: int | None

    def implied_over_prob(self) -> float | None:
        """Calculate implied probability for the over."""
        if self.over_odds is None:
            return None
        return _american_to_implied_prob(self.over_odds)

    def implied_under_prob(self) -> float | None:
        """Calculate implied probability for the under."""
        if self.under_odds is None:
            return None
        return _american_to_implied_prob(self.under_odds)


@dataclass
class PlayerProps:
    """All props for a single player."""

    player_id: str
    player_name: str
    first_name: str
    last_name: str
    team: str
    opponent: str
    game_id: str
    logo_url: str
    player_link: str
    lines: list[PropLine] = field(default_factory=list)

    def get_line(self, book: str, prop_type: str) -> PropLine | None:
        """Get a specific line by book and prop type."""
        for line in self.lines:
            if line.book == book and line.prop_type == prop_type:
                return line
        return None

    def best_over_odds(self, prop_type: str) -> PropLine | None:
        """Find the best over odds across all books for a prop type."""
        candidates = [l for l in self.lines if l.prop_type == prop_type and l.over_odds]
        if not candidates:
            return None
        return max(candidates, key=lambda x: x.over_odds or -999)

    def best_under_odds(self, prop_type: str) -> PropLine | None:
        """Find the best under odds across all books for a prop type."""
        candidates = [l for l in self.lines if l.prop_type == prop_type and l.under_odds]
        if not candidates:
            return None
        return max(candidates, key=lambda x: x.under_odds or -999)


def _american_to_implied_prob(odds: int) -> float:
    """Convert American odds to implied probability."""
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)


def _parse_odds(val: Any) -> int | None:
    """Parse odds value from JSON (can be string, int, or null)."""
    if val is None:
        return None
    if isinstance(val, int):
        return val
    if isinstance(val, str):
        try:
            return int(val)
        except ValueError:
            return None
    return None


def _parse_line(val: Any) -> float | None:
    """Parse line value from JSON (can be string, float, or null)."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        if val == "":
            return None
        try:
            return float(val)
        except ValueError:
            return None
    return None


class RotowirePropsScraper:
    """Scrape NBA player props from RotoWire."""

    def __init__(
        self,
        *,
        timeout: float = 30.0,
        user_agent: str | None = None,
        url: str = ROTOWIRE_PROPS_URL,
        client: httpx.Client | None = None,
    ) -> None:
        self.timeout = timeout
        self.url = url
        self.user_agent = user_agent or (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
        self.client = client

    def fetch_html(self) -> str:
        """Fetch the raw HTML from RotoWire props page."""
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
                f"RotoWire returned {exc.response.status_code} for {self.url}"
            ) from exc
        except httpx.RequestError as exc:
            raise RuntimeError(f"Failed to fetch {self.url}") from exc
        finally:
            if close_session:
                session.close()

    def parse_props(self, html: str) -> list[PlayerProps]:
        """Parse the HTML and extract player props data.

        The data is embedded as JSON arrays in the JavaScript code.
        Each prop type has its own data array with lines and odds.
        """
        # Extract all data arrays from the page
        # Format: data: [{"gameID":..., "playerID":..., ...}],
        pattern = r'data:\s*(\[{"gameID".*?}\])\s*,'
        matches = re.findall(pattern, html, re.DOTALL)

        if not matches:
            return []

        # Build a map of player_id -> PlayerProps
        players: dict[str, PlayerProps] = {}

        for match in matches:
            try:
                data = json.loads(match)
            except json.JSONDecodeError:
                continue

            for record in data:
                player_id = str(record.get("playerID", ""))
                if not player_id:
                    continue

                # Get or create player
                if player_id not in players:
                    players[player_id] = PlayerProps(
                        player_id=player_id,
                        player_name=record.get("name", ""),
                        first_name=record.get("firstName", ""),
                        last_name=record.get("lastName", ""),
                        team=record.get("team", ""),
                        opponent=record.get("opp", "").lstrip("@"),
                        game_id=str(record.get("gameID", "")),
                        logo_url=record.get("logo", ""),
                        player_link=record.get("playerLink", ""),
                    )

                player = players[player_id]

                # Extract lines for each sportsbook and prop type
                for book in SPORTSBOOKS:
                    for prop_type in PROP_TYPES:
                        line_key = f"{book}_{prop_type}"
                        over_key = f"{book}_{prop_type}Over"
                        under_key = f"{book}_{prop_type}Under"

                        line_val = _parse_line(record.get(line_key))
                        over_odds = _parse_odds(record.get(over_key))
                        under_odds = _parse_odds(record.get(under_key))

                        # Only add if we have at least the line
                        if line_val is not None:
                            # Check if we already have this line (avoid duplicates)
                            existing = player.get_line(book, prop_type)
                            if existing is None:
                                player.lines.append(
                                    PropLine(
                                        book=book,
                                        prop_type=prop_type,
                                        line=line_val,
                                        over_odds=over_odds,
                                        under_odds=under_odds,
                                    )
                                )

        return list(players.values())

    def scrape(self) -> list[PlayerProps]:
        """Fetch and parse RotoWire props."""
        html = self.fetch_html()
        return self.parse_props(html)


def scrape_rotowire_props() -> pd.DataFrame:
    """
    Fetch and normalize RotoWire NBA player props into a DataFrame.

    Returns a DataFrame with columns:
        - player_id: str (RotoWire's player ID)
        - player_name: str
        - team: str
        - opponent: str
        - game_id: str
        - book: str (sportsbook name)
        - prop_type: str (pts, reb, ast, etc.)
        - line: float
        - over_odds: int | None
        - under_odds: int | None
        - implied_over_prob: float | None
        - implied_under_prob: float | None
        - scraped_at: Timestamp
    """
    scraper = RotowirePropsScraper()
    players = scraper.scrape()
    scraped_at = pd.Timestamp.now(tz="UTC")

    records: list[dict] = []
    for player in players:
        for line in player.lines:
            records.append(
                {
                    "player_id": player.player_id,
                    "player_name": player.player_name,
                    "team": player.team,
                    "opponent": player.opponent,
                    "game_id": player.game_id,
                    "book": line.book,
                    "prop_type": line.prop_type,
                    "line": line.line,
                    "over_odds": line.over_odds,
                    "under_odds": line.under_odds,
                    "implied_over_prob": line.implied_over_prob(),
                    "implied_under_prob": line.implied_under_prob(),
                    "scraped_at": scraped_at,
                }
            )

    if not records:
        return pd.DataFrame(
            columns=[
                "player_id",
                "player_name",
                "team",
                "opponent",
                "game_id",
                "book",
                "prop_type",
                "line",
                "over_odds",
                "under_odds",
                "implied_over_prob",
                "implied_under_prob",
                "scraped_at",
            ]
        )

    return pd.DataFrame.from_records(records)


def scrape_props_wide() -> pd.DataFrame:
    """
    Fetch props in a wide format with one row per player.

    Each book+prop combination becomes its own columns:
        - draftkings_pts, draftkings_pts_over, draftkings_pts_under, etc.

    This format is useful for comparing lines across books.
    """
    scraper = RotowirePropsScraper()
    players = scraper.scrape()
    scraped_at = pd.Timestamp.now(tz="UTC")

    records: list[dict] = []
    for player in players:
        row: dict[str, Any] = {
            "player_id": player.player_id,
            "player_name": player.player_name,
            "team": player.team,
            "opponent": player.opponent,
            "game_id": player.game_id,
            "scraped_at": scraped_at,
        }

        for line in player.lines:
            prefix = f"{line.book}_{line.prop_type}"
            row[prefix] = line.line
            row[f"{prefix}_over"] = line.over_odds
            row[f"{prefix}_under"] = line.under_odds

        records.append(row)

    if not records:
        return pd.DataFrame()

    return pd.DataFrame.from_records(records)


def save_props_bronze(
    df: pd.DataFrame,
    game_date: date,
    data_root: Path | str | None = None,
) -> Path:
    """Save props DataFrame to bronze layer as parquet.

    File path: {data_root}/bronze/props/game_date={YYYY-MM-DD}/props_{timestamp}.parquet
    """
    if data_root is None:
        import os

        data_root = Path(os.environ.get("PROJECTIONS_DATA_ROOT", "data"))
    else:
        data_root = Path(data_root)

    date_str = game_date.isoformat()
    timestamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")

    out_dir = data_root / "bronze" / "props" / f"game_date={date_str}"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"props_{timestamp}.parquet"
    df.to_parquet(out_path, index=False)

    return out_path


__all__ = [
    "PropLine",
    "PlayerProps",
    "RotowirePropsScraper",
    "scrape_rotowire_props",
    "scrape_props_wide",
    "save_props_bronze",
    "PROP_TYPES",
    "SPORTSBOOKS",
]


if __name__ == "__main__":
    print("Fetching RotoWire player props...")
    df = scrape_rotowire_props()
    print(f"Found {len(df)} prop lines for {df['player_name'].nunique()} players")

    if not df.empty:
        print("\n=== Sample data (DraftKings points) ===")
        dk_pts = df[(df["book"] == "draftkings") & (df["prop_type"] == "pts")].head(10)
        print(
            dk_pts[
                ["player_name", "team", "opponent", "line", "over_odds", "under_odds"]
            ].to_string(index=False)
        )

        print("\n=== Prop types available ===")
        print(df["prop_type"].value_counts().to_string())

        print("\n=== Books available ===")
        print(df["book"].value_counts().to_string())
