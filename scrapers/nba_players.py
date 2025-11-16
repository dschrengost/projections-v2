"""Scraper for NBA.com players index (active rosters)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Iterable, List

import httpx

PLAYER_PAGE_URL = "https://www.nba.com/players"
NEXT_DATA_PATTERN = re.compile(
    r'<script id="__NEXT_DATA__" type="application/json">(?P<data>.+?)</script>',
    re.DOTALL,
)


@dataclass(frozen=True)
class PlayerProfile:
    """Normalized player entry from NBA.com."""

    person_id: int
    player_slug: str
    first_name: str
    last_name: str
    team_id: int | None
    team_slug: str | None
    team_abbreviation: str | None
    team_name: str | None
    jersey_number: str | None
    position: str | None
    height: str | None
    weight: str | None
    country: str | None
    roster_status: int


class NbaPlayersScraper:
    """Scrape the NBA.com players landing page for active roster data."""

    def __init__(
        self,
        *,
        timeout: float = 10.0,
        user_agent: str | None = None,
        players_url: str = PLAYER_PAGE_URL,
    ) -> None:
        self.timeout = timeout
        self.players_url = players_url
        self.user_agent = user_agent or (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )

    def fetch_players(self, *, active_only: bool = True) -> List[PlayerProfile]:
        """Fetch league-wide player profiles from NBA.com."""

        html = self._http_html(self.players_url)
        payload = self._extract_next_data(html)
        players = payload.get("props", {}).get("pageProps", {}).get("players")
        if not isinstance(players, list):
            raise RuntimeError("NBA.com response missing players array.")
        profiles = [self._build_profile(entry) for entry in players]
        if active_only:
            profiles = [profile for profile in profiles if profile.roster_status == 1]
        return profiles

    def _http_html(self, url: str) -> str:
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Referer": "https://www.nba.com/",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "close",
        }
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(url, headers=headers)
            response.raise_for_status()
            return response.text

    def _extract_next_data(self, html: str) -> dict:
        match = NEXT_DATA_PATTERN.search(html)
        if not match:
            raise RuntimeError("Unable to locate __NEXT_DATA__ payload on NBA.com players page.")
        try:
            return json.loads(match.group("data"))
        except json.JSONDecodeError as exc:
            raise RuntimeError("Failed to parse __NEXT_DATA__ JSON payload.") from exc

    def _build_profile(self, data: dict) -> PlayerProfile:
        return PlayerProfile(
            person_id=_safe_int(data.get("PERSON_ID"), default=0) or 0,
            player_slug=str(data.get("PLAYER_SLUG") or ""),
            first_name=str(data.get("PLAYER_FIRST_NAME") or ""),
            last_name=str(data.get("PLAYER_LAST_NAME") or ""),
            team_id=_safe_int(data.get("TEAM_ID")),
            team_slug=_safe_str(data.get("TEAM_SLUG")),
            team_abbreviation=_safe_str(data.get("TEAM_ABBREVIATION")),
            team_name=_safe_str(data.get("TEAM_NAME")),
            jersey_number=_safe_str(data.get("JERSEY_NUMBER")),
            position=_safe_str(data.get("POSITION")),
            height=_safe_str(data.get("HEIGHT")),
            weight=_safe_str(data.get("WEIGHT")),
            country=_safe_str(data.get("COUNTRY")),
            roster_status=_safe_int(data.get("ROSTER_STATUS"), default=0) or 0,
        )


def _safe_int(value: object, *, default: int | None = None) -> int | None:
    if value in (None, "", "null"):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_str(value: object) -> str | None:
    if value in (None, "", "null"):
        return None
    return str(value)


__all__ = ["NbaPlayersScraper", "PlayerProfile"]
