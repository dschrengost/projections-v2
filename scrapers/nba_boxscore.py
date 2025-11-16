"""Scraper for NBA.com liveData box score payloads."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List

import httpx

from scrapers.nba_schedule import NbaScheduleScraper

BOX_SCORE_URL_TEMPLATE = (
    "https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{game_id}.json"
)


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00") if value.endswith("Z") else value
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _safe_int(value: Any) -> int | None:
    if value in ("", None):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class NbaComArena:
    """Arena metadata embedded in the box score payload."""

    arena_id: int | None
    name: str | None
    city: str | None
    state: str | None
    country: str | None
    timezone: str | None


@dataclass(frozen=True)
class NbaComPlayerStatLine:
    """Individual player stat line from the NBA.com response."""

    person_id: str
    name: str
    first_name: str | None
    family_name: str | None
    jersey_number: str | None
    position: str | None
    starter: bool
    played: bool
    on_court: bool
    order: int | None
    status: str | None
    statistics: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NbaComTeamBoxScore:
    """Aggregated stats for a single team."""

    team_id: int
    team_name: str | None
    team_city: str | None
    team_tricode: str | None
    score: int | None
    in_bonus: bool
    timeouts_remaining: int | None
    statistics: Dict[str, Any] = field(default_factory=dict)
    periods: List[Dict[str, Any]] = field(default_factory=list)
    players: List[NbaComPlayerStatLine] = field(default_factory=list)


@dataclass(frozen=True)
class NbaComGameBoxScore:
    """Full NBA.com box score for a single game."""

    game_id: str
    game_code: str | None
    game_status: int | None
    game_status_text: str | None
    game_clock: str | None
    attendance: int | None
    duration: str | None
    sellout: bool | None
    regulation_periods: int | None
    period: int | None
    game_time_utc: datetime | None
    game_time_local: datetime | None
    game_time_home: datetime | None
    game_time_away: datetime | None
    arena: NbaComArena | None
    away: NbaComTeamBoxScore | None
    home: NbaComTeamBoxScore | None
    officials: List[str] = field(default_factory=list)


class NbaComBoxScoreScraper:
    """Scrape NBA.com box score data via the public liveData endpoints."""

    def __init__(
        self,
        *,
        timeout: float = 10.0,
        user_agent: str | None = None,
        box_score_url_template: str = BOX_SCORE_URL_TEMPLATE,
        schedule_scraper: NbaScheduleScraper | None = None,
    ) -> None:
        self.timeout = timeout
        self.user_agent = user_agent or (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
        self.box_score_url_template = box_score_url_template
        self.schedule_scraper = schedule_scraper or NbaScheduleScraper(
            timeout=timeout, user_agent=self.user_agent
        )

    def fetch_daily_box_scores(
        self, target_date: date, *, season: str | None = None
    ) -> List[NbaComGameBoxScore]:
        """Fetch all completed box scores for the supplied date."""

        scheduled_games = self.schedule_scraper.fetch_daily_schedule(
            target_date, season=season
        )
        results: List[NbaComGameBoxScore] = []
        for game in scheduled_games:
            if not game.game_id:
                continue
            box_score = self.fetch_box_score(game.game_id)
            if box_score:
                results.append(box_score)
        return results

    def fetch_box_score(self, game_id: str) -> NbaComGameBoxScore | None:
        """Fetch and normalize a single box score by game id."""

        url = self.box_score_url_template.format(game_id=game_id)
        try:
            payload = self._http_json(url)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in (403, 404):
                return None
            raise RuntimeError(
                f"HTTP error {exc.response.status_code} while fetching {game_id}"
            ) from exc
        except httpx.RequestError as exc:
            raise RuntimeError(f"Network failure while fetching {game_id}") from exc

        meta = payload.get("meta", {})
        code = meta.get("code")
        if code and code != 200:
            raise RuntimeError(f"NBA.com returned status {code} for game {game_id}")
        game_payload = payload.get("game")
        if not game_payload:
            return None
        return self._build_game(game_payload)

    def _http_json(self, url: str) -> Dict[str, Any]:
        headers = {
            "User-Agent": self.user_agent,
            "Referer": "https://www.nba.com/",
            "Accept": "application/json",
            "Origin": "https://www.nba.com",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "close",
        }
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(url, headers=headers)
            response.raise_for_status()
            return response.json()

    def _build_game(self, data: Dict[str, Any]) -> NbaComGameBoxScore:
        return NbaComGameBoxScore(
            game_id=data.get("gameId", ""),
            game_code=data.get("gameCode"),
            game_status=_safe_int(data.get("gameStatus")),
            game_status_text=data.get("gameStatusText"),
            game_clock=data.get("gameClock"),
            attendance=_safe_int(data.get("attendance")),
            duration=data.get("duration"),
            sellout=data.get("sellout"),
            regulation_periods=_safe_int(data.get("regulationPeriods")),
            period=_safe_int(data.get("period")),
            game_time_utc=_parse_datetime(data.get("gameTimeUTC")),
            game_time_local=_parse_datetime(data.get("gameTimeLocal")),
            game_time_home=_parse_datetime(data.get("gameTimeHome")),
            game_time_away=_parse_datetime(data.get("gameTimeAway")),
            arena=self._build_arena(data.get("arena")),
            away=self._build_team(data.get("awayTeam")),
            home=self._build_team(data.get("homeTeam")),
            officials=self._extract_officials(data.get("officials", [])),
        )

    def _build_arena(self, data: Dict[str, Any] | None) -> NbaComArena | None:
        if not data:
            return None
        return NbaComArena(
            arena_id=_safe_int(data.get("arenaId")),
            name=data.get("arenaName"),
            city=data.get("arenaCity"),
            state=data.get("arenaState"),
            country=data.get("arenaCountry"),
            timezone=data.get("arenaTimezone"),
        )

    def _build_team(self, data: Dict[str, Any] | None) -> NbaComTeamBoxScore | None:
        if not data:
            return None
        return NbaComTeamBoxScore(
            team_id=_safe_int(data.get("teamId")) or 0,
            team_name=data.get("teamName"),
            team_city=data.get("teamCity"),
            team_tricode=data.get("teamTricode"),
            score=_safe_int(data.get("score")),
            in_bonus=bool(data.get("inBonus")),
            timeouts_remaining=_safe_int(data.get("timeoutsRemaining")),
            statistics=dict(data.get("statistics") or {}),
            periods=list(data.get("periods") or []),
            players=[self._build_player(player) for player in data.get("players", [])],
        )

    def _build_player(self, data: Dict[str, Any]) -> NbaComPlayerStatLine:
        return NbaComPlayerStatLine(
            person_id=str(data.get("personId", "")),
            name=data.get("name", ""),
            first_name=data.get("firstName"),
            family_name=data.get("familyName"),
            jersey_number=data.get("jerseyNum"),
            position=data.get("position"),
            starter=bool(data.get("starter")),
            played=bool(data.get("played")),
            on_court=bool(data.get("oncourt")),
            order=_safe_int(data.get("order")),
            status=data.get("status"),
            statistics=dict(data.get("statistics") or {}),
        )

    def _extract_officials(self, officials: List[Dict[str, Any]]) -> List[str]:
        return [official.get("name", "") for official in officials if official.get("name")]


__all__ = [
    "NbaComArena",
    "NbaComBoxScoreScraper",
    "NbaComGameBoxScore",
    "NbaComPlayerStatLine",
    "NbaComTeamBoxScore",
]
