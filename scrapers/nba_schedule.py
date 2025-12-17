from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Dict, List
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


LEAGUE_SCHEDULE_URL = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"
TODAYS_SCOREBOARD_URL = "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json"
MOBILE_SCHEDULE_URL_TEMPLATE = (
    "https://data.nba.com/data/10s/v2015/json/mobile_teams/nba/{season_start}/league/00_full_schedule.json"
)


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00") if value.endswith("Z") else value
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


@dataclass(frozen=True)
class ScheduledTeam:
    """Minimal view of a team entry inside the schedule payload."""

    team_id: int
    team_name: str
    team_city: str
    team_tricode: str
    wins: int | None
    losses: int | None
    score: int | None
    seed: int | None = None
    slug: str | None = None


@dataclass(frozen=True)
class ScheduledGame:
    """Normalized representation of one scheduled NBA game."""

    game_id: str
    game_code: str
    status: int
    status_text: str
    game_label: str | None
    game_sub_label: str | None
    season_year: str | None
    game_time_utc: datetime | None
    is_neutral: bool
    series_game_number: str | None
    game_subtype: str | None
    week_number: int | None
    arena_name: str | None
    arena_city: str | None
    arena_state: str | None
    broadcasters: Dict[str, List[str]] = field(default_factory=dict)
    home_team: ScheduledTeam | None = None
    away_team: ScheduledTeam | None = None
    local_game_date: date | None = None


class NbaScheduleScraper:
    """Scrape NBA schedule data from nba.com endpoints."""

    def __init__(
        self,
        *,
        timeout: float = 10.0,
        user_agent: str | None = None,
        league_schedule_url: str = LEAGUE_SCHEDULE_URL,
        scoreboard_url: str = TODAYS_SCOREBOARD_URL,
        mobile_schedule_url_template: str = MOBILE_SCHEDULE_URL_TEMPLATE,
    ) -> None:
        self.timeout = timeout
        self.user_agent = user_agent or (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
        self.league_schedule_url = league_schedule_url
        self.scoreboard_url = scoreboard_url
        self.mobile_schedule_url_template = mobile_schedule_url_template

    def fetch_season_schedule(self, season: str | None = None) -> List[ScheduledGame]:
        if season:
            return self._fetch_mobile_schedule(season)

        payload = self._http_json(self.league_schedule_url)
        schedule = payload.get("leagueSchedule", {})
        season_year = schedule.get("seasonYear")
        games: List[ScheduledGame] = []
        for date_entry in schedule.get("gameDates", []):
            for game in date_entry.get("games", []):
                games.append(self._build_game_from_league(game, season_year))
        return games

    def fetch_daily_schedule(
        self, target_date: date | None = None, *, season: str | None = None
    ) -> List[ScheduledGame]:
        day = target_date or self._current_date()
        if season is None and target_date is None:
            try:
                games = self._fetch_scoreboard_games()
                # Validate: if scoreboard returns bad data (e.g., team_id=0), fall back to season schedule
                for g in games:
                    if not (g.home_team and g.home_team.team_id) or not (g.away_team and g.away_team.team_id):
                        games = []
                        break
            except RuntimeError:
                games = []
            else:
                filtered = [
                    game for game in games if game.game_time_utc and game.game_time_utc.date() == day
                ]
                if filtered:
                    return filtered
        season_games = self.fetch_season_schedule(season=season)
        results: List[ScheduledGame] = []
        for game in season_games:
            match_date = game.game_time_utc.date() if game.game_time_utc else None
            if match_date != day and game.local_game_date:
                match_date = game.local_game_date
            if match_date == day:
                results.append(game)
        return results

    def _fetch_scoreboard_games(self) -> List[ScheduledGame]:
        payload = self._http_json(self.scoreboard_url)
        scoreboard = payload.get("scoreboard", {})
        return [
            self._build_game_from_scoreboard(game)
            for game in scoreboard.get("games", [])
        ]

    def _build_game_from_league(self, game: Dict[str, Any], season_year: str | None) -> ScheduledGame:
        broadcasters = self._extract_broadcasters(game.get("broadcasters", {}))
        return ScheduledGame(
            game_id=game.get("gameId", ""),
            game_code=game.get("gameCode", ""),
            status=int(game.get("gameStatus", 0) or 0),
            status_text=game.get("gameStatusText", ""),
            game_label=game.get("gameLabel"),
            game_sub_label=game.get("gameSubLabel"),
            season_year=season_year,
            game_time_utc=_parse_datetime(game.get("gameDateTimeUTC")),
            is_neutral=bool(game.get("isNeutral", False)),
            series_game_number=game.get("seriesGameNumber"),
            game_subtype=game.get("gameSubtype"),
            week_number=self._safe_int(game.get("weekNumber")),
            arena_name=game.get("arenaName"),
            arena_city=game.get("arenaCity"),
            arena_state=game.get("arenaState"),
            broadcasters=broadcasters,
            home_team=self._build_team(game.get("homeTeam", {})),
            away_team=self._build_team(game.get("awayTeam", {})),
            local_game_date=self._parse_date(game.get("gameDateEst")),
        )

    def _build_game_from_scoreboard(self, game: Dict[str, Any]) -> ScheduledGame:
        return ScheduledGame(
            game_id=game.get("gameId", ""),
            game_code=game.get("gameCode", ""),
            status=int(game.get("gameStatus", 0) or 0),
            status_text=game.get("gameStatusText", ""),
            game_label=game.get("gameLabel"),
            game_sub_label=game.get("gameSubLabel"),
            season_year=None,
            game_time_utc=_parse_datetime(game.get("gameTimeUTC")),
            is_neutral=bool(game.get("isNeutral", False)),
            series_game_number=game.get("seriesGameNumber"),
            game_subtype=game.get("gameSubtype"),
            week_number=None,
            arena_name=None,
            arena_city=None,
            arena_state=None,
            broadcasters={},
            home_team=self._build_team(game.get("homeTeam", {})),
            away_team=self._build_team(game.get("awayTeam", {})),
            local_game_date=self._parse_date(game.get("gameDate")),
        )

    def _fetch_mobile_schedule(self, season: str) -> List[ScheduledGame]:
        season_start = season.split("-", 1)[0]
        url = self.mobile_schedule_url_template.format(season_start=season_start)
        payload = self._http_json(url)
        games: List[ScheduledGame] = []
        for block in payload.get("lscd", []):
            month = block.get("mscd", {})
            for game in month.get("g", []):
                games.append(self._build_game_from_mobile(game, season))
        return games

    def _build_game_from_mobile(self, game: Dict[str, Any], season_year: str) -> ScheduledGame:
        game_time = self._combine_mobile_datetime(game.get("gdtutc"), game.get("utctm"))
        broadcasters = self._extract_mobile_broadcasters(game.get("bd"))
        return ScheduledGame(
            game_id=game.get("gid", ""),
            game_code=game.get("gcode", ""),
            status=self._safe_int(game.get("st")) or 0,
            status_text=game.get("stt", ""),
            game_label=game.get("seri"),
            game_sub_label=None,
            season_year=season_year,
            game_time_utc=game_time,
            is_neutral=False,
            series_game_number=game.get("seq"),
            game_subtype=None,
            week_number=None,
            arena_name=game.get("an"),
            arena_city=game.get("ac"),
            arena_state=game.get("as"),
            broadcasters=broadcasters,
            home_team=self._build_mobile_team(game.get("h", {})),
            away_team=self._build_mobile_team(game.get("v", {})),
            local_game_date=self._parse_date(game.get("gdte")),
        )

    def _build_team(self, payload: Dict[str, Any]) -> ScheduledTeam | None:
        if not payload:
            return None
        return ScheduledTeam(
            team_id=int(payload.get("teamId") or 0),
            team_name=payload.get("teamName", ""),
            team_city=payload.get("teamCity", ""),
            team_tricode=payload.get("teamTricode", ""),
            wins=self._safe_int(payload.get("wins")),
            losses=self._safe_int(payload.get("losses")),
            score=self._safe_int(payload.get("score")),
            seed=self._safe_int(payload.get("seed")),
            slug=payload.get("teamSlug"),
        )

    def _extract_broadcasters(self, payload: Dict[str, Any]) -> Dict[str, List[str]]:
        results: Dict[str, List[str]] = {}
        for key, entries in payload.items():
            names = [
                entry.get("broadcasterDisplay", "").strip()
                for entry in entries
                if entry.get("broadcasterDisplay")
            ]
            if names:
                results[key] = names
        return results

    def _build_mobile_team(self, payload: Dict[str, Any]) -> ScheduledTeam | None:
        if not payload:
            return None
        wins, losses = self._parse_record(payload.get("re"))
        return ScheduledTeam(
            team_id=int(payload.get("tid") or 0),
            team_name=payload.get("tn", ""),
            team_city=payload.get("tc", ""),
            team_tricode=payload.get("ta", ""),
            wins=wins,
            losses=losses,
            score=self._safe_int(payload.get("s")),
            seed=None,
            slug=None,
        )

    def _extract_mobile_broadcasters(self, payload: Dict[str, Any] | None) -> Dict[str, List[str]]:
        if not payload:
            return {}
        groups: Dict[str, List[str]] = {}
        for entry in payload.get("b", []):
            scope = entry.get("scope") or "other"
            name = entry.get("disp")
            if not name:
                continue
            groups.setdefault(scope, []).append(name)
        return groups

    def _combine_mobile_datetime(
        self, date_str: str | None, time_str: str | None
    ) -> datetime | None:
        if not date_str or not time_str:
            return None
        try:
            parsed = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
        except ValueError:
            return None
        return parsed.replace(tzinfo=timezone.utc)

    def _http_json(self, url: str) -> Dict[str, Any]:
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "application/json",
            "Referer": "https://www.nba.com/schedule",
            "Origin": "https://www.nba.com",
        }
        request = Request(url, headers=headers)
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError) as exc:  # pragma: no cover - network errors
            raise RuntimeError(f"Failed to fetch {url}: {exc}") from exc

    def _current_date(self) -> date:
        return datetime.now(timezone.utc).date()

    @staticmethod
    def _safe_int(value: Any) -> int | None:
        if value is None or value == "":
            return None
        try:
            return int(value)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _parse_record(record: str | None) -> tuple[int | None, int | None]:
        if not record or "-" not in record:
            return None, None
        wins, losses = record.split("-", 1)
        try:
            return int(wins), int(losses)
        except ValueError:
            return None, None

    @staticmethod
    def _parse_date(value: str | None) -> date | None:
        if not value:
            return None
        normalized = value.replace("Z", "") if isinstance(value, str) else value
        for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f"):
            try:
                return datetime.strptime(normalized, fmt).date()
            except ValueError:
                continue
        try:
            return datetime.fromisoformat(value).date()
        except ValueError:
            return None


__all__ = [
    "LEAGUE_SCHEDULE_URL",
    "TODAYS_SCOREBOARD_URL",
    "NbaScheduleScraper",
    "ScheduledGame",
    "ScheduledTeam",
]
