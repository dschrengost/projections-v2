"""Scraper for stats.nba.com daily lineup payloads."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, Iterable

import httpx
import pandas as pd

NBA_DAILY_LINEUPS_URL_TEMPLATE = (
    "https://stats.nba.com/js/data/leaders/00_daily_lineups_{date}.json"
)

DEFAULT_HEADERS: Dict[str, str] = {
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "close",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
}

LINEUP_ROLE_PROJECTED = "projected_starter"
LINEUP_ROLE_CONFIRMED = "confirmed_starter"
LINEUP_ROLE_BENCH = "bench"
LINEUP_ROLE_OUT = "out"


def _season_start_from_day(day: date) -> int:
    return day.year if day.month >= 8 else day.year - 1


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_timestamp(value: str | None) -> pd.Timestamp | None:
    if not value:
        return None
    try:
        timestamp = pd.Timestamp(value)
    except ValueError:
        return None
    if timestamp.tzinfo is None:
        try:
            return timestamp.tz_localize("UTC")
        except TypeError:
            return None
    return timestamp.tz_convert("UTC")


def _normalize_role(
    lineup_status: str | None,
    roster_status: str | None,
    position: str | None,
) -> str:
    status = (lineup_status or "").strip().lower()
    roster = (roster_status or "").strip().lower()
    has_position = bool((position or "").strip())

    if roster in {"inactive", "out"} or status in {"out", "injured"}:
        return LINEUP_ROLE_OUT
    if has_position and status in {"confirmed", "starter", "starting"}:
        return LINEUP_ROLE_CONFIRMED
    if has_position and status in {"expected", "projected", "probable"}:
        return LINEUP_ROLE_PROJECTED
    return LINEUP_ROLE_BENCH


class NbaDailyLineupsScraper:
    """Fetch the NBA daily lineup JSON payload for a given date."""

    def __init__(
        self,
        *,
        timeout: float = 10.0,
        user_agent: str | None = None,
        url_template: str = NBA_DAILY_LINEUPS_URL_TEMPLATE,
        client: httpx.Client | None = None,
    ) -> None:
        self.timeout = timeout
        self.url_template = url_template
        self.user_agent = user_agent or (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
        self.client = client

    def fetch_daily_lineups(self, target_date: date) -> Dict[str, Any]:
        """Fetch the stats.nba.com payload for ``target_date``."""

        day_token = target_date.strftime("%Y%m%d")
        url = self.url_template.format(date=day_token)
        headers = dict(DEFAULT_HEADERS)
        headers["User-Agent"] = self.user_agent
        session = self.client
        close_session = False
        if session is None:
            session = httpx.Client(timeout=self.timeout)
            close_session = True
        try:
            response = session.get(url, headers=headers, timeout=self.timeout)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:  # pragma: no cover - HTTP error path
            raise RuntimeError(
                f"stats.nba.com returned {exc.response.status_code} for {url}"
            ) from exc
        except httpx.RequestError as exc:  # pragma: no cover - HTTP error path
            raise RuntimeError(f"Failed to fetch {url}") from exc
        finally:
            if close_session:
                session.close()
        return response.json()


@dataclass(frozen=True)
class DailyLineupRecord:
    game_id: int | None
    game_status: int | None
    game_status_text: str | None
    season_start: int
    date: pd.Timestamp
    team_id: int | None
    team_abbreviation: str | None
    is_home: bool
    player_id: int | None
    player_name: str | None
    first_name: str | None
    last_name: str | None
    position: str | None
    lineup_status: str | None
    roster_status: str | None
    lineup_role: str
    lineup_timestamp: pd.Timestamp | None
    ingested_ts: pd.Timestamp


def _iter_lineup_records(
    payload: Dict[str, Any],
    *,
    target_date: date,
    season_start: int,
    ingested_ts: pd.Timestamp,
) -> Iterable[DailyLineupRecord]:
    games = payload.get("games") or []
    normalized_day = pd.Timestamp(target_date).normalize()
    for game in games:
        game_id = _coerce_int(game.get("gameId"))
        status = game.get("gameStatus")
        status_text = game.get("gameStatusText")
        for side_key, is_home in (("homeTeam", True), ("awayTeam", False)):
            team = game.get(side_key) or {}
            team_id = _coerce_int(team.get("teamId"))
            team_abbrev = team.get("teamAbbreviation")
            for player in team.get("players") or []:
                player_id = _coerce_int(player.get("personId"))
                yield DailyLineupRecord(
                    game_id=game_id,
                    game_status=_coerce_int(status),
                    game_status_text=status_text,
                    season_start=season_start,
                    date=normalized_day,
                    team_id=team_id,
                    team_abbreviation=team_abbrev,
                    is_home=is_home,
                    player_id=player_id,
                    player_name=player.get("playerName"),
                    first_name=player.get("firstName"),
                    last_name=player.get("lastName"),
                    position=player.get("position"),
                    lineup_status=player.get("lineupStatus"),
                    roster_status=player.get("rosterStatus"),
                    lineup_role=_normalize_role(
                        player.get("lineupStatus"),
                        player.get("rosterStatus"),
                        player.get("position"),
                    ),
                    lineup_timestamp=_parse_timestamp(player.get("timestamp")),
                    ingested_ts=ingested_ts,
                )


def normalize_daily_lineups(
    payload: Dict[str, Any],
    *,
    target_date: date,
    season_start: int | None = None,
    ingested_ts: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Normalize the lineup JSON into a flat player-level dataframe."""

    inferred_season = season_start or _season_start_from_day(target_date)
    timestamp = (
        pd.Timestamp.now(tz="UTC")
        if ingested_ts is None
        else pd.to_datetime(ingested_ts, utc=True)
    )
    records = list(
        _iter_lineup_records(
            payload,
            target_date=target_date,
            season_start=inferred_season,
            ingested_ts=timestamp,
        )
    )
    columns = [
        "game_id",
        "game_status",
        "game_status_text",
        "season_start",
        "date",
        "team_id",
        "team_abbreviation",
        "is_home",
        "player_id",
        "player_name",
        "first_name",
        "last_name",
        "position",
        "lineup_status",
        "roster_status",
        "lineup_role",
        "lineup_timestamp",
        "ingested_ts",
    ]
    if not records:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame.from_records([record.__dict__ for record in records], columns=columns)


__all__ = [
    "DailyLineupRecord",
    "LINEUP_ROLE_BENCH",
    "LINEUP_ROLE_CONFIRMED",
    "LINEUP_ROLE_OUT",
    "LINEUP_ROLE_PROJECTED",
    "NbaDailyLineupsScraper",
    "normalize_daily_lineups",
]
