"""HTTP client for stats.nba.com tracking endpoints."""

from __future__ import annotations

from datetime import date
import logging
import time
from typing import Any, Dict, Mapping

import httpx

LOGGER = logging.getLogger(__name__)

NBA_STATS_TRACKING_URL = "https://stats.nba.com/stats/leaguedashptstats"

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

DEFAULT_HEADERS: Dict[str, str] = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "Host": "stats.nba.com",
    "Origin": "https://www.nba.com",
    "Pragma": "no-cache",
    "Referer": "https://www.nba.com/stats/",
    "DNT": "1",
}

RETRY_STATUS_CODES = {429}


class TrackingClientError(RuntimeError):
    """Raised when the tracking client fails."""


def _encode_game_date(game_date: date) -> str:
    return game_date.strftime("%m/%d/%Y")


def _default_params(
    *,
    season: str,
    season_type: str,
    pt_measure_type: str,
    per_mode: str,
    player_or_team: str,
    game_date: date,
) -> Dict[str, Any]:
    day_token = _encode_game_date(game_date)
    return {
        "PtMeasureType": pt_measure_type,
        "PerMode": per_mode,
        "PlayerOrTeam": player_or_team,
        "Season": season,
        "SeasonType": season_type,
        "DateFrom": day_token,
        "DateTo": day_token,
        "LastNGames": 0,
        "Month": 0,
        "OpponentTeamID": 0,
        "Outcome": "",
        "Location": "",
        "SeasonSegment": "",
        "VsConference": "",
        "VsDivision": "",
        "TeamID": 0,
        "Conference": "",
        "Division": "",
        "GameScope": "",
        "PlayerExperience": "",
        "PlayerPosition": "",
        "StarterBench": "",
        "DraftYear": "",
        "DraftPick": "",
        "College": "",
        "Country": "",
        "Height": "",
        "Weight": "",
        "PORound": 0,
        "PaceAdjust": "N",
        "Rank": "N",
        "LeagueID": "00",
        "PlusMinus": "N",
        "GameSegment": "",
        "VsPlayerID": "",
        "VsTeamID": "",
        "ClutchTime": "",
        "ShotClockRange": "",
        "DribbleRange": "",
        "TouchTimeRange": "",
        "DistanceRange": "",
    }


def fetch_leaguedashptstats(
    *,
    season: str,
    season_type: str,
    pt_measure_type: str,
    game_date: date,
    per_mode: str = "Totals",
    player_or_team: str = "Player",
    timeout: float = 15.0,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    user_agent: str | None = None,
    extra_headers: Mapping[str, str] | None = None,
    client: httpx.Client | None = None,
) -> Dict[str, Any]:
    """Fetch tracking payload for ``game_date`` across ``pt_measure_type``."""

    headers = dict(DEFAULT_HEADERS)
    headers["User-Agent"] = user_agent or DEFAULT_USER_AGENT
    if extra_headers:
        headers.update(extra_headers)

    params = _default_params(
        season=season,
        season_type=season_type,
        pt_measure_type=pt_measure_type,
        per_mode=per_mode,
        player_or_team=player_or_team,
        game_date=game_date,
    )
    session = client or httpx.Client(
        timeout=timeout, headers=headers, follow_redirects=True
    )
    close_session = client is None
    attempts = max(1, max_retries)
    try:
        for attempt in range(attempts):
            try:
                response = session.get(
                    NBA_STATS_TRACKING_URL,
                    params=params,
                    timeout=timeout,
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code
                should_retry = status_code in RETRY_STATUS_CODES or status_code >= 500
                LOGGER.warning(
                    "[tracking] stats.nba.com returned %s for %s %s (%s)",
                    status_code,
                    game_date.isoformat(),
                    pt_measure_type,
                    "retrying" if should_retry else "fatal",
                )
                if not should_retry or attempt >= attempts - 1:
                    raise TrackingClientError(
                        f"stats.nba.com returned {status_code} for "
                        f"{pt_measure_type} on {game_date.isoformat()}"
                    ) from exc
            except httpx.RequestError as exc:
                LOGGER.warning(
                    "[tracking] request error for %s %s: %s",
                    game_date.isoformat(),
                    pt_measure_type,
                    exc,
                )
                if attempt >= attempts - 1:
                    raise TrackingClientError(
                        f"request failed for {pt_measure_type} on "
                        f"{game_date.isoformat()}"
                    ) from exc
            if retry_delay > 0:
                time.sleep(retry_delay)
        raise TrackingClientError(
            f"exhausted retries for {pt_measure_type} on {game_date.isoformat()}"
        )
    finally:
        if close_session:
            session.close()
