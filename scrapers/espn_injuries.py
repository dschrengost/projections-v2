"""ESPN real-time injury scraper.

Fetches injury data from ESPN's game summary API, which updates faster than
the official NBA injury report PDFs. This catches late scratches like 
Donovan Mitchell that NBA.com misses.

API endpoints:
- Scoreboard: https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard
- Game summary: https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={eventId}
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
ESPN_SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"

# Map ESPN status to our canonical status
ESPN_STATUS_MAP = {
    "out": "OUT",
    "doubtful": "DOUBTFUL",
    "questionable": "Q",
    "probable": "PROB",
    "day-to-day": "Q",
}

# ESPN team ID to NBA team ID mapping
ESPN_TO_NBA_TEAM_ID = {
    "1": 1610612737,   # ATL
    "2": 1610612738,   # BOS
    "3": 1610612751,   # BKN
    "4": 1610612766,   # CHA
    "5": 1610612741,   # CHI
    "6": 1610612739,   # CLE
    "7": 1610612742,   # DAL
    "8": 1610612743,   # DEN
    "9": 1610612765,   # DET
    "10": 1610612744,  # GSW
    "11": 1610612745,  # HOU
    "12": 1610612754,  # IND
    "13": 1610612746,  # LAC
    "14": 1610612747,  # LAL
    "15": 1610612763,  # MEM
    "16": 1610612748,  # MIA
    "17": 1610612749,  # MIL
    "18": 1610612750,  # MIN
    "19": 1610612740,  # NOP
    "20": 1610612752,  # NYK
    "21": 1610612760,  # OKC
    "22": 1610612753,  # ORL
    "23": 1610612755,  # PHI
    "24": 1610612756,  # PHX
    "25": 1610612757,  # POR
    "26": 1610612758,  # SAC
    "27": 1610612759,  # SAS
    "28": 1610612761,  # TOR
    "29": 1610612762,  # UTA
    "30": 1610612764,  # WAS
}


def _fetch_json(url: str, params: dict | None = None, timeout: float = 10.0) -> dict:
    """Fetch JSON from ESPN API."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    }
    response = requests.get(url, params=params, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.json()


def fetch_todays_games() -> list[dict[str, Any]]:
    """Fetch today's games from ESPN scoreboard."""
    data = _fetch_json(ESPN_SCOREBOARD_URL)
    games = []
    for event in data.get("events", []):
        game_id = event.get("id")
        short_name = event.get("shortName", "")
        competitions = event.get("competitions", [])
        if not competitions:
            continue
        comp = competitions[0]
        home_team = None
        away_team = None
        for team in comp.get("competitors", []):
            if team.get("homeAway") == "home":
                home_team = team.get("team", {})
            else:
                away_team = team.get("team", {})
        games.append({
            "espn_event_id": game_id,
            "short_name": short_name,
            "home_team_id": home_team.get("id") if home_team else None,
            "home_team_abbr": home_team.get("abbreviation") if home_team else None,
            "away_team_id": away_team.get("id") if away_team else None,
            "away_team_abbr": away_team.get("abbreviation") if away_team else None,
        })
    return games


def fetch_game_injuries(espn_event_id: str) -> list[dict[str, Any]]:
    """Fetch injuries for a specific game from ESPN summary API."""
    data = _fetch_json(ESPN_SUMMARY_URL, params={"event": espn_event_id})
    injuries_list = []
    
    for team_injuries in data.get("injuries", []):
        team_info = team_injuries.get("team", {})
        espn_team_id = team_info.get("id")
        team_abbr = team_info.get("abbreviation")
        nba_team_id = ESPN_TO_NBA_TEAM_ID.get(espn_team_id)
        
        for injury in team_injuries.get("injuries", []):
            athlete = injury.get("athlete", {})
            status_raw = injury.get("status", "").lower()
            status = ESPN_STATUS_MAP.get(status_raw, status_raw.upper())
            
            injuries_list.append({
                "espn_player_id": athlete.get("id"),
                "player_name": athlete.get("displayName"),
                "espn_team_id": espn_team_id,
                "team_abbreviation": team_abbr,
                "nba_team_id": nba_team_id,
                "status": status,
                "status_raw": injury.get("status"),
                "injury_type": injury.get("type", {}).get("description"),
                "espn_event_id": espn_event_id,
            })
    
    return injuries_list


def scrape_espn_injuries(game_date: date | None = None) -> pd.DataFrame:
    """
    Scrape injuries for all of today's games from ESPN.
    
    Returns DataFrame with columns:
    - player_name
    - nba_team_id
    - team_abbreviation
    - status (OUT, Q, PROB, etc.)
    - status_raw
    - injury_type
    - as_of_ts
    - source
    """
    if game_date is None:
        game_date = date.today()
    
    as_of_ts = datetime.now(timezone.utc)
    
    logger.info(f"[espn-injuries] Fetching games for {game_date}")
    games = fetch_todays_games()
    logger.info(f"[espn-injuries] Found {len(games)} games")
    
    all_injuries = []
    for game in games:
        event_id = game["espn_event_id"]
        try:
            injuries = fetch_game_injuries(event_id)
            all_injuries.extend(injuries)
            if injuries:
                logger.info(
                    f"[espn-injuries] {game['short_name']}: {len(injuries)} injuries"
                )
        except Exception as e:
            logger.warning(f"[espn-injuries] Failed to fetch {event_id}: {e}")
    
    if not all_injuries:
        logger.info("[espn-injuries] No injuries found")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_injuries)
    df["as_of_ts"] = as_of_ts
    df["source"] = "espn"
    df["game_date"] = game_date
    
    # Log OUT players prominently
    out_players = df[df["status"] == "OUT"]
    if not out_players.empty:
        logger.info(f"[espn-injuries] OUT players: {out_players['player_name'].tolist()}")
    
    return df


def main(
    game_date: date | None = None,
    output_path: str | None = None,
) -> pd.DataFrame:
    """Main entry point for CLI usage."""
    df = scrape_espn_injuries(game_date)
    
    if output_path and not df.empty:
        df.to_parquet(output_path, index=False)
        logger.info(f"[espn-injuries] Wrote {len(df)} rows to {output_path}")
    
    return df


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    
    game_date = date.today()
    if len(sys.argv) > 1:
        game_date = date.fromisoformat(sys.argv[1])
    
    df = main(game_date)
    if not df.empty:
        print(df[["player_name", "team_abbreviation", "status", "injury_type"]].to_string())
