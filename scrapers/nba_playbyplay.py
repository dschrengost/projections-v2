"""Play-by-play scraper using nba_api v3 endpoint.

This module provides a scraper for NBA play-by-play data using the
nba_api library's PlayByPlayV3 endpoint. The data can be used for
building models that analyze game flow, player usage patterns, etc.

Example usage:
    >>> from scrapers.nba_playbyplay import NbaPlayByPlayScraper
    >>> scraper = NbaPlayByPlayScraper()
    >>> pbp_df = scraper.fetch_game_pbp("0022400001")  # Game ID
    >>> # Or fetch all games for a date
    >>> daily_pbp = scraper.fetch_daily_pbp(date(2025, 1, 2))
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional

import pandas as pd

from nba_api.stats.endpoints import playbyplayv3
from nba_api.stats.static import teams

from scrapers.nba_schedule import NbaScheduleScraper


@dataclass(frozen=True)
class PlayByPlayAction:
    """Single action from play-by-play data."""
    
    game_id: str
    action_number: int
    clock: str
    period: int
    team_id: Optional[int]
    team_tricode: Optional[str]
    person_id: Optional[int]
    player_name: Optional[str]
    player_name_i: Optional[str]
    x_legacy: Optional[float]
    y_legacy: Optional[float]
    shot_distance: Optional[float]
    shot_result: Optional[str]
    is_field_goal: bool
    score_home: str
    score_away: str
    points_total: Optional[int]
    location: Optional[str]
    description: str
    action_type: str
    sub_type: Optional[str]
    video_available: bool
    action_id: int


@dataclass
class GamePlayByPlay:
    """Complete play-by-play data for a single game."""
    
    game_id: str
    actions: List[PlayByPlayAction] = field(default_factory=list)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert actions to a pandas DataFrame."""
        if not self.actions:
            return pd.DataFrame()
        
        records = [
            {
                "game_id": a.game_id,
                "action_number": a.action_number,
                "clock": a.clock,
                "period": a.period,
                "team_id": a.team_id,
                "team_tricode": a.team_tricode,
                "person_id": a.person_id,
                "player_name": a.player_name,
                "player_name_i": a.player_name_i,
                "x_legacy": a.x_legacy,
                "y_legacy": a.y_legacy,
                "shot_distance": a.shot_distance,
                "shot_result": a.shot_result,
                "is_field_goal": a.is_field_goal,
                "score_home": a.score_home,
                "score_away": a.score_away,
                "points_total": a.points_total,
                "location": a.location,
                "description": a.description,
                "action_type": a.action_type,
                "sub_type": a.sub_type,
                "video_available": a.video_available,
                "action_id": a.action_id,
            }
            for a in self.actions
        ]
        return pd.DataFrame(records)


class NbaPlayByPlayScraper:
    """Scraper for NBA play-by-play data using nba_api's v3 endpoint.
    
    Uses the PlayByPlayV3 endpoint which provides detailed action-level
    data including shot locations, player information, and game flow.
    
    Attributes:
        request_delay: Delay between API requests to avoid rate limiting.
        timeout: Request timeout in seconds.
        schedule_scraper: Optional schedule scraper for daily lookups.
    """
    
    def __init__(
        self,
        *,
        request_delay: float = 0.6,
        timeout: float = 30.0,
        schedule_scraper: NbaScheduleScraper | None = None,
    ) -> None:
        """Initialize the play-by-play scraper.
        
        Args:
            request_delay: Seconds to wait between API requests.
            timeout: Request timeout in seconds.
            schedule_scraper: Scraper for fetching daily schedules.
        """
        self.request_delay = request_delay
        self.timeout = timeout
        self.schedule_scraper = schedule_scraper or NbaScheduleScraper(
            timeout=timeout
        )
    
    def fetch_game_pbp(
        self,
        game_id: str,
        *,
        start_period: int = 1,
        end_period: int = 10,
    ) -> GamePlayByPlay:
        """Fetch play-by-play data for a single game.
        
        Args:
            game_id: NBA game ID (10-digit string, e.g., "0022400001").
            start_period: First period to include (default 1).
            end_period: Last period to include (default 10 for OT games).
            
        Returns:
            GamePlayByPlay object containing all actions.
            
        Raises:
            RuntimeError: If the API request fails.
        """
        normalized_gid = str(game_id).zfill(10)
        print(f"[pbp] Fetching play-by-play for game {normalized_gid}")
        
        try:
            pbp = playbyplayv3.PlayByPlayV3(
                game_id=normalized_gid,
                start_period=start_period,
                end_period=end_period,
                timeout=self.timeout,
            )
            
            # Get the play-by-play data
            pbp_data = pbp.get_dict()
            
        except Exception as exc:
            raise RuntimeError(
                f"Failed to fetch play-by-play for game {normalized_gid}: {exc}"
            ) from exc
        
        actions = self._parse_actions(normalized_gid, pbp_data)
        return GamePlayByPlay(game_id=normalized_gid, actions=actions)
    
    def fetch_daily_pbp(
        self,
        target_date: date,
        *,
        season: str | None = None,
        completed_only: bool = True,
    ) -> List[GamePlayByPlay]:
        """Fetch play-by-play data for all games on a given date.
        
        Args:
            target_date: Date to fetch games for.
            season: Optional season string (e.g., "2024-25").
            completed_only: If True, only fetch completed games.
            
        Returns:
            List of GamePlayByPlay objects for each game.
        """
        scheduled_games = self.schedule_scraper.fetch_daily_schedule(
            target_date, season=season
        )
        
        results: List[GamePlayByPlay] = []
        for game in scheduled_games:
            if not game.game_id:
                continue
            
            # Skip non-completed games if requested
            if completed_only and game.game_status != 3:  # 3 = Final
                print(f"[pbp] Skipping non-final game {game.game_id} (status: {game.game_status})")
                continue
            
            try:
                pbp = self.fetch_game_pbp(game.game_id)
                results.append(pbp)
            except RuntimeError as exc:
                print(f"[pbp] Warning: {exc}; skipping {game.game_id}")
                continue
            
            # Delay between requests
            time.sleep(self.request_delay)
        
        return results
    
    def fetch_daily_pbp_dataframe(
        self,
        target_date: date,
        *,
        season: str | None = None,
        completed_only: bool = True,
    ) -> pd.DataFrame:
        """Fetch play-by-play data for all games and return as DataFrame.
        
        Convenience method that combines all games into a single DataFrame.
        
        Args:
            target_date: Date to fetch games for.
            season: Optional season string.
            completed_only: If True, only fetch completed games.
            
        Returns:
            Combined DataFrame with all play-by-play actions.
        """
        games = self.fetch_daily_pbp(
            target_date,
            season=season,
            completed_only=completed_only,
        )
        
        if not games:
            return pd.DataFrame()
        
        dfs = [game.to_dataframe() for game in games]
        return pd.concat(dfs, ignore_index=True)
    
    def _parse_actions(
        self, game_id: str, data: Dict[str, Any]
    ) -> List[PlayByPlayAction]:
        """Parse action data from the API response."""
        # The v3 endpoint returns data under game.actions
        game_data = data.get("game", {})
        play_by_play_data = game_data.get("actions", [])
        
        actions: List[PlayByPlayAction] = []
        for row in play_by_play_data:
            action = PlayByPlayAction(
                game_id=game_id,
                action_number=row.get("actionNumber", 0),
                clock=row.get("clock", ""),
                period=row.get("period", 0),
                team_id=row.get("teamId"),
                team_tricode=row.get("teamTricode"),
                person_id=row.get("personId"),
                player_name=row.get("playerName"),
                player_name_i=row.get("playerNameI"),
                x_legacy=row.get("xLegacy"),
                y_legacy=row.get("yLegacy"),
                shot_distance=row.get("shotDistance"),
                shot_result=row.get("shotResult"),
                is_field_goal=bool(row.get("isFieldGoal")),
                score_home=str(row.get("scoreHome", "")),
                score_away=str(row.get("scoreAway", "")),
                points_total=row.get("pointsTotal"),
                location=row.get("location"),
                description=row.get("description", ""),
                action_type=row.get("actionType", ""),
                sub_type=row.get("subType"),
                video_available=bool(row.get("videoAvailable")),
                action_id=row.get("actionId", 0),
            )
            actions.append(action)
        
        return actions


__all__ = [
    "GamePlayByPlay",
    "NbaPlayByPlayScraper",
    "PlayByPlayAction",
]
