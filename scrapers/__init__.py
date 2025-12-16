"""Scraper utilities for pulling external betting data."""

from .nba_boxscore import (
    NbaComArena,
    NbaComBoxScoreScraper,
    NbaComGameBoxScore,
    NbaComPlayerStatLine,
    NbaComTeamBoxScore,
)
from .nba_schedule import (
    NbaScheduleScraper,
    ScheduledGame,
    ScheduledTeam,
)
from .oddstrader import EventOdds, MarketLine, OddstraderScraper
from .nba_players import NbaPlayersScraper, PlayerProfile
from .nba_injuries import InjuryRecord, NBAInjuryScraper
from .nba_daily_lineups import (
    NbaDailyLineupsScraper,
    normalize_daily_lineups,
    LINEUP_ROLE_BENCH,
    LINEUP_ROLE_CONFIRMED,
    LINEUP_ROLE_OUT,
    LINEUP_ROLE_PROJECTED,
)

__all__ = [
    "EventOdds",
    "InjuryRecord",
    "NbaComArena",
    "NbaComBoxScoreScraper",
    "NbaComGameBoxScore",
    "NbaComPlayerStatLine",
    "NbaComTeamBoxScore",
    "NBAInjuryScraper",
    "MarketLine",
    "OddstraderScraper",
    "NbaScheduleScraper",
    "NbaPlayersScraper",
    "ScheduledGame",
    "ScheduledTeam",
    "PlayerProfile",
    "NbaDailyLineupsScraper",
    "normalize_daily_lineups",
    "LINEUP_ROLE_BENCH",
    "LINEUP_ROLE_CONFIRMED",
    "LINEUP_ROLE_OUT",
    "LINEUP_ROLE_PROJECTED",
]
