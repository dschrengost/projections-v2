"""Scraper utilities for pulling external betting data."""

from .basketball_reference import (
    BasketballReferenceScraper,
    GameBoxScore,
    PlayerBoxScore,
    TeamBoxScore,
)
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
    "BasketballReferenceScraper",
    "EventOdds",
    "InjuryRecord",
    "GameBoxScore",
    "NbaComArena",
    "NbaComBoxScoreScraper",
    "NbaComGameBoxScore",
    "NbaComPlayerStatLine",
    "NbaComTeamBoxScore",
    "NBAInjuryScraper",
    "MarketLine",
    "OddstraderScraper",
    "PlayerBoxScore",
    "NbaScheduleScraper",
    "NbaPlayersScraper",
    "TeamBoxScore",
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
