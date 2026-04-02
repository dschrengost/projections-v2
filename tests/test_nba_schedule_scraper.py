from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from scrapers.nba_schedule import NbaScheduleScraper, ScheduledGame, ScheduledTeam


def _team(team_id: int, tri: str, city: str, name: str) -> ScheduledTeam:
    return ScheduledTeam(
        team_id=team_id,
        team_name=name,
        team_city=city,
        team_tricode=tri,
        wins=None,
        losses=None,
        score=None,
    )


def _game(
    *,
    game_id: str,
    game_time_utc: datetime,
    local_game_date: date,
    home: ScheduledTeam,
    away: ScheduledTeam,
) -> ScheduledGame:
    return ScheduledGame(
        game_id=game_id,
        game_code="",
        status=1,
        status_text="Scheduled",
        game_label=None,
        game_sub_label=None,
        season_year="2025-26",
        game_time_utc=game_time_utc,
        is_neutral=False,
        series_game_number=None,
        game_subtype=None,
        week_number=None,
        arena_name=None,
        arena_city=None,
        arena_state=None,
        broadcasters={},
        home_team=home,
        away_team=away,
        local_game_date=local_game_date,
    )


def test_fetch_daily_schedule_prefers_scoreboard_for_explicit_today(monkeypatch: pytest.MonkeyPatch) -> None:
    scraper = NbaScheduleScraper()
    today = date(2026, 4, 1)

    scoreboard_game = _game(
        game_id="0022501003",
        game_time_utc=datetime(2026, 4, 2, 0, 0, tzinfo=timezone.utc),
        local_game_date=today,
        home=_team(1610612763, "MEM", "Memphis", "Grizzlies"),
        away=_team(1610612752, "NYK", "New York", "Knicks"),
    )
    stale_game = _game(
        game_id="0022501111",
        game_time_utc=datetime(2026, 4, 2, 0, 0, tzinfo=timezone.utc),
        local_game_date=today,
        home=_team(1610612763, "MEM", "Memphis", "Grizzlies"),
        away=_team(1610612742, "DAL", "Dallas", "Mavericks"),
    )

    monkeypatch.setattr(scraper, "_current_date", lambda: today)
    monkeypatch.setattr(scraper, "_fetch_scoreboard_games", lambda: [scoreboard_game])
    monkeypatch.setattr(scraper, "fetch_season_schedule", lambda season=None: [stale_game])

    out = scraper.fetch_daily_schedule(target_date=today)
    assert len(out) == 1
    assert out[0].game_id == "0022501003"
    assert out[0].away_team and out[0].away_team.team_tricode == "NYK"


def test_fetch_daily_schedule_non_today_uses_season_schedule(monkeypatch: pytest.MonkeyPatch) -> None:
    scraper = NbaScheduleScraper()
    today = date(2026, 4, 1)
    target_day = date(2026, 3, 31)

    season_game = _game(
        game_id="0022500999",
        game_time_utc=datetime(2026, 3, 31, 23, 0, tzinfo=timezone.utc),
        local_game_date=target_day,
        home=_team(1610612744, "GSW", "Golden State", "Warriors"),
        away=_team(1610612757, "POR", "Portland", "Trail Blazers"),
    )
    scoreboard_today = _game(
        game_id="0022501003",
        game_time_utc=datetime(2026, 4, 2, 0, 0, tzinfo=timezone.utc),
        local_game_date=today,
        home=_team(1610612763, "MEM", "Memphis", "Grizzlies"),
        away=_team(1610612752, "NYK", "New York", "Knicks"),
    )

    monkeypatch.setattr(scraper, "_current_date", lambda: today)
    monkeypatch.setattr(scraper, "_fetch_scoreboard_games", lambda: [scoreboard_today])
    monkeypatch.setattr(scraper, "fetch_season_schedule", lambda season=None: [season_game])

    out = scraper.fetch_daily_schedule(target_date=target_day)
    assert len(out) == 1
    assert out[0].game_id == "0022500999"
