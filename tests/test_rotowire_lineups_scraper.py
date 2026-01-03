from __future__ import annotations

from scrapers.rotowire_lineups import (
    LINEUP_ROLE_CONFIRMED,
    LINEUP_ROLE_OUT,
    RotowireLineupsScraper,
)


def test_rotowire_lineups_parser_handles_current_markup() -> None:
    html = """
    <html><body>
      <div class="lineup is-nba" data-lnum="1">
        <div class="lineup__meta flex-row"><div class="lineup__time">8:00 PM ET</div></div>
        <div class="lineup__box">
          <div class="lineup__top">
            <div class="lineup__teams">
              <a class="lineup__team is-visit"><div class="lineup__abbr">CHA</div></a>
              <a class="lineup__team is-home"><div class="lineup__abbr">MIL</div></a>
            </div>
          </div>

          <ul class="lineup__list is-visit">
            <li class="lineup__status is-confirmed">Confirmed Lineup</li>
            <li class="lineup__player">
              <div class="lineup__pos">G</div>
              <a title="LaMelo Ball" href="/basketball/player/lamelo-ball-0000">L. Ball</a>
            </li>
            <li class="lineup__player">
              <div class="lineup__pos">F</div>
              <a title="Miles Bridges" href="/basketball/player/miles-bridges-0001">M. Bridges</a>
            </li>
          </ul>

          <ul class="lineup__list is-home">
            <li class="lineup__status is-confirmed">Confirmed Lineup</li>
            <li class="lineup__player">
              <div class="lineup__pos">F</div>
              <a title="Giannis Antetokounmpo" href="/basketball/player/giannis-antetokounmpo-0002">G. Antetokounmpo</a>
            </li>
            <li class="lineup__player is-pct-play-0 has-injury-status" title="Very Unlikely To Play">
              <div class="lineup__pos">G</div>
              <a title="Gary Harris" href="/basketball/player/gary-harris-0003">G. Harris</a>
              <span class="lineup__inj">Out</span>
            </li>
          </ul>
        </div>
      </div>
    </body></html>
    """
    scraper = RotowireLineupsScraper()
    games = scraper.parse_lineups(html)

    assert len(games) == 1
    game = games[0]
    assert game.away_team == "CHA"
    assert game.home_team == "MIL"
    assert game.is_confirmed is True

    assert len(game.away_players) == 2
    assert len(game.home_players) == 2

    assert all(player.lineup_role == LINEUP_ROLE_CONFIRMED for player in game.away_players)
    assert game.home_players[0].lineup_role == LINEUP_ROLE_CONFIRMED
    assert game.home_players[1].lineup_role == LINEUP_ROLE_OUT

