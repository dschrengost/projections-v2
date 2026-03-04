from __future__ import annotations

from scrapers.rotowire_lineups import (
    LINEUP_ROLE_CONFIRMED,
    LINEUP_ROLE_OUT,
    LINEUP_ROLE_PROJECTED,
    RotowireLineupsScraper,
    scrape_rotowire_lineups,
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


def test_scrape_rotowire_lineups_keeps_doubtful_players_as_out(monkeypatch) -> None:
    html = """
    <html><body>
      <div class="lineup is-nba" data-lnum="1">
        <div class="lineup__box">
          <div class="lineup__teams">
            <a class="lineup__team is-visit"><div class="lineup__abbr">OKC</div></a>
            <a class="lineup__team is-home"><div class="lineup__abbr">NYK</div></a>
          </div>
          <ul class="lineup__list is-visit">
            <li class="lineup__status is-expected">Expected Lineup</li>
            <li class="lineup__player"><div class="lineup__pos">PG</div><a title="Player 1">Player 1</a></li>
            <li class="lineup__player"><div class="lineup__pos">SG</div><a title="Player 2">Player 2</a></li>
            <li class="lineup__player"><div class="lineup__pos">SF</div><a title="Player 3">Player 3</a></li>
            <li class="lineup__player"><div class="lineup__pos">PF</div><a title="Player 4">Player 4</a></li>
            <li class="lineup__player"><div class="lineup__pos">C</div><a title="Player 5">Player 5</a></li>
            <li class="lineup__player is-pct-play-25 has-injury-status" title="Unlikely To Play">
              <div class="lineup__pos">G</div>
              <a title="Ajay Mitchell">A. Mitchell</a>
              <span class="lineup__inj">Doubt</span>
            </li>
          </ul>
          <ul class="lineup__list is-home">
            <li class="lineup__status is-expected">Expected Lineup</li>
            <li class="lineup__player"><div class="lineup__pos">PG</div><a title="Home 1">Home 1</a></li>
            <li class="lineup__player"><div class="lineup__pos">SG</div><a title="Home 2">Home 2</a></li>
            <li class="lineup__player"><div class="lineup__pos">SF</div><a title="Home 3">Home 3</a></li>
            <li class="lineup__player"><div class="lineup__pos">PF</div><a title="Home 4">Home 4</a></li>
            <li class="lineup__player"><div class="lineup__pos">C</div><a title="Home 5">Home 5</a></li>
          </ul>
        </div>
      </div>
    </body></html>
    """
    games = RotowireLineupsScraper().parse_lineups(html)

    def _fake_scrape(self):  # noqa: ANN001
        return games

    monkeypatch.setattr(RotowireLineupsScraper, "scrape", _fake_scrape)
    df = scrape_rotowire_lineups()

    okc = df.loc[df["team_abbreviation"] == "OKC"].copy()
    ajay = okc.loc[okc["player_name"] == "Ajay Mitchell"]
    assert len(ajay) == 1
    assert ajay.iloc[0]["lineup_role"] == LINEUP_ROLE_OUT

    starter_count = int(okc["lineup_role"].isin([LINEUP_ROLE_PROJECTED, LINEUP_ROLE_CONFIRMED]).sum())
    assert starter_count == 5


def test_rotowire_parser_marks_class_only_doubtful_rows_as_out() -> None:
    html = """
    <html><body>
      <div class="lineup is-nba" data-lnum="1">
        <div class="lineup__box">
          <div class="lineup__teams">
            <a class="lineup__team is-visit"><div class="lineup__abbr">OKC</div></a>
            <a class="lineup__team is-home"><div class="lineup__abbr">NYK</div></a>
          </div>
          <ul class="lineup__list is-visit">
            <li class="lineup__status is-expected">Expected Lineup</li>
            <li class="lineup__player is-doubtful">
              <div class="lineup__pos">G</div>
              <a title="Ajay Mitchell">A. Mitchell</a>
            </li>
          </ul>
          <ul class="lineup__list is-home">
            <li class="lineup__status is-expected">Expected Lineup</li>
            <li class="lineup__player"><div class="lineup__pos">PG</div><a title="Home 1">Home 1</a></li>
          </ul>
        </div>
      </div>
    </body></html>
    """
    games = RotowireLineupsScraper().parse_lineups(html)
    assert len(games) == 1
    assert len(games[0].away_players) == 1
    assert games[0].away_players[0].lineup_role == LINEUP_ROLE_OUT
