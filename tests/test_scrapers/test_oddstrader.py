from __future__ import annotations

import json
from datetime import date, datetime, timezone
from typing import Any, Dict

import pytest

from scrapers.oddstrader import EventOdds, OddstraderScraper


def _scoreboard_html() -> str:
    initial_state = {
        "events": {
            "events": {
                "4532388": {
                    "eid": 4532388,
                    "dt": 1642289400000,
                    "participants": {
                        "home": {
                            "partid": 1001,
                            "ih": True,
                            "source": {"nn": "Milwaukee Bucks", "abbr": "MIL"},
                        },
                        "away": {
                            "partid": 1002,
                            "ih": False,
                            "source": {"nn": "Toronto Raptors", "abbr": "TOR"},
                        },
                    },
                },
                "4532393": {
                    "eid": 4532393,
                    "dt": 1642293000000,
                    "participants": {
                        "home": {
                            "partid": 2001,
                            "ih": True,
                            "source": {"nn": "Washington Wizards", "abbr": "WAS"},
                        },
                        "away": {
                            "partid": 2002,
                            "ih": False,
                            "source": {"nn": "Portland Trail Blazers", "abbr": "POR"},
                        },
                    },
                },
            }
        }
    }
    config_state = {
        "apigateway": {
            "headers": {
                "Authorization": "JWT fake",
                "X-Access-Policy": "policy-token",
            }
        },
        "endpoints": {"oddsv2Service": "https://ms.example.com/odds-v2"},
    }
    return (
        "<html><body><script>"
        f"window.__INITIAL_STATE__={json.dumps(initial_state)};"
        f"window.__config={json.dumps(config_state)};"
        "</script></body></html>"
    )


def _lines_payload() -> Dict[str, Any]:
    return {
        "data": {
            "A_CL": [
                # Event 1 spreads (FanDuel)
                {
                    "eid": 4532388,
                    "mtid": 401,
                    "partid": 1001,
                    "paid": 98,
                    "ap": -110,
                    "adj": -7.5,
                    "tim": "1642268407000",
                },
                {
                    "eid": 4532388,
                    "mtid": 401,
                    "partid": 1002,
                    "paid": 98,
                    "ap": -110,
                    "adj": 7.5,
                    "tim": "1642268407000",
                },
                # Event 1 totals (FanDuel over, Caesars under)
                {
                    "eid": 4532388,
                    "mtid": 402,
                    "partid": 15143,
                    "paid": 98,
                    "ap": -115,
                    "adj": 218.5,
                    "tim": "1642268407000",
                },
                {
                    "eid": 4532388,
                    "mtid": 402,
                    "partid": 15144,
                    "paid": 101,
                    "ap": -110,
                    "adj": 218.5,
                    "tim": "1642268500000",
                },
                # Event 1 moneyline (FanDuel only)
                {
                    "eid": 4532388,
                    "mtid": 83,
                    "partid": 1001,
                    "paid": 98,
                    "ap": -320,
                    "adj": 0,
                    "tim": "1642268407000",
                },
                {
                    "eid": 4532388,
                    "mtid": 83,
                    "partid": 1002,
                    "paid": 98,
                    "ap": 260,
                    "adj": 0,
                    "tim": "1642268407000",
                },
                # Event 2 spreads (no FanDuel, fallback to Caesars)
                {
                    "eid": 4532393,
                    "mtid": 401,
                    "partid": 2001,
                    "paid": 101,
                    "ap": -112,
                    "adj": -4.5,
                    "tim": "1642291111000",
                },
                {
                    "eid": 4532393,
                    "mtid": 401,
                    "partid": 2002,
                    "paid": 101,
                    "ap": -108,
                    "adj": 4.5,
                    "tim": "1642291111000",
                },
            ]
        }
    }


def test_fetch_daily_odds_prefers_fanduel_and_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    scraper = OddstraderScraper()

    scoreboard = _scoreboard_html()
    monkeypatch.setattr(scraper, "_http_get", lambda url: scoreboard)

    captured: Dict[str, Any] = {}

    def fake_http_json(url: str, headers: Dict[str, str]) -> Dict[str, Any]:
        captured["url"] = url
        captured["headers"] = headers
        return _lines_payload()

    monkeypatch.setattr(scraper, "_http_json", fake_http_json)

    results = scraper.fetch_daily_odds(date(2022, 1, 15))

    assert len(results) == 2
    first = next(ev for ev in results if ev.event_id == 4532388)
    second = next(ev for ev in results if ev.event_id == 4532393)

    assert first.markets["spread"]["home"].book == "FanDuel"
    assert first.markets["spread"]["home"].point == -7.5
    assert first.markets["total"]["over"].book == "FanDuel"
    assert first.markets["total"]["under"].book == "Caesars"
    assert first.markets["moneyline"]["home"].price == -320

    assert second.markets["spread"]["home"].book == "Caesars"
    assert second.markets["spread"]["away"].book == "Caesars"

    assert "eid:[4532388,4532393]" in captured["url"]
    assert captured["url"].startswith("https://ms.example.com/odds-v2/odds-v2-service")


def test_fetch_range_calls_each_day(monkeypatch: pytest.MonkeyPatch) -> None:
    scraper = OddstraderScraper()
    called = []

    def fake_scrape(day: date) -> list[EventOdds]:
        called.append(day)
        return [
            EventOdds(
                event_id=int(day.strftime("%Y%m%d")),
                scheduled=datetime(day.year, day.month, day.day, tzinfo=timezone.utc),
                home_team="Home",
                away_team="Away",
                markets={},
            )
        ]

    monkeypatch.setattr(scraper, "_scrape_single_date", fake_scrape)

    results = scraper.fetch_range(date(2022, 1, 15), date(2022, 1, 17))

    assert [d.day for d in called] == [15, 16, 17]
    assert len(results) == 3
    assert results[0].event_id == 20220115

    with pytest.raises(ValueError):
        scraper.fetch_range(date(2022, 1, 18), date(2022, 1, 17))
