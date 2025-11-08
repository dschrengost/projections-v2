from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from typer.testing import CliRunner

from projections import scrape
from scrapers.oddstrader import EventOdds, MarketLine

runner = CliRunner()


def _sample_event() -> EventOdds:
    line = MarketLine(
        market="spread",
        selection="home",
        price=-110,
        point=-4.5,
        book="FanDuel",
        updated_at=datetime(2022, 1, 15, tzinfo=timezone.utc),
    )
    return EventOdds(
        event_id=1,
        scheduled=datetime(2022, 1, 15, 1, tzinfo=timezone.utc),
        home_team="Home",
        away_team="Away",
        markets={"spread": {"home": line}},
    )


def test_cli_writes_json(tmp_path: Path, monkeypatch) -> None:
    called: list[date] = []

    class FakeScraper:
        def fetch_daily_odds(self, target_date):
            called.append(target_date)
            return [_sample_event()]

    monkeypatch.setattr(scrape, "OddstraderScraper", lambda: FakeScraper())

    output = tmp_path / "odds.json"
    result = runner.invoke(
        scrape.app,
        [
            "--start",
            "2022-01-15",
            "--end",
            "2022-01-16",
            "--out",
            str(output),
            "--pretty",
        ],
    )

    assert result.exit_code == 0
    assert [c.isoformat() for c in called] == ["2022-01-15", "2022-01-16"]
    data = json.loads(output.read_text())
    assert data[0]["home_team"] == "Home"
    assert data[0]["markets"]["spread"]["home"]["book"] == "FanDuel"


def test_cli_rejects_invalid_range() -> None:
    result = runner.invoke(
        scrape.app,
        [
            "--start",
            "2022-01-17",
            "--end",
            "2022-01-16",
        ],
    )
    assert result.exit_code != 0
    assert "greater than or equal" in result.stderr
