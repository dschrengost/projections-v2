from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from projections.etl import daily_lineups, injuries
from scrapers.nba_injuries import InjuryRecord


class DummyTeamResolver:
    def __init__(self, _schedule: pd.DataFrame) -> None:
        self.team_id = 1610612737

    def lookup_game_id(self, game_date: str, away: str | None, home: str | None) -> str:
        return "0001"

    def resolve_team_id(self, team: str | None) -> int:
        return self.team_id


class DummyPlayerResolver:
    def resolve(self, name: str | None) -> int | None:
        return 203500


def _injury_record() -> InjuryRecord:
    return InjuryRecord(
        report_time=datetime(2025, 11, 16, 14, 30, tzinfo=timezone.utc),
        report_url="https://example.com/report.pdf",
        game_date=datetime(2025, 11, 16).date(),
        game_time_et="7:00 PM ET",
        matchup="ATL @ BOS",
        team="Atlanta Hawks",
        player_name="Trae Young",
        current_status="Available",
        reason="",
    )


def test_injuries_etl_scrape_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeScraper:
        def __init__(self, timeout: float) -> None:
            self.timeout = timeout

        def __enter__(self) -> "FakeScraper":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - cleanup
            return None

        def fetch_range(self, start, end):
            return [_injury_record()]

    schedule_df = pd.DataFrame(
        {
            "game_id": ["0001"],
            "tip_ts": [pd.Timestamp("2025-11-17T00:00:00Z")],
        }
    )

    monkeypatch.setattr(injuries, "NBAInjuryScraper", FakeScraper)
    monkeypatch.setattr(injuries, "TeamResolver", DummyTeamResolver)
    monkeypatch.setattr(injuries, "_build_player_resolver", lambda timeout: DummyPlayerResolver())
    monkeypatch.setattr(
        injuries,
        "load_schedule_data",
        lambda *_args, **_kwargs: schedule_df,
    )
    monkeypatch.setattr(injuries, "enforce_schema", lambda df, *_args, **_kwargs: df)
    monkeypatch.setattr(injuries, "validate_with_pandera", lambda df, *_args, **_kwargs: df)
    monkeypatch.setattr(injuries, "select_injury_snapshot", lambda df: df)

    injuries.main(
        injuries_json=None,
        schedule=[],
        use_scraper=True,
        start=datetime(2025, 11, 16),
        end=datetime(2025, 11, 16),
        season=2025,
        month=11,
        data_root=tmp_path,
        bronze_root=None,
        bronze_out=None,
        silver_out=None,
        scraper_timeout=0.1,
        schedule_timeout=0.1,
        injury_timeout=0.1,
    )

    bronze_path = (
        tmp_path
        / "bronze"
        / "injuries_raw"
        / "season=2025"
        / "date=2025-11-16"
        / "injuries.parquet"
    )
    assert bronze_path.exists(), "Bronze parquet not written"
    bronze_df = pd.read_parquet(bronze_path)
    assert not bronze_df.empty
    assert set(bronze_df["player_name"]) == {"Trae Young"}

    silver_path = tmp_path / "silver" / "injuries_snapshot" / "season=2025" / "month=11"
    assert (silver_path / "injuries_snapshot.parquet").exists()


def test_daily_lineups_ingest_range(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {"games": [{"gameId": 123, "teams": []}]}

    class FakeLineupScraper:
        url_template = "https://example.test/{date}.json"

        def __init__(self, timeout: float) -> None:
            self.timeout = timeout

        def fetch_daily_lineups(self, target_date):
            return payload

    def fake_normalize(_payload, *, target_date, season_start, ingested_ts=None):
        return pd.DataFrame(
            [
                {
                    "game_id": 123,
                    "team_id": 1610612737,
                    "player_id": 203500,
                    "player_name": "Trae Young",
                    "date": pd.Timestamp(target_date),
                }
            ]
        )

    monkeypatch.setattr(daily_lineups, "NbaDailyLineupsScraper", FakeLineupScraper)
    monkeypatch.setattr(daily_lineups, "normalize_daily_lineups", fake_normalize)

    results = daily_lineups.ingest_range(
        start_day=pd.Timestamp("2025-11-16"),
        end_day=pd.Timestamp("2025-11-16"),
        season=2025,
        data_root=tmp_path,
        timeout=0.1,
        capture_payloads=True,
    )

    assert len(results) == 1
    bronze_path = (
        tmp_path
        / "bronze"
        / "daily_lineups"
        / "season=2025"
        / "date=2025-11-16"
        / "daily_lineups_raw.parquet"
    )
    assert bronze_path.exists()
    bronze_df = pd.read_parquet(bronze_path)
    assert bronze_df["season_start"].iloc[0] == 2025
    silver_path = (
        tmp_path
        / "silver"
        / "nba_daily_lineups"
        / "season=2025"
        / "date=2025-11-16"
        / "lineups.parquet"
    )
    assert silver_path.exists()
    silver_df = pd.read_parquet(silver_path)
    assert not silver_df.empty
    assert silver_df["game_id"].iloc[0] == 123
    assert results[0].payload == payload
