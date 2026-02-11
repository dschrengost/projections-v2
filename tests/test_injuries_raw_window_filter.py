from __future__ import annotations

import pandas as pd

from projections.etl.injuries import _build_injuries_raw


class _FakeTeamResolver:
    def lookup_game_id(
        self,
        game_date: str,
        away_tricode: str,
        home_tricode: str,
        *,
        tip_ts: pd.Timestamp | None = None,
    ) -> int:
        if game_date == "2025-12-23":
            return 22500700
        return 22500001

    def resolve_team_id(self, name: str | None) -> int:
        return 1610612737


class _FakePlayerResolver:
    def resolve(self, name: str | None) -> int:
        return 123456


def test_build_injuries_raw_filters_out_of_window_game_dates() -> None:
    records = [
        {
            "report_time": "2025-12-23T17:30:00Z",
            "matchup": "ATL @ BOS",
            "game_date": "2025-11-30",
            "team": "Atlanta Hawks",
            "player_name": "Player Old",
            "current_status": "Out",
            "reason": "Injury",
            "report_url": "https://example.com/old.pdf",
        },
        {
            "report_time": "2025-12-23T17:30:00Z",
            "matchup": "ATL @ BOS",
            "game_date": "2025-12-23",
            "team": "Atlanta Hawks",
            "player_name": "Player Current",
            "current_status": "Out",
            "reason": "Injury",
            "report_url": "https://example.com/current.pdf",
        },
    ]

    result = _build_injuries_raw(
        records,
        start=pd.Timestamp("2025-12-23"),
        end=pd.Timestamp("2025-12-23"),
        resolver=_FakeTeamResolver(),
        player_resolver=_FakePlayerResolver(),
    )

    assert len(result) == 1
    assert int(result.iloc[0]["game_id"]) == 22500700
    assert str(result.iloc[0]["player_name"]) == "Player Current"


def test_build_injuries_raw_uses_report_time_for_asof_when_scraped_late() -> None:
    records = [
        {
            "report_time": "2026-02-10T22:00:00Z",
            "matchup": "ATL @ BOS",
            "game_date": "2026-02-10",
            "team": "Atlanta Hawks",
            "player_name": "Player Current",
            "current_status": "Out",
            "reason": "Injury",
            "report_url": "https://example.com/current.pdf",
        },
    ]

    result = _build_injuries_raw(
        records,
        start=pd.Timestamp("2026-02-10"),
        end=pd.Timestamp("2026-02-10"),
        resolver=_FakeTeamResolver(),
        player_resolver=_FakePlayerResolver(),
    )

    assert len(result) == 1
    row = result.iloc[0]
    assert pd.Timestamp(row["report_ts"]) == pd.Timestamp("2026-02-10T22:00:00Z")
    assert pd.Timestamp(row["as_of_ts"]) == pd.Timestamp("2026-02-10T22:00:00Z")
