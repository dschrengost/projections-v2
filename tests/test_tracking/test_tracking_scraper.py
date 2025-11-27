from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from projections.data.nba import tracking
from scripts.tracking import scrape_tracking_raw

runner = CliRunner()


def _sample_payload(player_id: int = 100) -> dict:
    return {
        "resultSets": [
            {
                "name": "LeagueDashPtStats",
                "headers": [
                    "PLAYER_ID",
                    "PLAYER_NAME",
                    "TEAM_ID",
                    "TEAM_ABBREVIATION",
                    "GP",
                    "W",
                    "L",
                    "MIN",
                    "PTS_PER_TOUCH",
                    "TOUCHES",
                ],
                "rowSet": [
                    [
                        player_id,
                        f"Player {player_id}",
                        1610612749,
                        "MIL",
                        1,
                        1,
                        0,
                        30,
                        0.5,
                        12,
                    ]
                ],
            }
        ]
    }


def test_normalize_tracking_df_basic() -> None:
    frame = tracking.normalize_tracking_df(
        _sample_payload(),
        season="2024-25",
        season_type="Regular Season",
        game_date=date(2024, 10, 21),
        pt_measure_type="Possessions",
    )
    assert list(frame.columns)[:8] == [
        "season",
        "season_type",
        "game_date",
        "pt_measure_type",
        "player_id",
        "player_name",
        "team_id",
        "team_abbreviation",
    ]
    assert frame.loc[0, "player_id"] == 100
    assert frame.loc[0, "player_name"] == "Player 100"
    assert frame.loc[0, "pts_per_touch"] == 0.5
    assert frame.loc[0, "season"] == "2024-25"
    assert frame.loc[0, "pt_measure_type"] == "Possessions"


def test_write_tracking_partition_dedup(tmp_path: Path) -> None:
    base_frame = tracking.normalize_tracking_df(
        _sample_payload(1),
        season="2024-25",
        season_type="Regular Season",
        game_date=date(2024, 10, 21),
        pt_measure_type="Possessions",
    )
    result = tracking.write_tracking_partition(
        base_frame,
        data_root=tmp_path,
        season="2024-25",
        season_type="Regular Season",
        game_date=date(2024, 10, 21),
        pt_measure_type="Possessions",
    )
    assert result.rows_written == 1
    stored = pd.read_parquet(result.path)
    assert len(stored) == 1
    extra = pd.concat([base_frame, base_frame], ignore_index=True)
    extra.loc[1, "player_id"] = 2
    extra.loc[1, "player_name"] = "Player 2"
    dedup_result = tracking.write_tracking_partition(
        extra,
        data_root=tmp_path,
        season="2024-25",
        season_type="Regular Season",
        game_date=date(2024, 10, 21),
        pt_measure_type="Possessions",
    )
    assert dedup_result.rows_written == 1
    stored = pd.read_parquet(result.path)
    assert len(stored) == 2
    assert set(stored["player_id"]) == {1, 2}


def test_cli_backfill_writes_partitions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, date]] = []

    def fake_fetch(**kwargs):
        calls.append((kwargs["pt_measure_type"], kwargs["game_date"]))
        return _sample_payload(player_id=len(calls))

    monkeypatch.setattr(
        scrape_tracking_raw.tracking_client,
        "fetch_leaguedashptstats",
        fake_fetch,
    )
    result = runner.invoke(
        scrape_tracking_raw.app,
        [
            "backfill",
            "--season",
            "2024-25",
            "--start-date",
            "2024-10-21",
            "--end-date",
            "2024-10-22",
            "--pt-measure-type",
            "Possessions",
            "--pt-measure-type",
            "Passing",
            "--data-root",
            str(tmp_path),
            "--sleep-seconds",
            "0",
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(calls) == 4  # 2 dates x 2 measure types
    base = (
        tmp_path
        / "bronze"
        / "nba"
        / "tracking"
        / "season=2024-25"
        / "season_type=Regular+Season"
    )
    first_partition = (
        base / "game_date=2024-10-21" / "pt_measure_type=Possessions" / "part-00000.parquet"
    )
    second_partition = (
        base / "game_date=2024-10-22" / "pt_measure_type=Passing" / "part-00000.parquet"
    )
    assert first_partition.exists()
    assert second_partition.exists()
    df = pd.read_parquet(first_partition)
    assert not df.empty
    assert df.loc[0, "player_name"].startswith("Player")

