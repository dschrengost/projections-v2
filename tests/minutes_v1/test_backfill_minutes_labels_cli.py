from __future__ import annotations

from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from projections.cli import backfill_minutes_labels


def _season_from_day(day: pd.Timestamp) -> int:
    return day.year if day.month >= 8 else day.year - 1


def test_backfill_cli_respects_overwrite(monkeypatch, tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    runner = CliRunner()
    calls: dict[str, list[str]] = {"boxscores": [], "labels": []}

    def fake_boxscore_main(*, start, end, season, schedule, data_root, timeout=10.0):
        del end, schedule, timeout  # unused in fake
        day = pd.Timestamp(start).normalize()
        target = data_root / "bronze" / "boxscores_raw" / f"season={season}" / f"date={day.date().isoformat()}"
        target.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            {
                "game_id": [123],
                "payload": ["{}"],
                "tip_ts": ["2024-10-10T00:00:00Z"],
            }
        )
        df.to_parquet(target / "boxscores_raw.parquet", index=False)
        calls["boxscores"].append(day.date().isoformat())

    def fake_labels_main(*, start_date, end_date, data_root):
        del end_date
        day = pd.Timestamp(start_date).normalize()
        season = _season_from_day(day)
        target = (
            data_root
            / "gold"
            / "labels_minutes_v1"
            / f"season={season}"
            / f"game_date={day.date().isoformat()}"
        )
        target.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(
            {
                "game_id": [123],
                "player_id": [456],
                "game_date": [day],
                "actual_minutes": [30.0],
            }
        )
        frame.to_parquet(target / "labels.parquet", index=False)
        calls["labels"].append(day.date().isoformat())

    monkeypatch.setattr(backfill_minutes_labels.boxscores_etl, "main", fake_boxscore_main)
    monkeypatch.setattr(backfill_minutes_labels.labels_cli, "main", fake_labels_main)

    result = runner.invoke(
        backfill_minutes_labels.app,
        [
            "--start-date",
            "2024-10-10",
            "--end-date",
            "2024-10-11",
            "--data-root",
            str(data_root),
        ],
    )
    assert result.exit_code == 0, result.output
    assert calls == {
        "boxscores": ["2024-10-10", "2024-10-11"],
        "labels": ["2024-10-10", "2024-10-11"],
    }

    # Second run should skip both days because labels already exist.
    result_skip = runner.invoke(
        backfill_minutes_labels.app,
        [
            "--start-date",
            "2024-10-11",
            "--end-date",
            "2024-10-11",
            "--data-root",
            str(data_root),
        ],
    )
    assert result_skip.exit_code == 0, result_skip.output
    assert calls == {
        "boxscores": ["2024-10-10", "2024-10-11"],
        "labels": ["2024-10-10", "2024-10-11"],
    }

    # With overwrite we should rebuild the date and invoke both builders again.
    result_overwrite = runner.invoke(
        backfill_minutes_labels.app,
        [
            "--start-date",
            "2024-10-11",
            "--end-date",
            "2024-10-11",
            "--data-root",
            str(data_root),
            "--overwrite",
        ],
    )
    assert result_overwrite.exit_code == 0, result_overwrite.output
    assert calls == {
        "boxscores": ["2024-10-10", "2024-10-11", "2024-10-11"],
        "labels": ["2024-10-10", "2024-10-11", "2024-10-11"],
    }
