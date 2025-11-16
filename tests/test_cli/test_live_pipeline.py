from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from projections.cli import live_pipeline


class StageRecorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def record(self, name: str):
        def _inner(**kwargs: Any) -> None:
            self.calls.append((name, kwargs))

        return _inner


def test_live_pipeline_invokes_enabled_stages(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runner = CliRunner()
    recorder = StageRecorder()

    monkeypatch.setattr(live_pipeline.injuries_etl, "main", recorder.record("injuries"))
    monkeypatch.setattr(live_pipeline.daily_lineups_etl, "run", recorder.record("lineups"))
    monkeypatch.setattr(live_pipeline.odds_etl, "main", recorder.record("odds"))
    monkeypatch.setattr(live_pipeline.roster_etl, "main", recorder.record("roster"))

    result = runner.invoke(
        live_pipeline.app,
        [
            "--start",
            "2025-11-16",
            "--end",
            "2025-11-16",
            "--season",
            "2025",
            "--month",
            "11",
            f"--data-root={tmp_path}",
        ],
    )

    assert result.exit_code == 0, result.output
    stage_names = [name for name, _ in recorder.calls]
    assert stage_names == ["injuries", "lineups", "odds", "roster"]


def test_live_pipeline_skips_disabled_stages(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runner = CliRunner()
    recorder = StageRecorder()

    monkeypatch.setattr(live_pipeline.injuries_etl, "main", recorder.record("injuries"))
    monkeypatch.setattr(live_pipeline.daily_lineups_etl, "run", recorder.record("lineups"))
    monkeypatch.setattr(live_pipeline.odds_etl, "main", recorder.record("odds"))
    monkeypatch.setattr(live_pipeline.roster_etl, "main", recorder.record("roster"))

    result = runner.invoke(
        live_pipeline.app,
        [
            "--start",
            "2025-11-16",
            "--end",
            "2025-11-16",
            "--season",
            "2025",
            "--month",
            "11",
            f"--data-root={tmp_path}",
            "--skip-injuries",
            "--skip-lineups",
            "--skip-odds",
            "--skip-roster",
        ],
    )

    assert result.exit_code == 0, result.output
    assert not recorder.calls
