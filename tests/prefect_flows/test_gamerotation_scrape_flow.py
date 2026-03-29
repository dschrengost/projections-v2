from __future__ import annotations

import logging
from datetime import datetime as real_datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def test_run_gamerotation_scrape_passes_guardrail_flags(monkeypatch, tmp_path: Path) -> None:
    import prefect_flows.gamerotation_scrape as flow_mod

    captured: dict[str, object] = {}

    def _fake_run(cmd, **kwargs):  # noqa: ANN001, ANN003
        captured["cmd"] = cmd

        class _Result:
            stdout = ""
            stderr = ""
            returncode = 0

        return _Result()

    monkeypatch.setattr(flow_mod.subprocess, "run", _fake_run)

    flow_mod._run_gamerotation_scrape(
        start_date="2026-03-20",
        end_date="2026-03-20",
        data_root=tmp_path,
        overwrite=False,
        timeout_s=20.0,
        max_failure_rate=0.25,
        min_success_coverage=1.0,
        subprocess_timeout_s=120.0,
    )

    cmd = [str(x) for x in captured["cmd"]]  # type: ignore[index]
    assert "--max-failure-rate" in cmd
    assert "0.25" in cmd
    assert "--min-success-coverage" in cmd
    assert "1.0" in cmd


def test_gamerotation_flow_defaults_include_full_coverage_guard(monkeypatch) -> None:
    import prefect_flows.gamerotation_scrape as flow_mod

    captured: dict[str, object] = {}

    def _fake_task(**kwargs):  # noqa: ANN003
        captured.update(kwargs)
        return {"start_date": "2026-03-20", "end_date": "2026-03-20"}

    monkeypatch.setattr(flow_mod, "gamerotation_scrape_task", _fake_task)

    result = flow_mod.gamerotation_scrape_flow.fn(game_date="2026-03-20")

    assert result["start_date"] == "2026-03-20"
    assert result["end_date"] == "2026-03-20"
    assert captured["lookback_days"] == 14
    assert captured["min_success_coverage"] == 1.0


def test_gamerotation_task_defaults_to_rolling_lookback_window(monkeypatch, tmp_path: Path) -> None:
    import prefect_flows.gamerotation_scrape as flow_mod

    class _FakeDateTime:
        @classmethod
        def now(cls, tz=None):  # noqa: ANN001
            base = real_datetime(2026, 3, 25, 12, 0, tzinfo=ZoneInfo("America/New_York"))
            return base if tz is None else base.astimezone(tz)

    captured: dict[str, object] = {}

    def _fake_run(**kwargs):  # noqa: ANN003
        captured.update(kwargs)

    monkeypatch.setattr(flow_mod, "datetime", _FakeDateTime)
    monkeypatch.setattr(flow_mod, "_run_gamerotation_scrape", _fake_run)
    monkeypatch.setattr(flow_mod, "get_run_logger", lambda: logging.getLogger("gamerotation-test"))
    monkeypatch.setattr(flow_mod.paths, "get_data_root", lambda: tmp_path)

    result = flow_mod.gamerotation_scrape_task.fn(
        game_date=None,
        start_date=None,
        end_date=None,
        overwrite=False,
        lookback_days=14,
        timeout_s=20.0,
        max_failure_rate=0.5,
        min_success_coverage=1.0,
        subprocess_timeout_s=900.0,
    )

    assert result["start_date"] == "2026-03-11"
    assert result["end_date"] == "2026-03-24"
    assert captured["start_date"] == "2026-03-11"
    assert captured["end_date"] == "2026-03-24"
