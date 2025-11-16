from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from click.core import ParameterSource

from projections.cli.score_minutes_v1 import _apply_scoring_overrides
from projections.models.minutes_lgbm import _apply_training_overrides
from projections.minutes_v1.config import load_scoring_config, load_training_config


class _DummyCtx:
    """Minimal stand-in for typer.Context in tests."""

    def __init__(self, overrides: dict[str, ParameterSource] | None = None) -> None:
        self._overrides = overrides or {}

    def get_parameter_source(self, name: str) -> ParameterSource:
        return self._overrides.get(name, ParameterSource.DEFAULT)


def _write_yaml(path: Path, payload: str) -> Path:
    path.write_text(payload, encoding="utf-8")
    return path


def test_training_config_overrides_defaults(tmp_path) -> None:
    cfg_path = _write_yaml(
        tmp_path / "train.yaml",
        """
        run_id: yaml_run
        tolerance_relaxed: true
        train_start: 2025-01-01T00:00:00+00:00
        """,
    )
    cli_params = {
        "run_id": None,
        "tolerance_relaxed": False,
        "train_start": datetime(2024, 12, 1, tzinfo=timezone.utc),
    }
    merged = _apply_training_overrides(_DummyCtx(), cli_params, cfg_path)
    assert merged["run_id"] == "yaml_run"
    assert merged["tolerance_relaxed"] is True
    assert merged["train_start"] == datetime(2025, 1, 1, tzinfo=timezone.utc)


def test_training_config_respects_cli_overrides(tmp_path) -> None:
    cfg_path = _write_yaml(
        tmp_path / "train.yaml",
        """
        train_start: 2025-02-01T00:00:00+00:00
        """,
    )
    cli_params = {
        "train_start": datetime(2024, 12, 1, tzinfo=timezone.utc),
    }
    ctx = _DummyCtx({"train_start": ParameterSource.COMMANDLINE})
    merged = _apply_training_overrides(ctx, cli_params, cfg_path)
    assert merged["train_start"] == datetime(2024, 12, 1, tzinfo=timezone.utc)


def test_scoring_config_applies_overrides(tmp_path) -> None:
    cfg_path = _write_yaml(
        tmp_path / "score.yaml",
        """
        date: 2025-11-14T00:00:00+00:00
        end_date: 2025-11-15T00:00:00+00:00
        minutes_output: unconditional
        promotion_prior_enabled: false
        """,
    )
    cli_params = {
        "date": None,
        "end_date": None,
        "minutes_output": "both",
        "promotion_prior_enabled": True,
    }
    merged = _apply_scoring_overrides(_DummyCtx(), cli_params, cfg_path)
    assert merged["date"] == datetime(2025, 11, 14, tzinfo=timezone.utc)
    assert merged["end_date"] == datetime(2025, 11, 15, tzinfo=timezone.utc)
    assert merged["minutes_output"] == "unconditional"
    assert merged["promotion_prior_enabled"] is False


def test_scoring_config_respects_cli_flags(tmp_path) -> None:
    cfg_path = _write_yaml(
        tmp_path / "score.yaml",
        """
        minutes_output: conditional
        """,
    )
    cli_params = {
        "minutes_output": "unconditional",
    }
    ctx = _DummyCtx({"minutes_output": ParameterSource.COMMANDLINE})
    merged = _apply_scoring_overrides(ctx, cli_params, cfg_path)
    assert merged["minutes_output"] == "unconditional"


def test_load_training_config_supports_section(tmp_path) -> None:
    cfg_path = _write_yaml(
        tmp_path / "train.yaml",
        """
        minutes_training:
          run_id: nested
          train_start: 2024-12-01T00:00:00+00:00
        """,
    )
    cfg = load_training_config(cfg_path)
    assert cfg.run_id == "nested"
    assert cfg.train_start == datetime(2024, 12, 1, tzinfo=timezone.utc)


def test_load_scoring_config_supports_section(tmp_path) -> None:
    cfg_path = _write_yaml(
        tmp_path / "score.yaml",
        """
        minutes_scoring:
          date: 2025-11-14T00:00:00+00:00
        """,
    )
    cfg = load_scoring_config(cfg_path)
    assert cfg.date == datetime(2025, 11, 14, tzinfo=timezone.utc)
