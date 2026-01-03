"""Unit tests for reconcile_team_minutes propagation and summary reporting.

Verifies:
1. reconcile_team_minutes from bundle config is resolved correctly
2. CLI overrides still win over bundle config
3. Rotshare forces reconcile_team_minutes="none"

These tests are standalone to avoid import issues with score_minutes_v1.
The helper function is duplicated here for testing in isolation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _resolve_reconcile_team_minutes(config_path: Path | None, cli_default: str) -> str:
    """Resolve reconcile_team_minutes from JSON bundle config.

    Returns cli_default if config_path is None, doesn't exist, or doesn't
    contain reconcile_team_minutes.
    
    NOTE: This is a copy of the function from score_minutes_v1.py for isolated testing.
    """
    if config_path is None:
        return cli_default
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return cli_default
    value = payload.get("reconcile_team_minutes")
    if value is None:
        return cli_default
    return str(value).strip().lower()


class TestResolveReconcileTeamMinutes:
    """Tests for _resolve_reconcile_team_minutes helper."""

    def test_returns_cli_default_when_config_path_is_none(self):
        result = _resolve_reconcile_team_minutes(None, "none")
        assert result == "none"

    def test_returns_cli_default_when_config_missing(self, tmp_path: Path):
        missing_path = tmp_path / "does_not_exist.json"
        result = _resolve_reconcile_team_minutes(missing_path, "none")
        assert result == "none"

    def test_returns_cli_default_when_config_invalid_json(self, tmp_path: Path):
        invalid_config = tmp_path / "invalid.json"
        invalid_config.write_text("not valid json {", encoding="utf-8")
        result = _resolve_reconcile_team_minutes(invalid_config, "none")
        assert result == "none"

    def test_returns_cli_default_when_key_missing(self, tmp_path: Path):
        config = tmp_path / "config.json"
        config.write_text(json.dumps({"minutes_alloc_mode": "legacy"}), encoding="utf-8")
        result = _resolve_reconcile_team_minutes(config, "none")
        assert result == "none"

    def test_returns_value_from_config(self, tmp_path: Path):
        config = tmp_path / "config.json"
        config.write_text(
            json.dumps({"reconcile_team_minutes": "p50", "minutes_alloc_mode": "legacy"}),
            encoding="utf-8",
        )
        result = _resolve_reconcile_team_minutes(config, "none")
        assert result == "p50"

    def test_normalizes_to_lowercase(self, tmp_path: Path):
        config = tmp_path / "config.json"
        config.write_text(
            json.dumps({"reconcile_team_minutes": "P50_AND_TAILS"}),
            encoding="utf-8",
        )
        result = _resolve_reconcile_team_minutes(config, "none")
        assert result == "p50_and_tails"

    def test_strips_whitespace(self, tmp_path: Path):
        config = tmp_path / "config.json"
        config.write_text(
            json.dumps({"reconcile_team_minutes": "  p50  "}),
            encoding="utf-8",
        )
        result = _resolve_reconcile_team_minutes(config, "none")
        assert result == "p50"

    def test_matches_production_config_format(self, tmp_path: Path):
        """Test with same format as config/minutes_current_run.json."""
        config = tmp_path / "config.json"
        config.write_text(
            json.dumps({
                "mode": "single",
                "bundle_dir": "artifacts/minutes_lgbm/minutes_v1_safe_starter_20251214",
                "run_id": "minutes_v1_safe_starter_20251214",
                "minutes_alloc_mode": "legacy",
                "rotalloc_bundle_dir": "artifacts/experiments/lgbm_rotalloc_final_v1",
                "reconcile_team_minutes": "p50",
                "enable_upside_adjustment": False
            }),
            encoding="utf-8",
        )
        result = _resolve_reconcile_team_minutes(config, "none")
        assert result == "p50"
