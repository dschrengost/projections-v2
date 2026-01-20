"""Regression tests for CLI modules to ensure they load without BadParameter errors.

These tests verify that CLI modules using typer.BadParameter work correctly
after the migration from param_name= to param_hint=.
"""

import pytest


def test_score_minutes_v1_help_runs():
    """Verify score_minutes_v1 CLI loads and runs --help without crashing."""
    from typer.testing import CliRunner
    from projections.cli.score_minutes_v1 import app

    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Usage" in result.output or "usage" in result.output.lower()


def test_build_rotation_set_minutes_features_v1_help_runs():
    """Verify build_rotation_set_minutes_features_v1 CLI loads and runs --help without crashing."""
    from typer.testing import CliRunner
    from projections.cli.build_rotation_set_minutes_features_v1 import app

    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Usage" in result.output or "usage" in result.output.lower()


def test_build_starter_priors_help_runs():
    """Verify build_starter_priors CLI loads and runs --help without crashing."""
    from typer.testing import CliRunner
    from projections.cli.build_starter_priors import app

    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Usage" in result.output or "usage" in result.output.lower()


def test_badparameter_with_param_hint():
    """Verify that typer.BadParameter with param_hint= works correctly."""
    import typer
    
    # This should not raise TypeError about unexpected keyword argument
    try:
        raise typer.BadParameter("Test error message", param_hint="test_param")
    except typer.BadParameter as e:
        assert "Test error message" in str(e)
