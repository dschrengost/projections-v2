"""Tests for staleness monitoring module."""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
import pytest

from projections.validation.staleness import (
    DEFAULT_MAX_STALENESS,
    check_snapshot_staleness,
    check_all_snapshot_staleness,
)


def test_fresh_snapshots_pass() -> None:
    """Snapshots within threshold should not trigger warnings."""
    base = pd.Timestamp("2024-11-15T19:00:00Z")
    df = pd.DataFrame({
        "tip_ts": [base, base + timedelta(hours=3)],
        "injury_as_of_ts": [base - timedelta(hours=2), base + timedelta(hours=1)],
    })
    result = check_snapshot_staleness(df)
    assert result["checked"] is True
    assert result["stale_rows"] == 0


def test_stale_snapshots_detected() -> None:
    """Snapshots exceeding threshold should be flagged."""
    base = pd.Timestamp("2024-11-15T19:00:00Z")
    df = pd.DataFrame({
        "tip_ts": [base, base],
        "injury_as_of_ts": [base - timedelta(hours=10), base - timedelta(hours=2)],
    })
    with pytest.warns(RuntimeWarning, match="Staleness alert"):
        result = check_snapshot_staleness(df)
    assert result["stale_rows"] == 1


def test_custom_threshold() -> None:
    """Custom staleness threshold should be respected."""
    base = pd.Timestamp("2024-11-15T19:00:00Z")
    df = pd.DataFrame({
        "tip_ts": [base],
        "injury_as_of_ts": [base - timedelta(hours=2)],
    })
    # Default is 6 hours, so 2 hours should pass
    result = check_snapshot_staleness(df)
    assert result["stale_rows"] == 0

    # With 1 hour threshold, 2 hours should fail
    with pytest.warns(RuntimeWarning):
        result = check_snapshot_staleness(df, max_staleness=timedelta(hours=1))
    assert result["stale_rows"] == 1


def test_strict_mode_raises() -> None:
    """Strict mode should raise ValueError on stale data."""
    base = pd.Timestamp("2024-11-15T19:00:00Z")
    df = pd.DataFrame({
        "tip_ts": [base],
        "injury_as_of_ts": [base - timedelta(hours=10)],
    })
    with pytest.raises(ValueError, match="Staleness alert"):
        check_snapshot_staleness(df, warn_only=False)


def test_missing_columns_returns_unchecked() -> None:
    """Missing columns should return checked=False."""
    df = pd.DataFrame({"unrelated": [1, 2, 3]})
    result = check_snapshot_staleness(df)
    assert result["checked"] is False
    assert "missing columns" in result["reason"]


def test_empty_dataframe() -> None:
    """Empty DataFrame should return zero stale rows."""
    df = pd.DataFrame({"tip_ts": [], "injury_as_of_ts": []})
    result = check_snapshot_staleness(df)
    assert result["checked"] is True
    assert result["stale_rows"] == 0
    assert result["total_rows"] == 0


def test_check_all_snapshots() -> None:
    """Should check all common snapshot columns."""
    base = pd.Timestamp("2024-11-15T19:00:00Z")
    df = pd.DataFrame({
        "tip_ts": [base],
        "injury_as_of_ts": [base - timedelta(hours=2)],
        "odds_as_of_ts": [base - timedelta(hours=1)],
        "roster_as_of_ts": [base - timedelta(minutes=30)],
        "feature_as_of_ts": [base - timedelta(minutes=15)],
    })
    results = check_all_snapshot_staleness(df)
    assert "injury" in results
    assert "odds" in results
    assert "roster" in results
    assert "feature" in results
    assert all(r["checked"] for r in results.values() if "checked" in r)
