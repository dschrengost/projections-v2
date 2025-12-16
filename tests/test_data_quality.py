"""Tests for data quality validation module."""

from __future__ import annotations

import pandas as pd
import pytest

from projections.validation.data_quality import (
    FeatureBounds,
    MINUTES_FEATURE_BOUNDS,
    validate_feature_ranges,
)


def test_valid_features_pass() -> None:
    """Features within bounds should not produce violations."""
    df = pd.DataFrame({
        "minutes": [25.0, 30.0, 0.0, 48.0],
        "roll_mean_5": [20.0, 22.0, 15.0, 35.0],
        "prior_play_prob": [0.5, 0.8, 0.0, 1.0],
        "is_starter": [0, 1, 0, 1],
    })
    violations = validate_feature_ranges(df)
    assert violations == []


def test_out_of_range_above_detected() -> None:
    """Values above max should be flagged."""
    df = pd.DataFrame({"minutes": [50.0, 25.0, 30.0]})
    violations = validate_feature_ranges(df)
    assert len(violations) == 1
    assert "above 48" in violations[0].lower()


def test_out_of_range_below_detected() -> None:
    """Values below min should be flagged."""
    df = pd.DataFrame({"minutes": [-5.0, 20.0, 30.0]})
    violations = validate_feature_ranges(df)
    assert len(violations) == 1
    assert "below 0" in violations[0].lower()


def test_probability_bounds() -> None:
    """Probabilities outside [0, 1] should be flagged."""
    df = pd.DataFrame({"prior_play_prob": [0.5, 1.2, 0.8]})
    violations = validate_feature_ranges(df)
    assert len(violations) == 1
    assert "above 1" in violations[0].lower()


def test_strict_mode_raises() -> None:
    """Strict mode should raise ValueError on violations."""
    df = pd.DataFrame({"minutes": [-5.0]})
    with pytest.raises(ValueError, match="Data quality violations"):
        validate_feature_ranges(df, strict=True)


def test_custom_bounds() -> None:
    """Custom bounds should be respected."""
    custom_bounds = [
        FeatureBounds("custom_col", min_val=10, max_val=20),
    ]
    df = pd.DataFrame({"custom_col": [5, 15, 25]})
    violations = validate_feature_ranges(df, bounds=custom_bounds)
    assert len(violations) == 2  # One below 10, one above 20


def test_missing_column_ignored() -> None:
    """Columns not in DataFrame should be silently skipped."""
    df = pd.DataFrame({"unrelated": [1, 2, 3]})
    violations = validate_feature_ranges(df)
    assert violations == []


def test_null_values_allowed_by_default() -> None:
    """Null values should be allowed when allow_null=True."""
    df = pd.DataFrame({"minutes": [25.0, None, 30.0]})
    violations = validate_feature_ranges(df)
    assert violations == []


def test_null_values_flagged_when_disallowed() -> None:
    """Null values should be flagged when allow_null=False."""
    custom_bounds = [
        FeatureBounds("strict_col", allow_null=False),
    ]
    df = pd.DataFrame({"strict_col": [1.0, None, 2.0]})
    violations = validate_feature_ranges(df, bounds=custom_bounds)
    assert len(violations) == 1
    assert "null" in violations[0].lower()
