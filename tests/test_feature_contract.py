"""Unit tests for feature contract enforcement module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import json
from pathlib import Path
import tempfile

import sys
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from projections.models.minute_share.feature_contract import (
    ContractReport,
    CRITICAL_FEATURES,
    SEVERITY_CRITICAL,
    SEVERITY_IMPORTANT,
    SEVERITY_MINOR,
    load_expected_share_features,
    enforce_share_feature_contract,
    get_missing_feature_summary,
    get_severity_summary,
)


class TestLoadExpectedShareFeatures:
    """Tests for loading expected feature columns from bundle."""

    def test_load_from_dict_format(self, tmp_path):
        """Load columns from dict format with 'columns' key."""
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        cols_file = bundle / "feature_columns.json"
        cols_file.write_text(json.dumps({"columns": ["a", "b", "c"]}))
        
        result = load_expected_share_features(bundle)
        assert result == ["a", "b", "c"]

    def test_load_from_list_format(self, tmp_path):
        """Load columns from list format."""
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        cols_file = bundle / "feature_columns.json"
        cols_file.write_text(json.dumps(["x", "y", "z"]))
        
        result = load_expected_share_features(bundle)
        assert result == ["x", "y", "z"]

    def test_empty_bundle(self, tmp_path):
        """Empty bundle returns empty list."""
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        
        result = load_expected_share_features(bundle)
        assert result == []


class TestEnforceShareFeatureContract:
    """Tests for feature contract enforcement."""

    def test_all_columns_present(self):
        """When all columns present, no changes needed."""
        df = pd.DataFrame({
            "player_id": [1, 2],
            "feature_a": [1.0, 2.0],
            "feature_b": [3.0, 4.0],
        })
        expected = ["feature_a", "feature_b"]
        
        result, report = enforce_share_feature_contract(df, expected)
        
        assert report.n_expected == 2
        assert report.n_missing == 0
        assert len(report.missing_cols) == 0
        assert report.missing_feature_frac == 0.0

    def test_missing_columns_filled_with_zero(self):
        """Missing columns are added and filled with 0."""
        df = pd.DataFrame({
            "player_id": [1, 2],
            "feature_a": [1.0, 2.0],
        })
        expected = ["feature_a", "feature_b", "feature_c"]
        
        result, report = enforce_share_feature_contract(df, expected)
        
        assert "feature_b" in result.columns
        assert "feature_c" in result.columns
        assert list(result["feature_b"]) == [0.0, 0.0]
        assert list(result["feature_c"]) == [0.0, 0.0]
        assert report.n_missing == 2
        assert set(report.missing_cols) == {"feature_b", "feature_c"}
        assert report.missing_feature_frac == pytest.approx(2/3)

    def test_nan_indicator_creation(self):
        """_is_nan indicator columns are created from base column when indicator is missing."""
        df = pd.DataFrame({
            "player_id": [1, 2, 3],
            "roll_mean_5": [10.0, np.nan, 15.0],
            # roll_mean_5_is_nan is NOT in the df - it will be created
        })
        # The indicator column is expected but missing from df
        expected = ["roll_mean_5", "roll_mean_5_is_nan"]
        
        result, report = enforce_share_feature_contract(df, expected)
        
        assert "roll_mean_5_is_nan" in result.columns
        # NaN was at index 1, indicator should be created from base's NaN status  
        assert list(result["roll_mean_5_is_nan"]) == [0.0, 1.0, 0.0]
        # Base column should have NaN filled
        assert not result["roll_mean_5"].isna().any()
        assert report.nan_indicators_created == 1

    def test_nan_indicator_for_missing_base(self):
        """When base column is missing, indicator is set to 1.0 (data was missing)."""
        df = pd.DataFrame({
            "player_id": [1, 2],
        })
        expected = ["missing_feature", "missing_feature_is_nan"]
        
        result, report = enforce_share_feature_contract(df, expected)
        
        assert "missing_feature" in result.columns
        assert "missing_feature_is_nan" in result.columns
        # Base was missing, so indicator = 1.0 for all rows (footgun prevention)
        assert list(result["missing_feature"]) == [0.0, 0.0]
        assert list(result["missing_feature_is_nan"]) == [1.0, 1.0]
        # Track that base was missing
        assert "missing_feature" in report.base_missing_cols

    def test_existing_nans_filled(self):
        """Existing NaN values in columns are filled with 0."""
        df = pd.DataFrame({
            "player_id": [1, 2, 3],
            "feature_a": [1.0, np.nan, 3.0],
        })
        expected = ["feature_a"]
        
        result, report = enforce_share_feature_contract(df, expected)
        
        assert not result["feature_a"].isna().any()
        assert list(result["feature_a"]) == [1.0, 0.0, 3.0]

    def test_empty_expected_returns_original(self):
        """Empty expected list returns original df unchanged."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        
        result, report = enforce_share_feature_contract(df, [])
        
        assert report.n_expected == 0
        assert report.n_missing == 0
        assert result.equals(df)


class TestGetMissingFeatureSummary:
    """Tests for aggregating missing feature info across reports."""

    def test_aggregates_counts(self):
        """Counts are aggregated across reports."""
        reports = [
            ContractReport(n_missing=2, missing_cols=["a", "b"]),
            ContractReport(n_missing=2, missing_cols=["a", "c"]),
            ContractReport(n_missing=1, missing_cols=["a"]),
        ]
        
        result = get_missing_feature_summary(reports)
        
        assert result["a"] == 3  # In all 3 reports
        assert result["b"] == 1
        assert result["c"] == 1
        
    def test_sorted_by_count(self):
        """Results should be sorted by count descending."""
        reports = [
            ContractReport(n_missing=3, missing_cols=["rare", "common", "very_common"]),
            ContractReport(n_missing=2, missing_cols=["common", "very_common"]),
            ContractReport(n_missing=1, missing_cols=["very_common"]),
        ]
        
        result = get_missing_feature_summary(reports)
        
        keys = list(result.keys())
        assert keys[0] == "very_common"  # 3 occurrences
        assert keys[1] == "common"  # 2 occurrences
        assert keys[2] == "rare"  # 1 occurrence


class TestContractReport:
    """Tests for ContractReport dataclass."""

    def test_to_dict_truncates_missing_cols(self):
        """to_dict truncates missing_cols to 25."""
        report = ContractReport(
            n_expected=100,
            n_missing=50,
            missing_cols=[f"col_{i}" for i in range(50)],
        )
        
        d = report.to_dict()
        
        assert len(d["missing_cols"]) == 25

    def test_missing_feature_frac_calculation(self):
        """missing_feature_frac should be calculated correctly."""
        report = ContractReport(
            n_expected=100,
            n_missing=15,
            missing_feature_frac=0.15,
        )
        
        assert report.missing_feature_frac == 0.15


class TestSeverityTracking:
    """Tests for critical feature and severity tracking."""

    def test_critical_feature_detection(self):
        """Critical features should be tracked separately."""
        df = pd.DataFrame({
            "player_id": [1, 2],
            "other_feature": [1.0, 2.0],
        })
        # roll_mean_5 is a critical feature
        expected = ["other_feature", "roll_mean_5", "roll_mean_10"]
        
        result, report = enforce_share_feature_contract(df, expected)
        
        assert report.n_critical_missing >= 2
        assert "roll_mean_5" in report.critical_missing
        assert "roll_mean_10" in report.critical_missing

    def test_severity_minor_when_no_critical_missing(self):
        """Severity is minor when no critical features missing."""
        df = pd.DataFrame({
            "player_id": [1, 2],
            "other": [1.0, 2.0],
        })
        expected = ["other", "non_critical_a", "non_critical_b"]
        
        result, report = enforce_share_feature_contract(df, expected)
        
        # Only 2 non-critical missing -> minor
        assert report.severity == SEVERITY_MINOR

    def test_severity_important_when_some_critical_missing(self):
        """Severity is important when 1-2 critical features missing."""
        df = pd.DataFrame({
            "player_id": [1, 2],
        })
        # Only 1 critical feature
        expected = ["roll_mean_5"]
        
        result, report = enforce_share_feature_contract(df, expected)
        
        assert report.severity == SEVERITY_IMPORTANT

    def test_severity_critical_when_many_critical_missing(self):
        """Severity is critical when 3+ critical features missing."""
        df = pd.DataFrame({
            "player_id": [1, 2],
        })
        # 3+ critical features
        expected = ["roll_mean_5", "roll_mean_10", "min_last3", "starter_flag"]
        
        result, report = enforce_share_feature_contract(df, expected)
        
        assert report.n_critical_missing >= 3
        assert report.severity == SEVERITY_CRITICAL

    def test_get_severity_summary(self):
        """Severity summary aggregates across reports."""
        reports = [
            ContractReport(severity=SEVERITY_CRITICAL),
            ContractReport(severity=SEVERITY_IMPORTANT),
            ContractReport(severity=SEVERITY_MINOR),
            ContractReport(severity=SEVERITY_MINOR),
        ]
        
        summary = get_severity_summary(reports)
        
        assert summary[SEVERITY_CRITICAL] == 1
        assert summary[SEVERITY_IMPORTANT] == 1
        assert summary[SEVERITY_MINOR] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

