"""Tests for pipeline guardrail checks."""

import pandas as pd
import pytest

from projections.pipeline.guardrails import (
    GuardrailResult,
    check_data_freshness,
    check_feature_coverage,
    check_rates_output_sanity,
)


class TestCheckRatesOutputSanity:
    """Tests for check_rates_output_sanity."""

    def test_passes_with_valid_predictions(self):
        """Valid predictions should pass all checks."""
        df = pd.DataFrame({
            "pred_fga2_per_min": [0.3, 0.25, 0.4, 0.35],
            "pred_fga3_per_min": [0.15, 0.1, 0.2, 0.12],
            "pred_ast_per_min": [0.2, 0.15, 0.3, 0.25],
            "fpts_mean": [35.0, 25.0, 50.0, 30.0],
        })
        result = check_rates_output_sanity(df)
        assert result.passed
        assert len(result.warnings) == 0
        assert result.metrics["fga2_per_min_median"] > 0.15
        assert result.metrics["top_fpts"] == 50.0

    def test_warns_on_low_fga2_median(self):
        """Low fga2_per_min median should trigger warning."""
        df = pd.DataFrame({
            "pred_fga2_per_min": [0.05, 0.03, 0.08, 0.04],
            "fpts_mean": [35.0, 25.0, 50.0, 30.0],
        })
        result = check_rates_output_sanity(df, min_fga2_median=0.15)
        assert not result.passed
        assert any("fga2_per_min median" in w for w in result.warnings)

    def test_warns_on_low_top_fpts(self):
        """Low top FPTS should trigger warning."""
        df = pd.DataFrame({
            "pred_fga2_per_min": [0.3, 0.25, 0.4, 0.35],
            "fpts_mean": [30.0, 25.0, 35.0, 28.0],
        })
        result = check_rates_output_sanity(df, min_top_fpts=45.0)
        assert not result.passed
        assert any("fpts_mean" in w for w in result.warnings)

    def test_warns_on_nan_predictions(self):
        """NaN predictions should trigger warning."""
        df = pd.DataFrame({
            "pred_fga2_per_min": [0.3, None, 0.4, 0.35],
            "pred_ast_per_min": [0.2, 0.15, None, 0.25],
        })
        result = check_rates_output_sanity(df)
        assert "nan_rate" in result.metrics
        assert any("NaN" in w for w in result.warnings)

    def test_warns_on_high_zero_rate(self):
        """High zero rate should trigger warning."""
        df = pd.DataFrame({
            "pred_fga2_per_min": [0.0, 0.0, 0.0, 0.0],
            "pred_fga3_per_min": [0.0, 0.0, 0.0, 0.0],
        })
        result = check_rates_output_sanity(df, zero_tolerance=0.01)
        assert not result.passed
        assert any("zero rate" in w.lower() for w in result.warnings)

    def test_empty_dataframe_warns(self):
        """Empty dataframe should trigger warning."""
        df = pd.DataFrame(columns=["pred_fga2_per_min", "fpts_mean"])
        result = check_rates_output_sanity(df)
        assert not result.passed
        assert any("empty" in w.lower() for w in result.warnings)


class TestCheckFeatureCoverage:
    """Tests for check_feature_coverage."""

    def test_passes_with_complete_features(self):
        """Complete features should pass."""
        df = pd.DataFrame({
            "game_id": [1, 2, 3],
            "player_id": [101, 102, 103],
            "minutes_pred_p50": [25.0, 30.0, 20.0],
        })
        result = check_feature_coverage(df)
        assert result.passed
        assert result.metrics["row_count"] == 3

    def test_warns_on_missing_critical_cols(self):
        """Missing critical columns should trigger warning."""
        df = pd.DataFrame({
            "game_id": [1, 2, 3],
            "player_id": [101, 102, 103],
        })
        result = check_feature_coverage(df, critical_cols=["game_id", "minutes_pred_p50"])
        assert not result.passed
        assert any("minutes_pred_p50" in w for w in result.warnings)

    def test_warns_on_null_critical_cols(self):
        """Null values in critical columns should trigger warning."""
        df = pd.DataFrame({
            "game_id": [1, 2, None],
            "player_id": [101, 102, 103],
        })
        result = check_feature_coverage(df, critical_cols=["game_id"])
        assert not result.passed
        assert any("null" in w.lower() for w in result.warnings)

    def test_warns_on_low_row_count(self):
        """Row count below expected should trigger warning."""
        df = pd.DataFrame({
            "game_id": [1, 2, 3],
            "player_id": [101, 102, 103],
        })
        result = check_feature_coverage(df, expected_rows=100, min_row_ratio=0.9)
        assert not result.passed
        assert result.metrics["row_ratio"] < 0.1


class TestCheckDataFreshness:
    """Tests for check_data_freshness."""

    def test_passes_with_fresh_data(self):
        """Fresh data should pass."""
        now = pd.Timestamp.now(tz="UTC")
        df = pd.DataFrame({
            "feature_as_of_ts": [now - pd.Timedelta(hours=1), now - pd.Timedelta(hours=2)],
        })
        result = check_data_freshness(df, max_age_hours=24.0, reference_ts=now)
        assert result.passed
        assert result.metrics["max_age_hours"] < 24.0

    def test_warns_on_stale_data(self):
        """Stale data should trigger warning."""
        now = pd.Timestamp.now(tz="UTC")
        df = pd.DataFrame({
            "feature_as_of_ts": [now - pd.Timedelta(hours=48), now - pd.Timedelta(hours=50)],
        })
        result = check_data_freshness(df, max_age_hours=24.0, reference_ts=now)
        assert not result.passed
        assert any("hours old" in w for w in result.warnings)

    def test_skips_when_column_missing(self):
        """Missing timestamp column should skip check gracefully."""
        df = pd.DataFrame({
            "game_id": [1, 2, 3],
        })
        result = check_data_freshness(df)
        assert result.passed  # Not a failure, just skip
        assert any("not present" in w for w in result.warnings)
