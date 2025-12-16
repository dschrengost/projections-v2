"""Tests for rates_v1 schemas and validation."""

import pandas as pd
import pytest

from projections.rates_v1.schemas import (
    FEATURES_RATES_V1_COLUMNS,
    FEATURES_RATES_V1_SCHEMA,
    FeatureSchemaMismatchError,
    validate_rates_features,
)


class TestFeaturesRatesV1Schema:
    """Tests for FEATURES_RATES_V1_SCHEMA definition."""

    def test_schema_has_key_columns(self):
        """Schema should include game_id, player_id, team_id, game_date."""
        assert "game_id" in FEATURES_RATES_V1_COLUMNS
        assert "player_id" in FEATURES_RATES_V1_COLUMNS
        assert "team_id" in FEATURES_RATES_V1_COLUMNS
        assert "game_date" in FEATURES_RATES_V1_COLUMNS

    def test_schema_has_stage3_features(self):
        """Schema should include stage3 context features."""
        # Check some key features from each stage
        assert "minutes_pred_p50" in FEATURES_RATES_V1_COLUMNS
        assert "season_fga_per_min" in FEATURES_RATES_V1_COLUMNS
        assert "track_touches_per_min_szn" in FEATURES_RATES_V1_COLUMNS
        assert "vac_min_szn" in FEATURES_RATES_V1_COLUMNS
        assert "team_pace_szn" in FEATURES_RATES_V1_COLUMNS

    def test_schema_has_primary_key(self):
        """Schema should define a primary key."""
        assert FEATURES_RATES_V1_SCHEMA.primary_key == ("game_id", "player_id")


class TestValidateRatesFeatures:
    """Tests for validate_rates_features function."""

    def test_valid_features_pass(self):
        """Complete feature set should pass validation."""
        data = {col: [1.0] for col in FEATURES_RATES_V1_COLUMNS}
        data["game_id"] = [1]
        data["player_id"] = [1]
        data["team_id"] = [1]
        data["game_date"] = pd.to_datetime(["2025-01-01"])
        df = pd.DataFrame(data)

        result = validate_rates_features(df, strict=True)
        assert result == []

    def test_missing_columns_raise_in_strict_mode(self):
        """Missing columns should raise FeatureSchemaMismatchError in strict mode."""
        # Create df with only key columns
        df = pd.DataFrame({
            "game_id": [1],
            "player_id": [1],
            "team_id": [1],
            "game_date": pd.to_datetime(["2025-01-01"]),
        })

        with pytest.raises(FeatureSchemaMismatchError) as exc_info:
            validate_rates_features(df, strict=True)

        assert "missing required columns" in str(exc_info.value)

    def test_missing_columns_return_list_in_non_strict_mode(self):
        """Missing columns should return list in non-strict mode."""
        df = pd.DataFrame({
            "game_id": [1],
            "player_id": [1],
            "team_id": [1],
            "game_date": pd.to_datetime(["2025-01-01"]),
        })

        result = validate_rates_features(df, strict=False)
        assert isinstance(result, list)
        assert len(result) > 0
        assert "minutes_pred_p50" in result
