"""Tests for minutes feature builder verification."""

from __future__ import annotations

import pandas as pd
import pytest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from projections.cli.build_minutes_live import (
    REQUIRED_MINUTES_FEATURES,
    _verify_required_features,
)


class TestRequiredMinutesFeatures:
    """Tests for REQUIRED_MINUTES_FEATURES constant."""

    def test_contains_team_context(self):
        """Team context columns are required."""
        assert "team_pace_szn" in REQUIRED_MINUTES_FEATURES
        assert "team_off_rtg_szn" in REQUIRED_MINUTES_FEATURES
        assert "team_def_rtg_szn" in REQUIRED_MINUTES_FEATURES

    def test_contains_opp_context(self):
        """Opponent context columns are required."""
        assert "opp_pace_szn" in REQUIRED_MINUTES_FEATURES
        assert "opp_def_rtg_szn" in REQUIRED_MINUTES_FEATURES

    def test_contains_vacancy(self):
        """Vacancy feature columns are required."""
        assert "vac_min_szn" in REQUIRED_MINUTES_FEATURES
        assert "vac_min_guard_szn" in REQUIRED_MINUTES_FEATURES
        assert "vac_min_wing_szn" in REQUIRED_MINUTES_FEATURES
        assert "vac_min_big_szn" in REQUIRED_MINUTES_FEATURES

    def test_contains_trends(self):
        """Trend feature columns are required."""
        assert "roll_mean_5" in REQUIRED_MINUTES_FEATURES
        assert "roll_mean_10" in REQUIRED_MINUTES_FEATURES
        assert "min_last3" in REQUIRED_MINUTES_FEATURES
        assert "min_last5" in REQUIRED_MINUTES_FEATURES


class TestVerifyRequiredFeatures:
    """Tests for _verify_required_features function."""

    def test_all_required_present(self):
        """No warning when all required features present."""
        df = pd.DataFrame({col: [1.0] for col in REQUIRED_MINUTES_FEATURES})
        warnings = []
        
        _verify_required_features(df, "test_run", warnings)
        
        # No warnings should be added when all features present
        assert len([w for w in warnings if "Missing" in w]) == 0

    def test_missing_features_logged(self):
        """Missing features are logged as warning."""
        # Only include half the required features
        subset = list(REQUIRED_MINUTES_FEATURES)[:7]
        df = pd.DataFrame({col: [1.0] for col in subset})
        warnings = []
        
        _verify_required_features(df, "test_run", warnings)
        
        # Warning should be added
        assert len(warnings) == 1
        assert "Missing" in warnings[0]
        # All missing columns should be mentioned
        missing = REQUIRED_MINUTES_FEATURES - set(subset)
        for col in missing:
            assert col in warnings[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
