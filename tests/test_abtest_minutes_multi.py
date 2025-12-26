"""Unit tests for multi-slate A/B/C test utilities.

Tests cover:
- Feature contract enforcement
- Scaling within eligible set (sum-to-240 + caps)
- Slate integrity heuristics
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Add project root to path
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from projections.eval.minutes_alloc_abtest import scale_shares_to_240


class TestFeatureContractEnforcement:
    """Tests for feature column filling logic."""

    def test_ensure_feature_columns_all_present(self):
        """When all columns exist, no changes should be made."""
        from scripts.abtest_minutes_allocators import _ensure_feature_columns
        
        df = pd.DataFrame({
            "game_id": [1, 1],
            "team_id": [10, 10],
            "feature_a": [1.0, 2.0],
            "feature_b": [3.0, 4.0],
        })
        expected = ["feature_a", "feature_b"]
        
        result, n_expected, n_missing, missing = _ensure_feature_columns(df, expected)
        
        assert n_expected == 2
        assert n_missing == 0
        assert missing == []
        assert list(result["feature_a"]) == [1.0, 2.0]

    def test_ensure_feature_columns_fills_missing(self):
        """Missing columns should be filled with 0."""
        from scripts.abtest_minutes_allocators import _ensure_feature_columns
        
        df = pd.DataFrame({
            "game_id": [1, 1],
            "team_id": [10, 10],
            "feature_a": [1.0, 2.0],
        })
        expected = ["feature_a", "feature_b", "feature_c"]
        
        result, n_expected, n_missing, missing = _ensure_feature_columns(df, expected)
        
        assert n_expected == 3
        assert n_missing == 2
        assert set(missing) == {"feature_b", "feature_c"}
        assert "feature_b" in result.columns
        assert "feature_c" in result.columns
        assert list(result["feature_b"]) == [0.0, 0.0]
        assert list(result["feature_c"]) == [0.0, 0.0]

    def test_ensure_feature_columns_fills_nans(self):
        """NaN values in existing columns should be filled with 0."""
        from scripts.abtest_minutes_allocators import _ensure_feature_columns
        
        df = pd.DataFrame({
            "game_id": [1, 1, 1],
            "team_id": [10, 10, 10],
            "feature_a": [1.0, np.nan, 3.0],
        })
        expected = ["feature_a"]
        
        result, n_expected, n_missing, missing = _ensure_feature_columns(df, expected)
        
        assert n_missing == 0
        assert not result["feature_a"].isna().any()
        assert list(result["feature_a"]) == [1.0, 0.0, 3.0]

    def test_ensure_feature_columns_empty_expected(self):
        """Empty expected list should return original df unchanged."""
        from scripts.abtest_minutes_allocators import _ensure_feature_columns
        
        df = pd.DataFrame({"a": [1, 2]})
        
        result, n_expected, n_missing, missing = _ensure_feature_columns(df, [])
        
        assert n_expected == 0
        assert n_missing == 0
        assert missing == []


class TestScalingWithinEligibleSet:
    """Tests for scaling shares within eligible set (Allocator C logic)."""

    def test_scaling_within_eligible_sums_to_240(self):
        """Shares scaled within eligible set should sum to 240 per team."""
        # Create a team with 10 players, 8 eligible
        df = pd.DataFrame({
            "game_id": ["g1"] * 10,
            "team_id": ["t1"] * 10,
            "player_id": [f"p{i}" for i in range(10)],
            "share": [0.15, 0.14, 0.13, 0.12, 0.11, 0.10, 0.09, 0.08, 0.00, 0.00],
            "eligible": [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],  # Last 2 ineligible
        })
        
        # Zero out shares for ineligible
        df["eligible_share"] = np.where(df["eligible"] == 1, df["share"], 0.0)
        
        result, diag = scale_shares_to_240(
            df,
            shares_col="eligible_share",
            cap_max=48.0,
            require_positive_share=False,
            redistribute_after_cap=True,
        )
        
        # Check sum to 240
        team_sum = result.groupby(["game_id", "team_id"])["minutes_mean_A"].sum().iloc[0]
        assert abs(team_sum - 240.0) < 1e-6
        
        # Check ineligible players got 0 minutes
        ineligible_mins = result[df["eligible"] == 0]["minutes_mean_A"].sum()
        assert ineligible_mins == 0.0

    def test_scaling_respects_cap(self):
        """Cap should be enforced and excess redistributed."""
        # Create a star-heavy team with 8 players (realistic)
        df = pd.DataFrame({
            "game_id": ["g1"] * 8,
            "team_id": ["t1"] * 8,
            "player_id": [f"p{i}" for i in range(8)],
            "share": [0.35, 0.20, 0.15, 0.12, 0.08, 0.05, 0.03, 0.02],  # Star gets 35% share
        })
        
        result, diag = scale_shares_to_240(
            df,
            shares_col="share",
            cap_max=42.0,
            require_positive_share=False,
            redistribute_after_cap=True,
        )
        
        # Check sum to 240
        team_sum = result.groupby(["game_id", "team_id"])["minutes_mean_A"].sum().iloc[0]
        assert abs(team_sum - 240.0) < 1e-6
        
        # Check cap not exceeded
        max_mins = result["minutes_mean_A"].max()
        assert max_mins <= 42.0 + 1e-6

    def test_scaling_multi_team(self):
        """Each team should independently sum to 240."""
        df = pd.DataFrame({
            "game_id": ["g1"] * 8 + ["g1"] * 8,
            "team_id": ["t1"] * 8 + ["t2"] * 8,
            "player_id": [f"p{i}" for i in range(16)],
            "share": [0.15, 0.15, 0.15, 0.15, 0.1, 0.1, 0.1, 0.1] * 2,
        })
        
        result, diag = scale_shares_to_240(
            df,
            shares_col="share",
            cap_max=48.0,
            require_positive_share=False,
            redistribute_after_cap=True,
        )
        
        # Check each team sums to 240
        team_sums = result.groupby(["game_id", "team_id"])["minutes_mean_A"].sum()
        assert len(team_sums) == 2
        for team_sum in team_sums:
            assert abs(team_sum - 240.0) < 1e-6


class TestSlateIntegrityHeuristic:
    """Tests for slate integrity checks."""

    def test_integrity_valid_slate(self):
        """Valid slate should pass integrity check."""
        from scripts.abtest_minutes_allocators import _check_slate_integrity
        
        # 2 games, 4 teams, 10 players each = 40 players
        df = pd.DataFrame({
            "game_id": (["g1"] * 20) + (["g2"] * 20),
            "team_id": (["t1"] * 10 + ["t2"] * 10) + (["t3"] * 10 + ["t4"] * 10),
            "player_id": [f"p{i}" for i in range(40)],
        })
        
        is_valid, reason, counts = _check_slate_integrity(df)
        
        assert is_valid
        assert reason is None
        assert counts["n_games"] == 2
        assert counts["n_teams"] == 4
        assert counts["n_players"] == 40

    def test_integrity_missing_teams(self):
        """Slate with missing teams should fail."""
        from scripts.abtest_minutes_allocators import _check_slate_integrity
        
        # 2 games but only 3 teams (one game incomplete)
        df = pd.DataFrame({
            "game_id": (["g1"] * 20) + (["g2"] * 10),  # g2 has only 1 team
            "team_id": (["t1"] * 10 + ["t2"] * 10) + (["t3"] * 10),
            "player_id": [f"p{i}" for i in range(30)],
        })
        
        is_valid, reason, counts = _check_slate_integrity(df)
        
        assert not is_valid
        assert "missing_teams" in reason
        assert counts["n_games"] == 2
        assert counts["n_teams"] == 3  # Expected 4

    def test_integrity_no_games(self):
        """Empty slate should fail."""
        from scripts.abtest_minutes_allocators import _check_slate_integrity
        
        df = pd.DataFrame({
            "game_id": pd.Series([], dtype=str),
            "team_id": pd.Series([], dtype=str),
            "player_id": pd.Series([], dtype=str),
        })
        
        is_valid, reason, counts = _check_slate_integrity(df)
        
        assert not is_valid
        assert "no_games_or_teams" in reason

    def test_integrity_incomplete_roster(self):
        """Teams with too few players should fail."""
        from scripts.abtest_minutes_allocators import _check_slate_integrity
        
        # 1 game, 2 teams, but one team has only 3 players
        df = pd.DataFrame({
            "game_id": ["g1"] * 13,
            "team_id": ["t1"] * 10 + ["t2"] * 3,  # t2 has incomplete roster
            "player_id": [f"p{i}" for i in range(13)],
        })
        
        is_valid, reason, counts = _check_slate_integrity(df)
        
        assert not is_valid
        assert "incomplete_rosters" in reason


class TestQualityTierLogic:
    """Tests for quality tier assignment based on missing features."""

    def test_clean_tier_threshold(self):
        """Missing < 2% should be 'clean'."""
        from projections.eval.minutes_alloc_abtest_aggregate import (
            MISSING_FEATURE_CLEAN_THRESHOLD,
            MISSING_FEATURE_SKIP_THRESHOLD,
        )
        
        # With 100 expected features
        n_expected = 100
        n_missing_clean = 1  # 1% missing
        n_missing_degraded = 5  # 5% missing
        n_missing_skip = 15  # 15% missing
        
        frac_clean = n_missing_clean / n_expected
        frac_degraded = n_missing_degraded / n_expected
        frac_skip = n_missing_skip / n_expected
        
        assert frac_clean <= MISSING_FEATURE_CLEAN_THRESHOLD  # Should be clean
        assert frac_degraded > MISSING_FEATURE_CLEAN_THRESHOLD  # Should be degraded
        assert frac_degraded <= MISSING_FEATURE_SKIP_THRESHOLD  # Not skipped
        assert frac_skip > MISSING_FEATURE_SKIP_THRESHOLD  # Should be skipped


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
