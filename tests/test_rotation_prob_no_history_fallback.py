"""Regression test for rotation probability fallback when history is missing.

This test ensures that when player history features are missing (roll_mean_5, min_last1,
min_last3 all NaN), the rotation_prob fallback uses p50_pred to spread players across
different tiers rather than collapsing everyone to a single constant value.

This prevents templated minutes outputs where many players share identical (p10, p50, p90)
triplets due to rotation cap buckets.
"""

import numpy as np
import pandas as pd
import pytest

from projections.cli.score_minutes_v1 import _derive_rotation_prob_with_history


class TestRotationProbNoHistoryFallback:
    """Regression tests for p50_pred-based fallback when history features are missing."""

    @pytest.fixture
    def df_no_history(self) -> pd.DataFrame:
        """Create test DataFrame with no history features, only p50_pred."""
        return pd.DataFrame({
            "player_id": range(10),
            "p50_pred": [35.0, 28.0, 22.0, 15.0, 10.0, 7.0, 4.0, 2.5, 1.0, 0.5],
            # History features intentionally missing (will be treated as NaN)
        })

    @pytest.fixture
    def df_with_history(self) -> pd.DataFrame:
        """Create test DataFrame with history features present."""
        return pd.DataFrame({
            "player_id": range(5),
            "p50_pred": [20.0, 20.0, 20.0, 20.0, 20.0],
            "roll_mean_5": [15.0, 10.0, 5.0, np.nan, np.nan],
            "min_last1": [np.nan, np.nan, np.nan, 12.0, np.nan],
            "min_last3": [np.nan, np.nan, np.nan, np.nan, np.nan],
            "recent_start_pct_10": [0.0, 0.0, 0.0, 0.0, 0.0],
        })

    def test_no_history_spreads_across_tiers(self, df_no_history: pd.DataFrame):
        """When history is missing, rotation_prob should vary based on p50_pred."""
        prob, has_history = _derive_rotation_prob_with_history(df_no_history)

        # All players should have has_any_history = False
        assert not has_history.any(), "All players should report no history"

        # Rotation prob should NOT be a single constant value
        unique_probs = prob.unique()
        assert len(unique_probs) > 1, (
            f"rotation_prob should vary based on p50_pred, got single value: {unique_probs}"
        )

        # High p50_pred players should get higher rotation_prob
        assert prob.iloc[0] > prob.iloc[-1], (
            "Players with higher p50_pred should have higher rotation_prob"
        )

    def test_no_history_uses_p50_tiers(self, df_no_history: pd.DataFrame):
        """Verify specific tier boundaries are respected for p50_pred fallback."""
        prob, _ = _derive_rotation_prob_with_history(df_no_history)

        # p50_pred >= 28 should get >= 0.55 (no rotation caps)
        assert prob.iloc[0] >= 0.55, f"p50=35 should get >=0.55, got {prob.iloc[0]}"
        assert prob.iloc[1] >= 0.55, f"p50=28 should get >=0.55, got {prob.iloc[1]}"

        # p50_pred >= 20 should get >= 0.35 tier
        assert prob.iloc[2] >= 0.35, f"p50=22 should get >=0.35, got {prob.iloc[2]}"

        # Very low p50_pred should get low rotation_prob
        assert prob.iloc[-1] < 0.15, f"p50=0.5 should get <0.15, got {prob.iloc[-1]}"

    def test_history_present_ignores_p50_fallback(self, df_with_history: pd.DataFrame):
        """When history features are present, p50_pred fallback should not be used."""
        prob, has_history = _derive_rotation_prob_with_history(df_with_history)

        # First 4 players have some history (roll_mean_5 or min_last1 is not NaN)
        assert has_history.iloc[0], "Player 0 should have history (roll_mean_5=15)"
        assert has_history.iloc[3], "Player 3 should have history (min_last1=12)"

        # Players with history should get rotation_prob based on history rules, not p50
        # Player 0: roll_mean_5=15 >= 14 -> should get 0.85
        assert prob.iloc[0] == pytest.approx(0.85, abs=0.01), (
            f"Player 0 with roll_mean_5=15 should get ~0.85, got {prob.iloc[0]}"
        )

    def test_mixed_history_players(self, df_with_history: pd.DataFrame):
        """When some players have history and others don't, both paths are used."""
        prob, has_history = _derive_rotation_prob_with_history(df_with_history)

        # Last player has no history (all history features NaN or 0)
        assert not has_history.iloc[-1], "Last player should have no history"

        # Should have a mix of rotation_prob values
        unique_probs = prob.unique()
        assert len(unique_probs) >= 2, "Should have different probs for history vs no-history"
