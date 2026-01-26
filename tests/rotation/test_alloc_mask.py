"""Tests for alloc_mask construction logic (PR: fix minutes smear)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from projections.rotation.alloc_mask import build_alloc_mask_from_features


class TestBuildAllocMaskFromFeatures:
    """Tests for build_alloc_mask_from_features helper."""

    def test_empty_dataframe_returns_empty_array(self) -> None:
        """Empty input returns empty mask."""
        df = pd.DataFrame()
        result = build_alloc_mask_from_features(df)
        assert len(result) == 0
        assert result.dtype == bool

    def test_missing_columns_defaults_all_eligible(self) -> None:
        """When no signal columns exist, all players should be eligible via backfill."""
        df = pd.DataFrame({
            "player_id": [1, 2, 3, 4, 5],
            "game_id": [100] * 5,
            "team_id": [10] * 5,
        })
        result = build_alloc_mask_from_features(df, min_eligible=9)
        # All 5 players should be eligible (backfill kicks in since min_eligible > count)
        assert result.sum() == 5
        np.testing.assert_array_equal(result, [True, True, True, True, True])

    def test_out_players_excluded(self) -> None:
        """Players with is_out=1 should never be eligible."""
        df = pd.DataFrame({
            "player_id": [1, 2, 3, 4, 5],
            "is_out": [0, 1, 0, 1, 0],
            "prior_play_prob": [0.9, 0.9, 0.9, 0.9, 0.9],
        })
        result = build_alloc_mask_from_features(df, min_eligible=1)
        # Players 2 and 4 are OUT, should be False
        expected = np.array([True, False, True, False, True])
        np.testing.assert_array_equal(result, expected)

    def test_status_out_excluded(self) -> None:
        """Players with status='OUT' should be excluded regardless of is_out."""
        df = pd.DataFrame({
            "player_id": [1, 2, 3],
            "is_out": [0, 0, 0],
            "status": ["", "OUT", "out"],
            "prior_play_prob": [0.9, 0.9, 0.9],
        })
        result = build_alloc_mask_from_features(df, min_eligible=1)
        # Players 2 and 3 have OUT status
        expected = np.array([True, False, False])
        np.testing.assert_array_equal(result, expected)

    def test_starters_always_eligible(self) -> None:
        """Confirmed or projected starters should always be eligible."""
        df = pd.DataFrame({
            "player_id": list(range(1, 19)),
            "is_out": [0] * 18,
            "is_confirmed_starter": [1, 0, 0, 0, 0] + [0] * 13,
            "is_projected_starter": [0, 1, 0, 0, 0] + [0] * 13,
            "prior_play_prob": [0.01] * 18,  # Very low prob, wouldn't qualify otherwise
        })
        result = build_alloc_mask_from_features(df, min_eligible=2)
        # Starters (0, 1) should be eligible even with low prior_play_prob
        assert result[0] == True  # confirmed starter
        assert result[1] == True  # projected starter
        # min_eligible=2 is satisfied by starters, so no backfill needed
        assert result.sum() == 2

    def test_high_prior_prob_eligible(self) -> None:
        """Players with high prior_play_prob should be eligible."""
        df = pd.DataFrame({
            "player_id": list(range(1, 11)),
            "is_out": [0] * 10,
            "prior_play_prob": [0.90, 0.85, 0.25, 0.20, 0.19, 0.15, 0.10, 0.05, 0.01, 0.00],
        })
        result = build_alloc_mask_from_features(
            df, min_eligible=1, prior_play_prob_threshold=0.20
        )
        # Players 0-3 have prior_prob >= 0.20
        expected = np.array([True, True, True, True, False, False, False, False, False, False])
        np.testing.assert_array_equal(result, expected)

    def test_high_baseline_minutes_eligible(self) -> None:
        """Players with high baseline minutes should be eligible."""
        df = pd.DataFrame({
            "player_id": list(range(1, 11)),
            "is_out": [0] * 10,
            "prior_play_prob": [0.05] * 10,  # Low prob
            "baseline_p50": [35.0, 28.0, 10.0, 5.0, 4.0, 3.9, 2.0, 1.0, 0.5, 0.0],
        })
        result = build_alloc_mask_from_features(
            df,
            min_eligible=1,
            prior_play_prob_threshold=0.20,
            baseline_minutes_threshold=4.0,
            baseline_minutes_col="baseline_p50",
        )
        # Players 0-4 have baseline >= 4.0
        expected = np.array([True, True, True, True, True, False, False, False, False, False])
        np.testing.assert_array_equal(result, expected)

    def test_min_eligible_backfill(self) -> None:
        """When fewer than min_eligible are naturally eligible, backfill from not-out."""
        df = pd.DataFrame({
            "player_id": list(range(1, 19)),
            "is_out": [0] * 18,
            "prior_play_prob": [0.90, 0.85] + [0.05] * 16,  # Only 2 high prob
            "is_confirmed_starter": [0] * 18,
            "is_projected_starter": [0] * 18,
        })
        result = build_alloc_mask_from_features(df, min_eligible=9)
        # Should have exactly 9 eligible (2 natural + 7 backfilled)
        assert result.sum() == 9
        # First 2 (high prob) should definitely be eligible
        assert result[0] == True
        assert result[1] == True

    def test_backfill_respects_priority_order(self) -> None:
        """Backfill should add players in descending order of prior_play_prob."""
        df = pd.DataFrame({
            "player_id": list(range(1, 11)),
            "is_out": [0] * 10,
            "prior_play_prob": [0.19, 0.18, 0.17, 0.16, 0.15, 0.14, 0.13, 0.12, 0.11, 0.10],
        })
        result = build_alloc_mask_from_features(
            df, min_eligible=5, prior_play_prob_threshold=0.20
        )
        # No one naturally qualifies (all < 0.20), so backfill top 5 by prob
        assert result.sum() == 5
        # First 5 have highest prob
        expected = np.array([True, True, True, True, True, False, False, False, False, False])
        np.testing.assert_array_equal(result, expected)

    def test_max_eligible_trims_to_limit(self) -> None:
        """When max_eligible is set, trim eligible players to that limit."""
        df = pd.DataFrame({
            "player_id": list(range(1, 16)),
            "is_out": [0] * 15,
            "prior_play_prob": [0.95, 0.90, 0.85, 0.80, 0.75, 0.70] + [0.60] * 9,
            "is_confirmed_starter": [1, 1, 0, 0, 0, 0] + [0] * 9,
        })
        result = build_alloc_mask_from_features(
            df, min_eligible=5, max_eligible=8, prior_play_prob_threshold=0.50
        )
        # All 15 players have prob >= 0.50, but max_eligible=8
        assert result.sum() == 8
        # Starters (0, 1) should always be kept
        assert result[0] == True
        assert result[1] == True

    def test_realistic_team_game_scenario(self) -> None:
        """Simulate a real team-game: 18 roster, only 9 rotation players expected."""
        # Create realistic data: 5 starters + high rotation, 4 bench rotation, 9 deep bench
        df = pd.DataFrame({
            "player_id": list(range(100, 118)),
            "is_out": [0] * 18,
            "is_confirmed_starter": [1, 1, 1, 1, 1] + [0] * 13,
            "is_projected_starter": [0] * 18,
            "prior_play_prob": (
                [0.99, 0.99, 0.99, 0.99, 0.99]  # 5 starters
                + [0.85, 0.80, 0.75, 0.70]  # 4 bench rotation
                + [0.15, 0.12, 0.10, 0.08, 0.05, 0.03, 0.02, 0.01, 0.00]  # 9 deep bench
            ),
        })
        result = build_alloc_mask_from_features(
            df, min_eligible=9, prior_play_prob_threshold=0.20
        )
        # 5 starters + 4 bench rotation = 9 should naturally qualify
        assert result.sum() == 9
        # First 9 should be eligible, last 9 should not
        expected = np.array([True] * 9 + [False] * 9)
        np.testing.assert_array_equal(result, expected)

    def test_no_smear_scenario(self) -> None:
        """
        Ensure the alloc_mask prevents minutes smear by excluding low-signal players.
        
        This test ensures that end-of-bench players with very low play probability
        are excluded from the allocation mask, preventing the 240-minute budget
        from being spread across the entire roster.
        """
        df = pd.DataFrame({
            "player_id": list(range(1, 19)),
            "is_out": [0] * 18,
            "is_confirmed_starter": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "prior_play_prob": [
                0.99, 0.99, 0.99, 0.99, 0.99,  # starters
                0.90, 0.85, 0.75, 0.60,  # key bench
                0.10, 0.08, 0.05, 0.03, 0.02, 0.01, 0.01, 0.01, 0.00,  # deep bench
            ],
        })
        result = build_alloc_mask_from_features(
            df, min_eligible=9, prior_play_prob_threshold=0.20
        )
        
        # Key assertions for preventing smear:
        # 1. All 5 starters are eligible
        assert all(result[:5]), "All starters must be eligible"
        
        # 2. Key bench players (indices 5-8, prob >= 0.60) are eligible
        assert all(result[5:9]), "Key bench players must be eligible"
        
        # 3. Deep bench players (indices 9+, prob < 0.20) should NOT be eligible
        assert not any(result[9:]), "Deep bench players must NOT be eligible"
        
        # 4. Total eligible should be exactly 9 (no smear)
        assert result.sum() == 9, f"Expected 9 eligible, got {result.sum()}"
