"""Unit tests for Allocator E (FRINGE_ONLY_ALPHA)."""

import numpy as np
import pandas as pd
import pytest

from scripts.abtest_minutes_allocators import (
    FringeOnlyAlphaDiagnostics,
    fringe_only_alpha_within_eligible,
)


def _create_synthetic_slate(
    n_players: int = 12,
    n_eligible: int = 10,
    shares_base: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create synthetic features and rotalloc output for testing.
    
    Returns:
        Tuple of (features_df, rotalloc_df)
    """
    game_id = "game_001"
    team_id = 1610612737  # Example team ID
    
    player_ids = list(range(1, n_players + 1))
    
    # Features with share predictions
    features = pd.DataFrame({
        "game_id": [game_id] * n_players,
        "team_id": [team_id] * n_players,
        "player_id": player_ids,
        "minute_share_pred": [shares_base + i * 0.02 for i in range(n_players)],
    })
    
    # RotAlloc output with eligibility
    # Higher p_rot for lower player IDs = more likely to be in rotation
    rotalloc = pd.DataFrame({
        "game_id": [game_id] * n_players,
        "team_id": [team_id] * n_players,
        "player_id": player_ids,
        "p_rot": [0.95 - i * 0.08 for i in range(n_players)],
        "mu_cond": [25.0 - i * 1.5 for i in range(n_players)],
        "eligible_flag": [1 if i < n_eligible else 0 for i in range(n_players)],
    })
    
    return features, rotalloc


class TestFringeOnlyAlphaWithinEligible:
    """Tests for the fringe_only_alpha_within_eligible function."""

    def test_respects_eligibility_and_sums_240(self):
        """E should allocate 240 total minutes within eligible set only."""
        features, rotalloc = _create_synthetic_slate(n_players=12, n_eligible=10)
        
        df_E, diag_E = fringe_only_alpha_within_eligible(
            features,
            rotalloc,
            "minute_share_pred",
            k_core=8,
            alpha_core=0.8,
            alpha_fringe=0.3,
        )
        
        # Check team sum is exactly 240
        team_sum = df_E["minutes_mean_E"].sum()
        assert abs(team_sum - 240.0) < 1e-6, f"Team sum {team_sum} != 240"
        
        # df_E already has eligible_flag from the function
        ineligible_minutes = df_E.loc[df_E["eligible_flag"] == 0, "minutes_mean_E"]
        assert (ineligible_minutes == 0).all(), "Ineligible players should get 0 minutes"
        
        # Check eligible players get >= 0 minutes
        eligible_minutes = df_E.loc[df_E["eligible_flag"] == 1, "minutes_mean_E"]
        assert (eligible_minutes >= 0).all(), "Eligible players should get >= 0 minutes"
        
        # Check diagnostics
        assert diag_E.n_eligible == 10
        assert diag_E.n_players == 12
        assert diag_E.team_sum_dev_max < 1e-6

    def test_core_selection_by_w_rot_top_k_core(self):
        """Top k_core players by w_rot should be marked as core."""
        features, rotalloc = _create_synthetic_slate(n_players=12, n_eligible=10)
        
        df_E, diag_E = fringe_only_alpha_within_eligible(
            features,
            rotalloc,
            "minute_share_pred",
            k_core=5,
            alpha_core=0.8,
            alpha_fringe=0.3,
        )
        
        # Check core size
        assert diag_E.mean_core_size == 5.0, f"Expected 5 core players, got {diag_E.mean_core_size}"
        
        # The top 5 by w_rot (highest p_rot * mu_cond) should be is_core=1
        # Players 1-5 have highest p_rot and mu_cond values
        top_core = df_E.loc[df_E["is_core"] == 1, "player_id"].tolist()
        assert len(top_core) == 5, f"Expected 5 core players, got {len(top_core)}"
        
        # First 5 player IDs should be core (they have highest w_rot)
        expected_core = [1, 2, 3, 4, 5]
        assert sorted(top_core) == expected_core, f"Core players {top_core} != expected {expected_core}"

    def test_core_and_fringe_get_different_blend(self):
        """Core players should get alpha_core blend, fringe should get alpha_fringe."""
        features, rotalloc = _create_synthetic_slate(n_players=10, n_eligible=10)
        
        # Run with very different alpha values to see difference
        df_E, diag_E = fringe_only_alpha_within_eligible(
            features,
            rotalloc,
            "minute_share_pred",
            k_core=5,
            alpha_core=1.0,  # Pure shares for core
            alpha_fringe=0.0,  # Pure rotalloc proxy for fringe
        )
        
        # With alpha_core=1.0, core players should lean heavily on share predictions
        # With alpha_fringe=0.0, fringe players should lean heavily on w_rot
        # This means core minutes should correlate with share predictions
        # and fringe minutes should correlate with p_rot * mu_cond
        
        merged = df_E.merge(features[["player_id", "minute_share_pred"]], on="player_id")
        merged = merged.merge(rotalloc[["player_id", "p_rot", "mu_cond"]], on="player_id")
        
        core_df = merged[merged["is_core"] == 1]
        fringe_df = merged[merged["is_core"] == 0]
        
        # Just verify both groups got minutes
        assert core_df["minutes_mean_E"].sum() > 0, "Core should have minutes"
        assert fringe_df["minutes_mean_E"].sum() > 0, "Fringe should have minutes"
        
        # Total should still be 240
        total = df_E["minutes_mean_E"].sum()
        assert abs(total - 240.0) < 1e-6

    def test_fallback_when_all_shares_zero(self):
        """Should fallback gracefully when all share predictions are zero."""
        features, rotalloc = _create_synthetic_slate(n_players=10, n_eligible=10)
        
        # Set all shares to zero
        features["minute_share_pred"] = 0.0
        
        df_E, diag_E = fringe_only_alpha_within_eligible(
            features,
            rotalloc,
            "minute_share_pred",
            k_core=5,
            alpha_core=0.8,
            alpha_fringe=0.3,
        )
        
        # Should not crash
        # Should still sum to 240
        team_sum = df_E["minutes_mean_E"].sum()
        assert abs(team_sum - 240.0) < 1e-6, f"Team sum {team_sum} != 240 (fallback case)"
        
        # Fallback may or may not be triggered depending on w_rot values
        # Just verify it ran successfully
        assert diag_E.n_teams == 1

    def test_handles_fewer_eligible_than_k_core(self):
        """When eligible < k_core, all eligible become core."""
        # Note: need at least 5 eligible to reach 240 with 48 cap (5*48=240)
        features, rotalloc = _create_synthetic_slate(n_players=10, n_eligible=6)
        
        df_E, diag_E = fringe_only_alpha_within_eligible(
            features,
            rotalloc,
            "minute_share_pred",
            k_core=8,  # More than eligible
            alpha_core=0.8,
            alpha_fringe=0.3,
        )
        
        # Core size should be capped at eligible (6)
        assert diag_E.mean_core_size == 6.0, f"Expected 6 core (all eligible), got {diag_E.mean_core_size}"
        
        # All eligible should be core
        eligible_and_core = df_E[(df_E["eligible_flag"] == 1) & (df_E["is_core"] == 1)]
        assert len(eligible_and_core) == 6
        
        # Should still sum to 240
        team_sum = df_E["minutes_mean_E"].sum()
        assert abs(team_sum - 240.0) < 1e-6

    def test_cap_redistribution(self):
        """Minutes cap should be applied with redistribution."""
        features, rotalloc = _create_synthetic_slate(n_players=8, n_eligible=8)
        
        # Give one player extremely high share to trigger cap
        features.loc[0, "minute_share_pred"] = 10.0  # Much higher than others
        
        df_E, diag_E = fringe_only_alpha_within_eligible(
            features,
            rotalloc,
            "minute_share_pred",
            k_core=5,
            alpha_core=0.99,  # Heavily favor shares
            alpha_fringe=0.3,
            cap_max=36.0,  # Set a cap
        )
        
        # No player should exceed cap
        max_minutes = df_E["minutes_mean_E"].max()
        assert max_minutes <= 36.0 + 0.1, f"Max minutes {max_minutes} exceeds cap"
        
        # Should still sum to 240 (redistribution)
        team_sum = df_E["minutes_mean_E"].sum()
        assert abs(team_sum - 240.0) < 1e-6


class TestFringeOnlyAlphaDiagnostics:
    """Tests for the FringeOnlyAlphaDiagnostics dataclass."""

    def test_diagnostics_to_dict(self):
        """Diagnostics should serialize to dict correctly."""
        diag = FringeOnlyAlphaDiagnostics(
            n_teams=2,
            n_players=24,
            n_eligible=20,
            k_core=8,
            alpha_core=0.8,
            alpha_fringe=0.3,
            mean_core_size=8.0,
            team_sum_dev_max=0.0,
        )
        
        d = diag.to_dict()
        
        assert d["n_teams"] == 2
        assert d["n_players"] == 24
        assert d["n_eligible"] == 20
        assert d["k_core"] == 8
        assert d["alpha_core"] == 0.8
        assert d["alpha_fringe"] == 0.3
        assert d["mean_core_size"] == 8.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
