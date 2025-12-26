"""Unit tests for Allocator F (POWER_POSTPROCESS)."""

import numpy as np
import pandas as pd
import pytest

from scripts.abtest_minutes_allocators import (
    PowerPostprocessDiagnostics,
    postprocess_power_minutes,
)


def _create_synthetic_rotalloc(
    n_players: int = 10,
    n_eligible: int = 8,
    flat_top: bool = True,
) -> pd.DataFrame:
    """Create synthetic RotAlloc output for testing.
    
    Args:
        n_players: Total players
        n_eligible: Players in rotation (get minutes)
        flat_top: If True, create flat top (low concentration)
    """
    game_id = "game_001"
    team_id = 1610612737
    
    player_ids = list(range(1, n_players + 1))
    
    # Create minutes distribution
    if flat_top:
        # Flat top distribution - all eligible get similar minutes
        base_minutes = 240.0 / n_eligible
        minutes = [base_minutes if i < n_eligible else 0.0 for i in range(n_players)]
    else:
        # Concentrated distribution - top players get more
        minutes = []
        remaining = 240.0
        for i in range(n_players):
            if i >= n_eligible:
                minutes.append(0.0)
            elif i < 3:  # Top 3 get most
                m = min(40.0, remaining)
                minutes.append(m)
                remaining -= m
            else:
                m = min(remaining / (n_eligible - i), remaining)
                minutes.append(m)
                remaining -= m
    
    df = pd.DataFrame({
        "game_id": [game_id] * n_players,
        "team_id": [team_id] * n_players,
        "player_id": player_ids,
        "minutes_mean": minutes,
        "eligible_flag": [1 if i < n_eligible else 0 for i in range(n_players)],
    })
    
    return df


class TestPowerPostprocessMinutes:
    """Tests for postprocess_power_minutes function."""

    def test_preserves_sum_240(self):
        """Power transform should preserve sum-to-240."""
        df_B = _create_synthetic_rotalloc()
        
        for p in [1.0, 1.1, 1.2, 1.3, 1.5, 2.0]:
            df_F, diag = postprocess_power_minutes(df_B, p=p, cap_max=48.0)
            
            team_sum = df_F["minutes_mean_F"].sum()
            assert abs(team_sum - 240.0) < 1e-6, f"Team sum {team_sum} != 240 for p={p}"

    def test_p1_equals_original(self):
        """p=1.0 should produce same minutes as input (up to numerical precision)."""
        df_B = _create_synthetic_rotalloc()
        
        df_F, diag = postprocess_power_minutes(df_B, p=1.0, cap_max=48.0)
        
        # With p=1.0, output should equal input (both sum to 240)
        np.testing.assert_array_almost_equal(
            df_F["minutes_mean_F"].values,
            df_B["minutes_mean"].values,
            decimal=5,
            err_msg="p=1.0 should not change the distribution",
        )

    def test_increases_concentration_when_p_gt_1(self):
        """Higher p should increase top5_sum (more concentration)."""
        df_B = _create_synthetic_rotalloc(flat_top=True)
        
        top5_sums = []
        for p in [1.0, 1.1, 1.2, 1.3, 1.5]:
            df_F, _ = postprocess_power_minutes(df_B, p=p, cap_max=48.0)
            top5 = df_F.nlargest(5, "minutes_mean_F")["minutes_mean_F"].sum()
            top5_sums.append(top5)
        
        # Each higher p should give higher top5_sum (more concentrated)
        for i in range(len(top5_sums) - 1):
            assert top5_sums[i] <= top5_sums[i + 1] + 0.01, \
                f"top5_sum should increase with p: {top5_sums}"

    def test_ineligible_stay_zero(self):
        """Players with minutes_mean=0 (ineligible) should stay at 0."""
        df_B = _create_synthetic_rotalloc(n_players=12, n_eligible=8)
        
        # Verify some players start with 0 minutes
        assert (df_B["minutes_mean"] == 0).any(), "Test setup: need some 0-minute players"
        
        df_F, diag = postprocess_power_minutes(df_B, p=1.3, cap_max=48.0)
        
        # Check that all originally zero players remain zero
        zero_mask = df_B["minutes_mean"] == 0
        assert (df_F.loc[zero_mask, "minutes_mean_F"] == 0).all(), \
            "Players with 0 minutes should stay at 0"
        
        # Diagnostics should show 0 ineligible with nonzero
        assert diag.ineligible_nonzero_count == 0

    def test_respects_cap_max_after_redistribution(self):
        """All minutes should be <= cap_max after redistribution."""
        # Create uneven distribution that will hit the cap
        df_B = pd.DataFrame({
            "game_id": ["g1"] * 8,
            "team_id": [1] * 8,
            "player_id": [1, 2, 3, 4, 5, 6, 7, 8],
            "minutes_mean": [50.0, 45.0, 40.0, 35.0, 30.0, 20.0, 15.0, 5.0],  # sum=240
            "eligible_flag": [1] * 8,
        })
        
        cap = 40.0
        df_F, diag = postprocess_power_minutes(df_B, p=1.5, cap_max=cap)
        
        max_minutes = df_F["minutes_mean_F"].max()
        assert max_minutes <= cap + 0.01, f"Max {max_minutes} exceeds cap {cap}"
        
        # Sum should still be 240
        assert abs(df_F["minutes_mean_F"].sum() - 240.0) < 1e-6

    def test_handles_all_zeros(self):
        """Should not crash when all minutes are zero (edge case)."""
        df_B = pd.DataFrame({
            "game_id": ["g1"] * 5,
            "team_id": [1] * 5,
            "player_id": [1, 2, 3, 4, 5],
            "minutes_mean": [0.0, 0.0, 0.0, 0.0, 0.0],
            "eligible_flag": [0] * 5,
        })
        
        df_F, diag = postprocess_power_minutes(df_B, p=1.2, cap_max=48.0)
        
        # Should not crash, output should be all zeros
        assert (df_F["minutes_mean_F"] == 0).all()
        assert diag.fallback_used_count >= 1


class TestPowerPostprocessDiagnostics:
    """Tests for PowerPostprocessDiagnostics dataclass."""

    def test_diagnostics_to_dict(self):
        """Diagnostics should serialize to dict correctly."""
        diag = PowerPostprocessDiagnostics(
            n_teams=2,
            n_players=24,
            p_value=1.2,
            cap_applied_count=3,
            ineligible_nonzero_count=0,
            fallback_used_count=0,
            max_redistribution_rounds=1,
            team_sum_dev_max=0.0,
        )
        
        d = diag.to_dict()
        
        assert d["n_teams"] == 2
        assert d["n_players"] == 24
        assert d["p_value"] == 1.2
        assert d["cap_applied_count"] == 3
        assert d["ineligible_nonzero_count"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
