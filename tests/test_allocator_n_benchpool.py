"""Tests for allocate_mixture_benchpool (Allocator N: MIXTURE_BENCHPOOL).

These tests verify the core properties of the benchpool allocator:
1. Sum-to-240 per team
2. Mask (OUT players get 0 minutes)
3. Core/bench pool split behavior
"""

import numpy as np
import pytest


# Import the function to test
from scripts.abtest_minutes_allocators import (
    allocate_mixture_benchpool,
    BenchpoolDiagnostics,
)


class TestAllocateMixtureBenchpool:
    """Test suite for allocate_mixture_benchpool function."""

    def test_sum_to_240_single_team(self):
        """Test that minutes sum to 240 for a single team."""
        # 10 players on one team
        expected_minutes = np.array([30.0, 28.0, 25.0, 22.0, 20.0, 15.0, 10.0, 8.0, 5.0, 2.0])
        mask_inactive = np.array([False] * 10)
        team_ids = np.array([1] * 10)

        minutes, diag = allocate_mixture_benchpool(
            expected_minutes,
            mask_inactive,
            team_ids,
            bench_pool=80.0,
            core_k=6,
            cap_max=48.0,
        )

        assert minutes.shape == expected_minutes.shape
        assert np.isclose(minutes.sum(), 240.0, atol=1e-6), f"Sum = {minutes.sum()}, expected 240.0"
        assert diag.n_teams == 1
        assert diag.n_players == 10

    def test_sum_to_240_multiple_teams(self):
        """Test that each team sums to 240 with multiple teams."""
        # 2 teams, 8 players each
        expected_minutes = np.array([
            # Team A
            35.0, 30.0, 28.0, 25.0, 20.0, 15.0, 10.0, 5.0,
            # Team B  
            40.0, 32.0, 26.0, 22.0, 18.0, 12.0, 8.0, 4.0,
        ])
        mask_inactive = np.array([False] * 16)
        team_ids = np.array([1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2])

        minutes, diag = allocate_mixture_benchpool(
            expected_minutes,
            mask_inactive,
            team_ids,
            bench_pool=80.0,
            core_k=6,
            cap_max=48.0,
        )

        # Check each team sums to 240
        team_a_sum = minutes[:8].sum()
        team_b_sum = minutes[8:].sum()
        
        assert np.isclose(team_a_sum, 240.0, atol=1e-6), f"Team A sum = {team_a_sum}, expected 240.0"
        assert np.isclose(team_b_sum, 240.0, atol=1e-6), f"Team B sum = {team_b_sum}, expected 240.0"
        assert diag.n_teams == 2
        assert diag.n_players == 16

    def test_mask_inactive_players_get_zero(self):
        """Test that masked (OUT/inactive) players receive 0 minutes."""
        expected_minutes = np.array([30.0, 25.0, 20.0, 15.0, 10.0, 8.0, 5.0, 3.0])
        # Players at indices 2 and 6 are OUT
        mask_inactive = np.array([False, False, True, False, False, False, True, False])
        team_ids = np.array([1] * 8)

        minutes, diag = allocate_mixture_benchpool(
            expected_minutes,
            mask_inactive,
            team_ids,
            bench_pool=60.0,
            core_k=5,
            cap_max=48.0,
        )

        # Inactive players get 0
        assert minutes[2] == 0.0, f"Masked player at idx 2 got {minutes[2]} minutes"
        assert minutes[6] == 0.0, f"Masked player at idx 6 got {minutes[6]} minutes"
        
        # Active players get positive minutes
        active_indices = [0, 1, 3, 4, 5, 7]
        for idx in active_indices:
            assert minutes[idx] > 0, f"Active player at idx {idx} got {minutes[idx]} minutes"
        
        # Still sums to 240
        assert np.isclose(minutes.sum(), 240.0, atol=1e-6)

    def test_all_inactive_fallback(self):
        """Test fallback behavior when all players are inactive."""
        expected_minutes = np.array([30.0, 25.0, 20.0, 15.0, 10.0])
        mask_inactive = np.array([True, True, True, True, True])
        team_ids = np.array([1] * 5)

        minutes, diag = allocate_mixture_benchpool(
            expected_minutes,
            mask_inactive,
            team_ids,
            bench_pool=80.0,
            core_k=4,
            cap_max=48.0,
        )

        # Fallback: top player by E_minutes gets all 240
        assert minutes[0] == 240.0  # Player with highest E_minutes
        assert minutes.sum() == 240.0
        assert diag.fallback_count >= 1

    def test_core_bench_pool_split(self):
        """Test that core players get core_pool and bench players get bench_pool."""
        # Setup: 8 players, core_k=4, bench_pool=60 means core_pool=180
        expected_minutes = np.array([40.0, 35.0, 30.0, 25.0, 10.0, 8.0, 5.0, 3.0])
        mask_inactive = np.array([False] * 8)
        team_ids = np.array([1] * 8)

        minutes, diag = allocate_mixture_benchpool(
            expected_minutes,
            mask_inactive,
            team_ids,
            bench_pool=60.0,  # core_pool = 240 - 60 = 180
            core_k=4,
            cap_max=48.0,
        )

        # Core players (indices 0-3 by E_minutes ranking) should get more minutes
        core_minutes = minutes[:4].sum()
        bench_minutes = minutes[4:].sum()
        
        # After cap/redistribution, total still 240
        assert np.isclose(core_minutes + bench_minutes, 240.0, atol=1e-6)
        
        # Core should get significantly more than bench
        assert core_minutes > bench_minutes, (
            f"Core minutes ({core_minutes}) should exceed bench minutes ({bench_minutes})"
        )

    def test_cap_max_applied(self):
        """Test that cap_max is enforced and excess is redistributed."""
        # One player with very high E_minutes should get capped
        expected_minutes = np.array([50.0, 10.0, 8.0, 6.0, 4.0, 2.0])
        mask_inactive = np.array([False] * 6)
        team_ids = np.array([1] * 6)

        cap_max = 40.0
        minutes, diag = allocate_mixture_benchpool(
            expected_minutes,
            mask_inactive,
            team_ids,
            bench_pool=40.0,
            core_k=4,
            cap_max=cap_max,
        )

        # No player exceeds cap_max
        assert np.all(minutes <= cap_max + 1e-6), f"Max minutes = {minutes.max()}, expected <= {cap_max}"
        
        # Still sums to 240
        assert np.isclose(minutes.sum(), 240.0, atol=1e-6)

    def test_diagnostics_populated(self):
        """Test that diagnostics are correctly populated."""
        expected_minutes = np.array([30.0, 25.0, 20.0, 15.0, 10.0, 8.0, 5.0, 3.0])
        mask_inactive = np.array([False] * 8)
        team_ids = np.array([1] * 8)

        minutes, diag = allocate_mixture_benchpool(
            expected_minutes,
            mask_inactive,
            team_ids,
            bench_pool=80.0,
            core_k=5,
            cap_max=48.0,
        )

        assert isinstance(diag, BenchpoolDiagnostics)
        assert diag.bench_pool == 80.0
        assert diag.core_k == 5
        assert diag.n_teams == 1
        assert diag.n_players == 8
        assert diag.mean_core_size > 0
        assert diag.team_sum_dev_max < 1e-6  # Should be ~0 for valid allocation

    def test_core_k_larger_than_active_players(self):
        """Test behavior when core_k exceeds number of active players."""
        # Only 4 active players but core_k=6
        expected_minutes = np.array([30.0, 25.0, 20.0, 15.0, 10.0, 8.0])
        mask_inactive = np.array([False, False, False, False, True, True])
        team_ids = np.array([1] * 6)

        minutes, diag = allocate_mixture_benchpool(
            expected_minutes,
            mask_inactive,
            team_ids,
            bench_pool=60.0,
            core_k=6,  # More than 4 active players
            cap_max=48.0,
        )

        # Should still work, all 4 active players become "core"
        assert np.isclose(minutes.sum(), 240.0, atol=1e-6)
        assert minutes[4] == 0.0  # Inactive
        assert minutes[5] == 0.0  # Inactive
        assert diag.mean_core_size == 4.0  # Limited to actual active count

    def test_empty_team(self):
        """Test with no players (edge case)."""
        expected_minutes = np.array([])
        mask_inactive = np.array([], dtype=bool)
        team_ids = np.array([])

        minutes, diag = allocate_mixture_benchpool(
            expected_minutes,
            mask_inactive,
            team_ids,
            bench_pool=80.0,
            core_k=6,
            cap_max=48.0,
        )

        assert len(minutes) == 0
        assert diag.n_teams == 0
        assert diag.n_players == 0


class TestBenchpoolDiagnostics:
    """Test the BenchpoolDiagnostics dataclass."""

    def test_to_dict(self):
        """Test that to_dict produces valid dictionary."""
        diag = BenchpoolDiagnostics(
            n_teams=1,
            n_players=10,
            bench_pool=80.0,
            core_k=6,
        )
        
        d = diag.to_dict()
        assert isinstance(d, dict)
        assert d["n_teams"] == 1
        assert d["n_players"] == 10
        assert d["bench_pool"] == 80.0
        assert d["core_k"] == 6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
