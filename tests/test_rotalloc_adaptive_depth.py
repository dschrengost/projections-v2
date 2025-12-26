"""Tests for adaptive depth allocation in RotAlloc.

These tests verify the key invariants of the adaptive depth allocation:
1. Sum-to-240 constraint is maintained exactly
2. Fringe players are capped at fringe_cap_max
3. Core players get at least their fair share
4. 6th-8th man crush metric is bounded (protection against over-crushing)
5. bench_share_pred is clamped to [bench_share_min, bench_share_max]
"""

from __future__ import annotations

import numpy as np

from projections.models.rotalloc import (
    allocate_adaptive_depth,
    compute_bench_share_prior,
    compute_sixth_man_crush_metric,
)


class TestAllocateAdaptiveDepth:
    """Tests for the allocate_adaptive_depth function."""

    def test_sum_to_240_with_full_rotation(self) -> None:
        """Verify allocation sums exactly to 240 with a full rotation."""
        n = 12
        p_rot = np.array([0.95, 0.90, 0.85, 0.80, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15, 0.10])
        mu = np.array([35.0, 33.0, 31.0, 29.0, 27.0, 22.0, 18.0, 14.0, 10.0, 8.0, 5.0, 3.0])
        mask = np.ones(n, dtype=bool)

        minutes, diag = allocate_adaptive_depth(p_rot, mu, mask, bench_share_pred=0.15)

        assert abs(minutes.sum() - 240.0) < 1e-9
        assert diag["core_k"] >= 1
        assert diag["bench_share_actual"] >= 0.0

    def test_sum_to_240_with_injuries(self) -> None:
        """Verify allocation sums to 240 when some players are OUT (masked)."""
        n = 12
        p_rot = np.array([0.95, 0.90, 0.85, 0.80, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15, 0.10])
        mu = np.array([35.0, 33.0, 31.0, 29.0, 27.0, 22.0, 18.0, 14.0, 10.0, 8.0, 5.0, 3.0])
        mask = np.ones(n, dtype=bool)
        # Mark 3 players as OUT
        mask[[1, 3, 5]] = False

        minutes, diag = allocate_adaptive_depth(p_rot, mu, mask, bench_share_pred=0.20)

        assert abs(minutes.sum() - 240.0) < 1e-9
        # OUT players get 0 minutes
        assert (minutes[~mask] == 0.0).all()

    def test_fringe_players_capped(self) -> None:
        """Verify fringe players are capped at fringe_cap_max."""
        n = 10
        p_rot = np.linspace(0.95, 0.10, n)
        mu = np.linspace(35.0, 5.0, n)
        mask = np.ones(n, dtype=bool)
        fringe_cap = 12.0

        minutes, diag = allocate_adaptive_depth(
            p_rot, mu, mask,
            bench_share_pred=0.25,  # High bench share to get more fringe minutes
            fringe_cap_max=fringe_cap,
        )

        # Fringe players should be below fringe_cap_max
        core_k = diag["core_k"]
        fringe_minutes = minutes[core_k:]
        assert (fringe_minutes <= fringe_cap + 1e-9).all()

    def test_cap_max_respected(self) -> None:
        """Verify no player exceeds cap_max when feasible.

        Note: cap_max * n_players must be >= 240 for sum-to-240 to be feasible.
        With 10 players and cap_max=42, max = 420, so constraint is feasible.
        """
        n = 10  # Need enough players to make sum-to-240 feasible with cap
        p_rot = np.linspace(0.99, 0.50, n)
        mu = np.linspace(40.0, 20.0, n)
        mask = np.ones(n, dtype=bool)
        cap_max = 42.0

        minutes, _ = allocate_adaptive_depth(p_rot, mu, mask, cap_max=cap_max)

        # All players should respect cap_max when constraint is feasible
        assert (minutes <= cap_max + 1e-9).all()
        # Sum should still be 240
        assert abs(minutes.sum() - 240.0) < 1e-9

    def test_bench_share_clipping(self) -> None:
        """Verify bench_share_pred is clipped to [min, max]."""
        n = 10
        p_rot = np.linspace(0.95, 0.10, n)
        mu = np.linspace(35.0, 5.0, n)
        mask = np.ones(n, dtype=bool)

        # Test with extremely low bench_share
        _, diag_low = allocate_adaptive_depth(
            p_rot, mu, mask,
            bench_share_pred=-0.5,  # Below min
            bench_share_min=0.05,
        )
        assert diag_low["bench_share_actual"] >= 0.0

        # Test with extremely high bench_share
        _, diag_high = allocate_adaptive_depth(
            p_rot, mu, mask,
            bench_share_pred=0.99,  # Above max
            bench_share_max=0.35,
        )
        assert diag_high["bench_share_actual"] <= 0.35 + 1e-9

    def test_deterministic(self) -> None:
        """Verify allocation is deterministic."""
        n = 10
        p_rot = np.linspace(0.95, 0.10, n)
        mu = np.linspace(35.0, 5.0, n)
        mask = np.ones(n, dtype=bool)

        minutes1, _ = allocate_adaptive_depth(p_rot, mu, mask, bench_share_pred=0.15)
        minutes2, _ = allocate_adaptive_depth(p_rot, mu, mask, bench_share_pred=0.15)

        np.testing.assert_array_almost_equal(minutes1, minutes2)

    def test_single_player_gets_240(self) -> None:
        """Verify single player gets all 240 minutes."""
        p_rot = np.array([0.95])
        mu = np.array([35.0])
        mask = np.array([True])

        minutes, diag = allocate_adaptive_depth(p_rot, mu, mask)

        assert abs(minutes[0] - 240.0) < 1e-9
        assert diag["core_k"] == 1

    def test_higher_bench_share_reduces_core_minutes(self) -> None:
        """Verify higher bench_share reduces minutes for top players."""
        n = 10
        p_rot = np.linspace(0.95, 0.10, n)
        mu = np.linspace(35.0, 5.0, n)
        mask = np.ones(n, dtype=bool)

        minutes_low, _ = allocate_adaptive_depth(p_rot, mu, mask, bench_share_pred=0.10)
        minutes_high, _ = allocate_adaptive_depth(p_rot, mu, mask, bench_share_pred=0.30)

        # Top 3 players should have fewer minutes with higher bench_share
        assert minutes_high[:3].sum() < minutes_low[:3].sum()


class TestComputeBenchSharePrior:
    """Tests for the compute_bench_share_prior function."""

    def test_default_value_with_no_context(self) -> None:
        """Verify default bench_share is returned when no context is available."""
        bs = compute_bench_share_prior(
            team_bench_share_avg=None,
            spread=None,
            total=None,
            out_count=0,
            league_bench_share=0.15,
        )
        assert abs(bs - 0.15) < 1e-9

    def test_injuries_increase_bench_share(self) -> None:
        """Verify OUT players increase bench_share (minutes spread more widely)."""
        bs_no_out = compute_bench_share_prior(
            team_bench_share_avg=None, spread=None, total=None, out_count=0,
            league_bench_share=0.15, out_coef=0.015,
        )
        bs_with_out = compute_bench_share_prior(
            team_bench_share_avg=None, spread=None, total=None, out_count=3,
            league_bench_share=0.15, out_coef=0.015,
        )
        assert bs_with_out > bs_no_out

    def test_favorites_increase_bench_share(self) -> None:
        """Verify favorites (negative spread) rest starters more."""
        bs_underdog = compute_bench_share_prior(
            team_bench_share_avg=None, spread=10.0, total=None, out_count=0,
            league_bench_share=0.15, spread_coef=-0.002,
        )
        bs_favorite = compute_bench_share_prior(
            team_bench_share_avg=None, spread=-10.0, total=None, out_count=0,
            league_bench_share=0.15, spread_coef=-0.002,
        )
        # Favorites (negative spread) should have higher bench_share
        # spread_coef is negative, so -10 * -0.002 = +0.02 added to bench_share
        assert bs_favorite > bs_underdog

    def test_team_prior_weight(self) -> None:
        """Verify team_prior_weight blends team history with context."""
        # Team that plays deep bench more
        bs_deep_team = compute_bench_share_prior(
            team_bench_share_avg=0.25,
            spread=None, total=None, out_count=0,
            league_bench_share=0.15,
            team_prior_weight=0.6,
        )
        # Team that plays short rotation
        bs_shallow_team = compute_bench_share_prior(
            team_bench_share_avg=0.08,
            spread=None, total=None, out_count=0,
            league_bench_share=0.15,
            team_prior_weight=0.6,
        )
        assert bs_deep_team > bs_shallow_team


class TestComputeSixthManCrushMetric:
    """Tests for the compute_sixth_man_crush_metric function."""

    def test_no_crush_when_6th_man_gets_full_mu(self) -> None:
        """Verify crush metric is ~1.0 when 6th-8th man gets their expected minutes."""
        n = 10
        mu = np.array([35.0, 33.0, 31.0, 29.0, 27.0, 22.0, 18.0, 14.0, 10.0, 5.0])
        minutes = mu.copy()  # Everyone gets exactly their mu
        proxy = np.ones(n)  # Equal weights
        eligible = np.ones(n, dtype=bool)

        crush = compute_sixth_man_crush_metric(minutes, mu, proxy, eligible)

        assert abs(crush - 1.0) < 0.1  # ~1.0 means no crush

    def test_high_crush_when_6th_man_gets_few_minutes(self) -> None:
        """Verify high crush when 6th-8th man gets significantly fewer minutes."""
        n = 10
        mu = np.array([35.0, 33.0, 31.0, 29.0, 27.0, 22.0, 18.0, 14.0, 10.0, 5.0])
        # Top 5 get more, 6th-8th get crushed
        minutes = np.array([40.0, 38.0, 36.0, 34.0, 32.0, 11.0, 9.0, 7.0, 6.0, 5.0])
        proxy = np.linspace(1.0, 0.1, n)  # Higher proxy for top players
        eligible = np.ones(n, dtype=bool)

        crush = compute_sixth_man_crush_metric(minutes, mu, proxy, eligible)

        # Crush < 1.0 indicates 6th-8th man got fewer minutes than expected
        assert crush < 0.7

    def test_handles_small_team(self) -> None:
        """Verify crush metric handles teams with < 8 players."""
        n = 5
        mu = np.array([35.0, 33.0, 31.0, 29.0, 27.0])
        minutes = mu.copy()
        proxy = np.ones(n)
        eligible = np.ones(n, dtype=bool)

        crush = compute_sixth_man_crush_metric(minutes, mu, proxy, eligible)

        # Should return NaN or handle gracefully for small teams
        assert np.isnan(crush) or crush >= 0.0
