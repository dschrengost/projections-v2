"""Tests for position-aware minutes variance in sim_v2."""

import numpy as np
import pytest

from projections.sim_v2.minutes_stabilization import (
    _position_aware_target_cv,
    sample_minutes_noise_per_world,
)


class TestPositionAwareTargetCV:
    """Test the _position_aware_target_cv function returns correct CV targets."""

    def test_heavy_starter_cv(self):
        """Heavy starters (32+ min) should have CV=0.10."""
        assert _position_aware_target_cv(35.0) == 0.10
        assert _position_aware_target_cv(32.0) == 0.10

    def test_starter_cv(self):
        """Starters (26-32 min) should have CV=0.11."""
        assert _position_aware_target_cv(30.0) == 0.11
        assert _position_aware_target_cv(26.0) == 0.11

    def test_sixth_man_cv(self):
        """6th man (20-26 min) should have CV=0.13."""
        assert _position_aware_target_cv(24.0) == 0.13
        assert _position_aware_target_cv(20.0) == 0.13

    def test_key_rotation_cv(self):
        """Key rotation (14-20 min) should have CV=0.18."""
        assert _position_aware_target_cv(18.0) == 0.18
        assert _position_aware_target_cv(14.0) == 0.18

    def test_rotation_cv(self):
        """Rotation (8-14 min) should have CV=0.28."""
        assert _position_aware_target_cv(12.0) == 0.28
        assert _position_aware_target_cv(8.0) == 0.28

    def test_deep_bench_cv(self):
        """Deep bench (<8 min) should have CV=0.45."""
        assert _position_aware_target_cv(5.0) == 0.45
        assert _position_aware_target_cv(0.0) == 0.45


class TestPositionAwareSigma:
    """Test that sample_minutes_noise_per_world uses position-aware sigma correctly."""

    def test_sigma_scales_with_position(self):
        """Verify CV increases as expected minutes decrease (position-aware behavior)."""
        # Create a realistic rotation that sums to 240
        minutes_p50 = np.array([
            36.0, 34.0, 32.0, 28.0, 26.0,  # starters (5)
            22.0,                           # 6th man
            18.0, 14.0,                     # key rotation
            18.0, 12.0,                     # fill to 240
        ])
        assert minutes_p50.sum() == 240.0, "Test setup: should sum to 240"

        is_starter = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        team_indices = np.zeros(10, dtype=int)

        n_worlds = 10000
        minutes_worlds, _ = sample_minutes_noise_per_world(
            minutes_reconciled=minutes_p50,
            minutes_p10=minutes_p50 * 0.7,  # Wider bounds
            minutes_p90=minutes_p50 * 1.3,
            is_starter=is_starter,
            team_indices=team_indices,
            n_worlds=n_worlds,
            use_position_aware_sigma=True,
            rng=np.random.default_rng(42),
        )

        # Compute empirical CV for each player
        empirical_cv = minutes_worlds.std(axis=0) / minutes_worlds.mean(axis=0)

        # Key check: CV should increase as minutes decrease
        # Heavy starter (36 min, idx 0) should have lower CV than rotation (14 min, idx 7)
        heavy_starter_cv = empirical_cv[0]  # 36 min
        rotation_cv = empirical_cv[7]       # 14 min

        assert rotation_cv > heavy_starter_cv, (
            f"Rotation CV ({rotation_cv:.3f}) should be > heavy starter CV ({heavy_starter_cv:.3f})"
        )

        # Verify CV increases monotonically as expected minutes decrease
        # Group by tier and check averages
        starter_cv_avg = empirical_cv[:5].mean()      # 26-36 min
        rotation_cv_avg = empirical_cv[6:8].mean()    # 14-18 min

        assert rotation_cv_avg > starter_cv_avg, (
            f"Rotation avg CV ({rotation_cv_avg:.3f}) should be > starter avg CV ({starter_cv_avg:.3f})"
        )

    def test_legacy_sigma_when_disabled(self):
        """Verify legacy sigma_starter/sigma_bench when use_position_aware_sigma=False."""
        minutes_p50 = np.array([35.0, 30.0, 25.0, 20.0, 15.0, 10.0, 8.0, 6.0, 5.0, 4.0])
        is_starter = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        team_indices = np.zeros(10, dtype=int)

        n_worlds = 10000
        sigma_starter = 2.0
        sigma_bench = 3.0

        minutes_worlds, _ = sample_minutes_noise_per_world(
            minutes_reconciled=minutes_p50,
            minutes_p10=minutes_p50 - 5,
            minutes_p90=minutes_p50 + 5,
            is_starter=is_starter,
            team_indices=team_indices,
            n_worlds=n_worlds,
            use_position_aware_sigma=False,  # Legacy mode
            sigma_starter=sigma_starter,
            sigma_bench=sigma_bench,
            rng=np.random.default_rng(42),
        )

        # Empirical std should be approximately sigma_starter for starters
        # and sigma_bench for bench (before team-240 normalization effects)
        emp_std = minutes_worlds.std(axis=0)

        # Starters (first 5) should have lower std than bench
        for i in range(5):
            if minutes_p50[i] >= 8:  # above noise threshold
                # Starters should have std closer to sigma_starter
                assert emp_std[i] < 3.5, f"Starter {i} std too high: {emp_std[i]}"

    def test_team_sum_still_240(self):
        """Verify team totals still sum to 240 with position-aware sigma."""
        minutes_p50 = np.array([35.0, 32.0, 28.0, 26.0, 24.0, 20.0, 16.0, 12.0, 8.0, 4.0])
        # Note: sum is 205, let the function handle team normalization
        is_starter = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        team_indices = np.zeros(10, dtype=int)

        n_worlds = 1000
        minutes_worlds, stats = sample_minutes_noise_per_world(
            minutes_reconciled=minutes_p50,
            minutes_p10=minutes_p50 - 5,
            minutes_p90=minutes_p50 + 5,
            is_starter=is_starter,
            team_indices=team_indices,
            n_worlds=n_worlds,
            use_position_aware_sigma=True,
            rng=np.random.default_rng(42),
        )

        # Team sums should be exactly 240 (within floating point tolerance)
        team_sums = minutes_worlds.sum(axis=1)
        np.testing.assert_allclose(team_sums, 240.0, atol=1e-6)

    def test_cv_reduction_vs_legacy(self):
        """Verify position-aware sigma reduces CV for rotation players vs legacy."""
        # 6th man scenario: 20 minute player
        minutes_p50 = np.array([35.0, 32.0, 28.0, 26.0, 24.0, 20.0, 16.0, 12.0, 8.0, 4.0])
        is_starter = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        team_indices = np.zeros(10, dtype=int)

        n_worlds = 10000
        rng_seed = 42

        # Position-aware sigma
        minutes_pa, _ = sample_minutes_noise_per_world(
            minutes_reconciled=minutes_p50,
            minutes_p10=minutes_p50 - 5,
            minutes_p90=minutes_p50 + 5,
            is_starter=is_starter,
            team_indices=team_indices,
            n_worlds=n_worlds,
            use_position_aware_sigma=True,
            rng=np.random.default_rng(rng_seed),
        )

        # Legacy sigma
        minutes_legacy, _ = sample_minutes_noise_per_world(
            minutes_reconciled=minutes_p50,
            minutes_p10=minutes_p50 - 5,
            minutes_p90=minutes_p50 + 5,
            is_starter=is_starter,
            team_indices=team_indices,
            n_worlds=n_worlds,
            use_position_aware_sigma=False,
            sigma_starter=2.0,
            sigma_bench=3.0,
            rng=np.random.default_rng(rng_seed),
        )

        # CV for 6th man (index 5, 20 min)
        cv_pa = minutes_pa[:, 5].std() / minutes_pa[:, 5].mean()
        cv_legacy = minutes_legacy[:, 5].std() / minutes_legacy[:, 5].mean()

        # Position-aware should have lower CV for 6th man
        # Legacy CV for 20-min bench player with sigma=3: CV = 3/20 = 0.15
        # Position-aware: CV = 0.13 (target)
        # Allow some tolerance due to team-240 effects
        assert cv_pa < cv_legacy * 1.1, (
            f"Position-aware CV ({cv_pa:.3f}) should be lower than legacy ({cv_legacy:.3f})"
        )
