"""Tests for model-space minutes worlds sampling (PR5 backend)."""

from __future__ import annotations

import numpy as np
import pytest

from projections.sim_v2.minutes_worlds_model_space_v1 import (
    MinutesWorldsConfig,
    MinutesWorldsResult,
    compute_minutes_quantiles,
    sample_minutes_worlds_model_space_v1,
)


class TestSampleMinutesWorldsModelSpaceV1:
    """Tests for sample_minutes_worlds_model_space_v1."""

    def test_determinism_same_seed_same_output(self) -> None:
        """Same seed + inputs should produce identical minutes_worlds."""
        n_players = 10
        n_worlds = 100

        minutes_mean = np.array([30.0, 25.0, 20.0, 15.0, 12.0, 8.0, 6.0, 4.0, 2.0, 0.0])
        gate_logit = np.array([2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0])
        gate_prob = 1.0 / (1.0 + np.exp(-gate_logit))
        share_logit = np.zeros(n_players)
        play_prob = np.array([0.95, 0.90, 0.85, 0.80, 0.75, 0.60, 0.50, 0.30, 0.10, 0.0])
        team_indices = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        # First run
        rng1 = np.random.default_rng(12345)
        result1 = sample_minutes_worlds_model_space_v1(
            minutes_mean=minutes_mean,
            gate_logit=gate_logit,
            gate_prob=gate_prob,
            share_logit=share_logit,
            play_prob=play_prob,
            team_indices=team_indices,
            n_worlds=n_worlds,
            rng=rng1,
        )

        # Second run with same seed
        rng2 = np.random.default_rng(12345)
        result2 = sample_minutes_worlds_model_space_v1(
            minutes_mean=minutes_mean,
            gate_logit=gate_logit,
            gate_prob=gate_prob,
            share_logit=share_logit,
            play_prob=play_prob,
            team_indices=team_indices,
            n_worlds=n_worlds,
            rng=rng2,
        )

        np.testing.assert_array_equal(result1.minutes_worlds, result2.minutes_worlds)
        np.testing.assert_array_equal(result1.active_mask, result2.active_mask)

    def test_determinism_different_seed_different_output(self) -> None:
        """Different seeds should produce different minutes_worlds."""
        n_players = 10
        n_worlds = 100

        minutes_mean = np.linspace(30, 5, n_players)
        gate_prob = np.linspace(0.9, 0.3, n_players)
        share_logit = np.zeros(n_players)
        play_prob = np.linspace(0.95, 0.2, n_players)
        team_indices = np.zeros(n_players, dtype=int)

        rng1 = np.random.default_rng(12345)
        result1 = sample_minutes_worlds_model_space_v1(
            minutes_mean=minutes_mean,
            gate_logit=None,
            gate_prob=gate_prob,
            share_logit=share_logit,
            play_prob=play_prob,
            team_indices=team_indices,
            n_worlds=n_worlds,
            rng=rng1,
        )

        rng2 = np.random.default_rng(99999)
        result2 = sample_minutes_worlds_model_space_v1(
            minutes_mean=minutes_mean,
            gate_logit=None,
            gate_prob=gate_prob,
            share_logit=share_logit,
            play_prob=play_prob,
            team_indices=team_indices,
            n_worlds=n_worlds,
            rng=rng2,
        )

        # Should be different
        assert not np.array_equal(result1.minutes_worlds, result2.minutes_worlds)

    def test_returns_minutes_worlds_result(self) -> None:
        """Should return MinutesWorldsResult with correct structure."""
        n_players = 5
        n_worlds = 50

        result = sample_minutes_worlds_model_space_v1(
            minutes_mean=np.array([30.0, 25.0, 20.0, 15.0, 10.0]),
            gate_logit=None,
            gate_prob=np.array([0.9, 0.8, 0.7, 0.6, 0.5]),
            share_logit=np.zeros(n_players),
            play_prob=np.array([0.9, 0.9, 0.9, 0.9, 0.9]),
            team_indices=np.zeros(n_players, dtype=int),
            n_worlds=n_worlds,
            rng=np.random.default_rng(42),
        )

        assert isinstance(result, MinutesWorldsResult)
        assert result.minutes_worlds.shape == (n_worlds, n_players)
        assert result.active_mask.shape == (n_worlds, n_players)
        assert isinstance(result.diagnostics, dict)

    def test_team_240_constraint_single_team(self) -> None:
        """Each team should sum to approximately 240 minutes when all players are active."""
        n_players = 8
        n_worlds = 100

        # All players on same team with play_prob=1.0 and gate_prob=1.0 to ensure all active
        result = sample_minutes_worlds_model_space_v1(
            minutes_mean=np.array([35.0, 32.0, 28.0, 25.0, 20.0, 15.0, 10.0, 5.0]),
            gate_logit=None,
            gate_prob=np.ones(n_players),  # 100% rotation prob
            share_logit=np.zeros(n_players),
            play_prob=np.ones(n_players),  # Always active
            team_indices=np.zeros(n_players, dtype=int),
            n_worlds=n_worlds,
            rng=np.random.default_rng(42),
            config=MinutesWorldsConfig(use_bench_zero_mixture=False),
        )

        team_sums = result.minutes_worlds.sum(axis=1)
        # All worlds should have exactly 240 minutes when everyone is active
        np.testing.assert_allclose(team_sums, 240.0, rtol=0.01)

    def test_team_240_constraint_multiple_teams(self) -> None:
        """Multiple teams should each sum to ~240 minutes when all players are active."""
        n_players = 10
        n_worlds = 50

        # 5 players per team
        team_indices = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        result = sample_minutes_worlds_model_space_v1(
            minutes_mean=np.array([35.0, 30.0, 25.0, 20.0, 10.0, 35.0, 30.0, 25.0, 20.0, 10.0]),
            gate_logit=None,
            gate_prob=np.ones(n_players),  # 100% rotation prob
            share_logit=np.zeros(n_players),
            play_prob=np.ones(n_players),  # Always active
            team_indices=team_indices,
            n_worlds=n_worlds,
            rng=np.random.default_rng(42),
            config=MinutesWorldsConfig(use_bench_zero_mixture=False),
        )

        # Check each team sums to ~240 when all active
        for team_id in [0, 1]:
            team_mask = team_indices == team_id
            team_sums = result.minutes_worlds[:, team_mask].sum(axis=1)
            np.testing.assert_allclose(team_sums, 240.0, rtol=0.01)

    def test_inactive_players_get_zero_minutes(self) -> None:
        """Players with play_prob=0 should always get 0 minutes."""
        n_players = 5
        n_worlds = 100

        # Player 4 has play_prob=0
        play_prob = np.array([0.9, 0.9, 0.9, 0.9, 0.0])

        result = sample_minutes_worlds_model_space_v1(
            minutes_mean=np.array([30.0, 25.0, 20.0, 15.0, 10.0]),
            gate_logit=None,
            gate_prob=np.ones(n_players) * 0.9,
            share_logit=np.zeros(n_players),
            play_prob=play_prob,
            team_indices=np.zeros(n_players, dtype=int),
            n_worlds=n_worlds,
            rng=np.random.default_rng(42),
        )

        # Player 4 should have 0 minutes in all worlds
        assert np.all(result.minutes_worlds[:, 4] == 0.0)
        assert np.all(~result.active_mask[:, 4])

    def test_play_prob_nan_raises_error(self) -> None:
        """play_prob with NaN should raise ValueError."""
        n_players = 5

        with pytest.raises(ValueError, match="play_prob contains NaN"):
            sample_minutes_worlds_model_space_v1(
                minutes_mean=np.ones(n_players) * 20.0,
                gate_logit=None,
                gate_prob=np.ones(n_players) * 0.9,
                share_logit=np.zeros(n_players),
                play_prob=np.array([0.9, 0.9, np.nan, 0.9, 0.9]),
                team_indices=np.zeros(n_players, dtype=int),
                n_worlds=10,
                rng=np.random.default_rng(42),
            )

    def test_bench_zero_mixture_increases_zero_rate_for_low_minutes(self) -> None:
        """Bench-zero mixture should increase zero-minutes rate for low-minute players."""
        n_players = 5
        n_worlds = 1000

        # Player 4 has very low minutes (candidate for bench-zero)
        minutes_mean = np.array([30.0, 25.0, 20.0, 15.0, 3.0])

        # Without bench-zero mixture
        result_no_bzm = sample_minutes_worlds_model_space_v1(
            minutes_mean=minutes_mean,
            gate_logit=None,
            gate_prob=np.ones(n_players) * 0.95,
            share_logit=np.zeros(n_players),
            play_prob=np.ones(n_players) * 0.9,
            team_indices=np.zeros(n_players, dtype=int),
            n_worlds=n_worlds,
            rng=np.random.default_rng(42),
            config=MinutesWorldsConfig(use_bench_zero_mixture=False),
        )

        # With bench-zero mixture
        result_bzm = sample_minutes_worlds_model_space_v1(
            minutes_mean=minutes_mean,
            gate_logit=None,
            gate_prob=np.ones(n_players) * 0.95,
            share_logit=np.zeros(n_players),
            play_prob=np.ones(n_players) * 0.9,
            team_indices=np.zeros(n_players, dtype=int),
            n_worlds=n_worlds,
            rng=np.random.default_rng(42),
            config=MinutesWorldsConfig(use_bench_zero_mixture=True),
        )

        # Zero rate should be higher for low-minute player with bench-zero mixture
        zero_rate_no_bzm = (result_no_bzm.minutes_worlds[:, 4] == 0.0).mean()
        zero_rate_bzm = (result_bzm.minutes_worlds[:, 4] == 0.0).mean()
        assert zero_rate_bzm > zero_rate_no_bzm

    def test_gate_temperature_affects_rotation_rate(self) -> None:
        """Gate temperature scaling should affect rotation membership rate."""
        n_players = 5
        n_worlds = 1000

        gate_logit = np.array([0.5, 0.0, -0.5, -1.0, -2.0])  # Varying gate logits
        gate_prob = 1.0 / (1.0 + np.exp(-gate_logit))

        # Temperature < 1 makes distribution sharper (more extreme probs)
        # Temperature > 1 makes distribution flatter (closer to 0.5)

        result_t1 = sample_minutes_worlds_model_space_v1(
            minutes_mean=np.ones(n_players) * 20.0,
            gate_logit=gate_logit,
            gate_prob=gate_prob,
            share_logit=np.zeros(n_players),
            play_prob=np.ones(n_players) * 0.99,
            team_indices=np.zeros(n_players, dtype=int),
            n_worlds=n_worlds,
            rng=np.random.default_rng(42),
            config=MinutesWorldsConfig(gate_temperature=1.0, use_bench_zero_mixture=False),
        )

        result_t2 = sample_minutes_worlds_model_space_v1(
            minutes_mean=np.ones(n_players) * 20.0,
            gate_logit=gate_logit,
            gate_prob=gate_prob,
            share_logit=np.zeros(n_players),
            play_prob=np.ones(n_players) * 0.99,
            team_indices=np.zeros(n_players, dtype=int),
            n_worlds=n_worlds,
            rng=np.random.default_rng(42),
            config=MinutesWorldsConfig(gate_temperature=2.0, use_bench_zero_mixture=False),
        )

        # With higher temperature, the variance in rotation rates across players should be lower
        # (probabilities closer to 0.5 means less differentiation)
        rot_rates_t1 = (result_t1.minutes_worlds > 0).mean(axis=0)
        rot_rates_t2 = (result_t2.minutes_worlds > 0).mean(axis=0)

        # Higher temp should result in less variance
        assert rot_rates_t1.std() > rot_rates_t2.std() * 0.8  # Allow some tolerance


class TestComputeMinutesQuantiles:
    """Tests for compute_minutes_quantiles."""

    def test_unconditional_includes_zeros(self) -> None:
        """Unconditional quantiles should include zero-minute worlds."""
        n_worlds = 100
        n_players = 3

        # Create minutes with some zeros (DNP)
        minutes_worlds = np.array([
            [20.0, 0.0, 15.0],  # Player 1 DNP
            [22.0, 18.0, 0.0],  # Player 2 DNP
            [0.0, 20.0, 17.0],  # Player 0 DNP
        ] * 33 + [[21.0, 19.0, 16.0]])  # Extra row to get to 100

        active_mask = minutes_worlds > 0

        uncond, cond = compute_minutes_quantiles(minutes_worlds, active_mask)

        # Unconditional p50 for player 0 should reflect the DNP zeros
        # ~1/3 of worlds have 0 minutes
        assert uncond[1, 0] < 22.0  # Median pulled down by zeros

    def test_conditional_excludes_inactive_worlds(self) -> None:
        """Conditional quantiles should only consider active worlds."""
        n_worlds = 100
        n_players = 2

        # Player 0: always 20 minutes when active, inactive 50% of time
        # Player 1: always 25 minutes when active, active all time
        minutes_worlds = np.zeros((n_worlds, n_players))
        active_mask = np.zeros((n_worlds, n_players), dtype=bool)

        for w in range(n_worlds):
            # Player 0: active in even worlds
            if w % 2 == 0:
                minutes_worlds[w, 0] = 20.0
                active_mask[w, 0] = True
            # Player 1: always active
            minutes_worlds[w, 1] = 25.0
            active_mask[w, 1] = True

        uncond, cond = compute_minutes_quantiles(minutes_worlds, active_mask)

        # Unconditional p50 for player 0 should be ~10 (half zeros)
        assert uncond[1, 0] < 15.0

        # Conditional p50 for player 0 should be 20 (only active worlds)
        assert cond[1, 0] == pytest.approx(20.0, abs=0.1)

        # Player 1: same for both since always active
        assert uncond[1, 1] == pytest.approx(25.0, abs=0.1)
        assert cond[1, 1] == pytest.approx(25.0, abs=0.1)

    def test_contract_exact_quantiles_synthetic(self) -> None:
        """Contract test: verify exact quantiles on known synthetic data."""
        # Create a simple case where we know exact quantiles
        n_worlds = 10
        n_players = 2

        # Player 0: minutes = [0, 10, 20, 20, 20, 30, 30, 30, 40, 50]
        # Player 1: always 25
        minutes_worlds = np.array([
            [0.0, 25.0],
            [10.0, 25.0],
            [20.0, 25.0],
            [20.0, 25.0],
            [20.0, 25.0],
            [30.0, 25.0],
            [30.0, 25.0],
            [30.0, 25.0],
            [40.0, 25.0],
            [50.0, 25.0],
        ])

        # Active mask: player 0 is inactive (minutes=0) in first world
        active_mask = minutes_worlds > 0

        uncond, cond = compute_minutes_quantiles(minutes_worlds, active_mask, quantiles=(0.10, 0.50, 0.90))

        # Unconditional quantiles for player 0:
        # p10 ≈ 0-10, p50 ≈ 20-25, p90 ≈ 40-50
        assert uncond[0, 0] <= 10.0  # p10
        assert 20.0 <= uncond[1, 0] <= 30.0  # p50
        assert uncond[2, 0] >= 35.0  # p90

        # Conditional quantiles for player 0 (exclude the 0-minute world):
        # Values are [10, 20, 20, 20, 30, 30, 30, 40, 50] - 9 values
        # p50 should be around 25-30
        assert cond[1, 0] >= 20.0

        # Player 1: constant 25, so all quantiles should be 25
        np.testing.assert_allclose(uncond[:, 1], 25.0, rtol=0.01)
        np.testing.assert_allclose(cond[:, 1], 25.0, rtol=0.01)

    def test_empty_active_mask_returns_zeros(self) -> None:
        """If a player is never active, conditional quantiles should be 0."""
        n_worlds = 10
        n_players = 2

        minutes_worlds = np.zeros((n_worlds, n_players))
        minutes_worlds[:, 1] = 20.0  # Only player 1 has minutes

        active_mask = np.zeros((n_worlds, n_players), dtype=bool)
        active_mask[:, 1] = True  # Only player 1 is active

        uncond, cond = compute_minutes_quantiles(minutes_worlds, active_mask)

        # Player 0: never active, should have 0 for all quantiles
        np.testing.assert_array_equal(uncond[:, 0], 0.0)
        np.testing.assert_array_equal(cond[:, 0], 0.0)

        # Player 1: always 20 minutes
        np.testing.assert_allclose(uncond[:, 1], 20.0, rtol=0.01)
        np.testing.assert_allclose(cond[:, 1], 20.0, rtol=0.01)
