"""Tests for minutes stabilization: per-world noise + cheap team-240 projection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from projections.sim_v2.minutes_stabilization import (
    apply_pre_sim_qp_reconcile,
    sample_minutes_noise_per_world,
)


def _make_team_roster(
    n_players: int = 10,
    team_id: int = 0,
    total_minutes: float = 240.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create a fake team roster with realistic minutes distribution."""
    # Typical NBA rotation: 5 starters ~30-35 min, 3-4 bench ~15-20 min, 1-2 deep bench ~5-8 min
    if n_players >= 10:
        minutes_p50 = np.array([34, 33, 32, 30, 28, 22, 18, 14, 5, 4], dtype=float)
        is_starter = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=bool)
    else:
        # Proportional distribution
        minutes_p50 = np.linspace(34, 5, n_players)
        is_starter = np.zeros(n_players, dtype=bool)
        is_starter[:5] = True

    # Scale to exactly total_minutes
    current_sum = minutes_p50.sum()
    minutes_p50 = minutes_p50 * (total_minutes / current_sum)

    # Compute p10/p90 based on typical variance
    z90 = 1.28
    sigma = np.where(is_starter, 3.0, 4.0)
    minutes_p10 = np.maximum(minutes_p50 - z90 * sigma, 0.0)
    minutes_p90 = np.minimum(minutes_p50 + z90 * sigma, 48.0)

    team_indices = np.full(n_players, team_id, dtype=int)

    return minutes_p50, minutes_p10, minutes_p90, is_starter, team_indices


class TestSampleMinutesNoisePerWorld:
    """Tests for sample_minutes_noise_per_world."""

    def test_team_sum_equals_240(self) -> None:
        """Verify that team sums equal 240 for every world."""
        minutes_p50, minutes_p10, minutes_p90, is_starter, team_indices = _make_team_roster()
        n_worlds = 200

        minutes_worlds, stats = sample_minutes_noise_per_world(
            minutes_reconciled=minutes_p50,
            minutes_p10=minutes_p10,
            minutes_p90=minutes_p90,
            is_starter=is_starter,
            team_indices=team_indices,
            n_worlds=n_worlds,
            sigma_starter=2.0,
            sigma_bench=3.0,
            min_minutes_for_noise=8.0,
            cap_abs=6.0,
            rng=np.random.default_rng(42),
        )

        assert minutes_worlds.shape == (n_worlds, 10)

        # Check team sums
        team_sums = minutes_worlds.sum(axis=1)
        np.testing.assert_allclose(team_sums, 240.0, atol=1e-6)

        # No violations
        assert stats.sum_240_violations == 0

    def test_no_negative_minutes(self) -> None:
        """Verify that no player has negative minutes."""
        minutes_p50, minutes_p10, minutes_p90, is_starter, team_indices = _make_team_roster()
        n_worlds = 200

        minutes_worlds, _ = sample_minutes_noise_per_world(
            minutes_reconciled=minutes_p50,
            minutes_p10=minutes_p10,
            minutes_p90=minutes_p90,
            is_starter=is_starter,
            team_indices=team_indices,
            n_worlds=n_worlds,
            sigma_starter=2.0,
            sigma_bench=3.0,
            rng=np.random.default_rng(42),
        )

        assert np.all(minutes_worlds >= 0)

    def test_no_minutes_above_48(self) -> None:
        """Verify that no player has more than 48 minutes."""
        minutes_p50, minutes_p10, minutes_p90, is_starter, team_indices = _make_team_roster()
        n_worlds = 200

        minutes_worlds, _ = sample_minutes_noise_per_world(
            minutes_reconciled=minutes_p50,
            minutes_p10=minutes_p10,
            minutes_p90=minutes_p90,
            is_starter=is_starter,
            team_indices=team_indices,
            n_worlds=n_worlds,
            sigma_starter=2.0,
            sigma_bench=3.0,
            rng=np.random.default_rng(42),
        )

        assert np.all(minutes_worlds <= 48.0)

    def test_mean_close_to_reconciled(self) -> None:
        """Verify that mean minutes across worlds is close to reconciled (unbiased)."""
        minutes_p50, minutes_p10, minutes_p90, is_starter, team_indices = _make_team_roster()
        n_worlds = 500

        minutes_worlds, _ = sample_minutes_noise_per_world(
            minutes_reconciled=minutes_p50,
            minutes_p10=minutes_p10,
            minutes_p90=minutes_p90,
            is_starter=is_starter,
            team_indices=team_indices,
            n_worlds=n_worlds,
            sigma_starter=2.0,
            sigma_bench=3.0,
            min_minutes_for_noise=8.0,
            rng=np.random.default_rng(42),
        )

        mean_across_worlds = minutes_worlds.mean(axis=0)

        # For players with minutes >= 8, mean should be close to reconciled
        rotation_mask = minutes_p50 >= 8.0
        rotation_mean = mean_across_worlds[rotation_mask]
        rotation_reconciled = minutes_p50[rotation_mask]

        max_bias = np.abs(rotation_mean - rotation_reconciled).max()
        assert max_bias < 0.5, f"Max bias {max_bias} exceeds 0.5 minutes"

    def test_determinism_same_seed(self) -> None:
        """Verify that same seed produces identical results."""
        minutes_p50, minutes_p10, minutes_p90, is_starter, team_indices = _make_team_roster()
        n_worlds = 50

        minutes_worlds_1, _ = sample_minutes_noise_per_world(
            minutes_reconciled=minutes_p50,
            minutes_p10=minutes_p10,
            minutes_p90=minutes_p90,
            is_starter=is_starter,
            team_indices=team_indices,
            n_worlds=n_worlds,
            rng=np.random.default_rng(12345),
        )

        minutes_worlds_2, _ = sample_minutes_noise_per_world(
            minutes_reconciled=minutes_p50,
            minutes_p10=minutes_p10,
            minutes_p90=minutes_p90,
            is_starter=is_starter,
            team_indices=team_indices,
            n_worlds=n_worlds,
            rng=np.random.default_rng(12345),
        )

        np.testing.assert_array_equal(minutes_worlds_1, minutes_worlds_2)

    def test_different_seed_different_results(self) -> None:
        """Verify that different seeds produce different results."""
        minutes_p50, minutes_p10, minutes_p90, is_starter, team_indices = _make_team_roster()
        n_worlds = 50

        minutes_worlds_1, _ = sample_minutes_noise_per_world(
            minutes_reconciled=minutes_p50,
            minutes_p10=minutes_p10,
            minutes_p90=minutes_p90,
            is_starter=is_starter,
            team_indices=team_indices,
            n_worlds=n_worlds,
            rng=np.random.default_rng(12345),
        )

        minutes_worlds_2, _ = sample_minutes_noise_per_world(
            minutes_reconciled=minutes_p50,
            minutes_p10=minutes_p10,
            minutes_p90=minutes_p90,
            is_starter=is_starter,
            team_indices=team_indices,
            n_worlds=n_worlds,
            rng=np.random.default_rng(54321),
        )

        assert not np.allclose(minutes_worlds_1, minutes_worlds_2)

    def test_multiple_teams(self) -> None:
        """Verify team-240 constraint holds for multiple teams."""
        # Create 3 teams
        n_players_per_team = 10
        n_teams = 3

        all_minutes_p50 = []
        all_minutes_p10 = []
        all_minutes_p90 = []
        all_is_starter = []
        all_team_indices = []

        for t in range(n_teams):
            m50, m10, m90, is_s, t_idx = _make_team_roster(
                n_players=n_players_per_team, team_id=t
            )
            all_minutes_p50.append(m50)
            all_minutes_p10.append(m10)
            all_minutes_p90.append(m90)
            all_is_starter.append(is_s)
            all_team_indices.append(t_idx)

        minutes_p50 = np.concatenate(all_minutes_p50)
        minutes_p10 = np.concatenate(all_minutes_p10)
        minutes_p90 = np.concatenate(all_minutes_p90)
        is_starter = np.concatenate(all_is_starter)
        team_indices = np.concatenate(all_team_indices)

        n_worlds = 100
        minutes_worlds, stats = sample_minutes_noise_per_world(
            minutes_reconciled=minutes_p50,
            minutes_p10=minutes_p10,
            minutes_p90=minutes_p90,
            is_starter=is_starter,
            team_indices=team_indices,
            n_worlds=n_worlds,
            rng=np.random.default_rng(42),
        )

        assert minutes_worlds.shape == (n_worlds, n_teams * n_players_per_team)

        # Check each team sums to 240
        for t in range(n_teams):
            team_mask = team_indices == t
            team_sums = minutes_worlds[:, team_mask].sum(axis=1)
            np.testing.assert_allclose(
                team_sums, 240.0, atol=1e-6,
                err_msg=f"Team {t} sums not equal to 240"
            )

    def test_student_t_distribution(self) -> None:
        """Verify Student-t noise option works and produces heavier tails."""
        minutes_p50, minutes_p10, minutes_p90, is_starter, team_indices = _make_team_roster()
        n_worlds = 1000

        # Normal noise
        minutes_normal, _ = sample_minutes_noise_per_world(
            minutes_reconciled=minutes_p50,
            minutes_p10=minutes_p10,
            minutes_p90=minutes_p90,
            is_starter=is_starter,
            team_indices=team_indices,
            n_worlds=n_worlds,
            use_student_t=False,
            rng=np.random.default_rng(42),
        )

        # Student-t noise
        minutes_student_t, _ = sample_minutes_noise_per_world(
            minutes_reconciled=minutes_p50,
            minutes_p10=minutes_p10,
            minutes_p90=minutes_p90,
            is_starter=is_starter,
            team_indices=team_indices,
            n_worlds=n_worlds,
            use_student_t=True,
            t_df=5.0,
            rng=np.random.default_rng(42),
        )

        # Both should satisfy constraints
        np.testing.assert_allclose(minutes_normal.sum(axis=1), 240.0, atol=1e-6)
        np.testing.assert_allclose(minutes_student_t.sum(axis=1), 240.0, atol=1e-6)

        # Results should differ (different RNG paths)
        assert not np.allclose(minutes_normal, minutes_student_t)

    def test_empty_roster(self) -> None:
        """Verify empty roster is handled gracefully."""
        minutes_worlds, stats = sample_minutes_noise_per_world(
            minutes_reconciled=np.array([], dtype=float),
            minutes_p10=np.array([], dtype=float),
            minutes_p90=np.array([], dtype=float),
            is_starter=np.array([], dtype=bool),
            team_indices=np.array([], dtype=int),
            n_worlds=10,
            rng=np.random.default_rng(42),
        )

        assert minutes_worlds.shape == (10, 0)
        assert stats.n_teams == 0


class TestApplyPreSimQpReconcile:
    """Tests for pre-sim QP reconciliation."""

    def test_reconciles_to_240(self) -> None:
        """Verify that reconciliation produces team sums of 240."""
        # Create a DataFrame with slightly off team totals
        df = pd.DataFrame({
            "game_id": [1] * 10 + [1] * 10,
            "team_id": [100] * 10 + [200] * 10,
            "player_id": list(range(20)),
            "minutes_p50": [36, 34, 32, 30, 28, 22, 18, 14, 10, 8] * 2,  # Sum > 240
            "is_projected_starter": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0] * 2,
            "play_prob": [1.0] * 20,
            "minutes_p10": [30, 28, 26, 24, 22, 16, 12, 8, 4, 2] * 2,
            "minutes_p90": [42, 40, 38, 36, 34, 28, 24, 20, 16, 14] * 2,
        })

        result = apply_pre_sim_qp_reconcile(df, starter_weight=2.0)

        # Check team sums
        for team_id in [100, 200]:
            team_sum = result[result["team_id"] == team_id]["minutes_p50"].sum()
            assert abs(team_sum - 240.0) < 1.0, f"Team {team_id} sum {team_sum} not close to 240"

    def test_preserves_player_order(self) -> None:
        """Verify that player_id order is preserved."""
        df = pd.DataFrame({
            "game_id": [1] * 10,
            "team_id": [100] * 10,
            "player_id": list(range(10)),
            "minutes_p50": [36, 34, 32, 30, 28, 22, 18, 14, 10, 8],
            "is_projected_starter": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            "play_prob": [1.0] * 10,
            "minutes_p10": [30, 28, 26, 24, 22, 16, 12, 8, 4, 2],
            "minutes_p90": [42, 40, 38, 36, 34, 28, 24, 20, 16, 14],
        })

        result = apply_pre_sim_qp_reconcile(df)

        assert list(result["player_id"]) == list(df["player_id"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
