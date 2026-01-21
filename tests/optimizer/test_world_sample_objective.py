"""Unit tests for world sample objective mode in QuickBuild.

Tests verify:
1. WorldSampleConfig dataclass functionality
2. Seed reproducibility for world index sampling
3. Projection retrieval from worlds matrix with fallback
4. Default mode (mean projections) unchanged
"""

import numpy as np
import pytest

from projections.optimizer.quick_build import WorldSampleConfig


class TestWorldSampleConfig:
    """Tests for WorldSampleConfig dataclass."""

    def test_disabled_by_default(self):
        """Default config should be disabled."""
        cfg = WorldSampleConfig()
        assert cfg.enabled is False
        assert cfg.seed is None
        assert cfg.with_replacement is True

    def test_sample_world_index_reproducible(self):
        """Sampling with same seed should produce same indices."""
        n_worlds, n_players = 100, 20
        worlds = np.random.default_rng(42).random((n_worlds, n_players))
        player_index = {f"p{i}": i for i in range(n_players)}

        cfg1 = WorldSampleConfig(
            enabled=True,
            seed=12345,
            worlds_matrix=worlds,
            player_index=player_index,
        )
        cfg2 = WorldSampleConfig(
            enabled=True,
            seed=12345,
            worlds_matrix=worlds,
            player_index=player_index,
        )

        # Sample 5 times from each
        indices1 = [cfg1.sample_world_index() for _ in range(5)]
        indices2 = [cfg2.sample_world_index() for _ in range(5)]

        assert indices1 == indices2, "Same seed should produce identical indices"
        assert all(0 <= i < n_worlds for i in indices1), "Indices should be valid"

    def test_sample_world_index_records_sampled(self):
        """Sampling should record indices in _sampled_indices."""
        n_worlds, n_players = 50, 10
        worlds = np.random.default_rng(1).random((n_worlds, n_players))
        player_index = {f"p{i}": i for i in range(n_players)}

        cfg = WorldSampleConfig(
            enabled=True,
            seed=99,
            worlds_matrix=worlds,
            player_index=player_index,
        )

        cfg.sample_world_index()
        cfg.sample_world_index()
        cfg.sample_world_index()

        assert len(cfg._sampled_indices) == 3

    def test_get_world_projections_returns_values(self):
        """get_world_projections should return correct values from matrix."""
        worlds = np.array([
            [10.0, 20.0, 30.0],  # world 0
            [15.0, 25.0, 35.0],  # world 1
        ])
        player_index = {"p1": 0, "p2": 1, "p3": 2}

        cfg = WorldSampleConfig(
            enabled=True,
            worlds_matrix=worlds,
            player_index=player_index,
        )

        projs = cfg.get_world_projections(0, ["p1", "p2", "p3"])
        assert projs["p1"] == 10.0
        assert projs["p2"] == 20.0
        assert projs["p3"] == 30.0

        projs = cfg.get_world_projections(1, ["p1", "p3"])
        assert projs["p1"] == 15.0
        assert projs["p3"] == 35.0

    def test_get_world_projections_missing_player_fallback(self):
        """Missing players should fall back to mean_projections."""
        worlds = np.array([[10.0, 20.0]])
        player_index = {"p1": 0, "p2": 1}
        mean_projs = {"p1": 5.0, "p2": 10.0, "p3": 99.9}

        cfg = WorldSampleConfig(
            enabled=True,
            worlds_matrix=worlds,
            player_index=player_index,
            mean_projections=mean_projs,
        )

        projs = cfg.get_world_projections(0, ["p1", "p3"])
        assert projs["p1"] == 10.0  # from matrix
        assert projs["p3"] == 99.9  # from fallback

    def test_get_world_projections_missing_no_fallback(self):
        """Missing players with no fallback should return 0."""
        worlds = np.array([[10.0]])
        player_index = {"p1": 0}

        cfg = WorldSampleConfig(
            enabled=True,
            worlds_matrix=worlds,
            player_index=player_index,
            mean_projections=None,
        )

        projs = cfg.get_world_projections(0, ["p1", "p_missing"])
        assert projs["p1"] == 10.0
        assert projs["p_missing"] == 0.0

    def test_different_seeds_produce_different_sequences(self):
        """Different seeds should produce different sampling sequences."""
        n_worlds = 1000
        worlds = np.zeros((n_worlds, 1))
        player_index = {"p0": 0}

        cfg1 = WorldSampleConfig(
            enabled=True, seed=111, worlds_matrix=worlds, player_index=player_index
        )
        cfg2 = WorldSampleConfig(
            enabled=True, seed=222, worlds_matrix=worlds, player_index=player_index
        )

        indices1 = [cfg1.sample_world_index() for _ in range(10)]
        indices2 = [cfg2.sample_world_index() for _ in range(10)]

        # With 1000 worlds and 10 samples, these should almost certainly differ
        assert indices1 != indices2


class TestWorldSampleIntegration:
    """Integration-style tests for world sample mode (without full solver)."""

    def test_config_initialization_valid(self):
        """Valid config should initialize without errors."""
        worlds = np.ones((100, 10))
        player_index = {f"p{i}": i for i in range(10)}
        mean_projs = {f"p{i}": float(i) for i in range(10)}

        cfg = WorldSampleConfig(
            enabled=True,
            seed=42,
            with_replacement=True,
            worlds_matrix=worlds,
            player_index=player_index,
            mean_projections=mean_projs,
        )

        assert cfg.enabled
        assert cfg.seed == 42
        assert cfg.worlds_matrix.shape == (100, 10)
        assert len(cfg.player_index) == 10

    def test_sample_requires_matrix(self):
        """Sampling without matrix should raise ValueError."""
        cfg = WorldSampleConfig(enabled=True, seed=1)
        with pytest.raises(ValueError, match="worlds_matrix not set"):
            cfg.sample_world_index()
