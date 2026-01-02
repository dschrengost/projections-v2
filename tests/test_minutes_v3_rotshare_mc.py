"""Unit tests for minutes_v3 rotshare Monte Carlo quantiles."""

from __future__ import annotations

import numpy as np
import pytest

from projections.minutes_v3.rotshare_mc import (
    RotshareMonteCarloConfig,
    compute_minutes_quantiles_from_worlds,
    sample_team_minutes_worlds,
)


def test_sample_team_minutes_worlds_sums_to_240_per_world():
    base_share = np.array([0.3, 0.25, 0.2, 0.15, 0.1], dtype=float)
    play_prob = np.array([0.95, 0.9, 0.85, 0.8, 0.75], dtype=float)
    cfg = RotshareMonteCarloConfig(n_worlds=500, concentration=50.0, seed=1, min_active_players=5)
    worlds = sample_team_minutes_worlds(
        base_share=base_share,
        play_prob=play_prob,
        is_out=None,
        config=cfg,
        game_id=1,
        team_id=10,
    )
    assert worlds.shape == (500, 5)
    totals = worlds.sum(axis=1)
    assert float(np.max(np.abs(totals - 240.0))) == pytest.approx(0.0, abs=1e-6)


def test_compute_quantiles_monotonic_and_deterministic():
    base_share = np.array([0.4, 0.3, 0.2, 0.1], dtype=float)
    play_prob = np.array([0.95, 0.9, 0.4, 0.2], dtype=float)
    cfg = RotshareMonteCarloConfig(n_worlds=1000, concentration=30.0, seed=123, min_active_players=3)

    worlds_a = sample_team_minutes_worlds(
        base_share=base_share,
        play_prob=play_prob,
        is_out=None,
        config=cfg,
        game_id=7,
        team_id=99,
    )
    worlds_b = sample_team_minutes_worlds(
        base_share=base_share,
        play_prob=play_prob,
        is_out=None,
        config=cfg,
        game_id=7,
        team_id=99,
    )
    assert np.allclose(worlds_a, worlds_b)

    p10, p50, p90 = compute_minutes_quantiles_from_worlds(worlds_a, center="p50")
    assert np.all(np.isfinite(p10))
    assert np.all(np.isfinite(p50))
    assert np.all(np.isfinite(p90))
    assert np.all(p10 <= p50 + 1e-9)
    assert np.all(p50 <= p90 + 1e-9)
