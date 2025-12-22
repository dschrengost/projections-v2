from __future__ import annotations

import numpy as np

from projections.models.rotalloc import allocate_team_minutes


def test_allocate_sum_to_240_and_caps_respected() -> None:
    rng = np.random.RandomState(7)
    p_rot = rng.uniform(0.0, 1.0, size=10)
    mu = rng.uniform(0.0, 40.0, size=10)
    minutes = allocate_team_minutes(p_rot, mu, np.ones(10, dtype=bool), a=1.5, cap_max=48.0)
    assert abs(float(minutes.sum()) - 240.0) < 1e-9
    assert float(minutes.max()) <= 48.0 + 1e-9
    assert float(minutes.min()) >= -1e-9


def test_allocate_fallback_uniform_when_all_weights_zero() -> None:
    p_rot = np.zeros(6, dtype=float)
    mu = np.zeros(6, dtype=float)
    minutes = allocate_team_minutes(p_rot, mu, np.ones(6, dtype=bool), a=1.5, cap_max=48.0)
    assert abs(float(minutes.sum()) - 240.0) < 1e-9
    assert np.allclose(minutes, np.full(6, 40.0), atol=1e-9)


def test_allocate_deterministic() -> None:
    p_rot = np.array([0.9, 0.2, 0.7, 0.1, 0.3], dtype=float)
    mu = np.array([30.0, 10.0, 25.0, 5.0, 15.0], dtype=float)
    out1 = allocate_team_minutes(p_rot, mu, np.ones(5, dtype=bool), a=1.5, cap_max=48.0)
    out2 = allocate_team_minutes(p_rot, mu, np.ones(5, dtype=bool), a=1.5, cap_max=48.0)
    assert np.allclose(out1, out2, atol=0.0)


def test_allocate_single_player_gets_240() -> None:
    minutes = allocate_team_minutes(np.array([0.5]), np.array([12.3]), np.array([True]), a=1.5, cap_max=48.0)
    assert abs(float(minutes.sum()) - 240.0) < 1e-9
    assert abs(float(minutes[0]) - 240.0) < 1e-9


def test_allocate_caps_and_redistributes_leftover() -> None:
    p_rot = np.array([1.0, 0.9, 0.8, 0.1, 0.1, 0.1], dtype=float)
    mu = np.array([100.0, 10.0, 10.0, 10.0, 10.0, 10.0], dtype=float)
    minutes = allocate_team_minutes(p_rot, mu, np.ones(6, dtype=bool), a=1.5, cap_max=48.0)
    assert abs(float(minutes.sum()) - 240.0) < 1e-9
    assert float(minutes.max()) <= 48.0 + 1e-9
    # At least one player should hit the cap in this construction.
    assert np.any(np.isclose(minutes, 48.0, atol=1e-6))


def test_allocate_with_cutoff_prunes_and_sums() -> None:
    p_rot = np.array([0.9, 0.8, 0.1, 0.05], dtype=float)
    mu = np.array([30.0, 20.0, 10.0, 5.0], dtype=float)
    minutes = allocate_team_minutes(
        p_rot,
        mu,
        np.ones(4, dtype=bool),
        a=1.5,
        cap_max=48.0,
        p_cutoff=0.2,
    )
    assert abs(float(minutes.sum()) - 240.0) < 1e-9
    assert np.all(minutes[2:] == 0.0)


def test_allocate_cutoff_fallback_top1() -> None:
    p_rot = np.array([0.1, 0.05], dtype=float)
    mu = np.array([10.0, 20.0], dtype=float)
    minutes = allocate_team_minutes(
        p_rot,
        mu,
        np.ones(2, dtype=bool),
        a=1.5,
        cap_max=48.0,
        p_cutoff=0.9,
        use_expected_k=True,
        k_min=2,
        k_max=3,
    )
    assert abs(float(minutes.sum()) - 240.0) < 1e-9
    assert int((minutes > 0).sum()) == 1


def test_allocate_cutoff_expected_k_sum_to_240() -> None:
    p_rot = np.array([0.9, 0.6, 0.4, 0.2, 0.1], dtype=float)
    mu = np.array([30.0, 25.0, 20.0, 10.0, 5.0], dtype=float)
    minutes = allocate_team_minutes(
        p_rot,
        mu,
        np.ones(5, dtype=bool),
        a=1.5,
        cap_max=48.0,
        p_cutoff=0.3,
        use_expected_k=True,
        k_min=2,
        k_max=4,
    )
    assert abs(float(minutes.sum()) - 240.0) < 1e-9


def test_allocate_expected_k_small_team() -> None:
    p_rot = np.array([0.9, 0.1, 0.05], dtype=float)
    mu = np.array([25.0, 15.0, 5.0], dtype=float)
    minutes = allocate_team_minutes(
        p_rot,
        mu,
        np.ones(3, dtype=bool),
        a=2.0,
        cap_max=48.0,
        use_expected_k=True,
        k_min=2,
        k_max=5,
    )
    assert abs(float(minutes.sum()) - 240.0) < 1e-9
