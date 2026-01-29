from __future__ import annotations

import numpy as np

from projections.sim_v2.config import MinutesFeasibilityConfig
from projections.sim_v2.minutes_allocator import allocate_team_minutes_matrix
from projections.sim_v2.minutes_physics import apply_team_feasibility_gate


def test_feasibility_gate_resamples_until_team_worlds_are_feasible() -> None:
    rng = np.random.default_rng(123)

    n_worlds = 500
    baseline = np.array([36, 34, 32, 30, 28, 22, 18, 16, 14, 10], dtype=float)  # sums to 240
    p_play = np.array([0.98, 0.98, 0.98, 0.96, 0.96, 0.70, 0.60, 0.55, 0.30, 0.20], dtype=float)
    max_increase = np.array([12, 12, 12, 12, 12, 10, 8, 8, 4, 4], dtype=float)
    cap_upper = np.minimum(48.0, baseline + max_increase)

    active = rng.random(size=(n_worlds, baseline.size)) < p_play[None, :]
    group_map = {"T": np.arange(baseline.size, dtype=int)}

    cfg = MinutesFeasibilityConfig(
        enabled=True,
        min_active_players=8,
        min_sum_demand=210.0,
        max_resample_attempts=5,
        min_rotation_locks_active=None,
    )

    active2, diag = apply_team_feasibility_gate(
        active,
        play_prob=p_play,
        baseline_minutes=baseline,
        cap_upper=cap_upper,
        group_map=group_map,
        cfg=cfg,
        rng=rng,
        eligible_mask=None,
        rotation_lock_mask=None,
        target_total=240.0,
        eps=1e-6,
    )

    assert diag.enabled is True
    assert diag.n_team_worlds == n_worlds
    assert (active2.sum(axis=1) >= 8).all()
    assert ((active2.astype(float) * baseline[None, :]).sum(axis=1) >= 210.0).all()
    assert ((active2.astype(float) * cap_upper[None, :]).sum(axis=1) >= 240.0 - 1e-6).all()


def test_allocator_respects_increase_only_caps() -> None:
    # Baseline minutes sum to 240 for a single team.
    baseline = np.array([36, 34, 32, 30, 28, 22, 18, 16, 14, 10], dtype=float)
    max_increase = np.array([12, 12, 12, 12, 12, 10, 8, 8, 4, 4], dtype=float)
    hard_cap = 48.0

    # World A: only top-5 + last player are active (pathological sparse team).
    # Under increase-only caps this should be infeasible (can't reach 240 without sponging).
    active = np.array([[1, 1, 1, 1, 1, 0, 0, 0, 0, 1]], dtype=bool)
    demand = baseline[None, :] * active.astype(float)

    _allocated, stats = allocate_team_minutes_matrix(
        demand,
        active,
        priority=baseline,
        cap=hard_cap,
        max_increase=max_increase,
        baseline=baseline,
        target_total=240.0,
        k=3.0,
        eps=1e-6,
    )

    assert stats["n_cap_infeasible_rows"] == 1

    # World B: add one rotation player so the world becomes feasible, and ensure the deep
    # bench player still cannot soak up extreme minutes.
    active2 = np.array([[1, 1, 1, 1, 1, 1, 0, 0, 0, 1]], dtype=bool)
    demand2 = baseline[None, :] * active2.astype(float)
    allocated2, stats2 = allocate_team_minutes_matrix(
        demand2,
        active2,
        priority=baseline,
        cap=hard_cap,
        max_increase=max_increase,
        baseline=baseline,
        target_total=240.0,
        k=3.0,
        eps=1e-6,
    )

    assert stats2["n_cap_infeasible_rows"] == 0
    np.testing.assert_allclose(allocated2.sum(axis=1), 240.0, atol=1e-6)

    # Deep bench player (baseline=10, delta=4) should not absorb more than 14 minutes.
    assert float(allocated2[0, -1]) <= 14.0 + 1e-6
