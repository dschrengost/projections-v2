from __future__ import annotations

import numpy as np
import pytest

from projections.sim_v2.minutes_allocator import allocate_team_minutes_matrix


def test_allocator_respects_fixed_minutes_when_active() -> None:
    demand = np.array([[25.0, 25.0, 25.0, 25.0]], dtype=float)  # (W=1, N=4)
    active = np.array([[True, True, True, True]], dtype=bool)
    priority = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)

    fixed_mask = np.array([True, False, False, False], dtype=bool)
    fixed_minutes = np.array([30.0, 0.0, 0.0, 0.0], dtype=float)

    allocated, _stats = allocate_team_minutes_matrix(
        demand,
        active,
        priority=priority,
        cap=100.0,
        fixed_mask=fixed_mask,
        fixed_minutes=fixed_minutes,
        target_total=100.0,
    )

    assert allocated.shape == demand.shape
    assert abs(float(allocated[0, 0]) - 30.0) < 1e-6
    assert abs(float(allocated.sum()) - 100.0) < 1e-4
    assert abs(float(allocated[0, 1:].sum()) - 70.0) < 1e-4


def test_allocator_availability_overrides_fixed_minutes_when_inactive() -> None:
    demand = np.array([[25.0, 25.0, 25.0, 25.0]], dtype=float)
    active = np.array([[False, True, True, True]], dtype=bool)
    priority = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)

    fixed_mask = np.array([True, False, False, False], dtype=bool)
    fixed_minutes = np.array([30.0, 0.0, 0.0, 0.0], dtype=float)

    allocated, _stats = allocate_team_minutes_matrix(
        demand,
        active,
        priority=priority,
        cap=100.0,
        fixed_mask=fixed_mask,
        fixed_minutes=fixed_minutes,
        target_total=100.0,
    )

    assert abs(float(allocated[0, 0]) - 0.0) < 1e-6
    assert abs(float(allocated.sum()) - 100.0) < 1e-4
    assert abs(float(allocated[0, 1:].sum()) - 100.0) < 1e-4


def test_allocator_raises_when_fixed_sum_exceeds_target() -> None:
    demand = np.array([[10.0, 10.0, 10.0]], dtype=float)
    active = np.array([[True, True, True]], dtype=bool)
    priority = np.array([0.0, 0.0, 0.0], dtype=float)

    fixed_mask = np.array([True, True, False], dtype=bool)
    fixed_minutes = np.array([60.0, 50.0, 0.0], dtype=float)

    with pytest.raises(ValueError, match=r"locked minutes infeasible"):
        allocate_team_minutes_matrix(
            demand,
            active,
            priority=priority,
            cap=100.0,
            fixed_mask=fixed_mask,
            fixed_minutes=fixed_minutes,
            target_total=100.0,
        )
