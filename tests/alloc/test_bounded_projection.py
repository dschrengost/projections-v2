from __future__ import annotations

import numpy as np
import pytest

from projections.alloc.bounded_projection import ProjectionInfeasibleError, project_sum_with_bounds


def test_project_sum_with_bounds_random_feasible_vectors() -> None:
    rng = np.random.default_rng(7)

    for _ in range(50):
        n = int(rng.integers(6, 16))
        x = rng.normal(24.0, 9.0, size=n)
        lb = np.clip(rng.normal(0.0, 2.0, size=n), 0.0, 8.0)
        ub = np.clip(lb + rng.uniform(8.0, 38.0, size=n), lb + 0.1, 48.0)
        target = float(rng.uniform(lb.sum() + 1e-4, ub.sum() - 1e-4))
        weights = rng.uniform(0.3, 3.5, size=n)

        y = project_sum_with_bounds(x, target, lb, ub, weights)

        assert y.shape == (n,)
        assert np.all(y >= lb - 1e-8)
        assert np.all(y <= ub + 1e-8)
        assert abs(float(y.sum()) - target) <= 1e-5


def test_project_sum_with_bounds_respects_many_locks() -> None:
    x = np.array([30.0, 25.0, 15.0, 10.0, 8.0, 6.0], dtype=float)
    lb = np.array([24.0, 22.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    ub = np.array([24.0, 22.0, 40.0, 40.0, 40.0, 40.0], dtype=float)
    target = 120.0

    y = project_sum_with_bounds(x, target, lb, ub)

    assert abs(float(y[0]) - 24.0) <= 1e-9
    assert abs(float(y[1]) - 22.0) <= 1e-9
    assert np.all(y >= lb - 1e-9)
    assert np.all(y <= ub + 1e-9)
    assert abs(float(y.sum()) - target) <= 1e-6


def test_project_sum_with_bounds_caps_prevent_spike() -> None:
    # Player 0 has huge raw demand but hard cap should prevent 48+ spikes.
    x = np.array([80.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0], dtype=float)
    lb = np.zeros_like(x)
    ub = np.array([34.0, 34.0, 34.0, 34.0, 34.0, 34.0, 34.0, 34.0], dtype=float)

    y = project_sum_with_bounds(x, 240.0, lb, ub)

    assert np.all(y <= 34.0 + 1e-6)
    assert abs(float(y.sum()) - 240.0) <= 1e-6


def test_project_sum_with_bounds_raises_structured_infeasible_error() -> None:
    with pytest.raises(ProjectionInfeasibleError) as excinfo:
        project_sum_with_bounds(
            x=np.array([10.0, 10.0, 10.0], dtype=float),
            target_sum=20.0,
            lb=np.array([10.0, 10.0, 10.0], dtype=float),
            ub=np.array([48.0, 48.0, 48.0], dtype=float),
        )
    err = excinfo.value
    assert err.reason == "sum_lb_exceeds_target"
    assert err.to_dict()["n_items"] == 3

    with pytest.raises(ProjectionInfeasibleError) as excinfo2:
        project_sum_with_bounds(
            x=np.array([10.0, 10.0, 10.0], dtype=float),
            target_sum=240.0,
            lb=np.array([0.0, 0.0, 0.0], dtype=float),
            ub=np.array([20.0, 20.0, 20.0], dtype=float),
        )
    assert excinfo2.value.reason == "sum_ub_below_target"


def test_project_sum_with_bounds_is_deterministic() -> None:
    x = np.array([31.0, 28.0, 24.0, 19.0, 16.0, 12.0, 9.0, 5.0], dtype=float)
    lb = np.array([10.0, 8.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    ub = np.array([36.0, 34.0, 32.0, 30.0, 28.0, 24.0, 20.0, 18.0], dtype=float)
    w = np.array([2.0, 1.7, 1.3, 1.1, 1.0, 0.9, 0.8, 0.7], dtype=float)

    y1 = project_sum_with_bounds(x, 180.0, lb, ub, w)
    y2 = project_sum_with_bounds(x, 180.0, lb, ub, w)

    np.testing.assert_allclose(y1, y2, atol=1e-10)
