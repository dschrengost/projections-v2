from __future__ import annotations

import numpy as np

from projections.sim_v2.minutes_stabilization import (
    _project_to_capped_simplex,
    _project_to_capped_simplex_batch,
    _project_to_simplex_nonnegative,
    _project_to_simplex_nonnegative_batch,
)


def test_simplex_nonnegative_batch_matches_scalar_rowwise() -> None:
    rng = np.random.default_rng(0)
    w, n = 50, 13
    v = rng.normal(size=(w, n))

    active = rng.random(size=(w, n)) < 0.65
    active[0] = False  # include all-inactive edge case
    for r in range(1, w):
        if not active[r].any():
            active[r, int(rng.integers(0, n))] = True

    total = 1.0
    out_batch = _project_to_simplex_nonnegative_batch(v, active, total)

    out_scalar = np.zeros_like(v)
    for r in range(w):
        idxs = np.where(active[r])[0]
        if idxs.size == 0:
            continue
        out_scalar[r, idxs] = _project_to_simplex_nonnegative(v[r, idxs], total)

    np.testing.assert_allclose(out_batch, out_scalar, rtol=0.0, atol=1e-10)
    expected_sums = np.where(active.any(axis=1), total, 0.0)
    np.testing.assert_allclose(out_batch.sum(axis=1), expected_sums, rtol=0.0, atol=1e-10)
    assert np.all(out_batch[~active] == 0.0)
    assert np.all(out_batch >= 0.0)


def test_capped_simplex_batch_matches_scalar_rowwise_with_relaxed_cap() -> None:
    rng = np.random.default_rng(1)
    w, n = 40, 11
    v = rng.normal(size=(w, n))

    active = rng.random(size=(w, n)) < 0.35
    active[0] = False  # include all-inactive edge case
    for r in range(1, w):
        if not active[r].any():
            active[r, int(rng.integers(0, n))] = True

    total = 5.0
    cap = 2.0

    out_batch = _project_to_capped_simplex_batch(v, active, total, cap)

    out_scalar = np.zeros_like(v)
    for r in range(w):
        idxs = np.where(active[r])[0]
        if idxs.size == 0:
            continue
        cap_eff = max(cap, total / float(idxs.size))
        out_scalar[r, idxs] = _project_to_capped_simplex(v[r, idxs], total, cap_eff)

    np.testing.assert_allclose(out_batch, out_scalar, rtol=0.0, atol=1e-7)
    expected_sums = np.where(active.any(axis=1), total, 0.0)
    np.testing.assert_allclose(out_batch.sum(axis=1), expected_sums, rtol=0.0, atol=1e-6)
    assert np.all(out_batch[~active] == 0.0)
    assert np.all(out_batch >= 0.0)

    active_counts = active.sum(axis=1)
    feasible = (active_counts * cap) >= total
    feasible = feasible & active.any(axis=1)
    if feasible.any():
        assert float(out_batch[feasible].max()) <= cap + 1e-6
