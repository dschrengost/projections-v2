from __future__ import annotations

import numpy as np

from projections.sim_v2.minutes_allocator import allocate_team_minutes_matrix


def test_allocator_keeps_inactive_minutes_zero() -> None:
    n_worlds = 3
    n_players = 6

    # Demand includes inactive players (should be zeroed by active_mask).
    demand = np.full((n_worlds, n_players), 30.0, dtype=float)
    # Active mask: last player inactive in all worlds.
    active = np.ones((n_worlds, n_players), dtype=bool)
    active[:, -1] = False

    # Priority arbitrary; cap allows 5 actives to reach 240 exactly.
    priority = np.arange(1, n_players + 1, dtype=float)

    allocated, stats = allocate_team_minutes_matrix(
        demand,
        active,
        priority=priority,
        cap=48.0,
        target_total=240.0,
        k=3.0,
        eps=1e-6,
    )

    # Inactive players must be exactly 0.
    assert np.all(allocated[:, -1] == 0.0)

    # Team sum should still be 240 in feasible rows.
    np.testing.assert_allclose(allocated.sum(axis=1), 240.0, atol=1e-6)

    # Sanity: allocator does not mark any rows infeasible.
    assert stats["n_cap_infeasible_rows"] == 0
