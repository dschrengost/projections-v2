from __future__ import annotations

import numpy as np

from projections.sim_v2.minutes_stabilization import reconcile_team_minutes_active_softmax


def test_active_only_softmax_reconcile_sums_to_240_and_zeros_inactive() -> None:
    # 4 worlds, 8 players.
    m0 = np.array(
        [
            [32.0, 30.0, 28.0, 24.0, 20.0, 16.0, 12.0, 8.0],
            [36.0, 28.0, 26.0, 22.0, 18.0, 14.0, 10.0, 6.0],
            [40.0, 30.0, 20.0, 10.0, 5.0, 3.0, 2.0, 1.0],
            [34.0, 29.0, 27.0, 23.0, 19.0, 15.0, 11.0, 7.0],
        ],
        dtype=float,
    )
    active = np.ones_like(m0, dtype=bool)
    active[1, 0] = False
    active[2, 0] = False
    active[2, 1] = False

    out, stats = reconcile_team_minutes_active_softmax(m0, active, cap_minutes=None)

    assert int(stats["n_all_inactive"]) == 0
    assert np.all(out[~active] == 0.0)
    assert np.all(out >= 0.0)
    np.testing.assert_allclose(out.sum(axis=1), 240.0, rtol=0.0, atol=1e-6)


def test_active_only_softmax_reconcile_with_cap_41_sums_to_240() -> None:
    # Force a very concentrated proposal; cap should bind and renormalize.
    m0 = np.array([[200.0, 20.0, 10.0, 5.0, 3.0, 2.0, 1.0, 1.0, 1.0, 1.0]], dtype=float)
    active = np.ones_like(m0, dtype=bool)

    out, stats = reconcile_team_minutes_active_softmax(m0, active, cap_minutes=41.0)

    assert int(stats["n_cap_infeasible_rows"]) == 0
    assert int(stats["n_cap_bind_rows"]) == 1
    assert float(out.max()) <= 41.0 + 1e-6
    np.testing.assert_allclose(out.sum(axis=1), 240.0, rtol=0.0, atol=1e-6)
    assert np.all(out[~active] == 0.0)


def test_active_only_softmax_cap_infeasible_skips_cap_to_preserve_team_total() -> None:
    # With only 5 active players, a 41-minute cap is infeasible (5*41=205 < 240).
    m0 = np.full((2, 5), 1.0, dtype=float)
    active = np.ones_like(m0, dtype=bool)

    out, stats = reconcile_team_minutes_active_softmax(m0, active, cap_minutes=41.0)

    assert int(stats["n_cap_infeasible_rows"]) == 2
    np.testing.assert_allclose(out.sum(axis=1), 240.0, rtol=0.0, atol=1e-6)
    assert float(out.max()) > 41.0
