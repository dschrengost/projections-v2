from __future__ import annotations

import numpy as np

from projections.sim_v2.minutes_stabilization import recenter_team_minutes_to_conditional_means


def test_minutes_mean_recentering_preserves_team_total_and_reduces_mean_error() -> None:
    # 200 worlds, 10 players, all active.
    w = 200
    m0_row = np.array([41, 41, 30, 30, 25, 25, 20, 14, 7, 7], dtype=float)
    assert float(m0_row.sum()) == 240.0
    minutes = np.broadcast_to(m0_row[None, :], (w, len(m0_row))).copy()
    active = np.ones_like(minutes, dtype=bool)

    # Targets shift 12 minutes from the two capped players into the last two.
    targets = np.array([35, 35, 30, 30, 25, 25, 20, 14, 13, 13], dtype=float)
    assert float(targets.sum()) == 240.0

    before_means = minutes.mean(axis=0)
    before_err = np.max(np.abs(before_means - targets))

    out, stats = recenter_team_minutes_to_conditional_means(
        minutes,
        active,
        target_minutes_conditional=targets,
        total_minutes=240.0,
        cap_minutes=41.0,
        max_iters=15,
        step=1.0,
        tol=1e-3,
    )

    after_means = out.mean(axis=0)
    after_err = np.max(np.abs(after_means - targets))

    np.testing.assert_allclose(out.sum(axis=1), 240.0, rtol=0.0, atol=1e-6)
    assert np.all(out >= 0.0)
    assert float(out.max()) <= 41.0 + 1e-6
    assert float(after_err) < float(before_err)
    assert int(stats["n_iters"]) >= 1

