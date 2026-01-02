from __future__ import annotations

import numpy as np

from projections.sim_v2.minutes_stabilization import sample_minutes_noise_per_world


def test_sample_minutes_noise_respects_active_mask_and_team_240() -> None:
    # 6-player team, baseline sums to 240
    minutes_reconciled = np.array([40.0, 40.0, 40.0, 40.0, 40.0, 40.0], dtype=float)
    minutes_p10 = np.zeros_like(minutes_reconciled)
    minutes_p90 = np.full_like(minutes_reconciled, 48.0)
    is_starter = np.ones_like(minutes_reconciled, dtype=bool)
    team_indices = np.zeros_like(minutes_reconciled, dtype=int)

    n_worlds = 3
    active_mask = np.ones((n_worlds, len(minutes_reconciled)), dtype=bool)
    # Player 0 inactive in all worlds; their 40 minutes must flow to teammates.
    active_mask[:, 0] = False

    out, stats = sample_minutes_noise_per_world(
        minutes_reconciled=minutes_reconciled,
        minutes_p10=minutes_p10,
        minutes_p90=minutes_p90,
        is_starter=is_starter,
        team_indices=team_indices,
        n_worlds=n_worlds,
        active_mask=active_mask,
        sigma_starter=0.0,
        sigma_bench=0.0,
        min_minutes_for_noise=0.0,
        cap_abs=0.0,
        rng=np.random.default_rng(123),
    )

    assert out.shape == (n_worlds, len(minutes_reconciled))
    assert np.allclose(out[:, 0], 0.0)
    assert np.allclose(out.sum(axis=1), 240.0, atol=1e-6)
    assert float(out.max()) <= 48.0 + 1e-6
    assert stats.sum_240_violations == 0

