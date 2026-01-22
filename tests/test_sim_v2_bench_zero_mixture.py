import numpy as np

from projections.sim_v2.bench_zero_mixture import apply_bench_zero_mixture
from projections.sim_v2.minutes_stabilization import reconcile_team_minutes_active_softmax


def test_bench_zero_mixture_preserves_feasibility_and_team_240_after_reconcile() -> None:
    rng = np.random.default_rng(123)
    n_worlds = 200
    n_players = 10
    cap = 41.0
    total = 240.0

    # 5 core players + 5 low-minute players subject to dropout.
    minutes_target = np.array([34.0, 30.0, 28.0, 26.0, 24.0, 6.0, 5.0, 4.0, 3.0, 2.0])
    minutes_worlds = np.clip(
        minutes_target[None, :] + rng.normal(loc=0.0, scale=2.0, size=(n_worlds, n_players)),
        0.0,
        None,
    )
    active = np.ones((n_worlds, n_players), dtype=bool)

    group_map = {111: list(range(n_players))}
    play_prob = np.zeros(n_players, dtype=float)

    stats = apply_bench_zero_mixture(
        minutes_worlds,
        active,
        group_map=group_map,
        minutes_target=minutes_target,
        play_prob=play_prob,
        minutes_threshold=8.0,
        p_zero_base=0.95,
        p_zero_slope=0.0,
        cap_minutes=cap,
        total_minutes=total,
        rng=rng,
    )

    assert stats.min_active_needed == int(np.ceil(total / cap))
    assert (active.sum(axis=1) >= stats.min_active_needed).all()
    np.testing.assert_array_equal(minutes_worlds[~active], 0.0)

    reconciled, _ = reconcile_team_minutes_active_softmax(
        minutes_worlds,
        active,
        total_minutes=total,
        cap_minutes=cap,
        eps=1e-6,
        tol=1e-6,
    )
    np.testing.assert_allclose(reconciled.sum(axis=1), total, atol=1e-3)
    assert (reconciled >= 0.0).all()
    np.testing.assert_array_equal(reconciled[~active], 0.0)

