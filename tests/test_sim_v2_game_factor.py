import numpy as np

from projections.sim_v2.game_factor import apply_game_factor


def test_apply_game_factor_additive_preserves_player_mean_with_zero_mean_shocks() -> None:
    rng = np.random.default_rng(123)
    n_worlds = 2
    n_players = 4

    fpts = rng.normal(loc=20.0, scale=5.0, size=(n_worlds, n_players)).astype(float)
    active = np.ones_like(fpts, dtype=bool)
    game_ids = np.array([10, 10, 20, 20])
    basis = np.array([30.0, 10.0, 25.0, 5.0])

    # Zero-mean shocks per game across worlds -> additive preserves per-player sample mean exactly.
    game_shocks = np.array(
        [
            [-3.0, 1.5],
            [3.0, -1.5],
        ],
        dtype=float,
    )

    before = fpts.copy()
    apply_game_factor(
        fpts,
        active,
        game_ids=game_ids,
        beta_basis=basis,
        sigma=1.0,
        mode="additive",
        game_shocks=game_shocks,
    )

    np.testing.assert_allclose(fpts.mean(axis=0), before.mean(axis=0), atol=1e-12)

    # Per-game total adjustment equals the shock for each world (shares sum to 1 within game).
    delta = fpts - before
    for w in range(n_worlds):
        for gi, gid in enumerate(np.unique(game_ids)):
            idxs = np.where(game_ids == gid)[0]
            np.testing.assert_allclose(delta[w, idxs].sum(), game_shocks[w, gi], atol=1e-12)


def test_apply_game_factor_inactive_worlds_remain_zero_and_sum_to_shock() -> None:
    n_worlds = 3
    game_ids = np.array([1, 1, 1, 1])
    basis = np.array([10.0, 2.0, 1.0, 1.0])

    fpts = np.array(
        [
            [12.0, 8.0, 0.0, 0.0],
            [11.0, 0.0, 6.0, 0.0],
            [0.0, 0.0, 5.0, 4.0],
        ],
        dtype=float,
    )
    active = fpts > 0.0
    before = fpts.copy()
    game_shocks = np.array([[2.0], [-2.0], [1.0]], dtype=float)

    apply_game_factor(
        fpts,
        active,
        game_ids=game_ids,
        beta_basis=basis,
        sigma=1.0,
        mode="additive",
        game_shocks=game_shocks,
    )

    # Inactive player-worlds must remain exactly 0.
    np.testing.assert_array_equal(fpts[~active], 0.0)

    # Total adjustment among actives equals the shock for each world.
    delta = fpts - before
    for w in range(n_worlds):
        np.testing.assert_allclose(delta[w, active[w]].sum(), game_shocks[w, 0], atol=1e-12)
