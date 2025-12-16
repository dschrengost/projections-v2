from __future__ import annotations

import numpy as np
from scipy import stats

from projections.sim_v2.game_script import GameScriptConfig, sample_minutes_with_scripts


def test_sample_minutes_with_scripts_hits_input_quantiles_without_noise() -> None:
    # Force every player into the "close" script with no quantile noise so that
    # the output minutes are exactly at the configured quantile targets.
    config = GameScriptConfig(
        margin_std=0.0,
        spread_coef=0.0,
        quantile_noise_std=0.0,
        quantile_targets={"close": (0.9, 0.1)},
    )
    rng = np.random.default_rng(1337)

    minutes_p10 = np.array([28.0, 14.0], dtype=float)
    minutes_p50 = np.array([32.0, 18.0], dtype=float)
    minutes_p90 = np.array([44.0, 22.0], dtype=float)
    is_starter = np.array([1.0, 0.0], dtype=float)

    game_ids = np.array([1, 1], dtype=int)
    team_ids = np.array([10, 20], dtype=int)
    spreads_home = np.array([0.0, 0.0], dtype=float)
    home_team_ids = {1: 10}

    out = sample_minutes_with_scripts(
        minutes_p10=minutes_p10,
        minutes_p50=minutes_p50,
        minutes_p90=minutes_p90,
        is_starter=is_starter,
        game_ids=game_ids,
        team_ids=team_ids,
        spreads_home=spreads_home,
        home_team_ids=home_team_ids,
        n_worlds=1,
        config=config,
        rng=rng,
    )

    assert out.shape == (1, 2)
    np.testing.assert_allclose(out[0, 0], minutes_p90[0], rtol=0.0, atol=1e-6)  # starter @ p90
    np.testing.assert_allclose(out[0, 1], minutes_p10[1], rtol=0.0, atol=1e-6)  # bench @ p10


def test_sample_minutes_with_scripts_uses_asymmetric_sigmas() -> None:
    # With right-skewed quantiles (p90-p50 > p50-p10), the upper tail should be
    # scaled by a larger sigma than the lower tail.
    config = GameScriptConfig(
        margin_std=0.0,
        spread_coef=0.0,
        quantile_noise_std=0.0,
        quantile_targets={"close": (0.75, 0.75)},
    )
    rng = np.random.default_rng(1337)

    minutes_p10 = np.array([28.0], dtype=float)
    minutes_p50 = np.array([32.0], dtype=float)
    minutes_p90 = np.array([44.0], dtype=float)
    is_starter = np.array([1.0], dtype=float)

    out = sample_minutes_with_scripts(
        minutes_p10=minutes_p10,
        minutes_p50=minutes_p50,
        minutes_p90=minutes_p90,
        is_starter=is_starter,
        game_ids=np.array([1], dtype=int),
        team_ids=np.array([10], dtype=int),
        spreads_home=np.array([0.0], dtype=float),
        home_team_ids={1: 10},
        n_worlds=1,
        config=config,
        rng=rng,
    )

    z = float(stats.norm.ppf(0.75))
    sigma_old = (minutes_p90[0] - minutes_p10[0]) / 2.56
    expected_old = minutes_p50[0] + sigma_old * z

    # New split-normal mapping should land higher than the old symmetric mapping.
    assert float(out[0, 0]) > float(expected_old)

