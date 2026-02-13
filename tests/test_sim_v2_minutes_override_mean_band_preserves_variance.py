from __future__ import annotations

import numpy as np
import pandas as pd

from projections.alloc.bounded_projection import project_sum_with_bounds
from projections.overrides.minutes_overrides_v2 import MinutesOverrideV2Policy, apply_minutes_overrides_v2


def _toy_team_df() -> pd.DataFrame:
    rows = []
    for pid in range(1, 11):
        rows.append(
            {
                "game_id": 100,
                "team_id": 10,
                "player_id": pid,
                "minutes_mean": 24.0,
                "is_projected_starter": 1 if pid <= 5 else 0,
                "play_prob": 0.95,
            }
        )
    return pd.DataFrame(rows)


def _simulate_world_minutes(
    *,
    mu: np.ndarray,
    world_lb: np.ndarray,
    world_ub: np.ndarray,
    weight: np.ndarray,
    n_worlds: int = 4000,
    seed: int = 17,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    residuals = rng.normal(loc=0.0, scale=6.0, size=(n_worlds, mu.size))
    out = np.zeros((n_worlds, mu.size), dtype=float)
    for i in range(n_worlds):
        raw = mu + residuals[i]
        clipped = np.clip(raw, world_lb, world_ub)
        out[i] = project_sum_with_bounds(clipped, 240.0, world_lb, world_ub, weight)
    return out


def test_mean_band_preserves_world_variance() -> None:
    baseline = _toy_team_df()
    payload = {
        "overrides": [
            {
                "game_id": 100,
                "player_id": 1,
                "fields": {"override_mode": "band", "lb_minutes": 20.0, "ub_minutes": 24.0},
            }
        ]
    }
    resolved, _ = apply_minutes_overrides_v2(
        baseline,
        payload,
        policy=MinutesOverrideV2Policy(override_infeasible="error"),
        strict=True,
    )

    row = resolved.loc[resolved["player_id"] == 1].iloc[0]
    assert abs(float(row["mean_lb_minutes"]) - 20.0) <= 1e-9
    assert abs(float(row["mean_ub_minutes"]) - 24.0) <= 1e-9
    assert abs(float(row["world_lb_minutes"]) - 0.0) <= 1e-9
    assert abs(float(row["world_ub_minutes"]) - 48.0) <= 1e-9
    assert 20.0 - 1e-9 <= float(row["mu_minutes"]) <= 24.0 + 1e-9

    mu = resolved["mu_minutes"].to_numpy(dtype=float)
    world_lb = resolved["world_lb_minutes"].to_numpy(dtype=float)
    world_ub = resolved["world_ub_minutes"].to_numpy(dtype=float)
    weight = resolved["weight"].to_numpy(dtype=float)
    worlds = _simulate_world_minutes(mu=mu, world_lb=world_lb, world_ub=world_ub, weight=weight)

    np.testing.assert_allclose(worlds.sum(axis=1), 240.0, atol=1e-6)

    x = worlds[:, 0]
    p10, p50, p90 = np.quantile(x, [0.1, 0.5, 0.9])

    # Mean band must not become per-world clamp bounds.
    assert bool((x < 20.0 - 1e-6).any())
    assert bool((x > 24.0 + 1e-6).any())
    assert float(p50) > 20.3
    assert float(p90 - p10) >= 3.0
