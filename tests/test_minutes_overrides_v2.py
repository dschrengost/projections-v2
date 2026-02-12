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
                "play_prob": 0.9,
            }
        )
    return pd.DataFrame(rows)


def test_apply_minutes_overrides_v2_lock_and_caps_sum_to_240() -> None:
    baseline = _toy_team_df()

    overrides_payload = {
        "version": 1,
        "game_date": "2026-01-18",
        "overrides": [
            {"game_id": 100, "player_id": 1, "fields": {"minutes_target": 20.0, "minutes_lock": True}},
            *[
                {
                    "game_id": 100,
                    "player_id": pid,
                    "fields": {"minutes_cap": 34.0},
                }
                for pid in range(2, 11)
            ],
        ],
    }

    resolved, diag = apply_minutes_overrides_v2(
        baseline,
        overrides_payload,
        policy=MinutesOverrideV2Policy(override_infeasible="error"),
        seed=42,
        strict=True,
    )

    assert {"game_id", "team_id", "player_id", "mu_minutes", "lb_minutes", "ub_minutes"}.issubset(resolved.columns)

    one = resolved.loc[resolved["player_id"] == 1].iloc[0]
    assert abs(float(one["lb_minutes"]) - 20.0) <= 1e-9
    assert abs(float(one["ub_minutes"]) - 20.0) <= 1e-9
    assert abs(float(one["mu_minutes"]) - 20.0) <= 1e-6

    others = resolved.loc[resolved["player_id"] != 1]
    assert np.all(others["ub_minutes"].to_numpy(dtype=float) <= 34.0 + 1e-9)
    assert np.all(others["mu_minutes"].to_numpy(dtype=float) <= 34.0 + 1e-6)
    assert abs(float(resolved["mu_minutes"].sum()) - 240.0) <= 1e-6

    assert diag["team_diagnostics"]
    team_diag = diag["team_diagnostics"][0]
    assert abs(float(team_diag["sum_mu"]) - 240.0) <= 1e-6


def test_failure_mode_raise_x_to_20_does_not_make_other_48_or_x_zero() -> None:
    baseline = _toy_team_df()
    overrides_payload = {
        "overrides": [
            {"game_id": 100, "player_id": 1, "fields": {"minutes_target": 20.0, "minutes_lock": True}},
            *[
                {
                    "game_id": 100,
                    "player_id": pid,
                    "fields": {"minutes_cap": 34.0},
                }
                for pid in range(2, 11)
            ],
        ]
    }

    resolved, _ = apply_minutes_overrides_v2(
        baseline,
        overrides_payload,
        policy=MinutesOverrideV2Policy(override_infeasible="error"),
        seed=777,
        strict=True,
    )

    mu = resolved["mu_minutes"].to_numpy(dtype=float)
    lb = resolved["lb_minutes"].to_numpy(dtype=float)
    ub = resolved["ub_minutes"].to_numpy(dtype=float)
    w = resolved["weight"].to_numpy(dtype=float)

    # Extreme world sample: try to force player 1 below 0 and player 10 above 48.
    raw = mu.copy()
    raw[0] = -12.0
    raw[-1] = 80.0

    m_clip = np.clip(raw, lb, ub)
    m_world = project_sum_with_bounds(m_clip, 240.0, lb, ub, w)

    assert abs(float(m_world.sum()) - 240.0) <= 1e-6
    assert abs(float(m_world[0]) - 20.0) <= 1e-6  # locked player cannot become 0
    assert float(m_world.max()) <= 34.0 + 1e-6  # cap prevents 48+ spikes


def test_no_override_v2_regression_keeps_baseline_mu() -> None:
    baseline = _toy_team_df()

    resolved, diag = apply_minutes_overrides_v2(
        baseline,
        overrides_payload={"overrides": []},
        policy=MinutesOverrideV2Policy(override_infeasible="error"),
        seed=9,
        strict=True,
    )

    b = resolved["b_minutes"].to_numpy(dtype=float)
    mu = resolved["mu_minutes"].to_numpy(dtype=float)
    np.testing.assert_allclose(mu, b, atol=1e-10)

    assert diag["team_diagnostics"]
    assert diag["team_diagnostics"][0]["infeasible_action"] == "no_override_baseline"
