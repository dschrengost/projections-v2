from __future__ import annotations

import numpy as np
import pandas as pd

from projections.sim_v2.minutes_allocator import allocate_team_minutes


def test_allocate_team_minutes_sums_to_target_within_tolerance() -> None:
    df = pd.DataFrame(
        {
            "minutes_demand": [45.0, 45.0, 45.0, 45.0, 45.0, 45.0],  # sum=270
            "active": [1, 1, 1, 1, 1, 1],
            "priority": [45.0, 40.0, 35.0, 30.0, 20.0, 10.0],
        }
    )
    out = allocate_team_minutes(
        df,
        demand_col="minutes_demand",
        active_col="active",
        priority_col="priority",
        target_total=240.0,
    )
    assert out.shape == (len(df),)
    assert abs(float(out.sum()) - 240.0) <= 1e-3
    assert np.all(out >= -1e-9)
    assert np.all(out <= 48.0 + 1e-6)


def test_allocate_team_minutes_inactive_always_zero() -> None:
    df = pd.DataFrame(
        {
            "minutes_demand": [30.0, 30.0, 30.0, 30.0],
            "active": [1, 0, 1, 0],
            "priority": [30.0, 30.0, 30.0, 30.0],
        }
    )
    out = allocate_team_minutes(
        df,
        demand_col="minutes_demand",
        active_col="active",
        priority_col="priority",
        target_total=90.0,
    )
    assert out[1] == 0.0
    assert out[3] == 0.0
    assert abs(float(out.sum()) - 90.0) <= 1e-3


def test_allocate_team_minutes_priority_protects_when_shrinking() -> None:
    # Same demand for everyone, but player 0 has much higher priority; they should lose less.
    df = pd.DataFrame(
        {
            "minutes_demand": [45.0] * 6,
            "active": [1] * 6,
            "priority": [100.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    out = allocate_team_minutes(
        df,
        demand_col="minutes_demand",
        active_col="active",
        priority_col="priority",
        target_total=240.0,
        k=3.0,
    )
    assert abs(float(out.sum()) - 240.0) <= 1e-3
    assert out[0] > out[1]
    assert abs(out[0] - 45.0) < abs(out[1] - 45.0)


def test_allocate_team_minutes_respects_caps() -> None:
    df = pd.DataFrame(
        {
            "minutes_demand": [30.0] * 6,  # sum=180, need +60 minutes
            "active": [1] * 6,
            "priority": [1.0] * 6,
            "cap": [48.0, 48.0, 48.0, 48.0, 48.0, 31.0],  # last player capped low
        }
    )
    out = allocate_team_minutes(
        df,
        demand_col="minutes_demand",
        active_col="active",
        priority_col="priority",
        cap_col="cap",
        target_total=240.0,
    )
    assert abs(float(out.sum()) - 240.0) <= 1e-3
    assert out[-1] <= 31.0 + 1e-6
    assert np.all(out <= df["cap"].to_numpy(dtype=float) + 1e-6)


def test_allocate_team_minutes_all_demands_zero_returns_zeros() -> None:
    df = pd.DataFrame(
        {
            "minutes_demand": [0.0, 0.0, 0.0],
            "active": [1, 1, 1],
            "priority": [10.0, 5.0, 1.0],
        }
    )
    out = allocate_team_minutes(
        df,
        demand_col="minutes_demand",
        active_col="active",
        priority_col="priority",
        target_total=240.0,
    )
    assert np.all(out == 0.0)


def test_allocate_team_minutes_constant_priority_is_symmetric() -> None:
    df = pd.DataFrame(
        {
            "minutes_demand": [45.0] * 6,  # sum=270, need -30
            "active": [1] * 6,
            "priority": [1.0] * 6,  # constant
        }
    )
    out = allocate_team_minutes(
        df,
        demand_col="minutes_demand",
        active_col="active",
        priority_col="priority",
        target_total=240.0,
    )
    assert abs(float(out.sum()) - 240.0) <= 1e-3
    # With equal weights and no bounds binding, everyone shifts equally: 45 -> 40.
    assert np.allclose(out, 40.0, atol=1e-3)


def test_allocate_team_minutes_infeasible_caps_returns_max_possible() -> None:
    # 4 active players with cap=48 cannot reach 240; expect to return caps.
    df = pd.DataFrame(
        {
            "minutes_demand": [48.0, 48.0, 48.0, 48.0, 48.0],
            "active": [1, 1, 1, 1, 0],
            "priority": [10.0, 9.0, 8.0, 7.0, 6.0],
        }
    )
    out = allocate_team_minutes(
        df,
        demand_col="minutes_demand",
        active_col="active",
        priority_col="priority",
        target_total=240.0,
    )
    assert out[-1] == 0.0
    assert abs(float(out.sum()) - 192.0) <= 1e-6
