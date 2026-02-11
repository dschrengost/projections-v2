from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from projections.minutes_alloc.occupancy_sparse import (
    OccupancySparseConfig,
    apply_occupancy_sparse_allocation,
)


def _base_team_frame(*, all_out: bool = False) -> pd.DataFrame:
    n = 8
    status = ["OK"] * n
    is_out = [0] * n
    if all_out:
        status = ["OUT"] * n
        is_out = [1] * n
    else:
        status[6] = "OUT"
        is_out[6] = 1

    frame = pd.DataFrame(
        {
            "game_id": [101] * n,
            "team_id": [200] * n,
            "player_id": list(range(1, n + 1)),
            "play_prob": np.linspace(0.2, 0.95, n),
            "minutes_p10": np.linspace(6.0, 24.0, n),
            "minutes_p50": np.linspace(10.0, 32.0, n),
            "minutes_p90": np.linspace(14.0, 38.0, n),
            "status": status,
            "is_out": is_out,
            "lineup_role": ["PROJECTED_STARTER"] * 5 + ["BENCH"] * 3,
            "starter_flag": [1] * 5 + [0] * 3,
            "is_projected_starter": [1] * 5 + [0] * 3,
            "is_confirmed_starter": [0] * n,
            "is_starter": [1] * 5 + [0] * 3,
            "spread_home": [-4.5] * n,
            "total": [228.5] * n,
            "home_flag": [1] * n,
        }
    )
    return frame


def _deep_rotation_frame() -> pd.DataFrame:
    n = 13
    return pd.DataFrame(
        {
            "game_id": [102] * n,
            "team_id": [201] * n,
            "player_id": list(range(1, n + 1)),
            "play_prob": [0.98, 0.96, 0.95, 0.93, 0.91, 0.89, 0.87, 0.84, 0.82, 0.80, 0.79, 0.77, 0.75],
            "minutes_p10": np.linspace(20.0, 5.0, n),
            "minutes_p50": np.linspace(30.0, 10.0, n),
            "minutes_p90": np.linspace(36.0, 14.0, n),
            "status": ["OK"] * n,
            "is_out": [0] * n,
            "lineup_role": ["PROJECTED_STARTER"] * 5 + ["BENCH"] * (n - 5),
            "starter_flag": [1] * 5 + [0] * (n - 5),
            "is_projected_starter": [1] * 5 + [0] * (n - 5),
            "is_confirmed_starter": [0] * n,
            "is_starter": [1] * 5 + [0] * (n - 5),
            "spread_home": [-10.5] * n,
            "total": [239.5] * n,
            "home_flag": [1] * n,
        }
    )


def test_apply_occupancy_sparse_allocation_invariants() -> None:
    frame = _base_team_frame(all_out=False)
    cfg = OccupancySparseConfig(starter_floor=0.8)
    out, diag = apply_occupancy_sparse_allocation(frame, config=cfg)

    assert not out.empty
    assert set(
        [
            "minutes_occ",
            "play_prob_occ",
            "minutes_p10_occ",
            "minutes_p90_occ",
            "eligible_flag_occ",
            "out_flag_occ",
            "starter_flag_occ",
        ]
    ).issubset(out.columns)

    out_rows = out[out["out_flag_occ"] == 1]
    assert not out_rows.empty
    assert (pd.to_numeric(out_rows["minutes_occ"], errors="coerce").fillna(0.0) == 0.0).all()
    assert (pd.to_numeric(out_rows["play_prob_occ"], errors="coerce").fillna(0.0) == 0.0).all()

    active = out["out_flag_occ"] == 0
    assert float(pd.to_numeric(out.loc[active, "minutes_occ"], errors="coerce").fillna(0.0).sum()) == pytest.approx(240.0)

    p10 = pd.to_numeric(out["minutes_p10_occ"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    p50 = pd.to_numeric(out["minutes_occ"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    p90 = pd.to_numeric(out["minutes_p90_occ"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    assert np.all(p10 <= p50 + 1e-9)
    assert np.all(p50 <= p90 + 1e-9)

    starter_active = (out["starter_flag_occ"] == 1) & (out["out_flag_occ"] == 0) & (p50 > 0.0)
    starter_probs = pd.to_numeric(out.loc[starter_active, "play_prob_occ"], errors="coerce").fillna(0.0)
    assert not starter_probs.empty
    assert float(starter_probs.min()) >= 0.8

    assert not diag.empty
    assert float(pd.to_numeric(diag["team_minutes_sum_dev"], errors="coerce").fillna(0.0).max()) < 1e-6


def test_apply_occupancy_sparse_allocation_handles_all_out_team() -> None:
    frame = _base_team_frame(all_out=True)
    out, diag = apply_occupancy_sparse_allocation(frame, config=OccupancySparseConfig())

    assert float(pd.to_numeric(out["minutes_occ"], errors="coerce").fillna(0.0).sum()) == 0.0
    assert float(pd.to_numeric(out["play_prob_occ"], errors="coerce").fillna(0.0).sum()) == 0.0
    assert not diag.empty
    assert int(pd.to_numeric(diag["active_count"], errors="coerce").fillna(0).iloc[0]) == 0
    assert float(pd.to_numeric(diag["team_minutes_sum"], errors="coerce").fillna(0.0).iloc[0]) == 0.0


def test_occupancy_sparse_dynamic_k_max_expands_for_deep_team() -> None:
    frame = _deep_rotation_frame()

    static_cfg = OccupancySparseConfig(
        k_min=8,
        k_max=11,
        dynamic_k_bounds_enabled=False,
    )
    _, static_diag = apply_occupancy_sparse_allocation(frame, config=static_cfg)
    static_n_eligible = int(pd.to_numeric(static_diag["n_eligible"], errors="coerce").fillna(0).iloc[0])

    dynamic_cfg = OccupancySparseConfig(
        k_min=8,
        k_max=11,
        dynamic_k_bounds_enabled=True,
        dynamic_k_max_cap=13,
        dynamic_k_min_floor=7,
        dynamic_k_window=3,
        dynamic_depth_prob_floor=0.06,
        dynamic_depth_minutes_floor=4.0,
        dynamic_bench_share_midpoint=0.18,
        dynamic_bench_share_scale=25.0,
    )
    _, dynamic_diag = apply_occupancy_sparse_allocation(frame, config=dynamic_cfg)
    dynamic_n_eligible = int(pd.to_numeric(dynamic_diag["n_eligible"], errors="coerce").fillna(0).iloc[0])
    dynamic_k_max_eff = int(pd.to_numeric(dynamic_diag["k_max_eff"], errors="coerce").fillna(0).iloc[0])

    assert static_n_eligible <= 11
    assert dynamic_n_eligible > static_n_eligible
    assert dynamic_k_max_eff > 11


def test_occupancy_sparse_config_parses_dynamic_payload() -> None:
    cfg = OccupancySparseConfig.from_payload(
        {
            "dynamic_k_bounds_enabled": "true",
            "dynamic_k_max_cap": 14,
            "dynamic_k_min_floor": 6,
            "dynamic_k_window": 4,
            "dynamic_depth_prob_floor": 0.08,
            "dynamic_depth_minutes_floor": 5.0,
            "dynamic_bench_share_midpoint": 0.2,
            "dynamic_bench_share_scale": 30.0,
        }
    )
    assert cfg.dynamic_k_bounds_enabled is True
    assert cfg.dynamic_k_max_cap == 14
    assert cfg.dynamic_k_min_floor == 6
    assert cfg.dynamic_k_window == 4
    assert cfg.dynamic_depth_prob_floor == pytest.approx(0.08)
    assert cfg.dynamic_depth_minutes_floor == pytest.approx(5.0)
    assert cfg.dynamic_bench_share_midpoint == pytest.approx(0.2)
    assert cfg.dynamic_bench_share_scale == pytest.approx(30.0)
