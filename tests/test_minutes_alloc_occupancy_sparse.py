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
