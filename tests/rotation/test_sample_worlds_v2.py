from __future__ import annotations

import pandas as pd
import torch

from projections.rotation.game_transformer_v2 import FLOW_TARGET_COLUMNS_V1
from projections.rotation.sample_worlds_v2 import (
    check_world_contracts,
    project_flow_stats_to_contract,
    summarize_worlds_to_projections,
)


def _team_index(num_worlds: int) -> torch.Tensor:
    return torch.cat(
        [
            torch.zeros((num_worlds, 15), dtype=torch.long),
            torch.ones((num_worlds, 15), dtype=torch.long),
        ],
        dim=1,
    )


def test_project_flow_stats_to_contract_clamps_and_caps_makes() -> None:
    cols = list(FLOW_TARGET_COLUMNS_V1)
    flow = torch.zeros((1, 30, len(cols)), dtype=torch.float32)
    flow[..., cols.index("fga2")] = 3.0
    flow[..., cols.index("fg2m")] = 7.0
    flow[..., cols.index("fga3")] = -2.0
    flow[..., cols.index("fg3m")] = 2.0
    flow[..., cols.index("fta")] = 1.0
    flow[..., cols.index("ftm")] = 5.0

    out = project_flow_stats_to_contract(flow, flow_target_columns=cols)
    assert torch.min(out).item() >= 0.0
    assert torch.max(out[..., cols.index("fg2m")] - out[..., cols.index("fga2")]).item() <= 0.0
    assert torch.max(out[..., cols.index("fg3m")] - out[..., cols.index("fga3")]).item() <= 0.0
    assert torch.max(out[..., cols.index("ftm")] - out[..., cols.index("fta")]).item() <= 0.0


def test_check_world_contracts_reports_clean_and_violation_cases() -> None:
    cols = list(FLOW_TARGET_COLUMNS_V1)
    valid = torch.ones((2, 30), dtype=torch.bool)
    t_idx = _team_index(2)

    minutes = torch.zeros((2, 30), dtype=torch.float32)
    minutes[:, :5] = 48.0
    minutes[:, 15:20] = 48.0
    flow = torch.ones((2, 30, len(cols)), dtype=torch.float32)
    clean = check_world_contracts(
        minutes=minutes,
        flow_values=flow,
        valid_mask=valid,
        team_index=t_idx,
        flow_target_columns=cols,
    )
    assert clean["total_violations"] == 0

    bad_minutes = minutes.clone()
    bad_minutes[1, 0] = 60.0
    bad_flow = flow.clone()
    bad_flow[1, 0, cols.index("fg2m")] = 10.0
    bad_flow[1, 0, cols.index("fga2")] = 2.0
    bad_flow[1, 1, cols.index("fta")] = -1.0
    bad = check_world_contracts(
        minutes=bad_minutes,
        flow_values=bad_flow,
        valid_mask=valid,
        team_index=t_idx,
        flow_target_columns=cols,
    )
    assert bad["total_violations"] > 0
    assert bad["minutes_over_48"] > 0
    assert bad["fg2m_gt_fga2"] > 0


def test_check_world_contracts_flags_inactive_nonzero_stats_when_active_mask_provided() -> None:
    cols = list(FLOW_TARGET_COLUMNS_V1)
    valid = torch.ones((1, 30), dtype=torch.bool)
    t_idx = _team_index(1)
    active = torch.ones((1, 30), dtype=torch.bool)
    active[:, 0] = False
    minutes = torch.zeros((1, 30), dtype=torch.float32)
    minutes[:, :5] = 48.0
    minutes[:, 15:20] = 48.0
    flow = torch.zeros((1, 30, len(cols)), dtype=torch.float32)
    flow[:, 0, cols.index("fga2")] = 3.0
    out = check_world_contracts(
        minutes=minutes,
        flow_values=flow,
        valid_mask=valid,
        team_index=t_idx,
        flow_target_columns=cols,
        active_mask=active,
    )
    assert out["inactive_nonzero_stats"] > 0


def test_summarize_worlds_to_projections_emits_contract_columns_and_semantics() -> None:
    worlds = [
        {
            "world_idx": 0,
            "game_date": "2026-01-18",
            "game_id": 1001,
            "game_id_norm": "0000001001",
            "team_id": 10,
            "player_id": 101,
            "active": 1,
            "minutes": 30.0,
            "dk_fpts": 40.0,
            "pts": 20.0,
            "reb": 8.0,
            "ast": 5.0,
            "stl": 1.0,
            "blk": 1.0,
            "tov": 2.0,
        },
        {
            "world_idx": 1,
            "game_date": "2026-01-18",
            "game_id": 1001,
            "game_id_norm": "0000001001",
            "team_id": 10,
            "player_id": 101,
            "active": 0,
            "minutes": 0.0,
            "dk_fpts": 0.0,
            "pts": 0.0,
            "reb": 0.0,
            "ast": 0.0,
            "stl": 0.0,
            "blk": 0.0,
            "tov": 0.0,
        },
    ]
    df = summarize_worlds_to_projections(pd.DataFrame(worlds), sim_profile="game_transformer_v2")
    assert len(df) == 1
    row = df.iloc[0]
    assert row["sim_profile"] == "game_transformer_v2"
    assert row["n_worlds"] == 2
    assert row["sim_p_active"] == 0.5
    assert row["dk_fpts_mean"] == 40.0
    assert row["dk_fpts_mean_uncond"] == 20.0
    assert row["minutes_sim_mean"] == 30.0
    assert row["minutes_sim_mean_uncond"] == 15.0
    # Canonical fields are added via add_canonical_projection_fields.
    assert row["fpts_sim_uncond_mean"] == 20.0
    assert row["minutes_sim_uncond_mean"] == 15.0
