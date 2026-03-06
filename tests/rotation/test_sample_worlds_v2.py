from __future__ import annotations

import pandas as pd
import torch

from projections.rotation.game_transformer_v2 import FLOW_TARGET_COLUMNS_V1
from projections.rotation.sample_worlds_v2 import (
    MakeModelConfig,
    _align_flow_to_backbone_budgets,
    check_world_contracts,
    project_flow_stats_to_contract,
    sample_worlds_for_batch,
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


def test_align_flow_backbone_beta_binomial_ft_only_preserves_fg_legacy_and_contracts() -> None:
    cols = list(FLOW_TARGET_COLUMNS_V1)
    flow = torch.zeros((1, 30, len(cols)), dtype=torch.float32)
    valid = torch.ones((1, 30), dtype=torch.bool)
    active = torch.ones((1, 30), dtype=torch.bool)
    team_index = _team_index(1)

    flow[..., cols.index("fga2")] = 1.5
    flow[..., cols.index("fg2m")] = 0.75
    flow[..., cols.index("fga3")] = 1.0
    flow[..., cols.index("fg3m")] = 0.35
    flow[..., cols.index("fta")] = 0.8
    flow[..., cols.index("ftm")] = 0.5
    flow[..., cols.index("tov")] = 0.2
    flow[..., cols.index("oreb")] = 0.3

    backbone_fga = torch.tensor([[80.0, 78.0]], dtype=torch.float32)
    backbone_fta = torch.tensor([[22.0, 24.0]], dtype=torch.float32)
    backbone_tov = torch.tensor([[13.0, 14.0]], dtype=torch.float32)
    backbone_oreb = torch.tensor([[10.0, 9.0]], dtype=torch.float32)
    backbone_share3 = torch.tensor([[0.38, 0.36]], dtype=torch.float32)

    legacy = _align_flow_to_backbone_budgets(
        flow_values=flow,
        valid_mask=valid,
        team_index=team_index,
        active_mask=active,
        flow_target_columns=cols,
        backbone_fga=backbone_fga,
        backbone_fta=backbone_fta,
        backbone_tov=backbone_tov,
        backbone_oreb=backbone_oreb,
        backbone_three_pa_share=backbone_share3,
        make_model_config=MakeModelConfig(mode="legacy"),
    )
    torch.manual_seed(42)
    ft_only = _align_flow_to_backbone_budgets(
        flow_values=flow,
        valid_mask=valid,
        team_index=team_index,
        active_mask=active,
        flow_target_columns=cols,
        backbone_fga=backbone_fga,
        backbone_fta=backbone_fta,
        backbone_tov=backbone_tov,
        backbone_oreb=backbone_oreb,
        backbone_three_pa_share=backbone_share3,
        make_model_config=MakeModelConfig(mode="beta_binomial_ft"),
    )

    fga2 = ft_only[..., cols.index("fga2")]
    fg2m = ft_only[..., cols.index("fg2m")]
    fga3 = ft_only[..., cols.index("fga3")]
    fg3m = ft_only[..., cols.index("fg3m")]
    fta = ft_only[..., cols.index("fta")]
    ftm = ft_only[..., cols.index("ftm")]
    assert torch.all(fg2m <= fga2 + 1e-6)
    assert torch.all(fg3m <= fga3 + 1e-6)
    assert torch.all(ftm <= fta + 1e-6)
    assert torch.min(ft_only).item() >= 0.0

    # FT-only mode must keep FG make reconstruction identical to legacy.
    assert torch.allclose(
        ft_only[..., [cols.index("fg2m"), cols.index("fg3m")]],
        legacy[..., [cols.index("fg2m"), cols.index("fg3m")]],
        atol=1e-6,
    )


def test_sample_worlds_for_batch_honors_force_active_worlds_mask() -> None:
    cols = list(FLOW_TARGET_COLUMNS_V1)

    class _FlowHead:
        def sample(self, z: torch.Tensor, **_: object) -> torch.Tensor:
            return torch.ones_like(z, dtype=torch.float32)

    class _Out:
        def __init__(self, valid_flat: torch.Tensor, active_flat: torch.Tensor, minutes_flat: torch.Tensor, team_idx: torch.Tensor):
            self.player_states = torch.zeros((valid_flat.shape[0], 30, 4), dtype=torch.float32)
            self.team_states = torch.zeros((valid_flat.shape[0], 2, 4), dtype=torch.float32)
            self.game_state = torch.zeros((valid_flat.shape[0], 4), dtype=torch.float32)
            self.player_valid_mask = valid_flat
            self.player_team_index = team_idx
            self.active = type("Active", (), {"active_mask": active_flat})()
            self.minutes = type("Minutes", (), {"minutes": minutes_flat})()
            self.flow = None
            self.backbone = None
            self.usage_share = None
            self.efficiency = None

    class _Model:
        def __init__(self) -> None:
            self.flow_head = _FlowHead()
            self.flow_target_columns = cols
            self.enable_possession_backbone = False

        def __call__(self, pf: torch.Tensor, pvm: torch.Tensor, **_: object) -> _Out:
            bsz = int(pf.shape[0])
            valid_flat = pvm.reshape(bsz, -1).to(dtype=torch.bool)
            active_flat = torch.zeros_like(valid_flat, dtype=torch.bool)
            minutes_flat = torch.zeros((bsz, 30), dtype=torch.float32)
            # Keep team-minute contracts feasible: 5 players x 48 per team.
            minutes_flat[:, :5] = 48.0
            minutes_flat[:, 15:20] = 48.0
            team_idx = _team_index(bsz)
            return _Out(valid_flat=valid_flat, active_flat=active_flat, minutes_flat=minutes_flat, team_idx=team_idx)

    player_ids = torch.arange(1001, 1031, dtype=torch.long).reshape(1, 2, 15)
    team_ids = torch.tensor([[10, 20]], dtype=torch.long)
    player_valid_mask = torch.zeros((1, 2, 15), dtype=torch.bool)
    player_valid_mask[:, 0, :5] = True
    player_valid_mask[:, 1, :5] = True
    force_active_worlds = torch.zeros((1, 2, 15), dtype=torch.bool)
    force_active_worlds[:, 0, 0] = True  # starter-like slot
    force_active_worlds[:, 1, 0] = True  # manual force-in slot

    batch: dict[str, torch.Tensor | list[str]] = {
        "player_features": torch.zeros((1, 2, 15, 2), dtype=torch.float32),
        "player_valid_mask": player_valid_mask,
        "force_active_worlds": force_active_worlds,
        "game_features": torch.zeros((1, 0), dtype=torch.float32),
        "team_features": torch.zeros((1, 2, 0), dtype=torch.float32),
        "player_ids": player_ids,
        "team_ids": team_ids,
        "game_id_norm": ["1001"],
        "game_date": ["2026-01-18"],
    }

    worlds_df, checks = sample_worlds_for_batch(
        _Model(),
        batch,
        device=torch.device("cpu"),
        num_worlds=4,
        chunk_size=4,
        active_temperature=1.0,
        strict_contracts=True,
    )
    assert checks["total_violations"] == 0
    assert not worlds_df.empty

    forced_ids = {1001, 1016}
    for pid in forced_ids:
        pid_rows = worlds_df.loc[worlds_df["player_id"] == pid]
        assert len(pid_rows) == 4
        assert int(pid_rows["active"].min()) == 1
