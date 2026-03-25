from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from projections.rotation.game_transformer_v2 import FLOW_TARGET_COLUMNS_V1
from projections.rotation.sample_worlds_v2 import (
    MakeModelConfig,
    _align_flow_to_backbone_budgets,
    _build_world_rows,
    _compute_dk_fpts,
    _flow_idx,
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


def _build_world_rows_reference(
    *,
    batch: dict[str, torch.Tensor | list[str]],
    world_offset: int,
    minutes: torch.Tensor,
    active_mask: torch.Tensor,
    flow_values: torch.Tensor,
    flow_target_columns: list[str],
) -> pd.DataFrame:
    bsz = int(batch["player_features"].shape[0])  # type: ignore[index]
    n_worlds = int(minutes.shape[1])
    valid = batch["player_valid_mask"].cpu().numpy().astype(bool)  # type: ignore[index]
    player_ids = batch["player_ids"].cpu().numpy().astype(np.int64)  # type: ignore[index]
    team_ids = batch["team_ids"].cpu().numpy().astype(np.int64)  # type: ignore[index]
    game_ids = [str(v) for v in batch["game_id_norm"]]  # type: ignore[index]
    game_dates = [str(v) for v in batch["game_date"]]  # type: ignore[index]

    mins_np = minutes.cpu().numpy()
    active_np = active_mask.cpu().numpy().astype(bool)
    flow_np = flow_values.cpu().numpy()

    idx = {name: _flow_idx(flow_target_columns, name) for name in FLOW_TARGET_COLUMNS_V1}
    pf_idx = flow_target_columns.index("pf") if "pf" in flow_target_columns else None

    rows: list[dict[str, object]] = []
    for b_idx in range(bsz):
        valid_flat = np.concatenate([valid[b_idx, 0], valid[b_idx, 1]], axis=0)
        player_flat = np.concatenate([player_ids[b_idx, 0], player_ids[b_idx, 1]], axis=0)
        team_flat = np.concatenate(
            [
                np.full((15,), int(team_ids[b_idx, 0]), dtype=np.int64),
                np.full((15,), int(team_ids[b_idx, 1]), dtype=np.int64),
            ],
            axis=0,
        )
        for w_idx in range(n_worlds):
            flow_world = flow_np[b_idx, w_idx]
            fga2 = flow_world[:, idx["fga2"]]
            fg2m = flow_world[:, idx["fg2m"]]
            fga3 = flow_world[:, idx["fga3"]]
            fg3m = flow_world[:, idx["fg3m"]]
            fta = flow_world[:, idx["fta"]]
            ftm = flow_world[:, idx["ftm"]]
            oreb = flow_world[:, idx["oreb"]]
            dreb = flow_world[:, idx["dreb"]]
            ast = flow_world[:, idx["ast"]]
            stl = flow_world[:, idx["stl"]]
            blk = flow_world[:, idx["blk"]]
            tov = flow_world[:, idx["tov"]]
            pf = flow_world[:, int(pf_idx)] if pf_idx is not None else np.zeros_like(fga2)
            fga = fga2 + fga3
            fgm = fg2m + fg3m
            pts = 2.0 * fg2m + 3.0 * fg3m + ftm
            reb = oreb + dreb
            dk = _compute_dk_fpts(
                pts=torch.from_numpy(pts),
                reb=torch.from_numpy(reb),
                ast=torch.from_numpy(ast),
                stl=torch.from_numpy(stl),
                blk=torch.from_numpy(blk),
                tov=torch.from_numpy(tov),
            ).numpy()
            for p_idx in np.where(valid_flat)[0]:
                rows.append(
                    {
                        "world_idx": int(world_offset + w_idx),
                        "game_id": int(game_ids[b_idx]),
                        "game_id_norm": str(game_ids[b_idx]),
                        "game_date": str(game_dates[b_idx]),
                        "team_id": int(team_flat[p_idx]),
                        "player_id": int(player_flat[p_idx]),
                        "active": int(bool(active_np[b_idx, w_idx, p_idx])),
                        "minutes": float(mins_np[b_idx, w_idx, p_idx]),
                        "fga2": float(fga2[p_idx]),
                        "fg2m": float(fg2m[p_idx]),
                        "fga3": float(fga3[p_idx]),
                        "fg3m": float(fg3m[p_idx]),
                        "fta": float(fta[p_idx]),
                        "ftm": float(ftm[p_idx]),
                        "oreb": float(oreb[p_idx]),
                        "dreb": float(dreb[p_idx]),
                        "ast": float(ast[p_idx]),
                        "stl": float(stl[p_idx]),
                        "blk": float(blk[p_idx]),
                        "tov": float(tov[p_idx]),
                        "pf": float(pf[p_idx]),
                        "fga": float(fga[p_idx]),
                        "fgm": float(fgm[p_idx]),
                        "fg3a": float(fga3[p_idx]),
                        "pts": float(pts[p_idx]),
                        "reb": float(reb[p_idx]),
                        "plus_minus": 0.0,
                        "dk_fpts": float(dk[p_idx]),
                    }
                )
    return pd.DataFrame.from_records(rows)


def test_build_world_rows_vectorized_matches_reference() -> None:
    torch.manual_seed(7)
    cols = list(FLOW_TARGET_COLUMNS_V1)
    bsz = 2
    n_worlds = 4
    n_targets = len(cols)

    player_ids = torch.arange(1, 1 + bsz * 2 * 15, dtype=torch.long).reshape(bsz, 2, 15)
    valid = torch.zeros((bsz, 2, 15), dtype=torch.bool)
    valid[0, 0, :11] = True
    valid[0, 1, :9] = True
    valid[1, 0, :10] = True
    valid[1, 1, :8] = True

    batch: dict[str, torch.Tensor | list[str]] = {
        "player_features": torch.zeros((bsz, 2, 15, 3), dtype=torch.float32),
        "player_valid_mask": valid,
        "player_ids": player_ids,
        "team_ids": torch.tensor([[100, 200], [300, 400]], dtype=torch.long),
        "game_id_norm": ["0000001001", "0000001002"],
        "game_date": ["2026-03-10", "2026-03-11"],
    }
    minutes = torch.rand((bsz, n_worlds, 30), dtype=torch.float32) * 48.0
    active = torch.rand((bsz, n_worlds, 30), dtype=torch.float32) > 0.35
    flow = torch.rand((bsz, n_worlds, 30, n_targets), dtype=torch.float32) * 6.0

    out = _build_world_rows(
        batch=batch,
        world_offset=12,
        minutes=minutes,
        active_mask=active,
        flow_values=flow,
        flow_target_columns=cols,
    )
    ref = _build_world_rows_reference(
        batch=batch,
        world_offset=12,
        minutes=minutes,
        active_mask=active,
        flow_values=flow,
        flow_target_columns=cols,
    )

    sort_cols = ["world_idx", "game_id", "team_id", "player_id"]
    out = out.sort_values(sort_cols).reset_index(drop=True)
    ref = ref.sort_values(sort_cols).reset_index(drop=True)
    assert len(out) == len(ref) == int(valid.sum().item()) * n_worlds
    pd.testing.assert_frame_equal(out, ref, check_dtype=False, check_exact=False, rtol=1e-6, atol=1e-6)


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


def test_summarize_worlds_to_projections_handles_sparse_index_and_null_key_rows() -> None:
    worlds = pd.DataFrame(
        [
            {
                "world_idx": 0,
                "game_date": "2026-01-18",
                "game_id": 1001,
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
            {
                "world_idx": 2,
                "game_date": "2026-01-18",
                "game_id": None,  # Should be dropped (matches pandas groupby(dropna=True) behavior).
                "team_id": 10,
                "player_id": 101,
                "active": 1,
                "minutes": 10.0,
                "dk_fpts": 10.0,
                "pts": 5.0,
                "reb": 2.0,
                "ast": 1.0,
                "stl": 0.0,
                "blk": 0.0,
                "tov": 1.0,
            },
        ]
    )
    worlds.index = pd.Index([100, 300, 900])

    df = summarize_worlds_to_projections(worlds, sim_profile="game_transformer_v2")

    assert len(df) == 1
    row = df.iloc[0]
    assert row["game_id"] == 1001
    assert row["n_worlds"] == 2
    assert row["sim_p_active"] == 0.5
    assert row["dk_fpts_mean_uncond"] == 20.0


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


def test_sample_worlds_for_batch_zero_minute_forced_active_rows_remain_inactive() -> None:
    cols = list(FLOW_TARGET_COLUMNS_V1)

    class _FlowHead:
        def sample(self, z: torch.Tensor, **_: object) -> torch.Tensor:
            return torch.ones_like(z, dtype=torch.float32)

    class _Out:
        def __init__(
            self,
            valid_flat: torch.Tensor,
            active_flat: torch.Tensor,
            minutes_flat: torch.Tensor,
            team_idx: torch.Tensor,
        ) -> None:
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
            minutes_flat[:, 1:6] = 48.0
            minutes_flat[:, 15:20] = 48.0
            team_idx = _team_index(bsz)
            return _Out(
                valid_flat=valid_flat,
                active_flat=active_flat,
                minutes_flat=minutes_flat,
                team_idx=team_idx,
            )

    player_ids = torch.arange(1001, 1031, dtype=torch.long).reshape(1, 2, 15)
    player_valid_mask = torch.zeros((1, 2, 15), dtype=torch.bool)
    player_valid_mask[:, 0, :6] = True
    player_valid_mask[:, 1, :5] = True
    force_active_worlds = torch.zeros((1, 2, 15), dtype=torch.bool)
    force_active_worlds[:, 0, 0] = True

    batch: dict[str, torch.Tensor | list[str]] = {
        "player_features": torch.zeros((1, 2, 15, 2), dtype=torch.float32),
        "player_valid_mask": player_valid_mask,
        "force_active_worlds": force_active_worlds,
        "game_features": torch.zeros((1, 0), dtype=torch.float32),
        "team_features": torch.zeros((1, 2, 0), dtype=torch.float32),
        "player_ids": player_ids,
        "team_ids": torch.tensor([[10, 20]], dtype=torch.long),
        "game_id_norm": ["1001"],
        "game_date": ["2026-01-18"],
    }

    worlds_df, checks = sample_worlds_for_batch(
        _Model(),
        batch,
        device=torch.device("cpu"),
        num_worlds=2,
        chunk_size=2,
        active_temperature=1.0,
        strict_contracts=True,
    )
    assert checks["total_violations"] == 0
    forced_rows = worlds_df.loc[worlds_df["player_id"] == 1001]
    assert len(forced_rows) == 2
    assert int(forced_rows["active"].max()) == 0
    assert float(forced_rows["minutes"].max()) <= 1e-6
    assert float(forced_rows["dk_fpts"].max()) <= 1e-6


def test_sample_worlds_for_batch_applies_props_anchor_floor_for_manual_and_low_minute_starter() -> None:
    cols = list(FLOW_TARGET_COLUMNS_V1)

    class _FlowHead:
        def sample(self, z: torch.Tensor, **_: object) -> torch.Tensor:
            return torch.ones_like(z, dtype=torch.float32)

    class _Out:
        def __init__(
            self,
            valid_flat: torch.Tensor,
            active_flat: torch.Tensor,
            minutes_flat: torch.Tensor,
            team_idx: torch.Tensor,
        ):
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
            # Ten-player rotation baseline: 24 min each per team (sum=240).
            minutes_flat[:, :10] = 24.0
            minutes_flat[:, 15:25] = 24.0
            # Home slot0 simulates low-minute projected starter.
            minutes_flat[:, 0] = 4.0
            team_idx = _team_index(bsz)
            return _Out(valid_flat=valid_flat, active_flat=active_flat, minutes_flat=minutes_flat, team_idx=team_idx)

    player_ids = torch.arange(1001, 1031, dtype=torch.long).reshape(1, 2, 15)
    team_ids = torch.tensor([[10, 20]], dtype=torch.long)
    player_valid_mask = torch.zeros((1, 2, 15), dtype=torch.bool)
    player_valid_mask[:, 0, :10] = True
    player_valid_mask[:, 1, :10] = True
    force_active_worlds = torch.zeros((1, 2, 15), dtype=torch.bool)
    force_active_worlds[:, 0, 0] = True
    force_active_worlds[:, 0, 1] = True
    force_active_worlds[:, 1, 0] = True
    starter_force_active_worlds = torch.zeros((1, 2, 15), dtype=torch.bool)
    starter_force_active_worlds[:, 0, 0] = True
    starter_force_active_worlds[:, 0, 1] = True
    force_active_minutes_anchor = torch.zeros((1, 2, 15), dtype=torch.float32)
    force_active_minutes_anchor[:, 0, 0] = 40.0
    force_active_minutes_anchor[:, 0, 1] = 40.0
    force_active_minutes_anchor[:, 1, 0] = 40.0

    batch: dict[str, torch.Tensor | list[str]] = {
        "player_features": torch.zeros((1, 2, 15, 2), dtype=torch.float32),
        "player_valid_mask": player_valid_mask,
        "force_active_worlds": force_active_worlds,
        "starter_force_active_worlds": starter_force_active_worlds,
        "force_active_minutes_anchor": force_active_minutes_anchor,
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
        num_worlds=3,
        chunk_size=3,
        active_temperature=1.0,
        strict_contracts=True,
    )
    assert checks["total_violations"] == 0
    assert not worlds_df.empty

    # Default floor policy: 0.65 * 40 = 26 minutes.
    # 1001: starter + low-minute trigger (<10) => floor applies.
    starter_rows = worlds_df.loc[worlds_df["player_id"] == 1001]
    assert int(starter_rows["active"].min()) == 1
    assert float(starter_rows["minutes"].min()) >= 25.99

    # 1016: manual force-in (non-starter) => floor applies unconditionally.
    manual_rows = worlds_df.loc[worlds_df["player_id"] == 1016]
    assert int(manual_rows["active"].min()) == 1
    assert float(manual_rows["minutes"].min()) >= 25.99

    # 1002: starter with >=10 baseline minutes => no floor trigger.
    starter_not_low_rows = worlds_df.loc[worlds_df["player_id"] == 1002]
    assert int(starter_not_low_rows["active"].min()) == 1
    assert float(starter_not_low_rows["minutes"].max()) < 26.0

    team_minutes = (
        worlds_df.groupby(["world_idx", "team_id"], as_index=False)["minutes"]
        .sum()
        .sort_values(["world_idx", "team_id"])
        .reset_index(drop=True)
    )
    assert (team_minutes["minutes"] - 240.0).abs().max() <= 1e-3
