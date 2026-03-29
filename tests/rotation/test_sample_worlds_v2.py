from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from projections.rotation.game_transformer_v2 import FLOW_TARGET_COLUMNS_V1, GameTransformerV2Config
from projections.rotation.possession_backbone import FTA_POSS_COEFF
from projections.rotation.sample_worlds_v2 import (
    AstFactorizationRuntimeConfig,
    MakeModelConfig,
    MinutesUncertaintyConfig,
    _align_flow_to_backbone_budgets,
    _build_ast_override,
    _build_creator_reconcile_alpha,
    _resolve_team_points_budget,
    _reconcile_ast_to_team_budget,
    _build_world_rows,
    _compute_dk_fpts,
    _flow_idx,
    _resolve_team_opportunity_share,
    _reconcile_points_to_team_budget,
    _reconcile_opportunities_to_team_budget,
    _reconcile_rebounds_to_opportunity_budgets,
    _reweight_top_usage_alloc_weights,
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


def test_project_flow_stats_to_contract_applies_ast_override() -> None:
    cols = list(FLOW_TARGET_COLUMNS_V1)
    flow = torch.zeros((1, 30, len(cols)), dtype=torch.float32)
    flow[..., cols.index("ast")] = 1.5
    ast_override = torch.full((1, 30), 4.25, dtype=torch.float32)

    out = project_flow_stats_to_contract(
        flow,
        flow_target_columns=cols,
        ast_override=ast_override,
    )
    assert torch.allclose(out[..., cols.index("ast")], ast_override)


def test_build_ast_override_applies_blend_budget_and_temperature() -> None:
    cols = ["pts", "ast", "reb"]
    flow_projected = torch.zeros((1, 30, 3), dtype=torch.float32)
    flow_projected[0, 0, 1] = 6.0
    flow_projected[0, 1, 1] = 2.0
    flow_projected[0, 15, 1] = 4.0
    valid_mask = torch.zeros((1, 30), dtype=torch.bool)
    valid_mask[0, 0] = True
    valid_mask[0, 1] = True
    valid_mask[0, 15] = True
    team_index = torch.cat([torch.zeros((1, 15), dtype=torch.long), torch.ones((1, 15), dtype=torch.long)], dim=1)
    team_budget = torch.tensor([[12.0, 6.0]], dtype=torch.float32)
    ast_logits = torch.full((1, 30), -10.0, dtype=torch.float32)
    ast_logits[0, 0] = 1.0
    ast_logits[0, 1] = 0.0
    ast_logits[0, 15] = 0.0

    override = _build_ast_override(
        flow_projected_base=flow_projected,
        flow_contract_columns=cols,
        player_valid_mask=valid_mask,
        player_team_index=team_index,
        team_ast_budget=team_budget,
        assist_share_logits=ast_logits,
        ast_blend_gate=None,
        runtime_config=AstFactorizationRuntimeConfig(
            ast_blend_alpha=0.5,
            assist_share_temperature=0.5,
            team_ast_budget_blend_alpha=0.5,
        ),
    )
    assert override is not None
    # Home team budget is blended between factorized 12 and flow-implied 8 -> 10,
    # then mixed 50/50 with the original flow AST (8) -> 9. Away: (6,4)->5 then 50/50 with 4 -> 4.5.
    assert override.shape == (1, 30)
    assert float(override[0, :15].sum()) == pytest.approx(9.0, rel=1e-5)
    assert float(override[0, 15:].sum()) == pytest.approx(4.5, rel=1e-5)
    # Sharpened temperature plus blend should still keep player 0 above player 1.
    assert float(override[0, 0]) > float(override[0, 1])


def test_build_ast_override_prefers_learned_gate_over_scalar_blend() -> None:
    cols = ["fga2", "ast", "fta"]
    flow_projected = torch.zeros((1, 30, len(cols)), dtype=torch.float32)
    flow_projected[0, 0, 1] = 2.0
    flow_projected[0, 1, 1] = 2.0
    valid_mask = torch.zeros((1, 30), dtype=torch.bool)
    valid_mask[0, 0] = True
    valid_mask[0, 1] = True
    team_index = torch.cat([torch.zeros((1, 15), dtype=torch.long), torch.ones((1, 15), dtype=torch.long)], dim=1)
    team_budget = torch.tensor([[10.0, 0.0]], dtype=torch.float32)
    ast_logits = torch.full((1, 30), -10.0, dtype=torch.float32)
    ast_logits[0, 0] = 2.0
    ast_logits[0, 1] = 0.0
    ast_gate = torch.zeros((1, 30), dtype=torch.float32)
    ast_gate[0, 0] = 1.0
    ast_gate[0, 1] = 0.0

    override = _build_ast_override(
        flow_projected_base=flow_projected,
        flow_contract_columns=cols,
        player_valid_mask=valid_mask,
        player_team_index=team_index,
        team_ast_budget=team_budget,
        assist_share_logits=ast_logits,
        ast_blend_gate=ast_gate,
        runtime_config=AstFactorizationRuntimeConfig(ast_blend_alpha=0.0),
    )
    assert override is not None
    factorized_player0 = float(override[0, 0])
    flow_player1 = float(override[0, 1])
    assert factorized_player0 > 2.0
    assert flow_player1 == pytest.approx(2.0, rel=1e-6)


def test_reconcile_ast_to_team_budget_blends_flow_and_factorized_shares() -> None:
    cols = ["pts", "ast", "reb"]
    flow_projected = torch.zeros((1, 30, len(cols)), dtype=torch.float32)
    flow_projected[0, 0, cols.index("ast")] = 8.0
    flow_projected[0, 1, cols.index("ast")] = 2.0
    flow_projected[0, 15, cols.index("ast")] = 5.0
    flow_projected[0, 16, cols.index("ast")] = 5.0
    valid_mask = torch.zeros((1, 30), dtype=torch.bool)
    valid_mask[0, 0:2] = True
    valid_mask[0, 15:17] = True
    active_mask = valid_mask.clone()
    team_index = torch.cat([torch.zeros((1, 15), dtype=torch.long), torch.ones((1, 15), dtype=torch.long)], dim=1)
    team_budget = torch.tensor([[12.0, 10.0]], dtype=torch.float32)
    ast_logits = torch.full((1, 30), -10.0, dtype=torch.float32)
    ast_logits[0, 0] = 3.0
    ast_logits[0, 1] = 1.0
    ast_logits[0, 15] = 0.0
    ast_logits[0, 16] = 0.0

    out = _reconcile_ast_to_team_budget(
        flow_values=flow_projected,
        valid_mask=valid_mask,
        team_index=team_index,
        active_mask=active_mask,
        flow_target_columns=cols,
        team_ast_budget=team_budget,
        assist_share_logits=ast_logits,
        share_alpha=0.75,
        share_temperature=1.0,
    )

    ast_idx = cols.index("ast")
    home_ast = out[0, 0:2, ast_idx]
    away_ast = out[0, 15:17, ast_idx]
    assert float(home_ast.sum()) == pytest.approx(12.0, abs=1e-6)
    assert float(away_ast.sum()) == pytest.approx(10.0, abs=1e-6)
    assert float(home_ast[0]) > float(home_ast[1])


def test_reconcile_points_to_team_budget_scales_team_scoring_makes() -> None:
    cols = list(FLOW_TARGET_COLUMNS_V1)
    flow = torch.zeros((1, 30, len(cols)), dtype=torch.float32)
    fg2m_idx = cols.index("fg2m")
    fg3m_idx = cols.index("fg3m")
    ftm_idx = cols.index("ftm")
    fga2_idx = cols.index("fga2")
    fga3_idx = cols.index("fga3")
    fta_idx = cols.index("fta")

    valid_mask = torch.zeros((1, 30), dtype=torch.bool)
    active_mask = torch.zeros((1, 30), dtype=torch.bool)
    valid_mask[0, 0] = True
    valid_mask[0, 1] = True
    valid_mask[0, 15] = True
    valid_mask[0, 16] = True
    active_mask.copy_(valid_mask)
    team_index = _team_index(1)

    flow[0, 0, fga2_idx] = 10.0
    flow[0, 0, fg2m_idx] = 4.0
    flow[0, 1, fta_idx] = 6.0
    flow[0, 1, ftm_idx] = 3.0
    flow[0, 15, fga3_idx] = 6.0
    flow[0, 15, fg3m_idx] = 2.0
    flow[0, 16, fta_idx] = 4.0
    flow[0, 16, ftm_idx] = 2.0

    # Home starts at 11 pts, away starts at 8 pts.
    out = _reconcile_points_to_team_budget(
        flow_values=flow,
        valid_mask=valid_mask,
        team_index=team_index,
        active_mask=active_mask,
        flow_target_columns=cols,
        team_points_budget=torch.tensor([[22.0, 4.0]], dtype=torch.float32),
        budget_alpha=1.0,
    )

    home_pts = float((2.0 * out[0, :15, fg2m_idx] + 3.0 * out[0, :15, fg3m_idx] + out[0, :15, ftm_idx]).sum())
    away_pts = float((2.0 * out[0, 15:, fg2m_idx] + 3.0 * out[0, 15:, fg3m_idx] + out[0, 15:, ftm_idx]).sum())
    assert home_pts == pytest.approx(22.0, rel=1e-6)
    assert away_pts == pytest.approx(4.0, rel=1e-6)
    assert float(out[0, 0, fg2m_idx]) <= float(out[0, 0, fga2_idx]) + 1e-6
    assert float(out[0, 15, fg3m_idx]) <= float(out[0, 15, fga3_idx]) + 1e-6
    assert float(out[0, 1, ftm_idx]) <= float(out[0, 1, fta_idx]) + 1e-6
    assert float(out[0, 16, ftm_idx]) <= float(out[0, 16, fta_idx]) + 1e-6


def test_resolve_team_points_budget_market_implied_uses_game_features() -> None:
    cfg = GameTransformerV2Config(
        feature_columns=["feat"],
        feature_mean=[0.0],
        feature_std=[1.0],
        game_feature_columns=["vegas_total", "vegas_spread", "estimated_possessions"],
        team_feature_columns=[],
        team_points_budget_parameterization="market_implied",
    )
    game_features = torch.tensor([[230.0, -6.0, 99.0]], dtype=torch.float32)
    budget = _resolve_team_points_budget(
        model_config=cfg,
        game_features=game_features,
        team_points_budget_out=None,
    )
    assert budget is not None
    assert budget.shape == (1, 2)
    assert float(budget[0, 0]) == pytest.approx(118.0, abs=1e-6)
    assert float(budget[0, 1]) == pytest.approx(112.0, abs=1e-6)


def test_resolve_team_points_budget_team_ppp_implied_uses_ppp_and_possessions() -> None:
    cfg = GameTransformerV2Config(
        feature_columns=["feat"],
        feature_mean=[0.0],
        feature_std=[1.0],
        game_feature_columns=["vegas_total", "vegas_spread", "estimated_possessions"],
        team_feature_columns=[],
        team_points_budget_parameterization="team_ppp_implied",
    )
    game_features = torch.tensor([[230.0, -6.0, 99.0]], dtype=torch.float32)
    budget = _resolve_team_points_budget(
        model_config=cfg,
        game_features=game_features,
        team_points_budget_out=None,
        team_ppp_out=torch.tensor([[1.15, 1.08]], dtype=torch.float32),
        possession_out=torch.tensor([100.0], dtype=torch.float32),
    )
    assert budget is not None
    assert budget.shape == (1, 2)
    assert float(budget[0, 0]) == pytest.approx(115.0, abs=1e-6)
    assert float(budget[0, 1]) == pytest.approx(108.0, abs=1e-5)


def test_resolve_team_opportunity_share_market_implied_uses_game_features() -> None:
    cfg = GameTransformerV2Config(
        feature_columns=["feat"],
        feature_mean=[0.0],
        feature_std=[1.0],
        game_feature_columns=["vegas_total", "vegas_spread", "estimated_possessions"],
        team_feature_columns=[],
        team_opportunity_budget_parameterization="market_implied_share",
    )
    game_features = torch.tensor([[230.0, -6.0, 99.0]], dtype=torch.float32)
    share = _resolve_team_opportunity_share(
        model_config=cfg,
        game_features=game_features,
    )
    assert share is not None
    assert share.shape == (1, 2)
    assert float(share[0, 0]) == pytest.approx(118.0 / 230.0, abs=1e-6)
    assert float(share[0, 1]) == pytest.approx(112.0 / 230.0, abs=1e-6)
    assert float(share.sum()) == pytest.approx(1.0, abs=1e-6)


def test_reconcile_opportunities_to_team_budget_scales_side_fga_fta_and_preserves_game_totals() -> None:
    cols = list(FLOW_TARGET_COLUMNS_V1)
    flow = torch.zeros((1, 30, len(cols)), dtype=torch.float32)
    fga2_idx = cols.index("fga2")
    fg2m_idx = cols.index("fg2m")
    fga3_idx = cols.index("fga3")
    fg3m_idx = cols.index("fg3m")
    fta_idx = cols.index("fta")
    ftm_idx = cols.index("ftm")

    valid_mask = torch.zeros((1, 30), dtype=torch.bool)
    active_mask = torch.zeros((1, 30), dtype=torch.bool)
    valid_mask[0, 0] = True
    valid_mask[0, 1] = True
    valid_mask[0, 15] = True
    valid_mask[0, 16] = True
    active_mask.copy_(valid_mask)
    team_index = _team_index(1)

    flow[0, 0, fga2_idx] = 8.0
    flow[0, 0, fg2m_idx] = 4.0
    flow[0, 1, fta_idx] = 4.0
    flow[0, 1, ftm_idx] = 3.0
    flow[0, 15, fga3_idx] = 10.0
    flow[0, 15, fg3m_idx] = 4.0
    flow[0, 16, fta_idx] = 6.0
    flow[0, 16, ftm_idx] = 5.0

    out = _reconcile_opportunities_to_team_budget(
        flow_values=flow,
        valid_mask=valid_mask,
        team_index=team_index,
        active_mask=active_mask,
        flow_target_columns=cols,
        team_opportunity_share=torch.tensor([[0.75, 0.25]], dtype=torch.float32),
        budget_alpha=1.0,
    )

    home_fga = float((out[0, :15, fga2_idx] + out[0, :15, fga3_idx]).sum())
    away_fga = float((out[0, 15:, fga2_idx] + out[0, 15:, fga3_idx]).sum())
    home_fta = float(out[0, :15, fta_idx].sum())
    away_fta = float(out[0, 15:, fta_idx].sum())
    game_fga = float((out[0, :, fga2_idx] + out[0, :, fga3_idx]).sum())
    game_fta = float(out[0, :, fta_idx].sum())

    assert home_fga == pytest.approx(13.5, rel=1e-6)
    assert away_fga == pytest.approx(4.5, rel=1e-6)
    assert game_fga == pytest.approx(18.0, rel=1e-6)
    assert home_fta == pytest.approx(7.5, rel=1e-6)
    assert away_fta == pytest.approx(2.5, rel=1e-6)
    assert game_fta == pytest.approx(10.0, rel=1e-6)
    assert float(out[0, 0, fg2m_idx]) <= float(out[0, 0, fga2_idx]) + 1e-6
    assert float(out[0, 15, fg3m_idx]) <= float(out[0, 15, fga3_idx]) + 1e-6
    assert float(out[0, 1, ftm_idx]) <= float(out[0, 1, fta_idx]) + 1e-6
    assert float(out[0, 16, ftm_idx]) <= float(out[0, 16, fta_idx]) + 1e-6


def test_reconcile_opportunities_to_team_budget_can_preserve_possessions_by_adjusting_tov() -> None:
    cols = list(FLOW_TARGET_COLUMNS_V1)
    flow = torch.zeros((1, 30, len(cols)), dtype=torch.float32)
    fga2_idx = cols.index("fga2")
    fg2m_idx = cols.index("fg2m")
    fga3_idx = cols.index("fga3")
    fg3m_idx = cols.index("fg3m")
    fta_idx = cols.index("fta")
    ftm_idx = cols.index("ftm")
    tov_idx = cols.index("tov")

    valid_mask = torch.zeros((1, 30), dtype=torch.bool)
    active_mask = torch.zeros((1, 30), dtype=torch.bool)
    valid_mask[0, 0] = True
    valid_mask[0, 1] = True
    valid_mask[0, 15] = True
    valid_mask[0, 16] = True
    active_mask.copy_(valid_mask)
    team_index = _team_index(1)

    flow[0, 0, fga2_idx] = 6.0
    flow[0, 0, fg2m_idx] = 3.0
    flow[0, 1, fta_idx] = 4.0
    flow[0, 1, ftm_idx] = 3.0
    flow[0, 1, tov_idx] = 2.24
    flow[0, 15, fga3_idx] = 10.0
    flow[0, 15, fg3m_idx] = 4.0

    def _team_poss(t: torch.Tensor, start: int, end: int) -> float:
        fga = float((t[0, start:end, fga2_idx] + t[0, start:end, fga3_idx]).sum())
        fta = float(t[0, start:end, fta_idx].sum())
        tov = float(t[0, start:end, tov_idx].sum())
        return fga + tov + float(FTA_POSS_COEFF) * fta

    poss_home_before = _team_poss(flow, 0, 15)
    poss_away_before = _team_poss(flow, 15, 30)

    out = _reconcile_opportunities_to_team_budget(
        flow_values=flow,
        valid_mask=valid_mask,
        team_index=team_index,
        active_mask=active_mask,
        flow_target_columns=cols,
        team_opportunity_share=torch.tensor([[0.75, 0.25]], dtype=torch.float32),
        budget_alpha=1.0,
        preserve_possessions=True,
    )

    poss_home_after = _team_poss(out, 0, 15)
    poss_away_after = _team_poss(out, 15, 30)
    home_fga_before = float((flow[0, :15, fga2_idx] + flow[0, :15, fga3_idx]).sum())
    away_fga_before = float((flow[0, 15:, fga2_idx] + flow[0, 15:, fga3_idx]).sum())
    home_fga_after = float((out[0, :15, fga2_idx] + out[0, :15, fga3_idx]).sum())
    away_fga_after = float((out[0, 15:, fga2_idx] + out[0, 15:, fga3_idx]).sum())

    assert poss_home_before == pytest.approx(10.0, rel=1e-6)
    assert poss_away_before == pytest.approx(10.0, rel=1e-6)
    assert poss_home_after == pytest.approx(poss_home_before, rel=1e-6)
    assert poss_away_after == pytest.approx(poss_away_before, rel=1e-6)
    assert home_fga_after / (home_fga_after + away_fga_after) > home_fga_before / (home_fga_before + away_fga_before)
    assert float(out[0, :15, tov_idx].sum()) == pytest.approx(0.0, abs=1e-6)
    assert float(out[0, 15:, tov_idx].sum()) > float(flow[0, 15:, tov_idx].sum())


def test_reconcile_rebounds_to_opportunity_budgets_caps_and_redistributes() -> None:
    cols = list(FLOW_TARGET_COLUMNS_V1)
    flow = torch.zeros((1, 30, len(cols)), dtype=torch.float32)
    oreb_idx = cols.index("oreb")
    dreb_idx = cols.index("dreb")
    fga2_idx = cols.index("fga2")
    fg2m_idx = cols.index("fg2m")
    fga3_idx = cols.index("fga3")
    fg3m_idx = cols.index("fg3m")

    valid_mask = torch.zeros((1, 30), dtype=torch.bool)
    active_mask = torch.zeros((1, 30), dtype=torch.bool)
    valid_mask[0, 0] = True
    valid_mask[0, 1] = True
    valid_mask[0, 15] = True
    valid_mask[0, 16] = True
    active_mask.copy_(valid_mask)
    team_index = _team_index(1)

    flow[0, 0, oreb_idx] = 1.0
    flow[0, 1, oreb_idx] = 7.0
    flow[0, 15, oreb_idx] = 5.0
    flow[0, 16, oreb_idx] = 1.0
    flow[0, 0, dreb_idx] = 6.0
    flow[0, 1, dreb_idx] = 1.0
    flow[0, 15, dreb_idx] = 1.0
    flow[0, 16, dreb_idx] = 6.0

    flow[0, 0, fga2_idx] = 5.0
    flow[0, 0, fg2m_idx] = 2.0
    flow[0, 1, fga3_idx] = 4.0
    flow[0, 1, fg3m_idx] = 1.0
    flow[0, 15, fga2_idx] = 8.0
    flow[0, 15, fg2m_idx] = 1.0
    flow[0, 16, fga3_idx] = 2.0
    flow[0, 16, fg3m_idx] = 1.0

    team_oreb_budget = torch.tensor([[10.0, 4.0]], dtype=torch.float32)
    team_dreb_budget = torch.tensor([[12.0, 9.0]], dtype=torch.float32)
    oreb_share_logits = torch.full((1, 30), -9.0, dtype=torch.float32)
    dreb_share_logits = torch.full((1, 30), -9.0, dtype=torch.float32)
    oreb_share_logits[0, 0] = 2.0
    oreb_share_logits[0, 1] = 0.0
    oreb_share_logits[0, 15] = 0.0
    oreb_share_logits[0, 16] = 1.0
    dreb_share_logits[0, 0] = 0.0
    dreb_share_logits[0, 1] = 1.0
    dreb_share_logits[0, 15] = 2.0
    dreb_share_logits[0, 16] = 0.0

    out = _reconcile_rebounds_to_opportunity_budgets(
        flow_values=flow,
        valid_mask=valid_mask,
        team_index=team_index,
        active_mask=active_mask,
        flow_target_columns=cols,
        team_oreb_budget=team_oreb_budget,
        team_dreb_budget=team_dreb_budget,
        team_oreb_budget_gate=None,
        team_dreb_budget_gate=None,
        oreb_share_logits=oreb_share_logits,
        dreb_share_logits=dreb_share_logits,
        share_alpha=1.0,
        share_temperature=1.0,
    )

    home_oreb = out[0, :15, oreb_idx]
    away_oreb = out[0, 15:, oreb_idx]
    home_dreb = out[0, :15, dreb_idx]
    away_dreb = out[0, 15:, dreb_idx]
    expected_home_oreb = torch.softmax(torch.tensor([2.0, 0.0]), dim=0) * 6.0
    expected_away_oreb = torch.softmax(torch.tensor([0.0, 1.0]), dim=0) * 4.0
    expected_home_dreb = torch.softmax(torch.tensor([0.0, 1.0]), dim=0) * 8.0
    expected_away_dreb = torch.softmax(torch.tensor([2.0, 0.0]), dim=0) * 6.0

    assert float(home_oreb.sum()) == pytest.approx(6.0, rel=1e-6)
    assert float(away_oreb.sum()) == pytest.approx(4.0, rel=1e-6)
    assert float(home_dreb.sum()) == pytest.approx(8.0, rel=1e-6)
    assert float(away_dreb.sum()) == pytest.approx(6.0, rel=1e-6)
    assert torch.allclose(home_oreb[:2], expected_home_oreb, atol=1e-6)
    assert torch.allclose(away_oreb[:2], expected_away_oreb, atol=1e-6)
    assert torch.allclose(home_dreb[:2], expected_home_dreb, atol=1e-6)
    assert torch.allclose(away_dreb[:2], expected_away_dreb, atol=1e-6)
    assert float(home_oreb[0]) > float(home_oreb[1])
    assert float(home_dreb[1]) > float(home_dreb[0])
    assert float(away_oreb[1]) > float(away_oreb[0])
    assert float(away_dreb[0]) > float(away_dreb[1])


def test_reconcile_rebounds_to_opportunity_budgets_dreb_only_leaves_oreb_unchanged() -> None:
    cols = list(FLOW_TARGET_COLUMNS_V1)
    flow = torch.zeros((1, 30, len(cols)), dtype=torch.float32)
    oreb_idx = cols.index("oreb")
    dreb_idx = cols.index("dreb")
    fga2_idx = cols.index("fga2")
    fg2m_idx = cols.index("fg2m")

    valid_mask = torch.zeros((1, 30), dtype=torch.bool)
    active_mask = torch.zeros((1, 30), dtype=torch.bool)
    valid_mask[0, 0] = True
    valid_mask[0, 1] = True
    valid_mask[0, 15] = True
    valid_mask[0, 16] = True
    active_mask.copy_(valid_mask)
    team_index = _team_index(1)

    flow[0, 0, oreb_idx] = 1.0
    flow[0, 1, oreb_idx] = 3.0
    flow[0, 15, oreb_idx] = 2.0
    flow[0, 16, oreb_idx] = 6.0
    flow[0, 0, dreb_idx] = 6.0
    flow[0, 1, dreb_idx] = 1.0
    flow[0, 15, dreb_idx] = 1.0
    flow[0, 16, dreb_idx] = 6.0
    flow[0, 0, fga2_idx] = 5.0
    flow[0, 0, fg2m_idx] = 2.0
    flow[0, 15, fga2_idx] = 8.0
    flow[0, 15, fg2m_idx] = 1.0

    out = _reconcile_rebounds_to_opportunity_budgets(
        flow_values=flow,
        valid_mask=valid_mask,
        team_index=team_index,
        active_mask=active_mask,
        flow_target_columns=cols,
        team_oreb_budget=torch.tensor([[1.0, 1.0]], dtype=torch.float32),
        team_dreb_budget=torch.tensor([[12.0, 9.0]], dtype=torch.float32),
        team_oreb_budget_gate=None,
        team_dreb_budget_gate=None,
        oreb_share_logits=torch.zeros((1, 30), dtype=torch.float32),
        dreb_share_logits=torch.tensor(
            [[0.0, 1.0] + [0.0] * 13 + [2.0, 0.0] + [0.0] * 13],
            dtype=torch.float32,
        ),
        share_alpha=1.0,
        share_temperature=1.0,
        reconcile_mode="dreb_only",
    )

    assert torch.allclose(out[..., oreb_idx], flow[..., oreb_idx], atol=1e-6)
    assert not torch.allclose(out[..., dreb_idx], flow[..., dreb_idx], atol=1e-6)


def test_reconcile_rebounds_to_opportunity_budgets_dreb_rate_uses_opp_missed_budget() -> None:
    cols = list(FLOW_TARGET_COLUMNS_V1)
    flow = torch.zeros((1, 30, len(cols)), dtype=torch.float32)
    dreb_idx = cols.index("dreb")
    fga2_idx = cols.index("fga2")
    fg2m_idx = cols.index("fg2m")

    valid_mask = torch.zeros((1, 30), dtype=torch.bool)
    active_mask = torch.zeros((1, 30), dtype=torch.bool)
    valid_mask[0, 0] = True
    valid_mask[0, 1] = True
    valid_mask[0, 15] = True
    valid_mask[0, 16] = True
    active_mask.copy_(valid_mask)
    team_index = _team_index(1)

    flow[0, 0, dreb_idx] = 4.0
    flow[0, 1, dreb_idx] = 4.0
    flow[0, 15, dreb_idx] = 5.0
    flow[0, 16, dreb_idx] = 3.0
    flow[0, 0, fga2_idx] = 8.0
    flow[0, 0, fg2m_idx] = 2.0
    flow[0, 15, fga2_idx] = 10.0
    flow[0, 15, fg2m_idx] = 1.0

    out = _reconcile_rebounds_to_opportunity_budgets(
        flow_values=flow,
        valid_mask=valid_mask,
        team_index=team_index,
        active_mask=active_mask,
        flow_target_columns=cols,
        team_oreb_budget=torch.tensor([[1.0, 1.0]], dtype=torch.float32),
        team_dreb_budget=torch.tensor([[0.5, 0.25]], dtype=torch.float32),
        team_oreb_budget_gate=None,
        team_dreb_budget_gate=None,
        oreb_share_logits=torch.zeros((1, 30), dtype=torch.float32),
        dreb_share_logits=torch.tensor(
            [[0.0, 1.0] + [0.0] * 13 + [2.0, 0.0] + [0.0] * 13],
            dtype=torch.float32,
        ),
        share_alpha=1.0,
        share_temperature=1.0,
        reconcile_mode="dreb_only",
        budget_parameterization="dreb_rate",
    )

    home_dreb = out[0, :15, dreb_idx]
    away_dreb = out[0, 15:, dreb_idx]
    assert float(home_dreb.sum()) == pytest.approx(4.5, rel=1e-6)
    assert float(away_dreb.sum()) == pytest.approx(1.5, rel=1e-6)


def test_reconcile_rebounds_to_opportunity_budgets_dreb_budget_blend_alpha_mixes_flow_and_rate() -> None:
    cols = list(FLOW_TARGET_COLUMNS_V1)
    flow = torch.zeros((1, 30, len(cols)), dtype=torch.float32)
    dreb_idx = cols.index("dreb")
    fga2_idx = cols.index("fga2")
    fg2m_idx = cols.index("fg2m")

    valid_mask = torch.zeros((1, 30), dtype=torch.bool)
    active_mask = torch.zeros((1, 30), dtype=torch.bool)
    valid_mask[0, 0] = True
    valid_mask[0, 1] = True
    valid_mask[0, 15] = True
    valid_mask[0, 16] = True
    active_mask.copy_(valid_mask)
    team_index = _team_index(1)

    flow[0, 0, dreb_idx] = 4.0
    flow[0, 1, dreb_idx] = 4.0
    flow[0, 15, dreb_idx] = 5.0
    flow[0, 16, dreb_idx] = 3.0
    flow[0, 0, fga2_idx] = 8.0
    flow[0, 0, fg2m_idx] = 2.0
    flow[0, 15, fga2_idx] = 10.0
    flow[0, 15, fg2m_idx] = 1.0

    out = _reconcile_rebounds_to_opportunity_budgets(
        flow_values=flow,
        valid_mask=valid_mask,
        team_index=team_index,
        active_mask=active_mask,
        flow_target_columns=cols,
        team_oreb_budget=torch.tensor([[1.0, 1.0]], dtype=torch.float32),
        team_dreb_budget=torch.tensor([[0.5, 0.25]], dtype=torch.float32),
        team_oreb_budget_gate=None,
        team_dreb_budget_gate=None,
        oreb_share_logits=torch.zeros((1, 30), dtype=torch.float32),
        dreb_share_logits=torch.tensor(
            [[0.0, 1.0] + [0.0] * 13 + [2.0, 0.0] + [0.0] * 13],
            dtype=torch.float32,
        ),
        share_alpha=1.0,
        share_temperature=1.0,
        reconcile_mode="dreb_only",
        budget_parameterization="dreb_rate",
        dreb_budget_blend_alpha=0.5,
    )

    home_dreb = out[0, :15, dreb_idx]
    away_dreb = out[0, 15:, dreb_idx]
    assert float(home_dreb.sum()) == pytest.approx(6.25, rel=1e-6)
    assert float(away_dreb.sum()) == pytest.approx(3.75, rel=1e-6)


def test_reconcile_rebounds_to_opportunity_budgets_gate_overrides_scalar_budget_blend() -> None:
    cols = list(FLOW_TARGET_COLUMNS_V1)
    flow = torch.zeros((1, 30, len(cols)), dtype=torch.float32)
    dreb_idx = cols.index("dreb")
    fga2_idx = cols.index("fga2")
    fg2m_idx = cols.index("fg2m")

    valid_mask = torch.zeros((1, 30), dtype=torch.bool)
    active_mask = torch.zeros((1, 30), dtype=torch.bool)
    valid_mask[0, 0] = True
    valid_mask[0, 1] = True
    valid_mask[0, 15] = True
    valid_mask[0, 16] = True
    active_mask.copy_(valid_mask)
    team_index = _team_index(1)

    flow[0, 0, dreb_idx] = 4.0
    flow[0, 1, dreb_idx] = 4.0
    flow[0, 15, dreb_idx] = 5.0
    flow[0, 16, dreb_idx] = 3.0
    flow[0, 0, fga2_idx] = 8.0
    flow[0, 0, fg2m_idx] = 2.0
    flow[0, 15, fga2_idx] = 10.0
    flow[0, 15, fg2m_idx] = 1.0

    out = _reconcile_rebounds_to_opportunity_budgets(
        flow_values=flow,
        valid_mask=valid_mask,
        team_index=team_index,
        active_mask=active_mask,
        flow_target_columns=cols,
        team_oreb_budget=torch.tensor([[1.0, 1.0]], dtype=torch.float32),
        team_dreb_budget=torch.tensor([[0.5, 0.25]], dtype=torch.float32),
        team_oreb_budget_gate=None,
        team_dreb_budget_gate=torch.tensor([[0.5, 0.5]], dtype=torch.float32),
        oreb_share_logits=torch.zeros((1, 30), dtype=torch.float32),
        dreb_share_logits=torch.tensor(
            [[0.0, 1.0] + [0.0] * 13 + [2.0, 0.0] + [0.0] * 13],
            dtype=torch.float32,
        ),
        share_alpha=1.0,
        share_temperature=1.0,
        reconcile_mode="dreb_only",
        budget_parameterization="dreb_rate",
        dreb_budget_blend_alpha=0.0,
    )

    home_dreb = out[0, :15, dreb_idx]
    away_dreb = out[0, 15:, dreb_idx]
    assert float(home_dreb.sum()) == pytest.approx(6.25, rel=1e-6)
    assert float(away_dreb.sum()) == pytest.approx(3.75, rel=1e-6)


def test_reconcile_rebounds_to_opportunity_budgets_dreb_rate_residual_offsets_flow_rate() -> None:
    cols = list(FLOW_TARGET_COLUMNS_V1)
    flow = torch.zeros((1, 30, len(cols)), dtype=torch.float32)
    dreb_idx = cols.index("dreb")
    fga2_idx = cols.index("fga2")
    fg2m_idx = cols.index("fg2m")

    valid_mask = torch.zeros((1, 30), dtype=torch.bool)
    active_mask = torch.zeros((1, 30), dtype=torch.bool)
    valid_mask[0, 0] = True
    valid_mask[0, 1] = True
    valid_mask[0, 15] = True
    valid_mask[0, 16] = True
    active_mask.copy_(valid_mask)
    team_index = _team_index(1)

    flow[0, 0, dreb_idx] = 2.0
    flow[0, 1, dreb_idx] = 2.0
    flow[0, 15, dreb_idx] = 1.0
    flow[0, 16, dreb_idx] = 1.0
    flow[0, 0, fga2_idx] = 4.0
    flow[0, 0, fg2m_idx] = 0.0
    flow[0, 15, fga2_idx] = 8.0
    flow[0, 15, fg2m_idx] = 0.0

    out = _reconcile_rebounds_to_opportunity_budgets(
        flow_values=flow,
        valid_mask=valid_mask,
        team_index=team_index,
        active_mask=active_mask,
        flow_target_columns=cols,
        team_oreb_budget=torch.tensor([[1.0, 1.0]], dtype=torch.float32),
        team_dreb_budget=torch.tensor([[0.1, -0.2]], dtype=torch.float32),
        team_oreb_budget_gate=None,
        team_dreb_budget_gate=None,
        oreb_share_logits=torch.zeros((1, 30), dtype=torch.float32),
        dreb_share_logits=torch.tensor(
            [[0.0, 0.0] + [0.0] * 13 + [0.0, 0.0] + [0.0] * 13],
            dtype=torch.float32,
        ),
        share_alpha=0.0,
        share_temperature=1.0,
        reconcile_mode="dreb_only",
        budget_parameterization="dreb_rate_residual",
        dreb_budget_blend_alpha=1.0,
    )

    home_dreb = out[0, :15, dreb_idx]
    away_dreb = out[0, 15:, dreb_idx]
    assert float(home_dreb.sum()) == pytest.approx(4.8, rel=1e-6)
    assert float(away_dreb.sum()) == pytest.approx(1.2, rel=1e-6)


def test_reconcile_rebounds_to_opportunity_budgets_dreb_deterministic_applies_discount_to_opp_misses_minus_opp_oreb() -> None:
    cols = list(FLOW_TARGET_COLUMNS_V1)
    flow = torch.zeros((1, 30, len(cols)), dtype=torch.float32)
    oreb_idx = cols.index("oreb")
    dreb_idx = cols.index("dreb")
    fga2_idx = cols.index("fga2")
    fg2m_idx = cols.index("fg2m")

    valid_mask = torch.zeros((1, 30), dtype=torch.bool)
    active_mask = torch.zeros((1, 30), dtype=torch.bool)
    valid_mask[0, 0] = True
    valid_mask[0, 1] = True
    valid_mask[0, 15] = True
    valid_mask[0, 16] = True
    active_mask.copy_(valid_mask)
    team_index = _team_index(1)

    flow[0, 0, oreb_idx] = 1.0
    flow[0, 1, oreb_idx] = 1.0
    flow[0, 15, oreb_idx] = 3.0
    flow[0, 16, oreb_idx] = 0.0
    flow[0, 0, dreb_idx] = 4.0
    flow[0, 1, dreb_idx] = 4.0
    flow[0, 15, dreb_idx] = 5.0
    flow[0, 16, dreb_idx] = 3.0
    flow[0, 0, fga2_idx] = 8.0
    flow[0, 0, fg2m_idx] = 2.0
    flow[0, 15, fga2_idx] = 10.0
    flow[0, 15, fg2m_idx] = 1.0

    out = _reconcile_rebounds_to_opportunity_budgets(
        flow_values=flow,
        valid_mask=valid_mask,
        team_index=team_index,
        active_mask=active_mask,
        flow_target_columns=cols,
        team_oreb_budget=None,
        team_dreb_budget=None,
        team_oreb_budget_gate=None,
        team_dreb_budget_gate=None,
        oreb_share_logits=None,
        dreb_share_logits=torch.tensor(
            [[0.0, 1.0] + [0.0] * 13 + [2.0, 0.0] + [0.0] * 13],
            dtype=torch.float32,
        ),
        share_alpha=1.0,
        share_temperature=1.0,
        reconcile_mode="dreb_only",
        budget_parameterization="dreb_deterministic",
        dreb_deterministic_discount=0.9,
    )

    home_dreb = out[0, :15, dreb_idx]
    away_dreb = out[0, 15:, dreb_idx]
    assert float(home_dreb.sum()) == pytest.approx(5.4, rel=1e-6)
    assert float(away_dreb.sum()) == pytest.approx(3.6, rel=1e-6)


def test_reconcile_rebounds_to_opportunity_budgets_oreb_flow_budget_redistributes_without_changing_team_total() -> None:
    cols = list(FLOW_TARGET_COLUMNS_V1)
    flow = torch.zeros((1, 30, len(cols)), dtype=torch.float32)
    oreb_idx = cols.index("oreb")
    fga2_idx = cols.index("fga2")
    fg2m_idx = cols.index("fg2m")

    valid_mask = torch.zeros((1, 30), dtype=torch.bool)
    active_mask = torch.zeros((1, 30), dtype=torch.bool)
    valid_mask[0, 0] = True
    valid_mask[0, 1] = True
    valid_mask[0, 15] = True
    valid_mask[0, 16] = True
    active_mask.copy_(valid_mask)
    team_index = _team_index(1)

    flow[0, 0, oreb_idx] = 1.0
    flow[0, 1, oreb_idx] = 3.0
    flow[0, 15, oreb_idx] = 2.0
    flow[0, 16, oreb_idx] = 2.0
    flow[0, 0, fga2_idx] = 8.0
    flow[0, 0, fg2m_idx] = 2.0
    flow[0, 15, fga2_idx] = 10.0
    flow[0, 15, fg2m_idx] = 1.0

    out = _reconcile_rebounds_to_opportunity_budgets(
        flow_values=flow,
        valid_mask=valid_mask,
        team_index=team_index,
        active_mask=active_mask,
        flow_target_columns=cols,
        team_oreb_budget=None,
        team_dreb_budget=None,
        team_oreb_budget_gate=None,
        team_dreb_budget_gate=None,
        oreb_share_logits=torch.tensor(
            [[2.0, 0.0] + [0.0] * 13 + [0.0, 2.0] + [0.0] * 13],
            dtype=torch.float32,
        ),
        dreb_share_logits=None,
        share_alpha=1.0,
        share_temperature=1.0,
        reconcile_mode="oreb_only",
        budget_parameterization="dreb_deterministic",
        oreb_reconcile_use_flow_budget=True,
    )

    home_oreb = out[0, :15, oreb_idx]
    away_oreb = out[0, 15:, oreb_idx]
    assert float(home_oreb.sum()) == pytest.approx(4.0, rel=1e-6)
    assert float(away_oreb.sum()) == pytest.approx(4.0, rel=1e-6)
    assert float(home_oreb[0]) > float(home_oreb[1])
    assert float(away_oreb[1]) > float(away_oreb[0])


def test_reconcile_ast_to_team_budget_accepts_player_level_alpha_tensor() -> None:
    cols = ["pts", "ast", "reb"]
    flow_projected = torch.zeros((1, 30, len(cols)), dtype=torch.float32)
    flow_projected[0, 0, cols.index("ast")] = 8.0
    flow_projected[0, 1, cols.index("ast")] = 2.0
    valid_mask = torch.zeros((1, 30), dtype=torch.bool)
    valid_mask[0, 0:2] = True
    active_mask = valid_mask.clone()
    team_index = torch.cat([torch.zeros((1, 15), dtype=torch.long), torch.ones((1, 15), dtype=torch.long)], dim=1)
    team_budget = torch.tensor([[12.0, 0.0]], dtype=torch.float32)
    ast_logits = torch.full((1, 30), -10.0, dtype=torch.float32)
    ast_logits[0, 0] = 0.0
    ast_logits[0, 1] = 4.0
    alpha = torch.zeros((1, 30), dtype=torch.float32)
    alpha[0, 0] = 0.1
    alpha[0, 1] = 0.9

    out = _reconcile_ast_to_team_budget(
        flow_values=flow_projected,
        valid_mask=valid_mask,
        team_index=team_index,
        active_mask=active_mask,
        flow_target_columns=cols,
        team_ast_budget=team_budget,
        assist_share_logits=ast_logits,
        share_alpha=alpha,
        share_temperature=1.0,
    )
    ast_idx = cols.index("ast")
    home_ast = out[0, 0:2, ast_idx]
    assert float(home_ast.sum()) == pytest.approx(12.0, abs=1e-6)
    assert float(home_ast[1]) > float(home_ast[0])


def test_build_creator_reconcile_alpha_team_relative_prioritizes_top_creator() -> None:
    player_features = torch.zeros((1, 2, 15, 4), dtype=torch.float32)
    valid_mask = torch.zeros((1, 30), dtype=torch.bool)
    valid_mask[0, 0:3] = True
    valid_mask[0, 15:18] = True
    team_index = torch.cat([torch.zeros((1, 15), dtype=torch.long), torch.ones((1, 15), dtype=torch.long)], dim=1)

    # feature columns: an_ast_line, an_implied_minutes, prior_play_prob, started_proxy_rate_prior_20
    player_features[0, 0, 0, 0] = 8.0
    player_features[0, 0, 0, 1] = 32.0
    player_features[0, 0, 0, 2] = 0.95
    player_features[0, 0, 1, 0] = 6.0
    player_features[0, 0, 1, 1] = 30.0
    player_features[0, 0, 1, 2] = 0.95
    player_features[0, 0, 2, 0] = 3.0
    player_features[0, 0, 2, 1] = 24.0
    player_features[0, 0, 2, 2] = 0.95

    player_features[0, 1, 0, 0] = 9.0
    player_features[0, 1, 0, 1] = 34.0
    player_features[0, 1, 0, 2] = 0.95
    player_features[0, 1, 1, 0] = 7.0
    player_features[0, 1, 1, 1] = 31.0
    player_features[0, 1, 1, 2] = 0.95

    config = GameTransformerV2Config(
        feature_columns=[
            "an_ast_line",
            "an_implied_minutes",
            "prior_play_prob",
            "started_proxy_rate_prior_20",
        ],
        game_feature_columns=[],
        team_feature_columns=[],
        feature_mean=[0.0, 0.0, 0.0, 0.0],
        feature_std=[1.0, 1.0, 1.0, 1.0],
    )
    alpha = _build_creator_reconcile_alpha(
        player_features,
        valid_mask=valid_mask,
        team_index=team_index,
        config=config,
        runtime_config=AstFactorizationRuntimeConfig(
            creator_reconcile_alpha_enabled=True,
            creator_reconcile_alpha_max=0.6,
            creator_reconcile_ast_line_center=6.0,
            creator_reconcile_ast_line_scale=1.25,
            creator_reconcile_minutes_center=28.0,
            creator_reconcile_minutes_scale=5.0,
            creator_reconcile_prior_play_prob_floor=0.8,
            creator_reconcile_team_relative=True,
            creator_reconcile_team_power=1.5,
        ),
    )
    assert alpha is not None
    assert float(alpha[0, 0]) > float(alpha[0, 1]) > float(alpha[0, 2])
    assert float(alpha[0, 15]) > float(alpha[0, 16])


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


def test_reweight_top_usage_alloc_weights_boosts_top_players_and_preserves_simplex() -> None:
    weights = torch.tensor([[0.60, 0.25, 0.10, 0.05]], dtype=torch.float32)
    eligible = torch.tensor([[1, 1, 1, 1]], dtype=torch.bool)

    out = _reweight_top_usage_alloc_weights(
        weights,
        eligible_mask=eligible,
        top1_scale=1.10,
        top2_scale=1.05,
    )

    assert torch.allclose(out.sum(dim=1), torch.ones((1,), dtype=torch.float32), atol=1e-6)
    assert out[0, 0] > weights[0, 0]
    assert (out[0, 1] / out[0, 2]) > (weights[0, 1] / weights[0, 2])
    assert out[0, 2] < weights[0, 2]
    assert out[0, 3] < weights[0, 3]


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
    player_valid_mask[:, 0, :6] = True
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


def test_sample_worlds_for_batch_oracle_rotation_state_overrides_minutes_context() -> None:
    cols = list(FLOW_TARGET_COLUMNS_V1)

    class _FlowHead:
        def __init__(self) -> None:
            self.last_minutes_context: torch.Tensor | None = None

        def sample(self, z: torch.Tensor, **kwargs: object) -> torch.Tensor:
            minutes_context = kwargs.get("minutes_context")
            assert isinstance(minutes_context, torch.Tensor)
            self.last_minutes_context = minutes_context.detach().cpu()
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
        def __init__(self, flow_head: _FlowHead) -> None:
            self.flow_head = flow_head
            self.flow_target_columns = cols
            self.enable_possession_backbone = False

        def __call__(self, pf: torch.Tensor, pvm: torch.Tensor, **_: object) -> _Out:
            bsz = int(pf.shape[0])
            valid_flat = pvm.reshape(bsz, -1).to(dtype=torch.bool)
            active_flat = torch.zeros_like(valid_flat, dtype=torch.bool)
            minutes_flat = torch.zeros((bsz, 30), dtype=torch.float32)
            minutes_flat[:, 1:6] = 48.0
            minutes_flat[:, 15:20] = 48.0
            active_flat[:, 1:6] = True
            active_flat[:, 15:20] = True
            return _Out(valid_flat=valid_flat, active_flat=active_flat, minutes_flat=minutes_flat, team_idx=_team_index(bsz))

    player_ids = torch.arange(1001, 1031, dtype=torch.long).reshape(1, 2, 15)
    player_valid_mask = torch.zeros((1, 2, 15), dtype=torch.bool)
    player_valid_mask[:, 0, :6] = True
    player_valid_mask[:, 1, :5] = True
    y_minutes = torch.zeros((1, 2, 15), dtype=torch.float32)
    y_minutes[:, 0, 0:5] = 48.0
    y_minutes[:, 1, 0:5] = 48.0

    batch: dict[str, torch.Tensor | list[str]] = {
        "player_features": torch.zeros((1, 2, 15, 2), dtype=torch.float32),
        "player_valid_mask": player_valid_mask,
        "y_minutes": y_minutes,
        "game_features": torch.zeros((1, 0), dtype=torch.float32),
        "team_features": torch.zeros((1, 2, 0), dtype=torch.float32),
        "player_ids": player_ids,
        "team_ids": torch.tensor([[10, 20]], dtype=torch.long),
        "game_id_norm": ["1001"],
        "game_date": ["2026-01-18"],
    }

    flow_head = _FlowHead()
    worlds_df, checks = sample_worlds_for_batch(
        _Model(flow_head),
        batch,
        device=torch.device("cpu"),
        num_worlds=2,
        chunk_size=2,
        active_temperature=1.0,
        strict_contracts=True,
        oracle_rotation_state=True,
    )
    assert checks["total_violations"] == 0
    assert flow_head.last_minutes_context is not None
    assert float(flow_head.last_minutes_context[0, 0]) == 48.0
    assert float(flow_head.last_minutes_context[0, 5]) == 0.0

    promoted_rows = worlds_df.loc[worlds_df["player_id"] == 1001]
    displaced_rows = worlds_df.loc[worlds_df["player_id"] == 1006]
    assert len(promoted_rows) == 2
    assert int(promoted_rows["active"].min()) == 1
    assert float(promoted_rows["minutes"].min()) >= 47.99
    assert int(displaced_rows["active"].max()) == 0
    assert float(displaced_rows["minutes"].max()) <= 1e-6


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


def test_sample_worlds_for_batch_minutes_uncertainty_adds_same_signature_variance() -> None:
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
            self.minutes = type("Minutes", (), {"minutes": minutes_flat, "sigma": None})()
            self.flow = None
            self.backbone = None
            self.usage_share = None
            self.efficiency = None

    class _Model:
        def __init__(self) -> None:
            self.flow_head = _FlowHead()
            self.flow_target_columns = cols
            self.enable_possession_backbone = False
            self.gtv2_config = GameTransformerV2Config(
                feature_columns=["minutes_from_stints_std_prior_20", "dummy"],
                feature_mean=[0.0, 0.0],
                feature_std=[1.0, 1.0],
                game_feature_columns=[],
                team_feature_columns=[],
            )

        def __call__(self, pf: torch.Tensor, pvm: torch.Tensor, **_: object) -> _Out:
            bsz = int(pf.shape[0])
            valid_flat = pvm.reshape(bsz, -1).to(dtype=torch.bool)
            active_flat = torch.zeros_like(valid_flat, dtype=torch.bool)
            active_flat[:, :8] = True
            active_flat[:, 15:23] = True
            minutes_flat = torch.zeros((bsz, 30), dtype=torch.float32)
            minutes_flat[:, :5] = 36.0
            minutes_flat[:, 5:8] = 20.0
            minutes_flat[:, 15:20] = 36.0
            minutes_flat[:, 20:23] = 20.0
            return _Out(valid_flat=valid_flat, active_flat=active_flat, minutes_flat=minutes_flat, team_idx=_team_index(bsz))

    player_ids = torch.arange(1001, 1031, dtype=torch.long).reshape(1, 2, 15)
    player_valid_mask = torch.zeros((1, 2, 15), dtype=torch.bool)
    player_valid_mask[:, 0, :8] = True
    player_valid_mask[:, 1, :8] = True
    player_features = torch.zeros((1, 2, 15, 2), dtype=torch.float32)
    player_features[:, :, :, 0] = 2.5

    batch: dict[str, torch.Tensor | list[str]] = {
        "player_features": player_features,
        "player_valid_mask": player_valid_mask,
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
        num_worlds=8,
        chunk_size=8,
        active_temperature=1.0,
        strict_contracts=True,
        minutes_uncertainty_config=MinutesUncertaintyConfig(enabled=True, gaussian_scale=1.0),
    )
    assert checks["total_violations"] == 0
    player_rows = worlds_df.loc[worlds_df["player_id"] == 1001].sort_values("world_idx")
    assert len(player_rows) == 8
    assert int(player_rows["active"].nunique()) == 1
    assert int(player_rows["active"].iloc[0]) == 1
    assert float(player_rows["minutes"].std(ddof=0)) > 0.0
    protected_rows = worlds_df.loc[worlds_df["player_id"] == 1002].sort_values("world_idx")
    assert len(protected_rows) == 8
    assert float(protected_rows["minutes"].std(ddof=0)) <= 1e-5


def test_sample_worlds_for_batch_hard_masks_out_players() -> None:
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
            self.minutes = type("Minutes", (), {"minutes": minutes_flat, "sigma": None})()
            self.flow = None
            self.backbone = None
            self.usage_share = None
            self.efficiency = None

    class _Model:
        def __init__(self) -> None:
            self.flow_head = _FlowHead()
            self.flow_target_columns = cols
            self.enable_possession_backbone = False
            self.gtv2_config = GameTransformerV2Config(
                feature_columns=["is_out", "dummy"],
                feature_mean=[0.0, 0.0],
                feature_std=[1.0, 1.0],
                game_feature_columns=[],
                team_feature_columns=[],
            )

        def __call__(self, pf: torch.Tensor, pvm: torch.Tensor, **_: object) -> _Out:
            bsz = int(pf.shape[0])
            valid_flat = pvm.reshape(bsz, -1).to(dtype=torch.bool)
            active_flat = torch.zeros_like(valid_flat, dtype=torch.bool)
            active_flat[:, 1:6] = True
            active_flat[:, 15:20] = True
            minutes_flat = torch.zeros((bsz, 30), dtype=torch.float32)
            minutes_flat[:, 1:6] = 48.0
            minutes_flat[:, 15:20] = 48.0
            return _Out(
                valid_flat=valid_flat,
                active_flat=active_flat,
                minutes_flat=minutes_flat,
                team_idx=_team_index(bsz),
            )

    player_ids = torch.arange(1001, 1031, dtype=torch.long).reshape(1, 2, 15)
    player_valid_mask = torch.zeros((1, 2, 15), dtype=torch.bool)
    player_valid_mask[:, 0, :6] = True
    player_valid_mask[:, 1, :5] = True
    player_features = torch.zeros((1, 2, 15, 2), dtype=torch.float32)
    player_features[:, 0, 0, 0] = 1.0

    batch: dict[str, torch.Tensor | list[str]] = {
        "player_features": player_features,
        "player_valid_mask": player_valid_mask,
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
        num_worlds=4,
        chunk_size=4,
        active_temperature=1.0,
        strict_contracts=True,
    )
    assert checks["total_violations"] == 0
    out_rows = worlds_df.loc[worlds_df["player_id"] == 1001]
    assert len(out_rows) == 4
    assert int(out_rows["active"].max()) == 0
    assert float(out_rows["minutes"].max()) <= 1e-6
    assert float(out_rows["dk_fpts"].max()) <= 1e-6
