"""Game-transformer v2 foundation: game collation + backbone + joint heads."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset

from projections.rotation.assist_heads import (
    AstBlendGateHead,
    AstBlendGateHeadOutputs,
    AssistShareHead,
    AssistShareHeadOutputs,
    TeamAstBudgetHead,
    TeamAstBudgetHeadOutputs,
)
from projections.rotation.rebound_heads import (
    ReboundBudgetBlendGateHead,
    ReboundBudgetBlendGateHeadOutputs,
    ReboundShareHead,
    ReboundShareHeadOutputs,
    TeamReboundBudgetHead,
    TeamReboundBudgetHeadOutputs,
)
from projections.rotation.team_budget_heads import (
    TeamAdvantageHead,
    TeamAdvantageHeadOutputs,
    TeamPPPHead,
    TeamPPPHeadOutputs,
    TeamPointsBudgetHead,
    TeamPointsBudgetHeadOutputs,
)
from projections.rotation.joint_active_set import JointActiveSetHead, JointActiveSetOutputs
from projections.rotation.efficiency_head import EfficiencyHead, EfficiencyHeadOutputs
from projections.rotation.joint_game_flow import JointGameFlow, JointGameFlowOutputs
from projections.rotation.joint_minutes import JointMinutesHead, JointMinutesOutputs
from projections.rotation.possession_backbone import (
    PossessionHead,
    PossessionHeadOutputs,
    TeamEventBackbone,
    TeamEventBackboneOutputs,
    ThreePAShareHead,
)
from projections.rotation.usage_share_head import UsageShareHead, UsageShareHeadOutputs
from projections.rotation.set_model import zfill_game_id_series

MAX_PLAYERS_PER_TEAM = 15
TOTAL_PLAYERS_PER_GAME = 2 * MAX_PLAYERS_PER_TEAM
# With max_minutes_per_player=48, at least 5 players are required to make 240 feasible.
MIN_FEASIBLE_PLAYERS_PER_TEAM = 5
PROTECTED_PRIOR_PLAY_PROB_FLOOR = 0.938507
PROTECTED_PRIOR_MINUTES_FLOOR = 29.520922
OVERFLOW_RISK_WEIGHT_CONSECUTIVE_ACTIVE_DNP = 0.579943
OVERFLOW_RISK_WEIGHT_ACTIVE_BUT_DNP_RATE_LAST10 = 6.053079
OVERFLOW_RISK_WEIGHT_INACTIVE_STREAK_LEN = 0.117685
OVERFLOW_KEEP_WEIGHT_PRIOR_PLAY_PROB = 2.202986
OVERFLOW_KEEP_WEIGHT_PRIOR_MINUTES = 0.051353
FLOW_TARGET_COLUMNS_V1 = [
    "fga2",
    "fg2m",
    "fga3",
    "fg3m",
    "fta",
    "ftm",
    "oreb",
    "dreb",
    "ast",
    "stl",
    "blk",
    "tov",
]
FLOW_TARGET_COLUMNS_V2 = [
    "fga2",
    "fga3",
    "fta",
    "oreb",
    "dreb",
    "ast",
    "stl",
    "blk",
    "tov",
]
FLOW_TARGET_COLUMNS_WITH_PF = [*FLOW_TARGET_COLUMNS_V1, "pf"]
FLOW_TARGET_COLUMNS_V2_WITH_PF = [*FLOW_TARGET_COLUMNS_V2, "pf"]
FLOW_TARGET_SCHEMA_V1 = "v1"
FLOW_TARGET_SCHEMA_V2 = "v2"
FLOW_TARGET_SCHEMA_DEFAULT = FLOW_TARGET_SCHEMA_V1
FLOW_TARGET_SCHEMAS = (FLOW_TARGET_SCHEMA_V1, FLOW_TARGET_SCHEMA_V2)


def normalize_flow_target_schema(schema: str) -> str:
    value = str(schema).strip().lower()
    if value not in FLOW_TARGET_SCHEMAS:
        raise ValueError(f"Unsupported flow_target_schema={schema!r}. Expected one of {FLOW_TARGET_SCHEMAS}")
    return value


def flow_contract_columns(*, include_pf: bool) -> list[str]:
    return list(FLOW_TARGET_COLUMNS_WITH_PF if include_pf else FLOW_TARGET_COLUMNS_V1)


def flow_target_columns(*, include_pf: bool, schema: str = FLOW_TARGET_SCHEMA_DEFAULT) -> list[str]:
    schema_norm = normalize_flow_target_schema(schema)
    if schema_norm == FLOW_TARGET_SCHEMA_V1:
        return list(FLOW_TARGET_COLUMNS_WITH_PF if include_pf else FLOW_TARGET_COLUMNS_V1)
    return list(FLOW_TARGET_COLUMNS_V2_WITH_PF if include_pf else FLOW_TARGET_COLUMNS_V2)


def select_flow_columns(
    flow_values: torch.Tensor,
    *,
    source_columns: list[str],
    target_columns: list[str],
    fill_value: float | bool = 0.0,
) -> torch.Tensor:
    if flow_values.ndim < 1:
        raise ValueError("flow_values must have at least one dimension")
    if int(flow_values.shape[-1]) != int(len(source_columns)):
        raise ValueError(
            f"flow_values last dim ({flow_values.shape[-1]}) must match source_columns ({len(source_columns)})"
        )
    if not target_columns:
        return flow_values[..., :0]

    source_idx = {name: idx for idx, name in enumerate(source_columns)}
    if int(flow_values.shape[-1]) > 0:
        base = flow_values[..., :1]
    else:
        base = torch.zeros(
            (*flow_values.shape[:-1], 1),
            dtype=flow_values.dtype,
            device=flow_values.device,
        )
    cols: list[torch.Tensor] = []
    for name in target_columns:
        idx = source_idx.get(name)
        if idx is None:
            cols.append(torch.full_like(base, fill_value))
        else:
            cols.append(flow_values[..., idx:idx + 1])
    return torch.cat(cols, dim=-1)


def _resolve_rate_tensor(
    rate: torch.Tensor | float | None,
    *,
    reference: torch.Tensor,
    default: float,
) -> torch.Tensor:
    if rate is None:
        out = torch.full_like(reference, float(default))
    elif isinstance(rate, torch.Tensor):
        if rate.shape != reference.shape:
            raise ValueError(f"rate tensor shape {tuple(rate.shape)} must match reference shape {tuple(reference.shape)}")
        out = rate.to(device=reference.device, dtype=reference.dtype)
    else:
        out = torch.full_like(reference, float(rate))
    return torch.clamp(out, min=0.0, max=1.0)


def reconstruct_flow_to_contract(
    flow_values: torch.Tensor,
    *,
    flow_target_columns: list[str],
    contract_columns: list[str] | None = None,
    fg2_rate: torch.Tensor | float | None = None,
    fg3_rate: torch.Tensor | float | None = None,
    ft_rate: torch.Tensor | float | None = None,
    fg2_rate_default: float = 0.54,
    fg3_rate_default: float = 0.36,
    ft_rate_default: float = 0.77,
) -> torch.Tensor:
    if flow_values.ndim < 1:
        raise ValueError("flow_values must have at least one dimension")
    if int(flow_values.shape[-1]) != int(len(flow_target_columns)):
        raise ValueError(
            f"flow_values last dim ({flow_values.shape[-1]}) must match flow_target_columns ({len(flow_target_columns)})"
        )

    cleaned = torch.clamp(flow_values, min=0.0)
    idx = {name: i for i, name in enumerate(flow_target_columns)}
    zero = (
        torch.zeros(cleaned.shape[:-1], dtype=cleaned.dtype, device=cleaned.device)
        if int(cleaned.shape[-1]) == 0
        else torch.zeros_like(cleaned[..., 0])
    )

    def _col(name: str) -> torch.Tensor:
        col_idx = idx.get(name)
        if col_idx is None:
            return zero
        return cleaned[..., col_idx]

    fga2 = _col("fga2")
    fga3 = _col("fga3")
    fta = _col("fta")
    oreb = _col("oreb")
    dreb = _col("dreb")
    ast = _col("ast")
    stl = _col("stl")
    blk = _col("blk")
    tov = _col("tov")
    pf = _col("pf")

    if "fg2m" in idx:
        fg2m = torch.minimum(_col("fg2m"), fga2)
    else:
        fg2m = fga2 * _resolve_rate_tensor(fg2_rate, reference=fga2, default=float(fg2_rate_default))
    if "fg3m" in idx:
        fg3m = torch.minimum(_col("fg3m"), fga3)
    else:
        fg3m = fga3 * _resolve_rate_tensor(fg3_rate, reference=fga3, default=float(fg3_rate_default))
    if "ftm" in idx:
        ftm = torch.minimum(_col("ftm"), fta)
    else:
        ftm = fta * _resolve_rate_tensor(ft_rate, reference=fta, default=float(ft_rate_default))

    out_cols = list(contract_columns) if contract_columns is not None else flow_contract_columns(include_pf=("pf" in idx))
    values = {
        "fga2": fga2,
        "fg2m": fg2m,
        "fga3": fga3,
        "fg3m": fg3m,
        "fta": fta,
        "ftm": ftm,
        "oreb": oreb,
        "dreb": dreb,
        "ast": ast,
        "stl": stl,
        "blk": blk,
        "tov": tov,
        "pf": pf,
    }
    missing = [name for name in out_cols if name not in values]
    if missing:
        raise ValueError(f"Unsupported contract columns requested: {missing}")
    return torch.cat([values[name].unsqueeze(-1) for name in out_cols], dim=-1)


def _resolve_minutes_active_mask(
    predicted_active_mask: torch.Tensor,
    *,
    target_active_mask: torch.Tensor | None = None,
    player_team_index: torch.Tensor | None = None,
    minutes_use_target_active: bool = False,
    minutes_teacher_forcing_prob: float = 1.0,
    minutes_teacher_forcing_mode: str = "batch",
) -> torch.Tensor:
    """Choose the minutes-conditioning active mask.

    Supports training-time exposure to predicted active masks at batch, example,
    or team granularity while leaving inference behavior unchanged.
    """

    pred_mask = predicted_active_mask.to(dtype=torch.bool)
    if bool(minutes_use_target_active):
        if target_active_mask is None:
            raise ValueError("minutes_use_target_active=True requires target_active_mask")
        return target_active_mask.to(dtype=torch.bool)

    if target_active_mask is None:
        return pred_mask

    target_mask = target_active_mask.to(dtype=torch.bool)
    prob = float(min(1.0, max(0.0, minutes_teacher_forcing_prob)))
    if prob <= 0.0:
        return pred_mask
    if prob >= 1.0:
        return target_mask

    mode = str(minutes_teacher_forcing_mode).strip().lower()
    device = pred_mask.device
    if mode == "batch":
        use_target = bool(torch.rand((), device=device).item() < prob)
        return target_mask if use_target else pred_mask
    if mode == "example":
        mix_mask = (torch.rand((pred_mask.shape[0], 1), device=device) < prob).expand_as(pred_mask)
        return torch.where(mix_mask, target_mask, pred_mask)
    if mode == "team":
        if player_team_index is None:
            raise ValueError("minutes_teacher_forcing_mode='team' requires player_team_index")
        if player_team_index.shape != pred_mask.shape:
            raise ValueError("player_team_index must match active mask shape for team mixing")
        team_mix = torch.rand((pred_mask.shape[0], 2), device=device) < prob
        mix_mask = torch.gather(team_mix, dim=1, index=player_team_index.to(dtype=torch.long))
        return torch.where(mix_mask, target_mask, pred_mask)

    raise ValueError(f"Unsupported minutes_teacher_forcing_mode={minutes_teacher_forcing_mode!r}")


def _resolve_flow_conditioning_minutes(
    predicted_minutes: torch.Tensor,
    *,
    target_minutes: torch.Tensor | None = None,
    player_team_index: torch.Tensor | None = None,
    teacher_forcing_prob: float = 1.0,
    teacher_forcing_mode: str = "batch",
) -> torch.Tensor:
    """Choose flow-conditioning minutes source.

    This mirrors minutes teacher-forcing behavior and allows batch/example/team
    granularity mixing between target and predicted minutes during training.
    """

    pred = predicted_minutes.to(dtype=torch.float32)
    if target_minutes is None:
        return pred

    target = target_minutes.to(dtype=pred.dtype)
    prob = float(min(1.0, max(0.0, teacher_forcing_prob)))
    if prob <= 0.0:
        return pred
    if prob >= 1.0:
        return target

    mode = str(teacher_forcing_mode).strip().lower()
    device = pred.device
    if mode == "batch":
        use_target = bool(torch.rand((), device=device).item() < prob)
        return target if use_target else pred
    if mode == "example":
        mix_mask = (torch.rand((pred.shape[0], 1), device=device) < prob).expand_as(pred)
        return torch.where(mix_mask, target, pred)
    if mode == "team":
        if player_team_index is None:
            raise ValueError("teacher_forcing_mode='team' requires player_team_index")
        if player_team_index.shape != pred.shape:
            raise ValueError("player_team_index must match minutes shape for team mixing")
        team_mix = torch.rand((pred.shape[0], 2), device=device) < prob
        mix_mask = torch.gather(team_mix, dim=1, index=player_team_index.to(dtype=torch.long))
        return torch.where(mix_mask, target, pred)

    raise ValueError(f"Unsupported teacher_forcing_mode={teacher_forcing_mode!r}")


@dataclass(frozen=True)
class GameTransformerV2Config:
    feature_columns: list[str]
    feature_mean: list[float]
    feature_std: list[float]
    game_feature_columns: list[str]
    team_feature_columns: list[str]
    efficiency_sidecar_feature_columns: list[str] = field(default_factory=list)
    efficiency_sidecar_feature_mean: list[float] = field(default_factory=list)
    efficiency_sidecar_feature_std: list[float] = field(default_factory=list)
    d_model: int = 192
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 6
    dropout: float = 0.1
    ff_mult: float = 4.0
    min_active_count: int = 5
    max_active_count: int = 13
    active_threshold_minutes: float = 4.0
    total_minutes_per_team: float = 240.0
    max_minutes_per_player: float = 48.0
    enable_minutes_hurdle_head: bool = False
    minutes_hurdle_hidden: int = 64
    minutes_hurdle_sigma_floor: float = 0.5
    enable_minutes_role_head: bool = False
    minutes_role_use_context_for_preferences: bool = True
    minutes_role_hidden: int = 64
    minutes_role_embedding_dim: int = 32
    minutes_role_num_classes: int = 5
    enable_starter_promotion_head: bool = False
    starter_promotion_hidden_dim: int = 64
    flow_coupling_type: str = "affine"
    flow_num_blocks: int = 4
    flow_scale_clip: float = 3.0  # H1 fix: increased from 2.0 to reduce star under-projection
    flow_rqs_num_bins: int = 8
    flow_rqs_tail_bound: float = 40.0
    flow_rqs_min_bin_width: float = 1e-3
    flow_rqs_min_bin_height: float = 1e-3
    flow_rqs_min_derivative: float = 1e-3
    flow_mean_ctx_weight: float = 1.0
    flow_context_mode: str = "attention"  # H2 fix: "attention" instead of "mean" for star concentration
    flow_target_schema: str = FLOW_TARGET_SCHEMA_DEFAULT
    flow_use_minutes_conditioning: bool = False
    include_pf_in_flow_targets: bool = False
    enable_efficiency_head: bool = False
    efficiency_head_hidden: int = 128
    efficiency_ft_prior_mean: float = 0.77
    efficiency_ft_prior_strength: float = 6.0
    efficiency_fg2_prior_mean: float = 0.54
    efficiency_fg2_prior_strength: float = 8.0
    efficiency_fg3_prior_mean: float = 0.36
    efficiency_fg3_prior_strength: float = 8.0
    efficiency_market_context: bool = False
    efficiency_market_hidden: int = 32
    efficiency_market_alpha: float = 1.0
    efficiency_sidecar_hidden: int = 32
    efficiency_sidecar_alpha: float = 1.0
    enable_team_ppp_head: bool = False
    team_ppp_head_hidden: int = 128
    team_ppp_to_backbone: bool = False
    team_ppp_latent_hidden: int = 32
    team_ppp_backbone_alpha: float = 1.0
    team_ppp_to_efficiency: bool = False
    team_ppp_efficiency_alpha: float = 1.0
    team_ppp_direct_backbone_context: bool = False
    team_ppp_direct_efficiency_context: bool = False
    enable_team_advantage_head: bool = False
    team_advantage_head_hidden: int = 64
    team_advantage_direct_backbone_context: bool = False
    # Possession backbone (section 15 refactor)
    enable_possession_backbone: bool = False
    enable_three_pa_share: bool = False
    possession_head_hidden: int = 128
    possession_mu_mode: str = "absolute"
    possession_mu_baseline: float = 100.0
    enable_team_possession_split_head: bool = False
    team_possession_max_delta: float = 8.0
    backbone_hidden: int = 128
    three_pa_share_hidden: int = 64
    enable_usage_share_head: bool = False
    usage_share_head_hidden: int = 128
    enable_team_points_budget_head: bool = False
    team_points_budget_head_hidden: int = 128
    team_points_budget_parameterization: str = "absolute"
    team_points_budget_to_backbone: bool = False
    team_points_budget_latent_hidden: int = 32
    team_points_reconcile_budget: bool = False
    team_points_reconcile_alpha: float = 1.0
    team_opportunity_budget_parameterization: str = "absolute"
    team_opportunity_budget_to_backbone: bool = False
    team_opportunity_budget_latent_hidden: int = 32
    team_opportunity_budget_backbone_alpha: float = 1.0
    team_opportunity_reconcile_budget: bool = False
    team_opportunity_reconcile_alpha: float = 1.0
    team_opportunity_reconcile_preserve_possessions: bool = False
    enable_team_ast_budget_head: bool = False
    team_ast_budget_head_hidden: int = 128
    enable_assist_share_head: bool = False
    assist_share_head_hidden: int = 128
    enable_team_rebound_budget_head: bool = False
    team_rebound_budget_head_hidden: int = 128
    rebound_budget_parameterization: str = "absolute"
    rebound_oreb_rate_cap: float = 1.0
    rebound_dreb_rate_cap: float = 0.85
    rebound_dreb_deterministic_discount: float = 1.0
    rebound_oreb_budget_blend_alpha: float = 1.0
    rebound_dreb_budget_blend_alpha: float = 1.0
    rebound_oreb_reconcile_use_flow_budget: bool = False
    enable_rebound_budget_blend_gate: bool = False
    rebound_budget_blend_gate_hidden: int = 64
    rebound_budget_blend_gate_init_alpha: float = 0.25
    enable_rebound_share_head: bool = False
    rebound_share_head_hidden: int = 128
    rebound_share_condition_feature_columns: list[str] = field(default_factory=list)
    rebound_share_condition_hidden: int = 32
    assist_share_condition_feature_columns: list[str] = field(default_factory=list)
    assist_share_condition_hidden: int = 32
    enable_ast_blend_gate: bool = False
    ast_blend_gate_hidden: int = 128
    ast_blend_gate_init_alpha: float = 0.75
    assist_share_replace_flow_ast: bool = False
    assist_share_factorized_ast: bool = False
    assist_share_reconcile_ast_budget: bool = False
    assist_share_reconcile_alpha: float = 0.75
    assist_share_reconcile_temperature: float = 0.85
    rebound_factor_reconcile_oreb_dreb: bool = False
    rebound_factor_reconcile_mode: str = "both"
    rebound_factor_reconcile_alpha: float = 0.50
    rebound_factor_reconcile_temperature: float = 0.90
    overflow_protected_prior_play_prob_floor: float = PROTECTED_PRIOR_PLAY_PROB_FLOOR
    overflow_protected_prior_minutes_floor: float = PROTECTED_PRIOR_MINUTES_FLOOR
    overflow_risk_weight_consecutive_active_dnp: float = OVERFLOW_RISK_WEIGHT_CONSECUTIVE_ACTIVE_DNP
    overflow_risk_weight_active_but_dnp_rate_last10: float = OVERFLOW_RISK_WEIGHT_ACTIVE_BUT_DNP_RATE_LAST10
    overflow_risk_weight_inactive_streak_len: float = OVERFLOW_RISK_WEIGHT_INACTIVE_STREAK_LEN
    overflow_keep_weight_prior_play_prob: float = OVERFLOW_KEEP_WEIGHT_PRIOR_PLAY_PROB
    overflow_keep_weight_prior_minutes: float = OVERFLOW_KEEP_WEIGHT_PRIOR_MINUTES
    backbone_env_feature_columns: list[str] = field(default_factory=list)
    backbone_env_enrich_features: bool = False
    backbone_side_market_context: bool = False
    backbone_side_market_hidden: int = 32
    backbone_env_adapter_dim: int = 0
    backbone_env_adapter_hidden: int = 32
    enable_env_side_channel: bool = False
    env_side_channel_dim: int = 32
    env_side_channel_hidden: int = 64
    version: str = "game_transformer_v2"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GameTransformerV2Config":
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in dict(payload).items() if k in known}
        # Backward-compatible defaults for models trained before H1+H2 changes
        # If config doesn't have flow_context_mode, use "mean" (original behavior)
        if "flow_context_mode" not in filtered:
            filtered["flow_context_mode"] = "mean"
        # If config doesn't have flow_scale_clip, use 2.0 (original default)
        if "flow_scale_clip" not in filtered:
            filtered["flow_scale_clip"] = 2.0
        if "flow_rqs_num_bins" not in filtered:
            filtered["flow_rqs_num_bins"] = 8
        if "flow_rqs_tail_bound" not in filtered:
            filtered["flow_rqs_tail_bound"] = 40.0
        if "flow_rqs_min_bin_width" not in filtered:
            filtered["flow_rqs_min_bin_width"] = 1e-3
        if "flow_rqs_min_bin_height" not in filtered:
            filtered["flow_rqs_min_bin_height"] = 1e-3
        if "flow_rqs_min_derivative" not in filtered:
            filtered["flow_rqs_min_derivative"] = 1e-3
        if "flow_target_schema" not in filtered:
            filtered["flow_target_schema"] = FLOW_TARGET_SCHEMA_DEFAULT
        else:
            filtered["flow_target_schema"] = normalize_flow_target_schema(str(filtered["flow_target_schema"]))
        if "enable_minutes_hurdle_head" not in filtered:
            filtered["enable_minutes_hurdle_head"] = False
        if "minutes_hurdle_hidden" not in filtered:
            filtered["minutes_hurdle_hidden"] = 64
        if "minutes_hurdle_sigma_floor" not in filtered:
            filtered["minutes_hurdle_sigma_floor"] = 0.5
        if "enable_minutes_role_head" not in filtered:
            filtered["enable_minutes_role_head"] = False
        if "minutes_role_use_context_for_preferences" not in filtered:
            filtered["minutes_role_use_context_for_preferences"] = True
        if "minutes_role_hidden" not in filtered:
            filtered["minutes_role_hidden"] = 64
        if "minutes_role_embedding_dim" not in filtered:
            filtered["minutes_role_embedding_dim"] = 32
        if "minutes_role_num_classes" not in filtered:
            filtered["minutes_role_num_classes"] = 5
        if "enable_starter_promotion_head" not in filtered:
            filtered["enable_starter_promotion_head"] = False
        if "starter_promotion_hidden_dim" not in filtered:
            filtered["starter_promotion_hidden_dim"] = 64
        if "enable_efficiency_head" not in filtered:
            filtered["enable_efficiency_head"] = False
        if "efficiency_sidecar_feature_columns" not in filtered:
            filtered["efficiency_sidecar_feature_columns"] = []
        if "efficiency_sidecar_feature_mean" not in filtered:
            filtered["efficiency_sidecar_feature_mean"] = []
        if "efficiency_sidecar_feature_std" not in filtered:
            filtered["efficiency_sidecar_feature_std"] = []
        if "efficiency_market_context" not in filtered:
            filtered["efficiency_market_context"] = False
        if "efficiency_market_hidden" not in filtered:
            filtered["efficiency_market_hidden"] = 32
        if "efficiency_market_alpha" not in filtered:
            filtered["efficiency_market_alpha"] = 1.0
        if "efficiency_sidecar_hidden" not in filtered:
            filtered["efficiency_sidecar_hidden"] = 32
        if "efficiency_sidecar_alpha" not in filtered:
            filtered["efficiency_sidecar_alpha"] = 1.0
        if "enable_team_ppp_head" not in filtered:
            filtered["enable_team_ppp_head"] = False
        if "team_ppp_head_hidden" not in filtered:
            filtered["team_ppp_head_hidden"] = 128
        if "team_ppp_to_backbone" not in filtered:
            filtered["team_ppp_to_backbone"] = False
        if "team_ppp_latent_hidden" not in filtered:
            filtered["team_ppp_latent_hidden"] = 32
        if "team_ppp_backbone_alpha" not in filtered:
            filtered["team_ppp_backbone_alpha"] = 1.0
        if "team_ppp_to_efficiency" not in filtered:
            filtered["team_ppp_to_efficiency"] = False
        if "team_ppp_efficiency_alpha" not in filtered:
            filtered["team_ppp_efficiency_alpha"] = 1.0
        if "team_ppp_direct_backbone_context" not in filtered:
            filtered["team_ppp_direct_backbone_context"] = False
        if "team_ppp_direct_efficiency_context" not in filtered:
            filtered["team_ppp_direct_efficiency_context"] = False
        if "enable_team_advantage_head" not in filtered:
            filtered["enable_team_advantage_head"] = False
        if "team_advantage_head_hidden" not in filtered:
            filtered["team_advantage_head_hidden"] = 64
        if "team_advantage_direct_backbone_context" not in filtered:
            filtered["team_advantage_direct_backbone_context"] = False
        if "efficiency_head_hidden" not in filtered:
            filtered["efficiency_head_hidden"] = 128
        if "efficiency_ft_prior_mean" not in filtered:
            filtered["efficiency_ft_prior_mean"] = 0.77
        if "efficiency_ft_prior_strength" not in filtered:
            filtered["efficiency_ft_prior_strength"] = 6.0
        if "efficiency_fg2_prior_mean" not in filtered:
            filtered["efficiency_fg2_prior_mean"] = 0.54
        if "efficiency_fg2_prior_strength" not in filtered:
            filtered["efficiency_fg2_prior_strength"] = 8.0
        if "efficiency_fg3_prior_mean" not in filtered:
            filtered["efficiency_fg3_prior_mean"] = 0.36
        if "efficiency_fg3_prior_strength" not in filtered:
            filtered["efficiency_fg3_prior_strength"] = 8.0
        # Backbone defaults for models trained before possession refactor
        if "enable_possession_backbone" not in filtered:
            filtered["enable_possession_backbone"] = False
        if "enable_team_possession_split_head" not in filtered:
            filtered["enable_team_possession_split_head"] = False
        if "team_possession_max_delta" not in filtered:
            filtered["team_possession_max_delta"] = 8.0
        if "enable_three_pa_share" not in filtered:
            filtered["enable_three_pa_share"] = False
        if "possession_mu_mode" not in filtered:
            filtered["possession_mu_mode"] = "absolute"
        if "possession_mu_baseline" not in filtered:
            filtered["possession_mu_baseline"] = 100.0
        if "enable_usage_share_head" not in filtered:
            filtered["enable_usage_share_head"] = False
        if "usage_share_head_hidden" not in filtered:
            filtered["usage_share_head_hidden"] = 128
        if "enable_team_points_budget_head" not in filtered:
            filtered["enable_team_points_budget_head"] = False
        if "team_points_budget_head_hidden" not in filtered:
            filtered["team_points_budget_head_hidden"] = 128
        if "team_points_budget_parameterization" not in filtered:
            filtered["team_points_budget_parameterization"] = "absolute"
        if "team_points_budget_to_backbone" not in filtered:
            filtered["team_points_budget_to_backbone"] = False
        if "team_points_budget_latent_hidden" not in filtered:
            filtered["team_points_budget_latent_hidden"] = 32
        if "team_points_reconcile_budget" not in filtered:
            filtered["team_points_reconcile_budget"] = False
        if "team_points_reconcile_alpha" not in filtered:
            filtered["team_points_reconcile_alpha"] = 1.0
        if "team_opportunity_budget_parameterization" not in filtered:
            filtered["team_opportunity_budget_parameterization"] = "absolute"
        if "team_opportunity_budget_to_backbone" not in filtered:
            filtered["team_opportunity_budget_to_backbone"] = False
        if "team_opportunity_budget_latent_hidden" not in filtered:
            filtered["team_opportunity_budget_latent_hidden"] = 32
        if "team_opportunity_budget_backbone_alpha" not in filtered:
            filtered["team_opportunity_budget_backbone_alpha"] = 1.0
        if "team_opportunity_reconcile_budget" not in filtered:
            filtered["team_opportunity_reconcile_budget"] = False
        if "team_opportunity_reconcile_alpha" not in filtered:
            filtered["team_opportunity_reconcile_alpha"] = 1.0
        if "team_opportunity_reconcile_preserve_possessions" not in filtered:
            filtered["team_opportunity_reconcile_preserve_possessions"] = False
        if "enable_team_ast_budget_head" not in filtered:
            filtered["enable_team_ast_budget_head"] = False
        if "team_ast_budget_head_hidden" not in filtered:
            filtered["team_ast_budget_head_hidden"] = 128
        if "enable_assist_share_head" not in filtered:
            filtered["enable_assist_share_head"] = False
        if "assist_share_head_hidden" not in filtered:
            filtered["assist_share_head_hidden"] = 128
        if "enable_team_rebound_budget_head" not in filtered:
            filtered["enable_team_rebound_budget_head"] = False
        if "team_rebound_budget_head_hidden" not in filtered:
            filtered["team_rebound_budget_head_hidden"] = 128
        if "rebound_budget_parameterization" not in filtered:
            filtered["rebound_budget_parameterization"] = "absolute"
        if "rebound_oreb_rate_cap" not in filtered:
            filtered["rebound_oreb_rate_cap"] = 1.0
        if "rebound_dreb_rate_cap" not in filtered:
            filtered["rebound_dreb_rate_cap"] = 0.85
        if "rebound_dreb_deterministic_discount" not in filtered:
            filtered["rebound_dreb_deterministic_discount"] = 1.0
        if "rebound_oreb_budget_blend_alpha" not in filtered:
            filtered["rebound_oreb_budget_blend_alpha"] = 1.0
        if "rebound_dreb_budget_blend_alpha" not in filtered:
            filtered["rebound_dreb_budget_blend_alpha"] = 1.0
        if "rebound_oreb_reconcile_use_flow_budget" not in filtered:
            filtered["rebound_oreb_reconcile_use_flow_budget"] = False
        if "enable_rebound_budget_blend_gate" not in filtered:
            filtered["enable_rebound_budget_blend_gate"] = False
        if "rebound_budget_blend_gate_hidden" not in filtered:
            filtered["rebound_budget_blend_gate_hidden"] = 64
        if "rebound_budget_blend_gate_init_alpha" not in filtered:
            filtered["rebound_budget_blend_gate_init_alpha"] = 0.25
        if "enable_rebound_share_head" not in filtered:
            filtered["enable_rebound_share_head"] = False
        if "rebound_share_head_hidden" not in filtered:
            filtered["rebound_share_head_hidden"] = 128
        if "rebound_share_condition_feature_columns" not in filtered:
            filtered["rebound_share_condition_feature_columns"] = []
        if "rebound_share_condition_hidden" not in filtered:
            filtered["rebound_share_condition_hidden"] = 32
        if "assist_share_condition_feature_columns" not in filtered:
            filtered["assist_share_condition_feature_columns"] = []
        if "assist_share_condition_hidden" not in filtered:
            filtered["assist_share_condition_hidden"] = 32
        if "enable_ast_blend_gate" not in filtered:
            filtered["enable_ast_blend_gate"] = False
        if "ast_blend_gate_hidden" not in filtered:
            filtered["ast_blend_gate_hidden"] = 128
        if "ast_blend_gate_init_alpha" not in filtered:
            filtered["ast_blend_gate_init_alpha"] = 0.75
        if "assist_share_replace_flow_ast" not in filtered:
            filtered["assist_share_replace_flow_ast"] = False
        if "assist_share_factorized_ast" not in filtered:
            filtered["assist_share_factorized_ast"] = False
        if "assist_share_reconcile_ast_budget" not in filtered:
            filtered["assist_share_reconcile_ast_budget"] = False
        if "assist_share_reconcile_alpha" not in filtered:
            filtered["assist_share_reconcile_alpha"] = 0.75
        if "assist_share_reconcile_temperature" not in filtered:
            filtered["assist_share_reconcile_temperature"] = 0.85
        if "rebound_factor_reconcile_oreb_dreb" not in filtered:
            filtered["rebound_factor_reconcile_oreb_dreb"] = False
        if "rebound_factor_reconcile_mode" not in filtered:
            filtered["rebound_factor_reconcile_mode"] = "both"
        if "rebound_factor_reconcile_alpha" not in filtered:
            filtered["rebound_factor_reconcile_alpha"] = 0.50
        if "rebound_factor_reconcile_temperature" not in filtered:
            filtered["rebound_factor_reconcile_temperature"] = 0.90
        if "backbone_env_feature_columns" not in filtered:
            filtered["backbone_env_feature_columns"] = []
        if "backbone_env_enrich_features" not in filtered:
            filtered["backbone_env_enrich_features"] = False
        if "backbone_side_market_context" not in filtered:
            filtered["backbone_side_market_context"] = False
        if "backbone_side_market_hidden" not in filtered:
            filtered["backbone_side_market_hidden"] = 32
        if "backbone_env_adapter_dim" not in filtered:
            filtered["backbone_env_adapter_dim"] = 0
        if "backbone_env_adapter_hidden" not in filtered:
            filtered["backbone_env_adapter_hidden"] = 32
        if "enable_env_side_channel" not in filtered:
            filtered["enable_env_side_channel"] = False
        if "env_side_channel_dim" not in filtered:
            filtered["env_side_channel_dim"] = 32
        if "env_side_channel_hidden" not in filtered:
            filtered["env_side_channel_hidden"] = 64
        return cls(**filtered)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "GameTransformerV2Config":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


@dataclass(frozen=True)
class GameLevelExample:
    player_features: np.ndarray  # (2,15,F)
    efficiency_sidecar_features: np.ndarray  # (2,15,C)
    player_valid_mask: np.ndarray  # (2,15)
    player_ids: np.ndarray  # (2,15)
    team_ids: np.ndarray  # (2,)
    score_minutes_deterministic: np.ndarray  # (2,15)
    score_active_deterministic: np.ndarray  # (2,15)
    force_active_worlds: np.ndarray  # (2,15) hard guardrail mask for sampled worlds
    starter_force_active_worlds: np.ndarray  # (2,15) starter-only force-active mask
    force_active_minutes_anchor: np.ndarray  # (2,15) props-implied minutes anchor
    y_minutes: np.ndarray  # (2,15)
    flow_targets: np.ndarray  # (2,15,S)
    flow_observed_mask: np.ndarray  # (2,15,S)
    lineup_available: np.ndarray  # (2,15)
    game_features: np.ndarray  # (G,)
    team_features: np.ndarray  # (2,T)
    game_id_norm: str
    game_date: str


class GameLevelDataset(Dataset[GameLevelExample]):
    def __init__(self, examples: list[GameLevelExample]) -> None:
        if not examples:
            raise ValueError("GameLevelDataset requires >=1 example")
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> GameLevelExample:
        return self.examples[idx]


@dataclass(frozen=True)
class GameTransformerV2Outputs:
    game_state: torch.Tensor
    team_states: torch.Tensor
    player_states: torch.Tensor
    player_valid_mask: torch.Tensor
    player_team_index: torch.Tensor
    active: JointActiveSetOutputs
    minutes: JointMinutesOutputs
    flow: JointGameFlowOutputs | None
    efficiency: EfficiencyHeadOutputs | None = None
    possession: PossessionHeadOutputs | None = None
    backbone: TeamEventBackboneOutputs | None = None
    usage_share: UsageShareHeadOutputs | None = None
    team_ppp: TeamPPPHeadOutputs | None = None
    team_advantage: TeamAdvantageHeadOutputs | None = None
    team_points_budget: TeamPointsBudgetHeadOutputs | None = None
    team_ast_budget: TeamAstBudgetHeadOutputs | None = None
    assist_share: AssistShareHeadOutputs | None = None
    team_rebound_budget: TeamReboundBudgetHeadOutputs | None = None
    rebound_budget_blend_gate: ReboundBudgetBlendGateHeadOutputs | None = None
    rebound_share: ReboundShareHeadOutputs | None = None
    ast_blend_gate: AstBlendGateHeadOutputs | None = None
    env_context: torch.Tensor | None = None


def _numeric_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    parts: list[pd.Series] = []
    for col in cols:
        if col not in df.columns:
            parts.append(pd.Series(0.0, index=df.index, name=col, dtype="float32"))
            continue
        if pd.api.types.is_bool_dtype(df[col]):
            parts.append(df[col].astype("float32").rename(col))
        else:
            parts.append(pd.to_numeric(df[col], errors="coerce").rename(col))
    if not parts:
        return pd.DataFrame(index=df.index)
    return pd.concat(parts, axis=1)


def _numeric_frame_with_nans_for_missing(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    parts: list[pd.Series] = []
    for col in cols:
        if col not in df.columns:
            parts.append(pd.Series(np.nan, index=df.index, name=col, dtype="float32"))
            continue
        if pd.api.types.is_bool_dtype(df[col]):
            parts.append(df[col].astype("float32").rename(col))
        else:
            parts.append(pd.to_numeric(df[col], errors="coerce").rename(col))
    if not parts:
        return pd.DataFrame(index=df.index)
    return pd.concat(parts, axis=1)


def _resolve_home_away_team_ids(game_df: pd.DataFrame | pd.Series) -> tuple[int | None, int | None]:
    # Some callers may pass a single-row slice (Series) depending on how the
    # upstream indexing was constructed. Normalize to a 1-row DataFrame.
    if isinstance(game_df, pd.Series):
        game_df = game_df.to_frame().T

    # Explicit home/away ids are optional in some inference feature frames.
    home_raw = game_df["home_team_id"] if "home_team_id" in game_df.columns else pd.Series([], dtype="float64")
    away_raw = game_df["away_team_id"] if "away_team_id" in game_df.columns else pd.Series([], dtype="float64")
    home_series = pd.to_numeric(home_raw, errors="coerce").dropna().astype("int64")
    away_series = pd.to_numeric(away_raw, errors="coerce").dropna().astype("int64")
    home_id = int(home_series.mode().iloc[0]) if not home_series.empty else None
    away_id = int(away_series.mode().iloc[0]) if not away_series.empty else None
    if home_id is not None and away_id is not None and home_id != away_id:
        return home_id, away_id

    # Fallback using home_flag if explicit team ids are unavailable.
    if "home_flag" in game_df.columns:
        hf = pd.to_numeric(game_df["home_flag"], errors="coerce")
        home_rows = game_df.loc[hf == 1]
        away_rows = game_df.loc[hf == 0]
        if not home_rows.empty and not away_rows.empty:
            home = pd.to_numeric(home_rows["team_id"], errors="coerce").dropna().astype("int64")
            away = pd.to_numeric(away_rows["team_id"], errors="coerce").dropna().astype("int64")
            if not home.empty and not away.empty:
                return int(home.mode().iloc[0]), int(away.mode().iloc[0])

    teams = sorted(
        {
            int(v)
            for v in pd.to_numeric(game_df["team_id"], errors="coerce").dropna().astype("int64").tolist()
        }
    )
    if len(teams) >= 2:
        return teams[0], teams[1]
    if len(teams) == 1:
        return teams[0], None
    return None, None


def _sort_team_rows(
    team_df: pd.DataFrame,
    *,
    protected_prior_play_prob_floor: float,
    protected_prior_minutes_floor: float,
    risk_weight_consecutive_active_dnp: float,
    risk_weight_active_but_dnp_rate_last10: float,
    risk_weight_inactive_streak_len: float,
    keep_weight_prior_play_prob: float,
    keep_weight_prior_minutes: float,
) -> pd.DataFrame:
    out = team_df.copy()
    if "is_out" not in out.columns:
        out["is_out"] = 0.0
    if "lineup_starter_announced" not in out.columns:
        out["lineup_starter_announced"] = 0
    if "prior_play_prob" not in out.columns:
        out["prior_play_prob"] = 0.0
    if "minutes_from_stints_prior_20" not in out.columns:
        out["minutes_from_stints_prior_20"] = 0.0
    if "an_has_any_props" not in out.columns:
        out["an_has_any_props"] = 0.0
    if "an_implied_minutes" not in out.columns:
        out["an_implied_minutes"] = 0.0
    if "consecutive_active_dnp" not in out.columns:
        out["consecutive_active_dnp"] = 0.0
    if "active_but_dnp_rate_last10" not in out.columns:
        out["active_but_dnp_rate_last10"] = 0.0
    if "inactive_streak_len" not in out.columns:
        out["inactive_streak_len"] = 0.0

    out["is_out"] = pd.to_numeric(out["is_out"], errors="coerce").fillna(0.0)
    out["lineup_starter_announced"] = pd.to_numeric(out["lineup_starter_announced"], errors="coerce").fillna(0.0)
    out["prior_play_prob"] = pd.to_numeric(out["prior_play_prob"], errors="coerce").fillna(0.0)
    out["minutes_from_stints_prior_20"] = pd.to_numeric(out["minutes_from_stints_prior_20"], errors="coerce").fillna(0.0)
    out["an_has_any_props"] = pd.to_numeric(out["an_has_any_props"], errors="coerce").fillna(0.0)
    out["an_implied_minutes"] = pd.to_numeric(out["an_implied_minutes"], errors="coerce").fillna(0.0)
    out["consecutive_active_dnp"] = pd.to_numeric(out["consecutive_active_dnp"], errors="coerce").fillna(0.0)
    out["active_but_dnp_rate_last10"] = pd.to_numeric(out["active_but_dnp_rate_last10"], errors="coerce").fillna(0.0)
    out["inactive_streak_len"] = pd.to_numeric(out["inactive_streak_len"], errors="coerce").fillna(0.0)
    out["player_id"] = pd.to_numeric(out["player_id"], errors="coerce").fillna(0).astype("int64")

    starter = out["lineup_starter_announced"].ge(0.5)
    has_props = out["an_has_any_props"].ge(0.5)
    has_implied_minutes = out["an_implied_minutes"].gt(0.0)
    high_prior = out["prior_play_prob"].ge(float(protected_prior_play_prob_floor)) | out[
        "minutes_from_stints_prior_20"
    ].ge(float(protected_prior_minutes_floor))
    protected = starter | has_props | has_implied_minutes | high_prior
    out["overflow_protected"] = protected.astype(np.int8)

    # Lower values = higher risk of pre-tip DNP/zero-minute outcome.
    out["overflow_dnp_risk"] = (
        float(risk_weight_consecutive_active_dnp) * out["consecutive_active_dnp"].clip(lower=0.0, upper=20.0)
        + float(risk_weight_active_but_dnp_rate_last10) * out["active_but_dnp_rate_last10"].clip(lower=0.0, upper=1.0)
        + float(risk_weight_inactive_streak_len) * out["inactive_streak_len"].clip(lower=0.0, upper=20.0)
    )
    # Higher values = better keep candidates for overflow tie-breaks.
    out["overflow_keep_score"] = (
        float(keep_weight_prior_play_prob) * out["prior_play_prob"].clip(lower=0.0, upper=1.0)
        + float(keep_weight_prior_minutes) * out["minutes_from_stints_prior_20"].clip(lower=0.0, upper=48.0)
        - out["overflow_dnp_risk"]
    )

    return out.sort_values(
        by=[
            "is_out",
            "overflow_protected",
            "overflow_keep_score",
            "lineup_starter_announced",
            "an_has_any_props",
            "an_implied_minutes",
            "prior_play_prob",
            "minutes_from_stints_prior_20",
            "player_id",
        ],
        ascending=[True, False, False, False, False, False, False, False, True],
        kind="mergesort",
    )


def build_game_level_examples(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    game_feature_columns: list[str],
    team_feature_columns: list[str],
    efficiency_sidecar_feature_columns: list[str] | None = None,
    efficiency_sidecar_feature_mean: np.ndarray | None = None,
    efficiency_sidecar_feature_std: np.ndarray | None = None,
    flow_label_columns: list[str] | None = None,
    minutes_label_col: str = "minutes_label",
    max_players_per_team: int = MAX_PLAYERS_PER_TEAM,
    min_valid_players_per_team: int = MIN_FEASIBLE_PLAYERS_PER_TEAM,
    overflow_protected_prior_play_prob_floor: float = PROTECTED_PRIOR_PLAY_PROB_FLOOR,
    overflow_protected_prior_minutes_floor: float = PROTECTED_PRIOR_MINUTES_FLOOR,
    overflow_risk_weight_consecutive_active_dnp: float = OVERFLOW_RISK_WEIGHT_CONSECUTIVE_ACTIVE_DNP,
    overflow_risk_weight_active_but_dnp_rate_last10: float = OVERFLOW_RISK_WEIGHT_ACTIVE_BUT_DNP_RATE_LAST10,
    overflow_risk_weight_inactive_streak_len: float = OVERFLOW_RISK_WEIGHT_INACTIVE_STREAK_LEN,
    overflow_keep_weight_prior_play_prob: float = OVERFLOW_KEEP_WEIGHT_PRIOR_PLAY_PROB,
    overflow_keep_weight_prior_minutes: float = OVERFLOW_KEEP_WEIGHT_PRIOR_MINUTES,
) -> list[GameLevelExample]:
    """Convert flat player rows into per-game (home+away) training examples."""

    if max_players_per_team <= 0:
        raise ValueError("max_players_per_team must be > 0")
    if min_valid_players_per_team <= 0:
        raise ValueError("min_valid_players_per_team must be > 0")
    if min_valid_players_per_team > max_players_per_team:
        raise ValueError("min_valid_players_per_team must be <= max_players_per_team")
    if len(feature_columns) <= 0:
        raise ValueError("feature_columns must be non-empty")

    df = frame.copy().reset_index(drop=True)
    if "game_id_norm" not in df.columns:
        df["game_id_norm"] = zfill_game_id_series(df["game_id"])
    if "game_date" not in df.columns:
        raise ValueError("frame missing game_date")
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.normalize()
    if df["game_date"].isna().any():
        raise ValueError("frame has invalid game_date values")

    x = _numeric_frame(df, feature_columns).to_numpy(dtype="float32", copy=False)
    mean = np.asarray(feature_mean, dtype=np.float32)
    std = np.asarray(feature_std, dtype=np.float32)
    if mean.shape[0] != len(feature_columns) or std.shape[0] != len(feature_columns):
        raise ValueError("feature_mean/std must align with feature_columns")
    std = np.where(std <= 1e-6, 1.0, std)
    x = np.nan_to_num(x, nan=mean[None, :], posinf=mean[None, :], neginf=mean[None, :])
    x = (x - mean[None, :]) / std[None, :]

    efficiency_sidecar_cols = list(efficiency_sidecar_feature_columns or [])
    if efficiency_sidecar_cols:
        if efficiency_sidecar_feature_mean is None or efficiency_sidecar_feature_std is None:
            raise ValueError(
                "efficiency_sidecar_feature_mean/std are required when efficiency_sidecar_feature_columns are provided"
            )
        sidecar_mean = np.asarray(efficiency_sidecar_feature_mean, dtype=np.float32)
        sidecar_std = np.asarray(efficiency_sidecar_feature_std, dtype=np.float32)
        if sidecar_mean.shape[0] != len(efficiency_sidecar_cols) or sidecar_std.shape[0] != len(efficiency_sidecar_cols):
            raise ValueError("efficiency_sidecar_feature_mean/std must align with efficiency_sidecar_feature_columns")
        sidecar_std = np.where(sidecar_std <= 1e-6, 1.0, sidecar_std)
        sidecar = _numeric_frame(df, efficiency_sidecar_cols).to_numpy(dtype=np.float32, copy=False)
        sidecar = np.nan_to_num(
            sidecar,
            nan=sidecar_mean[None, :],
            posinf=sidecar_mean[None, :],
            neginf=sidecar_mean[None, :],
        )
        sidecar = (sidecar - sidecar_mean[None, :]) / sidecar_std[None, :]
    else:
        sidecar = np.zeros((len(df), 0), dtype=np.float32)

    if minutes_label_col not in df.columns:
        fallback = "minutes" if "minutes" in df.columns else None
        if fallback is None:
            raise ValueError(f"minutes label column not found: {minutes_label_col}")
        minutes_label_col = fallback

    y_minutes = pd.to_numeric(df[minutes_label_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    score_minutes_by_idx = (
        pd.to_numeric(df["gtv2_score_minutes_deterministic"], errors="coerce").to_numpy(dtype=np.float32)
        if "gtv2_score_minutes_deterministic" in df.columns
        else np.full(len(df), np.nan, dtype=np.float32)
    )
    score_active_by_idx = (
        pd.to_numeric(df["gtv2_score_active_deterministic"], errors="coerce").to_numpy(dtype=np.float32)
        if "gtv2_score_active_deterministic" in df.columns
        else np.full(len(df), np.nan, dtype=np.float32)
    )
    starter_signal = np.zeros(len(df), dtype=bool)
    for starter_col in ("lineup_starter_announced", "is_projected_starter", "is_confirmed_starter"):
        if starter_col in df.columns:
            starter_signal |= (
                pd.to_numeric(df[starter_col], errors="coerce")
                .fillna(0.0)
                .to_numpy(dtype=np.float32)
                >= 0.5
            )
    manual_force_in = np.zeros(len(df), dtype=bool)
    if "force_active_worlds" in df.columns:
        manual_force_in = (
            pd.to_numeric(df["force_active_worlds"], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
            >= 0.5
        )
    elif "manual_override_type" in df.columns:
        override_type = (
            df["manual_override_type"]
            .astype("string")
            .fillna("")
            .str.strip()
            .str.lower()
            .to_numpy(dtype=object)
        )
        manual_force_in = override_type == "force_in"
        if "manual_override_active" in df.columns:
            override_active = (
                pd.to_numeric(df["manual_override_active"], errors="coerce")
                .fillna(0.0)
                .to_numpy(dtype=np.float32)
                >= 0.5
            )
            manual_force_in &= override_active
    force_active_worlds = starter_signal | manual_force_in
    force_active_minutes_anchor = np.zeros(len(df), dtype=np.float32)
    if "an_implied_minutes" in df.columns:
        implied_minutes = (
            pd.to_numeric(df["an_implied_minutes"], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
        )
        implied_minutes = np.clip(implied_minutes, 0.0, 48.0)
        if "an_has_implied_minutes" in df.columns:
            has_implied = (
                pd.to_numeric(df["an_has_implied_minutes"], errors="coerce")
                .fillna(0.0)
                .to_numpy(dtype=np.float32)
                >= 0.5
            )
            implied_minutes = np.where(has_implied, implied_minutes, 0.0).astype(np.float32, copy=False)
        force_active_minutes_anchor = implied_minutes
    flow_cols = list(flow_label_columns or [])
    if flow_cols:
        flow_raw = _numeric_frame_with_nans_for_missing(df, flow_cols).to_numpy(dtype=np.float32, copy=False)
        flow_observed = np.isfinite(flow_raw)
        flow_values = np.nan_to_num(flow_raw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    else:
        flow_values = np.zeros((len(df), 0), dtype=np.float32)
        flow_observed = np.zeros((len(df), 0), dtype=bool)
    lineup_available = (
        pd.to_numeric(df.get("lineup_available", 0), errors="coerce").fillna(0).astype(int).to_numpy(dtype=np.int64) > 0
    )

    x_by_idx = x
    sidecar_by_idx = sidecar
    y_by_idx = y_minutes
    force_active_by_idx = force_active_worlds
    starter_force_active_by_idx = starter_signal
    force_active_minutes_anchor_by_idx = force_active_minutes_anchor
    flow_by_idx = flow_values
    flow_observed_by_idx = flow_observed
    lineup_by_idx = lineup_available

    game_feats_df = _numeric_frame(df, game_feature_columns) if game_feature_columns else pd.DataFrame(index=df.index)
    team_feats_df = _numeric_frame(df, team_feature_columns) if team_feature_columns else pd.DataFrame(index=df.index)

    examples: list[GameLevelExample] = []
    grouped = df.groupby(["game_id_norm", "game_date"], sort=False).indices
    for (game_id_norm, game_date), idx in grouped.items():
        # Pandas may return a scalar index for singleton groups (version-dependent).
        # Ensure we always index with a 1-D array so `df.iloc[...]` returns a DataFrame.
        idx_arr = np.atleast_1d(np.asarray(idx, dtype=np.int64))
        game_df = df.iloc[idx_arr]
        home_id, away_id = _resolve_home_away_team_ids(game_df)

        player_features = np.zeros((2, max_players_per_team, len(feature_columns)), dtype=np.float32)
        efficiency_sidecar_features = np.zeros((2, max_players_per_team, len(efficiency_sidecar_cols)), dtype=np.float32)
        player_valid = np.zeros((2, max_players_per_team), dtype=bool)
        player_ids = np.zeros((2, max_players_per_team), dtype=np.int64)
        team_ids = np.zeros((2,), dtype=np.int64)
        score_minutes_arr = np.full((2, max_players_per_team), np.nan, dtype=np.float32)
        score_active_arr = np.full((2, max_players_per_team), np.nan, dtype=np.float32)
        force_active_arr = np.zeros((2, max_players_per_team), dtype=bool)
        starter_force_active_arr = np.zeros((2, max_players_per_team), dtype=bool)
        force_active_minutes_anchor_arr = np.zeros((2, max_players_per_team), dtype=np.float32)
        y_minutes_arr = np.zeros((2, max_players_per_team), dtype=np.float32)
        flow_arr = np.zeros((2, max_players_per_team, len(flow_cols)), dtype=np.float32)
        flow_obs_arr = np.zeros((2, max_players_per_team, len(flow_cols)), dtype=bool)
        lineup_arr = np.zeros((2, max_players_per_team), dtype=bool)

        for side_idx, team_id in enumerate([home_id, away_id]):
            if team_id is None:
                continue
            team_rows = game_df.loc[pd.to_numeric(game_df["team_id"], errors="coerce") == int(team_id)]
            if team_rows.empty:
                continue
            team_rows = _sort_team_rows(
                team_rows,
                protected_prior_play_prob_floor=float(overflow_protected_prior_play_prob_floor),
                protected_prior_minutes_floor=float(overflow_protected_prior_minutes_floor),
                risk_weight_consecutive_active_dnp=float(overflow_risk_weight_consecutive_active_dnp),
                risk_weight_active_but_dnp_rate_last10=float(overflow_risk_weight_active_but_dnp_rate_last10),
                risk_weight_inactive_streak_len=float(overflow_risk_weight_inactive_streak_len),
                keep_weight_prior_play_prob=float(overflow_keep_weight_prior_play_prob),
                keep_weight_prior_minutes=float(overflow_keep_weight_prior_minutes),
            ).head(max_players_per_team)
            team_ids[side_idx] = int(team_id)
            local_idx = team_rows.index.to_numpy(dtype=np.int64)
            n = local_idx.shape[0]

            player_features[side_idx, :n] = x_by_idx[local_idx]
            efficiency_sidecar_features[side_idx, :n] = sidecar_by_idx[local_idx]
            player_valid[side_idx, :n] = True
            player_ids[side_idx, :n] = (
                pd.to_numeric(team_rows["player_id"], errors="coerce").fillna(0).astype("int64").to_numpy(dtype=np.int64)
            )
            score_minutes_arr[side_idx, :n] = score_minutes_by_idx[local_idx]
            score_active_arr[side_idx, :n] = score_active_by_idx[local_idx]
            force_active_arr[side_idx, :n] = force_active_by_idx[local_idx]
            starter_force_active_arr[side_idx, :n] = starter_force_active_by_idx[local_idx]
            force_active_minutes_anchor_arr[side_idx, :n] = force_active_minutes_anchor_by_idx[local_idx]
            y_minutes_arr[side_idx, :n] = y_by_idx[local_idx]
            if flow_cols:
                flow_arr[side_idx, :n, :] = flow_by_idx[local_idx, :]
                flow_obs_arr[side_idx, :n, :] = flow_observed_by_idx[local_idx, :]
            lineup_arr[side_idx, :n] = lineup_by_idx[local_idx]

        if bool(player_valid[0].sum() == 0 and player_valid[1].sum() == 0):
            continue

        # Drop malformed games where either side has no valid/non-positive team id rows,
        # or where a side cannot satisfy the 240-minute team constraint under 48-minute caps.
        if int(team_ids[0]) <= 0 or int(team_ids[1]) <= 0:
            continue
        if int(player_valid[0].sum()) < int(min_valid_players_per_team):
            continue
        if int(player_valid[1].sum()) < int(min_valid_players_per_team):
            continue

        if game_feature_columns:
            gvals = game_feats_df.iloc[idx_arr].mean(axis=0, skipna=True).to_numpy(dtype=np.float32)
            gvals = np.nan_to_num(gvals, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            gvals = np.zeros((0,), dtype=np.float32)

        if team_feature_columns:
            tvals = np.zeros((2, len(team_feature_columns)), dtype=np.float32)
            game_team_feats_df = team_feats_df.iloc[idx_arr]
            for side_idx, team_id in enumerate(team_ids.tolist()):
                if team_id <= 0:
                    continue
                mask = pd.to_numeric(game_df["team_id"], errors="coerce") == int(team_id)
                if bool(mask.any()):
                    vals = game_team_feats_df.loc[mask].mean(axis=0, skipna=True).to_numpy(dtype=np.float32)
                    tvals[side_idx] = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            tvals = np.zeros((2, 0), dtype=np.float32)

        examples.append(
            GameLevelExample(
                player_features=player_features,
                efficiency_sidecar_features=efficiency_sidecar_features,
                player_valid_mask=player_valid,
                player_ids=player_ids,
                team_ids=team_ids,
                score_minutes_deterministic=score_minutes_arr,
                score_active_deterministic=score_active_arr,
                force_active_worlds=force_active_arr,
                starter_force_active_worlds=starter_force_active_arr,
                force_active_minutes_anchor=force_active_minutes_anchor_arr,
                y_minutes=y_minutes_arr,
                flow_targets=flow_arr,
                flow_observed_mask=flow_obs_arr,
                lineup_available=lineup_arr,
                game_features=gvals,
                team_features=tvals,
                game_id_norm=str(game_id_norm),
                game_date=str(pd.Timestamp(game_date).date().isoformat()),
            )
        )

    if not examples:
        raise ValueError("No game-level examples were built from the provided frame")
    return examples


def collate_game_level_examples(batch: list[GameLevelExample]) -> dict[str, torch.Tensor | list[str]]:
    if not batch:
        raise ValueError("batch must be non-empty")

    bsz = len(batch)
    n_feat = batch[0].player_features.shape[2]
    n_sidecar_feat = batch[0].efficiency_sidecar_features.shape[2]
    n_flow = batch[0].flow_targets.shape[2]
    n_game_feat = batch[0].game_features.shape[0]
    n_team_feat = batch[0].team_features.shape[1]

    player_features = torch.zeros((bsz, 2, MAX_PLAYERS_PER_TEAM, n_feat), dtype=torch.float32)
    efficiency_sidecar_features = torch.zeros((bsz, 2, MAX_PLAYERS_PER_TEAM, n_sidecar_feat), dtype=torch.float32)
    player_valid_mask = torch.zeros((bsz, 2, MAX_PLAYERS_PER_TEAM), dtype=torch.bool)
    player_ids = torch.zeros((bsz, 2, MAX_PLAYERS_PER_TEAM), dtype=torch.long)
    team_ids = torch.zeros((bsz, 2), dtype=torch.long)
    score_minutes_deterministic = torch.full((bsz, 2, MAX_PLAYERS_PER_TEAM), float("nan"), dtype=torch.float32)
    score_active_deterministic = torch.full((bsz, 2, MAX_PLAYERS_PER_TEAM), float("nan"), dtype=torch.float32)
    force_active_worlds = torch.zeros((bsz, 2, MAX_PLAYERS_PER_TEAM), dtype=torch.bool)
    starter_force_active_worlds = torch.zeros((bsz, 2, MAX_PLAYERS_PER_TEAM), dtype=torch.bool)
    force_active_minutes_anchor = torch.zeros((bsz, 2, MAX_PLAYERS_PER_TEAM), dtype=torch.float32)
    y_minutes = torch.zeros((bsz, 2, MAX_PLAYERS_PER_TEAM), dtype=torch.float32)
    flow_targets = torch.zeros((bsz, 2, MAX_PLAYERS_PER_TEAM, n_flow), dtype=torch.float32)
    flow_observed_mask = torch.zeros((bsz, 2, MAX_PLAYERS_PER_TEAM, n_flow), dtype=torch.bool)
    lineup_available = torch.zeros((bsz, 2, MAX_PLAYERS_PER_TEAM), dtype=torch.bool)
    game_features = torch.zeros((bsz, n_game_feat), dtype=torch.float32)
    team_features = torch.zeros((bsz, 2, n_team_feat), dtype=torch.float32)

    game_ids: list[str] = []
    game_dates: list[str] = []

    for i, ex in enumerate(batch):
        player_features[i] = torch.from_numpy(ex.player_features.astype(np.float32, copy=False))
        if n_sidecar_feat > 0:
            efficiency_sidecar_features[i] = torch.from_numpy(
                ex.efficiency_sidecar_features.astype(np.float32, copy=False)
            )
        player_valid_mask[i] = torch.from_numpy(ex.player_valid_mask.astype(bool, copy=False))
        player_ids[i] = torch.from_numpy(ex.player_ids.astype(np.int64, copy=False))
        team_ids[i] = torch.from_numpy(ex.team_ids.astype(np.int64, copy=False))
        score_minutes_deterministic[i] = torch.from_numpy(
            ex.score_minutes_deterministic.astype(np.float32, copy=False)
        )
        score_active_deterministic[i] = torch.from_numpy(
            ex.score_active_deterministic.astype(np.float32, copy=False)
        )
        force_active_worlds[i] = torch.from_numpy(ex.force_active_worlds.astype(bool, copy=False))
        starter_force_active_worlds[i] = torch.from_numpy(ex.starter_force_active_worlds.astype(bool, copy=False))
        force_active_minutes_anchor[i] = torch.from_numpy(ex.force_active_minutes_anchor.astype(np.float32, copy=False))
        y_minutes[i] = torch.from_numpy(ex.y_minutes.astype(np.float32, copy=False))
        if n_flow > 0:
            flow_targets[i] = torch.from_numpy(ex.flow_targets.astype(np.float32, copy=False))
            flow_observed_mask[i] = torch.from_numpy(ex.flow_observed_mask.astype(bool, copy=False))
        lineup_available[i] = torch.from_numpy(ex.lineup_available.astype(bool, copy=False))
        if n_game_feat > 0:
            game_features[i] = torch.from_numpy(ex.game_features.astype(np.float32, copy=False))
        if n_team_feat > 0:
            team_features[i] = torch.from_numpy(ex.team_features.astype(np.float32, copy=False))
        game_ids.append(ex.game_id_norm)
        game_dates.append(ex.game_date)

    return {
        "player_features": player_features,
        "efficiency_sidecar_features": efficiency_sidecar_features,
        "player_valid_mask": player_valid_mask,
        "player_ids": player_ids,
        "team_ids": team_ids,
        "score_minutes_deterministic": score_minutes_deterministic,
        "score_active_deterministic": score_active_deterministic,
        "force_active_worlds": force_active_worlds,
        "starter_force_active_worlds": starter_force_active_worlds,
        "force_active_minutes_anchor": force_active_minutes_anchor,
        "y_minutes": y_minutes,
        "flow_targets": flow_targets,
        "flow_observed_mask": flow_observed_mask,
        "lineup_available": lineup_available,
        "game_features": game_features,
        "team_features": team_features,
        "game_id_norm": game_ids,
        "game_date": game_dates,
    }


class GameTransformerV2(nn.Module):
    """Cross-team game transformer with joint active-set and minutes heads."""

    def __init__(
        self,
        num_player_features: int,
        *,
        num_game_features: int,
        num_team_features: int,
        num_efficiency_sidecar_features: int = 0,
        game_feature_names: list[str] | tuple[str, ...] = (),
        backbone_env_feature_indices: list[int] | tuple[int, ...] = (),
        backbone_env_feature_names: list[str] | tuple[str, ...] = (),
        backbone_env_enrich_features: bool = False,
        backbone_side_market_context: bool = False,
        backbone_side_market_hidden: int = 32,
        enable_env_side_channel: bool = False,
        env_side_channel_dim: int = 32,
        env_side_channel_hidden: int = 64,
        d_model: int = 192,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 6,
        dropout: float = 0.1,
        ff_mult: float = 4.0,
        min_active_count: int = 5,
        max_active_count: int = 13,
        total_minutes_per_team: float = 240.0,
        max_minutes_per_player: float = 48.0,
        enable_minutes_hurdle_head: bool = False,
        minutes_hurdle_hidden: int = 64,
        minutes_hurdle_sigma_floor: float = 0.5,
        enable_minutes_role_head: bool = False,
        minutes_role_use_context_for_preferences: bool = True,
        minutes_role_hidden: int = 64,
        minutes_role_embedding_dim: int = 32,
        minutes_role_num_classes: int = 5,
        enable_starter_promotion_head: bool = False,
        starter_promotion_hidden_dim: int = 64,
        flow_coupling_type: str = "affine",
        flow_num_blocks: int = 4,
        flow_scale_clip: float = 3.0,
        flow_rqs_num_bins: int = 8,
        flow_rqs_tail_bound: float = 40.0,
        flow_rqs_min_bin_width: float = 1e-3,
        flow_rqs_min_bin_height: float = 1e-3,
        flow_rqs_min_derivative: float = 1e-3,
        flow_mean_ctx_weight: float = 1.0,
        flow_context_mode: str = "attention",
        flow_target_schema: str = FLOW_TARGET_SCHEMA_DEFAULT,
        flow_use_minutes_conditioning: bool = False,
        include_pf_in_flow_targets: bool = False,
        enable_efficiency_head: bool = False,
        efficiency_head_hidden: int = 128,
        efficiency_ft_prior_mean: float = 0.77,
        efficiency_ft_prior_strength: float = 6.0,
        efficiency_fg2_prior_mean: float = 0.54,
        efficiency_fg2_prior_strength: float = 8.0,
        efficiency_fg3_prior_mean: float = 0.36,
        efficiency_fg3_prior_strength: float = 8.0,
        efficiency_market_context: bool = False,
        efficiency_market_hidden: int = 32,
        efficiency_market_alpha: float = 1.0,
        efficiency_sidecar_hidden: int = 32,
        efficiency_sidecar_alpha: float = 1.0,
        enable_team_ppp_head: bool = False,
        team_ppp_head_hidden: int = 128,
        team_ppp_to_backbone: bool = False,
        team_ppp_latent_hidden: int = 32,
        team_ppp_backbone_alpha: float = 1.0,
        team_ppp_to_efficiency: bool = False,
        team_ppp_efficiency_alpha: float = 1.0,
        team_ppp_direct_backbone_context: bool = False,
        team_ppp_direct_efficiency_context: bool = False,
        enable_team_advantage_head: bool = False,
        team_advantage_head_hidden: int = 64,
        team_advantage_direct_backbone_context: bool = False,
        enable_possession_backbone: bool = False,
        enable_three_pa_share: bool = False,
        possession_head_hidden: int = 128,
        possession_mu_mode: str = "absolute",
        possession_mu_baseline: float = 100.0,
        enable_team_possession_split_head: bool = False,
        team_possession_max_delta: float = 8.0,
        backbone_hidden: int = 128,
        three_pa_share_hidden: int = 64,
        enable_usage_share_head: bool = False,
        usage_share_head_hidden: int = 128,
        enable_team_points_budget_head: bool = False,
        team_points_budget_head_hidden: int = 128,
        team_points_budget_parameterization: str = "absolute",
        team_points_budget_to_backbone: bool = False,
        team_points_budget_latent_hidden: int = 32,
        team_opportunity_budget_parameterization: str = "absolute",
        team_opportunity_budget_to_backbone: bool = False,
        team_opportunity_budget_latent_hidden: int = 32,
        team_opportunity_budget_backbone_alpha: float = 1.0,
        enable_team_ast_budget_head: bool = False,
        team_ast_budget_head_hidden: int = 128,
        enable_assist_share_head: bool = False,
        assist_share_head_hidden: int = 128,
        enable_team_rebound_budget_head: bool = False,
        team_rebound_budget_head_hidden: int = 128,
        rebound_budget_parameterization: str = "absolute",
        rebound_oreb_rate_cap: float = 1.0,
        rebound_dreb_rate_cap: float = 0.85,
        rebound_dreb_deterministic_discount: float = 1.0,
        enable_rebound_budget_blend_gate: bool = False,
        rebound_budget_blend_gate_hidden: int = 64,
        rebound_budget_blend_gate_init_alpha: float = 0.25,
        enable_rebound_share_head: bool = False,
        rebound_share_head_hidden: int = 128,
        rebound_share_condition_feature_indices: tuple[int, ...] = (),
        rebound_share_condition_feature_mean: tuple[float, ...] = (),
        rebound_share_condition_feature_std: tuple[float, ...] = (),
        rebound_share_condition_hidden: int = 32,
        assist_share_condition_feature_indices: tuple[int, ...] = (),
        assist_share_condition_feature_mean: tuple[float, ...] = (),
        assist_share_condition_feature_std: tuple[float, ...] = (),
        assist_share_condition_hidden: int = 32,
        enable_ast_blend_gate: bool = False,
        ast_blend_gate_hidden: int = 128,
        ast_blend_gate_init_alpha: float = 0.75,
        backbone_env_adapter_dim: int = 0,
        backbone_env_adapter_hidden: int = 32,
    ) -> None:
        super().__init__()
        if num_player_features <= 0:
            raise ValueError("num_player_features must be > 0")
        if num_heads <= 0 or d_model % num_heads != 0:
            raise ValueError("num_heads must divide d_model")

        self.num_player_features = int(num_player_features)
        self.num_game_features = int(num_game_features)
        self.num_team_features = int(num_team_features)
        self.num_efficiency_sidecar_features = int(num_efficiency_sidecar_features)
        self.game_feature_names = tuple(str(name) for name in game_feature_names)
        self.backbone_env_feature_indices = tuple(int(idx) for idx in backbone_env_feature_indices)
        self.backbone_env_feature_names = tuple(str(name) for name in backbone_env_feature_names)
        self.backbone_env_enrich_features = bool(backbone_env_enrich_features)
        self.backbone_side_market_context = bool(backbone_side_market_context)
        self.backbone_side_market_hidden = max(1, int(backbone_side_market_hidden))
        self.num_backbone_env_features = int(len(self.backbone_env_feature_indices))
        self.backbone_env_adapter_dim = max(0, int(backbone_env_adapter_dim))
        self.backbone_env_adapter_hidden = max(1, int(backbone_env_adapter_hidden))
        self.enable_env_side_channel = bool(enable_env_side_channel)
        self.env_side_channel_dim = max(1, int(env_side_channel_dim))
        self.env_side_channel_hidden = max(1, int(env_side_channel_hidden))
        self.d_model = int(d_model)
        self.flow_target_schema = normalize_flow_target_schema(flow_target_schema)
        self.flow_target_columns = flow_target_columns(
            include_pf=bool(include_pf_in_flow_targets),
            schema=self.flow_target_schema,
        )
        self.num_flow_stats = int(len(self.flow_target_columns))
        self.flow_use_minutes_conditioning = bool(flow_use_minutes_conditioning)

        self.player_proj = nn.Linear(int(num_player_features), int(d_model))
        self.game_proj = nn.Linear(int(num_game_features), int(d_model)) if int(num_game_features) > 0 else None
        self.team_proj = nn.Linear(int(num_team_features), int(d_model)) if int(num_team_features) > 0 else None

        self.game_token = nn.Parameter(torch.randn(int(d_model)) * 0.02)
        self.team_tokens = nn.Parameter(torch.randn(2, int(d_model)) * 0.02)

        # token types: 0=game, 1=team, 2=player
        self.token_type_embedding = nn.Embedding(3, int(d_model))
        # side ids: 0=home, 1=away, 2=neutral(game)
        self.side_embedding = nn.Embedding(3, int(d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=int(d_model),
            nhead=int(num_heads),
            dim_feedforward=int(round(float(ff_mult) * int(d_model))),
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=int(num_layers))
        self.final_norm = nn.LayerNorm(int(d_model))
        self.dropout = nn.Dropout(float(dropout))

        self.active_head = JointActiveSetHead(
            d_model=int(d_model),
            hidden_dim=int(hidden_dim),
            dropout=float(dropout),
            min_active_count=int(min_active_count),
            max_active_count=int(max_active_count),
        )
        self.minutes_head = JointMinutesHead(
            d_model=int(d_model),
            hidden_dim=int(hidden_dim),
            dropout=float(dropout),
            total_minutes_per_team=float(total_minutes_per_team),
            max_minutes_per_player=float(max_minutes_per_player),
            enable_role_head=bool(enable_minutes_role_head),
            use_role_context_for_preferences=bool(minutes_role_use_context_for_preferences),
            role_hidden_dim=int(minutes_role_hidden),
            role_embedding_dim=int(minutes_role_embedding_dim),
            num_role_classes=int(minutes_role_num_classes),
            enable_starter_promotion_head=bool(enable_starter_promotion_head),
            starter_promotion_hidden_dim=int(starter_promotion_hidden_dim),
            enable_hurdle_head=bool(enable_minutes_hurdle_head),
            hurdle_hidden_dim=int(minutes_hurdle_hidden),
            hurdle_sigma_floor=float(minutes_hurdle_sigma_floor),
        )
        self.flow_head = JointGameFlow(
            d_model=int(d_model),
            num_stats=int(self.num_flow_stats),
            hidden_dim=int(hidden_dim),
            dropout=float(dropout),
            num_blocks=int(flow_num_blocks),
            coupling_type=str(flow_coupling_type),
            scale_clip=float(flow_scale_clip),
            rqs_num_bins=int(flow_rqs_num_bins),
            rqs_tail_bound=float(flow_rqs_tail_bound),
            rqs_min_bin_width=float(flow_rqs_min_bin_width),
            rqs_min_bin_height=float(flow_rqs_min_bin_height),
            rqs_min_derivative=float(flow_rqs_min_derivative),
            mean_ctx_weight=float(flow_mean_ctx_weight),
            context_mode=str(flow_context_mode),
            use_minutes_context=bool(flow_use_minutes_conditioning),
            env_context_dim=self.env_side_channel_dim if self.enable_env_side_channel else 0,
        )
        self.enable_efficiency_head = bool(enable_efficiency_head)
        self.efficiency_market_context = bool(efficiency_market_context)
        self.efficiency_market_alpha = float(efficiency_market_alpha)
        self.efficiency_sidecar_alpha = float(efficiency_sidecar_alpha)
        self.enable_team_ppp_head = bool(enable_team_ppp_head)
        self.team_ppp_to_backbone = bool(team_ppp_to_backbone)
        self.team_ppp_backbone_alpha = float(team_ppp_backbone_alpha)
        self.team_ppp_to_efficiency = bool(team_ppp_to_efficiency)
        self.team_ppp_efficiency_alpha = float(team_ppp_efficiency_alpha)
        self.team_ppp_direct_backbone_context = bool(team_ppp_direct_backbone_context)
        self.team_ppp_direct_efficiency_context = bool(team_ppp_direct_efficiency_context)
        self.enable_team_advantage_head = bool(enable_team_advantage_head)
        self.team_advantage_direct_backbone_context = bool(team_advantage_direct_backbone_context)
        self.team_ppp_head: TeamPPPHead | None = None
        self.team_advantage_head: TeamAdvantageHead | None = None
        self.backbone_team_ppp_encoder: nn.Module | None = None
        self.efficiency_team_ppp_encoder: nn.Module | None = None
        self.efficiency_player_sidecar_encoder: nn.Module | None = None
        self.efficiency_head: EfficiencyHead | None = None
        self.efficiency_team_market_encoder: nn.Module | None = None
        if self.enable_team_ppp_head:
            self.team_ppp_head = TeamPPPHead(
                d_model=int(d_model),
                hidden_dim=int(team_ppp_head_hidden),
                dropout=float(dropout),
            )
        if self.enable_team_advantage_head:
            self.team_advantage_head = TeamAdvantageHead(
                d_model=int(d_model),
                hidden_dim=int(team_advantage_head_hidden),
                dropout=float(dropout),
            )
        if self.team_ppp_to_backbone:
            self.backbone_team_ppp_encoder = nn.Sequential(
                nn.LayerNorm(4),
                nn.Linear(4, max(1, int(team_ppp_latent_hidden))),
                nn.GELU(),
                nn.Linear(max(1, int(team_ppp_latent_hidden)), int(d_model)),
            )
        if self.team_ppp_to_efficiency:
            self.efficiency_team_ppp_encoder = nn.Sequential(
                nn.LayerNorm(4),
                nn.Linear(4, max(1, int(team_ppp_latent_hidden))),
                nn.GELU(),
                nn.Linear(max(1, int(team_ppp_latent_hidden)), int(d_model)),
            )
        if self.enable_efficiency_head:
            if self.num_efficiency_sidecar_features > 0:
                self.efficiency_player_sidecar_encoder = nn.Sequential(
                    nn.LayerNorm(self.num_efficiency_sidecar_features),
                    nn.Linear(self.num_efficiency_sidecar_features, max(1, int(efficiency_sidecar_hidden))),
                    nn.GELU(),
                    nn.Linear(max(1, int(efficiency_sidecar_hidden)), int(d_model)),
                )
            self.efficiency_head = EfficiencyHead(
                d_model=int(d_model),
                hidden_dim=int(efficiency_head_hidden),
                dropout=float(dropout),
                num_team_context_features=4 if self.team_ppp_direct_efficiency_context else 0,
                ft_prior_mean=float(efficiency_ft_prior_mean),
                ft_prior_strength=float(efficiency_ft_prior_strength),
                fg2_prior_mean=float(efficiency_fg2_prior_mean),
                fg2_prior_strength=float(efficiency_fg2_prior_strength),
                fg3_prior_mean=float(efficiency_fg3_prior_mean),
                fg3_prior_strength=float(efficiency_fg3_prior_strength),
            )
            if self.efficiency_market_context:
                team_market_context_dim = 6 if "estimated_possessions" in self.game_feature_names else 4
                self.efficiency_team_market_encoder = nn.Sequential(
                    nn.LayerNorm(team_market_context_dim),
                    nn.Linear(team_market_context_dim, max(1, int(efficiency_market_hidden))),
                    nn.GELU(),
                    nn.Linear(max(1, int(efficiency_market_hidden)), int(d_model)),
                )

        # Possession-coupled event backbone (section 15 refactor)
        self.enable_possession_backbone = bool(enable_possession_backbone)
        self.enable_three_pa_share = bool(enable_three_pa_share)
        self.possession_head: PossessionHead | None = None
        self.event_backbone: TeamEventBackbone | None = None
        self.three_pa_share_head: ThreePAShareHead | None = None
        self.backbone_side_market_encoder: nn.Module | None = None
        backbone_raw_context_dim = (
            int(num_game_features)
            + (2 * self.num_backbone_env_features)
            + self._derived_env_feature_count()
        )
        self.backbone_env_adapter: nn.Module | None = None
        backbone_context_dim = int(backbone_raw_context_dim)
        if self.backbone_env_adapter_dim > 0 and backbone_raw_context_dim > 0:
            self.backbone_env_adapter = nn.Sequential(
                nn.LayerNorm(backbone_raw_context_dim),
                nn.Linear(backbone_raw_context_dim, self.backbone_env_adapter_hidden),
                nn.GELU(),
                nn.Linear(self.backbone_env_adapter_hidden, self.backbone_env_adapter_dim),
            )
            backbone_context_dim = self.backbone_env_adapter_dim
        self.env_side_channel_encoder: nn.Module | None = None
        if self.enable_env_side_channel and backbone_raw_context_dim > 0:
            self.env_side_channel_encoder = nn.Sequential(
                nn.LayerNorm(backbone_raw_context_dim),
                nn.Linear(backbone_raw_context_dim, self.env_side_channel_hidden),
                nn.GELU(),
                nn.Linear(self.env_side_channel_hidden, self.env_side_channel_dim),
            )
            backbone_context_dim = self.env_side_channel_dim
        team_market_context_dim = self._derived_team_market_feature_count()
        if self.backbone_side_market_context and team_market_context_dim > 0:
            self.backbone_side_market_encoder = nn.Sequential(
                nn.LayerNorm(team_market_context_dim),
                nn.Linear(team_market_context_dim, self.backbone_side_market_hidden),
                nn.GELU(),
                nn.Linear(self.backbone_side_market_hidden, int(d_model)),
            )
        if self.enable_possession_backbone:
            self.possession_head = PossessionHead(
                d_model=int(d_model),
                hidden_dim=int(possession_head_hidden),
                dropout=float(dropout),
                mu_mode=str(possession_mu_mode),
                mu_baseline=float(possession_mu_baseline),
                enable_team_possession_split=bool(enable_team_possession_split_head),
                team_possession_max_delta=float(team_possession_max_delta),
                num_game_features=int(backbone_context_dim),
            )
            self.event_backbone = TeamEventBackbone(
                d_model=int(d_model),
                hidden_dim=int(backbone_hidden),
                dropout=float(dropout),
                num_game_features=int(backbone_context_dim),
                num_team_context_features=4 if self.team_ppp_direct_backbone_context else 0,
                num_advantage_features=3 if self.team_advantage_direct_backbone_context else 0,
            )
            if self.enable_three_pa_share:
                self.three_pa_share_head = ThreePAShareHead(
                    d_model=int(d_model),
                    hidden_dim=int(three_pa_share_hidden),
                    dropout=float(dropout),
                    num_game_features=int(backbone_context_dim),
                    num_team_context_features=4 if self.team_ppp_direct_backbone_context else 0,
                    num_advantage_features=3 if self.team_advantage_direct_backbone_context else 0,
                )

        self.enable_usage_share_head = bool(enable_usage_share_head)
        self.usage_share_head: UsageShareHead | None = None
        if self.enable_usage_share_head:
            self.usage_share_head = UsageShareHead(
                d_model=int(d_model),
                hidden_dim=int(usage_share_head_hidden),
                dropout=float(dropout),
            )
        self.enable_team_points_budget_head = bool(enable_team_points_budget_head)
        self.team_points_budget_parameterization = str(team_points_budget_parameterization).strip().lower()
        self.team_points_budget_to_backbone = bool(team_points_budget_to_backbone)
        self.team_points_budget_latent_hidden = max(1, int(team_points_budget_latent_hidden))
        self.team_points_budget_head: TeamPointsBudgetHead | None = None
        self.backbone_team_points_budget_encoder: nn.Module | None = None
        if self.enable_team_points_budget_head:
            self.team_points_budget_head = TeamPointsBudgetHead(
                d_model=int(d_model),
                hidden_dim=int(team_points_budget_head_hidden),
                dropout=float(dropout),
            )
        if self.team_points_budget_to_backbone:
            self.backbone_team_points_budget_encoder = nn.Sequential(
                nn.LayerNorm(4),
                nn.Linear(4, self.team_points_budget_latent_hidden),
                nn.GELU(),
                nn.Linear(self.team_points_budget_latent_hidden, int(d_model)),
            )
        self.team_opportunity_budget_parameterization = str(team_opportunity_budget_parameterization).strip().lower()
        self.team_opportunity_budget_to_backbone = bool(team_opportunity_budget_to_backbone)
        self.team_opportunity_budget_latent_hidden = max(1, int(team_opportunity_budget_latent_hidden))
        self.team_opportunity_budget_backbone_alpha = float(team_opportunity_budget_backbone_alpha)
        self.backbone_team_opportunity_budget_encoder: nn.Module | None = None
        if self.team_opportunity_budget_to_backbone:
            self.backbone_team_opportunity_budget_encoder = nn.Sequential(
                nn.LayerNorm(4),
                nn.Linear(4, self.team_opportunity_budget_latent_hidden),
                nn.GELU(),
                nn.Linear(self.team_opportunity_budget_latent_hidden, int(d_model)),
            )
        self.enable_team_ast_budget_head = bool(enable_team_ast_budget_head)
        self.team_ast_budget_head: TeamAstBudgetHead | None = None
        if self.enable_team_ast_budget_head:
            self.team_ast_budget_head = TeamAstBudgetHead(
                d_model=int(d_model),
                hidden_dim=int(team_ast_budget_head_hidden),
                dropout=float(dropout),
            )
        self.enable_assist_share_head = bool(enable_assist_share_head)
        self.assist_share_head: AssistShareHead | None = None
        self.assist_share_condition_feature_indices = tuple(int(idx) for idx in assist_share_condition_feature_indices)
        self.num_assist_share_condition_features = int(len(self.assist_share_condition_feature_indices))
        if len(assist_share_condition_feature_mean) != self.num_assist_share_condition_features:
            raise ValueError("assist_share_condition_feature_mean must align with assist_share_condition_feature_indices")
        if len(assist_share_condition_feature_std) != self.num_assist_share_condition_features:
            raise ValueError("assist_share_condition_feature_std must align with assist_share_condition_feature_indices")
        if self.num_assist_share_condition_features > 0:
            self.register_buffer(
                "_assist_share_condition_feature_indices",
                torch.tensor(self.assist_share_condition_feature_indices, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                "_assist_share_condition_feature_mean",
                torch.tensor(assist_share_condition_feature_mean, dtype=torch.float32),
                persistent=False,
            )
            self.register_buffer(
                "_assist_share_condition_feature_std",
                torch.tensor(assist_share_condition_feature_std, dtype=torch.float32),
                persistent=False,
            )
        else:
            self.register_buffer("_assist_share_condition_feature_indices", torch.empty(0, dtype=torch.long), persistent=False)
            self.register_buffer("_assist_share_condition_feature_mean", torch.empty(0, dtype=torch.float32), persistent=False)
            self.register_buffer("_assist_share_condition_feature_std", torch.empty(0, dtype=torch.float32), persistent=False)
        if self.enable_assist_share_head:
            self.assist_share_head = AssistShareHead(
                d_model=int(d_model),
                hidden_dim=int(assist_share_head_hidden),
                condition_dim=self.num_assist_share_condition_features,
                condition_hidden_dim=int(assist_share_condition_hidden),
                dropout=float(dropout),
            )
        self.enable_team_rebound_budget_head = bool(enable_team_rebound_budget_head)
        self.rebound_budget_parameterization = str(rebound_budget_parameterization).strip().lower()
        self.rebound_oreb_rate_cap = float(max(0.0, min(1.0, rebound_oreb_rate_cap)))
        self.rebound_dreb_rate_cap = float(max(0.0, min(1.0, rebound_dreb_rate_cap)))
        self.rebound_dreb_deterministic_discount = float(
            max(0.0, min(1.0, rebound_dreb_deterministic_discount))
        )
        self.team_rebound_budget_head: TeamReboundBudgetHead | None = None
        if self.enable_team_rebound_budget_head:
            self.team_rebound_budget_head = TeamReboundBudgetHead(
                d_model=int(d_model),
                hidden_dim=int(team_rebound_budget_head_hidden),
                dropout=float(dropout),
                budget_parameterization=self.rebound_budget_parameterization,
                oreb_rate_cap=self.rebound_oreb_rate_cap,
                dreb_rate_cap=self.rebound_dreb_rate_cap,
            )
        self.enable_rebound_budget_blend_gate = bool(enable_rebound_budget_blend_gate)
        self.rebound_budget_blend_gate_head: ReboundBudgetBlendGateHead | None = None
        if self.enable_rebound_budget_blend_gate:
            self.rebound_budget_blend_gate_head = ReboundBudgetBlendGateHead(
                d_model=int(d_model),
                hidden_dim=int(rebound_budget_blend_gate_hidden),
                dropout=float(dropout),
                init_alpha=float(rebound_budget_blend_gate_init_alpha),
            )
        self.enable_rebound_share_head = bool(enable_rebound_share_head)
        self.rebound_share_condition_feature_indices = tuple(int(idx) for idx in rebound_share_condition_feature_indices)
        self.num_rebound_share_condition_features = int(len(self.rebound_share_condition_feature_indices))
        if len(rebound_share_condition_feature_mean) != self.num_rebound_share_condition_features:
            raise ValueError("rebound_share_condition_feature_mean must align with rebound_share_condition_feature_indices")
        if len(rebound_share_condition_feature_std) != self.num_rebound_share_condition_features:
            raise ValueError("rebound_share_condition_feature_std must align with rebound_share_condition_feature_indices")
        if self.num_rebound_share_condition_features > 0:
            self.register_buffer(
                "_rebound_share_condition_feature_indices",
                torch.tensor(self.rebound_share_condition_feature_indices, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                "_rebound_share_condition_feature_mean",
                torch.tensor(rebound_share_condition_feature_mean, dtype=torch.float32),
                persistent=False,
            )
            self.register_buffer(
                "_rebound_share_condition_feature_std",
                torch.tensor(rebound_share_condition_feature_std, dtype=torch.float32),
                persistent=False,
            )
        else:
            self.register_buffer("_rebound_share_condition_feature_indices", torch.empty(0, dtype=torch.long), persistent=False)
            self.register_buffer("_rebound_share_condition_feature_mean", torch.empty(0, dtype=torch.float32), persistent=False)
            self.register_buffer("_rebound_share_condition_feature_std", torch.empty(0, dtype=torch.float32), persistent=False)
        self.rebound_share_head: ReboundShareHead | None = None
        if self.enable_rebound_share_head:
            self.rebound_share_head = ReboundShareHead(
                d_model=int(d_model),
                hidden_dim=int(rebound_share_head_hidden),
                condition_dim=self.num_rebound_share_condition_features,
                condition_hidden_dim=int(rebound_share_condition_hidden),
                dropout=float(dropout),
            )
        self.enable_ast_blend_gate = bool(enable_ast_blend_gate)
        self.ast_blend_gate_head: AstBlendGateHead | None = None
        if self.enable_ast_blend_gate:
            self.ast_blend_gate_head = AstBlendGateHead(
                d_model=int(d_model),
                hidden_dim=int(ast_blend_gate_hidden),
                condition_dim=self.num_assist_share_condition_features,
                condition_hidden_dim=int(assist_share_condition_hidden),
                dropout=float(dropout),
                init_alpha=float(ast_blend_gate_init_alpha),
            )

        token_type_ids = [0, 1, *([2] * MAX_PLAYERS_PER_TEAM), 1, *([2] * MAX_PLAYERS_PER_TEAM)]
        side_ids = [2, 0, *([0] * MAX_PLAYERS_PER_TEAM), 1, *([1] * MAX_PLAYERS_PER_TEAM)]
        team_index = [0] * MAX_PLAYERS_PER_TEAM + [1] * MAX_PLAYERS_PER_TEAM
        self.register_buffer("_token_type_ids", torch.tensor(token_type_ids, dtype=torch.long), persistent=False)
        self.register_buffer("_side_ids", torch.tensor(side_ids, dtype=torch.long), persistent=False)
        self.register_buffer("_player_team_index", torch.tensor(team_index, dtype=torch.long), persistent=False)

    def _build_sequence(
        self,
        player_features: torch.Tensor,
        game_features: torch.Tensor | None,
        team_features: torch.Tensor | None,
    ) -> torch.Tensor:
        if player_features.ndim != 4:
            raise ValueError("player_features must have shape (B,2,15,F)")
        if player_features.shape[1] != 2 or player_features.shape[2] != MAX_PLAYERS_PER_TEAM:
            raise ValueError(f"player_features must have shape (B,2,{MAX_PLAYERS_PER_TEAM},F)")

        bsz = player_features.shape[0]
        p_flat = torch.cat([player_features[:, 0], player_features[:, 1]], dim=1)
        p_tok = self.player_proj(p_flat)
        p_home = p_tok[:, :MAX_PLAYERS_PER_TEAM]
        p_away = p_tok[:, MAX_PLAYERS_PER_TEAM:]

        g_tok = self.game_token.unsqueeze(0).unsqueeze(1).expand(bsz, 1, -1)
        if self.game_proj is not None:
            if game_features is None:
                raise ValueError("game_features is required when num_game_features > 0")
            if game_features.ndim != 2 or game_features.shape != (bsz, self.num_game_features):
                raise ValueError("game_features must have shape (B,G)")
            g_tok = g_tok + self.game_proj(game_features).unsqueeze(1)

        t_base = self.team_tokens.unsqueeze(0).expand(bsz, -1, -1)
        if self.team_proj is not None:
            if team_features is None:
                raise ValueError("team_features is required when num_team_features > 0")
            if team_features.ndim != 3 or team_features.shape != (bsz, 2, self.num_team_features):
                raise ValueError("team_features must have shape (B,2,T)")
            t_base = t_base + self.team_proj(team_features)

        seq = torch.cat(
            [
                g_tok,
                t_base[:, 0:1, :],
                p_home,
                t_base[:, 1:2, :],
                p_away,
            ],
            dim=1,
        )
        if seq.shape[1] != 33:
            raise RuntimeError(f"Unexpected sequence length: {seq.shape[1]}")

        type_emb = self.token_type_embedding(self._token_type_ids).unsqueeze(0)
        side_emb = self.side_embedding(self._side_ids).unsqueeze(0)
        seq = seq + type_emb + side_emb
        return self.dropout(seq)

    def _derived_env_feature_count(self) -> int:
        names = set(self.backbone_env_feature_names)
        count = 0
        if "team_pace_szn" in names:
            count += 2
        if "team_off_rtg_szn" in names and "opp_def_rtg_szn" in names:
            count += 2
        if self.backbone_env_enrich_features:
            game_names = set(self.game_feature_names)
            if "team_pace_szn" in names:
                count += 1
            if "team_off_rtg_szn" in names and "team_def_rtg_szn" in names:
                count += 2
            if "vegas_total" in game_names and "vegas_spread" in game_names:
                count += 3
            if "vegas_total" in game_names and "estimated_possessions" in game_names:
                count += 1
        return count

    def _derived_team_market_feature_count(self) -> int:
        if not bool(self.backbone_side_market_context):
            return 0
        game_names = set(self.game_feature_names)
        if "vegas_total" not in game_names or "vegas_spread" not in game_names:
            return 0
        count = 4
        if "estimated_possessions" in game_names:
            count += 2
        return count

    def _build_backbone_environment_context(
        self,
        player_features: torch.Tensor,
        player_valid_mask: torch.Tensor,
        game_features: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if self.num_game_features <= 0 and self.num_backbone_env_features <= 0:
            return None

        pieces: list[torch.Tensor] = []
        derived: list[torch.Tensor] = []
        if self.num_game_features > 0:
            if game_features is None:
                raise ValueError("game_features is required when num_game_features > 0")
            pieces.append(game_features)
            if self.backbone_env_enrich_features:
                game_name_to_idx = {name: i for i, name in enumerate(self.game_feature_names)}
                total_idx = game_name_to_idx.get("vegas_total")
                spread_idx = game_name_to_idx.get("vegas_spread")
                poss_idx = game_name_to_idx.get("estimated_possessions")
                if total_idx is not None and spread_idx is not None:
                    total = game_features[:, total_idx : total_idx + 1]
                    spread = game_features[:, spread_idx : spread_idx + 1]
                    derived.append(spread.abs())
                    derived.append(0.5 * (total - spread))  # home implied total
                    derived.append(0.5 * (total + spread))  # away implied total
                if total_idx is not None and poss_idx is not None:
                    total = game_features[:, total_idx : total_idx + 1]
                    poss = game_features[:, poss_idx : poss_idx + 1].clamp_min(1.0)
                    derived.append(total / poss)

        if self.num_backbone_env_features > 0:
            idx = torch.as_tensor(
                self.backbone_env_feature_indices,
                device=player_features.device,
                dtype=torch.long,
            )
            env_feats = player_features.index_select(dim=-1, index=idx)
            valid = player_valid_mask.to(dtype=player_features.dtype).unsqueeze(-1)
            denom = valid.sum(dim=2).clamp_min(1.0)
            team_env = (env_feats * valid).sum(dim=2) / denom
            pieces.append(team_env.reshape(player_features.shape[0], -1))

            names = {name: i for i, name in enumerate(self.backbone_env_feature_names)}
            if "team_pace_szn" in names:
                pace = team_env[:, :, names["team_pace_szn"]]
                derived.append(pace.mean(dim=1, keepdim=True))
                derived.append((pace[:, 0] - pace[:, 1]).unsqueeze(1))
                if self.backbone_env_enrich_features:
                    derived.append((pace[:, 0] - pace[:, 1]).abs().unsqueeze(1))
            if "team_off_rtg_szn" in names and "opp_def_rtg_szn" in names:
                off = team_env[:, :, names["team_off_rtg_szn"]]
                opp_def = team_env[:, :, names["opp_def_rtg_szn"]]
                derived.append((off[:, 0] - opp_def[:, 0]).unsqueeze(1))
                derived.append((off[:, 1] - opp_def[:, 1]).unsqueeze(1))
            if self.backbone_env_enrich_features and "team_off_rtg_szn" in names and "team_def_rtg_szn" in names:
                off = team_env[:, :, names["team_off_rtg_szn"]]
                team_def = team_env[:, :, names["team_def_rtg_szn"]]
                derived.append((off[:, 1] - team_def[:, 0]).unsqueeze(1))
                derived.append((off[:, 0] - team_def[:, 1]).unsqueeze(1))

        if derived:
            pieces.append(torch.cat(derived, dim=1))

        if not pieces:
            return None
        raw_context = torch.cat(pieces, dim=1)
        if self.backbone_env_adapter is not None:
            return self.backbone_env_adapter(raw_context)
        return raw_context

    def _build_backbone_team_market_context(
        self,
        game_features: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if not bool(self.backbone_side_market_context):
            return None
        if self.num_game_features <= 0 or game_features is None:
            return None
        game_name_to_idx = {name: i for i, name in enumerate(self.game_feature_names)}
        total_idx = game_name_to_idx.get("vegas_total")
        spread_idx = game_name_to_idx.get("vegas_spread")
        if total_idx is None or spread_idx is None:
            return None

        total = game_features[:, total_idx : total_idx + 1]
        spread = game_features[:, spread_idx : spread_idx + 1]
        home_total = 0.5 * (total - spread)
        away_total = 0.5 * (total + spread)
        abs_spread = spread.abs()
        home_margin = home_total - away_total
        away_margin = away_total - home_total

        home_feats = [home_total, away_total, home_margin, abs_spread]
        away_feats = [away_total, home_total, away_margin, abs_spread]

        poss_idx = game_name_to_idx.get("estimated_possessions")
        if poss_idx is not None:
            poss = game_features[:, poss_idx : poss_idx + 1].clamp_min(1.0)
            home_ppp = home_total / poss
            away_ppp = away_total / poss
            home_feats.extend([home_ppp, away_ppp])
            away_feats.extend([away_ppp, home_ppp])

        return torch.stack(
            [
                torch.cat(home_feats, dim=1),
                torch.cat(away_feats, dim=1),
            ],
            dim=1,
        )

    def _build_backbone_team_points_budget_context(
        self,
        team_points_budget: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if not self.team_points_budget_to_backbone or team_points_budget is None:
            return None
        if team_points_budget.ndim != 2 or int(team_points_budget.shape[1]) != 2:
            raise ValueError("team_points_budget must have shape (B,2)")

        home_total = team_points_budget[:, 0:1]
        away_total = team_points_budget[:, 1:2]
        home_margin = home_total - away_total
        away_margin = away_total - home_total
        return torch.stack(
            [
                torch.cat([home_total, away_total, home_margin, home_margin.abs()], dim=1),
                torch.cat([away_total, home_total, away_margin, away_margin.abs()], dim=1),
            ],
            dim=1,
        )

    def _resolve_team_points_budget(
        self,
        *,
        game_features: torch.Tensor | None,
        team_points_budget_out: TeamPointsBudgetHeadOutputs | None,
        team_ppp_out: TeamPPPHeadOutputs | None = None,
    ) -> torch.Tensor | None:
        mode = str(getattr(self, "team_points_budget_parameterization", "absolute")).strip().lower()
        if mode == "market_implied":
            if self.num_game_features <= 0 or game_features is None:
                return None
            game_name_to_idx = {name: i for i, name in enumerate(self.game_feature_names)}
            total_idx = game_name_to_idx.get("vegas_total")
            spread_idx = game_name_to_idx.get("vegas_spread")
            if total_idx is None or spread_idx is None:
                return None
            total = game_features[:, total_idx : total_idx + 1]
            spread = game_features[:, spread_idx : spread_idx + 1]
            home_total = 0.5 * (total - spread)
            away_total = 0.5 * (total + spread)
            return torch.cat([home_total, away_total], dim=1)
        if mode == "team_ppp_implied":
            if team_ppp_out is None:
                return None
            if self.num_game_features <= 0 or game_features is None:
                return None
            game_name_to_idx = {name: i for i, name in enumerate(self.game_feature_names)}
            poss_idx = game_name_to_idx.get("estimated_possessions")
            if poss_idx is None:
                return None
            poss = game_features[:, poss_idx : poss_idx + 1].expand(-1, 2)
            return team_ppp_out.team_ppp * poss.clamp_min(1.0)
        if team_points_budget_out is None:
            return None
        return team_points_budget_out.team_points

    def _resolve_team_opportunity_share(
        self,
        *,
        game_features: torch.Tensor | None,
    ) -> torch.Tensor | None:
        mode = str(getattr(self, "team_opportunity_budget_parameterization", "absolute")).strip().lower()
        if mode != "market_implied_share":
            return None
        if self.num_game_features <= 0 or game_features is None:
            return None
        game_name_to_idx = {name: i for i, name in enumerate(self.game_feature_names)}
        total_idx = game_name_to_idx.get("vegas_total")
        spread_idx = game_name_to_idx.get("vegas_spread")
        if total_idx is None or spread_idx is None:
            return None
        total = game_features[:, total_idx : total_idx + 1].clamp_min(1e-6)
        spread = game_features[:, spread_idx : spread_idx + 1]
        home_total = 0.5 * (total - spread)
        away_total = 0.5 * (total + spread)
        home_share = torch.clamp(home_total / total, min=0.0, max=1.0)
        away_share = torch.clamp(away_total / total, min=0.0, max=1.0)
        share_sum = (home_share + away_share).clamp_min(1e-6)
        return torch.cat([home_share / share_sum, away_share / share_sum], dim=1)

    def _build_backbone_team_opportunity_budget_context(
        self,
        team_opportunity_share: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if not self.team_opportunity_budget_to_backbone or team_opportunity_share is None:
            return None
        if team_opportunity_share.ndim != 2 or int(team_opportunity_share.shape[1]) != 2:
            raise ValueError("team_opportunity_share must have shape (B,2)")

        home_share = team_opportunity_share[:, 0:1]
        away_share = team_opportunity_share[:, 1:2]
        home_gap = home_share - away_share
        away_gap = away_share - home_share
        abs_gap = home_gap.abs()
        return torch.stack(
            [
                torch.cat([home_share, away_share, home_gap, abs_gap], dim=1),
                torch.cat([away_share, home_share, away_gap, abs_gap], dim=1),
            ],
            dim=1,
        )

    def _build_team_ppp_context(
        self,
        team_ppp: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if team_ppp is None:
            return None
        if team_ppp.ndim != 2 or int(team_ppp.shape[1]) != 2:
            raise ValueError("team_ppp must have shape (B,2)")

        home_ppp = team_ppp[:, 0:1]
        away_ppp = team_ppp[:, 1:2]
        home_gap = home_ppp - away_ppp
        away_gap = away_ppp - home_ppp
        abs_gap = home_gap.abs()
        return torch.stack(
            [
                torch.cat([home_ppp, away_ppp, home_gap, abs_gap], dim=1),
                torch.cat([away_ppp, home_ppp, away_gap, abs_gap], dim=1),
            ],
            dim=1,
        )

    def _build_team_advantage_context(
        self,
        team_advantage: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if team_advantage is None:
            return None
        if team_advantage.ndim != 1:
            raise ValueError("team_advantage must have shape (B,)")
        home_adv = team_advantage.unsqueeze(1)
        away_adv = -home_adv
        abs_adv = home_adv.abs()
        return torch.stack(
            [
                torch.cat([home_adv, away_adv, abs_adv], dim=1),
                torch.cat([away_adv, home_adv, abs_adv], dim=1),
            ],
            dim=1,
        )

    def _build_env_side_channel_context(
        self,
        player_features: torch.Tensor,
        player_valid_mask: torch.Tensor,
        game_features: torch.Tensor | None,
    ) -> torch.Tensor | None:
        raw_context = self._build_backbone_environment_context(
            player_features=player_features,
            player_valid_mask=player_valid_mask,
            game_features=game_features,
        )
        if raw_context is None:
            return None
        if self.env_side_channel_encoder is not None:
            return self.env_side_channel_encoder(raw_context)
        return raw_context

    def _extract_assist_share_condition_features(self, player_features: torch.Tensor) -> torch.Tensor | None:
        if self.num_assist_share_condition_features <= 0:
            return None
        if player_features.ndim != 4:
            raise ValueError("player_features must have shape (B,2,15,F)")
        cond = torch.index_select(
            player_features,
            dim=-1,
            index=self._assist_share_condition_feature_indices.to(device=player_features.device),
        )
        mean = self._assist_share_condition_feature_mean.to(device=player_features.device, dtype=player_features.dtype)
        std = self._assist_share_condition_feature_std.to(device=player_features.device, dtype=player_features.dtype)
        std = torch.where(std.abs() > 1e-6, std, torch.ones_like(std))
        raw = (cond * std.view(1, 1, 1, -1)) + mean.view(1, 1, 1, -1)
        return torch.cat([raw[:, 0], raw[:, 1]], dim=1)

    def _extract_rebound_share_condition_features(self, player_features: torch.Tensor) -> torch.Tensor | None:
        if self.num_rebound_share_condition_features <= 0:
            return None
        if player_features.ndim != 4:
            raise ValueError("player_features must have shape (B,2,15,F)")
        cond = torch.index_select(
            player_features,
            dim=-1,
            index=self._rebound_share_condition_feature_indices.to(device=player_features.device),
        )
        mean = self._rebound_share_condition_feature_mean.to(device=player_features.device, dtype=player_features.dtype)
        std = self._rebound_share_condition_feature_std.to(device=player_features.device, dtype=player_features.dtype)
        std = torch.where(std.abs() > 1e-6, std, torch.ones_like(std))
        raw = (cond * std.view(1, 1, 1, -1)) + mean.view(1, 1, 1, -1)
        return torch.cat([raw[:, 0], raw[:, 1]], dim=1)

    def forward(
        self,
        player_features: torch.Tensor,
        player_valid_mask: torch.Tensor,
        *,
        game_features: torch.Tensor | None = None,
        team_features: torch.Tensor | None = None,
        efficiency_sidecar_features: torch.Tensor | None = None,
        sample_active: bool = False,
        active_temperature: float = 1.0,
        target_counts: torch.Tensor | None = None,
        use_target_counts: bool = False,
        target_active_mask: torch.Tensor | None = None,
        use_target_active_mask: bool = False,
        minutes_use_target_active: bool = False,
        minutes_teacher_forcing_prob: float = 1.0,
        minutes_teacher_forcing_mode: str = "batch",
        starter_hint_mask: torch.Tensor | None = None,
        starter_promotion_candidate_mask: torch.Tensor | None = None,
        run_flow: bool = False,
        flow_targets: torch.Tensor | None = None,
        flow_observed_mask: torch.Tensor | None = None,
        flow_minutes_target: torch.Tensor | None = None,
        flow_minutes_teacher_forcing_prob: float = 1.0,
        flow_minutes_teacher_forcing_mode: str = "batch",
        sample_backbone: bool = False,
        detach_backbone: bool = True,
    ) -> GameTransformerV2Outputs:
        """Forward pass for one batch of full-game tensors."""

        if player_valid_mask.ndim != 3:
            raise ValueError("player_valid_mask must have shape (B,2,15)")
        if player_valid_mask.shape[:3] != player_features.shape[:3]:
            raise ValueError("player_valid_mask must align with player_features")
        if self.num_efficiency_sidecar_features > 0:
            if efficiency_sidecar_features is None:
                raise ValueError(
                    "efficiency_sidecar_features is required when num_efficiency_sidecar_features > 0"
                )
            expected_shape = (
                player_features.shape[0],
                2,
                MAX_PLAYERS_PER_TEAM,
                self.num_efficiency_sidecar_features,
            )
            if tuple(efficiency_sidecar_features.shape) != expected_shape:
                raise ValueError(
                    "efficiency_sidecar_features must have shape "
                    f"{expected_shape}, got {tuple(efficiency_sidecar_features.shape)}"
                )

        bsz = player_features.shape[0]
        valid_home = player_valid_mask[:, 0].to(dtype=torch.bool)
        valid_away = player_valid_mask[:, 1].to(dtype=torch.bool)
        valid_flat = torch.cat([valid_home, valid_away], dim=1)
        player_team_index = self._player_team_index.unsqueeze(0).expand(bsz, -1)

        seq = self._build_sequence(player_features, game_features, team_features)
        seq_valid = torch.cat(
            [
                torch.ones((bsz, 1), dtype=torch.bool, device=seq.device),
                torch.ones((bsz, 1), dtype=torch.bool, device=seq.device),
                valid_home,
                torch.ones((bsz, 1), dtype=torch.bool, device=seq.device),
                valid_away,
            ],
            dim=1,
        )
        encoded = self.encoder(seq, src_key_padding_mask=~seq_valid)
        encoded = self.final_norm(encoded)

        game_state = encoded[:, 0, :]
        team_states = torch.stack([encoded[:, 1, :], encoded[:, 17, :]], dim=1)
        player_states = torch.cat([encoded[:, 2:17, :], encoded[:, 18:33, :]], dim=1)

        target_active_flat: torch.Tensor | None = None
        if target_active_mask is not None:
            if target_active_mask.ndim != 3 or target_active_mask.shape != player_valid_mask.shape:
                raise ValueError("target_active_mask must have shape (B,2,15)")
            target_active_flat = torch.cat(
                [target_active_mask[:, 0].to(dtype=torch.bool), target_active_mask[:, 1].to(dtype=torch.bool)],
                dim=1,
            )

        starter_hint_flat: torch.Tensor | None = None
        if starter_hint_mask is not None:
            if starter_hint_mask.ndim != 3 or starter_hint_mask.shape != player_valid_mask.shape:
                raise ValueError("starter_hint_mask must have shape (B,2,15)")
            starter_hint_flat = torch.cat(
                [starter_hint_mask[:, 0].to(dtype=torch.bool), starter_hint_mask[:, 1].to(dtype=torch.bool)],
                dim=1,
            )
        starter_promotion_candidate_flat: torch.Tensor | None = None
        if starter_promotion_candidate_mask is not None:
            if (
                starter_promotion_candidate_mask.ndim != 3
                or starter_promotion_candidate_mask.shape != player_valid_mask.shape
            ):
                raise ValueError("starter_promotion_candidate_mask must have shape (B,2,15)")
            starter_promotion_candidate_flat = torch.cat(
                [
                    starter_promotion_candidate_mask[:, 0].to(dtype=torch.bool),
                    starter_promotion_candidate_mask[:, 1].to(dtype=torch.bool),
                ],
                dim=1,
            )

        active_out = self.active_head(
            player_states,
            team_states,
            player_team_index,
            valid_flat,
            sample=bool(sample_active),
            temperature=float(active_temperature),
            target_counts=target_counts,
            use_target_counts=bool(use_target_counts),
            target_active_mask=target_active_flat,
            use_target_active_mask=bool(use_target_active_mask),
        )

        minutes_active_mask = _resolve_minutes_active_mask(
            active_out.active_mask,
            target_active_mask=target_active_flat,
            player_team_index=player_team_index,
            minutes_use_target_active=bool(minutes_use_target_active),
            minutes_teacher_forcing_prob=float(minutes_teacher_forcing_prob),
            minutes_teacher_forcing_mode=str(minutes_teacher_forcing_mode),
        )

        minutes_out = self.minutes_head(
            player_states,
            team_states,
            player_team_index,
            valid_flat,
            minutes_active_mask,
            starter_hint_mask=starter_hint_flat,
            starter_promotion_candidate_mask=starter_promotion_candidate_flat,
        )

        flow_out: JointGameFlowOutputs | None = None
        env_side_channel_context: torch.Tensor | None = None
        if self.enable_env_side_channel:
            env_side_channel_context = self._build_env_side_channel_context(
                player_features=player_features,
                player_valid_mask=player_valid_mask,
                game_features=game_features,
            )
        if bool(run_flow) or flow_targets is not None:
            if flow_targets is None:
                raise ValueError("run_flow=True requires flow_targets")
            if flow_targets.ndim != 4:
                raise ValueError("flow_targets must have shape (B,2,15,S)")
            if flow_targets.shape[:3] != player_features.shape[:3]:
                raise ValueError("flow_targets must align with player_features first 3 dims")
            if flow_targets.shape[3] != self.num_flow_stats:
                raise ValueError(
                    f"flow_targets stat dim mismatch: expected {self.num_flow_stats}, got {flow_targets.shape[3]}"
                )
            flow_flat = torch.cat([flow_targets[:, 0], flow_targets[:, 1]], dim=1)

            flow_obs_flat: torch.Tensor | None = None
            if flow_observed_mask is not None:
                if flow_observed_mask.shape != flow_targets.shape:
                    raise ValueError("flow_observed_mask must match flow_targets shape")
                flow_obs_flat = torch.cat(
                    [
                        flow_observed_mask[:, 0].to(dtype=torch.bool),
                        flow_observed_mask[:, 1].to(dtype=torch.bool),
                    ],
                    dim=1,
                )
            flow_minutes_flat: torch.Tensor | None = None
            if bool(self.flow_use_minutes_conditioning):
                target_minutes_flat: torch.Tensor | None = None
                if flow_minutes_target is not None:
                    if flow_minutes_target.shape != player_valid_mask.shape:
                        raise ValueError("flow_minutes_target must have shape (B,2,15)")
                    target_minutes_flat = torch.cat(
                        [flow_minutes_target[:, 0], flow_minutes_target[:, 1]],
                        dim=1,
                    )
                flow_minutes_flat = _resolve_flow_conditioning_minutes(
                    minutes_out.minutes,
                    target_minutes=target_minutes_flat,
                    player_team_index=player_team_index,
                    teacher_forcing_prob=float(flow_minutes_teacher_forcing_prob),
                    teacher_forcing_mode=str(flow_minutes_teacher_forcing_mode),
                )

            flow_out = self.flow_head(
                flow_flat,
                player_states=player_states,
                team_states=team_states,
                game_state=game_state,
                player_team_index=player_team_index,
                valid_mask=valid_flat,
                observed_mask=flow_obs_flat,
                minutes_context=flow_minutes_flat,
                env_context=env_side_channel_context,
            )

        team_ppp_out: TeamPPPHeadOutputs | None = None
        if self.enable_team_ppp_head and self.team_ppp_head is not None:
            team_ppp_out = self.team_ppp_head(
                team_states=team_states,
                game_state=game_state,
            )
        team_advantage_out: TeamAdvantageHeadOutputs | None = None
        if self.enable_team_advantage_head and self.team_advantage_head is not None:
            team_advantage_out = self.team_advantage_head(
                team_states=team_states,
                game_state=game_state,
                sample=bool(sample_backbone),
            )
        team_ppp_context = self._build_team_ppp_context(
            team_ppp_out.team_ppp if team_ppp_out is not None else None
        )
        team_advantage_context = self._build_team_advantage_context(
            (
                team_advantage_out.sampled_advantage
                if team_advantage_out is not None and team_advantage_out.sampled_advantage is not None
                else (team_advantage_out.mu if team_advantage_out is not None else None)
            )
        )

        efficiency_out: EfficiencyHeadOutputs | None = None
        if self.enable_efficiency_head and self.efficiency_head is not None:
            efficiency_player_states = player_states
            if self.efficiency_player_sidecar_encoder is not None and efficiency_sidecar_features is not None:
                sidecar_flat = torch.cat([efficiency_sidecar_features[:, 0], efficiency_sidecar_features[:, 1]], dim=1)
                sidecar_encoded = self.efficiency_player_sidecar_encoder(
                    sidecar_flat.to(device=player_states.device, dtype=player_states.dtype)
                )
                sidecar_encoded = sidecar_encoded * valid_flat.to(dtype=sidecar_encoded.dtype).unsqueeze(-1)
                efficiency_player_states = efficiency_player_states + float(self.efficiency_sidecar_alpha) * sidecar_encoded
            efficiency_team_states = team_states
            efficiency_team_market_context = self._build_backbone_team_market_context(game_features)
            if self.efficiency_team_market_encoder is not None and efficiency_team_market_context is not None:
                efficiency_team_states = efficiency_team_states + float(self.efficiency_market_alpha) * (
                    self.efficiency_team_market_encoder(
                        efficiency_team_market_context.to(
                            device=efficiency_team_states.device,
                            dtype=efficiency_team_states.dtype,
                        )
                    )
                )
            if self.efficiency_team_ppp_encoder is not None and team_ppp_context is not None:
                efficiency_team_states = efficiency_team_states + float(self.team_ppp_efficiency_alpha) * (
                    self.efficiency_team_ppp_encoder(
                        team_ppp_context.to(
                            device=efficiency_team_states.device,
                            dtype=efficiency_team_states.dtype,
                        )
                    )
                )
            efficiency_out = self.efficiency_head(
                player_states=efficiency_player_states,
                team_states=efficiency_team_states,
                game_state=game_state,
                player_team_index=player_team_index,
                valid_mask=valid_flat,
                team_context=(
                    team_ppp_context.to(device=player_states.device, dtype=player_states.dtype)
                    if self.team_ppp_direct_efficiency_context and team_ppp_context is not None
                    else None
                ),
            )

        # Possession-coupled event backbone (runs when backbone is enabled)
        poss_out: PossessionHeadOutputs | None = None
        backbone_out: TeamEventBackboneOutputs | None = None
        team_points_budget_out: TeamPointsBudgetHeadOutputs | None = None
        if self.enable_team_points_budget_head and self.team_points_budget_head is not None:
            team_points_budget_out = self.team_points_budget_head(
                team_states=team_states,
                game_state=game_state,
            )
        if self.enable_possession_backbone and self.possession_head is not None and self.event_backbone is not None:
            backbone_env_context = (
                env_side_channel_context
                if env_side_channel_context is not None
                else self._build_backbone_environment_context(
                    player_features=player_features,
                    player_valid_mask=player_valid_mask,
                    game_features=game_features,
                )
            )
            backbone_team_states = team_states
            backbone_team_market_context = self._build_backbone_team_market_context(game_features)
            if self.backbone_side_market_encoder is not None and backbone_team_market_context is not None:
                backbone_team_states = backbone_team_states + self.backbone_side_market_encoder(
                    backbone_team_market_context.to(
                        device=backbone_team_states.device,
                        dtype=backbone_team_states.dtype,
                    )
                )
            if self.backbone_team_ppp_encoder is not None and team_ppp_context is not None:
                backbone_team_states = backbone_team_states + float(self.team_ppp_backbone_alpha) * (
                    self.backbone_team_ppp_encoder(
                        team_ppp_context.to(
                            device=backbone_team_states.device,
                            dtype=backbone_team_states.dtype,
                        )
                    )
                )
            resolved_team_points_budget = self._resolve_team_points_budget(
                game_features=game_features,
                team_points_budget_out=team_points_budget_out,
                team_ppp_out=team_ppp_out,
            )
            resolved_team_opportunity_share = self._resolve_team_opportunity_share(
                game_features=game_features,
            )
            backbone_team_points_budget_context = self._build_backbone_team_points_budget_context(
                resolved_team_points_budget
            )
            if (
                self.backbone_team_points_budget_encoder is not None
                and backbone_team_points_budget_context is not None
            ):
                backbone_team_states = backbone_team_states + self.backbone_team_points_budget_encoder(
                    backbone_team_points_budget_context.to(
                        device=backbone_team_states.device,
                        dtype=backbone_team_states.dtype,
                    )
                )
            backbone_team_opportunity_context = self._build_backbone_team_opportunity_budget_context(
                resolved_team_opportunity_share
            )
            if (
                self.backbone_team_opportunity_budget_encoder is not None
                and backbone_team_opportunity_context is not None
            ):
                backbone_team_states = backbone_team_states + float(self.team_opportunity_budget_backbone_alpha) * (
                    self.backbone_team_opportunity_budget_encoder(
                        backbone_team_opportunity_context.to(
                            device=backbone_team_states.device,
                            dtype=backbone_team_states.dtype,
                        )
                    )
                )
            # Optionally detach encoder outputs to prevent backbone gradients
            # from destabilizing the flow head during phase2 warmup.
            # Set detach_backbone=False once the flow head is stable.
            if detach_backbone:
                game_state_bb = game_state.detach()
                team_states_bb = backbone_team_states.detach()
            else:
                game_state_bb = game_state
                team_states_bb = backbone_team_states
            poss_out = self.possession_head(
                game_state_bb, sample=bool(sample_backbone), game_features=backbone_env_context,
            )
            if getattr(poss_out, "team_poss", None) is not None:
                poss_for_backbone = poss_out.team_poss
            elif poss_out.sampled_poss is not None:
                poss_for_backbone = poss_out.sampled_poss
            else:
                # During training (no sampling), use the predicted mean
                poss_for_backbone = poss_out.mu
            backbone_out = self.event_backbone(
                team_states_bb, game_state_bb, poss_for_backbone,
                sample=bool(sample_backbone),
                game_features=backbone_env_context,
                team_context=(
                    team_ppp_context.to(device=team_states_bb.device, dtype=team_states_bb.dtype)
                    if self.team_ppp_direct_backbone_context and team_ppp_context is not None
                    else None
                ),
                advantage_context=(
                    team_advantage_context.to(device=team_states_bb.device, dtype=team_states_bb.dtype)
                    if self.team_advantage_direct_backbone_context and team_advantage_context is not None
                    else None
                ),
            )
            # Optional shot-mix latent
            if self.three_pa_share_head is not None and backbone_out is not None:
                three_pa_share = self.three_pa_share_head(
                    team_states_bb, game_state_bb, backbone_out.fga,
                    sample=bool(sample_backbone),
                    game_features=backbone_env_context,
                    team_context=(
                        team_ppp_context.to(device=team_states_bb.device, dtype=team_states_bb.dtype)
                        if self.team_ppp_direct_backbone_context and team_ppp_context is not None
                        else None
                    ),
                    advantage_context=(
                        team_advantage_context.to(device=team_states_bb.device, dtype=team_states_bb.dtype)
                        if self.team_advantage_direct_backbone_context and team_advantage_context is not None
                        else None
                    ),
                )
                backbone_out = TeamEventBackboneOutputs(
                    fga=backbone_out.fga,
                    fta=backbone_out.fta,
                    tov=backbone_out.tov,
                    oreb=backbone_out.oreb,
                    three_pa_share=three_pa_share,
                    poss_used=backbone_out.poss_used,
                )

        usage_share_out: UsageShareHeadOutputs | None = None
        if self.enable_usage_share_head and self.usage_share_head is not None:
            usage_share_out = self.usage_share_head(
                player_states=player_states,
                team_states=team_states,
                game_state=game_state,
                player_team_index=player_team_index,
            )
        team_ast_budget_out: TeamAstBudgetHeadOutputs | None = None
        if self.enable_team_ast_budget_head and self.team_ast_budget_head is not None:
            team_ast_budget_out = self.team_ast_budget_head(
                team_states=team_states,
                game_state=game_state,
            )
        assist_share_out: AssistShareHeadOutputs | None = None
        if self.enable_assist_share_head and self.assist_share_head is not None:
            assist_share_condition = self._extract_assist_share_condition_features(player_features)
            assist_share_out = self.assist_share_head(
                player_states=player_states,
                team_states=team_states,
                game_state=game_state,
                player_team_index=player_team_index,
                condition_features=assist_share_condition,
            )
        team_rebound_budget_out: TeamReboundBudgetHeadOutputs | None = None
        if self.enable_team_rebound_budget_head and self.team_rebound_budget_head is not None:
            team_rebound_budget_out = self.team_rebound_budget_head(
                team_states=team_states,
                game_state=game_state,
            )
        rebound_budget_blend_gate_out: ReboundBudgetBlendGateHeadOutputs | None = None
        if self.enable_rebound_budget_blend_gate and self.rebound_budget_blend_gate_head is not None:
            rebound_budget_blend_gate_out = self.rebound_budget_blend_gate_head(
                team_states=team_states,
                game_state=game_state,
            )
        rebound_share_out: ReboundShareHeadOutputs | None = None
        if self.enable_rebound_share_head and self.rebound_share_head is not None:
            rebound_share_condition = self._extract_rebound_share_condition_features(player_features)
            rebound_share_out = self.rebound_share_head(
                player_states=player_states,
                team_states=team_states,
                game_state=game_state,
                player_team_index=player_team_index,
                condition_features=rebound_share_condition,
            )
        ast_blend_gate_out: AstBlendGateHeadOutputs | None = None
        if self.enable_ast_blend_gate and self.ast_blend_gate_head is not None:
            assist_share_condition = self._extract_assist_share_condition_features(player_features)
            ast_blend_gate_out = self.ast_blend_gate_head(
                player_states=player_states,
                team_states=team_states,
                game_state=game_state,
                player_team_index=player_team_index,
                condition_features=assist_share_condition,
            )

        return GameTransformerV2Outputs(
            game_state=game_state,
            team_states=team_states,
            player_states=player_states,
            player_valid_mask=valid_flat,
            player_team_index=player_team_index,
            active=active_out,
            minutes=minutes_out,
            flow=flow_out,
            efficiency=efficiency_out,
            possession=poss_out,
            backbone=backbone_out,
            usage_share=usage_share_out,
            team_ppp=team_ppp_out,
            team_advantage=team_advantage_out,
            team_points_budget=team_points_budget_out,
            team_ast_budget=team_ast_budget_out,
            assist_share=assist_share_out,
            team_rebound_budget=team_rebound_budget_out,
            rebound_budget_blend_gate=rebound_budget_blend_gate_out,
            rebound_share=rebound_share_out,
            ast_blend_gate=ast_blend_gate_out,
            env_context=env_side_channel_context,
        )


def build_game_transformer_v2(config: GameTransformerV2Config) -> GameTransformerV2:
    requested_backbone_env_feature_columns = list(getattr(config, "backbone_env_feature_columns", []))
    feature_index = {name: idx for idx, name in enumerate(config.feature_columns)}
    backbone_env_feature_columns = [name for name in requested_backbone_env_feature_columns if name in feature_index]
    backbone_env_feature_indices = [feature_index[name] for name in backbone_env_feature_columns]
    requested_assist_share_condition_feature_columns = list(
        getattr(config, "assist_share_condition_feature_columns", [])
    )
    missing_assist_share_condition_feature_columns = [
        name for name in requested_assist_share_condition_feature_columns if name not in feature_index
    ]
    if missing_assist_share_condition_feature_columns:
        raise ValueError(
            "assist_share_condition_feature_columns missing from feature_columns: "
            f"{missing_assist_share_condition_feature_columns}"
        )
    assist_share_condition_feature_indices = [
        feature_index[name] for name in requested_assist_share_condition_feature_columns
    ]
    assist_share_condition_feature_mean = [
        float(config.feature_mean[idx]) for idx in assist_share_condition_feature_indices
    ]
    assist_share_condition_feature_std = [
        float(config.feature_std[idx]) for idx in assist_share_condition_feature_indices
    ]
    requested_rebound_share_condition_feature_columns = list(
        getattr(config, "rebound_share_condition_feature_columns", [])
    )
    missing_rebound_share_condition_feature_columns = [
        name for name in requested_rebound_share_condition_feature_columns if name not in feature_index
    ]
    if missing_rebound_share_condition_feature_columns:
        raise ValueError(
            "rebound_share_condition_feature_columns missing from feature_columns: "
            f"{missing_rebound_share_condition_feature_columns}"
        )
    rebound_share_condition_feature_indices = [
        feature_index[name] for name in requested_rebound_share_condition_feature_columns
    ]
    rebound_share_condition_feature_mean = [
        float(config.feature_mean[idx]) for idx in rebound_share_condition_feature_indices
    ]
    rebound_share_condition_feature_std = [
        float(config.feature_std[idx]) for idx in rebound_share_condition_feature_indices
    ]
    model = GameTransformerV2(
        num_player_features=len(config.feature_columns),
        num_game_features=len(config.game_feature_columns),
        num_team_features=len(config.team_feature_columns),
        num_efficiency_sidecar_features=len(getattr(config, "efficiency_sidecar_feature_columns", [])),
        game_feature_names=config.game_feature_columns,
        backbone_env_feature_indices=backbone_env_feature_indices,
        backbone_env_feature_names=backbone_env_feature_columns,
        backbone_env_enrich_features=bool(getattr(config, "backbone_env_enrich_features", False)),
        backbone_side_market_context=bool(getattr(config, "backbone_side_market_context", False)),
        backbone_side_market_hidden=int(getattr(config, "backbone_side_market_hidden", 32)),
        enable_env_side_channel=bool(getattr(config, "enable_env_side_channel", False)),
        env_side_channel_dim=int(getattr(config, "env_side_channel_dim", 32)),
        env_side_channel_hidden=int(getattr(config, "env_side_channel_hidden", 64)),
        d_model=int(config.d_model),
        hidden_dim=int(config.hidden_dim),
        num_layers=int(config.num_layers),
        num_heads=int(config.num_heads),
        dropout=float(config.dropout),
        ff_mult=float(config.ff_mult),
        min_active_count=int(config.min_active_count),
        max_active_count=int(config.max_active_count),
        total_minutes_per_team=float(config.total_minutes_per_team),
        max_minutes_per_player=float(config.max_minutes_per_player),
        enable_minutes_hurdle_head=bool(getattr(config, "enable_minutes_hurdle_head", False)),
        minutes_hurdle_hidden=int(getattr(config, "minutes_hurdle_hidden", 64)),
        minutes_hurdle_sigma_floor=float(getattr(config, "minutes_hurdle_sigma_floor", 0.5)),
        enable_minutes_role_head=bool(getattr(config, "enable_minutes_role_head", False)),
        minutes_role_use_context_for_preferences=bool(
            getattr(config, "minutes_role_use_context_for_preferences", True)
        ),
        minutes_role_hidden=int(getattr(config, "minutes_role_hidden", 64)),
        minutes_role_embedding_dim=int(getattr(config, "minutes_role_embedding_dim", 32)),
        minutes_role_num_classes=int(getattr(config, "minutes_role_num_classes", 5)),
        enable_starter_promotion_head=bool(getattr(config, "enable_starter_promotion_head", False)),
        starter_promotion_hidden_dim=int(getattr(config, "starter_promotion_hidden_dim", 64)),
        flow_coupling_type=str(config.flow_coupling_type),
        flow_num_blocks=int(config.flow_num_blocks),
        flow_scale_clip=float(config.flow_scale_clip),
        flow_rqs_num_bins=int(getattr(config, "flow_rqs_num_bins", 8)),
        flow_rqs_tail_bound=float(getattr(config, "flow_rqs_tail_bound", 40.0)),
        flow_rqs_min_bin_width=float(getattr(config, "flow_rqs_min_bin_width", 1e-3)),
        flow_rqs_min_bin_height=float(getattr(config, "flow_rqs_min_bin_height", 1e-3)),
        flow_rqs_min_derivative=float(getattr(config, "flow_rqs_min_derivative", 1e-3)),
        flow_mean_ctx_weight=float(config.flow_mean_ctx_weight),
        flow_context_mode=str(getattr(config, "flow_context_mode", "mean")),
        flow_target_schema=str(getattr(config, "flow_target_schema", FLOW_TARGET_SCHEMA_DEFAULT)),
        flow_use_minutes_conditioning=bool(getattr(config, "flow_use_minutes_conditioning", False)),
        include_pf_in_flow_targets=bool(config.include_pf_in_flow_targets),
        enable_efficiency_head=bool(getattr(config, "enable_efficiency_head", False)),
        efficiency_head_hidden=int(getattr(config, "efficiency_head_hidden", 128)),
        efficiency_ft_prior_mean=float(getattr(config, "efficiency_ft_prior_mean", 0.77)),
        efficiency_ft_prior_strength=float(getattr(config, "efficiency_ft_prior_strength", 6.0)),
        efficiency_fg2_prior_mean=float(getattr(config, "efficiency_fg2_prior_mean", 0.54)),
        efficiency_fg2_prior_strength=float(getattr(config, "efficiency_fg2_prior_strength", 8.0)),
        efficiency_fg3_prior_mean=float(getattr(config, "efficiency_fg3_prior_mean", 0.36)),
        efficiency_fg3_prior_strength=float(getattr(config, "efficiency_fg3_prior_strength", 8.0)),
        efficiency_market_context=bool(getattr(config, "efficiency_market_context", False)),
        efficiency_market_hidden=int(getattr(config, "efficiency_market_hidden", 32)),
        efficiency_market_alpha=float(getattr(config, "efficiency_market_alpha", 1.0)),
        efficiency_sidecar_hidden=int(getattr(config, "efficiency_sidecar_hidden", 32)),
        efficiency_sidecar_alpha=float(getattr(config, "efficiency_sidecar_alpha", 1.0)),
        enable_team_ppp_head=bool(getattr(config, "enable_team_ppp_head", False)),
        team_ppp_head_hidden=int(getattr(config, "team_ppp_head_hidden", 128)),
        team_ppp_to_backbone=bool(getattr(config, "team_ppp_to_backbone", False)),
        team_ppp_latent_hidden=int(getattr(config, "team_ppp_latent_hidden", 32)),
        team_ppp_backbone_alpha=float(getattr(config, "team_ppp_backbone_alpha", 1.0)),
        team_ppp_to_efficiency=bool(getattr(config, "team_ppp_to_efficiency", False)),
        team_ppp_efficiency_alpha=float(getattr(config, "team_ppp_efficiency_alpha", 1.0)),
        team_ppp_direct_backbone_context=bool(getattr(config, "team_ppp_direct_backbone_context", False)),
        team_ppp_direct_efficiency_context=bool(getattr(config, "team_ppp_direct_efficiency_context", False)),
        enable_team_advantage_head=bool(getattr(config, "enable_team_advantage_head", False)),
        team_advantage_head_hidden=int(getattr(config, "team_advantage_head_hidden", 64)),
        team_advantage_direct_backbone_context=bool(
            getattr(config, "team_advantage_direct_backbone_context", False)
        ),
        enable_possession_backbone=bool(getattr(config, "enable_possession_backbone", False)),
        enable_three_pa_share=bool(getattr(config, "enable_three_pa_share", False)),
        possession_head_hidden=int(getattr(config, "possession_head_hidden", 128)),
        possession_mu_mode=str(getattr(config, "possession_mu_mode", "absolute")),
        possession_mu_baseline=float(getattr(config, "possession_mu_baseline", 100.0)),
        enable_team_possession_split_head=bool(getattr(config, "enable_team_possession_split_head", False)),
        team_possession_max_delta=float(getattr(config, "team_possession_max_delta", 8.0)),
        backbone_hidden=int(getattr(config, "backbone_hidden", 128)),
        three_pa_share_hidden=int(getattr(config, "three_pa_share_hidden", 64)),
        enable_usage_share_head=bool(getattr(config, "enable_usage_share_head", False)),
        usage_share_head_hidden=int(getattr(config, "usage_share_head_hidden", 128)),
        enable_team_points_budget_head=bool(getattr(config, "enable_team_points_budget_head", False)),
        team_points_budget_head_hidden=int(getattr(config, "team_points_budget_head_hidden", 128)),
        team_points_budget_parameterization=str(
            getattr(config, "team_points_budget_parameterization", "absolute")
        ),
        team_points_budget_to_backbone=bool(getattr(config, "team_points_budget_to_backbone", False)),
        team_points_budget_latent_hidden=int(getattr(config, "team_points_budget_latent_hidden", 32)),
        team_opportunity_budget_parameterization=str(
            getattr(config, "team_opportunity_budget_parameterization", "absolute")
        ),
        team_opportunity_budget_to_backbone=bool(getattr(config, "team_opportunity_budget_to_backbone", False)),
        team_opportunity_budget_latent_hidden=int(getattr(config, "team_opportunity_budget_latent_hidden", 32)),
        team_opportunity_budget_backbone_alpha=float(
            getattr(config, "team_opportunity_budget_backbone_alpha", 1.0)
        ),
        enable_team_ast_budget_head=bool(getattr(config, "enable_team_ast_budget_head", False)),
        team_ast_budget_head_hidden=int(getattr(config, "team_ast_budget_head_hidden", 128)),
        enable_assist_share_head=bool(getattr(config, "enable_assist_share_head", False)),
        assist_share_head_hidden=int(getattr(config, "assist_share_head_hidden", 128)),
        enable_team_rebound_budget_head=bool(getattr(config, "enable_team_rebound_budget_head", False)),
        team_rebound_budget_head_hidden=int(getattr(config, "team_rebound_budget_head_hidden", 128)),
        rebound_budget_parameterization=str(getattr(config, "rebound_budget_parameterization", "absolute")),
        rebound_oreb_rate_cap=float(getattr(config, "rebound_oreb_rate_cap", 1.0)),
        rebound_dreb_rate_cap=float(getattr(config, "rebound_dreb_rate_cap", 0.85)),
        rebound_dreb_deterministic_discount=float(
            getattr(config, "rebound_dreb_deterministic_discount", 1.0)
        ),
        enable_rebound_budget_blend_gate=bool(getattr(config, "enable_rebound_budget_blend_gate", False)),
        rebound_budget_blend_gate_hidden=int(getattr(config, "rebound_budget_blend_gate_hidden", 64)),
        rebound_budget_blend_gate_init_alpha=float(getattr(config, "rebound_budget_blend_gate_init_alpha", 0.25)),
        enable_rebound_share_head=bool(getattr(config, "enable_rebound_share_head", False)),
        rebound_share_head_hidden=int(getattr(config, "rebound_share_head_hidden", 128)),
        rebound_share_condition_feature_indices=tuple(rebound_share_condition_feature_indices),
        rebound_share_condition_feature_mean=tuple(rebound_share_condition_feature_mean),
        rebound_share_condition_feature_std=tuple(rebound_share_condition_feature_std),
        rebound_share_condition_hidden=int(getattr(config, "rebound_share_condition_hidden", 32)),
        assist_share_condition_feature_indices=tuple(assist_share_condition_feature_indices),
        assist_share_condition_feature_mean=tuple(assist_share_condition_feature_mean),
        assist_share_condition_feature_std=tuple(assist_share_condition_feature_std),
        assist_share_condition_hidden=int(getattr(config, "assist_share_condition_hidden", 32)),
        enable_ast_blend_gate=bool(getattr(config, "enable_ast_blend_gate", False)),
        ast_blend_gate_hidden=int(getattr(config, "ast_blend_gate_hidden", 128)),
        ast_blend_gate_init_alpha=float(getattr(config, "ast_blend_gate_init_alpha", 0.75)),
        backbone_env_adapter_dim=int(getattr(config, "backbone_env_adapter_dim", 0)),
        backbone_env_adapter_hidden=int(getattr(config, "backbone_env_adapter_hidden", 32)),
    )
    setattr(model, "gtv2_config", config)
    return model
