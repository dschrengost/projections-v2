"""Initial world generation path for GameTransformerV2 using inverse flow sampling."""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from projections.rotation.possession_backbone import FTA_POSS_COEFF
from torch.utils.data import DataLoader

from projections.minutes import PLAY_THRESHOLD_MINUTES, ROTATION_THRESHOLD_MINUTES
from projections.projections_bundle import add_canonical_projection_fields
from projections import paths
from projections.rotation.game_transformer_v2 import (
    FLOW_TARGET_COLUMNS_V1,
    GameLevelDataset,
    GameTransformerV2Config,
    build_game_level_examples,
    build_game_transformer_v2,
    collate_game_level_examples,
    flow_contract_columns,
    reconstruct_flow_to_contract,
)
from projections.rotation.gtv2_promotion_hybrid import (
    BenchRiserHybridConfig,
    PromotionHybridConfig,
    assert_promotion_hybrid_compatible,
    blend_expert_predictions,
    blend_promotion_predictions,
    compute_bench_riser_candidate_mask,
    compute_starter_promotion_candidate_mask,
)
from projections.rotation.joint_minutes import project_minutes_capped_simplex
from projections.rotation.set_model import zfill_game_id_series

logger = logging.getLogger(__name__)

JOIN_KEYS = ["game_id", "team_id", "player_id", "game_date"]
DEFAULT_FORCE_ACTIVE_MINUTES_FLOOR_RATIO = 0.65
DEFAULT_FORCE_ACTIVE_MINUTES_FLOOR_MIN = 12.0
DEFAULT_FORCE_ACTIVE_MINUTES_FLOOR_MAX = 36.0
DEFAULT_STARTER_LOW_MINUTES_TRIGGER = 10.0
DEFAULT_ACTIVE_MINUTES_TOL = 1e-6


@dataclass(frozen=True)
class MakeModelConfig:
    mode: str = "legacy"
    use_learned_efficiency: bool = True
    legacy_use_efficiency_mean: bool = False
    bb_ft_prior_mean: float = 0.77
    bb_ft_prior_strength: float = 6.0
    bb_ft_concentration: float = 8.0
    bb_fg2_prior_mean: float = 0.54
    bb_fg2_prior_strength: float = 8.0
    bb_fg2_concentration: float = 10.0
    bb_fg3_prior_mean: float = 0.36
    bb_fg3_prior_strength: float = 8.0
    bb_fg3_concentration: float = 10.0


@dataclass(frozen=True)
class MinutesUncertaintyConfig:
    enabled: bool = False
    mode: str = "gaussian"
    gaussian_scale: float = 1.0
    min_sigma: float = 0.75
    max_sigma: float = 6.0
    fallback_sigma: float = 1.5
    use_hurdle_sigma: bool = True
    use_prior_std: bool = True
    preserve_top_k_per_team: int = 3
    full_sigma_at_minutes_or_below: float = 24.0
    zero_sigma_at_minutes_or_above: float = 32.0
    dirichlet_base_concentration: float = 24.0
    prior_std_columns: tuple[str, ...] = (
        "minutes_from_stints_std_prior_20",
        "minutes_from_stints_std_prior_10",
        "minutes_from_stints_std_prior_5",
    )


@dataclass(frozen=True)
class AstFactorizationRuntimeConfig:
    ast_blend_alpha: float = 1.0
    assist_share_temperature: float = 1.0
    team_ast_budget_blend_alpha: float = 1.0
    creator_reconcile_alpha_enabled: bool = False
    creator_reconcile_alpha_max: float = 0.5
    creator_reconcile_ast_line_center: float = 6.0
    creator_reconcile_ast_line_scale: float = 1.25
    creator_reconcile_minutes_center: float = 28.0
    creator_reconcile_minutes_scale: float = 5.0
    creator_reconcile_prior_play_prob_floor: float = 0.8
    creator_reconcile_starter_penalty_weight: float = 0.25
    creator_reconcile_team_relative: bool = False
    creator_reconcile_team_power: float = 1.0


def _assert_no_labels_in_forward_kwargs(kwargs: dict[str, Any]) -> None:
    """Hard guard: sampling forward pass must not consume label/target tensors."""
    forbidden_none = ["target_counts", "target_active_mask", "flow_targets", "flow_observed_mask"]
    for key in forbidden_none:
        if kwargs.get(key, None) is not None:
            raise RuntimeError(f"label leakage guard failed: `{key}` must be None in sampler forward")
    forbidden_false = ["use_target_counts", "use_target_active_mask", "minutes_use_target_active", "run_flow"]
    for key in forbidden_false:
        if bool(kwargs.get(key, False)):
            raise RuntimeError(f"label leakage guard failed: `{key}` must be False in sampler forward")


def _utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _decode_player_feature_column(
    player_features: torch.Tensor,
    *,
    config: GameTransformerV2Config | None,
    column_name: str,
) -> torch.Tensor | None:
    if config is None:
        return None
    try:
        idx = int(config.feature_columns.index(str(column_name)))
    except ValueError:
        return None
    if idx < 0 or idx >= int(player_features.shape[-1]):
        return None
    mean = float(config.feature_mean[idx]) if idx < len(config.feature_mean) else 0.0
    std = float(config.feature_std[idx]) if idx < len(config.feature_std) else 1.0
    scale = std if abs(std) > 1e-6 else 1.0
    return player_features[..., idx].to(dtype=torch.float32) * float(scale) + float(mean)


def _decode_out_player_mask(
    player_features: torch.Tensor,
    *,
    valid_mask: torch.Tensor,
    config: GameTransformerV2Config | None,
) -> torch.Tensor:
    decoded = _decode_player_feature_column(
        player_features,
        config=config,
        column_name="is_out",
    )
    if decoded is None:
        return torch.zeros_like(valid_mask, dtype=torch.bool)
    return decoded.ge(0.5) & valid_mask.to(dtype=torch.bool)


def _build_creator_reconcile_alpha(
    player_features: torch.Tensor,
    *,
    valid_mask: torch.Tensor,
    team_index: torch.Tensor | None,
    config: GameTransformerV2Config | None,
    runtime_config: AstFactorizationRuntimeConfig | None,
) -> torch.Tensor | None:
    cfg = runtime_config or AstFactorizationRuntimeConfig()
    if not bool(cfg.creator_reconcile_alpha_enabled):
        return None
    if player_features.ndim == 4:
        player_features = player_features.reshape(
            player_features.shape[0],
            -1,
            player_features.shape[-1],
        )
    if player_features.ndim != 3:
        raise ValueError("player_features must have shape (N, P, F) or (N, 2, 15, F)")
    ast_line = _decode_player_feature_column(player_features, config=config, column_name="an_ast_line")
    implied_minutes = _decode_player_feature_column(
        player_features,
        config=config,
        column_name="an_implied_minutes",
    )
    prior_play_prob = _decode_player_feature_column(
        player_features,
        config=config,
        column_name="prior_play_prob",
    )
    started_rate = _decode_player_feature_column(
        player_features,
        config=config,
        column_name="started_proxy_rate_prior_20",
    )
    if ast_line is None or implied_minutes is None or prior_play_prob is None:
        return None
    ast_scale = max(float(cfg.creator_reconcile_ast_line_scale), 1e-6)
    minutes_scale = max(float(cfg.creator_reconcile_minutes_scale), 1e-6)
    ast_score = torch.sigmoid((ast_line - float(cfg.creator_reconcile_ast_line_center)) / ast_scale)
    minutes_score = torch.sigmoid(
        (implied_minutes - float(cfg.creator_reconcile_minutes_center)) / minutes_scale
    )
    prior_score = (prior_play_prob >= float(cfg.creator_reconcile_prior_play_prob_floor)).to(dtype=torch.float32)
    if started_rate is None:
        starter_penalty = torch.ones_like(ast_score, dtype=torch.float32)
    else:
        starter_penalty = 1.0 - float(cfg.creator_reconcile_starter_penalty_weight) * started_rate.clamp(0.0, 1.0)
    raw_score = (
        ast_score.to(dtype=torch.float32)
        * minutes_score.to(dtype=torch.float32)
        * prior_score
        * starter_penalty.to(dtype=torch.float32)
    )
    if bool(cfg.creator_reconcile_team_relative):
        if team_index is None:
            raise ValueError("team_index is required when creator_reconcile_team_relative is enabled")
        if team_index.shape != valid_mask.shape:
            raise ValueError("team_index must align with valid_mask for team-relative creator alpha")
        rel_score = torch.zeros_like(raw_score)
        team_power = max(float(cfg.creator_reconcile_team_power), 1e-6)
        valid_bool = valid_mask.to(dtype=torch.bool)
        for side in (0, 1):
            side_mask = valid_bool & team_index.eq(side)
            side_scores = torch.where(side_mask, raw_score, torch.zeros_like(raw_score))
            side_max = side_scores.max(dim=1, keepdim=True).values.clamp(min=1e-6)
            side_rel = torch.where(side_mask, (side_scores / side_max) ** team_power, torch.zeros_like(raw_score))
            rel_score = rel_score + side_rel
        raw_score = rel_score
    alpha = float(cfg.creator_reconcile_alpha_max) * raw_score
    alpha = alpha.clamp(min=0.0, max=1.0)
    return alpha * valid_mask.to(dtype=alpha.dtype)


def _estimate_minutes_sigma(
    *,
    minutes_base: torch.Tensor,
    minutes_out: Any,
    player_features: torch.Tensor,
    valid_mask: torch.Tensor,
    config: GameTransformerV2Config | None,
    uncertainty_config: MinutesUncertaintyConfig,
) -> torch.Tensor:
    sigma = torch.full_like(minutes_base, float(uncertainty_config.fallback_sigma), dtype=torch.float32)
    sigma_source: torch.Tensor | None = None
    if bool(uncertainty_config.use_hurdle_sigma):
        hurdle_sigma = getattr(minutes_out, "sigma", None)
        if isinstance(hurdle_sigma, torch.Tensor):
            sigma_source = hurdle_sigma.to(device=minutes_base.device, dtype=torch.float32)
    if sigma_source is None and bool(uncertainty_config.use_prior_std):
        decoded_cols: list[torch.Tensor] = []
        for col in uncertainty_config.prior_std_columns:
            decoded = _decode_player_feature_column(player_features, config=config, column_name=str(col))
            if isinstance(decoded, torch.Tensor):
                decoded_cols.append(decoded.to(device=minutes_base.device, dtype=torch.float32))
        if decoded_cols:
            sigma_source = torch.stack(decoded_cols, dim=0).amax(dim=0)
    if sigma_source is not None:
        sigma = sigma_source
    sigma = sigma * float(uncertainty_config.gaussian_scale)
    sigma = sigma.clamp(min=float(uncertainty_config.min_sigma), max=float(uncertainty_config.max_sigma))
    return sigma * valid_mask.to(dtype=sigma.dtype)


def _sample_minutes_with_uncertainty(
    *,
    minutes_base: torch.Tensor,
    active_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    player_team_index: torch.Tensor,
    sigma: torch.Tensor,
    uncertainty_config: MinutesUncertaintyConfig,
    total_minutes_per_team: float = 240.0,
    max_minutes_per_player: float = 48.0,
) -> torch.Tensor:
    active = active_mask.to(dtype=torch.bool) & valid_mask.to(dtype=torch.bool)
    if not bool(active.any()):
        return minutes_base
    sigma_eff = sigma.to(dtype=torch.float32).clone()
    lo = float(uncertainty_config.full_sigma_at_minutes_or_below)
    hi = float(uncertainty_config.zero_sigma_at_minutes_or_above)
    if hi <= lo:
        taper = minutes_base.new_ones(minutes_base.shape, dtype=torch.float32)
    else:
        taper = ((float(hi) - minutes_base.to(dtype=torch.float32)) / float(hi - lo)).clamp(min=0.0, max=1.0)
    sigma_eff = sigma_eff * taper
    top_k = max(0, int(uncertainty_config.preserve_top_k_per_team))
    protected_mask = torch.zeros_like(active, dtype=torch.bool)
    if top_k > 0:
        for team_idx in (0, 1):
            team_mask = active & (player_team_index == team_idx)
            scores = minutes_base.to(dtype=torch.float32).masked_fill(~team_mask, float("-inf"))
            order = torch.argsort(scores, dim=1, descending=True)
            ranks = torch.arange(minutes_base.shape[1], device=minutes_base.device).unsqueeze(0).expand_as(order)
            top_take = ranks < min(top_k, int(minutes_base.shape[1]))
            protect_mask = torch.zeros_like(team_mask, dtype=torch.bool)
            protect_mask.scatter_(1, order, top_take)
            protect_mask = protect_mask & team_mask
            protected_mask = protected_mask | protect_mask
            sigma_eff = torch.where(protect_mask, torch.zeros_like(sigma_eff), sigma_eff)
    mode = str(uncertainty_config.mode).strip().lower()
    if mode == "residual_dirichlet":
        result = torch.where(protected_mask, minutes_base.to(dtype=torch.float32), torch.zeros_like(minutes_base))
        for team_idx in (0, 1):
            team_active = active & (player_team_index == team_idx)
            team_protected = protected_mask & (player_team_index == team_idx)
            team_unprotected = team_active & (~team_protected)
            protected_sum = result.masked_fill(~team_protected, 0.0).sum(dim=1, keepdim=True)
            remaining_total = (
                torch.full_like(protected_sum, float(total_minutes_per_team)) - protected_sum
            ).clamp(min=0.0)
            base_unprotected = minutes_base.to(dtype=torch.float32).masked_fill(~team_unprotected, 0.0)
            base_sum = base_unprotected.sum(dim=1, keepdim=True)
            shares = torch.where(
                base_sum > 1e-6,
                base_unprotected / base_sum.clamp(min=1e-6),
                torch.zeros_like(base_unprotected),
            )
            sigma_mean = torch.where(
                team_unprotected.any(dim=1, keepdim=True),
                sigma_eff.masked_fill(~team_unprotected, 0.0).sum(dim=1, keepdim=True)
                / team_unprotected.to(dtype=torch.float32).sum(dim=1, keepdim=True).clamp(min=1.0),
                torch.ones_like(remaining_total),
            )
            alpha0 = (float(uncertainty_config.dirichlet_base_concentration) / sigma_mean.clamp(min=1.0)).clamp(
                min=4.0,
                max=64.0,
            )
            alpha = (shares * alpha0).clamp(min=1e-3)
            gamma = torch.distributions.Gamma(alpha, torch.ones_like(alpha)).sample()
            gamma = gamma.masked_fill(~team_unprotected, 0.0)
            gamma_sum = gamma.sum(dim=1, keepdim=True)
            sampled_share = torch.where(
                gamma_sum > 1e-6,
                gamma / gamma_sum.clamp(min=1e-6),
                shares,
            )
            sampled_team = sampled_share * remaining_total
            result = torch.where(team_unprotected, sampled_team, result)
        return result * valid_mask.to(dtype=result.dtype)

    noise = torch.randn_like(minutes_base, dtype=torch.float32) * sigma_eff
    noisy_seed = torch.where(active, minutes_base.to(dtype=torch.float32) + noise, torch.zeros_like(minutes_base))
    noisy_seed = noisy_seed.clamp(min=0.0)
    sampled_minutes, _ = project_minutes_capped_simplex(
        noisy_seed,
        active,
        valid_mask,
        player_team_index,
        total_minutes_per_team=float(total_minutes_per_team),
        max_minutes_per_player=float(max_minutes_per_player),
    )
    if top_k > 0 and bool(protected_mask.any()):
        protected_minutes = torch.where(protected_mask, minutes_base.to(dtype=torch.float32), torch.zeros_like(sampled_minutes))
        result = sampled_minutes.clone()
        for team_idx in (0, 1):
            team_active = active & (player_team_index == team_idx)
            team_protected = protected_mask & (player_team_index == team_idx)
            team_unprotected = team_active & (~team_protected)
            protected_sum = protected_minutes.masked_fill(~team_protected, 0.0).sum(dim=1, keepdim=True)
            remaining_total = (
                torch.full_like(protected_sum, float(total_minutes_per_team)) - protected_sum
            ).clamp(min=0.0)
            current_unprotected = result.masked_fill(~team_unprotected, 0.0)
            current_sum = current_unprotected.sum(dim=1, keepdim=True)
            fallback_unprotected = minutes_base.to(dtype=torch.float32).masked_fill(~team_unprotected, 0.0)
            fallback_sum = fallback_unprotected.sum(dim=1, keepdim=True)
            scaled_unprotected = torch.where(
                current_sum > 1e-6,
                current_unprotected * (remaining_total / current_sum.clamp(min=1e-6)),
                torch.where(
                    fallback_sum > 1e-6,
                    fallback_unprotected * (remaining_total / fallback_sum.clamp(min=1e-6)),
                    current_unprotected,
                ),
            )
            result = torch.where(team_unprotected, scaled_unprotected, result)
            result = torch.where(team_protected, protected_minutes, result)
        sampled_minutes = result * valid_mask.to(dtype=result.dtype)
    return sampled_minutes


def _renormalize_player_features_for_model(
    player_features: torch.Tensor,
    *,
    source_config: GameTransformerV2Config | None,
    target_config: GameTransformerV2Config | None,
) -> torch.Tensor:
    if source_config is None or target_config is None:
        return player_features
    if list(source_config.feature_columns) != list(target_config.feature_columns):
        raise ValueError("Cannot renormalize expert features: feature_columns mismatch")
    source_mean = np.asarray(source_config.feature_mean, dtype=np.float32)
    source_std = np.asarray(source_config.feature_std, dtype=np.float32)
    target_mean = np.asarray(target_config.feature_mean, dtype=np.float32)
    target_std = np.asarray(target_config.feature_std, dtype=np.float32)
    if (
        source_mean.shape == target_mean.shape
        and source_std.shape == target_std.shape
        and np.allclose(source_mean, target_mean, atol=1e-6)
        and np.allclose(source_std, target_std, atol=1e-6)
    ):
        return player_features
    source_mean_t = torch.as_tensor(source_mean, dtype=player_features.dtype, device=player_features.device)
    source_std_t = torch.as_tensor(source_std, dtype=player_features.dtype, device=player_features.device)
    target_mean_t = torch.as_tensor(target_mean, dtype=player_features.dtype, device=player_features.device)
    target_std_t = torch.as_tensor(target_std, dtype=player_features.dtype, device=player_features.device)
    source_std_t = torch.where(source_std_t.abs() > 1e-6, source_std_t, torch.ones_like(source_std_t))
    target_std_t = torch.where(target_std_t.abs() > 1e-6, target_std_t, torch.ones_like(target_std_t))
    raw = player_features * source_std_t.view(1, 1, 1, -1) + source_mean_t.view(1, 1, 1, -1)
    return (raw - target_mean_t.view(1, 1, 1, -1)) / target_std_t.view(1, 1, 1, -1)


def _coerce_join_keys(df: pd.DataFrame, *, name: str) -> pd.DataFrame:
    out = df.copy()
    for col in ["game_id", "team_id", "player_id"]:
        if col not in out.columns:
            raise ValueError(f"{name} missing key column: {col}")
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
    if "game_date" not in out.columns:
        raise ValueError(f"{name} missing key column: game_date")
    game_date = out["game_date"]
    if pd.api.types.is_datetime64_any_dtype(game_date):
        normalized = pd.Series(game_date, copy=False)
    else:
        normalized = pd.to_datetime(game_date.astype("string"), errors="coerce")
    out["game_date"] = pd.Series(normalized, copy=False).dt.normalize()

    invalid = out[["game_id", "team_id", "player_id", "game_date"]].isna().any(axis=1)
    if invalid.any():
        raise ValueError(f"{name} has invalid key rows: {int(invalid.sum())}")
    return out


def _resolve_dataset_dir(value: str | None) -> Path:
    root = paths.get_data_root() / "training" / "datasets"
    if value:
        p = Path(value).expanduser()
        if p.exists():
            return p.resolve()
        p2 = root / value
        if p2.exists():
            return p2.resolve()
        raise FileNotFoundError(f"Dataset directory not found: {value}")

    candidates = sorted(root.glob("joint_rotation_rates_v1*"))
    if not candidates:
        raise FileNotFoundError(f"No joint_rotation_rates_v1* datasets found under {root}")
    return candidates[-1].resolve()


def _resolve_run_dir(value: str) -> Path:
    p = Path(value).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"Run directory not found: {value}")
    if not (p / "config.json").exists() or not (p / "model.pt").exists():
        raise FileNotFoundError(f"Run directory missing config.json/model.pt: {p}")
    return p.resolve()


def _split_val(df: pd.DataFrame, *, val_days: int) -> pd.DataFrame:
    days = sorted(pd.to_datetime(df["game_date"]).dropna().dt.normalize().unique().tolist())
    vd = max(1, int(val_days))
    val_dates = set(days[-vd:])
    return df.loc[pd.to_datetime(df["game_date"]).dt.normalize().isin(val_dates)].copy()


def _flow_idx(flow_target_columns: list[str], name: str) -> int:
    try:
        return int(flow_target_columns.index(name))
    except ValueError as exc:
        raise KeyError(f"Missing flow target column: {name}") from exc


def project_flow_stats_to_contract(
    flow_values: torch.Tensor,
    *,
    flow_target_columns: list[str],
    flow_contract_columns: list[str] | None = None,
    fg2_rate: torch.Tensor | float | None = None,
    fg3_rate: torch.Tensor | float | None = None,
    ft_rate: torch.Tensor | float | None = None,
    ast_override: torch.Tensor | None = None,
) -> torch.Tensor:
    """Project model flow outputs to contract stats used by worlds and diagnostics."""

    out = reconstruct_flow_to_contract(
        flow_values,
        flow_target_columns=flow_target_columns,
        contract_columns=flow_contract_columns,
        fg2_rate=fg2_rate,
        fg3_rate=fg3_rate,
        ft_rate=ft_rate,
    )
    if ast_override is not None:
        contract_cols = list(flow_contract_columns) if flow_contract_columns is not None else list(flow_target_columns)
        ast_idx = _flow_idx(contract_cols, "ast")
        if ast_override.shape != out[..., ast_idx].shape:
            raise ValueError("ast_override must align with projected flow ast column shape")
        out = out.clone()
        out[..., ast_idx] = ast_override
    return out


def _reconstruct_ast_from_heads(
    *,
    player_valid_mask: torch.Tensor,
    player_team_index: torch.Tensor,
    team_ast_budget: torch.Tensor | None,
    assist_share_logits: torch.Tensor | None,
    share_temperature: float = 1.0,
) -> torch.Tensor | None:
    if team_ast_budget is None or assist_share_logits is None:
        return None
    if team_ast_budget.ndim != 2 or team_ast_budget.shape[1] != 2:
        raise ValueError("team_ast_budget must have shape (N, 2)")
    if assist_share_logits.ndim != 2:
        raise ValueError("assist_share_logits must have shape (N, P)")
    if assist_share_logits.shape != player_valid_mask.shape or player_team_index.shape != player_valid_mask.shape:
        raise ValueError("assist-share reconstruction tensors must align on (N, P)")

    valid_mask = player_valid_mask.to(dtype=torch.bool)
    ast_share = torch.zeros_like(assist_share_logits)
    temp = max(float(share_temperature), 1e-6)
    for side in (0, 1):
        side_mask = valid_mask & player_team_index.eq(side)
        if bool(side_mask.any()):
            side_logits = (assist_share_logits / temp).masked_fill(~side_mask, float("-inf"))
            ast_share = ast_share + torch.softmax(side_logits, dim=1) * side_mask.to(dtype=assist_share_logits.dtype)
    return ast_share * torch.where(
        player_team_index.eq(0),
        team_ast_budget[:, 0].unsqueeze(1),
        team_ast_budget[:, 1].unsqueeze(1),
    )


def _team_sum_by_side(
    values: torch.Tensor,
    *,
    valid_mask: torch.Tensor,
    team_index: torch.Tensor,
) -> torch.Tensor:
    if values.shape != valid_mask.shape or values.shape != team_index.shape:
        raise ValueError("values, valid_mask, and team_index must align")
    outs: list[torch.Tensor] = []
    for side in (0, 1):
        side_mask = valid_mask.to(dtype=torch.bool) & team_index.eq(side)
        outs.append((values * side_mask.to(dtype=values.dtype)).sum(dim=1))
    return torch.stack(outs, dim=1)


def _build_ast_override(
    *,
    flow_projected_base: torch.Tensor,
    flow_contract_columns: list[str],
    player_valid_mask: torch.Tensor,
    player_team_index: torch.Tensor,
    team_ast_budget: torch.Tensor | None,
    assist_share_logits: torch.Tensor | None,
    ast_blend_gate: torch.Tensor | None,
    runtime_config: AstFactorizationRuntimeConfig | None,
) -> torch.Tensor | None:
    if team_ast_budget is None or assist_share_logits is None:
        return None
    cfg = runtime_config or AstFactorizationRuntimeConfig()
    ast_idx = _flow_idx(flow_contract_columns, "ast")
    flow_ast = flow_projected_base[..., ast_idx]
    blended_budget = team_ast_budget
    budget_alpha = float(cfg.team_ast_budget_blend_alpha)
    if budget_alpha < 1.0:
        flow_team_ast = _team_sum_by_side(
            flow_ast,
            valid_mask=player_valid_mask,
            team_index=player_team_index,
        )
        budget_alpha = min(max(budget_alpha, 0.0), 1.0)
        blended_budget = budget_alpha * team_ast_budget + (1.0 - budget_alpha) * flow_team_ast
    factorized_ast = _reconstruct_ast_from_heads(
        player_valid_mask=player_valid_mask,
        player_team_index=player_team_index,
        team_ast_budget=blended_budget,
        assist_share_logits=assist_share_logits,
        share_temperature=float(cfg.assist_share_temperature),
    )
    if factorized_ast is None:
        return None
    if ast_blend_gate is not None:
        if ast_blend_gate.shape != flow_ast.shape:
            raise ValueError("ast_blend_gate must align with projected flow ast shape")
        ast_alpha = ast_blend_gate.to(device=flow_ast.device, dtype=flow_ast.dtype).clamp(min=0.0, max=1.0)
        return ast_alpha * factorized_ast + (1.0 - ast_alpha) * flow_ast
    ast_alpha = min(max(float(cfg.ast_blend_alpha), 0.0), 1.0)
    if ast_alpha >= 1.0:
        return factorized_ast
    if ast_alpha <= 0.0:
        return flow_ast
    return ast_alpha * factorized_ast + (1.0 - ast_alpha) * flow_ast


def _reconcile_ast_to_team_budget(
    *,
    flow_values: torch.Tensor,
    valid_mask: torch.Tensor,
    team_index: torch.Tensor,
    active_mask: torch.Tensor,
    flow_target_columns: list[str],
    team_ast_budget: torch.Tensor | None,
    assist_share_logits: torch.Tensor | None,
    share_alpha: float | torch.Tensor,
    share_temperature: float,
) -> torch.Tensor:
    if team_ast_budget is None or assist_share_logits is None:
        return flow_values
    if flow_values.ndim != 3:
        raise ValueError("flow_values must have shape (N, 30, S)")
    if valid_mask.shape != team_index.shape or valid_mask.shape != active_mask.shape:
        raise ValueError("valid_mask/team_index/active_mask must align")
    if team_ast_budget.ndim != 2 or team_ast_budget.shape[1] != 2:
        raise ValueError("team_ast_budget must have shape (N, 2)")
    if assist_share_logits.shape != valid_mask.shape:
        raise ValueError("assist_share_logits must align with valid_mask")

    out = flow_values.clone()
    ast_idx = _flow_idx(flow_target_columns, "ast")
    old_ast = out[:, :, ast_idx]
    alpha_tensor: torch.Tensor | None = None
    if isinstance(share_alpha, torch.Tensor):
        if share_alpha.shape != valid_mask.shape:
            raise ValueError("share_alpha tensor must align with valid_mask")
        alpha_tensor = share_alpha.to(device=flow_values.device, dtype=flow_values.dtype).clamp(min=0.0, max=1.0)
    else:
        alpha = float(np.clip(float(share_alpha), 0.0, 1.0))
    temp = max(float(share_temperature), 1e-6)
    valid = valid_mask.to(dtype=torch.bool)
    active = active_mask.to(dtype=torch.bool)
    alloc_base = valid & active

    for side in (0, 1):
        side_mask = team_index.eq(side)
        elig = alloc_base & side_mask
        valid_side = valid & side_mask
        has_active = elig.any(dim=1, keepdim=True)
        elig = torch.where(has_active, elig, valid_side)

        flow_weights = _normalize_alloc_weights(old_ast, eligible_mask=elig)
        factorized_weights = _softmax_alloc_weights(assist_share_logits / temp, eligible_mask=elig)
        if alpha_tensor is None:
            blend_weights = (1.0 - alpha) * flow_weights + alpha * factorized_weights
        else:
            side_alpha = torch.where(side_mask, alpha_tensor, torch.zeros_like(alpha_tensor))
            blend_weights = (1.0 - side_alpha) * flow_weights + side_alpha * factorized_weights
        ast_weights = _normalize_alloc_weights(blend_weights, eligible_mask=elig)
        new_ast = ast_weights * torch.clamp(team_ast_budget[:, side], min=0.0).unsqueeze(1)
        side_f = side_mask.to(dtype=out.dtype)
        out[:, :, ast_idx] = out[:, :, ast_idx] * (1.0 - side_f) + new_ast * side_f
    return out


def _reconcile_points_to_team_budget(
    *,
    flow_values: torch.Tensor,
    valid_mask: torch.Tensor,
    team_index: torch.Tensor,
    active_mask: torch.Tensor,
    flow_target_columns: list[str],
    team_points_budget: torch.Tensor | None,
    budget_alpha: float,
) -> torch.Tensor:
    if team_points_budget is None:
        return flow_values
    if flow_values.ndim != 3:
        raise ValueError("flow_values must have shape (N, 30, S)")
    if valid_mask.shape != team_index.shape or valid_mask.shape != active_mask.shape:
        raise ValueError("valid_mask/team_index/active_mask must align")
    if team_points_budget.ndim != 2 or team_points_budget.shape[1] != 2:
        raise ValueError("team_points_budget must have shape (N, 2)")

    fg2m_idx = _flow_idx(flow_target_columns, "fg2m")
    fg3m_idx = _flow_idx(flow_target_columns, "fg3m")
    ftm_idx = _flow_idx(flow_target_columns, "ftm")
    fga2_idx = _flow_idx(flow_target_columns, "fga2")
    fga3_idx = _flow_idx(flow_target_columns, "fga3")
    fta_idx = _flow_idx(flow_target_columns, "fta")

    alpha = float(np.clip(float(budget_alpha), 0.0, 1.0))
    if alpha <= 0.0:
        return flow_values

    valid = valid_mask.to(dtype=torch.bool)
    active = active_mask.to(dtype=torch.bool)
    alloc_base = valid & active
    fg2m = flow_values[:, :, fg2m_idx]
    fg3m = flow_values[:, :, fg3m_idx]
    ftm = flow_values[:, :, ftm_idx]
    fga2 = flow_values[:, :, fga2_idx]
    fga3 = flow_values[:, :, fga3_idx]
    fta = flow_values[:, :, fta_idx]

    for side in (0, 1):
        side_mask = team_index.eq(side)
        elig = alloc_base & side_mask
        valid_side = valid & side_mask
        has_active = elig.any(dim=1, keepdim=True)
        elig = torch.where(has_active, elig, valid_side)
        side_f = elig.to(dtype=flow_values.dtype)

        team_fg2m = (fg2m * side_f).sum(dim=1)
        team_fg3m = (fg3m * side_f).sum(dim=1)
        team_ftm = (ftm * side_f).sum(dim=1)
        team_pts = 2.0 * team_fg2m + 3.0 * team_fg3m + team_ftm

        target_pts = torch.clamp(team_points_budget[:, side], min=0.0)
        blended_pts = (1.0 - alpha) * team_pts + alpha * target_pts
        scale = torch.where(
            team_pts > 1e-6,
            blended_pts / team_pts.clamp(min=1e-6),
            torch.ones_like(team_pts),
        ).unsqueeze(1)

        new_fg2m = torch.minimum(fga2, fg2m * scale)
        new_fg3m = torch.minimum(fga3, fg3m * scale)
        new_ftm = torch.minimum(fta, ftm * scale)
        fg2m = torch.where(side_mask, new_fg2m, fg2m)
        fg3m = torch.where(side_mask, new_fg3m, fg3m)
        ftm = torch.where(side_mask, new_ftm, ftm)
    out = flow_values.clone()
    out[:, :, fg2m_idx] = fg2m
    out[:, :, fg3m_idx] = fg3m
    out[:, :, ftm_idx] = ftm

    return out


def _resolve_team_points_budget(
    *,
    model_config: object,
    game_features: torch.Tensor,
    team_points_budget_out: torch.Tensor | None,
    team_ppp_out: torch.Tensor | None = None,
    possession_out: torch.Tensor | None = None,
) -> torch.Tensor | None:
    mode = str(getattr(model_config, "team_points_budget_parameterization", "absolute")).strip().lower()
    if mode == "market_implied":
        game_names = list(getattr(model_config, "game_feature_columns", []))
        if not game_names:
            return None
        total_idx = game_names.index("vegas_total") if "vegas_total" in game_names else -1
        spread_idx = game_names.index("vegas_spread") if "vegas_spread" in game_names else -1
        if total_idx < 0 or spread_idx < 0:
            return None
        total = game_features[:, total_idx : total_idx + 1]
        spread = game_features[:, spread_idx : spread_idx + 1]
        home_total = 0.5 * (total - spread)
        away_total = 0.5 * (total + spread)
        return torch.cat([home_total, away_total], dim=1)
    if mode == "team_ppp_implied":
        if team_ppp_out is None:
            return None
        if team_ppp_out.ndim != 2 or team_ppp_out.shape[1] != 2:
            raise ValueError("team_ppp_out must have shape (B, 2)")
        if possession_out is not None:
            if possession_out.ndim == 1:
                poss = possession_out.unsqueeze(1).expand(-1, 2)
            elif possession_out.ndim == 2 and possession_out.shape[1] == 2:
                poss = possession_out
            else:
                raise ValueError("possession_out must have shape (B,) or (B, 2)")
        else:
            game_names = list(getattr(model_config, "game_feature_columns", []))
            poss_idx = game_names.index("estimated_possessions") if "estimated_possessions" in game_names else -1
            if poss_idx < 0:
                return None
            poss = game_features[:, poss_idx : poss_idx + 1].expand(-1, 2)
        return team_ppp_out * poss.clamp_min(1.0)
    return team_points_budget_out


def _resolve_team_opportunity_share(
    *,
    model_config: object,
    game_features: torch.Tensor,
) -> torch.Tensor | None:
    mode = str(getattr(model_config, "team_opportunity_budget_parameterization", "absolute")).strip().lower()
    if mode != "market_implied_share":
        return None
    game_names = list(getattr(model_config, "game_feature_columns", []))
    if not game_names:
        return None
    total_idx = game_names.index("vegas_total") if "vegas_total" in game_names else -1
    spread_idx = game_names.index("vegas_spread") if "vegas_spread" in game_names else -1
    if total_idx < 0 or spread_idx < 0:
        return None
    total = game_features[:, total_idx : total_idx + 1].clamp_min(1e-6)
    spread = game_features[:, spread_idx : spread_idx + 1]
    home_total = 0.5 * (total - spread)
    away_total = 0.5 * (total + spread)
    home_share = torch.clamp(home_total / total, min=0.0, max=1.0)
    away_share = torch.clamp(away_total / total, min=0.0, max=1.0)
    share_sum = (home_share + away_share).clamp_min(1e-6)
    return torch.cat([home_share / share_sum, away_share / share_sum], dim=1)


def _reconcile_opportunities_to_team_budget(
    *,
    flow_values: torch.Tensor,
    valid_mask: torch.Tensor,
    team_index: torch.Tensor,
    active_mask: torch.Tensor,
    flow_target_columns: list[str],
    team_opportunity_share: torch.Tensor | None,
    budget_alpha: float,
    preserve_possessions: bool = False,
) -> torch.Tensor:
    if team_opportunity_share is None:
        return flow_values
    if flow_values.ndim != 3:
        raise ValueError("flow_values must have shape (N, 30, S)")
    if valid_mask.shape != team_index.shape or valid_mask.shape != active_mask.shape:
        raise ValueError("valid_mask/team_index/active_mask must align")
    if team_opportunity_share.ndim != 2 or team_opportunity_share.shape[1] != 2:
        raise ValueError("team_opportunity_share must have shape (N, 2)")

    fga2_idx = _flow_idx(flow_target_columns, "fga2")
    fg2m_idx = _flow_idx(flow_target_columns, "fg2m")
    fga3_idx = _flow_idx(flow_target_columns, "fga3")
    fg3m_idx = _flow_idx(flow_target_columns, "fg3m")
    fta_idx = _flow_idx(flow_target_columns, "fta")
    ftm_idx = _flow_idx(flow_target_columns, "ftm")
    tov_idx = _flow_idx(flow_target_columns, "tov")
    oreb_idx = _flow_idx(flow_target_columns, "oreb")

    alpha = float(np.clip(float(budget_alpha), 0.0, 1.0))
    if alpha <= 0.0:
        return flow_values

    valid = valid_mask.to(dtype=torch.bool)
    active = active_mask.to(dtype=torch.bool)
    alloc_base = valid & active
    fga2 = flow_values[:, :, fga2_idx]
    fg2m = flow_values[:, :, fg2m_idx]
    fga3 = flow_values[:, :, fga3_idx]
    fg3m = flow_values[:, :, fg3m_idx]
    fta = flow_values[:, :, fta_idx]
    ftm = flow_values[:, :, ftm_idx]
    tov = flow_values[:, :, tov_idx]
    oreb = flow_values[:, :, oreb_idx]

    team_fga2 = torch.zeros((flow_values.shape[0], 2), dtype=flow_values.dtype, device=flow_values.device)
    team_fga3 = torch.zeros_like(team_fga2)
    team_fta = torch.zeros_like(team_fga2)
    team_tov = torch.zeros_like(team_fga2)
    team_oreb = torch.zeros_like(team_fga2)
    for side in (0, 1):
        side_mask = team_index.eq(side)
        elig = alloc_base & side_mask
        valid_side = valid & side_mask
        has_active = elig.any(dim=1, keepdim=True)
        elig = torch.where(has_active, elig, valid_side)
        side_f = elig.to(dtype=flow_values.dtype)
        team_fga2[:, side] = (fga2 * side_f).sum(dim=1)
        team_fga3[:, side] = (fga3 * side_f).sum(dim=1)
        team_fta[:, side] = (fta * side_f).sum(dim=1)
        team_tov[:, side] = (tov * side_f).sum(dim=1)
        team_oreb[:, side] = (oreb * side_f).sum(dim=1)

    team_fga = team_fga2 + team_fga3
    game_fga = team_fga.sum(dim=1, keepdim=True)
    game_fta = team_fta.sum(dim=1, keepdim=True)
    target_fga = team_opportunity_share * game_fga
    target_fta = team_opportunity_share * game_fta
    blended_fga = (1.0 - alpha) * team_fga + alpha * target_fga
    blended_fta = (1.0 - alpha) * team_fta + alpha * target_fta

    for side in (0, 1):
        side_mask = team_index.eq(side)
        elig = alloc_base & side_mask
        valid_side = valid & side_mask
        has_active = elig.any(dim=1, keepdim=True)
        elig = torch.where(has_active, elig, valid_side)
        current_fga = team_fga[:, side].clamp(min=1e-6).unsqueeze(1)
        current_fta = team_fta[:, side].clamp(min=1e-6).unsqueeze(1)
        fga_scale = torch.where(
            team_fga[:, side].unsqueeze(1) > 1e-6,
            blended_fga[:, side].unsqueeze(1) / current_fga,
            torch.ones_like(current_fga),
        )
        fta_scale = torch.where(
            team_fta[:, side].unsqueeze(1) > 1e-6,
            blended_fta[:, side].unsqueeze(1) / current_fta,
            torch.ones_like(current_fta),
        )
        new_fga2 = fga2 * fga_scale
        new_fg2m = torch.minimum(new_fga2, fg2m * fga_scale)
        new_fga3 = fga3 * fga_scale
        new_fg3m = torch.minimum(new_fga3, fg3m * fga_scale)
        new_fta = fta * fta_scale
        new_ftm = torch.minimum(new_fta, ftm * fta_scale)
        if preserve_possessions:
            side_f = side_mask.to(dtype=flow_values.dtype)
            side_poss = (
                team_fga[:, side]
                - team_oreb[:, side]
                + team_tov[:, side]
                + float(FTA_POSS_COEFF) * team_fta[:, side]
            )
            side_capacity = (side_poss + team_oreb[:, side]).clamp(min=0.0).unsqueeze(1)
            side_fga2 = (new_fga2 * side_f).sum(dim=1, keepdim=True)
            side_fga3 = (new_fga3 * side_f).sum(dim=1, keepdim=True)
            side_fta = (new_fta * side_f).sum(dim=1, keepdim=True)
            proposed_used = side_fga2 + side_fga3 + float(FTA_POSS_COEFF) * side_fta
            capacity_scale = torch.where(
                proposed_used > side_capacity + 1e-6,
                side_capacity / proposed_used.clamp(min=1e-6),
                torch.ones_like(side_capacity),
            )
            new_fga2 = new_fga2 * capacity_scale
            new_fg2m = torch.minimum(new_fga2, new_fg2m * capacity_scale)
            new_fga3 = new_fga3 * capacity_scale
            new_fg3m = torch.minimum(new_fga3, new_fg3m * capacity_scale)
            new_fta = new_fta * capacity_scale
            new_ftm = torch.minimum(new_fta, new_ftm * capacity_scale)
            side_target_tov = (
                side_poss.unsqueeze(1)
                + team_oreb[:, side].unsqueeze(1)
                - ((new_fga2 * side_f).sum(dim=1, keepdim=True) + (new_fga3 * side_f).sum(dim=1, keepdim=True))
                - float(FTA_POSS_COEFF) * (new_fta * side_f).sum(dim=1, keepdim=True)
            ).clamp(min=0.0)
            w_tov = _normalize_alloc_weights(tov, eligible_mask=elig)
            new_tov = w_tov * side_target_tov
        else:
            new_tov = tov
        fga2 = torch.where(side_mask, new_fga2, fga2)
        fg2m = torch.where(side_mask, new_fg2m, fg2m)
        fga3 = torch.where(side_mask, new_fga3, fga3)
        fg3m = torch.where(side_mask, new_fg3m, fg3m)
        fta = torch.where(side_mask, new_fta, fta)
        ftm = torch.where(side_mask, new_ftm, ftm)
        tov = torch.where(side_mask, new_tov, tov)

    out = flow_values.clone()
    out[:, :, fga2_idx] = fga2
    out[:, :, fg2m_idx] = fg2m
    out[:, :, fga3_idx] = fga3
    out[:, :, fg3m_idx] = fg3m
    out[:, :, fta_idx] = fta
    out[:, :, ftm_idx] = ftm
    out[:, :, tov_idx] = tov
    return out


def _swap_team_side_values(team_values: torch.Tensor) -> torch.Tensor:
    if team_values.ndim != 2 or team_values.shape[1] != 2:
        raise ValueError("team_values must have shape (N, 2)")
    return torch.stack([team_values[:, 1], team_values[:, 0]], dim=1)


def _rebound_budget_uses_rate(parameterization: str, channel: str) -> bool:
    mode = str(parameterization).strip().lower()
    if channel == "oreb":
        return mode in {"oreb_rate", "both_rate"}
    if channel == "dreb":
        return mode in {"dreb_rate", "both_rate"}
    raise ValueError(f"unsupported rebound budget channel: {channel!r}")


def _rebound_budget_is_residual_rate(parameterization: str, channel: str) -> bool:
    mode = str(parameterization).strip().lower()
    if channel == "oreb":
        return False
    if channel == "dreb":
        return mode in {"dreb_rate_residual"}
    raise ValueError(f"unsupported rebound budget channel: {channel!r}")


def _rebound_budget_is_deterministic(parameterization: str, channel: str) -> bool:
    mode = str(parameterization).strip().lower()
    if channel == "oreb":
        return False
    if channel == "dreb":
        return mode in {"dreb_deterministic"}
    raise ValueError(f"unsupported rebound budget channel: {channel!r}")


def _reconstruct_rebounds_from_heads(
    *,
    player_valid_mask: torch.Tensor,
    player_team_index: torch.Tensor,
    team_oreb_budget: torch.Tensor | None,
    team_dreb_budget: torch.Tensor | None,
    team_oreb_budget_gate: torch.Tensor | None,
    team_dreb_budget_gate: torch.Tensor | None,
    oreb_share_logits: torch.Tensor | None,
    dreb_share_logits: torch.Tensor | None,
    share_temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if (
        team_oreb_budget is None
        or team_dreb_budget is None
        or oreb_share_logits is None
        or dreb_share_logits is None
    ):
        return None
    if team_oreb_budget.ndim != 2 or team_oreb_budget.shape[1] != 2:
        raise ValueError("team_oreb_budget must have shape (N, 2)")
    if team_dreb_budget.shape != team_oreb_budget.shape:
        raise ValueError("team_dreb_budget must match team_oreb_budget shape")
    if oreb_share_logits.shape != player_valid_mask.shape or dreb_share_logits.shape != player_valid_mask.shape:
        raise ValueError("rebound share logits must align with player_valid_mask")
    if player_team_index.shape != player_valid_mask.shape:
        raise ValueError("player_team_index must align with player_valid_mask")

    temp = max(float(share_temperature), 1e-6)
    valid_mask = player_valid_mask.to(dtype=torch.bool)
    oreb_share = torch.zeros_like(oreb_share_logits)
    dreb_share = torch.zeros_like(dreb_share_logits)
    for side in (0, 1):
        side_mask = valid_mask & player_team_index.eq(side)
        if bool(side_mask.any()):
            side_oreb_logits = (oreb_share_logits / temp).masked_fill(~side_mask, float("-inf"))
            side_dreb_logits = (dreb_share_logits / temp).masked_fill(~side_mask, float("-inf"))
            oreb_share = oreb_share + torch.softmax(side_oreb_logits, dim=1) * side_mask.to(dtype=oreb_share_logits.dtype)
            dreb_share = dreb_share + torch.softmax(side_dreb_logits, dim=1) * side_mask.to(dtype=dreb_share_logits.dtype)
    team_oreb = torch.where(
        player_team_index.eq(0),
        team_oreb_budget[:, 0].unsqueeze(1),
        team_oreb_budget[:, 1].unsqueeze(1),
    )
    team_dreb = torch.where(
        player_team_index.eq(0),
        team_dreb_budget[:, 0].unsqueeze(1),
        team_dreb_budget[:, 1].unsqueeze(1),
    )
    return oreb_share * team_oreb, dreb_share * team_dreb


def _reconcile_rebounds_to_opportunity_budgets(
    *,
    flow_values: torch.Tensor,
    valid_mask: torch.Tensor,
    team_index: torch.Tensor,
    active_mask: torch.Tensor,
    flow_target_columns: list[str],
    team_oreb_budget: torch.Tensor | None,
    team_dreb_budget: torch.Tensor | None,
    team_oreb_budget_gate: torch.Tensor | None,
    team_dreb_budget_gate: torch.Tensor | None,
    oreb_share_logits: torch.Tensor | None,
    dreb_share_logits: torch.Tensor | None,
    share_alpha: float,
    share_temperature: float,
    reconcile_mode: str = "both",
    budget_parameterization: str = "absolute",
    dreb_deterministic_discount: float = 1.0,
    oreb_reconcile_use_flow_budget: bool = False,
    oreb_budget_blend_alpha: float = 1.0,
    dreb_budget_blend_alpha: float = 1.0,
) -> torch.Tensor:
    mode = str(reconcile_mode).strip().lower()
    reconcile_oreb = mode in {"both", "oreb_only"}
    reconcile_dreb = mode in {"both", "dreb_only"}
    deterministic_dreb_budget = reconcile_dreb and _rebound_budget_is_deterministic(
        budget_parameterization,
        "dreb",
    )
    flow_oreb_budget = reconcile_oreb and bool(oreb_reconcile_use_flow_budget)
    if not reconcile_oreb and not reconcile_dreb:
        return flow_values
    if reconcile_oreb and (oreb_share_logits is None or (team_oreb_budget is None and not flow_oreb_budget)):
        return flow_values
    if reconcile_dreb and (dreb_share_logits is None or (team_dreb_budget is None and not deterministic_dreb_budget)):
        return flow_values
    if flow_values.ndim != 3:
        raise ValueError("flow_values must have shape (N, 30, S)")
    if valid_mask.shape != team_index.shape or valid_mask.shape != active_mask.shape:
        raise ValueError("valid_mask/team_index/active_mask must align")
    if reconcile_oreb and not flow_oreb_budget and (team_oreb_budget.ndim != 2 or team_oreb_budget.shape[1] != 2):
        raise ValueError("team_oreb_budget must have shape (N, 2)")
    if (
        reconcile_dreb
        and not deterministic_dreb_budget
        and (team_dreb_budget.ndim != 2 or team_dreb_budget.shape[1] != 2)
    ):
        raise ValueError("team_dreb_budget must have shape (N, 2)")
    if (
        reconcile_oreb
        and not flow_oreb_budget
        and team_oreb_budget_gate is not None
        and team_oreb_budget_gate.shape != team_oreb_budget.shape
    ):
        raise ValueError("team_oreb_budget_gate must match team_oreb_budget shape")
    if reconcile_dreb and team_dreb_budget_gate is not None and (
        team_dreb_budget_gate.ndim != 2 or team_dreb_budget_gate.shape[1] != 2
    ):
        raise ValueError("team_dreb_budget_gate must have shape (N, 2)")
    if (
        reconcile_oreb
        and reconcile_dreb
        and not flow_oreb_budget
        and not deterministic_dreb_budget
        and team_dreb_budget.shape != team_oreb_budget.shape
    ):
        raise ValueError("team_dreb_budget must match team_oreb_budget shape")
    if reconcile_oreb and oreb_share_logits.shape != valid_mask.shape:
        raise ValueError("oreb share logits must align with valid_mask")
    if reconcile_dreb and dreb_share_logits.shape != valid_mask.shape:
        raise ValueError("rebound share logits must align with valid_mask")

    out = flow_values.clone()
    oreb_idx = _flow_idx(flow_target_columns, "oreb")
    dreb_idx = _flow_idx(flow_target_columns, "dreb")
    fga2_idx = _flow_idx(flow_target_columns, "fga2")
    fg2m_idx = _flow_idx(flow_target_columns, "fg2m")
    fga3_idx = _flow_idx(flow_target_columns, "fga3")
    fg3m_idx = _flow_idx(flow_target_columns, "fg3m")

    old_oreb = flow_values[:, :, oreb_idx].clone()
    old_dreb = flow_values[:, :, dreb_idx].clone()
    own_missed = (flow_values[:, :, fga2_idx] - flow_values[:, :, fg2m_idx]) + (
        flow_values[:, :, fga3_idx] - flow_values[:, :, fg3m_idx]
    )
    own_missed = torch.clamp(own_missed, min=0.0)
    own_missed_team = _team_sum_by_side(
        own_missed,
        valid_mask=valid_mask,
        team_index=team_index,
    )
    opp_missed_team = _swap_team_side_values(own_missed_team)
    flow_oreb_team = _team_sum_by_side(
        old_oreb,
        valid_mask=valid_mask,
        team_index=team_index,
    )
    opp_oreb_team = _swap_team_side_values(flow_oreb_team)
    flow_dreb_team = _team_sum_by_side(
        old_dreb,
        valid_mask=valid_mask,
        team_index=team_index,
    )
    use_oreb_rate = _rebound_budget_uses_rate(budget_parameterization, "oreb")
    use_dreb_rate = _rebound_budget_uses_rate(budget_parameterization, "dreb")
    use_dreb_residual_rate = _rebound_budget_is_residual_rate(budget_parameterization, "dreb")
    if reconcile_oreb and flow_oreb_budget:
        target_oreb_budget = torch.minimum(torch.clamp(flow_oreb_team, min=0.0), own_missed_team)
    elif reconcile_oreb and use_oreb_rate:
        target_oreb_budget = torch.clamp(team_oreb_budget, min=0.0, max=1.0) * own_missed_team
    elif reconcile_oreb:
        target_oreb_budget = torch.minimum(torch.clamp(team_oreb_budget, min=0.0), own_missed_team)
    else:
        target_oreb_budget = None
    flow_dreb_rate = torch.nan_to_num(
        flow_dreb_team / opp_missed_team.clamp(min=1.0),
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
    if reconcile_dreb and deterministic_dreb_budget:
        dreb_discount = float(np.clip(float(dreb_deterministic_discount), 0.0, 1.0))
        target_dreb_budget = torch.minimum(
            torch.clamp(opp_missed_team - opp_oreb_team, min=0.0) * dreb_discount,
            opp_missed_team,
        )
    elif reconcile_dreb and use_dreb_residual_rate:
        target_dreb_budget = torch.clamp(flow_dreb_rate + team_dreb_budget, min=0.0, max=1.0) * opp_missed_team
    elif reconcile_dreb and use_dreb_rate:
        target_dreb_budget = torch.clamp(team_dreb_budget, min=0.0, max=1.0) * opp_missed_team
    elif reconcile_dreb:
        target_dreb_budget = torch.minimum(torch.clamp(team_dreb_budget, min=0.0), opp_missed_team)
    else:
        target_dreb_budget = None
    oreb_budget_alpha = (
        team_oreb_budget_gate.to(device=own_missed_team.device, dtype=own_missed_team.dtype).clamp(min=0.0, max=1.0)
        if (team_oreb_budget_gate is not None and not flow_oreb_budget)
        else torch.full_like(own_missed_team, float(np.clip(float(oreb_budget_blend_alpha), 0.0, 1.0)))
    )
    dreb_budget_alpha = (
        team_dreb_budget_gate.to(device=opp_missed_team.device, dtype=opp_missed_team.dtype).clamp(min=0.0, max=1.0)
        if team_dreb_budget_gate is not None
        else torch.full_like(opp_missed_team, float(np.clip(float(dreb_budget_blend_alpha), 0.0, 1.0)))
    )
    blended_oreb_budget = (
        torch.minimum(
            (1.0 - oreb_budget_alpha) * torch.minimum(torch.clamp(flow_oreb_team, min=0.0), own_missed_team)
            + oreb_budget_alpha * torch.minimum(torch.clamp(target_oreb_budget, min=0.0), own_missed_team),
            own_missed_team,
        )
        if reconcile_oreb and not flow_oreb_budget
        else torch.minimum(torch.clamp(flow_oreb_team, min=0.0), own_missed_team)
        if reconcile_oreb
        else None
    )
    blended_dreb_budget = (
        torch.minimum(
            (1.0 - dreb_budget_alpha) * torch.minimum(torch.clamp(flow_dreb_team, min=0.0), opp_missed_team)
            + dreb_budget_alpha * torch.minimum(torch.clamp(target_dreb_budget, min=0.0), opp_missed_team),
            opp_missed_team,
        )
        if reconcile_dreb
        else None
    )

    alpha = float(np.clip(float(share_alpha), 0.0, 1.0))
    temp = max(float(share_temperature), 1e-6)
    valid = valid_mask.to(dtype=torch.bool)
    active = active_mask.to(dtype=torch.bool)
    alloc_base = valid & active
    new_oreb_all = old_oreb.clone()
    new_dreb_all = old_dreb.clone()
    for side in (0, 1):
        side_mask = team_index.eq(side)
        elig = alloc_base & side_mask
        valid_side = valid & side_mask
        has_active = elig.any(dim=1, keepdim=True)
        elig = torch.where(has_active, elig, valid_side)

        side_f = side_mask.to(dtype=out.dtype)
        if reconcile_oreb:
            flow_oreb_weights = _normalize_alloc_weights(old_oreb, eligible_mask=elig)
            factorized_oreb_weights = _softmax_alloc_weights(oreb_share_logits / temp, eligible_mask=elig)
            oreb_weights = _normalize_alloc_weights(
                (1.0 - alpha) * flow_oreb_weights + alpha * factorized_oreb_weights,
                eligible_mask=elig,
            )
            new_oreb = oreb_weights * blended_oreb_budget[:, side].unsqueeze(1)
            new_oreb_all = new_oreb_all * (1.0 - side_f) + new_oreb * side_f
        if reconcile_dreb:
            flow_dreb_weights = _normalize_alloc_weights(old_dreb, eligible_mask=elig)
            factorized_dreb_weights = _softmax_alloc_weights(dreb_share_logits / temp, eligible_mask=elig)
            dreb_weights = _normalize_alloc_weights(
                (1.0 - alpha) * flow_dreb_weights + alpha * factorized_dreb_weights,
                eligible_mask=elig,
            )
            new_dreb = dreb_weights * blended_dreb_budget[:, side].unsqueeze(1)
            new_dreb_all = new_dreb_all * (1.0 - side_f) + new_dreb * side_f
    if reconcile_oreb:
        out[:, :, oreb_idx] = new_oreb_all
    if reconcile_dreb:
        out[:, :, dreb_idx] = new_dreb_all
    return out


def _normalize_alloc_weights(
    weights: torch.Tensor,
    *,
    eligible_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Normalize non-negative player weights within each world row.

    Falls back to uniform distribution over eligible players when weight mass is zero.
    """
    elig = eligible_mask.to(dtype=torch.bool)
    w = torch.where(elig, torch.clamp(weights, min=0.0), torch.zeros_like(weights))
    denom = w.sum(dim=1, keepdim=True)
    elig_count = elig.to(dtype=w.dtype).sum(dim=1, keepdim=True).clamp(min=1.0)
    uniform = elig.to(dtype=w.dtype) / elig_count
    return torch.where(denom > float(eps), w / denom.clamp(min=float(eps)), uniform)


def _softmax_alloc_weights(
    logits: torch.Tensor,
    *,
    eligible_mask: torch.Tensor,
) -> torch.Tensor:
    """Softmax allocation over eligible players with uniform fallback."""
    elig = eligible_mask.to(dtype=torch.bool)
    masked_logits = logits.masked_fill(~elig, -1e9)
    probs = torch.softmax(masked_logits, dim=1)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    denom = probs.sum(dim=1, keepdim=True)
    elig_count = elig.to(dtype=probs.dtype).sum(dim=1, keepdim=True).clamp(min=1.0)
    uniform = elig.to(dtype=probs.dtype) / elig_count
    return torch.where(denom > 1e-8, probs / denom.clamp(min=1e-8), uniform)


def _reweight_top_usage_alloc_weights(
    weights: torch.Tensor,
    *,
    eligible_mask: torch.Tensor,
    top1_scale: float = 1.0,
    top2_scale: float = 1.0,
    rank_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reweight top implied-usage players and renormalize.

    This is intended only as a research probe for decode-time share compression.
    """
    base = _normalize_alloc_weights(weights, eligible_mask=eligible_mask)
    if abs(float(top1_scale) - 1.0) < 1e-8 and abs(float(top2_scale) - 1.0) < 1e-8:
        return base

    elig = eligible_mask.to(dtype=torch.bool)
    if not bool(elig.any()):
        return base

    order_source = (
        _normalize_alloc_weights(rank_weights, eligible_mask=eligible_mask)
        if rank_weights is not None
        else base
    )
    scores = order_source.masked_fill(~elig, float("-inf"))
    order = torch.argsort(scores, dim=1, descending=True)

    scales = torch.ones_like(base)
    top1_idx = order[:, :1]
    scales.scatter_(1, top1_idx, scales.gather(1, top1_idx) * float(top1_scale))

    top2_idx = order[:, 1:2]
    top2_valid = elig.sum(dim=1, keepdim=True) > 1
    top2_multiplier = torch.where(
        top2_valid,
        torch.full(top2_idx.shape, float(top2_scale), dtype=base.dtype, device=base.device),
        torch.ones(top2_idx.shape, dtype=base.dtype, device=base.device),
    )
    scales.scatter_(1, top2_idx, scales.gather(1, top2_idx) * top2_multiplier)
    return _normalize_alloc_weights(base * scales, eligible_mask=eligible_mask)


def _apply_forced_active_minutes_floor(
    *,
    minutes: torch.Tensor,
    valid_mask: torch.Tensor,
    team_index: torch.Tensor,
    forced_active_mask: torch.Tensor,
    forced_minutes_anchor: torch.Tensor,
    floor_ratio: float,
    floor_min: float,
    floor_max: float,
    team_total_minutes: float = 240.0,
    max_minutes_per_player: float = 48.0,
) -> torch.Tensor:
    """Apply props-anchored floors for forced-active players while preserving team totals."""
    if minutes.ndim != 2:
        raise ValueError("minutes must have shape (N,30)")
    if valid_mask.shape != minutes.shape or team_index.shape != minutes.shape:
        raise ValueError("valid_mask/team_index must have shape (N,30)")
    if forced_active_mask.shape != minutes.shape or forced_minutes_anchor.shape != minutes.shape:
        raise ValueError("forced_active_mask/forced_minutes_anchor must have shape (N,30)")

    out = torch.clamp(minutes, min=0.0, max=float(max_minutes_per_player))
    valid = valid_mask.to(dtype=torch.bool)
    forced = forced_active_mask.to(dtype=torch.bool) & valid

    anchor = torch.clamp(forced_minutes_anchor.to(dtype=out.dtype), min=0.0, max=float(max_minutes_per_player))
    floor_vals = torch.clamp(
        anchor * float(floor_ratio),
        min=float(floor_min),
        max=float(floor_max),
    )
    floor_vals = torch.clamp(floor_vals, min=0.0, max=float(max_minutes_per_player))
    # Apply only when we actually have a props-implied anchor.
    floor_vals = torch.where(forced & anchor.gt(0.0), floor_vals, torch.zeros_like(floor_vals))
    out = torch.where(valid, out, torch.zeros_like(out))

    team_total = float(team_total_minutes)
    for side in (0, 1):
        side_mask = valid & team_index.eq(side)
        side_f = side_mask.to(dtype=out.dtype)

        floor_side = floor_vals * side_f
        # Keep floors feasible: if team floor mass exceeds 240, down-scale proportionally.
        floor_sum = floor_side.sum(dim=1, keepdim=True)
        floor_scale = torch.where(
            floor_sum > team_total,
            floor_sum.new_full(floor_sum.shape, team_total) / floor_sum.clamp(min=1e-8),
            torch.ones_like(floor_sum),
        )
        floor_side = torch.clamp(floor_side * floor_scale, min=0.0, max=float(max_minutes_per_player))

        out = torch.maximum(out, floor_side)
        out = torch.clamp(out, min=0.0, max=float(max_minutes_per_player))

        team_sum = (out * side_f).sum(dim=1, keepdim=True)
        excess = torch.clamp(team_sum - team_total, min=0.0)
        reducible = torch.clamp(out - floor_side, min=0.0) * side_f
        reducible_sum = reducible.sum(dim=1, keepdim=True)
        frac = torch.where(
            reducible_sum > 1e-8,
            excess / reducible_sum.clamp(min=1e-8),
            torch.zeros_like(excess),
        )
        frac = torch.clamp(frac, min=0.0, max=1.0)
        out = out - reducible * frac
        out = torch.clamp(out, min=0.0, max=float(max_minutes_per_player))

    return torch.where(valid, out, torch.zeros_like(out))


def _align_flow_to_backbone_budgets(
    *,
    flow_values: torch.Tensor,
    valid_mask: torch.Tensor,
    team_index: torch.Tensor,
    active_mask: torch.Tensor,
    flow_target_columns: list[str],
    backbone_fga: torch.Tensor,
    backbone_fta: torch.Tensor,
    backbone_tov: torch.Tensor,
    backbone_oreb: torch.Tensor,
    backbone_three_pa_share: torch.Tensor | None,
    eff_alpha_ft: torch.Tensor | None = None,
    eff_beta_ft: torch.Tensor | None = None,
    eff_alpha_fg2: torch.Tensor | None = None,
    eff_beta_fg2: torch.Tensor | None = None,
    eff_alpha_fg3: torch.Tensor | None = None,
    eff_beta_fg3: torch.Tensor | None = None,
    make_model_config: MakeModelConfig | None = None,
    usage_share_logits: torch.Tensor | None = None,
    allocation_source: str = "emergent",
    allocation_blend_alpha: float = 0.5,
    allocation_top_usage_top1_scale: float = 1.0,
    allocation_top_usage_top2_scale: float = 1.0,
) -> torch.Tensor:
    """Align player-level flow outputs to sampled backbone team event budgets.

    This enforces sampled per-team totals for FGA/FTA/TOV/OREB while preserving
    player-level share patterns from the sampled flow output.
    """
    if flow_values.ndim != 3:
        raise ValueError("flow_values must have shape (N, 30, S)")
    if valid_mask.shape != team_index.shape or valid_mask.shape != active_mask.shape:
        raise ValueError("valid_mask/team_index/active_mask must have shape (N, 30)")
    if backbone_fga.ndim != 2 or backbone_fga.shape[1] != 2:
        raise ValueError("backbone_fga/backbone_fta/backbone_tov/backbone_oreb must have shape (N, 2)")
    alloc_source = str(allocation_source).strip().lower()
    if alloc_source not in {"emergent", "usage_head", "blend"}:
        raise ValueError(f"unsupported allocation_source: {allocation_source}")
    if usage_share_logits is not None and usage_share_logits.shape[:2] != flow_values.shape[:2]:
        raise ValueError("usage_share_logits must have shape (N, 30, K)")
    if usage_share_logits is not None and usage_share_logits.shape[2] < 3:
        raise ValueError("usage_share_logits must include at least 3 targets: fga/fta/tov")
    blend_alpha = float(np.clip(float(allocation_blend_alpha), 0.0, 1.0))
    use_usage = alloc_source in {"usage_head", "blend"} and usage_share_logits is not None

    out = flow_values.clone()
    valid = valid_mask.to(dtype=torch.bool)
    active = active_mask.to(dtype=torch.bool)
    # Prefer active players for allocation; fallback to valid players if needed.
    alloc_base = valid & active

    fga2_idx = _flow_idx(flow_target_columns, "fga2")
    fg2m_idx = _flow_idx(flow_target_columns, "fg2m")
    fga3_idx = _flow_idx(flow_target_columns, "fga3")
    fg3m_idx = _flow_idx(flow_target_columns, "fg3m")
    fta_idx = _flow_idx(flow_target_columns, "fta")
    ftm_idx = _flow_idx(flow_target_columns, "ftm")
    oreb_idx = _flow_idx(flow_target_columns, "oreb")
    tov_idx = _flow_idx(flow_target_columns, "tov")

    old_fga2 = out[:, :, fga2_idx]
    old_fga3 = out[:, :, fga3_idx]
    old_fta = out[:, :, fta_idx]
    old_fg2m = out[:, :, fg2m_idx]
    old_fg3m = out[:, :, fg3m_idx]
    old_ftm = out[:, :, ftm_idx]

    fg2_pct = torch.where(old_fga2 > 1e-8, old_fg2m / old_fga2.clamp(min=1e-8), torch.zeros_like(old_fga2))
    fg3_pct = torch.where(old_fga3 > 1e-8, old_fg3m / old_fga3.clamp(min=1e-8), torch.zeros_like(old_fga3))
    ft_pct = torch.where(old_fta > 1e-8, old_ftm / old_fta.clamp(min=1e-8), torch.zeros_like(old_fta))
    fg2_pct = torch.clamp(fg2_pct, min=0.0, max=1.0)
    fg3_pct = torch.clamp(fg3_pct, min=0.0, max=1.0)
    ft_pct = torch.clamp(ft_pct, min=0.0, max=1.0)

    cfg = make_model_config or MakeModelConfig()
    if (
        str(cfg.mode).strip().lower() == "legacy"
        and bool(cfg.legacy_use_efficiency_mean)
        and eff_alpha_fg2 is not None
        and eff_beta_fg2 is not None
        and eff_alpha_fg3 is not None
        and eff_beta_fg3 is not None
        and eff_alpha_ft is not None
        and eff_beta_ft is not None
    ):
        fg2_pct = eff_alpha_fg2 / (eff_alpha_fg2 + eff_beta_fg2).clamp(min=1e-8)
        fg3_pct = eff_alpha_fg3 / (eff_alpha_fg3 + eff_beta_fg3).clamp(min=1e-8)
        ft_pct = eff_alpha_ft / (eff_alpha_ft + eff_beta_ft).clamp(min=1e-8)
        fg2_pct = torch.clamp(fg2_pct, min=0.0, max=1.0)
        fg3_pct = torch.clamp(fg3_pct, min=0.0, max=1.0)
        ft_pct = torch.clamp(ft_pct, min=0.0, max=1.0)

    for side in (0, 1):
        side_mask = team_index.eq(side)
        elig = alloc_base & side_mask
        valid_side = valid & side_mask

        # If active mask is empty for a row, fallback to valid players for that row.
        has_active = elig.any(dim=1, keepdim=True)
        elig = torch.where(has_active, elig, valid_side)

        budget_fga = torch.clamp(backbone_fga[:, side], min=0.0)
        budget_fta = torch.clamp(backbone_fta[:, side], min=0.0)
        budget_tov = torch.clamp(backbone_tov[:, side], min=0.0)
        budget_oreb = torch.clamp(backbone_oreb[:, side], min=0.0)

        if backbone_three_pa_share is not None:
            share3 = torch.clamp(backbone_three_pa_share[:, side], min=0.0, max=1.0)
        else:
            team_fga2 = (old_fga2 * valid_side.to(dtype=old_fga2.dtype)).sum(dim=1)
            team_fga3 = (old_fga3 * valid_side.to(dtype=old_fga3.dtype)).sum(dim=1)
            share3 = torch.where(
                (team_fga2 + team_fga3) > 1e-8,
                team_fga3 / (team_fga2 + team_fga3).clamp(min=1e-8),
                torch.full_like(team_fga2, 0.38),
            )
            share3 = torch.clamp(share3, min=0.0, max=1.0)

        budget_fga3 = budget_fga * share3
        budget_fga2 = budget_fga - budget_fga3

        w_fga2 = _normalize_alloc_weights(old_fga2, eligible_mask=elig)
        w_fga3 = _normalize_alloc_weights(old_fga3, eligible_mask=elig)
        w_fta = _normalize_alloc_weights(old_fta, eligible_mask=elig)
        w_tov = _normalize_alloc_weights(out[:, :, tov_idx], eligible_mask=elig)
        w_oreb = _normalize_alloc_weights(out[:, :, oreb_idx], eligible_mask=elig)

        if use_usage and usage_share_logits is not None:
            w_usage_fga = _softmax_alloc_weights(usage_share_logits[:, :, 0], eligible_mask=elig)
            w_usage_fta = _softmax_alloc_weights(usage_share_logits[:, :, 1], eligible_mask=elig)
            w_usage_tov = _softmax_alloc_weights(usage_share_logits[:, :, 2], eligible_mask=elig)
            if alloc_source == "usage_head":
                w_fga2 = w_usage_fga
                w_fga3 = w_usage_fga
                w_fta = w_usage_fta
                w_tov = w_usage_tov
            else:
                w_fga2 = (1.0 - blend_alpha) * w_fga2 + blend_alpha * w_usage_fga
                w_fga3 = (1.0 - blend_alpha) * w_fga3 + blend_alpha * w_usage_fga
                w_fta = (1.0 - blend_alpha) * w_fta + blend_alpha * w_usage_fta
                w_tov = (1.0 - blend_alpha) * w_tov + blend_alpha * w_usage_tov
                w_fga2 = _normalize_alloc_weights(w_fga2, eligible_mask=elig)
                w_fga3 = _normalize_alloc_weights(w_fga3, eligible_mask=elig)
                w_fta = _normalize_alloc_weights(w_fta, eligible_mask=elig)
                w_tov = _normalize_alloc_weights(w_tov, eligible_mask=elig)

        usage_rank_weights = _normalize_alloc_weights(
            w_fga2 + w_fga3 + w_fta + w_tov,
            eligible_mask=elig,
        )
        w_fga2 = _reweight_top_usage_alloc_weights(
            w_fga2,
            eligible_mask=elig,
            top1_scale=float(allocation_top_usage_top1_scale),
            top2_scale=float(allocation_top_usage_top2_scale),
            rank_weights=usage_rank_weights,
        )
        w_fga3 = _reweight_top_usage_alloc_weights(
            w_fga3,
            eligible_mask=elig,
            top1_scale=float(allocation_top_usage_top1_scale),
            top2_scale=float(allocation_top_usage_top2_scale),
            rank_weights=usage_rank_weights,
        )
        w_fta = _reweight_top_usage_alloc_weights(
            w_fta,
            eligible_mask=elig,
            top1_scale=float(allocation_top_usage_top1_scale),
            top2_scale=float(allocation_top_usage_top2_scale),
            rank_weights=usage_rank_weights,
        )
        w_tov = _reweight_top_usage_alloc_weights(
            w_tov,
            eligible_mask=elig,
            top1_scale=float(allocation_top_usage_top1_scale),
            top2_scale=float(allocation_top_usage_top2_scale),
            rank_weights=usage_rank_weights,
        )

        new_fga2 = w_fga2 * budget_fga2.unsqueeze(1)
        new_fga3 = w_fga3 * budget_fga3.unsqueeze(1)
        new_fta = w_fta * budget_fta.unsqueeze(1)
        new_tov = w_tov * budget_tov.unsqueeze(1)
        new_oreb = w_oreb * budget_oreb.unsqueeze(1)

        side_f = side_mask.to(dtype=out.dtype)
        out[:, :, fga2_idx] = out[:, :, fga2_idx] * (1.0 - side_f) + new_fga2 * side_f
        out[:, :, fga3_idx] = out[:, :, fga3_idx] * (1.0 - side_f) + new_fga3 * side_f
        out[:, :, fta_idx] = out[:, :, fta_idx] * (1.0 - side_f) + new_fta * side_f
        out[:, :, tov_idx] = out[:, :, tov_idx] * (1.0 - side_f) + new_tov * side_f
        out[:, :, oreb_idx] = out[:, :, oreb_idx] * (1.0 - side_f) + new_oreb * side_f

    mode = str(cfg.mode).strip().lower()
    if mode not in {"legacy", "beta_binomial_ft", "beta_binomial_fg", "beta_binomial_all"}:
        raise ValueError(f"unsupported make_model mode: {cfg.mode}")

    def _legacy_makes(attempts: torch.Tensor, rates: torch.Tensor) -> torch.Tensor:
        return torch.minimum(attempts, rates * attempts)

    def _sample_beta_binomial_makes(
        *,
        attempts: torch.Tensor,
        rates: torch.Tensor,
        prior_mean: float,
        prior_strength: float,
        concentration: float,
        alpha_pred: torch.Tensor | None = None,
        beta_pred: torch.Tensor | None = None,
    ) -> torch.Tensor:
        eps = 1e-4
        clipped_attempts = torch.clamp(attempts, min=0.0)
        if alpha_pred is not None and beta_pred is not None and bool(cfg.use_learned_efficiency):
            alpha = torch.clamp(alpha_pred, min=eps)
            beta = torch.clamp(beta_pred, min=eps)
        else:
            clipped_rates = torch.clamp(rates, min=eps, max=1.0 - eps)
            pm = float(np.clip(prior_mean, eps, 1.0 - eps))
            ps = max(float(prior_strength), eps)
            conc = max(float(concentration), eps)

            prior_alpha = clipped_attempts.new_full(clipped_attempts.shape, pm * ps)
            prior_beta = clipped_attempts.new_full(clipped_attempts.shape, (1.0 - pm) * ps)
            alpha = torch.clamp(prior_alpha + clipped_rates * conc, min=eps)
            beta = torch.clamp(prior_beta + (1.0 - clipped_rates) * conc, min=eps)

        n_floor = torch.floor(clipped_attempts)
        frac = torch.clamp(clipped_attempts - n_floor, min=0.0, max=1.0)
        n_int = n_floor + torch.bernoulli(frac)

        # Beta(alpha,beta) sample via Gamma ratio to avoid per-call distribution object overhead.
        gamma_alpha = torch._standard_gamma(alpha)
        gamma_beta = torch._standard_gamma(beta)
        p = gamma_alpha / (gamma_alpha + gamma_beta).clamp(min=eps)
        p = torch.clamp(p, min=eps, max=1.0 - eps)
        makes_int = torch.binomial(n_int, p)
        scale = torch.where(
            n_int > 0.0,
            clipped_attempts / n_int.clamp(min=1.0),
            torch.zeros_like(clipped_attempts),
        )
        makes = makes_int * scale
        return torch.minimum(clipped_attempts, torch.clamp(makes, min=0.0))

    use_beta_ft = mode in {"beta_binomial_ft", "beta_binomial_all"}
    use_beta_fg = mode in {"beta_binomial_fg", "beta_binomial_all"}
    if use_beta_fg:
        out[:, :, fg2m_idx] = _sample_beta_binomial_makes(
            attempts=out[:, :, fga2_idx],
            rates=fg2_pct,
            prior_mean=float(cfg.bb_fg2_prior_mean),
            prior_strength=float(cfg.bb_fg2_prior_strength),
            concentration=float(cfg.bb_fg2_concentration),
            alpha_pred=eff_alpha_fg2,
            beta_pred=eff_beta_fg2,
        )
        out[:, :, fg3m_idx] = _sample_beta_binomial_makes(
            attempts=out[:, :, fga3_idx],
            rates=fg3_pct,
            prior_mean=float(cfg.bb_fg3_prior_mean),
            prior_strength=float(cfg.bb_fg3_prior_strength),
            concentration=float(cfg.bb_fg3_concentration),
            alpha_pred=eff_alpha_fg3,
            beta_pred=eff_beta_fg3,
        )
    else:
        out[:, :, fg2m_idx] = _legacy_makes(out[:, :, fga2_idx], fg2_pct)
        out[:, :, fg3m_idx] = _legacy_makes(out[:, :, fga3_idx], fg3_pct)

    if use_beta_ft:
        out[:, :, ftm_idx] = _sample_beta_binomial_makes(
            attempts=out[:, :, fta_idx],
            rates=ft_pct,
            prior_mean=float(cfg.bb_ft_prior_mean),
            prior_strength=float(cfg.bb_ft_prior_strength),
            concentration=float(cfg.bb_ft_concentration),
            alpha_pred=eff_alpha_ft,
            beta_pred=eff_beta_ft,
        )
    else:
        out[:, :, ftm_idx] = _legacy_makes(out[:, :, fta_idx], ft_pct)
    return torch.clamp(out, min=0.0)


def check_world_contracts(
    *,
    minutes: torch.Tensor,
    flow_values: torch.Tensor,
    valid_mask: torch.Tensor,
    team_index: torch.Tensor,
    flow_target_columns: list[str],
    active_mask: torch.Tensor | None = None,
    tol: float = 1e-4,
) -> dict[str, int]:
    if minutes.ndim != 2 or valid_mask.shape != minutes.shape or team_index.shape != minutes.shape:
        raise ValueError("minutes/valid_mask/team_index must have shape (N,30)")
    if flow_values.ndim != 3 or flow_values.shape[:2] != minutes.shape:
        raise ValueError("flow_values must have shape (N,30,S)")
    if active_mask is not None and active_mask.shape != minutes.shape:
        raise ValueError("active_mask must have shape (N,30)")

    valid = valid_mask.to(dtype=torch.bool)
    minutes_valid = minutes * valid.to(dtype=minutes.dtype)
    flow_valid = flow_values * valid.unsqueeze(-1).to(dtype=flow_values.dtype)

    metrics_t: dict[str, torch.Tensor] = {
        "minutes_negative": (minutes_valid < -float(tol)).sum(),
        "minutes_over_48": (minutes_valid > (48.0 + float(tol))).sum(),
        "invalid_player_nonzero_minutes": ((~valid) & (minutes.abs() > float(tol))).sum(),
        "negative_stats": (flow_valid < -float(tol)).sum(),
    }
    team0_mask = valid & (team_index == 0)
    team1_mask = valid & (team_index == 1)
    team0_sum = (minutes * team0_mask.to(dtype=minutes.dtype)).sum(dim=1)
    team1_sum = (minutes * team1_mask.to(dtype=minutes.dtype)).sum(dim=1)
    metrics_t["team_minutes_not_240"] = (
        (team0_sum - 240.0).abs().gt(float(tol)).sum()
        + (team1_sum - 240.0).abs().gt(float(tol)).sum()
    )

    fg2m_idx = _flow_idx(flow_target_columns, "fg2m")
    fga2_idx = _flow_idx(flow_target_columns, "fga2")
    fg3m_idx = _flow_idx(flow_target_columns, "fg3m")
    fga3_idx = _flow_idx(flow_target_columns, "fga3")
    ftm_idx = _flow_idx(flow_target_columns, "ftm")
    fta_idx = _flow_idx(flow_target_columns, "fta")

    metrics_t["fg2m_gt_fga2"] = (
        flow_valid[..., fg2m_idx] > flow_valid[..., fga2_idx] + float(tol)
    ).sum()
    metrics_t["fg3m_gt_fga3"] = (
        flow_valid[..., fg3m_idx] > flow_valid[..., fga3_idx] + float(tol)
    ).sum()
    metrics_t["ftm_gt_fta"] = (
        flow_valid[..., ftm_idx] > flow_valid[..., fta_idx] + float(tol)
    ).sum()
    if active_mask is not None:
        inactive = (~active_mask.to(dtype=torch.bool)) & valid
        inactive_nonzero_stats = (
            flow_values.abs() * inactive.unsqueeze(-1).to(dtype=flow_values.dtype)
        ) > float(tol)
        metrics_t["inactive_nonzero_stats"] = inactive_nonzero_stats.sum()
        metrics_t["inactive_nonzero_fpts_proxy"] = (
            flow_values.sum(dim=-1).abs() * inactive.to(dtype=flow_values.dtype) > float(tol)
        ).sum()

    metric_keys = list(metrics_t.keys())
    metric_vals = torch.stack(
        [metrics_t[key].to(dtype=torch.int64, device=minutes.device) for key in metric_keys], dim=0
    ).cpu()
    out: dict[str, int] = {
        key: int(val) for key, val in zip(metric_keys, metric_vals.tolist(), strict=False)
    }
    out["total_violations"] = int(sum(out.values()))
    return out


def check_possession_symmetry(
    *,
    flow_values: torch.Tensor,
    valid_mask: torch.Tensor,
    team_index: torch.Tensor,
    flow_target_columns: list[str],
) -> dict[str, float]:
    """Compute possession symmetry diagnostics from sampled world stats.

    Returns a dict with:
      - poss_home_mean, poss_away_mean: average team possessions
      - poss_delta_abs_mean: mean |home - away| possession gap
      - poss_delta_abs_p95: p95 |home - away| possession gap
      - poss_delta_abs_max: max |home - away| possession gap
    """
    if flow_values.ndim != 3:
        raise ValueError("flow_values must have shape (N, 30, S)")

    valid = valid_mask.to(dtype=torch.bool)
    fga2_idx = _flow_idx(flow_target_columns, "fga2")
    fga3_idx = _flow_idx(flow_target_columns, "fga3")
    fta_idx = _flow_idx(flow_target_columns, "fta")
    oreb_idx = _flow_idx(flow_target_columns, "oreb")
    tov_idx = _flow_idx(flow_target_columns, "tov")

    poss_list: list[torch.Tensor] = []
    for side in (0, 1):
        mask = valid & (team_index == side)
        mask_f = mask.unsqueeze(-1).to(dtype=flow_values.dtype)
        fv = flow_values * mask_f
        fga = fv[:, :, fga2_idx].sum(dim=1) + fv[:, :, fga3_idx].sum(dim=1)
        fta = fv[:, :, fta_idx].sum(dim=1)
        oreb = fv[:, :, oreb_idx].sum(dim=1)
        tov = fv[:, :, tov_idx].sum(dim=1)
        poss = fga - oreb + tov + FTA_POSS_COEFF * fta
        poss_list.append(poss)

    poss_home, poss_away = poss_list
    delta = (poss_home - poss_away).abs()

    return {
        "poss_home_mean": float(poss_home.mean().item()),
        "poss_away_mean": float(poss_away.mean().item()),
        "poss_delta_abs_mean": float(delta.mean().item()),
        "poss_delta_abs_p95": float(torch.quantile(delta, 0.95).item()) if delta.numel() > 0 else 0.0,
        "poss_delta_abs_max": float(delta.max().item()) if delta.numel() > 0 else 0.0,
    }


def _compute_dk_fpts(
    *,
    pts: torch.Tensor,
    reb: torch.Tensor,
    ast: torch.Tensor,
    stl: torch.Tensor,
    blk: torch.Tensor,
    tov: torch.Tensor,
) -> torch.Tensor:
    base = pts + 1.25 * reb + 1.5 * ast + 2.0 * stl + 2.0 * blk - 0.5 * tov
    qualifying = torch.stack([pts, reb, ast, stl, blk], dim=-1).ge(10.0).sum(dim=-1)
    dd_bonus = (qualifying == 2).to(dtype=base.dtype) * 1.5
    td_bonus = (qualifying >= 3).to(dtype=base.dtype) * 3.0
    return base + dd_bonus + td_bonus


def _build_world_rows(
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
    if bsz <= 0 or n_worlds <= 0:
        return pd.DataFrame()

    valid = batch["player_valid_mask"].cpu().numpy().astype(bool)
    player_ids = batch["player_ids"].cpu().numpy().astype(np.int64)
    team_ids = batch["team_ids"].cpu().numpy().astype(np.int64)
    game_id_norm = np.asarray([str(v) for v in batch["game_id_norm"]], dtype=object)  # type: ignore[index]
    game_dates = np.asarray([str(v) for v in batch["game_date"]], dtype=object)  # type: ignore[index]
    game_ids = np.asarray([int(v) for v in game_id_norm], dtype=np.int64)

    mins_np = minutes.cpu().numpy()
    active_np = active_mask.cpu().numpy().astype(np.int8)
    flow_np = flow_values.cpu().numpy()

    valid_flat = valid.reshape(bsz, -1)
    players_flat = player_ids.reshape(bsz, -1)
    team_flat = np.repeat(team_ids.astype(np.int64), repeats=player_ids.shape[-1], axis=1)
    n_players_total = int(valid_flat.shape[1])

    # Expand batch metadata to one row per (batch, world, player), then mask invalid players.
    bw_game_idx = np.repeat(np.arange(bsz, dtype=np.int64), n_worlds)
    bw_world_idx = np.tile(np.arange(n_worlds, dtype=np.int64), bsz)
    valid_mask = np.repeat(valid_flat, repeats=n_worlds, axis=0).reshape(-1)
    if not bool(valid_mask.any()):
        return pd.DataFrame()

    world_idx_all = np.repeat(world_offset + bw_world_idx, repeats=n_players_total)
    game_idx_all = np.repeat(bw_game_idx, repeats=n_players_total)
    player_id_all = np.repeat(players_flat, repeats=n_worlds, axis=0).reshape(-1)
    team_id_all = np.repeat(team_flat, repeats=n_worlds, axis=0).reshape(-1)

    idx = {name: _flow_idx(flow_target_columns, name) for name in FLOW_TARGET_COLUMNS_V1}
    pf_idx = flow_target_columns.index("pf") if "pf" in flow_target_columns else None
    flow_flat = flow_np.reshape(-1, flow_np.shape[-1])

    fga2 = flow_flat[:, idx["fga2"]]
    fg2m = flow_flat[:, idx["fg2m"]]
    fga3 = flow_flat[:, idx["fga3"]]
    fg3m = flow_flat[:, idx["fg3m"]]
    fta = flow_flat[:, idx["fta"]]
    ftm = flow_flat[:, idx["ftm"]]
    oreb = flow_flat[:, idx["oreb"]]
    dreb = flow_flat[:, idx["dreb"]]
    ast = flow_flat[:, idx["ast"]]
    stl = flow_flat[:, idx["stl"]]
    blk = flow_flat[:, idx["blk"]]
    tov = flow_flat[:, idx["tov"]]
    pf = flow_flat[:, int(pf_idx)] if pf_idx is not None else np.zeros_like(fga2)
    fga = fga2 + fga3
    fgm = fg2m + fg3m
    pts = np.float32(2.0) * fg2m + np.float32(3.0) * fg3m + ftm
    reb = oreb + dreb

    # Numpy vectorized DK scoring to avoid per-player torch roundtrips.
    dk_base = (
        pts
        + np.float32(1.25) * reb
        + np.float32(1.5) * ast
        + np.float32(2.0) * stl
        + np.float32(2.0) * blk
        - np.float32(0.5) * tov
    )
    qualifying = (
        (pts >= 10.0).astype(np.int8)
        + (reb >= 10.0).astype(np.int8)
        + (ast >= 10.0).astype(np.int8)
        + (stl >= 10.0).astype(np.int8)
        + (blk >= 10.0).astype(np.int8)
    )
    dk = dk_base + (qualifying == 2).astype(dk_base.dtype) * np.float32(1.5) + (
        qualifying >= 3
    ).astype(dk_base.dtype) * np.float32(3.0)

    mins_flat = mins_np.reshape(-1)
    active_flat = active_np.reshape(-1)
    mask = valid_mask
    n_rows = int(mask.sum())
    out = pd.DataFrame(
        {
            "world_idx": world_idx_all[mask],
            "game_id": game_ids[game_idx_all[mask]],
            "game_id_norm": game_id_norm[game_idx_all[mask]],
            "game_date": game_dates[game_idx_all[mask]],
            "team_id": team_id_all[mask],
            "player_id": player_id_all[mask],
            "active": active_flat[mask],
            "minutes": mins_flat[mask],
            "fga2": fga2[mask],
            "fg2m": fg2m[mask],
            "fga3": fga3[mask],
            "fg3m": fg3m[mask],
            "fta": fta[mask],
            "ftm": ftm[mask],
            "oreb": oreb[mask],
            "dreb": dreb[mask],
            "ast": ast[mask],
            "stl": stl[mask],
            "blk": blk[mask],
            "tov": tov[mask],
            "pf": pf[mask],
            "fga": fga[mask],
            "fgm": fgm[mask],
            "fg3a": fga3[mask],
            "pts": pts[mask],
            "reb": reb[mask],
            "plus_minus": np.zeros(n_rows, dtype=np.float32),
            "dk_fpts": dk[mask],
        },
        copy=False,
    )
    return out


def _q(values: np.ndarray, q: float) -> float:
    if values.size <= 0:
        return 0.0
    return float(np.quantile(values, float(q)))


def _group_boundaries_from_sorted_keys(
    *key_arrays: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    """Group arbitrary key arrays via NumPy lexsort without pandas factorize."""
    if len(key_arrays) <= 0:
        empty = np.array([], dtype=np.int64)
        return empty, empty, empty, []
    row_count = int(len(key_arrays[0]))
    if row_count <= 0:
        empty = np.array([], dtype=np.int64)
        return empty, empty, empty, [np.array([], dtype=arr.dtype) for arr in key_arrays]

    arrays = [np.asarray(arr) for arr in key_arrays]
    for arr in arrays[1:]:
        if len(arr) != row_count:
            raise RuntimeError("key array lengths must match for grouping")

    order = np.lexsort(tuple(arr for arr in arrays[::-1]))
    sorted_arrays = [arr[order] for arr in arrays]
    group_starts_mask = np.ones(row_count, dtype=bool)
    if row_count > 1:
        group_starts_mask[1:] = False
        for arr in sorted_arrays:
            group_starts_mask[1:] |= arr[1:] != arr[:-1]
    starts = np.flatnonzero(group_starts_mask)
    ends = np.r_[starts[1:], row_count]
    unique_key_arrays = [arr[starts] for arr in sorted_arrays]
    return order.astype(np.int64, copy=False), starts, ends, unique_key_arrays


def summarize_worlds_to_projections(
    worlds_df: pd.DataFrame,
    *,
    sim_profile: str,
    play_threshold_minutes: float = float(PLAY_THRESHOLD_MINUTES),
    rotation_threshold_minutes: float = float(ROTATION_THRESHOLD_MINUTES),
) -> pd.DataFrame:
    required = {"world_idx", "game_date", "game_id", "team_id", "player_id", "active", "minutes", "dk_fpts"}
    missing = sorted(required - set(worlds_df.columns))
    if missing:
        raise ValueError(f"worlds_df missing required columns: {missing}")
    if worlds_df.empty:
        return worlds_df.copy()

    df = worlds_df.copy()
    df["active"] = pd.to_numeric(df["active"], errors="coerce").fillna(0).astype(int) > 0
    df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce").fillna(0.0)
    df["dk_fpts"] = pd.to_numeric(df["dk_fpts"], errors="coerce").fillna(0.0)
    for c in ("pts", "reb", "ast", "stl", "blk", "tov"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    key_cols = ["game_date", "game_id", "team_id", "player_id"]
    key_frame = df.loc[:, key_cols].copy()
    for col in ("game_id", "team_id", "player_id"):
        key_frame[col] = pd.to_numeric(key_frame[col], errors="coerce")
    valid_keys = ~key_frame.loc[:, key_cols].isna().any(axis=1).to_numpy(dtype=bool, copy=False)
    if not bool(np.any(valid_keys)):
        out = add_canonical_projection_fields(pd.DataFrame())
        return out

    key_frame = key_frame.loc[valid_keys].reset_index(drop=True)
    key_frame["game_date"] = key_frame["game_date"].astype(str)
    work = df.loc[valid_keys].reset_index(drop=True)
    order, starts, ends, unique_key_arrays = _group_boundaries_from_sorted_keys(
        key_frame["game_date"].to_numpy(dtype=str, copy=False),
        key_frame["game_id"].to_numpy(dtype=np.int64, copy=False),
        key_frame["team_id"].to_numpy(dtype=np.int64, copy=False),
        key_frame["player_id"].to_numpy(dtype=np.int64, copy=False),
    )
    if len(order) <= 0:
        out = add_canonical_projection_fields(pd.DataFrame())
        return out

    minutes_all = work["minutes"].to_numpy(dtype=float, copy=False)
    fpts_all = work["dk_fpts"].to_numpy(dtype=float, copy=False)
    active_all = work["active"].to_numpy(dtype=bool, copy=False)
    world_idx_all = pd.to_numeric(work["world_idx"], errors="coerce").to_numpy(
        dtype=float, copy=False
    )
    stat_arrays: dict[str, np.ndarray] = {}
    for stat_name in ("pts", "reb", "ast", "stl", "blk", "tov"):
        if stat_name in work.columns:
            stat_arrays[stat_name] = pd.to_numeric(
                work[stat_name], errors="coerce"
            ).fillna(0.0).to_numpy(dtype=float, copy=False)

    rows: list[dict[str, Any]] = []
    for group_idx, (start, end) in enumerate(zip(starts, ends, strict=False)):
        idx = order[start:end]
        game_date = str(unique_key_arrays[0][group_idx])
        game_id = int(unique_key_arrays[1][group_idx])
        team_id = int(unique_key_arrays[2][group_idx])
        player_id = int(unique_key_arrays[3][group_idx])
        minutes = minutes_all[idx]
        fpts = fpts_all[idx]
        active_mask = active_all[idx]
        world_idx = world_idx_all[idx]
        world_idx = world_idx[np.isfinite(world_idx)]
        n_worlds = int(max(1, np.unique(world_idx.astype(np.int64, copy=False)).size))

        # Spec semantics: conditional moments are over active worlds.
        cond_mask = active_mask
        if not bool(cond_mask.any()):
            cond_mask = minutes >= float(play_threshold_minutes)
        fpts_cond = fpts[cond_mask]
        mins_cond = minutes[cond_mask]

        payload: dict[str, Any] = {
            "game_date": game_date,
            "game_id": game_id,
            "team_id": team_id,
            "player_id": player_id,
            "play_prob": float(active_mask.mean()),
            "play_prob_raw": float(active_mask.mean()),
            "play_prob_eff": float(active_mask.mean()),
            "sim_p_available": float(active_mask.mean()),
            "sim_p_active": float(active_mask.mean()),
            "sim_p_rotation": float((minutes >= float(rotation_threshold_minutes)).mean()),
            "rotation_lock": False,
            "play_prob_policy_reason": "n/a",
            "bench_zero_p_zero": 0.0,
            "bench_zero_threshold_minutes": float("nan"),
            "minutes_mean": float(minutes.mean()),
            "minutes_sim_mean": float(mins_cond.mean()) if mins_cond.size > 0 else 0.0,
            "minutes_sim_std": float(mins_cond.std(ddof=0)) if mins_cond.size > 0 else 0.0,
            "minutes_sim_p10": _q(mins_cond, 0.10),
            "minutes_sim_p50": _q(mins_cond, 0.50),
            "minutes_sim_p90": _q(mins_cond, 0.90),
            "minutes_p10_cond": _q(mins_cond, 0.10),
            "minutes_p50_cond": _q(mins_cond, 0.50),
            "minutes_p90_cond": _q(mins_cond, 0.90),
            "minutes_sim_mean_uncond": float(minutes.mean()),
            "minutes_sim_std_uncond": float(minutes.std(ddof=0)),
            "minutes_sim_p10_uncond": _q(minutes, 0.10),
            "minutes_sim_p50_uncond": _q(minutes, 0.50),
            "minutes_sim_p90_uncond": _q(minutes, 0.90),
            "minutes_p10": _q(minutes, 0.10),
            "minutes_p50": _q(minutes, 0.50),
            "minutes_p90": _q(minutes, 0.90),
            "dk_fpts_mean_target": float(fpts.mean()),
            "dk_fpts_mean": float(fpts_cond.mean()) if fpts_cond.size > 0 else 0.0,
            "dk_fpts_std": float(fpts_cond.std(ddof=0)) if fpts_cond.size > 0 else 0.0,
            "dk_fpts_p05": _q(fpts_cond, 0.05),
            "dk_fpts_p10": _q(fpts_cond, 0.10),
            "dk_fpts_p25": _q(fpts_cond, 0.25),
            "dk_fpts_p50": _q(fpts_cond, 0.50),
            "dk_fpts_p75": _q(fpts_cond, 0.75),
            "dk_fpts_p90": _q(fpts_cond, 0.90),
            "dk_fpts_p95": _q(fpts_cond, 0.95),
            "dk_fpts_mean_uncond": float(fpts.mean()),
            "dk_fpts_std_uncond": float(fpts.std(ddof=0)),
            "dk_fpts_p05_uncond": _q(fpts, 0.05),
            "dk_fpts_p10_uncond": _q(fpts, 0.10),
            "dk_fpts_p25_uncond": _q(fpts, 0.25),
            "dk_fpts_p50_uncond": _q(fpts, 0.50),
            "dk_fpts_p75_uncond": _q(fpts, 0.75),
            "dk_fpts_p90_uncond": _q(fpts, 0.90),
            "dk_fpts_p95_uncond": _q(fpts, 0.95),
            "sim_profile": str(sim_profile),
            "n_worlds": int(n_worlds),
            # Keep legacy-prefixed aliases used by some API code paths.
            "sim_dk_fpts_mean": float(fpts_cond.mean()) if fpts_cond.size > 0 else 0.0,
            "sim_dk_fpts_std": float(fpts_cond.std(ddof=0)) if fpts_cond.size > 0 else 0.0,
            "sim_dk_fpts_p05": _q(fpts_cond, 0.05),
            "sim_dk_fpts_p10": _q(fpts_cond, 0.10),
            "sim_dk_fpts_p25": _q(fpts_cond, 0.25),
            "sim_dk_fpts_p50": _q(fpts_cond, 0.50),
            "sim_dk_fpts_p75": _q(fpts_cond, 0.75),
            "sim_dk_fpts_p90": _q(fpts_cond, 0.90),
            "sim_dk_fpts_p95": _q(fpts_cond, 0.95),
            "sim_minutes_sim_mean": float(mins_cond.mean()) if mins_cond.size > 0 else 0.0,
            "sim_minutes_sim_p10": _q(mins_cond, 0.10),
            "sim_minutes_sim_p50": _q(mins_cond, 0.50),
            "sim_minutes_sim_p90": _q(mins_cond, 0.90),
            "sim_minutes_sim_std": float(mins_cond.std(ddof=0)) if mins_cond.size > 0 else 0.0,
            "sim_minutes_sim_mean_uncond": float(minutes.mean()),
            "sim_minutes_sim_p10_uncond": _q(minutes, 0.10),
            "sim_minutes_sim_p50_uncond": _q(minutes, 0.50),
            "sim_minutes_sim_p90_uncond": _q(minutes, 0.90),
            "sim_minutes_sim_std_uncond": float(minutes.std(ddof=0)),
        }
        for stat_name, vals_all in stat_arrays.items():
            vals = vals_all[idx]
            payload[f"{stat_name}_mean"] = float(vals[cond_mask].mean()) if cond_mask.any() else 0.0
            payload[f"{stat_name}_mean_uncond"] = float(vals.mean())
            payload[f"sim_{stat_name}_mean_uncond"] = float(vals.mean())

        rows.append(payload)

    out = pd.DataFrame(rows)
    out = add_canonical_projection_fields(out)
    sort_cols = [c for c in ("game_date", "game_id", "team_id", "player_id") if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    return out


def sample_worlds_for_batch(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor | list[str]],
    *,
    device: torch.device,
    num_worlds: int,
    chunk_size: int,
    active_temperature: float,
    strict_contracts: bool,
    poss_symmetry_gate: float | None = None,
    attempt_conditioning_mode: str = "predicted_attempts",
    make_model_config: MakeModelConfig | None = None,
    allocation_source: str = "emergent",
    allocation_blend_alpha: float = 0.5,
    allocation_top_usage_top1_scale: float = 1.0,
    allocation_top_usage_top2_scale: float = 1.0,
    ast_factorization_runtime_config: AstFactorizationRuntimeConfig | None = None,
    force_active_minutes_floor_ratio: float = DEFAULT_FORCE_ACTIVE_MINUTES_FLOOR_RATIO,
    force_active_minutes_floor_min: float = DEFAULT_FORCE_ACTIVE_MINUTES_FLOOR_MIN,
    force_active_minutes_floor_max: float = DEFAULT_FORCE_ACTIVE_MINUTES_FLOOR_MAX,
    starter_low_minutes_trigger: float = DEFAULT_STARTER_LOW_MINUTES_TRIGGER,
    promotion_expert_model: torch.nn.Module | None = None,
    promotion_hybrid_config: PromotionHybridConfig | None = None,
    bench_expert_model: torch.nn.Module | None = None,
    bench_hybrid_config: BenchRiserHybridConfig | None = None,
    oracle_rotation_state: bool = False,
    minutes_uncertainty_config: MinutesUncertaintyConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    if not hasattr(model, "flow_head") or model.flow_head is None:  # type: ignore[attr-defined]
        raise RuntimeError("Model does not expose flow_head for inverse flow sampling")
    if (promotion_expert_model is None) != (promotion_hybrid_config is None):
        raise ValueError(
            "promotion_expert_model and promotion_hybrid_config must be provided together",
        )
    if (bench_expert_model is None) != (bench_hybrid_config is None):
        raise ValueError(
            "bench_expert_model and bench_hybrid_config must be provided together",
        )

    model_flow_target_columns = list(model.flow_target_columns)  # type: ignore[attr-defined]
    contract_flow_target_columns = flow_contract_columns(include_pf=("pf" in model_flow_target_columns))
    model_config = getattr(model, "gtv2_config", None)
    if model_config is not None and not isinstance(model_config, GameTransformerV2Config):
        model_config = None
    bsz = int(batch["player_features"].shape[0])  # type: ignore[index]
    if str(attempt_conditioning_mode) == "predicted_attempts":
        flow_targets_batch = batch.get("flow_targets")
        if isinstance(flow_targets_batch, torch.Tensor) and int(flow_targets_batch.shape[-1]) > 0:
            raise RuntimeError(
                "label leakage guard failed: flow label tensors present in sampler batch under predicted_attempts mode",
            )

    player_features = batch["player_features"].to(device=device)  # type: ignore[index]
    player_valid_mask = batch["player_valid_mask"].to(device=device)  # type: ignore[index]
    forced_active_worlds = batch.get("force_active_worlds")
    if isinstance(forced_active_worlds, torch.Tensor):
        forced_active_worlds = forced_active_worlds.to(device=device, dtype=torch.bool)
    else:
        forced_active_worlds = torch.zeros_like(player_valid_mask, dtype=torch.bool, device=device)
    starter_force_active_worlds = batch.get("starter_force_active_worlds")
    if isinstance(starter_force_active_worlds, torch.Tensor):
        starter_force_active_worlds = starter_force_active_worlds.to(device=device, dtype=torch.bool)
    else:
        starter_force_active_worlds = torch.zeros_like(player_valid_mask, dtype=torch.bool, device=device)
    forced_active_minutes_anchor = batch.get("force_active_minutes_anchor")
    if isinstance(forced_active_minutes_anchor, torch.Tensor):
        forced_active_minutes_anchor = forced_active_minutes_anchor.to(device=device, dtype=torch.float32)
    else:
        forced_active_minutes_anchor = torch.zeros_like(player_valid_mask, dtype=torch.float32, device=device)
    game_features = batch["game_features"].to(device=device)  # type: ignore[index]
    team_features = batch["team_features"].to(device=device)  # type: ignore[index]
    oracle_minutes = batch.get("y_minutes")
    if bool(oracle_rotation_state):
        if not isinstance(oracle_minutes, torch.Tensor):
            raise RuntimeError("oracle_rotation_state=True requires y_minutes in sampler batch")
        oracle_minutes = oracle_minutes.to(device=device, dtype=torch.float32)
    else:
        oracle_minutes = None

    frames: list[pd.DataFrame] = []
    contract_counter: Counter[str] = Counter()
    poss_sym_flow_parts: list[torch.Tensor] = []
    poss_sym_valid_parts: list[torch.Tensor] = []
    poss_sym_team_parts: list[torch.Tensor] = []
    total_worlds = max(1, int(num_worlds))
    chunk = max(1, int(chunk_size))
    for world_offset in range(0, total_worlds, chunk):
        n_worlds_chunk = min(chunk, total_worlds - world_offset)
        rep_player_features = player_features.repeat_interleave(n_worlds_chunk, dim=0)
        rep_player_valid_mask = player_valid_mask.repeat_interleave(n_worlds_chunk, dim=0)
        rep_forced_active_worlds = forced_active_worlds.repeat_interleave(n_worlds_chunk, dim=0)
        rep_starter_force_active_worlds = starter_force_active_worlds.repeat_interleave(n_worlds_chunk, dim=0)
        rep_forced_active_minutes_anchor = forced_active_minutes_anchor.repeat_interleave(n_worlds_chunk, dim=0)
        rep_game_features = game_features.repeat_interleave(n_worlds_chunk, dim=0)
        rep_team_features = team_features.repeat_interleave(n_worlds_chunk, dim=0)
        rep_oracle_minutes = (
            oracle_minutes.repeat_interleave(n_worlds_chunk, dim=0) if isinstance(oracle_minutes, torch.Tensor) else None
        )
        starter_promotion_candidate_worlds: torch.Tensor | None = None
        bench_riser_candidate_worlds: torch.Tensor | None = None
        if promotion_hybrid_config is not None:
            starter_promotion_candidate_worlds = compute_starter_promotion_candidate_mask(
                player_features=rep_player_features,
                player_valid_mask=rep_player_valid_mask,
                starter_hint_mask=rep_starter_force_active_worlds,
                config=promotion_hybrid_config,
            )
        if bench_hybrid_config is not None:
            bench_riser_candidate_worlds = compute_bench_riser_candidate_mask(
                player_features=rep_player_features,
                player_valid_mask=rep_player_valid_mask,
                starter_hint_mask=rep_starter_force_active_worlds,
                config=bench_hybrid_config,
            )

        with torch.no_grad():
            # Enable backbone sampling when possession backbone is present
            has_backbone = getattr(model, "enable_possession_backbone", False)
            forward_kwargs = {
                "game_features": rep_game_features,
                "team_features": rep_team_features,
                "sample_active": True,
                "active_temperature": float(active_temperature),
                "run_flow": False,
                "sample_backbone": bool(has_backbone),
                "target_counts": None,
                "use_target_counts": False,
                "target_active_mask": None,
                "use_target_active_mask": False,
                "minutes_use_target_active": False,
                "flow_targets": None,
                "flow_observed_mask": None,
                "starter_hint_mask": rep_starter_force_active_worlds,
                "starter_promotion_candidate_mask": starter_promotion_candidate_worlds,
            }
            _assert_no_labels_in_forward_kwargs(forward_kwargs)
            out = model(
                rep_player_features,
                rep_player_valid_mask,
                **forward_kwargs,
            )
            if out.flow is not None:
                raise RuntimeError("label leakage guard failed: sampler forward returned flow outputs")
            active_mask_for_sampling = out.active.active_mask
            minutes_for_sampling = out.minutes.minutes
            if (
                promotion_expert_model is not None
                and promotion_hybrid_config is not None
                and starter_promotion_candidate_worlds is not None
                and bool(starter_promotion_candidate_worlds.any())
            ):
                expert_forward_kwargs = dict(forward_kwargs)
                expert_forward_kwargs["sample_backbone"] = bool(
                    getattr(promotion_expert_model, "enable_possession_backbone", False)
                )
                expert_out = promotion_expert_model(
                    rep_player_features,
                    rep_player_valid_mask,
                    **expert_forward_kwargs,
                )
                promotion_candidate_flat = starter_promotion_candidate_worlds.reshape(
                    starter_promotion_candidate_worlds.shape[0],
                    -1,
                )
                blended_minutes, blended_active_mask = blend_promotion_predictions(
                    baseline_minutes=out.minutes.minutes,
                    baseline_active_mask=out.active.active_mask,
                    expert_minutes=expert_out.minutes.minutes,
                    expert_active_mask=expert_out.active.active_mask,
                    promotion_candidate_mask=promotion_candidate_flat,
                    uplift_only=bool(promotion_hybrid_config.uplift_only),
                    force_active_candidates=bool(
                        promotion_hybrid_config.force_active_candidates
                    ),
                )
                minutes_for_sampling, active_mask_for_sampling = project_minutes_capped_simplex(
                    blended_minutes,
                    blended_active_mask,
                    out.player_valid_mask,
                    out.player_team_index,
                    total_minutes_per_team=240.0,
                    max_minutes_per_player=48.0,
                )
            if (
                bench_expert_model is not None
                and bench_hybrid_config is not None
                and bench_riser_candidate_worlds is not None
                and bool(bench_riser_candidate_worlds.any())
            ):
                bench_expert_cfg = getattr(bench_expert_model, "gtv2_config", None)
                if bench_expert_cfg is not None and not isinstance(bench_expert_cfg, GameTransformerV2Config):
                    bench_expert_cfg = None
                bench_player_features = _renormalize_player_features_for_model(
                    rep_player_features,
                    source_config=model_config,
                    target_config=bench_expert_cfg,
                )
                expert_forward_kwargs = dict(forward_kwargs)
                expert_forward_kwargs["sample_backbone"] = bool(
                    getattr(bench_expert_model, "enable_possession_backbone", False)
                )
                bench_expert_out = bench_expert_model(
                    bench_player_features,
                    rep_player_valid_mask,
                    **expert_forward_kwargs,
                )
                bench_candidate_flat = bench_riser_candidate_worlds.reshape(
                    bench_riser_candidate_worlds.shape[0],
                    -1,
                )
                blended_minutes, blended_active_mask = blend_expert_predictions(
                    baseline_minutes=minutes_for_sampling,
                    baseline_active_mask=active_mask_for_sampling,
                    expert_minutes=bench_expert_out.minutes.minutes,
                    expert_active_mask=bench_expert_out.active.active_mask,
                    candidate_mask=bench_candidate_flat,
                    uplift_only=bool(bench_hybrid_config.uplift_only),
                    force_active_candidates=bool(bench_hybrid_config.force_active_candidates),
                )
                minutes_for_sampling, active_mask_for_sampling = project_minutes_capped_simplex(
                    blended_minutes,
                    blended_active_mask,
                    out.player_valid_mask,
                    out.player_team_index,
                    total_minutes_per_team=240.0,
                    max_minutes_per_player=48.0,
                )
            if rep_oracle_minutes is not None:
                oracle_minutes_flat = rep_oracle_minutes.reshape(rep_oracle_minutes.shape[0], -1)
                oracle_active_mask = oracle_minutes_flat.gt(float(DEFAULT_ACTIVE_MINUTES_TOL)) & out.player_valid_mask
                minutes_for_sampling, active_mask_for_sampling = project_minutes_capped_simplex(
                    oracle_minutes_flat,
                    oracle_active_mask,
                    out.player_valid_mask,
                    out.player_team_index,
                    total_minutes_per_team=240.0,
                    max_minutes_per_player=48.0,
                )
            z = torch.randn(
                (rep_player_features.shape[0], out.player_states.shape[1], len(model_flow_target_columns)),
                device=device,
                dtype=out.player_states.dtype,
            )
            flow_env_context = getattr(out, "env_context", None)
            if flow_env_context is not None and flow_env_context.shape[0] != rep_player_features.shape[0]:
                repeat_factor = rep_player_features.shape[0] // max(1, flow_env_context.shape[0])
                flow_env_context = flow_env_context.repeat_interleave(repeat_factor, dim=0)
            if flow_env_context is None and bool(getattr(model, "enable_env_side_channel", False)):
                flow_env_context = model._build_env_side_channel_context(  # type: ignore[attr-defined]
                    player_features=rep_player_features,
                    player_valid_mask=rep_player_valid_mask,
                    game_features=rep_game_features,
                )
            flow_raw = model.flow_head.sample(  # type: ignore[attr-defined]
                z,
                player_states=out.player_states,
                team_states=out.team_states,
                game_state=out.game_state,
                player_team_index=out.player_team_index,
                valid_mask=out.player_valid_mask,
                minutes_context=minutes_for_sampling,
                env_context=flow_env_context,
            )
            flow_projected_base = project_flow_stats_to_contract(
                flow_raw,
                flow_target_columns=model_flow_target_columns,
                flow_contract_columns=contract_flow_target_columns,
                fg2_rate=out.efficiency.mean_fg2 if out.efficiency is not None else None,
                fg3_rate=out.efficiency.mean_fg3 if out.efficiency is not None else None,
                ft_rate=out.efficiency.mean_ft if out.efficiency is not None else None,
            )
            ast_override = None
            if bool(getattr(model_config, "assist_share_replace_flow_ast", False)) or bool(
                getattr(model_config, "assist_share_factorized_ast", False)
            ):
                ast_override = _build_ast_override(
                    flow_projected_base=flow_projected_base,
                    flow_contract_columns=contract_flow_target_columns,
                    player_valid_mask=out.player_valid_mask,
                    player_team_index=out.player_team_index,
                    team_ast_budget=(
                        out.team_ast_budget.team_ast if getattr(out, "team_ast_budget", None) is not None else None
                    ),
                    assist_share_logits=(
                        out.assist_share.ast_logits if getattr(out, "assist_share", None) is not None else None
                    ),
                    ast_blend_gate=(
                        out.ast_blend_gate.gate if getattr(out, "ast_blend_gate", None) is not None else None
                    ),
                    runtime_config=ast_factorization_runtime_config,
                )
            flow_projected = flow_projected_base
            if ast_override is not None:
                flow_projected = flow_projected_base.clone()
                ast_idx = _flow_idx(contract_flow_target_columns, "ast")
                flow_projected[..., ast_idx] = ast_override
            forced_active_flat = (
                rep_forced_active_worlds.reshape(rep_forced_active_worlds.shape[0], -1)
                & rep_player_valid_mask.reshape(rep_player_valid_mask.shape[0], -1)
            )
            starter_forced_active_flat = (
                rep_starter_force_active_worlds.reshape(rep_starter_force_active_worlds.shape[0], -1)
                & rep_player_valid_mask.reshape(rep_player_valid_mask.shape[0], -1)
            )
            out_player_mask_flat = _decode_out_player_mask(
                rep_player_features.reshape(
                    rep_player_features.shape[0],
                    -1,
                    rep_player_features.shape[-1],
                ),
                valid_mask=out.player_valid_mask,
                config=model_config,
            )
            forced_active_flat = forced_active_flat & (~out_player_mask_flat)
            starter_forced_active_flat = starter_forced_active_flat & (~out_player_mask_flat)
            forced_minutes_anchor_flat = (
                rep_forced_active_minutes_anchor.reshape(rep_forced_active_minutes_anchor.shape[0], -1)
                * forced_active_flat.to(dtype=rep_forced_active_minutes_anchor.dtype)
            )
            if bool(out_player_mask_flat.any()):
                active_mask_for_sampling = active_mask_for_sampling & (~out_player_mask_flat)
                minutes_for_sampling, active_mask_for_sampling = project_minutes_capped_simplex(
                    minutes_for_sampling.masked_fill(out_player_mask_flat, 0.0),
                    active_mask_for_sampling,
                    out.player_valid_mask,
                    out.player_team_index,
                )
            sampled_active_mask = active_mask_for_sampling | forced_active_flat
            minutes_before_floor = minutes_for_sampling
            if minutes_uncertainty_config is not None and bool(minutes_uncertainty_config.enabled):
                sigma = _estimate_minutes_sigma(
                    minutes_base=minutes_for_sampling,
                    minutes_out=out.minutes,
                    player_features=rep_player_features.reshape(
                        rep_player_features.shape[0],
                        -1,
                        rep_player_features.shape[-1],
                    ),
                    valid_mask=out.player_valid_mask,
                    config=model_config,
                    uncertainty_config=minutes_uncertainty_config,
                )
                minutes_before_floor = _sample_minutes_with_uncertainty(
                    minutes_base=minutes_for_sampling,
                    active_mask=sampled_active_mask,
                    valid_mask=out.player_valid_mask,
                    player_team_index=out.player_team_index,
                    sigma=sigma,
                    uncertainty_config=minutes_uncertainty_config,
                )
            starter_low_minutes_mask = (
                starter_forced_active_flat
                & minutes_before_floor.lt(float(starter_low_minutes_trigger))
            )
            manual_forced_mask = forced_active_flat & (~starter_forced_active_flat)
            floor_target_mask = manual_forced_mask | starter_low_minutes_mask
            sampled_minutes = _apply_forced_active_minutes_floor(
                minutes=minutes_before_floor,
                valid_mask=out.player_valid_mask,
                team_index=out.player_team_index,
                forced_active_mask=floor_target_mask,
                forced_minutes_anchor=forced_minutes_anchor_flat,
                floor_ratio=float(force_active_minutes_floor_ratio),
                floor_min=float(force_active_minutes_floor_min),
                floor_max=float(force_active_minutes_floor_max),
            )
            # Enforce active semantics: players with effectively zero minutes are inactive.
            sampled_active_mask = sampled_active_mask & sampled_minutes.gt(
                float(DEFAULT_ACTIVE_MINUTES_TOL)
            )
            sampled_active_mask = sampled_active_mask & (~out_player_mask_flat)
            sampled_minutes = sampled_minutes.masked_fill(out_player_mask_flat, 0.0)
            # Enforce DNP semantics: inactive players contribute zero counting stats.
            flow_projected = flow_projected * sampled_active_mask.unsqueeze(-1).to(dtype=flow_projected.dtype)
            usage_share_logits: torch.Tensor | None = None
            if getattr(out, "usage_share", None) is not None:
                usage_share_logits = torch.stack(
                    [
                        out.usage_share.fga_logits,
                        out.usage_share.fta_logits,
                        out.usage_share.tov_logits,
                    ],
                    dim=-1,
                )
            alloc_source = str(allocation_source).strip().lower()
            if alloc_source in {"usage_head", "blend"} and usage_share_logits is None:
                if alloc_source == "usage_head":
                    raise RuntimeError("allocation_source=usage_head requires model.enable_usage_share_head")
                alloc_source = "emergent"
            # Couple player-level outputs to sampled backbone team event budgets.
            if has_backbone and out.backbone is not None:
                budget_fga = out.backbone.fga
                budget_fta = out.backbone.fta
                budget_tov = out.backbone.tov
                budget_oreb = out.backbone.oreb
                budget_three_pa_share = out.backbone.three_pa_share
                if str(attempt_conditioning_mode) == "true_attempts_upper_bound":
                    if "flow_targets" not in batch:
                        raise RuntimeError(
                            "attempt_conditioning_mode=true_attempts_upper_bound requires flow_targets in batch",
                        )
                    flow_targets_true = batch["flow_targets"].to(device=device)  # type: ignore[index]
                    if flow_targets_true.ndim != 4 or int(flow_targets_true.shape[-1]) < len(contract_flow_target_columns):
                        raise RuntimeError(
                            "true_attempts_upper_bound requires observed flow_targets with expected stat columns",
                        )
                    fga2_idx = _flow_idx(contract_flow_target_columns, "fga2")
                    fga3_idx = _flow_idx(contract_flow_target_columns, "fga3")
                    fta_idx = _flow_idx(contract_flow_target_columns, "fta")
                    tov_idx = _flow_idx(contract_flow_target_columns, "tov")
                    oreb_idx = _flow_idx(contract_flow_target_columns, "oreb")
                    true_fga2 = flow_targets_true[:, :, :, fga2_idx].sum(dim=2)
                    true_fga3 = flow_targets_true[:, :, :, fga3_idx].sum(dim=2)
                    true_fta = flow_targets_true[:, :, :, fta_idx].sum(dim=2)
                    true_tov = flow_targets_true[:, :, :, tov_idx].sum(dim=2)
                    true_oreb = flow_targets_true[:, :, :, oreb_idx].sum(dim=2)
                    true_fga = true_fga2 + true_fga3
                    true_share3 = torch.where(
                        true_fga > 1e-8,
                        true_fga3 / true_fga.clamp(min=1e-8),
                        torch.full_like(true_fga, 0.38),
                    )
                    budget_fga = true_fga.repeat_interleave(n_worlds_chunk, dim=0)
                    budget_fta = true_fta.repeat_interleave(n_worlds_chunk, dim=0)
                    budget_tov = true_tov.repeat_interleave(n_worlds_chunk, dim=0)
                    budget_oreb = true_oreb.repeat_interleave(n_worlds_chunk, dim=0)
                    budget_three_pa_share = true_share3.repeat_interleave(n_worlds_chunk, dim=0)
                flow_projected = _align_flow_to_backbone_budgets(
                    flow_values=flow_projected,
                    valid_mask=out.player_valid_mask,
                    team_index=out.player_team_index,
                    active_mask=sampled_active_mask,
                    flow_target_columns=contract_flow_target_columns,
                    backbone_fga=budget_fga,
                    backbone_fta=budget_fta,
                    backbone_tov=budget_tov,
                    backbone_oreb=budget_oreb,
                    backbone_three_pa_share=budget_three_pa_share,
                    eff_alpha_ft=out.efficiency.alpha_ft if out.efficiency is not None else None,
                    eff_beta_ft=out.efficiency.beta_ft if out.efficiency is not None else None,
                    eff_alpha_fg2=out.efficiency.alpha_fg2 if out.efficiency is not None else None,
                    eff_beta_fg2=out.efficiency.beta_fg2 if out.efficiency is not None else None,
                    eff_alpha_fg3=out.efficiency.alpha_fg3 if out.efficiency is not None else None,
                    eff_beta_fg3=out.efficiency.beta_fg3 if out.efficiency is not None else None,
                    make_model_config=make_model_config,
                    usage_share_logits=usage_share_logits,
                    allocation_source=alloc_source,
                    allocation_blend_alpha=float(allocation_blend_alpha),
                    allocation_top_usage_top1_scale=float(allocation_top_usage_top1_scale),
                    allocation_top_usage_top2_scale=float(allocation_top_usage_top2_scale),
                )
                if bool(getattr(model_config, "assist_share_reconcile_ast_budget", False)):
                    creator_share_alpha = _build_creator_reconcile_alpha(
                        player_features=player_features,
                        valid_mask=out.player_valid_mask,
                        team_index=out.player_team_index,
                        config=model_config,
                        runtime_config=ast_factorization_runtime_config,
                    )
                    flow_projected = _reconcile_ast_to_team_budget(
                        flow_values=flow_projected,
                        valid_mask=out.player_valid_mask,
                        team_index=out.player_team_index,
                        active_mask=sampled_active_mask,
                        flow_target_columns=contract_flow_target_columns,
                        team_ast_budget=(
                            out.team_ast_budget.team_ast if getattr(out, "team_ast_budget", None) is not None else None
                        ),
                        assist_share_logits=(
                            out.assist_share.ast_logits if getattr(out, "assist_share", None) is not None else None
                        ),
                        share_alpha=(
                            creator_share_alpha
                            if creator_share_alpha is not None
                            else float(getattr(model_config, "assist_share_reconcile_alpha", 0.75))
                        ),
                        share_temperature=float(
                            getattr(model_config, "assist_share_reconcile_temperature", 0.85)
                        ),
                    )
                if bool(getattr(model_config, "rebound_factor_reconcile_oreb_dreb", False)):
                    flow_projected = _reconcile_rebounds_to_opportunity_budgets(
                        flow_values=flow_projected,
                        valid_mask=out.player_valid_mask,
                        team_index=out.player_team_index,
                        active_mask=sampled_active_mask,
                        flow_target_columns=contract_flow_target_columns,
                        team_oreb_budget=(
                            out.team_rebound_budget.team_oreb
                            if getattr(out, "team_rebound_budget", None) is not None
                            else None
                        ),
                        team_dreb_budget=(
                            out.team_rebound_budget.team_dreb
                            if getattr(out, "team_rebound_budget", None) is not None
                            else None
                        ),
                        team_oreb_budget_gate=(
                            out.rebound_budget_blend_gate.oreb_gate
                            if getattr(out, "rebound_budget_blend_gate", None) is not None
                            else None
                        ),
                        team_dreb_budget_gate=(
                            out.rebound_budget_blend_gate.dreb_gate
                            if getattr(out, "rebound_budget_blend_gate", None) is not None
                            else None
                        ),
                        oreb_share_logits=(
                            out.rebound_share.oreb_logits
                            if getattr(out, "rebound_share", None) is not None
                            else None
                        ),
                        dreb_share_logits=(
                            out.rebound_share.dreb_logits
                            if getattr(out, "rebound_share", None) is not None
                            else None
                        ),
                        share_alpha=float(getattr(model_config, "rebound_factor_reconcile_alpha", 0.50)),
                        share_temperature=float(getattr(model_config, "rebound_factor_reconcile_temperature", 0.90)),
                        reconcile_mode=str(getattr(model_config, "rebound_factor_reconcile_mode", "both")),
                        budget_parameterization=str(
                            getattr(model_config, "rebound_budget_parameterization", "absolute")
                        ),
                        dreb_deterministic_discount=float(
                            getattr(model_config, "rebound_dreb_deterministic_discount", 1.0)
                        ),
                        oreb_reconcile_use_flow_budget=bool(
                            getattr(model_config, "rebound_oreb_reconcile_use_flow_budget", False)
                        ),
                        oreb_budget_blend_alpha=float(
                            getattr(model_config, "rebound_oreb_budget_blend_alpha", 1.0)
                        ),
                        dreb_budget_blend_alpha=float(
                            getattr(model_config, "rebound_dreb_budget_blend_alpha", 1.0)
                        ),
                    )
                if bool(getattr(model_config, "team_opportunity_reconcile_budget", False)):
                    resolved_team_opportunity_share = _resolve_team_opportunity_share(
                        model_config=model_config,
                        game_features=rep_game_features,
                    )
                    flow_projected = _reconcile_opportunities_to_team_budget(
                        flow_values=flow_projected,
                        valid_mask=out.player_valid_mask,
                        team_index=out.player_team_index,
                        active_mask=sampled_active_mask,
                        flow_target_columns=contract_flow_target_columns,
                        team_opportunity_share=resolved_team_opportunity_share,
                        budget_alpha=float(getattr(model_config, "team_opportunity_reconcile_alpha", 1.0)),
                        preserve_possessions=bool(
                            getattr(model_config, "team_opportunity_reconcile_preserve_possessions", False)
                        ),
                    )
                if bool(getattr(model_config, "team_points_reconcile_budget", False)):
                    resolved_team_points_budget = _resolve_team_points_budget(
                        model_config=model_config,
                        game_features=rep_game_features,
                        team_points_budget_out=(
                            out.team_points_budget.team_points
                            if getattr(out, "team_points_budget", None) is not None
                            else None
                        ),
                        team_ppp_out=(
                            out.team_ppp.team_ppp
                            if getattr(out, "team_ppp", None) is not None
                            else None
                        ),
                        possession_out=(
                            out.possession.team_poss
                            if getattr(out.possession, "team_poss", None) is not None
                            else (
                                out.possession.sampled_poss
                                if getattr(out.possession, "sampled_poss", None) is not None
                                else out.possession.mu
                            )
                        ),
                    )
                    flow_projected = _reconcile_points_to_team_budget(
                        flow_values=flow_projected,
                        valid_mask=out.player_valid_mask,
                        team_index=out.player_team_index,
                        active_mask=sampled_active_mask,
                        flow_target_columns=contract_flow_target_columns,
                        team_points_budget=resolved_team_points_budget,
                        budget_alpha=float(getattr(model_config, "team_points_reconcile_alpha", 1.0)),
                    )
                # Safety re-mask after backbone alignment (allocator may fallback to valid players).
                flow_projected = flow_projected * sampled_active_mask.unsqueeze(-1).to(
                    dtype=flow_projected.dtype
                )

        minutes = sampled_minutes.reshape(bsz, n_worlds_chunk, -1)
        active = sampled_active_mask.reshape(bsz, n_worlds_chunk, -1)
        flow_vals = flow_projected.reshape(
            bsz,
            n_worlds_chunk,
            out.player_states.shape[1],
            len(contract_flow_target_columns),
        )
        valid_flat = out.player_valid_mask.reshape(bsz, n_worlds_chunk, -1)
        team_flat = out.player_team_index.reshape(bsz, n_worlds_chunk, -1)

        checks = check_world_contracts(
            minutes=minutes.reshape(-1, minutes.shape[-1]),
            flow_values=flow_vals.reshape(-1, flow_vals.shape[-2], flow_vals.shape[-1]),
            valid_mask=valid_flat.reshape(-1, valid_flat.shape[-1]),
            team_index=team_flat.reshape(-1, team_flat.shape[-1]),
            flow_target_columns=contract_flow_target_columns,
            active_mask=active.reshape(-1, active.shape[-1]),
        )
        contract_counter.update(checks)
        n_violations = int(checks.get("total_violations", 0))
        if n_violations > 0:
            if strict_contracts:
                raise RuntimeError(f"World contract check failed: {checks}")
            logger.warning("World contract violations (non-strict): %s", checks)

        # Accumulate tensors for possession symmetry check (lightweight CPU copies)
        if has_backbone:
            poss_sym_flow_parts.append(flow_vals.reshape(-1, flow_vals.shape[-2], flow_vals.shape[-1]).cpu())
            poss_sym_valid_parts.append(valid_flat.reshape(-1, valid_flat.shape[-1]).cpu())
            poss_sym_team_parts.append(team_flat.reshape(-1, team_flat.shape[-1]).cpu())

        chunk_df = _build_world_rows(
            batch=batch,
            world_offset=int(world_offset),
            minutes=minutes,
            active_mask=active,
            flow_values=flow_vals,
            flow_target_columns=contract_flow_target_columns,
        )
        if not chunk_df.empty:
            frames.append(chunk_df)

    # -- Possession symmetry diagnostics (when backbone is enabled) --
    if has_backbone and poss_sym_flow_parts:
        all_flow = torch.cat(poss_sym_flow_parts, dim=0)
        all_valid = torch.cat(poss_sym_valid_parts, dim=0)
        all_team = torch.cat(poss_sym_team_parts, dim=0)
        poss_diag = check_possession_symmetry(
            flow_values=all_flow,
            valid_mask=all_valid,
            team_index=all_team,
            flow_target_columns=contract_flow_target_columns,
        )
        logger.info(
            "possession symmetry: home=%.1f  away=%.1f  |delta| mean=%.2f  p95=%.2f  max=%.2f",
            poss_diag["poss_home_mean"],
            poss_diag["poss_away_mean"],
            poss_diag["poss_delta_abs_mean"],
            poss_diag["poss_delta_abs_p95"],
            poss_diag["poss_delta_abs_max"],
        )
        # Add possession symmetry diagnostics to contract counter for upstream visibility
        for k, v in poss_diag.items():
            contract_counter[k] = v  # type: ignore[assignment]

        # Hard validation gate: p95(|Poss_home - Poss_away|) must be within threshold
        gate = poss_symmetry_gate
        if gate is not None and poss_diag["poss_delta_abs_p95"] > gate:
            msg = (
                f"Possession symmetry gate FAILED: "
                f"p95(|delta|)={poss_diag['poss_delta_abs_p95']:.2f} > {gate:.1f}"
            )
            logger.warning(msg)
            if strict_contracts:
                raise RuntimeError(msg)

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return out, dict(contract_counter)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--promotion-expert-run-dir", type=str, default=None)
    parser.add_argument("--promotion-prior-minutes-max", type=float, default=12.0)
    parser.add_argument("--promotion-hist-start-rate-max", type=float, default=0.20)
    parser.add_argument(
        "--promotion-blend-mode",
        type=str,
        default="uplift_only",
        choices=["uplift_only", "replace"],
        help="How to blend the promotion expert into the primary minutes/active outputs.",
    )
    parser.add_argument(
        "--promotion-force-active-candidates",
        type=int,
        default=0,
        choices=[0, 1],
        help="When enabled, promotion candidates are forced active after hybrid blending.",
    )
    parser.add_argument("--bench-expert-run-dir", type=str, default=None)
    parser.add_argument("--bench-prior-minutes-min", type=float, default=12.0)
    parser.add_argument("--bench-prior-play-prob-min", type=float, default=0.80)
    parser.add_argument("--bench-implied-minutes-min", type=float, default=12.0)
    parser.add_argument("--bench-hist-start-rate-max", type=float, default=0.35)
    parser.add_argument(
        "--bench-blend-mode",
        type=str,
        default="uplift_only",
        choices=["uplift_only", "replace"],
        help="How to blend the bench-riser expert into the primary minutes/active outputs.",
    )
    parser.add_argument(
        "--bench-force-active-candidates",
        type=int,
        default=0,
        choices=[0, 1],
        help="When enabled, bench-riser candidates are forced active after hybrid blending.",
    )
    parser.add_argument("--dataset-dir", type=str, default=None)
    parser.add_argument("--val-days", type=int, default=14)
    parser.add_argument("--num-games", type=int, default=4)
    parser.add_argument("--num-worlds", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--active-temperature", type=float, default=1.0)
    parser.add_argument(
        "--flow-mean-ctx-weight-override",
        type=float,
        default=None,
        help="Inference-only override for mean-pooled flow conditioner context weight.",
    )
    parser.add_argument(
        "--flow-scale-clip-override",
        type=float,
        default=None,
        help="Inference-only override for flow coupling scale_clip (default 2.0). "
        "Higher values allow more extreme scale factors in affine coupling blocks. "
        "Also respects GT_FLOW_SCALE_CLIP env var if CLI not set.",
    )
    parser.add_argument("--strict-contracts", action="store_true")
    parser.add_argument(
        "--poss-symmetry-gate",
        type=float,
        default=None,
        help="Hard gate for p95(|Poss_home - Poss_away|). Warn if exceeded; fail if --strict-contracts.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--attempt-conditioning-mode",
        type=str,
        default="predicted_attempts",
        choices=["predicted_attempts", "true_attempts_upper_bound"],
        help=(
            "Make sampling attempt-conditioning source. "
            "predicted_attempts: normal inference path. "
            "true_attempts_upper_bound: audit-only upper bound using true team attempts post-forward."
        ),
    )
    parser.add_argument(
        "--make-model",
        type=str,
        default="legacy",
        choices=["legacy", "beta_binomial_ft", "beta_binomial_fg", "beta_binomial_all"],
        help=(
            "Make reconstruction mode after backbone budget coupling. "
            "legacy=attempts*pct clipping; beta_binomial_* applies discrete conditional sampling."
        ),
    )
    parser.add_argument(
        "--allocation-source",
        type=str,
        default="emergent",
        choices=["emergent", "usage_head", "blend"],
        help=(
            "Player allocation source after backbone team budgets. "
            "emergent=use sampled flow masses; usage_head=use explicit usage-share logits; "
            "blend=convex blend of emergent and usage_head."
        ),
    )
    parser.add_argument(
        "--allocation-blend-alpha",
        type=float,
        default=0.5,
        help="Blend weight for usage_head when --allocation-source=blend.",
    )
    parser.add_argument(
        "--allocation-top-usage-top1-scale",
        type=float,
        default=1.0,
        help="Research-only decode multiplier for the top-1 implied usage player per team-world.",
    )
    parser.add_argument(
        "--allocation-top-usage-top2-scale",
        type=float,
        default=1.0,
        help="Research-only decode multiplier for the top-2 implied usage player per team-world.",
    )
    parser.add_argument("--bb-ft-prior-mean", type=float, default=0.77)
    parser.add_argument("--bb-ft-prior-strength", type=float, default=6.0)
    parser.add_argument("--bb-ft-concentration", type=float, default=8.0)
    parser.add_argument("--bb-fg2-prior-mean", type=float, default=0.54)
    parser.add_argument("--bb-fg2-prior-strength", type=float, default=8.0)
    parser.add_argument("--bb-fg2-concentration", type=float, default=10.0)
    parser.add_argument("--bb-fg3-prior-mean", type=float, default=0.36)
    parser.add_argument("--bb-fg3-prior-strength", type=float, default=8.0)
    parser.add_argument("--bb-fg3-concentration", type=float, default=10.0)
    parser.add_argument(
        "--bb-use-learned-efficiency",
        type=int,
        default=1,
        choices=[0, 1],
        help=(
            "When make-model is beta_binomial_*, use learned efficiency head alpha/beta if available "
            "(1=yes, 0=no fallback to flow-derived rates + priors)."
        ),
    )
    parser.add_argument(
        "--legacy-use-efficiency-mean",
        type=int,
        default=0,
        choices=[0, 1],
        help=(
            "When make-model is legacy, replace flow-derived make rates with efficiency head means "
            "if available (1=yes, 0=no)."
        ),
    )
    parser.add_argument("--out-parquet", type=str, default=None)
    parser.add_argument("--out-summary-json", type=str, default=None)
    parser.add_argument("--out-projections-parquet", type=str, default=None)
    parser.add_argument("--sim-profile-name", type=str, default="game_transformer_v2")
    parser.add_argument(
        "--oracle-rotation-state",
        action="store_true",
        help=(
            "Research-only diagnostic: replace predicted active/minutes state with label-derived "
            "rotation state, projected back to 240-minute team constraints, before flow/world sampling."
        ),
    )
    parser.add_argument("--minutes-uncertainty-enabled", action="store_true")
    parser.add_argument("--minutes-uncertainty-mode", type=str, default="gaussian", choices=["gaussian", "residual_dirichlet"])
    parser.add_argument("--minutes-uncertainty-gaussian-scale", type=float, default=1.0)
    parser.add_argument("--minutes-uncertainty-min-sigma", type=float, default=0.75)
    parser.add_argument("--minutes-uncertainty-max-sigma", type=float, default=6.0)
    parser.add_argument("--minutes-uncertainty-fallback-sigma", type=float, default=1.5)
    parser.add_argument("--minutes-uncertainty-use-hurdle-sigma", type=int, default=1, choices=[0, 1])
    parser.add_argument("--minutes-uncertainty-use-prior-std", type=int, default=1, choices=[0, 1])
    parser.add_argument("--minutes-uncertainty-preserve-top-k-per-team", type=int, default=3)
    parser.add_argument("--minutes-uncertainty-full-sigma-at-minutes-or-below", type=float, default=24.0)
    parser.add_argument("--minutes-uncertainty-zero-sigma-at-minutes-or-above", type=float, default=32.0)
    parser.add_argument("--minutes-uncertainty-dirichlet-base-concentration", type=float, default=24.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    run_dir = _resolve_run_dir(args.run_dir)
    dataset_dir = _resolve_dataset_dir(args.dataset_dir)

    config = GameTransformerV2Config.load(run_dir / "config.json")
    model = build_game_transformer_v2(config)
    setattr(model, "gtv2_config", config)
    state = torch.load(run_dir / "model.pt", map_location="cpu")
    model.load_state_dict(state)
    promotion_expert_model: torch.nn.Module | None = None
    promotion_hybrid_config: PromotionHybridConfig | None = None
    promotion_expert_run_dir: Path | None = None
    if args.promotion_expert_run_dir:
        promotion_expert_run_dir = _resolve_run_dir(args.promotion_expert_run_dir)
        promotion_expert_cfg = GameTransformerV2Config.load(promotion_expert_run_dir / "config.json")
        assert_promotion_hybrid_compatible(config, promotion_expert_cfg)
        promotion_expert_model = build_game_transformer_v2(promotion_expert_cfg)
        setattr(promotion_expert_model, "gtv2_config", promotion_expert_cfg)
        expert_state = torch.load(promotion_expert_run_dir / "model.pt", map_location="cpu")
        promotion_expert_model.load_state_dict(expert_state)
        promotion_hybrid_config = PromotionHybridConfig.from_model_config(
            config,
            prior_minutes_max=float(args.promotion_prior_minutes_max),
            hist_start_rate_max=float(args.promotion_hist_start_rate_max),
            uplift_only=(str(args.promotion_blend_mode).strip().lower() == "uplift_only"),
            force_active_candidates=bool(int(args.promotion_force_active_candidates)),
        )
    bench_expert_model: torch.nn.Module | None = None
    bench_hybrid_config: BenchRiserHybridConfig | None = None
    bench_expert_run_dir: Path | None = None
    if args.bench_expert_run_dir:
        bench_expert_run_dir = _resolve_run_dir(args.bench_expert_run_dir)
        bench_expert_cfg = GameTransformerV2Config.load(bench_expert_run_dir / "config.json")
        assert_promotion_hybrid_compatible(config, bench_expert_cfg)
        bench_expert_model = build_game_transformer_v2(bench_expert_cfg)
        setattr(bench_expert_model, "gtv2_config", bench_expert_cfg)
        bench_expert_state = torch.load(bench_expert_run_dir / "model.pt", map_location="cpu")
        bench_expert_model.load_state_dict(bench_expert_state)
        bench_hybrid_config = BenchRiserHybridConfig.from_model_config(
            config,
            prior_minutes_min=float(args.bench_prior_minutes_min),
            prior_play_prob_min=float(args.bench_prior_play_prob_min),
            implied_minutes_min=float(args.bench_implied_minutes_min),
            hist_start_rate_max=float(args.bench_hist_start_rate_max),
            uplift_only=(str(args.bench_blend_mode).strip().lower() == "uplift_only"),
            force_active_candidates=bool(int(args.bench_force_active_candidates)),
        )
    if args.flow_mean_ctx_weight_override is not None:
        if not hasattr(model, "flow_head") or model.flow_head is None:  # type: ignore[attr-defined]
            raise RuntimeError("Model does not expose flow_head for mean context override")
        model.flow_head.set_mean_ctx_weight(float(args.flow_mean_ctx_weight_override))  # type: ignore[attr-defined]

    # flow_scale_clip override: CLI takes precedence over env var
    scale_clip_override = args.flow_scale_clip_override
    if scale_clip_override is None:
        env_clip = os.environ.get("GT_FLOW_SCALE_CLIP")
        if env_clip is not None:
            scale_clip_override = float(env_clip)
    if scale_clip_override is not None:
        if not hasattr(model, "flow_head") or model.flow_head is None:  # type: ignore[attr-defined]
            raise RuntimeError("Model does not expose flow_head for scale_clip override")
        print(f"[scale_clip override] setting flow_head.scale_clip = {scale_clip_override}")
        model.flow_head.set_scale_clip(float(scale_clip_override))  # type: ignore[attr-defined]

    device = torch.device(str(args.device))
    model = model.to(device=device)
    model.eval()
    if promotion_expert_model is not None:
        promotion_expert_model = promotion_expert_model.to(device=device)
        promotion_expert_model.eval()
    if bench_expert_model is not None:
        bench_expert_model = bench_expert_model.to(device=device)
        bench_expert_model.eval()
    flow_model_target_columns = list(model.flow_target_columns)  # type: ignore[attr-defined]
    flow_contract_target_columns = flow_contract_columns(include_pf=("pf" in flow_model_target_columns))
    make_model_config = MakeModelConfig(
        mode=str(args.make_model),
        use_learned_efficiency=bool(int(args.bb_use_learned_efficiency)),
        legacy_use_efficiency_mean=bool(int(args.legacy_use_efficiency_mean)),
        bb_ft_prior_mean=float(args.bb_ft_prior_mean),
        bb_ft_prior_strength=float(args.bb_ft_prior_strength),
        bb_ft_concentration=float(args.bb_ft_concentration),
        bb_fg2_prior_mean=float(args.bb_fg2_prior_mean),
        bb_fg2_prior_strength=float(args.bb_fg2_prior_strength),
        bb_fg2_concentration=float(args.bb_fg2_concentration),
        bb_fg3_prior_mean=float(args.bb_fg3_prior_mean),
        bb_fg3_prior_strength=float(args.bb_fg3_prior_strength),
        bb_fg3_concentration=float(args.bb_fg3_concentration),
    )
    minutes_uncertainty_config = MinutesUncertaintyConfig(
        enabled=bool(args.minutes_uncertainty_enabled),
        mode=str(args.minutes_uncertainty_mode),
        gaussian_scale=float(args.minutes_uncertainty_gaussian_scale),
        min_sigma=float(args.minutes_uncertainty_min_sigma),
        max_sigma=float(args.minutes_uncertainty_max_sigma),
        fallback_sigma=float(args.minutes_uncertainty_fallback_sigma),
        use_hurdle_sigma=bool(int(args.minutes_uncertainty_use_hurdle_sigma)),
        use_prior_std=bool(int(args.minutes_uncertainty_use_prior_std)),
        preserve_top_k_per_team=int(args.minutes_uncertainty_preserve_top_k_per_team),
        full_sigma_at_minutes_or_below=float(args.minutes_uncertainty_full_sigma_at_minutes_or_below),
        zero_sigma_at_minutes_or_above=float(args.minutes_uncertainty_zero_sigma_at_minutes_or_above),
        dirichlet_base_concentration=float(args.minutes_uncertainty_dirichlet_base_concentration),
    )

    features_df = _coerce_join_keys(pd.read_parquet(dataset_dir / "features.parquet"), name="features")
    labels_minutes_df = _coerce_join_keys(pd.read_parquet(dataset_dir / "labels_minutes.parquet"), name="labels_minutes")

    label_overlap = [c for c in labels_minutes_df.columns if c in features_df.columns and c not in JOIN_KEYS]
    labels_for_merge = labels_minutes_df.drop(columns=label_overlap)
    merged = features_df.merge(labels_for_merge, on=JOIN_KEYS, how="left", validate="one_to_one")
    if str(args.attempt_conditioning_mode) == "true_attempts_upper_bound":
        labels_counts_path = dataset_dir / "labels_boxscore_counts.parquet"
        if not labels_counts_path.exists():
            raise FileNotFoundError(
                f"{labels_counts_path} is required for --attempt-conditioning-mode=true_attempts_upper_bound",
            )
        labels_counts_df = _coerce_join_keys(pd.read_parquet(labels_counts_path), name="labels_boxscore_counts")
        count_overlap = [c for c in labels_counts_df.columns if c in merged.columns and c not in JOIN_KEYS]
        labels_counts_for_merge = labels_counts_df.drop(columns=count_overlap)
        merged = merged.merge(labels_counts_for_merge, on=JOIN_KEYS, how="left", validate="one_to_one")
    merged["game_id_norm"] = zfill_game_id_series(merged["game_id"])
    val_df = _split_val(merged, val_days=int(args.val_days))

    examples = build_game_level_examples(
        val_df,
        feature_columns=list(config.feature_columns),
        feature_mean=np.asarray(config.feature_mean, dtype=np.float32),
        feature_std=np.asarray(config.feature_std, dtype=np.float32),
        game_feature_columns=list(config.game_feature_columns),
        team_feature_columns=list(config.team_feature_columns),
        flow_label_columns=flow_contract_target_columns
        if str(args.attempt_conditioning_mode) == "true_attempts_upper_bound"
        else None,
        minutes_label_col="minutes_label" if "minutes_label" in val_df.columns else "minutes",
        overflow_protected_prior_play_prob_floor=float(config.overflow_protected_prior_play_prob_floor),
        overflow_protected_prior_minutes_floor=float(config.overflow_protected_prior_minutes_floor),
        overflow_risk_weight_consecutive_active_dnp=float(config.overflow_risk_weight_consecutive_active_dnp),
        overflow_risk_weight_active_but_dnp_rate_last10=float(config.overflow_risk_weight_active_but_dnp_rate_last10),
        overflow_risk_weight_inactive_streak_len=float(config.overflow_risk_weight_inactive_streak_len),
        overflow_keep_weight_prior_play_prob=float(config.overflow_keep_weight_prior_play_prob),
        overflow_keep_weight_prior_minutes=float(config.overflow_keep_weight_prior_minutes),
    )
    max_games = max(1, int(args.num_games))
    examples = examples[:max_games]
    loader = DataLoader(
        GameLevelDataset(examples),
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        num_workers=max(0, int(args.num_workers)),
        collate_fn=collate_game_level_examples,
    )

    frames: list[pd.DataFrame] = []
    contract_counter: Counter[str] = Counter()
    for batch in loader:
        df_batch, checks = sample_worlds_for_batch(
            model,
            batch,
            device=device,
            num_worlds=int(args.num_worlds),
            chunk_size=int(args.chunk_size),
            active_temperature=float(args.active_temperature),
            strict_contracts=bool(args.strict_contracts),
            poss_symmetry_gate=getattr(args, "poss_symmetry_gate", None),
            attempt_conditioning_mode=str(args.attempt_conditioning_mode),
            make_model_config=make_model_config,
            allocation_source=str(args.allocation_source),
            allocation_blend_alpha=float(args.allocation_blend_alpha),
            allocation_top_usage_top1_scale=float(args.allocation_top_usage_top1_scale),
            allocation_top_usage_top2_scale=float(args.allocation_top_usage_top2_scale),
            promotion_expert_model=promotion_expert_model,
            promotion_hybrid_config=promotion_hybrid_config,
            bench_expert_model=bench_expert_model,
            bench_hybrid_config=bench_hybrid_config,
            oracle_rotation_state=bool(args.oracle_rotation_state),
            minutes_uncertainty_config=minutes_uncertainty_config,
        )
        frames.append(df_batch)
        contract_counter.update(checks)

    worlds_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out_parquet = Path(args.out_parquet).expanduser() if args.out_parquet else (
        run_dir / f"sample_worlds_v2_{_utc_now_compact()}.parquet"
    )
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    worlds_df.to_parquet(out_parquet, index=False)

    projections_df = summarize_worlds_to_projections(
        worlds_df,
        sim_profile=str(args.sim_profile_name),
    )
    out_projections = Path(args.out_projections_parquet).expanduser() if args.out_projections_parquet else (
        run_dir / "projections.parquet"
    )
    out_projections.parent.mkdir(parents=True, exist_ok=True)
    projections_df.to_parquet(out_projections, index=False)

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "dataset_dir": str(dataset_dir),
        "num_games": int(max_games),
        "num_worlds_per_game": int(args.num_worlds),
        "flow_mean_ctx_weight_effective": float(
            args.flow_mean_ctx_weight_override
            if args.flow_mean_ctx_weight_override is not None
            else getattr(model.flow_head, "mean_ctx_weight", float("nan"))  # type: ignore[attr-defined]
        ),
        "rows": int(len(worlds_df)),
        "flow_target_columns": flow_contract_target_columns,
        "flow_model_target_columns": flow_model_target_columns,
        "attempt_conditioning_mode": str(args.attempt_conditioning_mode),
        "allocation": {
            "source": str(args.allocation_source),
            "blend_alpha": float(args.allocation_blend_alpha),
            "top_usage_top1_scale": float(args.allocation_top_usage_top1_scale),
            "top_usage_top2_scale": float(args.allocation_top_usage_top2_scale),
        },
        "oracle_rotation_state": bool(args.oracle_rotation_state),
        "minutes_uncertainty": {
            "enabled": bool(minutes_uncertainty_config.enabled),
            "mode": str(minutes_uncertainty_config.mode),
            "gaussian_scale": float(minutes_uncertainty_config.gaussian_scale),
            "min_sigma": float(minutes_uncertainty_config.min_sigma),
            "max_sigma": float(minutes_uncertainty_config.max_sigma),
            "fallback_sigma": float(minutes_uncertainty_config.fallback_sigma),
            "use_hurdle_sigma": bool(minutes_uncertainty_config.use_hurdle_sigma),
            "use_prior_std": bool(minutes_uncertainty_config.use_prior_std),
            "preserve_top_k_per_team": int(minutes_uncertainty_config.preserve_top_k_per_team),
            "full_sigma_at_minutes_or_below": float(minutes_uncertainty_config.full_sigma_at_minutes_or_below),
            "zero_sigma_at_minutes_or_above": float(minutes_uncertainty_config.zero_sigma_at_minutes_or_above),
            "dirichlet_base_concentration": float(minutes_uncertainty_config.dirichlet_base_concentration),
            "prior_std_columns": list(minutes_uncertainty_config.prior_std_columns),
        },
        "make_model": {
            "mode": str(make_model_config.mode),
            "use_learned_efficiency": bool(make_model_config.use_learned_efficiency),
            "bb_ft_prior_mean": float(make_model_config.bb_ft_prior_mean),
            "bb_ft_prior_strength": float(make_model_config.bb_ft_prior_strength),
            "bb_ft_concentration": float(make_model_config.bb_ft_concentration),
            "bb_fg2_prior_mean": float(make_model_config.bb_fg2_prior_mean),
            "bb_fg2_prior_strength": float(make_model_config.bb_fg2_prior_strength),
            "bb_fg2_concentration": float(make_model_config.bb_fg2_concentration),
            "bb_fg3_prior_mean": float(make_model_config.bb_fg3_prior_mean),
            "bb_fg3_prior_strength": float(make_model_config.bb_fg3_prior_strength),
            "bb_fg3_concentration": float(make_model_config.bb_fg3_concentration),
        },
        "promotion_hybrid": {
            "enabled": bool(promotion_expert_model is not None),
            "expert_run_dir": str(promotion_expert_run_dir) if promotion_expert_run_dir is not None else None,
            "prior_minutes_max": (
                float(promotion_hybrid_config.prior_minutes_max) if promotion_hybrid_config is not None else None
            ),
            "hist_start_rate_max": (
                float(promotion_hybrid_config.hist_start_rate_max) if promotion_hybrid_config is not None else None
            ),
            "blend_mode": str(args.promotion_blend_mode),
            "force_active_candidates": bool(int(args.promotion_force_active_candidates)),
        },
        "bench_hybrid": {
            "enabled": bool(bench_expert_model is not None),
            "expert_run_dir": str(bench_expert_run_dir) if bench_expert_run_dir is not None else None,
            "prior_minutes_min": (
                float(bench_hybrid_config.prior_minutes_min) if bench_hybrid_config is not None else None
            ),
            "prior_play_prob_min": (
                float(bench_hybrid_config.prior_play_prob_min) if bench_hybrid_config is not None else None
            ),
            "implied_minutes_min": (
                float(bench_hybrid_config.implied_minutes_min) if bench_hybrid_config is not None else None
            ),
            "hist_start_rate_max": (
                float(bench_hybrid_config.hist_start_rate_max) if bench_hybrid_config is not None else None
            ),
            "blend_mode": str(args.bench_blend_mode),
            "force_active_candidates": bool(int(args.bench_force_active_candidates)),
        },
        "contract_checks": dict(contract_counter),
        "out_parquet": str(out_parquet),
        "out_projections_parquet": str(out_projections),
        "projection_rows": int(len(projections_df)),
    }
    out_summary = Path(args.out_summary_json).expanduser() if args.out_summary_json else (
        run_dir / f"sample_worlds_v2_summary_{_utc_now_compact()}.json"
    )
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
