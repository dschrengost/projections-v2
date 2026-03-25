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
) -> torch.Tensor:
    """Project model flow outputs to contract stats used by worlds and diagnostics."""

    return reconstruct_flow_to_contract(
        flow_values,
        flow_target_columns=flow_target_columns,
        contract_columns=flow_contract_columns,
        fg2_rate=fg2_rate,
        fg3_rate=fg3_rate,
        ft_rate=ft_rate,
    )


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
    force_active_minutes_floor_ratio: float = DEFAULT_FORCE_ACTIVE_MINUTES_FLOOR_RATIO,
    force_active_minutes_floor_min: float = DEFAULT_FORCE_ACTIVE_MINUTES_FLOOR_MIN,
    force_active_minutes_floor_max: float = DEFAULT_FORCE_ACTIVE_MINUTES_FLOOR_MAX,
    starter_low_minutes_trigger: float = DEFAULT_STARTER_LOW_MINUTES_TRIGGER,
) -> tuple[pd.DataFrame, dict[str, int]]:
    if not hasattr(model, "flow_head") or model.flow_head is None:  # type: ignore[attr-defined]
        raise RuntimeError("Model does not expose flow_head for inverse flow sampling")

    model_flow_target_columns = list(model.flow_target_columns)  # type: ignore[attr-defined]
    contract_flow_target_columns = flow_contract_columns(include_pf=("pf" in model_flow_target_columns))
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
            }
            _assert_no_labels_in_forward_kwargs(forward_kwargs)
            out = model(
                rep_player_features,
                rep_player_valid_mask,
                **forward_kwargs,
            )
            if out.flow is not None:
                raise RuntimeError("label leakage guard failed: sampler forward returned flow outputs")
            z = torch.randn(
                (rep_player_features.shape[0], out.player_states.shape[1], len(model_flow_target_columns)),
                device=device,
                dtype=out.player_states.dtype,
            )
            flow_raw = model.flow_head.sample(  # type: ignore[attr-defined]
                z,
                player_states=out.player_states,
                team_states=out.team_states,
                game_state=out.game_state,
                player_team_index=out.player_team_index,
                valid_mask=out.player_valid_mask,
                minutes_context=out.minutes.minutes,
            )
            flow_projected = project_flow_stats_to_contract(
                flow_raw,
                flow_target_columns=model_flow_target_columns,
                flow_contract_columns=contract_flow_target_columns,
                fg2_rate=out.efficiency.mean_fg2 if out.efficiency is not None else None,
                fg3_rate=out.efficiency.mean_fg3 if out.efficiency is not None else None,
                ft_rate=out.efficiency.mean_ft if out.efficiency is not None else None,
            )
            forced_active_flat = (
                rep_forced_active_worlds.reshape(rep_forced_active_worlds.shape[0], -1)
                & rep_player_valid_mask.reshape(rep_player_valid_mask.shape[0], -1)
            )
            starter_forced_active_flat = (
                rep_starter_force_active_worlds.reshape(rep_starter_force_active_worlds.shape[0], -1)
                & rep_player_valid_mask.reshape(rep_player_valid_mask.shape[0], -1)
            )
            forced_minutes_anchor_flat = (
                rep_forced_active_minutes_anchor.reshape(rep_forced_active_minutes_anchor.shape[0], -1)
                * forced_active_flat.to(dtype=rep_forced_active_minutes_anchor.dtype)
            )
            sampled_active_mask = out.active.active_mask | forced_active_flat
            minutes_before_floor = out.minutes.minutes
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
        if strict_contracts and int(checks.get("total_violations", 0)) > 0:
            raise RuntimeError(f"World contract check failed: {checks}")

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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    run_dir = _resolve_run_dir(args.run_dir)
    dataset_dir = _resolve_dataset_dir(args.dataset_dir)

    config = GameTransformerV2Config.load(run_dir / "config.json")
    model = build_game_transformer_v2(config)
    state = torch.load(run_dir / "model.pt", map_location="cpu")
    model.load_state_dict(state)
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
