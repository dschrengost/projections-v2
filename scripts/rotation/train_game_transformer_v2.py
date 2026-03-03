#!/usr/bin/env python3
"""Train Phase 1b GameTransformerV2 (joint active-set + joint minutes)."""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from projections import paths
from projections.rotation.game_transformer_v2 import (
    FLOW_TARGET_COLUMNS_V1,
    FLOW_TARGET_COLUMNS_WITH_PF,
    GameLevelDataset,
    GameTransformerV2Config,
    OVERFLOW_KEEP_WEIGHT_PRIOR_MINUTES,
    OVERFLOW_KEEP_WEIGHT_PRIOR_PLAY_PROB,
    OVERFLOW_RISK_WEIGHT_ACTIVE_BUT_DNP_RATE_LAST10,
    OVERFLOW_RISK_WEIGHT_CONSECUTIVE_ACTIVE_DNP,
    OVERFLOW_RISK_WEIGHT_INACTIVE_STREAK_LEN,
    PROTECTED_PRIOR_MINUTES_FLOOR,
    PROTECTED_PRIOR_PLAY_PROB_FLOOR,
    build_game_level_examples,
    build_game_transformer_v2,
    collate_game_level_examples,
)
from projections.rotation.joint_active_set import (
    build_active_set_labels,
    compute_active_set_losses,
)
from projections.rotation.possession_backbone import (
    compute_possession_truth,
)
from projections.rotation.training_losses import compute_crps_loss, compute_team_energy_score
from projections.rotation.set_model import zfill_game_id_series

JOIN_KEYS = ["game_id", "team_id", "player_id", "game_date"]

EXCLUDE_FEATURES = {
    "minutes",
    "minutes_label",
    "starter_flag_label",
    "starter_flag",
    "source",
    "first_in_time_real",
    "last_out_time_real",
    "time_unit_detected",
}

# Match proven v1 exclusion policy for known leakage-prone/same-game fields.
EXCLUDE_DNP_BLIND_FEATURES = {
    "min_last1",
    "min_last3",
    "min_last5",
    "roll_mean_3",
    "roll_mean_5",
    "roll_mean_10",
    "roll_iqr_5",
    "z_vs_10",
}
EXCLUDE_INJURY_STATUS_FEATURES = {"is_prob", "is_q"}
EXCLUDE_SAME_GAME_ROTATION_FEATURES = {
    "depth_6",
    "depth_10",
    "depth_14",
    "effective_n",
    "bench_conc_top1",
    "bench_conc_top2",
    "starter_pool_minutes",
    "bench_pool_minutes",
    "team_total_minutes_from_stints",
    "num_stints",
    "first_in_time_real",
    "last_out_time_real",
    "max_stint_len_real",
    "minutes_from_stints",
    "started_proxy",
    "rotation_team_missing",
    "rotation_missing",
    "rotation_player_row_missing_raw",
    "rotation_player_filled_zero",
}
EXCLUDE_UNSTABLE_FEATURES = {
    "vac_min_szn",
    "vac_min_guard_szn",
    "vac_min_wing_szn",
    "vac_min_big_szn",
}

DEFAULT_GAME_FEATURE_COLS = [
    "vegas_total",
    "vegas_spread",
    "estimated_possessions",
    "vegas_total_missing",
    "vegas_spread_missing",
    "estimated_possessions_missing",
]


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    phase2_flow_warmup: float
    phase2_anchor_weight: float
    phase2_a2_scale: float
    phase2_backoff_count: int
    train_skipped_batches: int
    train_instability_events: int
    train_total: float
    train_minutes_mae: float
    train_count_loss: float
    train_member_loss: float
    train_minutes_nll: float
    train_flow_nll: float
    train_crps_fpts: float
    train_team_energy: float
    train_count_acc: float
    train_poss_nll: float = 0.0
    train_backbone_nll: float = 0.0
    train_three_pa_nll: float = 0.0
    train_efficiency_nll: float = 0.0
    train_usage_share_nll: float = 0.0
    train_emergent_share_aux: float = 0.0
    train_ast_share_aux: float = 0.0
    train_reb_share_aux: float = 0.0
    train_ast_team_rate_aux: float = 0.0
    train_reb_opportunity_rate_aux: float = 0.0
    train_spread_aux: float = 0.0
    train_total_aux: float = 0.0
    train_efficiency_mean_aux: float = 0.0
    val_total: float = 0.0
    val_minutes_mae: float = 0.0
    val_count_loss: float = 0.0
    val_member_loss: float = 0.0
    val_minutes_nll: float = 0.0
    val_flow_nll: float = 0.0
    val_crps_fpts: float = 0.0
    val_team_energy: float = 0.0
    val_count_acc: float = 0.0
    val_poss_nll: float = 0.0
    val_backbone_nll: float = 0.0
    val_three_pa_nll: float = 0.0
    val_efficiency_nll: float = 0.0
    val_usage_share_nll: float = 0.0
    val_emergent_share_aux: float = 0.0
    val_ast_share_aux: float = 0.0
    val_reb_share_aux: float = 0.0
    val_ast_team_rate_aux: float = 0.0
    val_reb_opportunity_rate_aux: float = 0.0
    val_spread_aux: float = 0.0
    val_total_aux: float = 0.0
    val_efficiency_mean_aux: float = 0.0
    val_total_ex_possreg: float = 0.0
    train_poss_regression: float = 0.0
    val_poss_regression: float = 0.0


@dataclass(frozen=True)
class Phase2EpochWeights:
    w_minutes: float
    w_minutes_nll: float
    w_count: float
    w_member: float
    w_flow_nll: float
    w_crps_fpts: float
    w_team_energy: float
    flow_warmup: float
    anchor_weight: float
    run_phase2_flow: bool
    run_phase3_decision: bool


@dataclass(frozen=True)
class BackboneEpochWeights:
    w_poss_nll: float
    w_backbone_nll: float
    w_three_pa_nll: float
    w_poss_regression: float
    ramp_scale_poss: float
    ramp_scale_backbone: float
    ramp_scale_three_pa: float
    ramp_scale_poss_regression: float


@dataclass(frozen=True)
class Phase2StabilityConfig:
    nll_explosion_ratio: float = 3.0
    nll_explosion_abs: float = 25.0
    nll_ema_alpha: float = 0.1
    nll_backoff_consecutive_batches: int = 2
    max_backoffs_before_rollback: int = 3
    min_a2_scale: float = 0.125


@dataclass
class Phase2StabilityState:
    a2_scale: float = 1.0
    ema_gen_nll: float | None = None
    consecutive_explosions: int = 0
    backoff_count: int = 0
    rollback_requested: bool = False
    rollback_reason: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class EarlyStopConfig:
    patience: int = 0
    min_delta: float = 0.0
    min_epochs: int = 0
    min_coupled_epochs: int = 0


@dataclass
class EarlyStopState:
    best_metric: float = float("inf")
    best_epoch: int = 0
    bad_epochs: int = 0
    stop_requested: bool = False
    stop_epoch: int | None = None
    stop_reason: str | None = None


def _utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _coerce_join_keys(df: pd.DataFrame, *, name: str) -> pd.DataFrame:
    out = df.copy()
    for col in ["game_id", "team_id", "player_id"]:
        if col not in out.columns:
            raise ValueError(f"{name} missing key column: {col}")
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
    if "game_date" not in out.columns:
        raise ValueError(f"{name} missing key column: game_date")
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce").dt.normalize()

    invalid_keys = out[["game_id", "team_id", "player_id", "game_date"]].isna().any(axis=1)
    if invalid_keys.any():
        raise ValueError(f"{name} has invalid key rows: {int(invalid_keys.sum())}")
    return out


def _infer_feature_columns(features_df: pd.DataFrame, labels_minutes_df: pd.DataFrame) -> list[str]:
    excluded = {
        "game_id",
        "team_id",
        "player_id",
        "game_date",
        "game_id_norm",
    }
    excluded.update(labels_minutes_df.columns.tolist())
    excluded.update(EXCLUDE_FEATURES)
    excluded.update(EXCLUDE_DNP_BLIND_FEATURES)
    excluded.update(EXCLUDE_INJURY_STATUS_FEATURES)
    excluded.update(EXCLUDE_SAME_GAME_ROTATION_FEATURES)
    excluded.update(EXCLUDE_UNSTABLE_FEATURES)

    cols: list[str] = []
    for col in features_df.columns:
        if col in excluded:
            continue
        if pd.api.types.is_numeric_dtype(features_df[col]) or pd.api.types.is_bool_dtype(features_df[col]):
            cols.append(col)
    if not cols:
        raise ValueError("No numeric feature columns inferred")
    return cols


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


def _compute_feature_norm(train_df: pd.DataFrame, feature_columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x = _numeric_frame(train_df, feature_columns).to_numpy(dtype=np.float32, copy=False)
    mean = np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0).astype(np.float32)
    std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0).astype(np.float32)
    return mean, std


def _split_train_val(df: pd.DataFrame, *, val_days: int) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    days = sorted(pd.to_datetime(df["game_date"]).dropna().dt.normalize().unique().tolist())
    if len(days) < 2:
        raise ValueError("Need at least 2 distinct game dates for train/val split")
    vd = max(1, int(val_days))
    val_dates = set(days[-vd:])

    is_val = pd.to_datetime(df["game_date"]).dt.normalize().isin(val_dates)
    train = df.loc[~is_val].copy()
    val = df.loc[is_val].copy()
    if train.empty or val.empty:
        raise ValueError(f"Invalid split train={len(train)} val={len(val)}")

    meta = {
        "train_rows": int(len(train)),
        "val_rows": int(len(val)),
        "train_games": int(train[["game_id_norm", "game_date"]].drop_duplicates().shape[0]),
        "val_games": int(val[["game_id_norm", "game_date"]].drop_duplicates().shape[0]),
        "train_min_date": str(pd.to_datetime(train["game_date"]).min().date()),
        "train_max_date": str(pd.to_datetime(train["game_date"]).max().date()),
        "val_min_date": str(pd.to_datetime(val["game_date"]).min().date()),
        "val_max_date": str(pd.to_datetime(val["game_date"]).max().date()),
    }
    return train, val, meta


def _flatten_side(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 3 or x.shape[1] != 2:
        raise ValueError("expected x shape (B,2,N)")
    return torch.cat([x[:, 0], x[:, 1]], dim=1)


def _team_index(batch_size: int, n_team_slots: int, *, device: torch.device) -> torch.Tensor:
    return torch.cat(
        [
            torch.zeros((batch_size, n_team_slots), dtype=torch.long, device=device),
            torch.ones((batch_size, n_team_slots), dtype=torch.long, device=device),
        ],
        dim=1,
    )


def _masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    err = (pred - target).abs() * mask.to(dtype=pred.dtype)
    denom = mask.to(dtype=pred.dtype).sum().clamp(min=1.0)
    return err.sum() / denom


def _gaussian_nll(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, *, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    mask_f = mask.to(dtype=pred.dtype)
    var = float(sigma) * float(sigma)
    residual = target - pred
    per = 0.5 * ((residual * residual) / var + math.log(2.0 * math.pi * var))
    denom = mask_f.sum().clamp(min=1.0)
    return (per * mask_f).sum() / denom


def _beta_binomial_nll(
    *,
    attempts: torch.Tensor,
    makes: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    eps = 1e-6
    mask_f = mask.to(dtype=attempts.dtype)
    n = torch.clamp(torch.round(attempts), min=0.0)
    k = torch.clamp(torch.round(makes), min=0.0)
    k = torch.minimum(k, n)
    a = torch.clamp(alpha, min=eps)
    b = torch.clamp(beta, min=eps)

    log_comb = torch.lgamma(n + 1.0) - torch.lgamma(k + 1.0) - torch.lgamma(n - k + 1.0)
    log_beta_ratio = (
        torch.lgamma(k + a)
        + torch.lgamma(n - k + b)
        + torch.lgamma(a + b)
        - torch.lgamma(n + a + b)
        - torch.lgamma(a)
        - torch.lgamma(b)
    )
    nll = -(log_comb + log_beta_ratio)
    denom = mask_f.sum().clamp(min=1.0)
    return (nll * mask_f).sum() / denom


def _team_share_ce_loss(
    *,
    logits: torch.Tensor,
    attempts_true: torch.Tensor,
    valid_mask: torch.Tensor,
    team_index: torch.Tensor,
    observed_mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Cross-entropy between true within-team shares and model logits."""
    if logits.ndim != 2:
        raise ValueError("logits must have shape (B,P)")
    if attempts_true.shape != logits.shape:
        raise ValueError("attempts_true must match logits shape")
    if valid_mask.shape != logits.shape or team_index.shape != logits.shape:
        raise ValueError("valid_mask/team_index must match logits shape")
    if observed_mask is not None and observed_mask.shape != logits.shape:
        raise ValueError("observed_mask must match logits shape")

    valid = valid_mask.to(dtype=torch.bool)
    obs = observed_mask.to(dtype=torch.bool) if observed_mask is not None else valid
    attempts = torch.clamp(attempts_true, min=0.0)

    loss_sum = logits.new_zeros(())
    n_rows = logits.new_zeros(())
    for side in (0, 1):
        elig = valid & team_index.eq(side)
        label_mask = elig & obs
        target_mass = torch.where(label_mask, attempts, torch.zeros_like(attempts))
        team_total = target_mass.sum(dim=1)
        has_labels = team_total > float(eps)
        if not bool(has_labels.any()):
            continue

        masked_logits = logits.masked_fill(~elig, -1e9)
        log_probs = torch.log_softmax(masked_logits, dim=1)
        target_share = target_mass / team_total.unsqueeze(1).clamp(min=float(eps))
        loss_row = -(target_share * log_probs).sum(dim=1)
        has_f = has_labels.to(dtype=loss_row.dtype)
        loss_sum = loss_sum + (loss_row * has_f).sum()
        n_rows = n_rows + has_f.sum()

    return loss_sum / n_rows.clamp(min=1.0)


def _team_sum_by_side(
    *,
    values: torch.Tensor,
    valid_mask: torch.Tensor,
    team_index: torch.Tensor,
    observed_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sum flattened player values to (B, 2) team totals with a side-level observed mask."""
    if values.ndim != 2:
        raise ValueError("values must have shape (B,P)")
    if valid_mask.shape != values.shape or team_index.shape != values.shape:
        raise ValueError("valid_mask/team_index must match values shape")
    if observed_mask is not None and observed_mask.shape != values.shape:
        raise ValueError("observed_mask must match values shape")

    valid = valid_mask.to(dtype=torch.bool)
    obs = observed_mask.to(dtype=torch.bool) if observed_mask is not None else valid
    totals: list[torch.Tensor] = []
    seen: list[torch.Tensor] = []
    for side in (0, 1):
        elig = valid & team_index.eq(side)
        label_mask = elig & obs
        totals.append(torch.where(label_mask, values, torch.zeros_like(values)).sum(dim=1))
        seen.append(label_mask.any(dim=1))
    return torch.stack(totals, dim=1), torch.stack(seen, dim=1)


def _team_ratio_mse_loss(
    *,
    pred_numerator: torch.Tensor,
    pred_denominator: torch.Tensor,
    true_numerator: torch.Tensor,
    true_denominator: torch.Tensor,
    observed_mask: torch.Tensor,
    detach_pred_denominator: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Squared-error loss on team-level rates, optionally detaching the predicted denominator."""
    if pred_numerator.ndim != 2:
        raise ValueError("pred_numerator must have shape (B,2)")
    if pred_denominator.shape != pred_numerator.shape:
        raise ValueError("pred_denominator must match pred_numerator shape")
    if true_numerator.shape != pred_numerator.shape or true_denominator.shape != pred_numerator.shape:
        raise ValueError("true numerator/denominator must match pred_numerator shape")
    if observed_mask.shape != pred_numerator.shape:
        raise ValueError("observed_mask must match pred_numerator shape")

    denom_pred = pred_denominator.detach() if detach_pred_denominator else pred_denominator
    pred_rate = pred_numerator / denom_pred.clamp(min=1.0)
    true_rate = true_numerator / true_denominator.clamp(min=1.0)
    pred_rate = torch.clamp(torch.nan_to_num(pred_rate, nan=0.0, posinf=2.0, neginf=0.0), min=0.0, max=2.0)
    true_rate = torch.clamp(torch.nan_to_num(true_rate, nan=0.0, posinf=1.0, neginf=0.0), min=0.0, max=1.0)
    mask = observed_mask.to(dtype=torch.bool) & (true_denominator > float(eps))
    if not bool(mask.any()):
        return pred_numerator.new_zeros(())
    sq = torch.square(pred_rate - true_rate)
    mask_f = mask.to(dtype=sq.dtype)
    return (sq * mask_f).sum() / mask_f.sum().clamp(min=1.0)


def _team_fixed_opportunity_rate_mse_loss(
    *,
    pred_numerator: torch.Tensor,
    true_numerator: torch.Tensor,
    true_denominator: torch.Tensor,
    observed_mask: torch.Tensor,
    max_rate: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Bound a predicted team total by the observed opportunity budget before scoring rate error."""
    if pred_numerator.ndim != 2:
        raise ValueError("pred_numerator must have shape (B,2)")
    if true_numerator.shape != pred_numerator.shape or true_denominator.shape != pred_numerator.shape:
        raise ValueError("true numerator/denominator must match pred_numerator shape")
    if observed_mask.shape != pred_numerator.shape:
        raise ValueError("observed_mask must match pred_numerator shape")

    denom = true_denominator.clamp(min=1.0)
    pred_rate = pred_numerator / denom
    pred_rate = torch.clamp(torch.nan_to_num(pred_rate, nan=0.0, posinf=max_rate, neginf=0.0), min=0.0, max=max_rate)
    true_rate = true_numerator / denom
    true_rate = torch.clamp(torch.nan_to_num(true_rate, nan=0.0, posinf=max_rate, neginf=0.0), min=0.0, max=max_rate)
    mask = observed_mask.to(dtype=torch.bool) & (true_denominator > float(eps))
    if not bool(mask.any()):
        return pred_numerator.new_zeros(())
    sq = torch.square(pred_rate - true_rate)
    mask_f = mask.to(dtype=sq.dtype)
    return (sq * mask_f).sum() / mask_f.sum().clamp(min=1.0)


def _flow_index(flow_target_columns: list[str], name: str) -> int:
    try:
        return int(flow_target_columns.index(name))
    except ValueError as exc:
        raise KeyError(f"missing flow target column: {name}") from exc


def _project_flow_stats_to_contract(
    flow_values: torch.Tensor,
    *,
    flow_target_columns: list[str],
) -> torch.Tensor:
    fg2m_idx = _flow_index(flow_target_columns, "fg2m")
    fga2_idx = _flow_index(flow_target_columns, "fga2")
    fg3m_idx = _flow_index(flow_target_columns, "fg3m")
    fga3_idx = _flow_index(flow_target_columns, "fga3")
    ftm_idx = _flow_index(flow_target_columns, "ftm")
    fta_idx = _flow_index(flow_target_columns, "fta")
    y_pos = torch.clamp(flow_values, min=0.0)

    cols: list[torch.Tensor] = []
    for idx in range(y_pos.shape[-1]):
        col = y_pos[..., idx]
        if idx == fg2m_idx:
            col = torch.minimum(col, y_pos[..., fga2_idx])
        elif idx == fg3m_idx:
            col = torch.minimum(col, y_pos[..., fga3_idx])
        elif idx == ftm_idx:
            col = torch.minimum(col, y_pos[..., fta_idx])
        cols.append(col.unsqueeze(-1))
    return torch.cat(cols, dim=-1)


def _compute_dk_fpts_from_flow(
    flow_values: torch.Tensor,
    *,
    flow_target_columns: list[str],
) -> torch.Tensor:
    fg2m_idx = _flow_index(flow_target_columns, "fg2m")
    fg3m_idx = _flow_index(flow_target_columns, "fg3m")
    ftm_idx = _flow_index(flow_target_columns, "ftm")
    oreb_idx = _flow_index(flow_target_columns, "oreb")
    dreb_idx = _flow_index(flow_target_columns, "dreb")
    ast_idx = _flow_index(flow_target_columns, "ast")
    stl_idx = _flow_index(flow_target_columns, "stl")
    blk_idx = _flow_index(flow_target_columns, "blk")
    tov_idx = _flow_index(flow_target_columns, "tov")

    fg2m = flow_values[..., fg2m_idx]
    fg3m = flow_values[..., fg3m_idx]
    ftm = flow_values[..., ftm_idx]
    oreb = flow_values[..., oreb_idx]
    dreb = flow_values[..., dreb_idx]
    ast = flow_values[..., ast_idx]
    stl = flow_values[..., stl_idx]
    blk = flow_values[..., blk_idx]
    tov = flow_values[..., tov_idx]

    pts = 2.0 * fg2m + 3.0 * fg3m + ftm
    reb = oreb + dreb
    base = pts + 1.25 * reb + 1.5 * ast + 2.0 * stl + 2.0 * blk - 0.5 * tov
    qualifying = torch.stack([pts, reb, ast, stl, blk], dim=-1).ge(10.0).sum(dim=-1)
    dd_bonus = (qualifying == 2).to(dtype=base.dtype) * 1.5
    td_bonus = (qualifying >= 3).to(dtype=base.dtype) * 3.0
    return base + dd_bonus + td_bonus


def _sample_decision_fpts(
    model: nn.Module,
    *,
    context_out: Any,
    num_samples: int,
    active_temperature: float,
) -> torch.Tensor:
    if int(num_samples) <= 0:
        raise ValueError("num_samples must be > 0")
    if float(active_temperature) <= 0:
        raise ValueError("active_temperature must be > 0")
    if not hasattr(model, "flow_head") or not hasattr(model, "flow_target_columns"):
        raise ValueError("model must expose flow_head and flow_target_columns")

    # Reuse one context pass from the caller, then sample worlds from active/minutes/flow heads.
    ctx = context_out
    flow_head = model.flow_head  # type: ignore[attr-defined]
    flow_target_columns: list[str] = list(model.flow_target_columns)  # type: ignore[attr-defined]
    valid_flat = ctx.player_valid_mask.to(dtype=torch.bool)

    worlds: list[torch.Tensor] = []
    for _ in range(int(num_samples)):
        active_out = model.active_head(  # type: ignore[attr-defined]
            ctx.player_states,
            ctx.team_states,
            ctx.player_team_index,
            valid_flat,
            sample=False,
            temperature=float(active_temperature),
        )

        z = torch.randn(
            (
                ctx.player_states.shape[0],
                ctx.player_states.shape[1],
                len(flow_target_columns),
            ),
            dtype=ctx.player_states.dtype,
            device=ctx.player_states.device,
        )
        flow_samples = flow_head.sample(
            z,
            player_states=ctx.player_states,
            team_states=ctx.team_states,
            game_state=ctx.game_state,
            player_team_index=ctx.player_team_index,
            valid_mask=valid_flat,
            observed_mask=valid_flat.unsqueeze(-1).expand_as(z),
        )
        flow_samples = _project_flow_stats_to_contract(
            flow_samples,
            flow_target_columns=flow_target_columns,
        )
        flow_samples = flow_samples * active_out.active_mask.unsqueeze(-1).to(dtype=flow_samples.dtype)
        worlds.append(
            torch.nan_to_num(
                _compute_dk_fpts_from_flow(
                    flow_samples,
                    flow_target_columns=flow_target_columns,
                ),
                nan=0.0,
                posinf=200.0,
                neginf=-200.0,
            )
        )

    return torch.stack(worlds, dim=1)


def _phase2_flow_warmup_factor(
    epoch: int, *, warmup_epochs: int, delay_epochs: int = 0,
) -> float:
    if int(epoch) <= 0:
        raise ValueError("epoch must be >= 1")
    if int(epoch) <= int(delay_epochs):
        return 0.0
    effective_epoch = int(epoch) - int(delay_epochs)
    if int(warmup_epochs) <= 0:
        return 1.0
    return float(min(1.0, float(effective_epoch) / float(warmup_epochs)))


def _phase2_anchor_weight(
    flow_warmup: float,
    *,
    start_weight: float,
    end_weight: float,
) -> float:
    warm = float(min(max(flow_warmup, 0.0), 1.0))
    return float(end_weight + (start_weight - end_weight) * (1.0 - warm))


def _is_flow_param_name(name: str) -> bool:
    return bool(name.startswith("flow_head."))


def _is_backbone_head_param_name(name: str) -> bool:
    return bool(
        name.startswith("possession_head.")
        or name.startswith("event_backbone.")
        or name.startswith("three_pa_share_head.")
    )


def _is_encoder_param_name(name: str) -> bool:
    if name in {"game_token", "team_tokens"}:
        return True
    return bool(
        name.startswith("player_proj.")
        or name.startswith("game_proj.")
        or name.startswith("team_proj.")
        or name.startswith("token_type_embedding.")
        or name.startswith("side_embedding.")
        or name.startswith("encoder.")
        or name.startswith("final_norm.")
    )


def _resolve_ramped_loss_scale(
    *,
    epoch: int,
    ramp_epochs: int,
    start_scale: float,
) -> float:
    if int(epoch) <= 0:
        raise ValueError("epoch must be >= 1")
    if int(ramp_epochs) <= 1:
        return 1.0
    progress = min(1.0, float(int(epoch) - 1) / float(int(ramp_epochs) - 1))
    return float(start_scale + (1.0 - float(start_scale)) * progress)


def _masked_scaled_huber_loss(
    *,
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    scale: float,
    delta: float,
) -> torch.Tensor:
    """Huber loss on normalized errors, averaged over mask-selected rows."""
    if pred.shape != target.shape:
        raise ValueError("pred and target must have the same shape")
    if mask.shape != pred.shape:
        raise ValueError("mask must match pred shape")
    if float(scale) <= 0.0:
        raise ValueError("scale must be > 0")
    if float(delta) <= 0.0:
        raise ValueError("delta must be > 0")

    mask_b = mask.to(dtype=torch.bool)
    if not bool(mask_b.any()):
        return pred.new_zeros(())
    pred_scaled = pred / float(scale)
    target_scaled = target / float(scale)
    loss = F.huber_loss(pred_scaled, target_scaled, reduction="none", delta=float(delta))
    mask_f = mask_b.to(dtype=loss.dtype)
    return (loss * mask_f).sum() / mask_f.sum().clamp(min=1.0)


def _resolve_backbone_epoch_weights(
    *,
    epoch: int,
    enable_possession_backbone: bool,
    enable_three_pa_share: bool,
    w_poss_nll: float,
    w_backbone_nll: float,
    w_three_pa_nll: float,
    w_poss_regression: float,
    loss_ramp_epochs: int,
    poss_loss_start_scale: float,
    backbone_loss_start_scale: float,
    three_pa_loss_start_scale: float,
    poss_regression_start_scale: float,
) -> BackboneEpochWeights:
    if not bool(enable_possession_backbone):
        return BackboneEpochWeights(
            w_poss_nll=0.0,
            w_backbone_nll=0.0,
            w_three_pa_nll=0.0,
            w_poss_regression=0.0,
            ramp_scale_poss=0.0,
            ramp_scale_backbone=0.0,
            ramp_scale_three_pa=0.0,
            ramp_scale_poss_regression=0.0,
        )
    poss_scale = _resolve_ramped_loss_scale(
        epoch=int(epoch),
        ramp_epochs=int(loss_ramp_epochs),
        start_scale=float(poss_loss_start_scale),
    )
    backbone_scale = _resolve_ramped_loss_scale(
        epoch=int(epoch),
        ramp_epochs=int(loss_ramp_epochs),
        start_scale=float(backbone_loss_start_scale),
    )
    three_pa_scale = _resolve_ramped_loss_scale(
        epoch=int(epoch),
        ramp_epochs=int(loss_ramp_epochs),
        start_scale=float(three_pa_loss_start_scale),
    )
    poss_reg_scale = _resolve_ramped_loss_scale(
        epoch=int(epoch),
        ramp_epochs=int(loss_ramp_epochs),
        start_scale=float(poss_regression_start_scale),
    )
    return BackboneEpochWeights(
        w_poss_nll=float(w_poss_nll) * poss_scale,
        w_backbone_nll=float(w_backbone_nll) * backbone_scale,
        w_three_pa_nll=float(w_three_pa_nll) * three_pa_scale if bool(enable_three_pa_share) else 0.0,
        w_poss_regression=float(w_poss_regression) * poss_reg_scale,
        ramp_scale_poss=float(poss_scale),
        ramp_scale_backbone=float(backbone_scale),
        ramp_scale_three_pa=float(three_pa_scale) if bool(enable_three_pa_share) else 0.0,
        ramp_scale_poss_regression=float(poss_reg_scale),
    )


def _count_backbone_coupled_epochs(
    *,
    epoch: int,
    enable_possession_backbone: bool,
    backbone_detach_until_epoch: int,
) -> int:
    if int(epoch) <= 0:
        raise ValueError("epoch must be >= 1")
    if not bool(enable_possession_backbone):
        return int(epoch)
    detached_epochs = max(0, int(backbone_detach_until_epoch) - 1)
    return max(0, int(epoch) - detached_epochs)


def _resolve_phase2_epoch_weights(
    *,
    epoch: int,
    enable_phase2_flow: bool,
    enable_phase3_decision: bool,
    w_minutes: float,
    w_minutes_nll: float,
    w_count: float,
    w_member: float,
    w_flow_nll: float,
    w_crps_fpts: float,
    w_team_energy: float,
    flow_warmup_epochs: int,
    flow_delay_epochs: int = 0,
    anchor_start_weight: float,
    anchor_end_weight: float,
    a2_scale: float,
) -> Phase2EpochWeights:
    if not bool(enable_phase2_flow):
        return Phase2EpochWeights(
            w_minutes=float(w_minutes),
            w_minutes_nll=0.0,
            w_count=float(w_count),
            w_member=float(w_member),
            w_flow_nll=0.0,
            w_crps_fpts=0.0,
            w_team_energy=0.0,
            flow_warmup=0.0,
            anchor_weight=1.0,
            run_phase2_flow=False,
            run_phase3_decision=False,
        )

    flow_warmup = _phase2_flow_warmup_factor(
        int(epoch), warmup_epochs=int(flow_warmup_epochs), delay_epochs=int(flow_delay_epochs),
    )
    anchor_weight = _phase2_anchor_weight(
        flow_warmup,
        start_weight=float(anchor_start_weight),
        end_weight=float(anchor_end_weight),
    )
    a2 = max(float(a2_scale), 0.0)
    return Phase2EpochWeights(
        w_minutes=float(w_minutes) * anchor_weight,
        w_minutes_nll=float(w_minutes_nll) * a2,
        w_count=float(w_count) * anchor_weight,
        w_member=float(w_member) * anchor_weight,
        w_flow_nll=float(w_flow_nll) * flow_warmup * a2,
        w_crps_fpts=float(w_crps_fpts) if bool(enable_phase3_decision) else 0.0,
        w_team_energy=float(w_team_energy) if bool(enable_phase3_decision) else 0.0,
        flow_warmup=float(flow_warmup),
        anchor_weight=float(anchor_weight),
        run_phase2_flow=bool(flow_warmup > 0.0),
        run_phase3_decision=bool(enable_phase3_decision),
    )


def _update_phase2_nll_guard(
    *,
    epoch: int,
    batch_idx: int,
    gen_nll: float,
    config: Phase2StabilityConfig,
    state: Phase2StabilityState,
) -> tuple[bool, bool, float]:
    threshold = float(config.nll_explosion_abs)
    if state.ema_gen_nll is not None and math.isfinite(float(state.ema_gen_nll)):
        threshold = max(threshold, float(config.nll_explosion_ratio) * float(state.ema_gen_nll))

    exploded = (not math.isfinite(float(gen_nll))) or (float(gen_nll) > threshold)
    backoff_applied = False
    if exploded:
        state.consecutive_explosions += 1
        if state.consecutive_explosions >= max(1, int(config.nll_backoff_consecutive_batches)):
            prev_a2 = float(state.a2_scale)
            state.consecutive_explosions = 0
            state.a2_scale = max(float(config.min_a2_scale), float(state.a2_scale) * 0.5)
            state.backoff_count += 1
            backoff_applied = True
            state.events.append(
                {
                    "epoch": int(epoch),
                    "batch": int(batch_idx),
                    "event": "a2_backoff",
                    "gen_nll": float(gen_nll),
                    "threshold": float(threshold),
                    "a2_before": float(prev_a2),
                    "a2_after": float(state.a2_scale),
                    "backoff_count": int(state.backoff_count),
                }
            )
            if state.backoff_count >= max(1, int(config.max_backoffs_before_rollback)):
                state.rollback_requested = True
                state.rollback_reason = (
                    "phase2_instability_repeated_backoff_limit_reached"
                    f"(backoff_count={state.backoff_count})"
                )
                state.events.append(
                    {
                        "epoch": int(epoch),
                        "batch": int(batch_idx),
                        "event": "rollback_requested",
                        "reason": str(state.rollback_reason),
                    }
                )
    else:
        state.consecutive_explosions = 0
        if state.ema_gen_nll is None:
            state.ema_gen_nll = float(gen_nll)
        else:
            alpha = float(min(max(config.nll_ema_alpha, 0.0), 1.0))
            state.ema_gen_nll = (1.0 - alpha) * float(state.ema_gen_nll) + alpha * float(gen_nll)

    return bool(exploded), bool(backoff_applied), float(threshold)


def _stats_finite(stats: dict[str, float]) -> bool:
    keys = [
        "total",
        "minutes_mae",
        "count_loss",
        "member_loss",
        "minutes_nll",
        "flow_nll",
        "crps_fpts",
        "team_energy",
        "count_acc",
        "poss_nll",
        "backbone_nll",
        "three_pa_nll",
        "efficiency_nll",
        "usage_share_nll",
        "emergent_share_aux",
        "ast_share_aux",
        "reb_share_aux",
        "ast_team_rate_aux",
        "reb_opportunity_rate_aux",
    ]
    return all(math.isfinite(float(stats.get(k, float("nan")))) for k in keys)


def _update_early_stop(
    *,
    epoch: int,
    metric_value: float,
    coupled_epochs: int,
    config: EarlyStopConfig,
    state: EarlyStopState,
    metric_name: str = "val_total",
) -> bool:
    if int(config.patience) <= 0:
        return False
    if math.isfinite(float(metric_value)) and float(metric_value) < (float(state.best_metric) - float(config.min_delta)):
        state.best_metric = float(metric_value)
        state.best_epoch = int(epoch)
        state.bad_epochs = 0
        return False
    if int(epoch) < int(config.min_epochs):
        return False
    if int(coupled_epochs) < int(config.min_coupled_epochs):
        return False
    if state.best_epoch <= 0:
        return False
    state.bad_epochs += 1
    if int(state.bad_epochs) < int(config.patience):
        return False
    state.stop_requested = True
    state.stop_epoch = int(epoch)
    state.stop_reason = (
        f"no {str(metric_name)} improvement >= {float(config.min_delta):.6f} "
        f"for {int(config.patience)} epoch(s)"
    )
    return True


def _resolve_early_stop_metric_value(
    *,
    metric_name: str,
    val_total: float,
    val_poss_regression: float,
    w_poss_regression: float,
) -> float:
    name = str(metric_name).strip().lower()
    if name == "val_total":
        return float(val_total)
    if name == "val_total_ex_possreg":
        return float(val_total) - float(w_poss_regression) * float(val_poss_regression)
    raise ValueError(f"Unsupported early-stop metric: {metric_name}")


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    active_threshold: float,
    min_active_count: int,
    max_active_count: int,
    run_phase2_flow: bool,
    run_phase3_decision: bool,
    w_minutes: float,
    w_minutes_nll: float,
    w_count: float,
    w_member: float,
    w_flow_nll: float,
    w_crps_fpts: float,
    w_team_energy: float,
    minutes_nll_sigma: float,
    phase3_num_samples: int,
    phase3_active_temperature: float,
    phase3_stop_grad: bool,
    positive_weight: float,
    epoch_index: int,
    backbone_grad_clip_norm: float,
    flow_grad_clip_norm: float,
    encoder_grad_clip_norm: float = -1.0,
    backbone_head_grad_clip_norm: float = -1.0,
    w_poss_nll: float = 0.0,
    w_backbone_nll: float = 0.0,
    w_three_pa_nll: float = 0.0,
    w_poss_regression: float = 0.0,
    estimated_possessions_idx: int = -1,
    w_efficiency_nll: float = 0.0,
    w_usage_share_nll: float = 0.0,
    w_emergent_share_aux: float = 0.0,
    w_ast_share_aux: float = 0.0,
    w_reb_share_aux: float = 0.0,
    w_ast_team_rate_aux: float = 0.0,
    w_reb_opportunity_rate_aux: float = 0.0,
    w_spread_aux: float = 0.0,
    w_total_aux: float = 0.0,
    spread_total_aux_ramp_epochs: int = 0,
    spread_total_aux_start_scale: float = 1.0,
    spread_aux_target_scale: float = 10.0,
    total_aux_target_scale: float = 25.0,
    spread_aux_huber_delta: float = 1.0,
    total_aux_huber_delta: float = 1.0,
    w_efficiency_mean_aux: float = 0.0,
    vegas_total_idx: int = -1,
    vegas_spread_idx: int = -1,
    vegas_total_missing_idx: int = -1,
    vegas_spread_missing_idx: int = -1,
    enable_possession_backbone: bool = False,
    enable_efficiency_head: bool = False,
    enable_usage_share_head: bool = False,
    detach_backbone: bool = True,
    phase2_stability_config: Phase2StabilityConfig | None = None,
    phase2_stability_state: Phase2StabilityState | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    spread_total_aux_ramp = _resolve_ramped_loss_scale(
        epoch=int(epoch_index),
        ramp_epochs=int(spread_total_aux_ramp_epochs),
        start_scale=float(spread_total_aux_start_scale),
    )
    w_spread_aux_eff = float(w_spread_aux) * float(spread_total_aux_ramp)
    w_total_aux_eff = float(w_total_aux) * float(spread_total_aux_ramp)

    totals = {
        "total": 0.0,
        "minutes_mae": 0.0,
        "count_loss": 0.0,
        "member_loss": 0.0,
        "minutes_nll": 0.0,
        "flow_nll": 0.0,
        "crps_fpts": 0.0,
        "team_energy": 0.0,
        "count_acc": 0.0,
        "poss_nll": 0.0,
        "poss_regression": 0.0,
        "backbone_nll": 0.0,
        "three_pa_nll": 0.0,
        "efficiency_nll": 0.0,
        "usage_share_nll": 0.0,
        "emergent_share_aux": 0.0,
        "ast_share_aux": 0.0,
        "reb_share_aux": 0.0,
        "ast_team_rate_aux": 0.0,
        "reb_opportunity_rate_aux": 0.0,
        "spread_aux": 0.0,
        "total_aux": 0.0,
        "efficiency_mean_aux": 0.0,
        "steps": 0,
        "skipped_batches": 0,
        "instability_events": 0,
    }
    flow_params: list[nn.Parameter] = []
    encoder_params: list[nn.Parameter] = []
    backbone_head_params: list[nn.Parameter] = []
    base_params: list[nn.Parameter] = []
    if training:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if _is_flow_param_name(name):
                flow_params.append(param)
            elif _is_backbone_head_param_name(name):
                backbone_head_params.append(param)
            elif _is_encoder_param_name(name):
                encoder_params.append(param)
            else:
                base_params.append(param)

    for batch_idx, batch in enumerate(loader, start=1):
        player_features = batch["player_features"].to(device=device)
        player_valid_mask = batch["player_valid_mask"].to(device=device)
        y_minutes = batch["y_minutes"].to(device=device)
        flow_targets = batch["flow_targets"].to(device=device)
        flow_observed_mask = batch["flow_observed_mask"].to(device=device)
        game_features = batch["game_features"].to(device=device)
        team_features = batch["team_features"].to(device=device)

        y_flat = _flatten_side(y_minutes)
        valid_flat = _flatten_side(player_valid_mask)
        team_index = _team_index(batch_size=y_minutes.shape[0], n_team_slots=y_minutes.shape[2], device=device)

        label_targets = build_active_set_labels(
            y_flat,
            valid_flat,
            team_index,
            threshold=float(active_threshold),
            min_active_count=int(min_active_count),
            max_active_count=int(max_active_count),
        )
        target_active_mask_2d = label_targets.active_mask.view(y_minutes.shape[0], 2, y_minutes.shape[2])

        flow_mask = flow_observed_mask
        if bool(run_phase2_flow):
            # No flow likelihood terms for DNP rows; only score rows with observed count labels.
            dnp_mask = (y_minutes > 0.0).unsqueeze(-1)
            flow_mask = flow_mask & player_valid_mask.unsqueeze(-1) & dnp_mask
        flow_target_flat = torch.cat([flow_targets[:, 0], flow_targets[:, 1]], dim=1)
        flow_observed_flat = torch.cat([flow_observed_mask[:, 0], flow_observed_mask[:, 1]], dim=1)

        with torch.set_grad_enabled(training):
            out = model(
                player_features,
                player_valid_mask,
                game_features=game_features,
                team_features=team_features,
                sample_active=False,
                active_temperature=1.0,
                target_active_mask=target_active_mask_2d,
                minutes_use_target_active=True,
                run_flow=bool(run_phase2_flow),
                flow_targets=flow_targets if bool(run_phase2_flow) else None,
                flow_observed_mask=flow_mask if bool(run_phase2_flow) else None,
                detach_backbone=bool(detach_backbone),
            )
            active_losses = compute_active_set_losses(
                count_logits=out.active.count_logits,
                player_logits=out.active.player_logits,
                count_targets=label_targets.count_targets,
                active_targets=label_targets.active_mask,
                valid_mask=out.player_valid_mask,
                min_active_count=int(min_active_count),
                positive_weight=float(positive_weight),
            )
            minutes_mae = _masked_mae(out.minutes.minutes, y_flat, out.player_valid_mask)
            minutes_nll = _gaussian_nll(
                out.minutes.preferences,
                y_flat,
                label_targets.active_mask & out.player_valid_mask,
                sigma=float(minutes_nll_sigma),
            )
            if bool(run_phase2_flow):
                if out.flow is None:
                    raise RuntimeError("run_phase2_flow=True but model did not return flow outputs")
                flow_nll = out.flow.nll_mean
            else:
                flow_nll = torch.zeros((), dtype=minutes_mae.dtype, device=minutes_mae.device)

            if bool(run_phase3_decision):
                if not bool(run_phase2_flow):
                    raise RuntimeError("run_phase3_decision=True requires run_phase2_flow=True")
                if not hasattr(model, "flow_target_columns"):
                    raise RuntimeError("run_phase3_decision=True requires model.flow_target_columns")
                flow_target_columns: list[str] = list(model.flow_target_columns)  # type: ignore[attr-defined]
                sampled_fpts = _sample_decision_fpts(
                    model,
                    context_out=out,
                    num_samples=int(phase3_num_samples),
                    active_temperature=float(phase3_active_temperature),
                )
                if bool(phase3_stop_grad):
                    sampled_fpts = sampled_fpts.detach()
                target_flow = _project_flow_stats_to_contract(
                    flow_target_flat,
                    flow_target_columns=flow_target_columns,
                )
                target_fpts = _compute_dk_fpts_from_flow(
                    target_flow,
                    flow_target_columns=flow_target_columns,
                )
                decision_mask = flow_observed_flat.all(dim=-1) & out.player_valid_mask
                crps_fpts = compute_crps_loss(sampled_fpts, target_fpts, decision_mask)
                team_energy = compute_team_energy_score(
                    sampled_fpts,
                    target_fpts,
                    decision_mask,
                    out.player_team_index,
                    eps=1e-6,
                )
                crps_fpts = torch.nan_to_num(crps_fpts, nan=0.0, posinf=50.0, neginf=-50.0)
                team_energy = torch.nan_to_num(team_energy, nan=0.0, posinf=50.0, neginf=-50.0)
            else:
                crps_fpts = torch.zeros((), dtype=minutes_mae.dtype, device=minutes_mae.device)
                team_energy = torch.zeros((), dtype=minutes_mae.dtype, device=minutes_mae.device)

            # Possession backbone losses (section 15)
            poss_nll_loss = torch.zeros((), dtype=minutes_mae.dtype, device=minutes_mae.device)
            poss_regression_loss = torch.zeros((), dtype=minutes_mae.dtype, device=minutes_mae.device)
            backbone_nll_loss = torch.zeros((), dtype=minutes_mae.dtype, device=minutes_mae.device)
            three_pa_nll_loss = torch.zeros((), dtype=minutes_mae.dtype, device=minutes_mae.device)
            if bool(enable_possession_backbone) and out.possession is not None and out.backbone is not None:
                # Extract team-level truth counts from flow_targets (B, 2, 15, S)
                # Sum across players per team to get team totals
                ftc = list(model.flow_target_columns)  # type: ignore[attr-defined]
                required_cols = ["fga2", "fga3", "fta", "oreb", "tov"]
                missing_required_cols = [name for name in required_cols if name not in ftc]
                if missing_required_cols:
                    raise RuntimeError(
                        f"Missing required flow target columns for possession backbone: {missing_required_cols}",
                    )
                fga2_i = _flow_index(ftc, "fga2")
                fga3_i = _flow_index(ftc, "fga3")
                fta_i = _flow_index(ftc, "fta")
                oreb_i = _flow_index(ftc, "oreb")
                tov_i = _flow_index(ftc, "tov")
                max_required_idx = max(fga2_i, fga3_i, fta_i, oreb_i, tov_i)
                if flow_targets.ndim != 4 or int(flow_targets.shape[-1]) <= int(max_required_idx):
                    raise RuntimeError(
                        "Possession backbone requires populated flow_targets stats. "
                        f"Got flow_targets shape={tuple(flow_targets.shape)} with required index={max_required_idx}. "
                        "Use --enable-phase2-flow so labels_boxscore_counts are loaded.",
                    )

                # flow_targets is (B, 2, 15, S); valid players mask is player_valid_mask (B, 2, 15)
                ft = flow_targets  # (B, 2, 15, S)
                vm = player_valid_mask.unsqueeze(-1).to(dtype=ft.dtype)  # (B, 2, 15, 1)
                ft_masked = ft * vm

                # Team sums: (B, 2)
                fga_team = ft_masked[:, :, :, fga2_i].sum(dim=2) + ft_masked[:, :, :, fga3_i].sum(dim=2)
                fta_team = ft_masked[:, :, :, fta_i].sum(dim=2)
                oreb_team = ft_masked[:, :, :, oreb_i].sum(dim=2)
                tov_team = ft_masked[:, :, :, tov_i].sum(dim=2)

                # Only compute losses where flow labels are observed
                flow_obs_any = flow_observed_mask.any(dim=-1)  # (B, 2, 15) -> any stat observed
                team_has_labels = flow_obs_any.any(dim=2)  # (B, 2) -> team has at least one observed player
                game_has_labels = team_has_labels.all(dim=1)  # (B,) -> both teams have labels

                if game_has_labels.any():
                    # Possession truth
                    poss_true = compute_possession_truth(fga_team, oreb_team, tov_team, fta_team)  # (B,)
                    # Possession NLL (only on games with labels)
                    from projections.rotation.possession_backbone import PossessionHead
                    poss_nll_per_game = PossessionHead.nll_student_t(
                        poss_true, out.possession.mu, out.possession.sigma, out.possession.df,
                    )
                    poss_nll_loss = (poss_nll_per_game * game_has_labels.to(dtype=poss_nll_per_game.dtype)).sum() / game_has_labels.to(dtype=poss_nll_per_game.dtype).sum().clamp(min=1.0)

                    # Backbone rate NLL
                    backbone_nll_loss = model.event_backbone.nll_rates(  # type: ignore[attr-defined]
                        out.team_states,
                        out.game_state,
                        poss_true,
                        fta_true=fta_team,
                        tov_true=tov_team,
                        oreb_true=oreb_team,
                        game_features=game_features,
                    )

                    # 3PA share NLL (optional)
                    if hasattr(model, "three_pa_share_head") and model.three_pa_share_head is not None:  # type: ignore[attr-defined]
                        fga3_team = ft_masked[:, :, :, fga3_i].sum(dim=2)
                        three_pa_share_true = fga3_team / fga_team.clamp(min=1.0)
                        three_pa_nll_loss = model.three_pa_share_head.nll(  # type: ignore[attr-defined]
                            out.team_states,
                            out.game_state,
                            fga_team,
                            three_pa_share_true=three_pa_share_true,
                            game_features=game_features,
                        )

                    # Possession regression loss (Approach C): MSE(mu_P, estimated_possessions)
                    if float(w_poss_regression) > 0.0 and int(estimated_possessions_idx) >= 0:
                        est_poss = game_features[:, int(estimated_possessions_idx)]  # (B,)
                        mask_f = game_has_labels.to(dtype=poss_nll_loss.dtype)
                        n_valid = mask_f.sum().clamp(min=1.0)
                        poss_regression_loss = (
                            ((out.possession.mu - est_poss) ** 2 * mask_f).sum() / n_valid
                        )

            efficiency_nll_loss = torch.zeros((), dtype=minutes_mae.dtype, device=minutes_mae.device)
            efficiency_mean_aux_loss = torch.zeros((), dtype=minutes_mae.dtype, device=minutes_mae.device)
            if bool(enable_efficiency_head) and out.efficiency is not None:
                ftc = list(model.flow_target_columns)  # type: ignore[attr-defined]
                required_cols = ["fga2", "fg2m", "fga3", "fg3m", "fta", "ftm"]
                missing_required_cols = [name for name in required_cols if name not in ftc]
                if missing_required_cols:
                    raise RuntimeError(
                        f"Missing required flow target columns for efficiency head: {missing_required_cols}",
                    )
                fga2_i = _flow_index(ftc, "fga2")
                fg2m_i = _flow_index(ftc, "fg2m")
                fga3_i = _flow_index(ftc, "fga3")
                fg3m_i = _flow_index(ftc, "fg3m")
                fta_i = _flow_index(ftc, "fta")
                ftm_i = _flow_index(ftc, "ftm")
                max_required_idx = max(fga2_i, fg2m_i, fga3_i, fg3m_i, fta_i, ftm_i)
                if flow_targets.ndim != 4 or int(flow_targets.shape[-1]) <= int(max_required_idx):
                    raise RuntimeError(
                        "Efficiency head loss requires populated flow_targets stats. "
                        f"Got flow_targets shape={tuple(flow_targets.shape)} with required index={max_required_idx}. "
                        "Use --enable-phase2-flow so labels_boxscore_counts are loaded.",
                    )

                def _obs_mask(a_idx: int, m_idx: int) -> torch.Tensor:
                    return flow_observed_flat[..., a_idx] & flow_observed_flat[..., m_idx] & out.player_valid_mask

                ft_nll = _beta_binomial_nll(
                    attempts=flow_target_flat[..., fta_i],
                    makes=flow_target_flat[..., ftm_i],
                    alpha=out.efficiency.alpha_ft,
                    beta=out.efficiency.beta_ft,
                    mask=_obs_mask(fta_i, ftm_i),
                )
                fg2_nll = _beta_binomial_nll(
                    attempts=flow_target_flat[..., fga2_i],
                    makes=flow_target_flat[..., fg2m_i],
                    alpha=out.efficiency.alpha_fg2,
                    beta=out.efficiency.beta_fg2,
                    mask=_obs_mask(fga2_i, fg2m_i),
                )
                fg3_nll = _beta_binomial_nll(
                    attempts=flow_target_flat[..., fga3_i],
                    makes=flow_target_flat[..., fg3m_i],
                    alpha=out.efficiency.alpha_fg3,
                    beta=out.efficiency.beta_fg3,
                    mask=_obs_mask(fga3_i, fg3m_i),
                )
                efficiency_nll_loss = (ft_nll + fg2_nll + fg3_nll) / 3.0
                if float(w_efficiency_mean_aux) > 0.0:
                    def _rate_mse(a_idx: int, m_idx: int, pred_mean: torch.Tensor) -> torch.Tensor:
                        attempts = flow_target_flat[..., a_idx]
                        makes = flow_target_flat[..., m_idx]
                        mask = (
                            flow_observed_flat[..., a_idx]
                            & flow_observed_flat[..., m_idx]
                            & out.player_valid_mask
                            & (attempts > 0.0)
                        )
                        if not mask.any():
                            return torch.zeros((), dtype=minutes_mae.dtype, device=minutes_mae.device)
                        rate_true = makes / attempts.clamp(min=1.0)
                        err = (pred_mean - rate_true) ** 2
                        mask_f = mask.to(dtype=err.dtype)
                        return (err * mask_f).sum() / mask_f.sum().clamp(min=1.0)

                    ft_mse = _rate_mse(fta_i, ftm_i, out.efficiency.mean_ft)
                    fg2_mse = _rate_mse(fga2_i, fg2m_i, out.efficiency.mean_fg2)
                    fg3_mse = _rate_mse(fga3_i, fg3m_i, out.efficiency.mean_fg3)
                    efficiency_mean_aux_loss = (ft_mse + fg2_mse + fg3_mse) / 3.0

            usage_share_nll_loss = torch.zeros((), dtype=minutes_mae.dtype, device=minutes_mae.device)
            emergent_share_aux_loss = torch.zeros((), dtype=minutes_mae.dtype, device=minutes_mae.device)
            ast_share_aux_loss = torch.zeros((), dtype=minutes_mae.dtype, device=minutes_mae.device)
            reb_share_aux_loss = torch.zeros((), dtype=minutes_mae.dtype, device=minutes_mae.device)
            ast_team_rate_aux_loss = torch.zeros((), dtype=minutes_mae.dtype, device=minutes_mae.device)
            reb_opportunity_rate_aux_loss = torch.zeros((), dtype=minutes_mae.dtype, device=minutes_mae.device)
            spread_aux_loss = torch.zeros((), dtype=minutes_mae.dtype, device=minutes_mae.device)
            total_aux_loss = torch.zeros((), dtype=minutes_mae.dtype, device=minutes_mae.device)
            if bool(run_phase2_flow):
                ftc = list(model.flow_target_columns)  # type: ignore[attr-defined]
                fga2_i = _flow_index(ftc, "fga2")
                fg2m_i = _flow_index(ftc, "fg2m")
                fga3_i = _flow_index(ftc, "fga3")
                fg3m_i = _flow_index(ftc, "fg3m")
                fta_i = _flow_index(ftc, "fta")
                ftm_i = _flow_index(ftc, "ftm")
                oreb_i = _flow_index(ftc, "oreb")
                dreb_i = _flow_index(ftc, "dreb")
                ast_i = _flow_index(ftc, "ast")
                tov_i = _flow_index(ftc, "tov")

                fga_true = flow_target_flat[..., fga2_i] + flow_target_flat[..., fga3_i]
                fgm_true = flow_target_flat[..., fg2m_i] + flow_target_flat[..., fg3m_i]
                missed_fg_true = torch.clamp(fga_true - fgm_true, min=0.0)
                fta_true = flow_target_flat[..., fta_i]
                oreb_true = flow_target_flat[..., oreb_i]
                dreb_true = flow_target_flat[..., dreb_i]
                ast_true = flow_target_flat[..., ast_i]
                tov_true = flow_target_flat[..., tov_i]

                fga_obs = flow_observed_flat[..., fga2_i] & flow_observed_flat[..., fga3_i]
                fgm_obs = flow_observed_flat[..., fg2m_i] & flow_observed_flat[..., fg3m_i]
                missed_fg_obs = (
                    flow_observed_flat[..., fga2_i]
                    & flow_observed_flat[..., fga3_i]
                    & flow_observed_flat[..., fg2m_i]
                    & flow_observed_flat[..., fg3m_i]
                )
                fta_obs = flow_observed_flat[..., fta_i]
                oreb_obs = flow_observed_flat[..., oreb_i]
                dreb_obs = flow_observed_flat[..., dreb_i]
                ast_obs = flow_observed_flat[..., ast_i]
                tov_obs = flow_observed_flat[..., tov_i]

                if bool(enable_usage_share_head) and out.usage_share is not None:
                    loss_fga = _team_share_ce_loss(
                        logits=out.usage_share.fga_logits,
                        attempts_true=fga_true,
                        valid_mask=out.player_valid_mask,
                        team_index=out.player_team_index,
                        observed_mask=fga_obs,
                    )
                    loss_fta = _team_share_ce_loss(
                        logits=out.usage_share.fta_logits,
                        attempts_true=fta_true,
                        valid_mask=out.player_valid_mask,
                        team_index=out.player_team_index,
                        observed_mask=fta_obs,
                    )
                    loss_tov = _team_share_ce_loss(
                        logits=out.usage_share.tov_logits,
                        attempts_true=tov_true,
                        valid_mask=out.player_valid_mask,
                        team_index=out.player_team_index,
                        observed_mask=tov_obs,
                    )
                    usage_share_nll_loss = (loss_fga + loss_fta + loss_tov) / 3.0

                if (
                    float(w_emergent_share_aux) > 0.0
                    or float(w_ast_share_aux) > 0.0
                    or float(w_reb_share_aux) > 0.0
                    or float(w_ast_team_rate_aux) > 0.0
                    or float(w_reb_opportunity_rate_aux) > 0.0
                    or float(w_spread_aux) > 0.0
                    or float(w_total_aux) > 0.0
                ):
                    z0 = torch.zeros(
                        (
                            out.player_states.shape[0],
                            out.player_states.shape[1],
                            len(ftc),
                        ),
                        dtype=out.player_states.dtype,
                        device=out.player_states.device,
                    )
                    emergent_flow = model.flow_head.sample(  # type: ignore[attr-defined]
                        z0,
                        player_states=out.player_states,
                        team_states=out.team_states,
                        game_state=out.game_state,
                        player_team_index=out.player_team_index,
                        valid_mask=out.player_valid_mask,
                        observed_mask=out.player_valid_mask.unsqueeze(-1).expand_as(z0),
                    )
                    emergent_flow = _project_flow_stats_to_contract(
                        emergent_flow,
                        flow_target_columns=ftc,
                    )
                    fg2m_em = emergent_flow[..., fg2m_i]
                    fg3m_em = emergent_flow[..., fg3m_i]
                    fga_em = emergent_flow[..., fga2_i] + emergent_flow[..., fga3_i]
                    fgm_em = fg2m_em + fg3m_em
                    missed_fg_em = torch.clamp(fga_em - fgm_em, min=0.0)
                    fta_em = emergent_flow[..., fta_i]
                    ftm_em = emergent_flow[..., ftm_i]
                    oreb_em = emergent_flow[..., oreb_i]
                    dreb_em = emergent_flow[..., dreb_i]
                    ast_em = emergent_flow[..., ast_i]
                    tov_em = emergent_flow[..., tov_i]
                    if float(w_spread_aux) > 0.0 or float(w_total_aux) > 0.0:
                        if float(w_spread_aux) > 0.0 and int(vegas_spread_idx) < 0:
                            raise RuntimeError(
                                "w_spread_aux > 0 requires vegas_spread in --game-feature-cols",
                            )
                        if float(w_total_aux) > 0.0 and int(vegas_total_idx) < 0:
                            raise RuntimeError(
                                "w_total_aux > 0 requires vegas_total in --game-feature-cols",
                            )

                        pts_em = 2.0 * fg2m_em + 3.0 * fg3m_em + ftm_em
                        def _team_sum(values: torch.Tensor, team_id: int) -> torch.Tensor:
                            mask = out.player_valid_mask & (out.player_team_index == int(team_id))
                            return (values * mask.to(dtype=values.dtype)).sum(dim=1)

                        home_pts = _team_sum(pts_em, 0)
                        away_pts = _team_sum(pts_em, 1)
                        pred_total = home_pts + away_pts
                        pred_spread = home_pts - away_pts

                        if float(w_total_aux) > 0.0:
                            target_total = game_features[:, int(vegas_total_idx)]
                            total_mask = torch.ones_like(pred_total, dtype=torch.bool)
                            if int(vegas_total_missing_idx) >= 0:
                                total_mask = game_features[:, int(vegas_total_missing_idx)] < 0.5
                            total_aux_loss = _masked_scaled_huber_loss(
                                pred=pred_total,
                                target=target_total,
                                mask=total_mask,
                                scale=float(total_aux_target_scale),
                                delta=float(total_aux_huber_delta),
                            )

                        if float(w_spread_aux) > 0.0:
                            # Dataset vegas_spread follows book line sign (home favorite often negative).
                            # Convert to home-margin convention to match pred_spread = home_pts - away_pts.
                            target_spread = -game_features[:, int(vegas_spread_idx)]
                            spread_mask = torch.ones_like(pred_spread, dtype=torch.bool)
                            if int(vegas_spread_missing_idx) >= 0:
                                spread_mask = game_features[:, int(vegas_spread_missing_idx)] < 0.5
                            spread_aux_loss = _masked_scaled_huber_loss(
                                pred=pred_spread,
                                target=target_spread,
                                mask=spread_mask,
                                scale=float(spread_aux_target_scale),
                                delta=float(spread_aux_huber_delta),
                            )
                    if float(w_emergent_share_aux) > 0.0:
                        loss_fga_em = _team_share_ce_loss(
                            logits=torch.log(torch.clamp(fga_em, min=1e-6)),
                            attempts_true=fga_true,
                            valid_mask=out.player_valid_mask,
                            team_index=out.player_team_index,
                            observed_mask=fga_obs,
                        )
                        loss_fta_em = _team_share_ce_loss(
                            logits=torch.log(torch.clamp(fta_em, min=1e-6)),
                            attempts_true=fta_true,
                            valid_mask=out.player_valid_mask,
                            team_index=out.player_team_index,
                            observed_mask=fta_obs,
                        )
                        loss_tov_em = _team_share_ce_loss(
                            logits=torch.log(torch.clamp(tov_em, min=1e-6)),
                            attempts_true=tov_true,
                            valid_mask=out.player_valid_mask,
                            team_index=out.player_team_index,
                            observed_mask=tov_obs,
                        )
                        emergent_share_aux_loss = (loss_fga_em + loss_fta_em + loss_tov_em) / 3.0

                    if float(w_ast_share_aux) > 0.0:
                        ast_share_aux_loss = _team_share_ce_loss(
                            logits=torch.log(torch.clamp(ast_em, min=1e-6)),
                            attempts_true=ast_true,
                            valid_mask=out.player_valid_mask,
                            team_index=out.player_team_index,
                            observed_mask=ast_obs,
                        )

                    if (
                        float(w_reb_share_aux) > 0.0
                        or float(w_ast_team_rate_aux) > 0.0
                        or float(w_reb_opportunity_rate_aux) > 0.0
                    ):
                        oreb_true_team, oreb_true_seen = _team_sum_by_side(
                            values=oreb_true,
                            valid_mask=out.player_valid_mask,
                            team_index=out.player_team_index,
                            observed_mask=oreb_obs,
                        )
                        dreb_true_team, dreb_true_seen = _team_sum_by_side(
                            values=dreb_true,
                            valid_mask=out.player_valid_mask,
                            team_index=out.player_team_index,
                            observed_mask=dreb_obs,
                        )
                        ast_true_team, ast_true_seen = _team_sum_by_side(
                            values=ast_true,
                            valid_mask=out.player_valid_mask,
                            team_index=out.player_team_index,
                            observed_mask=ast_obs,
                        )
                        fgm_true_team, fgm_true_seen = _team_sum_by_side(
                            values=fgm_true,
                            valid_mask=out.player_valid_mask,
                            team_index=out.player_team_index,
                            observed_mask=fgm_obs,
                        )
                        # First soft-structure slice uses missed FGs only; missed FTs stay out of the opportunity budget.
                        own_missed_true_team, own_missed_true_seen = _team_sum_by_side(
                            values=missed_fg_true,
                            valid_mask=out.player_valid_mask,
                            team_index=out.player_team_index,
                            observed_mask=missed_fg_obs,
                        )
                        opp_missed_true_team = torch.stack(
                            [own_missed_true_team[:, 1], own_missed_true_team[:, 0]],
                            dim=1,
                        )
                        opp_missed_true_seen = torch.stack(
                            [own_missed_true_seen[:, 1], own_missed_true_seen[:, 0]],
                            dim=1,
                        )

                        oreb_em_team, _ = _team_sum_by_side(
                            values=oreb_em,
                            valid_mask=out.player_valid_mask,
                            team_index=out.player_team_index,
                        )
                        dreb_em_team, _ = _team_sum_by_side(
                            values=dreb_em,
                            valid_mask=out.player_valid_mask,
                            team_index=out.player_team_index,
                        )
                        ast_em_team, _ = _team_sum_by_side(
                            values=ast_em,
                            valid_mask=out.player_valid_mask,
                            team_index=out.player_team_index,
                        )

                    if float(w_reb_share_aux) > 0.0:
                        loss_oreb_em = _team_share_ce_loss(
                            logits=torch.log(torch.clamp(oreb_em, min=1e-6)),
                            attempts_true=oreb_true,
                            valid_mask=out.player_valid_mask,
                            team_index=out.player_team_index,
                            observed_mask=oreb_obs,
                        )
                        loss_dreb_em = _team_share_ce_loss(
                            logits=torch.log(torch.clamp(dreb_em, min=1e-6)),
                            attempts_true=dreb_true,
                            valid_mask=out.player_valid_mask,
                            team_index=out.player_team_index,
                            observed_mask=dreb_obs,
                        )
                        reb_share_aux_loss = (loss_oreb_em + loss_dreb_em) / 2.0

                    if float(w_ast_team_rate_aux) > 0.0:
                        # Score AST against the observed made-shot budget so the aux path cannot
                        # improve by co-moving the predicted denominator.
                        ast_team_rate_aux_loss = _team_fixed_opportunity_rate_mse_loss(
                            pred_numerator=ast_em_team,
                            true_numerator=ast_true_team,
                            true_denominator=fgm_true_team,
                            observed_mask=ast_true_seen & fgm_true_seen,
                        )

                    if float(w_reb_opportunity_rate_aux) > 0.0:
                        # Use observed miss budgets so rebound auxiliaries cannot inflate team totals
                        # by shrinking or expanding a self-generated opportunity denominator.
                        loss_oreb_rate = _team_fixed_opportunity_rate_mse_loss(
                            pred_numerator=oreb_em_team,
                            true_numerator=oreb_true_team,
                            true_denominator=own_missed_true_team,
                            observed_mask=oreb_true_seen & own_missed_true_seen,
                        )
                        loss_dreb_rate = _team_fixed_opportunity_rate_mse_loss(
                            pred_numerator=dreb_em_team,
                            true_numerator=dreb_true_team,
                            true_denominator=opp_missed_true_team,
                            observed_mask=dreb_true_seen & opp_missed_true_seen,
                        )
                        reb_opportunity_rate_aux_loss = (loss_oreb_rate + loss_dreb_rate) / 2.0

            total_loss = (
                float(w_minutes) * minutes_mae
                + float(w_count) * active_losses["count_loss"]
                + float(w_member) * active_losses["member_loss"]
                + float(w_minutes_nll) * minutes_nll
                + float(w_flow_nll) * flow_nll
                + float(w_crps_fpts) * crps_fpts
                + float(w_team_energy) * team_energy
                + float(w_poss_nll) * poss_nll_loss
                + float(w_poss_regression) * poss_regression_loss
                + float(w_backbone_nll) * backbone_nll_loss
                + float(w_three_pa_nll) * three_pa_nll_loss
                + float(w_efficiency_nll) * efficiency_nll_loss
                + float(w_usage_share_nll) * usage_share_nll_loss
                + float(w_emergent_share_aux) * emergent_share_aux_loss
                + float(w_ast_share_aux) * ast_share_aux_loss
                + float(w_reb_share_aux) * reb_share_aux_loss
                + float(w_ast_team_rate_aux) * ast_team_rate_aux_loss
                + float(w_reb_opportunity_rate_aux) * reb_opportunity_rate_aux_loss
                + float(w_spread_aux_eff) * spread_aux_loss
                + float(w_total_aux_eff) * total_aux_loss
                + float(w_efficiency_mean_aux) * efficiency_mean_aux_loss
            )

            skip_step = False
            if training and not bool(torch.isfinite(total_loss)):
                totals["instability_events"] += 1
                totals["skipped_batches"] += 1
                skip_step = True
            if (
                training
                and bool(run_phase2_flow)
                and float(w_minutes_nll + w_flow_nll) > 0.0
                and phase2_stability_config is not None
                and phase2_stability_state is not None
            ):
                gen_nll = float((minutes_nll + flow_nll).detach().item())
                exploded, backoff_applied, threshold = _update_phase2_nll_guard(
                    epoch=int(epoch_index),
                    batch_idx=int(batch_idx),
                    gen_nll=float(gen_nll),
                    config=phase2_stability_config,
                    state=phase2_stability_state,
                )
                if exploded:
                    totals["instability_events"] += 1
                    skip_step = True
                    totals["skipped_batches"] += 1
                    if backoff_applied:
                        print(
                            (
                                "[phase2][nll-guard] "
                                f"epoch={epoch_index:03d} batch={batch_idx:04d} "
                                f"gen_nll={gen_nll:.4f} threshold={threshold:.4f} "
                                f"a2_scale={phase2_stability_state.a2_scale:.4f} "
                                f"backoff_count={phase2_stability_state.backoff_count}"
                            ),
                            flush=True,
                        )
                    if phase2_stability_state.rollback_requested:
                        print(
                            (
                                "[phase2][rollback] "
                                f"epoch={epoch_index:03d} batch={batch_idx:04d} "
                                f"reason={phase2_stability_state.rollback_reason}"
                            ),
                            flush=True,
                        )

            if training:
                if skip_step or (
                    phase2_stability_state is not None and bool(phase2_stability_state.rollback_requested)
                ):
                    optimizer.zero_grad(set_to_none=True)
                else:
                    optimizer.zero_grad(set_to_none=True)
                    total_loss.backward()
                    encoder_clip = (
                        float(encoder_grad_clip_norm)
                        if float(encoder_grad_clip_norm) > 0.0
                        else float(backbone_grad_clip_norm)
                    )
                    backbone_head_clip = (
                        float(backbone_head_grad_clip_norm)
                        if float(backbone_head_grad_clip_norm) > 0.0
                        else float(backbone_grad_clip_norm)
                    )
                    if encoder_params:
                        torch.nn.utils.clip_grad_norm_(
                            encoder_params,
                            max_norm=max(0.0, float(encoder_clip)),
                        )
                    if backbone_head_params:
                        torch.nn.utils.clip_grad_norm_(
                            backbone_head_params,
                            max_norm=max(0.0, float(backbone_head_clip)),
                        )
                    if base_params:
                        torch.nn.utils.clip_grad_norm_(
                            base_params,
                            max_norm=max(0.0, float(backbone_grad_clip_norm)),
                        )
                    if bool(run_phase2_flow) and flow_params:
                        torch.nn.utils.clip_grad_norm_(
                            flow_params,
                            max_norm=max(0.0, float(flow_grad_clip_norm)),
                        )
                    optimizer.step()

        with torch.no_grad():
            pred_counts = out.active.sampled_counts
            count_acc = (pred_counts == label_targets.count_targets).to(dtype=torch.float32).mean()

            totals["total"] += float(total_loss.item())
            totals["minutes_mae"] += float(minutes_mae.item())
            totals["count_loss"] += float(active_losses["count_loss"].item())
            totals["member_loss"] += float(active_losses["member_loss"].item())
            totals["minutes_nll"] += float(minutes_nll.item())
            totals["flow_nll"] += float(flow_nll.item())
            totals["crps_fpts"] += float(crps_fpts.item())
            totals["team_energy"] += float(team_energy.item())
            totals["count_acc"] += float(count_acc.item())
            totals["poss_nll"] += float(poss_nll_loss.item())
            totals["poss_regression"] += float(poss_regression_loss.item())
            totals["backbone_nll"] += float(backbone_nll_loss.item())
            totals["three_pa_nll"] += float(three_pa_nll_loss.item())
            totals["efficiency_nll"] += float(efficiency_nll_loss.item())
            totals["usage_share_nll"] += float(usage_share_nll_loss.item())
            totals["emergent_share_aux"] += float(emergent_share_aux_loss.item())
            totals["ast_share_aux"] += float(ast_share_aux_loss.item())
            totals["reb_share_aux"] += float(reb_share_aux_loss.item())
            totals["ast_team_rate_aux"] += float(ast_team_rate_aux_loss.item())
            totals["reb_opportunity_rate_aux"] += float(reb_opportunity_rate_aux_loss.item())
            totals["spread_aux"] += float(spread_aux_loss.item())
            totals["total_aux"] += float(total_aux_loss.item())
            totals["efficiency_mean_aux"] += float(efficiency_mean_aux_loss.item())
            totals["steps"] += 1
        if phase2_stability_state is not None and bool(phase2_stability_state.rollback_requested):
            break

    denom = max(1, totals["steps"])
    return {
        "total": totals["total"] / denom,
        "minutes_mae": totals["minutes_mae"] / denom,
        "count_loss": totals["count_loss"] / denom,
        "member_loss": totals["member_loss"] / denom,
        "minutes_nll": totals["minutes_nll"] / denom,
        "flow_nll": totals["flow_nll"] / denom,
        "crps_fpts": totals["crps_fpts"] / denom,
        "team_energy": totals["team_energy"] / denom,
        "count_acc": totals["count_acc"] / denom,
        "poss_nll": totals["poss_nll"] / denom,
        "poss_regression": totals["poss_regression"] / denom,
        "backbone_nll": totals["backbone_nll"] / denom,
        "three_pa_nll": totals["three_pa_nll"] / denom,
        "efficiency_nll": totals["efficiency_nll"] / denom,
        "usage_share_nll": totals["usage_share_nll"] / denom,
        "emergent_share_aux": totals["emergent_share_aux"] / denom,
        "ast_share_aux": totals["ast_share_aux"] / denom,
        "reb_share_aux": totals["reb_share_aux"] / denom,
        "ast_team_rate_aux": totals["ast_team_rate_aux"] / denom,
        "reb_opportunity_rate_aux": totals["reb_opportunity_rate_aux"] / denom,
        "spread_aux": totals["spread_aux"] / denom,
        "total_aux": totals["total_aux"] / denom,
        "efficiency_mean_aux": totals["efficiency_mean_aux"] / denom,
        "skipped_batches": float(totals["skipped_batches"]),
        "instability_events": float(totals["instability_events"]),
        "rollback_requested": 1.0 if (phase2_stability_state and phase2_stability_state.rollback_requested) else 0.0,
    }


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument(
        "--init-model-pt",
        type=str,
        default=None,
        help="Optional checkpoint path for warm-start (for example Phase 1 model.pt).",
    )
    parser.add_argument("--val-days", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--active-threshold", type=float, default=4.0)
    parser.add_argument("--min-active-count", type=int, default=5)
    parser.add_argument("--max-active-count", type=int, default=13)
    parser.add_argument(
        "--overflow-protected-prior-play-prob-floor",
        type=float,
        default=PROTECTED_PRIOR_PLAY_PROB_FLOOR,
    )
    parser.add_argument(
        "--overflow-protected-prior-minutes-floor",
        type=float,
        default=PROTECTED_PRIOR_MINUTES_FLOOR,
    )
    parser.add_argument(
        "--overflow-risk-weight-consecutive-active-dnp",
        type=float,
        default=OVERFLOW_RISK_WEIGHT_CONSECUTIVE_ACTIVE_DNP,
    )
    parser.add_argument(
        "--overflow-risk-weight-active-but-dnp-rate-last10",
        type=float,
        default=OVERFLOW_RISK_WEIGHT_ACTIVE_BUT_DNP_RATE_LAST10,
    )
    parser.add_argument(
        "--overflow-risk-weight-inactive-streak-len",
        type=float,
        default=OVERFLOW_RISK_WEIGHT_INACTIVE_STREAK_LEN,
    )
    parser.add_argument(
        "--overflow-keep-weight-prior-play-prob",
        type=float,
        default=OVERFLOW_KEEP_WEIGHT_PRIOR_PLAY_PROB,
    )
    parser.add_argument(
        "--overflow-keep-weight-prior-minutes",
        type=float,
        default=OVERFLOW_KEEP_WEIGHT_PRIOR_MINUTES,
    )

    parser.add_argument("--w-minutes", type=float, default=1.0)
    parser.add_argument("--w-minutes-nll", type=float, default=1.0)
    parser.add_argument("--w-count", type=float, default=0.5)
    parser.add_argument("--w-member", type=float, default=0.5)
    parser.add_argument("--w-flow-nll", type=float, default=1.0)
    parser.add_argument("--minutes-nll-sigma", type=float, default=6.0)
    parser.add_argument("--active-positive-weight", type=float, default=1.0)
    parser.add_argument("--enable-phase2-flow", action="store_true")
    parser.add_argument("--phase2-flow-warmup-epochs", type=int, default=4)
    parser.add_argument(
        "--phase2-flow-delay-epochs",
        type=int,
        default=0,
        help="Completely disable flow head for the first N epochs (0 disables). "
        "Useful when warm-starting with expanded backbone inputs that need "
        "time to stabilize before flow training begins.",
    )
    parser.add_argument("--phase2-anchor-start-weight", type=float, default=1.0)
    parser.add_argument("--phase2-anchor-end-weight", type=float, default=0.5)
    parser.add_argument("--phase2-nll-guard-ratio", type=float, default=3.0)
    parser.add_argument("--phase2-nll-guard-abs", type=float, default=25.0)
    parser.add_argument("--phase2-nll-guard-ema-alpha", type=float, default=0.1)
    parser.add_argument("--phase2-nll-guard-consecutive-batches", type=int, default=2)
    parser.add_argument("--phase2-max-backoffs-before-rollback", type=int, default=3)
    parser.add_argument("--phase2-min-a2-scale", type=float, default=0.125)
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Stop after N consecutive non-improving validation epochs (0 disables).",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=0.0,
        help="Minimum val_total improvement required to reset patience.",
    )
    parser.add_argument(
        "--early-stop-min-epochs",
        type=int,
        default=0,
        help="Do not trigger early stopping before this epoch.",
    )
    parser.add_argument(
        "--early-stop-min-coupled-epochs",
        type=int,
        default=0,
        help="Do not trigger early stopping until backbone-coupled epochs reach this count.",
    )
    parser.add_argument(
        "--early-stop-metric",
        type=str,
        default="val_total",
        choices=["val_total", "val_total_ex_possreg"],
        help="Validation metric used for early stopping.",
    )
    parser.add_argument("--flow-num-blocks", type=int, default=4)
    parser.add_argument("--flow-scale-clip", type=float, default=3.0)  # H1 fix: 2.0 → 3.0
    parser.add_argument("--flow-context-mode", type=str, default="attention", choices=["mean", "attention"])  # H2 fix
    parser.add_argument("--backbone-grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--flow-grad-clip-norm", type=float, default=5.0)
    parser.add_argument(
        "--encoder-grad-clip-norm",
        type=float,
        default=-1.0,
        help="Gradient clip for shared encoder params. <=0 falls back to --backbone-grad-clip-norm.",
    )
    parser.add_argument(
        "--backbone-head-grad-clip-norm",
        type=float,
        default=-1.0,
        help="Gradient clip for possession/event/3PA backbone heads. <=0 falls back to --backbone-grad-clip-norm.",
    )
    parser.add_argument(
        "--encoder-lr-scale",
        type=float,
        default=1.0,
        help="LR multiplier for shared encoder params (encoder/projections/tokens).",
    )
    parser.add_argument(
        "--backbone-head-lr-scale",
        type=float,
        default=1.0,
        help="LR multiplier for possession/event/3PA backbone head params.",
    )
    parser.add_argument("--enable-phase3-decision", action="store_true")
    parser.add_argument("--w-crps-fpts", type=float, default=1.0)
    parser.add_argument("--w-team-energy", type=float, default=0.25)
    parser.add_argument("--phase3-num-samples", type=int, default=16)
    parser.add_argument("--phase3-active-temperature", type=float, default=1.0)
    parser.add_argument("--phase3-stop-grad", action="store_true")
    parser.add_argument("--enable-efficiency-head", action="store_true")
    parser.add_argument(
        "--efficiency-head-only",
        action="store_true",
        help="Freeze all non-efficiency params and train only efficiency_head.* weights.",
    )
    parser.add_argument("--w-efficiency-nll", type=float, default=1.0)
    parser.add_argument("--efficiency-head-hidden", type=int, default=128)
    parser.add_argument("--efficiency-ft-prior-mean", type=float, default=0.77)
    parser.add_argument("--efficiency-ft-prior-strength", type=float, default=6.0)
    parser.add_argument("--efficiency-fg2-prior-mean", type=float, default=0.54)
    parser.add_argument("--efficiency-fg2-prior-strength", type=float, default=8.0)
    parser.add_argument("--efficiency-fg3-prior-mean", type=float, default=0.36)
    parser.add_argument("--efficiency-fg3-prior-strength", type=float, default=8.0)
    parser.add_argument("--enable-usage-share-head", action="store_true")
    parser.add_argument(
        "--usage-share-head-only",
        action="store_true",
        help="Freeze all non-usage-share params and train only usage_share_head.* weights.",
    )
    parser.add_argument("--usage-share-head-hidden", type=int, default=128)
    parser.add_argument("--w-usage-share-nll", type=float, default=0.0)
    parser.add_argument(
        "--w-emergent-share-aux",
        type=float,
        default=0.0,
        help="Auxiliary emergent-share CE loss using zero-latent flow samples.",
    )
    parser.add_argument(
        "--w-ast-share-aux",
        type=float,
        default=0.0,
        help="Auxiliary AST share CE loss using the emergent zero-latent flow path.",
    )
    parser.add_argument(
        "--w-reb-share-aux",
        type=float,
        default=0.0,
        help="Auxiliary OREB/DREB share CE loss using the emergent zero-latent flow path.",
    )
    parser.add_argument(
        "--w-ast-team-rate-aux",
        type=float,
        default=0.0,
        help="Auxiliary team AST-to-FGM rate loss on the emergent zero-latent flow path.",
    )
    parser.add_argument(
        "--w-reb-opportunity-rate-aux",
        type=float,
        default=0.0,
        help="Auxiliary OREB/DREB capture-rate loss against missed-FG opportunity on the emergent zero-latent flow path.",
    )
    parser.add_argument(
        "--w-spread-aux",
        type=float,
        default=0.0,
        help="Auxiliary normalized Huber loss on home spread vs -vegas_spread using emergent zero-latent flow points.",
    )
    parser.add_argument(
        "--w-total-aux",
        type=float,
        default=0.0,
        help="Auxiliary normalized Huber loss on game total vs vegas_total using emergent zero-latent flow points.",
    )
    parser.add_argument(
        "--spread-total-aux-ramp-epochs",
        type=int,
        default=0,
        help="Linearly ramp spread/total aux weights over N epochs (0 disables).",
    )
    parser.add_argument(
        "--spread-total-aux-start-scale",
        type=float,
        default=1.0,
        help="Initial scale for spread/total aux weight ramp in [0,1].",
    )
    parser.add_argument(
        "--spread-aux-target-scale",
        type=float,
        default=10.0,
        help="Normalization scale (points) for spread aux error before Huber loss.",
    )
    parser.add_argument(
        "--total-aux-target-scale",
        type=float,
        default=25.0,
        help="Normalization scale (points) for total aux error before Huber loss.",
    )
    parser.add_argument(
        "--spread-aux-huber-delta",
        type=float,
        default=1.0,
        help="Huber delta on normalized spread error.",
    )
    parser.add_argument(
        "--total-aux-huber-delta",
        type=float,
        default=1.0,
        help="Huber delta on normalized total error.",
    )
    parser.add_argument(
        "--w-efficiency-mean-aux",
        type=float,
        default=0.0,
        help="Auxiliary MSE loss on efficiency head mean rates vs observed make rates.",
    )

    # Possession backbone (section 15)
    parser.add_argument("--enable-possession-backbone", action="store_true")
    parser.add_argument("--enable-three-pa-share", action="store_true")
    parser.add_argument("--w-poss-nll", type=float, default=1.0)
    parser.add_argument("--w-backbone-nll", type=float, default=1.0)
    parser.add_argument("--w-three-pa-nll", type=float, default=0.5)
    parser.add_argument(
        "--backbone-loss-ramp-epochs",
        type=int,
        default=0,
        help="Linearly ramp backbone losses to their target values over N epochs (0 disables).",
    )
    parser.add_argument(
        "--poss-loss-start-scale",
        type=float,
        default=1.0,
        help="Initial scale factor for --w-poss-nll when backbone loss ramping is enabled.",
    )
    parser.add_argument(
        "--backbone-loss-start-scale",
        type=float,
        default=1.0,
        help="Initial scale factor for --w-backbone-nll when backbone loss ramping is enabled.",
    )
    parser.add_argument(
        "--three-pa-loss-start-scale",
        type=float,
        default=1.0,
        help="Initial scale factor for --w-three-pa-nll when backbone loss ramping is enabled.",
    )
    parser.add_argument(
        "--w-poss-regression",
        type=float,
        default=0.0,
        help="Weight for MSE(mu_P, estimated_possessions) regression loss (Approach C). 0 disables.",
    )
    parser.add_argument(
        "--poss-regression-start-scale",
        type=float,
        default=1.0,
        help="Initial scale factor for --w-poss-regression when backbone loss ramping is enabled.",
    )
    parser.add_argument(
        "--possession-mu-mode",
        type=str,
        default="absolute",
        choices=["absolute", "baseline_delta"],
        help="Possession head mean parameterization: absolute mu or baseline + delta.",
    )
    parser.add_argument(
        "--possession-mu-baseline",
        type=float,
        default=100.0,
        help="Baseline possessions used when --possession-mu-mode=baseline_delta.",
    )
    parser.add_argument(
        "--backbone-detach-until-epoch",
        type=int,
        default=0,
        help="Detach backbone from encoder for the first N epochs (0=never detach). "
        "Allows flow head to stabilize before backbone gradients flow into the encoder.",
    )

    parser.add_argument(
        "--game-feature-cols",
        type=str,
        default=",".join(DEFAULT_GAME_FEATURE_COLS),
        help="Comma-separated game-level columns fed to the [GAME] token.",
    )
    parser.add_argument(
        "--team-feature-cols",
        type=str,
        default="",
        help="Comma-separated team-level columns fed to [TEAM_*] tokens.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _set_seed(int(args.seed))
    if bool(args.enable_phase3_decision) and not bool(args.enable_phase2_flow):
        raise ValueError("--enable-phase3-decision requires --enable-phase2-flow")
    if bool(args.enable_efficiency_head) and not bool(args.enable_phase2_flow):
        raise ValueError("--enable-efficiency-head requires --enable-phase2-flow")
    if bool(args.efficiency_head_only) and not bool(args.enable_efficiency_head):
        raise ValueError("--efficiency-head-only requires --enable-efficiency-head")
    if bool(args.enable_usage_share_head) and not bool(args.enable_phase2_flow):
        raise ValueError("--enable-usage-share-head requires --enable-phase2-flow")
    if bool(args.usage_share_head_only) and not bool(args.enable_usage_share_head):
        raise ValueError("--usage-share-head-only requires --enable-usage-share-head")
    if bool(args.efficiency_head_only) and bool(args.usage_share_head_only):
        raise ValueError("--efficiency-head-only and --usage-share-head-only are mutually exclusive")
    if (bool(args.enable_possession_backbone) or bool(args.enable_three_pa_share)) and not bool(args.enable_phase2_flow):
        raise ValueError(
            "--enable-possession-backbone/--enable-three-pa-share require --enable-phase2-flow "
            "(backbone supervision uses flow count labels)",
        )
    if bool(args.enable_three_pa_share) and not bool(args.enable_possession_backbone):
        raise ValueError("--enable-three-pa-share requires --enable-possession-backbone")
    if int(args.phase3_num_samples) <= 0:
        raise ValueError("--phase3-num-samples must be > 0")
    if float(args.phase3_active_temperature) <= 0:
        raise ValueError("--phase3-active-temperature must be > 0")
    if int(args.early_stop_patience) < 0:
        raise ValueError("--early-stop-patience must be >= 0")
    if float(args.early_stop_min_delta) < 0.0:
        raise ValueError("--early-stop-min-delta must be >= 0")
    if int(args.early_stop_min_epochs) < 0:
        raise ValueError("--early-stop-min-epochs must be >= 0")
    if int(args.early_stop_min_coupled_epochs) < 0:
        raise ValueError("--early-stop-min-coupled-epochs must be >= 0")
    if int(args.early_stop_min_coupled_epochs) > 0 and not bool(args.enable_possession_backbone):
        raise ValueError("--early-stop-min-coupled-epochs requires --enable-possession-backbone")
    if str(args.early_stop_metric) == "val_total_ex_possreg" and float(args.w_poss_regression) <= 0.0:
        raise ValueError("--early-stop-metric=val_total_ex_possreg requires --w-poss-regression > 0")
    if int(args.backbone_detach_until_epoch) < 0:
        raise ValueError("--backbone-detach-until-epoch must be >= 0")
    if int(args.backbone_loss_ramp_epochs) < 0:
        raise ValueError("--backbone-loss-ramp-epochs must be >= 0")
    if float(args.poss_loss_start_scale) < 0.0:
        raise ValueError("--poss-loss-start-scale must be >= 0")
    if float(args.backbone_loss_start_scale) < 0.0:
        raise ValueError("--backbone-loss-start-scale must be >= 0")
    if float(args.three_pa_loss_start_scale) < 0.0:
        raise ValueError("--three-pa-loss-start-scale must be >= 0")
    if float(args.encoder_lr_scale) <= 0.0:
        raise ValueError("--encoder-lr-scale must be > 0")
    if float(args.backbone_head_lr_scale) <= 0.0:
        raise ValueError("--backbone-head-lr-scale must be > 0")
    if int(args.efficiency_head_hidden) <= 0:
        raise ValueError("--efficiency-head-hidden must be > 0")
    if int(args.usage_share_head_hidden) <= 0:
        raise ValueError("--usage-share-head-hidden must be > 0")
    if float(args.efficiency_ft_prior_mean) <= 0.0 or float(args.efficiency_ft_prior_mean) >= 1.0:
        raise ValueError("--efficiency-ft-prior-mean must be in (0, 1)")
    if float(args.efficiency_fg2_prior_mean) <= 0.0 or float(args.efficiency_fg2_prior_mean) >= 1.0:
        raise ValueError("--efficiency-fg2-prior-mean must be in (0, 1)")
    if float(args.efficiency_fg3_prior_mean) <= 0.0 or float(args.efficiency_fg3_prior_mean) >= 1.0:
        raise ValueError("--efficiency-fg3-prior-mean must be in (0, 1)")
    if float(args.efficiency_ft_prior_strength) <= 0.0:
        raise ValueError("--efficiency-ft-prior-strength must be > 0")
    if float(args.efficiency_fg2_prior_strength) <= 0.0:
        raise ValueError("--efficiency-fg2-prior-strength must be > 0")
    if float(args.efficiency_fg3_prior_strength) <= 0.0:
        raise ValueError("--efficiency-fg3-prior-strength must be > 0")
    if float(args.w_efficiency_nll) < 0.0:
        raise ValueError("--w-efficiency-nll must be >= 0")
    if float(args.w_usage_share_nll) < 0.0:
        raise ValueError("--w-usage-share-nll must be >= 0")
    if float(args.w_emergent_share_aux) < 0.0:
        raise ValueError("--w-emergent-share-aux must be >= 0")
    if float(args.w_ast_share_aux) < 0.0:
        raise ValueError("--w-ast-share-aux must be >= 0")
    if float(args.w_reb_share_aux) < 0.0:
        raise ValueError("--w-reb-share-aux must be >= 0")
    if float(args.w_ast_team_rate_aux) < 0.0:
        raise ValueError("--w-ast-team-rate-aux must be >= 0")
    if float(args.w_reb_opportunity_rate_aux) < 0.0:
        raise ValueError("--w-reb-opportunity-rate-aux must be >= 0")
    if float(args.w_spread_aux) < 0.0:
        raise ValueError("--w-spread-aux must be >= 0")
    if float(args.w_total_aux) < 0.0:
        raise ValueError("--w-total-aux must be >= 0")
    if int(args.spread_total_aux_ramp_epochs) < 0:
        raise ValueError("--spread-total-aux-ramp-epochs must be >= 0")
    if float(args.spread_total_aux_start_scale) < 0.0 or float(args.spread_total_aux_start_scale) > 1.0:
        raise ValueError("--spread-total-aux-start-scale must be in [0, 1]")
    if float(args.spread_aux_target_scale) <= 0.0:
        raise ValueError("--spread-aux-target-scale must be > 0")
    if float(args.total_aux_target_scale) <= 0.0:
        raise ValueError("--total-aux-target-scale must be > 0")
    if float(args.spread_aux_huber_delta) <= 0.0:
        raise ValueError("--spread-aux-huber-delta must be > 0")
    if float(args.total_aux_huber_delta) <= 0.0:
        raise ValueError("--total-aux-huber-delta must be > 0")
    if float(args.w_efficiency_mean_aux) < 0.0:
        raise ValueError("--w-efficiency-mean-aux must be >= 0")
    if float(args.w_emergent_share_aux) > 0.0 and not bool(args.enable_phase2_flow):
        raise ValueError("--w-emergent-share-aux > 0 requires --enable-phase2-flow")
    if (
        float(args.w_ast_share_aux) > 0.0
        or float(args.w_reb_share_aux) > 0.0
        or float(args.w_ast_team_rate_aux) > 0.0
        or float(args.w_reb_opportunity_rate_aux) > 0.0
    ) and not bool(args.enable_phase2_flow):
        raise ValueError("AST/REB structure auxiliary losses require --enable-phase2-flow")
    if (float(args.w_spread_aux) > 0.0 or float(args.w_total_aux) > 0.0) and not bool(args.enable_phase2_flow):
        raise ValueError("--w-spread-aux/--w-total-aux require --enable-phase2-flow")
    if float(args.w_efficiency_mean_aux) > 0.0 and not bool(args.enable_efficiency_head):
        raise ValueError("--w-efficiency-mean-aux requires --enable-efficiency-head")
    if float(args.overflow_protected_prior_play_prob_floor) < 0.0 or float(args.overflow_protected_prior_play_prob_floor) > 1.0:
        raise ValueError("--overflow-protected-prior-play-prob-floor must be within [0, 1]")
    if float(args.overflow_protected_prior_minutes_floor) < 0.0:
        raise ValueError("--overflow-protected-prior-minutes-floor must be >= 0")
    if float(args.overflow_risk_weight_consecutive_active_dnp) < 0.0:
        raise ValueError("--overflow-risk-weight-consecutive-active-dnp must be >= 0")
    if float(args.overflow_risk_weight_active_but_dnp_rate_last10) < 0.0:
        raise ValueError("--overflow-risk-weight-active-but-dnp-rate-last10 must be >= 0")
    if float(args.overflow_risk_weight_inactive_streak_len) < 0.0:
        raise ValueError("--overflow-risk-weight-inactive-streak-len must be >= 0")
    if float(args.overflow_keep_weight_prior_play_prob) < 0.0:
        raise ValueError("--overflow-keep-weight-prior-play-prob must be >= 0")
    if float(args.overflow_keep_weight_prior_minutes) < 0.0:
        raise ValueError("--overflow-keep-weight-prior-minutes must be >= 0")

    dataset_dir = _resolve_dataset_dir(args.dataset_dir)
    features_path = dataset_dir / "features.parquet"
    labels_minutes_path = dataset_dir / "labels_minutes.parquet"
    labels_boxscore_counts_path = dataset_dir / "labels_boxscore_counts.parquet"
    if not features_path.exists() or not labels_minutes_path.exists():
        raise FileNotFoundError(f"Missing dataset files under {dataset_dir}")
    if bool(args.enable_phase2_flow) and not labels_boxscore_counts_path.exists():
        raise FileNotFoundError(f"Missing labels_boxscore_counts.parquet under {dataset_dir} for Phase 2 flow training")

    features_df = _coerce_join_keys(pd.read_parquet(features_path), name="features")
    labels_minutes_df = _coerce_join_keys(pd.read_parquet(labels_minutes_path), name="labels_minutes")
    if "minutes_label" not in labels_minutes_df.columns and "minutes" not in labels_minutes_df.columns:
        raise ValueError("labels_minutes must contain minutes_label or minutes column")

    label_overlap = [c for c in labels_minutes_df.columns if c in features_df.columns and c not in JOIN_KEYS]
    labels_for_merge = labels_minutes_df.drop(columns=label_overlap)
    merged = features_df.merge(
        labels_for_merge,
        on=JOIN_KEYS,
        how="left",
        validate="one_to_one",
    )
    flow_label_cols = list(FLOW_TARGET_COLUMNS_V1) if bool(args.enable_phase2_flow) else []
    if bool(args.enable_phase2_flow):
        labels_counts_df = _coerce_join_keys(
            pd.read_parquet(labels_boxscore_counts_path),
            name="labels_boxscore_counts",
        )
        labels_counts_df = labels_counts_df.drop_duplicates(subset=JOIN_KEYS, keep="last")
        count_overlap = [c for c in labels_counts_df.columns if c in merged.columns and c not in JOIN_KEYS]
        labels_counts_for_merge = labels_counts_df.drop(columns=count_overlap)
        merged = merged.merge(
            labels_counts_for_merge,
            on=JOIN_KEYS,
            how="left",
            validate="one_to_one",
        )
    else:
        flow_label_cols = []
    merged["game_id_norm"] = zfill_game_id_series(merged["game_id"])

    feature_cols = _infer_feature_columns(features_df, labels_minutes_df)
    train_df, val_df, split_meta = _split_train_val(merged, val_days=int(args.val_days))

    feature_mean, feature_std = _compute_feature_norm(train_df, feature_cols)
    game_feature_cols = [c.strip() for c in str(args.game_feature_cols).split(",") if c.strip()]
    team_feature_cols = [c.strip() for c in str(args.team_feature_cols).split(",") if c.strip()]

    estimated_possessions_idx = -1
    if "estimated_possessions" in game_feature_cols:
        estimated_possessions_idx = game_feature_cols.index("estimated_possessions")
    vegas_total_idx = game_feature_cols.index("vegas_total") if "vegas_total" in game_feature_cols else -1
    vegas_spread_idx = game_feature_cols.index("vegas_spread") if "vegas_spread" in game_feature_cols else -1
    vegas_total_missing_idx = (
        game_feature_cols.index("vegas_total_missing") if "vegas_total_missing" in game_feature_cols else -1
    )
    vegas_spread_missing_idx = (
        game_feature_cols.index("vegas_spread_missing") if "vegas_spread_missing" in game_feature_cols else -1
    )
    if float(args.w_poss_regression) > 0.0 and estimated_possessions_idx < 0:
        raise ValueError(
            "--w-poss-regression > 0 requires 'estimated_possessions' in --game-feature-cols"
        )
    if float(args.w_total_aux) > 0.0 and vegas_total_idx < 0:
        raise ValueError("--w-total-aux > 0 requires 'vegas_total' in --game-feature-cols")
    if float(args.w_spread_aux) > 0.0 and vegas_spread_idx < 0:
        raise ValueError("--w-spread-aux > 0 requires 'vegas_spread' in --game-feature-cols")

    train_examples = build_game_level_examples(
        train_df,
        feature_columns=feature_cols,
        feature_mean=feature_mean,
        feature_std=feature_std,
        game_feature_columns=game_feature_cols,
        team_feature_columns=team_feature_cols,
        flow_label_columns=flow_label_cols,
        minutes_label_col="minutes_label" if "minutes_label" in merged.columns else "minutes",
        overflow_protected_prior_play_prob_floor=float(args.overflow_protected_prior_play_prob_floor),
        overflow_protected_prior_minutes_floor=float(args.overflow_protected_prior_minutes_floor),
        overflow_risk_weight_consecutive_active_dnp=float(args.overflow_risk_weight_consecutive_active_dnp),
        overflow_risk_weight_active_but_dnp_rate_last10=float(args.overflow_risk_weight_active_but_dnp_rate_last10),
        overflow_risk_weight_inactive_streak_len=float(args.overflow_risk_weight_inactive_streak_len),
        overflow_keep_weight_prior_play_prob=float(args.overflow_keep_weight_prior_play_prob),
        overflow_keep_weight_prior_minutes=float(args.overflow_keep_weight_prior_minutes),
    )
    val_examples = build_game_level_examples(
        val_df,
        feature_columns=feature_cols,
        feature_mean=feature_mean,
        feature_std=feature_std,
        game_feature_columns=game_feature_cols,
        team_feature_columns=team_feature_cols,
        flow_label_columns=flow_label_cols,
        minutes_label_col="minutes_label" if "minutes_label" in merged.columns else "minutes",
        overflow_protected_prior_play_prob_floor=float(args.overflow_protected_prior_play_prob_floor),
        overflow_protected_prior_minutes_floor=float(args.overflow_protected_prior_minutes_floor),
        overflow_risk_weight_consecutive_active_dnp=float(args.overflow_risk_weight_consecutive_active_dnp),
        overflow_risk_weight_active_but_dnp_rate_last10=float(args.overflow_risk_weight_active_but_dnp_rate_last10),
        overflow_risk_weight_inactive_streak_len=float(args.overflow_risk_weight_inactive_streak_len),
        overflow_keep_weight_prior_play_prob=float(args.overflow_keep_weight_prior_play_prob),
        overflow_keep_weight_prior_minutes=float(args.overflow_keep_weight_prior_minutes),
    )

    train_loader = DataLoader(
        GameLevelDataset(train_examples),
        batch_size=max(1, int(args.batch_size)),
        shuffle=True,
        num_workers=max(0, int(args.num_workers)),
        collate_fn=collate_game_level_examples,
    )
    val_loader = DataLoader(
        GameLevelDataset(val_examples),
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        num_workers=max(0, int(args.num_workers)),
        collate_fn=collate_game_level_examples,
    )

    include_pf_in_flow_targets = False
    if bool(args.enable_phase2_flow):
        flow_label_cols = list(
            FLOW_TARGET_COLUMNS_WITH_PF if include_pf_in_flow_targets else FLOW_TARGET_COLUMNS_V1
        )

    config = GameTransformerV2Config(
        feature_columns=feature_cols,
        feature_mean=feature_mean.astype(np.float64).tolist(),
        feature_std=feature_std.astype(np.float64).tolist(),
        game_feature_columns=game_feature_cols,
        team_feature_columns=team_feature_cols,
        d_model=int(args.d_model),
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
        num_heads=int(args.num_heads),
        dropout=float(args.dropout),
        min_active_count=int(args.min_active_count),
        max_active_count=int(args.max_active_count),
        active_threshold_minutes=float(args.active_threshold),
        include_pf_in_flow_targets=bool(include_pf_in_flow_targets),
        flow_num_blocks=int(args.flow_num_blocks),
        flow_scale_clip=float(args.flow_scale_clip),
        flow_context_mode=str(args.flow_context_mode),
        enable_efficiency_head=bool(args.enable_efficiency_head),
        efficiency_head_hidden=int(args.efficiency_head_hidden),
        efficiency_ft_prior_mean=float(args.efficiency_ft_prior_mean),
        efficiency_ft_prior_strength=float(args.efficiency_ft_prior_strength),
        efficiency_fg2_prior_mean=float(args.efficiency_fg2_prior_mean),
        efficiency_fg2_prior_strength=float(args.efficiency_fg2_prior_strength),
        efficiency_fg3_prior_mean=float(args.efficiency_fg3_prior_mean),
        efficiency_fg3_prior_strength=float(args.efficiency_fg3_prior_strength),
        enable_usage_share_head=bool(args.enable_usage_share_head),
        usage_share_head_hidden=int(args.usage_share_head_hidden),
        enable_possession_backbone=bool(args.enable_possession_backbone),
        enable_three_pa_share=bool(args.enable_three_pa_share),
        possession_mu_mode=str(args.possession_mu_mode),
        possession_mu_baseline=float(args.possession_mu_baseline),
        overflow_protected_prior_play_prob_floor=float(args.overflow_protected_prior_play_prob_floor),
        overflow_protected_prior_minutes_floor=float(args.overflow_protected_prior_minutes_floor),
        overflow_risk_weight_consecutive_active_dnp=float(args.overflow_risk_weight_consecutive_active_dnp),
        overflow_risk_weight_active_but_dnp_rate_last10=float(args.overflow_risk_weight_active_but_dnp_rate_last10),
        overflow_risk_weight_inactive_streak_len=float(args.overflow_risk_weight_inactive_streak_len),
        overflow_keep_weight_prior_play_prob=float(args.overflow_keep_weight_prior_play_prob),
        overflow_keep_weight_prior_minutes=float(args.overflow_keep_weight_prior_minutes),
    )

    device = torch.device(str(args.device))
    model = build_game_transformer_v2(config).to(device=device)
    init_model_pt = Path(args.init_model_pt).expanduser().resolve() if args.init_model_pt else None
    if init_model_pt is not None:
        if not init_model_pt.exists():
            raise FileNotFoundError(f"init checkpoint not found: {init_model_pt}")
        init_state = torch.load(init_model_pt, map_location=device)
        # Filter out keys with shape mismatches (e.g. backbone heads expanded
        # by num_game_features via Approach A).  Those params keep their random
        # init and are reported below.
        model_state = model.state_dict()
        shape_mismatched: list[str] = []
        for key in list(init_state.keys()):
            if key in model_state and init_state[key].shape != model_state[key].shape:
                shape_mismatched.append(key)
                del init_state[key]
        if shape_mismatched:
            print(
                f"[warm-start] shape-mismatched keys skipped ({len(shape_mismatched)}): {shape_mismatched}",
                flush=True,
            )
        missing, unexpected = model.load_state_dict(init_state, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected keys in init checkpoint {init_model_pt}: {unexpected}")
        if missing:
            print(
                f"[warm-start] missing keys from init checkpoint ({len(missing)}): {missing}",
                flush=True,
            )
        print(f"[warm-start] loaded init checkpoint: {init_model_pt}", flush=True)
    if bool(args.efficiency_head_only):
        n_trainable = 0
        n_total = 0
        for name, param in model.named_parameters():
            n_total += 1
            is_eff = name.startswith("efficiency_head.")
            param.requires_grad = bool(is_eff)
            if param.requires_grad:
                n_trainable += 1
        print(
            f"[efficiency-head-only] trainable_param_tensors={n_trainable} total_param_tensors={n_total}",
            flush=True,
        )
    if bool(args.usage_share_head_only):
        n_trainable = 0
        n_total = 0
        for name, param in model.named_parameters():
            n_total += 1
            is_usage = name.startswith("usage_share_head.")
            param.requires_grad = bool(is_usage)
            if param.requires_grad:
                n_trainable += 1
        print(
            f"[usage-share-head-only] trainable_param_tensors={n_trainable} total_param_tensors={n_total}",
            flush=True,
        )
    optimizer_groups: dict[str, list[nn.Parameter]] = {
        "encoder": [],
        "backbone_heads": [],
        "base": [],
    }
    optimizer_group_tensor_counts = {key: 0 for key in optimizer_groups}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if _is_encoder_param_name(name):
            group_name = "encoder"
        elif _is_backbone_head_param_name(name):
            group_name = "backbone_heads"
        else:
            group_name = "base"
        optimizer_groups[group_name].append(param)
        optimizer_group_tensor_counts[group_name] += 1

    optimizer_param_groups: list[dict[str, Any]] = []
    base_lr = float(args.lr)
    if optimizer_groups["encoder"]:
        optimizer_param_groups.append(
            {
                "params": optimizer_groups["encoder"],
                "lr": base_lr * float(args.encoder_lr_scale),
                "weight_decay": float(args.weight_decay),
            }
        )
    if optimizer_groups["backbone_heads"]:
        optimizer_param_groups.append(
            {
                "params": optimizer_groups["backbone_heads"],
                "lr": base_lr * float(args.backbone_head_lr_scale),
                "weight_decay": float(args.weight_decay),
            }
        )
    if optimizer_groups["base"]:
        optimizer_param_groups.append(
            {
                "params": optimizer_groups["base"],
                "lr": base_lr,
                "weight_decay": float(args.weight_decay),
            }
        )
    optimizer = torch.optim.AdamW(optimizer_param_groups)
    print(
        (
            "[optimizer] "
            f"base_lr={base_lr:.6g} "
            f"encoder_tensors={optimizer_group_tensor_counts['encoder']} "
            f"encoder_lr={base_lr * float(args.encoder_lr_scale):.6g} "
            f"backbone_head_tensors={optimizer_group_tensor_counts['backbone_heads']} "
            f"backbone_head_lr={base_lr * float(args.backbone_head_lr_scale):.6g} "
            f"base_tensors={optimizer_group_tensor_counts['base']}"
        ),
        flush=True,
    )

    out_dir = Path(args.out_dir).expanduser() if args.out_dir else (
        paths.get_data_root() / "training" / "runs" / f"game_transformer_v2_{_utc_now_compact()}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    history: list[EpochMetrics] = []
    best_val = float("inf")
    best_epoch = -1
    rollback_triggered = False
    stable_checkpoint_path = out_dir / "checkpoint_stable.pt"
    torch.save(model.state_dict(), stable_checkpoint_path)
    stable_checkpoint_epoch = 0
    early_stop_cfg = EarlyStopConfig(
        patience=int(args.early_stop_patience),
        min_delta=float(args.early_stop_min_delta),
        min_epochs=int(args.early_stop_min_epochs),
        min_coupled_epochs=int(args.early_stop_min_coupled_epochs),
    )
    early_stop_state = EarlyStopState()

    phase2_guard_cfg = Phase2StabilityConfig(
        nll_explosion_ratio=float(args.phase2_nll_guard_ratio),
        nll_explosion_abs=float(args.phase2_nll_guard_abs),
        nll_ema_alpha=float(args.phase2_nll_guard_ema_alpha),
        nll_backoff_consecutive_batches=int(args.phase2_nll_guard_consecutive_batches),
        max_backoffs_before_rollback=int(args.phase2_max_backoffs_before_rollback),
        min_a2_scale=float(args.phase2_min_a2_scale),
    )
    phase2_guard_state = Phase2StabilityState(a2_scale=1.0)

    for epoch in range(1, int(args.epochs) + 1):
        phase2_weights = _resolve_phase2_epoch_weights(
            epoch=int(epoch),
            enable_phase2_flow=bool(args.enable_phase2_flow),
            enable_phase3_decision=bool(args.enable_phase3_decision),
            w_minutes=float(args.w_minutes),
            w_minutes_nll=float(args.w_minutes_nll),
            w_count=float(args.w_count),
            w_member=float(args.w_member),
            w_flow_nll=float(args.w_flow_nll),
            w_crps_fpts=float(args.w_crps_fpts),
            w_team_energy=float(args.w_team_energy),
            flow_warmup_epochs=int(args.phase2_flow_warmup_epochs),
            flow_delay_epochs=int(args.phase2_flow_delay_epochs),
            anchor_start_weight=float(args.phase2_anchor_start_weight),
            anchor_end_weight=float(args.phase2_anchor_end_weight),
            a2_scale=float(phase2_guard_state.a2_scale),
        )
        backbone_weights = _resolve_backbone_epoch_weights(
            epoch=int(epoch),
            enable_possession_backbone=bool(args.enable_possession_backbone),
            enable_three_pa_share=bool(args.enable_three_pa_share),
            w_poss_nll=float(args.w_poss_nll),
            w_backbone_nll=float(args.w_backbone_nll),
            w_three_pa_nll=float(args.w_three_pa_nll),
            w_poss_regression=float(args.w_poss_regression),
            loss_ramp_epochs=int(args.backbone_loss_ramp_epochs),
            poss_loss_start_scale=float(args.poss_loss_start_scale),
            backbone_loss_start_scale=float(args.backbone_loss_start_scale),
            three_pa_loss_start_scale=float(args.three_pa_loss_start_scale),
            poss_regression_start_scale=float(args.poss_regression_start_scale),
        )
        coupled_epochs = _count_backbone_coupled_epochs(
            epoch=int(epoch),
            enable_possession_backbone=bool(args.enable_possession_backbone),
            backbone_detach_until_epoch=int(args.backbone_detach_until_epoch),
        )

        train_stats = _run_epoch(
            model,
            train_loader,
            device=device,
            optimizer=optimizer,
            active_threshold=float(args.active_threshold),
            min_active_count=int(args.min_active_count),
            max_active_count=int(args.max_active_count),
            run_phase2_flow=bool(phase2_weights.run_phase2_flow),
            run_phase3_decision=bool(phase2_weights.run_phase3_decision),
            w_minutes=float(phase2_weights.w_minutes),
            w_minutes_nll=float(phase2_weights.w_minutes_nll),
            w_count=float(phase2_weights.w_count),
            w_member=float(phase2_weights.w_member),
            w_flow_nll=float(phase2_weights.w_flow_nll),
            w_crps_fpts=float(phase2_weights.w_crps_fpts),
            w_team_energy=float(phase2_weights.w_team_energy),
            minutes_nll_sigma=float(args.minutes_nll_sigma),
            phase3_num_samples=int(args.phase3_num_samples),
            phase3_active_temperature=float(args.phase3_active_temperature),
            phase3_stop_grad=bool(args.phase3_stop_grad),
            positive_weight=float(args.active_positive_weight),
            epoch_index=int(epoch),
            backbone_grad_clip_norm=float(args.backbone_grad_clip_norm),
            flow_grad_clip_norm=float(args.flow_grad_clip_norm),
            encoder_grad_clip_norm=float(args.encoder_grad_clip_norm),
            backbone_head_grad_clip_norm=float(args.backbone_head_grad_clip_norm),
            w_poss_nll=float(backbone_weights.w_poss_nll),
            w_backbone_nll=float(backbone_weights.w_backbone_nll),
            w_three_pa_nll=float(backbone_weights.w_three_pa_nll),
            w_efficiency_nll=float(args.w_efficiency_nll) if bool(args.enable_efficiency_head) else 0.0,
            w_usage_share_nll=float(args.w_usage_share_nll) if bool(args.enable_usage_share_head) else 0.0,
            w_emergent_share_aux=float(args.w_emergent_share_aux),
            w_ast_share_aux=float(args.w_ast_share_aux),
            w_reb_share_aux=float(args.w_reb_share_aux),
            w_ast_team_rate_aux=float(args.w_ast_team_rate_aux),
            w_reb_opportunity_rate_aux=float(args.w_reb_opportunity_rate_aux),
            w_spread_aux=float(args.w_spread_aux),
            w_total_aux=float(args.w_total_aux),
            spread_total_aux_ramp_epochs=int(args.spread_total_aux_ramp_epochs),
            spread_total_aux_start_scale=float(args.spread_total_aux_start_scale),
            spread_aux_target_scale=float(args.spread_aux_target_scale),
            total_aux_target_scale=float(args.total_aux_target_scale),
            spread_aux_huber_delta=float(args.spread_aux_huber_delta),
            total_aux_huber_delta=float(args.total_aux_huber_delta),
            w_efficiency_mean_aux=float(args.w_efficiency_mean_aux),
            vegas_total_idx=int(vegas_total_idx),
            vegas_spread_idx=int(vegas_spread_idx),
            vegas_total_missing_idx=int(vegas_total_missing_idx),
            vegas_spread_missing_idx=int(vegas_spread_missing_idx),
            enable_possession_backbone=bool(args.enable_possession_backbone),
            enable_efficiency_head=bool(args.enable_efficiency_head),
            enable_usage_share_head=bool(args.enable_usage_share_head),
            detach_backbone=bool(int(epoch) < int(args.backbone_detach_until_epoch)),
            phase2_stability_config=phase2_guard_cfg if bool(args.enable_phase2_flow) else None,
            phase2_stability_state=phase2_guard_state if bool(args.enable_phase2_flow) else None,
            w_poss_regression=float(backbone_weights.w_poss_regression),
            estimated_possessions_idx=int(estimated_possessions_idx),
        )
        rollback_requested = bool(train_stats.get("rollback_requested", 0.0) > 0.0)
        if rollback_requested:
            rollback_triggered = True
            if stable_checkpoint_path.exists():
                state = torch.load(stable_checkpoint_path, map_location=device)
                model.load_state_dict(state)
                torch.save(model.state_dict(), out_dir / "model_rollback.pt")
            val_stats = {
                "total": float("nan"),
                "minutes_mae": float("nan"),
                "count_loss": float("nan"),
                "member_loss": float("nan"),
                "minutes_nll": float("nan"),
                "flow_nll": float("nan"),
                "crps_fpts": float("nan"),
                "team_energy": float("nan"),
                "count_acc": float("nan"),
                "poss_nll": float("nan"),
                "backbone_nll": float("nan"),
                "three_pa_nll": float("nan"),
                "efficiency_nll": float("nan"),
                "usage_share_nll": float("nan"),
                "emergent_share_aux": float("nan"),
                "ast_share_aux": float("nan"),
                "reb_share_aux": float("nan"),
                "ast_team_rate_aux": float("nan"),
                "reb_opportunity_rate_aux": float("nan"),
                "poss_regression": float("nan"),
                "skipped_batches": 0.0,
                "instability_events": 0.0,
                "rollback_requested": 1.0,
            }
        else:
            val_stats = _run_epoch(
                model,
                val_loader,
                device=device,
                optimizer=None,
                active_threshold=float(args.active_threshold),
                min_active_count=int(args.min_active_count),
                max_active_count=int(args.max_active_count),
                run_phase2_flow=bool(phase2_weights.run_phase2_flow),
                run_phase3_decision=bool(phase2_weights.run_phase3_decision),
                w_minutes=float(phase2_weights.w_minutes),
                w_minutes_nll=float(phase2_weights.w_minutes_nll),
                w_count=float(phase2_weights.w_count),
                w_member=float(phase2_weights.w_member),
                w_flow_nll=float(phase2_weights.w_flow_nll),
                w_crps_fpts=float(phase2_weights.w_crps_fpts),
                w_team_energy=float(phase2_weights.w_team_energy),
                minutes_nll_sigma=float(args.minutes_nll_sigma),
                phase3_num_samples=int(args.phase3_num_samples),
                phase3_active_temperature=float(args.phase3_active_temperature),
                phase3_stop_grad=bool(args.phase3_stop_grad),
                positive_weight=float(args.active_positive_weight),
                epoch_index=int(epoch),
                backbone_grad_clip_norm=float(args.backbone_grad_clip_norm),
                flow_grad_clip_norm=float(args.flow_grad_clip_norm),
                encoder_grad_clip_norm=float(args.encoder_grad_clip_norm),
                backbone_head_grad_clip_norm=float(args.backbone_head_grad_clip_norm),
                w_poss_nll=float(backbone_weights.w_poss_nll),
                w_backbone_nll=float(backbone_weights.w_backbone_nll),
                w_three_pa_nll=float(backbone_weights.w_three_pa_nll),
                w_efficiency_nll=float(args.w_efficiency_nll) if bool(args.enable_efficiency_head) else 0.0,
                w_usage_share_nll=float(args.w_usage_share_nll) if bool(args.enable_usage_share_head) else 0.0,
                w_emergent_share_aux=float(args.w_emergent_share_aux),
                w_ast_share_aux=float(args.w_ast_share_aux),
                w_reb_share_aux=float(args.w_reb_share_aux),
                w_ast_team_rate_aux=float(args.w_ast_team_rate_aux),
                w_reb_opportunity_rate_aux=float(args.w_reb_opportunity_rate_aux),
                w_spread_aux=float(args.w_spread_aux),
                w_total_aux=float(args.w_total_aux),
                spread_total_aux_ramp_epochs=int(args.spread_total_aux_ramp_epochs),
                spread_total_aux_start_scale=float(args.spread_total_aux_start_scale),
                spread_aux_target_scale=float(args.spread_aux_target_scale),
                total_aux_target_scale=float(args.total_aux_target_scale),
                spread_aux_huber_delta=float(args.spread_aux_huber_delta),
                total_aux_huber_delta=float(args.total_aux_huber_delta),
                w_efficiency_mean_aux=float(args.w_efficiency_mean_aux),
                vegas_total_idx=int(vegas_total_idx),
                vegas_spread_idx=int(vegas_spread_idx),
                vegas_total_missing_idx=int(vegas_total_missing_idx),
                vegas_spread_missing_idx=int(vegas_spread_missing_idx),
                enable_possession_backbone=bool(args.enable_possession_backbone),
                enable_efficiency_head=bool(args.enable_efficiency_head),
                enable_usage_share_head=bool(args.enable_usage_share_head),
                detach_backbone=bool(int(epoch) < int(args.backbone_detach_until_epoch)),
                w_poss_regression=float(backbone_weights.w_poss_regression),
                estimated_possessions_idx=int(estimated_possessions_idx),
            )

        val_total_ex_possreg = _resolve_early_stop_metric_value(
            metric_name="val_total_ex_possreg",
            val_total=float(val_stats["total"]),
            val_poss_regression=float(val_stats.get("poss_regression", 0.0)),
            w_poss_regression=float(backbone_weights.w_poss_regression),
        )
        metrics = EpochMetrics(
            epoch=epoch,
            phase2_flow_warmup=float(phase2_weights.flow_warmup),
            phase2_anchor_weight=float(phase2_weights.anchor_weight),
            phase2_a2_scale=float(phase2_guard_state.a2_scale),
            phase2_backoff_count=int(phase2_guard_state.backoff_count),
            train_skipped_batches=int(train_stats.get("skipped_batches", 0.0)),
            train_instability_events=int(train_stats.get("instability_events", 0.0)),
            train_total=train_stats["total"],
            train_minutes_mae=train_stats["minutes_mae"],
            train_count_loss=train_stats["count_loss"],
            train_member_loss=train_stats["member_loss"],
            train_minutes_nll=train_stats["minutes_nll"],
            train_flow_nll=train_stats["flow_nll"],
            train_crps_fpts=train_stats["crps_fpts"],
            train_team_energy=train_stats["team_energy"],
            train_count_acc=train_stats["count_acc"],
            train_poss_nll=train_stats.get("poss_nll", 0.0),
            train_backbone_nll=train_stats.get("backbone_nll", 0.0),
            train_three_pa_nll=train_stats.get("three_pa_nll", 0.0),
            train_efficiency_nll=train_stats.get("efficiency_nll", 0.0),
            train_usage_share_nll=train_stats.get("usage_share_nll", 0.0),
            train_emergent_share_aux=train_stats.get("emergent_share_aux", 0.0),
            train_ast_share_aux=train_stats.get("ast_share_aux", 0.0),
            train_reb_share_aux=train_stats.get("reb_share_aux", 0.0),
            train_ast_team_rate_aux=train_stats.get("ast_team_rate_aux", 0.0),
            train_reb_opportunity_rate_aux=train_stats.get("reb_opportunity_rate_aux", 0.0),
            train_spread_aux=train_stats.get("spread_aux", 0.0),
            train_total_aux=train_stats.get("total_aux", 0.0),
            train_efficiency_mean_aux=train_stats.get("efficiency_mean_aux", 0.0),
            train_poss_regression=train_stats.get("poss_regression", 0.0),
            val_total=val_stats["total"],
            val_minutes_mae=val_stats["minutes_mae"],
            val_count_loss=val_stats["count_loss"],
            val_member_loss=val_stats["member_loss"],
            val_minutes_nll=val_stats["minutes_nll"],
            val_flow_nll=val_stats["flow_nll"],
            val_crps_fpts=val_stats["crps_fpts"],
            val_team_energy=val_stats["team_energy"],
            val_count_acc=val_stats["count_acc"],
            val_poss_nll=val_stats.get("poss_nll", 0.0),
            val_backbone_nll=val_stats.get("backbone_nll", 0.0),
            val_three_pa_nll=val_stats.get("three_pa_nll", 0.0),
            val_efficiency_nll=val_stats.get("efficiency_nll", 0.0),
            val_usage_share_nll=val_stats.get("usage_share_nll", 0.0),
            val_emergent_share_aux=val_stats.get("emergent_share_aux", 0.0),
            val_ast_share_aux=val_stats.get("ast_share_aux", 0.0),
            val_reb_share_aux=val_stats.get("reb_share_aux", 0.0),
            val_ast_team_rate_aux=val_stats.get("ast_team_rate_aux", 0.0),
            val_reb_opportunity_rate_aux=val_stats.get("reb_opportunity_rate_aux", 0.0),
            val_spread_aux=val_stats.get("spread_aux", 0.0),
            val_total_aux=val_stats.get("total_aux", 0.0),
            val_efficiency_mean_aux=val_stats.get("efficiency_mean_aux", 0.0),
            val_total_ex_possreg=float(val_total_ex_possreg),
            val_poss_regression=val_stats.get("poss_regression", 0.0),
        )
        history.append(metrics)

        msg = (
            f"epoch={epoch:03d} "
            f"train_total={metrics.train_total:.4f} val_total={metrics.val_total:.4f} "
            f"val_minutes_mae={metrics.val_minutes_mae:.4f} val_count_acc={metrics.val_count_acc:.4f} "
            f"phase2_warmup={metrics.phase2_flow_warmup:.3f} "
            f"anchor={metrics.phase2_anchor_weight:.3f} a2={metrics.phase2_a2_scale:.3f}"
        )
        if bool(args.enable_phase2_flow):
            msg = (
                f"{msg} "
                f"val_minutes_nll={metrics.val_minutes_nll:.4f} "
                f"val_flow_nll={metrics.val_flow_nll:.4f} "
                f"skipped_batches={metrics.train_skipped_batches} "
                f"instability_events={metrics.train_instability_events}"
            )
        if bool(args.enable_phase3_decision):
            msg = (
                f"{msg} "
                f"val_crps_fpts={metrics.val_crps_fpts:.4f} "
                f"val_team_energy={metrics.val_team_energy:.4f}"
            )
        if bool(args.enable_efficiency_head):
            msg = f"{msg} val_efficiency_nll={metrics.val_efficiency_nll:.4f}"
        if bool(args.enable_usage_share_head):
            msg = f"{msg} val_usage_share_nll={metrics.val_usage_share_nll:.4f}"
        if float(args.w_emergent_share_aux) > 0.0:
            msg = f"{msg} val_emergent_share_aux={metrics.val_emergent_share_aux:.4f}"
        if float(args.w_ast_share_aux) > 0.0:
            msg = f"{msg} val_ast_share_aux={metrics.val_ast_share_aux:.4f}"
        if float(args.w_reb_share_aux) > 0.0:
            msg = f"{msg} val_reb_share_aux={metrics.val_reb_share_aux:.4f}"
        if float(args.w_ast_team_rate_aux) > 0.0:
            msg = f"{msg} val_ast_team_rate_aux={metrics.val_ast_team_rate_aux:.4f}"
        if float(args.w_reb_opportunity_rate_aux) > 0.0:
            msg = f"{msg} val_reb_opp_rate_aux={metrics.val_reb_opportunity_rate_aux:.4f}"
        if float(args.w_spread_aux) > 0.0:
            msg = f"{msg} val_spread_aux={metrics.val_spread_aux:.4f}"
        if float(args.w_total_aux) > 0.0:
            msg = f"{msg} val_total_aux={metrics.val_total_aux:.4f}"
        if float(args.w_efficiency_mean_aux) > 0.0:
            msg = f"{msg} val_eff_mean_aux={metrics.val_efficiency_mean_aux:.4f}"
        if bool(args.enable_possession_backbone):
            bb_detached = int(epoch) < int(args.backbone_detach_until_epoch)
            msg = (
                f"{msg} "
                f"val_poss_nll={metrics.val_poss_nll:.4f} "
                f"val_backbone_nll={metrics.val_backbone_nll:.4f} "
                f"bb_detach={'Y' if bb_detached else 'N'} "
                f"bb_coupled_epochs={coupled_epochs} "
                f"w_poss={backbone_weights.w_poss_nll:.4f} "
                f"w_bb={backbone_weights.w_backbone_nll:.4f}"
            )
            if bool(args.enable_three_pa_share):
                msg = (
                    f"{msg} "
                    f"val_three_pa_nll={metrics.val_three_pa_nll:.4f} "
                    f"w_3pa={backbone_weights.w_three_pa_nll:.4f}"
                )
            if float(args.w_poss_regression) > 0.0:
                msg = (
                    f"{msg} "
                    f"val_poss_reg={metrics.val_poss_regression:.4f} "
                    f"w_poss_reg={backbone_weights.w_poss_regression:.4f} "
                    f"val_total_ex_possreg={metrics.val_total_ex_possreg:.4f}"
                )
        print(msg, flush=True)

        # Don't update best_val during flow delay: val_total is not comparable
        # across the delay boundary because it excludes flow_nll during delay.
        in_flow_delay = (
            bool(args.enable_phase2_flow)
            and int(args.phase2_flow_delay_epochs) > 0
            and int(epoch) <= int(args.phase2_flow_delay_epochs)
        )
        if (
            not in_flow_delay
            and math.isfinite(float(metrics.val_total))
            and metrics.val_total < best_val
        ):
            best_val = metrics.val_total
            best_epoch = epoch
            torch.save(model.state_dict(), out_dir / "model.pt")

        if rollback_requested:
            print(
                (
                    "[phase2][rollback] "
                    f"stopped_at_epoch={epoch:03d} "
                    f"reason={phase2_guard_state.rollback_reason} "
                    f"stable_checkpoint_epoch={stable_checkpoint_epoch:03d}"
                ),
                flush=True,
            )
            break

        if (
            int(early_stop_cfg.patience) > 0
            and not in_flow_delay
            and int(train_stats.get("instability_events", 0.0)) <= 0
        ):
            early_stop_metric_value = _resolve_early_stop_metric_value(
                metric_name=str(args.early_stop_metric),
                val_total=float(metrics.val_total),
                val_poss_regression=float(metrics.val_poss_regression),
                w_poss_regression=float(backbone_weights.w_poss_regression),
            )
            should_stop = _update_early_stop(
                epoch=int(epoch),
                metric_value=float(early_stop_metric_value),
                coupled_epochs=int(coupled_epochs),
                config=early_stop_cfg,
                state=early_stop_state,
                metric_name=str(args.early_stop_metric),
            )
            if should_stop:
                print(
                    (
                        "[early-stop] "
                        f"stopped_at_epoch={epoch:03d} "
                        f"best_epoch={early_stop_state.best_epoch:03d} "
                        f"best_{str(args.early_stop_metric)}={early_stop_state.best_metric:.6f} "
                        f"reason={early_stop_state.stop_reason}"
                    ),
                    flush=True,
                )
                break

        if (
            bool(args.enable_phase2_flow)
            and int(train_stats.get("instability_events", 0.0)) > 0
        ):
            continue
        if _stats_finite(train_stats) and _stats_finite(val_stats):
            torch.save(model.state_dict(), stable_checkpoint_path)
            stable_checkpoint_epoch = int(epoch)

    if not (out_dir / "model.pt").exists():
        torch.save(model.state_dict(), out_dir / "model.pt")

    config.save(out_dir / "config.json")
    (out_dir / "history.json").write_text(
        json.dumps([asdict(m) for m in history], indent=2, sort_keys=True),
        encoding="utf-8",
    )

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(dataset_dir),
        "features_path": str(features_path),
        "labels_minutes_path": str(labels_minutes_path),
        "labels_boxscore_counts_path": str(labels_boxscore_counts_path) if bool(args.enable_phase2_flow) else None,
        "num_feature_columns": int(len(feature_cols)),
        "flow_target_columns": flow_label_cols,
        "phase2_flow_enabled": bool(args.enable_phase2_flow),
        "efficiency_head_enabled": bool(args.enable_efficiency_head),
        "efficiency_head_only": bool(args.efficiency_head_only),
        "usage_share_head_enabled": bool(args.enable_usage_share_head),
        "usage_share_head_only": bool(args.usage_share_head_only),
        "ast_reb_structure_aux": {
            "w_ast_share_aux": float(args.w_ast_share_aux),
            "w_reb_share_aux": float(args.w_reb_share_aux),
            "w_ast_team_rate_aux": float(args.w_ast_team_rate_aux),
            "w_reb_opportunity_rate_aux": float(args.w_reb_opportunity_rate_aux),
        },
        "spread_total_aux": {
            "w_spread_aux": float(args.w_spread_aux),
            "w_total_aux": float(args.w_total_aux),
            "ramp_epochs": int(args.spread_total_aux_ramp_epochs),
            "start_scale": float(args.spread_total_aux_start_scale),
            "spread_target_scale": float(args.spread_aux_target_scale),
            "total_target_scale": float(args.total_aux_target_scale),
            "spread_huber_delta": float(args.spread_aux_huber_delta),
            "total_huber_delta": float(args.total_aux_huber_delta),
        },
        "efficiency_mean_aux": {
            "w_efficiency_mean_aux": float(args.w_efficiency_mean_aux),
        },
        "phase3_decision_enabled": bool(args.enable_phase3_decision),
        "phase2_schedule": {
            "flow_warmup_epochs": int(args.phase2_flow_warmup_epochs),
            "anchor_start_weight": float(args.phase2_anchor_start_weight),
            "anchor_end_weight": float(args.phase2_anchor_end_weight),
        },
        "phase3_decision": {
            "w_crps_fpts": float(args.w_crps_fpts),
            "w_team_energy": float(args.w_team_energy),
            "num_samples": int(args.phase3_num_samples),
            "active_temperature": float(args.phase3_active_temperature),
            "stop_grad": bool(args.phase3_stop_grad),
        },
        "phase2_stability": {
            "rollback_triggered": bool(rollback_triggered),
            "rollback_reason": str(phase2_guard_state.rollback_reason) if rollback_triggered else None,
            "stable_checkpoint_epoch": int(stable_checkpoint_epoch),
            "final_a2_scale": float(phase2_guard_state.a2_scale),
            "backoff_count": int(phase2_guard_state.backoff_count),
            "events": phase2_guard_state.events[-50:],
            "config": asdict(phase2_guard_cfg),
        },
        "backbone_stabilization": {
            "detach_until_epoch": int(args.backbone_detach_until_epoch),
            "loss_ramp_epochs": int(args.backbone_loss_ramp_epochs),
            "poss_loss_start_scale": float(args.poss_loss_start_scale),
            "backbone_loss_start_scale": float(args.backbone_loss_start_scale),
            "three_pa_loss_start_scale": float(args.three_pa_loss_start_scale),
            "encoder_lr_scale": float(args.encoder_lr_scale),
            "backbone_head_lr_scale": float(args.backbone_head_lr_scale),
            "backbone_grad_clip_norm": float(args.backbone_grad_clip_norm),
            "encoder_grad_clip_norm": float(args.encoder_grad_clip_norm),
            "backbone_head_grad_clip_norm": float(args.backbone_head_grad_clip_norm),
            "w_poss_regression": float(args.w_poss_regression),
            "poss_regression_start_scale": float(args.poss_regression_start_scale),
        },
        "early_stopping": {
            "enabled": bool(int(early_stop_cfg.patience) > 0),
            "metric": str(args.early_stop_metric),
            "config": asdict(early_stop_cfg),
            "best_epoch": int(early_stop_state.best_epoch),
            "best_metric": float(early_stop_state.best_metric),
            "bad_epochs": int(early_stop_state.bad_epochs),
            "stopped_early": bool(early_stop_state.stop_requested),
            "stop_epoch": int(early_stop_state.stop_epoch) if early_stop_state.stop_epoch is not None else None,
            "stop_reason": str(early_stop_state.stop_reason) if early_stop_state.stop_reason is not None else None,
        },
        "split": split_meta,
        "train_examples": int(len(train_examples)),
        "val_examples": int(len(val_examples)),
        "best_val_total": float(best_val),
        "best_epoch": int(best_epoch),
        "init_model_pt": str(init_model_pt) if init_model_pt is not None else None,
        "args": vars(args),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps({"out_dir": str(out_dir), "best_epoch": best_epoch, "best_val_total": best_val}, indent=2), flush=True)


if __name__ == "__main__":
    main()
