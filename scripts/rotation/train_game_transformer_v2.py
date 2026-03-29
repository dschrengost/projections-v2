#!/usr/bin/env python3
"""Train Phase 1b GameTransformerV2 (joint active-set + joint minutes)."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from projections import paths
from projections.rotation.game_transformer_v2 import (
    FLOW_TARGET_COLUMNS_V1,
    FLOW_TARGET_COLUMNS_WITH_PF,
    FLOW_TARGET_SCHEMA_DEFAULT,
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
    flow_target_columns,
    normalize_flow_target_schema,
    reconstruct_flow_to_contract,
    select_flow_columns,
)
from projections.rotation.joint_active_set import (
    build_active_set_labels,
    compute_active_set_losses,
)
from projections.rotation.joint_minutes import build_minutes_role_targets
from projections.rotation.possession_backbone import (
    compute_possession_truth,
    compute_possession_truth_per_team,
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
DEFAULT_BACKBONE_ENV_FEATURE_COLS = [
    "is_b2b",
    "team_pace_szn",
    "team_off_rtg_szn",
    "team_def_rtg_szn",
    "opp_pace_szn",
    "opp_def_rtg_szn",
]
DEFAULT_ASSIST_SHARE_CONDITION_FEATURE_COLS = [
    "an_ast_line",
    "an_implied_minutes",
    "prior_play_prob",
    "started_proxy_rate_prior_20",
]
DEFAULT_BACKBONE_ENV_ADAPTER_HIDDEN = 32


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    phase2_flow_warmup: float
    phase2_anchor_weight: float
    phase2_a2_scale: float
    minutes_teacher_forcing_prob: float
    flow_minutes_teacher_forcing_prob: float
    phase2_backoff_count: int
    train_skipped_batches: int
    train_instability_events: int
    train_total: float
    train_minutes_mae: float
    train_count_loss: float
    train_member_loss: float
    train_minutes_nll: float
    train_minutes_hurdle_nll: float
    train_role_loss: float
    train_role_acc: float
    train_sparse_starter_underpred_loss: float
    train_bench_riser_underpred_loss: float
    train_starter_promotion_loss: float
    train_flow_nll: float
    train_crps_fpts: float
    train_team_energy: float
    train_count_acc: float
    train_poss_nll: float = 0.0
    train_backbone_nll: float = 0.0
    train_three_pa_nll: float = 0.0
    train_efficiency_nll: float = 0.0
    train_team_efficiency_ppp_aux: float = 0.0
    train_team_ppp_aux: float = 0.0
    train_team_advantage_aux: float = 0.0
    train_usage_share_nll: float = 0.0
    train_team_possession_aux: float = 0.0
    train_team_points_budget_aux: float = 0.0
    train_team_ast_budget_aux: float = 0.0
    train_assist_share_aux: float = 0.0
    train_assist_share_recon_aux: float = 0.0
    train_ast_blend_gate_aux: float = 0.0
    train_emergent_share_aux: float = 0.0
    train_ast_share_aux: float = 0.0
    train_reb_share_aux: float = 0.0
    train_ast_team_rate_aux: float = 0.0
    train_reb_opportunity_rate_aux: float = 0.0
    train_team_rebound_budget_rate_aux: float = 0.0
    train_rebound_budget_blend_gate_aux: float = 0.0
    train_spread_aux: float = 0.0
    train_total_aux: float = 0.0
    train_props_pts_aux: float = 0.0
    train_props_reb_aux: float = 0.0
    train_props_ast_aux: float = 0.0
    train_direct_pts_aux: float = 0.0
    train_direct_reb_aux: float = 0.0
    train_direct_ast_aux: float = 0.0
    train_direct_stl_aux: float = 0.0
    train_direct_blk_aux: float = 0.0
    train_direct_tov_aux: float = 0.0
    train_direct_boxscore_aux: float = 0.0
    train_direct_opportunity_aux: float = 0.0
    train_flow_anchor_nonast_aux: float = 0.0
    train_efficiency_mean_aux: float = 0.0
    val_total: float = 0.0
    val_minutes_mae: float = 0.0
    val_count_loss: float = 0.0
    val_member_loss: float = 0.0
    val_minutes_nll: float = 0.0
    val_minutes_hurdle_nll: float = 0.0
    val_role_loss: float = 0.0
    val_role_acc: float = 0.0
    val_sparse_starter_underpred_loss: float = 0.0
    val_bench_riser_underpred_loss: float = 0.0
    val_starter_promotion_loss: float = 0.0
    val_flow_nll: float = 0.0
    val_crps_fpts: float = 0.0
    val_team_energy: float = 0.0
    val_count_acc: float = 0.0
    val_poss_nll: float = 0.0
    val_backbone_nll: float = 0.0
    val_three_pa_nll: float = 0.0
    val_efficiency_nll: float = 0.0
    val_team_efficiency_ppp_aux: float = 0.0
    val_team_ppp_aux: float = 0.0
    val_team_advantage_aux: float = 0.0
    val_usage_share_nll: float = 0.0
    val_team_possession_aux: float = 0.0
    val_team_points_budget_aux: float = 0.0
    val_team_ast_budget_aux: float = 0.0
    val_assist_share_aux: float = 0.0
    val_assist_share_recon_aux: float = 0.0
    val_ast_blend_gate_aux: float = 0.0
    val_emergent_share_aux: float = 0.0
    val_ast_share_aux: float = 0.0
    val_reb_share_aux: float = 0.0
    val_ast_team_rate_aux: float = 0.0
    val_reb_opportunity_rate_aux: float = 0.0
    val_team_rebound_budget_rate_aux: float = 0.0
    val_rebound_budget_blend_gate_aux: float = 0.0
    val_spread_aux: float = 0.0
    val_total_aux: float = 0.0
    val_props_pts_aux: float = 0.0
    val_props_reb_aux: float = 0.0
    val_props_ast_aux: float = 0.0
    val_direct_pts_aux: float = 0.0
    val_direct_reb_aux: float = 0.0
    val_direct_ast_aux: float = 0.0
    val_direct_stl_aux: float = 0.0
    val_direct_blk_aux: float = 0.0
    val_direct_tov_aux: float = 0.0
    val_direct_boxscore_aux: float = 0.0
    val_direct_opportunity_aux: float = 0.0
    val_flow_anchor_nonast_aux: float = 0.0
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
class MinutesCheckpointCandidate:
    epoch: int
    val_minutes_mae: float
    checkpoint_metric_value: float
    val_total: float
    checkpoint_path: str


@dataclass(frozen=True)
class SparseRerankMetrics:
    sparse_next_up_underpred_rate: float
    active_count_mae: float
    starter_sparse_pred_minutes_mean: float
    sparse_score: float


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


def _resolve_training_device(device_arg: str) -> torch.device:
    value = str(device_arg).strip().lower()
    if value in {"", "auto"}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and bool(mps_backend.is_available()):
            return torch.device("mps")
        return torch.device("cpu")
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(f"--device={device_arg!r} requested CUDA, but CUDA is not available")
    if value == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is None or not bool(mps_backend.is_available()):
            raise ValueError("--device='mps' requested, but MPS is not available")
    return torch.device(device_arg)


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


def _load_feature_columns_override(path_value: str) -> list[str]:
    path = Path(path_value).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        cols = payload.get("columns")
        if cols is None:
            cols = payload.get("feature_columns")
    else:
        cols = payload
    if not isinstance(cols, list) or not all(isinstance(col, str) for col in cols):
        raise ValueError(f"Invalid feature columns payload at {path}")
    if not cols:
        raise ValueError(f"Feature columns override at {path} is empty")
    return [str(col) for col in cols]


def _add_efficiency_sidecar_interaction_features(
    df: pd.DataFrame,
    *,
    sidecar_feature_columns: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    if not sidecar_feature_columns:
        return df, []

    out = df.copy()
    derived_cols: list[str] = []
    windows = ("5", "10", "20")
    interaction_specs = [
        ("fg2_pct_prior_{w}", "opp_fg2_pct_allowed_prior_{w}", "fg2_pct_matchup_delta_{w}"),
        ("fg3_pct_prior_{w}", "opp_fg3_pct_allowed_prior_{w}", "fg3_pct_matchup_delta_{w}"),
        ("efg_pct_prior_{w}", "opp_efg_pct_allowed_prior_{w}", "efg_pct_matchup_delta_{w}"),
        ("fta_per_min_prior_{w}", "opp_fta_rate_allowed_prior_{w}", "fta_rate_matchup_delta_{w}"),
        ("three_pa_share_prior_{w}", "opp_three_pa_share_allowed_prior_{w}", "three_pa_share_matchup_delta_{w}"),
    ]
    available = set(out.columns)
    requested = set(sidecar_feature_columns)

    for lhs_tmpl, rhs_tmpl, out_tmpl in interaction_specs:
        for window in windows:
            lhs = lhs_tmpl.format(w=window)
            rhs = rhs_tmpl.format(w=window)
            if lhs not in available or rhs not in available:
                continue
            if lhs not in requested and rhs not in requested:
                continue
            out_col = out_tmpl.format(w=window)
            out[out_col] = pd.to_numeric(out[lhs], errors="coerce").fillna(0.0) - pd.to_numeric(
                out[rhs], errors="coerce"
            ).fillna(0.0)
            derived_cols.append(out_col)

    if {"team_off_rtg_szn", "opp_def_rtg_szn"}.issubset(available):
        out["team_off_vs_opp_def_delta"] = (
            pd.to_numeric(out["team_off_rtg_szn"], errors="coerce").fillna(0.0)
            - pd.to_numeric(out["opp_def_rtg_szn"], errors="coerce").fillna(0.0)
        )
        derived_cols.append("team_off_vs_opp_def_delta")

    return out, derived_cols


def _exclude_feature_patterns(
    feature_cols: list[str],
    *,
    exclude_patterns: list[str],
) -> list[str]:
    if not exclude_patterns:
        return list(feature_cols)
    compiled = [re.compile(str(pattern)) for pattern in exclude_patterns if str(pattern).strip()]
    if not compiled:
        return list(feature_cols)

    kept: list[str] = []
    for col in feature_cols:
        if any(pattern.search(col) for pattern in compiled):
            continue
        kept.append(col)
    if not kept:
        raise ValueError("All inferred feature columns were excluded by feature patterns")
    return kept


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


def _build_lineup_available_example_sampling_weights(
    examples: list[Any],
    *,
    lineup_available_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Build per-example train sampling weights from lineup-available coverage."""
    if not examples:
        return (
            torch.ones((0,), dtype=torch.double),
            {
                "lineup_weight_target": float(max(1.0, float(lineup_available_weight))),
                "lineup_fraction_mean": 0.0,
                "lineup_fraction_min": 0.0,
                "lineup_fraction_max": 0.0,
                "sample_weight_mean": 1.0,
                "sample_weight_min": 1.0,
                "sample_weight_max": 1.0,
            },
        )

    target_weight = float(max(1.0, float(lineup_available_weight)))
    lineup_fraction = np.zeros((len(examples),), dtype=np.float64)

    for idx, ex in enumerate(examples):
        valid = np.asarray(ex.player_valid_mask, dtype=bool)
        if valid.size == 0:
            continue
        lineup = np.asarray(ex.lineup_available, dtype=bool)
        if lineup.shape != valid.shape:
            raise ValueError("lineup_available shape must match player_valid_mask per example")
        valid_count = int(valid.sum())
        if valid_count <= 0:
            continue
        lineup_count = int(np.logical_and(lineup, valid).sum())
        lineup_fraction[idx] = float(lineup_count) / float(valid_count)

    sample_weights = 1.0 + (target_weight - 1.0) * lineup_fraction
    meta = {
        "lineup_weight_target": target_weight,
        "lineup_fraction_mean": float(lineup_fraction.mean()),
        "lineup_fraction_min": float(lineup_fraction.min()),
        "lineup_fraction_max": float(lineup_fraction.max()),
        "sample_weight_mean": float(sample_weights.mean()),
        "sample_weight_min": float(sample_weights.min()),
        "sample_weight_max": float(sample_weights.max()),
    }
    return torch.as_tensor(sample_weights, dtype=torch.double), meta


def _build_sparse_candidate_example_sampling_weights(
    examples: list[Any],
    *,
    feature_columns: list[str],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    sparse_candidate_weight: float,
    prior_minutes_max: float,
    hist_start_rate_max: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Build per-example train sampling weights for broad sparse-starter candidate games."""
    if not examples:
        return (
            torch.ones((0,), dtype=torch.double),
            {
                "sparse_candidate_weight_target": float(max(1.0, float(sparse_candidate_weight))),
                "candidate_game_rate": 0.0,
                "candidate_player_fraction_mean": 0.0,
                "candidate_player_fraction_min": 0.0,
                "candidate_player_fraction_max": 0.0,
                "sample_weight_mean": 1.0,
                "sample_weight_min": 1.0,
                "sample_weight_max": 1.0,
            },
        )

    target_weight = float(max(1.0, float(sparse_candidate_weight)))
    if target_weight <= 1.0:
        ones = torch.ones((len(examples),), dtype=torch.double)
        return (
            ones,
            {
                "sparse_candidate_weight_target": target_weight,
                "candidate_game_rate": 0.0,
                "candidate_player_fraction_mean": 0.0,
                "candidate_player_fraction_min": 0.0,
                "candidate_player_fraction_max": 0.0,
                "sample_weight_mean": 1.0,
                "sample_weight_min": 1.0,
                "sample_weight_max": 1.0,
            },
        )

    idx_by_name = {name: idx for idx, name in enumerate(feature_columns)}
    prior_idx = idx_by_name.get("minutes_from_stints_prior_20", -1)
    hist_idxs = [
        idx_by_name[name]
        for name in ("recent_start_pct_10", "started_proxy_rate_prior_10", "started_proxy_rate_prior_20")
        if name in idx_by_name
    ]
    if prior_idx < 0 or not hist_idxs:
        raise ValueError(
            "sparse candidate sampler requires minutes_from_stints_prior_20 and at least one historical start-rate feature"
        )

    prior_mean = float(feature_mean[int(prior_idx)])
    prior_std = float(feature_std[int(prior_idx)]) if abs(float(feature_std[int(prior_idx)])) > 1e-6 else 1.0
    hist_means = [float(feature_mean[int(idx)]) for idx in hist_idxs]
    hist_stds = [float(feature_std[int(idx)]) if abs(float(feature_std[int(idx)])) > 1e-6 else 1.0 for idx in hist_idxs]

    candidate_present = np.zeros((len(examples),), dtype=np.float64)
    candidate_fraction = np.zeros((len(examples),), dtype=np.float64)

    for idx, ex in enumerate(examples):
        valid = np.asarray(ex.player_valid_mask, dtype=bool)
        starter_hint = np.asarray(ex.starter_force_active_worlds, dtype=bool)
        features = np.asarray(ex.player_features, dtype=np.float32)
        if valid.size == 0 or features.ndim != 3:
            continue
        if valid.shape != starter_hint.shape or valid.shape != features.shape[:2]:
            raise ValueError("example masks/features must align for sparse candidate sampler")

        prior_minutes = features[..., int(prior_idx)].astype(np.float64) * prior_std + prior_mean
        hist_parts = [
            features[..., int(hist_idx)].astype(np.float64) * hist_std + hist_mean
            for hist_idx, hist_mean, hist_std in zip(hist_idxs, hist_means, hist_stds, strict=True)
        ]
        hist_rate = np.maximum.reduce(hist_parts)
        candidate_mask = (
            valid
            & starter_hint
            & (prior_minutes <= float(prior_minutes_max))
            & (hist_rate <= float(hist_start_rate_max))
        )
        valid_count = int(valid.sum())
        candidate_count = int(candidate_mask.sum())
        if candidate_count > 0:
            candidate_present[idx] = 1.0
            candidate_fraction[idx] = float(candidate_count) / float(max(1, valid_count))

    sample_weights = 1.0 + (target_weight - 1.0) * candidate_present
    meta = {
        "sparse_candidate_weight_target": target_weight,
        "candidate_game_rate": float(candidate_present.mean()),
        "candidate_player_fraction_mean": float(candidate_fraction.mean()),
        "candidate_player_fraction_min": float(candidate_fraction.min()),
        "candidate_player_fraction_max": float(candidate_fraction.max()),
        "sample_weight_mean": float(sample_weights.mean()),
        "sample_weight_min": float(sample_weights.min()),
        "sample_weight_max": float(sample_weights.max()),
    }
    return torch.as_tensor(sample_weights, dtype=torch.double), meta


def _build_bench_riser_example_sampling_weights(
    examples: list[Any],
    *,
    feature_columns: list[str],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    bench_candidate_weight: float,
    prior_minutes_min: float,
    hist_start_rate_max: float,
    prior_play_prob_min: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Build per-example train sampling weights for bench-riser candidate games."""
    if not examples:
        return (
            torch.ones((0,), dtype=torch.double),
            {
                "bench_candidate_weight_target": float(max(1.0, float(bench_candidate_weight))),
                "candidate_game_rate": 0.0,
                "candidate_player_fraction_mean": 0.0,
                "candidate_player_fraction_min": 0.0,
                "candidate_player_fraction_max": 0.0,
                "sample_weight_mean": 1.0,
                "sample_weight_min": 1.0,
                "sample_weight_max": 1.0,
            },
        )

    target_weight = float(max(1.0, float(bench_candidate_weight)))
    if target_weight <= 1.0:
        ones = torch.ones((len(examples),), dtype=torch.double)
        return (
            ones,
            {
                "bench_candidate_weight_target": target_weight,
                "candidate_game_rate": 0.0,
                "candidate_player_fraction_mean": 0.0,
                "candidate_player_fraction_min": 0.0,
                "candidate_player_fraction_max": 0.0,
                "sample_weight_mean": 1.0,
                "sample_weight_min": 1.0,
                "sample_weight_max": 1.0,
            },
        )

    idx_by_name = {name: idx for idx, name in enumerate(feature_columns)}
    prior_idx = idx_by_name.get("minutes_from_stints_prior_20", -1)
    play_prob_idx = idx_by_name.get("prior_play_prob", -1)
    hist_idxs = [
        idx_by_name[name]
        for name in ("recent_start_pct_10", "started_proxy_rate_prior_10", "started_proxy_rate_prior_20")
        if name in idx_by_name
    ]
    if prior_idx < 0 or play_prob_idx < 0 or not hist_idxs:
        raise ValueError(
            "bench-riser sampler requires minutes_from_stints_prior_20, prior_play_prob, "
            "and at least one historical start-rate feature"
        )

    prior_mean = float(feature_mean[int(prior_idx)])
    prior_std = float(feature_std[int(prior_idx)]) if abs(float(feature_std[int(prior_idx)])) > 1e-6 else 1.0
    play_prob_mean = float(feature_mean[int(play_prob_idx)])
    play_prob_std = float(feature_std[int(play_prob_idx)]) if abs(float(feature_std[int(play_prob_idx)])) > 1e-6 else 1.0
    hist_means = [float(feature_mean[int(idx)]) for idx in hist_idxs]
    hist_stds = [float(feature_std[int(idx)]) if abs(float(feature_std[int(idx)])) > 1e-6 else 1.0 for idx in hist_idxs]

    candidate_present = np.zeros((len(examples),), dtype=np.float64)
    candidate_fraction = np.zeros((len(examples),), dtype=np.float64)

    for idx, ex in enumerate(examples):
        valid = np.asarray(ex.player_valid_mask, dtype=bool)
        starter_hint = np.asarray(ex.starter_force_active_worlds, dtype=bool)
        features = np.asarray(ex.player_features, dtype=np.float32)
        if valid.size == 0 or features.ndim != 3:
            continue
        if valid.shape != starter_hint.shape or valid.shape != features.shape[:2]:
            raise ValueError("example masks/features must align for bench-riser candidate sampler")

        prior_minutes = features[..., int(prior_idx)].astype(np.float64) * prior_std + prior_mean
        prior_play_prob = features[..., int(play_prob_idx)].astype(np.float64) * play_prob_std + play_prob_mean
        hist_parts = [
            features[..., int(hist_idx)].astype(np.float64) * hist_std + hist_mean
            for hist_idx, hist_mean, hist_std in zip(hist_idxs, hist_means, hist_stds, strict=True)
        ]
        hist_rate = np.maximum.reduce(hist_parts)
        candidate_mask = (
            valid
            & (~starter_hint)
            & (prior_minutes >= float(prior_minutes_min))
            & (prior_play_prob >= float(prior_play_prob_min))
            & (hist_rate <= float(hist_start_rate_max))
        )
        valid_count = int(valid.sum())
        candidate_count = int(candidate_mask.sum())
        if candidate_count > 0:
            candidate_present[idx] = 1.0
            candidate_fraction[idx] = float(candidate_count) / float(max(1, valid_count))

    sample_weights = 1.0 + (target_weight - 1.0) * candidate_present
    meta = {
        "bench_candidate_weight_target": target_weight,
        "candidate_game_rate": float(candidate_present.mean()),
        "candidate_player_fraction_mean": float(candidate_fraction.mean()),
        "candidate_player_fraction_min": float(candidate_fraction.min()),
        "candidate_player_fraction_max": float(candidate_fraction.max()),
        "sample_weight_mean": float(sample_weights.mean()),
        "sample_weight_min": float(sample_weights.min()),
        "sample_weight_max": float(sample_weights.max()),
    }
    return torch.as_tensor(sample_weights, dtype=torch.double), meta


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


def _minutes_hurdle_nll(
    *,
    pred_minutes: torch.Tensor,
    target_minutes: torch.Tensor,
    zero_logits: torch.Tensor,
    sigma: torch.Tensor,
    valid_mask: torch.Tensor,
    zero_threshold: float,
) -> torch.Tensor:
    if pred_minutes.shape != target_minutes.shape:
        raise ValueError("pred_minutes and target_minutes must have the same shape")
    if zero_logits.shape != pred_minutes.shape or sigma.shape != pred_minutes.shape:
        raise ValueError("zero_logits/sigma must match pred_minutes shape")
    if valid_mask.shape != pred_minutes.shape:
        raise ValueError("valid_mask must match pred_minutes shape")
    if float(zero_threshold) < 0.0:
        raise ValueError("zero_threshold must be >= 0")

    valid = valid_mask.to(dtype=torch.bool)
    valid_f = valid.to(dtype=pred_minutes.dtype)

    # Hurdle gate: probability mass at zero minutes.
    is_zero_target = (target_minutes <= float(zero_threshold)).to(dtype=pred_minutes.dtype)
    zero_bce_per = F.binary_cross_entropy_with_logits(
        zero_logits,
        is_zero_target,
        reduction="none",
    )
    zero_bce = (zero_bce_per * valid_f).sum() / valid_f.sum().clamp(min=1.0)

    # Positive branch: Gaussian likelihood over strictly positive-minute rows.
    pos_mask = valid & (target_minutes > float(zero_threshold))
    pos_mask_f = pos_mask.to(dtype=pred_minutes.dtype)
    sigma_safe = torch.clamp(sigma, min=1e-4)
    residual = target_minutes - pred_minutes
    gaussian_per = 0.5 * (
        (residual * residual) / (sigma_safe * sigma_safe)
        + 2.0 * torch.log(sigma_safe)
        + math.log(2.0 * math.pi)
    )
    gaussian = (gaussian_per * pos_mask_f).sum() / pos_mask_f.sum().clamp(min=1.0)

    return zero_bce + gaussian


def _raw_feature_from_normalized(
    player_features_flat: torch.Tensor,
    *,
    feature_idx: int,
    feature_mean: np.ndarray | None,
    feature_std: np.ndarray | None,
) -> torch.Tensor | None:
    if int(feature_idx) < 0 or int(feature_idx) >= int(player_features_flat.shape[-1]):
        return None
    val = player_features_flat[..., int(feature_idx)]
    if feature_mean is None or feature_std is None:
        return val
    mean_i = float(feature_mean[int(feature_idx)])
    std_i = float(feature_std[int(feature_idx)])
    if not math.isfinite(std_i) or std_i <= 1e-6:
        std_i = 1.0
    return val * std_i + mean_i


def _build_minutes_role_targets_contextual(
    *,
    y_minutes: torch.Tensor,
    valid_mask: torch.Tensor,
    active_threshold: float,
    lineup_starter_announced: torch.Tensor | None,
    historical_start_rate: torch.Tensor | None,
    prior_minutes: torch.Tensor | None,
) -> torch.Tensor:
    """Starter-aware role buckets."""

    if y_minutes.shape != valid_mask.shape:
        raise ValueError("y_minutes and valid_mask must have matching shape")
    minutes = y_minutes.to(dtype=torch.float32)
    valid = valid_mask.to(dtype=torch.bool)
    active = (minutes >= float(active_threshold)) & valid
    starter_now = (
        lineup_starter_announced.to(dtype=torch.float32) >= 0.5
        if lineup_starter_announced is not None
        else torch.zeros_like(minutes, dtype=torch.bool)
    )
    hist_start = (
        torch.clamp(historical_start_rate.to(dtype=torch.float32), min=0.0, max=1.0)
        if historical_start_rate is not None
        else torch.zeros_like(minutes, dtype=torch.float32)
    )
    prior_min = (
        torch.clamp(prior_minutes.to(dtype=torch.float32), min=0.0, max=48.0)
        if prior_minutes is not None
        else torch.zeros_like(minutes, dtype=torch.float32)
    )

    out = torch.zeros_like(minutes, dtype=torch.long)
    rotation_bench = active & ~starter_now & (minutes >= 18.0)
    fringe_active = active & ~starter_now & ~rotation_bench
    starter_core = active & starter_now & ((hist_start >= 0.45) | (prior_min >= 24.0))
    starter_fillin = active & starter_now & ~starter_core

    out = torch.where(fringe_active, torch.full_like(out, 1), out)
    out = torch.where(rotation_bench, torch.full_like(out, 2), out)
    out = torch.where(starter_fillin, torch.full_like(out, 3), out)
    out = torch.where(starter_core, torch.full_like(out, 4), out)
    return out


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
    target_weights: torch.Tensor | None = None,
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
    if target_weights is not None and target_weights.shape != logits.shape:
        raise ValueError("target_weights must match logits shape")

    valid = valid_mask.to(dtype=torch.bool)
    obs = observed_mask.to(dtype=torch.bool) if observed_mask is not None else valid
    attempts = torch.clamp(attempts_true, min=0.0)
    weights = (
        torch.clamp(target_weights, min=0.0).to(dtype=attempts.dtype)
        if target_weights is not None
        else torch.ones_like(attempts)
    )

    loss_sum = logits.new_zeros(())
    n_rows = logits.new_zeros(())
    for side in (0, 1):
        elig = valid & team_index.eq(side)
        label_mask = elig & obs
        target_mass = torch.where(label_mask, attempts * weights, torch.zeros_like(attempts))
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


def _playmaker_importance_weights(
    *,
    line_raw: torch.Tensor | None,
    prior_play_prob_raw: torch.Tensor | None,
    line_center: float,
    line_scale: float,
    max_weight: float,
) -> torch.Tensor | None:
    if line_raw is None:
        return None
    line_strength = torch.sigmoid((line_raw - float(line_center)) / max(float(line_scale), 1e-6))
    weights = 1.0 + max(float(max_weight) - 1.0, 0.0) * line_strength
    if prior_play_prob_raw is not None:
        weights = 1.0 + (weights - 1.0) * torch.clamp(prior_play_prob_raw, min=0.0, max=1.0)
    return torch.clamp(weights, min=1.0, max=max(float(max_weight), 1.0))


def _asymmetric_weighted_masked_scaled_huber_loss(
    *,
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    scale: float,
    delta: float,
    weights: torch.Tensor | None = None,
    underprediction_weight: float = 1.0,
) -> torch.Tensor:
    base_weights = (
        torch.ones_like(pred, dtype=pred.dtype)
        if weights is None
        else torch.clamp(weights.to(dtype=pred.dtype), min=0.0)
    )
    under_mult = torch.where(
        target > pred,
        torch.full_like(pred, max(float(underprediction_weight), 1.0)),
        torch.ones_like(pred),
    )
    return _weighted_masked_scaled_huber_loss(
        pred=pred,
        target=target,
        mask=mask,
        scale=scale,
        delta=delta,
        weights=base_weights * under_mult,
    )


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
    flow_contract_columns: list[str] | None = None,
    fg2_rate: torch.Tensor | float | None = None,
    fg3_rate: torch.Tensor | float | None = None,
    ft_rate: torch.Tensor | float | None = None,
    ast_override: torch.Tensor | None = None,
) -> torch.Tensor:
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
        ast_idx = _flow_index(contract_cols, "ast")
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
) -> torch.Tensor | None:
    if team_ast_budget is None or assist_share_logits is None:
        return None
    if team_ast_budget.ndim != 2 or team_ast_budget.shape[1] != 2:
        raise ValueError("team_ast_budget must have shape (B, 2)")
    if assist_share_logits.ndim != 2:
        raise ValueError("assist_share_logits must have shape (B, P)")
    if assist_share_logits.shape != player_valid_mask.shape or player_team_index.shape != player_valid_mask.shape:
        raise ValueError("assist-share reconstruction tensors must align on (B, P)")

    valid_mask = player_valid_mask.to(dtype=torch.bool)
    ast_share = torch.zeros_like(assist_share_logits)
    for side in (0, 1):
        side_mask = valid_mask & player_team_index.eq(side)
        if bool(side_mask.any()):
            side_logits = assist_share_logits.masked_fill(~side_mask, float("-inf"))
            ast_share = ast_share + torch.softmax(side_logits, dim=1) * side_mask.to(dtype=assist_share_logits.dtype)
    return ast_share * torch.where(
        player_team_index.eq(0),
        team_ast_budget[:, 0].unsqueeze(1),
        team_ast_budget[:, 1].unsqueeze(1),
    )


def _build_ast_override(
    *,
    flow_projected_base: torch.Tensor,
    flow_contract_columns: list[str],
    player_valid_mask: torch.Tensor,
    player_team_index: torch.Tensor,
    team_ast_budget: torch.Tensor | None,
    assist_share_logits: torch.Tensor | None,
    ast_blend_gate: torch.Tensor | None,
) -> torch.Tensor | None:
    factorized_ast = _reconstruct_ast_from_heads(
        player_valid_mask=player_valid_mask,
        player_team_index=player_team_index,
        team_ast_budget=team_ast_budget,
        assist_share_logits=assist_share_logits,
    )
    if factorized_ast is None:
        return None
    ast_idx = _flow_index(flow_contract_columns, "ast")
    flow_ast = flow_projected_base[..., ast_idx]
    if ast_blend_gate is None:
        return factorized_ast
    if ast_blend_gate.shape != flow_ast.shape:
        raise ValueError("ast_blend_gate must align with projected flow ast shape")
    gate = ast_blend_gate.to(device=flow_ast.device, dtype=flow_ast.dtype).clamp(min=0.0, max=1.0)
    return gate * factorized_ast + (1.0 - gate) * flow_ast


def _normalize_alloc_weights(
    weights: torch.Tensor,
    *,
    eligible_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
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
    elig = eligible_mask.to(dtype=torch.bool)
    masked_logits = logits.masked_fill(~elig, -1e9)
    probs = torch.softmax(masked_logits, dim=1)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    denom = probs.sum(dim=1, keepdim=True)
    elig_count = elig.to(dtype=probs.dtype).sum(dim=1, keepdim=True).clamp(min=1.0)
    uniform = elig.to(dtype=probs.dtype) / elig_count
    return torch.where(denom > 1e-8, probs / denom.clamp(min=1e-8), uniform)


def _reconcile_ast_to_team_budget(
    *,
    flow_projected: torch.Tensor,
    flow_contract_columns: list[str],
    player_valid_mask: torch.Tensor,
    player_team_index: torch.Tensor,
    active_mask: torch.Tensor,
    team_ast_budget: torch.Tensor | None,
    assist_share_logits: torch.Tensor | None,
    share_alpha: float,
    share_temperature: float,
) -> torch.Tensor:
    if team_ast_budget is None or assist_share_logits is None:
        return flow_projected
    if assist_share_logits.shape != player_valid_mask.shape:
        raise ValueError("assist_share_logits must align with player_valid_mask")
    ast_idx = _flow_index(flow_contract_columns, "ast")
    old_ast = flow_projected[..., ast_idx]
    alpha = float(np.clip(float(share_alpha), 0.0, 1.0))
    temp = max(float(share_temperature), 1e-6)
    valid = player_valid_mask.to(dtype=torch.bool)
    active = active_mask.to(dtype=torch.bool)
    alloc_base = valid & active
    out = flow_projected.clone()
    for side in (0, 1):
        side_mask = player_team_index.eq(side)
        elig = alloc_base & side_mask
        valid_side = valid & side_mask
        has_active = elig.any(dim=1, keepdim=True)
        elig = torch.where(has_active, elig, valid_side)
        flow_weights = _normalize_alloc_weights(old_ast, eligible_mask=elig)
        factorized_weights = _softmax_alloc_weights(assist_share_logits / temp, eligible_mask=elig)
        ast_weights = _normalize_alloc_weights(
            (1.0 - alpha) * flow_weights + alpha * factorized_weights,
            eligible_mask=elig,
        )
        new_ast = ast_weights * torch.clamp(team_ast_budget[:, side], min=0.0).unsqueeze(1)
        side_f = side_mask.to(dtype=out.dtype)
        out[..., ast_idx] = out[..., ast_idx] * (1.0 - side_f) + new_ast * side_f
    return out


def _reconcile_points_to_team_budget(
    *,
    flow_projected: torch.Tensor,
    flow_contract_columns: list[str],
    player_valid_mask: torch.Tensor,
    player_team_index: torch.Tensor,
    active_mask: torch.Tensor,
    team_points_budget: torch.Tensor | None,
    budget_alpha: float,
) -> torch.Tensor:
    if team_points_budget is None:
        return flow_projected
    if team_points_budget.ndim != 2 or team_points_budget.shape[1] != 2:
        raise ValueError("team_points_budget must have shape (B, 2)")

    fg2m_idx = _flow_index(flow_contract_columns, "fg2m")
    fg3m_idx = _flow_index(flow_contract_columns, "fg3m")
    ftm_idx = _flow_index(flow_contract_columns, "ftm")
    fga2_idx = _flow_index(flow_contract_columns, "fga2")
    fga3_idx = _flow_index(flow_contract_columns, "fga3")
    fta_idx = _flow_index(flow_contract_columns, "fta")

    alpha = float(np.clip(float(budget_alpha), 0.0, 1.0))
    if alpha <= 0.0:
        return flow_projected

    valid = player_valid_mask.to(dtype=torch.bool)
    active = active_mask.to(dtype=torch.bool)
    alloc_base = valid & active
    fg2m = flow_projected[..., fg2m_idx]
    fg3m = flow_projected[..., fg3m_idx]
    ftm = flow_projected[..., ftm_idx]
    fga2 = flow_projected[..., fga2_idx]
    fga3 = flow_projected[..., fga3_idx]
    fta = flow_projected[..., fta_idx]
    for side in (0, 1):
        side_mask = player_team_index.eq(side)
        elig = alloc_base & side_mask
        valid_side = valid & side_mask
        has_active = elig.any(dim=1, keepdim=True)
        elig = torch.where(has_active, elig, valid_side)
        side_f = elig.to(dtype=flow_projected.dtype)

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
    out = flow_projected.clone()
    out[..., fg2m_idx] = fg2m
    out[..., fg3m_idx] = fg3m
    out[..., ftm_idx] = ftm
    return out


def _resolve_team_points_budget(
    *,
    model_cfg: GameTransformerV2Config,
    game_features: torch.Tensor,
    team_points_budget_out: torch.Tensor | None,
    team_ppp_out: torch.Tensor | None = None,
    possession_out: torch.Tensor | None = None,
) -> torch.Tensor | None:
    mode = str(getattr(model_cfg, "team_points_budget_parameterization", "absolute")).strip().lower()
    if mode == "market_implied":
        game_names = list(getattr(model_cfg, "game_feature_columns", []))
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
            game_names = list(getattr(model_cfg, "game_feature_columns", []))
            poss_idx = game_names.index("estimated_possessions") if "estimated_possessions" in game_names else -1
            if poss_idx < 0:
                return None
            poss = game_features[:, poss_idx : poss_idx + 1].expand(-1, 2)
        return team_ppp_out * poss.clamp_min(1.0)
    return team_points_budget_out


def _resolve_team_opportunity_share(
    *,
    model_cfg: GameTransformerV2Config,
    game_features: torch.Tensor,
) -> torch.Tensor | None:
    mode = str(getattr(model_cfg, "team_opportunity_budget_parameterization", "absolute")).strip().lower()
    if mode != "market_implied_share":
        return None
    game_names = list(getattr(model_cfg, "game_feature_columns", []))
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
    flow_projected: torch.Tensor,
    flow_contract_columns: list[str],
    player_valid_mask: torch.Tensor,
    player_team_index: torch.Tensor,
    active_mask: torch.Tensor,
    team_opportunity_share: torch.Tensor | None,
    budget_alpha: float,
    preserve_possessions: bool = False,
) -> torch.Tensor:
    if team_opportunity_share is None:
        return flow_projected
    if team_opportunity_share.ndim != 2 or team_opportunity_share.shape[1] != 2:
        raise ValueError("team_opportunity_share must have shape (B, 2)")

    fga2_idx = _flow_index(flow_contract_columns, "fga2")
    fg2m_idx = _flow_index(flow_contract_columns, "fg2m")
    fga3_idx = _flow_index(flow_contract_columns, "fga3")
    fg3m_idx = _flow_index(flow_contract_columns, "fg3m")
    fta_idx = _flow_index(flow_contract_columns, "fta")
    ftm_idx = _flow_index(flow_contract_columns, "ftm")
    tov_idx = _flow_index(flow_contract_columns, "tov")
    oreb_idx = _flow_index(flow_contract_columns, "oreb")

    alpha = float(np.clip(float(budget_alpha), 0.0, 1.0))
    if alpha <= 0.0:
        return flow_projected

    valid = player_valid_mask.to(dtype=torch.bool)
    active = active_mask.to(dtype=torch.bool)
    alloc_base = valid & active
    fga2 = flow_projected[..., fga2_idx]
    fg2m = flow_projected[..., fg2m_idx]
    fga3 = flow_projected[..., fga3_idx]
    fg3m = flow_projected[..., fg3m_idx]
    fta = flow_projected[..., fta_idx]
    ftm = flow_projected[..., ftm_idx]
    tov = flow_projected[..., tov_idx]
    oreb = flow_projected[..., oreb_idx]

    team_fga2 = torch.zeros((flow_projected.shape[0], 2), dtype=flow_projected.dtype, device=flow_projected.device)
    team_fga3 = torch.zeros_like(team_fga2)
    team_fta = torch.zeros_like(team_fga2)
    team_tov = torch.zeros_like(team_fga2)
    team_oreb = torch.zeros_like(team_fga2)
    for side in (0, 1):
        side_mask = player_team_index.eq(side)
        elig = alloc_base & side_mask
        valid_side = valid & side_mask
        has_active = elig.any(dim=1, keepdim=True)
        elig = torch.where(has_active, elig, valid_side)
        side_f = elig.to(dtype=flow_projected.dtype)
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
        side_mask = player_team_index.eq(side)
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
            side_f = side_mask.to(dtype=flow_projected.dtype)
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

    out = flow_projected.clone()
    out[..., fga2_idx] = fga2
    out[..., fg2m_idx] = fg2m
    out[..., fga3_idx] = fga3
    out[..., fg3m_idx] = fg3m
    out[..., fta_idx] = fta
    out[..., ftm_idx] = ftm
    out[..., tov_idx] = tov
    return out


def _swap_team_side_values(team_values: torch.Tensor) -> torch.Tensor:
    if team_values.ndim != 2 or team_values.shape[1] != 2:
        raise ValueError("team_values must have shape (B, 2)")
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


def _reconcile_rebounds_to_opportunity_budgets(
    *,
    flow_projected: torch.Tensor,
    flow_contract_columns: list[str],
    player_valid_mask: torch.Tensor,
    player_team_index: torch.Tensor,
    active_mask: torch.Tensor,
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
        return flow_projected
    if reconcile_oreb and (oreb_share_logits is None or (team_oreb_budget is None and not flow_oreb_budget)):
        return flow_projected
    if reconcile_dreb and (dreb_share_logits is None or (team_dreb_budget is None and not deterministic_dreb_budget)):
        return flow_projected
    if player_valid_mask.shape != player_team_index.shape or player_valid_mask.shape != active_mask.shape:
        raise ValueError("player_valid_mask/player_team_index/active_mask must align")
    if reconcile_oreb and not flow_oreb_budget and (team_oreb_budget.ndim != 2 or team_oreb_budget.shape[1] != 2):
        raise ValueError("team_oreb_budget must have shape (B, 2)")
    if (
        reconcile_dreb
        and not deterministic_dreb_budget
        and (team_dreb_budget.ndim != 2 or team_dreb_budget.shape[1] != 2)
    ):
        raise ValueError("team_dreb_budget must have shape (B, 2)")
    if (
        reconcile_oreb
        and not flow_oreb_budget
        and team_oreb_budget_gate is not None
        and team_oreb_budget_gate.shape != team_oreb_budget.shape
    ):
        raise ValueError("team_oreb_budget_gate must match team_oreb_budget")
    if reconcile_dreb and team_dreb_budget_gate is not None and (
        team_dreb_budget_gate.ndim != 2 or team_dreb_budget_gate.shape[1] != 2
    ):
        raise ValueError("team_dreb_budget_gate must have shape (B, 2)")
    if (
        reconcile_oreb
        and reconcile_dreb
        and not flow_oreb_budget
        and not deterministic_dreb_budget
        and team_dreb_budget.shape != team_oreb_budget.shape
    ):
        raise ValueError("team_dreb_budget must match team_oreb_budget")
    if reconcile_oreb and oreb_share_logits.shape != player_valid_mask.shape:
        raise ValueError("oreb share logits must align with player_valid_mask")
    if reconcile_dreb and dreb_share_logits.shape != player_valid_mask.shape:
        raise ValueError("rebound share logits must align with player_valid_mask")

    oreb_idx = _flow_index(flow_contract_columns, "oreb")
    dreb_idx = _flow_index(flow_contract_columns, "dreb")
    fga2_idx = _flow_index(flow_contract_columns, "fga2")
    fg2m_idx = _flow_index(flow_contract_columns, "fg2m")
    fga3_idx = _flow_index(flow_contract_columns, "fga3")
    fg3m_idx = _flow_index(flow_contract_columns, "fg3m")

    out = flow_projected.clone()
    old_oreb = flow_projected[..., oreb_idx].clone()
    old_dreb = flow_projected[..., dreb_idx].clone()
    own_missed = (flow_projected[..., fga2_idx] - flow_projected[..., fg2m_idx]) + (
        flow_projected[..., fga3_idx] - flow_projected[..., fg3m_idx]
    )
    own_missed = torch.clamp(own_missed, min=0.0)
    own_missed_team, _ = _team_sum_by_side(
        values=own_missed,
        valid_mask=player_valid_mask,
        team_index=player_team_index,
        observed_mask=player_valid_mask,
    )
    opp_missed_team = _swap_team_side_values(own_missed_team)
    flow_oreb_team, _ = _team_sum_by_side(
        values=old_oreb,
        valid_mask=player_valid_mask,
        team_index=player_team_index,
        observed_mask=player_valid_mask,
    )
    opp_oreb_team = _swap_team_side_values(flow_oreb_team)
    flow_dreb_team, _ = _team_sum_by_side(
        values=old_dreb,
        valid_mask=player_valid_mask,
        team_index=player_team_index,
        observed_mask=player_valid_mask,
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
    valid = player_valid_mask.to(dtype=torch.bool)
    active = active_mask.to(dtype=torch.bool)
    alloc_base = valid & active
    new_oreb_all = old_oreb.clone()
    new_dreb_all = old_dreb.clone()
    for side in (0, 1):
        side_mask = player_team_index.eq(side)
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
        out[..., oreb_idx] = new_oreb_all
    if reconcile_dreb:
        out[..., dreb_idx] = new_dreb_all
    return out


def _ast_blend_gate_targets(
    *,
    flow_ast: torch.Tensor,
    factorized_ast: torch.Tensor,
    target_ast: torch.Tensor,
    valid_mask: torch.Tensor,
    eps: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor]:
    if flow_ast.shape != factorized_ast.shape or flow_ast.shape != target_ast.shape or flow_ast.shape != valid_mask.shape:
        raise ValueError("AST gate target tensors must align on shape")
    denom = factorized_ast - flow_ast
    solvable = valid_mask.to(dtype=torch.bool) & denom.abs().ge(float(eps))
    safe_denom = torch.where(solvable, denom, torch.ones_like(denom))
    alpha = torch.zeros_like(flow_ast, dtype=flow_ast.dtype)
    alpha = torch.where(
        solvable,
        torch.clamp((target_ast - flow_ast) / safe_denom, min=0.0, max=1.0),
        alpha,
    )
    return alpha, solvable


def _team_budget_blend_gate_targets(
    *,
    flow_team_budget: torch.Tensor,
    factorized_team_budget: torch.Tensor,
    target_team_budget: torch.Tensor,
    observed_mask: torch.Tensor,
    eps: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        flow_team_budget.shape != factorized_team_budget.shape
        or flow_team_budget.shape != target_team_budget.shape
        or flow_team_budget.shape != observed_mask.shape
    ):
        raise ValueError("team budget blend gate tensors must align on shape")
    denom = factorized_team_budget - flow_team_budget
    solvable = observed_mask.to(dtype=torch.bool) & denom.abs().ge(float(eps))
    safe_denom = torch.where(solvable, denom, torch.ones_like(denom))
    alpha = torch.zeros_like(flow_team_budget, dtype=flow_team_budget.dtype)
    alpha = torch.where(
        solvable,
        torch.clamp((target_team_budget - flow_team_budget) / safe_denom, min=0.0, max=1.0),
        alpha,
    )
    return alpha, solvable


def _mask_ast_from_flow_targets(
    flow_targets: torch.Tensor,
    flow_observed_mask: torch.Tensor,
    *,
    flow_target_columns: list[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    ast_idx = _flow_index(flow_target_columns, "ast")
    masked_targets = flow_targets.clone()
    masked_observed = flow_observed_mask.clone()
    masked_targets[..., ast_idx] = 0.0
    masked_observed[..., ast_idx] = False
    return masked_targets, masked_observed


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
    flow_contract_columns: list[str],
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
    flow_model_target_columns: list[str] = list(model.flow_target_columns)  # type: ignore[attr-defined]
    model_cfg = getattr(model, "gtv2_config", None)
    factorized_ast = bool(getattr(model_cfg, "assist_share_factorized_ast", False))
    replace_flow_ast = bool(getattr(model_cfg, "assist_share_replace_flow_ast", False)) or factorized_ast
    reconcile_ast_budget = bool(getattr(model_cfg, "assist_share_reconcile_ast_budget", False))
    reconcile_rebound_budget = bool(getattr(model_cfg, "rebound_factor_reconcile_oreb_dreb", False))
    valid_flat = ctx.player_valid_mask.to(dtype=torch.bool)
    env_context = getattr(ctx, "env_context", None)

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
                len(flow_model_target_columns),
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
            minutes_context=ctx.minutes.minutes,
            env_context=env_context,
        )
        fg2_rate = ctx.efficiency.mean_fg2 if getattr(ctx, "efficiency", None) is not None else None
        fg3_rate = ctx.efficiency.mean_fg3 if getattr(ctx, "efficiency", None) is not None else None
        ft_rate = ctx.efficiency.mean_ft if getattr(ctx, "efficiency", None) is not None else None
        ast_override = None
        if replace_flow_ast:
            flow_projected_base = _project_flow_stats_to_contract(
                flow_samples,
                flow_target_columns=flow_model_target_columns,
                flow_contract_columns=flow_contract_columns,
                fg2_rate=fg2_rate,
                fg3_rate=fg3_rate,
                ft_rate=ft_rate,
            )
            ast_override = _build_ast_override(
                flow_projected_base=flow_projected_base,
                flow_contract_columns=flow_contract_columns,
                player_valid_mask=ctx.player_valid_mask,
                player_team_index=ctx.player_team_index,
                team_ast_budget=ctx.team_ast_budget.team_ast if getattr(ctx, "team_ast_budget", None) is not None else None,
                assist_share_logits=ctx.assist_share.ast_logits if getattr(ctx, "assist_share", None) is not None else None,
                ast_blend_gate=ctx.ast_blend_gate.gate if getattr(ctx, "ast_blend_gate", None) is not None else None,
            )
        flow_samples = _project_flow_stats_to_contract(
            flow_samples,
            flow_target_columns=flow_model_target_columns,
            flow_contract_columns=flow_contract_columns,
            fg2_rate=fg2_rate,
            fg3_rate=fg3_rate,
            ft_rate=ft_rate,
            ast_override=ast_override,
        )
        if reconcile_ast_budget:
            flow_samples = _reconcile_ast_to_team_budget(
                flow_projected=flow_samples,
                flow_contract_columns=flow_contract_columns,
                player_valid_mask=ctx.player_valid_mask,
                player_team_index=ctx.player_team_index,
                active_mask=active_out.active_mask,
                team_ast_budget=ctx.team_ast_budget.team_ast if getattr(ctx, "team_ast_budget", None) is not None else None,
                assist_share_logits=ctx.assist_share.ast_logits if getattr(ctx, "assist_share", None) is not None else None,
                share_alpha=float(getattr(model_cfg, "assist_share_reconcile_alpha", 0.75)),
                share_temperature=float(getattr(model_cfg, "assist_share_reconcile_temperature", 0.85)),
            )
        if reconcile_rebound_budget:
            flow_samples = _reconcile_rebounds_to_opportunity_budgets(
                flow_projected=flow_samples,
                flow_contract_columns=flow_contract_columns,
                player_valid_mask=ctx.player_valid_mask,
                player_team_index=ctx.player_team_index,
                active_mask=active_out.active_mask,
                team_oreb_budget=(
                    ctx.team_rebound_budget.team_oreb if getattr(ctx, "team_rebound_budget", None) is not None else None
                ),
                team_dreb_budget=(
                    ctx.team_rebound_budget.team_dreb if getattr(ctx, "team_rebound_budget", None) is not None else None
                ),
                team_oreb_budget_gate=(
                    ctx.rebound_budget_blend_gate.oreb_gate
                    if getattr(ctx, "rebound_budget_blend_gate", None) is not None
                    else None
                ),
                team_dreb_budget_gate=(
                    ctx.rebound_budget_blend_gate.dreb_gate
                    if getattr(ctx, "rebound_budget_blend_gate", None) is not None
                    else None
                ),
                oreb_share_logits=(
                    ctx.rebound_share.oreb_logits if getattr(ctx, "rebound_share", None) is not None else None
                ),
                dreb_share_logits=(
                    ctx.rebound_share.dreb_logits if getattr(ctx, "rebound_share", None) is not None else None
                ),
                share_alpha=float(getattr(model_cfg, "rebound_factor_reconcile_alpha", 0.50)),
                share_temperature=float(getattr(model_cfg, "rebound_factor_reconcile_temperature", 0.90)),
                reconcile_mode=str(getattr(model_cfg, "rebound_factor_reconcile_mode", "both")),
                budget_parameterization=str(getattr(model_cfg, "rebound_budget_parameterization", "absolute")),
                dreb_deterministic_discount=float(
                    getattr(model_cfg, "rebound_dreb_deterministic_discount", 1.0)
                ),
                oreb_reconcile_use_flow_budget=bool(
                    getattr(model_cfg, "rebound_oreb_reconcile_use_flow_budget", False)
                ),
                oreb_budget_blend_alpha=float(getattr(model_cfg, "rebound_oreb_budget_blend_alpha", 1.0)),
                dreb_budget_blend_alpha=float(getattr(model_cfg, "rebound_dreb_budget_blend_alpha", 1.0)),
            )
        flow_samples = flow_samples * active_out.active_mask.unsqueeze(-1).to(dtype=flow_samples.dtype)
        worlds.append(
            torch.nan_to_num(
                _compute_dk_fpts_from_flow(
                    flow_samples,
                    flow_target_columns=flow_contract_columns,
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
        or name.startswith("team_points_budget_head.")
        or name.startswith("backbone_team_points_budget_encoder.")
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


def _resolve_minutes_teacher_forcing_prob(
    *,
    epoch: int,
    start_prob: float,
    end_prob: float,
    ramp_epochs: int,
) -> float:
    if int(epoch) <= 0:
        raise ValueError("epoch must be >= 1")
    start = float(min(1.0, max(0.0, start_prob)))
    end = float(min(1.0, max(0.0, end_prob)))
    if int(ramp_epochs) <= 1:
        return end
    progress = min(1.0, float(int(epoch) - 1) / float(int(ramp_epochs) - 1))
    return float(start + (end - start) * progress)


def _parse_prefix_csv(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    parts = [str(part).strip() for part in str(value).split(",")]
    return tuple(part for part in parts if part)


def _matches_prefix(name: str, prefixes: tuple[str, ...]) -> bool:
    return bool(prefixes) and any(name.startswith(prefix) for prefix in prefixes)


def _apply_partial_checkpoint(
    *,
    model: nn.Module,
    checkpoint_path: Path,
    device: torch.device,
    prefixes: tuple[str, ...],
    label: str,
) -> None:
    if not prefixes:
        raise ValueError(f"{label} requires at least one parameter prefix")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"{label} checkpoint not found: {checkpoint_path}")
    raw_state = torch.load(checkpoint_path, map_location=device)
    model_state = model.state_dict()
    graft_state: dict[str, torch.Tensor] = {}
    shape_mismatched: list[str] = []
    missing_in_model: list[str] = []
    for key, value in raw_state.items():
        if not _matches_prefix(str(key), prefixes):
            continue
        if key not in model_state:
            missing_in_model.append(str(key))
            continue
        if model_state[key].shape != value.shape:
            shape_mismatched.append(str(key))
            continue
        graft_state[str(key)] = value
    if not graft_state:
        raise RuntimeError(
            f"{label} checkpoint {checkpoint_path} had no compatible parameters for prefixes {list(prefixes)}"
        )
    missing, unexpected = model.load_state_dict(graft_state, strict=False)
    if unexpected:
        raise RuntimeError(f"{label} produced unexpected keys from {checkpoint_path}: {unexpected}")
    loaded_keys = sorted(graft_state.keys())
    print(
        f"[{label}] loaded {len(loaded_keys)} tensors from {checkpoint_path} using prefixes {list(prefixes)}",
        flush=True,
    )
    if shape_mismatched:
        print(
            f"[{label}] shape-mismatched keys skipped ({len(shape_mismatched)}): {shape_mismatched}",
            flush=True,
        )
    if missing_in_model:
        print(
            f"[{label}] keys absent from target model skipped ({len(missing_in_model)}): {missing_in_model}",
            flush=True,
        )
    relevant_missing = [name for name in missing if _matches_prefix(str(name), prefixes)]
    if relevant_missing:
        raise RuntimeError(f"{label} failed to load requested prefixes from {checkpoint_path}: {relevant_missing}")


def _freeze_parameter_prefixes(model: nn.Module, *, prefixes: tuple[str, ...], label: str) -> None:
    if not prefixes:
        return
    frozen = 0
    total = 0
    for name, param in model.named_parameters():
        if _matches_prefix(str(name), prefixes):
            total += 1
            param.requires_grad = False
            frozen += 1
    if frozen == 0:
        raise RuntimeError(f"{label} requested freeze prefixes {list(prefixes)} but matched no parameters")
    print(f"[{label}] froze {frozen} parameter tensors for prefixes {list(prefixes)}", flush=True)


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


def _weighted_masked_scaled_huber_loss(
    *,
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    scale: float,
    delta: float,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Weighted Huber loss on normalized errors, averaged over mask-selected rows."""
    if pred.shape != target.shape:
        raise ValueError("pred and target must have the same shape")
    if mask.shape != pred.shape:
        raise ValueError("mask must match pred shape")
    if weights.shape != pred.shape:
        raise ValueError("weights must match pred shape")
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
    w = torch.where(mask_b, torch.clamp(weights.to(dtype=loss.dtype), min=0.0), torch.zeros_like(loss))
    denom = w.sum().clamp(min=1e-6)
    return (loss * w).sum() / denom


def _compute_named_direct_stat_losses(
    *,
    stat_specs: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]],
    delta: float,
) -> dict[str, torch.Tensor]:
    losses: dict[str, torch.Tensor] = {}
    for name, (pred, target, mask, scale) in stat_specs.items():
        losses[name] = _masked_scaled_huber_loss(
            pred=pred,
            target=target,
            mask=mask,
            scale=float(scale),
            delta=float(delta),
        )
    return losses


def _mean_named_losses(
    losses: dict[str, torch.Tensor],
    names: tuple[str, ...],
) -> torch.Tensor:
    if not names:
        raise ValueError("names must be non-empty")
    if not losses:
        raise ValueError("losses must be non-empty")
    tensors = [losses[name] for name in names if name in losses]
    if not tensors:
        return next(iter(losses.values())).new_zeros(())
    return torch.stack(tensors).mean()


def _load_reference_teacher_model(
    *,
    run_dir: Path,
    device: torch.device,
) -> tuple[GameTransformerV2Config, nn.Module]:
    config_path = Path(run_dir) / "config.json"
    checkpoint_path = Path(run_dir) / "checkpoint_stable.pt"
    if not config_path.exists():
        raise FileNotFoundError(f"missing teacher config: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"missing teacher checkpoint: {checkpoint_path}")
    config = GameTransformerV2Config.load(config_path)
    model = build_game_transformer_v2(config)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    setattr(model, "gtv2_config", config)
    model = model.to(device=device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return config, model


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
        "minutes_hurdle_nll",
        "flow_nll",
        "crps_fpts",
        "team_energy",
        "count_acc",
        "poss_nll",
        "backbone_nll",
        "three_pa_nll",
        "efficiency_nll",
        "usage_share_nll",
        "team_ast_budget_aux",
        "assist_share_aux",
        "assist_share_recon_aux",
        "ast_blend_gate_aux",
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
    val_minutes_mae: float,
) -> float:
    name = str(metric_name).strip().lower()
    if name == "val_total":
        return float(val_total)
    if name == "val_total_ex_possreg":
        return float(val_total) - float(w_poss_regression) * float(val_poss_regression)
    if name == "val_minutes_mae":
        return float(val_minutes_mae)
    raise ValueError(f"Unsupported early-stop metric: {metric_name}")


def _record_topk_minutes_checkpoint(
    *,
    candidates: list[MinutesCheckpointCandidate],
    epoch: int,
    val_minutes_mae: float,
    checkpoint_metric_value: float,
    val_total: float,
    checkpoint_path: Path,
    top_k: int,
) -> list[MinutesCheckpointCandidate]:
    if int(top_k) <= 0 or not math.isfinite(float(val_minutes_mae)):
        return list(candidates)
    updated = list(candidates)
    updated.append(
        MinutesCheckpointCandidate(
            epoch=int(epoch),
            val_minutes_mae=float(val_minutes_mae),
            checkpoint_metric_value=float(checkpoint_metric_value),
            val_total=float(val_total),
            checkpoint_path=str(checkpoint_path),
        )
    )
    updated.sort(key=lambda item: (item.val_minutes_mae, item.epoch))
    keep = updated[: max(1, int(top_k))]
    keep_paths = {item.checkpoint_path for item in keep}
    for item in updated[max(1, int(top_k)) :]:
        if item.checkpoint_path in keep_paths:
            continue
        path = Path(item.checkpoint_path)
        if path.exists():
            path.unlink()
    return keep


def _compute_sparse_rerank_score(
    *,
    sparse_next_up_underpred_rate: float,
    active_count_mae: float,
    starter_sparse_pred_minutes_mean: float,
    target_starter_sparse_minutes: float,
    weight_sparse_underpred: float,
    weight_active_count_mae: float,
    weight_starter_sparse_shortfall: float,
) -> float:
    values = (
        float(sparse_next_up_underpred_rate),
        float(active_count_mae),
        float(starter_sparse_pred_minutes_mean),
    )
    if not all(math.isfinite(v) for v in values):
        return float("inf")
    starter_shortfall = max(0.0, float(target_starter_sparse_minutes) - float(starter_sparse_pred_minutes_mean))
    return float(
        float(weight_sparse_underpred) * float(sparse_next_up_underpred_rate)
        + float(weight_active_count_mae) * float(active_count_mae)
        + float(weight_starter_sparse_shortfall) * float(starter_shortfall)
    )


def _select_sparse_rerank_candidate(
    *,
    candidates: list[dict[str, Any]],
    minutes_mae_tolerance: float,
) -> dict[str, Any] | None:
    if not candidates:
        return None
    best_minutes_mae = min(float(item["val_minutes_mae"]) for item in candidates if math.isfinite(float(item["val_minutes_mae"])))
    eligible = [
        item for item in candidates if float(item["val_minutes_mae"]) <= float(best_minutes_mae) + float(minutes_mae_tolerance)
    ]
    if not eligible:
        eligible = sorted(candidates, key=lambda item: (float(item["val_minutes_mae"]), int(item["epoch"])))[:1]
    eligible.sort(
        key=lambda item: (
            float(item["sparse_rerank"]["sparse_score"]),
            float(item["val_minutes_mae"]),
            int(item["epoch"]),
        )
    )
    return eligible[0]


def _evaluate_sparse_checkpoint_candidate(
    *,
    model: nn.Module,
    checkpoint_path: Path,
    device: torch.device,
    val_loader: DataLoader,
    val_df: pd.DataFrame,
    active_threshold: float,
    low_minutes_threshold: float,
    sparse_prior_play_prob_max: float,
    sparse_prior_minutes_max: float,
    starter_promotion_prior_minutes_max: float,
    starter_promotion_hist_start_rate_max: float,
    next_up_actual_min: float,
    next_up_pred_min: float,
    estimated_possessions_idx: int | None,
    prior_minutes_feature_idx: int | None,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    target_starter_sparse_minutes: float,
    weight_sparse_underpred: float,
    weight_active_count_mae: float,
    weight_starter_sparse_shortfall: float,
) -> SparseRerankMetrics:
    from scripts.rotation.eval_game_transformer_v2 import (
        _active_count_calibration,
        _attach_sparse_context,
        _build_sparse_context_frame,
        _predict,
        _sparse_rotation_metrics,
    )

    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state)
    model = model.to(device=device)

    player_df, team_df = _predict(
        model,
        val_loader,
        device=device,
        active_threshold=float(active_threshold),
        estimated_possessions_idx=estimated_possessions_idx,
        prior_minutes_feature_idx=prior_minutes_feature_idx,
        feature_mean=feature_mean,
        feature_std=feature_std,
        starter_promotion_prior_minutes_max=float(starter_promotion_prior_minutes_max),
    )
    sparse_context_df = _build_sparse_context_frame(val_df)
    player_eval_df = _attach_sparse_context(player_df, sparse_context_df)
    sparse_diag = _sparse_rotation_metrics(
        player_eval_df,
        active_threshold=float(active_threshold),
        low_minutes_threshold=float(low_minutes_threshold),
        sparse_prior_play_prob_max=float(sparse_prior_play_prob_max),
        sparse_prior_minutes_max=float(sparse_prior_minutes_max),
        starter_promotion_prior_minutes_max=float(starter_promotion_prior_minutes_max),
        starter_promotion_hist_start_rate_max=float(starter_promotion_hist_start_rate_max),
        next_up_actual_min=float(next_up_actual_min),
        next_up_pred_min=float(next_up_pred_min),
    )
    active_count_cal = _active_count_calibration(team_df)
    sparse_next_up_underpred_rate = float(
        (sparse_diag.get("failure_rates", {}) or {}).get(
            "starter_promotion_next_up_underprediction_rate",
            (sparse_diag.get("failure_rates", {}) or {}).get("sparse_next_up_underprediction_rate", float("inf")),
        )
    )
    active_count_mae = float((active_count_cal or {}).get("mae", float("inf")))
    starter_sparse_pred_minutes_mean = float(
        ((sparse_diag.get("slices", {}) or {}).get("starter_promotion_candidate", {}) or {}
    ).get(
        "pred_minutes_mean",
        (((sparse_diag.get("slices", {}) or {}).get("starter_sparse_prior", {}) or {}).get(
            "pred_minutes_mean",
            float("inf"),
        )),
    ))
    sparse_score = _compute_sparse_rerank_score(
        sparse_next_up_underpred_rate=sparse_next_up_underpred_rate,
        active_count_mae=active_count_mae,
        starter_sparse_pred_minutes_mean=starter_sparse_pred_minutes_mean,
        target_starter_sparse_minutes=float(target_starter_sparse_minutes),
        weight_sparse_underpred=float(weight_sparse_underpred),
        weight_active_count_mae=float(weight_active_count_mae),
        weight_starter_sparse_shortfall=float(weight_starter_sparse_shortfall),
    )
    return SparseRerankMetrics(
        sparse_next_up_underpred_rate=sparse_next_up_underpred_rate,
        active_count_mae=active_count_mae,
        starter_sparse_pred_minutes_mean=starter_sparse_pred_minutes_mean,
        sparse_score=sparse_score,
    )


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    active_threshold: float,
    min_active_count: int,
    max_active_count: int,
    flow_label_columns: list[str],
    run_phase2_flow: bool,
    run_phase3_decision: bool,
    w_minutes: float,
    w_minutes_nll: float,
    w_minutes_hurdle_nll: float,
    w_role_loss: float,
    w_starter_promotion_loss: float,
    w_sparse_starter_underpred_loss: float,
    w_bench_riser_underpred_loss: float,
    minutes_role_target_scheme: str,
    w_count: float,
    w_member: float,
    w_flow_nll: float,
    w_crps_fpts: float,
    w_team_energy: float,
    minutes_nll_sigma: float,
    minutes_hurdle_zero_threshold: float,
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
    w_team_efficiency_ppp_aux: float = 0.0,
    team_efficiency_ppp_target_scale: float = 0.12,
    w_team_ppp_aux: float = 0.0,
    team_ppp_target_scale: float = 0.12,
    w_team_advantage_aux: float = 0.0,
    team_advantage_target_scale: float = 8.0,
    w_usage_share_nll: float = 0.0,
    w_team_possession_aux: float = 0.0,
    w_team_points_budget_aux: float = 0.0,
    w_team_ast_budget_aux: float = 0.0,
    w_assist_share_aux: float = 0.0,
    w_assist_share_recon_aux: float = 0.0,
    w_ast_blend_gate_aux: float = 0.0,
    w_emergent_share_aux: float = 0.0,
    w_ast_share_aux: float = 0.0,
    w_reb_share_aux: float = 0.0,
    w_ast_team_rate_aux: float = 0.0,
    w_reb_opportunity_rate_aux: float = 0.0,
    w_team_rebound_budget_rate_aux: float = 0.0,
    w_rebound_budget_blend_gate_aux: float = 0.0,
    w_spread_aux: float = 0.0,
    w_total_aux: float = 0.0,
    w_props_pts_aux: float = 0.0,
    w_props_reb_aux: float = 0.0,
    w_props_ast_aux: float = 0.0,
    w_direct_pts_aux: float = 0.0,
    w_direct_reb_aux: float = 0.0,
    w_direct_ast_aux: float = 0.0,
    w_direct_stl_aux: float = 0.0,
    w_direct_blk_aux: float = 0.0,
    w_direct_tov_aux: float = 0.0,
    w_direct_boxscore_aux: float = 0.0,
    w_direct_opportunity_aux: float = 0.0,
    flow_anchor_teacher_model: nn.Module | None = None,
    w_flow_anchor_nonast_aux: float = 0.0,
    flow_anchor_target_scale: float = 8.0,
    spread_total_aux_ramp_epochs: int = 0,
    spread_total_aux_start_scale: float = 1.0,
    props_aux_ramp_epochs: int = 0,
    props_aux_start_scale: float = 1.0,
    direct_stat_aux_ramp_epochs: int = 0,
    direct_stat_aux_start_scale: float = 1.0,
    spread_aux_target_scale: float = 10.0,
    total_aux_target_scale: float = 25.0,
    props_pts_target_scale: float = 8.0,
    props_reb_target_scale: float = 4.0,
    props_ast_target_scale: float = 3.0,
    props_pts_aux_min_line: float = 0.0,
    props_reb_aux_min_line: float = 0.0,
    props_ast_aux_min_line: float = 0.0,
    team_ast_budget_target_scale: float = 8.0,
    assist_share_recon_target_scale: float = 3.0,
    assist_playmaker_line_center: float = 5.5,
    assist_playmaker_line_scale: float = 1.0,
    assist_playmaker_max_weight: float = 3.0,
    assist_underprediction_weight: float = 2.0,
    ast_blend_gate_target_eps: float = 0.25,
    rebound_budget_blend_gate_target_eps: float = 0.25,
    direct_pts_target_scale: float = 8.0,
    direct_reb_target_scale: float = 4.0,
    direct_ast_target_scale: float = 3.0,
    direct_stl_target_scale: float = 1.5,
    direct_blk_target_scale: float = 1.5,
    direct_tov_target_scale: float = 2.0,
    direct_fg3m_target_scale: float = 2.5,
    direct_ftm_target_scale: float = 3.0,
    direct_fga_target_scale: float = 8.0,
    direct_fta_target_scale: float = 4.0,
    spread_aux_huber_delta: float = 1.0,
    total_aux_huber_delta: float = 1.0,
    props_aux_huber_delta: float = 1.0,
    direct_stat_aux_huber_delta: float = 1.0,
    props_aux_confidence_min: float = 0.05,
    w_efficiency_mean_aux: float = 0.0,
    feature_mean: np.ndarray | None = None,
    feature_std: np.ndarray | None = None,
    an_pts_line_idx: int = -1,
    an_reb_line_idx: int = -1,
    an_ast_line_idx: int = -1,
    an_has_pts_idx: int = -1,
    an_has_reb_idx: int = -1,
    an_has_ast_idx: int = -1,
    an_pts_books_idx: int = -1,
    an_reb_books_idx: int = -1,
    an_ast_books_idx: int = -1,
    an_props_market_count_idx: int = -1,
    prior_play_prob_idx: int = -1,
    lineup_starter_announced_idx: int = -1,
    recent_start_pct_10_idx: int = -1,
    started_proxy_rate_prior_10_idx: int = -1,
    started_proxy_rate_prior_20_idx: int = -1,
    minutes_from_stints_prior_20_idx: int = -1,
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
    minutes_teacher_forcing_prob: float = 1.0,
    minutes_teacher_forcing_mode: str = "batch",
    flow_minutes_teacher_forcing_prob: float = 1.0,
    flow_minutes_teacher_forcing_mode: str = "batch",
    sparse_starter_loss_prior_play_prob_max: float = 0.20,
    sparse_starter_loss_prior_minutes_max: float = 6.0,
    sparse_starter_loss_hist_start_rate_max: float = 0.20,
    sparse_starter_loss_actual_min_threshold: float = 20.0,
    bench_riser_loss_prior_minutes_min: float = 0.0,
    bench_riser_loss_prior_play_prob_min: float = 0.0,
    bench_riser_loss_hist_start_rate_max: float = 0.50,
    bench_riser_loss_actual_min_threshold: float = 20.0,
    starter_promotion_prior_minutes_max: float = 18.0,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    minutes_teacher_forcing_prob = float(min(1.0, max(0.0, minutes_teacher_forcing_prob)))
    flow_minutes_teacher_forcing_prob = float(min(1.0, max(0.0, flow_minutes_teacher_forcing_prob)))
    spread_total_aux_ramp = _resolve_ramped_loss_scale(
        epoch=int(epoch_index),
        ramp_epochs=int(spread_total_aux_ramp_epochs),
        start_scale=float(spread_total_aux_start_scale),
    )
    w_spread_aux_eff = float(w_spread_aux) * float(spread_total_aux_ramp)
    w_total_aux_eff = float(w_total_aux) * float(spread_total_aux_ramp)
    props_aux_ramp = _resolve_ramped_loss_scale(
        epoch=int(epoch_index),
        ramp_epochs=int(props_aux_ramp_epochs),
        start_scale=float(props_aux_start_scale),
    )
    w_props_pts_aux_eff = float(w_props_pts_aux) * float(props_aux_ramp)
    w_props_reb_aux_eff = float(w_props_reb_aux) * float(props_aux_ramp)
    w_props_ast_aux_eff = float(w_props_ast_aux) * float(props_aux_ramp)
    direct_stat_aux_ramp = _resolve_ramped_loss_scale(
        epoch=int(epoch_index),
        ramp_epochs=int(direct_stat_aux_ramp_epochs),
        start_scale=float(direct_stat_aux_start_scale),
    )
    w_direct_pts_aux_eff = float(w_direct_pts_aux) * float(direct_stat_aux_ramp)
    w_direct_reb_aux_eff = float(w_direct_reb_aux) * float(direct_stat_aux_ramp)
    w_direct_ast_aux_eff = float(w_direct_ast_aux) * float(direct_stat_aux_ramp)
    w_direct_stl_aux_eff = float(w_direct_stl_aux) * float(direct_stat_aux_ramp)
    w_direct_blk_aux_eff = float(w_direct_blk_aux) * float(direct_stat_aux_ramp)
    w_direct_tov_aux_eff = float(w_direct_tov_aux) * float(direct_stat_aux_ramp)
    w_direct_boxscore_aux_eff = float(w_direct_boxscore_aux) * float(direct_stat_aux_ramp)
    w_direct_opportunity_aux_eff = float(w_direct_opportunity_aux) * float(direct_stat_aux_ramp)
    feature_mean_arr = np.asarray(feature_mean, dtype=np.float32) if feature_mean is not None else None
    feature_std_arr = np.asarray(feature_std, dtype=np.float32) if feature_std is not None else None
    if not hasattr(model, "flow_target_columns"):
        raise RuntimeError("_run_epoch requires model.flow_target_columns")
    flow_model_columns: list[str] = list(model.flow_target_columns)  # type: ignore[attr-defined]
    model_cfg = getattr(model, "gtv2_config", None)
    factorized_ast_mode = bool(getattr(model_cfg, "assist_share_factorized_ast", False))
    flow_label_columns_full = list(flow_label_columns)
    if bool(run_phase2_flow) and not flow_label_columns_full:
        raise RuntimeError("run_phase2_flow=True requires non-empty flow_label_columns")

    totals = {
        "total": 0.0,
        "minutes_mae": 0.0,
        "count_loss": 0.0,
        "member_loss": 0.0,
        "minutes_nll": 0.0,
        "minutes_hurdle_nll": 0.0,
        "role_loss": 0.0,
        "role_acc": 0.0,
        "starter_promotion_loss": 0.0,
        "sparse_starter_underpred_loss": 0.0,
        "bench_riser_underpred_loss": 0.0,
        "flow_nll": 0.0,
        "crps_fpts": 0.0,
        "team_energy": 0.0,
        "count_acc": 0.0,
        "poss_nll": 0.0,
        "poss_regression": 0.0,
        "backbone_nll": 0.0,
        "three_pa_nll": 0.0,
        "efficiency_nll": 0.0,
        "team_efficiency_ppp_aux": 0.0,
        "team_ppp_aux": 0.0,
        "team_advantage_aux": 0.0,
        "usage_share_nll": 0.0,
        "team_possession_aux": 0.0,
        "team_points_budget_aux": 0.0,
        "team_ast_budget_aux": 0.0,
        "assist_share_aux": 0.0,
        "assist_share_recon_aux": 0.0,
        "ast_blend_gate_aux": 0.0,
        "emergent_share_aux": 0.0,
        "ast_share_aux": 0.0,
        "reb_share_aux": 0.0,
        "ast_team_rate_aux": 0.0,
        "reb_opportunity_rate_aux": 0.0,
        "team_rebound_budget_rate_aux": 0.0,
        "rebound_budget_blend_gate_aux": 0.0,
        "spread_aux": 0.0,
        "total_aux": 0.0,
        "props_pts_aux": 0.0,
        "props_reb_aux": 0.0,
        "props_ast_aux": 0.0,
        "direct_pts_aux": 0.0,
        "direct_reb_aux": 0.0,
        "direct_ast_aux": 0.0,
        "direct_stl_aux": 0.0,
        "direct_blk_aux": 0.0,
        "direct_tov_aux": 0.0,
        "direct_boxscore_aux": 0.0,
        "direct_opportunity_aux": 0.0,
        "flow_anchor_nonast_aux": 0.0,
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
        player_features_flat = torch.cat([player_features[:, 0], player_features[:, 1]], dim=1)
        raw_lineup_starter_announced = _raw_feature_from_normalized(
            player_features_flat,
            feature_idx=int(lineup_starter_announced_idx),
            feature_mean=feature_mean_arr,
            feature_std=feature_std_arr,
        )
        raw_recent_start_pct_10 = _raw_feature_from_normalized(
            player_features_flat,
            feature_idx=int(recent_start_pct_10_idx),
            feature_mean=feature_mean_arr,
            feature_std=feature_std_arr,
        )
        raw_started_proxy_rate_prior_10 = _raw_feature_from_normalized(
            player_features_flat,
            feature_idx=int(started_proxy_rate_prior_10_idx),
            feature_mean=feature_mean_arr,
            feature_std=feature_std_arr,
        )
        raw_started_proxy_rate_prior_20 = _raw_feature_from_normalized(
            player_features_flat,
            feature_idx=int(started_proxy_rate_prior_20_idx),
            feature_mean=feature_mean_arr,
            feature_std=feature_std_arr,
        )
        raw_prior_minutes = _raw_feature_from_normalized(
            player_features_flat,
            feature_idx=int(minutes_from_stints_prior_20_idx),
            feature_mean=feature_mean_arr,
            feature_std=feature_std_arr,
        )
        raw_prior_play_prob = _raw_feature_from_normalized(
            player_features_flat,
            feature_idx=int(prior_play_prob_idx),
            feature_mean=feature_mean_arr,
            feature_std=feature_std_arr,
        )
        player_valid_mask = batch["player_valid_mask"].to(device=device)
        starter_hint_mask = batch["starter_force_active_worlds"].to(device=device)
        y_minutes = batch["y_minutes"].to(device=device)
        flow_targets_full = batch["flow_targets"].to(device=device)
        flow_observed_mask_full = batch["flow_observed_mask"].to(device=device)
        game_features = batch["game_features"].to(device=device)
        team_features = batch["team_features"].to(device=device)
        if int(flow_targets_full.shape[-1]) != int(len(flow_label_columns_full)):
            raise RuntimeError(
                "Batch flow label width does not match configured flow_label_columns: "
                f"shape[-1]={flow_targets_full.shape[-1]} columns={len(flow_label_columns_full)}"
            )
        flow_targets_model = select_flow_columns(
            flow_targets_full,
            source_columns=flow_label_columns_full,
            target_columns=flow_model_columns,
            fill_value=0.0,
        )
        flow_observed_model = select_flow_columns(
            flow_observed_mask_full,
            source_columns=flow_label_columns_full,
            target_columns=flow_model_columns,
            fill_value=False,
        ).to(dtype=torch.bool)
        if factorized_ast_mode:
            flow_targets_model, flow_observed_model = _mask_ast_from_flow_targets(
                flow_targets_model,
                flow_observed_model,
                flow_target_columns=flow_model_columns,
            )

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
        starter_hint_flat = _flatten_side(starter_hint_mask).to(dtype=torch.bool)
        starter_promotion_candidate_mask = starter_hint_mask.to(dtype=torch.bool)
        if raw_prior_minutes is not None:
            starter_promotion_candidate_mask = starter_promotion_candidate_mask & (
                raw_prior_minutes.view_as(starter_hint_flat).view_as(starter_hint_mask)
                <= float(starter_promotion_prior_minutes_max)
            )
        starter_promotion_candidate_flat = _flatten_side(starter_promotion_candidate_mask).to(dtype=torch.bool)
        hist_start_rate: torch.Tensor | None = None
        hist_start_sources = [
            x
            for x in (raw_recent_start_pct_10, raw_started_proxy_rate_prior_10, raw_started_proxy_rate_prior_20)
            if x is not None
        ]
        if hist_start_sources:
            hist_start_rate = hist_start_sources[0]
            for hist_source in hist_start_sources[1:]:
                hist_start_rate = torch.maximum(hist_start_rate, hist_source)
        if str(minutes_role_target_scheme) == "starter_context":
            role_targets = _build_minutes_role_targets_contextual(
                y_minutes=y_flat,
                valid_mask=valid_flat,
                active_threshold=float(active_threshold),
                lineup_starter_announced=raw_lineup_starter_announced,
                historical_start_rate=hist_start_rate,
                prior_minutes=raw_prior_minutes,
            )
        else:
            role_targets = build_minutes_role_targets(
                y_flat,
                valid_flat,
                team_index,
                active_threshold=float(active_threshold),
            )

        flow_mask_model = flow_observed_model
        if bool(run_phase2_flow):
            # No flow likelihood terms for DNP rows; only score rows with observed count labels.
            dnp_mask = (y_minutes > 0.0).unsqueeze(-1)
            flow_mask_model = flow_mask_model & player_valid_mask.unsqueeze(-1) & dnp_mask
        flow_target_flat = torch.cat([flow_targets_full[:, 0], flow_targets_full[:, 1]], dim=1)
        flow_observed_flat = torch.cat([flow_observed_mask_full[:, 0], flow_observed_mask_full[:, 1]], dim=1)

        with torch.set_grad_enabled(training):
            out = model(
                player_features,
                player_valid_mask,
                game_features=game_features,
                team_features=team_features,
                efficiency_sidecar_features=batch.get("efficiency_sidecar_features"),
                sample_active=False,
                active_temperature=1.0,
                target_active_mask=target_active_mask_2d,
                starter_hint_mask=starter_hint_mask,
                starter_promotion_candidate_mask=starter_promotion_candidate_mask,
                minutes_use_target_active=False,
                minutes_teacher_forcing_prob=float(minutes_teacher_forcing_prob),
                minutes_teacher_forcing_mode=str(minutes_teacher_forcing_mode),
                run_flow=bool(run_phase2_flow),
                flow_targets=flow_targets_model if bool(run_phase2_flow) else None,
                flow_observed_mask=flow_mask_model if bool(run_phase2_flow) else None,
                flow_minutes_target=y_minutes if bool(run_phase2_flow) else None,
                flow_minutes_teacher_forcing_prob=float(flow_minutes_teacher_forcing_prob),
                flow_minutes_teacher_forcing_mode=str(flow_minutes_teacher_forcing_mode),
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
            minutes_hurdle_nll = torch.zeros((), dtype=minutes_mae.dtype, device=minutes_mae.device)
            if float(w_minutes_hurdle_nll) > 0.0:
                if out.minutes.zero_logits is None or out.minutes.sigma is None:
                    raise RuntimeError(
                        "w_minutes_hurdle_nll > 0 requires minutes hurdle outputs "
                        "(enable with --enable-minutes-hurdle-head).",
                    )
                minutes_hurdle_nll = _minutes_hurdle_nll(
                    pred_minutes=out.minutes.minutes,
                    target_minutes=y_flat,
                    zero_logits=out.minutes.zero_logits,
                    sigma=out.minutes.sigma,
                    valid_mask=out.player_valid_mask,
                    zero_threshold=float(minutes_hurdle_zero_threshold),
                )
            role_loss = torch.zeros((), dtype=minutes_mae.dtype, device=minutes_mae.device)
            role_acc = torch.zeros((), dtype=minutes_mae.dtype, device=minutes_mae.device)
            if out.minutes.role_logits is not None:
                role_mask = out.player_valid_mask.to(dtype=torch.bool)
                role_logits = out.minutes.role_logits
                role_ce = F.cross_entropy(
                    role_logits.view(-1, role_logits.shape[-1]),
                    role_targets.view(-1),
                    reduction="none",
                ).view_as(role_targets)
                role_mask_f = role_mask.to(dtype=role_ce.dtype)
                role_loss = (role_ce * role_mask_f).sum() / role_mask_f.sum().clamp(min=1.0)
                role_pred = torch.argmax(role_logits, dim=-1)
                role_acc = (
                    ((role_pred == role_targets) & role_mask).to(dtype=minutes_mae.dtype).sum()
                    / role_mask.to(dtype=minutes_mae.dtype).sum().clamp(min=1.0)
                )
            starter_promotion_loss = torch.zeros((), dtype=minutes_mae.dtype, device=minutes_mae.device)
            if float(w_starter_promotion_loss) > 0.0:
                if out.minutes.starter_promotion_delta is None:
                    raise RuntimeError(
                        "w_starter_promotion_loss > 0 requires starter promotion outputs "
                        "(enable with --enable-starter-promotion-head).",
                    )
                promo_mask = starter_promotion_candidate_flat & out.player_valid_mask.to(dtype=torch.bool)
                if raw_prior_minutes is not None:
                    promo_target = torch.relu(y_flat - raw_prior_minutes)
                else:
                    promo_target = torch.relu(y_flat)
                if bool(promo_mask.any()):
                    promo_err = F.huber_loss(
                        out.minutes.starter_promotion_delta,
                        promo_target,
                        reduction="none",
                        delta=4.0,
                    )
                    promo_mask_f = promo_mask.to(dtype=promo_err.dtype)
                    starter_promotion_loss = (promo_err * promo_mask_f).sum() / promo_mask_f.sum().clamp(min=1.0)
            sparse_starter_underpred_loss = torch.zeros((), dtype=minutes_mae.dtype, device=minutes_mae.device)
            if float(w_sparse_starter_underpred_loss) > 0.0:
                sparse_mask = out.player_valid_mask.to(dtype=torch.bool)
                if raw_lineup_starter_announced is not None:
                    sparse_mask = sparse_mask & (raw_lineup_starter_announced >= 0.5)
                else:
                    sparse_mask = sparse_mask & torch.zeros_like(sparse_mask)
                if raw_prior_play_prob is not None and float(sparse_starter_loss_prior_play_prob_max) < 1.0:
                    sparse_mask = sparse_mask & (raw_prior_play_prob <= float(sparse_starter_loss_prior_play_prob_max))
                if raw_prior_minutes is not None:
                    sparse_mask = sparse_mask & (raw_prior_minutes <= float(sparse_starter_loss_prior_minutes_max))
                if hist_start_rate is not None:
                    sparse_mask = sparse_mask & (hist_start_rate <= float(sparse_starter_loss_hist_start_rate_max))
                sparse_mask = sparse_mask & (y_flat >= float(sparse_starter_loss_actual_min_threshold))
                if bool(sparse_mask.any()):
                    underpred = torch.relu(y_flat - out.minutes.minutes)
                    sparse_mask_f = sparse_mask.to(dtype=underpred.dtype)
                    sparse_starter_underpred_loss = (underpred * sparse_mask_f).sum() / sparse_mask_f.sum().clamp(min=1.0)
            bench_riser_underpred_loss = torch.zeros((), dtype=minutes_mae.dtype, device=minutes_mae.device)
            if float(w_bench_riser_underpred_loss) > 0.0:
                bench_mask = out.player_valid_mask.to(dtype=torch.bool) & (~starter_hint_flat)
                if raw_prior_minutes is not None:
                    bench_mask = bench_mask & (raw_prior_minutes >= float(bench_riser_loss_prior_minutes_min))
                if raw_prior_play_prob is not None:
                    bench_mask = bench_mask & (raw_prior_play_prob >= float(bench_riser_loss_prior_play_prob_min))
                if hist_start_rate is not None:
                    bench_mask = bench_mask & (hist_start_rate <= float(bench_riser_loss_hist_start_rate_max))
                bench_mask = bench_mask & (y_flat >= float(bench_riser_loss_actual_min_threshold))
                if bool(bench_mask.any()):
                    bench_underpred = torch.relu(y_flat - out.minutes.minutes)
                    bench_mask_f = bench_mask.to(dtype=bench_underpred.dtype)
                    bench_riser_underpred_loss = (bench_underpred * bench_mask_f).sum() / bench_mask_f.sum().clamp(min=1.0)
            if bool(run_phase2_flow):
                if out.flow is None:
                    raise RuntimeError("run_phase2_flow=True but model did not return flow outputs")
                flow_nll = out.flow.nll_mean
            else:
                flow_nll = torch.zeros((), dtype=minutes_mae.dtype, device=minutes_mae.device)

            if bool(run_phase3_decision):
                if not bool(run_phase2_flow):
                    raise RuntimeError("run_phase3_decision=True requires run_phase2_flow=True")
                sampled_fpts = _sample_decision_fpts(
                    model,
                    context_out=out,
                    num_samples=int(phase3_num_samples),
                    active_temperature=float(phase3_active_temperature),
                    flow_contract_columns=flow_label_columns_full,
                )
                if bool(phase3_stop_grad):
                    sampled_fpts = sampled_fpts.detach()
                target_flow = _project_flow_stats_to_contract(
                    flow_target_flat,
                    flow_target_columns=flow_label_columns_full,
                    flow_contract_columns=flow_label_columns_full,
                )
                target_fpts = _compute_dk_fpts_from_flow(
                    target_flow,
                    flow_target_columns=flow_label_columns_full,
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
            team_possession_aux_loss = torch.zeros((), dtype=minutes_mae.dtype, device=minutes_mae.device)
            backbone_nll_loss = torch.zeros((), dtype=minutes_mae.dtype, device=minutes_mae.device)
            three_pa_nll_loss = torch.zeros((), dtype=minutes_mae.dtype, device=minutes_mae.device)
            team_efficiency_ppp_aux_loss = torch.zeros((), dtype=minutes_mae.dtype, device=minutes_mae.device)
            team_ppp_aux_loss = torch.zeros((), dtype=minutes_mae.dtype, device=minutes_mae.device)
            team_advantage_aux_loss = torch.zeros((), dtype=minutes_mae.dtype, device=minutes_mae.device)
            if bool(enable_possession_backbone) and out.possession is not None and out.backbone is not None:
                # Extract team-level truth counts from flow_targets (B, 2, 15, S)
                # Sum across players per team to get team totals
                ftc = flow_label_columns_full
                required_cols = ["fga2", "fg2m", "fga3", "fg3m", "fta", "ftm", "oreb", "tov"]
                missing_required_cols = [name for name in required_cols if name not in ftc]
                if missing_required_cols:
                    raise RuntimeError(
                        f"Missing required flow target columns for possession backbone: {missing_required_cols}",
                    )
                fga2_i = _flow_index(ftc, "fga2")
                fg2m_i = _flow_index(ftc, "fg2m")
                fga3_i = _flow_index(ftc, "fga3")
                fg3m_i = _flow_index(ftc, "fg3m")
                fta_i = _flow_index(ftc, "fta")
                ftm_i = _flow_index(ftc, "ftm")
                oreb_i = _flow_index(ftc, "oreb")
                tov_i = _flow_index(ftc, "tov")
                max_required_idx = max(fga2_i, fg2m_i, fga3_i, fg3m_i, fta_i, ftm_i, oreb_i, tov_i)
                if flow_targets_full.ndim != 4 or int(flow_targets_full.shape[-1]) <= int(max_required_idx):
                    raise RuntimeError(
                        "Possession backbone requires populated flow_targets stats. "
                        f"Got flow_targets shape={tuple(flow_targets_full.shape)} with required index={max_required_idx}. "
                        "Use --enable-phase2-flow so labels_boxscore_counts are loaded.",
                    )

                # flow_targets is (B, 2, 15, S); valid players mask is player_valid_mask (B, 2, 15)
                ft = flow_targets_full  # (B, 2, 15, S)
                vm = player_valid_mask.unsqueeze(-1).to(dtype=ft.dtype)  # (B, 2, 15, 1)
                ft_masked = ft * vm

                # Team sums: (B, 2)
                fga_team = ft_masked[:, :, :, fga2_i].sum(dim=2) + ft_masked[:, :, :, fga3_i].sum(dim=2)
                fta_team = ft_masked[:, :, :, fta_i].sum(dim=2)
                oreb_team = ft_masked[:, :, :, oreb_i].sum(dim=2)
                tov_team = ft_masked[:, :, :, tov_i].sum(dim=2)

                # Only compute losses where flow labels are observed
                flow_obs_any = flow_observed_mask_full.any(dim=-1)  # (B, 2, 15) -> any stat observed
                team_has_labels = flow_obs_any.any(dim=2)  # (B, 2) -> team has at least one observed player
                game_has_labels = team_has_labels.all(dim=1)  # (B,) -> both teams have labels

                if game_has_labels.any():
                    backbone_env_context = None
                    if bool(getattr(model, "enable_env_side_channel", False)):  # type: ignore[attr-defined]
                        backbone_env_context = model._build_env_side_channel_context(  # type: ignore[attr-defined]
                            player_features=player_features,
                            player_valid_mask=player_valid_mask,
                            game_features=game_features,
                        )
                    if backbone_env_context is None:
                        backbone_env_context = model._build_backbone_environment_context(  # type: ignore[attr-defined]
                            player_features=player_features,
                            player_valid_mask=player_valid_mask,
                            game_features=game_features,
                        )
                    backbone_team_context = None
                    if bool(getattr(model, "team_ppp_direct_backbone_context", False)) and getattr(out, "team_ppp", None) is not None:
                        backbone_team_context = model._build_team_ppp_context(out.team_ppp.team_ppp)  # type: ignore[attr-defined]
                    backbone_advantage_context = None
                    if bool(getattr(model, "team_advantage_direct_backbone_context", False)) and getattr(out, "team_advantage", None) is not None:
                        backbone_advantage_context = model._build_team_advantage_context(out.team_advantage.mu)  # type: ignore[attr-defined]
                    # Possession truth
                    poss_true = compute_possession_truth(fga_team, oreb_team, tov_team, fta_team)  # (B,)
                    poss_true_team = compute_possession_truth_per_team(fga_team, oreb_team, tov_team, fta_team)
                    true_team_pts = (
                        2.0 * ft_masked[:, :, :, fg2m_i].sum(dim=2)
                        + 3.0 * ft_masked[:, :, :, fg3m_i].sum(dim=2)
                        + ft_masked[:, :, :, ftm_i].sum(dim=2)
                    )
                    true_team_ppp = true_team_pts / poss_true_team.clamp(min=1.0)
                    true_team_margin = true_team_pts[:, 0] - true_team_pts[:, 1]
                    if float(w_team_ppp_aux) > 0.0:
                        if getattr(out, "team_ppp", None) is None:
                            raise RuntimeError("w_team_ppp_aux > 0 requires model.enable_team_ppp_head")
                        team_ppp_aux_loss = _masked_scaled_huber_loss(
                            pred=out.team_ppp.team_ppp,
                            target=true_team_ppp,
                            mask=team_has_labels,
                            scale=float(team_ppp_target_scale),
                            delta=float(direct_stat_aux_huber_delta),
                        )
                    if float(w_team_advantage_aux) > 0.0:
                        if getattr(out, "team_advantage", None) is None:
                            raise RuntimeError("w_team_advantage_aux > 0 requires model.enable_team_advantage_head")
                        team_advantage_aux_loss = _masked_scaled_huber_loss(
                            pred=out.team_advantage.mu,
                            target=true_team_margin,
                            mask=game_has_labels,
                            scale=float(team_advantage_target_scale),
                            delta=float(direct_stat_aux_huber_delta),
                        )
                    if bool(enable_efficiency_head) and out.efficiency is not None and float(w_team_efficiency_ppp_aux) > 0.0:
                        pred_team_pts = torch.zeros_like(poss_true_team)
                        for side in (0, 1):
                            side_mask_f = (out.player_team_index.eq(side) & out.player_valid_mask).to(
                                dtype=flow_target_flat.dtype
                            )
                            pred_team_pts[:, side] = (
                                2.0 * (out.efficiency.mean_fg2 * flow_target_flat[..., fga2_i] * side_mask_f).sum(dim=1)
                                + 3.0 * (out.efficiency.mean_fg3 * flow_target_flat[..., fga3_i] * side_mask_f).sum(dim=1)
                                + (out.efficiency.mean_ft * flow_target_flat[..., fta_i] * side_mask_f).sum(dim=1)
                            )
                        pred_team_ppp = pred_team_pts / poss_true_team.clamp(min=1.0)
                        team_efficiency_ppp_aux_loss = _masked_scaled_huber_loss(
                            pred=pred_team_ppp,
                            target=true_team_ppp,
                            mask=team_has_labels,
                            scale=float(team_efficiency_ppp_target_scale),
                            delta=float(direct_stat_aux_huber_delta),
                        )
                    # Possession NLL (only on games with labels)
                    from projections.rotation.possession_backbone import PossessionHead
                    poss_nll_per_game = PossessionHead.nll_student_t(
                        poss_true, out.possession.mu, out.possession.sigma, out.possession.df,
                    )
                    poss_nll_loss = (poss_nll_per_game * game_has_labels.to(dtype=poss_nll_per_game.dtype)).sum() / game_has_labels.to(dtype=poss_nll_per_game.dtype).sum().clamp(min=1.0)
                    if float(w_team_possession_aux) > 0.0 and getattr(out.possession, "team_poss", None) is not None:
                        team_possession_aux_loss = _masked_scaled_huber_loss(
                            pred=out.possession.team_poss,
                            target=poss_true_team,
                            mask=team_has_labels,
                            scale=4.0,
                            delta=1.0,
                        )

                    # Backbone rate NLL
                    backbone_nll_loss = model.event_backbone.nll_rates(  # type: ignore[attr-defined]
                        out.team_states,
                        out.game_state,
                        poss_true_team if getattr(out.possession, "team_poss", None) is not None else poss_true,
                        fta_true=fta_team,
                        tov_true=tov_team,
                        oreb_true=oreb_team,
                        game_features=backbone_env_context,
                        team_context=backbone_team_context,
                        advantage_context=backbone_advantage_context,
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
                            game_features=backbone_env_context,
                            team_context=backbone_team_context,
                            advantage_context=backbone_advantage_context,
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
                ftc = flow_label_columns_full
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
                if flow_targets_full.ndim != 4 or int(flow_targets_full.shape[-1]) <= int(max_required_idx):
                    raise RuntimeError(
                        "Efficiency head loss requires populated flow_targets stats. "
                        f"Got flow_targets shape={tuple(flow_targets_full.shape)} with required index={max_required_idx}. "
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

            def _raw_player_feature(feature_idx: int) -> torch.Tensor | None:
                return _raw_feature_from_normalized(
                    player_features_flat,
                    feature_idx=int(feature_idx),
                    feature_mean=feature_mean_arr,
                    feature_std=feature_std_arr,
                )

            usage_share_nll_loss = torch.zeros((), dtype=minutes_mae.dtype, device=minutes_mae.device)
            team_points_budget_aux_loss = torch.zeros((), dtype=minutes_mae.dtype, device=minutes_mae.device)
            team_ast_budget_aux_loss = torch.zeros((), dtype=minutes_mae.dtype, device=minutes_mae.device)
            assist_share_aux_loss = torch.zeros((), dtype=minutes_mae.dtype, device=minutes_mae.device)
            assist_share_recon_aux_loss = torch.zeros((), dtype=minutes_mae.dtype, device=minutes_mae.device)
            ast_blend_gate_aux_loss = torch.zeros((), dtype=minutes_mae.dtype, device=minutes_mae.device)
            emergent_share_aux_loss = torch.zeros((), dtype=minutes_mae.dtype, device=minutes_mae.device)
            ast_share_aux_loss = torch.zeros((), dtype=minutes_mae.dtype, device=minutes_mae.device)
            reb_share_aux_loss = torch.zeros((), dtype=minutes_mae.dtype, device=minutes_mae.device)
            ast_team_rate_aux_loss = torch.zeros((), dtype=minutes_mae.dtype, device=minutes_mae.device)
            reb_opportunity_rate_aux_loss = torch.zeros((), dtype=minutes_mae.dtype, device=minutes_mae.device)
            team_rebound_budget_rate_aux_loss = torch.zeros((), dtype=minutes_mae.dtype, device=minutes_mae.device)
            rebound_budget_blend_gate_aux_loss = torch.zeros((), dtype=minutes_mae.dtype, device=minutes_mae.device)
            spread_aux_loss = torch.zeros((), dtype=minutes_mae.dtype, device=minutes_mae.device)
            total_aux_loss = torch.zeros((), dtype=minutes_mae.dtype, device=minutes_mae.device)
            props_pts_aux_loss = torch.zeros((), dtype=minutes_mae.dtype, device=minutes_mae.device)
            props_reb_aux_loss = torch.zeros((), dtype=minutes_mae.dtype, device=minutes_mae.device)
            props_ast_aux_loss = torch.zeros((), dtype=minutes_mae.dtype, device=minutes_mae.device)
            direct_pts_aux_loss = torch.zeros((), dtype=minutes_mae.dtype, device=minutes_mae.device)
            direct_reb_aux_loss = torch.zeros((), dtype=minutes_mae.dtype, device=minutes_mae.device)
            direct_ast_aux_loss = torch.zeros((), dtype=minutes_mae.dtype, device=minutes_mae.device)
            direct_stl_aux_loss = torch.zeros((), dtype=minutes_mae.dtype, device=minutes_mae.device)
            direct_blk_aux_loss = torch.zeros((), dtype=minutes_mae.dtype, device=minutes_mae.device)
            direct_tov_aux_loss = torch.zeros((), dtype=minutes_mae.dtype, device=minutes_mae.device)
            direct_boxscore_aux_loss = torch.zeros((), dtype=minutes_mae.dtype, device=minutes_mae.device)
            direct_opportunity_aux_loss = torch.zeros((), dtype=minutes_mae.dtype, device=minutes_mae.device)
            flow_anchor_nonast_aux_loss = torch.zeros((), dtype=minutes_mae.dtype, device=minutes_mae.device)
            if float(w_team_points_budget_aux) > 0.0:
                if int(vegas_total_idx) < 0 or int(vegas_spread_idx) < 0:
                    raise RuntimeError(
                        "w_team_points_budget_aux > 0 requires vegas_total and vegas_spread in --game-feature-cols",
                    )
                if getattr(out, "team_points_budget", None) is None:
                    raise RuntimeError(
                        "w_team_points_budget_aux > 0 requires model.enable_team_points_budget_head",
                    )
                target_total = game_features[:, int(vegas_total_idx)]
                target_spread = game_features[:, int(vegas_spread_idx)]
                target_home = 0.5 * (target_total - target_spread)
                target_away = 0.5 * (target_total + target_spread)
                target_team_points = torch.stack([target_home, target_away], dim=1)
                team_points_mask = torch.ones_like(target_team_points, dtype=torch.bool)
                if int(vegas_total_missing_idx) >= 0:
                    total_mask = (game_features[:, int(vegas_total_missing_idx)] < 0.5).unsqueeze(1)
                    team_points_mask = team_points_mask & total_mask
                if int(vegas_spread_missing_idx) >= 0:
                    spread_mask = (game_features[:, int(vegas_spread_missing_idx)] < 0.5).unsqueeze(1)
                    team_points_mask = team_points_mask & spread_mask
                team_points_budget_aux_loss = _masked_scaled_huber_loss(
                    pred=out.team_points_budget.team_points,
                    target=target_team_points,
                    mask=team_points_mask,
                    scale=float(total_aux_target_scale),
                    delta=float(total_aux_huber_delta),
                )
            if bool(run_phase2_flow):
                ftc = flow_label_columns_full
                fga2_i = _flow_index(ftc, "fga2")
                fg2m_i = _flow_index(ftc, "fg2m")
                fga3_i = _flow_index(ftc, "fga3")
                fg3m_i = _flow_index(ftc, "fg3m")
                fta_i = _flow_index(ftc, "fta")
                ftm_i = _flow_index(ftc, "ftm")
                oreb_i = _flow_index(ftc, "oreb")
                dreb_i = _flow_index(ftc, "dreb")
                ast_i = _flow_index(ftc, "ast")
                stl_i = _flow_index(ftc, "stl")
                blk_i = _flow_index(ftc, "blk")
                tov_i = _flow_index(ftc, "tov")

                fga_true = flow_target_flat[..., fga2_i] + flow_target_flat[..., fga3_i]
                fgm_true = flow_target_flat[..., fg2m_i] + flow_target_flat[..., fg3m_i]
                missed_fg_true = torch.clamp(fga_true - fgm_true, min=0.0)
                fta_true = flow_target_flat[..., fta_i]
                oreb_true = flow_target_flat[..., oreb_i]
                dreb_true = flow_target_flat[..., dreb_i]
                ast_true = flow_target_flat[..., ast_i]
                stl_true = flow_target_flat[..., stl_i]
                blk_true = flow_target_flat[..., blk_i]
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
                stl_obs = flow_observed_flat[..., stl_i]
                blk_obs = flow_observed_flat[..., blk_i]
                tov_obs = flow_observed_flat[..., tov_i]
                ast_line_raw = _raw_player_feature(int(an_ast_line_idx)) if int(an_ast_line_idx) >= 0 else None
                prior_play_prob_raw = _raw_player_feature(int(prior_play_prob_idx)) if int(prior_play_prob_idx) >= 0 else None
                ast_player_weights = _playmaker_importance_weights(
                    line_raw=ast_line_raw,
                    prior_play_prob_raw=prior_play_prob_raw,
                    line_center=float(assist_playmaker_line_center),
                    line_scale=float(assist_playmaker_line_scale),
                    max_weight=float(assist_playmaker_max_weight),
                )

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

                ast_true_team, ast_true_seen = _team_sum_by_side(
                    values=ast_true,
                    valid_mask=out.player_valid_mask,
                    team_index=out.player_team_index,
                    observed_mask=ast_obs,
                )
                if float(w_team_ast_budget_aux) > 0.0 and out.team_ast_budget is not None:
                    team_ast_team_weights = None
                    if ast_player_weights is not None:
                        team_weight_parts: list[torch.Tensor] = []
                        for side in (0, 1):
                            side_mask = out.player_valid_mask & out.player_team_index.eq(side)
                            side_weight = torch.where(side_mask, ast_player_weights, torch.zeros_like(ast_player_weights)).amax(dim=1)
                            team_weight_parts.append(side_weight)
                        team_ast_team_weights = torch.stack(team_weight_parts, dim=1)
                    team_ast_budget_aux_loss = _asymmetric_weighted_masked_scaled_huber_loss(
                        pred=out.team_ast_budget.team_ast,
                        target=ast_true_team,
                        mask=ast_true_seen,
                        scale=float(team_ast_budget_target_scale),
                        delta=float(direct_stat_aux_huber_delta),
                        weights=team_ast_team_weights,
                        underprediction_weight=float(assist_underprediction_weight),
                    )
                if float(w_assist_share_aux) > 0.0 and out.assist_share is not None:
                    assist_share_aux_loss = _team_share_ce_loss(
                        logits=out.assist_share.ast_logits,
                        attempts_true=ast_true,
                        valid_mask=out.player_valid_mask,
                        team_index=out.player_team_index,
                        observed_mask=ast_obs,
                        target_weights=ast_player_weights,
                    )
                if (
                    float(w_assist_share_recon_aux) > 0.0
                    and out.assist_share is not None
                    and out.team_ast_budget is not None
                ):
                    ast_recon = _reconstruct_ast_from_heads(
                        player_valid_mask=out.player_valid_mask,
                        player_team_index=out.player_team_index,
                        team_ast_budget=out.team_ast_budget.team_ast,
                        assist_share_logits=out.assist_share.ast_logits,
                    )
                    if ast_recon is None:
                        raise RuntimeError("AST reconstruction unexpectedly missing despite enabled AST heads")
                    assist_share_recon_aux_loss = _asymmetric_weighted_masked_scaled_huber_loss(
                        pred=ast_recon,
                        target=ast_true,
                        mask=out.player_valid_mask & ast_obs,
                        scale=float(assist_share_recon_target_scale),
                        delta=float(direct_stat_aux_huber_delta),
                        weights=ast_player_weights,
                        underprediction_weight=float(assist_underprediction_weight),
                    )

                if (
                    float(w_ast_blend_gate_aux) > 0.0
                    or float(w_emergent_share_aux) > 0.0
                    or float(w_ast_share_aux) > 0.0
                    or float(w_reb_share_aux) > 0.0
                    or float(w_ast_team_rate_aux) > 0.0
                    or float(w_reb_opportunity_rate_aux) > 0.0
                    or float(w_team_rebound_budget_rate_aux) > 0.0
                    or float(w_spread_aux) > 0.0
                    or float(w_total_aux) > 0.0
                    or float(w_props_pts_aux) > 0.0
                    or float(w_props_reb_aux) > 0.0
                    or float(w_props_ast_aux) > 0.0
                    or float(w_direct_pts_aux) > 0.0
                    or float(w_direct_reb_aux) > 0.0
                    or float(w_direct_ast_aux) > 0.0
                    or float(w_direct_stl_aux) > 0.0
                    or float(w_direct_blk_aux) > 0.0
                    or float(w_direct_tov_aux) > 0.0
                    or float(w_direct_boxscore_aux) > 0.0
                    or float(w_direct_opportunity_aux) > 0.0
                ):
                    model_cfg = getattr(model, "gtv2_config", None)
                    replace_flow_ast = bool(getattr(model_cfg, "assist_share_replace_flow_ast", False)) or bool(
                        getattr(model_cfg, "assist_share_factorized_ast", False)
                    )
                    reconcile_ast_budget = bool(getattr(model_cfg, "assist_share_reconcile_ast_budget", False))
                    reconcile_rebound_budget = bool(getattr(model_cfg, "rebound_factor_reconcile_oreb_dreb", False))
                    z0 = torch.zeros(
                        (
                            out.player_states.shape[0],
                            out.player_states.shape[1],
                            len(flow_model_columns),
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
                        minutes_context=out.minutes.minutes,
                        env_context=getattr(out, "env_context", None),
                    )
                    emergent_flow_projected = _project_flow_stats_to_contract(
                        emergent_flow,
                        flow_target_columns=flow_model_columns,
                        flow_contract_columns=ftc,
                        fg2_rate=out.efficiency.mean_fg2 if out.efficiency is not None else None,
                        fg3_rate=out.efficiency.mean_fg3 if out.efficiency is not None else None,
                        ft_rate=out.efficiency.mean_ft if out.efficiency is not None else None,
                    )
                    ast_override = None
                    factorized_ast = None
                    if replace_flow_ast:
                        factorized_ast = _reconstruct_ast_from_heads(
                            player_valid_mask=out.player_valid_mask,
                            player_team_index=out.player_team_index,
                            team_ast_budget=out.team_ast_budget.team_ast if out.team_ast_budget is not None else None,
                            assist_share_logits=out.assist_share.ast_logits if out.assist_share is not None else None,
                        )
                        ast_override = _build_ast_override(
                            flow_projected_base=emergent_flow_projected,
                            flow_contract_columns=ftc,
                            player_valid_mask=out.player_valid_mask,
                            player_team_index=out.player_team_index,
                            team_ast_budget=out.team_ast_budget.team_ast if out.team_ast_budget is not None else None,
                            assist_share_logits=out.assist_share.ast_logits if out.assist_share is not None else None,
                            ast_blend_gate=out.ast_blend_gate.gate if getattr(out, "ast_blend_gate", None) is not None else None,
                        )
                    if (
                        float(w_ast_blend_gate_aux) > 0.0
                        and getattr(out, "ast_blend_gate", None) is not None
                        and factorized_ast is not None
                    ):
                        flow_ast_base = emergent_flow_projected[..., ast_i]
                        gate_target, gate_mask = _ast_blend_gate_targets(
                            flow_ast=flow_ast_base,
                            factorized_ast=factorized_ast,
                            target_ast=ast_true,
                            valid_mask=out.player_valid_mask & ast_obs,
                            eps=float(ast_blend_gate_target_eps),
                        )
                        if bool(gate_mask.any()):
                            gate_weights = (
                                ast_player_weights.to(dtype=flow_ast_base.dtype)
                                if ast_player_weights is not None
                                else torch.ones_like(flow_ast_base)
                            )
                            gate_loss_per = F.binary_cross_entropy_with_logits(
                                out.ast_blend_gate.gate_logits,
                                gate_target.to(dtype=out.ast_blend_gate.gate_logits.dtype),
                                reduction="none",
                            )
                            gate_mask_f = gate_mask.to(dtype=gate_loss_per.dtype)
                            ast_blend_gate_aux_loss = (
                                gate_loss_per * gate_weights * gate_mask_f
                            ).sum() / (gate_weights * gate_mask_f).sum().clamp(min=1.0)
                    emergent_flow = _project_flow_stats_to_contract(
                        emergent_flow,
                        flow_target_columns=flow_model_columns,
                        flow_contract_columns=ftc,
                        fg2_rate=out.efficiency.mean_fg2 if out.efficiency is not None else None,
                        fg3_rate=out.efficiency.mean_fg3 if out.efficiency is not None else None,
                        ft_rate=out.efficiency.mean_ft if out.efficiency is not None else None,
                        ast_override=ast_override,
                    )
                    if reconcile_ast_budget:
                        emergent_flow = _reconcile_ast_to_team_budget(
                            flow_projected=emergent_flow,
                            flow_contract_columns=ftc,
                            player_valid_mask=out.player_valid_mask,
                            player_team_index=out.player_team_index,
                            active_mask=out.active.active_mask,
                            team_ast_budget=(
                                out.team_ast_budget.team_ast if out.team_ast_budget is not None else None
                            ),
                            assist_share_logits=(
                                out.assist_share.ast_logits if out.assist_share is not None else None
                            ),
                            share_alpha=float(getattr(model_cfg, "assist_share_reconcile_alpha", 0.75)),
                            share_temperature=float(
                                getattr(model_cfg, "assist_share_reconcile_temperature", 0.85)
                            ),
                        )
                    rebound_flow_oreb_team_base = None
                    rebound_flow_dreb_team_base = None
                    rebound_target_oreb_budget = None
                    rebound_target_dreb_budget = None
                    rebound_opp_missed_team_base = None
                    if getattr(out, "team_rebound_budget", None) is not None:
                        rebound_oreb_base = emergent_flow[..., oreb_i]
                        rebound_dreb_base = emergent_flow[..., dreb_i]
                        rebound_missed_fg_base = (
                            emergent_flow[..., fga2_i]
                            - emergent_flow[..., fg2m_i]
                            + emergent_flow[..., fga3_i]
                            - emergent_flow[..., fg3m_i]
                        ).clamp(min=0.0)
                        rebound_own_missed_team_base, _ = _team_sum_by_side(
                            values=rebound_missed_fg_base,
                            valid_mask=out.player_valid_mask,
                            team_index=out.player_team_index,
                            observed_mask=out.player_valid_mask,
                        )
                        rebound_opp_missed_team_base = _swap_team_side_values(rebound_own_missed_team_base)
                        rebound_flow_oreb_team_base, _ = _team_sum_by_side(
                            values=rebound_oreb_base,
                            valid_mask=out.player_valid_mask,
                            team_index=out.player_team_index,
                            observed_mask=out.player_valid_mask,
                        )
                        rebound_flow_dreb_team_base, _ = _team_sum_by_side(
                            values=rebound_dreb_base,
                            valid_mask=out.player_valid_mask,
                            team_index=out.player_team_index,
                            observed_mask=out.player_valid_mask,
                        )
                        rebound_budget_mode = str(
                            getattr(model_cfg, "rebound_budget_parameterization", "absolute")
                        ).strip().lower()
                        if _rebound_budget_uses_rate(rebound_budget_mode, "oreb"):
                            rebound_target_oreb_budget = (
                                out.team_rebound_budget.team_oreb.clamp(min=0.0, max=1.0) * rebound_own_missed_team_base
                            )
                        else:
                            rebound_target_oreb_budget = torch.minimum(
                                out.team_rebound_budget.team_oreb.clamp(min=0.0),
                                rebound_own_missed_team_base,
                            )
                        if _rebound_budget_is_residual_rate(rebound_budget_mode, "dreb"):
                            rebound_flow_dreb_rate_base = torch.nan_to_num(
                                rebound_flow_dreb_team_base / rebound_opp_missed_team_base.clamp(min=1.0),
                                nan=0.0,
                                posinf=1.0,
                                neginf=0.0,
                            )
                            rebound_target_dreb_budget = torch.clamp(
                                rebound_flow_dreb_rate_base + out.team_rebound_budget.team_dreb,
                                min=0.0,
                                max=1.0,
                            ) * rebound_opp_missed_team_base
                        elif _rebound_budget_uses_rate(rebound_budget_mode, "dreb"):
                            rebound_target_dreb_budget = (
                                out.team_rebound_budget.team_dreb.clamp(min=0.0, max=1.0) * rebound_opp_missed_team_base
                            )
                        else:
                            rebound_target_dreb_budget = torch.minimum(
                                out.team_rebound_budget.team_dreb.clamp(min=0.0),
                                rebound_opp_missed_team_base,
                            )
                    if reconcile_rebound_budget:
                        emergent_flow = _reconcile_rebounds_to_opportunity_budgets(
                            flow_projected=emergent_flow,
                            flow_contract_columns=ftc,
                            player_valid_mask=out.player_valid_mask,
                            player_team_index=out.player_team_index,
                            active_mask=out.active.active_mask,
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
                            share_alpha=float(getattr(model_cfg, "rebound_factor_reconcile_alpha", 0.50)),
                            share_temperature=float(getattr(model_cfg, "rebound_factor_reconcile_temperature", 0.90)),
                            reconcile_mode=str(getattr(model_cfg, "rebound_factor_reconcile_mode", "both")),
                            budget_parameterization=str(
                                getattr(model_cfg, "rebound_budget_parameterization", "absolute")
                            ),
                            dreb_deterministic_discount=float(
                                getattr(model_cfg, "rebound_dreb_deterministic_discount", 1.0)
                            ),
                            oreb_reconcile_use_flow_budget=bool(
                                getattr(model_cfg, "rebound_oreb_reconcile_use_flow_budget", False)
                            ),
                            oreb_budget_blend_alpha=float(
                                getattr(model_cfg, "rebound_oreb_budget_blend_alpha", 1.0)
                            ),
                            dreb_budget_blend_alpha=float(
                                getattr(model_cfg, "rebound_dreb_budget_blend_alpha", 1.0)
                            ),
                        )
                    if bool(getattr(model_cfg, "team_opportunity_reconcile_budget", False)):
                        resolved_team_opportunity_share = _resolve_team_opportunity_share(
                            model_cfg=model_cfg,
                            game_features=game_features,
                        )
                        emergent_flow = _reconcile_opportunities_to_team_budget(
                            flow_projected=emergent_flow,
                            flow_contract_columns=ftc,
                            player_valid_mask=out.player_valid_mask,
                            player_team_index=out.player_team_index,
                            active_mask=out.active.active_mask,
                            team_opportunity_share=resolved_team_opportunity_share,
                            budget_alpha=float(getattr(model_cfg, "team_opportunity_reconcile_alpha", 1.0)),
                            preserve_possessions=bool(
                                getattr(model_cfg, "team_opportunity_reconcile_preserve_possessions", False)
                            ),
                        )
                    if bool(getattr(model_cfg, "team_points_reconcile_budget", False)):
                        resolved_team_points_budget = _resolve_team_points_budget(
                            model_cfg=model_cfg,
                            game_features=game_features,
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
                                else out.possession.mu
                            ),
                        )
                        emergent_flow = _reconcile_points_to_team_budget(
                            flow_projected=emergent_flow,
                            flow_contract_columns=ftc,
                            player_valid_mask=out.player_valid_mask,
                            player_team_index=out.player_team_index,
                            active_mask=out.active.active_mask,
                            team_points_budget=resolved_team_points_budget,
                            budget_alpha=float(getattr(model_cfg, "team_points_reconcile_alpha", 1.0)),
                        )
                    if (
                        float(w_flow_anchor_nonast_aux) > 0.0
                        and flow_anchor_teacher_model is not None
                    ):
                        with torch.no_grad():
                            teacher_out = flow_anchor_teacher_model(
                                player_features,
                                player_valid_mask,
                                game_features=game_features,
                                team_features=team_features,
                                efficiency_sidecar_features=batch.get("efficiency_sidecar_features"),
                                sample_active=False,
                                active_temperature=1.0,
                                target_active_mask=target_active_mask_2d,
                                starter_hint_mask=starter_hint_mask,
                                starter_promotion_candidate_mask=starter_promotion_candidate_mask,
                                minutes_use_target_active=False,
                                minutes_teacher_forcing_prob=float(minutes_teacher_forcing_prob),
                                minutes_teacher_forcing_mode=str(minutes_teacher_forcing_mode),
                                run_flow=True,
                                flow_targets=flow_targets_model,
                                flow_observed_mask=flow_mask_model,
                                flow_minutes_target=y_minutes,
                                flow_minutes_teacher_forcing_prob=float(flow_minutes_teacher_forcing_prob),
                                flow_minutes_teacher_forcing_mode=str(flow_minutes_teacher_forcing_mode),
                                detach_backbone=bool(detach_backbone),
                            )
                            teacher_cfg = getattr(flow_anchor_teacher_model, "gtv2_config", None)
                            teacher_flow_columns: list[str] = list(
                                getattr(flow_anchor_teacher_model, "flow_target_columns")
                            )
                            teacher_z0 = torch.zeros(
                                (
                                    teacher_out.player_states.shape[0],
                                    teacher_out.player_states.shape[1],
                                    len(teacher_flow_columns),
                                ),
                                dtype=teacher_out.player_states.dtype,
                                device=teacher_out.player_states.device,
                            )
                            teacher_flow = flow_anchor_teacher_model.flow_head.sample(  # type: ignore[attr-defined]
                                teacher_z0,
                                player_states=teacher_out.player_states,
                                team_states=teacher_out.team_states,
                                game_state=teacher_out.game_state,
                                player_team_index=teacher_out.player_team_index,
                                valid_mask=teacher_out.player_valid_mask,
                                observed_mask=teacher_out.player_valid_mask.unsqueeze(-1).expand_as(teacher_z0),
                                minutes_context=teacher_out.minutes.minutes,
                                env_context=getattr(teacher_out, "env_context", None),
                            )
                            teacher_flow = _project_flow_stats_to_contract(
                                teacher_flow,
                                flow_target_columns=teacher_flow_columns,
                                flow_contract_columns=ftc,
                                fg2_rate=teacher_out.efficiency.mean_fg2 if teacher_out.efficiency is not None else None,
                                fg3_rate=teacher_out.efficiency.mean_fg3 if teacher_out.efficiency is not None else None,
                                ft_rate=teacher_out.efficiency.mean_ft if teacher_out.efficiency is not None else None,
                            )
                            if bool(getattr(teacher_cfg, "assist_share_reconcile_ast_budget", False)):
                                teacher_flow = _reconcile_ast_to_team_budget(
                                    flow_projected=teacher_flow,
                                    flow_contract_columns=ftc,
                                    player_valid_mask=teacher_out.player_valid_mask,
                                    player_team_index=teacher_out.player_team_index,
                                    active_mask=teacher_out.active.active_mask,
                                    team_ast_budget=(
                                        teacher_out.team_ast_budget.team_ast
                                        if teacher_out.team_ast_budget is not None
                                        else None
                                    ),
                                    assist_share_logits=(
                                        teacher_out.assist_share.ast_logits
                                        if teacher_out.assist_share is not None
                                        else None
                                    ),
                                    share_alpha=float(getattr(teacher_cfg, "assist_share_reconcile_alpha", 0.75)),
                                    share_temperature=float(
                                        getattr(teacher_cfg, "assist_share_reconcile_temperature", 0.85)
                                    ),
                                )
                        anchor_mask = out.player_valid_mask.unsqueeze(-1).expand_as(emergent_flow).clone()
                        anchor_mask[..., ast_i] = False
                        if bool(anchor_mask.any()):
                            flow_anchor_nonast_aux_loss = _masked_scaled_huber_loss(
                                pred=emergent_flow,
                                target=teacher_flow.to(dtype=emergent_flow.dtype),
                                mask=anchor_mask,
                                scale=float(flow_anchor_target_scale),
                                delta=float(direct_stat_aux_huber_delta),
                            )
                    fg2m_em = emergent_flow[..., fg2m_i]
                    fg3m_em = emergent_flow[..., fg3m_i]
                    fga_em = emergent_flow[..., fga2_i] + emergent_flow[..., fga3_i]
                    fta_em = emergent_flow[..., fta_i]
                    ftm_em = emergent_flow[..., ftm_i]
                    oreb_em = emergent_flow[..., oreb_i]
                    dreb_em = emergent_flow[..., dreb_i]
                    ast_em = emergent_flow[..., ast_i]
                    stl_em = emergent_flow[..., stl_i]
                    blk_em = emergent_flow[..., blk_i]
                    tov_em = emergent_flow[..., tov_i]
                    pts_em = 2.0 * fg2m_em + 3.0 * fg3m_em + ftm_em
                    reb_em = oreb_em + dreb_em

                    if (
                        float(w_props_pts_aux) > 0.0
                        or float(w_props_reb_aux) > 0.0
                        or float(w_props_ast_aux) > 0.0
                    ):
                        market_count_raw = _raw_player_feature(int(an_props_market_count_idx))
                        prior_play_prob_raw = _raw_player_feature(int(prior_play_prob_idx))
                        if prior_play_prob_raw is None:
                            prior_play_prob_raw = torch.ones_like(pts_em)
                        prior_play_prob_raw = torch.clamp(prior_play_prob_raw, min=0.0, max=1.0)

                        def _prop_aux_loss(
                            *,
                            pred: torch.Tensor,
                            line_idx: int,
                            has_idx: int,
                            books_idx: int,
                            line_center: float,
                            line_scale: float,
                            target_scale: float,
                            min_line: float,
                        ) -> torch.Tensor:
                            line_raw = _raw_player_feature(int(line_idx))
                            if line_raw is None:
                                return torch.zeros((), dtype=pred.dtype, device=pred.device)
                            has_raw = _raw_player_feature(int(has_idx))
                            has_market = line_raw.gt(0.0) if has_raw is None else has_raw.ge(0.5)
                            line_ok = torch.isfinite(line_raw) & line_raw.gt(0.0)
                            if float(min_line) > 0.0:
                                line_ok = line_ok & line_raw.ge(float(min_line))
                            mask = out.player_valid_mask & has_market & line_ok
                            if not bool(mask.any()):
                                return torch.zeros((), dtype=pred.dtype, device=pred.device)

                            books_raw = _raw_player_feature(int(books_idx))
                            if books_raw is None:
                                books_strength = torch.full_like(line_raw, 0.5)
                            else:
                                books_strength = torch.clamp((books_raw - 1.0) / 2.0, min=0.0, max=1.0)
                            if market_count_raw is None:
                                market_strength = torch.full_like(line_raw, 0.5)
                            else:
                                market_strength = torch.clamp((market_count_raw - 1.0) / 6.0, min=0.0, max=1.0)
                            line_strength = torch.sigmoid((line_raw - float(line_center)) / max(float(line_scale), 1e-6))
                            conf = 0.60 * line_strength + 0.20 * books_strength + 0.20 * market_strength
                            conf = torch.clamp(conf * prior_play_prob_raw, min=float(props_aux_confidence_min), max=1.0)
                            return _weighted_masked_scaled_huber_loss(
                                pred=pred,
                                target=line_raw,
                                mask=mask,
                                scale=float(target_scale),
                                delta=float(props_aux_huber_delta),
                                weights=conf,
                            )

                        if float(w_props_pts_aux) > 0.0:
                            props_pts_aux_loss = _prop_aux_loss(
                                pred=pts_em,
                                line_idx=int(an_pts_line_idx),
                                has_idx=int(an_has_pts_idx),
                                books_idx=int(an_pts_books_idx),
                                line_center=16.0,
                                line_scale=6.0,
                                target_scale=float(props_pts_target_scale),
                                min_line=float(props_pts_aux_min_line),
                            )
                        if float(w_props_reb_aux) > 0.0:
                            props_reb_aux_loss = _prop_aux_loss(
                                pred=reb_em,
                                line_idx=int(an_reb_line_idx),
                                has_idx=int(an_has_reb_idx),
                                books_idx=int(an_reb_books_idx),
                                line_center=6.0,
                                line_scale=2.5,
                                target_scale=float(props_reb_target_scale),
                                min_line=float(props_reb_aux_min_line),
                            )
                        if float(w_props_ast_aux) > 0.0:
                            props_ast_aux_loss = _prop_aux_loss(
                                pred=ast_em,
                                line_idx=int(an_ast_line_idx),
                                has_idx=int(an_has_ast_idx),
                                books_idx=int(an_ast_books_idx),
                                line_center=4.0,
                                line_scale=2.0,
                                target_scale=float(props_ast_target_scale),
                                min_line=float(props_ast_aux_min_line),
                            )

                    if float(w_spread_aux) > 0.0 or float(w_total_aux) > 0.0:
                        if float(w_spread_aux) > 0.0 and int(vegas_spread_idx) < 0:
                            raise RuntimeError(
                                "w_spread_aux > 0 requires vegas_spread in --game-feature-cols",
                            )
                        if float(w_total_aux) > 0.0 and int(vegas_total_idx) < 0:
                            raise RuntimeError(
                                "w_total_aux > 0 requires vegas_total in --game-feature-cols",
                            )

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
                        or float(w_team_rebound_budget_rate_aux) > 0.0
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

                    if float(w_team_rebound_budget_rate_aux) > 0.0 and getattr(out, "team_rebound_budget", None) is not None:
                        budget_mode = str(
                            getattr(getattr(model, "gtv2_config", None), "rebound_budget_parameterization", "absolute")
                        ).strip().lower()
                        budget_rate_terms: list[torch.Tensor] = []
                        if budget_mode in {"oreb_rate", "both_rate"}:
                            budget_rate_terms.append(
                                _team_fixed_opportunity_rate_mse_loss(
                                    pred_numerator=out.team_rebound_budget.team_oreb * own_missed_true_team,
                                    true_numerator=oreb_true_team,
                                    true_denominator=own_missed_true_team,
                                    observed_mask=oreb_true_seen & own_missed_true_seen,
                                )
                            )
                        if budget_mode in {"dreb_rate", "both_rate"}:
                            budget_rate_terms.append(
                                _team_fixed_opportunity_rate_mse_loss(
                                    pred_numerator=out.team_rebound_budget.team_dreb * opp_missed_true_team,
                                    true_numerator=dreb_true_team,
                                    true_denominator=opp_missed_true_team,
                                    observed_mask=dreb_true_seen & opp_missed_true_seen,
                                )
                            )
                        if (
                            budget_mode == "dreb_rate_residual"
                            and rebound_flow_dreb_team_base is not None
                            and rebound_opp_missed_team_base is not None
                        ):
                            flow_dreb_rate_base = torch.nan_to_num(
                                rebound_flow_dreb_team_base / rebound_opp_missed_team_base.clamp(min=1.0),
                                nan=0.0,
                                posinf=1.0,
                                neginf=0.0,
                            )
                            residual_dreb_rate = torch.clamp(
                                flow_dreb_rate_base + out.team_rebound_budget.team_dreb,
                                min=0.0,
                                max=1.0,
                            )
                            budget_rate_terms.append(
                                _team_fixed_opportunity_rate_mse_loss(
                                    pred_numerator=residual_dreb_rate * opp_missed_true_team,
                                    true_numerator=dreb_true_team,
                                    true_denominator=opp_missed_true_team,
                                    observed_mask=dreb_true_seen & opp_missed_true_seen,
                                )
                            )
                        if budget_rate_terms:
                            team_rebound_budget_rate_aux_loss = torch.stack(budget_rate_terms).mean()

                    if (
                        float(w_rebound_budget_blend_gate_aux) > 0.0
                        and getattr(out, "rebound_budget_blend_gate", None) is not None
                        and rebound_flow_oreb_team_base is not None
                        and rebound_flow_dreb_team_base is not None
                        and rebound_target_oreb_budget is not None
                        and rebound_target_dreb_budget is not None
                    ):
                        gate_terms: list[torch.Tensor] = []
                        reconcile_mode = str(getattr(model_cfg, "rebound_factor_reconcile_mode", "both")).strip().lower()
                        if reconcile_mode in {"both", "oreb_only"}:
                            oreb_gate_target, oreb_gate_mask = _team_budget_blend_gate_targets(
                                flow_team_budget=rebound_flow_oreb_team_base,
                                factorized_team_budget=rebound_target_oreb_budget,
                                target_team_budget=oreb_true_team,
                                observed_mask=oreb_true_seen & own_missed_true_seen,
                                eps=float(rebound_budget_blend_gate_target_eps),
                            )
                            if bool(oreb_gate_mask.any()):
                                gate_loss = F.binary_cross_entropy_with_logits(
                                    out.rebound_budget_blend_gate.oreb_gate_logits,
                                    oreb_gate_target.to(dtype=out.rebound_budget_blend_gate.oreb_gate_logits.dtype),
                                    reduction="none",
                                )
                                gate_mask_f = oreb_gate_mask.to(dtype=gate_loss.dtype)
                                gate_terms.append((gate_loss * gate_mask_f).sum() / gate_mask_f.sum().clamp(min=1.0))
                        if reconcile_mode in {"both", "dreb_only"}:
                            dreb_gate_target, dreb_gate_mask = _team_budget_blend_gate_targets(
                                flow_team_budget=rebound_flow_dreb_team_base,
                                factorized_team_budget=rebound_target_dreb_budget,
                                target_team_budget=dreb_true_team,
                                observed_mask=dreb_true_seen & opp_missed_true_seen,
                                eps=float(rebound_budget_blend_gate_target_eps),
                            )
                            if bool(dreb_gate_mask.any()):
                                gate_loss = F.binary_cross_entropy_with_logits(
                                    out.rebound_budget_blend_gate.dreb_gate_logits,
                                    dreb_gate_target.to(dtype=out.rebound_budget_blend_gate.dreb_gate_logits.dtype),
                                    reduction="none",
                                )
                                gate_mask_f = dreb_gate_mask.to(dtype=gate_loss.dtype)
                                gate_terms.append((gate_loss * gate_mask_f).sum() / gate_mask_f.sum().clamp(min=1.0))
                        if gate_terms:
                            rebound_budget_blend_gate_aux_loss = torch.stack(gate_terms).mean()

                    direct_mask_base = out.player_valid_mask
                    direct_named_losses: dict[str, torch.Tensor] = {}
                    include_flow_ast_direct = not factorized_ast_mode
                    if float(w_direct_pts_aux) > 0.0:
                        pts_true = 2.0 * flow_target_flat[..., fg2m_i] + 3.0 * flow_target_flat[..., fg3m_i] + flow_target_flat[..., ftm_i]
                        pts_obs = (
                            flow_observed_flat[..., fg2m_i]
                            & flow_observed_flat[..., fg3m_i]
                            & flow_observed_flat[..., ftm_i]
                        )
                        direct_pts_aux_loss = _masked_scaled_huber_loss(
                            pred=pts_em,
                            target=pts_true,
                            mask=direct_mask_base & pts_obs,
                            scale=float(direct_pts_target_scale),
                            delta=float(direct_stat_aux_huber_delta),
                        )
                    if float(w_direct_reb_aux) > 0.0:
                        reb_true = flow_target_flat[..., oreb_i] + flow_target_flat[..., dreb_i]
                        reb_obs = flow_observed_flat[..., oreb_i] & flow_observed_flat[..., dreb_i]
                        direct_reb_aux_loss = _masked_scaled_huber_loss(
                            pred=reb_em,
                            target=reb_true,
                            mask=direct_mask_base & reb_obs,
                            scale=float(direct_reb_target_scale),
                            delta=float(direct_stat_aux_huber_delta),
                        )
                    if float(w_direct_ast_aux) > 0.0 and include_flow_ast_direct:
                        direct_ast_aux_loss = _masked_scaled_huber_loss(
                            pred=ast_em,
                            target=ast_true,
                            mask=direct_mask_base & ast_obs,
                            scale=float(direct_ast_target_scale),
                            delta=float(direct_stat_aux_huber_delta),
                        )
                    if float(w_direct_stl_aux) > 0.0:
                        direct_stl_aux_loss = _masked_scaled_huber_loss(
                            pred=stl_em,
                            target=stl_true,
                            mask=direct_mask_base & stl_obs,
                            scale=float(direct_stl_target_scale),
                            delta=float(direct_stat_aux_huber_delta),
                        )
                    if float(w_direct_blk_aux) > 0.0:
                        direct_blk_aux_loss = _masked_scaled_huber_loss(
                            pred=blk_em,
                            target=blk_true,
                            mask=direct_mask_base & blk_obs,
                            scale=float(direct_blk_target_scale),
                            delta=float(direct_stat_aux_huber_delta),
                        )
                    if float(w_direct_tov_aux) > 0.0:
                        direct_tov_aux_loss = _masked_scaled_huber_loss(
                            pred=tov_em,
                            target=tov_true,
                            mask=direct_mask_base & tov_obs,
                            scale=float(direct_tov_target_scale),
                            delta=float(direct_stat_aux_huber_delta),
                        )
                    if float(w_direct_boxscore_aux) > 0.0 or float(w_direct_opportunity_aux) > 0.0:
                        pts_true = 2.0 * flow_target_flat[..., fg2m_i] + 3.0 * flow_target_flat[..., fg3m_i] + flow_target_flat[..., ftm_i]
                        pts_obs = (
                            flow_observed_flat[..., fg2m_i]
                            & flow_observed_flat[..., fg3m_i]
                            & flow_observed_flat[..., ftm_i]
                        )
                        reb_true = flow_target_flat[..., oreb_i] + flow_target_flat[..., dreb_i]
                        reb_obs = flow_observed_flat[..., oreb_i] & flow_observed_flat[..., dreb_i]
                        stat_specs: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]] = {
                            "pts": (
                                pts_em,
                                pts_true,
                                direct_mask_base & pts_obs,
                                float(direct_pts_target_scale),
                            ),
                            "reb": (
                                reb_em,
                                reb_true,
                                direct_mask_base & reb_obs,
                                float(direct_reb_target_scale),
                            ),
                            "stl": (
                                stl_em,
                                stl_true,
                                direct_mask_base & stl_obs,
                                float(direct_stl_target_scale),
                            ),
                            "blk": (
                                blk_em,
                                blk_true,
                                direct_mask_base & blk_obs,
                                float(direct_blk_target_scale),
                            ),
                            "fg3m": (
                                fg3m_em,
                                flow_target_flat[..., fg3m_i],
                                direct_mask_base & flow_observed_flat[..., fg3m_i],
                                float(direct_fg3m_target_scale),
                            ),
                            "ftm": (
                                ftm_em,
                                flow_target_flat[..., ftm_i],
                                direct_mask_base & flow_observed_flat[..., ftm_i],
                                float(direct_ftm_target_scale),
                            ),
                            "tov": (
                                tov_em,
                                tov_true,
                                direct_mask_base & tov_obs,
                                float(direct_tov_target_scale),
                            ),
                            "fga": (
                                fga_em,
                                fga_true,
                                direct_mask_base & fga_obs,
                                float(direct_fga_target_scale),
                            ),
                            "fta": (
                                fta_em,
                                fta_true,
                                direct_mask_base & fta_obs,
                                float(direct_fta_target_scale),
                            ),
                        }
                        if include_flow_ast_direct:
                            stat_specs["ast"] = (
                                ast_em,
                                ast_true,
                                direct_mask_base & ast_obs,
                                float(direct_ast_target_scale),
                            )
                        direct_named_losses = _compute_named_direct_stat_losses(
                            stat_specs=stat_specs,
                            delta=float(direct_stat_aux_huber_delta),
                        )
                        if float(w_direct_boxscore_aux) > 0.0:
                            boxscore_loss_names = ("pts", "reb", "stl", "blk", "fg3m", "ftm", "tov")
                            if include_flow_ast_direct:
                                boxscore_loss_names = ("pts", "reb", "ast", "stl", "blk", "fg3m", "ftm", "tov")
                            direct_boxscore_aux_loss = _mean_named_losses(direct_named_losses, boxscore_loss_names)
                        if float(w_direct_opportunity_aux) > 0.0:
                            direct_opportunity_aux_loss = _mean_named_losses(
                                direct_named_losses,
                                ("fga", "fta"),
                            )

            total_loss = (
                float(w_minutes) * minutes_mae
                + float(w_count) * active_losses["count_loss"]
                + float(w_member) * active_losses["member_loss"]
                + float(w_minutes_nll) * minutes_nll
                + float(w_minutes_hurdle_nll) * minutes_hurdle_nll
                + float(w_role_loss) * role_loss
                + float(w_starter_promotion_loss) * starter_promotion_loss
                + float(w_sparse_starter_underpred_loss) * sparse_starter_underpred_loss
                + float(w_bench_riser_underpred_loss) * bench_riser_underpred_loss
                + float(w_flow_nll) * flow_nll
                + float(w_crps_fpts) * crps_fpts
                + float(w_team_energy) * team_energy
                + float(w_poss_nll) * poss_nll_loss
                + float(w_poss_regression) * poss_regression_loss
                + float(w_team_efficiency_ppp_aux) * team_efficiency_ppp_aux_loss
                + float(w_team_ppp_aux) * team_ppp_aux_loss
                + float(w_team_advantage_aux) * team_advantage_aux_loss
                + float(w_team_possession_aux) * team_possession_aux_loss
                + float(w_backbone_nll) * backbone_nll_loss
                + float(w_three_pa_nll) * three_pa_nll_loss
                + float(w_efficiency_nll) * efficiency_nll_loss
                + float(w_usage_share_nll) * usage_share_nll_loss
                + float(w_team_points_budget_aux) * team_points_budget_aux_loss
                + float(w_team_ast_budget_aux) * team_ast_budget_aux_loss
                + float(w_assist_share_aux) * assist_share_aux_loss
                + float(w_assist_share_recon_aux) * assist_share_recon_aux_loss
                + float(w_ast_blend_gate_aux) * ast_blend_gate_aux_loss
                + float(w_emergent_share_aux) * emergent_share_aux_loss
                + float(w_ast_share_aux) * ast_share_aux_loss
                + float(w_reb_share_aux) * reb_share_aux_loss
                + float(w_ast_team_rate_aux) * ast_team_rate_aux_loss
                + float(w_reb_opportunity_rate_aux) * reb_opportunity_rate_aux_loss
                + float(w_team_rebound_budget_rate_aux) * team_rebound_budget_rate_aux_loss
                + float(w_rebound_budget_blend_gate_aux) * rebound_budget_blend_gate_aux_loss
                + float(w_spread_aux_eff) * spread_aux_loss
                + float(w_total_aux_eff) * total_aux_loss
                + float(w_props_pts_aux_eff) * props_pts_aux_loss
                + float(w_props_reb_aux_eff) * props_reb_aux_loss
                + float(w_props_ast_aux_eff) * props_ast_aux_loss
                + float(w_direct_pts_aux_eff) * direct_pts_aux_loss
                + float(w_direct_reb_aux_eff) * direct_reb_aux_loss
                + float(w_direct_ast_aux_eff) * direct_ast_aux_loss
                + float(w_direct_stl_aux_eff) * direct_stl_aux_loss
                + float(w_direct_blk_aux_eff) * direct_blk_aux_loss
                + float(w_direct_tov_aux_eff) * direct_tov_aux_loss
                + float(w_direct_boxscore_aux_eff) * direct_boxscore_aux_loss
                + float(w_direct_opportunity_aux_eff) * direct_opportunity_aux_loss
                + float(w_flow_anchor_nonast_aux) * flow_anchor_nonast_aux_loss
                + float(w_efficiency_mean_aux) * efficiency_mean_aux_loss
            )

            skip_step = False
            if training and not bool(total_loss.requires_grad):
                totals["skipped_batches"] += 1
                skip_step = True
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
            totals["minutes_hurdle_nll"] += float(minutes_hurdle_nll.item())
            totals["role_loss"] += float(role_loss.item())
            totals["role_acc"] += float(role_acc.item())
            totals["starter_promotion_loss"] += float(starter_promotion_loss.item())
            totals["sparse_starter_underpred_loss"] += float(sparse_starter_underpred_loss.item())
            totals["bench_riser_underpred_loss"] += float(bench_riser_underpred_loss.item())
            totals["flow_nll"] += float(flow_nll.item())
            totals["crps_fpts"] += float(crps_fpts.item())
            totals["team_energy"] += float(team_energy.item())
            totals["count_acc"] += float(count_acc.item())
            totals["poss_nll"] += float(poss_nll_loss.item())
            totals["poss_regression"] += float(poss_regression_loss.item())
            totals["team_ppp_aux"] += float(team_ppp_aux_loss.item())
            totals["team_advantage_aux"] += float(team_advantage_aux_loss.item())
            totals["team_possession_aux"] += float(team_possession_aux_loss.item())
            totals["backbone_nll"] += float(backbone_nll_loss.item())
            totals["three_pa_nll"] += float(three_pa_nll_loss.item())
            totals["efficiency_nll"] += float(efficiency_nll_loss.item())
            totals["team_efficiency_ppp_aux"] += float(team_efficiency_ppp_aux_loss.item())
            totals["usage_share_nll"] += float(usage_share_nll_loss.item())
            totals["team_points_budget_aux"] += float(team_points_budget_aux_loss.item())
            totals["team_ast_budget_aux"] += float(team_ast_budget_aux_loss.item())
            totals["assist_share_aux"] += float(assist_share_aux_loss.item())
            totals["assist_share_recon_aux"] += float(assist_share_recon_aux_loss.item())
            totals["ast_blend_gate_aux"] += float(ast_blend_gate_aux_loss.item())
            totals["emergent_share_aux"] += float(emergent_share_aux_loss.item())
            totals["ast_share_aux"] += float(ast_share_aux_loss.item())
            totals["reb_share_aux"] += float(reb_share_aux_loss.item())
            totals["ast_team_rate_aux"] += float(ast_team_rate_aux_loss.item())
            totals["reb_opportunity_rate_aux"] += float(reb_opportunity_rate_aux_loss.item())
            totals["team_rebound_budget_rate_aux"] += float(team_rebound_budget_rate_aux_loss.item())
            totals["rebound_budget_blend_gate_aux"] += float(rebound_budget_blend_gate_aux_loss.item())
            totals["spread_aux"] += float(spread_aux_loss.item())
            totals["total_aux"] += float(total_aux_loss.item())
            totals["props_pts_aux"] += float(props_pts_aux_loss.item())
            totals["props_reb_aux"] += float(props_reb_aux_loss.item())
            totals["props_ast_aux"] += float(props_ast_aux_loss.item())
            totals["direct_pts_aux"] += float(direct_pts_aux_loss.item())
            totals["direct_reb_aux"] += float(direct_reb_aux_loss.item())
            totals["direct_ast_aux"] += float(direct_ast_aux_loss.item())
            totals["direct_stl_aux"] += float(direct_stl_aux_loss.item())
            totals["direct_blk_aux"] += float(direct_blk_aux_loss.item())
            totals["direct_tov_aux"] += float(direct_tov_aux_loss.item())
            totals["direct_boxscore_aux"] += float(direct_boxscore_aux_loss.item())
            totals["direct_opportunity_aux"] += float(direct_opportunity_aux_loss.item())
            totals["flow_anchor_nonast_aux"] += float(flow_anchor_nonast_aux_loss.item())
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
        "minutes_hurdle_nll": totals["minutes_hurdle_nll"] / denom,
        "role_loss": totals["role_loss"] / denom,
        "role_acc": totals["role_acc"] / denom,
        "starter_promotion_loss": totals["starter_promotion_loss"] / denom,
        "sparse_starter_underpred_loss": totals["sparse_starter_underpred_loss"] / denom,
        "bench_riser_underpred_loss": totals["bench_riser_underpred_loss"] / denom,
        "flow_nll": totals["flow_nll"] / denom,
        "crps_fpts": totals["crps_fpts"] / denom,
        "team_energy": totals["team_energy"] / denom,
        "count_acc": totals["count_acc"] / denom,
        "poss_nll": totals["poss_nll"] / denom,
        "poss_regression": totals["poss_regression"] / denom,
        "team_ppp_aux": totals["team_ppp_aux"] / denom,
        "team_advantage_aux": totals["team_advantage_aux"] / denom,
        "team_possession_aux": totals["team_possession_aux"] / denom,
        "backbone_nll": totals["backbone_nll"] / denom,
        "three_pa_nll": totals["three_pa_nll"] / denom,
        "efficiency_nll": totals["efficiency_nll"] / denom,
        "usage_share_nll": totals["usage_share_nll"] / denom,
        "team_points_budget_aux": totals["team_points_budget_aux"] / denom,
        "team_ast_budget_aux": totals["team_ast_budget_aux"] / denom,
        "assist_share_aux": totals["assist_share_aux"] / denom,
        "assist_share_recon_aux": totals["assist_share_recon_aux"] / denom,
        "ast_blend_gate_aux": totals["ast_blend_gate_aux"] / denom,
        "emergent_share_aux": totals["emergent_share_aux"] / denom,
        "ast_share_aux": totals["ast_share_aux"] / denom,
        "reb_share_aux": totals["reb_share_aux"] / denom,
        "ast_team_rate_aux": totals["ast_team_rate_aux"] / denom,
        "reb_opportunity_rate_aux": totals["reb_opportunity_rate_aux"] / denom,
        "team_rebound_budget_rate_aux": totals["team_rebound_budget_rate_aux"] / denom,
        "rebound_budget_blend_gate_aux": totals["rebound_budget_blend_gate_aux"] / denom,
        "spread_aux": totals["spread_aux"] / denom,
        "total_aux": totals["total_aux"] / denom,
        "props_pts_aux": totals["props_pts_aux"] / denom,
        "props_reb_aux": totals["props_reb_aux"] / denom,
        "props_ast_aux": totals["props_ast_aux"] / denom,
        "direct_pts_aux": totals["direct_pts_aux"] / denom,
        "direct_reb_aux": totals["direct_reb_aux"] / denom,
        "direct_ast_aux": totals["direct_ast_aux"] / denom,
        "direct_stl_aux": totals["direct_stl_aux"] / denom,
        "direct_blk_aux": totals["direct_blk_aux"] / denom,
        "direct_tov_aux": totals["direct_tov_aux"] / denom,
        "direct_boxscore_aux": totals["direct_boxscore_aux"] / denom,
        "direct_opportunity_aux": totals["direct_opportunity_aux"] / denom,
        "flow_anchor_nonast_aux": totals["flow_anchor_nonast_aux"] / denom,
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
        "--feature-columns-json",
        type=str,
        default=None,
        help="Optional JSON file providing the exact generic player feature_columns to use instead of inference.",
    )
    parser.add_argument(
        "--exclude-feature-pattern",
        action="append",
        default=None,
        help="Regex pattern for inferred player feature columns to exclude. Repeatable.",
    )
    parser.add_argument(
        "--context-priors-curated-only",
        action="store_true",
        help=(
            "Exclude raw context-bucket prior columns and keep only curated matched-context "
            "outputs such as ctx_minutes_from_stints_prior_* and ctx_started_proxy_rate_prior_*."
        ),
    )
    parser.add_argument(
        "--init-model-pt",
        type=str,
        default=None,
        help="Optional checkpoint path for warm-start (for example Phase 1 model.pt).",
    )
    parser.add_argument(
        "--graft-model-pt",
        type=str,
        default=None,
        help="Optional secondary checkpoint used to overwrite selected parameter prefixes after --init-model-pt.",
    )
    parser.add_argument(
        "--graft-prefixes",
        type=str,
        default="",
        help="Comma-separated parameter prefixes to copy from --graft-model-pt.",
    )
    parser.add_argument(
        "--freeze-prefixes",
        type=str,
        default="",
        help="Comma-separated parameter prefixes to freeze after warm-start/graft loading.",
    )
    parser.add_argument("--val-days", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", help="Torch device: auto, cpu, cuda, cuda:0, mps.")

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
    parser.add_argument(
        "--w-minutes-hurdle-nll",
        type=float,
        default=0.0,
        help="Auxiliary hurdle minutes NLL (zero-mass BCE + positive Gaussian NLL).",
    )
    parser.add_argument("--w-count", type=float, default=0.5)
    parser.add_argument("--w-member", type=float, default=0.5)
    parser.add_argument("--w-flow-nll", type=float, default=1.0)
    parser.add_argument("--minutes-nll-sigma", type=float, default=6.0)
    parser.add_argument(
        "--minutes-hurdle-zero-threshold",
        type=float,
        default=0.5,
        help="Minutes <= threshold are treated as zero-minute targets for hurdle BCE.",
    )
    parser.add_argument("--enable-minutes-hurdle-head", action="store_true")
    parser.add_argument("--minutes-hurdle-hidden", type=int, default=64)
    parser.add_argument("--minutes-hurdle-sigma-floor", type=float, default=0.5)
    parser.add_argument("--enable-minutes-role-head", action="store_true")
    parser.add_argument(
        "--disable-minutes-role-context-for-preferences",
        action="store_true",
        help="Keep role supervision enabled but do not feed predicted role context into the minutes preference head.",
    )
    parser.add_argument("--minutes-role-hidden", type=int, default=64)
    parser.add_argument("--minutes-role-embedding-dim", type=int, default=32)
    parser.add_argument("--minutes-role-num-classes", type=int, default=5)
    parser.add_argument(
        "--minutes-role-target-scheme",
        type=str,
        default="team_rank",
        choices=("team_rank", "starter_context"),
    )
    parser.add_argument(
        "--w-role-loss",
        type=float,
        default=0.0,
        help="Auxiliary CE loss for team-relative role-state prediction.",
    )
    parser.add_argument("--enable-starter-promotion-head", action="store_true")
    parser.add_argument("--starter-promotion-hidden-dim", type=int, default=64)
    parser.add_argument("--w-starter-promotion-loss", type=float, default=0.0)
    parser.add_argument("--starter-promotion-prior-minutes-max", type=float, default=18.0)
    parser.add_argument("--w-sparse-starter-underpred-loss", type=float, default=0.0)
    parser.add_argument("--sparse-starter-loss-prior-play-prob-max", type=float, default=0.20)
    parser.add_argument("--sparse-starter-loss-prior-minutes-max", type=float, default=6.0)
    parser.add_argument("--sparse-starter-loss-hist-start-rate-max", type=float, default=0.20)
    parser.add_argument("--sparse-starter-loss-actual-min-threshold", type=float, default=20.0)
    parser.add_argument("--w-bench-riser-underpred-loss", type=float, default=0.0)
    parser.add_argument("--bench-riser-loss-prior-minutes-min", type=float, default=8.0)
    parser.add_argument("--bench-riser-loss-prior-play-prob-min", type=float, default=0.80)
    parser.add_argument("--bench-riser-loss-hist-start-rate-max", type=float, default=0.50)
    parser.add_argument("--bench-riser-loss-actual-min-threshold", type=float, default=20.0)
    parser.add_argument("--active-positive-weight", type=float, default=1.0)
    parser.add_argument(
        "--lineup-available-sample-weight",
        type=float,
        default=1.0,
        help=(
            "Train sampler upweight for lineup_available=1 coverage. "
            "1.0 disables; values >1.0 oversample examples with higher lineup coverage."
        ),
    )
    parser.add_argument(
        "--sparse-candidate-sample-weight",
        type=float,
        default=1.0,
        help=(
            "Train sampler upweight for game examples containing at least one broad sparse-starter "
            "candidate under the same pre-tip features used by the promotion overlay."
        ),
    )
    parser.add_argument("--sparse-candidate-sample-prior-minutes-max", type=float, default=14.0)
    parser.add_argument("--sparse-candidate-sample-hist-start-rate-max", type=float, default=0.25)
    parser.add_argument("--bench-candidate-sample-weight", type=float, default=1.0)
    parser.add_argument("--bench-candidate-sample-prior-minutes-min", type=float, default=8.0)
    parser.add_argument("--bench-candidate-sample-prior-play-prob-min", type=float, default=0.80)
    parser.add_argument("--bench-candidate-sample-hist-start-rate-max", type=float, default=0.35)
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
    parser.add_argument("--minutes-teacher-forcing-prob-start", type=float, default=1.0)
    parser.add_argument("--minutes-teacher-forcing-prob-end", type=float, default=1.0)
    parser.add_argument("--minutes-teacher-forcing-ramp-epochs", type=int, default=1)
    parser.add_argument(
        "--minutes-teacher-forcing-mode",
        type=str,
        default="batch",
        choices=("batch", "example", "team"),
    )
    parser.add_argument(
        "--flow-use-minutes-conditioning",
        action="store_true",
        help="Condition flow coupling blocks on per-player minutes.",
    )
    parser.add_argument("--flow-minutes-teacher-forcing-prob-start", type=float, default=1.0)
    parser.add_argument("--flow-minutes-teacher-forcing-prob-end", type=float, default=1.0)
    parser.add_argument("--flow-minutes-teacher-forcing-ramp-epochs", type=int, default=1)
    parser.add_argument(
        "--flow-minutes-teacher-forcing-mode",
        type=str,
        default="batch",
        choices=("batch", "example", "team"),
    )
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
        choices=["val_total", "val_total_ex_possreg", "val_minutes_mae"],
        help="Validation metric used for early stopping.",
    )
    parser.add_argument(
        "--best-checkpoint-metric",
        type=str,
        default="val_total",
        choices=["val_total", "val_total_ex_possreg", "val_minutes_mae"],
        help="Validation metric used to decide which checkpoint is written to model.pt.",
    )
    parser.add_argument(
        "--enable-sparse-checkpoint-rerank",
        action="store_true",
        help="After training, rerank a shortlist of low val_minutes_mae checkpoints using sparse-rotation diagnostics.",
    )
    parser.add_argument(
        "--checkpoint-topk-by-minutes",
        type=int,
        default=5,
        help="How many low val_minutes_mae checkpoints to retain for sparse-aware reranking.",
    )
    parser.add_argument(
        "--checkpoint-minutes-mae-tolerance",
        type=float,
        default=0.08,
        help="Only checkpoints within best val_minutes_mae + tolerance are eligible for sparse-aware final selection.",
    )
    parser.add_argument("--checkpoint-low-minutes-threshold", type=float, default=8.0)
    parser.add_argument("--checkpoint-sparse-prior-play-prob-max", type=float, default=0.20)
    parser.add_argument("--checkpoint-sparse-prior-minutes-max", type=float, default=6.0)
    parser.add_argument("--checkpoint-starter-promotion-prior-minutes-max", type=float, default=12.0)
    parser.add_argument("--checkpoint-starter-promotion-hist-start-rate-max", type=float, default=0.20)
    parser.add_argument("--checkpoint-next-up-actual-min", type=float, default=20.0)
    parser.add_argument("--checkpoint-next-up-pred-min", type=float, default=10.0)
    parser.add_argument(
        "--checkpoint-sparse-target-starter-sparse-minutes",
        type=float,
        default=5.0,
        help="Sparse rerank only penalizes starter-sparse predicted minutes below this level.",
    )
    parser.add_argument("--checkpoint-sparse-weight-underpred", type=float, default=6.0)
    parser.add_argument("--checkpoint-sparse-weight-active-count-mae", type=float, default=1.0)
    parser.add_argument("--checkpoint-sparse-weight-starter-shortfall", type=float, default=0.5)
    parser.add_argument("--flow-coupling-type", type=str, default="affine", choices=["affine", "rqs"])
    parser.add_argument("--flow-num-blocks", type=int, default=4)
    parser.add_argument("--flow-scale-clip", type=float, default=3.0)  # H1 fix: 2.0 → 3.0
    parser.add_argument("--flow-rqs-num-bins", type=int, default=8)
    parser.add_argument("--flow-rqs-tail-bound", type=float, default=40.0)
    parser.add_argument("--flow-rqs-min-bin-width", type=float, default=1e-3)
    parser.add_argument("--flow-rqs-min-bin-height", type=float, default=1e-3)
    parser.add_argument("--flow-rqs-min-derivative", type=float, default=1e-3)
    parser.add_argument("--flow-context-mode", type=str, default="attention", choices=["mean", "attention"])  # H2 fix
    parser.add_argument(
        "--flow-target-schema",
        type=str,
        default=FLOW_TARGET_SCHEMA_DEFAULT,
        choices=["v1", "v2"],
        help="Flow training target schema: v1 includes make columns; v2 models attempts/other stats and reconstructs makes.",
    )
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
    parser.add_argument("--enable-team-points-budget-head", action="store_true")
    parser.add_argument("--team-points-budget-head-hidden", type=int, default=128)
    parser.add_argument("--enable-team-ppp-head", action="store_true")
    parser.add_argument("--team-ppp-head-hidden", type=int, default=128)
    parser.add_argument("--team-ppp-to-backbone", action="store_true")
    parser.add_argument("--team-ppp-latent-hidden", type=int, default=32)
    parser.add_argument("--team-ppp-backbone-alpha", type=float, default=1.0)
    parser.add_argument("--team-ppp-to-efficiency", action="store_true")
    parser.add_argument("--team-ppp-efficiency-alpha", type=float, default=1.0)
    parser.add_argument("--team-ppp-direct-backbone-context", action="store_true")
    parser.add_argument("--team-ppp-direct-efficiency-context", action="store_true")
    parser.add_argument("--enable-team-advantage-head", action="store_true")
    parser.add_argument("--team-advantage-head-hidden", type=int, default=64)
    parser.add_argument("--team-advantage-direct-backbone-context", action="store_true")
    parser.add_argument(
        "--team-points-budget-parameterization",
        type=str,
        default="absolute",
        choices=["absolute", "market_implied", "team_ppp_implied"],
        help="Source for the side-specific team points budget: learned absolute head, direct market implied totals, or learned team PPP times possessions.",
    )
    parser.add_argument(
        "--team-points-budget-to-backbone",
        action="store_true",
        help="Encode predicted side-specific team points budgets back into backbone team states.",
    )
    parser.add_argument("--team-points-budget-latent-hidden", type=int, default=32)
    parser.add_argument(
        "--team-points-reconcile-budget",
        action="store_true",
        help="Reconcile generated player scoring makes toward the predicted side-specific team points budget.",
    )
    parser.add_argument("--team-points-reconcile-alpha", type=float, default=1.0)
    parser.add_argument(
        "--team-opportunity-budget-parameterization",
        type=str,
        default="absolute",
        choices=["absolute", "market_implied_share"],
        help="Source for side-specific FGA/FTA opportunity split: leave unchanged or derive share from market implied team totals.",
    )
    parser.add_argument(
        "--team-opportunity-reconcile-budget",
        action="store_true",
        help="Reconcile generated team FGA/FTA opportunity split toward the configured side-specific share.",
    )
    parser.add_argument("--team-opportunity-reconcile-alpha", type=float, default=1.0)
    parser.add_argument(
        "--team-opportunity-reconcile-preserve-possessions",
        action="store_true",
        help="Reconcile opportunity rates while solving TOV as the residual so team possessions stay unchanged.",
    )
    parser.add_argument(
        "--team-opportunity-budget-to-backbone",
        action="store_true",
        help="Encode side-specific opportunity-share context into backbone team states before event generation.",
    )
    parser.add_argument("--team-opportunity-budget-latent-hidden", type=int, default=32)
    parser.add_argument("--team-opportunity-budget-backbone-alpha", type=float, default=1.0)
    parser.add_argument("--enable-team-ast-budget-head", action="store_true")
    parser.add_argument("--team-ast-budget-head-hidden", type=int, default=128)
    parser.add_argument("--enable-assist-share-head", action="store_true")
    parser.add_argument("--assist-share-head-hidden", type=int, default=128)
    parser.add_argument("--enable-team-rebound-budget-head", action="store_true")
    parser.add_argument("--team-rebound-budget-head-hidden", type=int, default=128)
    parser.add_argument("--enable-rebound-budget-blend-gate", action="store_true")
    parser.add_argument("--rebound-budget-blend-gate-hidden", type=int, default=64)
    parser.add_argument("--rebound-budget-blend-gate-init-alpha", type=float, default=0.25)
    parser.add_argument(
        "--rebound-budget-parameterization",
        type=str,
        default="absolute",
        choices=["absolute", "dreb_rate", "dreb_rate_residual", "dreb_deterministic", "oreb_rate", "both_rate"],
        help=(
            "Interpret rebound budget behavior as absolute totals, direct opportunity capture rates, "
            "a residual DREB capture-rate delta on top of flow, or deterministic DREB from opponent misses minus opponent OREB."
        ),
    )
    parser.add_argument(
        "--rebound-oreb-rate-cap",
        type=float,
        default=1.0,
        help="Upper cap for OREB capture-rate outputs when the rebound budget head uses an OREB rate parameterization.",
    )
    parser.add_argument(
        "--rebound-dreb-rate-cap",
        type=float,
        default=0.85,
        help="Upper cap for DREB capture-rate outputs, or max absolute DREB residual-rate delta when using dreb_rate_residual.",
    )
    parser.add_argument(
        "--rebound-dreb-deterministic-discount",
        type=float,
        default=1.0,
        help="Fixed scalar discount applied to deterministic DREB budgets to account for dead-ball or non-player rebounds.",
    )
    parser.add_argument(
        "--rebound-oreb-reconcile-use-flow-budget",
        action="store_true",
        help="When OREB reconciliation is enabled, keep the current team OREB total and only redistribute it with rebound-share logits.",
    )
    parser.add_argument("--enable-rebound-share-head", action="store_true")
    parser.add_argument("--rebound-share-head-hidden", type=int, default=128)
    parser.add_argument(
        "--rebound-share-condition-feature-cols",
        type=str,
        default="an_reb_line,an_implied_minutes,prior_play_prob,started_proxy_rate_prior_20",
    )
    parser.add_argument("--rebound-share-condition-hidden", type=int, default=32)
    parser.add_argument(
        "--assist-share-condition-feature-cols",
        type=str,
        default=",".join(DEFAULT_ASSIST_SHARE_CONDITION_FEATURE_COLS),
        help="Comma-separated player-level feature columns unnormalized and fed directly into AssistShareHead.",
    )
    parser.add_argument(
        "--assist-share-condition-hidden",
        type=int,
        default=32,
        help="Hidden size for the AssistShareHead condition encoder.",
    )
    parser.add_argument("--enable-ast-blend-gate", action="store_true")
    parser.add_argument("--ast-blend-gate-hidden", type=int, default=128)
    parser.add_argument(
        "--ast-blend-gate-init-alpha",
        type=float,
        default=0.75,
        help="Initial sigmoid blend toward factorized AST when the learned AST gate is enabled.",
    )
    parser.add_argument(
        "--assist-share-replace-flow-ast",
        action="store_true",
        help="Replace the projected flow AST channel with reconstructed team_ast_budget * assist_share outputs.",
    )
    parser.add_argument(
        "--assist-share-factorized-ast",
        action="store_true",
        help="Remove AST from flow supervision and source AST only from team_ast_budget * assist_share.",
    )
    parser.add_argument(
        "--assist-share-reconcile-ast-budget",
        action="store_true",
        help=(
            "Reconcile AST to team_ast_budget using a share blend of assist-share logits and emergent flow AST shares, "
            "instead of direct AST override/blending."
        ),
    )
    parser.add_argument(
        "--assist-share-reconcile-alpha",
        type=float,
        default=0.75,
        help="Blend weight on factorized assist-share weights inside AST budget reconciliation.",
    )
    parser.add_argument(
        "--assist-share-reconcile-temperature",
        type=float,
        default=0.85,
        help="Temperature applied to assist-share logits inside AST budget reconciliation.",
    )
    parser.add_argument(
        "--rebound-factor-reconcile-oreb-dreb",
        action="store_true",
        help=(
            "Reconcile OREB/DREB to explicit rebound budgets using rebound-share logits and "
            "opportunity-capped team budgets."
        ),
    )
    parser.add_argument(
        "--rebound-factor-reconcile-alpha",
        type=float,
        default=0.50,
        help="Blend weight on factorized rebound-share weights inside OREB/DREB reconciliation.",
    )
    parser.add_argument(
        "--rebound-factor-reconcile-mode",
        type=str,
        default="both",
        choices=["both", "dreb_only", "oreb_only"],
        help="Which rebound channels the OREB/DREB reconciliation is allowed to overwrite.",
    )
    parser.add_argument(
        "--rebound-factor-reconcile-temperature",
        type=float,
        default=0.90,
        help="Temperature applied to rebound-share logits inside OREB/DREB reconciliation.",
    )
    parser.add_argument(
        "--rebound-oreb-budget-blend-alpha",
        type=float,
        default=1.0,
        help="Blend weight on the factorized OREB team budget versus the old flow-implied team OREB total.",
    )
    parser.add_argument(
        "--rebound-dreb-budget-blend-alpha",
        type=float,
        default=1.0,
        help="Blend weight on the factorized DREB team budget versus the old flow-implied team DREB total.",
    )
    parser.add_argument(
        "--rebound-budget-blend-gate-target-eps",
        type=float,
        default=0.25,
        help="Minimum team-budget separation required to supervise the learned rebound budget blend gate.",
    )
    parser.add_argument("--w-usage-share-nll", type=float, default=0.0)
    parser.add_argument(
        "--w-team-points-budget-aux",
        type=float,
        default=0.0,
        help="Auxiliary normalized Huber loss on predicted side-specific team implied totals from vegas total/spread.",
    )
    parser.add_argument(
        "--w-team-ast-budget-aux",
        type=float,
        default=0.0,
        help="Auxiliary normalized Huber loss on predicted team AST budget per side.",
    )
    parser.add_argument(
        "--w-assist-share-aux",
        type=float,
        default=0.0,
        help="Auxiliary within-team AST share CE loss using explicit assist_share_head logits.",
    )
    parser.add_argument(
        "--w-assist-share-recon-aux",
        type=float,
        default=0.0,
        help="Auxiliary normalized Huber loss on AST reconstructed from team AST budget and assist-share logits.",
    )
    parser.add_argument(
        "--w-ast-blend-gate-aux",
        type=float,
        default=0.0,
        help="Auxiliary BCE loss supervising the learned AST blend gate against the blend coefficient that best matches observed AST.",
    )
    parser.add_argument(
        "--ast-blend-gate-target-eps",
        type=float,
        default=0.25,
        help="Minimum |factorized_ast - flow_ast| required to supervise the learned AST blend gate on a player-row.",
    )
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
        "--w-team-rebound-budget-rate-aux",
        type=float,
        default=0.0,
        help="Auxiliary rate loss directly on factorized rebound budget head outputs for any rate-parameterized rebound channels.",
    )
    parser.add_argument(
        "--w-rebound-budget-blend-gate-aux",
        type=float,
        default=0.0,
        help="Auxiliary BCE loss supervising the learned rebound budget blend gate against the blend coefficient that best matches observed team rebounds.",
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
        "--w-props-pts-aux",
        type=float,
        default=0.0,
        help="Confidence-weighted auxiliary normalized Huber loss on emergent player points vs AN points line.",
    )
    parser.add_argument(
        "--w-props-reb-aux",
        type=float,
        default=0.0,
        help="Confidence-weighted auxiliary normalized Huber loss on emergent player rebounds vs AN rebounds line.",
    )
    parser.add_argument(
        "--w-props-ast-aux",
        type=float,
        default=0.0,
        help="Confidence-weighted auxiliary normalized Huber loss on emergent player assists vs AN assists line.",
    )
    parser.add_argument(
        "--w-direct-pts-aux",
        type=float,
        default=0.0,
        help="Direct normalized Huber loss on emergent player points vs observed points labels.",
    )
    parser.add_argument(
        "--w-direct-reb-aux",
        type=float,
        default=0.0,
        help="Direct normalized Huber loss on emergent player rebounds vs observed rebound labels.",
    )
    parser.add_argument(
        "--w-direct-ast-aux",
        type=float,
        default=0.0,
        help="Direct normalized Huber loss on emergent player assists vs observed assist labels.",
    )
    parser.add_argument(
        "--w-direct-stl-aux",
        type=float,
        default=0.0,
        help="Direct normalized Huber loss on emergent player steals vs observed steal labels.",
    )
    parser.add_argument(
        "--w-direct-blk-aux",
        type=float,
        default=0.0,
        help="Direct normalized Huber loss on emergent player blocks vs observed block labels.",
    )
    parser.add_argument(
        "--w-direct-tov-aux",
        type=float,
        default=0.0,
        help="Direct normalized Huber loss on emergent player turnovers vs observed turnover labels.",
    )
    parser.add_argument(
        "--w-direct-boxscore-aux",
        type=float,
        default=0.0,
        help="Mean direct normalized Huber loss across box-score stats (PTS/REB/AST/STL/BLK/3PM/FTM/TOV).",
    )
    parser.add_argument(
        "--w-direct-opportunity-aux",
        type=float,
        default=0.0,
        help="Mean direct normalized Huber loss across opportunity stats (FGA/FTA).",
    )
    parser.add_argument(
        "--flow-anchor-teacher-run-dir",
        type=str,
        default="",
        help="Optional teacher run directory for non-AST emergent-flow anchoring.",
    )
    parser.add_argument(
        "--w-flow-anchor-nonast-aux",
        type=float,
        default=0.0,
        help="Anchor non-AST emergent flow channels to a frozen teacher branch.",
    )
    parser.add_argument(
        "--flow-anchor-target-scale",
        type=float,
        default=8.0,
        help="Normalization scale for the non-AST flow anchor Huber loss.",
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
        "--props-aux-ramp-epochs",
        type=int,
        default=0,
        help="Linearly ramp props-line aux weights over N epochs (0 disables).",
    )
    parser.add_argument(
        "--props-aux-start-scale",
        type=float,
        default=1.0,
        help="Initial scale for props-line aux weight ramp in [0,1].",
    )
    parser.add_argument(
        "--direct-stat-aux-ramp-epochs",
        type=int,
        default=0,
        help="Linearly ramp direct stat aux weights over N epochs (0 disables).",
    )
    parser.add_argument(
        "--direct-stat-aux-start-scale",
        type=float,
        default=1.0,
        help="Initial scale for direct stat aux weight ramp in [0,1].",
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
        "--props-pts-target-scale",
        type=float,
        default=8.0,
        help="Normalization scale (points) for points props aux error before Huber loss.",
    )
    parser.add_argument(
        "--props-reb-target-scale",
        type=float,
        default=4.0,
        help="Normalization scale (rebounds) for rebound props aux error before Huber loss.",
    )
    parser.add_argument(
        "--props-ast-target-scale",
        type=float,
        default=3.0,
        help="Normalization scale (assists) for assist props aux error before Huber loss.",
    )
    parser.add_argument(
        "--props-pts-aux-min-line",
        type=float,
        default=0.0,
        help="Only apply points props aux when the line is at least this value (0 disables thresholding).",
    )
    parser.add_argument(
        "--props-reb-aux-min-line",
        type=float,
        default=0.0,
        help="Only apply rebounds props aux when the line is at least this value (0 disables thresholding).",
    )
    parser.add_argument(
        "--props-ast-aux-min-line",
        type=float,
        default=0.0,
        help="Only apply assists props aux when the line is at least this value (0 disables thresholding).",
    )
    parser.add_argument(
        "--team-ast-budget-target-scale",
        type=float,
        default=8.0,
        help="Target scale for team AST budget auxiliary loss.",
    )
    parser.add_argument(
        "--assist-share-recon-target-scale",
        type=float,
        default=3.0,
        help="Target scale for AST reconstruction auxiliary loss.",
    )
    parser.add_argument("--assist-playmaker-line-center", type=float, default=5.5)
    parser.add_argument("--assist-playmaker-line-scale", type=float, default=1.0)
    parser.add_argument("--assist-playmaker-max-weight", type=float, default=3.0)
    parser.add_argument("--assist-underprediction-weight", type=float, default=2.0)
    parser.add_argument(
        "--direct-pts-target-scale",
        type=float,
        default=8.0,
        help="Normalization scale (points) for direct points aux error before Huber loss.",
    )
    parser.add_argument(
        "--direct-reb-target-scale",
        type=float,
        default=4.0,
        help="Normalization scale (rebounds) for direct rebound aux error before Huber loss.",
    )
    parser.add_argument(
        "--direct-ast-target-scale",
        type=float,
        default=3.0,
        help="Normalization scale (assists) for direct assist aux error before Huber loss.",
    )
    parser.add_argument(
        "--direct-stl-target-scale",
        type=float,
        default=1.5,
        help="Normalization scale (steals) for direct steal aux error before Huber loss.",
    )
    parser.add_argument(
        "--direct-blk-target-scale",
        type=float,
        default=1.5,
        help="Normalization scale (blocks) for direct block aux error before Huber loss.",
    )
    parser.add_argument(
        "--direct-tov-target-scale",
        type=float,
        default=2.0,
        help="Normalization scale (turnovers) for direct turnover aux error before Huber loss.",
    )
    parser.add_argument(
        "--direct-fg3m-target-scale",
        type=float,
        default=2.5,
        help="Normalization scale (made threes) for grouped direct box-score aux error before Huber loss.",
    )
    parser.add_argument(
        "--direct-ftm-target-scale",
        type=float,
        default=3.0,
        help="Normalization scale (made free throws) for grouped direct box-score aux error before Huber loss.",
    )
    parser.add_argument(
        "--direct-fga-target-scale",
        type=float,
        default=8.0,
        help="Normalization scale (field goal attempts) for grouped direct opportunity aux error before Huber loss.",
    )
    parser.add_argument(
        "--direct-fta-target-scale",
        type=float,
        default=4.0,
        help="Normalization scale (free throw attempts) for grouped direct opportunity aux error before Huber loss.",
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
        "--props-aux-huber-delta",
        type=float,
        default=1.0,
        help="Huber delta on normalized props aux error.",
    )
    parser.add_argument(
        "--direct-stat-aux-huber-delta",
        type=float,
        default=1.0,
        help="Huber delta on normalized direct stat aux errors.",
    )
    parser.add_argument(
        "--props-aux-confidence-min",
        type=float,
        default=0.05,
        help="Minimum confidence weight in [0,1] for rows included in props aux losses.",
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
    parser.add_argument("--enable-team-possession-split-head", action="store_true")
    parser.add_argument("--w-poss-nll", type=float, default=1.0)
    parser.add_argument("--w-backbone-nll", type=float, default=1.0)
    parser.add_argument("--w-three-pa-nll", type=float, default=0.5)
    parser.add_argument("--w-team-possession-aux", type=float, default=0.0)
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
        "--team-possession-max-delta",
        type=float,
        default=8.0,
        help="Maximum absolute home-away possession delta the split head can emit.",
    )
    parser.add_argument(
        "--backbone-detach-until-epoch",
        type=int,
        default=0,
        help="Detach backbone from encoder for the first N epochs (0=never detach). "
        "Allows flow head to stabilize before backbone gradients flow into the encoder.",
    )
    parser.add_argument("--efficiency-market-context", action="store_true")
    parser.add_argument("--efficiency-market-hidden", type=int, default=32)
    parser.add_argument("--efficiency-market-alpha", type=float, default=1.0)
    parser.add_argument(
        "--efficiency-sidecar-feature-cols",
        type=str,
        default="",
        help="Comma-separated player-level shooting/matchup prior columns routed only into the efficiency sidecar path.",
    )
    parser.add_argument(
        "--efficiency-sidecar-add-interactions",
        action="store_true",
        help="Add engineered offense-vs-defense matchup deltas for the efficiency sidecar from the requested sidecar priors.",
    )
    parser.add_argument("--efficiency-sidecar-hidden", type=int, default=32)
    parser.add_argument("--efficiency-sidecar-alpha", type=float, default=1.0)
    parser.add_argument("--w-team-ppp-aux", type=float, default=0.0)
    parser.add_argument("--team-ppp-target-scale", type=float, default=0.12)
    parser.add_argument("--w-team-advantage-aux", type=float, default=0.0)
    parser.add_argument("--team-advantage-target-scale", type=float, default=8.0)
    parser.add_argument("--w-team-efficiency-ppp-aux", type=float, default=0.0)
    parser.add_argument("--team-efficiency-ppp-target-scale", type=float, default=0.12)

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
    parser.add_argument(
        "--backbone-env-feature-cols",
        type=str,
        default=",".join(DEFAULT_BACKBONE_ENV_FEATURE_COLS),
        help="Comma-separated player-level environment columns pooled late and fed only to possession/backbone heads.",
    )
    parser.add_argument(
        "--backbone-env-enrich-features",
        action="store_true",
        help="Add richer derived late-fused environment features (implied totals, spread magnitude, matchup deltas).",
    )
    parser.add_argument(
        "--backbone-side-market-context",
        action="store_true",
        help="Inject side-specific implied-total/spread features into backbone team states before event generation.",
    )
    parser.add_argument(
        "--backbone-side-market-hidden",
        type=int,
        default=32,
        help="Hidden width for the side-specific market-context encoder added to backbone team states.",
    )
    parser.add_argument(
        "--enable-env-side-channel",
        action="store_true",
        help="Encode late-fused environment context through a dedicated side-channel MLP and condition flow/backbone on it.",
    )
    parser.add_argument(
        "--env-side-channel-dim",
        type=int,
        default=32,
        help="Embedding size for the environment side-channel encoder.",
    )
    parser.add_argument(
        "--env-side-channel-hidden",
        type=int,
        default=64,
        help="Hidden size for the environment side-channel encoder.",
    )
    parser.add_argument(
        "--backbone-env-adapter-dim",
        type=int,
        default=0,
        help="Optional compressed embedding size for late-fused backbone env context; 0 disables the adapter.",
    )
    parser.add_argument(
        "--backbone-env-adapter-hidden",
        type=int,
        default=DEFAULT_BACKBONE_ENV_ADAPTER_HIDDEN,
        help="Hidden size for the optional late-fused backbone env adapter.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _set_seed(int(args.seed))
    flow_target_schema = normalize_flow_target_schema(str(args.flow_target_schema))
    if bool(args.enable_phase3_decision) and not bool(args.enable_phase2_flow):
        raise ValueError("--enable-phase3-decision requires --enable-phase2-flow")
    if float(args.w_minutes_hurdle_nll) > 0.0 and not bool(args.enable_minutes_hurdle_head):
        raise ValueError("--w-minutes-hurdle-nll > 0 requires --enable-minutes-hurdle-head")
    if float(args.w_role_loss) > 0.0 and not bool(args.enable_minutes_role_head):
        raise ValueError("--w-role-loss > 0 requires --enable-minutes-role-head")
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
    if int(args.flow_num_blocks) <= 0:
        raise ValueError("--flow-num-blocks must be > 0")
    if float(args.flow_scale_clip) <= 0.0:
        raise ValueError("--flow-scale-clip must be > 0")
    if int(args.flow_rqs_num_bins) <= 1:
        raise ValueError("--flow-rqs-num-bins must be > 1")
    if float(args.flow_rqs_tail_bound) <= 0.0:
        raise ValueError("--flow-rqs-tail-bound must be > 0")
    if float(args.flow_rqs_min_bin_width) <= 0.0:
        raise ValueError("--flow-rqs-min-bin-width must be > 0")
    if float(args.flow_rqs_min_bin_height) <= 0.0:
        raise ValueError("--flow-rqs-min-bin-height must be > 0")
    if float(args.flow_rqs_min_derivative) <= 0.0:
        raise ValueError("--flow-rqs-min-derivative must be > 0")
    if float(args.flow_rqs_min_bin_width) * float(args.flow_rqs_num_bins) >= 1.0:
        raise ValueError("--flow-rqs-min-bin-width * --flow-rqs-num-bins must be < 1")
    if float(args.flow_rqs_min_bin_height) * float(args.flow_rqs_num_bins) >= 1.0:
        raise ValueError("--flow-rqs-min-bin-height * --flow-rqs-num-bins must be < 1")
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
    if str(args.best_checkpoint_metric) == "val_total_ex_possreg" and float(args.w_poss_regression) <= 0.0:
        raise ValueError("--best-checkpoint-metric=val_total_ex_possreg requires --w-poss-regression > 0")
    if int(args.checkpoint_topk_by_minutes) <= 0:
        raise ValueError("--checkpoint-topk-by-minutes must be > 0")
    if float(args.checkpoint_minutes_mae_tolerance) < 0.0:
        raise ValueError("--checkpoint-minutes-mae-tolerance must be >= 0")
    if float(args.checkpoint_low_minutes_threshold) < 0.0:
        raise ValueError("--checkpoint-low-minutes-threshold must be >= 0")
    if not (0.0 <= float(args.checkpoint_sparse_prior_play_prob_max) <= 1.0):
        raise ValueError("--checkpoint-sparse-prior-play-prob-max must be in [0, 1]")
    if float(args.checkpoint_sparse_prior_minutes_max) < 0.0:
        raise ValueError("--checkpoint-sparse-prior-minutes-max must be >= 0")
    if float(args.checkpoint_starter_promotion_prior_minutes_max) < 0.0:
        raise ValueError("--checkpoint-starter-promotion-prior-minutes-max must be >= 0")
    if not (0.0 <= float(args.checkpoint_starter_promotion_hist_start_rate_max) <= 1.0):
        raise ValueError("--checkpoint-starter-promotion-hist-start-rate-max must be in [0, 1]")
    if float(args.checkpoint_next_up_actual_min) < 0.0:
        raise ValueError("--checkpoint-next-up-actual-min must be >= 0")
    if float(args.checkpoint_next_up_pred_min) < 0.0:
        raise ValueError("--checkpoint-next-up-pred-min must be >= 0")
    if float(args.checkpoint_sparse_target_starter_sparse_minutes) < 0.0:
        raise ValueError("--checkpoint-sparse-target-starter-sparse-minutes must be >= 0")
    if float(args.checkpoint_sparse_weight_underpred) < 0.0:
        raise ValueError("--checkpoint-sparse-weight-underpred must be >= 0")
    if float(args.checkpoint_sparse_weight_active_count_mae) < 0.0:
        raise ValueError("--checkpoint-sparse-weight-active-count-mae must be >= 0")
    if float(args.checkpoint_sparse_weight_starter_shortfall) < 0.0:
        raise ValueError("--checkpoint-sparse-weight-starter-shortfall must be >= 0")
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
    if int(args.minutes_hurdle_hidden) <= 0:
        raise ValueError("--minutes-hurdle-hidden must be > 0")
    if float(args.minutes_hurdle_sigma_floor) <= 0.0:
        raise ValueError("--minutes-hurdle-sigma-floor must be > 0")
    if float(args.minutes_hurdle_zero_threshold) < 0.0:
        raise ValueError("--minutes-hurdle-zero-threshold must be >= 0")
    if float(args.w_minutes_hurdle_nll) < 0.0:
        raise ValueError("--w-minutes-hurdle-nll must be >= 0")
    if int(args.minutes_role_hidden) <= 0:
        raise ValueError("--minutes-role-hidden must be > 0")
    if int(args.minutes_role_embedding_dim) <= 0:
        raise ValueError("--minutes-role-embedding-dim must be > 0")
    if int(args.minutes_role_num_classes) < 2:
        raise ValueError("--minutes-role-num-classes must be >= 2")
    if float(args.w_role_loss) < 0.0:
        raise ValueError("--w-role-loss must be >= 0")
    if float(args.w_starter_promotion_loss) > 0.0 and not bool(args.enable_starter_promotion_head):
        raise ValueError("--w-starter-promotion-loss > 0 requires --enable-starter-promotion-head")
    if int(args.starter_promotion_hidden_dim) <= 0:
        raise ValueError("--starter-promotion-hidden-dim must be > 0")
    if float(args.w_starter_promotion_loss) < 0.0:
        raise ValueError("--w-starter-promotion-loss must be >= 0")
    if float(args.starter_promotion_prior_minutes_max) < 0.0:
        raise ValueError("--starter-promotion-prior-minutes-max must be >= 0")
    if float(args.w_sparse_starter_underpred_loss) < 0.0:
        raise ValueError("--w-sparse-starter-underpred-loss must be >= 0")
    if not (0.0 <= float(args.sparse_starter_loss_prior_play_prob_max) <= 1.0):
        raise ValueError("--sparse-starter-loss-prior-play-prob-max must be in [0, 1]")
    if float(args.sparse_starter_loss_prior_minutes_max) < 0.0:
        raise ValueError("--sparse-starter-loss-prior-minutes-max must be >= 0")
    if not (0.0 <= float(args.sparse_starter_loss_hist_start_rate_max) <= 1.0):
        raise ValueError("--sparse-starter-loss-hist-start-rate-max must be in [0, 1]")
    if float(args.sparse_starter_loss_actual_min_threshold) < 0.0:
        raise ValueError("--sparse-starter-loss-actual-min-threshold must be >= 0")
    if float(args.w_bench_riser_underpred_loss) < 0.0:
        raise ValueError("--w-bench-riser-underpred-loss must be >= 0")
    if float(args.bench_riser_loss_prior_minutes_min) < 0.0:
        raise ValueError("--bench-riser-loss-prior-minutes-min must be >= 0")
    if not (0.0 <= float(args.bench_riser_loss_prior_play_prob_min) <= 1.0):
        raise ValueError("--bench-riser-loss-prior-play-prob-min must be in [0, 1]")
    if not (0.0 <= float(args.bench_riser_loss_hist_start_rate_max) <= 1.0):
        raise ValueError("--bench-riser-loss-hist-start-rate-max must be in [0, 1]")
    if float(args.bench_riser_loss_actual_min_threshold) < 0.0:
        raise ValueError("--bench-riser-loss-actual-min-threshold must be >= 0")
    if float(args.bench_candidate_sample_weight) < 1.0:
        raise ValueError("--bench-candidate-sample-weight must be >= 1")
    if float(args.bench_candidate_sample_prior_minutes_min) < 0.0:
        raise ValueError("--bench-candidate-sample-prior-minutes-min must be >= 0")
    if not (0.0 <= float(args.bench_candidate_sample_prior_play_prob_min) <= 1.0):
        raise ValueError("--bench-candidate-sample-prior-play-prob-min must be in [0, 1]")
    if not (0.0 <= float(args.bench_candidate_sample_hist_start_rate_max) <= 1.0):
        raise ValueError("--bench-candidate-sample-hist-start-rate-max must be in [0, 1]")
    if int(args.usage_share_head_hidden) <= 0:
        raise ValueError("--usage-share-head-hidden must be > 0")
    if int(args.team_points_budget_head_hidden) <= 0:
        raise ValueError("--team-points-budget-head-hidden must be > 0")
    if int(args.team_ppp_head_hidden) <= 0:
        raise ValueError("--team-ppp-head-hidden must be > 0")
    if int(args.team_advantage_head_hidden) <= 0:
        raise ValueError("--team-advantage-head-hidden must be > 0")
    if int(args.team_ppp_latent_hidden) <= 0:
        raise ValueError("--team-ppp-latent-hidden must be > 0")
    if not (0.0 <= float(args.team_ppp_backbone_alpha) <= 1.0):
        raise ValueError("--team-ppp-backbone-alpha must be in [0, 1]")
    if not (0.0 <= float(args.team_ppp_efficiency_alpha) <= 1.0):
        raise ValueError("--team-ppp-efficiency-alpha must be in [0, 1]")
    if int(args.team_points_budget_latent_hidden) <= 0:
        raise ValueError("--team-points-budget-latent-hidden must be > 0")
    if not (0.0 <= float(args.team_points_reconcile_alpha) <= 1.0):
        raise ValueError("--team-points-reconcile-alpha must be in [0, 1]")
    if int(args.team_opportunity_budget_latent_hidden) <= 0:
        raise ValueError("--team-opportunity-budget-latent-hidden must be > 0")
    if not (0.0 <= float(args.team_opportunity_budget_backbone_alpha) <= 1.0):
        raise ValueError("--team-opportunity-budget-backbone-alpha must be in [0, 1]")
    if not (0.0 <= float(args.team_opportunity_reconcile_alpha) <= 1.0):
        raise ValueError("--team-opportunity-reconcile-alpha must be in [0, 1]")
    if int(args.team_ast_budget_head_hidden) <= 0:
        raise ValueError("--team-ast-budget-head-hidden must be > 0")
    if int(args.assist_share_head_hidden) <= 0:
        raise ValueError("--assist-share-head-hidden must be > 0")
    if int(args.team_rebound_budget_head_hidden) <= 0:
        raise ValueError("--team-rebound-budget-head-hidden must be > 0")
    if int(args.rebound_budget_blend_gate_hidden) <= 0:
        raise ValueError("--rebound-budget-blend-gate-hidden must be > 0")
    if not (0.0 < float(args.rebound_budget_blend_gate_init_alpha) < 1.0):
        raise ValueError("--rebound-budget-blend-gate-init-alpha must be in (0, 1)")
    if float(args.rebound_oreb_rate_cap) < 0.0 or float(args.rebound_oreb_rate_cap) > 1.0:
        raise ValueError("--rebound-oreb-rate-cap must be within [0, 1]")
    if float(args.rebound_dreb_rate_cap) < 0.0 or float(args.rebound_dreb_rate_cap) > 1.0:
        raise ValueError("--rebound-dreb-rate-cap must be within [0, 1]")
    if float(args.rebound_dreb_deterministic_discount) < 0.0 or float(args.rebound_dreb_deterministic_discount) > 1.0:
        raise ValueError("--rebound-dreb-deterministic-discount must be within [0, 1]")
    if not (0.0 <= float(args.rebound_oreb_budget_blend_alpha) <= 1.0):
        raise ValueError("--rebound-oreb-budget-blend-alpha must be in [0, 1]")
    if not (0.0 <= float(args.rebound_dreb_budget_blend_alpha) <= 1.0):
        raise ValueError("--rebound-dreb-budget-blend-alpha must be in [0, 1]")
    if float(args.rebound_budget_blend_gate_target_eps) < 0.0:
        raise ValueError("--rebound-budget-blend-gate-target-eps must be >= 0")
    if int(args.rebound_share_head_hidden) <= 0:
        raise ValueError("--rebound-share-head-hidden must be > 0")
    if int(args.rebound_share_condition_hidden) <= 0:
        raise ValueError("--rebound-share-condition-hidden must be > 0")
    if int(args.assist_share_condition_hidden) <= 0:
        raise ValueError("--assist-share-condition-hidden must be > 0")
    if int(args.ast_blend_gate_hidden) <= 0:
        raise ValueError("--ast-blend-gate-hidden must be > 0")
    if not (0.0 < float(args.ast_blend_gate_init_alpha) < 1.0):
        raise ValueError("--ast-blend-gate-init-alpha must be in (0, 1)")
    if not (0.0 <= float(args.assist_share_reconcile_alpha) <= 1.0):
        raise ValueError("--assist-share-reconcile-alpha must be in [0, 1]")
    if float(args.assist_share_reconcile_temperature) <= 0.0:
        raise ValueError("--assist-share-reconcile-temperature must be > 0")
    if not (0.0 <= float(args.rebound_factor_reconcile_alpha) <= 1.0):
        raise ValueError("--rebound-factor-reconcile-alpha must be in [0, 1]")
    if str(args.rebound_factor_reconcile_mode) not in {"both", "dreb_only", "oreb_only"}:
        raise ValueError("--rebound-factor-reconcile-mode must be one of: both, dreb_only, oreb_only")
    if float(args.rebound_factor_reconcile_temperature) <= 0.0:
        raise ValueError("--rebound-factor-reconcile-temperature must be > 0")
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
    if float(args.lineup_available_sample_weight) < 1.0:
        raise ValueError("--lineup-available-sample-weight must be >= 1.0")
    if float(args.sparse_candidate_sample_weight) < 1.0:
        raise ValueError("--sparse-candidate-sample-weight must be >= 1.0")
    if float(args.sparse_candidate_sample_prior_minutes_max) < 0.0:
        raise ValueError("--sparse-candidate-sample-prior-minutes-max must be >= 0")
    if not (0.0 <= float(args.sparse_candidate_sample_hist_start_rate_max) <= 1.0):
        raise ValueError("--sparse-candidate-sample-hist-start-rate-max must be in [0, 1]")
    if bool(args.flow_use_minutes_conditioning) and not bool(args.enable_phase2_flow):
        raise ValueError("--flow-use-minutes-conditioning requires --enable-phase2-flow")
    if float(args.flow_minutes_teacher_forcing_prob_start) < 0.0 or float(args.flow_minutes_teacher_forcing_prob_start) > 1.0:
        raise ValueError("--flow-minutes-teacher-forcing-prob-start must be in [0, 1]")
    if float(args.flow_minutes_teacher_forcing_prob_end) < 0.0 or float(args.flow_minutes_teacher_forcing_prob_end) > 1.0:
        raise ValueError("--flow-minutes-teacher-forcing-prob-end must be in [0, 1]")
    if int(args.flow_minutes_teacher_forcing_ramp_epochs) < 1:
        raise ValueError("--flow-minutes-teacher-forcing-ramp-epochs must be >= 1")
    if float(args.w_efficiency_nll) < 0.0:
        raise ValueError("--w-efficiency-nll must be >= 0")
    if int(args.efficiency_market_hidden) <= 0:
        raise ValueError("--efficiency-market-hidden must be > 0")
    if not (0.0 <= float(args.efficiency_market_alpha) <= 1.0):
        raise ValueError("--efficiency-market-alpha must be in [0, 1]")
    if int(args.efficiency_sidecar_hidden) <= 0:
        raise ValueError("--efficiency-sidecar-hidden must be > 0")
    if float(args.efficiency_sidecar_alpha) < 0.0:
        raise ValueError("--efficiency-sidecar-alpha must be >= 0")
    if float(args.w_team_ppp_aux) < 0.0:
        raise ValueError("--w-team-ppp-aux must be >= 0")
    if float(args.team_ppp_target_scale) <= 0.0:
        raise ValueError("--team-ppp-target-scale must be > 0")
    if float(args.w_team_advantage_aux) < 0.0:
        raise ValueError("--w-team-advantage-aux must be >= 0")
    if float(args.team_advantage_target_scale) <= 0.0:
        raise ValueError("--team-advantage-target-scale must be > 0")
    if float(args.w_team_efficiency_ppp_aux) < 0.0:
        raise ValueError("--w-team-efficiency-ppp-aux must be >= 0")
    if float(args.team_efficiency_ppp_target_scale) <= 0.0:
        raise ValueError("--team-efficiency-ppp-target-scale must be > 0")
    if str(args.efficiency_sidecar_feature_cols).strip() and not bool(args.enable_efficiency_head):
        raise ValueError("--efficiency-sidecar-feature-cols requires --enable-efficiency-head")
    if float(args.w_usage_share_nll) < 0.0:
        raise ValueError("--w-usage-share-nll must be >= 0")
    if float(args.w_team_possession_aux) < 0.0:
        raise ValueError("--w-team-possession-aux must be >= 0")
    if float(args.team_possession_max_delta) <= 0.0:
        raise ValueError("--team-possession-max-delta must be > 0")
    if float(args.w_team_points_budget_aux) < 0.0:
        raise ValueError("--w-team-points-budget-aux must be >= 0")
    if float(args.w_team_ast_budget_aux) < 0.0:
        raise ValueError("--w-team-ast-budget-aux must be >= 0")
    if float(args.w_assist_share_aux) < 0.0:
        raise ValueError("--w-assist-share-aux must be >= 0")
    if float(args.w_assist_share_recon_aux) < 0.0:
        raise ValueError("--w-assist-share-recon-aux must be >= 0")
    if float(args.w_ast_blend_gate_aux) < 0.0:
        raise ValueError("--w-ast-blend-gate-aux must be >= 0")
    if float(args.ast_blend_gate_target_eps) < 0.0:
        raise ValueError("--ast-blend-gate-target-eps must be >= 0")
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
    if float(args.w_team_rebound_budget_rate_aux) < 0.0:
        raise ValueError("--w-team-rebound-budget-rate-aux must be >= 0")
    if float(args.w_rebound_budget_blend_gate_aux) < 0.0:
        raise ValueError("--w-rebound-budget-blend-gate-aux must be >= 0")
    if float(args.w_spread_aux) < 0.0:
        raise ValueError("--w-spread-aux must be >= 0")
    if float(args.w_total_aux) < 0.0:
        raise ValueError("--w-total-aux must be >= 0")
    if float(args.w_props_pts_aux) < 0.0:
        raise ValueError("--w-props-pts-aux must be >= 0")
    if float(args.w_props_reb_aux) < 0.0:
        raise ValueError("--w-props-reb-aux must be >= 0")
    if float(args.w_props_ast_aux) < 0.0:
        raise ValueError("--w-props-ast-aux must be >= 0")
    if float(args.w_direct_pts_aux) < 0.0:
        raise ValueError("--w-direct-pts-aux must be >= 0")
    if float(args.w_direct_reb_aux) < 0.0:
        raise ValueError("--w-direct-reb-aux must be >= 0")
    if float(args.w_direct_ast_aux) < 0.0:
        raise ValueError("--w-direct-ast-aux must be >= 0")
    if float(args.w_direct_stl_aux) < 0.0:
        raise ValueError("--w-direct-stl-aux must be >= 0")
    if float(args.w_direct_blk_aux) < 0.0:
        raise ValueError("--w-direct-blk-aux must be >= 0")
    if float(args.w_direct_tov_aux) < 0.0:
        raise ValueError("--w-direct-tov-aux must be >= 0")
    if float(args.w_direct_boxscore_aux) < 0.0:
        raise ValueError("--w-direct-boxscore-aux must be >= 0")
    if float(args.w_direct_opportunity_aux) < 0.0:
        raise ValueError("--w-direct-opportunity-aux must be >= 0")
    if int(args.spread_total_aux_ramp_epochs) < 0:
        raise ValueError("--spread-total-aux-ramp-epochs must be >= 0")
    if float(args.spread_total_aux_start_scale) < 0.0 or float(args.spread_total_aux_start_scale) > 1.0:
        raise ValueError("--spread-total-aux-start-scale must be in [0, 1]")
    if int(args.props_aux_ramp_epochs) < 0:
        raise ValueError("--props-aux-ramp-epochs must be >= 0")
    if float(args.props_aux_start_scale) < 0.0 or float(args.props_aux_start_scale) > 1.0:
        raise ValueError("--props-aux-start-scale must be in [0, 1]")
    if int(args.direct_stat_aux_ramp_epochs) < 0:
        raise ValueError("--direct-stat-aux-ramp-epochs must be >= 0")
    if float(args.direct_stat_aux_start_scale) < 0.0 or float(args.direct_stat_aux_start_scale) > 1.0:
        raise ValueError("--direct-stat-aux-start-scale must be in [0, 1]")
    if float(args.spread_aux_target_scale) <= 0.0:
        raise ValueError("--spread-aux-target-scale must be > 0")
    if float(args.total_aux_target_scale) <= 0.0:
        raise ValueError("--total-aux-target-scale must be > 0")
    if float(args.props_pts_target_scale) <= 0.0:
        raise ValueError("--props-pts-target-scale must be > 0")
    if float(args.props_reb_target_scale) <= 0.0:
        raise ValueError("--props-reb-target-scale must be > 0")
    if float(args.props_ast_target_scale) <= 0.0:
        raise ValueError("--props-ast-target-scale must be > 0")
    if float(args.props_pts_aux_min_line) < 0.0:
        raise ValueError("--props-pts-aux-min-line must be >= 0")
    if float(args.props_reb_aux_min_line) < 0.0:
        raise ValueError("--props-reb-aux-min-line must be >= 0")
    if float(args.props_ast_aux_min_line) < 0.0:
        raise ValueError("--props-ast-aux-min-line must be >= 0")
    if float(args.direct_pts_target_scale) <= 0.0:
        raise ValueError("--direct-pts-target-scale must be > 0")
    if float(args.team_ast_budget_target_scale) <= 0.0:
        raise ValueError("--team-ast-budget-target-scale must be > 0")
    if float(args.assist_share_recon_target_scale) <= 0.0:
        raise ValueError("--assist-share-recon-target-scale must be > 0")
    if float(args.assist_playmaker_line_scale) <= 0.0:
        raise ValueError("--assist-playmaker-line-scale must be > 0")
    if float(args.assist_playmaker_max_weight) < 1.0:
        raise ValueError("--assist-playmaker-max-weight must be >= 1")
    if float(args.assist_underprediction_weight) < 1.0:
        raise ValueError("--assist-underprediction-weight must be >= 1")
    if float(args.direct_reb_target_scale) <= 0.0:
        raise ValueError("--direct-reb-target-scale must be > 0")
    if float(args.direct_ast_target_scale) <= 0.0:
        raise ValueError("--direct-ast-target-scale must be > 0")
    if float(args.direct_stl_target_scale) <= 0.0:
        raise ValueError("--direct-stl-target-scale must be > 0")
    if float(args.direct_blk_target_scale) <= 0.0:
        raise ValueError("--direct-blk-target-scale must be > 0")
    if float(args.direct_tov_target_scale) <= 0.0:
        raise ValueError("--direct-tov-target-scale must be > 0")
    if float(args.direct_fg3m_target_scale) <= 0.0:
        raise ValueError("--direct-fg3m-target-scale must be > 0")
    if float(args.direct_ftm_target_scale) <= 0.0:
        raise ValueError("--direct-ftm-target-scale must be > 0")
    if float(args.direct_fga_target_scale) <= 0.0:
        raise ValueError("--direct-fga-target-scale must be > 0")
    if float(args.direct_fta_target_scale) <= 0.0:
        raise ValueError("--direct-fta-target-scale must be > 0")
    if float(args.spread_aux_huber_delta) <= 0.0:
        raise ValueError("--spread-aux-huber-delta must be > 0")
    if float(args.total_aux_huber_delta) <= 0.0:
        raise ValueError("--total-aux-huber-delta must be > 0")
    if float(args.props_aux_huber_delta) <= 0.0:
        raise ValueError("--props-aux-huber-delta must be > 0")
    if float(args.direct_stat_aux_huber_delta) <= 0.0:
        raise ValueError("--direct-stat-aux-huber-delta must be > 0")
    if float(args.props_aux_confidence_min) < 0.0 or float(args.props_aux_confidence_min) > 1.0:
        raise ValueError("--props-aux-confidence-min must be in [0, 1]")
    if float(args.w_efficiency_mean_aux) < 0.0:
        raise ValueError("--w-efficiency-mean-aux must be >= 0")
    if float(args.w_emergent_share_aux) > 0.0 and not bool(args.enable_phase2_flow):
        raise ValueError("--w-emergent-share-aux > 0 requires --enable-phase2-flow")
    if float(args.w_team_points_budget_aux) > 0.0 and not bool(args.enable_team_points_budget_head):
        raise ValueError("--w-team-points-budget-aux > 0 requires --enable-team-points-budget-head")
    if float(args.w_team_ppp_aux) > 0.0 and not bool(args.enable_team_ppp_head):
        raise ValueError("--w-team-ppp-aux > 0 requires --enable-team-ppp-head")
    if float(args.w_team_advantage_aux) > 0.0 and not bool(args.enable_team_advantage_head):
        raise ValueError("--w-team-advantage-aux > 0 requires --enable-team-advantage-head")
    if float(args.w_team_possession_aux) > 0.0 and not bool(args.enable_team_possession_split_head):
        raise ValueError("--w-team-possession-aux > 0 requires --enable-team-possession-split-head")
    if bool(args.team_ppp_to_backbone) and not bool(args.enable_possession_backbone):
        raise ValueError("--team-ppp-to-backbone requires --enable-possession-backbone")
    if bool(args.team_ppp_direct_backbone_context):
        if not bool(args.enable_possession_backbone):
            raise ValueError("--team-ppp-direct-backbone-context requires --enable-possession-backbone")
        if not bool(args.enable_team_ppp_head):
            raise ValueError("--team-ppp-direct-backbone-context requires --enable-team-ppp-head")
    if bool(args.team_ppp_direct_efficiency_context):
        if not bool(args.enable_efficiency_head):
            raise ValueError("--team-ppp-direct-efficiency-context requires --enable-efficiency-head")
        if not bool(args.enable_team_ppp_head):
            raise ValueError("--team-ppp-direct-efficiency-context requires --enable-team-ppp-head")
    if bool(args.team_advantage_direct_backbone_context):
        if not bool(args.enable_possession_backbone):
            raise ValueError("--team-advantage-direct-backbone-context requires --enable-possession-backbone")
        if not bool(args.enable_team_advantage_head):
            raise ValueError("--team-advantage-direct-backbone-context requires --enable-team-advantage-head")
    if bool(args.team_ppp_to_efficiency) and not bool(args.enable_efficiency_head):
        raise ValueError("--team-ppp-to-efficiency requires --enable-efficiency-head")
    if bool(args.team_points_budget_to_backbone):
        if (
            str(args.team_points_budget_parameterization).strip().lower() == "absolute"
            and not bool(args.enable_team_points_budget_head)
        ):
            raise ValueError(
                "--team-points-budget-to-backbone requires --enable-team-points-budget-head when using absolute budgets"
            )
        if not bool(args.enable_possession_backbone):
            raise ValueError("--team-points-budget-to-backbone requires --enable-possession-backbone")
    if (
        bool(args.team_points_reconcile_budget)
        and str(args.team_points_budget_parameterization).strip().lower() == "absolute"
        and not bool(args.enable_team_points_budget_head)
    ):
        raise ValueError(
            "--team-points-reconcile-budget requires --enable-team-points-budget-head when using absolute budgets"
        )
    if bool(args.team_opportunity_budget_to_backbone):
        if not bool(args.enable_possession_backbone):
            raise ValueError("--team-opportunity-budget-to-backbone requires --enable-possession-backbone")
        if str(args.team_opportunity_budget_parameterization).strip().lower() == "absolute":
            raise ValueError(
                "--team-opportunity-budget-to-backbone currently requires "
                "--team-opportunity-budget-parameterization=market_implied_share"
            )
    if (
        float(args.w_team_ast_budget_aux) > 0.0
        or float(args.w_assist_share_aux) > 0.0
        or float(args.w_assist_share_recon_aux) > 0.0
    ) and not bool(args.enable_phase2_flow):
        raise ValueError("AST factorization auxiliary losses require --enable-phase2-flow")
    if (
        float(args.w_team_ast_budget_aux) > 0.0 or float(args.w_assist_share_recon_aux) > 0.0
    ) and not bool(args.enable_team_ast_budget_head):
        raise ValueError("AST budget losses require --enable-team-ast-budget-head")
    if (
        float(args.w_assist_share_aux) > 0.0 or float(args.w_assist_share_recon_aux) > 0.0
    ) and not bool(args.enable_assist_share_head):
        raise ValueError("Assist share losses require --enable-assist-share-head")
    if bool(args.assist_share_factorized_ast):
        if not bool(args.enable_phase2_flow):
            raise ValueError("--assist-share-factorized-ast requires --enable-phase2-flow")
        if not bool(args.enable_team_ast_budget_head):
            raise ValueError("--assist-share-factorized-ast requires --enable-team-ast-budget-head")
        if not bool(args.enable_assist_share_head):
            raise ValueError("--assist-share-factorized-ast requires --enable-assist-share-head")
    if bool(args.assist_share_reconcile_ast_budget):
        if not bool(args.enable_phase2_flow):
            raise ValueError("--assist-share-reconcile-ast-budget requires --enable-phase2-flow")
        if not bool(args.enable_team_ast_budget_head):
            raise ValueError("--assist-share-reconcile-ast-budget requires --enable-team-ast-budget-head")
        if not bool(args.enable_assist_share_head):
            raise ValueError("--assist-share-reconcile-ast-budget requires --enable-assist-share-head")
    if bool(args.rebound_factor_reconcile_oreb_dreb):
        if not bool(args.enable_phase2_flow):
            raise ValueError("--rebound-factor-reconcile-oreb-dreb requires --enable-phase2-flow")
        deterministic_dreb_only = (
            str(args.rebound_factor_reconcile_mode).strip().lower() == "dreb_only"
            and str(args.rebound_budget_parameterization).strip().lower() == "dreb_deterministic"
        )
        if not deterministic_dreb_only and not bool(args.enable_team_rebound_budget_head):
            raise ValueError("--rebound-factor-reconcile-oreb-dreb requires --enable-team-rebound-budget-head")
        if not bool(args.enable_rebound_share_head):
            raise ValueError("--rebound-factor-reconcile-oreb-dreb requires --enable-rebound-share-head")
    if bool(args.enable_rebound_budget_blend_gate) and not bool(args.enable_team_rebound_budget_head):
        raise ValueError("--enable-rebound-budget-blend-gate requires --enable-team-rebound-budget-head")
    if bool(args.enable_ast_blend_gate):
        if not bool(args.assist_share_factorized_ast):
            raise ValueError("--enable-ast-blend-gate requires --assist-share-factorized-ast")
        if not bool(args.enable_team_ast_budget_head):
            raise ValueError("--enable-ast-blend-gate requires --enable-team-ast-budget-head")
        if not bool(args.enable_assist_share_head):
            raise ValueError("--enable-ast-blend-gate requires --enable-assist-share-head")
    if float(args.w_ast_blend_gate_aux) > 0.0 and not bool(args.enable_ast_blend_gate):
        raise ValueError("--w-ast-blend-gate-aux > 0 requires --enable-ast-blend-gate")
    if (
        float(args.w_ast_share_aux) > 0.0
        or float(args.w_reb_share_aux) > 0.0
        or float(args.w_ast_team_rate_aux) > 0.0
        or float(args.w_reb_opportunity_rate_aux) > 0.0
        or float(args.w_rebound_budget_blend_gate_aux) > 0.0
    ) and not bool(args.enable_phase2_flow):
        raise ValueError("AST/REB structure auxiliary losses require --enable-phase2-flow")
    if float(args.w_rebound_budget_blend_gate_aux) > 0.0 and not bool(args.enable_rebound_budget_blend_gate):
        raise ValueError("--w-rebound-budget-blend-gate-aux > 0 requires --enable-rebound-budget-blend-gate")
    if (float(args.w_spread_aux) > 0.0 or float(args.w_total_aux) > 0.0) and not bool(args.enable_phase2_flow):
        raise ValueError("--w-spread-aux/--w-total-aux require --enable-phase2-flow")
    if (
        float(args.w_props_pts_aux) > 0.0
        or float(args.w_props_reb_aux) > 0.0
        or float(args.w_props_ast_aux) > 0.0
    ) and not bool(args.enable_phase2_flow):
        raise ValueError("--w-props-*-aux requires --enable-phase2-flow")
    if (
        float(args.w_direct_pts_aux) > 0.0
        or float(args.w_direct_reb_aux) > 0.0
        or float(args.w_direct_ast_aux) > 0.0
        or float(args.w_direct_stl_aux) > 0.0
        or float(args.w_direct_blk_aux) > 0.0
        or float(args.w_direct_tov_aux) > 0.0
        or float(args.w_direct_boxscore_aux) > 0.0
        or float(args.w_direct_opportunity_aux) > 0.0
    ) and not bool(args.enable_phase2_flow):
        raise ValueError("--w-direct-*-aux requires --enable-phase2-flow")
    if float(args.w_efficiency_mean_aux) > 0.0 and not bool(args.enable_efficiency_head):
        raise ValueError("--w-efficiency-mean-aux requires --enable-efficiency-head")
    if float(args.w_team_efficiency_ppp_aux) > 0.0 and not bool(args.enable_efficiency_head):
        raise ValueError("--w-team-efficiency-ppp-aux > 0 requires --enable-efficiency-head")
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

    if args.feature_columns_json:
        feature_cols = _load_feature_columns_override(str(args.feature_columns_json))
    else:
        feature_cols = _infer_feature_columns(features_df, labels_minutes_df)
        exclude_feature_patterns = list(args.exclude_feature_pattern or [])
        if bool(args.context_priors_curated_only):
            exclude_feature_patterns.extend(
                [
                    r"(^|_)ctx_same_pos_(thin|normal|deep)_prior_",
                    r"_ctx_same_pos_(thin|normal|deep)_prior_",
                ]
            )
        feature_cols = _exclude_feature_patterns(
            feature_cols,
            exclude_patterns=exclude_feature_patterns,
        )
    train_df, val_df, split_meta = _split_train_val(merged, val_days=int(args.val_days))

    feature_mean, feature_std = _compute_feature_norm(train_df, feature_cols)
    feature_index = {col: idx for idx, col in enumerate(feature_cols)}
    efficiency_sidecar_feature_cols = [
        c.strip() for c in str(args.efficiency_sidecar_feature_cols).split(",") if c.strip()
    ]
    if bool(args.efficiency_sidecar_add_interactions):
        merged, derived_efficiency_sidecar_feature_cols = _add_efficiency_sidecar_interaction_features(
            merged,
            sidecar_feature_columns=efficiency_sidecar_feature_cols,
        )
        efficiency_sidecar_feature_cols = list(
            dict.fromkeys([*efficiency_sidecar_feature_cols, *derived_efficiency_sidecar_feature_cols])
        )
    overlapping_efficiency_sidecar_feature_cols = [
        c for c in efficiency_sidecar_feature_cols if c in feature_index
    ]
    if overlapping_efficiency_sidecar_feature_cols:
        raise ValueError(
            "efficiency sidecar columns must be excluded from the generic player feature stack: "
            f"{overlapping_efficiency_sidecar_feature_cols}"
        )
    efficiency_sidecar_feature_mean = np.zeros((len(efficiency_sidecar_feature_cols),), dtype=np.float32)
    efficiency_sidecar_feature_std = np.ones((len(efficiency_sidecar_feature_cols),), dtype=np.float32)
    if efficiency_sidecar_feature_cols:
        efficiency_sidecar_feature_mean, efficiency_sidecar_feature_std = _compute_feature_norm(
            train_df,
            efficiency_sidecar_feature_cols,
        )
    an_pts_line_idx = int(feature_index.get("an_pts_line", -1))
    an_reb_line_idx = int(feature_index.get("an_reb_line", -1))
    an_ast_line_idx = int(feature_index.get("an_ast_line", -1))
    an_has_pts_idx = int(feature_index.get("an_has_pts", -1))
    an_has_reb_idx = int(feature_index.get("an_has_reb", -1))
    an_has_ast_idx = int(feature_index.get("an_has_ast", -1))
    an_pts_books_idx = int(feature_index.get("an_pts_books", -1))
    an_reb_books_idx = int(feature_index.get("an_reb_books", -1))
    an_ast_books_idx = int(feature_index.get("an_ast_books", -1))
    an_props_market_count_idx = int(feature_index.get("an_props_market_count", -1))
    prior_play_prob_idx = int(feature_index.get("prior_play_prob", -1))
    lineup_starter_announced_idx = int(feature_index.get("lineup_starter_announced", -1))
    recent_start_pct_10_idx = int(feature_index.get("recent_start_pct_10", -1))
    started_proxy_rate_prior_10_idx = int(feature_index.get("started_proxy_rate_prior_10", -1))
    started_proxy_rate_prior_20_idx = int(feature_index.get("started_proxy_rate_prior_20", -1))
    minutes_from_stints_prior_20_idx = int(feature_index.get("minutes_from_stints_prior_20", -1))
    game_feature_cols = [c.strip() for c in str(args.game_feature_cols).split(",") if c.strip()]
    team_feature_cols = [c.strip() for c in str(args.team_feature_cols).split(",") if c.strip()]
    backbone_env_feature_cols = [c.strip() for c in str(args.backbone_env_feature_cols).split(",") if c.strip()]
    assist_share_condition_feature_cols = [
        c.strip() for c in str(args.assist_share_condition_feature_cols).split(",") if c.strip()
    ]
    rebound_share_condition_feature_cols = [
        c.strip() for c in str(args.rebound_share_condition_feature_cols).split(",") if c.strip()
    ]

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
    if float(args.w_team_points_budget_aux) > 0.0 and (vegas_total_idx < 0 or vegas_spread_idx < 0):
        raise ValueError("--w-team-points-budget-aux > 0 requires 'vegas_total' and 'vegas_spread' in --game-feature-cols")
    if (
        str(args.team_points_budget_parameterization).strip().lower() == "market_implied"
        and (
            bool(args.team_points_budget_to_backbone)
            or bool(args.team_points_reconcile_budget)
        )
        and (vegas_total_idx < 0 or vegas_spread_idx < 0)
    ):
        raise ValueError(
            "market-implied team points budgets require 'vegas_total' and 'vegas_spread' in --game-feature-cols"
        )
    if (
        str(args.team_points_budget_parameterization).strip().lower() == "team_ppp_implied"
        and (
            bool(args.team_points_budget_to_backbone)
            or bool(args.team_points_reconcile_budget)
        )
        and not bool(args.enable_team_ppp_head)
    ):
        raise ValueError(
            "team_ppp_implied team points budgets require --enable-team-ppp-head"
        )
    if (
        str(args.team_points_budget_parameterization).strip().lower() == "team_ppp_implied"
        and bool(args.team_points_budget_to_backbone)
        and "estimated_possessions" not in game_feature_cols
    ):
        raise ValueError(
            "team_ppp_implied team points budget backbone context requires 'estimated_possessions' in --game-feature-cols"
        )
    if (
        str(args.team_opportunity_budget_parameterization).strip().lower() == "market_implied_share"
        and (
            bool(args.team_opportunity_reconcile_budget)
            or bool(args.team_opportunity_budget_to_backbone)
        )
        and (vegas_total_idx < 0 or vegas_spread_idx < 0)
    ):
        raise ValueError(
            "market-implied team opportunity share requires 'vegas_total' and 'vegas_spread' in --game-feature-cols"
        )
    if (
        float(args.w_team_ast_budget_aux) > 0.0
        or float(args.w_assist_share_aux) > 0.0
        or float(args.w_assist_share_recon_aux) > 0.0
    ) and an_ast_line_idx < 0:
        raise ValueError("AST factorization losses require feature column 'an_ast_line'")
    if bool(args.enable_assist_share_head):
        missing_assist_share_condition_feature_cols = [
            name for name in assist_share_condition_feature_cols if name not in feature_index
        ]
        if missing_assist_share_condition_feature_cols:
            raise ValueError(
                "Assist-share conditioning columns missing from feature set: "
                f"{missing_assist_share_condition_feature_cols}"
            )
    if bool(args.enable_rebound_share_head):
        missing_rebound_share_condition_feature_cols = [
            name for name in rebound_share_condition_feature_cols if name not in feature_index
        ]
        if missing_rebound_share_condition_feature_cols:
            raise ValueError(
                "Rebound-share conditioning columns missing from feature set: "
                f"{missing_rebound_share_condition_feature_cols}"
            )
    if float(args.w_poss_regression) > 0.0 and estimated_possessions_idx < 0:
        raise ValueError(
            "--w-poss-regression > 0 requires 'estimated_possessions' in --game-feature-cols"
        )
    if float(args.w_total_aux) > 0.0 and vegas_total_idx < 0:
        raise ValueError("--w-total-aux > 0 requires 'vegas_total' in --game-feature-cols")
    if float(args.w_spread_aux) > 0.0 and vegas_spread_idx < 0:
        raise ValueError("--w-spread-aux > 0 requires 'vegas_spread' in --game-feature-cols")
    if float(args.w_props_pts_aux) > 0.0 and an_pts_line_idx < 0:
        raise ValueError("--w-props-pts-aux > 0 requires feature column 'an_pts_line'")
    if float(args.w_props_reb_aux) > 0.0 and an_reb_line_idx < 0:
        raise ValueError("--w-props-reb-aux > 0 requires feature column 'an_reb_line'")
    if float(args.w_props_ast_aux) > 0.0 and an_ast_line_idx < 0:
        raise ValueError("--w-props-ast-aux > 0 requires feature column 'an_ast_line'")

    train_examples = build_game_level_examples(
        train_df,
        feature_columns=feature_cols,
        feature_mean=feature_mean,
        feature_std=feature_std,
        game_feature_columns=game_feature_cols,
        team_feature_columns=team_feature_cols,
        efficiency_sidecar_feature_columns=efficiency_sidecar_feature_cols,
        efficiency_sidecar_feature_mean=efficiency_sidecar_feature_mean,
        efficiency_sidecar_feature_std=efficiency_sidecar_feature_std,
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
        efficiency_sidecar_feature_columns=efficiency_sidecar_feature_cols,
        efficiency_sidecar_feature_mean=efficiency_sidecar_feature_mean,
        efficiency_sidecar_feature_std=efficiency_sidecar_feature_std,
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

    device = _resolve_training_device(str(args.device))
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        device_name = torch.cuda.get_device_name(device)
        print(f"[train_gtv2] device={device} ({device_name})", flush=True)
    else:
        print(f"[train_gtv2] device={device}", flush=True)

    loader_kwargs: dict[str, Any] = {"pin_memory": bool(device.type == "cuda")}
    if max(0, int(args.num_workers)) > 0:
        loader_kwargs["multiprocessing_context"] = "spawn"
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    train_dataset = GameLevelDataset(train_examples)
    train_sampler: WeightedRandomSampler | None = None
    combined_sample_weights: torch.Tensor | None = None
    if float(args.lineup_available_sample_weight) > 1.0:
        sample_weights, sampling_meta = _build_lineup_available_example_sampling_weights(
            train_examples,
            lineup_available_weight=float(args.lineup_available_sample_weight),
        )
        combined_sample_weights = sample_weights if combined_sample_weights is None else (combined_sample_weights * sample_weights)
        print(
            (
                "[train_gtv2] lineup curriculum sampler enabled: "
                f"target_weight={sampling_meta['lineup_weight_target']:.3f} "
                f"lineup_fraction_mean={sampling_meta['lineup_fraction_mean']:.3f} "
                f"sample_weight_range=[{sampling_meta['sample_weight_min']:.3f}, "
                f"{sampling_meta['sample_weight_max']:.3f}]"
            ),
            flush=True,
        )
    if float(args.sparse_candidate_sample_weight) > 1.0:
        sparse_sample_weights, sparse_sampling_meta = _build_sparse_candidate_example_sampling_weights(
            train_examples,
            feature_columns=feature_cols,
            feature_mean=feature_mean,
            feature_std=feature_std,
            sparse_candidate_weight=float(args.sparse_candidate_sample_weight),
            prior_minutes_max=float(args.sparse_candidate_sample_prior_minutes_max),
            hist_start_rate_max=float(args.sparse_candidate_sample_hist_start_rate_max),
        )
        combined_sample_weights = (
            sparse_sample_weights
            if combined_sample_weights is None
            else (combined_sample_weights * sparse_sample_weights)
        )
        print(
            (
                "[train_gtv2] sparse candidate sampler enabled: "
                f"target_weight={sparse_sampling_meta['sparse_candidate_weight_target']:.3f} "
                f"candidate_game_rate={sparse_sampling_meta['candidate_game_rate']:.3f} "
                f"sample_weight_range=[{sparse_sampling_meta['sample_weight_min']:.3f}, "
                f"{sparse_sampling_meta['sample_weight_max']:.3f}]"
            ),
            flush=True,
        )
    if float(args.bench_candidate_sample_weight) > 1.0:
        bench_sample_weights, bench_sampling_meta = _build_bench_riser_example_sampling_weights(
            train_examples,
            feature_columns=feature_cols,
            feature_mean=feature_mean,
            feature_std=feature_std,
            bench_candidate_weight=float(args.bench_candidate_sample_weight),
            prior_minutes_min=float(args.bench_candidate_sample_prior_minutes_min),
            hist_start_rate_max=float(args.bench_candidate_sample_hist_start_rate_max),
            prior_play_prob_min=float(args.bench_candidate_sample_prior_play_prob_min),
        )
        combined_sample_weights = (
            bench_sample_weights
            if combined_sample_weights is None
            else (combined_sample_weights * bench_sample_weights)
        )
        print(
            (
                "[train_gtv2] bench-riser candidate sampler enabled: "
                f"target_weight={bench_sampling_meta['bench_candidate_weight_target']:.3f} "
                f"candidate_game_rate={bench_sampling_meta['candidate_game_rate']:.3f} "
                f"sample_weight_range=[{bench_sampling_meta['sample_weight_min']:.3f}, "
                f"{bench_sampling_meta['sample_weight_max']:.3f}]"
            ),
            flush=True,
        )
    if combined_sample_weights is not None:
        train_sampler = WeightedRandomSampler(
            weights=combined_sample_weights,
            num_samples=len(train_dataset),
            replacement=True,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=max(1, int(args.batch_size)),
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=max(0, int(args.num_workers)),
        collate_fn=collate_game_level_examples,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        GameLevelDataset(val_examples),
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        num_workers=max(0, int(args.num_workers)),
        collate_fn=collate_game_level_examples,
        **loader_kwargs,
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
        efficiency_sidecar_feature_columns=efficiency_sidecar_feature_cols,
        efficiency_sidecar_feature_mean=efficiency_sidecar_feature_mean.astype(np.float64).tolist(),
        efficiency_sidecar_feature_std=efficiency_sidecar_feature_std.astype(np.float64).tolist(),
        backbone_env_feature_columns=backbone_env_feature_cols,
        backbone_env_enrich_features=bool(args.backbone_env_enrich_features),
        backbone_side_market_context=bool(args.backbone_side_market_context),
        backbone_side_market_hidden=int(args.backbone_side_market_hidden),
        enable_env_side_channel=bool(args.enable_env_side_channel),
        env_side_channel_dim=int(args.env_side_channel_dim),
        env_side_channel_hidden=int(args.env_side_channel_hidden),
        backbone_env_adapter_dim=int(args.backbone_env_adapter_dim),
        backbone_env_adapter_hidden=int(args.backbone_env_adapter_hidden),
        d_model=int(args.d_model),
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
        num_heads=int(args.num_heads),
        dropout=float(args.dropout),
        min_active_count=int(args.min_active_count),
        max_active_count=int(args.max_active_count),
        active_threshold_minutes=float(args.active_threshold),
        enable_minutes_hurdle_head=bool(args.enable_minutes_hurdle_head),
        minutes_hurdle_hidden=int(args.minutes_hurdle_hidden),
        minutes_hurdle_sigma_floor=float(args.minutes_hurdle_sigma_floor),
        enable_minutes_role_head=bool(args.enable_minutes_role_head),
        minutes_role_use_context_for_preferences=not bool(args.disable_minutes_role_context_for_preferences),
        minutes_role_hidden=int(args.minutes_role_hidden),
        minutes_role_embedding_dim=int(args.minutes_role_embedding_dim),
        minutes_role_num_classes=int(args.minutes_role_num_classes),
        enable_starter_promotion_head=bool(args.enable_starter_promotion_head),
        starter_promotion_hidden_dim=int(args.starter_promotion_hidden_dim),
        include_pf_in_flow_targets=bool(include_pf_in_flow_targets),
        flow_coupling_type=str(args.flow_coupling_type),
        flow_num_blocks=int(args.flow_num_blocks),
        flow_scale_clip=float(args.flow_scale_clip),
        flow_rqs_num_bins=int(args.flow_rqs_num_bins),
        flow_rqs_tail_bound=float(args.flow_rqs_tail_bound),
        flow_rqs_min_bin_width=float(args.flow_rqs_min_bin_width),
        flow_rqs_min_bin_height=float(args.flow_rqs_min_bin_height),
        flow_rqs_min_derivative=float(args.flow_rqs_min_derivative),
        flow_context_mode=str(args.flow_context_mode),
        flow_target_schema=str(flow_target_schema),
        flow_use_minutes_conditioning=bool(args.flow_use_minutes_conditioning),
        enable_efficiency_head=bool(args.enable_efficiency_head),
        efficiency_head_hidden=int(args.efficiency_head_hidden),
        efficiency_ft_prior_mean=float(args.efficiency_ft_prior_mean),
        efficiency_ft_prior_strength=float(args.efficiency_ft_prior_strength),
        efficiency_fg2_prior_mean=float(args.efficiency_fg2_prior_mean),
        efficiency_fg2_prior_strength=float(args.efficiency_fg2_prior_strength),
        efficiency_fg3_prior_mean=float(args.efficiency_fg3_prior_mean),
        efficiency_fg3_prior_strength=float(args.efficiency_fg3_prior_strength),
        efficiency_market_context=bool(args.efficiency_market_context),
        efficiency_market_hidden=int(args.efficiency_market_hidden),
        efficiency_market_alpha=float(args.efficiency_market_alpha),
        efficiency_sidecar_hidden=int(args.efficiency_sidecar_hidden),
        efficiency_sidecar_alpha=float(args.efficiency_sidecar_alpha),
        enable_team_ppp_head=bool(args.enable_team_ppp_head),
        team_ppp_head_hidden=int(args.team_ppp_head_hidden),
        team_ppp_to_backbone=bool(args.team_ppp_to_backbone),
        team_ppp_latent_hidden=int(args.team_ppp_latent_hidden),
        team_ppp_backbone_alpha=float(args.team_ppp_backbone_alpha),
        team_ppp_to_efficiency=bool(args.team_ppp_to_efficiency),
        team_ppp_efficiency_alpha=float(args.team_ppp_efficiency_alpha),
        team_ppp_direct_backbone_context=bool(args.team_ppp_direct_backbone_context),
        team_ppp_direct_efficiency_context=bool(args.team_ppp_direct_efficiency_context),
        enable_team_advantage_head=bool(args.enable_team_advantage_head),
        team_advantage_head_hidden=int(args.team_advantage_head_hidden),
        team_advantage_direct_backbone_context=bool(args.team_advantage_direct_backbone_context),
        enable_usage_share_head=bool(args.enable_usage_share_head),
        usage_share_head_hidden=int(args.usage_share_head_hidden),
        enable_team_points_budget_head=bool(args.enable_team_points_budget_head),
        team_points_budget_head_hidden=int(args.team_points_budget_head_hidden),
        team_points_budget_parameterization=str(args.team_points_budget_parameterization),
        team_points_budget_to_backbone=bool(args.team_points_budget_to_backbone),
        team_points_budget_latent_hidden=int(args.team_points_budget_latent_hidden),
        team_points_reconcile_budget=bool(args.team_points_reconcile_budget),
        team_points_reconcile_alpha=float(args.team_points_reconcile_alpha),
        team_opportunity_budget_parameterization=str(args.team_opportunity_budget_parameterization),
        team_opportunity_budget_to_backbone=bool(args.team_opportunity_budget_to_backbone),
        team_opportunity_budget_latent_hidden=int(args.team_opportunity_budget_latent_hidden),
        team_opportunity_budget_backbone_alpha=float(args.team_opportunity_budget_backbone_alpha),
        team_opportunity_reconcile_budget=bool(args.team_opportunity_reconcile_budget),
        team_opportunity_reconcile_alpha=float(args.team_opportunity_reconcile_alpha),
        team_opportunity_reconcile_preserve_possessions=bool(args.team_opportunity_reconcile_preserve_possessions),
        enable_team_ast_budget_head=bool(args.enable_team_ast_budget_head),
        team_ast_budget_head_hidden=int(args.team_ast_budget_head_hidden),
        enable_assist_share_head=bool(args.enable_assist_share_head),
        assist_share_head_hidden=int(args.assist_share_head_hidden),
        assist_share_condition_feature_columns=list(assist_share_condition_feature_cols),
        assist_share_condition_hidden=int(args.assist_share_condition_hidden),
        enable_team_rebound_budget_head=bool(args.enable_team_rebound_budget_head),
        team_rebound_budget_head_hidden=int(args.team_rebound_budget_head_hidden),
        rebound_budget_parameterization=str(args.rebound_budget_parameterization),
        rebound_oreb_rate_cap=float(args.rebound_oreb_rate_cap),
        rebound_dreb_rate_cap=float(args.rebound_dreb_rate_cap),
        rebound_dreb_deterministic_discount=float(args.rebound_dreb_deterministic_discount),
        rebound_oreb_reconcile_use_flow_budget=bool(args.rebound_oreb_reconcile_use_flow_budget),
        rebound_oreb_budget_blend_alpha=float(args.rebound_oreb_budget_blend_alpha),
        rebound_dreb_budget_blend_alpha=float(args.rebound_dreb_budget_blend_alpha),
        enable_rebound_budget_blend_gate=bool(args.enable_rebound_budget_blend_gate),
        rebound_budget_blend_gate_hidden=int(args.rebound_budget_blend_gate_hidden),
        rebound_budget_blend_gate_init_alpha=float(args.rebound_budget_blend_gate_init_alpha),
        enable_rebound_share_head=bool(args.enable_rebound_share_head),
        rebound_share_head_hidden=int(args.rebound_share_head_hidden),
        rebound_share_condition_feature_columns=list(rebound_share_condition_feature_cols),
        rebound_share_condition_hidden=int(args.rebound_share_condition_hidden),
        enable_ast_blend_gate=bool(args.enable_ast_blend_gate),
        ast_blend_gate_hidden=int(args.ast_blend_gate_hidden),
        ast_blend_gate_init_alpha=float(args.ast_blend_gate_init_alpha),
        assist_share_replace_flow_ast=bool(args.assist_share_replace_flow_ast),
        assist_share_factorized_ast=bool(args.assist_share_factorized_ast),
        assist_share_reconcile_ast_budget=bool(args.assist_share_reconcile_ast_budget),
        assist_share_reconcile_alpha=float(args.assist_share_reconcile_alpha),
        assist_share_reconcile_temperature=float(args.assist_share_reconcile_temperature),
        rebound_factor_reconcile_oreb_dreb=bool(args.rebound_factor_reconcile_oreb_dreb),
        rebound_factor_reconcile_mode=str(args.rebound_factor_reconcile_mode),
        rebound_factor_reconcile_alpha=float(args.rebound_factor_reconcile_alpha),
        rebound_factor_reconcile_temperature=float(args.rebound_factor_reconcile_temperature),
        enable_possession_backbone=bool(args.enable_possession_backbone),
        enable_three_pa_share=bool(args.enable_three_pa_share),
        possession_mu_mode=str(args.possession_mu_mode),
        possession_mu_baseline=float(args.possession_mu_baseline),
        enable_team_possession_split_head=bool(args.enable_team_possession_split_head),
        team_possession_max_delta=float(args.team_possession_max_delta),
        overflow_protected_prior_play_prob_floor=float(args.overflow_protected_prior_play_prob_floor),
        overflow_protected_prior_minutes_floor=float(args.overflow_protected_prior_minutes_floor),
        overflow_risk_weight_consecutive_active_dnp=float(args.overflow_risk_weight_consecutive_active_dnp),
        overflow_risk_weight_active_but_dnp_rate_last10=float(args.overflow_risk_weight_active_but_dnp_rate_last10),
        overflow_risk_weight_inactive_streak_len=float(args.overflow_risk_weight_inactive_streak_len),
        overflow_keep_weight_prior_play_prob=float(args.overflow_keep_weight_prior_play_prob),
        overflow_keep_weight_prior_minutes=float(args.overflow_keep_weight_prior_minutes),
    )

    model = build_game_transformer_v2(config).to(device=device)
    init_model_pt = Path(args.init_model_pt).expanduser().resolve() if args.init_model_pt else None
    graft_model_pt = Path(args.graft_model_pt).expanduser().resolve() if args.graft_model_pt else None
    graft_prefixes = _parse_prefix_csv(args.graft_prefixes)
    freeze_prefixes = _parse_prefix_csv(args.freeze_prefixes)
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
    if graft_model_pt is not None:
        _apply_partial_checkpoint(
            model=model,
            checkpoint_path=graft_model_pt,
            device=device,
            prefixes=graft_prefixes,
            label="graft",
        )
    if freeze_prefixes:
        _freeze_parameter_prefixes(
            model,
            prefixes=freeze_prefixes,
            label="freeze",
        )
    if bool(args.efficiency_head_only):
        n_trainable = 0
        n_total = 0
        for name, param in model.named_parameters():
            n_total += 1
            is_eff = (
                name.startswith("efficiency_head.")
                or name.startswith("efficiency_player_sidecar_encoder.")
                or name.startswith("efficiency_team_market_encoder.")
                or name.startswith("efficiency_team_ppp_encoder.")
            )
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
    flow_anchor_teacher_model: nn.Module | None = None
    if float(args.w_flow_anchor_nonast_aux) > 0.0:
        if not str(args.flow_anchor_teacher_run_dir).strip():
            raise ValueError("--w-flow-anchor-nonast-aux > 0 requires --flow-anchor-teacher-run-dir")
        teacher_cfg, flow_anchor_teacher_model = _load_reference_teacher_model(
            run_dir=Path(str(args.flow_anchor_teacher_run_dir)).expanduser(),
            device=device,
        )
        teacher_flow_columns = list(getattr(flow_anchor_teacher_model, "flow_target_columns"))
        student_flow_columns = list(getattr(model, "flow_target_columns"))
        if teacher_flow_columns != student_flow_columns:
            raise RuntimeError(
                "Flow-anchor teacher flow_target_columns mismatch: "
                f"teacher={teacher_flow_columns} student={student_flow_columns}"
            )
        print(
            (
                "[flow-anchor] "
                f"teacher_run_dir={Path(str(args.flow_anchor_teacher_run_dir)).expanduser()} "
                f"w_nonast={float(args.w_flow_anchor_nonast_aux):.4f} "
                f"target_scale={float(args.flow_anchor_target_scale):.4f} "
                f"teacher_assist_reconcile={bool(getattr(teacher_cfg, 'assist_share_reconcile_ast_budget', False))}"
            ),
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
    checkpoint_candidates_dir = out_dir / "checkpoint_candidates"
    checkpoint_candidates_dir.mkdir(parents=True, exist_ok=True)

    history: list[EpochMetrics] = []
    minutes_checkpoint_candidates: list[MinutesCheckpointCandidate] = []
    best_checkpoint_value = float("inf")
    best_val_total = float("inf")
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
        minutes_teacher_forcing_prob = _resolve_minutes_teacher_forcing_prob(
            epoch=int(epoch),
            start_prob=float(args.minutes_teacher_forcing_prob_start),
            end_prob=float(args.minutes_teacher_forcing_prob_end),
            ramp_epochs=int(args.minutes_teacher_forcing_ramp_epochs),
        )
        flow_minutes_teacher_forcing_prob = _resolve_minutes_teacher_forcing_prob(
            epoch=int(epoch),
            start_prob=float(args.flow_minutes_teacher_forcing_prob_start),
            end_prob=float(args.flow_minutes_teacher_forcing_prob_end),
            ramp_epochs=int(args.flow_minutes_teacher_forcing_ramp_epochs),
        )
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
            flow_label_columns=flow_label_cols,
            run_phase2_flow=bool(phase2_weights.run_phase2_flow),
            run_phase3_decision=bool(phase2_weights.run_phase3_decision),
            w_minutes=float(phase2_weights.w_minutes),
            w_minutes_nll=float(phase2_weights.w_minutes_nll),
            w_minutes_hurdle_nll=float(args.w_minutes_hurdle_nll),
            w_role_loss=float(args.w_role_loss),
            w_starter_promotion_loss=float(args.w_starter_promotion_loss),
            w_sparse_starter_underpred_loss=float(args.w_sparse_starter_underpred_loss),
            w_bench_riser_underpred_loss=float(args.w_bench_riser_underpred_loss),
            minutes_role_target_scheme=str(args.minutes_role_target_scheme),
            w_count=float(phase2_weights.w_count),
            w_member=float(phase2_weights.w_member),
            w_flow_nll=float(phase2_weights.w_flow_nll),
            w_crps_fpts=float(phase2_weights.w_crps_fpts),
            w_team_energy=float(phase2_weights.w_team_energy),
            minutes_nll_sigma=float(args.minutes_nll_sigma),
            minutes_hurdle_zero_threshold=float(args.minutes_hurdle_zero_threshold),
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
            w_team_efficiency_ppp_aux=float(args.w_team_efficiency_ppp_aux),
            w_usage_share_nll=float(args.w_usage_share_nll) if bool(args.enable_usage_share_head) else 0.0,
            w_team_ppp_aux=float(args.w_team_ppp_aux),
            w_team_advantage_aux=float(args.w_team_advantage_aux),
            w_team_possession_aux=float(args.w_team_possession_aux),
            w_team_points_budget_aux=float(args.w_team_points_budget_aux),
            w_team_ast_budget_aux=float(args.w_team_ast_budget_aux),
            w_assist_share_aux=float(args.w_assist_share_aux),
            w_assist_share_recon_aux=float(args.w_assist_share_recon_aux),
            w_ast_blend_gate_aux=float(args.w_ast_blend_gate_aux),
            w_emergent_share_aux=float(args.w_emergent_share_aux),
            w_ast_share_aux=float(args.w_ast_share_aux),
            w_reb_share_aux=float(args.w_reb_share_aux),
            w_ast_team_rate_aux=float(args.w_ast_team_rate_aux),
            w_reb_opportunity_rate_aux=float(args.w_reb_opportunity_rate_aux),
            w_team_rebound_budget_rate_aux=float(args.w_team_rebound_budget_rate_aux),
            w_rebound_budget_blend_gate_aux=float(args.w_rebound_budget_blend_gate_aux),
            w_spread_aux=float(args.w_spread_aux),
            w_total_aux=float(args.w_total_aux),
            w_props_pts_aux=float(args.w_props_pts_aux),
            w_props_reb_aux=float(args.w_props_reb_aux),
            w_props_ast_aux=float(args.w_props_ast_aux),
            w_direct_pts_aux=float(args.w_direct_pts_aux),
            w_direct_reb_aux=float(args.w_direct_reb_aux),
            w_direct_ast_aux=float(args.w_direct_ast_aux),
            w_direct_stl_aux=float(args.w_direct_stl_aux),
            w_direct_blk_aux=float(args.w_direct_blk_aux),
            w_direct_tov_aux=float(args.w_direct_tov_aux),
            w_direct_boxscore_aux=float(args.w_direct_boxscore_aux),
            w_direct_opportunity_aux=float(args.w_direct_opportunity_aux),
            flow_anchor_teacher_model=flow_anchor_teacher_model,
            w_flow_anchor_nonast_aux=float(args.w_flow_anchor_nonast_aux),
            flow_anchor_target_scale=float(args.flow_anchor_target_scale),
            spread_total_aux_ramp_epochs=int(args.spread_total_aux_ramp_epochs),
            spread_total_aux_start_scale=float(args.spread_total_aux_start_scale),
            props_aux_ramp_epochs=int(args.props_aux_ramp_epochs),
            props_aux_start_scale=float(args.props_aux_start_scale),
            direct_stat_aux_ramp_epochs=int(args.direct_stat_aux_ramp_epochs),
            direct_stat_aux_start_scale=float(args.direct_stat_aux_start_scale),
            spread_aux_target_scale=float(args.spread_aux_target_scale),
            total_aux_target_scale=float(args.total_aux_target_scale),
            props_pts_target_scale=float(args.props_pts_target_scale),
            props_reb_target_scale=float(args.props_reb_target_scale),
            props_ast_target_scale=float(args.props_ast_target_scale),
            props_pts_aux_min_line=float(args.props_pts_aux_min_line),
            props_reb_aux_min_line=float(args.props_reb_aux_min_line),
            props_ast_aux_min_line=float(args.props_ast_aux_min_line),
            team_ast_budget_target_scale=float(args.team_ast_budget_target_scale),
            assist_share_recon_target_scale=float(args.assist_share_recon_target_scale),
            assist_playmaker_line_center=float(args.assist_playmaker_line_center),
            assist_playmaker_line_scale=float(args.assist_playmaker_line_scale),
            assist_playmaker_max_weight=float(args.assist_playmaker_max_weight),
            assist_underprediction_weight=float(args.assist_underprediction_weight),
            ast_blend_gate_target_eps=float(args.ast_blend_gate_target_eps),
            rebound_budget_blend_gate_target_eps=float(args.rebound_budget_blend_gate_target_eps),
            direct_pts_target_scale=float(args.direct_pts_target_scale),
            direct_reb_target_scale=float(args.direct_reb_target_scale),
            direct_ast_target_scale=float(args.direct_ast_target_scale),
            direct_stl_target_scale=float(args.direct_stl_target_scale),
            direct_blk_target_scale=float(args.direct_blk_target_scale),
            direct_tov_target_scale=float(args.direct_tov_target_scale),
            direct_fg3m_target_scale=float(args.direct_fg3m_target_scale),
            direct_ftm_target_scale=float(args.direct_ftm_target_scale),
            direct_fga_target_scale=float(args.direct_fga_target_scale),
            direct_fta_target_scale=float(args.direct_fta_target_scale),
            spread_aux_huber_delta=float(args.spread_aux_huber_delta),
            total_aux_huber_delta=float(args.total_aux_huber_delta),
            props_aux_huber_delta=float(args.props_aux_huber_delta),
            direct_stat_aux_huber_delta=float(args.direct_stat_aux_huber_delta),
            props_aux_confidence_min=float(args.props_aux_confidence_min),
                w_efficiency_mean_aux=float(args.w_efficiency_mean_aux),
                team_efficiency_ppp_target_scale=float(args.team_efficiency_ppp_target_scale),
                team_ppp_target_scale=float(args.team_ppp_target_scale),
                team_advantage_target_scale=float(args.team_advantage_target_scale),
                feature_mean=feature_mean,
            feature_std=feature_std,
            an_pts_line_idx=int(an_pts_line_idx),
            an_reb_line_idx=int(an_reb_line_idx),
            an_ast_line_idx=int(an_ast_line_idx),
            an_has_pts_idx=int(an_has_pts_idx),
            an_has_reb_idx=int(an_has_reb_idx),
            an_has_ast_idx=int(an_has_ast_idx),
            an_pts_books_idx=int(an_pts_books_idx),
            an_reb_books_idx=int(an_reb_books_idx),
            an_ast_books_idx=int(an_ast_books_idx),
            an_props_market_count_idx=int(an_props_market_count_idx),
            prior_play_prob_idx=int(prior_play_prob_idx),
            lineup_starter_announced_idx=int(lineup_starter_announced_idx),
            recent_start_pct_10_idx=int(recent_start_pct_10_idx),
            started_proxy_rate_prior_10_idx=int(started_proxy_rate_prior_10_idx),
            started_proxy_rate_prior_20_idx=int(started_proxy_rate_prior_20_idx),
            minutes_from_stints_prior_20_idx=int(minutes_from_stints_prior_20_idx),
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
            minutes_teacher_forcing_prob=float(minutes_teacher_forcing_prob),
            minutes_teacher_forcing_mode=str(args.minutes_teacher_forcing_mode),
            flow_minutes_teacher_forcing_prob=float(flow_minutes_teacher_forcing_prob),
            flow_minutes_teacher_forcing_mode=str(args.flow_minutes_teacher_forcing_mode),
            sparse_starter_loss_prior_play_prob_max=float(args.sparse_starter_loss_prior_play_prob_max),
            sparse_starter_loss_prior_minutes_max=float(args.sparse_starter_loss_prior_minutes_max),
            sparse_starter_loss_hist_start_rate_max=float(args.sparse_starter_loss_hist_start_rate_max),
            sparse_starter_loss_actual_min_threshold=float(args.sparse_starter_loss_actual_min_threshold),
            bench_riser_loss_prior_minutes_min=float(args.bench_riser_loss_prior_minutes_min),
            bench_riser_loss_prior_play_prob_min=float(args.bench_riser_loss_prior_play_prob_min),
            bench_riser_loss_hist_start_rate_max=float(args.bench_riser_loss_hist_start_rate_max),
            bench_riser_loss_actual_min_threshold=float(args.bench_riser_loss_actual_min_threshold),
            starter_promotion_prior_minutes_max=float(args.starter_promotion_prior_minutes_max),
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
                "minutes_hurdle_nll": float("nan"),
                "role_loss": float("nan"),
                "role_acc": float("nan"),
                "starter_promotion_loss": float("nan"),
                "sparse_starter_underpred_loss": float("nan"),
                "flow_nll": float("nan"),
                "crps_fpts": float("nan"),
                "team_energy": float("nan"),
                "count_acc": float("nan"),
                "poss_nll": float("nan"),
                "backbone_nll": float("nan"),
                "three_pa_nll": float("nan"),
                "efficiency_nll": float("nan"),
                "team_efficiency_ppp_aux": float("nan"),
                "team_ppp_aux": float("nan"),
                "usage_share_nll": float("nan"),
                "team_points_budget_aux": float("nan"),
                "team_ast_budget_aux": float("nan"),
                "assist_share_aux": float("nan"),
                "assist_share_recon_aux": float("nan"),
                "ast_blend_gate_aux": float("nan"),
                "emergent_share_aux": float("nan"),
                "ast_share_aux": float("nan"),
                "reb_share_aux": float("nan"),
                "ast_team_rate_aux": float("nan"),
                "reb_opportunity_rate_aux": float("nan"),
                "spread_aux": float("nan"),
                "total_aux": float("nan"),
                "props_pts_aux": float("nan"),
                "props_reb_aux": float("nan"),
                "props_ast_aux": float("nan"),
                "direct_pts_aux": float("nan"),
                "direct_reb_aux": float("nan"),
                "direct_ast_aux": float("nan"),
                "direct_stl_aux": float("nan"),
                "direct_blk_aux": float("nan"),
                "direct_tov_aux": float("nan"),
                "direct_boxscore_aux": float("nan"),
                "direct_opportunity_aux": float("nan"),
                "flow_anchor_nonast_aux": float("nan"),
                "efficiency_mean_aux": float("nan"),
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
                flow_label_columns=flow_label_cols,
                run_phase2_flow=bool(phase2_weights.run_phase2_flow),
                run_phase3_decision=bool(phase2_weights.run_phase3_decision),
                w_minutes=float(phase2_weights.w_minutes),
                w_minutes_nll=float(phase2_weights.w_minutes_nll),
                w_minutes_hurdle_nll=float(args.w_minutes_hurdle_nll),
                w_role_loss=float(args.w_role_loss),
                w_starter_promotion_loss=float(args.w_starter_promotion_loss),
                w_sparse_starter_underpred_loss=float(args.w_sparse_starter_underpred_loss),
                w_bench_riser_underpred_loss=float(args.w_bench_riser_underpred_loss),
                minutes_role_target_scheme=str(args.minutes_role_target_scheme),
                w_count=float(phase2_weights.w_count),
                w_member=float(phase2_weights.w_member),
                w_flow_nll=float(phase2_weights.w_flow_nll),
                w_crps_fpts=float(phase2_weights.w_crps_fpts),
                w_team_energy=float(phase2_weights.w_team_energy),
                minutes_nll_sigma=float(args.minutes_nll_sigma),
                minutes_hurdle_zero_threshold=float(args.minutes_hurdle_zero_threshold),
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
                w_team_efficiency_ppp_aux=float(args.w_team_efficiency_ppp_aux),
                w_usage_share_nll=float(args.w_usage_share_nll) if bool(args.enable_usage_share_head) else 0.0,
                w_team_ppp_aux=float(args.w_team_ppp_aux),
                w_team_advantage_aux=float(args.w_team_advantage_aux),
                w_team_possession_aux=float(args.w_team_possession_aux),
                w_team_points_budget_aux=float(args.w_team_points_budget_aux),
                w_team_ast_budget_aux=float(args.w_team_ast_budget_aux),
                w_assist_share_aux=float(args.w_assist_share_aux),
                w_assist_share_recon_aux=float(args.w_assist_share_recon_aux),
                w_ast_blend_gate_aux=float(args.w_ast_blend_gate_aux),
                w_emergent_share_aux=float(args.w_emergent_share_aux),
                w_ast_share_aux=float(args.w_ast_share_aux),
                w_reb_share_aux=float(args.w_reb_share_aux),
                w_ast_team_rate_aux=float(args.w_ast_team_rate_aux),
                w_reb_opportunity_rate_aux=float(args.w_reb_opportunity_rate_aux),
                w_team_rebound_budget_rate_aux=float(args.w_team_rebound_budget_rate_aux),
                w_rebound_budget_blend_gate_aux=float(args.w_rebound_budget_blend_gate_aux),
                w_spread_aux=float(args.w_spread_aux),
                w_total_aux=float(args.w_total_aux),
                w_props_pts_aux=float(args.w_props_pts_aux),
                w_props_reb_aux=float(args.w_props_reb_aux),
                w_props_ast_aux=float(args.w_props_ast_aux),
                w_direct_pts_aux=float(args.w_direct_pts_aux),
                w_direct_reb_aux=float(args.w_direct_reb_aux),
                w_direct_ast_aux=float(args.w_direct_ast_aux),
                w_direct_stl_aux=float(args.w_direct_stl_aux),
                w_direct_blk_aux=float(args.w_direct_blk_aux),
                w_direct_tov_aux=float(args.w_direct_tov_aux),
                w_direct_boxscore_aux=float(args.w_direct_boxscore_aux),
                w_direct_opportunity_aux=float(args.w_direct_opportunity_aux),
                flow_anchor_teacher_model=flow_anchor_teacher_model,
                w_flow_anchor_nonast_aux=float(args.w_flow_anchor_nonast_aux),
                flow_anchor_target_scale=float(args.flow_anchor_target_scale),
                spread_total_aux_ramp_epochs=int(args.spread_total_aux_ramp_epochs),
                spread_total_aux_start_scale=float(args.spread_total_aux_start_scale),
                props_aux_ramp_epochs=int(args.props_aux_ramp_epochs),
                props_aux_start_scale=float(args.props_aux_start_scale),
                direct_stat_aux_ramp_epochs=int(args.direct_stat_aux_ramp_epochs),
                direct_stat_aux_start_scale=float(args.direct_stat_aux_start_scale),
                spread_aux_target_scale=float(args.spread_aux_target_scale),
                total_aux_target_scale=float(args.total_aux_target_scale),
                props_pts_target_scale=float(args.props_pts_target_scale),
                props_reb_target_scale=float(args.props_reb_target_scale),
                props_ast_target_scale=float(args.props_ast_target_scale),
                props_pts_aux_min_line=float(args.props_pts_aux_min_line),
                props_reb_aux_min_line=float(args.props_reb_aux_min_line),
                props_ast_aux_min_line=float(args.props_ast_aux_min_line),
                team_ast_budget_target_scale=float(args.team_ast_budget_target_scale),
                assist_share_recon_target_scale=float(args.assist_share_recon_target_scale),
                assist_playmaker_line_center=float(args.assist_playmaker_line_center),
                assist_playmaker_line_scale=float(args.assist_playmaker_line_scale),
                assist_playmaker_max_weight=float(args.assist_playmaker_max_weight),
                assist_underprediction_weight=float(args.assist_underprediction_weight),
                ast_blend_gate_target_eps=float(args.ast_blend_gate_target_eps),
                rebound_budget_blend_gate_target_eps=float(args.rebound_budget_blend_gate_target_eps),
                direct_pts_target_scale=float(args.direct_pts_target_scale),
                direct_reb_target_scale=float(args.direct_reb_target_scale),
                direct_ast_target_scale=float(args.direct_ast_target_scale),
                direct_stl_target_scale=float(args.direct_stl_target_scale),
                direct_blk_target_scale=float(args.direct_blk_target_scale),
                direct_tov_target_scale=float(args.direct_tov_target_scale),
                direct_fg3m_target_scale=float(args.direct_fg3m_target_scale),
                direct_ftm_target_scale=float(args.direct_ftm_target_scale),
                direct_fga_target_scale=float(args.direct_fga_target_scale),
                direct_fta_target_scale=float(args.direct_fta_target_scale),
                spread_aux_huber_delta=float(args.spread_aux_huber_delta),
                total_aux_huber_delta=float(args.total_aux_huber_delta),
                props_aux_huber_delta=float(args.props_aux_huber_delta),
                direct_stat_aux_huber_delta=float(args.direct_stat_aux_huber_delta),
                props_aux_confidence_min=float(args.props_aux_confidence_min),
                w_efficiency_mean_aux=float(args.w_efficiency_mean_aux),
                team_efficiency_ppp_target_scale=float(args.team_efficiency_ppp_target_scale),
                team_ppp_target_scale=float(args.team_ppp_target_scale),
                team_advantage_target_scale=float(args.team_advantage_target_scale),
                feature_mean=feature_mean,
                feature_std=feature_std,
                an_pts_line_idx=int(an_pts_line_idx),
                an_reb_line_idx=int(an_reb_line_idx),
                an_ast_line_idx=int(an_ast_line_idx),
                an_has_pts_idx=int(an_has_pts_idx),
                an_has_reb_idx=int(an_has_reb_idx),
                an_has_ast_idx=int(an_has_ast_idx),
                an_pts_books_idx=int(an_pts_books_idx),
                an_reb_books_idx=int(an_reb_books_idx),
                an_ast_books_idx=int(an_ast_books_idx),
                an_props_market_count_idx=int(an_props_market_count_idx),
                prior_play_prob_idx=int(prior_play_prob_idx),
                lineup_starter_announced_idx=int(lineup_starter_announced_idx),
                recent_start_pct_10_idx=int(recent_start_pct_10_idx),
                started_proxy_rate_prior_10_idx=int(started_proxy_rate_prior_10_idx),
                started_proxy_rate_prior_20_idx=int(started_proxy_rate_prior_20_idx),
                minutes_from_stints_prior_20_idx=int(minutes_from_stints_prior_20_idx),
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
                minutes_teacher_forcing_prob=1.0,
                minutes_teacher_forcing_mode=str(args.minutes_teacher_forcing_mode),
                flow_minutes_teacher_forcing_prob=0.0 if bool(args.flow_use_minutes_conditioning) else 1.0,
                flow_minutes_teacher_forcing_mode=str(args.flow_minutes_teacher_forcing_mode),
                sparse_starter_loss_prior_play_prob_max=float(args.sparse_starter_loss_prior_play_prob_max),
                sparse_starter_loss_prior_minutes_max=float(args.sparse_starter_loss_prior_minutes_max),
                sparse_starter_loss_hist_start_rate_max=float(args.sparse_starter_loss_hist_start_rate_max),
                sparse_starter_loss_actual_min_threshold=float(args.sparse_starter_loss_actual_min_threshold),
                bench_riser_loss_prior_minutes_min=float(args.bench_riser_loss_prior_minutes_min),
                bench_riser_loss_prior_play_prob_min=float(args.bench_riser_loss_prior_play_prob_min),
                bench_riser_loss_hist_start_rate_max=float(args.bench_riser_loss_hist_start_rate_max),
                bench_riser_loss_actual_min_threshold=float(args.bench_riser_loss_actual_min_threshold),
                starter_promotion_prior_minutes_max=float(args.starter_promotion_prior_minutes_max),
            )

        val_total_ex_possreg = _resolve_early_stop_metric_value(
            metric_name="val_total_ex_possreg",
            val_total=float(val_stats["total"]),
            val_poss_regression=float(val_stats.get("poss_regression", 0.0)),
            w_poss_regression=float(backbone_weights.w_poss_regression),
            val_minutes_mae=float(val_stats["minutes_mae"]),
        )
        metrics = EpochMetrics(
            epoch=epoch,
            phase2_flow_warmup=float(phase2_weights.flow_warmup),
            phase2_anchor_weight=float(phase2_weights.anchor_weight),
            phase2_a2_scale=float(phase2_guard_state.a2_scale),
            minutes_teacher_forcing_prob=float(minutes_teacher_forcing_prob),
            flow_minutes_teacher_forcing_prob=float(flow_minutes_teacher_forcing_prob),
            phase2_backoff_count=int(phase2_guard_state.backoff_count),
            train_skipped_batches=int(train_stats.get("skipped_batches", 0.0)),
            train_instability_events=int(train_stats.get("instability_events", 0.0)),
            train_total=train_stats["total"],
            train_minutes_mae=train_stats["minutes_mae"],
            train_count_loss=train_stats["count_loss"],
            train_member_loss=train_stats["member_loss"],
            train_minutes_nll=train_stats["minutes_nll"],
            train_minutes_hurdle_nll=train_stats.get("minutes_hurdle_nll", 0.0),
            train_role_loss=train_stats.get("role_loss", 0.0),
            train_role_acc=train_stats.get("role_acc", 0.0),
            train_sparse_starter_underpred_loss=train_stats.get("sparse_starter_underpred_loss", 0.0),
            train_bench_riser_underpred_loss=train_stats.get("bench_riser_underpred_loss", 0.0),
            train_starter_promotion_loss=train_stats.get("starter_promotion_loss", 0.0),
            train_flow_nll=train_stats["flow_nll"],
            train_crps_fpts=train_stats["crps_fpts"],
            train_team_energy=train_stats["team_energy"],
            train_count_acc=train_stats["count_acc"],
            train_poss_nll=train_stats.get("poss_nll", 0.0),
            train_backbone_nll=train_stats.get("backbone_nll", 0.0),
            train_three_pa_nll=train_stats.get("three_pa_nll", 0.0),
            train_efficiency_nll=train_stats.get("efficiency_nll", 0.0),
            train_team_efficiency_ppp_aux=train_stats.get("team_efficiency_ppp_aux", 0.0),
            train_team_ppp_aux=train_stats.get("team_ppp_aux", 0.0),
            train_team_advantage_aux=train_stats.get("team_advantage_aux", 0.0),
            train_usage_share_nll=train_stats.get("usage_share_nll", 0.0),
            train_team_possession_aux=train_stats.get("team_possession_aux", 0.0),
            train_team_points_budget_aux=train_stats.get("team_points_budget_aux", 0.0),
            train_team_ast_budget_aux=train_stats.get("team_ast_budget_aux", 0.0),
            train_assist_share_aux=train_stats.get("assist_share_aux", 0.0),
            train_assist_share_recon_aux=train_stats.get("assist_share_recon_aux", 0.0),
            train_ast_blend_gate_aux=train_stats.get("ast_blend_gate_aux", 0.0),
            train_emergent_share_aux=train_stats.get("emergent_share_aux", 0.0),
            train_ast_share_aux=train_stats.get("ast_share_aux", 0.0),
            train_reb_share_aux=train_stats.get("reb_share_aux", 0.0),
            train_ast_team_rate_aux=train_stats.get("ast_team_rate_aux", 0.0),
            train_reb_opportunity_rate_aux=train_stats.get("reb_opportunity_rate_aux", 0.0),
            train_team_rebound_budget_rate_aux=train_stats.get("team_rebound_budget_rate_aux", 0.0),
            train_rebound_budget_blend_gate_aux=train_stats.get("rebound_budget_blend_gate_aux", 0.0),
            train_spread_aux=train_stats.get("spread_aux", 0.0),
            train_total_aux=train_stats.get("total_aux", 0.0),
            train_props_pts_aux=train_stats.get("props_pts_aux", 0.0),
            train_props_reb_aux=train_stats.get("props_reb_aux", 0.0),
            train_props_ast_aux=train_stats.get("props_ast_aux", 0.0),
            train_direct_pts_aux=train_stats.get("direct_pts_aux", 0.0),
            train_direct_reb_aux=train_stats.get("direct_reb_aux", 0.0),
            train_direct_ast_aux=train_stats.get("direct_ast_aux", 0.0),
            train_direct_stl_aux=train_stats.get("direct_stl_aux", 0.0),
            train_direct_blk_aux=train_stats.get("direct_blk_aux", 0.0),
            train_direct_tov_aux=train_stats.get("direct_tov_aux", 0.0),
            train_direct_boxscore_aux=train_stats.get("direct_boxscore_aux", 0.0),
            train_direct_opportunity_aux=train_stats.get("direct_opportunity_aux", 0.0),
            train_flow_anchor_nonast_aux=train_stats.get("flow_anchor_nonast_aux", 0.0),
            train_efficiency_mean_aux=train_stats.get("efficiency_mean_aux", 0.0),
            train_poss_regression=train_stats.get("poss_regression", 0.0),
            val_total=val_stats["total"],
            val_minutes_mae=val_stats["minutes_mae"],
            val_count_loss=val_stats["count_loss"],
            val_member_loss=val_stats["member_loss"],
            val_minutes_nll=val_stats["minutes_nll"],
            val_minutes_hurdle_nll=val_stats.get("minutes_hurdle_nll", 0.0),
            val_role_loss=val_stats.get("role_loss", 0.0),
            val_role_acc=val_stats.get("role_acc", 0.0),
            val_sparse_starter_underpred_loss=val_stats.get("sparse_starter_underpred_loss", 0.0),
            val_bench_riser_underpred_loss=val_stats.get("bench_riser_underpred_loss", 0.0),
            val_starter_promotion_loss=val_stats.get("starter_promotion_loss", 0.0),
            val_flow_nll=val_stats["flow_nll"],
            val_crps_fpts=val_stats["crps_fpts"],
            val_team_energy=val_stats["team_energy"],
            val_count_acc=val_stats["count_acc"],
            val_poss_nll=val_stats.get("poss_nll", 0.0),
            val_backbone_nll=val_stats.get("backbone_nll", 0.0),
            val_three_pa_nll=val_stats.get("three_pa_nll", 0.0),
            val_efficiency_nll=val_stats.get("efficiency_nll", 0.0),
            val_team_efficiency_ppp_aux=val_stats.get("team_efficiency_ppp_aux", 0.0),
            val_team_ppp_aux=val_stats.get("team_ppp_aux", 0.0),
            val_team_advantage_aux=val_stats.get("team_advantage_aux", 0.0),
            val_usage_share_nll=val_stats.get("usage_share_nll", 0.0),
            val_team_possession_aux=val_stats.get("team_possession_aux", 0.0),
            val_team_points_budget_aux=val_stats.get("team_points_budget_aux", 0.0),
            val_team_ast_budget_aux=val_stats.get("team_ast_budget_aux", 0.0),
            val_assist_share_aux=val_stats.get("assist_share_aux", 0.0),
            val_assist_share_recon_aux=val_stats.get("assist_share_recon_aux", 0.0),
            val_ast_blend_gate_aux=val_stats.get("ast_blend_gate_aux", 0.0),
            val_emergent_share_aux=val_stats.get("emergent_share_aux", 0.0),
            val_ast_share_aux=val_stats.get("ast_share_aux", 0.0),
            val_reb_share_aux=val_stats.get("reb_share_aux", 0.0),
            val_ast_team_rate_aux=val_stats.get("ast_team_rate_aux", 0.0),
            val_reb_opportunity_rate_aux=val_stats.get("reb_opportunity_rate_aux", 0.0),
            val_team_rebound_budget_rate_aux=val_stats.get("team_rebound_budget_rate_aux", 0.0),
            val_rebound_budget_blend_gate_aux=val_stats.get("rebound_budget_blend_gate_aux", 0.0),
            val_spread_aux=val_stats.get("spread_aux", 0.0),
            val_total_aux=val_stats.get("total_aux", 0.0),
            val_props_pts_aux=val_stats.get("props_pts_aux", 0.0),
            val_props_reb_aux=val_stats.get("props_reb_aux", 0.0),
            val_props_ast_aux=val_stats.get("props_ast_aux", 0.0),
            val_direct_pts_aux=val_stats.get("direct_pts_aux", 0.0),
            val_direct_reb_aux=val_stats.get("direct_reb_aux", 0.0),
            val_direct_ast_aux=val_stats.get("direct_ast_aux", 0.0),
            val_direct_stl_aux=val_stats.get("direct_stl_aux", 0.0),
            val_direct_blk_aux=val_stats.get("direct_blk_aux", 0.0),
            val_direct_tov_aux=val_stats.get("direct_tov_aux", 0.0),
            val_direct_boxscore_aux=val_stats.get("direct_boxscore_aux", 0.0),
            val_direct_opportunity_aux=val_stats.get("direct_opportunity_aux", 0.0),
            val_flow_anchor_nonast_aux=val_stats.get("flow_anchor_nonast_aux", 0.0),
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
        if bool(args.flow_use_minutes_conditioning):
            msg = f"{msg} flow_mtf={metrics.flow_minutes_teacher_forcing_prob:.3f}"
        if bool(args.enable_phase2_flow):
            msg = (
                f"{msg} "
                f"val_minutes_nll={metrics.val_minutes_nll:.4f} "
                f"val_flow_nll={metrics.val_flow_nll:.4f} "
                f"skipped_batches={metrics.train_skipped_batches} "
                f"instability_events={metrics.train_instability_events}"
            )
        if float(args.w_minutes_hurdle_nll) > 0.0:
            msg = f"{msg} val_minutes_hurdle_nll={metrics.val_minutes_hurdle_nll:.4f}"
        if float(args.w_role_loss) > 0.0:
            msg = f"{msg} val_role_loss={metrics.val_role_loss:.4f} val_role_acc={metrics.val_role_acc:.4f}"
        if float(args.w_sparse_starter_underpred_loss) > 0.0:
            msg = f"{msg} val_sparse_starter_underpred_loss={metrics.val_sparse_starter_underpred_loss:.4f}"
        if float(args.w_bench_riser_underpred_loss) > 0.0:
            msg = f"{msg} val_bench_riser_underpred_loss={metrics.val_bench_riser_underpred_loss:.4f}"
        if float(args.w_starter_promotion_loss) > 0.0:
            msg = f"{msg} val_starter_promotion_loss={metrics.val_starter_promotion_loss:.4f}"
        if bool(args.enable_phase3_decision):
            msg = (
                f"{msg} "
                f"val_crps_fpts={metrics.val_crps_fpts:.4f} "
                f"val_team_energy={metrics.val_team_energy:.4f}"
            )
        if bool(args.enable_efficiency_head):
            msg = f"{msg} val_efficiency_nll={metrics.val_efficiency_nll:.4f}"
        if float(args.w_team_ppp_aux) > 0.0:
            msg = f"{msg} val_team_ppp_aux={metrics.val_team_ppp_aux:.4f}"
        if float(args.w_team_advantage_aux) > 0.0:
            msg = f"{msg} val_team_advantage_aux={metrics.val_team_advantage_aux:.4f}"
        if float(args.w_team_efficiency_ppp_aux) > 0.0:
            msg = f"{msg} val_team_eff_ppp_aux={metrics.val_team_efficiency_ppp_aux:.4f}"
        if bool(args.enable_usage_share_head):
            msg = f"{msg} val_usage_share_nll={metrics.val_usage_share_nll:.4f}"
        if float(args.w_team_points_budget_aux) > 0.0:
            msg = f"{msg} val_team_points_budget_aux={metrics.val_team_points_budget_aux:.4f}"
        if float(args.w_team_ast_budget_aux) > 0.0:
            msg = f"{msg} val_team_ast_budget_aux={metrics.val_team_ast_budget_aux:.4f}"
        if float(args.w_assist_share_aux) > 0.0:
            msg = f"{msg} val_assist_share_aux={metrics.val_assist_share_aux:.4f}"
        if float(args.w_assist_share_recon_aux) > 0.0:
            msg = f"{msg} val_assist_share_recon_aux={metrics.val_assist_share_recon_aux:.4f}"
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
        if float(args.w_team_rebound_budget_rate_aux) > 0.0:
            msg = f"{msg} val_team_reb_budget_rate_aux={metrics.val_team_rebound_budget_rate_aux:.4f}"
        if float(args.w_rebound_budget_blend_gate_aux) > 0.0:
            msg = f"{msg} val_reb_budget_gate_aux={metrics.val_rebound_budget_blend_gate_aux:.4f}"
        if float(args.w_spread_aux) > 0.0:
            msg = f"{msg} val_spread_aux={metrics.val_spread_aux:.4f}"
        if float(args.w_total_aux) > 0.0:
            msg = f"{msg} val_total_aux={metrics.val_total_aux:.4f}"
        if float(args.w_props_pts_aux) > 0.0:
            msg = f"{msg} val_props_pts_aux={metrics.val_props_pts_aux:.4f}"
        if float(args.w_props_reb_aux) > 0.0:
            msg = f"{msg} val_props_reb_aux={metrics.val_props_reb_aux:.4f}"
        if float(args.w_props_ast_aux) > 0.0:
            msg = f"{msg} val_props_ast_aux={metrics.val_props_ast_aux:.4f}"
        if (
            float(args.w_direct_pts_aux) > 0.0
            or float(args.w_direct_reb_aux) > 0.0
            or float(args.w_direct_ast_aux) > 0.0
            or float(args.w_direct_stl_aux) > 0.0
            or float(args.w_direct_blk_aux) > 0.0
            or float(args.w_direct_tov_aux) > 0.0
            or float(args.w_direct_boxscore_aux) > 0.0
            or float(args.w_direct_opportunity_aux) > 0.0
        ):
            msg = (
                f"{msg} "
                f"val_direct_pts_aux={metrics.val_direct_pts_aux:.4f} "
                f"val_direct_reb_aux={metrics.val_direct_reb_aux:.4f} "
                f"val_direct_ast_aux={metrics.val_direct_ast_aux:.4f} "
                f"val_direct_stl_aux={metrics.val_direct_stl_aux:.4f} "
                f"val_direct_blk_aux={metrics.val_direct_blk_aux:.4f} "
                f"val_direct_tov_aux={metrics.val_direct_tov_aux:.4f} "
                f"val_direct_boxscore_aux={metrics.val_direct_boxscore_aux:.4f} "
                f"val_direct_opportunity_aux={metrics.val_direct_opportunity_aux:.4f}"
            )
        if float(args.w_flow_anchor_nonast_aux) > 0.0:
            msg = f"{msg} val_flow_anchor_nonast_aux={metrics.val_flow_anchor_nonast_aux:.4f}"
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
        checkpoint_metric_value = _resolve_early_stop_metric_value(
            metric_name=str(args.best_checkpoint_metric),
            val_total=float(metrics.val_total),
            val_poss_regression=float(metrics.val_poss_regression),
            w_poss_regression=float(backbone_weights.w_poss_regression),
            val_minutes_mae=float(metrics.val_minutes_mae),
        )
        if (
            not in_flow_delay
            and math.isfinite(float(checkpoint_metric_value))
            and float(checkpoint_metric_value) < float(best_checkpoint_value)
        ):
            best_checkpoint_value = float(checkpoint_metric_value)
            best_val_total = float(metrics.val_total)
            best_epoch = epoch
            torch.save(model.state_dict(), out_dir / "model.pt")
        if not in_flow_delay and bool(args.enable_sparse_checkpoint_rerank):
            candidate_path = checkpoint_candidates_dir / f"epoch_{int(epoch):03d}.pt"
            torch.save(model.state_dict(), candidate_path)
            minutes_checkpoint_candidates = _record_topk_minutes_checkpoint(
                candidates=minutes_checkpoint_candidates,
                epoch=int(epoch),
                val_minutes_mae=float(metrics.val_minutes_mae),
                checkpoint_metric_value=float(checkpoint_metric_value),
                val_total=float(metrics.val_total),
                checkpoint_path=candidate_path,
                top_k=int(args.checkpoint_topk_by_minutes),
            )

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
                val_minutes_mae=float(metrics.val_minutes_mae),
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

    sparse_checkpoint_rerank_summary: dict[str, Any] = {
        "enabled": bool(args.enable_sparse_checkpoint_rerank),
        "topk_by_minutes": int(args.checkpoint_topk_by_minutes),
        "minutes_mae_tolerance": float(args.checkpoint_minutes_mae_tolerance),
        "weights": {
            "sparse_underpred": float(args.checkpoint_sparse_weight_underpred),
            "active_count_mae": float(args.checkpoint_sparse_weight_active_count_mae),
            "starter_sparse_shortfall": float(args.checkpoint_sparse_weight_starter_shortfall),
        },
        "starter_promotion_thresholds": {
            "prior_minutes_max": float(args.checkpoint_starter_promotion_prior_minutes_max),
            "hist_start_rate_max": float(args.checkpoint_starter_promotion_hist_start_rate_max),
        },
        "target_starter_sparse_minutes": float(args.checkpoint_sparse_target_starter_sparse_minutes),
        "candidates": [asdict(item) for item in minutes_checkpoint_candidates],
        "evaluated_candidates": [],
        "selected_candidate": None,
    }
    if bool(args.enable_sparse_checkpoint_rerank) and minutes_checkpoint_candidates:
        evaluated_candidates: list[dict[str, Any]] = []
        for candidate in minutes_checkpoint_candidates:
            rerank_metrics = _evaluate_sparse_checkpoint_candidate(
                model=model,
                checkpoint_path=Path(candidate.checkpoint_path),
                device=device,
                val_loader=val_loader,
                val_df=val_df,
                active_threshold=float(args.active_threshold),
                low_minutes_threshold=float(args.checkpoint_low_minutes_threshold),
                sparse_prior_play_prob_max=float(args.checkpoint_sparse_prior_play_prob_max),
                sparse_prior_minutes_max=float(args.checkpoint_sparse_prior_minutes_max),
                starter_promotion_prior_minutes_max=float(args.checkpoint_starter_promotion_prior_minutes_max),
                starter_promotion_hist_start_rate_max=float(args.checkpoint_starter_promotion_hist_start_rate_max),
                next_up_actual_min=float(args.checkpoint_next_up_actual_min),
                next_up_pred_min=float(args.checkpoint_next_up_pred_min),
                estimated_possessions_idx=int(estimated_possessions_idx) if int(estimated_possessions_idx) >= 0 else None,
                prior_minutes_feature_idx=(
                    int(minutes_from_stints_prior_20_idx) if int(minutes_from_stints_prior_20_idx) >= 0 else None
                ),
                feature_mean=np.asarray(feature_mean, dtype=np.float32),
                feature_std=np.asarray(feature_std, dtype=np.float32),
                target_starter_sparse_minutes=float(args.checkpoint_sparse_target_starter_sparse_minutes),
                weight_sparse_underpred=float(args.checkpoint_sparse_weight_underpred),
                weight_active_count_mae=float(args.checkpoint_sparse_weight_active_count_mae),
                weight_starter_sparse_shortfall=float(args.checkpoint_sparse_weight_starter_shortfall),
            )
            evaluated_candidates.append(
                {
                    **asdict(candidate),
                    "sparse_rerank": asdict(rerank_metrics),
                }
            )
        selected_candidate = _select_sparse_rerank_candidate(
            candidates=evaluated_candidates,
            minutes_mae_tolerance=float(args.checkpoint_minutes_mae_tolerance),
        )
        sparse_checkpoint_rerank_summary["evaluated_candidates"] = evaluated_candidates
        sparse_checkpoint_rerank_summary["selected_candidate"] = selected_candidate
        if selected_candidate is not None:
            selected_path = Path(str(selected_candidate["checkpoint_path"]))
            state = torch.load(selected_path, map_location="cpu")
            torch.save(state, out_dir / "model.pt")
            best_epoch = int(selected_candidate["epoch"])
            best_checkpoint_value = float(selected_candidate["checkpoint_metric_value"])
            best_val_total = float(selected_candidate["val_total"])
            print(
                (
                    "[checkpoint-rerank] "
                    f"selected_epoch={int(selected_candidate['epoch']):03d} "
                    f"val_minutes_mae={float(selected_candidate['val_minutes_mae']):.4f} "
                    f"sparse_score={float(selected_candidate['sparse_rerank']['sparse_score']):.4f} "
                    f"sparse_underpred={float(selected_candidate['sparse_rerank']['sparse_next_up_underpred_rate']):.4f} "
                    f"active_count_mae={float(selected_candidate['sparse_rerank']['active_count_mae']):.4f} "
                    f"starter_sparse_pred_minutes={float(selected_candidate['sparse_rerank']['starter_sparse_pred_minutes_mean']):.4f}"
                ),
                flush=True,
            )

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
        "flow_target_schema": str(flow_target_schema),
        "flow_target_columns": flow_label_cols,
        "flow_model_target_columns": (
            flow_target_columns(
                include_pf=bool(include_pf_in_flow_targets),
                schema=str(flow_target_schema),
            )
            if bool(args.enable_phase2_flow)
            else []
        ),
        "phase2_flow_enabled": bool(args.enable_phase2_flow),
        "minutes_hurdle_head_enabled": bool(args.enable_minutes_hurdle_head),
        "minutes_role_head_enabled": bool(args.enable_minutes_role_head),
        "minutes_hurdle": {
            "w_minutes_hurdle_nll": float(args.w_minutes_hurdle_nll),
            "zero_threshold": float(args.minutes_hurdle_zero_threshold),
            "hidden_dim": int(args.minutes_hurdle_hidden),
            "sigma_floor": float(args.minutes_hurdle_sigma_floor),
        },
        "minutes_role": {
            "w_role_loss": float(args.w_role_loss),
            "use_context_for_preferences": not bool(args.disable_minutes_role_context_for_preferences),
            "hidden_dim": int(args.minutes_role_hidden),
            "embedding_dim": int(args.minutes_role_embedding_dim),
            "num_classes": int(args.minutes_role_num_classes),
            "target_scheme": str(args.minutes_role_target_scheme),
        },
        "starter_promotion": {
            "enabled": bool(args.enable_starter_promotion_head),
            "w_starter_promotion_loss": float(args.w_starter_promotion_loss),
            "hidden_dim": int(args.starter_promotion_hidden_dim),
            "prior_minutes_max": float(args.starter_promotion_prior_minutes_max),
        },
        "sparse_starter_underpred": {
            "weight": float(args.w_sparse_starter_underpred_loss),
            "prior_play_prob_max": float(args.sparse_starter_loss_prior_play_prob_max),
            "prior_minutes_max": float(args.sparse_starter_loss_prior_minutes_max),
            "hist_start_rate_max": float(args.sparse_starter_loss_hist_start_rate_max),
            "actual_minutes_threshold": float(args.sparse_starter_loss_actual_min_threshold),
        },
        "bench_riser_underpred": {
            "weight": float(args.w_bench_riser_underpred_loss),
            "prior_minutes_min": float(args.bench_riser_loss_prior_minutes_min),
            "prior_play_prob_min": float(args.bench_riser_loss_prior_play_prob_min),
            "hist_start_rate_max": float(args.bench_riser_loss_hist_start_rate_max),
            "actual_minutes_threshold": float(args.bench_riser_loss_actual_min_threshold),
        },
        "efficiency_head_enabled": bool(args.enable_efficiency_head),
        "efficiency_head_only": bool(args.efficiency_head_only),
        "efficiency_market": {
            "enabled": bool(args.efficiency_market_context),
            "hidden_dim": int(args.efficiency_market_hidden),
            "alpha": float(args.efficiency_market_alpha),
            "sidecar_feature_columns": efficiency_sidecar_feature_cols,
            "sidecar_hidden_dim": int(args.efficiency_sidecar_hidden),
            "sidecar_alpha": float(args.efficiency_sidecar_alpha),
            "w_team_efficiency_ppp_aux": float(args.w_team_efficiency_ppp_aux),
            "team_efficiency_ppp_target_scale": float(args.team_efficiency_ppp_target_scale),
        },
        "team_ppp": {
            "enabled": bool(args.enable_team_ppp_head),
            "hidden_dim": int(args.team_ppp_head_hidden),
            "to_backbone": bool(args.team_ppp_to_backbone),
            "direct_backbone_context": bool(args.team_ppp_direct_backbone_context),
            "latent_hidden": int(args.team_ppp_latent_hidden),
            "backbone_alpha": float(args.team_ppp_backbone_alpha),
            "to_efficiency": bool(args.team_ppp_to_efficiency),
            "efficiency_alpha": float(args.team_ppp_efficiency_alpha),
            "direct_efficiency_context": bool(args.team_ppp_direct_efficiency_context),
            "w_team_ppp_aux": float(args.w_team_ppp_aux),
            "team_ppp_target_scale": float(args.team_ppp_target_scale),
        },
        "team_advantage": {
            "enabled": bool(args.enable_team_advantage_head),
            "hidden_dim": int(args.team_advantage_head_hidden),
            "direct_backbone_context": bool(args.team_advantage_direct_backbone_context),
            "w_team_advantage_aux": float(args.w_team_advantage_aux),
            "team_advantage_target_scale": float(args.team_advantage_target_scale),
        },
        "usage_share_head_enabled": bool(args.enable_usage_share_head),
        "usage_share_head_only": bool(args.usage_share_head_only),
        "team_points_budget": {
            "enabled": bool(args.enable_team_points_budget_head),
            "hidden_dim": int(args.team_points_budget_head_hidden),
            "parameterization": str(args.team_points_budget_parameterization),
            "to_backbone": bool(args.team_points_budget_to_backbone),
            "latent_hidden": int(args.team_points_budget_latent_hidden),
            "reconcile_budget": bool(args.team_points_reconcile_budget),
            "reconcile_alpha": float(args.team_points_reconcile_alpha),
            "w_team_points_budget_aux": float(args.w_team_points_budget_aux),
        },
        "team_opportunity_budget": {
            "parameterization": str(args.team_opportunity_budget_parameterization),
            "to_backbone": bool(args.team_opportunity_budget_to_backbone),
            "latent_hidden": int(args.team_opportunity_budget_latent_hidden),
            "backbone_alpha": float(args.team_opportunity_budget_backbone_alpha),
            "reconcile_budget": bool(args.team_opportunity_reconcile_budget),
            "reconcile_alpha": float(args.team_opportunity_reconcile_alpha),
            "reconcile_preserve_possessions": bool(args.team_opportunity_reconcile_preserve_possessions),
        },
        "ast_factorization_heads": {
            "team_ast_budget_head_enabled": bool(args.enable_team_ast_budget_head),
            "team_ast_budget_head_hidden": int(args.team_ast_budget_head_hidden),
            "assist_share_head_enabled": bool(args.enable_assist_share_head),
            "assist_share_head_hidden": int(args.assist_share_head_hidden),
            "assist_share_condition_feature_columns": list(assist_share_condition_feature_cols),
            "assist_share_condition_hidden": int(args.assist_share_condition_hidden),
            "ast_blend_gate_enabled": bool(args.enable_ast_blend_gate),
            "ast_blend_gate_hidden": int(args.ast_blend_gate_hidden),
            "ast_blend_gate_init_alpha": float(args.ast_blend_gate_init_alpha),
            "w_ast_blend_gate_aux": float(args.w_ast_blend_gate_aux),
            "ast_blend_gate_target_eps": float(args.ast_blend_gate_target_eps),
            "assist_share_replace_flow_ast": bool(args.assist_share_replace_flow_ast),
            "assist_share_factorized_ast": bool(args.assist_share_factorized_ast),
            "assist_share_reconcile_ast_budget": bool(args.assist_share_reconcile_ast_budget),
            "assist_share_reconcile_alpha": float(args.assist_share_reconcile_alpha),
            "assist_share_reconcile_temperature": float(args.assist_share_reconcile_temperature),
            "w_team_ast_budget_aux": float(args.w_team_ast_budget_aux),
            "w_assist_share_aux": float(args.w_assist_share_aux),
            "w_assist_share_recon_aux": float(args.w_assist_share_recon_aux),
            "team_ast_budget_target_scale": float(args.team_ast_budget_target_scale),
            "assist_share_recon_target_scale": float(args.assist_share_recon_target_scale),
            "assist_playmaker_line_center": float(args.assist_playmaker_line_center),
            "assist_playmaker_line_scale": float(args.assist_playmaker_line_scale),
            "assist_playmaker_max_weight": float(args.assist_playmaker_max_weight),
            "assist_underprediction_weight": float(args.assist_underprediction_weight),
        },
        "rebound_factorization_heads": {
            "team_rebound_budget_head_enabled": bool(args.enable_team_rebound_budget_head),
            "team_rebound_budget_head_hidden": int(args.team_rebound_budget_head_hidden),
            "rebound_budget_parameterization": str(args.rebound_budget_parameterization),
            "rebound_oreb_rate_cap": float(args.rebound_oreb_rate_cap),
            "rebound_dreb_rate_cap": float(args.rebound_dreb_rate_cap),
            "rebound_dreb_deterministic_discount": float(args.rebound_dreb_deterministic_discount),
            "rebound_oreb_reconcile_use_flow_budget": bool(args.rebound_oreb_reconcile_use_flow_budget),
            "rebound_oreb_budget_blend_alpha": float(args.rebound_oreb_budget_blend_alpha),
            "rebound_dreb_budget_blend_alpha": float(args.rebound_dreb_budget_blend_alpha),
            "rebound_budget_blend_gate_enabled": bool(args.enable_rebound_budget_blend_gate),
            "rebound_budget_blend_gate_hidden": int(args.rebound_budget_blend_gate_hidden),
            "rebound_budget_blend_gate_init_alpha": float(args.rebound_budget_blend_gate_init_alpha),
            "rebound_share_head_enabled": bool(args.enable_rebound_share_head),
            "rebound_share_head_hidden": int(args.rebound_share_head_hidden),
            "rebound_share_condition_feature_columns": list(rebound_share_condition_feature_cols),
            "rebound_share_condition_hidden": int(args.rebound_share_condition_hidden),
            "rebound_factor_reconcile_oreb_dreb": bool(args.rebound_factor_reconcile_oreb_dreb),
            "rebound_factor_reconcile_mode": str(args.rebound_factor_reconcile_mode),
            "rebound_factor_reconcile_alpha": float(args.rebound_factor_reconcile_alpha),
            "rebound_factor_reconcile_temperature": float(args.rebound_factor_reconcile_temperature),
        },
        "ast_reb_structure_aux": {
            "w_ast_share_aux": float(args.w_ast_share_aux),
            "w_reb_share_aux": float(args.w_reb_share_aux),
            "w_ast_team_rate_aux": float(args.w_ast_team_rate_aux),
            "w_reb_opportunity_rate_aux": float(args.w_reb_opportunity_rate_aux),
            "w_team_rebound_budget_rate_aux": float(args.w_team_rebound_budget_rate_aux),
            "w_rebound_budget_blend_gate_aux": float(args.w_rebound_budget_blend_gate_aux),
            "rebound_budget_blend_gate_target_eps": float(args.rebound_budget_blend_gate_target_eps),
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
        "props_line_aux": {
            "w_props_pts_aux": float(args.w_props_pts_aux),
            "w_props_reb_aux": float(args.w_props_reb_aux),
            "w_props_ast_aux": float(args.w_props_ast_aux),
            "props_pts_aux_min_line": float(args.props_pts_aux_min_line),
            "props_reb_aux_min_line": float(args.props_reb_aux_min_line),
            "props_ast_aux_min_line": float(args.props_ast_aux_min_line),
            "ramp_epochs": int(args.props_aux_ramp_epochs),
            "start_scale": float(args.props_aux_start_scale),
            "props_pts_target_scale": float(args.props_pts_target_scale),
            "props_reb_target_scale": float(args.props_reb_target_scale),
            "props_ast_target_scale": float(args.props_ast_target_scale),
            "props_aux_huber_delta": float(args.props_aux_huber_delta),
            "props_aux_confidence_min": float(args.props_aux_confidence_min),
            "feature_indices": {
                "an_pts_line_idx": int(an_pts_line_idx),
                "an_reb_line_idx": int(an_reb_line_idx),
                "an_ast_line_idx": int(an_ast_line_idx),
                "an_has_pts_idx": int(an_has_pts_idx),
                "an_has_reb_idx": int(an_has_reb_idx),
                "an_has_ast_idx": int(an_has_ast_idx),
                "an_pts_books_idx": int(an_pts_books_idx),
                "an_reb_books_idx": int(an_reb_books_idx),
                "an_ast_books_idx": int(an_ast_books_idx),
                "an_props_market_count_idx": int(an_props_market_count_idx),
                "prior_play_prob_idx": int(prior_play_prob_idx),
            },
        },
        "direct_stat_aux": {
            "w_direct_pts_aux": float(args.w_direct_pts_aux),
            "w_direct_reb_aux": float(args.w_direct_reb_aux),
            "w_direct_ast_aux": float(args.w_direct_ast_aux),
            "w_direct_stl_aux": float(args.w_direct_stl_aux),
            "w_direct_blk_aux": float(args.w_direct_blk_aux),
            "w_direct_tov_aux": float(args.w_direct_tov_aux),
            "w_direct_boxscore_aux": float(args.w_direct_boxscore_aux),
            "w_direct_opportunity_aux": float(args.w_direct_opportunity_aux),
            "ramp_epochs": int(args.direct_stat_aux_ramp_epochs),
            "start_scale": float(args.direct_stat_aux_start_scale),
            "target_scales": {
                "pts": float(args.direct_pts_target_scale),
                "reb": float(args.direct_reb_target_scale),
                "ast": float(args.direct_ast_target_scale),
                "stl": float(args.direct_stl_target_scale),
                "blk": float(args.direct_blk_target_scale),
                "tov": float(args.direct_tov_target_scale),
                "fg3m": float(args.direct_fg3m_target_scale),
                "ftm": float(args.direct_ftm_target_scale),
                "fga": float(args.direct_fga_target_scale),
                "fta": float(args.direct_fta_target_scale),
            },
            "huber_delta": float(args.direct_stat_aux_huber_delta),
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
        "best_val_total": float(best_val_total),
        "best_epoch": int(best_epoch),
        "best_checkpoint_metric": str(args.best_checkpoint_metric),
        "best_checkpoint_metric_value": float(best_checkpoint_value),
        "sparse_checkpoint_rerank": sparse_checkpoint_rerank_summary,
        "init_model_pt": str(init_model_pt) if init_model_pt is not None else None,
        "args": vars(args),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "best_epoch": best_epoch,
                "best_val_total": best_val_total,
                "best_checkpoint_metric": str(args.best_checkpoint_metric),
                "best_checkpoint_metric_value": best_checkpoint_value,
                "sparse_checkpoint_rerank_enabled": bool(args.enable_sparse_checkpoint_rerank),
                "sparse_checkpoint_rerank_selected_epoch": (
                    sparse_checkpoint_rerank_summary["selected_candidate"]["epoch"]
                    if sparse_checkpoint_rerank_summary.get("selected_candidate") is not None
                    else None
                ),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
