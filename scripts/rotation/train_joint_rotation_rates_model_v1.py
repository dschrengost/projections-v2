#!/usr/bin/env python3
"""Train a joint rotation+minutes+rates set model (v1).

Input dataset contract (from build_joint_rotation_rates_dataset_v1):
  - features.parquet
  - labels_minutes.parquet
  - labels_rates.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from projections import paths
from projections.rotation.joint_set_model_v1 import (
    JointRotationRatesModelConfig,
    build_joint_model,
)
from projections.rotation.set_model import zfill_game_id_series
from projections.rotation.training_losses import (
    build_in_rotation_labels,
    compute_anti_smear_penalty,
    compute_k_regularizer,
    compute_minutes_out_loss,
)

KEY_COLS = ["game_id", "team_id", "player_id"]
JOIN_KEYS = ["game_id", "team_id", "player_id", "game_date"]

RATE_TARGETS = [
    "fga2_per_min",
    "fga3_per_min",
    "fta_per_min",
    "ast_per_min",
    "tov_per_min",
    "oreb_per_min",
    "dreb_per_min",
    "stl_per_min",
    "blk_per_min",
]
EFFICIENCY_TARGETS = ["fg2_pct_label", "fg3_pct_label", "ft_pct_label"]
EFFICIENCY_LOWS = [0.3, 0.2, 0.5]
EFFICIENCY_HIGHS = [0.75, 0.55, 0.95]

TEAM_EMBED_TEAM_IDX_COL = "team_id_idx"
TEAM_EMBED_OPP_IDX_COL = "opp_id_idx"
PLAYER_EMBED_IDX_COL = "player_id_idx"
PLAYER_TEAM_HASH_IDX_COL = "player_team_hash_idx"

# Same exclusion policy as rotation trainer where possible.
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


def _utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_sha() -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=paths.get_project_root())  # noqa: S603,S607
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return None


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


def _coerce_join_keys(df: pd.DataFrame, *, name: str, require_game_date: bool) -> pd.DataFrame:
    out = df.copy()
    for col in KEY_COLS:
        if col not in out.columns:
            raise ValueError(f"{name} missing key column: {col}")
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
    if require_game_date:
        if "game_date" not in out.columns:
            raise ValueError(f"{name} missing key column: game_date")
        out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce").dt.normalize()
    elif "game_date" in out.columns:
        out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce").dt.normalize()

    invalid = out[KEY_COLS].isna().any(axis=1)
    if invalid.any():
        raise ValueError(f"{name} has {int(invalid.sum())} rows with invalid numeric keys")
    if require_game_date and out["game_date"].isna().any():
        raise ValueError(f"{name} has {int(out['game_date'].isna().sum())} rows with invalid game_date")
    return out


def _assert_unique_keys(df: pd.DataFrame, *, keys: list[str], name: str) -> None:
    dupes = df.duplicated(subset=keys, keep=False)
    if dupes.any():
        sample = df.loc[dupes, keys].head(5).to_dict(orient="records")
        raise ValueError(f"{name} has duplicated key rows; sample={sample}")


def _infer_feature_columns(features_df: pd.DataFrame, *, labels_minutes_df: pd.DataFrame, labels_rates_df: pd.DataFrame) -> list[str]:
    excluded = {
        "game_id",
        "team_id",
        "player_id",
        "game_date",
        "game_id_norm",
        TEAM_EMBED_TEAM_IDX_COL,
        TEAM_EMBED_OPP_IDX_COL,
        PLAYER_EMBED_IDX_COL,
        PLAYER_TEAM_HASH_IDX_COL,
    }
    excluded.update(set(labels_minutes_df.columns))
    excluded.update(set(labels_rates_df.columns))
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
        raise ValueError("No numeric feature columns found after exclusion rules")
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


def _build_team_id_vocab(df: pd.DataFrame) -> list[int]:
    ids: list[pd.Series] = [pd.to_numeric(df["team_id"], errors="coerce").astype("Int64")]
    for col in ["opponent_team_id", "home_team_id", "away_team_id"]:
        if col in df.columns:
            ids.append(pd.to_numeric(df[col], errors="coerce").astype("Int64"))
    all_ids = pd.concat(ids, ignore_index=True).dropna().astype("int64")
    vocab = sorted({int(v) for v in all_ids.to_numpy().tolist()})
    if not vocab:
        raise ValueError("Empty team id vocab")
    return vocab


def _build_player_id_vocab(df: pd.DataFrame) -> list[int]:
    series = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64").dropna().astype("int64")
    vocab = sorted({int(v) for v in series.to_numpy().tolist()})
    if not vocab:
        raise ValueError("Empty player id vocab")
    return vocab


def _hash_player_team_pair(player_id: int, team_id: int, buckets: int) -> int:
    payload = f"{int(player_id)}:{int(team_id)}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    value = int.from_bytes(digest, byteorder="little", signed=False)
    return int(value % max(1, buckets - 1)) + 1


def _add_team_embedding_indices(df: pd.DataFrame, *, team_id_vocab: list[int]) -> pd.DataFrame:
    mapping = {int(team_id): i + 1 for i, team_id in enumerate(team_id_vocab)}
    out = df.copy()
    out[TEAM_EMBED_TEAM_IDX_COL] = (
        pd.to_numeric(out["team_id"], errors="coerce").astype("Int64").map(mapping).fillna(0).astype("int64")
    )
    opp = out["opponent_team_id"] if "opponent_team_id" in out.columns else pd.Series(pd.NA, index=out.index)
    out[TEAM_EMBED_OPP_IDX_COL] = pd.to_numeric(opp, errors="coerce").astype("Int64").map(mapping).fillna(0).astype("int64")
    return out


def _add_player_embedding_indices(df: pd.DataFrame, *, player_id_vocab: list[int]) -> pd.DataFrame:
    mapping = {int(player_id): i + 1 for i, player_id in enumerate(player_id_vocab)}
    out = df.copy()
    out[PLAYER_EMBED_IDX_COL] = (
        pd.to_numeric(out["player_id"], errors="coerce").astype("Int64").map(mapping).fillna(0).astype("int64")
    )
    return out


def _add_player_team_hash_indices(df: pd.DataFrame, *, buckets: int) -> pd.DataFrame:
    if buckets <= 1:
        raise ValueError("player_team_hash_buckets must be > 1")
    out = df.copy()
    p = pd.to_numeric(out["player_id"], errors="coerce").astype("Int64")
    t = pd.to_numeric(out["team_id"], errors="coerce").astype("Int64")
    vals = np.zeros(len(out), dtype=np.int64)
    for i, (pv, tv) in enumerate(zip(p.to_numpy(), t.to_numpy(), strict=False)):
        if pd.isna(pv) or pd.isna(tv):
            vals[i] = 0
            continue
        vals[i] = _hash_player_team_pair(int(pv), int(tv), int(buckets))
    out[PLAYER_TEAM_HASH_IDX_COL] = vals
    return out


def _resolve_split(
    team_games: pd.DataFrame,
    *,
    val_start_date: str | None,
    val_end_date: str | None,
    val_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    unique_days = sorted(pd.to_datetime(team_games["game_date"]).dropna().dt.normalize().unique().tolist())
    if not unique_days:
        raise ValueError("No game_date values available for split")

    if val_start_date or val_end_date:
        val_start = pd.Timestamp(val_start_date).normalize() if val_start_date else pd.Timestamp(unique_days[0]).normalize()
        val_end = pd.Timestamp(val_end_date).normalize() if val_end_date else pd.Timestamp(unique_days[-1]).normalize()
        is_val = (team_games["game_date"] >= val_start) & (team_games["game_date"] <= val_end)
    else:
        vd = max(int(val_days), 1)
        val_dates = set(pd.to_datetime(unique_days[-vd:]).tolist())
        is_val = team_games["game_date"].isin(val_dates)

    val = team_games.loc[is_val].copy()
    train = team_games.loc[~is_val].copy()
    if train.empty or val.empty:
        raise ValueError(f"Invalid split: train={len(train)} val={len(val)}")

    meta = {
        "train_team_games": int(len(train)),
        "val_team_games": int(len(val)),
        "train_game_dates": int(pd.to_datetime(train["game_date"]).nunique()),
        "val_game_dates": int(pd.to_datetime(val["game_date"]).nunique()),
        "train_min_date": str(pd.to_datetime(train["game_date"]).min().date()),
        "train_max_date": str(pd.to_datetime(train["game_date"]).max().date()),
        "val_min_date": str(pd.to_datetime(val["game_date"]).min().date()),
        "val_max_date": str(pd.to_datetime(val["game_date"]).max().date()),
    }
    return train, val, meta


@dataclass(frozen=True)
class TeamGameExample:
    x: np.ndarray
    team_idx: np.ndarray
    opp_idx: np.ndarray
    player_idx: np.ndarray
    player_team_idx: np.ndarray
    alloc_mask: np.ndarray
    prior_w: np.ndarray
    router_features: np.ndarray
    y_minutes: np.ndarray
    y_rates: np.ndarray
    y_eff: np.ndarray
    rates_mask: np.ndarray
    eff_mask: np.ndarray


class TeamGameDataset(Dataset[TeamGameExample]):
    def __init__(self, examples: list[TeamGameExample]) -> None:
        if not examples:
            raise ValueError("TeamGameDataset requires >=1 example")
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> TeamGameExample:
        return self.examples[idx]


def _collate_team_games(
    batch: list[TeamGameExample],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    max_n = max(ex.x.shape[0] for ex in batch)
    num_features = batch[0].x.shape[1]
    n_rates = batch[0].y_rates.shape[1]
    n_eff = batch[0].y_eff.shape[1]

    x = torch.zeros((len(batch), max_n, num_features), dtype=torch.float32)
    team_idx = torch.zeros((len(batch), max_n), dtype=torch.long)
    opp_idx = torch.zeros((len(batch), max_n), dtype=torch.long)
    player_idx = torch.zeros((len(batch), max_n), dtype=torch.long)
    player_team_idx = torch.zeros((len(batch), max_n), dtype=torch.long)
    alloc_mask = torch.zeros((len(batch), max_n), dtype=torch.bool)
    prior_w = torch.zeros((len(batch), max_n), dtype=torch.float32)
    router_features = torch.zeros((len(batch), 3), dtype=torch.float32)
    y_minutes = torch.zeros((len(batch), max_n), dtype=torch.float32)
    y_rates = torch.zeros((len(batch), max_n, n_rates), dtype=torch.float32)
    y_eff = torch.zeros((len(batch), max_n, n_eff), dtype=torch.float32)
    rates_mask = torch.zeros((len(batch), max_n, n_rates), dtype=torch.bool)
    eff_mask = torch.zeros((len(batch), max_n, n_eff), dtype=torch.bool)
    mask = torch.zeros((len(batch), max_n), dtype=torch.bool)

    for i, ex in enumerate(batch):
        n = ex.x.shape[0]
        x[i, :n] = torch.from_numpy(ex.x)
        team_idx[i, :n] = torch.from_numpy(ex.team_idx)
        opp_idx[i, :n] = torch.from_numpy(ex.opp_idx)
        player_idx[i, :n] = torch.from_numpy(ex.player_idx)
        player_team_idx[i, :n] = torch.from_numpy(ex.player_team_idx)
        alloc_mask[i, :n] = torch.from_numpy(ex.alloc_mask)
        prior_w[i, :n] = torch.from_numpy(ex.prior_w)
        router_features[i] = torch.from_numpy(ex.router_features.astype("float32"))
        y_minutes[i, :n] = torch.from_numpy(ex.y_minutes)
        y_rates[i, :n] = torch.from_numpy(ex.y_rates)
        y_eff[i, :n] = torch.from_numpy(ex.y_eff)
        rates_mask[i, :n] = torch.from_numpy(ex.rates_mask)
        eff_mask[i, :n] = torch.from_numpy(ex.eff_mask)
        mask[i, :n] = True

    return (
        x,
        team_idx,
        opp_idx,
        player_idx,
        player_team_idx,
        alloc_mask,
        prior_w,
        router_features,
        y_minutes,
        y_rates,
        y_eff,
        rates_mask,
        eff_mask,
        mask,
    )


def _build_team_game_examples(
    df: pd.DataFrame,
    *,
    feature_cols: list[str],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    prior_weight_col: str,
) -> list[TeamGameExample]:
    feats = _numeric_frame(df, feature_cols).to_numpy(dtype="float32", copy=False)
    feats = np.nan_to_num(feats, nan=feature_mean[None, :], posinf=feature_mean[None, :], neginf=feature_mean[None, :])
    feats = (feats - feature_mean[None, :]) / feature_std[None, :]

    team_idx_all = pd.to_numeric(df.get(TEAM_EMBED_TEAM_IDX_COL, 0), errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    opp_idx_all = pd.to_numeric(df.get(TEAM_EMBED_OPP_IDX_COL, 0), errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    player_idx_all = pd.to_numeric(df.get(PLAYER_EMBED_IDX_COL, 0), errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    player_team_idx_all = (
        pd.to_numeric(df.get(PLAYER_TEAM_HASH_IDX_COL, 0), errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    )

    y_minutes_all = pd.to_numeric(df["minutes_label"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    is_out_all = pd.to_numeric(df.get("is_out", 0), errors="coerce").fillna(0).astype(int).to_numpy(dtype=np.int64) > 0
    alloc_mask_all = ~is_out_all

    prior_w_all = (
        pd.to_numeric(df.get(prior_weight_col, 0.0), errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
        .to_numpy(dtype=np.float32)
    )
    prior20_all = (
        pd.to_numeric(df.get("minutes_from_stints_prior_20", 0.0), errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
        .to_numpy(dtype=np.float32)
    )

    y_rates_cols = []
    y_rates_masks = []
    rates_loss_row = pd.to_numeric(df.get("rates_loss_eligible", 0), errors="coerce").fillna(0).to_numpy(dtype=np.int64) > 0
    for col in RATE_TARGETS:
        arr = pd.to_numeric(df.get(col, np.nan), errors="coerce").to_numpy(dtype=np.float32)
        valid = np.isfinite(arr) & rates_loss_row
        y_rates_cols.append(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0))
        y_rates_masks.append(valid)
    y_rates_all = np.stack(y_rates_cols, axis=1).astype(np.float32)
    rates_mask_all = np.stack(y_rates_masks, axis=1).astype(bool)

    y_eff_cols = []
    y_eff_masks = []
    for col in EFFICIENCY_TARGETS:
        arr = pd.to_numeric(df.get(col, np.nan), errors="coerce").to_numpy(dtype=np.float32)
        valid = np.isfinite(arr) & rates_loss_row
        y_eff_cols.append(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0))
        y_eff_masks.append(valid)
    y_eff_all = np.stack(y_eff_cols, axis=1).astype(np.float32)
    eff_mask_all = np.stack(y_eff_masks, axis=1).astype(bool)

    examples: list[TeamGameExample] = []
    grouped = df.groupby(["game_id_norm", "team_id"], sort=False).indices
    for _, idx in grouped.items():
        idx_arr = np.asarray(idx, dtype=np.int64)
        alloc_mask = alloc_mask_all[idx_arr]
        prior20_group = prior20_all[idx_arr].astype(np.float64, copy=False)
        out_mask = ~alloc_mask
        vacated = float(np.sum(prior20_group[out_mask]))
        k = int(min(5, prior20_group.shape[0]))
        if k > 0:
            top_idx = np.argsort(prior20_group)[::-1][:k]
            starters_out_proxy = int(np.sum(out_mask[top_idx].astype(np.int64)))
        else:
            starters_out_proxy = 0
        team_out_count = int(np.sum(out_mask.astype(np.int64)))
        router = np.array(
            [
                float(np.clip(vacated / 60.0, 0.0, 1.0)),
                float(np.clip(team_out_count / 10.0, 0.0, 1.0)),
                float(np.clip(starters_out_proxy / 5.0, 0.0, 1.0)),
            ],
            dtype=np.float32,
        )
        examples.append(
            TeamGameExample(
                x=feats[idx_arr],
                team_idx=team_idx_all[idx_arr],
                opp_idx=opp_idx_all[idx_arr],
                player_idx=player_idx_all[idx_arr],
                player_team_idx=player_team_idx_all[idx_arr],
                alloc_mask=alloc_mask,
                prior_w=prior_w_all[idx_arr],
                router_features=router,
                y_minutes=y_minutes_all[idx_arr],
                y_rates=y_rates_all[idx_arr],
                y_eff=y_eff_all[idx_arr],
                rates_mask=rates_mask_all[idx_arr],
                eff_mask=eff_mask_all[idx_arr],
            )
        )
    return examples


def _masked_mae_2d(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    err = (pred - y).abs() * mask.to(dtype=pred.dtype)
    denom = mask.to(dtype=pred.dtype).sum().clamp(min=1.0)
    return err.sum() / denom


def _masked_mae_3d(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    err = (pred - y).abs() * mask.to(dtype=pred.dtype)
    denom = mask.to(dtype=pred.dtype).sum().clamp(min=1.0)
    return err.sum() / denom


def _resolve_k_target_tensor(
    in_rotation: torch.Tensor,
    *,
    k_target_fixed: float,
    k_target_source: str,
    k_target_blend: float,
) -> torch.Tensor:
    """Resolve per-team K targets used by the K regularizer."""
    label_k = in_rotation.sum(dim=1)
    if k_target_source == "label":
        return label_k
    fixed = torch.full_like(label_k, float(k_target_fixed))
    if k_target_source == "blend":
        w = float(np.clip(k_target_blend, 0.0, 1.0))
        return (1.0 - w) * fixed + w * label_k
    return fixed


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    lambda_minutes: float,
    lambda_rates: float,
    lambda_eff: float,
    lambda_rot: float,
    rot_target_threshold: float,
    rotation_minutes_threshold: float,
    gate_bce_weight: float,
    minutes_out_weight: float,
    k_target: float,
    k_target_source: str,
    k_target_blend: float,
    k_reg_weight: float,
    anti_smear_weight: float,
    anti_smear_floor: float,
) -> dict[str, float]:
    model.eval()
    total = 0.0
    n_batches = 0
    minutes_sum = 0.0
    rates_sum = 0.0
    eff_sum = 0.0
    rot_sum = 0.0
    gate_bce_sum = 0.0
    minutes_out_sum = 0.0
    k_reg_sum = 0.0
    anti_smear_sum = 0.0
    with torch.no_grad():
        for (
            x,
            team_idx,
            opp_idx,
            player_idx,
            player_team_idx,
            alloc_mask,
            prior_w,
            router_features,
            y_minutes,
            y_rates,
            y_eff,
            rates_mask,
            eff_mask,
            mask,
        ) in loader:
            x = x.to(device)
            team_idx = team_idx.to(device)
            opp_idx = opp_idx.to(device)
            player_idx = player_idx.to(device)
            player_team_idx = player_team_idx.to(device)
            alloc_mask = alloc_mask.to(device)
            prior_w = prior_w.to(device)
            router_features = router_features.to(device)
            y_minutes = y_minutes.to(device)
            y_rates = y_rates.to(device)
            y_eff = y_eff.to(device)
            rates_mask = rates_mask.to(device)
            eff_mask = eff_mask.to(device)
            mask = mask.to(device)

            pred_minutes, gate_logits, _, pred_rates, pred_eff, _, _ = model(
                x,
                team_idx,
                opp_idx,
                mask,
                player_idx=player_idx,
                player_team_idx=player_team_idx,
                alloc_mask=alloc_mask,
                prior_weights=prior_w,
                router_features=router_features,
            )

            minutes_loss = _masked_mae_2d(pred_minutes, y_minutes, mask)
            rates_loss = _masked_mae_3d(pred_rates, y_rates, rates_mask)
            eff_loss = _masked_mae_3d(pred_eff, y_eff, eff_mask)
            rot_loss = torch.tensor(0.0, device=device)
            if lambda_rot > 0.0 and gate_logits is not None:
                final_mask = alloc_mask & mask
                y_rot = (y_minutes >= float(rot_target_threshold)).to(dtype=pred_minutes.dtype)
                per_el = F.binary_cross_entropy_with_logits(gate_logits, y_rot, reduction="none")
                rot_loss = (per_el * final_mask.to(dtype=per_el.dtype)).sum() / final_mask.to(dtype=per_el.dtype).sum().clamp(min=1.0)

            loss_gate_bce = torch.tensor(0.0, device=device)
            loss_minutes_out = torch.tensor(0.0, device=device)
            loss_k_reg = torch.tensor(0.0, device=device)
            loss_anti_smear = torch.tensor(0.0, device=device)
            if gate_logits is not None:
                final_mask_gate = alloc_mask & mask
                in_rotation = build_in_rotation_labels(y_minutes, float(rotation_minutes_threshold), final_mask_gate)
                if gate_bce_weight > 0:
                    n_pos = in_rotation.sum().clamp(min=1.0)
                    n_neg = (final_mask_gate.to(dtype=in_rotation.dtype) - in_rotation).sum().clamp(min=1.0)
                    pos_weight = (n_neg / n_pos).clamp(min=0.5, max=2.0)
                    bce_per_el = F.binary_cross_entropy_with_logits(
                        gate_logits,
                        in_rotation,
                        pos_weight=pos_weight.expand_as(gate_logits),
                        reduction="none",
                    )
                    mask_f = final_mask_gate.to(dtype=bce_per_el.dtype)
                    loss_gate_bce = (bce_per_el * mask_f).sum() / mask_f.sum().clamp(min=1.0)
                if minutes_out_weight > 0:
                    loss_minutes_out = compute_minutes_out_loss(pred_minutes, in_rotation, final_mask_gate)
                if k_reg_weight > 0:
                    k_target_tensor = _resolve_k_target_tensor(
                        in_rotation,
                        k_target_fixed=float(k_target),
                        k_target_source=str(k_target_source),
                        k_target_blend=float(k_target_blend),
                    )
                    loss_k_reg = compute_k_regularizer(gate_logits, final_mask_gate, k_target_tensor)
                if anti_smear_weight > 0:
                    loss_anti_smear = compute_anti_smear_penalty(
                        pred_minutes,
                        gate_logits,
                        final_mask_gate,
                        float(anti_smear_floor),
                    )

            batch_total = (
                float(lambda_minutes) * minutes_loss
                + float(lambda_rates) * rates_loss
                + float(lambda_eff) * eff_loss
                + float(lambda_rot) * rot_loss
                + float(gate_bce_weight) * loss_gate_bce
                + float(minutes_out_weight) * loss_minutes_out
                + float(k_reg_weight) * loss_k_reg
                + float(anti_smear_weight) * loss_anti_smear
            )
            total += float(batch_total.item())
            minutes_sum += float(minutes_loss.item())
            rates_sum += float(rates_loss.item())
            eff_sum += float(eff_loss.item())
            rot_sum += float(rot_loss.item())
            gate_bce_sum += float(loss_gate_bce.item())
            minutes_out_sum += float(loss_minutes_out.item())
            k_reg_sum += float(loss_k_reg.item())
            anti_smear_sum += float(loss_anti_smear.item())
            n_batches += 1

    if n_batches == 0:
        return {
            "loss": float("nan"),
            "minutes_mae": float("nan"),
            "rates_mae": float("nan"),
            "eff_mae": float("nan"),
            "rot_bce": float("nan"),
            "gate_bce": float("nan"),
            "minutes_out": float("nan"),
            "k_reg": float("nan"),
            "anti_smear": float("nan"),
        }
    denom = float(n_batches)
    return {
        "loss": total / denom,
        "minutes_mae": minutes_sum / denom,
        "rates_mae": rates_sum / denom,
        "eff_mae": eff_sum / denom,
        "rot_bce": rot_sum / denom,
        "gate_bce": gate_bce_sum / denom,
        "minutes_out": minutes_out_sum / denom,
        "k_reg": k_reg_sum / denom,
        "anti_smear": anti_smear_sum / denom,
    }


def _set_minutes_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for param in model.minutes_model.parameters():  # type: ignore[attr-defined]
        param.requires_grad = bool(trainable)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset-dir", type=str, required=True, help="Joint dataset dir with features/labels parquet files.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(paths.get_data_root() / "artifacts" / "joint_rotation_rates_v1" / "runs"),
        help="Directory where run artifacts will be written.",
    )
    parser.add_argument("--run-tag", type=str, default="joint_rotation_rates_v1", help="Run id prefix.")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", help="Torch device: auto, cpu, cuda, cuda:0, mps.")
    parser.add_argument("--val-start-date", type=str, default=None)
    parser.add_argument("--val-end-date", type=str, default=None)
    parser.add_argument("--val-days", type=int, default=14)
    parser.add_argument("--max-team-games", type=int, default=None, help="Optional cap of training team-games.")
    parser.add_argument("--lambda-minutes", type=float, default=1.0)
    parser.add_argument("--lambda-rates", type=float, default=0.6)
    parser.add_argument("--lambda-eff", type=float, default=0.2)
    parser.add_argument("--lambda-rot", type=float, default=0.4)
    parser.add_argument("--rot-target-threshold", type=float, default=8.0)
    parser.add_argument("--rotation-minutes-threshold", type=float, default=6.0)
    parser.add_argument("--gate-bce-weight", type=float, default=1.0)
    parser.add_argument("--minutes-out-weight", type=float, default=0.25)
    parser.add_argument("--k-target", type=float, default=9.5)
    parser.add_argument("--k-target-source", type=str, choices=["fixed", "label", "blend"], default="fixed")
    parser.add_argument("--k-target-blend", type=float, default=0.5)
    parser.add_argument("--k-reg-weight", type=float, default=0.05)
    parser.add_argument("--anti-smear-weight", type=float, default=0.05)
    parser.add_argument("--anti-smear-floor", type=float, default=4.0)
    parser.add_argument("--freeze-minutes-epochs", type=int, default=0, help="Train rates heads only for first K epochs.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--prior-weight-col", type=str, default="minutes_from_stints_prior_20")
    parser.add_argument("--prior-weight-floor", type=float, default=1.0)
    parser.add_argument("--use-prior-head", action="store_true", help="Multiply allocation weights by prior_weight_col.")
    parser.add_argument("--use-team-embeddings", action="store_true")
    parser.add_argument("--team-embedding-dim", type=int, default=8)
    parser.add_argument("--use-player-embeddings", action="store_true")
    parser.add_argument("--player-embedding-dim", type=int, default=16)
    parser.add_argument("--use-player-team-embeddings", action="store_true")
    parser.add_argument("--player-team-hash-buckets", type=int, default=16384)
    parser.add_argument("--player-team-embedding-dim", type=int, default=8)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-transformer-layers", type=int, default=2)
    parser.add_argument("--num-attention-heads", type=int, default=4)
    parser.add_argument("--alloc-activation", type=str, default="entmax")
    parser.add_argument("--entmax-alpha", type=float, default=1.5)
    parser.add_argument("--share-temperature", type=float, default=1.0)
    parser.add_argument("--rates-hidden-dim", type=int, default=128)
    args = parser.parse_args()

    _set_seed(int(args.seed))
    device = _resolve_training_device(str(args.device))
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        device_name = torch.cuda.get_device_name(device)
        print(f"[joint_train] device={device} ({device_name})")
    else:
        print(f"[joint_train] device={device}")
    if float(args.gate_bce_weight) < 0:
        raise ValueError("--gate-bce-weight must be >= 0")
    if float(args.minutes_out_weight) < 0:
        raise ValueError("--minutes-out-weight must be >= 0")
    if float(args.k_target) < 0:
        raise ValueError("--k-target must be >= 0")
    if str(args.k_target_source) == "blend" and not (0.0 <= float(args.k_target_blend) <= 1.0):
        raise ValueError("--k-target-blend must be in [0,1] when --k-target-source=blend")
    if float(args.k_reg_weight) < 0:
        raise ValueError("--k-reg-weight must be >= 0")
    if float(args.anti_smear_weight) < 0:
        raise ValueError("--anti-smear-weight must be >= 0")
    if float(args.anti_smear_floor) < 0:
        raise ValueError("--anti-smear-floor must be >= 0")
    if int(args.num_workers) < 0:
        raise ValueError("--num-workers must be >= 0")

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    features_path = dataset_dir / "features.parquet"
    labels_minutes_path = dataset_dir / "labels_minutes.parquet"
    labels_rates_path = dataset_dir / "labels_rates.parquet"
    for path in [features_path, labels_minutes_path, labels_rates_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset input file: {path}")

    print(f"[joint_train] dataset_dir={dataset_dir}")
    features_df = pd.read_parquet(features_path)
    labels_minutes_df = pd.read_parquet(labels_minutes_path)
    labels_rates_df = pd.read_parquet(labels_rates_path)
    features_df = _coerce_join_keys(features_df, name="features", require_game_date=True)
    labels_minutes_df = _coerce_join_keys(labels_minutes_df, name="labels_minutes", require_game_date=True)
    labels_rates_df = _coerce_join_keys(labels_rates_df, name="labels_rates", require_game_date=True)
    _assert_unique_keys(features_df, keys=JOIN_KEYS, name="features")
    _assert_unique_keys(labels_minutes_df, keys=JOIN_KEYS, name="labels_minutes")
    _assert_unique_keys(labels_rates_df, keys=JOIN_KEYS, name="labels_rates")

    minute_col = "minutes_label"
    if minute_col not in labels_minutes_df.columns:
        raise ValueError(f"labels_minutes missing required column: {minute_col}")

    keep_minutes = labels_minutes_df.loc[:, JOIN_KEYS + [minute_col]]
    rate_cols_present = [c for c in RATE_TARGETS + EFFICIENCY_TARGETS + ["rates_loss_eligible"] if c in labels_rates_df.columns]
    keep_rates = labels_rates_df.loc[:, JOIN_KEYS + rate_cols_present]

    merged = features_df.merge(keep_minutes, on=JOIN_KEYS, how="left", validate="one_to_one")
    merged = merged.merge(keep_rates, on=JOIN_KEYS, how="left", validate="one_to_one")
    merged["game_id_norm"] = zfill_game_id_series(merged["game_id"])
    merged[minute_col] = pd.to_numeric(merged[minute_col], errors="coerce")
    if merged[minute_col].isna().any():
        missing = int(merged[minute_col].isna().sum())
        raise ValueError(f"Merged frame has {missing} rows with missing {minute_col}")

    # Optional embedding indices.
    team_vocab: list[int] | None = None
    if bool(args.use_team_embeddings):
        team_vocab = _build_team_id_vocab(merged)
        merged = _add_team_embedding_indices(merged, team_id_vocab=team_vocab)
    else:
        merged[TEAM_EMBED_TEAM_IDX_COL] = 0
        merged[TEAM_EMBED_OPP_IDX_COL] = 0
    player_vocab: list[int] | None = None
    if bool(args.use_player_embeddings):
        player_vocab = _build_player_id_vocab(merged)
        merged = _add_player_embedding_indices(merged, player_id_vocab=player_vocab)
    else:
        merged[PLAYER_EMBED_IDX_COL] = 0
    if bool(args.use_player_team_embeddings):
        merged = _add_player_team_hash_indices(merged, buckets=int(args.player_team_hash_buckets))
    else:
        merged[PLAYER_TEAM_HASH_IDX_COL] = 0

    feature_cols = _infer_feature_columns(
        merged,
        labels_minutes_df=labels_minutes_df,
        labels_rates_df=labels_rates_df,
    )
    print(f"[joint_train] feature columns: {len(feature_cols)}")

    team_games = (
        merged.loc[:, ["game_id_norm", "team_id", "game_date"]]
        .drop_duplicates()
        .sort_values(["game_date", "game_id_norm", "team_id"])
        .reset_index(drop=True)
    )
    train_tg, val_tg, split_meta = _resolve_split(
        team_games,
        val_start_date=args.val_start_date,
        val_end_date=args.val_end_date,
        val_days=int(args.val_days),
    )
    print(
        "[joint_train] split:",
        f"train_team_games={split_meta['train_team_games']}",
        f"val_team_games={split_meta['val_team_games']}",
        f"train_dates={split_meta['train_min_date']}..{split_meta['train_max_date']}",
        f"val_dates={split_meta['val_min_date']}..{split_meta['val_max_date']}",
    )

    train_key_tuples = set(zip(train_tg["game_id_norm"], train_tg["team_id"], strict=False))
    val_key_tuples = set(zip(val_tg["game_id_norm"], val_tg["team_id"], strict=False))
    row_keys = pd.Series(list(zip(merged["game_id_norm"], merged["team_id"], strict=False)), index=merged.index)
    train_df = merged.loc[row_keys.isin(train_key_tuples)].copy()
    val_df = merged.loc[row_keys.isin(val_key_tuples)].copy()

    if args.max_team_games is not None and args.max_team_games > 0:
        max_tg = int(args.max_team_games)
        train_tg_sample = train_tg.iloc[:max_tg].copy()
        sampled_keys = set(zip(train_tg_sample["game_id_norm"], train_tg_sample["team_id"], strict=False))
        before = int(train_tg.shape[0])
        train_row_keys = pd.Series(list(zip(train_df["game_id_norm"], train_df["team_id"], strict=False)), index=train_df.index)
        train_df = train_df.loc[train_row_keys.isin(sampled_keys)].copy()
        train_tg = train_tg_sample
        print(f"[joint_train] max_team_games cap: {before}->{len(train_tg)}")

    train_num = _numeric_frame(train_df, feature_cols)
    feature_mean = train_num.mean(axis=0).to_numpy(dtype=np.float32)
    feature_std = train_num.std(axis=0).replace(0.0, 1.0).fillna(1.0).to_numpy(dtype=np.float32)
    feature_mean = np.nan_to_num(feature_mean, nan=0.0, posinf=0.0, neginf=0.0)
    feature_std = np.nan_to_num(feature_std, nan=1.0, posinf=1.0, neginf=1.0)
    feature_std = np.where(feature_std <= 1e-6, 1.0, feature_std)

    train_examples = _build_team_game_examples(
        train_df,
        feature_cols=feature_cols,
        feature_mean=feature_mean,
        feature_std=feature_std,
        prior_weight_col=str(args.prior_weight_col),
    )
    val_examples = _build_team_game_examples(
        val_df,
        feature_cols=feature_cols,
        feature_mean=feature_mean,
        feature_std=feature_std,
        prior_weight_col=str(args.prior_weight_col),
    )
    print(f"[joint_train] examples: train={len(train_examples)} val={len(val_examples)}")

    loader_kwargs: dict[str, Any] = {"pin_memory": bool(device.type == "cuda")}
    if int(args.num_workers) > 0:
        # Avoid Linux fork-related worker crashes with torch + large object graphs.
        loader_kwargs["multiprocessing_context"] = "spawn"
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        TeamGameDataset(train_examples),
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        collate_fn=_collate_team_games,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        TeamGameDataset(val_examples),
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=_collate_team_games,
        **loader_kwargs,
    )

    config = JointRotationRatesModelConfig(
        feature_columns=feature_cols,
        feature_mean=feature_mean.astype(float).tolist(),
        feature_std=feature_std.astype(float).tolist(),
        use_prior_head=bool(args.use_prior_head),
        prior_weight_col=str(args.prior_weight_col),
        prior_weight_floor=float(args.prior_weight_floor),
        use_team_embeddings=bool(args.use_team_embeddings),
        team_id_vocab=team_vocab,
        team_embedding_dim=int(args.team_embedding_dim),
        use_player_embeddings=bool(args.use_player_embeddings),
        player_id_vocab=player_vocab,
        player_embedding_dim=int(args.player_embedding_dim),
        use_player_team_embeddings=bool(args.use_player_team_embeddings),
        player_team_hash_buckets=int(args.player_team_hash_buckets),
        player_team_embedding_dim=int(args.player_team_embedding_dim),
        embed_dim=int(args.embed_dim),
        hidden_dim=int(args.hidden_dim),
        dropout=float(args.dropout),
        num_transformer_layers=int(args.num_transformer_layers),
        num_attention_heads=int(args.num_attention_heads),
        use_gate_head=True,
        num_experts=1,
        router_feature_dim=3,
        router_temperature=1.0,
        router_mode="soft",
        router_gumbel=False,
        router_gumbel_tau=1.0,
        alloc_activation=str(args.alloc_activation),
        entmax_alpha=float(args.entmax_alpha),
        share_temperature=float(args.share_temperature),
        total_minutes=240.0,
        eps=1e-6,
        rates_hidden_dim=int(args.rates_hidden_dim),
        rate_target_names=RATE_TARGETS,
        efficiency_target_names=EFFICIENCY_TARGETS,
        efficiency_lows=EFFICIENCY_LOWS,
        efficiency_highs=EFFICIENCY_HIGHS,
    )
    model = build_joint_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    history: list[dict[str, Any]] = []
    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, int(args.epochs) + 1):
        freeze_now = epoch <= int(args.freeze_minutes_epochs)
        _set_minutes_backbone_trainable(model, not freeze_now)
        model.train()
        train_total = 0.0
        train_minutes = 0.0
        train_rates = 0.0
        train_eff = 0.0
        train_rot = 0.0
        train_gate_bce = 0.0
        train_minutes_out = 0.0
        train_k_reg = 0.0
        train_anti_smear = 0.0
        n_batches = 0
        for (
            x,
            team_idx,
            opp_idx,
            player_idx,
            player_team_idx,
            alloc_mask,
            prior_w,
            router_features,
            y_minutes,
            y_rates,
            y_eff,
            rates_mask,
            eff_mask,
            mask,
        ) in train_loader:
            x = x.to(device)
            team_idx = team_idx.to(device)
            opp_idx = opp_idx.to(device)
            player_idx = player_idx.to(device)
            player_team_idx = player_team_idx.to(device)
            alloc_mask = alloc_mask.to(device)
            prior_w = prior_w.to(device)
            router_features = router_features.to(device)
            y_minutes = y_minutes.to(device)
            y_rates = y_rates.to(device)
            y_eff = y_eff.to(device)
            rates_mask = rates_mask.to(device)
            eff_mask = eff_mask.to(device)
            mask = mask.to(device)

            pred_minutes, gate_logits, _, pred_rates, pred_eff, _, _ = model(
                x,
                team_idx,
                opp_idx,
                mask,
                player_idx=player_idx,
                player_team_idx=player_team_idx,
                alloc_mask=alloc_mask,
                prior_weights=prior_w,
                router_features=router_features,
            )
            minutes_loss = _masked_mae_2d(pred_minutes, y_minutes, mask)
            rates_loss = _masked_mae_3d(pred_rates, y_rates, rates_mask)
            eff_loss = _masked_mae_3d(pred_eff, y_eff, eff_mask)
            rot_loss = torch.tensor(0.0, device=device)
            if float(args.lambda_rot) > 0.0 and gate_logits is not None:
                final_mask = alloc_mask & mask
                y_rot = (y_minutes >= float(args.rot_target_threshold)).to(dtype=pred_minutes.dtype)
                per_el = F.binary_cross_entropy_with_logits(gate_logits, y_rot, reduction="none")
                rot_loss = (per_el * final_mask.to(dtype=per_el.dtype)).sum() / final_mask.to(dtype=per_el.dtype).sum().clamp(min=1.0)

            loss_gate_bce = torch.tensor(0.0, device=device)
            loss_minutes_out = torch.tensor(0.0, device=device)
            loss_k_reg = torch.tensor(0.0, device=device)
            loss_anti_smear = torch.tensor(0.0, device=device)
            if gate_logits is not None:
                final_mask_gate = alloc_mask & mask
                in_rotation = build_in_rotation_labels(y_minutes, float(args.rotation_minutes_threshold), final_mask_gate)
                if float(args.gate_bce_weight) > 0:
                    n_pos = in_rotation.sum().clamp(min=1.0)
                    n_neg = (final_mask_gate.to(dtype=in_rotation.dtype) - in_rotation).sum().clamp(min=1.0)
                    pos_weight = (n_neg / n_pos).clamp(min=0.5, max=2.0)
                    bce_per_el = F.binary_cross_entropy_with_logits(
                        gate_logits,
                        in_rotation,
                        pos_weight=pos_weight.expand_as(gate_logits),
                        reduction="none",
                    )
                    mask_f = final_mask_gate.to(dtype=bce_per_el.dtype)
                    loss_gate_bce = (bce_per_el * mask_f).sum() / mask_f.sum().clamp(min=1.0)
                if float(args.minutes_out_weight) > 0:
                    loss_minutes_out = compute_minutes_out_loss(pred_minutes, in_rotation, final_mask_gate)
                if float(args.k_reg_weight) > 0:
                    k_target_tensor = _resolve_k_target_tensor(
                        in_rotation,
                        k_target_fixed=float(args.k_target),
                        k_target_source=str(args.k_target_source),
                        k_target_blend=float(args.k_target_blend),
                    )
                    loss_k_reg = compute_k_regularizer(gate_logits, final_mask_gate, k_target_tensor)
                if float(args.anti_smear_weight) > 0:
                    loss_anti_smear = compute_anti_smear_penalty(
                        pred_minutes,
                        gate_logits,
                        final_mask_gate,
                        float(args.anti_smear_floor),
                    )

            total_loss = (
                float(args.lambda_minutes) * minutes_loss
                + float(args.lambda_rates) * rates_loss
                + float(args.lambda_eff) * eff_loss
                + float(args.lambda_rot) * rot_loss
                + float(args.gate_bce_weight) * loss_gate_bce
                + float(args.minutes_out_weight) * loss_minutes_out
                + float(args.k_reg_weight) * loss_k_reg
                + float(args.anti_smear_weight) * loss_anti_smear
            )

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            if float(args.grad_clip_norm) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip_norm))
            optimizer.step()

            train_total += float(total_loss.item())
            train_minutes += float(minutes_loss.item())
            train_rates += float(rates_loss.item())
            train_eff += float(eff_loss.item())
            train_rot += float(rot_loss.item())
            train_gate_bce += float(loss_gate_bce.item())
            train_minutes_out += float(loss_minutes_out.item())
            train_k_reg += float(loss_k_reg.item())
            train_anti_smear += float(loss_anti_smear.item())
            n_batches += 1

        train_metrics = {
            "loss": train_total / max(n_batches, 1),
            "minutes_mae": train_minutes / max(n_batches, 1),
            "rates_mae": train_rates / max(n_batches, 1),
            "eff_mae": train_eff / max(n_batches, 1),
            "rot_bce": train_rot / max(n_batches, 1),
            "gate_bce": train_gate_bce / max(n_batches, 1),
            "minutes_out": train_minutes_out / max(n_batches, 1),
            "k_reg": train_k_reg / max(n_batches, 1),
            "anti_smear": train_anti_smear / max(n_batches, 1),
        }
        val_metrics = _evaluate(
            model,
            val_loader,
            device=device,
            lambda_minutes=float(args.lambda_minutes),
            lambda_rates=float(args.lambda_rates),
            lambda_eff=float(args.lambda_eff),
            lambda_rot=float(args.lambda_rot),
            rot_target_threshold=float(args.rot_target_threshold),
            rotation_minutes_threshold=float(args.rotation_minutes_threshold),
            gate_bce_weight=float(args.gate_bce_weight),
            minutes_out_weight=float(args.minutes_out_weight),
            k_target=float(args.k_target),
            k_target_source=str(args.k_target_source),
            k_target_blend=float(args.k_target_blend),
            k_reg_weight=float(args.k_reg_weight),
            anti_smear_weight=float(args.anti_smear_weight),
            anti_smear_floor=float(args.anti_smear_floor),
        )
        row = {
            "epoch": int(epoch),
            "freeze_minutes_backbone": bool(freeze_now),
            "train_loss": float(train_metrics["loss"]),
            "train_minutes_mae": float(train_metrics["minutes_mae"]),
            "train_rates_mae": float(train_metrics["rates_mae"]),
            "train_eff_mae": float(train_metrics["eff_mae"]),
            "train_rot_bce": float(train_metrics["rot_bce"]),
            "train_gate_bce": float(train_metrics["gate_bce"]),
            "train_minutes_out": float(train_metrics["minutes_out"]),
            "train_k_reg": float(train_metrics["k_reg"]),
            "train_anti_smear": float(train_metrics["anti_smear"]),
            "val_loss": float(val_metrics["loss"]),
            "val_minutes_mae": float(val_metrics["minutes_mae"]),
            "val_rates_mae": float(val_metrics["rates_mae"]),
            "val_eff_mae": float(val_metrics["eff_mae"]),
            "val_rot_bce": float(val_metrics["rot_bce"]),
            "val_gate_bce": float(val_metrics["gate_bce"]),
            "val_minutes_out": float(val_metrics["minutes_out"]),
            "val_k_reg": float(val_metrics["k_reg"]),
            "val_anti_smear": float(val_metrics["anti_smear"]),
        }
        history.append(row)
        print(
            "[joint_train] epoch",
            f"{epoch}/{args.epochs}",
            f"train_loss={row['train_loss']:.4f}",
            f"val_loss={row['val_loss']:.4f}",
            f"val_minutes_mae={row['val_minutes_mae']:.4f}",
            f"val_rates_mae={row['val_rates_mae']:.4f}",
            f"val_eff_mae={row['val_eff_mae']:.4f}",
            f"val_smear={row['val_anti_smear']:.4f}",
        )
        if float(row["val_loss"]) < best_val:
            best_val = float(row["val_loss"])
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("No best model state captured")
    model.load_state_dict(best_state)

    run_id = f"{args.run_tag}_{_utc_now_compact()}"
    runs_root = Path(args.out_dir).expanduser().resolve()
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), run_dir / "model.pt")
    config.save(run_dir / "config.json")
    _save_json(run_dir / "feature_columns.json", {"columns": feature_cols})

    manifest = {
        "run_id": run_id,
        "created_at": _utc_now_iso(),
        "git_sha": _git_sha(),
        "dataset_dir": str(dataset_dir),
        "counts": {
            "train_team_games": split_meta["train_team_games"],
            "val_team_games": split_meta["val_team_games"],
            "train_examples": len(train_examples),
            "val_examples": len(val_examples),
            "feature_columns": len(feature_cols),
        },
        "split": split_meta,
        "args": vars(args),
        "history": history,
        "best": min(history, key=lambda r: float(r["val_loss"])) if history else None,
    }
    _save_json(run_dir / "manifest.json", manifest)

    print(f"[joint_train] wrote model        -> {run_dir / 'model.pt'}")
    print(f"[joint_train] wrote config       -> {run_dir / 'config.json'}")
    print(f"[joint_train] wrote feature cols -> {run_dir / 'feature_columns.json'}")
    print(f"[joint_train] wrote manifest     -> {run_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
