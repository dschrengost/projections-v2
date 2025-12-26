#!/usr/bin/env python3
"""DeepSets-style team-allocation minutes model training script.

This script trains a CPU-friendly neural network that learns to allocate exactly
240 minutes per team-game using a differentiable constraint layer.

Example usage:
    # Train on a parquet with features + labels
    uv run python -m scripts.experiments.minutes_teamalloc_deepsets \
        --train-parquet data/gold/features_minutes_v1/season=2024/month=12/features.parquet \
        --out-dir artifacts/experiments/team_alloc_v1 \
        --epochs 10 \
        --hidden 128 \
        --seed 7

    # With inference on new data
    uv run python -m scripts.experiments.minutes_teamalloc_deepsets \
        --train-parquet data/gold/features_minutes_v1/season=2024/month=11/features.parquet \
        --infer-parquet data/gold/features_minutes_v1/season=2024/month=12/features.parquet \
        --out-dir artifacts/experiments/team_alloc_v1 \
        --epochs 10

    # Override feature columns
    uv run python -m scripts.experiments.minutes_teamalloc_deepsets \
        --train-parquet data.parquet \
        --out-dir output/ \
        --features "minutes_avg_7d,minutes_avg_14d,starter_flag"

    # Reduce DNP smear (penalize false-positive minutes on minutes_actual == 0)
    uv run python -m scripts.experiments.minutes_teamalloc_deepsets \
        --train-parquet data.parquet \
        --out-dir output/ \
        --split-mode time --val-frac 0.2 \
        --lambda-fp-dnp 0.3 --fp-threshold 0.25 --fp-power 2

    # Hybrid allocation guardrail (restrict allocation to an eligible subset by prior minutes)
    uv run python -m scripts.experiments.minutes_teamalloc_deepsets \
        --train-parquet data.parquet \
        --out-dir output/ \
        --split-mode time --val-frac 0.2 \
        --hybrid-topk 11 --hybrid-prior-col sum_min_7d

Data Requirements:
    The input parquet must contain:
    - game_id: game identifier
    - team_id: team identifier
    - player_id: player identifier
    - label column (default 'minutes'): actual minutes played
    - numeric feature columns

Overtime Handling:
    Team total > ~242 indicates overtime. Options:
    - 'drop': exclude OT team-games from training
    - 'scale': scale player minutes to sum to 240 (default)

Output Artifacts:
    - model_checkpoint.pt: model state dict + metadata
    - val_predictions.parquet: validation predictions
    - infer_predictions.parquet: inference predictions (if --infer-parquet given)
"""

from __future__ import annotations

import argparse
import math
import logging
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from projections.models.team_alloc_deepsets import (
    AllocationLoss,
    DeepSetsAllocator,
    TeamAllocConfig,
    build_eligibility_mask,
    compute_metrics,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Key columns that should never be used as features
KEY_COLUMNS = {"game_id", "player_id", "team_id"}
EXCLUDED_COLUMNS = {
    "game_id",
    "player_id",
    "team_id",
    "home_team_id",
    "away_team_id",
    "opponent_team_id",
    "season",
    "game_date",
    "tip_ts",
    "feature_as_of_ts",
    "horizon_min",
}

# Regulation total minutes per team
REGULATION_MINUTES = 240.0
# Overtime detection threshold (5 OT minutes buffer)
OVERTIME_THRESHOLD = 242.0


@dataclass
class TeamGameBatch:
    """A batch of team-games for training/inference.

    All tensors have shape [B, P] or [B, P, F] where:
        B = batch size (number of team-games)
        P = max players per team
        F = number of features
    """

    features: torch.Tensor  # [B, P, F]
    mask: torch.Tensor  # [B, P] - 1.0 for real players, 0.0 for padding
    minutes_actual: torch.Tensor  # [B, P]
    prior_minutes: torch.Tensor  # [B, P] - hybrid eligibility prior (0 if unavailable)
    game_ids: np.ndarray  # [B]
    team_ids: np.ndarray  # [B]
    game_dates: np.ndarray  # [B] - None if unavailable
    player_ids: np.ndarray  # [B, P] - -1 for padding


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer_feature_columns(
    df: pd.DataFrame,
    label_col: str,
    explicit_features: list[str] | None = None,
) -> list[str]:
    """Infer feature columns from dataframe.

    If explicit_features is provided, use those (validating they exist).
    Otherwise, auto-detect numeric columns excluding keys and label.
    """
    if explicit_features:
        missing = set(explicit_features) - set(df.columns)
        if missing:
            raise ValueError(f"Explicit features not found in data: {missing}")
        return list(explicit_features)

    # Auto-detect: all numeric columns except excluded ones
    blocked = EXCLUDED_COLUMNS | {label_col}
    candidates = []
    for col in df.columns:
        if col in blocked:
            continue
        # Heuristic: don't treat ID columns as numeric features.
        if col.endswith("_id"):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            candidates.append(col)

    if not candidates:
        raise ValueError(
            "No feature columns detected. Provide explicit --features or check data."
        )

    return sorted(candidates)


def handle_overtime(
    df: pd.DataFrame,
    label_col: str,
    mode: str,
) -> pd.DataFrame:
    """Handle overtime games by drop or scale.

    Args:
        df: DataFrame with player rows
        label_col: column containing actual minutes
        mode: 'drop' or 'scale'

    Returns:
        DataFrame with overtime handled
    """
    # Compute team totals
    team_totals = df.groupby(["game_id", "team_id"])[label_col].transform("sum")
    ot_mask = team_totals > OVERTIME_THRESHOLD

    n_ot_rows = ot_mask.sum()
    n_ot_teams = df.loc[ot_mask, ["game_id", "team_id"]].drop_duplicates().shape[0]

    if n_ot_teams == 0:
        logger.info("No overtime games detected")
        return df

    if mode == "drop":
        logger.info(f"Dropping {n_ot_teams} OT team-games ({n_ot_rows} player rows)")
        return df[~ot_mask].copy()

    elif mode == "scale":
        logger.info(f"Scaling {n_ot_teams} OT team-games to 240 minutes")
        df = df.copy()
        # Scale factor: 240 / actual_total
        scale_factor = REGULATION_MINUTES / team_totals
        # Only apply to OT games
        df.loc[ot_mask, label_col] = df.loc[ot_mask, label_col] * scale_factor[ot_mask]
        return df

    else:
        raise ValueError(f"Unknown ot_mode: {mode}. Use 'drop' or 'scale'.")


def fill_missing_values(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Fill NaN values in feature columns with 0, logging counts."""
    df = df.copy()
    nan_counts = {}
    for col in feature_cols:
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            nan_counts[col] = nan_count
            df[col] = df[col].fillna(0.0)

    if nan_counts:
        total_nans = sum(nan_counts.values())
        logger.warning(f"Filled {total_nans} NaN values across {len(nan_counts)} columns")
        for col, count in sorted(nan_counts.items(), key=lambda x: -x[1])[:5]:
            logger.warning(f"  {col}: {count} NaNs")

    return df


def add_missingness_indicators(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    suffix: str = "_is_nan",
) -> tuple[pd.DataFrame, list[str]]:
    """Add missingness indicator columns for NaNs in feature columns.

    For each feature column with any NaN values, creates a new float32 indicator
    column `<feature><suffix>` that is 1.0 where the original value is NaN.
    """
    df = df.copy()
    new_cols: list[str] = []

    for col in feature_cols:
        if col not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if not df[col].isna().any():
            continue
        ind_col = f"{col}{suffix}"
        if ind_col in df.columns:
            continue
        df[ind_col] = df[col].isna().astype(np.float32)
        new_cols.append(ind_col)

    if new_cols:
        logger.info(f"Added {len(new_cols)} missingness indicator columns (suffix={suffix})")

    return df, new_cols


def ensure_infer_feature_columns(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    indicator_suffix: str = "_is_nan",
) -> pd.DataFrame:
    """Ensure inference dataframe contains all feature columns.

    - Missing base feature columns are created and filled with 0.0.
    - Missingness indicator columns (ending with indicator_suffix) are created from
      the corresponding base column's NaN mask; if the base column is missing, the
      indicator is set to 1.0.
    """
    df = df.copy()

    # Create base columns first (so indicators can be computed).
    for col in feature_cols:
        if col.endswith(indicator_suffix):
            base = col[: -len(indicator_suffix)]
            if base not in df.columns:
                df[base] = 0.0
        elif col not in df.columns:
            df[col] = 0.0

    # Create indicators (or overwrite if present but base is missingness changed).
    for col in feature_cols:
        if not col.endswith(indicator_suffix):
            continue
        base = col[: -len(indicator_suffix)]
        if base in df.columns:
            df[col] = df[base].isna().astype(np.float32)
        else:
            df[col] = 1.0

    return df


def group_by_team_game(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    max_players: int,
    truncate_sort_col: str | None = None,
    *,
    prior_col: str | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Group dataframe by (game_id, team_id) for batching.

    Returns:
        List of team-game dicts with features, labels, ids
        Count of teams truncated due to >max_players
    """
    grouped = df.groupby(["game_id", "team_id"])
    team_games = []
    truncated_count = 0

    for (game_id, team_id), group in grouped:
        if truncate_sort_col and truncate_sort_col in group.columns:
            group = group.sort_values(
                truncate_sort_col, ascending=False, na_position="last"
            )
        else:
            # Deterministic ordering without using labels.
            group = group.sort_values("player_id")

        if len(group) > max_players:
            truncated_count += 1
            group = group.head(max_players)

        game_date = None
        if "game_date" in group.columns:
            game_date = group["game_date"].iloc[0]

        team_games.append(
            {
                "game_id": game_id,
                "team_id": team_id,
                "game_date": game_date,
                "player_ids": group["player_id"].values,
                "features": group[feature_cols].values.astype(np.float32),
                "minutes": group[label_col].values.astype(np.float32),
                "prior_minutes": (
                    pd.to_numeric(group[prior_col], errors="coerce").fillna(0.0).values.astype(np.float32)
                    if (prior_col and prior_col in group.columns)
                    else np.zeros(len(group), dtype=np.float32)
                ),
            }
        )

    if truncated_count > 0:
        pct = 100 * truncated_count / len(team_games)
        sort_desc = (
            f"top by {truncate_sort_col}"
            if truncate_sort_col
            else "lowest by player_id"
        )
        logger.warning(
            f"Truncated {truncated_count} teams ({pct:.1f}%) to {max_players} players "
            f"(keeping {sort_desc}; label_col={label_col} is never used for truncation)"
        )

    return team_games, truncated_count


def split_train_val(
    team_games: list[dict[str, Any]],
    val_frac: float,
    seed: int,
    *,
    mode: str = "random",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split team-games into train/val by team-game key.

    Ensures players from the same team-game are never in both splits.
    """
    if mode not in {"random", "time"}:
        raise ValueError(f"Unknown split mode: {mode}. Use 'random' or 'time'.")

    indices = np.arange(len(team_games))
    if mode == "random":
        rng = np.random.RandomState(seed)
        rng.shuffle(indices)
    else:
        missing_date = [
            {"game_id": tg.get("game_id"), "team_id": tg.get("team_id")}
            for tg in team_games
            if tg.get("game_date") is None
        ]
        if missing_date:
            raise ValueError(
                "Time split requested but some team-games are missing game_date. "
                f"missing_count={len(missing_date)} sample={missing_date[:5]}"
            )

        dt = pd.to_datetime([tg["game_date"] for tg in team_games], errors="raise")
        # Stable sort: date, game_id, team_id
        game_ids = np.asarray([tg["game_id"] for tg in team_games], dtype=np.int64)
        team_ids = np.asarray([tg["team_id"] for tg in team_games], dtype=np.int64)
        order = np.lexsort((team_ids, game_ids, dt.view("int64")))
        indices = indices[order]

    n_val = int(len(team_games) * val_frac)
    val_indices = set(indices[-n_val:]) if mode == "time" else set(indices[:n_val])

    train = [tg for i, tg in enumerate(team_games) if i not in val_indices]
    val = [tg for i, tg in enumerate(team_games) if i in val_indices]

    if mode == "time" and val:
        train_dt = pd.to_datetime([tg["game_date"] for tg in train])
        val_dt = pd.to_datetime([tg["game_date"] for tg in val])
        logger.info(
            "Time split: "
            f"train={len(train)} ({train_dt.min().date()}..{train_dt.max().date()}) | "
            f"val={len(val)} ({val_dt.min().date()}..{val_dt.max().date()})"
        )
    else:
        logger.info(f"Split: {len(train)} train, {len(val)} val team-games")
    return train, val


def build_batch(
    team_games: list[dict[str, Any]],
    max_players: int,
    n_features: int,
) -> TeamGameBatch:
    """Build a padded batch from team-game dicts."""
    B = len(team_games)

    # Initialize tensors
    features = np.zeros((B, max_players, n_features), dtype=np.float32)
    mask = np.zeros((B, max_players), dtype=np.float32)
    minutes = np.zeros((B, max_players), dtype=np.float32)
    prior_minutes = np.zeros((B, max_players), dtype=np.float32)
    game_ids = np.zeros(B, dtype=np.int64)
    team_ids = np.zeros(B, dtype=np.int64)
    game_dates = np.full(B, None, dtype=object)
    player_ids = np.full((B, max_players), -1, dtype=np.int64)

    for b, tg in enumerate(team_games):
        n_players = len(tg["player_ids"])
        features[b, :n_players] = tg["features"]
        mask[b, :n_players] = 1.0
        minutes[b, :n_players] = tg["minutes"]
        prior_minutes[b, :n_players] = tg.get("prior_minutes", np.zeros(n_players, dtype=np.float32))
        game_ids[b] = tg["game_id"]
        team_ids[b] = tg["team_id"]
        game_dates[b] = tg.get("game_date")
        player_ids[b, :n_players] = tg["player_ids"]

    return TeamGameBatch(
        features=torch.from_numpy(features),
        mask=torch.from_numpy(mask),
        minutes_actual=torch.from_numpy(minutes),
        prior_minutes=torch.from_numpy(prior_minutes),
        game_ids=game_ids,
        team_ids=team_ids,
        game_dates=game_dates,
        player_ids=player_ids,
    )


def iterate_batches(
    team_games: list[dict[str, Any]],
    batch_size: int,
    max_players: int,
    n_features: int,
    shuffle: bool = True,
    seed: int | None = None,
) -> list[TeamGameBatch]:
    """Yield batches of team-games."""
    indices = list(range(len(team_games)))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(indices)

    batches = []
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        batch_tgs = [team_games[i] for i in batch_indices]
        batches.append(build_batch(batch_tgs, max_players, n_features))

    return batches


def _require_non_empty_team_masks(batch: TeamGameBatch) -> None:
    """Fail fast if any team-game has no valid players (mask_sum == 0)."""
    mask_sum = batch.mask.sum(dim=1)
    empty = torch.nonzero(mask_sum <= 0, as_tuple=False).squeeze(-1)
    if empty.numel() == 0:
        return

    offenders = []
    for i in empty.tolist()[:10]:
        offenders.append(
            {
                "b": int(i),
                "game_id": int(batch.game_ids[i]),
                "team_id": int(batch.team_ids[i]),
                "mask_sum": float(mask_sum[i].item()),
            }
        )
    raise ValueError(f"Found empty team masks (mask_sum==0). offenders={offenders}")


def _log_one_batch_debug(
    *,
    prefix: str,
    batch: TeamGameBatch,
    gate_logits: torch.Tensor,
    share_scores: torch.Tensor,
    minutes_hat: torch.Tensor,
    alloc_eps: float,
) -> None:
    """Log targeted allocator diagnostics for a single batch."""
    B, P, F = batch.features.shape

    mask = batch.mask
    mask_sum = mask.sum(dim=1)  # [B]
    valid_teams = mask_sum > 0

    gate_prob = torch.sigmoid(gate_logits)
    masked_gate = gate_prob[mask == 1]
    masked_scores = share_scores[mask == 1]

    team_min_sum = (minutes_hat * mask).sum(dim=1)  # [B]
    valid_team_sums = team_min_sum[valid_teams]

    frac_sum_lt_1 = float(((valid_team_sums < 1.0).float().mean().item())) if valid_team_sums.numel() else 0.0

    # Mirror allocator internals to detect numerator/denominator collapse.
    masked_share = share_scores.masked_fill(mask == 0, -1e9)
    shifted_share = masked_share - masked_share.max(dim=1, keepdim=True).values
    exp_scores = torch.exp(shifted_share) * mask
    weights = (gate_prob + alloc_eps) * exp_scores
    weight_sum = weights.sum(dim=1)  # [B]

    def _finite(x: torch.Tensor) -> bool:
        return bool(torch.isfinite(x).all().item())

    def _stats(x: torch.Tensor) -> tuple[float, float, float]:
        if x.numel() == 0:
            return float("nan"), float("nan"), float("nan")
        return float(x.min().item()), float(x.mean().item()), float(x.max().item())

    logger.info(
        f"{prefix} batch_shape B={B} P={P} F={F} | "
        f"mask_sum min/mean/max={_stats(mask_sum)}"
    )
    logger.info(
        f"{prefix} gate_prob(masked) min/mean/max={_stats(masked_gate)} | "
        f"share_scores(masked) min/mean/max={_stats(masked_scores)}"
    )
    logger.info(
        f"{prefix} minutes_sum min/mean/max={_stats(valid_team_sums)} | "
        f"frac(teams sum<1.0)={frac_sum_lt_1:.3f}"
    )
    logger.info(
        f"{prefix} weight_sum min/mean/max={_stats(weight_sum[valid_teams])} | "
        f"finite gate={_finite(gate_prob)} scores={_finite(share_scores)} minutes={_finite(minutes_hat)}"
    )


def train_epoch(
    model: DeepSetsAllocator,
    batches: list[TeamGameBatch],
    optimizer: torch.optim.Optimizer,
    loss_fn: AllocationLoss,
    *,
    debug_first_batch: bool = False,
    grad_clip_norm: float = 0.0,
) -> dict[str, float]:
    """Train for one epoch, return average losses."""
    model.train()
    total_losses = {k: 0.0 for k in ["gate_bce", "minutes_loss", "entropy", "sparsity", "fp_dnp", "total"]}
    n_batches = 0

    for batch in batches:
        _require_non_empty_team_masks(batch)
        optimizer.zero_grad()

        gate_logits, share_scores, minutes_hat = model(batch.features, batch.mask)

        if debug_first_batch and n_batches == 0:
            _log_one_batch_debug(
                prefix="[train_debug]",
                batch=batch,
                gate_logits=gate_logits,
                share_scores=share_scores,
                minutes_hat=minutes_hat,
                alloc_eps=float(model.config.alloc_eps),
            )

        loss, components = loss_fn(
            gate_logits, minutes_hat, batch.minutes_actual, batch.mask
        )
        if debug_first_batch and n_batches == 0 and float(getattr(loss_fn, "lambda_fp_dnp", 0.0)) > 0:
            logger.info(
                "[train_debug] fp_dnp penalty "
                f"value={components.get('fp_dnp', 0.0):.6f} "
                f"(lambda={float(loss_fn.lambda_fp_dnp):.3g} thr={float(loss_fn.fp_threshold):.3g} p={int(loss_fn.fp_power)})"
            )

        loss.backward()
        if grad_clip_norm and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()

        for k, v in components.items():
            total_losses[k] += v
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in total_losses.items()}


@torch.no_grad()
def evaluate(
    model: DeepSetsAllocator,
    batches: list[TeamGameBatch],
    loss_fn: AllocationLoss,
    top_k: int = 8,
    gate_threshold: float = 0.5,
    *,
    debug_first_batch: bool = False,
) -> tuple[dict[str, float], dict[str, float]]:
    """Evaluate model, return losses and metrics."""
    model.eval()
    total_losses = {k: 0.0 for k in ["gate_bce", "minutes_loss", "entropy", "sparsity", "fp_dnp", "total"]}
    all_metrics = {
        "mae": 0.0,
        "gate_precision": 0.0,
        "gate_recall": 0.0,
        "gate_f1": 0.0,
        f"top_{top_k}_overlap": 0.0,
        "max_sum_deviation": 0.0,
    }
    n_batches = 0

    for batch in batches:
        _require_non_empty_team_masks(batch)
        gate_logits, share_scores, minutes_hat = model(batch.features, batch.mask)

        if debug_first_batch and n_batches == 0:
            _log_one_batch_debug(
                prefix="[val_debug]",
                batch=batch,
                gate_logits=gate_logits,
                share_scores=share_scores,
                minutes_hat=minutes_hat,
                alloc_eps=float(model.config.alloc_eps),
            )

        _, components = loss_fn(
            gate_logits, minutes_hat, batch.minutes_actual, batch.mask
        )
        if debug_first_batch and n_batches == 0 and float(getattr(loss_fn, "lambda_fp_dnp", 0.0)) > 0:
            logger.info(
                "[val_debug] fp_dnp penalty "
                f"value={components.get('fp_dnp', 0.0):.6f} "
                f"(lambda={float(loss_fn.lambda_fp_dnp):.3g} thr={float(loss_fn.fp_threshold):.3g} p={int(loss_fn.fp_power)})"
            )

        metrics = compute_metrics(
            gate_logits,
            minutes_hat,
            batch.minutes_actual,
            batch.mask,
            top_k=top_k,
            gate_threshold=gate_threshold,
        )

        for k, v in components.items():
            total_losses[k] += v
        for k, v in metrics.items():
            if k in all_metrics:
                if k == "max_sum_deviation":
                    all_metrics[k] = max(all_metrics[k], v)
                else:
                    all_metrics[k] += v
        n_batches += 1

    avg_losses = {k: v / max(n_batches, 1) for k, v in total_losses.items()}
    avg_metrics = {
        k: v / max(n_batches, 1) if k != "max_sum_deviation" else v
        for k, v in all_metrics.items()
    }

    return avg_losses, avg_metrics


@torch.no_grad()
def predict_to_dataframe(
    model: DeepSetsAllocator,
    batches: list[TeamGameBatch],
    *,
    hybrid_topk: int = 0,
    hybrid_prior_threshold: float = float("-inf"),
) -> pd.DataFrame:
    """Generate predictions as a dataframe."""
    model.eval()
    rows = []

    hybrid_enabled = (hybrid_topk > 0) or math.isfinite(hybrid_prior_threshold)

    for batch in batches:
        _require_non_empty_team_masks(batch)
        gate_logits, share_scores, minutes_hat = model(batch.features, batch.mask)

        minutes_hat_hybrid = None
        if hybrid_enabled:
            eligible_mask = build_eligibility_mask(
                batch.prior_minutes,
                batch.mask,
                topk=int(hybrid_topk),
                prior_threshold=float(hybrid_prior_threshold),
            )
            minutes_hat_hybrid = model._allocate_minutes(
                gate_logits,
                share_scores,
                batch.mask,
                eligible_mask=eligible_mask,
            )

        B, P = minutes_hat.shape
        for b in range(B):
            for p in range(P):
                if batch.mask[b, p] == 0:
                    continue  # skip padding
                rows.append({
                    "game_id": int(batch.game_ids[b]),
                    "team_id": int(batch.team_ids[b]),
                    "game_date": batch.game_dates[b],
                    "player_id": int(batch.player_ids[b, p]),
                    "minutes_pred": float(minutes_hat[b, p]),
                    "minutes_actual": float(batch.minutes_actual[b, p]),
                    "gate_prob": float(torch.sigmoid(gate_logits[b, p])),
                })
                if minutes_hat_hybrid is not None:
                    rows[-1]["minutes_pred_hybrid"] = float(minutes_hat_hybrid[b, p])
                    rows[-1]["hybrid_prior_minutes"] = float(batch.prior_minutes[b, p])

    return pd.DataFrame(rows)


def save_checkpoint(
    path: Path,
    model: DeepSetsAllocator,
    config: TeamAllocConfig,
    feature_cols: list[str],
    args: argparse.Namespace,
) -> None:
    """Save model checkpoint with metadata."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config.to_metadata(),
        "feature_cols": feature_cols,
        "max_players": args.max_players,
        "hyperparams": {
            "hidden": args.hidden,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "grad_clip_norm": args.grad_clip_norm,
            "epochs": args.epochs,
            "alloc_eps": args.alloc_eps,
            "denom_eps": args.denom_eps,
            "alpha_gate": args.alpha_gate,
            "no_gate_bias_init": args.no_gate_bias_init,
            "no_gate_pos_weight": args.no_gate_pos_weight,
            "beta_min": args.beta_min,
            "gamma_entropy": args.gamma_entropy,
            "delta_sparsity": args.delta_sparsity,
            "lambda_fp_dnp": args.lambda_fp_dnp,
            "fp_threshold": args.fp_threshold,
            "fp_power": args.fp_power,
            "ot_mode": args.ot_mode,
            "split_mode": args.split_mode,
            "val_frac": args.val_frac,
            "seed": args.seed,
            "truncate_sort_col": args.truncate_sort_col,
            "drop_features": args.drop_features,
            "filter_horizon_min": args.filter_horizon_min,
            "allow_duplicate_player_rows": args.allow_duplicate_player_rows,
            "gate_threshold": args.gate_threshold,
            "hybrid_topk": args.hybrid_topk,
            "hybrid_prior_threshold": args.hybrid_prior_threshold,
            "hybrid_prior_col": args.hybrid_prior_col,
        },
    }
    torch.save(checkpoint, path)
    logger.info(f"Saved checkpoint to {path}")


def prepare_infer_data(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    max_players: int,
    truncate_sort_col: str | None,
    *,
    prior_col: str | None = None,
) -> list[dict[str, Any]]:
    """Prepare inference data, filling missing features with 0."""
    df = df.copy()

    df = ensure_infer_feature_columns(df, feature_cols)

    # Fill any NaNs
    df = fill_missing_values(df, feature_cols)

    # Create fake label if missing (for batching structure)
    if label_col not in df.columns:
        df[label_col] = np.nan

    team_games, _ = group_by_team_game(
        df,
        feature_cols,
        label_col,
        max_players,
        truncate_sort_col=truncate_sort_col,
        prior_col=prior_col,
    )
    return team_games


def _estimate_play_rate(team_games: list[dict[str, Any]]) -> tuple[float, int, int]:
    """Estimate play rate (minutes > 0) on the provided team-games."""
    pos = 0
    total = 0
    for tg in team_games:
        mins = np.asarray(tg["minutes"])
        total += int(mins.size)
        pos += int((mins > 0).sum())
    neg = total - pos
    p_play = (pos / total) if total > 0 else 0.0
    return float(p_play), int(pos), int(neg)


def _log_val_prediction_metrics(
    preds: pd.DataFrame,
    *,
    top_k: int,
    name: str,
    pred_col: str = "minutes_pred",
) -> None:
    """Compute and log diagnostic metrics from a predictions dataframe."""
    required = {"game_id", "team_id", "player_id", pred_col, "minutes_actual"}
    missing = required - set(preds.columns)
    if missing:
        logger.warning(f"{name}: skipping parquet metrics (missing columns): {sorted(missing)}")
        return

    df = preds.copy()
    df[pred_col] = pd.to_numeric(df[pred_col], errors="coerce")
    df["minutes_actual"] = pd.to_numeric(df["minutes_actual"], errors="coerce")

    labeled = df["minutes_actual"].notna()
    if not labeled.any():
        logger.info(f"{name}: no labels present in predictions; skipping label-based metrics")
        return

    # If there are duplicate (game_id, team_id, player_id) rows, report and de-duplicate for metrics.
    dup = int(df[labeled].duplicated(subset=["game_id", "team_id", "player_id"]).sum())
    if dup > 0:
        logger.warning(
            f"{name}: found {dup} duplicate (game_id, team_id, player_id) rows; "
            "aggregating by mean for metrics"
        )
        agg: dict[str, tuple[str, str]] = {
            pred_col: (pred_col, "mean"),
            "minutes_actual": ("minutes_actual", "mean"),
        }
        if "gate_prob" in df.columns:
            agg["gate_prob"] = ("gate_prob", "mean")
        if "game_date" in df.columns:
            agg["game_date"] = ("game_date", "first")
        df = df.groupby(["game_id", "team_id", "player_id"], as_index=False).agg(**agg)

    abs_err = (df[pred_col] - df["minutes_actual"]).abs()
    mae = float(abs_err.mean())
    mask_10 = df["minutes_actual"] >= 10
    n_10 = int(mask_10.sum())
    mae_10 = float(abs_err[mask_10].mean()) if n_10 > 0 else float("nan")

    smear = (df[pred_col] > 0) & (df[pred_col] < 3)
    smear_count = int(smear.sum())
    smear_frac = float(smear.mean())
    smear_per_team = df.assign(_smear=smear.astype(int)).groupby(["game_id", "team_id"])["_smear"].sum()
    smear_avg_per_team = float(smear_per_team.mean()) if len(smear_per_team) else 0.0

    smear_dnp = smear & (df["minutes_actual"] == 0)
    smear_dnp_count = int(smear_dnp.sum())
    smear_dnp_frac = float(smear_dnp.mean())

    dnp = df["minutes_actual"] == 0
    if dnp.any():
        mean_pred_on_dnp = float(df.loc[dnp, pred_col].mean())
        p90_pred_on_dnp = float(df.loc[dnp, pred_col].quantile(0.9))
    else:
        mean_pred_on_dnp = float("nan")
        p90_pred_on_dnp = float("nan")

    # Leak DNP minutes per team-game: sum of predicted minutes where minutes_actual == 0.
    team_totals = df.groupby(["game_id", "team_id"])[pred_col].sum()
    dnp_totals = df.loc[dnp].groupby(["game_id", "team_id"])[pred_col].sum()
    dnp_totals = dnp_totals.reindex(team_totals.index, fill_value=0.0)
    if len(dnp_totals):
        leak_p50, leak_p90, leak_p99 = dnp_totals.quantile([0.5, 0.9, 0.99]).tolist()
    else:
        leak_p50 = leak_p90 = leak_p99 = float("nan")

    # Top-k overlap from dataframe (membership only).
    overlaps: list[float] = []
    team_sizes = df.groupby(["game_id", "team_id"]).size()
    for _, g in df.groupby(["game_id", "team_id"]):
        k = min(top_k, len(g))
        if k <= 0:
            continue
        top_actual = set(g.nlargest(k, "minutes_actual")["player_id"].tolist())
        top_pred = set(g.nlargest(k, pred_col)["player_id"].tolist())
        overlaps.append(len(top_actual & top_pred) / k)
    topk_overlap = float(np.mean(overlaps)) if overlaps else 0.0

    frac_small_candidate = float((team_sizes < top_k).mean()) if len(team_sizes) else 0.0
    if len(team_sizes):
        size_min = int(team_sizes.min())
        size_p50 = float(team_sizes.quantile(0.5))
        size_p90 = float(team_sizes.quantile(0.9))
        size_max = int(team_sizes.max())
    else:
        size_min = 0
        size_p50 = float("nan")
        size_p90 = float("nan")
        size_max = 0

    logger.info(
        f"{name}: parquet metrics ({pred_col}) | "
        f"MAE={mae:.3f} | MAE>=10={mae_10:.3f} (n={n_10}) | "
        f"Top-{top_k} overlap={topk_overlap:.3f} | "
        f"smear(0<pred<3) count={smear_count} frac={smear_frac:.3f} avg/team={smear_avg_per_team:.2f} | "
        f"smear on DNP count={smear_dnp_count} frac={smear_dnp_frac:.3f} | "
        f"leak_dnp(team sum on y==0) p50/p90/p99={leak_p50:.1f}/{leak_p90:.1f}/{leak_p99:.1f} | "
        f"DNP pred mean/p90={mean_pred_on_dnp:.3f}/{p90_pred_on_dnp:.3f} | "
        f"teams(n<{top_k}) frac={frac_small_candidate:.3f} | "
        f"team_size min/p50/p90/max={size_min}/{size_p50:.1f}/{size_p90:.1f}/{size_max}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train DeepSets-style team-allocation minutes model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--train-parquet",
        type=Path,
        required=True,
        help="Path to training parquet file",
    )
    parser.add_argument(
        "--infer-parquet",
        type=Path,
        default=None,
        help="Optional path to inference parquet file",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for model and predictions",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default="minutes",
        help="Label column name (default: minutes)",
    )
    parser.add_argument(
        "--features",
        type=str,
        default=None,
        help="Comma-separated feature column names (auto-detect if not provided)",
    )
    parser.add_argument(
        "--drop-features",
        type=str,
        default=None,
        help="Comma-separated feature names to drop (e.g. sum_min_7d,roll_mean_10)",
    )
    parser.add_argument(
        "--filter-horizon-min",
        type=int,
        default=None,
        help="If present, filter input rows to a single horizon_min value",
    )
    parser.add_argument(
        "--allow-duplicate-player-rows",
        action="store_true",
        help="Allow duplicate (game_id, team_id, player_id) rows (not recommended)",
    )
    parser.add_argument(
        "--max-players",
        type=int,
        default=21,
        help="Maximum players per team (default: 21)",
    )
    parser.add_argument(
        "--truncate-sort-col",
        type=str,
        default="auto",
        help=(
            "Column to sort by when truncating to --max-players. "
            "Use 'auto' to pick a pre-game proxy (default: auto)."
        ),
    )
    parser.add_argument(
        "--hidden",
        type=int,
        default=128,
        help="Hidden dimension (default: 128)",
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=2,
        help="Number of encoder layers (default: 2)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate (default: 0.1)",
    )
    parser.add_argument(
        "--alloc-eps",
        type=float,
        default=1e-4,
        help="Allocator baseline added to gate to prevent collapse (default: 1e-4)",
    )
    parser.add_argument(
        "--denom-eps",
        type=float,
        default=1e-12,
        help="Allocator denominator clamp for numerical stability (default: 1e-12)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay for optimizer (default: 1e-4)",
    )
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=1.0,
        help="Global grad norm clip (0 disables; default: 1.0)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size in team-games (default: 128)",
    )
    parser.add_argument(
        "--alpha-gate",
        type=float,
        default=0.5,
        help="Weight for gate BCE loss (default: 0.5)",
    )
    parser.add_argument(
        "--no-gate-bias-init",
        action="store_true",
        help="Disable initializing gate_head bias from empirical play rate",
    )
    parser.add_argument(
        "--no-gate-pos-weight",
        action="store_true",
        help="Disable pos_weight for gate BCE (class imbalance handling)",
    )
    parser.add_argument(
        "--beta-min",
        type=float,
        default=1.0,
        help="Weight for minutes loss (default: 1.0)",
    )
    parser.add_argument(
        "--gamma-entropy",
        type=float,
        default=0.01,
        help="Weight for entropy penalty (default: 0.01)",
    )
    parser.add_argument(
        "--delta-sparsity",
        type=float,
        default=0.0,
        help="Weight for L1 sparsity penalty on gates (default: 0.0)",
    )
    parser.add_argument(
        "--lambda-fp-dnp",
        type=float,
        default=0.0,
        help=(
            "Weight for false-positive DNP minutes penalty (minutes_actual==0). "
            "Adds lambda * mean(relu(minutes_pred - fp_threshold)^fp_power) (default: 0.0)"
        ),
    )
    parser.add_argument(
        "--fp-threshold",
        type=float,
        default=0.25,
        help="Minutes threshold for false-positive DNP penalty (default: 0.25)",
    )
    parser.add_argument(
        "--fp-power",
        type=int,
        choices=[1, 2],
        default=2,
        help="Power for false-positive DNP penalty: 1 or 2 (default: 2)",
    )
    parser.add_argument(
        "--ot-mode",
        type=str,
        choices=["drop", "scale"],
        default="scale",
        help="Overtime handling: drop or scale (default: scale)",
    )
    parser.add_argument(
        "--split-mode",
        type=str,
        choices=["random", "time"],
        default="random",
        help="Validation split mode: random or time (latest fraction by game_date)",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.15,
        help="Validation fraction (default: 0.15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed (default: 7)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Top-k for rotation overlap metric (default: 8)",
    )
    parser.add_argument(
        "--gate-threshold",
        type=float,
        default=0.5,
        help="Threshold for gate metrics only; does not affect minutes allocation (default: 0.5)",
    )
    parser.add_argument(
        "--hybrid-topk",
        type=int,
        default=0,
        help="If >0, restrict allocation to top-K by prior minutes (default: 0 = off)",
    )
    parser.add_argument(
        "--hybrid-prior-threshold",
        type=float,
        default=float("-inf"),
        help="If finite, require prior minutes >= threshold for eligibility (default: -inf = off)",
    )
    parser.add_argument(
        "--hybrid-prior-col",
        type=str,
        default=None,
        help=(
            "Optional dataframe column to use as prior minutes for hybrid eligibility "
            "(e.g. sum_min_7d, roll_mean_10, min_last5, or an LGBM minutes column). "
            "If omitted, auto-selects a proxy if available."
        ),
    )
    parser.add_argument(
        "--debug-batch",
        type=str,
        choices=["none", "first_epoch", "each_epoch"],
        default="first_epoch",
        help="Log allocator diagnostics for the first batch (default: first_epoch)",
    )

    args = parser.parse_args()

    # Setup
    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Loss hyperparams: "
        f"alpha_gate={args.alpha_gate} beta_min={args.beta_min} "
        f"gamma_entropy={args.gamma_entropy} delta_sparsity={args.delta_sparsity} | "
        f"lambda_fp_dnp={args.lambda_fp_dnp} fp_threshold={args.fp_threshold} fp_power={args.fp_power}"
    )

    # Load and validate training data
    logger.info(f"Loading training data from {args.train_parquet}")
    if not args.train_parquet.exists():
        logger.error(f"Training parquet not found: {args.train_parquet}")
        return 1

    train_df = pd.read_parquet(args.train_parquet)
    logger.info(f"Loaded {len(train_df)} rows")

    # Optional horizon filter (useful for multi-horizon training parquets)
    if args.filter_horizon_min is not None:
        if "horizon_min" not in train_df.columns:
            raise ValueError("--filter-horizon-min provided but horizon_min column not found")
        before = len(train_df)
        train_df = train_df[train_df["horizon_min"] == args.filter_horizon_min].copy()
        logger.info(
            f"Filtered horizon_min={args.filter_horizon_min}: {before} -> {len(train_df)} rows"
        )

    # Validate required columns
    required_cols = {"game_id", "team_id", "player_id", args.label_col}
    missing = required_cols - set(train_df.columns)
    if missing:
        logger.error(f"Missing required columns: {missing}")
        return 1

    # Parse explicit features if provided
    explicit_features = None
    if args.features:
        explicit_features = [f.strip() for f in args.features.split(",")]

    # Infer feature columns
    feature_cols = infer_feature_columns(train_df, args.label_col, explicit_features)
    if args.drop_features:
        to_drop = {f.strip() for f in args.drop_features.split(",") if f.strip()}
        if to_drop:
            missing_drop = sorted([f for f in to_drop if f not in feature_cols])
            if missing_drop:
                logger.warning(f"Requested --drop-features not in feature set: {missing_drop}")
            feature_cols = [f for f in feature_cols if f not in to_drop]
    logger.info(f"Using {len(feature_cols)} feature columns")
    if len(feature_cols) <= 10:
        logger.info(f"Features: {feature_cols}")

    # Handle overtime
    train_df = handle_overtime(train_df, args.label_col, args.ot_mode)

    # Fail fast on duplicate player rows (these inflate overlap and break the set semantics).
    key_cols = ["game_id", "team_id", "player_id"]
    dup_rows = int(train_df.duplicated(subset=key_cols).sum())
    if dup_rows > 0:
        msg = (
            "Found duplicate (game_id, team_id, player_id) rows in training data "
            f"(dup_rows={dup_rows}). This will treat duplicates as separate players and "
            "can inflate overlap/metrics. Filter/dedupe your input (e.g. pick one horizon_min)."
        )
        if args.allow_duplicate_player_rows:
            logger.warning(msg)
        else:
            raise ValueError(msg)

    # Add missingness indicators BEFORE filling NaNs (so the model can learn missingness).
    train_df, missing_indicators = add_missingness_indicators(train_df, feature_cols)
    if missing_indicators:
        feature_cols = feature_cols + missing_indicators
        logger.info(f"Using {len(feature_cols)} feature columns after indicators")

    # Fill missing values
    train_df = fill_missing_values(train_df, feature_cols)

    dropped_features = set()
    if args.drop_features:
        dropped_features = {f.strip() for f in args.drop_features.split(",") if f.strip()}

    hybrid_enabled = (args.hybrid_topk > 0) or math.isfinite(args.hybrid_prior_threshold)
    hybrid_prior_col = None
    if hybrid_enabled:
        if args.hybrid_prior_col is not None:
            if args.hybrid_prior_col not in train_df.columns:
                raise ValueError(f"--hybrid-prior-col not found in data: {args.hybrid_prior_col}")
            hybrid_prior_col = args.hybrid_prior_col
        else:
            for cand in ("sum_min_7d", "roll_mean_10", "min_last5"):
                if cand in dropped_features:
                    continue
                if cand in train_df.columns:
                    hybrid_prior_col = cand
                    break
        if hybrid_prior_col:
            logger.info(
                "Hybrid allocation enabled: "
                f"topk={args.hybrid_topk} prior_threshold={args.hybrid_prior_threshold} prior_col={hybrid_prior_col}"
            )
        else:
            logger.warning(
                "Hybrid allocation enabled but no prior column available; "
                "falling back to 0 priors (eligibility will be arbitrary for ties). "
                f"topk={args.hybrid_topk} prior_threshold={args.hybrid_prior_threshold}"
            )

    truncate_sort_col = None
    if args.truncate_sort_col != "auto":
        if args.truncate_sort_col not in train_df.columns:
            raise ValueError(f"--truncate-sort-col not found in data: {args.truncate_sort_col}")
        truncate_sort_col = args.truncate_sort_col
        if truncate_sort_col == args.label_col:
            logger.warning(
                "truncate_sort_col == label_col (label leakage risk). "
                "Prefer a pre-game proxy like prior minutes / play prob."
            )
    else:
        for cand in ("prior_play_prob", "sum_min_7d", "min_last5", "min_last3"):
            if cand in dropped_features:
                continue
            if cand in train_df.columns and cand != args.label_col:
                truncate_sort_col = cand
                break
        if truncate_sort_col:
            logger.info(f"Auto-selected truncate_sort_col={truncate_sort_col}")
        else:
            logger.info("Auto-selected truncate_sort_col=None (fallback to player_id ordering)")

    # Group by team-game
    team_games, truncated = group_by_team_game(
        train_df,
        feature_cols,
        args.label_col,
        args.max_players,
        truncate_sort_col=truncate_sort_col,
        prior_col=hybrid_prior_col,
    )
    logger.info(f"Created {len(team_games)} team-games")

    # Split train/val
    train_tgs, val_tgs = split_train_val(
        team_games, args.val_frac, args.seed, mode=args.split_mode
    )

    # Estimate play rate on the actual training batches (post-truncation), for gate init/loss weighting.
    p_play, pos, neg = _estimate_play_rate(train_tgs)
    if pos + neg > 0:
        logger.info(f"Train gate targets: p_play={p_play:.3f} (pos={pos}, neg={neg})")

    # Build batches
    n_features = len(feature_cols)
    train_batches = iterate_batches(
        train_tgs, args.batch_size, args.max_players, n_features, shuffle=True, seed=args.seed
    )
    val_batches = iterate_batches(
        val_tgs, args.batch_size, args.max_players, n_features, shuffle=False
    )

    logger.info(f"Train: {len(train_batches)} batches, Val: {len(val_batches)} batches")

    # Initialize model
    config = TeamAllocConfig(
        n_features=n_features,
        hidden_dim=args.hidden,
        n_layers=args.n_layers,
        dropout=args.dropout,
        alloc_eps=args.alloc_eps,
        denom_eps=args.denom_eps,
    )
    model = DeepSetsAllocator(config)
    logger.info(f"Model initialized: {sum(p.numel() for p in model.parameters())} parameters")

    if not args.no_gate_bias_init and (pos + neg) > 0:
        p = float(np.clip(p_play, 1e-4, 1.0 - 1e-4))
        gate_bias = float(np.log(p / (1.0 - p)))
        with torch.no_grad():
            model.gate_head.bias.fill_(gate_bias)
        logger.info(f"Initialized gate_head.bias={gate_bias:.3f} from p_play={p:.3f}")

    # Loss and optimizer
    gate_pos_weight = None
    if not args.no_gate_pos_weight and pos > 0:
        if neg > pos:
            gate_pos_weight = float(neg / pos)
            logger.info(f"Using gate BCE pos_weight={gate_pos_weight:.3f}")
        else:
            logger.info(
                "Skipping gate BCE pos_weight (positives not minority): "
                f"pos={pos} neg={neg}"
            )

    loss_fn = AllocationLoss(
        alpha_gate=args.alpha_gate,
        beta_min=args.beta_min,
        gamma_entropy=args.gamma_entropy,
        delta_sparsity=args.delta_sparsity,
        gate_pos_weight=gate_pos_weight,
        lambda_fp_dnp=args.lambda_fp_dnp,
        fp_threshold=args.fp_threshold,
        fp_power=args.fp_power,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Training loop
    logger.info("Starting training...")
    for epoch in range(args.epochs):
        debug_this_epoch = (args.debug_batch == "each_epoch") or (
            args.debug_batch == "first_epoch" and epoch == 0
        )

        # Reshuffle training batches each epoch
        train_batches = iterate_batches(
            train_tgs, args.batch_size, args.max_players, n_features,
            shuffle=True, seed=args.seed + epoch
        )

        train_losses = train_epoch(
            model,
            train_batches,
            optimizer,
            loss_fn,
            debug_first_batch=debug_this_epoch,
            grad_clip_norm=args.grad_clip_norm,
        )
        val_losses, val_metrics = evaluate(
            model,
            val_batches,
            loss_fn,
            top_k=args.top_k,
            gate_threshold=args.gate_threshold,
            debug_first_batch=debug_this_epoch,
        )

        # Log results
        logger.info(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Train Loss: {train_losses['total']:.4f} | "
            f"Val Loss: {val_losses['total']:.4f}"
        )
        logger.info(
            f"  Val MAE: {val_metrics['mae']:.2f} min | "
            f"Gate F1: {val_metrics['gate_f1']:.3f} "
            f"(P={val_metrics['gate_precision']:.3f}, R={val_metrics['gate_recall']:.3f})"
        )
        logger.info(
            f"  Top-{args.top_k} Overlap: {val_metrics[f'top_{args.top_k}_overlap']:.3f} | "
            f"Max Sum Dev: {val_metrics['max_sum_deviation']:.4f}"
        )

    # Save checkpoint
    checkpoint_path = args.out_dir / "model_checkpoint.pt"
    save_checkpoint(checkpoint_path, model, config, feature_cols, args)

    # Save validation predictions
    logger.info("Generating validation predictions...")
    expected_val_rows = int(sum(int(b.mask.sum().item()) for b in val_batches))
    val_preds = predict_to_dataframe(
        model,
        val_batches,
        hybrid_topk=int(args.hybrid_topk),
        hybrid_prior_threshold=float(args.hybrid_prior_threshold),
    )
    val_preds_path = args.out_dir / "val_predictions.parquet"
    val_preds.to_parquet(val_preds_path, index=False)
    logger.info(f"Saved {len(val_preds)} val predictions to {val_preds_path}")
    if len(val_preds) != expected_val_rows:
        logger.warning(
            "val_predictions row count mismatch (possible player drop). "
            f"expected_rows={expected_val_rows} actual_rows={len(val_preds)}"
        )

    dup = int(val_preds.duplicated(subset=["game_id", "team_id", "player_id"]).sum())
    if dup > 0:
        logger.warning(f"val_predictions has {dup} duplicate (game_id, team_id, player_id) rows")

    # Sanity check: verify sum-to-240 on predictions
    pred_sums = val_preds.groupby(["game_id", "team_id"])["minutes_pred"].sum()
    dev = (pred_sums - 240.0).abs()
    max_dev = dev.max()
    logger.info(f"Val predictions max sum deviation from 240: {max_dev:.6f}")
    if len(dev) > 0:
        q50, q90, q99 = dev.quantile([0.5, 0.9, 0.99]).tolist()
        frac_gt_1e6 = float((dev > 1e-6).mean())
        logger.info(
            "Val predictions sum deviation distribution "
            f"(min={dev.min():.6g}, p50={q50:.6g}, p90={q90:.6g}, p99={q99:.6g}, max={max_dev:.6g}, "
            f"frac>1e-6={frac_gt_1e6:.3f})"
        )

    if max_dev > 1e-3:
        logger.warning(f"Sum-to-240 constraint violation detected: {max_dev:.6f}")

    # Recompute from parquet on disk (guards against write-time schema issues)
    val_preds_disk = pd.read_parquet(val_preds_path)
    disk_sums = val_preds_disk.groupby(["game_id", "team_id"])["minutes_pred"].sum()
    disk_max_dev = (disk_sums - 240.0).abs().max()
    logger.info(f"Val predictions (from parquet) max sum deviation from 240: {disk_max_dev:.6f}")
    _log_val_prediction_metrics(val_preds_disk, top_k=args.top_k, name="val_predictions", pred_col="minutes_pred")
    if "minutes_pred_hybrid" in val_preds_disk.columns:
        hybrid_sums = val_preds_disk.groupby(["game_id", "team_id"])["minutes_pred_hybrid"].sum()
        hybrid_dev = (hybrid_sums - 240.0).abs()
        logger.info(
            "Val predictions (hybrid) max sum deviation from 240: "
            f"{float(hybrid_dev.max() if len(hybrid_dev) else 0.0):.6f}"
        )
        _log_val_prediction_metrics(
            val_preds_disk,
            top_k=args.top_k,
            name="val_predictions_hybrid",
            pred_col="minutes_pred_hybrid",
        )

    # Inference if requested
    if args.infer_parquet:
        if not args.infer_parquet.exists():
            logger.error(f"Inference parquet not found: {args.infer_parquet}")
            return 1

        logger.info(f"Loading inference data from {args.infer_parquet}")
        infer_df = pd.read_parquet(args.infer_parquet)

        if args.filter_horizon_min is not None and "horizon_min" in infer_df.columns:
            before = len(infer_df)
            infer_df = infer_df[infer_df["horizon_min"] == args.filter_horizon_min].copy()
            logger.info(
                f"Filtered inference horizon_min={args.filter_horizon_min}: "
                f"{before} -> {len(infer_df)} rows"
            )

        key_cols = ["game_id", "team_id", "player_id"]
        dup_rows_infer = int(infer_df.duplicated(subset=key_cols).sum())
        if dup_rows_infer > 0:
            msg = (
                "Found duplicate (game_id, team_id, player_id) rows in inference data "
                f"(dup_rows={dup_rows_infer}). This will treat duplicates as separate players."
            )
            if args.allow_duplicate_player_rows:
                logger.warning(msg)
            else:
                raise ValueError(msg)

        infer_tgs = prepare_infer_data(
            infer_df,
            feature_cols,
            args.label_col,
            args.max_players,
            truncate_sort_col=truncate_sort_col,
            prior_col=hybrid_prior_col,
        )
        infer_batches = iterate_batches(
            infer_tgs, args.batch_size, args.max_players, n_features, shuffle=False
        )

        logger.info(f"Generating predictions for {len(infer_tgs)} team-games...")
        expected_infer_rows = int(sum(int(b.mask.sum().item()) for b in infer_batches))
        infer_preds = predict_to_dataframe(
            model,
            infer_batches,
            hybrid_topk=int(args.hybrid_topk),
            hybrid_prior_threshold=float(args.hybrid_prior_threshold),
        )
        infer_preds_path = args.out_dir / "infer_predictions.parquet"
        infer_preds.to_parquet(infer_preds_path, index=False)
        logger.info(f"Saved {len(infer_preds)} infer predictions to {infer_preds_path}")
        if len(infer_preds) != expected_infer_rows:
            logger.warning(
                "infer_predictions row count mismatch (possible player drop). "
                f"expected_rows={expected_infer_rows} actual_rows={len(infer_preds)}"
            )

        # Sanity check inference sums
        infer_sums = infer_preds.groupby(["game_id", "team_id"])["minutes_pred"].sum()
        dev_infer = (infer_sums - 240.0).abs()
        max_dev_infer = dev_infer.max()
        logger.info(f"Infer predictions max sum deviation from 240: {max_dev_infer:.6f}")

        infer_preds_disk = pd.read_parquet(infer_preds_path)
        infer_sums_disk = infer_preds_disk.groupby(["game_id", "team_id"])["minutes_pred"].sum()
        disk_max_dev_infer = (infer_sums_disk - 240.0).abs().max()
        logger.info(f"Infer predictions (from parquet) max sum deviation from 240: {disk_max_dev_infer:.6f}")
        _log_val_prediction_metrics(infer_preds_disk, top_k=args.top_k, name="infer_predictions", pred_col="minutes_pred")
        if "minutes_pred_hybrid" in infer_preds_disk.columns:
            hybrid_sums_infer = infer_preds_disk.groupby(["game_id", "team_id"])["minutes_pred_hybrid"].sum()
            hybrid_dev_infer = (hybrid_sums_infer - 240.0).abs()
            logger.info(
                "Infer predictions (hybrid) max sum deviation from 240: "
                f"{float(hybrid_dev_infer.max() if len(hybrid_dev_infer) else 0.0):.6f}"
            )
            _log_val_prediction_metrics(
                infer_preds_disk,
                top_k=args.top_k,
                name="infer_predictions_hybrid",
                pred_col="minutes_pred_hybrid",
            )

    logger.info("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
