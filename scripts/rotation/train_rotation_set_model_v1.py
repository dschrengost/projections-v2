#!/usr/bin/env python3
"""Train a permutation-invariant TEAM-SET minutes model on rotation_train_v1.

This trains on team-games: each example is a (game_id, team_id) with a variable-size set of players.

Model variants:
  - deepsets: φ(player) -> mean pool -> ρ([φ_i, pool]) -> per-player logits
  - settransformer: attention blocks over player embeddings -> per-player logits

Output minutes are constrained to sum to 240 per team-game:
  logits -> softplus -> normalize to 240 (mask-aware)

OT / label totals handling:
  - Labels are rescaled per team-game to sum exactly to 240 minutes.

Example:
  uv run python scripts/rotation/train_rotation_set_model_v1.py \
    --dataset-dir /home/daniel/projections-data/training/datasets/rotation_train_v1_20260103 \
    --model deepsets --epochs 2 --batch-size 32 --lr 1e-3 --device cpu
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# Add project root to path for script execution from anywhere.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from projections.rotation.set_model import (  # noqa: E402
    RotationSetModelConfig,
    build_model,
    zfill_game_id_series,
)


TEAM_TOTAL_MINUTES = 240.0
REGULATION_TOLERANCE = (238.0, 242.0)
TEAM_TOTAL_MINUTES_COL = "team_total_minutes_from_stints"
TEAM_ID_COLS = {"home_team_id", "away_team_id", "opponent_team_id"}
TEAM_EMBED_TEAM_IDX_COL = "team_id_idx"
TEAM_EMBED_OPP_IDX_COL = "opp_id_idx"
DEFAULT_MIN_TEAM_MINUTES_FROM_STINTS = 200.0
DEFAULT_MAX_TEAM_MINUTES_GAP = 2.0


def _utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _git_sha() -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT)  # noqa: S603,S607
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


def _infer_label_column(labels_df: pd.DataFrame) -> str:
    candidates = [c for c in labels_df.columns if c.lower() in {"minutes", "min", "target_minutes"}]
    for c in candidates:
        if c in labels_df.columns:
            return c
    numeric = [
        c
        for c in labels_df.columns
        if c not in {"game_id", "team_id", "player_id"}
        and pd.api.types.is_numeric_dtype(labels_df[c])
        and "minute" in c.lower()
    ]
    if len(numeric) == 1:
        return numeric[0]
    raise ValueError(f"Could not infer label column; candidates={candidates}, numeric_minute_like={numeric}")


# Features to exclude from training - these only count games where player got minutes (>0),
# ignoring DNPs. This is misleading for rotation prediction since we're trying to predict
# who plays vs DNPs. Use minutes_from_stints_prior_* instead which correctly includes DNPs as 0.
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

# Injury status flags that conflate "available to play" with "will get minutes".
# is_prob/is_q indicate injury report status, NOT rotation likelihood.
# The model learns these as strong positive signals for minutes because in training data,
# players with PROB/Q status who appear in the dataset typically ARE rotation players.
# But at inference, deep bench players can have PROB status and should still get ~0 minutes.
# The priors (minutes_from_stints_prior_*) already capture who actually plays.
EXCLUDE_INJURY_STATUS_FEATURES = {
    "is_prob",
    "is_q",
}

# Same-game rotation features that are ONLY available post-game (from gamerotation data).
# These are labels/targets, not features - including them is data leakage.
# At inference time, we don't have this data; we only have priors (_prior_5, _prior_10, _prior_20).
EXCLUDE_SAME_GAME_ROTATION_FEATURES = {
    # Team-level same-game shape metrics
    "depth_6",
    "depth_10",
    "depth_14",
    "effective_n",
    "bench_conc_top1",
    "bench_conc_top2",
    "starter_pool_minutes",
    "bench_pool_minutes",
    "team_total_minutes_from_stints",
    # Player-level same-game stint data
    "num_stints",
    "first_in_time_real",
    "last_out_time_real",
    "max_stint_len_real",
    "minutes_from_stints",
    "started_proxy",
    # Missing flags for same-game data
    "rotation_team_missing",
    "rotation_missing",
    "rotation_player_row_missing_raw",
    "rotation_player_filled_zero",
}

# These features are currently unstable/misaligned between the historical minutes
# training dataset and live minutes feature builder (large distribution shift).
# Including them has been observed to produce overly-flat live allocations.
EXCLUDE_UNSTABLE_FEATURES = {
    "vac_min_szn",
    "vac_min_guard_szn",
    "vac_min_wing_szn",
    "vac_min_big_szn",
}


def _infer_feature_columns(features_df: pd.DataFrame, *, labels_df: pd.DataFrame, label_col: str) -> list[str]:
    cols: list[str] = []
    excluded = {"game_id", "team_id", "player_id", "game_id_norm", label_col}
    excluded.update(TEAM_ID_COLS)
    excluded.update({TEAM_EMBED_TEAM_IDX_COL, TEAM_EMBED_OPP_IDX_COL})
    excluded.update(set(labels_df.columns))
    excluded.update(EXCLUDE_DNP_BLIND_FEATURES)
    excluded.update(EXCLUDE_INJURY_STATUS_FEATURES)
    excluded.update(EXCLUDE_SAME_GAME_ROTATION_FEATURES)
    excluded.update(EXCLUDE_UNSTABLE_FEATURES)
    for col in features_df.columns:
        if col in excluded:
            continue
        if pd.api.types.is_numeric_dtype(features_df[col]) or pd.api.types.is_bool_dtype(features_df[col]):
            cols.append(col)
    if not cols:
        raise ValueError("No numeric feature columns found in features dataframe")
    return cols


def _coerce_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["game_id_norm"] = zfill_game_id_series(out["game_id"])
    out["team_id"] = pd.to_numeric(out["team_id"], errors="coerce").astype("Int64")
    out["player_id"] = pd.to_numeric(out["player_id"], errors="coerce").astype("Int64")
    if out["game_id_norm"].isna().any() or out["team_id"].isna().any() or out["player_id"].isna().any():
        raise ValueError("Invalid keys found after coercion")
    return out


def _build_team_id_vocab(df: pd.DataFrame) -> list[int]:
    ids: list[pd.Series] = [pd.to_numeric(df["team_id"], errors="coerce").astype("Int64")]
    for col in ["opponent_team_id", "home_team_id", "away_team_id"]:
        if col in df.columns:
            ids.append(pd.to_numeric(df[col], errors="coerce").astype("Int64"))
    all_ids = pd.concat(ids, ignore_index=True).dropna().astype("int64")
    vocab = sorted({int(v) for v in all_ids.to_numpy().tolist()})
    if not vocab:
        raise ValueError("Empty team_id vocab; check dataset team id columns")
    return vocab


def _add_team_embedding_indices(df: pd.DataFrame, *, team_id_vocab: list[int]) -> pd.DataFrame:
    mapping = {int(team_id): i + 1 for i, team_id in enumerate(team_id_vocab)}
    out = df.copy()
    out[TEAM_EMBED_TEAM_IDX_COL] = (
        pd.to_numeric(out["team_id"], errors="coerce").astype("Int64").map(mapping).fillna(0).astype("int64")
    )
    if "opponent_team_id" in out.columns:
        out[TEAM_EMBED_OPP_IDX_COL] = (
            pd.to_numeric(out["opponent_team_id"], errors="coerce")
            .astype("Int64")
            .map(mapping)
            .fillna(0)
            .astype("int64")
        )
    else:
        out[TEAM_EMBED_OPP_IDX_COL] = 0
    return out


def _rescale_labels_to_240(df: pd.DataFrame, *, label_col: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = df.copy()
    label_sums = out.groupby(["game_id_norm", "team_id"], sort=False)[label_col].sum(min_count=1).rename(
        "label_minutes_sum_raw"
    )
    out = out.merge(label_sums.reset_index(), on=["game_id_norm", "team_id"], how="left")
    if out["label_minutes_sum_raw"].isna().any():
        raise ValueError("Found team-games with missing label totals after grouping")
    bad = out["label_minutes_sum_raw"] <= 0
    if bad.any():
        out = out.loc[~bad].copy()
    scale = TEAM_TOTAL_MINUTES / out["label_minutes_sum_raw"].astype("float64")
    out[label_col] = out[label_col].astype("float64") * scale

    # OT diagnostics (regulation tolerance only for reporting).
    tol_lo, tol_hi = REGULATION_TOLERANCE
    team_minutes_source = f"sum({label_col})"
    unique_team_minutes = label_sums.dropna()
    if TEAM_TOTAL_MINUTES_COL in out.columns:
        unique_team_minutes = (
            pd.to_numeric(out[TEAM_TOTAL_MINUTES_COL], errors="coerce")
            .groupby([out["game_id_norm"], out["team_id"]], sort=False)
            .mean()
            .dropna()
        )
        team_minutes_source = TEAM_TOTAL_MINUTES_COL
    ot_rate = (
        float(((unique_team_minutes < tol_lo) | (unique_team_minutes > tol_hi)).mean())
        if len(unique_team_minutes)
        else float("nan")
    )
    raw_sum_stats = {
        "min": float(label_sums.min()) if len(label_sums) else None,
        "p50": float(label_sums.median()) if len(label_sums) else None,
        "mean": float(label_sums.mean()) if len(label_sums) else None,
        "max": float(label_sums.max()) if len(label_sums) else None,
    }
    payload = {
        "label_rescale": {"type": "per_team_game", "target_sum": TEAM_TOTAL_MINUTES},
        "regulation_tolerance": [tol_lo, tol_hi],
        "team_minutes_source_for_ot_rate": team_minutes_source,
        "team_games_total": int(len(unique_team_minutes)),
        "team_games_ot_rate": ot_rate,
        "label_minutes_sum_raw": raw_sum_stats,
    }
    return out, payload


def _split_by_game_date(df: pd.DataFrame, *, val_frac: float) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if not (0.0 < val_frac < 1.0):
        raise ValueError("val_frac must be in (0, 1)")

    if "game_date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["game_date"]):
        game_dates = df.loc[:, ["game_id_norm", "game_date"]].drop_duplicates()
        game_dates = game_dates.dropna(subset=["game_date"]).sort_values(["game_date", "game_id_norm"])
        sort_key = "game_date"
    else:
        game_dates = df.loc[:, ["game_id_norm"]].drop_duplicates().sort_values(["game_id_norm"])
        game_dates["game_date"] = pd.NaT
        sort_key = "game_id_norm"

    n_games = len(game_dates)
    n_val = max(1, int(round(n_games * val_frac)))
    val_games = set(game_dates.tail(n_val)["game_id_norm"].astype(str))

    train_df = df.loc[~df["game_id_norm"].astype(str).isin(val_games)].copy()
    val_df = df.loc[df["game_id_norm"].astype(str).isin(val_games)].copy()

    meta: dict[str, Any] = {
        "split": {"type": "time", "sort_key": sort_key, "val_frac": float(val_frac)},
        "counts": {"games_total": int(n_games), "games_val": int(n_val)},
    }
    if "game_date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["game_date"]):
        meta["date_ranges"] = {
            "train_min": train_df["game_date"].min().date().isoformat() if len(train_df) else None,
            "train_max": train_df["game_date"].max().date().isoformat() if len(train_df) else None,
            "val_min": val_df["game_date"].min().date().isoformat() if len(val_df) else None,
            "val_max": val_df["game_date"].max().date().isoformat() if len(val_df) else None,
        }
    return train_df, val_df, meta


def _numeric_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    frame = df.loc[:, cols].copy()
    for col in cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame.fillna(0.0)


def _filter_invalid_team_games_for_training(
    df: pd.DataFrame,
    *,
    label_col: str,
    min_team_minutes_from_stints: float,
    max_team_minutes_gap: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if TEAM_TOTAL_MINUTES_COL not in df.columns:
        raise ValueError(f"Missing required column for validation: {TEAM_TOTAL_MINUTES_COL}")
    if "minutes_from_stints" not in df.columns:
        raise ValueError("Missing required column for validation: minutes_from_stints")

    group_cols = ["game_id_norm", "team_id"]
    n_players = df.groupby(group_cols, sort=False).size().rename("n_players")
    team_total = (
        pd.to_numeric(df[TEAM_TOTAL_MINUTES_COL], errors="coerce")
        .groupby([df["game_id_norm"], df["team_id"]], sort=False)
        .mean()
        .rename("stints_team_total")
    )
    minutes_from_stints_sum = (
        pd.to_numeric(df["minutes_from_stints"], errors="coerce")
        .fillna(0.0)
        .groupby([df["game_id_norm"], df["team_id"]], sort=False)
        .sum()
        .rename("minutes_from_stints_sum")
    )
    label_sum = (
        pd.to_numeric(df[label_col], errors="coerce")
        .fillna(0.0)
        .groupby([df["game_id_norm"], df["team_id"]], sort=False)
        .sum()
        .rename("label_sum")
    )
    summary = pd.concat([n_players, team_total, minutes_from_stints_sum, label_sum], axis=1).reset_index()
    summary["coverage_gap"] = (summary["stints_team_total"] - summary["minutes_from_stints_sum"]).abs()
    summary["label_gap"] = (summary["stints_team_total"] - summary["label_sum"]).abs()

    stints_ok = summary["stints_team_total"].notna() & (summary["stints_team_total"] >= float(min_team_minutes_from_stints))
    coverage_ok = summary["coverage_gap"].notna() & (summary["coverage_gap"] <= float(max_team_minutes_gap))
    labels_ok = summary["label_gap"].notna() & (summary["label_gap"] <= float(max_team_minutes_gap))
    keep = stints_ok & coverage_ok & labels_ok

    reason_stints_low = ~stints_ok
    reason_missing_coverage = stints_ok & ~coverage_ok
    reason_labels_incomplete = stints_ok & coverage_ok & ~labels_ok

    kept_pairs = set(zip(summary.loc[keep, "game_id_norm"].astype(str), summary.loc[keep, "team_id"].astype(int)))
    pair_series = list(zip(df["game_id_norm"].astype(str), df["team_id"].astype(int)))
    keep_mask = pd.Series([p in kept_pairs for p in pair_series], index=df.index)
    filtered = df.loc[keep_mask].copy()
    filtered[label_col] = pd.to_numeric(filtered[label_col], errors="coerce").fillna(0.0).astype("float64")

    worst_rows = (
        summary.loc[~keep]
        .sort_values(["label_sum", "n_players"], ascending=[True, True])
        .head(20)
        .loc[:, ["game_id_norm", "team_id", "n_players", "label_sum", "stints_team_total", "minutes_from_stints_sum", "coverage_gap", "label_gap"]]
    )
    worst_examples: list[dict[str, Any]] = []
    for row in worst_rows.to_dict(orient="records"):
        worst_examples.append(
            {
                "game_id_norm": str(row["game_id_norm"]),
                "team_id": int(row["team_id"]),
                "n_players": int(row["n_players"]),
                "label_sum": float(row["label_sum"]) if row["label_sum"] is not None else None,
                "stints_team_total": float(row["stints_team_total"]) if row["stints_team_total"] is not None else None,
                "minutes_from_stints_sum": float(row["minutes_from_stints_sum"]) if row["minutes_from_stints_sum"] is not None else None,
                "coverage_gap": float(row["coverage_gap"]) if row["coverage_gap"] is not None else None,
                "label_gap": float(row["label_gap"]) if row["label_gap"] is not None else None,
            }
        )

    def _stats(series: pd.Series) -> dict[str, float | None]:
        clean = pd.to_numeric(series, errors="coerce").dropna()
        if clean.empty:
            return {"min": None, "p10": None, "p50": None, "mean": None, "p90": None, "max": None}
        return {
            "min": float(clean.min()),
            "p10": float(clean.quantile(0.1)),
            "p50": float(clean.quantile(0.5)),
            "mean": float(clean.mean()),
            "p90": float(clean.quantile(0.9)),
            "max": float(clean.max()),
        }

    meta: dict[str, Any] = {
        "thresholds": {
            "min_team_minutes_from_stints": float(min_team_minutes_from_stints),
            "max_team_minutes_gap": float(max_team_minutes_gap),
        },
        "team_games_total": int(len(summary)),
        "team_games_kept": int(keep.sum()),
        "team_games_dropped_by_reason": {
            "stints_team_total_too_low": int(reason_stints_low.sum()),
            "missing_player_coverage": int(reason_missing_coverage.sum()),
            "labels_incomplete_vs_stints": int(reason_labels_incomplete.sum()),
        },
        "summaries": {
            "n_players": _stats(summary["n_players"]),
            "stints_team_total": _stats(summary["stints_team_total"]),
            "minutes_from_stints_sum": _stats(summary["minutes_from_stints_sum"]),
            "label_sum": _stats(summary["label_sum"]),
        },
        "summaries_kept": {
            "n_players": _stats(summary.loc[keep, "n_players"]),
            "stints_team_total": _stats(summary.loc[keep, "stints_team_total"]),
            "minutes_from_stints_sum": _stats(summary.loc[keep, "minutes_from_stints_sum"]),
            "label_sum": _stats(summary.loc[keep, "label_sum"]),
        },
        "worst_examples": worst_examples,
    }
    return filtered, meta


@dataclass(frozen=True)
class TeamGameExample:
    x: np.ndarray
    team_idx: np.ndarray
    opp_idx: np.ndarray
    alloc_mask: np.ndarray
    prior_w: np.ndarray
    vacated_minutes_prior_20_total: float
    team_out_count: int
    starters_out_proxy: int
    y: np.ndarray


class TeamGameDataset(Dataset[TeamGameExample]):
    def __init__(self, examples: list[TeamGameExample]) -> None:
        if not examples:
            raise ValueError("TeamGameDataset requires at least one example")
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
]:
    max_n = max(ex.x.shape[0] for ex in batch)
    num_features = batch[0].x.shape[1]
    x = torch.zeros((len(batch), max_n, num_features), dtype=torch.float32)
    team_idx = torch.zeros((len(batch), max_n), dtype=torch.long)
    opp_idx = torch.zeros((len(batch), max_n), dtype=torch.long)
    alloc_mask = torch.zeros((len(batch), max_n), dtype=torch.bool)
    prior_w = torch.zeros((len(batch), max_n), dtype=torch.float32)
    vacated_minutes = torch.zeros((len(batch),), dtype=torch.float32)
    router_features = torch.zeros((len(batch), 3), dtype=torch.float32)
    y = torch.zeros((len(batch), max_n), dtype=torch.float32)
    mask = torch.zeros((len(batch), max_n), dtype=torch.bool)
    for i, ex in enumerate(batch):
        n = ex.x.shape[0]
        x[i, :n] = torch.from_numpy(ex.x)
        team_idx[i, :n] = torch.from_numpy(ex.team_idx)
        opp_idx[i, :n] = torch.from_numpy(ex.opp_idx)
        alloc_mask[i, :n] = torch.from_numpy(ex.alloc_mask)
        prior_w[i, :n] = torch.from_numpy(ex.prior_w)
        vacated_minutes[i] = float(ex.vacated_minutes_prior_20_total)
        router_features[i, 0] = float(np.clip(float(ex.vacated_minutes_prior_20_total) / 60.0, 0.0, 1.0))
        router_features[i, 1] = float(np.clip(float(ex.team_out_count) / 10.0, 0.0, 1.0))
        router_features[i, 2] = float(np.clip(float(ex.starters_out_proxy) / 5.0, 0.0, 1.0))
        y[i, :n] = torch.from_numpy(ex.y)
        mask[i, :n] = True
    return x, team_idx, opp_idx, alloc_mask, prior_w, vacated_minutes, router_features, y, mask


def _build_team_game_examples(
    df: pd.DataFrame,
    *,
    feature_cols: list[str],
    team_idx_col: str,
    opp_idx_col: str,
    label_col: str,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    prior_weight_col: str | None = None,
) -> list[TeamGameExample]:
    feats = _numeric_frame(df, feature_cols).to_numpy(dtype="float32", copy=False)
    feats = (feats - feature_mean) / feature_std
    team_idx_all = pd.to_numeric(df[team_idx_col], errors="coerce").fillna(0).to_numpy(dtype=np.int64, copy=False)
    opp_idx_all = pd.to_numeric(df[opp_idx_col], errors="coerce").fillna(0).to_numpy(dtype=np.int64, copy=False)
    labels = pd.to_numeric(df[label_col], errors="coerce").fillna(0.0).to_numpy(dtype="float32", copy=False)
    is_out = pd.to_numeric(df.get("is_out", pd.Series(0, index=df.index)), errors="coerce").fillna(0).astype(int)
    alloc_mask_all = (is_out == 0).to_numpy(dtype=bool, copy=False)

    if prior_weight_col and prior_weight_col in df.columns:
        prior_w_all = (
            pd.to_numeric(df[prior_weight_col], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype="float32")
        )
    else:
        prior_w_all = np.zeros(len(df), dtype="float32")

    # Injury regime weighting: vacated minutes based on OUT players' prior minutes.
    if "minutes_from_stints_prior_20" in df.columns:
        prior20_all = (
            pd.to_numeric(df["minutes_from_stints_prior_20"], errors="coerce")
            .fillna(0.0)
            .clip(lower=0.0)
            .to_numpy(dtype="float32", copy=False)
        )
    else:
        prior20_all = np.zeros(len(df), dtype="float32")

    examples: list[TeamGameExample] = []
    for _, idx in df.groupby(["game_id_norm", "team_id"], sort=False).indices.items():
        idx_arr = np.asarray(idx, dtype=np.int64)
        x = feats[idx_arr]
        team_idx = team_idx_all[idx_arr]
        opp_idx = opp_idx_all[idx_arr]
        alloc_mask = alloc_mask_all[idx_arr]
        prior_w = prior_w_all[idx_arr]
        out_mask = ~alloc_mask
        vacated_minutes = float(np.sum(prior20_all[idx_arr][out_mask].astype("float64")))
        team_out_count = int(np.sum(out_mask.astype(np.int64)))
        prior20_group = prior20_all[idx_arr].astype("float64", copy=False)
        k = int(min(5, prior20_group.shape[0]))
        if k > 0:
            top_idx = np.argsort(prior20_group)[::-1][:k]
            starters_out_proxy = int(np.sum(out_mask[top_idx].astype(np.int64)))
        else:
            starters_out_proxy = 0
        y = labels[idx_arr]
        examples.append(
            TeamGameExample(
                x=x,
                team_idx=team_idx,
                opp_idx=opp_idx,
                alloc_mask=alloc_mask,
                prior_w=prior_w,
                vacated_minutes_prior_20_total=vacated_minutes,
                team_out_count=team_out_count,
                starters_out_proxy=starters_out_proxy,
                y=y,
            )
        )
    return examples


def _masked_mae(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    err = (pred - y).abs() * mask.to(dtype=pred.dtype)
    denom = mask.sum().clamp(min=1)
    return err.sum() / denom


def _focal_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    gamma: float = 2.0,
    alpha: float = 0.25,
) -> torch.Tensor:
    """Per-element focal BCE (no reduction)."""

    # Standard BCE term.
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    prob = torch.sigmoid(logits)
    pt = torch.where(targets >= 0.5, prob, 1.0 - prob)
    focal = (1.0 - pt).clamp(min=0.0) ** float(gamma)
    alpha_t = torch.where(targets >= 0.5, float(alpha), 1.0 - float(alpha))
    return alpha_t * focal * bce


def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    use_gate_head: bool,
    lambda_rot: float,
    lambda_fp: float,
    inj_weight_scale: float,
    rot_target_mode: str,
    rot_target_threshold: float,
    rot_target_slope: float,
    fp_minutes_threshold: float,
    rot_loss_type: str,
    focal_gamma: float,
    focal_alpha: float,
) -> dict[str, Any]:
    model.eval()
    total_abs = 0.0
    total_count = 0.0
    team_maes: list[float] = []
    rot_loss_sum = 0.0
    rot_loss_denom = 0.0
    fp_loss_sum = 0.0
    fp_loss_denom = 0.0
    tail_sum = 0.0
    rot_size8_sum = 0.0
    n_teams = 0.0
    # Gate calibration bins: [0,0.2), [0.2,0.4), ..., [0.8,1.0]
    gate_bins = torch.tensor([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], device=device)
    gate_bin_pos = torch.zeros((len(gate_bins) - 1,), dtype=torch.float64)
    gate_bin_cnt = torch.zeros((len(gate_bins) - 1,), dtype=torch.float64)
    router_pi_soft_sum: np.ndarray | None = None
    router_entropy_soft_sum = 0.0
    router_eff_soft_sum = 0.0
    router_max_pi_soft_gt_08 = 0.0
    router_rows_soft = 0.0
    router_pi_soft_sum_inj: np.ndarray | None = None
    router_entropy_soft_sum_inj = 0.0
    router_eff_soft_sum_inj = 0.0
    router_max_pi_soft_gt_08_inj = 0.0
    router_rows_soft_inj = 0.0

    router_pi_used_sum: np.ndarray | None = None
    router_assign_counts: np.ndarray | None = None
    router_max_pi_used_gt_08 = 0.0
    router_rows_used = 0.0
    router_pi_used_sum_inj: np.ndarray | None = None
    router_assign_counts_inj: np.ndarray | None = None
    router_max_pi_used_gt_08_inj = 0.0
    router_rows_used_inj = 0.0
    with torch.no_grad():
        for x, team_idx, opp_idx, alloc_mask, prior_w, vacated_minutes, router_features_batch, y, mask in loader:
            x = x.to(device)
            team_idx = team_idx.to(device)
            opp_idx = opp_idx.to(device)
            alloc_mask = alloc_mask.to(device)
            prior_w = prior_w.to(device)
            vacated_minutes = vacated_minutes.to(device)
            router_features_batch = router_features_batch.to(device)
            y = y.to(device)
            mask = mask.to(device)
            if use_gate_head:
                if not hasattr(model, "forward_with_aux"):
                    raise RuntimeError("use_gate_head=True requires model.forward_with_aux")
                router_feature_dim = int(getattr(model, "router_feature_dim", 0))
                router_features = None
                if router_feature_dim > 0:
                    base_feat = router_features_batch
                    if base_feat.shape[1] >= router_feature_dim:
                        router_features = base_feat[:, :router_feature_dim]
                    else:
                        router_features = torch.zeros(
                            (base_feat.shape[0], router_feature_dim), device=device, dtype=base_feat.dtype
                        )
                        router_features[:, : base_feat.shape[1]] = base_feat

                pred, gate_logits, _, router_pi_soft, router_pi_used = model.forward_with_aux(  # type: ignore[attr-defined]
                    x,
                    team_idx,
                    opp_idx,
                    mask,
                    alloc_mask=alloc_mask,
                    prior_weights=prior_w,
                    router_features=router_features,
                )
                if gate_logits is None:
                    raise RuntimeError("use_gate_head=True but model returned gate_logits=None")
            else:
                pred = model(x, team_idx, opp_idx, mask, alloc_mask=alloc_mask, prior_weights=prior_w)
                gate_logits = None
                router_pi_soft = None
                router_pi_used = None

            abs_err = (pred - y).abs() * mask.to(dtype=pred.dtype)
            total_abs += float(abs_err.sum().item())
            total_count += float(mask.sum().item())
            per_team = abs_err.sum(dim=1) / mask.sum(dim=1).clamp(min=1).to(dtype=abs_err.dtype)
            team_maes.extend(per_team.cpu().numpy().tolist())

            # Team-level diagnostics (computed on alloc_mask, which excludes OUT players).
            final_mask = alloc_mask & mask
            pred_team = pred * final_mask.to(dtype=pred.dtype)
            # tail minutes relative to predicted top-9
            top9_sum = torch.topk(pred_team, k=min(9, pred_team.shape[1]), dim=1).values.sum(dim=1)
            tail_sum += float((240.0 - top9_sum).sum().item())
            rot_size8_sum += float((pred_team >= 8.0).sum(dim=1).to(dtype=torch.float32).sum().item())
            n_teams += float(pred_team.shape[0])

            if (lambda_rot > 0 or lambda_fp > 0) and use_gate_head:
                w_inj = 1.0 + float(inj_weight_scale) * torch.clamp(vacated_minutes / 60.0, min=0.0, max=1.0)
                w_inj = w_inj.view(-1, 1)

                y_rot_hard = (y >= float(rot_target_threshold)).to(dtype=pred.dtype)
                if rot_target_mode == "soft":
                    y_rot = torch.sigmoid((y - float(rot_target_threshold)) / float(rot_target_slope)).to(dtype=pred.dtype)
                else:
                    y_rot = y_rot_hard

                rot_mask_f = final_mask.to(dtype=pred.dtype)
                if lambda_rot > 0:
                    if rot_loss_type == "focal":
                        per_el = _focal_bce_with_logits(
                            gate_logits,  # type: ignore[arg-type]
                            y_rot,
                            gamma=float(focal_gamma),
                            alpha=float(focal_alpha),
                        )
                    else:
                        per_el = torch.nn.functional.binary_cross_entropy_with_logits(
                            gate_logits,  # type: ignore[arg-type]
                            y_rot,
                            reduction="none",
                        )
                    weights = rot_mask_f * w_inj
                    rot_loss_sum += float((per_el * weights).sum().item())
                    rot_loss_denom += float(weights.sum().item())

                    # Gate calibration bins.
                    gate_prob = torch.sigmoid(gate_logits)  # type: ignore[arg-type]
                    gate_prob = gate_prob[final_mask]
                    y_true = y_rot_hard[final_mask].to(dtype=torch.float32)
                    if gate_prob.numel() > 0:
                        idx = torch.bucketize(gate_prob.clamp(0.0, 1.0), gate_bins, right=False) - 1
                        idx = idx.clamp(min=0, max=len(gate_bin_cnt) - 1)
                        for b in range(len(gate_bin_cnt)):
                            sel = idx == b
                            if bool(sel.any()):
                                gate_bin_pos[b] += float(y_true[sel].sum().item())
                                gate_bin_cnt[b] += float(sel.sum().item())

                if lambda_fp > 0:
                    excess = torch.relu(pred - float(fp_minutes_threshold)) ** 2
                    fp = excess * (1.0 - y_rot_hard)
                    weights = rot_mask_f * w_inj
                    fp_loss_sum += float((fp * weights).sum().item())
                    fp_loss_denom += float(weights.sum().item())

            if router_pi_soft is not None:
                pi = router_pi_soft.detach().cpu().numpy().astype("float64", copy=False)
                if router_pi_soft_sum is None:
                    router_pi_soft_sum = np.zeros((pi.shape[1],), dtype="float64")
                router_pi_soft_sum += pi.sum(axis=0)
                router_rows_soft += float(pi.shape[0])
                entropy = -(pi * np.log(np.clip(pi, 1e-12, 1.0))).sum(axis=1)
                router_entropy_soft_sum += float(entropy.sum())
                router_eff_soft_sum += float(np.exp(entropy).sum())
                router_max_pi_soft_gt_08 += float((pi.max(axis=1) > 0.8).sum())
                if router_features_batch.shape[1] >= 3:
                    inj_sel = (router_features_batch[:, 1] >= 0.2) & (router_features_batch[:, 2] >= 0.2)
                    inj_sel_np = inj_sel.detach().cpu().numpy().astype(bool)
                    if bool(np.any(inj_sel_np)):
                        pi_inj = pi[inj_sel_np]
                        if router_pi_soft_sum_inj is None:
                            router_pi_soft_sum_inj = np.zeros((pi_inj.shape[1],), dtype="float64")
                        router_pi_soft_sum_inj += pi_inj.sum(axis=0)
                        router_rows_soft_inj += float(pi_inj.shape[0])
                        entropy_inj = -(pi_inj * np.log(np.clip(pi_inj, 1e-12, 1.0))).sum(axis=1)
                        router_entropy_soft_sum_inj += float(entropy_inj.sum())
                        router_eff_soft_sum_inj += float(np.exp(entropy_inj).sum())
                        router_max_pi_soft_gt_08_inj += float((pi_inj.max(axis=1) > 0.8).sum())

            if router_pi_used is not None:
                pi_used = router_pi_used.detach().cpu().numpy().astype("float64", copy=False)
                if router_pi_used_sum is None:
                    router_pi_used_sum = np.zeros((pi_used.shape[1],), dtype="float64")
                router_pi_used_sum += pi_used.sum(axis=0)
                assign = np.argmax(pi_used, axis=1)
                if router_assign_counts is None:
                    router_assign_counts = np.zeros((pi_used.shape[1],), dtype="int64")
                router_assign_counts += np.bincount(assign, minlength=pi_used.shape[1])
                router_rows_used += float(pi_used.shape[0])
                router_max_pi_used_gt_08 += float((pi_used.max(axis=1) > 0.8).sum())
                if router_features_batch.shape[1] >= 3:
                    inj_sel = (router_features_batch[:, 1] >= 0.2) & (router_features_batch[:, 2] >= 0.2)
                    inj_sel_np = inj_sel.detach().cpu().numpy().astype(bool)
                    if bool(np.any(inj_sel_np)):
                        pi_inj = pi_used[inj_sel_np]
                        if router_pi_used_sum_inj is None:
                            router_pi_used_sum_inj = np.zeros((pi_inj.shape[1],), dtype="float64")
                        router_pi_used_sum_inj += pi_inj.sum(axis=0)
                        assign_inj = np.argmax(pi_inj, axis=1)
                        if router_assign_counts_inj is None:
                            router_assign_counts_inj = np.zeros((pi_inj.shape[1],), dtype="int64")
                        router_assign_counts_inj += np.bincount(assign_inj, minlength=pi_inj.shape[1])
                        router_rows_used_inj += float(pi_inj.shape[0])
                        router_max_pi_used_gt_08_inj += float((pi_inj.max(axis=1) > 0.8).sum())

    mae = total_abs / max(total_count, 1.0)
    team_mae = float(np.mean(team_maes)) if team_maes else float("nan")
    out: dict[str, Any] = {
        "mae": float(mae),
        "team_mae": float(team_mae),
        "tail_mean_team240_top9": float(tail_sum / max(n_teams, 1.0)),
        "rotation_size8_mean": float(rot_size8_sum / max(n_teams, 1.0)),
    }
    if use_gate_head and (lambda_rot > 0):
        out["rot_loss"] = float(rot_loss_sum / max(rot_loss_denom, 1.0))
        out["gate_calibration"] = {
            f"[{float(gate_bins[i]):.1f},{float(gate_bins[i+1]):.1f})": (
                float(gate_bin_pos[i] / gate_bin_cnt[i]) if gate_bin_cnt[i] > 0 else None
            )
            for i in range(len(gate_bin_cnt))
        }
    if use_gate_head and (lambda_fp > 0):
        out["fp_loss"] = float(fp_loss_sum / max(fp_loss_denom, 1.0))
    if router_pi_soft_sum is not None and router_rows_soft > 0:
        out["router_pi_mean"] = (router_pi_soft_sum / router_rows_soft).tolist()
        out["router_entropy_mean"] = float(router_entropy_soft_sum / router_rows_soft)
        out["router_effective_num_experts"] = float(router_eff_soft_sum / router_rows_soft)
        out["router_frac_max_pi_gt_0.8"] = float(router_max_pi_soft_gt_08 / router_rows_soft)
    if router_pi_soft_sum_inj is not None and router_rows_soft_inj > 0:
        out["router_pi_mean_injury_slice"] = (router_pi_soft_sum_inj / router_rows_soft_inj).tolist()
        out["router_entropy_mean_injury_slice"] = float(router_entropy_soft_sum_inj / router_rows_soft_inj)
        out["router_effective_num_experts_injury_slice"] = float(router_eff_soft_sum_inj / router_rows_soft_inj)
        out["router_frac_max_pi_gt_0.8_injury_slice"] = float(router_max_pi_soft_gt_08_inj / router_rows_soft_inj)

    if router_pi_used_sum is not None and router_rows_used > 0:
        out["router_pi_used_mean"] = (router_pi_used_sum / router_rows_used).tolist()
        if router_assign_counts is not None:
            assign_frac = router_assign_counts.astype("float64") / float(router_rows_used)
            out["router_assign_frac"] = assign_frac.tolist()
            p = np.clip(assign_frac, 1e-12, 1.0)
            out["router_assign_effective_num_experts"] = float(np.exp(-(p * np.log(p)).sum()))
        out["router_frac_max_pi_used_gt_0.8"] = float(router_max_pi_used_gt_08 / router_rows_used)
    if router_pi_used_sum_inj is not None and router_rows_used_inj > 0:
        out["router_pi_used_mean_injury_slice"] = (router_pi_used_sum_inj / router_rows_used_inj).tolist()
        if router_assign_counts_inj is not None:
            assign_frac = router_assign_counts_inj.astype("float64") / float(router_rows_used_inj)
            out["router_assign_frac_injury_slice"] = assign_frac.tolist()
            p = np.clip(assign_frac, 1e-12, 1.0)
            out["router_assign_effective_num_experts_injury_slice"] = float(np.exp(-(p * np.log(p)).sum()))
        out["router_frac_max_pi_used_gt_0.8_injury_slice"] = float(router_max_pi_used_gt_08_inj / router_rows_used_inj)
    return out


def _post_train_diagnostics(model: torch.nn.Module, loader: DataLoader, *, device: torch.device) -> dict[str, Any]:
    model.eval()
    dust_count = 0
    total_rows = 0
    top8_shares: list[float] = []
    dnp_preds: list[float] = []

    with torch.no_grad():
        for x, team_idx, opp_idx, alloc_mask, prior_w, vacated_minutes, router_features_batch, y, mask in loader:
            x = x.to(device)
            team_idx = team_idx.to(device)
            opp_idx = opp_idx.to(device)
            alloc_mask = alloc_mask.to(device)
            prior_w = prior_w.to(device)
            vacated_minutes = vacated_minutes.to(device)
            router_features_batch = router_features_batch.to(device)
            y = y.to(device)
            mask = mask.to(device)

            router_feature_dim = int(getattr(model, "router_feature_dim", 0))
            router_features = None
            if router_feature_dim > 0:
                base_feat = router_features_batch
                if base_feat.shape[1] >= router_feature_dim:
                    router_features = base_feat[:, :router_feature_dim]
                else:
                    router_features = torch.zeros((base_feat.shape[0], router_feature_dim), device=device, dtype=base_feat.dtype)
                    router_features[:, : base_feat.shape[1]] = base_feat

            pred = model(
                x,
                team_idx,
                opp_idx,
                mask,
                alloc_mask=alloc_mask,
                prior_weights=prior_w,
                router_features=router_features,
            )

            pred_np = pred.cpu().numpy()
            y_np = y.cpu().numpy()
            mask_np = mask.cpu().numpy()

            for i in range(pred_np.shape[0]):
                m = mask_np[i].astype(bool)
                if not m.any():
                    continue
                p = pred_np[i][m]
                yy = y_np[i][m]
                dust_count += int((p < 0.25).sum())
                total_rows += int(len(p))
                topk = np.sort(p)[::-1][:8]
                top8_shares.append(float(topk.sum() / TEAM_TOTAL_MINUTES))
                if (yy == 0).any():
                    dnp_preds.extend(p[yy == 0].tolist())

    dust_rate = float(dust_count / total_rows) if total_rows else float("nan")
    top8_share_mean = float(np.mean(top8_shares)) if top8_shares else float("nan")
    dnp_pred_median = float(np.median(dnp_preds)) if dnp_preds else None
    return {"dust_rate": dust_rate, "top8_share_mean": top8_share_mean, "dnp_pred_median": dnp_pred_median}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="/home/daniel/projections-data/training/datasets/rotation_train_v1_20260103",
        help="Path containing features.parquet and labels.parquet.",
    )
    parser.add_argument("--model", type=str, choices=["deepsets", "settransformer"], default="deepsets")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out-dir", type=str, default="artifacts/rotation_set_minutes")
    parser.add_argument(
        "--alloc-activation",
        type=str,
        choices=["entmax", "entmax15", "sparsemax", "softmax", "softplus"],
        default="entmax",
        help="Share distribution activation (entmax default; softplus is legacy).",
    )
    parser.add_argument(
        "--use-gate-head",
        action="store_true",
        help="Enable a two-headed allocator: gate_logits (membership) + share_logits (relative allocation).",
    )
    parser.add_argument(
        "--moe-experts",
        type=int,
        default=1,
        help="Number of experts for a Mixture-of-Experts policy head (requires --use-gate-head).",
    )
    parser.add_argument(
        "--lambda-router",
        "--lambda-router-load-balance",
        type=float,
        default=0.01,
        dest="lambda_router_load_balance",
        help="Global diversity regularizer weight for the router (load-balance; applied when --moe-experts > 1).",
    )
    parser.add_argument(
        "--lambda-router-entropy",
        type=float,
        default=0.0,
        help="Per-sample router entropy penalty weight (minimize entropy => more decisive routing).",
    )
    parser.add_argument(
        "--router-temperature",
        type=float,
        default=1.0,
        help="Softmax temperature for router logits (lower => more decisive routing).",
    )
    parser.add_argument(
        "--router-mode",
        type=str,
        choices=["soft", "top1_st", "top2_st"],
        default="soft",
        help="Router mode for MoE: soft mixing or straight-through hard routing.",
    )
    router_gumbel_group = parser.add_mutually_exclusive_group()
    router_gumbel_group.add_argument(
        "--router-gumbel",
        action="store_true",
        help="Enable Gumbel noise for hard routing during training (ignored for eval/inference).",
    )
    router_gumbel_group.add_argument(
        "--no-router-gumbel",
        action="store_false",
        dest="router_gumbel",
        help="Disable Gumbel noise for hard routing during training.",
    )
    parser.set_defaults(router_gumbel=False)
    parser.add_argument(
        "--router-gumbel-tau",
        type=float,
        default=1.0,
        help="Scale for Gumbel noise added to router logits when --router-gumbel is enabled.",
    )
    parser.add_argument(
        "--router-reg-type",
        type=str,
        choices=["load_balance", "entropy"],
        default="load_balance",
        help="Router regularizer type (applied when --moe-experts > 1).",
    )
    parser.add_argument("--entmax-alpha", type=float, default=1.5, help="Alpha for entmax (alloc-activation=entmax).")
    parser.add_argument(
        "--share-temperature",
        type=float,
        default=1.0,
        help="Temperature for share logits before simplex activation (lower => sharper).",
    )
    parser.add_argument("--lambda-rot", type=float, default=0.6, help="Membership (rotation) loss weight.")
    parser.add_argument("--lambda-fp", type=float, default=0.05, help="False-positive minutes penalty weight.")
    parser.add_argument(
        "--fp-minutes-threshold",
        type=float,
        default=6.0,
        help="m0 threshold for false-positive penalty: relu(pred_minutes - m0)^2 on non-rotation players.",
    )
    parser.add_argument(
        "--inj-weight-scale",
        type=float,
        default=1.0,
        help="Upweight rotation losses in injury regimes: 1 + scale * clip(vacated_prior20/60, 0, 1).",
    )
    parser.add_argument(
        "--rot-target-mode",
        type=str,
        choices=["hard", "soft"],
        default="hard",
        help="Rotation membership target: hard step or soft sigmoid ramp around threshold.",
    )
    parser.add_argument("--rot-target-threshold", type=float, default=8.0, help="Minutes threshold for rotation label.")
    parser.add_argument("--rot-target-slope", type=float, default=2.0, help="Slope for soft rotation target.")
    parser.add_argument(
        "--rot-loss-type",
        type=str,
        choices=["bce", "focal"],
        default="bce",
        help="Rotation membership loss type.",
    )
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="Focal loss gamma (rot-loss-type=focal).")
    parser.add_argument("--focal-alpha", type=float, default=0.25, help="Focal loss alpha (rot-loss-type=focal).")
    parser.add_argument(
        "--use-prior-head",
        action="store_true",
        help="Multiply logits by a per-row prior weight (e.g. minutes_from_stints_prior_20) before normalization.",
    )
    parser.add_argument(
        "--prior-weight-col",
        type=str,
        default="minutes_from_stints_prior_20",
        help="Column to use as the prior weight when --use-prior-head is enabled.",
    )
    parser.add_argument(
        "--prior-weight-floor",
        type=float,
        default=1.0,
        help="Non-negative floor added to prior weights when --use-prior-head is enabled.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--max-team-games", type=int, default=None, help="Optional cap for faster smoke runs.")
    parser.add_argument(
        "--min-team-minutes-from-stints",
        type=float,
        default=DEFAULT_MIN_TEAM_MINUTES_FROM_STINTS,
        help="Drop team-games with team_total_minutes_from_stints below this threshold.",
    )
    parser.add_argument(
        "--max-team-minutes-gap",
        type=float,
        default=DEFAULT_MAX_TEAM_MINUTES_GAP,
        help="Max allowed |team_total_minutes_from_stints - (label_sum or minutes_from_stints_sum)| to keep a team-game.",
    )
    args = parser.parse_args()

    _set_seed(int(args.seed))
    device = torch.device(args.device)

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    features_path = dataset_dir / "features.parquet"
    labels_path = dataset_dir / "labels.parquet"
    if not features_path.exists():
        raise FileNotFoundError(f"Missing features.parquet at {features_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels.parquet at {labels_path}")

    features_df = pd.read_parquet(features_path)
    labels_df = pd.read_parquet(labels_path)
    label_col = _infer_label_column(labels_df)
    target_col = "target_minutes"

    features_df = _coerce_keys(features_df)
    labels_df = _coerce_keys(labels_df)
    labels_keep = labels_df.loc[:, ["game_id_norm", "team_id", "player_id", label_col]].rename(
        columns={label_col: target_col}
    )
    labels_keep = labels_keep.drop_duplicates(subset=["game_id_norm", "team_id", "player_id"], keep="last")
    merged = features_df.merge(labels_keep, on=["game_id_norm", "team_id", "player_id"], how="left")

    merged, qc_meta = _filter_invalid_team_games_for_training(
        merged,
        label_col=target_col,
        min_team_minutes_from_stints=float(args.min_team_minutes_from_stints),
        max_team_minutes_gap=float(args.max_team_minutes_gap),
    )
    dropped_by_reason = qc_meta.get("team_games_dropped_by_reason", {})
    print(
        "[rotation_set_minutes] Team-game QC:",
        f"total={qc_meta['team_games_total']:,}",
        f"kept={qc_meta['team_games_kept']:,}",
        f"dropped={qc_meta['team_games_total']-qc_meta['team_games_kept']:,}",
    )
    print(f"[rotation_set_minutes] Dropped by reason: {dropped_by_reason}")
    worst = qc_meta.get("worst_examples", [])[:5]
    if worst:
        print("[rotation_set_minutes] Worst examples (first 5):")
        for row in worst:
            print(
                " ",
                row.get("game_id_norm"),
                row.get("team_id"),
                f"n_players={row.get('n_players')}",
                f"label_sum={row.get('label_sum')}",
                f"stints_total={row.get('stints_team_total')}",
                f"coverage_gap={row.get('coverage_gap')}",
                f"label_gap={row.get('label_gap')}",
            )

    merged, ot_meta = _rescale_labels_to_240(merged, label_col=target_col)
    team_id_vocab = _build_team_id_vocab(merged)
    merged = _add_team_embedding_indices(merged, team_id_vocab=team_id_vocab)

    feature_cols = _infer_feature_columns(features_df, labels_df=labels_df, label_col=target_col)
    use_prior_head = bool(args.use_prior_head)
    prior_weight_col = str(args.prior_weight_col)
    prior_weight_floor = float(args.prior_weight_floor)
    if prior_weight_floor < 0:
        raise ValueError("--prior-weight-floor must be >= 0")
    if use_prior_head and prior_weight_col not in merged.columns:
        raise ValueError(
            f"--use-prior-head requires --prior-weight-col={prior_weight_col!r} to exist in dataset features"
        )
    if use_prior_head and prior_weight_col not in feature_cols:
        feature_cols.append(prior_weight_col)
    train_df, val_df, split_meta = _split_by_game_date(merged, val_frac=float(args.val_frac))
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    # Split summary (cal split not used for this trainer; n_cal=0).
    tg_train = train_df.groupby(["game_id_norm", "team_id"], sort=False).ngroups
    tg_val = val_df.groupby(["game_id_norm", "team_id"], sort=False).ngroups
    date_ranges = split_meta.get("date_ranges", {})
    print(
        "[rotation_set_minutes] split:",
        f"rows_train={len(train_df):,}",
        f"rows_val={len(val_df):,}",
        "rows_cal=0",
        f"team_games_train={tg_train:,}",
        f"team_games_val={tg_val:,}",
        f"val_frac={split_meta.get('split', {}).get('val_frac', float('nan'))}",
        f"train_dates={date_ranges.get('train_min')}..{date_ranges.get('train_max')}",
        f"val_dates={date_ranges.get('val_min')}..{date_ranges.get('val_max')}",
    )

    if args.max_team_games is not None:
        max_tg = int(args.max_team_games)
        if max_tg <= 0:
            raise ValueError("--max-team-games must be positive")
        train_groups = list(train_df.groupby(["game_id_norm", "team_id"], sort=False).indices.items())
        val_groups = list(val_df.groupby(["game_id_norm", "team_id"], sort=False).indices.items())
        random.Random(args.seed).shuffle(train_groups)
        random.Random(args.seed).shuffle(val_groups)
        train_keep = train_groups[:max_tg]
        val_keep = val_groups[: max(1, int(round(max_tg * 0.25)))]
        train_idx = np.concatenate([np.asarray(idx, dtype=np.int64) for _, idx in train_keep]) if train_keep else np.array([], dtype=np.int64)
        val_idx = np.concatenate([np.asarray(idx, dtype=np.int64) for _, idx in val_keep]) if val_keep else np.array([], dtype=np.int64)
        train_df = train_df.iloc[train_idx].copy()
        val_df = val_df.iloc[val_idx].copy()
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

    train_frame = _numeric_frame(train_df, feature_cols)
    mean = train_frame.mean(axis=0).to_numpy(dtype="float32", copy=False)
    std = train_frame.std(axis=0, ddof=0).to_numpy(dtype="float32", copy=False)
    std = np.where(std < 1e-6, 1.0, std).astype("float32")

    train_examples = _build_team_game_examples(
        train_df,
        feature_cols=feature_cols,
        team_idx_col=TEAM_EMBED_TEAM_IDX_COL,
        opp_idx_col=TEAM_EMBED_OPP_IDX_COL,
        label_col=target_col,
        feature_mean=mean,
        feature_std=std,
        prior_weight_col=prior_weight_col if use_prior_head else None,
    )
    val_examples = _build_team_game_examples(
        val_df,
        feature_cols=feature_cols,
        team_idx_col=TEAM_EMBED_TEAM_IDX_COL,
        opp_idx_col=TEAM_EMBED_OPP_IDX_COL,
        label_col=target_col,
        feature_mean=mean,
        feature_std=std,
        prior_weight_col=prior_weight_col if use_prior_head else None,
    )

    train_loader = DataLoader(
        TeamGameDataset(train_examples),
        batch_size=int(args.batch_size),
        shuffle=True,
        collate_fn=_collate_team_games,
        num_workers=0,
        generator=torch.Generator().manual_seed(int(args.seed)),
    )
    val_loader = DataLoader(
        TeamGameDataset(val_examples),
        batch_size=int(args.batch_size),
        shuffle=False,
        collate_fn=_collate_team_games,
        num_workers=0,
    )

    config = RotationSetModelConfig(
        model=args.model,
        feature_columns=feature_cols,
        feature_mean=mean.astype(float).tolist(),
        feature_std=std.astype(float).tolist(),
        use_prior_head=use_prior_head,
        prior_weight_col=prior_weight_col,
        prior_weight_floor=prior_weight_floor,
        use_team_embeddings=True,
        team_id_vocab=team_id_vocab,
        team_embedding_dim=8,
        use_gate_head=bool(args.use_gate_head),
        num_experts=int(args.moe_experts),
        router_feature_dim=(3 if int(args.moe_experts) > 1 else 0),
        router_temperature=float(args.router_temperature),
        router_mode=str(args.router_mode),
        router_gumbel=bool(args.router_gumbel),
        router_gumbel_tau=float(args.router_gumbel_tau),
        alloc_activation=str(args.alloc_activation),
        entmax_alpha=float(args.entmax_alpha),
        share_temperature=float(args.share_temperature),
        lambda_rot=float(args.lambda_rot),
        lambda_fp=float(args.lambda_fp),
        lambda_router=float(args.lambda_router_load_balance),
        lambda_router_load_balance=float(args.lambda_router_load_balance),
        lambda_router_entropy=float(args.lambda_router_entropy),
        router_reg_type=str(args.router_reg_type),
        inj_weight_scale=float(args.inj_weight_scale),
        rot_target_mode=str(args.rot_target_mode),
        rot_target_threshold=float(args.rot_target_threshold),
        rot_target_slope=float(args.rot_target_slope),
        fp_minutes_threshold=float(args.fp_minutes_threshold),
        rot_loss_type=str(args.rot_loss_type),
        focal_gamma=float(args.focal_gamma),
        focal_alpha=float(args.focal_alpha),
    )
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr))

    use_gate_head = bool(args.use_gate_head)
    moe_experts = int(args.moe_experts)
    lambda_router_load_balance = float(args.lambda_router_load_balance)
    lambda_router_entropy = float(args.lambda_router_entropy)
    router_reg_type = str(args.router_reg_type)
    lambda_rot = float(args.lambda_rot)
    lambda_fp = float(args.lambda_fp)
    inj_weight_scale = float(args.inj_weight_scale)
    rot_target_mode = str(args.rot_target_mode)
    rot_target_threshold = float(args.rot_target_threshold)
    rot_target_slope = float(args.rot_target_slope)
    fp_minutes_threshold = float(args.fp_minutes_threshold)
    rot_loss_type = str(args.rot_loss_type)
    focal_gamma = float(args.focal_gamma)
    focal_alpha = float(args.focal_alpha)

    if moe_experts < 1:
        raise ValueError("--moe-experts must be >= 1")
    if moe_experts > 1 and not use_gate_head:
        raise ValueError("--use-gate-head is required when --moe-experts > 1")
    if moe_experts > 1 and str(args.model) != "settransformer":
        raise ValueError("--moe-experts > 1 is only supported for --model settransformer")
    if lambda_router_load_balance < 0:
        raise ValueError("--lambda-router-load-balance must be >= 0")
    if lambda_router_entropy < 0:
        raise ValueError("--lambda-router-entropy must be >= 0")
    if float(args.router_temperature) <= 0:
        raise ValueError("--router-temperature must be > 0")
    if float(args.router_gumbel_tau) <= 0:
        raise ValueError("--router-gumbel-tau must be > 0")

    if (lambda_rot > 0 or lambda_fp > 0) and not use_gate_head:
        raise ValueError("--use-gate-head is required when --lambda-rot or --lambda-fp is non-zero")
    if rot_target_mode == "soft" and rot_target_slope <= 0:
        raise ValueError("--rot-target-slope must be > 0 when --rot-target-mode=soft")
    if fp_minutes_threshold < 0:
        raise ValueError("--fp-minutes-threshold must be >= 0")
    if inj_weight_scale < 0:
        raise ValueError("--inj-weight-scale must be >= 0")
    if lambda_rot < 0 or lambda_fp < 0:
        raise ValueError("--lambda-rot/--lambda-fp must be >= 0")
    if float(args.share_temperature) <= 0:
        raise ValueError("--share-temperature must be > 0")

    best_val = float("inf")
    best_state: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        running_loss = 0.0
        running_loss_min = 0.0
        running_loss_rot = 0.0
        running_loss_fp = 0.0
        running_loss_router = 0.0
        batches = 0
        for x, team_idx, opp_idx, alloc_mask, prior_w, vacated_minutes, router_features_batch, y, mask in train_loader:
            x = x.to(device)
            team_idx = team_idx.to(device)
            opp_idx = opp_idx.to(device)
            alloc_mask = alloc_mask.to(device)
            prior_w = prior_w.to(device)
            vacated_minutes = vacated_minutes.to(device)
            router_features_batch = router_features_batch.to(device)
            y = y.to(device)
            mask = mask.to(device)

            optimizer.zero_grad(set_to_none=True)
            if use_gate_head:
                router_feature_dim = int(getattr(model, "router_feature_dim", 0))
                router_features = None
                if router_feature_dim > 0:
                    base_feat = router_features_batch
                    if base_feat.shape[1] >= router_feature_dim:
                        router_features = base_feat[:, :router_feature_dim]
                    else:
                        router_features = torch.zeros((base_feat.shape[0], router_feature_dim), device=device, dtype=base_feat.dtype)
                        router_features[:, : base_feat.shape[1]] = base_feat

                pred, gate_logits, _, router_pi_soft, router_pi_used = model.forward_with_aux(  # type: ignore[attr-defined]
                    x,
                    team_idx,
                    opp_idx,
                    mask,
                    alloc_mask=alloc_mask,
                    prior_weights=prior_w if use_prior_head else None,
                    router_features=router_features,
                )
                if gate_logits is None:
                    raise RuntimeError("use_gate_head=True but model returned gate_logits=None")
            else:
                pred = model(
                    x,
                    team_idx,
                    opp_idx,
                    mask,
                    alloc_mask=alloc_mask,
                    prior_weights=prior_w if use_prior_head else None,
                    router_features=router_features_batch[:, : int(getattr(model, "router_feature_dim", 0))]
                    if int(getattr(model, "router_feature_dim", 0)) > 0
                    else None,
                )
                gate_logits = None
                router_pi_soft = None
                router_pi_used = None

            loss_min = _masked_mae(pred, y, mask)
            loss_rot = pred.new_tensor(0.0)
            loss_fp = pred.new_tensor(0.0)
            loss_router = pred.new_tensor(0.0)
            loss_router_lb = pred.new_tensor(0.0)
            loss_router_entropy = pred.new_tensor(0.0)
            if (lambda_rot > 0 or lambda_fp > 0) and use_gate_head:
                final_mask = alloc_mask & mask
                rot_mask_f = final_mask.to(dtype=pred.dtype)
                w_inj = 1.0 + float(inj_weight_scale) * torch.clamp(vacated_minutes / 60.0, min=0.0, max=1.0)
                w_inj = w_inj.view(-1, 1)

                y_rot_hard = (y >= float(rot_target_threshold)).to(dtype=pred.dtype)
                if rot_target_mode == "soft":
                    y_rot = torch.sigmoid((y - float(rot_target_threshold)) / float(rot_target_slope)).to(dtype=pred.dtype)
                else:
                    y_rot = y_rot_hard

                weights = rot_mask_f * w_inj
                if lambda_rot > 0:
                    if rot_loss_type == "focal":
                        per_el = _focal_bce_with_logits(
                            gate_logits,  # type: ignore[arg-type]
                            y_rot,
                            gamma=float(focal_gamma),
                            alpha=float(focal_alpha),
                        )
                    else:
                        per_el = torch.nn.functional.binary_cross_entropy_with_logits(
                            gate_logits,  # type: ignore[arg-type]
                            y_rot,
                            reduction="none",
                        )
                    denom = weights.sum().clamp(min=1.0)
                    loss_rot = (per_el * weights).sum() / denom

                if lambda_fp > 0:
                    excess = torch.relu(pred - float(fp_minutes_threshold)) ** 2
                    fp = excess * (1.0 - y_rot_hard)
                    denom = weights.sum().clamp(min=1.0)
                    loss_fp = (fp * weights).sum() / denom

            if router_pi_soft is not None and moe_experts > 1:
                if lambda_router_load_balance > 0:
                    if router_reg_type == "entropy":
                        pi = router_pi_soft.clamp(min=1e-9)
                        entropy = -(pi * torch.log(pi)).sum(dim=1)
                        max_entropy = float(np.log(float(moe_experts)))
                        loss_router_lb = (max_entropy - entropy).mean()
                    else:
                        mean_pi = router_pi_soft.mean(dim=0)
                        target = 1.0 / float(moe_experts)
                        loss_router_lb = ((mean_pi - target) ** 2).sum()
                if lambda_router_entropy > 0:
                    pi = router_pi_soft.clamp(min=1e-9)
                    loss_router_entropy = -(pi * torch.log(pi)).sum(dim=1).mean()

                loss_router = float(lambda_router_load_balance) * loss_router_lb + float(lambda_router_entropy) * loss_router_entropy

            loss = (
                loss_min
                + float(lambda_rot) * loss_rot
                + float(lambda_fp) * loss_fp
                + loss_router
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += float(loss.item())
            running_loss_min += float(loss_min.item())
            running_loss_rot += float(loss_rot.item())
            running_loss_fp += float(loss_fp.item())
            running_loss_router += float(loss_router.item())
            batches += 1

        train_eval = _evaluate(
            model,
            train_loader,
            device=device,
            use_gate_head=use_gate_head,
            lambda_rot=lambda_rot,
            lambda_fp=lambda_fp,
            inj_weight_scale=inj_weight_scale,
            rot_target_mode=rot_target_mode,
            rot_target_threshold=rot_target_threshold,
            rot_target_slope=rot_target_slope,
            fp_minutes_threshold=fp_minutes_threshold,
            rot_loss_type=rot_loss_type,
            focal_gamma=focal_gamma,
            focal_alpha=focal_alpha,
        )
        val_eval = _evaluate(
            model,
            val_loader,
            device=device,
            use_gate_head=use_gate_head,
            lambda_rot=lambda_rot,
            lambda_fp=lambda_fp,
            inj_weight_scale=inj_weight_scale,
            rot_target_mode=rot_target_mode,
            rot_target_threshold=rot_target_threshold,
            rot_target_slope=rot_target_slope,
            fp_minutes_threshold=fp_minutes_threshold,
            rot_loss_type=rot_loss_type,
            focal_gamma=focal_gamma,
            focal_alpha=focal_alpha,
        )
        row = {
            "epoch": epoch,
            "train_loss": running_loss / max(batches, 1),
            "train_loss_min": running_loss_min / max(batches, 1),
            "train_loss_rot": running_loss_rot / max(batches, 1),
            "train_loss_fp": running_loss_fp / max(batches, 1),
            "train_loss_router": running_loss_router / max(batches, 1),
            "train_mae": float(train_eval["mae"]),
            "val_mae": float(val_eval["mae"]),
            "val_team_mae": float(val_eval["team_mae"]),
            "val_tail_mean_team240_top9": float(val_eval["tail_mean_team240_top9"]),
            "val_rotation_size8_mean": float(val_eval["rotation_size8_mean"]),
        }
        if "rot_loss" in val_eval:
            row["val_rot_loss"] = float(val_eval["rot_loss"])
        if "fp_loss" in val_eval:
            row["val_fp_loss"] = float(val_eval["fp_loss"])
        if "router_pi_mean" in val_eval:
            row["val_router_pi_mean"] = list(val_eval["router_pi_mean"])
            row["val_router_effective_num_experts"] = float(val_eval.get("router_effective_num_experts", float("nan")))
            row["val_router_frac_max_pi_gt_0.8"] = float(val_eval.get("router_frac_max_pi_gt_0.8", float("nan")))
            if "router_effective_num_experts_injury_slice" in val_eval:
                row["val_router_effective_num_experts_injury_slice"] = float(
                    val_eval.get("router_effective_num_experts_injury_slice", float("nan"))
                )
                row["val_router_frac_max_pi_gt_0.8_injury_slice"] = float(
                    val_eval.get("router_frac_max_pi_gt_0.8_injury_slice", float("nan"))
                )
            if "router_pi_used_mean" in val_eval:
                row["val_router_pi_used_mean"] = list(val_eval["router_pi_used_mean"])
                row["val_router_frac_max_pi_used_gt_0.8"] = float(
                    val_eval.get("router_frac_max_pi_used_gt_0.8", float("nan"))
                )
            if "router_assign_frac" in val_eval:
                row["val_router_assign_frac"] = list(val_eval["router_assign_frac"])
                row["val_router_assign_effective_num_experts"] = float(
                    val_eval.get("router_assign_effective_num_experts", float("nan"))
                )
            if "router_pi_used_mean_injury_slice" in val_eval:
                row["val_router_pi_used_mean_injury_slice"] = list(val_eval["router_pi_used_mean_injury_slice"])
                row["val_router_frac_max_pi_used_gt_0.8_injury_slice"] = float(
                    val_eval.get("router_frac_max_pi_used_gt_0.8_injury_slice", float("nan"))
                )
            if "router_assign_frac_injury_slice" in val_eval:
                row["val_router_assign_frac_injury_slice"] = list(val_eval["router_assign_frac_injury_slice"])
                row["val_router_assign_effective_num_experts_injury_slice"] = float(
                    val_eval.get("router_assign_effective_num_experts_injury_slice", float("nan"))
                )
        history.append(row)
        msg = (
            f"[epoch {epoch:03d}] loss={row['train_loss']:.4f} (min={row['train_loss_min']:.4f} "
            f"rot={row['train_loss_rot']:.4f} fp={row['train_loss_fp']:.4f} router={row['train_loss_router']:.4f}) "
            f"train_mae={row['train_mae']:.4f} val_mae={row['val_mae']:.4f} "
            f"val_tail={row['val_tail_mean_team240_top9']:.2f} val_rot8={row['val_rotation_size8_mean']:.2f}"
        )
        if "val_rot_loss" in row:
            msg += f" val_rot_loss={row['val_rot_loss']:.4f}"
        if "val_fp_loss" in row:
            msg += f" val_fp_loss={row['val_fp_loss']:.4f}"
        if "val_router_effective_num_experts" in row:
            msg += (
                f" val_router_eff={row['val_router_effective_num_experts']:.2f}"
                f" val_router_maxpi80={row['val_router_frac_max_pi_gt_0.8']:.3f}"
            )
        if "val_router_assign_effective_num_experts" in row:
            msg += (
                f" val_router_assign_eff={row['val_router_assign_effective_num_experts']:.2f}"
                f" val_router_maxpi_used80={row.get('val_router_frac_max_pi_used_gt_0.8', float('nan')):.3f}"
            )
        if "val_router_effective_num_experts_injury_slice" in row:
            msg += (
                f" val_router_eff_inj={row['val_router_effective_num_experts_injury_slice']:.2f}"
                f" val_router_maxpi80_inj={row['val_router_frac_max_pi_gt_0.8_injury_slice']:.3f}"
            )
        if "val_router_assign_effective_num_experts_injury_slice" in row:
            msg += (
                f" val_router_assign_eff_inj={row['val_router_assign_effective_num_experts_injury_slice']:.2f}"
                f" val_router_maxpi_used80_inj={row.get('val_router_frac_max_pi_used_gt_0.8_injury_slice', float('nan')):.3f}"
            )
        print(msg)
        if use_gate_head and "gate_calibration" in val_eval:
            print(f"  val_gate_calibration={val_eval['gate_calibration']}")
        if "val_router_pi_mean" in row:
            pi_str = ", ".join(f"{float(v):.3f}" for v in row["val_router_pi_mean"][:8])
            print(f"  val_router_pi_mean=[{pi_str}]")
        if "val_router_assign_frac" in row:
            frac_str = ", ".join(f"{float(v):.3f}" for v in row["val_router_assign_frac"][:8])
            print(f"  val_router_assign_frac=[{frac_str}]")

        if row["val_mae"] < best_val:
            best_val = row["val_mae"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("No best_state captured")

    model.load_state_dict(best_state)
    val_best = _evaluate(
        model,
        val_loader,
        device=device,
        use_gate_head=use_gate_head,
        lambda_rot=lambda_rot,
        lambda_fp=lambda_fp,
        inj_weight_scale=inj_weight_scale,
        rot_target_mode=rot_target_mode,
        rot_target_threshold=rot_target_threshold,
        rot_target_slope=rot_target_slope,
        fp_minutes_threshold=fp_minutes_threshold,
        rot_loss_type=rot_loss_type,
        focal_gamma=focal_gamma,
        focal_alpha=focal_alpha,
    )
    diagnostics_val = _post_train_diagnostics(model, val_loader, device=device)

    run_id = f"{config.version}_{args.model}_{_utc_now_compact()}"
    out_root = Path(args.out_dir)
    run_dir = (out_root / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.save(best_state, run_dir / "model.pt")
    config.save(run_dir / "config.json")
    (run_dir / "feature_columns.json").write_text(
        json.dumps({"columns": feature_cols}, indent=2, sort_keys=True), encoding="utf-8"
    )

    final_metrics = {"best_val_mae": float(best_val), "best_val_team_mae": float(val_best["team_mae"]), "final_epoch": int(args.epochs)}
    final_metrics.update({f"last_{k}": v for k, v in history[-1].items() if k != "epoch"})

    manifest: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "dataset_dir": str(dataset_dir),
        "inputs": {"features": str(features_path), "labels": str(labels_path)},
        "model": config.to_dict(),
        "label_handling": ot_meta,
        "dataset_qc": qc_meta,
        "split": split_meta,
        "counts": {
            "rows_train": int(len(train_df)),
            "rows_val": int(len(val_df)),
            "team_games_train": int(len(train_examples)),
            "team_games_val": int(len(val_examples)),
        },
        "metrics": final_metrics,
        "diagnostics": {"val": diagnostics_val},
        "history": history,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print("\n[rotation_set_minutes] Done")
    print(f"  run_id: {run_id}")
    print(f"  best_val_mae: {best_val:.4f}")
    print(f"  val_dust_rate: {diagnostics_val['dust_rate']:.4f}")
    print(f"  val_top8_share_mean: {diagnostics_val['top8_share_mean']:.4f}")
    print(f"  val_dnp_pred_median: {diagnostics_val['dnp_pred_median']}")
    print(f"  artifacts: {run_dir}")


if __name__ == "__main__":
    main()
