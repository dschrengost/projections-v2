"""
Generate injury behavior report for usage shares models.

Compares baseline rate_weighted vs LGBM vs NN across overall and bucketed metrics,
with special focus on vacancy/injury reallocation behavior.

Usage:
    uv run python -m scripts.usage_shares_v1.model_report \
        --data-root /home/daniel/projections-data \
        --run-id 20251215_220429 \
        --targets fga,tov \
        --backends baseline,lgbm,nn \
        --start-date 2024-11-01 \
        --end-date 2025-02-01 \
        --split val \
        --max-examples 25
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import typer
from scipy import stats as scipy_stats

from projections.paths import data_path
from projections.usage_shares_v1.features import (
    GROUP_COLS,
    add_derived_features,
)
from projections.usage_shares_v1.metrics import compute_baseline_log_weights
from projections.usage_shares_v1.production import (
    UsageSharesBundle,
    load_bundle,
    predict_log_weights,
)

app = typer.Typer(add_completion=False, help=__doc__)


# =============================================================================
# Data Loading (same as train scripts)
# =============================================================================


def load_training_data(
    data_root: Path,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """Load usage_shares_training_base partitions for the date range."""
    root = data_root / "gold" / "usage_shares_training_base"
    frames: list[pd.DataFrame] = []

    for season_dir in root.glob("season=*"):
        for day_dir in season_dir.glob("game_date=*"):
            try:
                day_str = day_dir.name.split("=", 1)[1]
                day = pd.Timestamp(day_str).normalize()
            except (ValueError, IndexError):
                continue
            if day < start_date or day > end_date:
                continue
            path = day_dir / "usage_shares_training_base.parquet"
            if path.exists():
                frames.append(pd.read_parquet(path))

    if not frames:
        raise FileNotFoundError(
            f"No usage_shares_training_base partitions found for {start_date.date()}..{end_date.date()}"
        )

    df = pd.concat(frames, ignore_index=True)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.normalize()
    return df


def apply_train_val_split(
    df: pd.DataFrame,
    split: str,
    val_days: int = 30,
) -> pd.DataFrame:
    """Apply train/val split matching training scripts (tail_days)."""
    unique_dates = sorted(df["game_date"].unique())
    if len(unique_dates) <= val_days:
        raise ValueError(f"Not enough dates for val split: {len(unique_dates)} <= {val_days}")

    val_start_date = unique_dates[-val_days]

    if split == "train":
        return df[df["game_date"] < val_start_date].copy()
    elif split == "val":
        return df[df["game_date"] >= val_start_date].copy()
    elif split == "all":
        return df.copy()
    else:
        raise ValueError(f"Unknown split: {split}")


# =============================================================================
# Metrics Computation
# =============================================================================


@dataclass
class GlobalMetrics:
    """Global metrics for a target/backend combination."""

    mae_share: float
    kl: float
    top1_acc: float
    top2_capture: float
    spearman_rank: float
    n_groups: int
    n_rows: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "mae_share": round(self.mae_share, 6),
            "kl": round(self.kl, 6),
            "top1_acc": round(self.top1_acc, 4),
            "top2_capture": round(self.top2_capture, 4),
            "spearman_rank": round(self.spearman_rank, 4),
            "n_groups": self.n_groups,
            "n_rows": self.n_rows,
        }


@dataclass
class BucketMetrics:
    """Metrics for a specific bucket."""

    bucket_name: str
    bucket_range: str
    mae_share: float
    kl: float
    top1_acc: float
    top2_capture: float
    spearman_rank: float
    n_groups: int
    n_rows: int
    improvement_vs_baseline_mae: float | None = None
    improvement_vs_baseline_pct: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "bucket_name": self.bucket_name,
            "bucket_range": self.bucket_range,
            "mae_share": round(self.mae_share, 6),
            "kl": round(self.kl, 6),
            "top1_acc": round(self.top1_acc, 4),
            "top2_capture": round(self.top2_capture, 4),
            "spearman_rank": round(self.spearman_rank, 4),
            "n_groups": self.n_groups,
            "n_rows": self.n_rows,
        }
        if self.improvement_vs_baseline_mae is not None:
            d["improvement_vs_baseline_mae"] = round(self.improvement_vs_baseline_mae, 6)
        if self.improvement_vs_baseline_pct is not None:
            d["improvement_vs_baseline_pct"] = round(self.improvement_vs_baseline_pct, 2)
        return d


@dataclass
class ReallocationSanity:
    """Reallocation sanity metrics for high-vacancy games.
    
    Two perspectives:
    - ABS (absolute): Who has highest PREDICTED share? Should be starters.
    - DELTA (change): Who got biggest INCREASE vs baseline? May include bench players.
    """
    # ABSOLUTE top-share player stats
    abs_frac_starter: float
    abs_frac_top2_minutes: float
    abs_avg_minutes_rank: float
    
    # DELTA (vs baseline) top-beneficiary stats  
    delta_frac_starter: float
    delta_frac_top2_minutes: float
    delta_avg_minutes_rank: float
    
    n_games: int
    per_game_details: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "abs_frac_starter": round(self.abs_frac_starter, 4),
            "abs_frac_top2_minutes": round(self.abs_frac_top2_minutes, 4),
            "abs_avg_minutes_rank": round(self.abs_avg_minutes_rank, 4),
            "delta_frac_starter": round(self.delta_frac_starter, 4),
            "delta_frac_top2_minutes": round(self.delta_frac_top2_minutes, 4),
            "delta_avg_minutes_rank": round(self.delta_avg_minutes_rank, 4),
            "n_games": self.n_games,
        }


@dataclass
class WorstCaseExample:
    """A worst-case team-game example."""

    game_id: int
    team_id: int
    game_date: str
    kl: float
    vac_min_szn: float
    vac_fga_szn: float
    players: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "team_id": self.team_id,
            "game_date": self.game_date,
            "kl": round(self.kl, 6),
            "vac_min_szn": round(self.vac_min_szn, 4) if pd.notna(self.vac_min_szn) else None,
            "vac_fga_szn": round(self.vac_fga_szn, 4) if pd.notna(self.vac_fga_szn) else None,
            "players": self.players,
        }


def compute_shares_from_log_weights(
    df: pd.DataFrame,
    log_weights: np.ndarray,
) -> np.ndarray:
    """Convert log-weights to shares within each (game_id, team_id) group."""
    eps = 1e-9
    weights = np.exp(log_weights)

    working = df[GROUP_COLS].copy()
    working["weight"] = weights

    group_sums = working.groupby(GROUP_COLS)["weight"].transform("sum")
    shares = weights / group_sums.clip(lower=eps).values

    return shares


def compute_global_metrics(
    df: pd.DataFrame,
    share_pred: np.ndarray,
    target: str,
) -> GlobalMetrics:
    """Compute global metrics for a target."""
    share_col = f"share_{target}"
    eps = 1e-9

    working = df[GROUP_COLS + [share_col]].copy()
    working["share_pred"] = share_pred
    working["share_true"] = working[share_col]

    # Per-row MAE
    working["abs_err"] = (working["share_pred"] - working["share_true"]).abs()

    # Group-level aggregations
    group_agg = working.groupby(GROUP_COLS).agg(
        mae=("abs_err", "mean"),
    ).reset_index()

    # KL per group
    def kl_per_group(g: pd.DataFrame) -> float:
        s_true = g["share_true"].values + eps
        s_pred = g["share_pred"].values + eps
        return float(np.sum(s_true * np.log(s_true / s_pred)))

    kl_series = working.groupby(GROUP_COLS).apply(kl_per_group, include_groups=False)

    # Top-1 accuracy
    def top1_match(g: pd.DataFrame) -> float:
        pred_top = g["share_pred"].idxmax()
        true_top = g["share_true"].idxmax()
        return 1.0 if pred_top == true_top else 0.0

    top1_series = working.groupby(GROUP_COLS).apply(top1_match, include_groups=False)

    # Top-2 capture: do predicted top2 contain true top2?
    def top2_capture(g: pd.DataFrame) -> float:
        pred_top2 = set(g["share_pred"].nlargest(2).index)
        true_top2 = set(g["share_true"].nlargest(2).index)
        return 1.0 if true_top2.issubset(pred_top2) else 0.0

    top2_series = working.groupby(GROUP_COLS).apply(top2_capture, include_groups=False)

    # Spearman rank correlation per group
    def spearman_per_group(g: pd.DataFrame) -> float:
        if len(g) < 3:
            return np.nan
        # Check for constant input (causes scipy warning)
        if g["share_true"].nunique() <= 1 or g["share_pred"].nunique() <= 1:
            return np.nan
        corr, _ = scipy_stats.spearmanr(g["share_true"], g["share_pred"])
        return corr

    spearman_series = working.groupby(GROUP_COLS).apply(spearman_per_group, include_groups=False)

    return GlobalMetrics(
        mae_share=float(group_agg["mae"].mean()),
        kl=float(kl_series.mean()),
        top1_acc=float(top1_series.mean()),
        top2_capture=float(top2_series.mean()),
        spearman_rank=float(spearman_series.dropna().mean()),
        n_groups=len(group_agg),
        n_rows=len(working),
    )


def compute_bucketed_metrics(
    df: pd.DataFrame,
    share_pred: np.ndarray,
    target: str,
    bucket_col: str,
    n_buckets: int = 4,
) -> list[BucketMetrics]:
    """Compute metrics bucketed by a column (e.g., vac_min_szn)."""
    if bucket_col not in df.columns:
        return []

    share_col = f"share_{target}"
    working = df[GROUP_COLS + [share_col, bucket_col]].copy()
    working["share_pred"] = share_pred

    # Use group-level bucket value (same for all players in group)
    group_bucket = working.groupby(GROUP_COLS)[bucket_col].first()

    # Create quantile buckets
    try:
        bucket_labels = [f"Q{i}" for i in range(1, n_buckets + 1)]
        bucket_values = pd.qcut(group_bucket, n_buckets, labels=bucket_labels, duplicates="drop")
    except ValueError:
        # Not enough unique values for quantile cut
        return []

    bucket_map = bucket_values.to_dict()
    working["bucket"] = working.set_index(GROUP_COLS).index.map(lambda x: bucket_map.get(x, None))

    # Get bucket ranges
    bucket_ranges: dict[str, str] = {}
    for bucket_label in bucket_labels:
        mask = bucket_values == bucket_label
        if mask.any():
            vals = group_bucket[mask]
            bucket_ranges[bucket_label] = f"{vals.min():.2f}-{vals.max():.2f}"

    results = []
    for bucket_label in bucket_labels:
        bucket_mask = working["bucket"] == bucket_label
        if not bucket_mask.any():
            continue

        bucket_df = working[bucket_mask].copy()

        eps = 1e-9
        bucket_df["share_true"] = bucket_df[share_col]
        bucket_df["abs_err"] = (bucket_df["share_pred"] - bucket_df["share_true"]).abs()

        group_agg = bucket_df.groupby(GROUP_COLS).agg(mae=("abs_err", "mean")).reset_index()

        def kl_per_group(g: pd.DataFrame) -> float:
            s_true = g["share_true"].values + eps
            s_pred = g["share_pred"].values + eps
            return float(np.sum(s_true * np.log(s_true / s_pred)))

        kl_series = bucket_df.groupby(GROUP_COLS).apply(kl_per_group, include_groups=False)

        def top1_match(g: pd.DataFrame) -> float:
            return 1.0 if g["share_pred"].idxmax() == g["share_true"].idxmax() else 0.0

        def top2_capture(g: pd.DataFrame) -> float:
            pred_top2 = set(g["share_pred"].nlargest(2).index)
            true_top2 = set(g["share_true"].nlargest(2).index)
            return 1.0 if true_top2.issubset(pred_top2) else 0.0

        def spearman_per_group(g: pd.DataFrame) -> float:
            if len(g) < 3:
                return np.nan
            # Check for constant input (causes scipy warning)
            if g["share_true"].nunique() <= 1 or g["share_pred"].nunique() <= 1:
                return np.nan
            corr, _ = scipy_stats.spearmanr(g["share_true"], g["share_pred"])
            return corr

        top1_series = bucket_df.groupby(GROUP_COLS).apply(top1_match, include_groups=False)
        top2_series = bucket_df.groupby(GROUP_COLS).apply(top2_capture, include_groups=False)
        spearman_series = bucket_df.groupby(GROUP_COLS).apply(spearman_per_group, include_groups=False)

        results.append(
            BucketMetrics(
                bucket_name=f"{bucket_col}_{bucket_label}",
                bucket_range=bucket_ranges.get(bucket_label, ""),
                mae_share=float(group_agg["mae"].mean()),
                kl=float(kl_series.mean()),
                top1_acc=float(top1_series.mean()),
                top2_capture=float(top2_series.mean()),
                spearman_rank=float(spearman_series.dropna().mean()),
                n_groups=len(group_agg),
                n_rows=len(bucket_df),
            )
        )

    return results


def compute_reallocation_sanity(
    df: pd.DataFrame,
    share_baseline: np.ndarray,
    share_model: np.ndarray,
    target: str,
    vacancy_col: str = "vac_min_szn",
    top_percentile: float = 0.10,
) -> ReallocationSanity:
    """
    Compute reallocation sanity score for high-vacancy games.

    Reports TWO perspectives:
    - ABS: Who has highest ABSOLUTE predicted share? (Should be starters/high-min)
    - DELTA: Who got biggest INCREASE vs baseline? (May include bench players)
    """
    share_col = f"share_{target}"

    working = df[GROUP_COLS + [share_col, vacancy_col, "is_starter", "minutes_pred_p50"]].copy()
    working["share_baseline"] = share_baseline
    working["share_model"] = share_model
    working["share_change"] = share_model - share_baseline

    # Get group-level vacancy
    group_vacancy = working.groupby(GROUP_COLS)[vacancy_col].first()

    # Find high-vacancy threshold (top percentile)
    threshold = group_vacancy.quantile(1 - top_percentile)
    high_vac_groups = set(group_vacancy[group_vacancy >= threshold].index)

    if not high_vac_groups:
        return ReallocationSanity(
            abs_frac_starter=np.nan,
            abs_frac_top2_minutes=np.nan,
            abs_avg_minutes_rank=np.nan,
            delta_frac_starter=np.nan,
            delta_frac_top2_minutes=np.nan,
            delta_avg_minutes_rank=np.nan,
            n_games=0,
        )

    # Filter to high-vacancy games
    working["group_key"] = list(zip(working["game_id"], working["team_id"]))
    high_vac_df = working[working["group_key"].isin(high_vac_groups)].copy()

    # Per-game analysis - collect BOTH ABS and DELTA metrics
    abs_is_starter = []
    abs_is_top2_minutes = []
    abs_minutes_ranks = []
    
    delta_is_starter = []
    delta_is_top2_minutes = []
    delta_minutes_ranks = []
    
    per_game_details = []

    for (game_id, team_id), group in high_vac_df.groupby(GROUP_COLS):
        # Minutes rank within team (1 = highest)
        minutes_ranks = group["minutes_pred_p50"].rank(ascending=False, method="min")
        
        # ABS: player with highest ABSOLUTE predicted share
        abs_top_idx = group["share_model"].idxmax()
        abs_top_player = group.loc[abs_top_idx]
        abs_top_minutes_rank = minutes_ranks.loc[abs_top_idx]
        
        abs_is_starter.append(abs_top_player["is_starter"] == 1)
        abs_is_top2_minutes.append(abs_top_minutes_rank <= 2)
        abs_minutes_ranks.append(abs_top_minutes_rank)
        
        # DELTA: player with highest CHANGE from baseline
        delta_top_idx = group["share_change"].idxmax()
        delta_top_player = group.loc[delta_top_idx]
        delta_top_minutes_rank = minutes_ranks.loc[delta_top_idx]
        
        delta_is_starter.append(delta_top_player["is_starter"] == 1)
        delta_is_top2_minutes.append(delta_top_minutes_rank <= 2)
        delta_minutes_ranks.append(delta_top_minutes_rank)

        per_game_details.append(
            {
                "game_id": int(game_id),
                "team_id": int(team_id),
                "abs_player_id": int(group.loc[abs_top_idx, "player_id"])
                if "player_id" in group.columns else None,
                "abs_is_starter": bool(abs_top_player["is_starter"] == 1),
                "abs_minutes_rank": int(abs_top_minutes_rank),
                "abs_share": round(float(abs_top_player["share_model"]), 4),
                "delta_player_id": int(group.loc[delta_top_idx, "player_id"])
                if "player_id" in group.columns else None,
                "delta_is_starter": bool(delta_top_player["is_starter"] == 1),
                "delta_minutes_rank": int(delta_top_minutes_rank),
                "delta_share_change": round(float(delta_top_player["share_change"]), 4),
            }
        )

    return ReallocationSanity(
        abs_frac_starter=float(np.mean(abs_is_starter)),
        abs_frac_top2_minutes=float(np.mean(abs_is_top2_minutes)),
        abs_avg_minutes_rank=float(np.mean(abs_minutes_ranks)),
        delta_frac_starter=float(np.mean(delta_is_starter)),
        delta_frac_top2_minutes=float(np.mean(delta_is_top2_minutes)),
        delta_avg_minutes_rank=float(np.mean(delta_minutes_ranks)),
        n_games=len(abs_is_starter),
        per_game_details=per_game_details,
    )


def find_worst_case_examples(
    df: pd.DataFrame,
    share_pred: np.ndarray,
    share_baseline: np.ndarray,
    target: str,
    topk: int = 6,
    max_examples: int = 25,
) -> list[WorstCaseExample]:
    """Find worst-case team-games by KL divergence."""
    share_col = f"share_{target}"
    eps = 1e-9

    working = df.copy()
    working["share_pred"] = share_pred
    working["share_baseline"] = share_baseline
    working["share_true"] = working[share_col]

    # Compute KL per group
    def kl_per_group(g: pd.DataFrame) -> float:
        s_true = g["share_true"].values + eps
        s_pred = g["share_pred"].values + eps
        return float(np.sum(s_true * np.log(s_true / s_pred)))

    kl_series = working.groupby(GROUP_COLS).apply(kl_per_group, include_groups=False)

    # Get worst groups
    worst_groups = kl_series.nlargest(max_examples)

    examples = []
    for (game_id, team_id), kl_val in worst_groups.items():
        group = working[(working["game_id"] == game_id) & (working["team_id"] == team_id)]

        # Get vacancy features (same for all players in group)
        vac_min = group["vac_min_szn"].iloc[0] if "vac_min_szn" in group.columns else np.nan
        vac_fga = group["vac_fga_szn"].iloc[0] if "vac_fga_szn" in group.columns else np.nan
        game_date = group["game_date"].iloc[0]

        # Get top-k players by true share
        top_players = group.nlargest(topk, "share_true")
        players = []
        for _, row in top_players.iterrows():
            players.append(
                {
                    "player_id": int(row["player_id"]) if "player_id" in row else None,
                    "minutes_pred_p50": round(float(row.get("minutes_pred_p50", 0)), 1),
                    "is_starter": int(row.get("is_starter", 0)),
                    "role_cluster": int(row.get("track_role_cluster", -1))
                    if "track_role_cluster" in row
                    else None,
                    "true_share": round(float(row["share_true"]), 4),
                    "pred_share": round(float(row["share_pred"]), 4),
                    "baseline_share": round(float(row["share_baseline"]), 4),
                }
            )

        examples.append(
            WorstCaseExample(
                game_id=int(game_id),
                team_id=int(team_id),
                game_date=str(game_date.date()) if hasattr(game_date, "date") else str(game_date),
                kl=kl_val,
                vac_min_szn=vac_min,
                vac_fga_szn=vac_fga,
                players=players,
            )
        )

    return examples


# =============================================================================
# Counterfactual Sensitivity (Optional)
# =============================================================================


@dataclass
class CounterfactualSensitivity:
    """Results from counterfactual vacancy perturbation."""

    mean_abs_share_change: float
    share_change_concentration_starters: float
    share_change_concentration_top2_minutes: float
    n_games: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean_abs_share_change": round(self.mean_abs_share_change, 6),
            "share_change_concentration_starters": round(self.share_change_concentration_starters, 4),
            "share_change_concentration_top2_minutes": round(self.share_change_concentration_top2_minutes, 4),
            "n_games": self.n_games,
        }


def compute_counterfactual_sensitivity(
    df: pd.DataFrame,
    bundle: UsageSharesBundle,
    target: str,
    vacancy_col: str = "vac_min_szn",
    perturbation_std: float = 1.0,
    top_percentile: float = 0.10,
) -> CounterfactualSensitivity:
    """
    Test model sensitivity to vacancy feature perturbations.

    For high-vacancy games, increase vacancy features by perturbation_std
    and measure how shares change.
    """
    # Get high-vacancy games
    group_vacancy = df.groupby(GROUP_COLS)[vacancy_col].first()
    threshold = group_vacancy.quantile(1 - top_percentile)
    high_vac_groups = set(group_vacancy[group_vacancy >= threshold].index)

    if not high_vac_groups:
        return CounterfactualSensitivity(
            mean_abs_share_change=np.nan,
            share_change_concentration_starters=np.nan,
            share_change_concentration_top2_minutes=np.nan,
            n_games=0,
        )

    # Filter to high-vacancy games
    df["group_key"] = list(zip(df["game_id"], df["team_id"]))
    high_vac_df = df[df["group_key"].isin(high_vac_groups)].copy()

    # Original predictions
    log_weights_orig = predict_log_weights(bundle, high_vac_df, target)
    shares_orig = compute_shares_from_log_weights(high_vac_df, log_weights_orig)

    # Perturbed predictions (increase vacancy by 1 std)
    perturbed_df = high_vac_df.copy()
    vacancy_std = df[vacancy_col].std()
    for col in ["vac_min_szn", "vac_fga_szn", "vac_min_guard_szn", "vac_min_wing_szn", "vac_min_big_szn"]:
        if col in perturbed_df.columns:
            perturbed_df[col] = perturbed_df[col] + perturbation_std * vacancy_std

    log_weights_pert = predict_log_weights(bundle, perturbed_df, target)
    shares_pert = compute_shares_from_log_weights(high_vac_df, log_weights_pert)  # Use original group structure

    # Compute share changes
    high_vac_df["share_orig"] = shares_orig
    high_vac_df["share_pert"] = shares_pert
    high_vac_df["share_change"] = shares_pert - shares_orig
    high_vac_df["abs_share_change"] = np.abs(shares_pert - shares_orig)

    # Mean absolute share change per game
    mean_change_per_game = high_vac_df.groupby(GROUP_COLS)["abs_share_change"].mean()
    mean_abs_share_change = float(mean_change_per_game.mean())

    # Concentration: do changes go to starters / high-minutes players?
    # For each game, check if the biggest share increase goes to starter / top-2 minutes
    concentration_starters = []
    concentration_top2_minutes = []

    for _, group in high_vac_df.groupby(GROUP_COLS):
        top_beneficiary_idx = group["share_change"].idxmax()
        top_beneficiary = group.loc[top_beneficiary_idx]

        minutes_ranks = group["minutes_pred_p50"].rank(ascending=False, method="min")
        beneficiary_minutes_rank = minutes_ranks.loc[top_beneficiary_idx]

        concentration_starters.append(top_beneficiary["is_starter"] == 1)
        concentration_top2_minutes.append(beneficiary_minutes_rank <= 2)

    return CounterfactualSensitivity(
        mean_abs_share_change=mean_abs_share_change,
        share_change_concentration_starters=float(np.mean(concentration_starters)),
        share_change_concentration_top2_minutes=float(np.mean(concentration_top2_minutes)),
        n_games=len(concentration_starters),
    )


# =============================================================================
# Report Generation
# =============================================================================


@dataclass
class TargetReport:
    """Report for a single target."""

    target: str
    global_metrics: dict[str, GlobalMetrics]  # backend -> metrics
    bucketed_metrics: dict[str, list[BucketMetrics]]  # backend -> list of bucket metrics
    reallocation_sanity: dict[str, ReallocationSanity]  # backend -> sanity metrics
    worst_case_examples: dict[str, list[WorstCaseExample]]  # backend -> examples
    counterfactual: dict[str, CounterfactualSensitivity] | None = None  # backend -> sensitivity

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "target": self.target,
            "global_metrics": {k: v.to_dict() for k, v in self.global_metrics.items()},
            "bucketed_metrics": {k: [b.to_dict() for b in v] for k, v in self.bucketed_metrics.items()},
            "reallocation_sanity": {k: v.to_dict() for k, v in self.reallocation_sanity.items()},
            "worst_case_examples": {k: [e.to_dict() for e in v] for k, v in self.worst_case_examples.items()},
        }
        if self.counterfactual:
            d["counterfactual"] = {k: v.to_dict() for k, v in self.counterfactual.items()}
        return d


@dataclass
class FullReport:
    """Full model comparison report."""

    metadata: dict[str, Any]
    targets: dict[str, TargetReport]  # target -> report

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata,
            "targets": {k: v.to_dict() for k, v in self.targets.items()},
        }


def generate_target_report(
    df: pd.DataFrame,
    target: str,
    backends: list[str],
    bundles: dict[str, UsageSharesBundle | None],
    topk: int,
    max_examples: int,
    do_counterfactual: bool = False,
) -> TargetReport:
    """Generate report for a single target."""
    # Filter to valid rows for this target
    # NaN validity is treated as valid (handles newer data without explicit flags)
    share_col = f"share_{target}"
    valid_col = f"share_{target}_valid"
    
    if valid_col in df.columns:
        explicit_valid = df[valid_col].fillna(True)  # NaN = assume valid
    else:
        explicit_valid = pd.Series(True, index=df.index)
    share_finite = df[share_col].notna() & np.isfinite(df[share_col])
    valid_mask = explicit_valid & share_finite
    target_df = df[valid_mask].copy()

    if len(target_df) == 0:
        raise ValueError(f"No valid rows for target {target}")

    typer.echo(f"  [{target}] {len(target_df):,} valid rows, {target_df.groupby(GROUP_COLS).ngroups:,} groups")

    # Compute predictions for all backends
    predictions: dict[str, np.ndarray] = {}
    shares: dict[str, np.ndarray] = {}

    for backend in backends:
        if backend == "baseline":
            log_weights = compute_baseline_log_weights(target_df, target)
        else:
            bundle = bundles.get(backend)
            if bundle is None:
                typer.echo(f"  [{target}] Skipping {backend}: bundle not loaded")
                continue
            log_weights = predict_log_weights(bundle, target_df, target)

        predictions[backend] = log_weights
        shares[backend] = compute_shares_from_log_weights(target_df, log_weights)

    if "baseline" not in shares:
        raise ValueError("Baseline predictions required")

    # Global metrics
    global_metrics: dict[str, GlobalMetrics] = {}
    for backend, share_pred in shares.items():
        global_metrics[backend] = compute_global_metrics(target_df, share_pred, target)

    # Bucketed metrics (vacancy-focused)
    bucketed_metrics: dict[str, list[BucketMetrics]] = {}
    bucket_cols = ["vac_min_szn", "vac_fga_szn", "is_starter", "minutes_pred_p50"]

    for backend, share_pred in shares.items():
        all_bucket_metrics: list[BucketMetrics] = []
        for bucket_col in bucket_cols:
            if bucket_col not in target_df.columns:
                continue
            bucket_metrics = compute_bucketed_metrics(target_df, share_pred, target, bucket_col)

            # Add improvement vs baseline
            if backend != "baseline" and "baseline" in shares:
                baseline_bucket_metrics = compute_bucketed_metrics(
                    target_df, shares["baseline"], target, bucket_col
                )
                baseline_mae_map = {b.bucket_name: b.mae_share for b in baseline_bucket_metrics}
                for bm in bucket_metrics:
                    baseline_mae = baseline_mae_map.get(bm.bucket_name)
                    if baseline_mae and baseline_mae > 0:
                        bm.improvement_vs_baseline_mae = baseline_mae - bm.mae_share
                        bm.improvement_vs_baseline_pct = (baseline_mae - bm.mae_share) / baseline_mae * 100

            all_bucket_metrics.extend(bucket_metrics)
        bucketed_metrics[backend] = all_bucket_metrics

    # Reallocation sanity (for models vs baseline)
    reallocation_sanity: dict[str, ReallocationSanity] = {}
    for backend, share_pred in shares.items():
        if backend == "baseline":
            continue
        reallocation_sanity[backend] = compute_reallocation_sanity(
            target_df, shares["baseline"], share_pred, target
        )

    # Worst-case examples
    worst_case_examples: dict[str, list[WorstCaseExample]] = {}
    for backend, share_pred in shares.items():
        worst_case_examples[backend] = find_worst_case_examples(
            target_df, share_pred, shares["baseline"], target, topk, max_examples
        )

    # Counterfactual (optional, for non-baseline backends)
    counterfactual: dict[str, CounterfactualSensitivity] | None = None
    if do_counterfactual:
        counterfactual = {}
        for backend in backends:
            if backend == "baseline":
                continue
            bundle = bundles.get(backend)
            if bundle is None:
                continue
            counterfactual[backend] = compute_counterfactual_sensitivity(target_df, bundle, target)

    return TargetReport(
        target=target,
        global_metrics=global_metrics,
        bucketed_metrics=bucketed_metrics,
        reallocation_sanity=reallocation_sanity,
        worst_case_examples=worst_case_examples,
        counterfactual=counterfactual,
    )


# =============================================================================
# Console Output
# =============================================================================


def print_global_metrics_table(report: FullReport) -> None:
    """Print global metrics table."""
    typer.echo("\n" + "=" * 80)
    typer.echo("GLOBAL METRICS")
    typer.echo("=" * 80)

    for target, target_report in report.targets.items():
        typer.echo(f"\n{target.upper()}")
        typer.echo("-" * 70)
        typer.echo(f"{'Backend':<12} {'MAE':>10} {'KL':>10} {'Top1':>8} {'Top2':>8} {'Spearman':>10}")
        typer.echo("-" * 70)

        baseline_mae = target_report.global_metrics.get("baseline", GlobalMetrics(0, 0, 0, 0, 0, 0, 0)).mae_share

        for backend, metrics in target_report.global_metrics.items():
            improvement = ""
            if backend != "baseline" and baseline_mae > 0:
                pct = (baseline_mae - metrics.mae_share) / baseline_mae * 100
                improvement = f" ({pct:+.1f}%)"

            typer.echo(
                f"{backend:<12} {metrics.mae_share:>10.5f} {metrics.kl:>10.5f} "
                f"{metrics.top1_acc:>7.1%} {metrics.top2_capture:>7.1%} "
                f"{metrics.spearman_rank:>10.3f}{improvement}"
            )


def print_bucketed_metrics_summary(report: FullReport) -> None:
    """Print bucketed metrics summary focusing on vacancy buckets."""
    typer.echo("\n" + "=" * 80)
    typer.echo("VACANCY-BUCKETED METRICS (vac_min_szn)")
    typer.echo("=" * 80)

    for target, target_report in report.targets.items():
        typer.echo(f"\n{target.upper()}")

        # Get vac_min_szn buckets only
        for backend, buckets in target_report.bucketed_metrics.items():
            vac_buckets = [b for b in buckets if b.bucket_name.startswith("vac_min_szn")]
            if not vac_buckets:
                continue

            typer.echo(f"\n  {backend}:")
            typer.echo(f"  {'Bucket':<25} {'Range':>15} {'MAE':>10} {'vs BL':>12} {'KL':>10}")
            typer.echo("  " + "-" * 75)

            for b in vac_buckets:
                improvement_str = ""
                if b.improvement_vs_baseline_pct is not None:
                    improvement_str = f"{b.improvement_vs_baseline_pct:+.1f}%"

                typer.echo(
                    f"  {b.bucket_name:<25} {b.bucket_range:>15} {b.mae_share:>10.5f} "
                    f"{improvement_str:>12} {b.kl:>10.5f}"
                )


def print_reallocation_sanity(report: FullReport) -> None:
    """Print reallocation sanity summary."""
    typer.echo("\n" + "=" * 80)
    typer.echo("REALLOCATION SANITY (High-Vacancy Games)")
    typer.echo("=" * 80)
    typer.echo("ABS = player with highest PREDICTED share (should be starters)")
    typer.echo("DELTA = player with biggest INCREASE from baseline (may include bench)")
    
    typer.echo("\n--- ABSOLUTE TOP-SHARE PLAYER (Model's #1 pick) ---")
    typer.echo(f"{'Target':<10} {'Backend':<12} {'% Starter':>12} {'% Top2 Min':>12} {'Avg Rank':>10} {'N Games':>10}")
    typer.echo("-" * 70)

    for target, target_report in report.targets.items():
        for backend, sanity in target_report.reallocation_sanity.items():
            typer.echo(
                f"{target:<10} {backend:<12} {sanity.abs_frac_starter:>11.1%} "
                f"{sanity.abs_frac_top2_minutes:>11.1%} {sanity.abs_avg_minutes_rank:>10.1f} "
                f"{sanity.n_games:>10}"
            )

    typer.echo("\n--- DELTA TOP-BENEFICIARY (Biggest gain vs baseline) ---")
    typer.echo(f"{'Target':<10} {'Backend':<12} {'% Starter':>12} {'% Top2 Min':>12} {'Avg Rank':>10} {'N Games':>10}")
    typer.echo("-" * 70)

    for target, target_report in report.targets.items():
        for backend, sanity in target_report.reallocation_sanity.items():
            typer.echo(
                f"{target:<10} {backend:<12} {sanity.delta_frac_starter:>11.1%} "
                f"{sanity.delta_frac_top2_minutes:>11.1%} {sanity.delta_avg_minutes_rank:>10.1f} "
                f"{sanity.n_games:>10}"
            )


def print_worst_examples_summary(report: FullReport, n_print: int = 10) -> None:
    """Print worst-case examples summary."""
    typer.echo("\n" + "=" * 80)
    typer.echo(f"WORST-CASE EXAMPLES (Top {n_print} by KL)")
    typer.echo("=" * 80)

    for target, target_report in report.targets.items():
        for backend, examples in target_report.worst_case_examples.items():
            if not examples:
                continue
            typer.echo(f"\n{target.upper()} / {backend}")
            typer.echo("-" * 70)

            for ex in examples[:n_print]:
                vac_str = f"{ex.vac_min_szn:.2f}" if pd.notna(ex.vac_min_szn) else "N/A"
                typer.echo(
                    f"  game={ex.game_id} team={ex.team_id} date={ex.game_date} "
                    f"KL={ex.kl:.4f} vac_min={vac_str}"
                )


# =============================================================================
# Main Command
# =============================================================================


@app.command()
def main(
    data_root: Path = typer.Option(
        None,
        help="Root data directory (defaults to PROJECTIONS_DATA_ROOT).",
    ),
    run_id: str = typer.Option(
        ...,
        help="Run ID for model artifacts.",
    ),
    targets: str = typer.Option(
        "fga,tov",
        help="Comma-separated list of targets to analyze.",
    ),
    backends: str = typer.Option(
        "baseline,lgbm,nn",
        help="Comma-separated list of backends to compare.",
    ),
    start_date: str = typer.Option(
        ...,
        help="Start date (YYYY-MM-DD) inclusive.",
    ),
    end_date: str = typer.Option(
        ...,
        help="End date (YYYY-MM-DD) inclusive.",
    ),
    split: str = typer.Option(
        "val",
        help="Split to analyze: train, val, or all.",
    ),
    val_days: int = typer.Option(
        30,
        help="Number of trailing days for validation split.",
    ),
    topk: int = typer.Option(
        6,
        help="Number of top players to show in worst-case examples.",
    ),
    max_examples: int = typer.Option(
        25,
        help="Maximum number of worst-case examples to collect.",
    ),
    out: Path = typer.Option(
        None,
        help="Output path for JSON report.",
    ),
    csv_out: Path = typer.Option(
        None,
        help="Output path for CSV summary.",
    ),
    counterfactual: bool = typer.Option(
        False,
        help="Run counterfactual vacancy sensitivity analysis.",
    ),
    seed: int = typer.Option(
        1337,
        help="Random seed for reproducibility.",
    ),
) -> None:
    """Generate injury behavior report for usage shares models."""
    np.random.seed(seed)

    root = data_root or data_path()
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    target_list = [t.strip() for t in targets.split(",")]
    backend_list = [b.strip() for b in backends.split(",")]

    # Generate report ID
    report_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{run_id}"

    typer.echo(f"[report] Loading data from {start.date()} to {end.date()}...")
    df = load_training_data(root, start, end)
    typer.echo(f"[report] Loaded {len(df):,} rows")

    # Add derived features
    df = add_derived_features(df)

    # Apply split
    df = apply_train_val_split(df, split, val_days)
    typer.echo(f"[report] Using {split} split: {len(df):,} rows")

    # Load model bundles
    bundles: dict[str, UsageSharesBundle | None] = {"baseline": None}

    for backend in backend_list:
        if backend == "baseline":
            continue
        try:
            bundle = load_bundle(root, run_id=run_id, backend=backend)
            bundles[backend] = bundle
            typer.echo(f"[report] Loaded {backend} bundle from run {run_id}")
        except FileNotFoundError as e:
            typer.echo(f"[report] Warning: Could not load {backend} bundle: {e}")
            bundles[backend] = None

    # Generate reports per target
    target_reports: dict[str, TargetReport] = {}

    for target in target_list:
        typer.echo(f"\n[report] Analyzing target: {target}")
        try:
            target_reports[target] = generate_target_report(
                df,
                target,
                backend_list,
                bundles,
                topk,
                max_examples,
                do_counterfactual=counterfactual,
            )
        except ValueError as e:
            typer.echo(f"[report] Skipping {target}: {e}")

    # Build full report
    metadata = {
        "report_id": report_id,
        "run_id": run_id,
        "date_range": [start.date().isoformat(), end.date().isoformat()],
        "split": split,
        "val_days": val_days,
        "targets": target_list,
        "backends": backend_list,
        "n_rows": len(df),
        "created_at": datetime.now().isoformat(),
    }

    full_report = FullReport(metadata=metadata, targets=target_reports)

    # Print console summary
    print_global_metrics_table(full_report)
    print_bucketed_metrics_summary(full_report)
    print_reallocation_sanity(full_report)
    print_worst_examples_summary(full_report)

    # Save JSON report
    if out is None:
        reports_dir = root / "artifacts" / "usage_shares_v1" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        out = reports_dir / f"{report_id}.json"

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(full_report.to_dict(), indent=2))
    typer.echo(f"\n[report] JSON report saved to {out}")

    # Save CSV summary (optional)
    if csv_out:
        rows = []
        for target, tr in full_report.targets.items():
            for backend, gm in tr.global_metrics.items():
                rows.append(
                    {
                        "target": target,
                        "backend": backend,
                        "mae_share": gm.mae_share,
                        "kl": gm.kl,
                        "top1_acc": gm.top1_acc,
                        "top2_capture": gm.top2_capture,
                        "spearman_rank": gm.spearman_rank,
                        "n_groups": gm.n_groups,
                    }
                )
        csv_df = pd.DataFrame(rows)
        csv_df.to_csv(csv_out, index=False)
        typer.echo(f"[report] CSV summary saved to {csv_out}")


if __name__ == "__main__":
    app()
