#!/usr/bin/env python
"""Walk-forward cross-validation for minutes models.

This script runs multiple training folds, each with a progressively advancing
time window to simulate production deployment. Results are aggregated to
provide robust performance estimates.

Supports two model types:
- lgbm: Standard LightGBM quantile model (projections.models.minutes_lgbm)
- minute_share: Minute share model with team normalization (projections.cli.train_minute_share)

Features:
- NBA season-aware: automatically skips off-season periods (July-September)
- Full MLFlow integration: logs aggregated metrics, feature importance, per-fold results
- Configurable windows for train/cal/val splits

Usage:
    # Standard LightGBM model
    uv run python scripts/walk_forward_minutes.py \
        --features /path/to/features.parquet \
        --start 2022-10-18 \
        --end 2025-11-30 \
        --train-months 12 \
        --cal-weeks 3 \
        --val-weeks 2 \
        --step-weeks 4 \
        --season-aware

    # Minute share model
    uv run python scripts/walk_forward_minutes.py \
        --features /path/to/features.parquet \
        --start 2022-10-18 \
        --end 2025-11-30 \
        --model-type minute_share \
        --train-months 12 \
        --val-weeks 2 \
        --step-weeks 4

This will create folds like:
    Fold 1: Train [2022-10 → 2023-11] | Cal [2023-11 → 2023-12] | Val [2023-12]
    Fold 2: Train [2022-10 → 2023-12] | Cal [2023-12 → 2024-01] | Val [2024-01]
    ...
(Off-season folds with validation in July-September are automatically skipped)
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd
import typer


class ModelType(str, Enum):
    """Supported model types for walk-forward CV."""

    LGBM = "lgbm"
    MINUTE_SHARE = "minute_share"


# Model-specific configuration
MODEL_CONFIG = {
    ModelType.LGBM: {
        "module": "projections.models.minutes_lgbm",
        "artifact_dir": "artifacts/minutes_lgbm",
        "uses_calibration": True,
        "metric_keys": [
            "val_mae_p50",
            "val_p10_cond_playable",
            "val_p90_cond_playable",
            "val_winkler_cond_playable",
            "val_inside_cond_playable",
            "train_rows",
            "cal_rows",
            "val_rows",
        ],
    },
    ModelType.MINUTE_SHARE: {
        "module": "projections.cli.train_minute_share",
        "artifact_dir": "artifacts/minute_share",
        "uses_calibration": False,
        "metric_keys": [
            # Overall regression
            "mae_minutes",
            "mae_shares",
            "rmse_minutes",
            # MAE by bucket
            "mae_dnp",
            "mae_garbage_time",
            "mae_bench",
            "mae_rotation",
            "mae_core",
            "mae_stars",
            # DNP classification
            "dnp_auc",
            "dnp_accuracy",
            "dnp_precision",
            "dnp_recall",
            # Rotation classification
            "rotation_accuracy",
            "rotation_precision",
            "rotation_recall",
            # Share distribution
            "raw_pred_share_sum_pre_norm_mean",
            "raw_pred_share_sum_pre_norm_std",
            "norm_share_sum_post_norm_mean",
            "norm_share_sum_post_norm_std",
            # Legacy alias (backward compat)
            "team_share_sum_mean",
            "team_share_sum_std",
            "max_share_mean",
            "max_share_std",
            # Sample sizes
            "n_samples",
            "n_dnp_actual",
            "n_rotation_actual",
        ],
    },
}

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# MLflow configuration
MLFLOW_TRACKING_URI = "sqlite:////home/daniel/projections-data/mlflow/mlflow.db"
MLFLOW_EXPERIMENT_NAME = "minutes_v1_walk_forward"

# NBA season boundaries (approximate)
# Regular season: mid-October to mid-April
# Playoffs: mid-April to mid-June
# Off-season: mid-June to mid-October
NBA_OFFSEASON_MONTHS = {7, 8, 9}  # July, August, September

app = typer.Typer(help=__doc__)


@dataclass
class FoldConfig:
    """Configuration for a single walk-forward fold."""

    fold_id: str
    train_start: datetime
    train_end: datetime
    cal_start: datetime | None
    cal_end: datetime | None
    val_start: datetime
    val_end: datetime

    def to_args(self, model_type: ModelType) -> list[str]:
        """Convert to CLI arguments for the specified model type."""
        if model_type == ModelType.LGBM:
            return [
                "--train-start", self.train_start.strftime("%Y-%m-%d"),
                "--train-end", self.train_end.strftime("%Y-%m-%d"),
                "--cal-start", self.cal_start.strftime("%Y-%m-%d") if self.cal_start else "",
                "--cal-end", self.cal_end.strftime("%Y-%m-%d") if self.cal_end else "",
                "--val-start", self.val_start.strftime("%Y-%m-%d"),
                "--val-end", self.val_end.strftime("%Y-%m-%d"),
                "--fold-id", self.fold_id,
            ]
        elif model_type == ModelType.MINUTE_SHARE:
            # minute_share doesn't use calibration window
            return [
                "--train-start", self.train_start.strftime("%Y-%m-%d"),
                "--train-end", self.train_end.strftime("%Y-%m-%d"),
                "--val-start", self.val_start.strftime("%Y-%m-%d"),
                "--val-end", self.val_end.strftime("%Y-%m-%d"),
            ]
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def to_dict(self) -> dict[str, str | None]:
        return {
            "fold_id": self.fold_id,
            "train_start": self.train_start.isoformat(),
            "train_end": self.train_end.isoformat(),
            "cal_start": self.cal_start.isoformat() if self.cal_start else None,
            "cal_end": self.cal_end.isoformat() if self.cal_end else None,
            "val_start": self.val_start.isoformat(),
            "val_end": self.val_end.isoformat(),
        }


@dataclass
class FoldResult:
    """Results from a single fold."""

    fold_id: str
    config: FoldConfig
    success: bool
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


def is_offseason(date: datetime) -> bool:
    """Check if a date falls in the NBA off-season (July-September)."""
    return date.month in NBA_OFFSEASON_MONTHS


def generate_folds(
    *,
    data_start: datetime,
    data_end: datetime,
    min_train_months: int,
    cal_weeks: int,
    val_weeks: int,
    step_weeks: int,
    season_aware: bool = True,
    model_type: ModelType = ModelType.LGBM,
) -> list[FoldConfig]:
    """Generate walk-forward fold configurations.

    Parameters
    ----------
    data_start
        Earliest date in the dataset.
    data_end
        Latest date in the dataset.
    min_train_months
        Minimum training window size in months.
    cal_weeks
        Calibration window size in weeks (ignored for minute_share model).
    val_weeks
        Validation window size in weeks.
    step_weeks
        How many weeks to advance between folds.
    season_aware
        If True, skip folds where validation window overlaps NBA off-season.
    model_type
        Which model type to generate folds for.

    Returns
    -------
    List of FoldConfig objects.
    """
    folds: list[FoldConfig] = []
    skipped_offseason = 0

    uses_calibration = MODEL_CONFIG[model_type]["uses_calibration"]

    # Start with minimum training window
    train_start = data_start

    # Calculate first possible val_end (need min_train + cal + val)
    min_train_end = train_start + timedelta(days=min_train_months * 30)

    fold_num = 1
    if uses_calibration:
        current_val_end = min_train_end + timedelta(weeks=cal_weeks + val_weeks)
    else:
        current_val_end = min_train_end + timedelta(weeks=val_weeks)

    while current_val_end <= data_end:
        # Work backwards from val_end to determine windows
        val_end = current_val_end
        val_start = val_end - timedelta(weeks=val_weeks)

        if uses_calibration:
            cal_end = val_start - timedelta(days=1)
            cal_start = cal_end - timedelta(weeks=cal_weeks) + timedelta(days=1)
            train_end = cal_start - timedelta(days=1)
        else:
            cal_start = None
            cal_end = None
            train_end = val_start - timedelta(days=1)

        # Skip if train window is too short
        if train_end < min_train_end:
            current_val_end += timedelta(weeks=step_weeks)
            continue

        # Skip if validation period is in off-season
        if season_aware and (is_offseason(val_start) or is_offseason(val_end)):
            skipped_offseason += 1
            current_val_end += timedelta(weeks=step_weeks)
            continue

        fold = FoldConfig(
            fold_id=f"fold_{fold_num:02d}",
            train_start=train_start,
            train_end=train_end,
            cal_start=cal_start,
            cal_end=cal_end,
            val_start=val_start,
            val_end=val_end,
        )
        folds.append(fold)

        fold_num += 1
        current_val_end += timedelta(weeks=step_weeks)

    if skipped_offseason > 0:
        typer.echo(f"[season-aware] Skipped {skipped_offseason} off-season folds")

    return folds


def run_fold(
    fold: FoldConfig,
    features_path: Path,
    run_id_prefix: str,
    extra_args: list[str],
    model_type: ModelType = ModelType.LGBM,
) -> FoldResult:
    """Run a single training fold."""

    config = MODEL_CONFIG[model_type]
    run_id = f"{run_id_prefix}_{fold.fold_id}"

    cmd = [
        "uv", "run", "python", "-m", config["module"],
        "--features", str(features_path),
        "--run-id", run_id,
        *fold.to_args(model_type),
        *extra_args,
    ]

    typer.echo(f"\n{'='*60}")
    typer.echo(f"[{model_type.value}] Running {fold.fold_id}: Train [{fold.train_start.date()} → {fold.train_end.date()}]")
    if fold.cal_start and fold.cal_end:
        typer.echo(f"  Cal [{fold.cal_start.date()} → {fold.cal_end.date()}]")
    typer.echo(f"  Val [{fold.val_start.date()} → {fold.val_end.date()}]")
    typer.echo(f"{'='*60}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout per fold
        )

        # Parse metrics from artifacts if successful
        metrics = {}
        metrics_path = Path(f"{config['artifact_dir']}/{run_id}/metrics.json")
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)

        if result.returncode == 0:
            typer.echo(f"✓ {fold.fold_id} completed successfully")
            return FoldResult(
                fold_id=fold.fold_id,
                config=fold,
                success=True,
                metrics=metrics,
            )
        else:
            # Check if it's just a coverage failure (still has metrics)
            if metrics:
                typer.echo(f"⚠ {fold.fold_id} completed with warnings (coverage check failed)")
                return FoldResult(
                    fold_id=fold.fold_id,
                    config=fold,
                    success=True,  # Still consider it successful if we got metrics
                    metrics=metrics,
                )
            else:
                typer.echo(f"✗ {fold.fold_id} failed: {result.stderr[-500:]}", err=True)
                return FoldResult(
                    fold_id=fold.fold_id,
                    config=fold,
                    success=False,
                    error=result.stderr[-1000:],
                )

    except subprocess.TimeoutExpired:
        typer.echo(f"✗ {fold.fold_id} timed out", err=True)
        return FoldResult(
            fold_id=fold.fold_id,
            config=fold,
            success=False,
            error="Timeout after 600 seconds",
        )
    except Exception as e:
        typer.echo(f"✗ {fold.fold_id} error: {e}", err=True)
        return FoldResult(
            fold_id=fold.fold_id,
            config=fold,
            success=False,
            error=str(e),
        )


def aggregate_results(
    results: list[FoldResult],
    model_type: ModelType = ModelType.LGBM,
) -> dict[str, Any]:
    """Aggregate metrics across all folds."""

    successful = [r for r in results if r.success and r.metrics]
    if not successful:
        return {"error": "No successful folds"}

    # Get metric keys for this model type
    metric_keys = MODEL_CONFIG[model_type]["metric_keys"]

    aggregated: dict[str, Any] = {
        "n_folds": len(results),
        "n_successful": len(successful),
        "n_failed": len(results) - len(successful),
        "model_type": model_type.value,
    }

    for key in metric_keys:
        values = [r.metrics.get(key) for r in successful if r.metrics.get(key) is not None]
        if values:
            # Handle boolean metrics
            if all(isinstance(v, bool) for v in values):
                aggregated[f"{key}_pass_rate"] = sum(values) / len(values)
            else:
                aggregated[f"{key}_mean"] = sum(values) / len(values)
                aggregated[f"{key}_min"] = min(values)
                aggregated[f"{key}_max"] = max(values)
                aggregated[f"{key}_std"] = (
                    (sum((v - aggregated[f"{key}_mean"])**2 for v in values) / len(values)) ** 0.5
                    if len(values) > 1 else 0.0
                )

    return aggregated


def print_summary(
    results: list[FoldResult],
    aggregated: dict[str, Any],
    model_type: ModelType = ModelType.LGBM,
) -> None:
    """Print a summary of walk-forward results."""

    typer.echo("\n" + "="*70)
    typer.echo(f"WALK-FORWARD CROSS-VALIDATION SUMMARY [{model_type.value}]")
    typer.echo("="*70)

    # Per-fold results - format depends on model type
    typer.echo("\nPer-Fold Results:")
    typer.echo("-"*70)

    if model_type == ModelType.LGBM:
        typer.echo(f"{'Fold':<10} {'MAE':<8} {'P10 Cov':<10} {'P90 Cov':<10} {'Winkler':<10} {'Status':<8}")
        typer.echo("-"*70)
        for r in results:
            if r.success and r.metrics:
                mae = r.metrics.get("val_mae_p50", r.metrics.get("val_mae_overall", 0))
                p10 = r.metrics.get("val_p10_cond_playable", 0)
                p90 = r.metrics.get("val_p90_cond_playable", 0)
                wink = r.metrics.get("val_winkler_cond_playable", 0)
                typer.echo(f"{r.fold_id:<10} {mae:<8.3f} {p10:<10.1%} {p90:<10.1%} {wink:<10.2f} {'✓':<8}")
            else:
                typer.echo(f"{r.fold_id:<10} {'-':<8} {'-':<10} {'-':<10} {'-':<10} {'✗':<8}")
    else:  # MINUTE_SHARE
        typer.echo(f"{'Fold':<10} {'MAE Min':<10} {'DNP AUC':<10} {'Rot Acc':<10} {'RawSum':<10} {'Status':<8}")
        typer.echo("-"*70)
        for r in results:
            if r.success and r.metrics:
                mae_min = r.metrics.get("mae_minutes", 0)
                dnp_auc = r.metrics.get("dnp_auc")
                dnp_auc_str = f"{dnp_auc:.3f}" if dnp_auc is not None else "N/A"
                rot_acc = r.metrics.get("rotation_accuracy", 0)
                raw_sum = r.metrics.get("team_share_sum_mean", 0)
                typer.echo(f"{r.fold_id:<10} {mae_min:<10.3f} {dnp_auc_str:<10} {rot_acc:<10.3f} {raw_sum:<10.3f} {'✓':<8}")
            else:
                typer.echo(f"{r.fold_id:<10} {'-':<10} {'-':<10} {'-':<10} {'-':<10} {'✗':<8}")

    # Aggregated metrics
    typer.echo("\n" + "="*70)
    typer.echo("AGGREGATED METRICS (across all successful folds)")
    typer.echo("="*70)
    typer.echo(f"\nFolds: {aggregated.get('n_successful', 0)}/{aggregated.get('n_folds', 0)} successful")

    if model_type == ModelType.LGBM:
        if "val_mae_p50_mean" in aggregated:
            typer.echo("\nMAE (minutes):")
            typer.echo(f"  Mean: {aggregated['val_mae_p50_mean']:.3f}")
            typer.echo(f"  Range: [{aggregated['val_mae_p50_min']:.3f}, {aggregated['val_mae_p50_max']:.3f}]")
            typer.echo(f"  Std: {aggregated['val_mae_p50_std']:.3f}")

        if "val_p10_cond_playable_mean" in aggregated:
            typer.echo("\nP10 Coverage (target: 10%):")
            typer.echo(f"  Mean: {aggregated['val_p10_cond_playable_mean']:.1%}")
            typer.echo(f"  Range: [{aggregated['val_p10_cond_playable_min']:.1%}, {aggregated['val_p10_cond_playable_max']:.1%}]")

        if "val_p90_cond_playable_mean" in aggregated:
            typer.echo("\nP90 Coverage (target: 90%):")
            typer.echo(f"  Mean: {aggregated['val_p90_cond_playable_mean']:.1%}")
            typer.echo(f"  Range: [{aggregated['val_p90_cond_playable_min']:.1%}, {aggregated['val_p90_cond_playable_max']:.1%}]")

        if "val_winkler_cond_playable_mean" in aggregated:
            typer.echo("\nWinkler Score (lower is better):")
            typer.echo(f"  Mean: {aggregated['val_winkler_cond_playable_mean']:.2f}")
            typer.echo(f"  Range: [{aggregated['val_winkler_cond_playable_min']:.2f}, {aggregated['val_winkler_cond_playable_max']:.2f}]")

    else:  # MINUTE_SHARE
        if "mae_minutes_mean" in aggregated:
            typer.echo("\nMAE (minutes):")
            typer.echo(f"  Mean: {aggregated['mae_minutes_mean']:.3f}")
            typer.echo(f"  Range: [{aggregated['mae_minutes_min']:.3f}, {aggregated['mae_minutes_max']:.3f}]")
            typer.echo(f"  Std: {aggregated['mae_minutes_std']:.3f}")

        if "mae_shares_mean" in aggregated:
            typer.echo("\nMAE (shares):")
            typer.echo(f"  Mean: {aggregated['mae_shares_mean']:.4f}")
            typer.echo(f"  Range: [{aggregated['mae_shares_min']:.4f}, {aggregated['mae_shares_max']:.4f}]")

        # MAE by bucket
        bucket_names = ["dnp", "garbage_time", "bench", "rotation", "core", "stars"]
        typer.echo("\nMAE by Bucket:")
        for bucket in bucket_names:
            key = f"mae_{bucket}_mean"
            if key in aggregated and aggregated[key] is not None:
                typer.echo(f"  {bucket:12s}: {aggregated[key]:.2f}")

        # DNP classification
        if "dnp_auc_mean" in aggregated and aggregated["dnp_auc_mean"] is not None:
            typer.echo("\nDNP Classification:")
            typer.echo(f"  AUC:       {aggregated['dnp_auc_mean']:.3f}")
        if "dnp_accuracy_mean" in aggregated:
            typer.echo(f"  Accuracy:  {aggregated['dnp_accuracy_mean']:.3f}")
        if "dnp_precision_mean" in aggregated and aggregated["dnp_precision_mean"] is not None:
            typer.echo(f"  Precision: {aggregated['dnp_precision_mean']:.3f}")
        if "dnp_recall_mean" in aggregated and aggregated["dnp_recall_mean"] is not None:
            typer.echo(f"  Recall:    {aggregated['dnp_recall_mean']:.3f}")

        # Rotation classification
        if "rotation_accuracy_mean" in aggregated:
            typer.echo("\nRotation Classification (>=15 min):")
            typer.echo(f"  Accuracy:  {aggregated['rotation_accuracy_mean']:.3f}")
        if "rotation_precision_mean" in aggregated and aggregated["rotation_precision_mean"] is not None:
            typer.echo(f"  Precision: {aggregated['rotation_precision_mean']:.3f}")
        if "rotation_recall_mean" in aggregated and aggregated["rotation_recall_mean"] is not None:
            typer.echo(f"  Recall:    {aggregated['rotation_recall_mean']:.3f}")

        # Share distribution
        if "team_share_sum_mean_mean" in aggregated:
            typer.echo("\nShare Distribution:")
            typer.echo(f"  Team sum mean: {aggregated['team_share_sum_mean_mean']:.3f}")
        if "max_share_mean_mean" in aggregated:
            typer.echo(f"  Max share mean: {aggregated['max_share_mean_mean']:.3f}")


@app.command()
def main(
    features: Path = typer.Option(
        ...,
        "--features",
        help="Path to features parquet file.",
    ),
    start_date: datetime = typer.Option(
        ...,
        "--start",
        help="Start date of your dataset (YYYY-MM-DD).",
    ),
    end_date: datetime = typer.Option(
        ...,
        "--end",
        help="End date of your dataset (YYYY-MM-DD).",
    ),
    model_type: ModelType = typer.Option(
        ModelType.LGBM,
        "--model-type",
        help="Model type to train: 'lgbm' (standard) or 'minute_share' (share-based).",
    ),
    train_months: int = typer.Option(
        12,
        "--train-months",
        help="Minimum training window size in months.",
    ),
    cal_weeks: int = typer.Option(
        3,
        "--cal-weeks",
        help="Calibration window size in weeks (ignored for minute_share).",
    ),
    val_weeks: int = typer.Option(
        2,
        "--val-weeks",
        help="Validation window size in weeks.",
    ),
    step_weeks: int = typer.Option(
        4,
        "--step-weeks",
        help="How many weeks to advance between folds.",
    ),
    run_id_prefix: str = typer.Option(
        "walkfwd",
        "--run-id-prefix",
        help="Prefix for run IDs.",
    ),
    output: Path = typer.Option(
        None,
        "--output",
        help="Optional path to save results JSON.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Only show fold configurations without running.",
    ),
    tolerance_relaxed: bool = typer.Option(
        False,
        "--tolerance-relaxed",
        is_flag=True,
        help="Use relaxed coverage tolerance (lgbm only).",
    ),
    allow_guard_failure: bool = typer.Option(
        True,
        "--allow-guard-failure/--strict",
        help="Allow runs to continue even if coverage checks fail (lgbm only).",
    ),
    season_aware: bool = typer.Option(
        True,
        "--season-aware/--all-seasons",
        help="Skip off-season folds (July-September). Default: enabled.",
    ),
) -> None:
    """Run walk-forward cross-validation for the minutes model."""

    if not features.exists():
        typer.echo(f"Features file not found: {features}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"[walk-forward] Model type: {model_type.value}")

    # Generate folds
    folds = generate_folds(
        data_start=start_date,
        data_end=end_date,
        min_train_months=train_months,
        cal_weeks=cal_weeks,
        val_weeks=val_weeks,
        step_weeks=step_weeks,
        season_aware=season_aware,
        model_type=model_type,
    )

    if not folds:
        typer.echo("No valid folds could be generated with the given parameters.", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Generated {len(folds)} walk-forward folds:")
    for fold in folds:
        if fold.cal_start and fold.cal_end:
            typer.echo(f"  {fold.fold_id}: Train [{fold.train_start.date()} → {fold.train_end.date()}] | "
                       f"Cal [{fold.cal_start.date()} → {fold.cal_end.date()}] | "
                       f"Val [{fold.val_start.date()} → {fold.val_end.date()}]")
        else:
            typer.echo(f"  {fold.fold_id}: Train [{fold.train_start.date()} → {fold.train_end.date()}] | "
                       f"Val [{fold.val_start.date()} → {fold.val_end.date()}]")

    if dry_run:
        typer.echo("\n--dry-run specified, not executing folds.")
        return

    # Build extra args (model-type specific)
    extra_args = []
    if model_type == ModelType.LGBM:
        if tolerance_relaxed:
            extra_args.append("--tolerance-relaxed")
        if allow_guard_failure:
            extra_args.append("--allow-guard-failure")

    # Run each fold
    results: list[FoldResult] = []
    for fold in folds:
        result = run_fold(fold, features, run_id_prefix, extra_args, model_type)
        results.append(result)

    # Aggregate and print summary
    aggregated = aggregate_results(results, model_type)
    print_summary(results, aggregated, model_type)

    # Save results if requested
    if output:
        output_data = {
            "model_type": model_type.value,
            "folds": [
                {
                    "config": r.config.to_dict(),
                    "success": r.success,
                    "metrics": r.metrics,
                    "error": r.error,
                }
                for r in results
            ],
            "aggregated": aggregated,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        typer.echo(f"\nResults saved to {output}")

    # Log aggregated results to MLFlow
    _log_walk_forward_to_mlflow(
        run_id_prefix=run_id_prefix,
        results=results,
        aggregated=aggregated,
        params={
            "model_type": model_type.value,
            "features_path": str(features),
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "train_months": train_months,
            "cal_weeks": cal_weeks,
            "val_weeks": val_weeks,
            "step_weeks": step_weeks,
            "season_aware": season_aware,
            "tolerance_relaxed": tolerance_relaxed,
        },
        output_path=output,
    )

    # Exit with error if no successful folds
    if aggregated.get("n_successful", 0) == 0:
        raise typer.Exit(code=1)


def _log_walk_forward_to_mlflow(
    *,
    run_id_prefix: str,
    results: list[FoldResult],
    aggregated: dict[str, Any],
    params: dict[str, Any],
    output_path: Path | None,
) -> None:
    """Log walk-forward CV results to MLFlow as a parent run with nested child runs."""
    if not MLFLOW_AVAILABLE:
        typer.echo("[mlflow] MLFlow not available, skipping aggregated logging", err=True)
        return

    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

        run_name = f"{run_id_prefix}_summary"
        
        with mlflow.start_run(run_name=run_name) as parent_run:
            # Log parameters
            mlflow.log_params(params)
            mlflow.log_param("n_folds", len(results))
            
            # Log aggregated metrics
            for key, value in aggregated.items():
                if isinstance(value, (int, float)) and value is not None:
                    mlflow.log_metric(f"agg_{key}", float(value))

            # Log per-fold metrics as a table
            fold_data = []
            for r in results:
                fold_row = {
                    "fold_id": r.fold_id,
                    "success": r.success,
                    "val_start": r.config.val_start.strftime("%Y-%m-%d"),
                    "val_end": r.config.val_end.strftime("%Y-%m-%d"),
                }
                if r.metrics:
                    fold_row.update({
                        "mae": r.metrics.get("val_mae_overall"),
                        "p10_cov": r.metrics.get("val_p10_cond_playable"),
                        "p90_cov": r.metrics.get("val_p90_cond_playable"),
                        "winkler": r.metrics.get("val_winkler_cond_playable"),
                        "val_rows": r.metrics.get("val_rows"),
                    })
                fold_data.append(fold_row)
            
            # Save fold table as artifact
            fold_df = pd.DataFrame(fold_data)
            fold_table_path = Path(f"/tmp/{run_id_prefix}_fold_results.csv")
            fold_df.to_csv(fold_table_path, index=False)
            mlflow.log_artifact(str(fold_table_path))
            
            # Log individual fold metrics as nested runs
            for r in results:
                if r.success and r.metrics:
                    with mlflow.start_run(run_name=r.fold_id, nested=True):
                        mlflow.log_param("fold_id", r.fold_id)
                        mlflow.log_param("val_start", r.config.val_start.strftime("%Y-%m-%d"))
                        mlflow.log_param("val_end", r.config.val_end.strftime("%Y-%m-%d"))
                        mlflow.log_param("train_start", r.config.train_start.strftime("%Y-%m-%d"))
                        mlflow.log_param("train_end", r.config.train_end.strftime("%Y-%m-%d"))
                        
                        for key, value in r.metrics.items():
                            if isinstance(value, (int, float)) and value is not None:
                                mlflow.log_metric(key, float(value))
            
            # Log the full results JSON if available
            if output_path and output_path.exists():
                mlflow.log_artifact(str(output_path))

            typer.echo(f"\n[mlflow] Logged walk-forward run '{run_name}' to experiment '{MLFLOW_EXPERIMENT_NAME}'")
            typer.echo(f"[mlflow] Parent run ID: {parent_run.info.run_id}")

    except Exception as e:
        typer.echo(f"[mlflow] Warning: Failed to log to MLFlow: {e}", err=True)


if __name__ == "__main__":
    app()
