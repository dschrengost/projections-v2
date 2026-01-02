#!/usr/bin/env python
"""Walk-forward cross-validation for rates_v1 models.

This script repeatedly trains `scripts.rates.train_rates_v1` on progressively
advancing time windows (expanding training set) and aggregates per-target
metrics across folds.

Usage:
    uv run python scripts/walk_forward_rates.py \
        --start 2023-10-01 \
        --end 2025-12-01 \
        --feature-set stage4_recency \
        --train-months 6 \
        --cal-weeks 3 \
        --val-weeks 2 \
        --step-weeks 4 \
        --season-aware
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import typer

from projections.paths import data_path

NBA_OFFSEASON_MONTHS = {7, 8, 9}

app = typer.Typer(add_completion=False, help=__doc__)


def _overlaps_offseason(start: datetime, end: datetime) -> bool:
    day = start.date()
    end_day = end.date()
    while day < end_day:
        if day.month in NBA_OFFSEASON_MONTHS:
            return True
        day += timedelta(days=1)
    return False


def _available_game_dates(root: Path, start: datetime, end: datetime) -> pd.DatetimeIndex:
    base = root / "gold" / "rates_training_base"
    if not base.exists():
        raise FileNotFoundError(f"Missing rates_training_base root at {base}")

    days: set[pd.Timestamp] = set()
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    for season_dir in base.glob("season=*"):
        for day_dir in season_dir.glob("game_date=*"):
            try:
                day = pd.Timestamp(day_dir.name.split("=", 1)[1]).normalize()
            except ValueError:
                continue
            if day < start_ts or day > end_ts:
                continue
            candidate = day_dir / "rates_training_base.parquet"
            if candidate.exists():
                days.add(day)

    if not days:
        raise FileNotFoundError(
            f"No rates_training_base partitions matched [{start_ts.date()}..{end_ts.date()}]."
        )
    return pd.DatetimeIndex(sorted(days))


def _snap_up(dates: pd.DatetimeIndex, dt: datetime) -> datetime | None:
    idx = int(dates.searchsorted(pd.Timestamp(dt).normalize(), side="left"))
    if idx >= len(dates):
        return None
    return pd.Timestamp(dates[idx]).to_pydatetime()


def _snap_down(dates: pd.DatetimeIndex, dt: datetime) -> datetime | None:
    idx = int(dates.searchsorted(pd.Timestamp(dt).normalize(), side="right")) - 1
    if idx < 0:
        return None
    return pd.Timestamp(dates[idx]).to_pydatetime()


@dataclass(frozen=True)
class FoldConfig:
    fold_id: str
    train_start: datetime
    train_end: datetime
    cal_end: datetime
    val_end: datetime

    @property
    def cal_start(self) -> datetime:
        return self.train_end

    @property
    def val_start(self) -> datetime:
        return self.cal_end

    def to_train_args(self) -> list[str]:
        return [
            "--start-date",
            self.train_start.strftime("%Y-%m-%d"),
            "--end-date",
            self.val_end.strftime("%Y-%m-%d"),
            "--train-end-date",
            self.train_end.strftime("%Y-%m-%d"),
            "--cal-end-date",
            self.cal_end.strftime("%Y-%m-%d"),
        ]

    def to_dict(self) -> dict[str, str]:
        return {
            "fold_id": self.fold_id,
            "train_start": self.train_start.date().isoformat(),
            "train_end": self.train_end.date().isoformat(),
            "cal_start": self.cal_start.date().isoformat(),
            "cal_end": self.cal_end.date().isoformat(),
            "val_start": self.val_start.date().isoformat(),
            "val_end": self.val_end.date().isoformat(),
        }


@dataclass
class FoldResult:
    fold_id: str
    config: FoldConfig
    run_id: str
    success: bool
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


def generate_folds(
    *,
    data_start: datetime,
    data_end: datetime,
    available_dates: pd.DatetimeIndex,
    min_train_months: int,
    cal_weeks: int,
    val_weeks: int,
    step_weeks: int,
    season_aware: bool,
) -> list[FoldConfig]:
    folds: list[FoldConfig] = []
    train_start = data_start
    step = timedelta(weeks=step_weeks)
    cal_span = timedelta(weeks=cal_weeks)
    val_span = timedelta(weeks=val_weeks)

    fold_num = 1
    train_end = train_start + timedelta(days=min_train_months * 30)
    while True:
        snapped_train_end = _snap_up(available_dates, train_end)
        if snapped_train_end is None or snapped_train_end > data_end:
            break

        cal_end_candidate = snapped_train_end + cal_span
        snapped_cal_end = _snap_down(available_dates, cal_end_candidate)
        if snapped_cal_end is None:
            break
        if snapped_cal_end < snapped_train_end:
            snapped_cal_end = snapped_train_end

        val_end_candidate = snapped_cal_end + val_span - timedelta(days=1)
        snapped_val_end = _snap_down(available_dates, val_end_candidate)
        if snapped_val_end is None or snapped_val_end > data_end:
            break

        if snapped_val_end < snapped_cal_end:
            train_end = snapped_train_end + step
            continue

        if season_aware and _overlaps_offseason(snapped_cal_end, snapped_val_end + timedelta(days=1)):
            train_end = snapped_train_end + step
            continue

        fold_id = f"fold_{fold_num:03d}_{snapped_val_end.strftime('%Y%m%d')}"
        folds.append(
            FoldConfig(
                fold_id=fold_id,
                train_start=train_start,
                train_end=snapped_train_end,
                cal_end=snapped_cal_end,
                val_end=snapped_val_end,
            )
        )
        fold_num += 1
        train_end = snapped_train_end + step

    if not folds:
        raise typer.BadParameter("No folds generated; widen the date window or reduce train/cal/val sizes.")
    return folds


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _weighted_mae(items: list[tuple[float, int]]) -> float | None:
    numer = 0.0
    denom = 0
    for mae, n in items:
        numer += mae * n
        denom += n
    if denom <= 0:
        return None
    return float(numer / denom)


def _weighted_rmse(items: list[tuple[float, int]]) -> float | None:
    numer = 0.0
    denom = 0
    for rmse, n in items:
        numer += (rmse**2) * n
        denom += n
    if denom <= 0:
        return None
    return float((numer / denom) ** 0.5)


def _aggregate_metrics(fold_results: list[FoldResult]) -> dict[str, Any]:
    # train_rates_v1 writes metrics as metrics[target] = {val_mae, val_rmse, val_n, ...}
    per_target: dict[str, dict[str, Any]] = {}
    for fr in fold_results:
        if not fr.success:
            continue
        for target, m in fr.metrics.items():
            if not isinstance(m, dict):
                continue
            val_mae = m.get("val_mae")
            val_rmse = m.get("val_rmse")
            val_n = m.get("val_n")
            if val_mae is None or val_rmse is None or val_n in (None, 0):
                continue
            bucket = per_target.setdefault(target, {"val_mae": [], "val_rmse": []})
            bucket["val_mae"].append((float(val_mae), int(val_n)))
            bucket["val_rmse"].append((float(val_rmse), int(val_n)))

    summary_targets: dict[str, dict[str, Any]] = {}
    for target, buckets in sorted(per_target.items()):
        mae = _weighted_mae(buckets["val_mae"])
        rmse = _weighted_rmse(buckets["val_rmse"])
        n_total = int(sum(n for _, n in buckets["val_mae"]))
        summary_targets[target] = {"val_mae": mae, "val_rmse": rmse, "val_n": n_total}

    overall_maes = [v["val_mae"] for v in summary_targets.values() if v["val_mae"] is not None]
    overall_rmses = [v["val_rmse"] for v in summary_targets.values() if v["val_rmse"] is not None]

    return {
        "n_folds_success": sum(1 for fr in fold_results if fr.success),
        "n_folds_total": len(fold_results),
        "overall_val_mae_mean": float(sum(overall_maes) / len(overall_maes)) if overall_maes else None,
        "overall_val_rmse_mean": float(sum(overall_rmses) / len(overall_rmses)) if overall_rmses else None,
        "targets": summary_targets,
    }


def _run_fold(
    fold: FoldConfig,
    *,
    data_root: Path,
    output_root: Path,
    run_tag: str,
    feature_set: str,
    allow_minutes_actual_fallback: bool,
) -> FoldResult:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_id = f"{run_tag}_{feature_set}_{fold.fold_id}_{timestamp}"
    run_dir = output_root / run_id
    cmd = [
        sys.executable,
        "-m",
        "scripts.rates.train_rates_v1",
        *fold.to_train_args(),
        "--data-root",
        str(data_root),
        "--output-root",
        str(output_root),
        "--run-id",
        run_id,
        "--run-tag",
        run_tag,
        "--feature-set",
        feature_set,
        "--no-register",
    ]
    if not allow_minutes_actual_fallback:
        cmd.append("--no-minutes-actual-fallback")

    typer.echo(f"[rates-walk] {fold.fold_id} train_end={fold.train_end.date()} val_end={fold.val_end.date()}")
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        return FoldResult(
            fold_id=fold.fold_id,
            config=fold,
            run_id=run_id,
            success=False,
            metrics={},
            error=f"train_rates_v1 exited with code {proc.returncode}",
        )

    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return FoldResult(
            fold_id=fold.fold_id,
            config=fold,
            run_id=run_id,
            success=False,
            metrics={},
            error=f"missing metrics.json at {metrics_path}",
        )
    return FoldResult(
        fold_id=fold.fold_id,
        config=fold,
        run_id=run_id,
        success=True,
        metrics=_load_json(metrics_path),
        error=None,
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@app.command()
def main(
    start: str = typer.Option("2023-10-01", help="Dataset start date YYYY-MM-DD."),
    end: str = typer.Option(..., help="Dataset end date YYYY-MM-DD."),
    data_root: Path | None = typer.Option(None, help="Root containing gold/rates_training_base."),
    output_dir: Path | None = typer.Option(
        None,
        help="Directory to write fold outputs and summary (defaults under <data_root>/artifacts).",
    ),
    feature_set: str = typer.Option(
        "stage4_recency",
        help="rates_v1 feature set key (stage0/stage1/stage2_tracking/stage3_context/stage4_recency).",
        case_sensitive=False,
    ),
    run_tag: str = typer.Option("rates_v1_walk_forward", help="Prefix for per-fold run_id."),
    train_months: int = typer.Option(6, min=1, help="Minimum training window size in months (~30d)."),
    cal_weeks: int = typer.Option(3, min=0, help="Calibration window size in weeks."),
    val_weeks: int = typer.Option(2, min=1, help="Validation window size in weeks."),
    step_weeks: int = typer.Option(4, min=1, help="Weeks to advance the cutoff between folds."),
    season_aware: bool = typer.Option(True, "--season-aware/--no-season-aware", help="Skip folds with offseason validation."),
    allow_minutes_actual_fallback: bool = typer.Option(
        True,
        "--allow-minutes-actual-fallback/--no-minutes-actual-fallback",
        help="Allow fallback to minutes_actual when minutes_pred_* are missing (leaky).",
    ),
) -> None:
    root = data_root or data_path()
    start_dt = pd.Timestamp(start).to_pydatetime()
    end_dt = pd.Timestamp(end).to_pydatetime()
    if end_dt <= start_dt:
        raise typer.BadParameter("--end must be after --start")

    base_out = output_dir or (root / "artifacts" / "rates_v1" / "walk_forward" / datetime.utcnow().strftime("%Y%m%d_%H%M%S"))
    base_out.mkdir(parents=True, exist_ok=True)

    available_dates = _available_game_dates(root, start_dt, end_dt)
    folds = generate_folds(
        data_start=start_dt,
        data_end=end_dt,
        available_dates=available_dates,
        min_train_months=train_months,
        cal_weeks=cal_weeks,
        val_weeks=val_weeks,
        step_weeks=step_weeks,
        season_aware=season_aware,
    )

    fold_results: list[FoldResult] = []
    for fold in folds:
        fold_results.append(
            _run_fold(
                fold,
                data_root=root,
                output_root=base_out,
                run_tag=run_tag,
                feature_set=feature_set.lower(),
                allow_minutes_actual_fallback=allow_minutes_actual_fallback,
            )
        )

    summary = _aggregate_metrics(fold_results)
    payload = {
        "run_tag": run_tag,
        "feature_set": feature_set.lower(),
        "data_start": start,
        "data_end": end,
        "windows": {
            "train_months": train_months,
            "cal_weeks": cal_weeks,
            "val_weeks": val_weeks,
            "step_weeks": step_weeks,
            "season_aware": season_aware,
        },
        "folds": [
            {
                **fr.config.to_dict(),
                "run_id": fr.run_id,
                "success": fr.success,
                "error": fr.error,
            }
            for fr in fold_results
        ],
        "summary": summary,
    }
    _write_json(base_out / "walk_forward_summary.json", payload)
    typer.echo(f"[rates-walk] wrote {base_out / 'walk_forward_summary.json'}")
    typer.echo(f"[rates-walk] summary overall_val_mae_mean={summary['overall_val_mae_mean']}")


if __name__ == "__main__":
    app()
