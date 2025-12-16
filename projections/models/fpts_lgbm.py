"""LightGBM regression pipeline for fantasy points per minute."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import typer
from sklearn.impute import SimpleImputer

from projections import paths
from projections.fpts_v1.datasets import FptsDatasetBuilder, MinutesSource
from projections.fpts_v1.eval import (
    DEFAULT_BASELINE_PRIOR,
    baseline_per_minute,
    evaluate_model_vs_baseline,
)
from projections.models.minutes_features import infer_feature_columns
from projections.registry.manifest import (
    load_manifest,
    save_manifest,
    register_model,
)

app = typer.Typer(help=__doc__)
BASELINE_PRIOR_COLUMN = DEFAULT_BASELINE_PRIOR
_LEAKY_COLUMN_NAMES = {
    "actual_minutes",
    "actual_fpts",
    "fpts_per_min_actual",
    "fpts_per_min_game",
    "fpts_baseline",
    "fpts_baseline_per_min",
}
_LEAKY_SUBSTRINGS = ("actual_", "_game", "baseline", "label")


def _is_leaky_column(name: str) -> bool:
    lower = name.lower()
    if name in _LEAKY_COLUMN_NAMES:
        return True
    return any(token in lower for token in _LEAKY_SUBSTRINGS)


def _assert_no_leaky_features(columns: Sequence[str]) -> None:
    for col in columns:
        if _is_leaky_column(col):
            raise AssertionError(f"Leaky feature detected in training columns: {col}")


def _infer_safe_feature_columns(
    frame: pd.DataFrame,
    *,
    target_col: str,
    extra_excluded: Iterable[str] | None = None,
) -> list[str]:
    excluded = set(extra_excluded or [])
    excluded.add(target_col)
    safe: list[str] = []
    for col in frame.columns:
        if col in excluded:
            continue
        if _is_leaky_column(col):
            continue
        safe.append(col)
    if not safe:
        raise ValueError("Feature selection removed every column; adjust safe feature rules.")
    _assert_no_leaky_features(safe)
    return safe


def _normalize_date(value: datetime | str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC")
    return ts.tz_localize(None).normalize()


@dataclass(frozen=True)
class DateWindow:
    start: pd.Timestamp
    end: pd.Timestamp

    def slice(self, df: pd.DataFrame) -> pd.DataFrame:
        mask = (df["game_date"] >= self.start) & (df["game_date"] <= self.end)
        return df.loc[mask].copy()

    def to_metadata(self) -> dict[str, str]:
        return {"start": self.start.date().isoformat(), "end": self.end.date().isoformat()}


def _window(
    start: datetime,
    end: datetime,
) -> DateWindow:
    start_day = _normalize_date(start)
    end_day = _normalize_date(end)
    if end_day < start_day:
        raise ValueError("Window end must be on/after start date.")
    return DateWindow(start=start_day, end=end_day)


def _ensure_run_dir(run_id: str, artifact_root: Path) -> Path:
    run_dir = (artifact_root / run_id).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


@app.command()
def main(
    run_id: str = typer.Option(..., "--run-id", help="Unique identifier for the training run."),
    data_root: Path = typer.Option(
        paths.get_data_root(), help="Root directory containing bronze/silver/gold partitions."
    ),
    artifact_root: Path = typer.Option(Path("artifacts/fpts_lgbm"), help="Where to store model artifacts."),
    train_start: datetime = typer.Option(..., help="Training window start date (YYYY-MM-DD)."),
    train_end: datetime = typer.Option(..., help="Training window end date (YYYY-MM-DD)."),
    cal_start: datetime | None = typer.Option(
        None,
        help="Calibration window start (defaults to the day after --train-end).",
    ),
    cal_end: datetime = typer.Option(..., help="Calibration window end date."),
    val_start: datetime | None = typer.Option(
        None,
        help="Optional validation window start (omit for train/cal only).",
    ),
    val_end: datetime | None = typer.Option(
        None,
        help="Optional validation window end date (required if --val-start is set).",
    ),
    scoring_system: str = typer.Option(
        "dk", "--scoring-system", help="Fantasy scoring system (currently dk)."
    ),
    minutes_source: str = typer.Option(
        "predicted",
        "--minutes-source",
        help="predicted (default) loads minutes_v2 backfills; actual uses realized minutes (optimistic).",
    ),
    minutes_run_id: str | None = typer.Option(
        None,
        "--minutes-run-id",
        help="Minutes prediction log run identifier (defaults to production minutes run).",
    ),
    learning_rate: float = typer.Option(0.05, help="LightGBM learning rate."),
    num_leaves: int = typer.Option(63, help="LightGBM num_leaves."),
    n_estimators: int = typer.Option(600, help="LightGBM n_estimators."),
    subsample: float = typer.Option(0.85, help="LightGBM subsample."),
    colsample_bytree: float = typer.Option(0.85, help="LightGBM colsample_bytree."),
    random_state: int = typer.Option(42, help="Random seed."),
) -> None:
    if cal_start is None:
        cal_start = _normalize_date(train_end + timedelta(days=1))
    if val_start is None and val_end is not None:
        raise typer.BadParameter("--val-start is required when --val-end is provided.")
    if val_start is not None and val_end is None:
        raise typer.BadParameter("--val-end is required when --val-start is provided.")

    train_window = _window(train_start, train_end)
    cal_window = _window(cal_start, cal_end)
    val_window = _window(val_start, val_end) if val_start and val_end else None
    minutes_key = minutes_source.strip().lower()
    if minutes_key not in ("predicted", "actual"):
        raise typer.BadParameter("--minutes-source must be 'predicted' or 'actual'.")

    global_start = min(
        train_window.start,
        cal_window.start,
        val_window.start if val_window else cal_window.start,
    )
    window_end_candidates = [train_window.end, cal_window.end]
    if val_window:
        window_end_candidates.append(val_window.end)
    global_end = max(window_end_candidates)

    typer.echo(
        f"[fpts] building dataset from {global_start.date()} to {global_end.date()} "
        f"(minutes_source={minutes_key}, scoring={scoring_system})"
    )
    if minutes_key == "actual":
        typer.echo(
            "[fpts] WARNING: minutes_source=actual leaks realized minutes; use only for experiments.",
            err=True,
        )
    minutes_enum: MinutesSource = "predicted" if minutes_key == "predicted" else "actual"

    builder = FptsDatasetBuilder(
        data_root=data_root,
        scoring_system=scoring_system,
        minutes_source=minutes_enum,
        minutes_run_id=minutes_run_id,
    )
    dataset = builder.build(global_start, global_end)
    if dataset.empty:
        raise RuntimeError("Dataset is empty after filtering requested windows.")
    dataset["game_date"] = pd.to_datetime(dataset["game_date"]).dt.normalize()
    coverage_start = dataset["game_date"].min()
    coverage_end = dataset["game_date"].max()
    coverage_msg = (
        f"Available game_date range: {coverage_start.date().isoformat()} → "
        f"{coverage_end.date().isoformat()} ({len(dataset)} rows)."
    )

    splits: dict[str, pd.DataFrame] = {
        "train": train_window.slice(dataset),
        "cal": cal_window.slice(dataset),
    }
    if val_window is not None:
        splits["val"] = val_window.slice(dataset)
    split_windows: dict[str, DateWindow] = {
        "train": train_window,
        "cal": cal_window,
    }
    if val_window is not None:
        split_windows["val"] = val_window
    for name, frame in splits.items():
        if frame.empty:
            window = split_windows[name]
            raise RuntimeError(
                f"{name} split is empty. Requested window: "
                f"{window.start.date().isoformat()} → {window.end.date().isoformat()}. "
                f"{coverage_msg} Ensure PROJECTIONS_DATA_ROOT contains boxscores/features "
                "for the requested calibration/validation range."
            )

    target_col = "fpts_per_min_actual"
    excluded = {"actual_minutes", "actual_fpts", "fpts_baseline_per_min"}
    inferred_columns = infer_feature_columns(
        splits["train"], target_col=target_col, excluded=excluded
    )
    feature_columns = [
        col for col in inferred_columns if not _is_leaky_column(col)
    ]
    if not feature_columns:
        feature_columns = _infer_safe_feature_columns(
            splits["train"], target_col=target_col, extra_excluded=excluded
        )
    _assert_no_leaky_features(feature_columns)
    typer.echo(f"[fpts] training LightGBM on {len(feature_columns)} safe features.")

    imputer = SimpleImputer(strategy="median")
    X_train = splits["train"][feature_columns]
    y_train = splits["train"][target_col]
    imputer.fit(X_train)
    X_train_imputed = imputer.transform(X_train)
    model = lgb.LGBMRegressor(
        objective="regression_l1",
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        num_leaves=num_leaves,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
    )
    model.fit(X_train_imputed, y_train)

    split_metrics: dict[str, Any] = {}
    for name, frame in splits.items():
        matrix = imputer.transform(frame[feature_columns])
        preds = model.predict(matrix)
        baseline_preds = baseline_per_minute(frame, prior_col=BASELINE_PRIOR_COLUMN)
        split_metrics[name] = evaluate_model_vs_baseline(
            frame,
            model_preds=preds,
            baseline_preds=baseline_preds,
        )

    run_dir = _ensure_run_dir(run_id, artifact_root)
    joblib.dump(
        {
            "model": model,
            "imputer": imputer,
            "feature_columns": feature_columns,
            "metadata": {
                "run_id": run_id,
                "scoring_system": scoring_system,
                "minutes_source": minutes_key,
                "random_state": random_state,
            },
        },
        run_dir / "model.joblib",
    )

    config = {
        "run_id": run_id,
        "data_root": str(data_root),
        "artifact_root": str(run_dir),
        "minutes_source": minutes_key,
        "scoring_system": scoring_system,
        "learning_rate": learning_rate,
        "num_leaves": num_leaves,
        "n_estimators": n_estimators,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "random_state": random_state,
        "train_window": train_window.to_metadata(),
        "cal_window": cal_window.to_metadata(),
        "val_window": val_window.to_metadata() if val_window else None,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(split_metrics, indent=2), encoding="utf-8")

    report_lines = [
        f"# FPTS per Minute LightGBM — {run_id}",
        "",
        "## Model vs Baseline",
        "",
        (
            "| Split | Rows | Baseline MAE/min | Model MAE/min | Δ MAE/min | "
            "Baseline MAE FPTS | Model MAE FPTS | Δ MAE FPTS |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    ordered_splits = [name for name in ("train", "cal", "val") if name in split_metrics]
    for name in ordered_splits:
        metrics = split_metrics[name]
        report_lines.append(
            f"| {name} | {metrics['rows']:,} | "
            f"{metrics['baseline_mae_per_min']:.4f} | {metrics['model_mae_per_min']:.4f} | "
            f"{metrics['delta_mae_per_min']:.4f} | "
            f"{metrics['baseline_mae_fpts']:.3f} | {metrics['model_mae_fpts']:.3f} | "
            f"{metrics['delta_mae_fpts']:.3f} |"
        )
    anchor_split = "val" if "val" in split_metrics else "cal"
    anchor_metrics = split_metrics[anchor_split]
    delta_fpts = anchor_metrics["delta_mae_fpts"]
    direction = "beats" if delta_fpts > 0 else "trails"
    report_lines.append("")
    report_lines.append(
        f"Model {direction} the baseline on {anchor_split} MAE by "
        f"{abs(delta_fpts):.3f} {scoring_system.upper()} points."
    )
    if minutes_key == "actual":
        report_lines.append(
            "**Warning:** `minutes_source=\"actual\"` leaks realized minutes and should"
            " only be used for experiments."
        )
    report_lines.append("")
    report_lines.append("## Bucket Summaries")
    for name in ordered_splits:
        buckets = split_metrics[name].get("buckets") or {}
        if not buckets:
            continue
        report_lines.append(f"### {name.title()}")
        report_lines.append("")
        report_lines.append(
            "| Bucket | Rows | Baseline MAE/min | Model MAE/min | Δ MAE/min | "
            "Baseline MAE FPTS | Model MAE FPTS | Δ MAE FPTS |"
        )
        report_lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for bucket, payload in buckets.items():
            report_lines.append(
                f"| {bucket} | {payload['rows']:,} | "
                f"{payload['baseline_mae_per_min']:.4f} | {payload['model_mae_per_min']:.4f} | "
                f"{payload['delta_mae_per_min']:.4f} | "
                f"{payload['baseline_mae_fpts']:.3f} | {payload['model_mae_fpts']:.3f} | "
                f"{payload['delta_mae_fpts']:.3f} |"
            )
        report_lines.append("")
    (run_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")

    val_metrics = split_metrics.get("val", split_metrics["cal"])
    typer.echo(
        f"[fpts] training complete. Validation MAE/min="
        f"{val_metrics['model_mae_per_min']:.4f} "
        f"(baseline {val_metrics['baseline_mae_per_min']:.4f}). "
        f"Artifacts: {run_dir}"
    )

    # Auto-register model in registry
    try:
        manifest = load_manifest()
        windows_meta = {
            "train": train_window.to_metadata(),
            "cal": cal_window.to_metadata(),
        }
        register_model(
            manifest,
            model_name="fpts_v1_lgbm",
            version=run_id,
            run_id=run_id,
            artifact_path=str(run_dir),
            training_start=windows_meta["train"]["start"],
            training_end=windows_meta["train"]["end"],
            feature_schema_version="v1",
            metrics={
                "val_mae_per_min": val_metrics["model_mae_per_min"],
                "val_mae_fpts": val_metrics["model_mae_fpts"],
                "baseline_mae_per_min": val_metrics["baseline_mae_per_min"],
                "delta_mae_fpts": val_metrics["delta_mae_fpts"],
            },
            description=f"Train {windows_meta['train']['start']} to {windows_meta['train']['end']} | {scoring_system}",
        )
        save_manifest(manifest)
        typer.echo(f"[registry] Registered fpts_v1_lgbm v{run_id} (stage=dev)")
    except Exception as e:
        typer.echo(f"[registry] Warning: Failed to register model: {e}", err=True)


if __name__ == "__main__":  # pragma: no cover
    app()
