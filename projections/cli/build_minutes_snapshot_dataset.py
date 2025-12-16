"""Build a snapshot-rich Minutes V1 training parquet from live features + gold labels.

The dual minutes trainer (`projections/cli/train_minutes_dual.py`) expects a single
parquet with:
  - many snapshots per `(game_id, player_id, team_id)` via `feature_as_of_ts`
  - a populated `minutes` target (gold labels)

This CLI discovers live `features.parquet` runs under:
  `<data_root>/live/features_minutes_v1/<YYYY-MM-DD>/run=*/features.parquet`
joins gold labels from:
  `<data_root>/gold/labels_minutes_v1/season=*/game_date=*/labels.parquet`
and writes a labeled snapshot table for downstream horizonization + training.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import typer

from projections import paths
from projections.minutes_v1.training_snapshots import (
    discover_label_paths,
    discover_live_feature_paths,
    filter_to_labeled_rows,
    load_live_feature_snapshots,
    load_parquet_files,
    merge_features_with_labels,
    normalize_snapshot_timestamps,
)

UTC = timezone.utc

app = typer.Typer(help=__doc__)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _run_stamp() -> str:
    return datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")


def _normalize_day(value: datetime) -> date:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.normalize().date()


def _format_click_datetime(value: datetime) -> str:
    """Format a datetime in a Click-compatible string (no timezone suffix)."""

    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.strftime("%Y-%m-%dT%H:%M:%S")


def _build_snapshot_dataset(
    *,
    start_date: date,
    end_date: date,
    out_dir: Path,
    live_features_root: Path,
    labels_root: Path,
    max_runs_per_day: int | None,
    keep_post_tip: bool,
) -> Path:
    out_dir = out_dir.expanduser().resolve()
    live_features_root = live_features_root.expanduser().resolve()
    labels_root = labels_root.expanduser().resolve()

    feature_paths = discover_live_feature_paths(
        live_features_root,
        start_date=start_date,
        end_date=end_date,
        max_runs_per_day=max_runs_per_day,
    )
    if not feature_paths:
        raise FileNotFoundError(
            f"No live minutes feature files found under {live_features_root} for {start_date}..{end_date}."
        )

    typer.echo(f"[snapshot-dataset] loading {len(feature_paths)} live feature files…")
    features = load_live_feature_snapshots(feature_paths)
    if features.empty:
        raise RuntimeError("Loaded zero live feature rows; aborting.")

    features = normalize_snapshot_timestamps(features)
    if not keep_post_tip and {"tip_ts", "feature_as_of_ts"}.issubset(features.columns):
        tip_ts = pd.to_datetime(features["tip_ts"], utc=True, errors="coerce")
        as_of_ts = pd.to_datetime(features["feature_as_of_ts"], utc=True, errors="coerce")
        mask = tip_ts.notna() & as_of_ts.notna() & (as_of_ts <= tip_ts)
        features = features.loc[mask].copy()

    label_discovery = discover_label_paths(labels_root, start_date=start_date, end_date=end_date)
    if not label_discovery.paths:
        raise FileNotFoundError(
            f"No gold label files found under {labels_root} for {start_date}..{end_date}."
        )
    if label_discovery.missing_days:
        typer.echo(
            f"[snapshot-dataset] warning: missing gold labels for {len(label_discovery.missing_days)} day(s): "
            + ", ".join(label_discovery.missing_days[:10])
            + ("…" if len(label_discovery.missing_days) > 10 else ""),
            err=True,
        )

    labels = load_parquet_files(label_discovery.paths)
    if labels.empty:
        raise RuntimeError("Loaded zero label rows; aborting.")

    merged = merge_features_with_labels(features, labels)
    labeled = filter_to_labeled_rows(merged)
    if labeled.empty:
        raise RuntimeError("After joining labels, zero labeled rows remain. Check date window and label availability.")

    run_dir = out_dir / f"run={_run_stamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "features.parquet"
    labeled.to_parquet(out_path, index=False)

    key_cols = ["game_id", "player_id", "team_id"]
    snapshot_stats: dict[str, Any] = {}
    if {"feature_as_of_ts", *key_cols}.issubset(labeled.columns):
        per_key = labeled.groupby(key_cols)["feature_as_of_ts"].nunique(dropna=False)
        snapshot_stats = {
            "keys": int(len(per_key)),
            "mean": float(per_key.mean()),
            "p50": float(per_key.quantile(0.50)),
            "p90": float(per_key.quantile(0.90)),
            "max": int(per_key.max()),
        }

    manifest = {
        "created_at": datetime.now(tz=UTC).isoformat(),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "live_features_root": str(live_features_root),
        "labels_root": str(labels_root),
        "max_runs_per_day": max_runs_per_day,
        "keep_post_tip": keep_post_tip,
        "counts": {
            "feature_files": len(feature_paths),
            "feature_rows": int(len(features)),
            "label_files": len(label_discovery.paths),
            "label_rows": int(len(labels)),
            "labeled_rows": int(len(labeled)),
            "games": int(labeled["game_id"].nunique()) if "game_id" in labeled.columns else None,
        },
        "snapshot_stats": snapshot_stats,
        "missing_label_days": label_discovery.missing_days,
        "paths": {"features": str(out_path)},
    }
    _write_json(run_dir / "manifest.json", manifest)
    typer.echo(f"[snapshot-dataset] wrote {len(labeled):,} labeled snapshot rows -> {out_path}")
    return out_path


@app.command("build")
def build_cmd(
    start_date: datetime = typer.Option(..., "--start-date", help="Start date (inclusive, YYYY-MM-DD)."),
    end_date: datetime = typer.Option(..., "--end-date", help="End date (inclusive, YYYY-MM-DD)."),
    out_dir: Path = typer.Option(
        paths.data_path("training", "snapshots_minutes_v1"),
        "--out-dir",
        help="Directory to write the labeled snapshot parquet (run=<ts>/features.parquet).",
    ),
    live_features_root: Path = typer.Option(
        paths.data_path("live", "features_minutes_v1"),
        "--live-features-root",
        help="Root directory of live minutes feature runs.",
    ),
    labels_root: Path = typer.Option(
        paths.data_path("gold", "labels_minutes_v1"),
        "--labels-root",
        help="Root directory of gold minutes labels partitions.",
    ),
    max_runs_per_day: int | None = typer.Option(
        None,
        "--max-runs-per-day",
        min=1,
        help="Optional cap on run=*/features.parquet files loaded per day (keeps latest N by directory sort).",
    ),
    keep_post_tip: bool = typer.Option(
        False,
        "--keep-post-tip",
        help="Keep snapshots with feature_as_of_ts > tip_ts (default drops them).",
        is_flag=True,
    ),
) -> None:
    start_day = _normalize_day(start_date)
    end_day = _normalize_day(end_date)
    _build_snapshot_dataset(
        start_date=start_day,
        end_date=end_day,
        out_dir=out_dir,
        live_features_root=live_features_root,
        labels_root=labels_root,
        max_runs_per_day=max_runs_per_day,
        keep_post_tip=keep_post_tip,
    )


@app.command("train-dual")
def train_dual_cmd(
    start_date: datetime = typer.Option(..., "--start-date", help="Start date (inclusive, YYYY-MM-DD) for snapshot discovery."),
    end_date: datetime = typer.Option(..., "--end-date", help="End date (inclusive, YYYY-MM-DD) for snapshot discovery."),
    early_run_id: str = typer.Option(..., "--early-run-id", help="Run id for the early minutes model bundle."),
    late_run_id: str = typer.Option(..., "--late-run-id", help="Run id for the late minutes model bundle."),
    train_end: datetime = typer.Option(..., "--train-end", help="Tip timestamp cutoff for the training split (UTC)."),
    cal_end: datetime = typer.Option(..., "--cal-end", help="Tip timestamp cutoff for the calibration split (UTC)."),
    val_end: datetime = typer.Option(..., "--val-end", help="Tip timestamp cutoff for the validation split (UTC)."),
    out_dir: Path = typer.Option(
        paths.data_path("training", "snapshots_minutes_v1"),
        "--out-dir",
        help="Directory to write the labeled snapshot parquet (run=<ts>/features.parquet).",
    ),
    dual_out_dir: Path = typer.Option(
        Path("data/training/horizons_minutes_v1"),
        "--dual-out-dir",
        help="Directory passed to `train_minutes_dual` for horizonized datasets.",
    ),
    artifact_root: Path = typer.Option(
        Path("artifacts/minutes_lgbm"),
        "--artifact-root",
        help="Root directory to write trained model bundles.",
    ),
    trainer_config: Path | None = typer.Option(
        None,
        "--trainer-config",
        help="Optional YAML config forwarded to the minutes_lgbm trainer for shared parameters.",
    ),
    live_features_root: Path = typer.Option(
        paths.data_path("live", "features_minutes_v1"),
        "--live-features-root",
        help="Root directory of live minutes feature runs.",
    ),
    labels_root: Path = typer.Option(
        paths.data_path("gold", "labels_minutes_v1"),
        "--labels-root",
        help="Root directory of gold minutes labels partitions.",
    ),
    max_runs_per_day: int | None = typer.Option(
        None,
        "--max-runs-per-day",
        min=1,
        help="Optional cap on run=*/features.parquet files loaded per day (keeps latest N by directory sort).",
    ),
    keep_post_tip: bool = typer.Option(
        False,
        "--keep-post-tip",
        help="Keep snapshots with feature_as_of_ts > tip_ts (default drops them).",
        is_flag=True,
    ),
    max_snapshot_age_hours: float | None = typer.Option(
        12.0,
        "--max-snapshot-age-hours",
        min=0.0,
        help="Forwarded to horizon selection freshness guard (set 0 to disable).",
    ),
) -> None:
    start_day = _normalize_day(start_date)
    end_day = _normalize_day(end_date)
    snapshot_path = _build_snapshot_dataset(
        start_date=start_day,
        end_date=end_day,
        out_dir=out_dir,
        live_features_root=live_features_root,
        labels_root=labels_root,
        max_runs_per_day=max_runs_per_day,
        keep_post_tip=keep_post_tip,
    )

    cmd = [
        sys.executable,
        "-m",
        "projections.cli.train_minutes_dual",
        "--features-path",
        str(snapshot_path),
        "--early-run-id",
        early_run_id,
        "--late-run-id",
        late_run_id,
        "--train-end",
        _format_click_datetime(train_end),
        "--cal-end",
        _format_click_datetime(cal_end),
        "--val-end",
        _format_click_datetime(val_end),
        "--out-dir",
        str(dual_out_dir),
        "--artifact-root",
        str(artifact_root),
    ]
    if trainer_config is not None:
        cmd.extend(["--trainer-config", str(trainer_config)])
    if max_snapshot_age_hours is not None:
        cmd.extend(["--max-snapshot-age-hours", str(max_snapshot_age_hours)])

    typer.echo("[snapshot-dataset] launching dual training: " + " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":  # pragma: no cover
    app()
