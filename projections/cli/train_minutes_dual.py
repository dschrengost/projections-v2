"""Train a dual (early/late) Minutes V1 LightGBM bundle from horizonized snapshots."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import typer

from projections.minutes_v1.horizons import (
    DEFAULT_EARLY_HORIZONS_MINUTES,
    DEFAULT_HORIZONS_MINUTES,
    DEFAULT_LATE_HORIZONS_MINUTES,
    TipTimeSplit,
    add_odds_missing_indicator,
    assign_game_splits,
    build_horizon_rows,
    filter_horizons,
)

app = typer.Typer(help=__doc__)

UTC = timezone.utc


def _parse_int_list(raw: str, *, label: str) -> list[int]:
    values: list[int] = []
    for token in (raw or "").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            values.append(int(token))
        except ValueError as exc:
            raise typer.BadParameter(f"Invalid {label} integer: {token!r}") from exc
    if not values:
        raise typer.BadParameter(f"{label} cannot be empty")
    return values


def _read_parquet_tree(path: Path) -> pd.DataFrame:
    path = path.expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Missing parquet input at {path}")
    if path.is_file():
        return pd.read_parquet(path)
    files = sorted(path.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files discovered under {path}")
    frames = [pd.read_parquet(file) for file in files]
    return pd.concat(frames, ignore_index=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


@dataclass(frozen=True)
class DualDatasetPaths:
    horizons_path: Path
    early_path: Path
    late_path: Path


def _build_datasets(
    features: pd.DataFrame,
    *,
    out_dir: Path,
    horizons_minutes: list[int],
    early_horizons: list[int],
    late_horizons: list[int],
    snapshot_ts_col: str,
    max_snapshot_age_hours: float | None,
    train_end: datetime,
    cal_end: datetime,
    val_end: datetime,
) -> DualDatasetPaths:
    out_dir.mkdir(parents=True, exist_ok=True)

    horizon_df = build_horizon_rows(
        features,
        horizons_minutes=horizons_minutes,
        snapshot_ts_col=snapshot_ts_col,
        max_snapshot_age_hours=max_snapshot_age_hours,
    )
    if horizon_df.empty:
        raise RuntimeError("Horizon selection produced zero rows; check snapshot coverage and horizons.")

    horizon_df = add_odds_missing_indicator(horizon_df)
    split = TipTimeSplit.from_bounds(train_end=train_end, cal_end=cal_end, val_end=val_end)
    horizon_df = assign_game_splits(horizon_df, split, output_col="split")
    horizon_df = horizon_df[horizon_df["split"].isin(["train", "cal", "val"])].copy()
    if horizon_df.empty:
        raise RuntimeError("All rows fell outside train/cal/val bounds; check cutoff timestamps.")

    split_counts = horizon_df["split"].value_counts().to_dict()
    typer.echo(f"[dual-train] split counts: {split_counts}")
    if int(split_counts.get("train", 0)) == 0:
        raise RuntimeError("Split assignment produced zero train rows; check --train-end / input coverage.")
    if int(split_counts.get("cal", 0)) == 0:
        raise RuntimeError("Split assignment produced zero cal rows; check --cal-end / input coverage.")
    if int(split_counts.get("val", 0)) == 0:
        typer.echo("[dual-train] warning: split assignment produced zero val rows.", err=True)

    early_df = filter_horizons(horizon_df, early_horizons)
    late_df = filter_horizons(horizon_df, late_horizons)
    if early_df.empty:
        raise RuntimeError("Early dataset is empty after filtering horizons.")
    if late_df.empty:
        raise RuntimeError("Late dataset is empty after filtering horizons.")

    early_split_counts = early_df["split"].value_counts().to_dict()
    late_split_counts = late_df["split"].value_counts().to_dict()
    typer.echo(f"[dual-train] early split counts: {early_split_counts}")
    typer.echo(f"[dual-train] late  split counts: {late_split_counts}")

    if int(early_split_counts.get("train", 0)) == 0:
        raise RuntimeError(
            "Early dataset has zero train rows after horizon filtering. "
            "This usually means early snapshots (e.g., 90+ minutes) only exist after --train-end. "
            "Move --train-end later or expand snapshot coverage."
        )
    if int(early_split_counts.get("cal", 0)) == 0:
        raise RuntimeError(
            "Early dataset has zero calibration rows after horizon filtering; conformal calibration requires cal rows. "
            "Move --cal-end later or expand snapshot coverage."
        )
    if int(early_split_counts.get("val", 0)) == 0:
        typer.echo("[dual-train] warning: early dataset has zero val rows.", err=True)

    if int(late_split_counts.get("train", 0)) == 0:
        raise RuntimeError(
            "Late dataset has zero train rows after horizon filtering. "
            "Move --train-end later or expand snapshot coverage."
        )
    if int(late_split_counts.get("cal", 0)) == 0:
        raise RuntimeError(
            "Late dataset has zero calibration rows after horizon filtering; conformal calibration requires cal rows. "
            "Move --cal-end later or expand snapshot coverage."
        )
    if int(late_split_counts.get("val", 0)) == 0:
        typer.echo("[dual-train] warning: late dataset has zero val rows.", err=True)

    horizons_path = out_dir / "horizon_rows.parquet"
    early_path = out_dir / "early_rows.parquet"
    late_path = out_dir / "late_rows.parquet"

    horizon_df.to_parquet(horizons_path, index=False)
    early_df.to_parquet(early_path, index=False)
    late_df.to_parquet(late_path, index=False)

    manifest = {
        "created_at": datetime.now(tz=UTC).isoformat(),
        "snapshot_ts_col": snapshot_ts_col,
        "max_snapshot_age_hours": max_snapshot_age_hours,
        "horizons_minutes": horizons_minutes,
        "early_horizons_minutes": early_horizons,
        "late_horizons_minutes": late_horizons,
        "split_bounds": {
            "train_end": train_end.replace(tzinfo=UTC).isoformat(),
            "cal_end": cal_end.replace(tzinfo=UTC).isoformat(),
            "val_end": val_end.replace(tzinfo=UTC).isoformat(),
        },
        "counts": {
            "horizon_rows": int(len(horizon_df)),
            "early_rows": int(len(early_df)),
            "late_rows": int(len(late_df)),
            "games": int(horizon_df["game_id"].nunique()) if "game_id" in horizon_df.columns else None,
        },
        "paths": {
            "horizons": str(horizons_path),
            "early": str(early_path),
            "late": str(late_path),
        },
    }
    _write_json(out_dir / "manifest.json", manifest)
    return DualDatasetPaths(horizons_path=horizons_path, early_path=early_path, late_path=late_path)


def _run_minutes_trainer(
    *,
    dataset_path: Path,
    run_id: str,
    artifact_root: Path,
    trainer_config: Path | None,
    fold_id: str | None,
) -> None:
    cmd = [
        sys.executable,
        "-m",
        "projections.models.minutes_lgbm",
        "--features",
        str(dataset_path),
        "--run-id",
        run_id,
        "--artifact-root",
        str(artifact_root),
        "--split-col",
        "split",
    ]
    if trainer_config is not None:
        cmd.extend(["--config", str(trainer_config)])
    if fold_id is not None:
        cmd.extend(["--fold-id", fold_id])
    typer.echo("[dual-train] running: " + " ".join(cmd))
    subprocess.run(cmd, check=True)


@app.command()
def main(
    features_path: Path = typer.Option(..., "--features-path", help="Parquet file/dir containing snapshot-rich minutes features."),
    out_dir: Path = typer.Option(
        Path("data/training/horizons_minutes_v1"),
        "--out-dir",
        help="Directory to write horizonized datasets (horizon_rows.parquet, early_rows.parquet, late_rows.parquet).",
    ),
    early_run_id: str = typer.Option(..., "--early-run-id", help="Run id for the early minutes model bundle."),
    late_run_id: str = typer.Option(..., "--late-run-id", help="Run id for the late minutes model bundle."),
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
    train_end: datetime = typer.Option(..., "--train-end", help="Tip timestamp cutoff for the training split (UTC)."),
    cal_end: datetime = typer.Option(..., "--cal-end", help="Tip timestamp cutoff for the calibration split (UTC)."),
    val_end: datetime = typer.Option(..., "--val-end", help="Tip timestamp cutoff for the validation split (UTC)."),
    horizons_minutes: str = typer.Option(
        ",".join(str(x) for x in DEFAULT_HORIZONS_MINUTES),
        "--horizons-minutes",
        help="Comma-separated horizons (minutes before tip) to sample.",
    ),
    early_horizons_minutes: str = typer.Option(
        ",".join(str(x) for x in DEFAULT_EARLY_HORIZONS_MINUTES),
        "--early-horizons-minutes",
        help="Comma-separated horizons (minutes before tip) used for the early model.",
    ),
    late_horizons_minutes: str = typer.Option(
        ",".join(str(x) for x in DEFAULT_LATE_HORIZONS_MINUTES),
        "--late-horizons-minutes",
        help="Comma-separated horizons (minutes before tip) used for the late model.",
    ),
    snapshot_ts_col: str = typer.Option(
        "feature_as_of_ts",
        "--snapshot-ts-col",
        help="Column name used to order snapshots for horizon selection (typically feature_as_of_ts or snapshot_ts).",
    ),
    max_snapshot_age_hours: float | None = typer.Option(
        12.0,
        "--max-snapshot-age-hours",
        min=0.0,
        help="Optional freshness guard: require selected snapshot_ts >= tip_ts - max_age_hours (set 0 to disable).",
    ),
    fold_id_prefix: str = typer.Option(
        "dual",
        "--fold-id-prefix",
        help="Optional fold id prefix recorded in metrics for each bundle.",
    ),
    smoke_inference: bool = typer.Option(
        False,
        "--smoke-inference",
        help="After training, load both bundles and run a tiny inference sample using dual routing.",
        is_flag=True,
    ),
    smoke_rows: int = typer.Option(
        8,
        "--smoke-rows",
        min=1,
        help="Rows to score in smoke inference (uses horizon_rows.parquet).",
    ),
) -> None:
    out_dir = out_dir.expanduser().resolve()
    artifact_root = artifact_root.expanduser().resolve()
    trainer_config = trainer_config.expanduser().resolve() if trainer_config else None

    horizons_list = _parse_int_list(horizons_minutes, label="--horizons-minutes")
    early_list = _parse_int_list(early_horizons_minutes, label="--early-horizons-minutes")
    late_list = _parse_int_list(late_horizons_minutes, label="--late-horizons-minutes")

    max_age = None if (max_snapshot_age_hours is None or max_snapshot_age_hours <= 0) else float(max_snapshot_age_hours)

    features = _read_parquet_tree(features_path)
    typer.echo(f"[dual-train] loaded {len(features):,} source rows from {features_path}")

    stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    run_out_dir = out_dir / f"run={stamp}"
    paths = _build_datasets(
        features,
        out_dir=run_out_dir,
        horizons_minutes=horizons_list,
        early_horizons=early_list,
        late_horizons=late_list,
        snapshot_ts_col=snapshot_ts_col,
        max_snapshot_age_hours=max_age,
        train_end=train_end,
        cal_end=cal_end,
        val_end=val_end,
    )
    typer.echo(f"[dual-train] wrote datasets under {run_out_dir}")

    _run_minutes_trainer(
        dataset_path=paths.early_path,
        run_id=early_run_id,
        artifact_root=artifact_root,
        trainer_config=trainer_config,
        fold_id=f"{fold_id_prefix}_early",
    )
    _run_minutes_trainer(
        dataset_path=paths.late_path,
        run_id=late_run_id,
        artifact_root=artifact_root,
        trainer_config=trainer_config,
        fold_id=f"{fold_id_prefix}_late",
    )

    if smoke_inference:
        from projections.cli.score_minutes_v1 import _score_rows_dual as score_rows_dual  # local import to keep CLI lean

        early_bundle = joblib.load(artifact_root / early_run_id / "lgbm_quantiles.joblib")
        late_bundle = joblib.load(artifact_root / late_run_id / "lgbm_quantiles.joblib")
        sample_df = pd.read_parquet(paths.horizons_path)
        sample_df = sample_df.dropna(subset=["tip_ts", "feature_as_of_ts", "time_to_tip_min"]).copy()
        sample_df = sample_df.sort_values("time_to_tip_min", ascending=True, kind="mergesort").reset_index(drop=True)
        if not sample_df.empty:
            import numpy as np

            picks = np.linspace(0, len(sample_df) - 1, num=min(smoke_rows, len(sample_df)), dtype=int)
            picks = sorted(set(int(i) for i in picks))
            sample_df = sample_df.iloc[picks].copy()
        scored = score_rows_dual(
            sample_df,
            early_bundle=early_bundle,
            late_bundle=late_bundle,
            late_threshold_min=60.0,
            blend_band_min=30.0,
        )
        preview_cols = [
            "game_id",
            "player_id",
            "team_id",
            "time_to_tip_min",
            "minutes_model_used",
            "minutes_model_late_weight",
            "minutes_p50",
            "play_prob",
        ]
        available = [col for col in preview_cols if col in scored.columns]
        typer.echo("[dual-train] smoke inference preview:")
        typer.echo(scored[available].head(smoke_rows).to_string(index=False))


if __name__ == "__main__":  # pragma: no cover
    app()
