"""Residual calibration debugger for FPTS v2 + residual model.

Example:
    uv run python -m scripts.sim_v2.debug_residual_calibration_fpts_v2 \
      --data-root /home/daniel/projections-data \
      --fpts-run-id fpts_v2_stage0_20251129_062655
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import typer

from projections.fpts_v2.features import CATEGORICAL_FEATURES_DEFAULT, build_fpts_design_matrix
from projections.fpts_v2.loader import load_fpts_and_residual
from projections.sim_v2.residuals import select_bucket

app = typer.Typer(add_completion=False)


def _parse_date(value: str) -> date:
    try:
        return datetime.fromisoformat(value).date()
    except ValueError as exc:
        raise typer.BadParameter(f"Invalid date: {value}") from exc


def _iter_partitions(root: Path, start: pd.Timestamp, end: pd.Timestamp) -> list[Path]:
    base = root / "gold" / "fpts_training_base"
    partitions: list[Path] = []
    for season_dir in base.glob("season=*"):
        for day_dir in season_dir.glob("game_date=*"):
            try:
                day = pd.Timestamp(day_dir.name.split("=", 1)[1]).normalize()
            except ValueError:
                continue
            if day < start or day > end:
                continue
            candidate = day_dir / "fpts_training_base.parquet"
            if candidate.exists():
                partitions.append(candidate)
    return sorted(partitions)


def _load_training_base(root: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    paths = _iter_partitions(root, start, end)
    if not paths:
        raise FileNotFoundError("No fpts_training_base partitions in range.")
    frames = [pd.read_parquet(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.normalize()
    return df


def _prob_in_range(z: np.ndarray, threshold: float) -> float:
    if z.size == 0:
        return float("nan")
    return float((np.abs(z) < threshold).mean())


def _bucket_metrics(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    by_bucket: dict[str, dict[str, float]] = {}
    for bucket_name, grp in df.groupby("resid_bucket_name"):
        z = grp["z"].to_numpy()
        by_bucket[bucket_name] = {
            "n": int(len(z)),
            "z_mean": float(np.nanmean(z)) if len(z) else float("nan"),
            "z_std": float(np.nanstd(z)) if len(z) else float("nan"),
            "p_abs_lt_1": _prob_in_range(z, 1.0),
            "p_abs_lt_2": _prob_in_range(z, 2.0),
            "p_abs_lt_3": _prob_in_range(z, 3.0),
        }
    return by_bucket


@app.command()
def main(
    data_root: Path = typer.Option(..., "--data-root"),
    fpts_run_id: str = typer.Option(..., "--fpts-run-id"),
    start_date: Optional[str] = typer.Option(None, "--start-date"),
    end_date: Optional[str] = typer.Option(None, "--end-date"),
    use_val_only: bool = typer.Option(True, "--use-val-only/--no-use-val-only"),
) -> None:
    bundle, resid_model = load_fpts_and_residual(fpts_run_id, data_root=data_root)
    meta_window = bundle.meta.get("date_window") or {}
    meta_start = meta_window.get("start")
    meta_end = meta_window.get("end")
    meta_train_end = meta_window.get("train_end")
    meta_cal_end = meta_window.get("cal_end")
    if not (meta_start and meta_end and meta_train_end and meta_cal_end):
        raise RuntimeError("meta.date_window missing start/end/train_end/cal_end")

    start_dt = _parse_date(start_date) if start_date else _parse_date(str(meta_start))
    end_dt = _parse_date(end_date) if end_date else _parse_date(str(meta_end))
    cal_end_dt = _parse_date(str(meta_cal_end))

    typer.echo(
        f"[resid_calib] run_id={fpts_run_id} window=({start_dt}→{end_dt}) use_val_only={use_val_only}"
    )

    start_ts = pd.Timestamp(start_dt).normalize()
    end_ts = pd.Timestamp(end_dt).normalize()
    cal_end_ts = pd.Timestamp(cal_end_dt).normalize()

    df = _load_training_base(data_root, start_ts, end_ts)
    seasons = sorted(df["season"].dropna().unique().tolist())
    typer.echo(
        f"[resid_calib] loaded rows={len(df):,} seasons={seasons} date_range=({df['game_date'].min().date()}→{df['game_date'].max().date()})"
    )

    if use_val_only:
        mask = (df["game_date"] > cal_end_ts) & (df["game_date"] <= end_ts)
        df = df.loc[mask].copy()
    if df.empty:
        typer.echo("[resid_calib] no rows after filtering; exiting.")
        raise typer.Exit(code=0)
    required = ["dk_fpts_actual", "minutes_p50", "is_starter"]
    df = df.dropna(subset=[c for c in required if c in df.columns])
    typer.echo(f"[resid_calib] eval_rows={len(df):,} (val-only={use_val_only})")

    features = build_fpts_design_matrix(
        df,
        bundle.feature_cols,
        categorical_cols=CATEGORICAL_FEATURES_DEFAULT,
        fill_missing_with_zero=True,
    )
    num_iter = getattr(bundle.model, "best_iteration", None) or getattr(bundle.model, "best_iteration_", None)
    preds = bundle.model.predict(
        features.values, num_iteration=int(num_iter) if num_iter and num_iter > 0 else None
    )
    df["dk_fpts_pred"] = preds

    def _bucket_info(row: pd.Series):
        bucket = select_bucket(row, resid_model)
        if bucket is None:
            return pd.Series({"resid_bucket_name": None, "resid_sigma": np.nan})
        return pd.Series({"resid_bucket_name": bucket.name, "resid_sigma": bucket.sigma})

    bucket_info = df.apply(_bucket_info, axis=1)
    df = pd.concat([df, bucket_info], axis=1)

    before_drop = len(df)
    df = df[df["resid_sigma"].notna() & (df["resid_sigma"] > 0)]
    dropped = before_drop - len(df)
    if dropped:
        typer.echo(f"[resid_calib] dropped {dropped} rows with no valid residual bucket or sigma<=0")

    df["resid"] = pd.to_numeric(df["dk_fpts_actual"], errors="coerce") - df["dk_fpts_pred"]
    df["z"] = df["resid"] / df["resid_sigma"]
    df = df[df["z"].notna()]
    if df.empty:
        typer.echo("[resid_calib] no rows with valid z-scores; exiting.")
        raise typer.Exit(code=0)

    z = df["z"].to_numpy()
    overall = {
        "n": int(len(z)),
        "z_mean": float(np.mean(z)),
        "z_std": float(np.std(z)),
        "p_abs_lt_1": _prob_in_range(z, 1.0),
        "p_abs_lt_2": _prob_in_range(z, 2.0),
        "p_abs_lt_3": _prob_in_range(z, 3.0),
    }
    typer.echo(
        f"[resid_calib][overall] n={overall['n']} z_mean={overall['z_mean']:.3f} z_std={overall['z_std']:.3f}"
    )
    typer.echo(
        f"[resid_calib][overall] p(|z|<1)={overall['p_abs_lt_1']:.3f} "
        f"p(|z|<2)={overall['p_abs_lt_2']:.3f} p(|z|<3)={overall['p_abs_lt_3']:.3f}"
    )

    bucket_stats = _bucket_metrics(df)
    if bucket_stats:
        typer.echo("Bucket             n    z_mean   z_std   p<1   p<2   p<3")
        for name, stats in bucket_stats.items():
            typer.echo(
                f"{name:<18} {stats['n']:>5}  {stats['z_mean'] if not np.isnan(stats['z_mean']) else float('nan'):+6.3f}  "
                f"{stats['z_std'] if not np.isnan(stats['z_std']) else float('nan'):6.3f}  "
                f"{stats['p_abs_lt_1']:5.3f} {stats['p_abs_lt_2']:5.3f} {stats['p_abs_lt_3']:5.3f}"
            )

    out_path = (
        data_root
        / "artifacts"
        / "fpts_v2"
        / "runs"
        / fpts_run_id
        / "residual_calibration.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": fpts_run_id,
        "date_window": {
            "start": str(meta_start),
            "end": str(meta_end),
            "train_end": str(meta_train_end),
            "cal_end": str(meta_cal_end),
        },
        "eval_window": {
            "start": start_dt.isoformat(),
            "end": end_dt.isoformat(),
            "use_val_only": use_val_only,
        },
        "overall": overall,
        "by_bucket": bucket_stats,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    typer.echo(f"[resid_calib] wrote calibration report to {out_path}")


if __name__ == "__main__":
    app()
