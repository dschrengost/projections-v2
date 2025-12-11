"""Fit an empirical residual model for a specific FPTS v2 run.

Example:
    uv run python -m scripts.fpts.fit_residual_model \
      --data-root /home/daniel/projections-data \
      --fpts-run-id fpts_v2_stage0_20251129_062655
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import typer

from projections.fpts_v2.current import _get_data_root, load_fpts_bundle
from projections.fpts_v2.features import CATEGORICAL_FEATURES_DEFAULT, build_fpts_design_matrix

app = typer.Typer(add_completion=False)


BUCKETS = [
    {"name": "starter_under_24", "is_starter": 1, "min_minutes": 0.0, "max_minutes": 24.0},
    {"name": "starter_24_32", "is_starter": 1, "min_minutes": 24.0, "max_minutes": 32.0},
    {"name": "starter_32_plus", "is_starter": 1, "min_minutes": 32.0, "max_minutes": None},
    {"name": "bench_under_12", "is_starter": 0, "min_minutes": 0.0, "max_minutes": 12.0},
    {"name": "bench_12_20", "is_starter": 0, "min_minutes": 12.0, "max_minutes": 20.0},
    {"name": "bench_20_28", "is_starter": 0, "min_minutes": 20.0, "max_minutes": 28.0},
    {"name": "bench_28_plus", "is_starter": 0, "min_minutes": 28.0, "max_minutes": None},
]


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
        raise FileNotFoundError("No fpts_training_base partitions matched the requested window.")
    frames = [pd.read_parquet(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.normalize()
    return df


def _split_by_date(
    df: pd.DataFrame, train_end: pd.Timestamp, cal_end: pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = df[df["game_date"] <= train_end].copy()
    cal_df = df[(df["game_date"] > train_end) & (df["game_date"] <= cal_end)].copy()
    val_df = df[df["game_date"] > cal_end].copy()
    return train_df, cal_df, val_df


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not mask.any():
        return float("nan")
    return float(np.abs(y_true[mask] - y_pred[mask]).mean())


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not mask.any():
        return float("nan")
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def _read_meta(run_dir: Path) -> dict[str, Any]:
    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _winsorize(arr: np.ndarray, quantile: float) -> np.ndarray:
    if arr.size == 0:
        return arr
    lo = np.quantile(arr, 1 - quantile)
    hi = np.quantile(arr, quantile)
    return np.clip(arr, lo, hi)


@app.command()
def main(
    data_root: Path = typer.Option(None, "--data-root"),
    fpts_run_id: str = typer.Option(..., "--fpts-run-id"),
    output_json: Optional[Path] = typer.Option(
        None,
        "--output-json",
        help="Defaults to <data_root>/artifacts/sim_v1/fpts_residual_model.json",
    ),
    min_play_prob: float = typer.Option(0.1, "--min-play-prob"),
    min_minutes_for_bucketing: float = typer.Option(0.0, "--min-minutes-for-bucketing"),
    winsor_quantile: float = typer.Option(0.99, "--winsor-quantile"),
    min_bucket_n: int = typer.Option(200, "--min-bucket-n"),
) -> None:
    root = data_root or _get_data_root()
    run_dir = root / "artifacts" / "fpts_v2" / "runs" / fpts_run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    bundle = load_fpts_bundle(fpts_run_id, data_root=root)
    meta = _read_meta(run_dir)
    date_window = meta.get("date_window") or {}
    start_date = date_window.get("start") or meta.get("start_date")
    end_date = date_window.get("end") or meta.get("end_date")
    train_end_date = date_window.get("train_end") or meta.get("train_end_date")
    cal_end_date = date_window.get("cal_end") or meta.get("cal_end_date")
    if not (start_date and end_date and train_end_date and cal_end_date):
        raise RuntimeError("Missing date_window in meta.json; expected start/end/train_end/cal_end.")

    start_ts = pd.Timestamp(_parse_date(str(start_date))).normalize()
    end_ts = pd.Timestamp(_parse_date(str(end_date))).normalize()
    train_end_ts = pd.Timestamp(_parse_date(str(train_end_date))).normalize()
    cal_end_ts = pd.Timestamp(_parse_date(str(cal_end_date))).normalize()

    typer.echo(
        f"[resid] run_id={fpts_run_id} window=({start_ts.date()} to {end_ts.date()}) "
        f"train_end={train_end_ts.date()} cal_end={cal_end_ts.date()} feature_cols={len(bundle.feature_cols)}"
    )

    df = _load_training_base(root, start_ts, end_ts)
    if "dk_fpts_actual" not in df.columns:
        raise RuntimeError("dk_fpts_actual column missing in training base.")

    train_df, cal_df, val_df = _split_by_date(df, train_end_ts, cal_end_ts)
    if val_df.empty:
        raise RuntimeError("Validation slice is empty; cannot fit residual model.")

    val_df["minutes_p50"] = pd.to_numeric(val_df.get("minutes_p50"), errors="coerce")
    val_df["minutes_actual"] = pd.to_numeric(val_df.get("minutes_actual"), errors="coerce")
    val_df["play_prob"] = pd.to_numeric(val_df.get("play_prob"), errors="coerce")
    val_df["is_starter"] = pd.to_numeric(val_df.get("is_starter"), errors="coerce")

    features = build_fpts_design_matrix(
        val_df,
        bundle.feature_cols,
        categorical_cols=CATEGORICAL_FEATURES_DEFAULT,
        fill_missing_with_zero=True,
    )
    num_iter = getattr(bundle.model, "best_iteration", None) or getattr(bundle.model, "best_iteration_", None)
    preds = bundle.model.predict(
        features.values, num_iteration=int(num_iter) if num_iter and num_iter > 0 else None
    )
    val_df["fpts_pred"] = preds

    label = pd.to_numeric(val_df["dk_fpts_actual"], errors="coerce")
    val_df["residual"] = label - val_df["fpts_pred"]

    mask = label.notna()
    mask &= val_df["play_prob"].fillna(0.0) >= min_play_prob
    mask &= val_df["minutes_p50"].fillna(0.0) >= min_minutes_for_bucketing
    if "minutes_actual" in val_df.columns:
        mask &= val_df["minutes_actual"].fillna(0.0) >= 1.0

    df_resid = val_df.loc[mask].copy()
    if df_resid.empty:
        raise RuntimeError("No rows after filtering for residual fit.")

    buckets_out: list[dict[str, Any]] = []
    for bucket in BUCKETS:
        sub = df_resid.copy()
        sub = sub[sub["is_starter"] == bucket["is_starter"]]
        if bucket["min_minutes"] is not None:
            sub = sub[sub["minutes_actual"] >= bucket["min_minutes"]]
        if bucket["max_minutes"] is not None:
            sub = sub[sub["minutes_actual"] < bucket["max_minutes"]]
        r = pd.to_numeric(sub["residual"], errors="coerce").dropna().to_numpy()
        if r.size < min_bucket_n:
            continue
        r_clip = _winsorize(r, winsor_quantile)
        sigma = float(r_clip.std(ddof=1))
        buckets_out.append(
            {
                "name": bucket["name"],
                "min_minutes": bucket["min_minutes"],
                "max_minutes": bucket["max_minutes"],
                "is_starter": bucket["is_starter"],
                "sigma": sigma,
                "nu": 5,
                "n": int(r.size),
            }
        )
        typer.echo(f"[resid] bucket {bucket['name']} n={r.size} sigma={sigma:.3f}")

    r_all = pd.to_numeric(df_resid["residual"], errors="coerce").dropna().to_numpy()
    r_all_clip = _winsorize(r_all, winsor_quantile)
    sigma_default = float(r_all_clip.std(ddof=1)) if r_all_clip.size > 0 else 0.0
    nu_default = 5

    mae_val = _mae(label.to_numpy(), preds)
    rmse_val = _rmse(label.to_numpy(), preds)
    typer.echo(f"[resid] val MAE={mae_val:.3f} RMSE={rmse_val:.3f} n={int(label.notna().sum())}")

    payload = {
        "buckets": buckets_out,
        "sigma_default": sigma_default,
        "nu_default": nu_default,
    }

    target_path = output_json or (root / "artifacts" / "sim_v1" / "fpts_residual_model.json")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    typer.echo(f"[resid] wrote residual model to {target_path}")


if __name__ == "__main__":
    app()
