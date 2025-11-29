"""Compare fpts_v2 model against simple baselines on the same val split.

Example:
    uv run python -m scripts.fpts.compare_baselines \
      --data-root      /home/daniel/projections-data \
      --start-date     2023-10-24 \
      --end-date       2025-11-26 \
      --train-end-date 2024-06-30 \
      --cal-end-date   2025-03-01 \
      --fpts-run-id    fpts_v2_stage0_20251129_062655
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import typer

from projections.fpts_v2.current import load_fpts_bundle, _get_data_root
from projections.fpts_v2.features import build_fpts_design_matrix, CATEGORICAL_FEATURES_DEFAULT

app = typer.Typer(add_completion=False)


def _parse_date(value: str) -> date:
    try:
        return datetime.fromisoformat(value).date()
    except ValueError as exc:
        raise typer.BadParameter(f"Invalid date: {value}") from exc


def _iter_days(start: date, end: date):
    cursor = start
    while cursor <= end:
        yield cursor
        cursor += timedelta(days=1)


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


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not mask.any():
        return float("nan")
    y_t = y_true[mask]
    y_p = y_pred[mask]
    denom = np.sum((y_t - y_t.mean()) ** 2)
    if denom <= 0:
        return float("nan")
    return float(1.0 - np.sum((y_t - y_p) ** 2) / denom)


def _fit_minutes_baseline(train_df: pd.DataFrame, cal_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    working = pd.concat([train_df, cal_df], ignore_index=True)
    y = pd.to_numeric(working["dk_fpts_actual"], errors="coerce").to_numpy()
    minutes = pd.to_numeric(working["minutes_p50"], errors="coerce").to_numpy()
    # Design matrix: [1, minutes_p50]
    X = np.vstack([np.ones_like(minutes), minutes]).T
    mask = ~np.isnan(y) & ~np.isnan(minutes)
    if not mask.any():
        return np.array([np.nan, np.nan]), np.array([])
    coef, *_ = np.linalg.lstsq(X[mask], y[mask], rcond=None)
    return coef, mask


def _predict_minutes_baseline(minutes: np.ndarray, coef: np.ndarray) -> np.ndarray:
    return coef[0] + coef[1] * minutes


@app.command()
def main(
    data_root: Path = typer.Option(None, "--data-root"),
    start_date: str = typer.Option(..., "--start-date"),
    end_date: str = typer.Option(..., "--end-date"),
    train_end_date: str = typer.Option(..., "--train-end-date"),
    cal_end_date: str = typer.Option(..., "--cal-end-date"),
    fpts_run_id: str = typer.Option(..., "--fpts-run-id"),
) -> None:
    root = data_root or _get_data_root()
    start_dt = _parse_date(start_date)
    end_dt = _parse_date(end_date)
    train_end_dt = _parse_date(train_end_date)
    cal_end_dt = _parse_date(cal_end_date)

    start_ts = pd.Timestamp(start_dt).normalize()
    end_ts = pd.Timestamp(end_dt).normalize()
    train_end_ts = pd.Timestamp(train_end_dt).normalize()
    cal_end_ts = pd.Timestamp(cal_end_dt).normalize()

    df = _load_training_base(root, start_ts, end_ts)
    if "dk_fpts_actual" not in df.columns:
        raise RuntimeError("dk_fpts_actual column missing in training base.")

    bundle = load_fpts_bundle(fpts_run_id, data_root=root)
    feature_cols = bundle.feature_cols

    df, cal_df, val_df = _split_by_date(df, train_end_ts, cal_end_ts)
    train_cal_df = pd.concat([df, cal_df], ignore_index=True)

    # Season FPTS-per-minute baseline using train+cal
    work = train_cal_df.sort_values(["season", "player_id", "game_date", "game_id"]).copy()
    work["minutes_actual"] = pd.to_numeric(work.get("minutes_actual"), errors="coerce")
    work["dk_fpts_actual"] = pd.to_numeric(work["dk_fpts_actual"], errors="coerce")
    work["fpts_cum"] = work.groupby(["season", "player_id"])["dk_fpts_actual"].cumsum().shift(1)
    work["minutes_cum"] = work.groupby(["season", "player_id"])["minutes_actual"].cumsum().shift(1)
    denom_global = work["minutes_actual"].replace(0, np.nan).sum()
    global_fpm = work["dk_fpts_actual"].sum() / denom_global if denom_global and not np.isnan(denom_global) else 0.0
    work["season_fpts_per_min_prior"] = work["fpts_cum"] / work["minutes_cum"]
    work["season_fpts_per_min_prior"] = work["season_fpts_per_min_prior"].fillna(global_fpm)

    val_df = val_df.merge(
        work[["season", "player_id", "game_date", "season_fpts_per_min_prior"]],
        on=["season", "player_id", "game_date"],
        how="left",
    )
    val_df["minutes_p50"] = pd.to_numeric(val_df.get("minutes_p50"), errors="coerce")
    val_df["season_fpts_per_min_prior"] = val_df["season_fpts_per_min_prior"].fillna(global_fpm)
    val_df["fpts_baseline_season"] = val_df["season_fpts_per_min_prior"] * val_df["minutes_p50"]

    # Minutes linear baseline
    coef_minutes, _ = _fit_minutes_baseline(df, cal_df)
    val_minutes = pd.to_numeric(val_df.get("minutes_p50"), errors="coerce").to_numpy()
    val_df["fpts_baseline_minutes"] = _predict_minutes_baseline(val_minutes, coef_minutes)

    # Model predictions on val
    val_features = build_fpts_design_matrix(
        val_df,
        feature_cols,
        categorical_cols=CATEGORICAL_FEATURES_DEFAULT,
        fill_missing_with_zero=True,
    )
    num_iter = getattr(bundle.model, "best_iteration", None) or getattr(bundle.model, "best_iteration_", None)
    val_preds = bundle.model.predict(
        val_features.values, num_iteration=int(num_iter) if num_iter and num_iter > 0 else None
    )
    val_df["fpts_gbm"] = val_preds

    y_true = pd.to_numeric(val_df["dk_fpts_actual"], errors="coerce").to_numpy()

    metrics = {
        "gbm": {
            "mae": _mae(y_true, val_preds),
            "rmse": _rmse(y_true, val_preds),
            "r2": _r2(y_true, val_preds),
            "n": int(np.isfinite(y_true).sum()),
        },
        "baseline_season": {
            "mae": _mae(y_true, val_df["fpts_baseline_season"].to_numpy()),
            "rmse": _rmse(y_true, val_df["fpts_baseline_season"].to_numpy()),
            "r2": _r2(y_true, val_df["fpts_baseline_season"].to_numpy()),
            "n": int(np.isfinite(y_true).sum()),
        },
        "baseline_minutes": {
            "mae": _mae(y_true, val_df["fpts_baseline_minutes"].to_numpy()),
            "rmse": _rmse(y_true, val_df["fpts_baseline_minutes"].to_numpy()),
            "r2": _r2(y_true, val_df["fpts_baseline_minutes"].to_numpy()),
            "n": int(np.isfinite(y_true).sum()),
        },
    }

    typer.echo("Model              MAE      RMSE      R2        n")
    for key, label in (("gbm", "gbm_fpts_v2"), ("baseline_season", "baseline_season"), ("baseline_minutes", "baseline_minutes")):
        m = metrics[key]
        typer.echo(
            f"{label:<18} {m['mae']:.3f}  {m['rmse']:.3f}  {m['r2']:.3f}  {m['n']}"
        )

    out_dir = root / "artifacts" / "fpts_v2" / "runs" / fpts_run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_payload = {
        "val": metrics,
        "meta": {
            "start_date": start_dt.isoformat(),
            "end_date": end_dt.isoformat(),
            "train_end_date": train_end_dt.isoformat(),
            "cal_end_date": cal_end_dt.isoformat(),
            "run_id": fpts_run_id,
        },
    }
    (out_dir / "baseline_compare.json").write_text(json.dumps(out_payload, indent=2), encoding="utf-8")
    typer.echo(f"[compare_baselines] wrote {out_dir / 'baseline_compare.json'}")


if __name__ == "__main__":
    app()
