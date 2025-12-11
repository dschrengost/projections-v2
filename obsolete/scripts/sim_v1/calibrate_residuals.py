from __future__ import annotations

import json
import math
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import typer

from projections.fpts_v2.current import _get_data_root, _load_current_run_id, load_current_fpts_bundle
from projections.sim_v1.residuals import ResidualModel, fit_residual_model, to_json

app = typer.Typer(add_completion=False)


def _parse_date(value: str) -> date:
    try:
        return datetime.fromisoformat(value).date()
    except ValueError as exc:
        raise typer.BadParameter(f"Invalid date format: {value} (expected YYYY-MM-DD)") from exc


def _season_from_date(day: date) -> int:
    """Season keyed by the year the season starts (Aug–Jul)."""

    return day.year if day.month >= 8 else day.year - 1


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
        raise FileNotFoundError(f"No fpts_training_base partitions found between {start.date()} and {end.date()}")
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


def _resolve_label_col(df: pd.DataFrame) -> str:
    for candidate in ("dk_fpts", "dk_fpts_actual", "fpts_dk_label"):
        if candidate in df.columns:
            return candidate
    raise RuntimeError("Could not find a fantasy points label column (dk_fpts, dk_fpts_actual, or fpts_dk_label).")


def _prepare_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing}")

    feature_frame = df[feature_cols].copy()
    feature_frame = feature_frame.apply(pd.to_numeric, errors="coerce")

    categorical_cols = {"track_role_cluster", "track_role_is_low_minutes"}
    for col in categorical_cols & set(feature_frame.columns):
        feature_frame[col] = feature_frame[col].fillna(-1).astype(int)

    feature_frame = feature_frame.fillna(0.0)
    return feature_frame


def _score_predictions(df: pd.DataFrame, feature_cols: list[str], model) -> np.ndarray:
    feature_frame = _prepare_features(df, feature_cols)
    mask = feature_frame.notna().all(axis=1)
    preds = np.full(len(df), np.nan)
    if mask.any():
        num_iter = getattr(model, "best_iteration", None) or getattr(model, "best_iteration_", None)
        preds[mask] = model.predict(
            feature_frame.loc[mask].values, num_iteration=int(num_iter) if num_iter and num_iter > 0 else None
        )
    return preds


def _compute_metrics(y_true: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    mask = ~np.isnan(y_true) & ~np.isnan(preds)
    if not mask.any():
        return {"mae": math.nan, "rmse": math.nan, "n": 0}
    err = preds[mask] - y_true[mask]
    mae = float(np.abs(err).mean())
    rmse = float(np.sqrt((err ** 2).mean()))
    return {"mae": mae, "rmse": rmse, "n": int(mask.sum())}


def _load_minutes_for_rates(root: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    def _iter_days(s: pd.Timestamp, e: pd.Timestamp):
        current = s.normalize()
        while current <= e:
            yield current
            current += pd.Timedelta(days=1)

    frames: list[pd.DataFrame] = []
    for day in _iter_days(start, end):
        season = _season_from_date(day.date())
        path = (
            root
            / "gold"
            / "minutes_for_rates"
            / f"season={season}"
            / f"game_date={day.date().isoformat()}"
            / "minutes_for_rates.parquet"
        )
        if path.exists():
            frames.append(pd.read_parquet(path))
    if not frames:
        return pd.DataFrame()

    minutes = pd.concat(frames, ignore_index=True)
    minutes["game_date"] = pd.to_datetime(minutes["game_date"]).dt.normalize()
    return minutes


def _ensure_minutes(df: pd.DataFrame, minutes_preds: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    if "minutes_pred_p50" in working.columns:
        return working

    if not minutes_preds.empty:
        key_cols = ["season", "game_date", "team_id", "player_id"]
        join_cols = [
            c
            for c in ("minutes_pred_p50", "minutes_pred_play_prob", "minutes_pred_p10", "minutes_pred_p90")
            if c in minutes_preds.columns
        ]
        minutes_slice = minutes_preds[key_cols + join_cols].drop_duplicates(subset=key_cols)
        merged = working.merge(minutes_slice, on=key_cols, how="left", suffixes=("", "_minutes"))
        if "minutes_pred_p50" in merged.columns and merged["minutes_pred_p50"].notna().any():
            return merged

    if "minutes_actual" in working.columns:
        typer.echo("[sim_residuals] minutes_pred_p50 missing; falling back to minutes_actual")
        working["minutes_pred_p50"] = working["minutes_actual"]
    else:
        typer.echo("[sim_residuals] missing minutes_pred_p50 and minutes_actual; minutes left empty")
        working["minutes_pred_p50"] = np.nan
    return working


def _log_bucket_summary(model: ResidualModel) -> None:
    typer.echo("[sim_residuals] bucket scales:")
    for bucket in model.buckets:
        typer.echo(
            f"  - {bucket.name}: n={bucket.n} sigma={bucket.sigma:.3f} "
            f"nu={bucket.nu} minutes=[{bucket.min_minutes}, {bucket.max_minutes or 'inf'}) "
            f"is_starter={bucket.is_starter}"
        )
    typer.echo(
        f"[sim_residuals] defaults: sigma_default={model.sigma_default:.3f} nu_default={model.nu_default}"
    )


@app.command()
def main(
    data_root: Path = typer.Option(..., "--data-root"),
    start_date: str = typer.Option(..., "--start-date"),
    end_date: str = typer.Option(..., "--end-date"),
    train_end_date: str = typer.Option(..., "--train-end-date"),
    cal_end_date: str = typer.Option(..., "--cal-end-date"),
    output_path: Optional[Path] = typer.Option(
        None,
        "--output-path",
        help="Defaults to PROJECTIONS_DATA_ROOT/artifacts/sim_v1/fpts_residual_model.json",
    ),
) -> None:
    """
    Calibrate an empirical residual model for FPTS predictions.
    """

    root = data_root or _get_data_root()
    start_dt = _parse_date(start_date)
    end_dt = _parse_date(end_date)
    train_end_dt = _parse_date(train_end_date)
    cal_end_dt = _parse_date(cal_end_date)
    start = pd.Timestamp(start_dt).normalize()
    end = pd.Timestamp(end_dt).normalize()
    train_end = pd.Timestamp(train_end_dt).normalize()
    cal_end = pd.Timestamp(cal_end_dt).normalize()

    bundle = load_current_fpts_bundle(data_root=root)
    run_id = _load_current_run_id()
    typer.echo(f"[sim_residuals] using fpts_v2 run_id={run_id} feature_cols={len(bundle.feature_cols)}")

    df = _load_training_base(root, start, end)
    label_col = _resolve_label_col(df)

    preds = _score_predictions(df, bundle.feature_cols, bundle.model)
    df["dk_fpts_pred"] = preds

    minutes_preds = _load_minutes_for_rates(root, start, end)
    df = _ensure_minutes(df, minutes_preds)
    if "is_starter" not in df.columns:
        typer.echo("[sim_residuals] is_starter missing; defaulting to bench=0 for all rows")
        df["is_starter"] = 0

    train_df, cal_df, _ = _split_by_date(df, train_end, cal_end)
    fit_df = pd.concat([train_df, cal_df], ignore_index=True)

    model = fit_residual_model(
        fit_df,
        fpts_pred_col="dk_fpts_pred",
        fpts_label_col=label_col,
        minutes_col="minutes_pred_p50",
        is_starter_col="is_starter",
    )

    _log_bucket_summary(model)

    y_train = pd.to_numeric(train_df[label_col], errors="coerce").to_numpy()
    y_cal = pd.to_numeric(cal_df[label_col], errors="coerce").to_numpy()
    preds_train = pd.to_numeric(train_df["dk_fpts_pred"], errors="coerce").to_numpy()
    preds_cal = pd.to_numeric(cal_df["dk_fpts_pred"], errors="coerce").to_numpy()
    metrics_train = _compute_metrics(y_train, preds_train)
    metrics_cal = _compute_metrics(y_cal, preds_cal)
    typer.echo(
        f"[sim_residuals] train mae={metrics_train['mae']:.3f} rmse={metrics_train['rmse']:.3f} n={metrics_train['n']}"
    )
    typer.echo(
        f"[sim_residuals] cal mae={metrics_cal['mae']:.3f} rmse={metrics_cal['rmse']:.3f} n={metrics_cal['n']}"
    )

    target_path = output_path or (root / "artifacts" / "sim_v1" / "fpts_residual_model.json")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    payload = to_json(model)
    target_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    typer.echo(f"[sim_residuals] wrote residual model to {target_path}")


if __name__ == "__main__":
    app()
