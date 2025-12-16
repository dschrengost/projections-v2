from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import typer

from projections.fpts_v2.current import _get_data_root, _load_current_run_id, load_current_fpts_bundle

app = typer.Typer(add_completion=False)

DEFAULT_START = "2023-10-01"
DEFAULT_END = "2025-11-26"


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


def _compute_metrics(y_true: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    err = preds - y_true
    mae = float(np.abs(err).mean())
    rmse = float(math.sqrt(np.mean(err ** 2)))
    return {"mae": mae, "rmse": rmse, "n": int(len(y_true))}


@app.command()
def main(
    data_root: Optional[Path] = typer.Option(None, help="Root containing gold/fpts_training_base."),
    start_date: str = typer.Option(DEFAULT_START, help="Start date for evaluation window (YYYY-MM-DD)."),
    end_date: str = typer.Option(DEFAULT_END, help="End date for evaluation window (YYYY-MM-DD)."),
    limit: Optional[int] = typer.Option(10, help="Limit rows to display from the sample output."),
) -> None:
    root = data_root or _get_data_root()
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()

    run_id = _load_current_run_id()
    bundle = load_current_fpts_bundle(data_root=root)
    typer.echo(f"[fpts_debug] run_id={run_id} feature_set={bundle.meta.get('feature_set')}")

    df = _load_training_base(root, start, end)

    missing = [c for c in bundle.feature_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing expected feature columns in base data: {missing}")

    feature_frame = df[bundle.feature_cols].copy()
    feature_frame = feature_frame.apply(pd.to_numeric, errors="coerce")
    categorical_cols = {"track_role_cluster", "track_role_is_low_minutes"}
    for col in categorical_cols & set(feature_frame.columns):
        feature_frame[col] = feature_frame[col].fillna(-1).astype(int)
    feature_frame = feature_frame.fillna(0.0)

    if "dk_fpts_actual" in df.columns:
        label_col = "dk_fpts_actual"
    elif "fpts_dk_label" in df.columns:
        label_col = "fpts_dk_label"
    else:
        raise RuntimeError("Could not find fpts label column (dk_fpts_actual or fpts_dk_label).")

    labels = pd.to_numeric(df[label_col], errors="coerce")

    mask = feature_frame.notna().all(axis=1) & labels.notna()
    if not mask.any():
        raise RuntimeError("No rows available after dropping NaNs in features/label.")

    X = feature_frame.loc[mask]
    y = labels.loc[mask].values
    num_iter = bundle.model.best_iteration
    preds = bundle.model.predict(X.values, num_iteration=num_iter if num_iter and num_iter > 0 else None)
    metrics = _compute_metrics(y, preds)
    typer.echo(f"[fpts_debug] rows_used={metrics['n']} dropped={len(df) - metrics['n']}")
    typer.echo(f"[fpts_debug] quick_eval mae={metrics['mae']:.3f} rmse={metrics['rmse']:.3f} n={metrics['n']}")

    sample_cols = ["player_id", "minutes_p50", label_col]
    sample = df.loc[mask, sample_cols].copy()
    sample["fpts_pred"] = preds
    if limit is not None:
        sample = sample.head(limit)

    typer.echo(sample.to_string(index=False))


if __name__ == "__main__":
    app()
