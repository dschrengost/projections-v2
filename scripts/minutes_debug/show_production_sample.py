"""Inspect a small batch of predictions from the production minutes bundle."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from projections import paths
from projections.minutes_v1.datasets import KEY_COLUMNS, deduplicate_latest
from projections.minutes_v1.production import load_production_minutes_bundle
from projections.models import minutes_lgbm as ml

app = typer.Typer(help=__doc__)


def _load_features(features_root: Path, day: datetime) -> pd.DataFrame:
    season_path = features_root / f"season={day.year}" / f"month={day.month:02d}" / "features.parquet"
    if not season_path.exists():
        raise FileNotFoundError(f"Features parquet missing at {season_path}")
    df = pd.read_parquet(season_path)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    slice_df = df[df["game_date"] == day.date()].copy()
    if slice_df.empty:
        raise RuntimeError(f"No feature rows found for {day.date()} in {season_path}")
    return slice_df


@app.command()
def main(
    date: datetime = typer.Option(..., help="Slate date (YYYY-MM-DD)."),
    rows: int = typer.Option(5, help="Number of rows to display."),
    features_root: Path = typer.Option(paths.data_path("gold", "features_minutes_v1"), help="Root of gold features."),
) -> None:
    features_root = features_root.expanduser().resolve()
    bundle = load_production_minutes_bundle()
    run_dir = bundle.get("run_dir")
    run_id = bundle.get("run_id")
    feature_cols = bundle["feature_columns"]

    typer.echo(f"Production bundle run_id={run_id} at {run_dir}")

    day_df = _load_features(features_root, date)
    filtered = ml._filter_out_players(day_df)
    deduped = deduplicate_latest(filtered, key_cols=KEY_COLUMNS, order_cols=["feature_as_of_ts"])
    preds = ml.modeling.predict_quantiles(bundle["quantiles"], deduped[feature_cols])
    p10_raw = np.minimum(preds[0.1], preds[0.5])
    p90_raw = np.maximum(preds[0.9], preds[0.5])
    p10_cal, p90_cal = bundle["calibrator"].calibrate(p10_raw, p90_raw)
    scored = deduped.copy()
    scored["p10_pred"] = p10_cal
    scored["p50_pred"] = preds[0.5]
    scored["p90_pred"] = p90_cal
    scored = ml.apply_conformal(
        scored,
        bundle["bucket_offsets"],
        mode=bundle["conformal_mode"],
        bucket_mode=bundle.get("bucket_mode", "none"),
    )
    scored["minutes_p10"] = scored["p10_adj"]
    scored["minutes_p50"] = scored["p50_adj"]
    scored["minutes_p90"] = scored["p90_adj"]
    preview_cols = [
        "game_id",
        "player_id",
        "team_id",
        "status",
        "minutes_p10",
        "minutes_p50",
        "minutes_p90",
    ]
    available = [col for col in preview_cols if col in scored.columns]
    typer.echo(scored[available].head(rows).to_string(index=False))


if __name__ == "__main__":  # pragma: no cover
    app()
