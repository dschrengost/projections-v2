"""
Inspect minutes_v1 feature slice for a single game_date.

Example:
    uv run python -m scripts.minutes.debug_features_for_date \
      --data-root  /home/daniel/projections-data \
      --game-date  2023-10-24 \
      --season-type "Regular Season"
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from projections.cli import score_minutes_v1
from projections.paths import data_path

app = typer.Typer(add_completion=False)


@app.command()
def main(
    data_root: Optional[Path] = typer.Option(
        None, help="Root containing data (defaults to PROJECTIONS_DATA_ROOT or ./data)."
    ),
    game_date: str = typer.Option(..., help="Game date YYYY-MM-DD."),
    season_type: str = typer.Option("Regular Season", help="Season type (placeholder for logging)."),
) -> None:
    root = data_root or data_path()
    day = pd.Timestamp(game_date).date()
    features_root = root / "gold" / "features_minutes_v1"

    # Load production bundle to retrieve feature columns.
    bundle, _, _, _ = score_minutes_v1._resolve_bundle_artifacts(
        bundle_dir=None,
        config_path=score_minutes_v1.DEFAULT_BUNDLE_CONFIG,
    )
    feature_cols = bundle.get("feature_columns") or bundle.get("features") or []
    if not feature_cols:
        typer.echo("[features-debug] bundle missing feature_columns; aborting.", err=True)
        raise typer.Exit(code=1)

    try:
        features = score_minutes_v1._load_feature_slice(
            day,
            day,
            features_root=features_root,
            features_path=None,
            run_id=None,
        )
    except FileNotFoundError as exc:
        typer.echo(f"[features-debug] ERROR loading features: {exc}", err=True)
        raise typer.Exit(code=1)

    features["game_date"] = pd.to_datetime(features["game_date"]).dt.date
    day_slice = features.loc[features["game_date"] == day].copy()
    if day_slice.empty:
        typer.echo(f"[features-debug] game_date={day} has zero rows in features_minutes_v1.", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"[features-debug] game_date={day} rows={len(day_slice)} season_type='{season_type}'")
    typer.echo(f"[features-debug] first 20 columns: {day_slice.columns.tolist()[:20]}")

    used_feats = [c for c in feature_cols if c in day_slice.columns]
    typer.echo(f"[features-debug] using {len(used_feats)} / {len(feature_cols)} feature columns present in df")
    if not used_feats:
        raise typer.Exit(code=1)

    nan_frac = day_slice[used_feats].isna().mean().sort_values(ascending=False)
    std_vals = day_slice[used_feats].std(numeric_only=True).sort_values(ascending=False)

    typer.echo("\n[NaN fraction top 20]")
    typer.echo(nan_frac.head(20).to_string())

    typer.echo("\n[Std (variance) top 20]")
    typer.echo(std_vals.head(20).to_string())

    typer.echo("\n[Std (variance) bottom 20]")
    typer.echo(std_vals.tail(20).to_string())


if __name__ == "__main__":
    app()
