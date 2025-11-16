"""CLI to build starter slot priors for promotion calibration."""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
import typer

from projections import paths
from projections.minutes_v1.prior_starters import (
    StarterPriorArtifacts,
    build_player_starter_history,
    build_starter_slot_priors,
    load_feature_frames,
)

app = typer.Typer(help=__doc__)

DEFAULT_FEATURES_ROOT = paths.data_path("gold", "features_minutes_v1")
DEFAULT_OUT_ROOT = paths.data_path("gold", "minutes_priors")


@app.command()
def main(
    features: List[Path] = typer.Option(
        [],
        "--features",
        help="Explicit parquet file(s) or directories containing features.",
    ),
    features_root: Path = typer.Option(
        DEFAULT_FEATURES_ROOT,
        "--features-root",
        help="Root directory scanned for season=*/month=*/features.parquet when --features is empty.",
    ),
    out_path: Path = typer.Option(
        DEFAULT_OUT_ROOT / "starter_slot_priors.parquet",
        "--out",
        help="Where to write starter slot priors parquet.",
    ),
    history_out: Path = typer.Option(
        DEFAULT_OUT_ROOT / "player_starter_history.parquet",
        "--history-out",
        help="Where to write per-player starter history parquet.",
    ),
    min_minutes: float = typer.Option(
        0.0,
        "--min-minutes",
        help="Exclude starts with minutes <= this threshold when building priors.",
    ),
) -> None:
    if not features:
        if not features_root.exists():
            raise typer.BadParameter(f"--features-root {features_root} does not exist", param_name="features_root")
        candidate_paths = sorted(features_root.rglob("features.parquet"))
        if not candidate_paths:
            raise typer.BadParameter(
                f"No features.parquet files found under {features_root}", param_name="features"
            )
        feature_paths = candidate_paths
    else:
        feature_paths = features
    frames = load_feature_frames([path.expanduser() for path in feature_paths])
    slot_priors = build_starter_slot_priors(frames, min_minutes=min_minutes)
    history = build_player_starter_history(frames)

    out_path = out_path.expanduser()
    history_out = history_out.expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    history_out.parent.mkdir(parents=True, exist_ok=True)
    slot_priors.to_parquet(out_path, index=False)
    history.to_parquet(history_out, index=False)
    typer.echo(f"[starter-priors] wrote {len(slot_priors)} priors -> {out_path}")
    typer.echo(f"[starter-priors] wrote {len(history)} player history rows -> {history_out}")


if __name__ == "__main__":  # pragma: no cover
    app()
