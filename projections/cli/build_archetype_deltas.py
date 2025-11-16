"""Build archetype delta artifacts for Minutes V1 training."""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
import typer

from projections import paths
from projections.archetypes.deltas import ArchetypeDeltaConfig, build_archetype_deltas, load_config

app = typer.Typer(help=__doc__)


def _candidate_season_partitions(season: str) -> list[str]:
    normalized = season.strip()
    parts = [normalized]
    if "-" in normalized:
        start = normalized.split("-", 1)[0]
        if start and start not in parts:
            parts.append(start)
    return parts


def _seasoned_path(root: Path, season: str, filename: str) -> Path:
    for candidate in _candidate_season_partitions(season):
        path = root / f"season={candidate}" / filename
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing {filename} for season '{season}' under {root}")


@app.command()
def main(
    seasons: List[str] = typer.Option(..., "--seasons", help="Season labels to process."),
    labels_root: Path = typer.Option(
        paths.data_path("labels"), help="Root dir housing season=YYYY label parquets."
    ),
    injury_root: Path = typer.Option(
        paths.data_path("gold", "injury_features"),
        help="Root dir for injury_features season partitions.",
    ),
    roles_root: Path = typer.Option(
        paths.data_path("gold", "minutes_roles"), help="Root dir for roles season partitions."
    ),
    out_root: Path = typer.Option(
        paths.data_path("gold", "features_minutes_v1"), help="Destination root for archetype delta artifacts."
    ),
    config_path: Path | None = typer.Option(
        Path("config/minutes_archetype_deltas.yaml"),
        "--config",
        help="YAML config overriding delta thresholds.",
    ),
) -> None:
    """Compute (role_p, role_t) minutes deltas for each requested season."""

    if not seasons:
        raise typer.BadParameter("At least one --seasons value must be supplied.")
    config = load_config(config_path)
    for season in seasons:
        labels = pd.read_parquet(_seasoned_path(labels_root, season, "boxscore_labels.parquet"))
        injuries = pd.read_parquet(_seasoned_path(injury_root, season, "injury_features.parquet"))
        roles = pd.read_parquet(_seasoned_path(roles_root, season, "roles.parquet"))
        deltas = build_archetype_deltas(
            labels,
            injuries,
            roles,
            season_label=season,
            config=config,
        )
        dest = out_root / f"season={season}" / "archetype_deltas.parquet"
        dest.parent.mkdir(parents=True, exist_ok=True)
        deltas.to_parquet(dest, index=False)
        typer.echo(f"[deltas] wrote {len(deltas):,} rows to {dest}")


if __name__ == "__main__":  # pragma: no cover
    app()
