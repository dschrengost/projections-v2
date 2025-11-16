"""Build role_key assignments used by archetype delta features."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import pandas as pd
import typer

from projections import paths
from projections.archetypes.roles import build_roles_table

app = typer.Typer(help=__doc__)


def _candidate_season_partitions(season: str) -> list[str]:
    normalized = season.strip()
    parts = [normalized]
    if "-" in normalized:
        start = normalized.split("-", 1)[0]
        if start and start not in parts:
            parts.append(start)
    return parts


def _labels_path(labels_root: Path, season: str) -> Path:
    for candidate in _candidate_season_partitions(season):
        path = labels_root / f"season={candidate}" / "boxscore_labels.parquet"
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No labels parquet found for season '{season}' under {labels_root}."
    )


def _read_parquet_tree(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing parquet input at {path}")
    if path.is_file():
        return pd.read_parquet(path)
    files = sorted(path.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {path}")
    frames = [pd.read_parquet(file) for file in files]
    return pd.concat(frames, ignore_index=True)


def _filter_roster(roster: pd.DataFrame | None, game_ids: Iterable[int]) -> pd.DataFrame | None:
    if roster is None or roster.empty:
        return None
    normalized_ids = pd.Series(game_ids, dtype="Int64").dropna().astype(int).tolist()
    if not normalized_ids:
        return None
    roster = roster.copy()
    roster["game_id"] = pd.to_numeric(roster["game_id"], errors="coerce")
    mask = roster["game_id"].isin(normalized_ids)
    sliced = roster.loc[mask].copy()
    return sliced if not sliced.empty else None


@app.command()
def main(
    seasons: List[str] = typer.Option(..., "--seasons", help="Season labels to process."),
    labels_root: Path = typer.Option(
        paths.data_path("labels"),
        help="Root directory containing season=YYYY label partitions.",
    ),
    roster_root: Path | None = typer.Option(
        paths.data_path("silver", "roster_nightly"),
        help="Optional roster_nightly parquet dir for resolving listed_pos (default: <data_root>/silver/roster_nightly).",
    ),
    out_root: Path = typer.Option(
        paths.data_path("gold", "minutes_roles"),
        help="Output root for season=YYYY/roles.parquet artifacts.",
    ),
) -> None:
    """Compute position_group/starter_tier role keys for each requested season."""

    if not seasons:
        raise typer.BadParameter("At least one --season value must be provided.")

    roster_df: pd.DataFrame | None = None
    if roster_root is not None and roster_root.exists():
        roster_df = _read_parquet_tree(roster_root)
    elif roster_root is not None:
        typer.echo(f"[roles] roster root {roster_root} not found; continuing without position hints.")

    for season in seasons:
        labels_path = _labels_path(labels_root, season)
        labels = pd.read_parquet(labels_path)
        if labels.empty:
            typer.echo(f"[roles] warning: labels slice for season {season} is empty; skipping.")
            continue
        roster_slice = _filter_roster(roster_df, labels["game_id"].unique().tolist())
        roles = build_roles_table(labels, season_label=season, roster_nightly=roster_slice)
        dest = out_root / f"season={season}" / "roles.parquet"
        dest.parent.mkdir(parents=True, exist_ok=True)
        roles.to_parquet(dest, index=False)
        typer.echo(f"[roles] wrote {len(roles):,} rows to {dest}")


if __name__ == "__main__":  # pragma: no cover
    app()
