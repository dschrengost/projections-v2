"""
Build season-to-date tracking features and role clusters for rates training.

Reads bronze/nba/tracking per game_date, computes pre-game season aggregates, assigns
coarse role clusters, and writes gold/tracking_roles partitions.

Example:
    uv run python -m scripts.tracking.build_tracking_roles \\
        --data-root  /home/daniel/projections-data \\
        --start-date 2023-10-01 \\
        --end-date   2025-11-26
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from projections.paths import data_path
from projections.tracking.roles import (
    assign_tracking_roles,
    compute_cumulative_tracking,
    load_game_lookup,
    load_tracking_window,
    write_tracking_partitions,
)

app = typer.Typer(add_completion=False)


@app.command()
def main(
    start_date: str = typer.Option(..., help="Start date YYYY-MM-DD (inclusive)."),
    end_date: str = typer.Option(..., help="End date YYYY-MM-DD (inclusive)."),
    data_root: Optional[Path] = typer.Option(None, help="Root containing tracking data (defaults PROJECTIONS_DATA_ROOT or ./data)."),
    output_root: Optional[Path] = typer.Option(
        None, help="Output root for gold/tracking_roles (defaults under data_root/gold)."
    ),
    overwrite_existing: bool = typer.Option(
        True,
        "--overwrite-existing/--no-overwrite-existing",
        help="Overwrite existing tracking_roles partitions (default True).",
    ),
) -> None:
    root = data_root or data_path()
    out_root = output_root or (root / "gold" / "tracking_roles")
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()

    typer.echo(
        f"[tracking_roles] building from {start.date()} to {end.date()} data_root={root} out_root={out_root}"
    )
    game_lookup = load_game_lookup(root, start, end)
    if game_lookup.empty:
        typer.echo("[tracking_roles] warning: no labels/minutes found for game_id mapping; game_id may be missing.")
    raw = load_tracking_window(root, start, end, game_lookup)
    if raw.empty:
        typer.echo("[tracking_roles] no tracking data found in window; nothing written.")
        raise typer.Exit()

    typer.echo(f"[tracking_roles] loaded {len(raw):,} tracking rows across {raw['game_date'].nunique()} dates")
    cumulative = compute_cumulative_tracking(raw)
    with_roles = assign_tracking_roles(cumulative)
    written, skipped = write_tracking_partitions(with_roles, out_root, overwrite_existing=overwrite_existing)
    typer.echo(
        f"[tracking_roles] wrote_dates={written} skipped_existing={skipped} rows={len(with_roles):,} into {out_root}"
    )


if __name__ == "__main__":
    app()
