#!/usr/bin/env python3
"""Bronze retention/compaction placeholder (safe post-gold-freeze).

Do not delete bronze history needed to reproduce existing gold slates. Once gold
manifests are stable, this script can be extended to prune/compact partitions
older than a rolling window while keeping any snapshot referenced by manifests.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import typer

from projections import paths

app = typer.Typer(help=__doc__)


def cleanup_old_bronze_partitions(data_root: Path, *, days_to_keep: int = 30) -> None:
    """Prune/compact bronze partitions older than N days (post-gold freeze)."""
    if days_to_keep < 0:
        raise ValueError("days_to_keep must be >= 0")
    _ = date.today() - timedelta(days=days_to_keep)
    raise NotImplementedError(
        "Retention is intentionally deferred until gold/slates manifests are the source of truth."
    )


@app.command()
def main(
    data_root: Path = typer.Option(
        paths.get_data_root(),
        "--data-root",
        help="Base data directory (defaults to PROJECTIONS_DATA_ROOT or ./data).",
    ),
    days_to_keep: int = typer.Option(30, "--days-to-keep", min=0, help="Rolling window to keep in bronze."),
) -> None:
    cleanup_old_bronze_partitions(data_root.resolve(), days_to_keep=days_to_keep)


if __name__ == "__main__":  # pragma: no cover
    app()
