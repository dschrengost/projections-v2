"""
Force-rescore projections_minutes_v1 for a date range and immediately run projections-vs-labels debug.

Example (single-day rescore + debug):
    uv run python -m scripts.minutes.rescore_and_debug \
      --data-root  /home/daniel/projections-data \
      --start-date 2023-10-24 \
      --end-date   2023-10-24
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
import typer

import scripts.minutes.backfill_projections_minutes_v1 as backfill_mod
import scripts.minutes.debug_projections_vs_labels as debug_mod
from projections.paths import data_path

app = typer.Typer(add_completion=False)


@app.command()
def main(
    data_root: Optional[Path] = typer.Option(
        None, help="Root containing data (defaults to PROJECTIONS_DATA_ROOT or ./data)."
    ),
    start_date: str = typer.Option(..., help="Start date YYYY-MM-DD (inclusive)."),
    end_date: str = typer.Option(..., help="End date YYYY-MM-DD (inclusive)."),
    overwrite_existing: bool = typer.Option(
        True,
        "--overwrite-existing/--no-overwrite-existing",
        help="Force recompute projections even if outputs already exist (default True).",
    ),
    debug_describe: bool = typer.Option(
        True,
        "--debug-describe/--no-debug-describe",
        help="Print raw vs reconciled minutes_p50 describe() during scoring (default True).",
    ),
) -> None:
    root = data_root or data_path()
    start = pd.Timestamp(start_date).date()
    end = pd.Timestamp(end_date).date()

    typer.echo(
        f"[rescore-debug] scoring {start}..{end} overwrite_existing={overwrite_existing} debug_describe={debug_describe}"
    )

    backfill_mod.main(  # type: ignore[arg-type]
        start_date=start_date,
        end_date=end_date,
        data_root=root,
        season_type="Regular Season",
        overwrite_existing=overwrite_existing,
        bundle_run_id=None,
        debug_describe=debug_describe,
    )

    typer.echo(f"[rescore-debug] scoring complete; running projections_vs_labels debug for {start}..{end}")
    debug_mod.main(  # type: ignore[arg-type]
        data_root=root,
        start_date=start_date,
        end_date=end_date,
        season_type="Regular Season",
    )


if __name__ == "__main__":
    app()
