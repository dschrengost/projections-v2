"""
Validate minutes_for_rates coverage over a date window.

Loads gold/minutes_for_rates partitions and reports missing days plus NULL counts
for minutes_pred_* columns.

Example:
    uv run python -m scripts.rates.check_minutes_for_rates_coverage \\
        --start-date 2023-10-24 \\
        --end-date   2025-12-01
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from projections.paths import data_path

app = typer.Typer(add_completion=False)


def _iter_days(start: pd.Timestamp, end: pd.Timestamp):
    cur = start
    while cur <= end:
        yield cur
        cur += pd.Timedelta(days=1)


def _season_from_day(day: pd.Timestamp) -> int:
    return day.year if day.month >= 8 else day.year - 1


def _partition_path(root: Path, day: pd.Timestamp) -> Path:
    token = day.date().isoformat()
    season = _season_from_day(day)
    return root / "gold" / "minutes_for_rates" / f"season={season}" / f"game_date={token}" / "minutes_for_rates.parquet"


@app.command()
def main(
    start_date: str = typer.Option(..., help="Start date YYYY-MM-DD (inclusive)"),
    end_date: str = typer.Option(..., help="End date YYYY-MM-DD (inclusive)"),
    data_root: Optional[Path] = typer.Option(None, help="Data root (defaults to PROJECTIONS_DATA_ROOT or ./data)"),
) -> None:
    root = data_root or data_path()
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()

    frames: list[pd.DataFrame] = []
    missing: list[str] = []
    for day in _iter_days(start, end):
        path = _partition_path(root, day)
        if not path.exists():
            missing.append(day.date().isoformat())
            continue
        frames.append(pd.read_parquet(path))

    if missing:
        typer.echo(f"[coverage] missing {len(missing)} dates (sample: {', '.join(missing[:10])}{'...' if len(missing) > 10 else ''})")
    else:
        typer.echo("[coverage] no missing partitions in window.")

    if not frames:
        typer.echo("[coverage] no data loaded; aborting.")
        raise typer.Exit(code=1)

    df = pd.concat(frames, ignore_index=True)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.normalize()
    total_rows = len(df)
    pred_cols = ["minutes_pred_p10", "minutes_pred_p50", "minutes_pred_p90", "minutes_pred_play_prob"]
    na_counts = {col: int(df[col].isna().sum()) for col in pred_cols if col in df.columns}

    typer.echo(f"[coverage] rows={total_rows:,} dates={len(frames)} window={start.date()}..{end.date()}")
    for col, count in na_counts.items():
        typer.echo(f"[coverage] {col} nulls: {count}")

    if na_counts.get("minutes_pred_p50", 0) > 0:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
