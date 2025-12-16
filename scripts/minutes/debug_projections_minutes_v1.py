"""
Audit coverage for gold/projections_minutes_v1 between two dates.

Example:
    uv run python -m scripts.minutes.debug_projections_minutes_v1 \
      --data-root  /home/daniel/projections-data \
      --start-date 2023-10-01 \
      --end-date   2024-06-30

Force-rescore + debug in one step:
    uv run python -m scripts.minutes.rescore_and_debug \
      --data-root /home/daniel/projections-data \
      --start-date 2023-10-24 \
      --end-date   2023-10-24
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from projections.paths import data_path

app = typer.Typer(add_completion=False)


def _iter_days(start: date, end: date):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def _load_partition(root: Path, day: date) -> pd.DataFrame | None:
    iso = day.isoformat()
    candidates = [
        root / f"game_date={iso}" / "minutes.parquet",
        root / iso / "minutes.parquet",
    ]
    for path in candidates:
        if path.exists():
            return pd.read_parquet(path)
    return None


def _nan_frac(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns or len(df) == 0:
        return 100.0
    return float(df[col].isna().mean() * 100)


@app.command()
def main(
    data_root: Optional[Path] = typer.Option(
        None, help="Root containing data (defaults to PROJECTIONS_DATA_ROOT or ./data)."
    ),
    start_date: str = typer.Option(..., help="Start date YYYY-MM-DD (inclusive)."),
    end_date: str = typer.Option(..., help="End date YYYY-MM-DD (inclusive)."),
) -> None:
    root = data_root or data_path()
    start = pd.Timestamp(start_date).date()
    end = pd.Timestamp(end_date).date()
    proj_root = root / "gold" / "projections_minutes_v1"

    frames: list[pd.DataFrame] = []
    per_day_counts: list[tuple[date, int]] = []
    for day in _iter_days(start, end):
        df = _load_partition(proj_root, day)
        if df is None:
            continue
        df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
        per_day_counts.append((day, len(df)))
        frames.append(df)

    if not frames:
        typer.echo("[debug] no projections_minutes_v1 partitions found in the requested window.")
        raise typer.Exit()

    all_df = pd.concat(frames, ignore_index=True)
    seasons = all_df["season"] if "season" in all_df.columns else all_df["game_date"].apply(lambda d: d.year)

    typer.echo(f"[debug] total rows: {len(all_df):,}")
    typer.echo(f"[debug] distinct seasons: {len(pd.unique(seasons))}")
    typer.echo(f"[debug] distinct game_dates: {all_df['game_date'].nunique()}")
    if "game_id" in all_df.columns:
        typer.echo(f"[debug] distinct game_ids: {all_df['game_id'].nunique()}")

    for col in ["minutes_p10", "minutes_p50", "minutes_p90", "play_prob"]:
        typer.echo(f"[NaN] {col:<15}: {_nan_frac(all_df, col):.3f}%")

    typer.echo("\n[rows by game_date]")
    for day, count in sorted(per_day_counts, key=lambda x: x[0]):
        typer.echo(f"{day.isoformat()}: {count}")

    typer.echo("\n[distributions]")
    for col in ["minutes_p50", "play_prob"]:
        if col in all_df.columns:
            typer.echo(f"{col} describe():")
            typer.echo(all_df[col].describe().to_string())
        else:
            typer.echo(f"{col} missing from projections.")


if __name__ == "__main__":
    app()
