"""
Check coverage of gold/projections_minutes_v1 over a date window.

Reports which dates have gold projections minutes partitions and whether daily
artifacts exist for missing dates.

Example:
    PROJECTIONS_DATA_ROOT=/home/daniel/projections-data \\
    uv run python -m scripts.minutes.check_projections_minutes_v1_coverage \\
        --start-date 2023-10-24 \\
        --end-date   2025-12-01
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from projections.cli import score_minutes_v1
from projections.paths import data_path

app = typer.Typer(add_completion=False)


def _iter_days(start: pd.Timestamp, end: pd.Timestamp):
    cur = start
    while cur <= end:
        yield cur
        cur += pd.Timedelta(days=1)


def _gold_paths(root: Path, day: pd.Timestamp) -> list[Path]:
    token = day.date().isoformat()
    return [
        root / "gold" / "projections_minutes_v1" / f"game_date={token}" / score_minutes_v1.OUTPUT_FILENAME,
        root / "gold" / "projections_minutes_v1" / token / score_minutes_v1.OUTPUT_FILENAME,
    ]


def _daily_path(day: pd.Timestamp) -> Path:
    return Path(score_minutes_v1.DEFAULT_DAILY_ROOT) / day.date().isoformat() / score_minutes_v1.OUTPUT_FILENAME


@app.command()
def main(
    start_date: str = typer.Option(..., help="Start date YYYY-MM-DD (inclusive)"),
    end_date: str = typer.Option(..., help="End date YYYY-MM-DD (inclusive)"),
    data_root: Optional[Path] = typer.Option(
        None, help="Data root (defaults to PROJECTIONS_DATA_ROOT or ./data)"
    ),
) -> None:
    root = data_root or data_path()
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()

    records: list[dict[str, object]] = []
    for day in _iter_days(start, end):
        gold_paths = _gold_paths(root, day)
        gold_exists = any(p.exists() for p in gold_paths)
        row_count = None
        if gold_exists:
            for p in gold_paths:
                if p.exists():
                    try:
                        row_count = len(pd.read_parquet(p))
                        break
                    except Exception:
                        row_count = None
                        break
        daily_path = _daily_path(day)
        daily_exists = daily_path.exists()
        records.append(
            {
                "game_date": day.date(),
                "gold_exists": gold_exists,
                "row_count": row_count,
                "daily_exists": daily_exists,
            }
        )

    df = pd.DataFrame.from_records(records)
    total = len(df)
    present = int(df["gold_exists"].sum())
    missing = total - present
    typer.echo(f"[coverage] dates={total} present={present} missing={missing}")

    missing_df = df[~df["gold_exists"]]
    if missing:
        sample = pd.concat([missing_df.head(5), missing_df.tail(5)]).drop_duplicates(subset=["game_date"])
        typer.echo("\nMissing dates (sample):")
        typer.echo(sample.to_string(index=False))
        missing_with_daily = missing_df[missing_df["daily_exists"]]
        typer.echo(f"\nMissing but daily artifacts present: {len(missing_with_daily)}")
    else:
        typer.echo("[coverage] no missing dates.")


if __name__ == "__main__":
    app()
