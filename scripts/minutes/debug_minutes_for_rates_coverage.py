"""
Inspect minutes_for_rates coverage over a date range.

For each date, report whether a ``gold/minutes_for_rates`` parquet exists and the row
count. Summarizes present vs. missing dates and prints a small table for quick visual
inspection.

Example:
    uv run python -m scripts.minutes.debug_minutes_for_rates_coverage \\
        --data-root  /home/daniel/projections-data \\
        --start-date 2023-10-01 \\
        --end-date   2025-11-26
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from projections.cli.score_minutes_v1 import _season_from_date
from projections.paths import data_path

app = typer.Typer(add_completion=False)


def _iter_days(start: pd.Timestamp, end: pd.Timestamp):
    cur = start
    while cur <= end:
        yield cur
        cur += pd.Timedelta(days=1)


def _minutes_for_rates_path(root: Path, day: pd.Timestamp) -> Path:
    season = _season_from_date(day.date())
    token = day.date().isoformat()
    return root / "gold" / "minutes_for_rates" / f"season={season}" / f"game_date={token}" / "minutes_for_rates.parquet"


def _safe_row_count(path: Path) -> int | None:
    try:
        return len(pd.read_parquet(path))
    except Exception:
        return None


@app.command()
def main(
    start_date: str = typer.Option(..., help="Start date YYYY-MM-DD (inclusive)"),
    end_date: str = typer.Option(..., help="End date YYYY-MM-DD (inclusive)"),
    data_root: Optional[Path] = typer.Option(None, help="Data root (defaults PROJECTIONS_DATA_ROOT or ./data)"),
) -> None:
    root = data_root or data_path()
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    typer.echo(
        f"[coverage] scanning minutes_for_rates from {start.date()} to {end.date()} under {root}"
    )

    records: list[dict[str, object]] = []
    for day in _iter_days(start, end):
        path = _minutes_for_rates_path(root, day)
        exists = path.exists()
        records.append(
            {
                "game_date": day.date(),
                "season": _season_from_date(day.date()),
                "exists": exists,
                "row_count": _safe_row_count(path) if exists else None,
            }
        )

    total_dates = len(records)
    present = sum(1 for r in records if r["exists"])
    missing = total_dates - present
    typer.echo(f"[coverage] dates={total_dates} present={present} missing={missing}")
    if total_dates == 0:
        return

    df = pd.DataFrame.from_records(records)
    df.sort_values("game_date", inplace=True)
    missing_df = df[~df["exists"]]

    preview = pd.concat([df.head(3), df.tail(3)], ignore_index=True).drop_duplicates(subset=["game_date"])
    if not missing_df.empty:
        # Include all missing days if manageable; otherwise preview head/tail to keep output small.
        if len(missing_df) > 20:
            missing_preview = pd.concat([missing_df.head(10), missing_df.tail(10)], ignore_index=True)
        else:
            missing_preview = missing_df
        preview = (
            pd.concat([preview, missing_preview], ignore_index=True)
            .drop_duplicates(subset=["game_date"])
            .sort_values("game_date")
        )

    print_df = preview[["game_date", "exists", "row_count"]].rename(columns={"exists": "present"})
    typer.echo("\ngame_date    present  row_count")
    typer.echo(print_df.to_string(index=False))

    if not missing_df.empty:
        missing_list = [str(d.date()) if isinstance(d, pd.Timestamp) else str(d) for d in missing_df["game_date"]]
        sample_missing = ", ".join(missing_list[:10])
        suffix = "..." if len(missing_list) > 10 else ""
        typer.echo(f"\n[coverage] missing dates ({len(missing_list)}): {sample_missing}{suffix}")


if __name__ == "__main__":
    app()
