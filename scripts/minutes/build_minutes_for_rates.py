"""
Materialize minutes model predictions for rates training.

Reads existing minutes projections (production bundle outputs) **per game_date** from
``gold/projections_minutes_v1/game_date=YYYY-MM-DD/minutes.parquet`` and writes a thin
table keyed for rates joins:
- season, game_id, game_date, team_id, player_id
- minutes_pred_p10, minutes_pred_p50, minutes_pred_p90, minutes_pred_play_prob

Output (default):
- gold/minutes_for_rates/season=YYYY/game_date=YYYY-MM-DD/minutes_for_rates.parquet

Behavior:
- Loop every date in [start_date, end_date].
- When the per-date projections parquet exists, select the minutes/play_prob columns and
  write the matching ``minutes_for_rates`` partition.
- When the projections parquet is missing, log a warning and continue (do not crash).
- At the end, log how many dates were written vs. skipped due to missing projections.

Example:
    uv run python -m scripts.minutes.build_minutes_for_rates \\
        --start-date 2023-10-01 \\
        --end-date   2025-11-26 \\
        --data-root  /home/daniel/projections-data \\
        --output-root /home/daniel/projections-data/gold/minutes_for_rates
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from projections.paths import data_path

app = typer.Typer(add_completion=False)


def _season_from_day(day: pd.Timestamp) -> int:
    return day.year if day.month >= 8 else day.year - 1


def _iter_days(start: pd.Timestamp, end: pd.Timestamp):
    cur = start
    while cur <= end:
        yield cur
        cur += pd.Timedelta(days=1)


def _source_path(data_root: Path, day: pd.Timestamp) -> Path:
    date_token = day.date().isoformat()
    return data_root / "gold" / "projections_minutes_v1" / f"game_date={date_token}" / "minutes.parquet"


def _load_minutes(data_root: Path, day: pd.Timestamp) -> pd.DataFrame | None:
    path = _source_path(data_root, day)
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.normalize()
    df["season"] = df["game_date"].apply(_season_from_day)
    df.rename(
        columns={
            "minutes_p10": "minutes_pred_p10",
            "minutes_p50": "minutes_pred_p50",
            "minutes_p90": "minutes_pred_p90",
            "play_prob": "minutes_pred_play_prob",
        },
        inplace=True,
    )
    keep = [
        "season",
        "game_id",
        "game_date",
        "team_id",
        "player_id",
        "minutes_pred_p10",
        "minutes_pred_p50",
        "minutes_pred_p90",
        "minutes_pred_play_prob",
    ]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise KeyError(f"Source minutes missing columns: {missing} at {path}")
    return df[keep]


def _write_partition(df: pd.DataFrame, output_root: Path) -> None:
    grouped = df.groupby(["season", "game_date"])
    for (season, game_date), frame in grouped:
        out_dir = output_root / f"season={int(season)}" / f"game_date={pd.Timestamp(game_date).date().isoformat()}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "minutes_for_rates.parquet"
        frame.to_parquet(out_path, index=False)


@app.command()
def main(
    start_date: str = typer.Option(..., help="Start date YYYY-MM-DD (inclusive)"),
    end_date: str = typer.Option(..., help="End date YYYY-MM-DD (inclusive)"),
    data_root: Optional[Path] = typer.Option(None, help="Data root (defaults PROJECTIONS_DATA_ROOT or ./data)"),
    output_root: Optional[Path] = typer.Option(
        None, help="Output root (defaults to <data_root>/gold/minutes_for_rates)"
    ),
) -> None:
    root = data_root or data_path()
    out_root = output_root or (root / "gold" / "minutes_for_rates")
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()

    typer.echo(
        f"[minutes_for_rates] scoring window {start.date()} to {end.date()} "
        f"from projections_minutes_v1 into {out_root}"
    )
    written_dates = 0
    skipped_missing = 0
    total_rows = 0
    for day in _iter_days(start, end):
        df = _load_minutes(root, day)
        if df is None:
            typer.echo(
                f"[minutes_for_rates] {day.date()}: missing gold/projections_minutes_v1 parquet; skipping.",
                err=True,
            )
            skipped_missing += 1
            continue
        _write_partition(df, out_root)
        written_dates += 1
        total_rows += len(df)
    if written_dates == 0:
        typer.echo("[minutes_for_rates] no source minutes found; nothing written.")
        return
    typer.echo(
        f"[minutes_for_rates] wrote {total_rows:,} rows across {written_dates} dates into {out_root}"
    )
    typer.echo(
        f"[minutes_for_rates] summary written_dates={written_dates} skipped_missing_projections={skipped_missing}"
    )


if __name__ == "__main__":
    app()
