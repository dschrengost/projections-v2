"""
Quick sanity checks for gold/rates_training_base partitions.

Loads partitions over a date range (optionally filtered by season) and prints:
- Row counts (total, by season, by game_date)
- NaN fractions per column (marking >5%)
- Basic distributions for minutes_actual and per-minute targets
- Starter vs bench splits for select stats
- Small sampled rows for a couple of game dates

Examples:
- Quarter slice:
    uv run python -m scripts.rates.debug_training_base \
        --start-date 2024-01-01 \
        --end-date   2024-03-31 \
        --data-root  /home/daniel/projections-data

- Full season:
    uv run python -m scripts.rates.debug_training_base \
        --start-date 2023-10-01 \
        --end-date   2024-06-30 \
        --data-root  /home/daniel/projections-data
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import typer

from projections.paths import data_path

app = typer.Typer(add_completion=False)

TARGET_COLS = [
    "fga2_per_min",
    "fga3_per_min",
    "fta_per_min",
    "ast_per_min",
    "tov_per_min",
    "oreb_per_min",
    "dreb_per_min",
    "stl_per_min",
    "blk_per_min",
]


def _iter_partitions(
    root: Path, start: pd.Timestamp, end: pd.Timestamp, season: int | None
) -> list[Path]:
    partitions: list[Path] = []
    base = root / "gold" / "rates_training_base"
    if not base.exists():
        raise FileNotFoundError(f"Missing rates_training_base root at {base}")
    for season_dir in sorted(base.glob("season=*")):
        season_val = int(season_dir.name.split("=", 1)[1])
        if season is not None and season_val != season:
            continue
        for day_dir in season_dir.glob("game_date=*"):
            try:
                day = pd.Timestamp(day_dir.name.split("=", 1)[1]).normalize()
            except ValueError:
                continue
            if start <= day <= end:
                candidate = day_dir / "rates_training_base.parquet"
                if candidate.exists():
                    partitions.append(candidate)
    return partitions


def _load_partitions(paths: list[Path]) -> pd.DataFrame:
    if not paths:
        raise FileNotFoundError("No rates_training_base partitions matched the requested window.")
    frames = [pd.read_parquet(p) for p in paths]
    return pd.concat(frames, ignore_index=True)


def _nan_audit(df: pd.DataFrame) -> None:
    typer.echo("\n[NaN audit]")
    for col in df.columns:
        frac = float(df[col].isna().mean())
        marker = " !!" if frac > 0.05 else ""
        extra = " (100% missing)" if frac == 1.0 else ""
        typer.echo(f"{col:25s}: {frac:0.3%}{marker}{extra}")


def _describe_targets(df: pd.DataFrame) -> None:
    cols = ["minutes_actual"] + TARGET_COLS
    subset = df[cols]
    typer.echo("\n[Distributions]")
    typer.echo(subset.describe().to_string())


def _starter_splits(df: pd.DataFrame) -> None:
    typer.echo("\n[Starter vs bench]")
    df["fga_per_min_total"] = df["fga2_per_min"] + df["fga3_per_min"]
    df["reb_per_min_total"] = df["oreb_per_min"] + df["dreb_per_min"]
    counts = df["is_starter"].value_counts(dropna=False)
    typer.echo("counts by is_starter:")
    typer.echo(counts.to_string())
    grouped = (
        df.groupby("is_starter")[["minutes_actual", "fga_per_min_total", "ast_per_min", "reb_per_min_total"]]
        .agg(["mean", "std"])
        .rename_axis("is_starter")
    )
    typer.echo(grouped.to_string())


def _sample_rows(df: pd.DataFrame, rng: np.random.Generator) -> None:
    typer.echo("\n[Sample rows]")
    unique_dates = df["game_date"].dropna().unique()
    if len(unique_dates) == 0:
        typer.echo("No game_date values found for sampling.")
        return
    sample_dates = rng.choice(unique_dates, size=min(2, len(unique_dates)), replace=False)
    for date in sample_dates:
        slice_df = df[df["game_date"] == date]
        teams = slice_df["team_id"].unique()
        team_sample = rng.choice(teams, size=min(2, len(teams)), replace=False)
        display_cols = [
            "game_date",
            "team_id",
            "opponent_id",
            "player_id",
            "position_primary",
            "is_starter",
            "minutes_actual",
            "fga2_per_min",
            "fga3_per_min",
            "ast_per_min",
            "oreb_per_min",
            "dreb_per_min",
        ]
        typer.echo(f"\nDate: {pd.Timestamp(date).date().isoformat()}")
        for team in team_sample:
            typer.echo(f"  Team {team}:")
            sample = slice_df[slice_df["team_id"] == team].head(5)[display_cols]
            typer.echo(sample.to_string(index=False))


@app.command()
def main(
    start_date: str = typer.Option(..., help="Start date YYYY-MM-DD (inclusive)"),
    end_date: str = typer.Option(..., help="End date YYYY-MM-DD (inclusive)"),
    season: Optional[int] = typer.Option(None, help="Optional season filter (e.g., 2025)"),
    data_root: Optional[Path] = typer.Option(
        None, help="Root containing gold/rates_training_base (defaults to PROJECTIONS_DATA_ROOT or ./data)."
    ),
) -> None:
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    root = data_root or data_path()
    typer.echo(f"[debug] loading rates_training_base from {start.date()} to {end.date()} (season={season})")
    paths = _iter_partitions(root, start, end, season)
    df = _load_partitions(paths)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.normalize()

    typer.echo(f"[debug] total rows: {len(df):,}")
    typer.echo("[debug] rows by season:")
    typer.echo(df.groupby("season").size().to_string())
    typer.echo("[debug] rows by game_date:")
    typer.echo(df.groupby("game_date").size().to_string())

    _nan_audit(df)
    _describe_targets(df)
    _starter_splits(df)
    _sample_rows(df, np.random.default_rng(seed=0))


if __name__ == "__main__":
    app()
