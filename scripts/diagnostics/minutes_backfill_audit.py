"""
Audit coverage of minutes predictions for historical backfill.

For each game_date in a range, reports:
  - num_games from schedule
  - injuries_snapshot coverage (has data <= tip_ts per game)
  - roster_nightly coverage
  - odds_snapshot coverage (optional)
  - features_minutes_v1 exists
  - projections_minutes_v1 exists and non-empty

Usage:
    uv run python -m scripts.diagnostics.minutes_backfill_audit \
        --data-root /home/daniel/projections-data \
        --start-date 2024-10-22 \
        --end-date 2025-02-01
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from projections.paths import data_path

app = typer.Typer(add_completion=False, help=__doc__)


def _season_from_date(d: pd.Timestamp) -> int:
    """NBA season start year (Aug-Jul)."""
    return d.year if d.month >= 8 else d.year - 1


def _iter_days(start: pd.Timestamp, end: pd.Timestamp):
    """Iterate over days in [start, end] inclusive."""
    day = start.normalize()
    end_day = end.normalize()
    while day <= end_day:
        yield day
        day += pd.Timedelta(days=1)


def _load_schedule(data_root: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Load schedule for the date range."""
    schedule_root = data_root / "silver" / "schedule"
    frames: list[pd.DataFrame] = []
    for season_dir in schedule_root.glob("season=*"):
        for month_dir in season_dir.glob("month=*"):
            path = month_dir / "schedule.parquet"
            if path.exists():
                frames.append(pd.read_parquet(path))
    if not frames:
        return pd.DataFrame(columns=["game_id", "game_date", "tip_ts"])
    df = pd.concat(frames, ignore_index=True)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.normalize()
    df = df[(df["game_date"] >= start) & (df["game_date"] <= end)]
    return df


def _load_silver_data(data_root: Path, dataset: str, season: int) -> pd.DataFrame:
    """Load all silver data for a dataset and season."""
    root = data_root / "silver" / dataset / f"season={season}"
    if not root.exists():
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for month_dir in root.glob("month=*"):
        for pq_file in month_dir.glob("*.parquet"):
            try:
                frames.append(pd.read_parquet(pq_file))
            except Exception:
                continue

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


@app.command()
def main(
    data_root: Optional[Path] = typer.Option(
        None, help="Root data directory (defaults to PROJECTIONS_DATA_ROOT)."
    ),
    start_date: str = typer.Option(..., help="Start date YYYY-MM-DD (inclusive)."),
    end_date: str = typer.Option(..., help="End date YYYY-MM-DD (inclusive)."),
    sample_games: int = typer.Option(
        3, help="Number of sample missing game_ids to show per date."
    ),
) -> None:
    root = data_root or data_path()
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()

    typer.echo(f"[audit] data_root: {root}")
    typer.echo(f"[audit] range: {start.date()} to {end.date()}")

    # Load schedule
    typer.echo("[audit] loading schedule...")
    schedule = _load_schedule(root, start, end)
    if schedule.empty:
        typer.echo("[audit] ERROR: no schedule data found")
        raise typer.Exit(1)

    # Group games by date
    schedule["game_date"] = pd.to_datetime(schedule["game_date"]).dt.normalize()
    schedule["tip_ts"] = pd.to_datetime(schedule["tip_ts"], utc=True, errors="coerce")
    games_by_date = schedule.groupby("game_date")["game_id"].apply(set).to_dict()

    # Pre-load silver datasets per season
    seasons = sorted(set(_season_from_date(d) for d in _iter_days(start, end)))
    silver_cache: dict[tuple[str, int], pd.DataFrame] = {}

    for season in seasons:
        for dataset in ["injuries_snapshot", "roster_nightly", "odds_snapshot"]:
            typer.echo(f"[audit] loading {dataset} for season={season}...")
            silver_cache[(dataset, season)] = _load_silver_data(root, dataset, season)

    # Audit each date
    results: list[dict] = []
    missing_features_dates: list[pd.Timestamp] = []
    missing_projections_dates: list[pd.Timestamp] = []

    for day in _iter_days(start, end):
        day_str = day.date().isoformat()
        season = _season_from_date(day)

        # Get games for this date
        game_ids = games_by_date.get(day, set())
        num_games = len(game_ids)

        if num_games == 0:
            continue  # Skip dates with no games

        # Check features_minutes_v1
        features_path = root / "gold" / "features_minutes_v1" / f"season={season}" / f"month={day.month:02d}" / "features.parquet"
        features_exists = features_path.exists()
        features_has_date = False
        if features_exists:
            try:
                feat_df = pd.read_parquet(features_path, columns=["game_date"])
                feat_df["game_date"] = pd.to_datetime(feat_df["game_date"]).dt.normalize()
                features_has_date = (feat_df["game_date"] == day).any()
            except Exception:
                pass

        # Check projections_minutes_v1
        proj_path1 = root / "gold" / "projections_minutes_v1" / day_str / "minutes.parquet"
        proj_path2 = root / "gold" / "projections_minutes_v1" / f"game_date={day_str}" / "minutes.parquet"
        proj_path = proj_path1 if proj_path1.exists() else (proj_path2 if proj_path2.exists() else None)
        proj_exists = proj_path is not None
        proj_rows = 0
        if proj_path:
            try:
                proj_df = pd.read_parquet(proj_path)
                proj_rows = len(proj_df)
            except Exception:
                pass

        # Check silver datasets coverage per game
        injuries_df = silver_cache.get(("injuries_snapshot", season), pd.DataFrame())
        roster_df = silver_cache.get(("roster_nightly", season), pd.DataFrame())
        odds_df = silver_cache.get(("odds_snapshot", season), pd.DataFrame())

        # Count games with coverage (has any row for game_id)
        def count_covered_games(df: pd.DataFrame, game_ids: set) -> tuple[int, list]:
            if df.empty or "game_id" not in df.columns:
                return 0, list(game_ids)[:sample_games]
            df_games = set(pd.to_numeric(df["game_id"], errors="coerce").dropna().astype(int).unique())
            covered = game_ids & df_games
            missing = list(game_ids - df_games)[:sample_games]
            return len(covered), missing

        injuries_covered, injuries_missing = count_covered_games(injuries_df, game_ids)
        roster_covered, roster_missing = count_covered_games(roster_df, game_ids)
        odds_covered, odds_missing = count_covered_games(odds_df, game_ids)

        result = {
            "game_date": day_str,
            "num_games": num_games,
            "injuries_coverage": f"{injuries_covered}/{num_games}",
            "roster_coverage": f"{roster_covered}/{num_games}",
            "odds_coverage": f"{odds_covered}/{num_games}",
            "features_exists": features_has_date,
            "proj_exists": proj_exists,
            "proj_rows": proj_rows,
            "needs_backfill": not features_has_date or proj_rows == 0,
        }
        results.append(result)

        if not features_has_date:
            missing_features_dates.append(day)
        if proj_rows == 0:
            missing_projections_dates.append(day)

    # Summary
    df = pd.DataFrame(results)
    total_dates = len(df)
    needs_backfill = df["needs_backfill"].sum()

    typer.echo(f"\n[audit] Summary: {total_dates} dates with games")
    typer.echo(f"[audit] Features missing: {len(missing_features_dates)}")
    typer.echo(f"[audit] Projections empty: {len(missing_projections_dates)}")
    typer.echo(f"[audit] Total needing backfill: {needs_backfill}")

    # Show dates needing backfill
    if needs_backfill > 0:
        typer.echo("\n[audit] Dates needing backfill:")
        backfill_df = df[df["needs_backfill"]]
        for _, row in backfill_df.iterrows():
            typer.echo(
                f"  {row['game_date']}: games={row['num_games']} "
                f"injuries={row['injuries_coverage']} roster={row['roster_coverage']} "
                f"odds={row['odds_coverage']} "
                f"features={row['features_exists']} proj_rows={row['proj_rows']}"
            )

    # Full coverage stats
    typer.echo("\n[audit] Input coverage summary:")
    for col in ["injuries_coverage", "roster_coverage", "odds_coverage"]:
        full_coverage = df[col].apply(lambda x: x.split("/")[0] == x.split("/")[1]).sum()
        typer.echo(f"  {col}: {full_coverage}/{total_dates} dates have full coverage")


if __name__ == "__main__":
    app()
