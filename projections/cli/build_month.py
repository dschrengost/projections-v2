"""One-shot builder that materializes bronze→gold artifacts for one or more months."""

from __future__ import annotations

import calendar
from datetime import datetime
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd
import typer

from projections import paths
from projections.minutes_v1.smoke_dataset import SmokeDatasetBuilder
from projections.pipelines import build_features_minutes_v1 as features_cli

app = typer.Typer(help=__doc__)


def _month_bounds(year: int, month: int) -> tuple[datetime, datetime]:
    if month < 1 or month > 12:
        raise typer.BadParameter("month must be in [1, 12]", param_hint="month")
    last_day = calendar.monthrange(year, month)[1]
    start = datetime(year, month, 1)
    end = datetime(year, month, last_day)
    return start, end


def _season_strings(season_start: int, season_label: str | None, season_year: str | None) -> tuple[str, str]:
    label = season_label or str(season_start)
    year_suffix = season_year or f"{season_start}-{(season_start + 1) % 100:02d}"
    return label, year_suffix


def _default_json_paths(
    data_root: Path,
    season_year_slug: str,
    season_start: int,
) -> tuple[Path, Path, Path, Path]:
    injuries = data_root / f"nba_injuries_{season_year_slug}.json"
    schedule = data_root / f"nba_schedule_{season_year_slug}.json"
    boxscores = data_root / "raw" / f"nba_boxscores_{season_year_slug}.json"
    odds = data_root / "raw" / f"oddstrader_season-{season_start}-{season_start + 1}.json"
    return injuries, schedule, odds, boxscores


def _month_range(
    start_year: int,
    start_month: int,
    end_year: int,
    end_month: int,
) -> Iterable[Tuple[int, int]]:
    start_period = pd.Period(f"{start_year}-{start_month:02d}", freq="M")
    end_period = pd.Period(f"{end_year}-{end_month:02d}", freq="M")
    if end_period < start_period:
        raise typer.BadParameter(
            "end-month/year must be on or after the start month",
            param_hint="end-month/end-year",
        )
    for period in pd.period_range(start_period, end_period, freq="M"):
        yield period.year, period.month


def _process_month(
    *,
    year_value: int,
    month_value: int,
    data_root: Path,
    season_start: int,
    season_label: str,
    season_year_str: str,
    injuries_path: Path,
    schedule_path: Path,
    odds_path: Path,
    boxscores_path: Path,
    skip_bronze: bool,
    skip_gold: bool,
) -> None:
    start_dt, end_dt = _month_bounds(year_value, month_value)
    start_ts = pd.Timestamp(start_dt)
    end_ts = pd.Timestamp(end_dt)

    if not skip_bronze:
        typer.echo(
            f"[month-build] ({year_value}-{month_value:02d}) building bronze/silver using "
            f"{injuries_path.name} / {odds_path.name}"
        )
        builder = SmokeDatasetBuilder(
            start_date=start_ts,
            end_date=end_ts,
            data_dir=data_root,
            season_label=season_label,
            season_year=season_year_str,
            injuries_path=injuries_path,
            schedule_path=schedule_path,
            odds_path=odds_path,
            boxscores_path=boxscores_path,
        )
        builder.run()
    else:
        typer.echo(f"[month-build] ({year_value}-{month_value:02d}) skipping bronze/silver build.")

    if skip_gold:
        typer.echo(f"[month-build] ({year_value}-{month_value:02d}) skipping gold feature build.")
        return

    typer.echo(
        f"[month-build] ({year_value}-{month_value:02d}) building gold features (season={season_start}, month={month_value:02d})"
    )
    features_cli.main(
        start=start_dt,
        end=end_dt,
        data_root=data_root,
        season=season_start,
        month=month_value,
        out=None,
    )
    typer.echo(f"[month-build] ({year_value}-{month_value:02d}) completed bronze→gold build.")


@app.command()
def main(
    year: int = typer.Argument(..., help="Calendar year for the first month (e.g., 2024)."),
    month: int = typer.Argument(..., help="Calendar month number (1-12)."),
    end_year: int | None = typer.Option(None, help="Optional calendar year for the last month."),
    end_month: int | None = typer.Option(None, help="Optional month number for the last month."),
    data_root: Path = typer.Option(
        paths.get_data_root(),
        help="Root directory containing data/ artifacts (defaults to PROJECTIONS_DATA_ROOT or ./data).",
    ),
    season: int | None = typer.Option(
        None,
        help="Season start year (defaults to --year). Controls season= folders.",
    ),
    season_label: str | None = typer.Option(None, help="Override season label used in partition names."),
    season_year: str | None = typer.Option(None, help="Override season year string (e.g., 2024-25)."),
    injuries_json: Path | None = typer.Option(None, help="Path to nba_injuries_*.json."),
    schedule_json: Path | None = typer.Option(None, help="Path to nba_schedule_*.json."),
    odds_json: Path | None = typer.Option(None, help="Path to oddstrader JSON dump."),
    boxscores_json: Path | None = typer.Option(None, help="Path to nba_boxscores_*.json."),
    skip_bronze: bool = typer.Option(False, "--skip-bronze", help="Reuse existing bronze/silver snapshots."),
    skip_gold: bool = typer.Option(False, "--skip-gold", help="Skip the gold feature build."),
) -> None:
    season_start = season or year
    label, season_year_str = _season_strings(season_start, season_label, season_year)
    season_slug = season_year_str.replace("-", "_")

    default_injuries, default_schedule, default_odds, default_boxscores = _default_json_paths(
        data_root, season_slug, season_start
    )

    injuries_path = injuries_json or default_injuries
    schedule_path = schedule_json or default_schedule
    odds_path = odds_json or default_odds
    boxscores_path = boxscores_json or default_boxscores

    end_year_value = end_year or year
    end_month_value = end_month or month
    months = list(_month_range(year, month, end_year_value, end_month_value))
    typer.echo(
        f"[month-build] Processing {len(months)} month(s) from {months[0][0]}-{months[0][1]:02d} "
        f"to {months[-1][0]}-{months[-1][1]:02d}"
    )
    for idx, (year_value, month_value) in enumerate(months, start=1):
        typer.echo(f"[month-build] ---- Month {idx}/{len(months)} ----")
        _process_month(
            year_value=year_value,
            month_value=month_value,
            data_root=data_root,
            season_start=season_start,
            season_label=label,
            season_year_str=season_year_str,
            injuries_path=injuries_path,
            schedule_path=schedule_path,
            odds_path=odds_path,
            boxscores_path=boxscores_path,
            skip_bronze=skip_bronze,
            skip_gold=skip_gold,
        )


if __name__ == "__main__":  # pragma: no cover
    app()
