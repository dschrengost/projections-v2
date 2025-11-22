"""Live pipeline orchestrator for the core ETLs."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
import typer

from projections import paths
from projections.etl import daily_lineups as daily_lineups_etl
from projections.etl import injuries as injuries_etl
from projections.etl import odds as odds_etl
from projections.etl import roster_nightly as roster_etl

app = typer.Typer(help="Run injuries, daily lineups, odds, and roster ETLs sequentially.")


def _normalize_day(value: Optional[datetime]) -> pd.Timestamp:
    if value is None:
        return pd.Timestamp.utcnow().tz_localize("UTC").tz_convert(None).normalize()
    return pd.Timestamp(value).normalize()


def _resolve_season(target_day: pd.Timestamp, season_override: Optional[int]) -> int:
    if season_override is not None:
        return season_override
    return int(target_day.year)


def _resolve_month(target_day: pd.Timestamp, month_override: Optional[int]) -> int:
    if month_override is not None:
        return month_override
    return int(target_day.month)


def _echo_stage(message: str) -> None:
    typer.echo(f"[live] {message}")


@app.command()
def run(  # noqa: PLR0913, PLR0917 - orchestrator with many knobs
    start: Optional[datetime] = typer.Option(
        None,
        "--start",
        help="Start date inclusive (YYYY-MM-DD). Defaults to today if omitted.",
    ),
    end: Optional[datetime] = typer.Option(
        None,
        "--end",
        help="End date inclusive (YYYY-MM-DD). Defaults to --start when omitted.",
    ),
    season: Optional[int] = typer.Option(None, "--season", help="Season label (e.g., 2025)."),
    month: Optional[int] = typer.Option(None, "--month", help="Month partition (1-12)."),
    data_root: Path = typer.Option(
        paths.get_data_root(),
        "--data-root",
        help="Base data directory (defaults to PROJECTIONS_DATA_ROOT or ./data).",
    ),
    schedule: List[str] = typer.Option(
        [],
        "--schedule",
        help="Optional schedule parquet glob(s) shared across ETLs (falls back to NBA API when empty).",
    ),
    roster: List[str] = typer.Option(
        [],
        "--roster",
        help="Optional roster parquet(s) when seeding roster_nightly from prior polls.",
    ),
    lineups_dir: Optional[Path] = typer.Option(
        None,
        "--lineups-dir",
        help="Optional override for the daily lineups silver directory consumed by roster nightly.",
    ),
    injuries: bool = typer.Option(True, "--injuries/--skip-injuries", help="Run the injuries ETL stage."),
    lineups: bool = typer.Option(True, "--lineups/--skip-lineups", help="Run the daily lineups ETL stage."),
    odds: bool = typer.Option(True, "--odds/--skip-odds", help="Run the odds ETL stage."),
    run_roster: bool = typer.Option(True, "--run-roster/--skip-roster", help="Run the roster nightly stage."),
    schedule_timeout: float = typer.Option(10.0, "--schedule-timeout", help="Timeout (seconds) for NBA schedule API fallback."),
    injury_timeout: float = typer.Option(15.0, "--injury-timeout", help="HTTP timeout (seconds) for NBA injury PDF scraping."),
    injury_player_timeout: float = typer.Option(10.0, "--injury-player-timeout", help="Timeout (seconds) for NBA player resolver."),
    lineups_timeout: float = typer.Option(10.0, "--lineups-timeout", help="HTTP timeout (seconds) for stats.nba.com daily lineups."),
    odds_timeout: float = typer.Option(10.0, "--odds-timeout", help="HTTP timeout (seconds) for Oddstrader."),
    roster_timeout: float = typer.Option(10.0, "--roster-timeout", help="Timeout (seconds) for roster fallback scraper."),
) -> None:
    """Run the core scrapers for the requested window."""

    start_day = _normalize_day(start)
    end_day = _normalize_day(end) if end else start_day
    if end_day < start_day:
        raise typer.BadParameter("--end must be on/after --start.")

    season_value = _resolve_season(start_day, season)
    month_value = _resolve_month(start_day, month)
    data_root = data_root.resolve()

    _echo_stage(
        f"window={start_day.date()}->{end_day.date()} season={season_value} month={month_value} root={data_root}"
    )

    start_dt = start_day.to_pydatetime()
    end_dt = end_day.to_pydatetime()

    if injuries:
        _echo_stage("running injuries ETL")
        injuries_etl.main(
            injuries_json=None,
            schedule=schedule,
            use_scraper=True,
            start=start_dt,
            end=end_dt,
            season=season_value,
            month=month_value,
            data_root=data_root,
            bronze_root=None,
            bronze_out=None,
            silver_out=None,
            scraper_timeout=injury_player_timeout,
            schedule_timeout=schedule_timeout,
            injury_timeout=injury_timeout,
        )
    else:
        _echo_stage("skipping injuries stage")

    if lineups:
        _echo_stage("running daily lineups ETL")
        try:
            daily_lineups_etl.run(
                start=start_dt,
                end=end_dt,
                season=season_value,
                data_root=data_root,
                bronze_root=None,
                silver_root=None,
                timeout=lineups_timeout,
            )
        except Exception as exc:  # pragma: no cover - keep pipeline alive when NBA feed lags
            typer.echo(f"[live] warning: daily lineups failed ({exc}); continuing without lineups", err=True)
    else:
        _echo_stage("skipping daily lineups stage")

    if odds:
        _echo_stage("running odds ETL")
        odds_etl.main(
            start=start_dt,
            end=end_dt,
            season=season_value,
            month=month_value,
            schedule=schedule,
            data_root=data_root,
            bronze_root=None,
            bronze_out=None,
            silver_out=None,
            scraper_timeout=odds_timeout,
            schedule_timeout=schedule_timeout,
        )
    else:
        _echo_stage("skipping odds stage")

    if run_roster:
        _echo_stage("running roster nightly ETL")
        roster_etl.main(
            roster=roster,
            schedule=schedule,
            start=start_dt,
            end=end_dt,
            season=season_value,
            month=month_value,
            data_root=data_root,
            bronze_root=None,
            bronze_out=None,
            out=None,
            lineups_dir=lineups_dir,
            scrape_missing=True,
            roster_timeout=roster_timeout,
            schedule_timeout=schedule_timeout,
        )
    else:
        _echo_stage("skipping roster nightly stage")

    _echo_stage("live pipeline completed successfully")


if __name__ == "__main__":  # pragma: no cover - manual invocation
    app()
