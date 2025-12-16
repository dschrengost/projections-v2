"""CLI for scraping NBA tracking data into bronze partitions."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import time
from typing import Iterable, Sequence

import typer

from projections import paths
from projections.data.nba import tracking, tracking_client

app = typer.Typer(help="Scrape stats.nba.com tracking payloads into bronze.")

DEFAULT_SEASON_TYPE = "Regular Season"


def _iter_dates(start: date, end: date) -> Iterable[date]:
    cursor = start
    while cursor <= end:
        yield cursor
        cursor += timedelta(days=1)


def _resolve_measure_types(values: Sequence[str] | None) -> list[str]:
    if not values:
        return list(tracking.SUPPORTED_PT_MEASURE_TYPES)
    supported = set(tracking.SUPPORTED_PT_MEASURE_TYPES)
    invalid = sorted({value for value in values if value not in supported})
    if invalid:
        valid = ", ".join(tracking.SUPPORTED_PT_MEASURE_TYPES)
        raise typer.BadParameter(
            f"Unsupported pt_measure_type values: {', '.join(invalid)} "
            f"(valid options: {valid})"
        )
    # preserve CLI ordering while deduplicating
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            ordered.append(value)
            seen.add(value)
    return ordered


def _resolve_data_root(data_root: Path | None) -> Path:
    return (data_root or paths.get_data_root()).resolve()


def _parse_date(value: str, *, param_name: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise typer.BadParameter(
            f"Invalid date supplied for {param_name}: {value}"
        ) from exc


def _scrape_day(
    *,
    season: str,
    season_type: str,
    game_date: date,
    measure_types: Sequence[str],
    data_root: Path,
    bronze_root: Path | None,
    max_retries: int,
    sleep_seconds: float,
    timeout: float,
    dry_run: bool,
) -> None:
    for measure in measure_types:
        try:
            payload = tracking_client.fetch_leaguedashptstats(
                season=season,
                season_type=season_type,
                pt_measure_type=measure,
                game_date=game_date,
                timeout=timeout,
                max_retries=max_retries,
                retry_delay=sleep_seconds,
            )
        except tracking_client.TrackingClientError as exc:
            typer.secho(
                f"[tracking] failed to fetch {measure} for {game_date}: {exc}",
                fg=typer.colors.RED,
            )
            continue
        frame = tracking.normalize_tracking_df(
            payload,
            season=season,
            season_type=season_type,
            game_date=game_date,
            pt_measure_type=measure,
        )
        if frame.empty:
            typer.secho(
                f"[tracking] {game_date} {measure}: no rows returned.",
                fg=typer.colors.YELLOW,
            )
        elif dry_run:
            typer.echo(
                f"[tracking] {game_date} {measure}: {len(frame)} rows (dry-run)."
            )
        else:
            result = tracking.write_tracking_partition(
                frame,
                data_root=data_root,
                season=season,
                season_type=season_type,
                game_date=game_date,
                pt_measure_type=measure,
                bronze_root=bronze_root,
            )
            typer.echo(
                f"[tracking] wrote {result.rows_written} new rows "
                f"(total={result.total_rows}) -> {result.path}"
            )
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)


@app.command()
def backfill(
    season: str = typer.Option(..., "--season", help='Season token (e.g. "2024-25").'),
    season_type: str = typer.Option(
        DEFAULT_SEASON_TYPE,
        "--season-type",
        help='Season type label (e.g. "Regular Season").',
    ),
    start_date: str = typer.Option(
        ..., "--start-date", help="Start date (YYYY-MM-DD)."
    ),
    end_date: str | None = typer.Option(
        None,
        "--end-date",
        help="End date (YYYY-MM-DD). Defaults to --start-date when omitted.",
    ),
    pt_measure_type: list[str] | None = typer.Option(
        None,
        "--pt-measure-type",
        "-m",
        help="Filter to a subset of PtMeasureType values.",
        show_default=False,
    ),
    data_root: Path | None = typer.Option(
        None,
        "--data-root",
        help="Override PROJECTIONS_DATA_ROOT (defaults to ../projections-data).",
    ),
    bronze_root: Path | None = typer.Option(
        None,
        "--bronze-root",
        help="Optional override for bronze/nba/tracking root.",
    ),
    max_retries: int = typer.Option(3, "--max-retries", min=1),
    sleep_seconds: float = typer.Option(
        1.0,
        "--sleep-seconds",
        help="Delay between requests and retry backoff (seconds).",
        min=0.0,
    ),
    timeout: float = typer.Option(15.0, "--timeout", help="HTTP timeout (seconds)."),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Log planned fetches without writing files.",
    ),
) -> None:
    """Backfill tracking data for a date range."""

    start_day = _parse_date(start_date, param_name="--start-date")
    end_day = _parse_date(end_date, param_name="--end-date") if end_date else start_day
    if end_day < start_day:
        raise typer.BadParameter("--end-date must be >= --start-date")

    measure_types = _resolve_measure_types(pt_measure_type)
    resolved_root = _resolve_data_root(data_root)
    run_dates = list(_iter_dates(start_day, end_day))
    typer.echo(
        f"[tracking] backfill {season} {season_type} "
        f"{run_dates[0]}→{run_dates[-1]} ({len(run_dates)} days, "
        f"{len(measure_types)} measures)"
    )
    for day in run_dates:
        _scrape_day(
            season=season,
            season_type=season_type,
            game_date=day,
            measure_types=measure_types,
            data_root=resolved_root,
            bronze_root=bronze_root,
            max_retries=max_retries,
            sleep_seconds=sleep_seconds,
            timeout=timeout,
            dry_run=dry_run,
        )


@app.command("run-day")
def run_day(
    season: str = typer.Option(..., "--season", help='Season token (e.g. "2024-25").'),
    season_type: str = typer.Option(
        DEFAULT_SEASON_TYPE,
        "--season-type",
        help='Season type label (e.g. "Regular Season").',
    ),
    target_date: str = typer.Option(..., "--date", help="Game date (YYYY-MM-DD)."),
    pt_measure_type: list[str] | None = typer.Option(
        None,
        "--pt-measure-type",
        "-m",
        help="Filter to a subset of PtMeasureType values.",
        show_default=False,
    ),
    data_root: Path | None = typer.Option(
        None,
        "--data-root",
        help="Override PROJECTIONS_DATA_ROOT (defaults to ../projections-data).",
    ),
    bronze_root: Path | None = typer.Option(
        None,
        "--bronze-root",
        help="Optional override for bronze/nba/tracking root.",
    ),
    max_retries: int = typer.Option(3, "--max-retries", min=1),
    sleep_seconds: float = typer.Option(
        1.0,
        "--sleep-seconds",
        help="Delay between requests and retry backoff (seconds).",
        min=0.0,
    ),
    timeout: float = typer.Option(15.0, "--timeout", help="HTTP timeout (seconds)."),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Log planned fetches without writing files.",
    ),
) -> None:
    """Scrape an individual date for incremental updates."""

    measure_types = _resolve_measure_types(pt_measure_type)
    resolved_root = _resolve_data_root(data_root)
    target_day = _parse_date(target_date, param_name="--date")
    typer.echo(
        f"[tracking] run-day {season} {season_type} {target_day} "
        f"({len(measure_types)} measures)"
    )
    _scrape_day(
        season=season,
        season_type=season_type,
        game_date=target_day,
        measure_types=measure_types,
        data_root=resolved_root,
        bronze_root=bronze_root,
        max_retries=max_retries,
        sleep_seconds=sleep_seconds,
        timeout=timeout,
        dry_run=dry_run,
    )


if __name__ == "__main__":
    app()
