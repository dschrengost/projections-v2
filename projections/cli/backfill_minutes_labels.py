"""Backfill minutes_v1 labels by wrapping the existing boxscore + label builders.

This CLI ensures bronze boxscore payloads exist by invoking
``projections.etl.boxscores.main`` (the scraper that writes
``bronze/boxscores_raw`` alongside legacy ``labels/boxscore_labels.parquet``) and
then reuses ``projections.cli.build_minutes_labels.main`` to materialize
``gold/labels_minutes_v1/season=YYYY/game_date=YYYY-MM-DD/labels.parquet``.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
import typer

from projections import paths
from projections.cli import build_minutes_labels as labels_cli
from projections.etl import boxscores as boxscores_etl
from projections.etl import storage

app = typer.Typer(help=__doc__)
LABEL_FILENAME = labels_cli.LABEL_FILENAME


def _normalize_day(value: datetime | str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC")
    return ts.tz_localize(None).normalize()


def _season_from_day(day: pd.Timestamp) -> int:
    return day.year if day.month >= 8 else day.year - 1


def _iter_days(start: pd.Timestamp, end: pd.Timestamp) -> Iterable[pd.Timestamp]:
    return storage.iter_days(start, end)


def _bronze_boxscore_path(data_root: Path, day: pd.Timestamp) -> Path:
    season = _season_from_day(day)
    return storage.bronze_partition_path(
        "boxscores_raw",
        data_root=data_root,
        season=season,
        target_date=day.date(),
    )


def _labels_path(data_root: Path, day: pd.Timestamp) -> Path:
    season = _season_from_day(day)
    return (
        data_root
        / "gold"
        / "labels_minutes_v1"
        / f"season={season}"
        / f"game_date={day.date().isoformat()}"
        / LABEL_FILENAME
    )


def _ensure_boxscores(
    *,
    data_root: Path,
    day: pd.Timestamp,
    overwrite: bool,
) -> tuple[bool, str | None]:
    raw_path = _bronze_boxscore_path(data_root, day)
    if raw_path.exists() and not overwrite:
        return True, "exists"
    if raw_path.exists() and overwrite:
        raw_path.unlink(missing_ok=True)
    season = _season_from_day(day)
    try:
        boxscores_etl.main(
            start=day.to_pydatetime(),
            end=day.to_pydatetime(),
            season=season,
            schedule=[],
            data_root=data_root,
            timeout=10.0,  # Explicit to avoid Typer Option default when called programmatically
        )
    except Exception as exc:  # pragma: no cover - surfaced via summary + tests
        return False, str(exc)
    if raw_path.exists():
        return True, "fetched"
    return False, "boxscore raw partition missing after ETL"


def _build_labels_for_day(
    *,
    data_root: Path,
    day: pd.Timestamp,
    overwrite: bool,
) -> tuple[bool, str | None]:
    labels_target = _labels_path(data_root, day)
    if labels_target.exists() and not overwrite:
        return False, "labels_exist"
    if labels_target.exists() and overwrite:
        labels_target.unlink(missing_ok=True)
    try:
        labels_cli.main(
            start_date=day.to_pydatetime(),
            end_date=day.to_pydatetime(),
            data_root=data_root,
        )
    except Exception as exc:  # pragma: no cover - reported by CLI summary
        return False, str(exc)
    if labels_target.exists():
        return True, "written"
    return False, "labels parquet missing after builder"


@app.command()
def main(
    start_date: datetime = typer.Option(..., "--start-date", help="Inclusive backfill start date (YYYY-MM-DD)."),
    end_date: datetime | None = typer.Option(None, "--end-date", help="Inclusive backfill end date (defaults to start-date)."),
    data_root: Path = typer.Option(
        paths.get_data_root(),
        "--data-root",
        help="Root directory containing bronze/silver/gold data (defaults to PROJECTIONS_DATA_ROOT).",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite/--no-overwrite",
        help="Re-scrape boxscores and rewrite labels even when partitions already exist.",
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Emit per-step logs."),
) -> None:
    start_day = _normalize_day(start_date)
    end_input = end_date or start_date
    end_day = _normalize_day(end_input)
    if end_day < start_day:
        raise typer.BadParameter("--end-date must be on or after --start-date.")

    data_root = data_root.expanduser().resolve()

    processed: list[str] = []
    skipped_existing: list[str] = []
    failed_boxscores: list[tuple[str, str]] = []
    failed_labels: list[tuple[str, str]] = []

    for day in _iter_days(start_day, end_day):
        label_path = _labels_path(data_root, day)
        iso = day.date().isoformat()
        if label_path.exists() and not overwrite:
            skipped_existing.append(iso)
            if verbose:
                typer.echo(f"[labels-backfill] date={iso} skip=labels_exist")
            continue

        ok_box, message = _ensure_boxscores(data_root=data_root, day=day, overwrite=overwrite)
        if not ok_box:
            failed_boxscores.append((iso, message or "unknown_error"))
            typer.echo(
                f"[labels-backfill] date={iso} status=boxscore_etl_failed error={message}",
                err=True,
            )
            continue
        elif verbose:
            typer.echo(
                f"[labels-backfill] date={iso} status=boxscores_ready source={message}",
            )

        ok_labels, label_message = _build_labels_for_day(
            data_root=data_root,
            day=day,
            overwrite=overwrite,
        )
        if ok_labels:
            processed.append(iso)
            typer.echo(
                f"[labels-backfill] date={iso} status=labels_written target={_labels_path(data_root, day)}",
            )
        else:
            if label_message == "labels_exist":
                skipped_existing.append(iso)
                if verbose:
                    typer.echo(f"[labels-backfill] date={iso} skip=labels_exist_post_etl")
            else:
                failed_labels.append((iso, label_message or "unknown_error"))
                typer.echo(
                    f"[labels-backfill] date={iso} status=labels_failed error={label_message}",
                    err=True,
                )

    typer.echo(
        "[labels-backfill] summary: processed={processed_count} skipped_existing={skipped_count} "
        "failed_boxscores={failed_box} failed_labels={failed_lbl}".format(
            processed_count=len(processed),
            skipped_count=len(skipped_existing),
            failed_box=len(failed_boxscores),
            failed_lbl=len(failed_labels),
        )
    )

    if failed_boxscores or failed_labels:
        raise typer.Exit(code=1)


if __name__ == "__main__":  # pragma: no cover
    app()
