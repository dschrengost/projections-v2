"""Backfill helper to score FPTS projections over a date range."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import typer

from projections.cli.score_fpts_v1 import (
    DEFAULT_FEATURES_ROOT,
    DEFAULT_MINUTES_ROOT,
    DEFAULT_OUT_ROOT,
    DEFAULT_FPTS_ARTIFACT_ROOT,
    LATEST_POINTER,
    MINUTES_SUMMARY,
    OUTPUT_FILENAME,
    load_fpts_bundle_context,
    resolve_features_path,
    resolve_minutes_run,
    score_fpts_for_date,
)
from projections.fpts_v1.production import (
    DEFAULT_PRODUCTION_CONFIG as DEFAULT_FPTS_CONFIG,
)
from projections.minutes_v1.production import (
    DEFAULT_PRODUCTION_CONFIG as DEFAULT_MINUTES_CONFIG,
    resolve_production_run_dir,
)

app = typer.Typer(help=__doc__)


def _iter_days(start_day: date, end_day: date) -> Iterable[date]:
    current = start_day
    while current <= end_day:
        yield current
        current += timedelta(days=1)


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _default_minutes_run_id(config_path: Path) -> str | None:
    try:
        _, run_id = resolve_production_run_dir(config_path)
        return run_id
    except Exception:
        return None


def _select_minutes_run(
    day_dir: Path,
    desired_model_run: str | None,
) -> str | None:
    if desired_model_run is None:
        return None
    if not day_dir.exists():
        return None
    for entry in sorted(day_dir.glob("run=*")):
        summary = _load_json(entry / MINUTES_SUMMARY)
        if summary and summary.get("model_run_id") == desired_model_run:
            return entry.name.split("=", 1)[1]
    day_summary = _load_json(day_dir / MINUTES_SUMMARY)
    if day_summary and day_summary.get("model_run_id") == desired_model_run:
        pointer = day_dir / LATEST_POINTER
        if pointer.exists():
            try:
                payload = json.loads(pointer.read_text(encoding="utf-8"))
                return payload.get("run_id")
            except json.JSONDecodeError:
                return None
    return None


def _fpts_output_exists(out_root: Path, slate_day: date, run_id: str) -> bool:
    run_dir = out_root / slate_day.isoformat() / f"run={run_id}"
    return (run_dir / OUTPUT_FILENAME).exists()


@app.command()
def main(
    start_date: datetime = typer.Option(..., "--start-date", help="Inclusive start date (YYYY-MM-DD)."),
    end_date: datetime = typer.Option(..., "--end-date", help="Inclusive end date (YYYY-MM-DD)."),
    minutes_run_id: str | None = typer.Option(
        None,
        "--minutes-run-id",
        help="Minutes model run id to target (defaults to config/minutes_current_run.json).",
    ),
    minutes_config: Path = typer.Option(
        DEFAULT_MINUTES_CONFIG,
        "--minutes-config",
        help="Minutes production config (used when --minutes-run-id is omitted).",
    ),
    fpts_run_id: str | None = typer.Option(
        None,
        "--fpts-run-id",
        help="Override the FPTS run id. Defaults to config/fpts_current_run.json.",
    ),
    fpts_artifact_root: Path = typer.Option(
        DEFAULT_FPTS_ARTIFACT_ROOT,
        "--fpts-artifact-root",
        help="Artifact root for FPTS runs (when --fpts-run-id is provided).",
    ),
    fpts_config: Path = typer.Option(
        DEFAULT_FPTS_CONFIG,
        "--fpts-config",
        help="FPTS production config (used when --fpts-run-id is omitted).",
    ),
    minutes_root: Path = typer.Option(
        DEFAULT_MINUTES_ROOT,
        "--minutes-root",
        help="Directory containing minutes outputs (artifacts/minutes_v1/daily).",
    ),
    live_features_root: Path = typer.Option(
        DEFAULT_FEATURES_ROOT,
        "--live-features-root",
        help="Directory containing live feature runs.",
    ),
    out_root: Path = typer.Option(
        DEFAULT_OUT_ROOT,
        "--out-root",
        help="Destination root for FPTS gold outputs.",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite/--no-overwrite",
        help="Whether to overwrite existing FPTS outputs for a run/date.",
    ),
) -> None:
    start_day = start_date.date()
    end_day = end_date.date()
    if end_day < start_day:
        raise typer.BadParameter("--end-date must be on/after --start-date.")

    minutes_root = minutes_root.expanduser().resolve()
    live_features_root = live_features_root.expanduser().resolve()
    out_root = out_root.expanduser().resolve()
    bundle_ctx = load_fpts_bundle_context(fpts_run_id, fpts_artifact_root, fpts_config)
    target_minutes_model = minutes_run_id or _default_minutes_run_id(minutes_config)

    if target_minutes_model:
        typer.echo(f"[fpts-backfill] targeting minutes model run: {target_minutes_model}")
    else:
        typer.echo("[fpts-backfill] no minutes model run specified; using latest pointer per day.")

    for day in _iter_days(start_day, end_day):
        day_dir = minutes_root / day.isoformat()
        run_hint = _select_minutes_run(day_dir, target_minutes_model)
        try:
            minutes_run_dir, resolved_run_id = resolve_minutes_run(minutes_root, day, run_hint)
        except FileNotFoundError as exc:
            typer.echo(f"[fpts-backfill] {day}: {exc}", err=True)
            continue

        if not overwrite and _fpts_output_exists(out_root, day, resolved_run_id):
            typer.echo(
                f"[fpts-backfill] {day}: skipping run={resolved_run_id} (already scored)."
            )
            continue

        try:
            features_file = resolve_features_path(
                live_features_root,
                day,
                resolved_run_id,
                None,
            )
        except FileNotFoundError as exc:
            typer.echo(f"[fpts-backfill] {day}: {exc}", err=True)
            continue

        try:
            score_fpts_for_date(
                slate_day=day,
                run_id=resolved_run_id,
                minutes_root=minutes_root,
                live_features_root=live_features_root,
                features_path=None,
                out_root=out_root,
                bundle_ctx=bundle_ctx,
                resolved_minutes_dir=minutes_run_dir,
                resolved_run_id=resolved_run_id,
                resolved_features_path=features_file,
                quiet=False,
            )
        except RuntimeError as exc:
            typer.echo(f"[fpts-backfill] {day}: {exc}", err=True)
            continue


if __name__ == "__main__":  # pragma: no cover
    app()
