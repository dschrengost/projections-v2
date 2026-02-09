"""Build/update RealGM player-id crosswalk from live minutes artifacts."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import typer

from projections import paths
from projections.minutes.depth_chart_crosswalk import (
    refresh_realgm_player_crosswalk_from_minutes,
    summarize_crosswalk_json,
)

app = typer.Typer(help=__doc__)


def _resolve_run_id(day_dir: Path, run_id: str | None) -> str | None:
    if run_id:
        return run_id
    pointer = day_dir / "latest_run.json"
    if not pointer.exists():
        return None
    try:
        payload = json.loads(pointer.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(payload, dict) and payload.get("run_id"):
        return str(payload["run_id"])
    return None


def _read_as_of_from_manifest(run_dir: Path) -> pd.Timestamp | None:
    manifest = run_dir / "manifest.json"
    if not manifest.exists():
        return None
    try:
        payload: Any = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    ts = pd.to_datetime(payload.get("as_of_ts"), utc=True, errors="coerce")
    if ts is None or pd.isna(ts):
        ts = pd.to_datetime(payload.get("run_as_of_ts"), utc=True, errors="coerce")
    if ts is None or pd.isna(ts):
        return None
    return pd.Timestamp(ts)


@app.command()
def run(
    game_date: str = typer.Option(..., "--date", "-d", help="Slate date YYYY-MM-DD."),
    run_id: str | None = typer.Option(None, "--run-id", help="Minutes run_id (defaults to latest pointer)."),
    minutes_path: Path | None = typer.Option(
        None,
        "--minutes-path",
        help="Optional explicit minutes parquet path.",
    ),
    as_of_ts: str | None = typer.Option(
        None,
        "--as-of-ts",
        help="Optional override as_of timestamp (ISO-8601 UTC).",
    ),
    data_root: Path = typer.Option(
        paths.get_data_root(),
        "--data-root",
        help="Data root (defaults to PROJECTIONS_DATA_ROOT).",
    ),
) -> None:
    slate_day = date.fromisoformat(game_date)
    day_dir = data_root / "artifacts" / "minutes_v1" / "daily" / slate_day.isoformat()
    resolved_run = _resolve_run_id(day_dir, run_id)
    if minutes_path is None:
        if not resolved_run:
            raise typer.BadParameter(
                f"Unable to resolve run_id for {slate_day.isoformat()} (pass --run-id or --minutes-path)."
            )
        run_dir = day_dir / f"run={resolved_run}"
        minutes_path = run_dir / "minutes.parquet"
    else:
        run_dir = minutes_path.parent

    if not minutes_path.exists():
        raise typer.BadParameter(f"Minutes parquet not found: {minutes_path}")

    minutes_df = pd.read_parquet(minutes_path)
    as_of = pd.to_datetime(as_of_ts, utc=True, errors="coerce") if as_of_ts else _read_as_of_from_manifest(run_dir)
    as_of_value = None if as_of is None or pd.isna(as_of) else pd.Timestamp(as_of)

    diag = refresh_realgm_player_crosswalk_from_minutes(
        minutes_df,
        data_root=data_root,
        as_of_ts=as_of_value,
    )
    typer.echo(f"[dc-crosswalk] {summarize_crosswalk_json(diag)}")


if __name__ == "__main__":
    app()
