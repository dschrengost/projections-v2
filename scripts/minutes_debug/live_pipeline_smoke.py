"""Smoke test helper for the live minutes pipeline."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
import typer

from projections import paths
from projections.minutes_v1.production import load_production_minutes_bundle

app = typer.Typer(help=__doc__)

FEATURE_FILENAME = "features.parquet"


def _run_cmd(args: Iterable[str]) -> None:
    typer.echo(f"[smoke] Running: {' '.join(args)}")
    subprocess.run(list(args), check=True)


def _resolve_run_dir(day_dir: Path) -> Path:
    pointer = day_dir / LATEST_POINTER
    if pointer.exists():
        payload = json.loads(pointer.read_text(encoding="utf-8"))
        run_id = payload.get("run_id")
        if run_id:
            candidate = day_dir / f"run={run_id}"
            if candidate.exists():
                return candidate
    candidates = sorted(p for p in day_dir.glob("run=*") if p.is_dir())
    if not candidates:
        raise RuntimeError(f"No run directories found in {day_dir}")
    return candidates[-1]


def _format_ts(ts: datetime | None) -> str:
    target = ts or datetime.now(timezone.utc)
    if target.tzinfo:
        target = target.astimezone(timezone.utc).replace(tzinfo=None)
    return target.strftime("%Y-%m-%dT%H:%M:%S")


LATEST_POINTER = "latest_run.json"


@app.command()
def main(
    date: datetime = typer.Option(..., help="Slate date (YYYY-MM-DD)."),
    run_as_of_ts: datetime | None = typer.Option(
        None, help="Information timestamp (defaults to now, UTC)."
    ),
    data_root: Path = typer.Option(
        paths.get_data_root(), help="Data root that mirrors production (bronze/silver/gold)."
    ),
    rows: int = typer.Option(5, help="Rows to preview from the live feature slice."),
    keep_artifacts: bool = typer.Option(
        False, "--keep-artifacts/--discard-artifacts", help="Retain temporary outputs for inspection."
    ),
) -> None:
    data_root = data_root.expanduser()
    run_ts_str = _format_ts(run_as_of_ts)
    day_slug = date.strftime("%Y-%m-%d")

    temp_dir: str | None = None
    if keep_artifacts:
        root_dir = Path(tempfile.mkdtemp(prefix="live_smoke_"))
    else:
        temp_dir = tempfile.mkdtemp(prefix="live_smoke_")
        root_dir = Path(temp_dir)
    features_root = root_dir / "features"
    minutes_root = root_dir / "minutes"
    logs_root = root_dir / "logs"
    features_root.mkdir(parents=True, exist_ok=True)
    minutes_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)

    builder_cmd = [
        sys.executable,
        "-m",
        "projections.cli.build_minutes_live",
        "--date",
        day_slug,
        "--run-as-of-ts",
        run_ts_str,
        "--data-root",
        str(data_root),
        "--out-root",
        str(features_root),
        "--skip-active-roster",
    ]
    _run_cmd(builder_cmd)

    day_dir = features_root / day_slug
    run_dir = _resolve_run_dir(day_dir)
    feature_path = run_dir / FEATURE_FILENAME
    if not feature_path.exists():
        raise FileNotFoundError(f"Expected features at {feature_path}")

    typer.echo(f"[smoke] Features written to {feature_path}")
    live_features = pd.read_parquet(feature_path)
    bundle = load_production_minutes_bundle()
    bundle = dict(bundle)
    feature_columns = bundle["feature_columns"]
    missing = [col for col in feature_columns if col not in live_features.columns]
    if missing:
        raise RuntimeError(f"Live features missing columns: {', '.join(missing)}")

    preview_cols = [
        "game_id",
        "player_id",
        "team_id",
        "rotation_minutes_std_5g",
        "role_change_rate_10g",
        "season_phase",
        "depth_same_pos_active",
        "blowout_risk_score",
        "close_game_score",
        "starter_flag",
    ]
    available_preview = [col for col in preview_cols if col in live_features.columns]
    typer.echo(f"[smoke] Previewing {len(available_preview)} feature columns:")
    typer.echo(live_features[available_preview].head(rows).to_string(index=False))

    run_id = run_dir.name.split("=", 1)[1]
    score_cmd = [
        sys.executable,
        "-m",
        "projections.cli.score_minutes_v1",
        "--date",
        day_slug,
        "--mode",
        "live",
        "--features-path",
        str(feature_path),
        "--run-id",
        run_id,
        "--artifact-root",
        str(minutes_root),
        "--prediction-logs-root",
        str(logs_root),
    ]
    _run_cmd(score_cmd)

    scored_dir = minutes_root / day_slug / f"run={run_id}"
    scored_path = scored_dir / "minutes.parquet"
    if scored_path.exists():
        scored = pd.read_parquet(scored_path)
        typer.echo(f"[smoke] Scored {len(scored)} rows -> {scored_path}")
    else:
        typer.echo("[smoke] Scoring produced no rows (locked slate?).")

    if not keep_artifacts and temp_dir:
        typer.echo(f"[smoke] Cleaning up {temp_dir}")
        try:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            typer.echo("[smoke] Warning: failed to clean temporary directory.", err=True)
    else:
        typer.echo(f"[smoke] Outputs retained under {root_dir}")


if __name__ == "__main__":  # pragma: no cover
    app()
