"""Healthcheck for live minutes_v1 projections against salaries."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
import typer

from projections import paths

PROJECTIONS_FILENAME = "minutes.parquet"
SUMMARY_FILENAME = "summary.json"
LATEST_POINTER = "latest_run.json"
DEFAULT_PROJECTIONS_ROOT = paths.data_path("artifacts", "minutes_v1", "daily")
DEFAULT_QC_ROOT = paths.data_path("artifacts", "minutes_v1", "live_qc")
OUT_MINUTES_THRESHOLD = 1.0

REQUIRED_PROJECTION_COLUMNS = {
    "player_id",
    "team_id",
    "game_id",
    "game_date",
    "status",
    "starter_flag",
    "minutes_p50",
}

app = typer.Typer(help=__doc__)


def _normalize_date(value: datetime) -> date:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.date()


def _load_pointer(pointer_path: Path) -> str | None:
    try:
        payload = json.loads(pointer_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return payload.get("run_id")


def _resolve_projections_path(root: Path, target_day: date) -> Path:
    root = root.expanduser()
    if root.is_file():
        return root

    # Allow passing a day/run directory directly.
    direct_parquet = root / PROJECTIONS_FILENAME
    if direct_parquet.exists():
        return direct_parquet

    day_dir = root / target_day.isoformat()
    if not day_dir.exists():
        raise FileNotFoundError(f"No projections found for {target_day} under {root}")

    pointer = day_dir / LATEST_POINTER
    if pointer.exists():
        run_id = _load_pointer(pointer)
        if run_id:
            candidate = day_dir / f"run={run_id}" / PROJECTIONS_FILENAME
            if candidate.exists():
                return candidate

    default_candidate = day_dir / PROJECTIONS_FILENAME
    if default_candidate.exists():
        return default_candidate

    run_dirs = sorted(day_dir.glob("run=*"), reverse=True)
    for run_dir in run_dirs:
        candidate = run_dir / PROJECTIONS_FILENAME
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Unable to locate projections parquet under {day_dir}")


def _ensure_columns(df: pd.DataFrame, required: Iterable[str], *, label: str) -> None:
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"{label} missing required columns: {', '.join(sorted(missing))}")


def _load_salaries(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    player_col: str | None = None
    for candidate in ("player_id", "PLAYER_ID", "PlayerID", "id", "ID"):
        if candidate in df.columns:
            player_col = candidate
            break
    if player_col is None:
        raise ValueError("Salaries CSV is missing a player_id column.")
    ids = pd.to_numeric(df[player_col], errors="coerce").dropna().astype("Int64")
    return ids


def _team_minutes_violations(df: pd.DataFrame) -> dict[int, float]:
    grouped = (
        df.groupby("team_id")["minutes_p50"]
        .sum(min_count=1)
        .to_dict()
    )
    violations: dict[int, float] = {}
    for team_id, total in grouped.items():
        if total < 225 or total > 255:
            violations[int(team_id)] = float(total)
    return violations


def _out_status_violations(df: pd.DataFrame) -> pd.DataFrame:
    statuses = df["status"].astype(str).str.lower()
    flagged = df.loc[(statuses.str.startswith("out")) & (df["minutes_p50"] > OUT_MINUTES_THRESHOLD)].copy()
    return flagged


@app.command()
def main(
    game_date: datetime = typer.Option(..., "--game-date", help="Target slate date (YYYY-MM-DD)."),
    salaries_path: Path = typer.Option(..., "--salaries-path", exists=True, readable=True),
    projections_root: Path = typer.Option(
        DEFAULT_PROJECTIONS_ROOT,
        "--projections-root",
        help="Root of projections_minutes_v1 outputs (daily/run directories or a single parquet).",
    ),
    qc_root: Path = typer.Option(
        DEFAULT_QC_ROOT,
        "--qc-root",
        help="Where to write the QC summary JSON.",
    ),
) -> None:
    target_day = _normalize_date(game_date)

    try:
        projections_path = _resolve_projections_path(projections_root, target_day)
    except FileNotFoundError as exc:
        typer.echo(f"[live-qc] ERROR: {exc}", err=True)
        raise typer.Exit(code=1)

    projections = pd.read_parquet(projections_path)
    _ensure_columns(projections, REQUIRED_PROJECTION_COLUMNS, label="Projections")
    projections["game_date"] = pd.to_datetime(projections["game_date"]).dt.date
    day_slice = projections.loc[projections["game_date"] == target_day].copy()
    if day_slice.empty:
        typer.echo("[live-qc] ERROR: Found zero projection rows for requested game_date.", err=True)
        raise typer.Exit(code=1)
    day_slice["minutes_p50"] = pd.to_numeric(day_slice["minutes_p50"], errors="coerce")
    day_slice["team_id"] = pd.to_numeric(day_slice["team_id"], errors="coerce").astype("Int64")
    day_slice["player_id"] = pd.to_numeric(day_slice["player_id"], errors="coerce").astype("Int64")

    try:
        salaried_players = _load_salaries(salaries_path)
    except ValueError as exc:
        typer.echo(f"[live-qc] ERROR: {exc}", err=True)
        raise typer.Exit(code=1)

    salaried_set = set(salaried_players.dropna().astype(int).tolist())
    projection_set = set(day_slice["player_id"].dropna().astype(int).tolist())
    missing_players = sorted(salaried_set - projection_set)

    team_minutes_issues = _team_minutes_violations(day_slice)
    out_rows = _out_status_violations(day_slice)

    summary = {
        "game_date": target_day.isoformat(),
        "projections_path": str(projections_path),
        "salaries_path": str(salaries_path),
        "checks": {
            "missing_players": {
                "count": len(missing_players),
                "player_ids": missing_players,
            },
            "team_minutes_out_of_range": team_minutes_issues,
            "out_status_minutes": out_rows[
                ["player_id", "team_id", "minutes_p50", "status"]
            ]
            .to_dict(orient="records"),
        },
    }

    qc_dir = qc_root / f"game_date={target_day.isoformat()}"
    qc_dir.mkdir(parents=True, exist_ok=True)
    summary_path = qc_dir / SUMMARY_FILENAME
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    failures = bool(missing_players or team_minutes_issues or not out_rows.empty)
    if failures:
        if missing_players:
            typer.echo(
                f"[live-qc] Missing projections for {len(missing_players)} salaried player(s): {missing_players}",
                err=True,
            )
        if team_minutes_issues:
            typer.echo(
                "[live-qc] Team minutes outside [225, 255]: "
                + ", ".join(f"{team}:{total:.1f}" for team, total in sorted(team_minutes_issues.items())),
                err=True,
            )
        if not out_rows.empty:
            typer.echo(
                f"[live-qc] OUT players with minutes>={OUT_MINUTES_THRESHOLD}: "
                + ", ".join(
                    f"{int(row.player_id)} ({row.minutes_p50:.1f}m)"
                    for _, row in out_rows.iterrows()
                ),
                err=True,
            )
        typer.echo(f"[live-qc] Summary written to {summary_path}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"[live-qc] All checks passed for {target_day} (summary -> {summary_path})")


if __name__ == "__main__":  # pragma: no cover
    app()
