#!/usr/bin/env python3
"""Backfill gold/slates snapshots for historical games.

This script loads historical schedule rows from silver, then calls the same
freezer used for the live ``freeze-slates`` timer to materialize immutable
per-game snapshots under ``<DATA_ROOT>/gold/slates``.

It is resumable by default: if the target parquet + manifest files already
exist for a game/snapshot type, they are skipped (unless ``--force``).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import typer

from projections import paths
from projections.cli import freeze_slates as freezer

app = typer.Typer(help=__doc__)

UTC = timezone.utc


def _atomic_write_text(path: Path, payload: str, *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        tmp.write_text(payload, encoding=encoding)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _status_path(out_root: Path) -> Path:
    return out_root / "_backfill_slates_status.json"


def _load_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "last_run": None,
            "games_completed": 0,
            "games_failed": 0,
            "snapshots_frozen": 0,
            "snapshots_skipped_existing": 0,
            "snapshots_failed": 0,
            "failures": [],
        }
    return json.loads(path.read_text(encoding="utf-8"))


def _write_status(path: Path, status: dict[str, Any]) -> None:
    status["last_run"] = datetime.now(UTC).isoformat()
    _atomic_write_text(path, json.dumps(status, indent=2, sort_keys=True))


def _read_parquet_tree(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.is_file():
        return pd.read_parquet(path)
    files = sorted(path.rglob("*.parquet"))
    if not files:
        return pd.DataFrame()
    frames = [pd.read_parquet(file) for file in files]
    return pd.concat(frames, ignore_index=True)


def _parse_date(value: str, *, arg_name: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:  # noqa: BLE001
        raise typer.BadParameter(f"{arg_name} must be YYYY-MM-DD") from exc


def _season_for_game_date(game_date: date) -> int:
    return game_date.year if game_date.month >= 8 else game_date.year - 1


@dataclass(frozen=True)
class Candidate:
    season: int
    game_id: int
    game_date: date
    tip_ts: pd.Timestamp
    row: dict[str, Any]


def _load_schedule_candidates(
    data_root: Path,
    *,
    seasons: list[int],
    start_date: date,
    end_date: date,
) -> list[Candidate]:
    candidates: list[Candidate] = []
    for season in seasons:
        schedule_root = data_root / "silver" / "schedule" / f"season={season}"
        schedule_df = _read_parquet_tree(schedule_root)
        if schedule_df.empty:
            continue
        schedule_df["game_id"] = pd.to_numeric(schedule_df.get("game_id"), errors="coerce").astype("Int64")
        schedule_df["tip_ts"] = pd.to_datetime(schedule_df.get("tip_ts"), utc=True, errors="coerce")
        if "game_date" in schedule_df.columns:
            schedule_df["game_date"] = pd.to_datetime(schedule_df.get("game_date"), errors="coerce").dt.normalize()
        else:
            schedule_df["game_date"] = schedule_df["tip_ts"].dt.tz_convert(freezer.ET_TZ).dt.tz_localize(None).dt.normalize()

        window_start = pd.Timestamp(start_date).normalize()
        window_end = pd.Timestamp(end_date).normalize()
        schedule_df = schedule_df.dropna(subset=["game_id", "tip_ts", "game_date"])
        schedule_df = schedule_df[
            (schedule_df["game_date"] >= window_start) & (schedule_df["game_date"] <= window_end)
        ].copy()
        schedule_df.sort_values(["game_date", "tip_ts", "game_id"], inplace=True, kind="mergesort")

        for record in schedule_df.to_dict("records"):
            game_id_value = int(record["game_id"])
            tip_ts_value = pd.to_datetime(record.get("tip_ts"), utc=True, errors="coerce")
            game_date_value = pd.to_datetime(record.get("game_date"), errors="coerce")
            if pd.isna(tip_ts_value) or pd.isna(game_date_value):
                continue
            if getattr(game_date_value, "tzinfo", None) is not None:
                game_date_value = game_date_value.tz_convert(freezer.ET_TZ).tz_localize(None)
            game_date_value = game_date_value.normalize()
            candidates.append(
                Candidate(
                    season=season,
                    game_id=game_id_value,
                    game_date=game_date_value.date(),
                    tip_ts=pd.Timestamp(tip_ts_value),
                    row=record,
                )
            )
    return candidates


def _snapshot_paths(out_root: Path, candidate: Candidate, snapshot_type: str) -> tuple[Path, Path]:
    out_path = (
        out_root / f"season={candidate.season}" / f"game_date={candidate.game_date.isoformat()}" / f"game_id={candidate.game_id}"
    )
    parquet_path = out_path / f"{snapshot_type}.parquet"
    manifest_path = out_path / f"manifest.{snapshot_type}.json"
    return parquet_path, manifest_path


@app.command()
def backfill(
    season: int | None = typer.Option(
        None, "--season", help="Optional schedule season partition override (e.g., 2025)."
    ),
    date_str: str | None = typer.Option(None, "--date", help="Single ET date (YYYY-MM-DD)."),
    start: str | None = typer.Option(None, "--start", help="Start ET date (YYYY-MM-DD)."),
    end: str | None = typer.Option(None, "--end", help="End ET date (YYYY-MM-DD)."),
    data_root: Path = typer.Option(
        paths.get_data_root(),
        "--data-root",
        help="Base data directory (defaults to PROJECTIONS_DATA_ROOT or ./data).",
    ),
    out_root: Path | None = typer.Option(
        None,
        "--out-root",
        help="Optional override for gold/slates root (defaults to <data_root>/gold/slates).",
    ),
    snapshot_type: str = typer.Option(
        "both",
        "--snapshot-type",
        help="Which snapshot(s) to freeze: 'lock', 'pretip', or 'both'.",
    ),
    history_days: int | None = typer.Option(
        None, "--history-days", min=1, help="Optional rolling history window (in days) for label context."
    ),
    allow_empty_history: bool = typer.Option(
        True,
        "--allow-empty-history/--require-history",
        help="Allow freezing even if no historical labels exist before the game date.",
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite existing gold snapshots."),
    resume: bool = typer.Option(
        True,
        "--resume/--no-resume",
        help="Skip snapshots that already exist on disk (parquet + manifest).",
    ),
    repair_partials: bool = typer.Option(
        True,
        "--repair-partials/--no-repair-partials",
        help="If only one of parquet/manifest exists, overwrite to repair the snapshot.",
    ),
    dry_run: bool = typer.Option(
        True,
        "--dry-run/--no-dry-run",
        help="When enabled, only prints what would be frozen.",
    ),
    max_games: int | None = typer.Option(None, "--max-games", min=1, help="Optional cap on processed games."),
    stop_on_error: bool = typer.Option(False, "--stop-on-error", help="Abort on first error."),
    status_file: Path | None = typer.Option(
        None,
        "--status-file",
        help="Optional JSON status file path (defaults to <out_root>/_backfill_slates_status.json).",
    ),
) -> None:
    data_root = data_root.resolve()
    out_root = (out_root or (data_root / "gold" / "slates")).resolve()

    if date_str is not None:
        if start is not None or end is not None:
            raise typer.BadParameter("Use either --date or --start/--end, not both.")
        start_date = _parse_date(date_str, arg_name="--date")
        end_date = start_date
    else:
        if start is None or end is None:
            raise typer.BadParameter("Provide either --date or both --start and --end.")
        start_date = _parse_date(start, arg_name="--start")
        end_date = _parse_date(end, arg_name="--end")

    if end_date < start_date:
        raise typer.BadParameter("--end must be >= --start.")

    requested = snapshot_type.strip().lower()
    requested_types = {"lock", "pretip"} if requested == "both" else {requested}
    if not requested_types.issubset({"lock", "pretip"}):
        raise typer.BadParameter("--snapshot-type must be 'lock', 'pretip', or 'both'.")

    if season is not None:
        seasons = [season]
    else:
        seasons = sorted({_season_for_game_date(start_date), _season_for_game_date(end_date)})

    status_path = (status_file or _status_path(out_root)).resolve()
    status = _load_status(status_path)
    _write_status(status_path, status)

    candidates = _load_schedule_candidates(data_root, seasons=seasons, start_date=start_date, end_date=end_date)
    if not candidates:
        typer.echo(f"[slates-backfill] no schedule rows found for {start_date}..{end_date} seasons={seasons}")
        return

    typer.echo(
        f"[slates-backfill] seasons={seasons} dates={start_date}..{end_date} "
        f"games={len(candidates)} snapshot_type={requested} dry_run={dry_run} force={force} resume={resume}"
    )

    processed_games = 0
    for candidate in candidates:
        if max_games is not None and processed_games >= max_games:
            break

        schedule_row = freezer.ScheduleRow(
            season=candidate.season,
            game_id=candidate.game_id,
            game_date=candidate.game_date,
            tip_ts=candidate.tip_ts,
            schedule_row=candidate.row,
        )

        any_failed = False
        any_frozen = False
        any_skipped = False

        for snap in ("lock", "pretip"):
            if snap not in requested_types:
                continue
            parquet_path, manifest_path = _snapshot_paths(out_root, candidate, snap)
            exists_pair = parquet_path.exists() and manifest_path.exists()
            exists_any = parquet_path.exists() or manifest_path.exists()
            needs_repair = repair_partials and exists_any and not exists_pair

            if resume and exists_pair and not force:
                status["snapshots_skipped_existing"] = int(status.get("snapshots_skipped_existing", 0)) + 1
                any_skipped = True
                continue

            if dry_run:
                action = "repair" if needs_repair else "freeze"
                typer.echo(
                    f"[slates-backfill] {action} season={candidate.season} game_id={candidate.game_id} "
                    f"date={candidate.game_date} snapshot={snap}"
                )
                continue

            try:
                freezer._freeze_game_snapshot(
                    schedule_row,
                    snapshot_type=snap,
                    data_root=data_root,
                    out_root=out_root,
                    force=force or needs_repair,
                    history_days=history_days,
                    require_history=not allow_empty_history,
                )
            except Exception as exc:  # noqa: BLE001
                any_failed = True
                status["snapshots_failed"] = int(status.get("snapshots_failed", 0)) + 1
                status.setdefault("failures", []).append(
                    {
                        "season": candidate.season,
                        "game_id": candidate.game_id,
                        "game_date": candidate.game_date.isoformat(),
                        "snapshot_type": snap,
                        "error": str(exc),
                    }
                )
                typer.echo(
                    f"[slates-backfill] error season={candidate.season} game_id={candidate.game_id} "
                    f"snapshot={snap}: {exc}",
                    err=True,
                )
                _write_status(status_path, status)
                if stop_on_error:
                    raise
            else:
                any_frozen = True
                status["snapshots_frozen"] = int(status.get("snapshots_frozen", 0)) + 1

        processed_games += 1
        if any_failed:
            status["games_failed"] = int(status.get("games_failed", 0)) + 1
        else:
            status["games_completed"] = int(status.get("games_completed", 0)) + 1

        if any_frozen or any_skipped or any_failed:
            _write_status(status_path, status)

    typer.echo(
        "[slates-backfill] done "
        f"games_completed={status.get('games_completed', 0)} "
        f"games_failed={status.get('games_failed', 0)} "
        f"snapshots_frozen={status.get('snapshots_frozen', 0)} "
        f"snapshots_skipped_existing={status.get('snapshots_skipped_existing', 0)} "
        f"snapshots_failed={status.get('snapshots_failed', 0)} "
        f"status_file={status_path}"
    )


if __name__ == "__main__":  # pragma: no cover
    app()
