"""Show live Questionable/Doubtful players with conditional vs expected minutes."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
import typer

from projections import paths
from projections.labels import derive_starter_flag_labels
from projections.minutes_v1.production import load_production_minutes_bundle
from scripts.minutes_debug.status_eval_minutes import score_with_bundle
from projections.cli import build_minutes_live as build_minutes_live_cli


UTC = timezone.utc
app = typer.Typer(help=__doc__)

STATUS_CANDIDATES: tuple[str, ...] = ("status", "injury_status", "availability_status")
CONTEXT_COLUMNS: dict[str, str] = {
    "rotation_minutes_std_5g": "rot_std_5g",
    "role_change_rate_10g": "role_chg_10g",
    "depth_same_pos_active": "depth_same_pos",
    "blowout_risk_score": "blowout_risk",
    "season_phase": "season_phase",
}


def _resolve_status_column(df: pd.DataFrame, override: str | None = None) -> str:
    if override:
        if override not in df.columns:
            raise ValueError(f"Requested status column '{override}' not present in features.")
        return override
    for candidate in STATUS_CANDIDATES:
        if candidate in df.columns:
            return candidate
    raise ValueError(f"Unable to find status column; expected one of {', '.join(STATUS_CANDIDATES)}")


def _normalize_run_timestamp(value: datetime | None) -> pd.Timestamp:
    candidate = value if value is not None else datetime.now(tz=UTC)
    if candidate.tzinfo is None:
        candidate = candidate.replace(tzinfo=UTC)
    return build_minutes_live_cli._normalize_run_timestamp(candidate)


def _invoke_build_minutes_live(
    *,
    date: datetime,
    run_ts: pd.Timestamp,
    data_root: Path,
    out_root: Path,
    validate_active_roster: bool,
    enforce_active_roster: bool,
    lock_buffer_minutes: int,
) -> None:
    run_str = run_ts.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%S")
    cmd = [
        sys.executable,
        "-m",
        "projections.cli.build_minutes_live",
        "--date",
        date.strftime("%Y-%m-%d"),
        "--run-as-of-ts",
        run_str,
        "--data-root",
        str(data_root),
        "--out-root",
        str(out_root),
        "--lock-buffer-minutes",
        str(lock_buffer_minutes),
    ]
    if not validate_active_roster:
        cmd.append("--skip-active-roster")
    if enforce_active_roster:
        cmd.append("--enforce-active-roster")
    typer.echo(f"[minutes-live-q] Running build_minutes_live via: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _load_live_features(
    *,
    date: datetime,
    run_as_of_ts: datetime | None,
    data_root: Path,
    out_root: Path,
    features_path: Path | None,
    keep_build_dir: bool,
    validate_active_roster: bool,
    enforce_active_roster: bool,
    lock_buffer_minutes: int,
) -> tuple[pd.DataFrame, Path, Path | None]:
    if features_path is not None:
        resolved = features_path.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Live features parquet missing at {resolved}")
        typer.echo(f"[minutes-live-q] Loading pre-built features from {resolved}")
        return pd.read_parquet(resolved), resolved, None

    work_root = out_root.expanduser().resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix="minutes_live_debug_", dir=str(work_root)))
    run_ts = _normalize_run_timestamp(run_as_of_ts)
    typer.echo(
        f"[minutes-live-q] Building live features for {date.date()} run_ts={run_ts.isoformat()} into {temp_dir}"
    )
    _invoke_build_minutes_live(
        date=date,
        run_ts=run_ts,
        data_root=data_root,
        out_root=temp_dir,
        validate_active_roster=validate_active_roster,
        enforce_active_roster=enforce_active_roster,
        lock_buffer_minutes=lock_buffer_minutes,
    )
    day = build_minutes_live_cli._normalize_day(date)
    day_dir = temp_dir / day.strftime("%Y-%m-%d")
    run_id = build_minutes_live_cli._format_run_id(run_ts)
    feature_path = day_dir / f"run={run_id}" / build_minutes_live_cli.FEATURE_FILENAME
    if not feature_path.exists():
        raise FileNotFoundError(f"Expected live feature parquet missing at {feature_path}")
    df = pd.read_parquet(feature_path)
    if not keep_build_dir:
        shutil.rmtree(temp_dir, ignore_errors=True)
        temp_dir = None
    return df, feature_path, temp_dir


def _qish_mask(status_series: pd.Series) -> pd.Series:
    normalized = status_series.fillna("").astype(str).str.upper()
    return normalized.str.contains("Q") | normalized.str.contains("DOUBT") | normalized.str.contains("GTD")


def _format_context(row: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for column, label in CONTEXT_COLUMNS.items():
        value = row.get(column)
        if value is None or pd.isna(value):
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            parts.append(f"{label}={float(value):.2f}")
        else:
            parts.append(f"{label}={value}")
    play_prob = row.get("play_prob")
    if play_prob is not None and not pd.isna(play_prob):
        parts.insert(0, f"play_prob={float(play_prob):.2f}")
    return ", ".join(parts)


def _print_players(rows: Iterable[Mapping[str, Any]], status_col: str) -> None:
    header = f"{'player':<24} {'team':<5} {'game':<10} {'status':<8} {'starter':<7} {'p50_cond':>9} {'p50_expected':>14}  context"
    typer.echo(header)
    typer.echo("-" * len(header))
    for row in rows:
        player_label = str(row.get("player_name") or row.get("player_id") or "?")[:24]
        team_label = str(row.get("team_id") or "?")
        game_label = str(row.get("game_id") or "?")
        status_label = str(row.get(status_col) or "?")
        starter_val = pd.to_numeric(row.get("starter_flag"), errors="coerce")
        starter_flag = int(starter_val) if starter_val is not None and not pd.isna(starter_val) else 0
        p50 = float(row.get("minutes_p50") or 0.0)
        expected = row.get("minutes_expected_p50")
        expected_str = f"{float(expected):.1f}" if expected is not None and not pd.isna(expected) else "--"
        context = _format_context(row)
        typer.echo(
            f"{player_label:<24} {team_label:<5} {game_label:<10} {status_label:<8} {starter_flag:<7d} "
            f"{p50:9.1f} {expected_str:>14}  {context}"
        )


@app.command()
def main(
    date: datetime = typer.Option(..., "--date", help="Slate date (YYYY-MM-DD)."),
    run_as_of_ts: datetime | None = typer.Option(
        None,
        "--run-as-of-ts",
        help="Information timestamp used for the live slice (defaults to now UTC).",
    ),
    data_root: Path = typer.Option(
        paths.get_data_root(),
        "--data-root",
        help="Base data directory (defaults to PROJECTIONS_DATA_ROOT or ./data).",
    ),
    features_path: Path | None = typer.Option(
        None,
        "--features-path",
        help="Optional explicit live features parquet to score instead of building new features.",
    ),
    work_root: Path = typer.Option(
        Path("/tmp/minutes_live_debug"),
        "--work-root",
        help="Scratch directory used when building live features on the fly.",
    ),
    keep_build_dir: bool = typer.Option(
        False,
        "--keep-build-dir",
        help="Keep the temporary live features directory for inspection instead of deleting it.",
    ),
    validate_active_roster: bool = typer.Option(
        True,
        "--validate-active-roster/--skip-active-roster",
        help="Toggle NBA.com active roster validation during live feature construction.",
    ),
    enforce_active_roster: bool = typer.Option(
        False,
        "--enforce-active-roster",
        help="Drop players missing from the active roster snapshot during live building.",
    ),
    lock_buffer_minutes: int = typer.Option(0, "--lock-buffer-minutes", help="Skip games already past lock by this many minutes."),
    status_column: str | None = typer.Option(None, "--status-column", help="Override the status column name when filtering."),
    limit: int | None = typer.Option(None, "--limit", help="Maximum number of players to display."),
) -> None:
    date = date if date.tzinfo is None else date.astimezone(UTC).replace(tzinfo=None)
    data_root = data_root.expanduser().resolve()
    work_root = work_root.expanduser().resolve()

    feature_df, source_path, temp_dir = _load_live_features(
        date=date,
        run_as_of_ts=run_as_of_ts,
        data_root=data_root,
        out_root=work_root,
        features_path=features_path,
        keep_build_dir=keep_build_dir,
        validate_active_roster=validate_active_roster,
        enforce_active_roster=enforce_active_roster,
        lock_buffer_minutes=lock_buffer_minutes,
    )
    typer.echo(f"[minutes-live-q] Scoring live slice from {source_path}")
    if "starter_flag" not in feature_df.columns:
        feature_df = derive_starter_flag_labels(feature_df, output_col="starter_flag")
    status_col = _resolve_status_column(feature_df, override=status_column)

    loaded = load_production_minutes_bundle()
    if loaded.get("mode") == "dual":
        bundle = dict(loaded.get("late_bundle") or {})
    else:
        bundle = dict(loaded)
    scored = score_with_bundle(feature_df, bundle)
    mask = _qish_mask(scored[status_col])
    q_players = scored.loc[mask].copy()
    if q_players.empty:
        typer.echo("[minutes-live-q] No Questionable/Doubtful players detected in the live slice.")
        return
    q_players.sort_values(["starter_flag", "minutes_p50"], ascending=[False, False], inplace=True)
    if limit is not None:
        q_players = q_players.head(limit)
    _print_players(q_players.to_dict(orient="records"), status_col=status_col)
    if keep_build_dir and temp_dir is not None:
        typer.echo(f"[minutes-live-q] Temporary run preserved under {temp_dir}")


if __name__ == "__main__":  # pragma: no cover
    app()
