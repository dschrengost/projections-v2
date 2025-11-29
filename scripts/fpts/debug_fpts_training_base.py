"""Inspect fpts_training_base partitions and check feature leakage.

Example:
    uv run python -m scripts.fpts.debug_fpts_training_base \
      --data-root /home/daniel/projections-data \
      --start-date 2023-10-24 \
      --end-date   2025-11-26 \
      --fpts-run-id fpts_v2_stage0_20251129_062655
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from projections.fpts_v2.current import _get_data_root

app = typer.Typer(add_completion=False)


def _parse_date(value: str) -> date:
    try:
        return datetime.fromisoformat(value).date()
    except ValueError as exc:
        raise typer.BadParameter(f"Invalid date: {value}") from exc


def _season_from_date(day: date) -> int:
    return day.year if day.month >= 8 else day.year - 1


def _iter_days(start: date, end: date):
    cursor = start
    while cursor <= end:
        yield cursor
        cursor += timedelta(days=1)


def _iter_partitions(root: Path, start: pd.Timestamp, end: pd.Timestamp) -> list[Path]:
    base = root / "gold" / "fpts_training_base"
    partitions: list[Path] = []
    for season_dir in base.glob("season=*"):
        for day_dir in season_dir.glob("game_date=*"):
            try:
                day = pd.Timestamp(day_dir.name.split("=", 1)[1]).normalize()
            except ValueError:
                continue
            if day < start or day > end:
                continue
            candidate = day_dir / "fpts_training_base.parquet"
            if candidate.exists():
                partitions.append(candidate)
    return sorted(partitions)


def _load_feature_cols(run_dir: Path) -> list[str]:
    feature_cols_path = run_dir / "feature_cols.json"
    if not feature_cols_path.exists():
        return []
    payload = json.loads(feature_cols_path.read_text(encoding="utf-8"))
    return list(payload.get("feature_cols") or [])


def _load_meta(run_dir: Path) -> dict:
    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _leakage_check(feature_cols: list[str]) -> list[str]:
    banned_patterns = {
        "minutes_actual",
        "dk_fpts_actual",
        "pts",
        "fgm",
        "fga",
        "fg3m",
        "fg3a",
        "fta",
        "ftm",
        "reb",
        "ast",
        "stl",
        "blk",
        "tov",
    }
    banned_found = [col for col in feature_cols for banned in banned_patterns if col == banned]
    return banned_found


def _nan_fraction(df: pd.DataFrame, cols: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for col in cols:
        if col not in df.columns:
            out[col] = 1.0
            continue
        series = df[col]
        out[col] = float(series.isna().mean())
    return out


@app.command()
def main(
    data_root: Path = typer.Option(None, "--data-root"),
    start_date: str = typer.Option(..., "--start-date"),
    end_date: str = typer.Option(..., "--end-date"),
    fpts_run_id: Optional[str] = typer.Option(None, "--fpts-run-id"),
) -> None:
    root = data_root or _get_data_root()
    start_dt = _parse_date(start_date)
    end_dt = _parse_date(end_date)
    start_ts = pd.Timestamp(start_dt).normalize()
    end_ts = pd.Timestamp(end_dt).normalize()

    paths = _iter_partitions(root, start_ts, end_ts)
    if not paths:
        typer.echo("[debug_fpts] no fpts_training_base partitions found in range.")
        raise typer.Exit(code=0)

    frames = [pd.read_parquet(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.normalize()

    seasons = sorted(df["season"].dropna().unique().tolist())
    typer.echo(
        f"[debug_fpts] rows={len(df):,} seasons={seasons} date_range=({df['game_date'].min().date()} to {df['game_date'].max().date()})"
    )
    typer.echo(f"[debug_fpts] columns ({len(df.columns)}): {sorted(df.columns)}")

    feature_cols: list[str] = []
    meta: dict = {}
    if fpts_run_id:
        run_dir = root / "artifacts" / "fpts_v2" / "runs" / fpts_run_id
        feature_cols = _load_feature_cols(run_dir)
        meta = _load_meta(run_dir)
        typer.echo(f"[debug_fpts] run_id={fpts_run_id} feature_cols={len(feature_cols)}")
        if meta:
            typer.echo(f"[debug_fpts] meta={meta}")
        if feature_cols:
            typer.echo(f"[debug_fpts] feature_cols list: {feature_cols}")

    key_cols = [
        "dk_fpts_actual",
        "minutes_p10",
        "minutes_p50",
        "minutes_p90",
        "minutes_actual",
        "minutes_pred_play_prob",
        "pred_fga2_per_min",
        "pred_fga3_per_min",
        "pred_fta_per_min",
        "pred_ast_per_min",
        "pred_tov_per_min",
        "pred_oreb_per_min",
        "pred_dreb_per_min",
        "pred_stl_per_min",
        "pred_blk_per_min",
        "home_flag",
        "team_itt",
        "opp_itt",
    ]
    nan_fracs = _nan_fraction(df, key_cols)
    typer.echo("[debug_fpts] NaN fractions for key columns:")
    for col, frac in nan_fracs.items():
        typer.echo(f"  {col}: {frac:.3f}")

    if feature_cols:
        banned = _leakage_check(feature_cols)
        if banned:
            typer.echo(f"[leakage_check] banned feature columns found: {sorted(set(banned))}")
        else:
            typer.echo("[leakage_check] no banned columns in feature_cols")


if __name__ == "__main__":
    app()
