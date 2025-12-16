"""Aggregate sim_v2 FPTS worlds into projection quantiles."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import typer

from projections.paths import data_path

app = typer.Typer(add_completion=False, help="Aggregate sim_v2 worlds into projections.")

FPTS_CANDIDATES: tuple[str, ...] = ("dk_fpts_world", "dk_fpts_sim", "dk_fpts")
PROFILE_COLUMNS: tuple[str, ...] = ("sim_profile", "profile")
DATE_COLUMNS: tuple[str, ...] = ("game_date", "date")
OUTPUT_FILENAME = "sim_v2_projections.parquet"
STAT_COLUMNS: tuple[str, ...] = ("pts_world", "reb_world", "ast_world", "stl_world", "blk_world", "tov_world")
DEFAULT_OUT_ROOT = data_path("artifacts", "sim_v2", "projections")
MINUTES_COLUMN = "minutes_sim"


def _parse_date(value: str) -> date:
    try:
        return datetime.fromisoformat(value).date()
    except ValueError as exc:
        raise typer.BadParameter(f"Invalid date: {value}") from exc


def _date_range(start: date, end: date) -> Iterable[date]:
    if start > end:
        raise typer.BadParameter(f"start_date {start} is after end_date {end}")
    delta = (end - start).days
    for offset in range(delta + 1):
        yield start + timedelta(days=offset)


def _resolve_date_column(columns: Sequence[str]) -> str:
    for candidate in DATE_COLUMNS:
        if candidate in columns:
            return candidate
    raise ValueError(f"Expected one of date columns {DATE_COLUMNS}, found: {columns}")


def _resolve_fpts_column(columns: Sequence[str]) -> str:
    for candidate in FPTS_CANDIDATES:
        if candidate in columns:
            return candidate
    raise ValueError(f"Expected FPTS column in {FPTS_CANDIDATES}, found: {columns}")


def _load_worlds_partition(partition_dir: Path) -> pd.DataFrame | None:
    if not partition_dir.exists():
        typer.echo(f"[sim_v2] skip {partition_dir.name}: partition missing at {partition_dir}")
        return None
    paths = sorted(partition_dir.glob("*.parquet"))
    if not paths:
        typer.echo(f"[sim_v2] skip {partition_dir.name}: no parquet files found")
        return None
    frames = [pd.read_parquet(path) for path in paths]
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def _filter_profile(df: pd.DataFrame, profile: str | None) -> pd.DataFrame:
    if not profile:
        return df
    for col in PROFILE_COLUMNS:
        if col in df.columns:
            filtered = df[df[col] == profile].copy()
            if filtered.empty:
                typer.echo(f"[sim_v2] no rows after filtering {col} == {profile}")
            return filtered
    return df


def _aggregate_for_date(
    game_date: date, *, worlds_root: Path, profile: str | None, include_std: bool
) -> tuple[pd.DataFrame, dict] | tuple[None, None]:
    partition_dir = worlds_root / f"game_date={game_date.isoformat()}"
    df = _load_worlds_partition(partition_dir)
    if df is None or df.empty:
        return None, None

    df = _filter_profile(df, profile)
    if df.empty:
        typer.echo(f"[sim_v2] {game_date}: no worlds rows after profile filter")
        return None, None

    date_col = _resolve_date_column(df.columns)
    df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()

    key_cols = [date_col, "game_id", "team_id", "player_id"]
    profile_col = next((col for col in PROFILE_COLUMNS if col in df.columns), None)
    if profile_col:
        key_cols.append(profile_col)
    missing = [col for col in key_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required key columns: {missing}")

    fpts_col = _resolve_fpts_column(df.columns)
    for stat_col in STAT_COLUMNS:
        if stat_col not in df.columns:
            df[stat_col] = pd.Series(np.nan, index=df.index, dtype=float)
        else:
            df[stat_col] = pd.to_numeric(df[stat_col], errors="coerce")
    has_minutes = MINUTES_COLUMN in df.columns

    def _agg_func(group: pd.DataFrame) -> pd.Series:
        fpts_series = group[fpts_col]
        quantiles = fpts_series.quantile([0.05, 0.10, 0.25, 0.50, 0.75, 0.95])
        payload = {
            "dk_fpts_mean": float(fpts_series.mean()),
            "dk_fpts_p05": float(quantiles.loc[0.05]),
            "dk_fpts_p10": float(quantiles.loc[0.10]),
            "dk_fpts_p25": float(quantiles.loc[0.25]),
            "dk_fpts_p50": float(quantiles.loc[0.50]),
            "dk_fpts_p75": float(quantiles.loc[0.75]),
            "dk_fpts_p95": float(quantiles.loc[0.95]),
            "pts_mean": float(group["pts_world"].mean()),
            "reb_mean": float(group["reb_world"].mean()),
            "ast_mean": float(group["ast_world"].mean()),
            "stl_mean": float(group["stl_world"].mean()),
            "blk_mean": float(group["blk_world"].mean()),
            "tov_mean": float(group["tov_world"].mean()),
        }
        if has_minutes:
            payload["minutes_sim_mean"] = float(group[MINUTES_COLUMN].mean())
        if include_std:
            payload["dk_fpts_std"] = float(fpts_series.std(ddof=0))
        return pd.Series(payload)

    grouped = df.groupby(key_cols, dropna=False).apply(_agg_func).reset_index()

    summary = {
        "game_date": game_date.isoformat(),
        "fpts_column": fpts_col,
        "date_column": date_col,
        "n_rows": int(grouped.shape[0]),
        "n_worlds": int(df["world_id"].nunique()) if "world_id" in df.columns else None,
    }
    return grouped, summary


@app.command()
def main(
    start_date: str = typer.Option(..., "--start-date", help="Inclusive start date (YYYY-MM-DD)."),
    end_date: str = typer.Option(..., "--end-date", help="Inclusive end date (YYYY-MM-DD)."),
    profile: str = typer.Option("baseline", "--profile", help="Sim profile to filter when available."),
    worlds_root: Path | None = typer.Option(
        None,
        "--worlds-root",
        help="Root directory of sim_v2 worlds (default: <data_root>/artifacts/sim_v2/worlds_fpts_v2).",
    ),
    out_root: Path | None = typer.Option(
        None,
        "--out-root",
        help="Output root for projections (default: <data_root>/artifacts/sim_v2/projections).",
    ),
    run_id: str | None = typer.Option(None, "--run-id", help="Run identifier (default: current UTC timestamp)."),
    include_std: bool = typer.Option(True, "--std/--no-std", help="Compute fpts_std in outputs."),
) -> None:
    start = _parse_date(start_date)
    end = _parse_date(end_date)

    worlds_root = worlds_root or data_path("artifacts", "sim_v2", "worlds_fpts_v2")
    out_root = out_root or DEFAULT_OUT_ROOT
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    worlds_root = Path(worlds_root)
    out_root = Path(out_root)

    processed_any = False
    for game_day in _date_range(start, end):
        df_out, summary = _aggregate_for_date(
            game_day, worlds_root=worlds_root, profile=profile, include_std=include_std
        )
        if df_out is None or summary is None:
            continue

        out_dir = out_root / f"date={game_day.isoformat()}"
        if run_id:
            out_dir = out_dir / f"run={run_id}"
        out_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = out_dir / OUTPUT_FILENAME
        df_out.to_parquet(parquet_path, index=False)

        summary_payload = {
            **summary,
            "profile": profile,
            "run_id": run_id,
            "worlds_root": str(worlds_root),
            "output_path": str(parquet_path),
        }
        (out_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")
        latest_pointer = out_dir.parent / "latest_run.json"
        latest_pointer.write_text(json.dumps({"run_id": run_id}, indent=2, sort_keys=True), encoding="utf-8")

        sample_cols = [
            summary["date_column"],
            "player_id",
            "dk_fpts_mean",
            "dk_fpts_p10",
            "dk_fpts_p50",
            "dk_fpts_p95",
        ]
        sample = df_out[[col for col in sample_cols if col in df_out.columns]].head(5)
        typer.echo(f"[sim_v2] wrote {parquet_path} rows={df_out.shape[0]} n_worlds={summary_payload['n_worlds']}")
        typer.echo(sample.to_string(index=False))
        processed_any = True

    if not processed_any:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
