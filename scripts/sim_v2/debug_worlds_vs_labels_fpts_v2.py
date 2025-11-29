"""Evaluate simulated worlds against DK FPTS labels for a date range."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import typer

app = typer.Typer(add_completion=False)


def _parse_date(value: str) -> date:
    try:
        return datetime.fromisoformat(value).date()
    except ValueError as exc:
        raise typer.BadParameter(f"Invalid date: {value}") from exc


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


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not mask.any():
        return float("nan")
    return float(np.abs(y_true[mask] - y_pred[mask]).mean())


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not mask.any():
        return float("nan")
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if mask.sum() < 2:
        return float("nan")
    y_t = y_true[mask]
    y_p = y_pred[mask]
    denom = np.sum((y_t - y_t.mean()) ** 2)
    if denom <= 0:
        return float("nan")
    return float(1.0 - np.sum((y_t - y_p) ** 2) / denom)


@app.command()
def main(
    data_root: Path = typer.Option(..., "--data-root"),
    worlds_root: Path = typer.Option(..., "--worlds-root"),
    start_date: str = typer.Option(..., "--start-date"),
    end_date: str = typer.Option(..., "--end-date"),
    n_worlds: int = typer.Option(2000, "--n-worlds"),
) -> None:
    start_dt = _parse_date(start_date)
    end_dt = _parse_date(end_date)
    start_ts = pd.Timestamp(start_dt).normalize()
    end_ts = pd.Timestamp(end_dt).normalize()

    partitions = _iter_partitions(data_root, start_ts, end_ts)
    if not partitions:
        typer.echo("[debug_worlds_v2] no base partitions in range; exiting.")
        raise typer.Exit(code=0)
    base_frames = [pd.read_parquet(p) for p in partitions]
    base_df = pd.concat(base_frames, ignore_index=True)
    base_df["game_date"] = pd.to_datetime(base_df["game_date"]).dt.normalize()
    base_keep_cols = ["game_date", "game_id", "team_id", "player_id", "dk_fpts_actual", "is_starter", "minutes_p50"]
    base_df = base_df[[c for c in base_keep_cols if c in base_df.columns]]

    worlds_metrics = []

    for game_date, base_slice in base_df.groupby("game_date"):
        date_token = str(game_date.date())
        date_world_dir = worlds_root / f"game_date={date_token}"
        if not date_world_dir.exists():
            typer.echo(f"[debug_worlds_v2] missing worlds for {date_token}; skipping.")
            continue
        world_files = sorted(date_world_dir.glob("world=*.parquet"))[:n_worlds]
        if not world_files:
            typer.echo(f"[debug_worlds_v2] no world files for {date_token}; skipping.")
            continue

        for wf in world_files:
            world_df = pd.read_parquet(wf)
            join_cols = ["game_id", "team_id", "player_id"]
            merged = base_slice.merge(world_df, on=join_cols, how="inner", suffixes=("_base", ""))
            if merged.empty or "dk_fpts_world" not in merged.columns:
                continue
            label = pd.to_numeric(merged["dk_fpts_actual"], errors="coerce").to_numpy()
            world_vals = pd.to_numeric(merged["dk_fpts_world"], errors="coerce").to_numpy()
            mae = _mae(label, world_vals)
            rmse = _rmse(label, world_vals)
            r2 = _r2(label, world_vals)
            worlds_metrics.append({"world_id": int(merged.get("world_id", pd.Series([0])).iloc[0]), "mae": mae, "rmse": rmse, "r2": r2})

    if not worlds_metrics:
        typer.echo("[debug_worlds_v2] no worlds evaluated; exiting.")
        raise typer.Exit(code=0)

    maes = np.array([m["mae"] for m in worlds_metrics], dtype=float)
    rmses = np.array([m["rmse"] for m in worlds_metrics], dtype=float)
    r2s = np.array([m["r2"] for m in worlds_metrics], dtype=float)
    summary = {
        "worlds_evaluated": len(worlds_metrics),
        "mae_mean": float(np.nanmean(maes)),
        "mae_median": float(np.nanmedian(maes)),
        "rmse_mean": float(np.nanmean(rmses)),
        "rmse_median": float(np.nanmedian(rmses)),
        "r2_mean": float(np.nanmean(r2s)),
        "r2_median": float(np.nanmedian(r2s)),
    }

    typer.echo(
        f"[debug_worlds_v2] worlds={summary['worlds_evaluated']} "
        f"mae_mean={summary['mae_mean']:.3f} rmse_mean={summary['rmse_mean']:.3f} "
        f"r2_mean={summary['r2_mean']:.3f}"
    )

    out_path = (
        data_root
        / "artifacts"
        / "sim_v2"
        / f"debug_worlds_vs_labels_{start_dt}_{end_dt}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_payload = {
        "summary": summary,
        "world_metrics": worlds_metrics,
        "meta": {
            "start_date": start_dt.isoformat(),
            "end_date": end_dt.isoformat(),
            "worlds_root": str(worlds_root),
            "n_worlds": n_worlds,
        },
    }
    out_path.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")
    typer.echo(f"[debug_worlds_v2] wrote {out_path}")


if __name__ == "__main__":
    app()
