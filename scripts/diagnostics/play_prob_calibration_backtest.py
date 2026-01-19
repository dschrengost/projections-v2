"""Backtest play_prob calibration vs actual played labels.

This script compares predicted `play_prob` from minutes_v1 gold outputs against
NBA.com boxscore labels (played := minutes > 0) and prints:
- Per-slate Brier score
- Calibration buckets (mean predicted vs empirical played rate)

Usage:
  uv run python scripts/diagnostics/play_prob_calibration_backtest.py --start 2026-01-01 --end 2026-01-10
  uv run python scripts/diagnostics/play_prob_calibration_backtest.py --date 2026-01-08 --date 2026-01-09
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as date_cls
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from projections.paths import data_path
from projections.pipeline import control_plane

app = typer.Typer(add_completion=False)


def _season_from_date(day: date_cls) -> int:
    return day.year if day.month >= 8 else day.year - 1


def _iter_dates(start: date_cls, end: date_cls) -> list[date_cls]:
    out: list[date_cls] = []
    cur = start
    while cur <= end:
        out.append(cur)
        cur = cur.fromordinal(cur.toordinal() + 1)
    return out


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    if y.size == 0:
        return float("nan")
    return float(np.mean((y - p) ** 2))


def _ece(y: np.ndarray, p: np.ndarray, *, bins: int) -> float:
    if y.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(p)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
        if not np.any(mask):
            continue
        mean_p = float(np.mean(p[mask]))
        mean_y = float(np.mean(y[mask]))
        ece += (int(mask.sum()) / total) * abs(mean_p - mean_y)
    return float(ece)


@dataclass(frozen=True)
class SlateResult:
    date: str
    run_id: str
    n_pred: int
    n_label: int
    n_join: int
    brier: float
    ece: float
    mean_p: float
    mean_y: float
    pct_p0: float
    pct_p1: float


def _load_minutes_play_prob(root: Path, game_date: str) -> tuple[pd.DataFrame, str] | None:
    minutes_day = root / "gold" / "projections_minutes_v1" / f"game_date={game_date}"
    if not minutes_day.exists():
        return None
    run_id = control_plane.read_promoted_run_id(minutes_day)
    if run_id is None and control_plane.allow_unpromoted_run_reads():
        run_dirs = sorted([p for p in minutes_day.glob("run=*") if p.is_dir()], reverse=True)
        if run_dirs:
            run_id = run_dirs[0].name.split("=", 1)[1]
    if not run_id:
        return None

    run_dir = minutes_day / f"run={run_id}"
    minutes_path = run_dir / "minutes.parquet"
    if not minutes_path.exists():
        return None
    df = pd.read_parquet(minutes_path)
    if "player_id" not in df.columns:
        return None
    if "play_prob" not in df.columns:
        df["play_prob"] = 1.0
    df = df[["player_id", "play_prob"]].copy()
    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce")
    df = df.dropna(subset=["player_id"]).copy()
    df["player_id"] = df["player_id"].astype(int)
    df["play_prob"] = pd.to_numeric(df["play_prob"], errors="coerce").fillna(1.0).clip(0.0, 1.0)
    return df, str(run_id)


def _load_played_labels(root: Path, game_date: str) -> pd.DataFrame | None:
    day = pd.Timestamp(game_date).date()
    season = _season_from_date(day)
    path = root / "labels" / f"season={season}" / "boxscore_labels.parquet"
    if not path.exists():
        return None
    labels = pd.read_parquet(path)
    if labels.empty:
        return None
    if "game_date" not in labels.columns or "player_id" not in labels.columns or "minutes" not in labels.columns:
        return None
    labels["game_date"] = pd.to_datetime(labels["game_date"]).dt.date
    day_labels = labels.loc[labels["game_date"] == day, ["player_id", "minutes"]].copy()
    if day_labels.empty:
        return None
    day_labels["player_id"] = pd.to_numeric(day_labels["player_id"], errors="coerce")
    day_labels = day_labels.dropna(subset=["player_id"]).copy()
    day_labels["player_id"] = day_labels["player_id"].astype(int)
    day_labels["minutes"] = pd.to_numeric(day_labels["minutes"], errors="coerce").fillna(0.0).astype(float)
    out = day_labels.groupby("player_id", as_index=False)["minutes"].max()
    out["played"] = (out["minutes"] > 0.0).astype(int)
    return out[["player_id", "played"]]


def _bucket_table(y: np.ndarray, p: np.ndarray, *, bins: int) -> pd.DataFrame:
    edges = np.linspace(0.0, 1.0, bins + 1)
    bucket = np.digitize(p, edges[1:-1], right=False)
    df = pd.DataFrame({"bucket": bucket, "p": p, "y": y})
    out = (
        df.groupby("bucket", as_index=False)
        .agg(n=("y", "size"), mean_p=("p", "mean"), mean_y=("y", "mean"))
        .sort_values("bucket")
    )
    out["lo"] = edges[out["bucket"].to_numpy(dtype=int)]
    out["hi"] = edges[out["bucket"].to_numpy(dtype=int) + 1]
    out = out[["lo", "hi", "n", "mean_p", "mean_y"]]
    return out


@app.command()
def main(
    start: str | None = typer.Option(None, "--start", help="Start date (YYYY-MM-DD)."),
    end: str | None = typer.Option(None, "--end", help="End date (YYYY-MM-DD)."),
    date: list[str] = typer.Option([], "--date", help="Specific date(s); can repeat."),
    data_root: Path | None = typer.Option(None, "--data-root", help="Defaults to PROJECTIONS_DATA_ROOT."),
    bins: int = typer.Option(10, "--bins", help="Number of calibration buckets."),
) -> None:
    root = data_root or data_path()

    if date:
        dates = [pd.Timestamp(d).date() for d in date]
    else:
        if not start or not end:
            raise typer.BadParameter("Provide --date or both --start and --end.")
        dates = _iter_dates(pd.Timestamp(start).date(), pd.Timestamp(end).date())

    results: list[SlateResult] = []
    all_y: list[np.ndarray] = []
    all_p: list[np.ndarray] = []

    for day in dates:
        date_str = day.isoformat()
        pred_payload = _load_minutes_play_prob(root, date_str)
        if pred_payload is None:
            typer.echo(f"[play_prob] {date_str}: missing minutes predictions; skipping", err=True)
            continue
        pred_df, run_id = pred_payload
        labels_df = _load_played_labels(root, date_str)
        if labels_df is None:
            typer.echo(f"[play_prob] {date_str}: missing boxscore labels; skipping", err=True)
            continue

        joined = pred_df.merge(labels_df, on="player_id", how="inner")
        if joined.empty:
            typer.echo(f"[play_prob] {date_str}: no join rows; skipping", err=True)
            continue

        y = joined["played"].to_numpy(dtype=float)
        p = joined["play_prob"].to_numpy(dtype=float)
        slate_brier = _brier(y, p)
        slate_ece = _ece(y, p, bins=bins)
        results.append(
            SlateResult(
                date=date_str,
                run_id=run_id,
                n_pred=int(len(pred_df)),
                n_label=int(len(labels_df)),
                n_join=int(len(joined)),
                brier=slate_brier,
                ece=slate_ece,
                mean_p=float(np.mean(p)),
                mean_y=float(np.mean(y)),
                pct_p0=float(np.mean(p <= 1e-9)),
                pct_p1=float(np.mean(p >= 1.0 - 1e-9)),
            )
        )
        all_y.append(y)
        all_p.append(p)

    if not results:
        raise typer.Exit(code=2)

    summary = pd.DataFrame([r.__dict__ for r in results]).sort_values("date")
    typer.echo("\n## Per-slate play_prob calibration")
    typer.echo(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    y_all = np.concatenate(all_y) if all_y else np.array([], dtype=float)
    p_all = np.concatenate(all_p) if all_p else np.array([], dtype=float)
    typer.echo("\n## Overall")
    typer.echo(
        f"n={len(y_all)} brier={_brier(y_all, p_all):.4f} ece@{bins}={_ece(y_all, p_all, bins=bins):.4f} "
        f"mean_p={float(np.mean(p_all)):.4f} mean_y={float(np.mean(y_all)):.4f}"
    )

    buckets = _bucket_table(y_all, p_all, bins=bins)
    typer.echo("\n## Calibration buckets")
    typer.echo(buckets.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":  # pragma: no cover
    app()
