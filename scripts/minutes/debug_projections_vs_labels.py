"""
Compare minutes_v1 projections vs labels over a date range.

Example:
    uv run python -m scripts.minutes.debug_projections_vs_labels \
      --data-root  /home/daniel/projections-data \
      --start-date 2023-10-24 \
      --end-date   2023-10-24

With rescore + debug combined:
    uv run python -m scripts.minutes.rescore_and_debug \
      --data-root /home/daniel/projections-data \
      --start-date 2023-10-24 \
      --end-date   2023-10-24
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import typer

from projections.paths import data_path

app = typer.Typer(add_completion=False)


def _iter_days(start: date, end: date):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def _load_proj_partition(root: Path, day: date) -> pd.DataFrame | None:
    """Try both flat and game_date= layouts."""

    iso = day.isoformat()
    candidates = [
        root / iso / "minutes.parquet",
        root / f"game_date={iso}" / "minutes.parquet",
    ]
    for path in candidates:
        if path.exists():
            try:
                df = pd.read_parquet(path)
            except Exception:
                continue
            df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
            return df
    return None


def _season_start(day: pd.Timestamp) -> int:
    return day.year if day.month >= 8 else day.year - 1


def _load_labels_partition(labels_root: Path, day: date) -> pd.DataFrame | None:
    ts = pd.Timestamp(day)
    season = _season_start(ts)
    iso = day.isoformat()
    labels_path = labels_root / f"season={season}" / f"game_date={iso}" / "labels.parquet"
    if not labels_path.exists():
        return None
    df = pd.read_parquet(labels_path)
    if df.empty:
        return None
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    if "minutes" in df.columns and "minutes_actual" not in df.columns:
        df = df.rename(columns={"minutes": "minutes_actual"})
    df["minutes_actual"] = pd.to_numeric(df["minutes_actual"], errors="coerce")
    df = df.dropna(subset=["minutes_actual"])
    numeric_cols = ["game_id", "player_id", "team_id", "season"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "season" not in df.columns:
        df["season"] = season
    return df


def _normalize_season_column(df: pd.DataFrame) -> pd.DataFrame:
    if "season" not in df.columns:
        if "game_date" in df.columns:
            df["season"] = pd.to_datetime(df["game_date"]).apply(lambda ts: ts.year if ts.month >= 8 else ts.year - 1)
        return df

    def _coerce_season(val: object) -> float:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return np.nan
        text = str(val)
        try:
            return float(text)
        except ValueError:
            if "-" in text:
                prefix = text.split("-", 1)[0]
                try:
                    return float(prefix)
                except ValueError:
                    return np.nan
        return np.nan

    df["season"] = df["season"].apply(_coerce_season)
    df["season"] = df["season"].fillna(
        pd.to_datetime(df["game_date"]).apply(lambda ts: ts.year if ts.month >= 8 else ts.year - 1)
        if "game_date" in df.columns
        else np.nan
    )
    df["season"] = df["season"].astype("Int64")
    return df


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.abs(y_pred - y_true).mean())


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(((y_pred - y_true) ** 2).mean()))


@app.command()
def main(
    data_root: Optional[Path] = typer.Option(
        None, help="Root containing data (defaults to PROJECTIONS_DATA_ROOT or ./data)."
    ),
    start_date: str = typer.Option(..., help="Start date YYYY-MM-DD (inclusive)."),
    end_date: str = typer.Option(..., help="End date YYYY-MM-DD (inclusive)."),
    season_type: str = typer.Option("Regular Season", help="Unused placeholder for compatibility/logging."),
) -> None:
    root = data_root or data_path()
    start = pd.Timestamp(start_date).date()
    end = pd.Timestamp(end_date).date()
    proj_root = root / "gold" / "projections_minutes_v1"
    labels_root = root / "gold" / "labels_minutes_v1"

    merged_frames: list[pd.DataFrame] = []

    for day in _iter_days(start, end):
        proj = _load_proj_partition(proj_root, day)
        labels = _load_labels_partition(labels_root, day)
        if proj is None:
            typer.echo(f"[{day}] projections missing; skip", err=True)
            continue
        if labels is None:
            typer.echo(f"[{day}] labels missing; skip", err=True)
            continue

        proj = _normalize_season_column(proj)
        labels = _normalize_season_column(labels)
        for col in ("game_id", "player_id", "team_id"):
            if col in proj.columns:
                proj[col] = pd.to_numeric(proj[col], errors="coerce")
            if col in labels.columns:
                labels[col] = pd.to_numeric(labels[col], errors="coerce")

        join_cols = [c for c in ["season", "game_id", "team_id", "player_id"] if c in proj.columns and c in labels.columns]
        if len(join_cols) < 3:
            typer.echo(f"[{day}] insufficient join keys ({join_cols}); skip", err=True)
            continue

        merged = proj.merge(labels, on=join_cols, how="inner", suffixes=("_pred", "_actual"))
        if merged.empty or "minutes_p50" not in merged.columns or "minutes_actual" not in merged.columns:
            typer.echo(f"[{day}] merged empty or missing minutes columns; skip", err=True)
            continue

        merged["minutes_p50"] = pd.to_numeric(merged["minutes_p50"], errors="coerce")
        merged["minutes_actual"] = pd.to_numeric(merged["minutes_actual"], errors="coerce")
        merged = merged.dropna(subset=["minutes_p50", "minutes_actual"])
        if merged.empty:
            typer.echo(f"[{day}] merged rows all NaN after coercion; skip", err=True)
            continue

        y_true = merged["minutes_actual"].to_numpy()
        y_pred = merged["minutes_p50"].to_numpy()
        corr = float(pd.Series(y_pred).corr(pd.Series(y_true)))
        day_mae = _mae(y_true, y_pred)
        day_rmse = _rmse(y_true, y_pred)
        typer.echo(
            f"[{day}] merged_rows={len(merged)} mae={day_mae:.4f} rmse={day_rmse:.4f} corr={corr:.4f}"
        )
        typer.echo(merged[["minutes_p50", "minutes_actual"]].describe().to_string())
        merged_frames.append(merged)
        if "is_starter" in merged.columns:
            for flag, label in [(1, "starters"), (0, "bench")]:
                sub = merged[merged["is_starter"] == flag]
                if sub.empty:
                    continue
                mae_s = _mae(sub["minutes_actual"].to_numpy(), sub["minutes_p50"].to_numpy())
                rmse_s = _rmse(sub["minutes_actual"].to_numpy(), sub["minutes_p50"].to_numpy())
                typer.echo(f"[{day}][{label}] n={len(sub)} mae={mae_s:.4f} rmse={rmse_s:.4f}")

    if not merged_frames:
        typer.echo("[summary] no merged rows across requested window.")
        raise typer.Exit()

    all_merged = pd.concat(merged_frames, ignore_index=True)
    y_true = all_merged["minutes_actual"].to_numpy()
    y_pred = all_merged["minutes_p50"].to_numpy()
    overall_corr = float(pd.Series(y_pred).corr(pd.Series(y_true)))
    overall_mae = _mae(y_true, y_pred)
    overall_rmse = _rmse(y_true, y_pred)
    typer.echo(
        f"[summary] total_merged={len(all_merged)} overall_mae={overall_mae:.4f} "
        f"overall_rmse={overall_rmse:.4f} overall_corr={overall_corr:.4f}"
    )
    typer.echo("minutes_p50 vs minutes_actual describe():")
    typer.echo(all_merged[["minutes_p50", "minutes_actual"]].describe().to_string())
    if "is_starter" in all_merged.columns:
        for flag, label in [(1, "starters"), (0, "bench")]:
            sub = all_merged[all_merged["is_starter"] == flag]
            if sub.empty:
                continue
            mae_s = _mae(sub["minutes_actual"].to_numpy(), sub["minutes_p50"].to_numpy())
            rmse_s = _rmse(sub["minutes_actual"].to_numpy(), sub["minutes_p50"].to_numpy())
            typer.echo(f"[summary][{label}] n={len(sub)} mae={mae_s:.4f} rmse={rmse_s:.4f}")


if __name__ == "__main__":
    app()
