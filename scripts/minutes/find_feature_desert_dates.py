"""
Detect "feature desert" dates for minutes_v1 features and write a CSV summary.

Example:
    uv run python -m scripts.minutes.find_feature_desert_dates \
      --data-root  /home/daniel/projections-data \
      --start-date 2022-10-01 \
      --end-date   2025-02-28 \
      --output-csv /home/daniel/projections-data/artifacts/minutes_v1/feature_deserts.csv
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import typer

from projections.cli import score_minutes_v1
from projections.paths import data_path

app = typer.Typer(add_completion=False)

MINUTES_ROLLS = [
    "min_last1",
    "min_last3",
    "min_last5",
    "roll_mean_3",
    "roll_mean_5",
    "roll_mean_10",
    "sum_min_7d",
    "z_vs_10",
    "rotation_minutes_std_5g",
    "recent_start_pct_10",
]
ODDS = ["spread_home", "total", "blowout_index", "blowout_risk_score", "close_game_score"]
DISPERSION = ["team_minutes_dispersion_prior"]


def _iter_days(start: date, end: date):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def _group_stats(df: pd.DataFrame, cols: list[str]) -> tuple[float, float]:
    present = [c for c in cols if c in df.columns]
    if not present:
        return 1.0, np.nan
    nan_fracs = [df[c].isna().mean() for c in present]
    stds: list[float] = []
    for c in present:
        try:
            val = df[c].std()
        except Exception:
            continue
        try:
            val_f = float(val)
        except (TypeError, ValueError):
            continue
        if pd.isna(val_f):
            continue
        stds.append(val_f)
    nan_mean = float(np.mean(nan_fracs)) if nan_fracs else 1.0
    std_mean = float(np.mean(stds)) if stds else np.nan
    return nan_mean, std_mean


@app.command()
def main(
    data_root: Optional[Path] = typer.Option(
        None, help="Root containing data (defaults to PROJECTIONS_DATA_ROOT or ./data)."
    ),
    start_date: str = typer.Option(..., help="Start date YYYY-MM-DD (inclusive)."),
    end_date: str = typer.Option(..., help="End date YYYY-MM-DD (inclusive)."),
    season_type: str = typer.Option("Regular Season", help="Unused placeholder for logging."),
    output_csv: Path = typer.Option(..., help="Output CSV path for desert summary."),
) -> None:
    root = data_root or data_path()
    start = pd.Timestamp(start_date).date()
    end = pd.Timestamp(end_date).date()
    features_root = root / "gold" / "features_minutes_v1"

    records: list[dict[str, object]] = []
    for day in _iter_days(start, end):
        try:
            features = score_minutes_v1._load_feature_slice(
                day,
                day,
                features_root=features_root,
                features_path=None,
                run_id=None,
            )
        except FileNotFoundError:
            typer.echo(f"[desert] {day}: no features slice; skipping")
            continue
        features["game_date"] = pd.to_datetime(features["game_date"]).dt.date
        day_df = features.loc[features["game_date"] == day].copy()
        if day_df.empty:
            typer.echo(f"[desert] {day}: features slice empty; skipping")
            continue

        rolls_nan_raw, rolls_std = _group_stats(day_df, MINUTES_ROLLS)
        odds_nan_raw, odds_std = _group_stats(day_df, ODDS)
        disp_nan_raw, disp_std = _group_stats(day_df, DISPERSION)

        rolls_nan = 1.0 if pd.isna(rolls_nan_raw) else float(rolls_nan_raw)
        odds_nan = 1.0 if pd.isna(odds_nan_raw) else float(odds_nan_raw)
        disp_nan = 1.0 if pd.isna(disp_nan_raw) else float(disp_nan_raw)

        rolls_bad = (rolls_nan >= 0.5) or (rolls_std is not None and rolls_std < 1.0)
        odds_bad = (odds_nan >= 0.5) or (odds_std is not None and odds_std < 0.5)
        is_desert = bool(rolls_bad and odds_bad)
        is_partial_desert = bool(rolls_bad and not odds_bad)

        typer.echo(
            f"[desert-check] {day} rows={len(day_df)} "
            f"rolls_nan={rolls_nan:.2f} odds_nan={odds_nan:.2f} disp_nan={disp_nan:.2f} "
            f"is_desert={is_desert} is_partial_desert={is_partial_desert}"
        )

        records.append(
            {
                "game_date": day,
                "rows": len(day_df),
                "minutes_rolls_nan_mean": rolls_nan,
                "minutes_rolls_std_mean": rolls_std,
                "odds_nan_mean": odds_nan,
                "odds_std_mean": odds_std,
                "dispersion_nan_mean": disp_nan,
                "dispersion_std_mean": disp_std,
                "is_desert": bool(is_desert),
                "is_partial_desert": bool(is_partial_desert),
            }
        )

    if not records:
        typer.echo("[desert] no records written; nothing to save.")
        raise typer.Exit()

    df_out = pd.DataFrame(records)
    for col in ("is_desert", "is_partial_desert"):
        if col in df_out.columns:
            df_out[col] = df_out[col].fillna(False).astype(bool)
    output_csv = output_csv.expanduser()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_csv, index=False)
    typer.echo(f"[desert] wrote {len(df_out)} rows to {output_csv}")


if __name__ == "__main__":
    app()
