"""
Validate rates_training_base coverage and leakage over a date window.

Checks that minutes_pred_* are present, flags any rows that would fall back to
minutes_actual, and reports basic NULL counts for tracking/context features.

Example:
    uv run python -m scripts.rates.check_rates_training_base_coverage \\
        --start-date 2023-10-24 \\
        --end-date   2025-12-01
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from projections.paths import data_path

app = typer.Typer(add_completion=False)


def _iter_partitions(root: Path, start: pd.Timestamp, end: pd.Timestamp) -> list[Path]:
    base = root / "gold" / "rates_training_base"
    paths: list[Path] = []
    for season_dir in sorted(base.glob("season=*")):
        for day_dir in sorted(season_dir.glob("game_date=*")):
            try:
                day = pd.Timestamp(day_dir.name.split("=", 1)[1]).normalize()
            except ValueError:
                continue
            if start <= day <= end:
                candidate = day_dir / "rates_training_base.parquet"
                if candidate.exists():
                    paths.append(candidate)
    return paths


@app.command()
def main(
    start_date: str = typer.Option(..., help="Start date YYYY-MM-DD (inclusive)"),
    end_date: str = typer.Option(..., help="End date YYYY-MM-DD (inclusive)"),
    data_root: Optional[Path] = typer.Option(None, help="Data root (defaults to PROJECTIONS_DATA_ROOT or ./data)"),
) -> None:
    root = data_root or data_path()
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()

    paths = _iter_partitions(root, start, end)
    if not paths:
        typer.echo("[coverage] no rates_training_base partitions found in window.")
        raise typer.Exit(code=1)

    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.normalize()
    total_rows = len(df)

    pred_cols = ["minutes_pred_p10", "minutes_pred_p50", "minutes_pred_p90", "minutes_pred_play_prob"]
    na_pred = {c: int(df[c].isna().sum()) for c in pred_cols if c in df.columns}
    leaky_rows = 0
    if {"minutes_pred_p50", "minutes_actual"} <= set(df.columns):
        leaky_mask = df["minutes_pred_p50"].isna() & df["minutes_actual"].notna()
        leaky_rows = int(leaky_mask.sum())

    typer.echo(f"[coverage] rows={total_rows:,} dates={len(paths)} window={start.date()}..{end.date()}")
    for col, count in na_pred.items():
        typer.echo(f"[coverage] {col} nulls: {count}")
    typer.echo(f"[coverage] leaky rows (minutes_pred missing, minutes_actual present): {leaky_rows}")

    tracking_cols = [
        "track_touches_per_min_szn",
        "track_sec_per_touch_szn",
        "track_pot_ast_per_min_szn",
        "track_drives_per_min_szn",
        "track_role_cluster",
        "track_role_is_low_minutes",
    ]
    context_cols = [
        "vac_min_szn",
        "vac_fga_szn",
        "vac_ast_szn",
        "team_pace_szn",
        "opp_pace_szn",
        "team_off_rtg_szn",
        "team_def_rtg_szn",
        "opp_def_rtg_szn",
    ]

    for group_name, cols in (("tracking", tracking_cols), ("context", context_cols)):
        present = [c for c in cols if c in df.columns]
        if not present:
            typer.echo(f"[coverage] {group_name}: no columns present")
            continue
        null_counts = {c: int(df[c].isna().sum()) for c in present}
        typer.echo(f"[coverage] {group_name} nulls: " + ", ".join(f"{c}={n}" for c, n in null_counts.items()))

    if na_pred.get("minutes_pred_p50", 0) > 0 or leaky_rows > 0:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
