"""
Usage Share Prep - Diagnostics for opportunity-share labels (FGA/FTA/TOV).

This script is intentionally lightweight and read-only: it inspects the existing
gold `rates_training_base` dataset and derives player-game opportunity totals
and within-team shares:

  - fga = (fga2_per_min + fga3_per_min) * minutes_actual
  - fta = fta_per_min * minutes_actual
  - tov = tov_per_min * minutes_actual
  - share_x = x_i / sum_team(x)

Usage:
  uv run python -m scripts.diagnostics.usage_share_prep --start-date 2025-11-01 --end-date 2025-11-03
  uv run python -m scripts.diagnostics.usage_share_prep --start-date 2025-11-03 --end-date 2025-11-03 --max-days 1
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import typer
from rich.console import Console

from projections.paths import data_path

app = typer.Typer(add_completion=False, help=__doc__)
console = Console()


def _parse_date(value: str) -> date:
    try:
        return datetime.fromisoformat(value).date()
    except ValueError as exc:
        raise typer.BadParameter(f"Invalid date: {value}") from exc


def _iter_rates_training_partitions(
    base: Path, start: pd.Timestamp, end: pd.Timestamp, *, max_days: int | None
) -> list[Path]:
    root = base / "gold" / "rates_training_base"
    partitions: list[Path] = []
    days_seen = 0
    for season_dir in sorted(root.glob("season=*")):
        for day_dir in sorted(season_dir.glob("game_date=*")):
            try:
                day = pd.Timestamp(day_dir.name.split("=", 1)[1]).normalize()
            except ValueError:
                continue
            if day < start or day > end:
                continue
            parquet_path = day_dir / "rates_training_base.parquet"
            if parquet_path.exists():
                partitions.append(parquet_path)
                days_seen += 1
                if max_days is not None and days_seen >= max_days:
                    return partitions
    return partitions


def _safe_divide(numer: pd.Series, denom: pd.Series) -> pd.Series:
    denom_safe = denom.replace(0, np.nan)
    out = numer / denom_safe
    return out.fillna(0.0)


@dataclass(frozen=True)
class SharePrepResult:
    df: pd.DataFrame
    team_totals: pd.DataFrame


def build_share_labels_from_rates_training_base(df: pd.DataFrame) -> SharePrepResult:
    required = {
        "game_id",
        "team_id",
        "player_id",
        "minutes_actual",
        "fga2_per_min",
        "fga3_per_min",
        "fta_per_min",
        "tov_per_min",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"rates_training_base missing required columns: {', '.join(sorted(missing))}")

    working = df.copy()
    for col in ("minutes_actual", "fga2_per_min", "fga3_per_min", "fta_per_min", "tov_per_min"):
        working[col] = pd.to_numeric(working[col], errors="coerce").fillna(0.0)

    working["fga2"] = (working["fga2_per_min"] * working["minutes_actual"]).clip(lower=0.0)
    working["fga3"] = (working["fga3_per_min"] * working["minutes_actual"]).clip(lower=0.0)
    working["fga"] = (working["fga2"] + working["fga3"]).clip(lower=0.0)
    working["fta"] = (working["fta_per_min"] * working["minutes_actual"]).clip(lower=0.0)
    working["tov"] = (working["tov_per_min"] * working["minutes_actual"]).clip(lower=0.0)

    keys = ["game_id", "team_id"]
    team_totals = (
        working.groupby(keys, as_index=False)[["fga", "fta", "tov", "minutes_actual"]]
        .sum()
        .rename(columns={"minutes_actual": "team_minutes"})
    )
    working = working.merge(team_totals, on=keys, how="left", suffixes=("", "_team"))
    working["fga_share"] = _safe_divide(working["fga"], working["fga_team"])
    working["fta_share"] = _safe_divide(working["fta"], working["fta_team"])
    working["tov_share"] = _safe_divide(working["tov"], working["tov_team"])

    return SharePrepResult(df=working, team_totals=team_totals)


def _pct_non_null(df: pd.DataFrame, cols: Iterable[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for c in cols:
        if c not in df.columns:
            out[c] = float("nan")
            continue
        out[c] = float(100.0 * (1.0 - df[c].isna().mean())) if len(df) else float("nan")
    return out


@app.command()
def main(
    start_date: str = typer.Option(..., "--start-date"),
    end_date: str = typer.Option(..., "--end-date"),
    max_days: int | None = typer.Option(
        None,
        "--max-days",
        help="Optional cap on the number of partitions loaded (useful for quick sanity checks).",
    ),
    data_root: Path | None = typer.Option(
        None,
        "--data-root",
        help="Override PROJECTIONS_DATA_ROOT (defaults to env var or ./data).",
    ),
) -> None:
    start = pd.Timestamp(_parse_date(start_date)).normalize()
    end = pd.Timestamp(_parse_date(end_date)).normalize()
    if end < start:
        raise typer.BadParameter("--end-date must be >= --start-date")

    root = data_root or data_path()
    paths = _iter_rates_training_partitions(root, start, end, max_days=max_days)
    if not paths:
        raise FileNotFoundError(f"No rates_training_base partitions found for {start.date()}..{end.date()} under {root}")

    console.print(f"usage_shares: loading {len(paths)} partitions from {root}/gold/rates_training_base")
    frames = [pd.read_parquet(p) for p in paths]
    base = pd.concat(frames, ignore_index=True)
    console.print(f"usage_shares: rows={len(base):,} cols={len(base.columns)}")

    built = build_share_labels_from_rates_training_base(base)
    df = built.df

    # Team totals sanity
    team = built.team_totals
    for col in ("fga", "fta", "tov", "team_minutes"):
        if col not in team.columns:
            continue
        s = team[col]
        q = s.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).to_dict()
        console.print(
            f"usage_shares: team {col}: n={len(s):,} "
            f"p05={q.get(0.05, float('nan')):.2f} p50={q.get(0.5, float('nan')):.2f} p95={q.get(0.95, float('nan')):.2f}"
        )

    # Share sum-to-1 checks
    sums = (
        df.groupby(["game_id", "team_id"], as_index=False)[["fga_share", "fta_share", "tov_share"]]
        .sum()
        .rename(columns={"fga_share": "sum_fga_share", "fta_share": "sum_fta_share", "tov_share": "sum_tov_share"})
    )
    for col in ("sum_fga_share", "sum_fta_share", "sum_tov_share"):
        s = sums[col]
        max_abs_err = float((s - 1.0).abs().max()) if len(s) else float("nan")
        console.print(f"usage_shares: {col}: max_abs_err_vs_1={max_abs_err:.6f}")

    # Feature availability quick glance
    feature_cols = [
        "minutes_pred_p50",
        "minutes_pred_play_prob",
        "is_starter",
        "position_primary",
        "team_itt",
        "opp_itt",
        "spread_close",
        "total_close",
        "vac_min_szn",
        "vac_fga_szn",
        "vac_ast_szn",
        "track_role_cluster",
    ]
    avail = _pct_non_null(df, feature_cols)
    console.print("usage_shares: feature availability (% non-null):")
    for k, v in avail.items():
        console.print(f"  - {k}: {v:.1f}%")


if __name__ == "__main__":
    app()
