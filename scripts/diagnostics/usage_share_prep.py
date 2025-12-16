"""
Usage Share Prep - Diagnostics for opportunity-share labels (FGA/FTA/TOV).

This script is intentionally lightweight and read-only: it inspects the existing
gold `rates_training_base` or `usage_shares_training_base` dataset and derives
player-game opportunity totals and within-team shares:

  - fga = (fga2_per_min + fga3_per_min) * minutes_actual
  - fta = fta_per_min * minutes_actual
  - tov = tov_per_min * minutes_actual
  - share_x = x_i / sum_team(x)

Diagnostics printed:
  - Share sums per team-game (max deviation from 1)
  - Missingness per feature
  - Distribution of shares (top-1 share, Herfindahl H = Σ share_i²)
  - Counts dropped due to as_of_ts / tip_ts violations (if usage_shares_training_base)

Usage:
  uv run python -m scripts.diagnostics.usage_share_prep --start-date 2025-11-01 --end-date 2025-11-03
  uv run python -m scripts.diagnostics.usage_share_prep --start-date 2025-11-03 --end-date 2025-11-03 --max-days 1
  uv run python -m scripts.diagnostics.usage_share_prep --start-date 2025-11-01 --end-date 2025-11-03 --source usage_shares
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


def _iter_training_partitions(
    base: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    max_days: int | None,
    source: str = "rates",
) -> list[Path]:
    """
    Iterate over training base partitions.

    Args:
        source: "rates" for rates_training_base, "usage_shares" for usage_shares_training_base
    """
    if source == "usage_shares":
        root = base / "gold" / "usage_shares_training_base"
        filename = "usage_shares_training_base.parquet"
    else:
        root = base / "gold" / "rates_training_base"
        filename = "rates_training_base.parquet"

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
            parquet_path = day_dir / filename
            if parquet_path.exists():
                partitions.append(parquet_path)
                days_seen += 1
                if max_days is not None and days_seen >= max_days:
                    return partitions
    return partitions


def _iter_rates_training_partitions(
    base: Path, start: pd.Timestamp, end: pd.Timestamp, *, max_days: int | None
) -> list[Path]:
    """Backward-compatible wrapper."""
    return _iter_training_partitions(base, start, end, max_days=max_days, source="rates")


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


def _compute_herfindahl(df: pd.DataFrame, share_col: str) -> pd.Series:
    """Compute Herfindahl index (H = Σ share_i²) per team-game."""
    df_copy = df.copy()
    df_copy["_share_sq"] = df_copy[share_col] ** 2
    return df_copy.groupby(["game_id", "team_id"])["_share_sq"].sum()


def _compute_top1_share(df: pd.DataFrame, share_col: str) -> pd.Series:
    """Compute max share per team-game."""
    return df.groupby(["game_id", "team_id"])[share_col].max()


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
    source: str = typer.Option(
        "rates",
        "--source",
        help="Data source: 'rates' for rates_training_base, 'usage_shares' for usage_shares_training_base.",
    ),
) -> None:
    start = pd.Timestamp(_parse_date(start_date)).normalize()
    end = pd.Timestamp(_parse_date(end_date)).normalize()
    if end < start:
        raise typer.BadParameter("--end-date must be >= --start-date")

    root = data_root or data_path()
    paths = _iter_training_partitions(root, start, end, max_days=max_days, source=source)
    if not paths:
        source_name = "usage_shares_training_base" if source == "usage_shares" else "rates_training_base"
        raise FileNotFoundError(f"No {source_name} partitions found for {start.date()}..{end.date()} under {root}")

    source_name = "usage_shares_training_base" if source == "usage_shares" else "rates_training_base"
    console.print(f"usage_shares: loading {len(paths)} partitions from {root}/gold/{source_name}")
    frames = [pd.read_parquet(p) for p in paths]
    base = pd.concat(frames, ignore_index=True)
    console.print(f"usage_shares: rows={len(base):,} cols={len(base.columns)}")

    # For usage_shares_training_base, shares are already computed
    if source == "usage_shares":
        df = base.copy()
        # Rename to expected column names for diagnostics
        share_cols_map = {
            "share_fga": "fga_share",
            "share_fta": "fta_share",
            "share_tov": "tov_share",
        }
        for old, new in share_cols_map.items():
            if old in df.columns and new not in df.columns:
                df[new] = df[old]

        # Check leak_safe / tip_ts violations
        console.print("")
        console.print("[bold]== Anti-leak diagnostics ==[/bold]")
        if "leak_safe" in df.columns:
            n_leak_safe = df["leak_safe"].sum()
            n_total = len(df)
            n_violations = n_total - n_leak_safe
            console.print(f"usage_shares: leak_safe rows: {n_leak_safe:,}/{n_total:,} ({100*n_leak_safe/n_total:.1f}%)")
            console.print(f"usage_shares: rows missing tip_ts (potential leak): {n_violations:,}")
        else:
            console.print("usage_shares: leak_safe column not present")

        if "tip_ts_present" in df.columns:
            n_with_tip = df["tip_ts_present"].sum()
            console.print(f"usage_shares: rows with tip_ts: {n_with_tip:,}/{len(df):,}")

        # Team totals from precomputed columns
        team = df.groupby(["game_id", "team_id"], as_index=False).agg(
            fga=("fga", "sum") if "fga" in df.columns else ("share_fga", "count"),
            fta=("fta", "sum") if "fta" in df.columns else ("share_fta", "count"),
            tov=("tov", "sum") if "tov" in df.columns else ("share_tov", "count"),
            team_minutes=("minutes_actual", "sum") if "minutes_actual" in df.columns else ("share_fga", "count"),
        )
    else:
        built = build_share_labels_from_rates_training_base(base)
        df = built.df
        team = built.team_totals

    console.print("")
    console.print("[bold]== Team totals sanity ==[/bold]")
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
    console.print("")
    console.print("[bold]== Share sum-to-1 checks ==[/bold]")
    share_col_names = ["fga_share", "fta_share", "tov_share"]
    existing_share_cols = [c for c in share_col_names if c in df.columns]
    if existing_share_cols:
        sums = (
            df.groupby(["game_id", "team_id"], as_index=False)[existing_share_cols]
            .sum()
        )
        for col in existing_share_cols:
            s = sums[col]
            max_abs_err = float((s - 1.0).abs().max()) if len(s) else float("nan")
            console.print(f"usage_shares: sum_{col}: max_abs_err_vs_1={max_abs_err:.6f}")

    # Herfindahl concentration
    console.print("")
    console.print("[bold]== Herfindahl concentration (H = Σ share_i²) ==[/bold]")
    for share_col in existing_share_cols:
        h = _compute_herfindahl(df, share_col)
        console.print(
            f"usage_shares: {share_col} H: mean={h.mean():.4f} "
            f"p10={h.quantile(0.1):.4f} p50={h.quantile(0.5):.4f} p90={h.quantile(0.9):.4f}"
        )
    console.print("  (Typical NBA game H ~ 0.10-0.20; lower = more dispersed)")

    # Top-1 share
    console.print("")
    console.print("[bold]== Top-1 share per team-game ==[/bold]")
    for share_col in existing_share_cols:
        top1 = _compute_top1_share(df, share_col)
        console.print(
            f"usage_shares: {share_col} top-1: mean={top1.mean():.4f} "
            f"p10={top1.quantile(0.1):.4f} p50={top1.quantile(0.5):.4f} p90={top1.quantile(0.9):.4f}"
        )

    # Feature availability quick glance
    console.print("")
    console.print("[bold]== Feature availability (% non-null) ==[/bold]")
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
        "tip_ts",
        "leak_safe",
    ]
    avail = _pct_non_null(df, feature_cols)
    for k, v in avail.items():
        console.print(f"  - {k}: {v:.1f}%")


if __name__ == "__main__":
    app()
