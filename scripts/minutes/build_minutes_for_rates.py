"""
Materialize minutes model predictions for rates training.

Reads existing minutes projections (production bundle outputs) **per game_date** from
``gold/projections_minutes_v1/game_date=YYYY-MM-DD/minutes.parquet`` and writes a thin
table keyed for rates joins:
- season, game_id, game_date, team_id, player_id
- minutes_pred_p10, minutes_pred_p50, minutes_pred_p90, minutes_pred_play_prob

Output (default):
- gold/minutes_for_rates/season=YYYY/game_date=YYYY-MM-DD/minutes_for_rates.parquet

Behavior:
- Loop every date in [start_date, end_date].
- When the per-date projections parquet exists, select the minutes/play_prob columns and
  write the matching ``minutes_for_rates`` partition.
- When the projections parquet is missing, log a warning and continue (do not crash).
- At the end, log how many dates were written vs. skipped due to missing projections.

Example:
    uv run python -m scripts.minutes.build_minutes_for_rates \\
        --start-date 2023-10-01 \\
        --end-date   2025-11-26 \\
        --data-root  /home/daniel/projections-data \\
        --reconcile-mode p50_and_tails \\
        --output-root /home/daniel/projections-data/gold/minutes_for_rates
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
import typer

from projections.minutes_v1.reconcile import (
    ReconcileConfig,
    load_reconcile_config,
    reconcile_minutes_p50_all,
)
from projections.paths import data_path

app = typer.Typer(add_completion=False)

DEFAULT_RECONCILE_CONFIG = Path("config/minutes_l2_reconcile.yaml")
ReconcileMode = Literal["none", "p50", "p50_and_tails"]


def _season_from_day(day: pd.Timestamp) -> int:
    return day.year if day.month >= 8 else day.year - 1


def _iter_days(start: pd.Timestamp, end: pd.Timestamp):
    cur = start
    while cur <= end:
        yield cur
        cur += pd.Timedelta(days=1)


def _source_path(data_root: Path, day: pd.Timestamp) -> Path:
    date_token = day.date().isoformat()
    return data_root / "gold" / "projections_minutes_v1" / f"game_date={date_token}" / "minutes.parquet"


def _load_minutes(data_root: Path, day: pd.Timestamp) -> pd.DataFrame | None:
    path = _source_path(data_root, day)
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.normalize()
    df["season"] = df["game_date"].apply(_season_from_day)
    return df


def _apply_reconciliation(
    df: pd.DataFrame,
    reconcile_mode: ReconcileMode,
    reconcile_config: ReconcileConfig,
) -> pd.DataFrame:
    """Apply L2 reconciliation to minutes predictions per game/team."""
    if reconcile_mode == "none":
        return df

    # reconcile_minutes_p50_all expects columns: game_id, team_id, player_id, minutes_p50, minutes_p10, minutes_p90, play_prob, is_starter
    # It also needs is_starter for rotation detection
    required = ["game_id", "team_id", "player_id", "minutes_p50", "minutes_p10", "minutes_p90"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        typer.echo(f"[reconcile] Warning: missing columns {missing}, skipping reconciliation", err=True)
        return df

    # Ensure play_prob and is_starter exist
    if "play_prob" not in df.columns:
        df["play_prob"] = 1.0
    if "is_starter" not in df.columns:
        # Try to derive from is_confirmed_starter or is_projected_starter
        df["is_starter"] = (
            df.get("is_confirmed_starter", pd.Series(False, index=df.index)).fillna(False).astype(bool)
            | df.get("is_projected_starter", pd.Series(False, index=df.index)).fillna(False).astype(bool)
        ).astype(int)

    result = reconcile_minutes_p50_all(df, config=reconcile_config)
    return result


def _prepare_output(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns for rates and select output columns."""
    df = df.copy()
    df.rename(
        columns={
            "minutes_p10": "minutes_pred_p10",
            "minutes_p50": "minutes_pred_p50",
            "minutes_p90": "minutes_pred_p90",
            "play_prob": "minutes_pred_play_prob",
        },
        inplace=True,
    )
    keep = [
        "season",
        "game_id",
        "game_date",
        "team_id",
        "player_id",
        "minutes_pred_p10",
        "minutes_pred_p50",
        "minutes_pred_p90",
        "minutes_pred_play_prob",
    ]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise KeyError(f"Output missing columns: {missing}")
    return df[keep]


def _write_partition(df: pd.DataFrame, output_root: Path) -> None:
    grouped = df.groupby(["season", "game_date"])
    for (season, game_date), frame in grouped:
        out_dir = output_root / f"season={int(season)}" / f"game_date={pd.Timestamp(game_date).date().isoformat()}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "minutes_for_rates.parquet"
        frame.to_parquet(out_path, index=False)


@app.command()
def main(
    start_date: str = typer.Option(..., help="Start date YYYY-MM-DD (inclusive)"),
    end_date: str = typer.Option(..., help="End date YYYY-MM-DD (inclusive)"),
    data_root: Optional[Path] = typer.Option(None, help="Data root (defaults PROJECTIONS_DATA_ROOT or ./data)"),
    output_root: Optional[Path] = typer.Option(
        None, help="Output root (defaults to <data_root>/gold/minutes_for_rates)"
    ),
    reconcile_mode: ReconcileMode = typer.Option(
        "p50_and_tails",
        "--reconcile-mode",
        help="Reconciliation mode: none, p50 (reconcile median only), or p50_and_tails (also clamp tails).",
    ),
    reconcile_config_path: Path = typer.Option(
        DEFAULT_RECONCILE_CONFIG,
        "--reconcile-config",
        help="Path to L2 reconcile config YAML",
    ),
) -> None:
    root = data_root or data_path()
    out_root = output_root or (root / "gold" / "minutes_for_rates")
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()

    # Load reconcile config if needed
    reconcile_config: ReconcileConfig | None = None
    if reconcile_mode != "none":
        if not reconcile_config_path.exists():
            typer.echo(f"[minutes_for_rates] ERROR: reconcile config not found at {reconcile_config_path}", err=True)
            raise typer.Exit(1)
        base_config = load_reconcile_config(reconcile_config_path)
        clamp_tails = reconcile_mode == "p50_and_tails"
        reconcile_config = replace(base_config, clamp_tails=clamp_tails)
        typer.echo(
            f"[minutes_for_rates] reconciliation enabled: mode={reconcile_mode} clamp_tails={reconcile_config.clamp_tails}"
        )

    typer.echo(
        f"[minutes_for_rates] scoring window {start.date()} to {end.date()} "
        f"from projections_minutes_v1 into {out_root}"
    )
    written_dates = 0
    skipped_missing = 0
    total_rows = 0
    reconcile_failures = 0

    for day in _iter_days(start, end):
        df = _load_minutes(root, day)
        if df is None:
            typer.echo(
                f"[minutes_for_rates] {day.date()}: missing gold/projections_minutes_v1 parquet; skipping.",
                err=True,
            )
            skipped_missing += 1
            continue

        # Apply reconciliation if enabled
        if reconcile_config is not None:
            try:
                df = _apply_reconciliation(df, reconcile_mode, reconcile_config)
            except Exception as e:
                typer.echo(f"[minutes_for_rates] {day.date()}: reconciliation failed: {e}", err=True)
                reconcile_failures += 1
                # Continue with unreconciled data

        # Prepare output columns
        df = _prepare_output(df)

        _write_partition(df, out_root)
        written_dates += 1
        total_rows += len(df)

    if written_dates == 0:
        typer.echo("[minutes_for_rates] no source minutes found; nothing written.")
        return
    typer.echo(
        f"[minutes_for_rates] wrote {total_rows:,} rows across {written_dates} dates into {out_root}"
    )
    summary_parts = [f"written_dates={written_dates}", f"skipped_missing_projections={skipped_missing}"]
    if reconcile_mode != "none":
        summary_parts.append(f"reconcile_mode={reconcile_mode}")
        if reconcile_failures > 0:
            summary_parts.append(f"reconcile_failures={reconcile_failures}")
    typer.echo(f"[minutes_for_rates] summary {' '.join(summary_parts)}")


if __name__ == "__main__":
    app()
