"""
Compute same-team DK FPTS correlations from actual games over a date range.

Usage:
    uv run python -m scripts.sim_v2.eval_team_corr_actual \
        --start-date 2025-11-01 \
        --end-date 2025-12-04 \
        [--actual-root /custom/path] \
        [--max-games 2000]
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional, Sequence

import duckdb
import numpy as np
import pandas as pd
import typer

from projections.paths import data_path

app = typer.Typer(add_completion=False)


def _default_actual_root() -> Path:
    return data_path() / "gold" / "fpts_training_base"


def _load_actual_fpts(start_date: str, end_date: str, actual_root: Optional[Path]) -> pd.DataFrame:
    """Load actual DK FPTS from fpts_training_base."""

    root = (actual_root or _default_actual_root()).expanduser().resolve()
    pattern = str(root / "**" / "fpts_training_base.parquet")
    con = duckdb.connect()
    try:
        df = con.execute(
            f"""
            SELECT
              game_date,
              game_id,
              team_id,
              player_id,
              dk_fpts_actual
            FROM read_parquet('{pattern}', hive_partitioning=true)
            WHERE game_date >= '{start_date}' AND game_date <= '{end_date}'
            """
        ).df()
    finally:
        con.close()
    missing_cols = {"game_date", "game_id", "team_id", "player_id", "dk_fpts_actual"} - set(df.columns)
    if missing_cols:
        raise RuntimeError(f"Missing expected columns in fpts_training_base: {missing_cols}")
    return df


def _compute_corr(
    df: pd.DataFrame,
    max_games: int,
    min_games_per_player: int,
    winsor_frac: float,
    min_shared_games_pair: int,
) -> tuple[list[float], int, int, int]:
    """Compute teammate correlations across game-team samples."""

    df = df.copy()
    df["game_team_key"] = (
        pd.to_datetime(df["game_date"]).dt.date.astype(str)
        + "_"
        + df["game_id"].astype(str)
        + "_"
        + df["team_id"].astype(str)
    )
    unique_keys = df["game_team_key"].unique().tolist()
    if max_games and len(unique_keys) > max_games:
        unique_keys = random.sample(unique_keys, max_games)
    df = df[df["game_team_key"].isin(unique_keys)].copy()

    pivot = df.pivot_table(index="game_team_key", columns="player_id", values="dk_fpts_actual")
    # Drop players with no data across sampled games
    pivot = pivot.dropna(axis=1, how="all")
    # Enforce minimum games per player
    games_per_player = pivot.notna().sum(axis=0)
    eligible_players = games_per_player[games_per_player >= min_games_per_player].index
    pivot = pivot[eligible_players]
    if pivot.shape[1] < 2 or pivot.shape[0] < 2:
        return [], pivot.shape[0], pivot.shape[1]

    corr_mat = pivot.corr().values
    n_players = corr_mat.shape[0]

    presence = pivot.notna().astype(int)
    co_counts_mat = presence.T.dot(presence).values  # shared game-teams per pair

    off_diag_mask = ~np.eye(n_players, dtype=bool)
    shared_counts_off = co_counts_mat[off_diag_mask]
    corr_off = corr_mat[off_diag_mask]

    valid_pairs_mask = shared_counts_off >= min_shared_games_pair
    corr_filtered = corr_off[valid_pairs_mask]
    shared_filtered = shared_counts_off[valid_pairs_mask]

    if corr_filtered.size == 0:
        return [], pivot.shape[0], pivot.shape[1], 0

    arr = np.array(corr_filtered, dtype=float)
    if winsor_frac > 0.0 and 0.0 < winsor_frac < 0.5 and arr.size > 0:
        lower = np.quantile(arr, winsor_frac)
        upper = np.quantile(arr, 1.0 - winsor_frac)
        arr = np.clip(arr, lower, upper)

    return arr.tolist(), pivot.shape[0], pivot.shape[1], int(corr_filtered.size)


def _summarize(vals: Sequence[float]) -> pd.DataFrame:
    arr = np.array(vals, dtype=float)
    stats = {
        "mean_corr": arr.mean(),
        "median": np.median(arr),
        "p10": np.percentile(arr, 10),
        "p25": np.percentile(arr, 25),
        "p75": np.percentile(arr, 75),
        "p90": np.percentile(arr, 90),
    }
    return pd.DataFrame([stats])


@app.command()
def main(
    start_date: str = typer.Option(..., help="Start date (YYYY-MM-DD, inclusive)."),
    end_date: str = typer.Option(..., help="End date (YYYY-MM-DD, inclusive)."),
    actual_root: Optional[Path] = typer.Option(
        None, help="Override path to actual fpts data (defaults to gold/fpts_training_base)."
    ),
    max_games: int = typer.Option(2000, help="Max game-team samples to use (randomly sampled if exceeded)."),
    min_games_per_player: int = typer.Option(
        5,
        help="Minimum number of game-teams a player must appear in to be included in the correlation matrix.",
    ),
    winsor_frac: float = typer.Option(
        0.0,
        help="Optional winsorization fraction to clip extreme correlations (e.g., 0.02 clips 2% on each side).",
    ),
    min_shared_games_pair: int = typer.Option(
        3,
        help="Minimum shared game-teams required for a player pair to be included in correlation summary.",
    ),
) -> None:
    df = _load_actual_fpts(start_date, end_date, actual_root)
    if df.empty:
        typer.echo("[actual_corr] no rows found; exiting.")
        raise typer.Exit(code=0)

    corrs, n_games, n_players, n_pairs = _compute_corr(
        df,
        max_games=max_games,
        min_games_per_player=min_games_per_player,
        winsor_frac=winsor_frac,
        min_shared_games_pair=min_shared_games_pair,
    )
    if not corrs:
        typer.echo("[actual_corr] no correlations computed (not enough players/games); exiting.")
        raise typer.Exit(code=0)

    summary = _summarize(corrs)
    typer.echo(
        f"[actual_corr] date_range={start_date}→{end_date} games_used={n_games} players_used={n_players} "
        f"pairs_used={n_pairs} "
        f"mean_corr={summary.loc[0, 'mean_corr']:.3f} median={summary.loc[0, 'median']:.3f}"
    )
    print("\nSummary:\n", summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    app()
