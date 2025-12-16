"""
Evaluate same-team FPTS correlation in sim_v2 worlds.

Loads simulated worlds, computes teammate FPTS correlation per game/team, and summarizes.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional, Sequence, Tuple

import duckdb
import numpy as np
import pandas as pd
import typer

from projections.paths import data_path

app = typer.Typer(add_completion=False)


def _default_worlds_root() -> Path:
    return data_path() / "artifacts" / "sim_v2" / "worlds_fpts_v2"


def _load_worlds(
    worlds_root: Path, start_date: str, end_date: str, profile_name: Optional[str]
) -> pd.DataFrame:
    """Load worlds between start/end dates (inclusive)."""

    pattern = str(worlds_root / "**" / "*.parquet")
    con = duckdb.connect()
    cols = [
        "game_date",
        "game_id",
        "team_id",
        "player_id",
        "world_id",
        "dk_fpts_world",
    ]
    select_cols = ", ".join(cols)
    query = f"""
        SELECT {select_cols}
        FROM read_parquet('{pattern}', hive_partitioning=true)
        WHERE game_date >= '{start_date}' AND game_date <= '{end_date}'
    """
    df = con.execute(query).df()

    # Optional profile filtering if present
    if profile_name and "profile" in df.columns:
        df = df[df["profile"] == profile_name]
    if profile_name and "profile_name" in df.columns:
        df = df[df["profile_name"] == profile_name]

    return df


def _mean_team_corr(
    df: pd.DataFrame, max_games: Optional[int] = None
) -> Tuple[list[float], Sequence[Tuple[pd.Timestamp, int, int]]]:
    """Compute mean teammate correlation per (date, game_id, team_id)."""

    group_keys = ["game_date", "game_id", "team_id"]
    grouped = list(df.groupby(group_keys))
    if max_games is not None and max_games > 0 and len(grouped) > max_games:
        grouped = random.sample(grouped, max_games)

    corrs: list[float] = []
    meta: list[Tuple[pd.Timestamp, int, int]] = []
    for (g_date, g_id, t_id), grp in grouped:
        if grp["player_id"].nunique() < 2 or grp["world_id"].nunique() < 2:
            continue
        pivot = grp.pivot_table(index="world_id", columns="player_id", values="dk_fpts_world")
        pivot = pivot.dropna(axis=1, how="any")
        if pivot.shape[1] < 2:
            continue
        corr = pivot.corr()
        mask = ~np.eye(corr.shape[0], dtype=bool)
        off_diag = corr.values[mask]
        if off_diag.size == 0:
            continue
        corrs.append(float(off_diag.mean()))
        meta.append((pd.to_datetime(g_date), int(g_id), int(t_id)))
    return corrs, meta


def _summarize(values: Sequence[float]) -> pd.DataFrame:
    arr = np.array(values, dtype=float)
    stats = {
        "mean_corr": arr.mean(),
        "median": np.median(arr),
        "p10": np.percentile(arr, 10),
        "p25": np.percentile(arr, 25),
        "p75": np.percentile(arr, 75),
        "p90": np.percentile(arr, 90),
    }
    df = pd.DataFrame([stats])
    return df


@app.command()
def main(
    start_date: str = typer.Option(..., help="Start date (YYYY-MM-DD, inclusive)."),
    end_date: str = typer.Option(..., help="End date (YYYY-MM-DD, inclusive)."),
    profile_name: Optional[str] = typer.Option(
        None, help="Optional profile name filter if present in worlds."
    ),
    worlds_root: Optional[Path] = typer.Option(
        None, help="Override worlds root; defaults to artifacts/sim_v2/worlds_fpts_v2."
    ),
    max_games: int = typer.Option(500, help="Max (date, game_id, team_id) groups to sample for speed."),
) -> None:
    root = (worlds_root or _default_worlds_root()).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Worlds root not found at {root}")

    typer.echo(
        f"[eval] loading worlds {start_date} to {end_date} root={root} "
        f"profile={profile_name or 'any'} max_groups={max_games}"
    )
    df = _load_worlds(root, start_date, end_date, profile_name)
    if df.empty:
        typer.echo("[eval] no worlds found in range; exiting.")
        raise typer.Exit(code=0)

    corrs, meta = _mean_team_corr(df, max_games=max_games)
    if not corrs:
        typer.echo("[eval] no valid groups with >=2 players/worlds; exiting.")
        raise typer.Exit(code=0)

    summary = _summarize(corrs)
    typer.echo(
        f"[eval] groups={len(corrs)} mean_corr={summary.loc[0, 'mean_corr']:.3f} "
        f"median={summary.loc[0, 'median']:.3f} "
        f"p25={summary.loc[0, 'p25']:.3f} p75={summary.loc[0, 'p75']:.3f}"
    )
    print("\nSummary:\n", summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    app()
