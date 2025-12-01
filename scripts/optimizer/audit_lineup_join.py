"""Audit that lineup p*_id values join to FPTS gold for a slate.

Usage:
  PROJECTIONS_DATA_ROOT=/home/daniel/projections-data \
  UV_PROJECT_ENVIRONMENT=.venv-user uv run \
    python -m scripts.optimizer.audit_lineup_join \
      --game-date 2025-11-21 \
      --site dk
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import typer

app = typer.Typer(add_completion=False)


def _resolve_data_root() -> Path:
    import os

    root = os.environ.get("PROJECTIONS_DATA_ROOT")
    if root:
        return Path(root).expanduser().resolve()
    return (Path.cwd() / "data").resolve()


def _latest_fpts_path(game_date: str, run_hint: Optional[str], data_root: Path) -> Path:
    base = data_root / "gold" / "projections_fpts_v1" / game_date
    if run_hint:
        candidate = base / f"run={run_hint}" / "fpts.parquet"
        if not candidate.exists():
            raise FileNotFoundError(f"Requested run {run_hint} not found at {candidate}")
        return candidate
    runs = sorted(glob.glob(str(base / "run=*")))
    if not runs:
        raise FileNotFoundError(f"No FPTS runs found under {base}")
    latest = Path(runs[-1]) / "fpts.parquet"
    if not latest.exists():
        raise FileNotFoundError(f"Latest FPTS parquet missing at {latest}")
    return latest


def _collect_slot_ids(df: pd.DataFrame) -> pd.Series:
    slot_cols = [f"p{i}_id" for i in range(1, 9)]
    missing = [c for c in slot_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Lineups CSV missing slot columns: {missing}")
    ids = []
    for col in slot_cols:
        series = pd.to_numeric(df[col], errors="coerce")
        ids.append(series)
    all_ids = pd.concat(ids, ignore_index=True)
    return all_ids.dropna().astype(int)


@app.command()
def main(
    game_date: str = typer.Option(..., help="Game date YYYY-MM-DD."),
    site: str = typer.Option("dk", help="Site (dk/fd)."),
    lineups_csv: Optional[Path] = typer.Option(None, help="Path to lineups CSV."),
    fpts_run: Optional[str] = typer.Option(None, help="Specific FPTS run id (defaults to latest)."),
) -> None:
    data_root = _resolve_data_root()
    default_lineups = (
        Path("artifacts")
        / "lineups"
        / site
        / f"game_date={game_date}"
        / "lineups_pointproj.csv"
    )
    lineup_path = lineups_csv or default_lineups
    if not lineup_path.exists():
        raise FileNotFoundError(f"Lineups CSV not found: {lineup_path}")

    lineups_df = pd.read_csv(lineup_path)
    slot_ids = _collect_slot_ids(lineups_df)
    unique_ids = set(slot_ids.tolist())

    fpts_path = _latest_fpts_path(game_date, fpts_run, data_root)
    fpts_df = pd.read_parquet(fpts_path)
    fpts_ids = set(pd.to_numeric(fpts_df.get("player_id"), errors="coerce").dropna().astype(int).tolist())

    missing_ids = sorted(unique_ids - fpts_ids)
    summary = (
        f"slots={len(slot_ids)} unique_ids={len(unique_ids)} matched_ids={len(unique_ids) - len(missing_ids)}"
    )
    if missing_ids:
        print(f"[audit] WARNING: {summary} missing_ids={missing_ids}")
        sys.exit(1)
    else:
        print(f"[audit] OK: {summary} missing_ids=[]")


if __name__ == "__main__":
    app()
