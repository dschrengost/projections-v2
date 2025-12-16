"""Debug join between minutes_v1 and rates_v1_live for sim_v2."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
import typer

from projections.paths import data_path
from scripts.sim_v2.generate_worlds_fpts_v2 import build_rates_mean_fpts

app = typer.Typer(add_completion=False, help="Diagnose minutes_v1 x rates_v1_live join issues.")

JOIN_KEYS = ["game_date", "game_id", "team_id", "player_id"]


def _parse_date(value: str) -> date:
    try:
        return datetime.fromisoformat(value).date()
    except ValueError as exc:
        raise typer.BadParameter(f"Invalid date: {value}") from exc


def _load_minutes(root: Path, game_date: date, run_id: str) -> pd.DataFrame:
    path = root / "daily" / game_date.isoformat() / f"run={run_id}" / "minutes.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Minutes parquet not found at {path}")
    return pd.read_parquet(path)


def _load_rates(root: Path, game_date: date, run_id: str) -> pd.DataFrame:
    base = root / game_date.isoformat()
    candidate = base / f"run={run_id}" / "rates.parquet"
    if not candidate.exists():
        raise FileNotFoundError(f"Rates parquet not found at {candidate}")
    return pd.read_parquet(candidate)


def _print_shapes_and_columns(minutes_df: pd.DataFrame, rates_df: pd.DataFrame) -> None:
    typer.echo(f"Minutes shape: {minutes_df.shape}")
    typer.echo(f"Rates shape:   {rates_df.shape}")
    typer.echo(f"Minutes columns ({len(minutes_df.columns)}): {sorted(minutes_df.columns.tolist())}")
    typer.echo(f"Rates columns   ({len(rates_df.columns)}): {sorted(rates_df.columns.tolist())}")


def _print_key_dtypes(minutes_df: pd.DataFrame, rates_df: pd.DataFrame) -> None:
    minutes_missing = [k for k in JOIN_KEYS if k not in minutes_df.columns]
    rates_missing = [k for k in JOIN_KEYS if k not in rates_df.columns]

    typer.echo("Join key dtypes (minutes):")
    if minutes_missing:
        typer.echo(f"  missing columns: {minutes_missing}")
    else:
        typer.echo(minutes_df[JOIN_KEYS].dtypes)

    typer.echo("Join key dtypes (rates):")
    if rates_missing:
        typer.echo(f"  missing columns: {rates_missing}")
    else:
        typer.echo(rates_df[JOIN_KEYS].dtypes)


def _diagnose_join(minutes_df: pd.DataFrame, rates_df: pd.DataFrame, game_date: date) -> None:
    minutes_missing = [k for k in JOIN_KEYS if k not in minutes_df.columns]
    rates_missing = [k for k in JOIN_KEYS if k not in rates_df.columns]

    minutes_merge = minutes_df.copy()
    rates_merge = rates_df.copy()

    if minutes_missing or rates_missing:
        typer.echo(
            f"Adding placeholder columns for diagnostic merge "
            f"(minutes_missing={minutes_missing}, rates_missing={rates_missing})"
        )
    for key in minutes_missing:
        minutes_merge[key] = pd.NA
    for key in rates_missing:
        if key == "game_date":
            rates_merge[key] = pd.Timestamp(game_date)
        else:
            rates_merge[key] = pd.NA

    if "game_date" in minutes_merge.columns:
        minutes_merge["game_date"] = pd.to_datetime(minutes_merge["game_date"]).dt.normalize()
    if "game_date" in rates_merge.columns:
        rates_merge["game_date"] = pd.to_datetime(rates_merge["game_date"]).dt.normalize()

    merged = minutes_merge.merge(
        rates_merge, on=JOIN_KEYS, how="outer", indicator=True, suffixes=("_min", "_rate")
    )
    counts = merged["_merge"].value_counts(dropna=False).to_dict()
    typer.echo(f"Join indicator counts: {counts}")

    inner = merged[merged["_merge"] == "both"].copy()
    typer.echo(f"Inner-join rows: {inner.shape[0]}")

    def _sample(label: str) -> pd.DataFrame:
        return merged.loc[merged["_merge"] == label, JOIN_KEYS].head(10)

    typer.echo("Sample left_only (in minutes, missing in rates):")
    typer.echo(_sample("left_only"))

    typer.echo("Sample right_only (in rates, missing in minutes):")
    typer.echo(_sample("right_only"))

    typer.echo("Sample both (matching rows):")
    typer.echo(inner[JOIN_KEYS].head(10))


@app.command()
def main(
    date: str = typer.Option(..., "--date", help="Game date (YYYY-MM-DD)."),
    minutes_run_id: str = typer.Option(..., "--minutes-run-id", help="Run id for minutes_v1 daily."),
    rates_run_id: str = typer.Option(..., "--rates-run-id", help="Run id for rates_v1_live."),
    minutes_root: Path | None = typer.Option(
        None, "--minutes-root", help="Root for minutes_v1 (default: <data_root>/artifacts/minutes_v1)."
    ),
    rates_root: Path | None = typer.Option(
        None, "--rates-root", help="Root for rates_v1_live (default: <data_root>/gold/rates_v1_live)."
    ),
) -> None:
    game_date = _parse_date(date)
    minutes_root = minutes_root or data_path("artifacts", "minutes_v1")
    rates_root = rates_root or data_path("gold", "rates_v1_live")

    minutes_df = _load_minutes(Path(minutes_root), game_date, minutes_run_id)
    rates_df = _load_rates(Path(rates_root), game_date, rates_run_id)

    _print_shapes_and_columns(minutes_df, rates_df)
    _print_key_dtypes(minutes_df, rates_df)
    _diagnose_join(minutes_df, rates_df, game_date)

    try:
        mu_df = build_rates_mean_fpts(minutes_df, rates_df)
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"build_rates_mean_fpts raised: {exc}")
        return

    typer.echo(f"mu_df rows: {mu_df.shape[0]}")
    typer.echo(mu_df[JOIN_KEYS].head(10))
    if not mu_df.empty and "dk_fpts_mean" in mu_df.columns:
        typer.echo(mu_df["dk_fpts_mean"].describe())


if __name__ == "__main__":
    app()
