"""Generate FPTS v2 worlds using residual model noise."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import typer

from projections.fpts_v2.features import CATEGORICAL_FEATURES_DEFAULT, build_fpts_design_matrix
from projections.fpts_v2.loader import load_fpts_and_residual
from projections.sim_v2.residuals import sample_residuals_with_team_factor

app = typer.Typer(add_completion=False)


def _parse_date(value: str) -> date:
    try:
        return datetime.fromisoformat(value).date()
    except ValueError as exc:
        raise typer.BadParameter(f"Invalid date: {value}") from exc


def _iter_partitions(root: Path, start: pd.Timestamp, end: pd.Timestamp) -> list[Path]:
    base = root / "gold" / "fpts_training_base"
    partitions: list[Path] = []
    for season_dir in base.glob("season=*"):
        for day_dir in season_dir.glob("game_date=*"):
            try:
                day = pd.Timestamp(day_dir.name.split("=", 1)[1]).normalize()
            except ValueError:
                continue
            if day < start or day > end:
                continue
            candidate = day_dir / "fpts_training_base.parquet"
            if candidate.exists():
                partitions.append(candidate)
    return sorted(partitions)


def _load_base(root: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    paths = _iter_partitions(root, start, end)
    if not paths:
        raise FileNotFoundError("No fpts_training_base partitions found in date range.")
    frames = [pd.read_parquet(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.normalize()
    return df


@app.command()
def main(
    data_root: Path = typer.Option(..., "--data-root"),
    start_date: str = typer.Option(..., "--start-date"),
    end_date: str = typer.Option(..., "--end-date"),
    fpts_run_id: str = typer.Option(..., "--fpts-run-id"),
    n_worlds: int = typer.Option(2000, "--n-worlds"),
    output_root: Optional[Path] = typer.Option(
        None,
        "--output-root",
        help="Defaults to <data_root>/artifacts/sim_v2/worlds_fpts_v2",
    ),
    min_play_prob: float = typer.Option(0.05, "--min-play-prob"),
    seed: int = typer.Option(1234, "--seed"),
    team_factor_sigma: float = typer.Option(
        0.0, "--team-factor-sigma", help="Std dev of team latent factor; 0 disables correlation."
    ),
    team_factor_gamma: float = typer.Option(
        1.0, "--team-factor-gamma", help="Exponent for alpha weights based on dk_fpts_mean."
    ),
) -> None:
    root = data_root
    start_dt = _parse_date(start_date)
    end_dt = _parse_date(end_date)
    start_ts = pd.Timestamp(start_dt).normalize()
    end_ts = pd.Timestamp(end_dt).normalize()

    bundle, residual_model = load_fpts_and_residual(fpts_run_id, data_root=root)
    typer.echo(f"[sim_v2] run_id={fpts_run_id} feature_cols={len(bundle.feature_cols)}")
    typer.echo(f"[sim_v2] team_factor_sigma={team_factor_sigma} alpha_gamma={team_factor_gamma}")

    df = _load_base(root, start_ts, end_ts)
    output_base = output_root or (root / "artifacts" / "sim_v2" / "worlds_fpts_v2")

    for game_date, date_df in df.groupby("game_date"):
        date_df = date_df.copy()
        date_df["minutes_p50"] = pd.to_numeric(date_df.get("minutes_p50"), errors="coerce")
        date_df["is_starter"] = pd.to_numeric(date_df.get("is_starter"), errors="coerce")
        date_df["play_prob"] = pd.to_numeric(date_df.get("play_prob"), errors="coerce")
        date_df = date_df[date_df["play_prob"].fillna(0.0) >= min_play_prob]
        if date_df.empty:
            continue
        date_df = date_df.reset_index(drop=True)

        features = build_fpts_design_matrix(
            date_df,
            bundle.feature_cols,
            categorical_cols=CATEGORICAL_FEATURES_DEFAULT,
            fill_missing_with_zero=True,
        )
        mu = bundle.model.predict(features.values)
        date_df["dk_fpts_mean"] = mu

        typer.echo(
            f"[sim_v2] {game_date.date()} rows={len(date_df)} "
            f"dk_fpts_mean min/med/max={mu.min():.2f}/{np.median(mu):.2f}/{mu.max():.2f}"
        )

        out_dir = output_base / f"game_date={game_date.date()}"
        out_dir.mkdir(parents=True, exist_ok=True)

        for world_id in range(n_worlds):
            rng = np.random.default_rng(seed + world_id)
            eps = sample_residuals_with_team_factor(
                date_df,
                residual_model,
                rng,
                dk_fpts_col="dk_fpts_mean",
                minutes_col="minutes_p50",
                is_starter_col="is_starter",
                game_id_col="game_id",
                team_id_col="team_id",
                team_factor_sigma=team_factor_sigma,
                alpha_gamma=team_factor_gamma,
            )
            dk_fpts_world = date_df["dk_fpts_mean"].to_numpy() + eps
            world_df = date_df[
                [
                    "game_date",
                    "game_id",
                    "team_id",
                    "player_id",
                    "is_starter",
                    "minutes_p50",
                    "dk_fpts_mean",
                ]
                + ([c for c in ["dk_fpts_actual"] if c in date_df.columns])
            ].copy()
            world_df["world_id"] = world_id
            world_df["dk_fpts_world"] = dk_fpts_world
            out_path = out_dir / f"world={world_id:04d}.parquet"
            world_df.to_parquet(out_path, index=False)


if __name__ == "__main__":
    app()
