"""Generate FPTS v2 worlds using residual model noise."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import typer

from projections.fpts_v2.features import CATEGORICAL_FEATURES_DEFAULT, build_fpts_design_matrix
from projections.fpts_v2.loader import load_fpts_and_residual, load_fpts_bundle
from projections.fpts_v2.scoring import compute_dk_fpts
from projections.sim_v2.noise import load_rates_noise_params
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


def _resolve_rate_columns(df: pd.DataFrame, targets: list[str]) -> dict[str, str]:
    """
    Map target names to columns in df (prefers exact match, then pred_<target>).
    """

    mapping: dict[str, str] = {}
    for t in targets:
        if t in df.columns:
            mapping[t] = t
        else:
            pred_col = f"pred_{t}"
            if pred_col in df.columns:
                mapping[t] = pred_col
    return mapping


def _compute_fpts_from_stats(stats: dict[str, np.ndarray]) -> np.ndarray:
    """
    Compute DK FPTS from simulated stat totals.
    Uses simple approximations for makes:
      fgm = fga2 + fga3 (assume makes = attempts)
      fg3m = fga3
      ftm = 0.75 * fta
    """

    n = len(next(iter(stats.values()))) if stats else 0
    zeros = np.zeros(n, dtype=float)

    fga2 = stats.get("fga2", zeros)
    fga3 = stats.get("fga3", zeros)
    fta = stats.get("fta", zeros)
    ast = stats.get("ast", zeros)
    tov = stats.get("tov", zeros)
    oreb = stats.get("oreb", zeros)
    dreb = stats.get("dreb", zeros)
    stl = stats.get("stl", zeros)
    blk = stats.get("blk", zeros)

    reb = oreb + dreb
    ftm = 0.75 * fta
    pts = 2.0 * fga2 + 3.0 * fga3 + ftm

    fgm = fga2 + fga3
    fga = fga2 + fga3
    fg3m = fga3
    fg3a = fga3

    df = pd.DataFrame(
        {
            "pts": pts,
            "fgm": fgm,
            "fga": fga,
            "fg3m": fg3m,
            "fg3a": fg3a,
            "ftm": ftm,
            "fta": fta,
            "reb": reb,
            "oreb": oreb,
            "dreb": dreb,
            "ast": ast,
            "stl": stl,
            "blk": blk,
            "tov": tov,
            "pf": zeros,
            "plus_minus": zeros,
        }
    )
    return compute_dk_fpts(df).to_numpy()


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
    use_rates_noise: bool = typer.Option(
        True,
        "--use-rates-noise/--no-rates-noise",
        help="If True, use rates noise (team+player shocks per stat) instead of FPTS residuals.",
    ),
    rates_noise_split: str = typer.Option(
        "val",
        "--rates-noise-split",
        help="Split key for rates noise params (default: val).",
    ),
    rates_run_id: Optional[str] = typer.Option(
        None,
        "--rates-run-id",
        help="Optional rates run id for noise lookup (defaults to current rates head).",
    ),
) -> None:
    root = data_root
    start_dt = _parse_date(start_date)
    end_dt = _parse_date(end_date)
    start_ts = pd.Timestamp(start_dt).normalize()
    end_ts = pd.Timestamp(end_dt).normalize()

    if use_rates_noise:
        bundle = load_fpts_bundle(fpts_run_id, data_root=root)
        residual_model = None
        typer.echo(f"[sim_v2] run_id={fpts_run_id} feature_cols={len(bundle.feature_cols)}")
        typer.echo(f"[sim_v2] using rates noise; skipping FPTS residual model for run_id={fpts_run_id}")
    else:
        bundle, residual_model = load_fpts_and_residual(fpts_run_id, data_root=root)
        typer.echo(f"[sim_v2] run_id={fpts_run_id} feature_cols={len(bundle.feature_cols)}")
        typer.echo(f"[sim_v2] team_factor_sigma={team_factor_sigma} alpha_gamma={team_factor_gamma}")

    noise_params = None
    noise_path = None
    stat_targets: list[str] = []
    if use_rates_noise:
        noise_params, noise_path = load_rates_noise_params(
            data_root=root, run_id=rates_run_id, split=rates_noise_split
        )
        stat_targets = list(noise_params.keys())
        typer.echo(
            f"[sim_v2] using rates noise run_id={rates_run_id or 'current'} split={rates_noise_split} "
            f"targets={len(stat_targets)} path={noise_path}"
        )

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

        out_dir = output_base / f"game_date={game_date.date()}"
        out_dir.mkdir(parents=True, exist_ok=True)

        if use_rates_noise and noise_params is not None:
            mapping = _resolve_rate_columns(date_df, stat_targets)
            missing_targets = [t for t in stat_targets if t not in mapping]
            if missing_targets:
                typer.echo(f"[sim_v2] warning: missing rate columns for targets={missing_targets}; skipping.")
            if not mapping:
                continue

            mu_stats: dict[str, np.ndarray] = {}
            for target, col in mapping.items():
                rates = pd.to_numeric(date_df[col], errors="coerce").to_numpy()
                mins = date_df["minutes_p50"].to_numpy()
                mu_stats[target] = np.clip(rates * mins, 0.0, None)

            base_mu: dict[str, np.ndarray] = {}
            for target, vals in mu_stats.items():
                base = target.replace("_per_min", "")
                base_mu[base] = vals

            dk_fpts_mean = _compute_fpts_from_stats(base_mu)
            date_df["dk_fpts_mean"] = dk_fpts_mean

            typer.echo(
                f"[sim_v2] {game_date.date()} rows={len(date_df)} "
                f"dk_fpts_mean (rates) min/med/max={dk_fpts_mean.min():.2f}/{np.median(dk_fpts_mean):.2f}/{dk_fpts_mean.max():.2f}"
            )

            # Precompute team grouping once per date.
            game_ids = date_df["game_id"].to_numpy()
            team_ids = date_df["team_id"].to_numpy()
            group_map: dict[tuple[int, int], np.ndarray] = {}
            for idx, key in enumerate(zip(game_ids, team_ids)):
                group_map.setdefault(key, []).append(idx)
            group_map = {k: np.array(v, dtype=int) for k, v in group_map.items()}

            for world_id in range(n_worlds):
                rng = np.random.default_rng(seed + world_id)
                stat_totals: dict[str, np.ndarray] = {}
                for target, col in mapping.items():
                    params = noise_params.get(target, {})
                    sigma_team = float(params.get("sigma_team", 0.0) or 0.0)
                    sigma_player = float(params.get("sigma_player", 0.0) or 0.0)

                    mu = mu_stats[target]
                    team_shock = np.zeros_like(mu)
                    if sigma_team > 0:
                        for key, idxs in group_map.items():
                            ts = rng.normal(loc=0.0, scale=sigma_team)
                            team_shock[idxs] = ts
                    player_eps = rng.normal(loc=0.0, scale=sigma_player, size=len(mu))
                    total = np.clip(mu + team_shock + player_eps, 0.0, None)
                    base = target.replace("_per_min", "")
                    stat_totals[base] = total

                dk_fpts_world = _compute_fpts_from_stats(stat_totals)
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

                for base, values in stat_totals.items():
                    world_df[f"{base}_sim"] = values
                # Derived stats
                if "oreb" in stat_totals and "dreb" in stat_totals:
                    world_df["reb_sim"] = stat_totals["oreb"] + stat_totals["dreb"]

                out_path = out_dir / f"world={world_id:04d}.parquet"
                world_df.to_parquet(out_path, index=False)
        else:
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
