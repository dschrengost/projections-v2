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
from projections.paths import data_path
from projections.sim_v2.config import DEFAULT_PROFILES_PATH, SimV2Profile, load_sim_v2_profile
from projections.sim_v2.minutes_noise import (
    build_sigma_per_player,
    load_minutes_noise_params,
    status_bucket_from_raw,
)
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
    Accepts 1D arrays (players) or 2D arrays (worlds, players); output mirrors input shape.
    """

    if not stats:
        return np.array([])
    sample = next(iter(stats.values()))
    zeros = np.zeros_like(sample)

    def _prep(name: str) -> np.ndarray:
        return stats.get(name, zeros)

    fga2 = _prep("fga2")
    fga3 = _prep("fga3")
    fta = _prep("fta")
    ast = _prep("ast")
    tov = _prep("tov")
    oreb = _prep("oreb")
    dreb = _prep("dreb")
    stl = _prep("stl")
    blk = _prep("blk")

    reb = oreb + dreb
    ftm = 0.75 * fta
    pts = 2.0 * fga2 + 3.0 * fga3 + ftm

    fgm = fga2 + fga3
    fga = fga2 + fga3
    fg3m = fga3
    fg3a = fga3

    shaped_like = sample.shape
    flat = lambda arr: arr.reshape(-1)
    df = pd.DataFrame(
        {
            "pts": flat(pts),
            "fgm": flat(fgm),
            "fga": flat(fga),
            "fg3m": flat(fg3m),
            "fg3a": flat(fg3a),
            "ftm": flat(ftm),
            "fta": flat(fta),
            "reb": flat(reb),
            "oreb": flat(oreb),
            "dreb": flat(dreb),
            "ast": flat(ast),
            "stl": flat(stl),
            "blk": flat(blk),
            "tov": flat(tov),
            "pf": flat(np.zeros_like(pts)),
            "plus_minus": flat(np.zeros_like(pts)),
        }
    )
    fpts_flat = compute_dk_fpts(df).to_numpy()
    return fpts_flat.reshape(shaped_like)


def _resolve_minutes_column(df: pd.DataFrame) -> str:
    for candidate in ("minutes_p50_cond", "minutes_p50", "minutes_pred_p50"):
        if candidate in df.columns:
            return candidate
    raise KeyError("Missing minutes column (expected minutes_p50_cond/minutes_p50/minutes_pred_p50)")


def _ensure_status_bucket(df: pd.DataFrame) -> pd.DataFrame:
    if "status_bucket" in df.columns:
        df["status_bucket"] = df["status_bucket"].apply(status_bucket_from_raw)
        return df
    for col in ("status", "injury_status", "availability_status"):
        if col in df.columns:
            df["status_bucket"] = df[col].apply(status_bucket_from_raw)
            return df
    df["status_bucket"] = "healthy"
    return df


@app.command()
def main(
    start_date: str = typer.Option(..., "--start-date"),
    end_date: str = typer.Option(..., "--end-date"),
    n_worlds: int = typer.Option(2000, "--n-worlds"),
    profile: str = typer.Option("baseline", "--profile", help="Name of sim_v2 profile to load."),
    data_root: Optional[Path] = typer.Option(None, "--data-root", help="Data root (default: PROJECTIONS_DATA_ROOT/./data)."),
    profiles_path: Optional[Path] = typer.Option(None, "--profiles-path", help="Override path to sim_v2 profiles JSON."),
    output_root: Optional[Path] = typer.Option(
        None,
        "--output-root",
        help="Defaults to <data_root>/artifacts/sim_v2/worlds_fpts_v2",
    ),
    fpts_run_id: Optional[str] = typer.Option(None, "--fpts-run-id", help="Override FPTS run id (otherwise profile)."),
    use_rates_noise: Optional[bool] = typer.Option(
        None,
        "--use-rates-noise/--no-rates-noise",
        help="Override rates noise toggle (otherwise profile).",
    ),
    rates_noise_split: Optional[str] = typer.Option(
        None,
        "--rates-noise-split",
        help="Override rates noise split (otherwise profile).",
    ),
    rates_run_id: Optional[str] = typer.Option(
        None,
        "--rates-run-id",
        help="Override rates run id for noise lookup (otherwise profile).",
    ),
    use_minutes_noise: Optional[bool] = typer.Option(
        None,
        "--use-minutes-noise/--no-minutes-noise",
        help="Override minutes noise toggle (otherwise profile).",
    ),
    minutes_noise_run_id: Optional[str] = typer.Option(
        None,
        "--minutes-noise-run-id",
        help="Override minutes run id for noise lookup (otherwise profile).",
    ),
    minutes_sigma_min: Optional[float] = typer.Option(
        None,
        "--minutes-sigma-min",
        help="Optional override to floor per-bucket sigmas when sampling minutes (otherwise profile).",
    ),
    seed: Optional[int] = typer.Option(None, "--seed", help="Override RNG seed (otherwise profile)."),
    min_play_prob: Optional[float] = typer.Option(None, "--min-play-prob", help="Override minimum play_prob filter."),
    team_factor_sigma: Optional[float] = typer.Option(
        None, "--team-factor-sigma", help="Override team latent factor sigma for residual model path."
    ),
    team_factor_gamma: Optional[float] = typer.Option(
        None, "--team-factor-gamma", help="Override alpha exponent for residual model path."
    ),
) -> None:
    profile_cfg = load_sim_v2_profile(profile=profile, profiles_path=profiles_path)

    def _resolve(value, override, label):
        if override is not None and override != value:
            typer.echo(f"[sim_v2] override {label}: profile={value} -> cli={override}")
            return override
        return value

    fpts_run = _resolve(profile_cfg.fpts_run_id, fpts_run_id, "fpts_run_id")
    use_rates_noise_eff = profile_cfg.use_rates_noise if use_rates_noise is None else use_rates_noise
    rates_run = _resolve(profile_cfg.rates_run_id, rates_run_id, "rates_run_id") if use_rates_noise_eff else None
    rates_split = _resolve(profile_cfg.rates_noise_split, rates_noise_split, "rates_noise_split") if use_rates_noise_eff else None
    use_minutes_noise_eff = profile_cfg.use_minutes_noise if use_minutes_noise is None else use_minutes_noise
    minutes_run = _resolve(profile_cfg.minutes_run_id, minutes_noise_run_id, "minutes_run_id") if use_minutes_noise_eff else None
    minutes_sigma_min_eff = (
        _resolve(profile_cfg.minutes_sigma_min, minutes_sigma_min, "minutes_sigma_min")
        if use_minutes_noise_eff
        else profile_cfg.minutes_sigma_min
    )
    seed_eff = seed if seed is not None else (profile_cfg.seed or 1234)
    min_play_prob_eff = min_play_prob if min_play_prob is not None else profile_cfg.min_play_prob
    team_factor_sigma_eff = team_factor_sigma if team_factor_sigma is not None else profile_cfg.team_factor_sigma
    team_factor_gamma_eff = team_factor_gamma if team_factor_gamma is not None else profile_cfg.team_factor_gamma
    worlds_per_chunk = max(1, profile_cfg.worlds_per_chunk)

    root = data_root or data_path()
    start_dt = _parse_date(start_date)
    end_dt = _parse_date(end_date)
    start_ts = pd.Timestamp(start_dt).normalize()
    end_ts = pd.Timestamp(end_dt).normalize()

    typer.echo(
        f"[sim_v2] profile={profile_cfg.name} config={profiles_path or DEFAULT_PROFILES_PATH} "
        f"worlds_per_chunk={worlds_per_chunk} seed={seed_eff}"
    )
    typer.echo(
        f"[sim_v2] fpts_run_id={fpts_run} rates_run_id={rates_run} minutes_run_id={minutes_run} "
        f"use_rates_noise={use_rates_noise_eff} split={rates_split} "
        f"use_minutes_noise={use_minutes_noise_eff} sigma_min={minutes_sigma_min_eff} min_play_prob={min_play_prob_eff}"
    )

    minutes_noise_params = None
    if use_minutes_noise_eff:
        minutes_noise_params = load_minutes_noise_params(data_root=root, minutes_run_id=minutes_run)
        typer.echo(
            f"[sim_v2] using minutes noise run_id={minutes_noise_params.run_id} "
            f"sigma_min={minutes_sigma_min_eff:.3f} path={minutes_noise_params.source_path}"
        )

    if use_rates_noise_eff:
        bundle = load_fpts_bundle(fpts_run, data_root=root)
        residual_model = None
        typer.echo(f"[sim_v2] run_id={fpts_run} feature_cols={len(bundle.feature_cols)}")
        typer.echo(f"[sim_v2] using rates noise; skipping FPTS residual model for run_id={fpts_run}")
    else:
        bundle, residual_model = load_fpts_and_residual(fpts_run, data_root=root)
        typer.echo(f"[sim_v2] run_id={fpts_run} feature_cols={len(bundle.feature_cols)}")
        typer.echo(f"[sim_v2] team_factor_sigma={team_factor_sigma_eff} alpha_gamma={team_factor_gamma_eff}")

    noise_params = None
    noise_path = None
    stat_targets: list[str] = []
    if use_rates_noise_eff:
        noise_params, noise_path = load_rates_noise_params(
            data_root=root, run_id=rates_run, split=rates_split or "val"
        )
        stat_targets = list(noise_params.keys())
        typer.echo(
            f"[sim_v2] using rates noise run_id={rates_run or 'current'} split={rates_split or 'val'} "
            f"targets={len(stat_targets)} path={noise_path}"
        )

    df = _load_base(root, start_ts, end_ts)
    output_base = output_root or (root / "artifacts" / "sim_v2" / "worlds_fpts_v2")

    for game_date, date_df in df.groupby("game_date"):
        date_df = date_df.copy()
        try:
            minutes_col = _resolve_minutes_column(date_df)
        except KeyError:
            typer.echo(f"[sim_v2] {game_date.date()} missing minutes columns; skipping.")
            continue
        date_df[minutes_col] = pd.to_numeric(date_df[minutes_col], errors="coerce")
        date_df["is_starter"] = pd.to_numeric(date_df.get("is_starter"), errors="coerce")
        if "play_prob" in date_df.columns:
            date_df["play_prob"] = pd.to_numeric(date_df["play_prob"], errors="coerce")
        else:
            date_df["play_prob"] = 1.0
        date_df = _ensure_status_bucket(date_df)
        date_df = date_df[date_df[minutes_col].notna()]
        date_df = date_df[date_df["play_prob"].fillna(0.0) >= min_play_prob_eff]
        if date_df.empty:
            continue
        date_df = date_df.reset_index(drop=True)

        out_dir = output_base / f"game_date={game_date.date()}"
        out_dir.mkdir(parents=True, exist_ok=True)

        date_seed = seed_eff + int(pd.Timestamp(game_date).toordinal())

        if use_rates_noise_eff and noise_params is not None:
            mapping = _resolve_rate_columns(date_df, stat_targets)
            missing_targets = [t for t in stat_targets if t not in mapping]
            if missing_targets:
                typer.echo(f"[sim_v2] warning: missing rate columns for targets={missing_targets}; skipping.")
            if not mapping:
                continue

            minutes_center = date_df[minutes_col].to_numpy(dtype=float)
            play_prob_arr = date_df["play_prob"].fillna(1.0).to_numpy(dtype=float)

            sigma_minutes = None
            if use_minutes_noise_eff and minutes_noise_params is not None:
                sigma_minutes = build_sigma_per_player(
                    date_df,
                    minutes_noise_params,
                    minutes_col=minutes_col,
                    starter_col="is_starter",
                    status_col="status_bucket",
                )
                if minutes_sigma_min_eff is not None:
                    sigma_minutes = np.maximum(sigma_minutes, minutes_sigma_min_eff)

            rate_arrays: dict[str, np.ndarray] = {}
            base_mu: dict[str, np.ndarray] = {}
            for target, col in mapping.items():
                rates = pd.to_numeric(date_df[col], errors="coerce").to_numpy()
                rates = np.nan_to_num(rates, nan=0.0)
                rate_arrays[target] = rates
                base = target.replace("_per_min", "")
                base_mu[base] = np.clip(rates * minutes_center, 0.0, None)

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

            rng = np.random.default_rng(date_seed)
            world_fpts_samples: list[np.ndarray] = []
            for chunk_start in range(0, n_worlds, worlds_per_chunk):
                chunk_size = min(worlds_per_chunk, n_worlds - chunk_start)
                if use_minutes_noise_eff and sigma_minutes is not None:
                    eps_minutes = rng.standard_normal(size=(chunk_size, len(minutes_center)))
                    u_active = rng.random(size=(chunk_size, len(minutes_center)))
                    minutes_worlds = np.maximum(minutes_center[None, :] + eps_minutes * sigma_minutes[None, :], 0.0)
                    active_mask = u_active < play_prob_arr[None, :]
                    minutes_worlds = minutes_worlds * active_mask
                    # TODO: reconcile minutes to per-team caps (e.g., 240) per world.
                else:
                    minutes_worlds = np.repeat(minutes_center[None, :], chunk_size, axis=0)

                stat_totals: dict[str, np.ndarray] = {}
                for target, rates in rate_arrays.items():
                    params = noise_params.get(target, {})
                    sigma_team = float(params.get("sigma_team", 0.0) or 0.0)
                    sigma_player = float(params.get("sigma_player", 0.0) or 0.0)

                    mu = np.clip(rates[None, :] * minutes_worlds, 0.0, None)
                    team_shock = np.zeros_like(mu)
                    if sigma_team > 0:
                        for key, idxs in group_map.items():
                            ts = rng.normal(loc=0.0, scale=sigma_team, size=chunk_size)
                            team_shock[:, idxs] = ts[:, None]
                    player_eps = rng.normal(loc=0.0, scale=sigma_player, size=mu.shape)
                    total = np.clip(mu + team_shock + player_eps, 0.0, None)
                    base = target.replace("_per_min", "")
                    stat_totals[base] = total

                dk_fpts_worlds = _compute_fpts_from_stats(stat_totals)
                world_fpts_samples.append(dk_fpts_worlds)

                base_cols = [
                    "game_date",
                    "game_id",
                    "team_id",
                    "player_id",
                    "is_starter",
                    minutes_col,
                    "dk_fpts_mean",
                ]
                if minutes_col != "minutes_p50" and "minutes_p50" in date_df.columns:
                    base_cols.append("minutes_p50")
                if "play_prob" in date_df.columns:
                    base_cols.append("play_prob")
                base_cols = list(dict.fromkeys(base_cols))

                for offset in range(chunk_size):
                    world_id = chunk_start + offset
                    world_df = date_df[base_cols + ([c for c in ["dk_fpts_actual"] if c in date_df.columns])].copy()
                    world_df["world_id"] = world_id
                    world_df["minutes_sim"] = minutes_worlds[offset]
                    world_df["dk_fpts_world"] = dk_fpts_worlds[offset]

                    for base, values in stat_totals.items():
                        world_df[f"{base}_sim"] = values[offset]
                    # Derived stats
                    if "oreb" in stat_totals and "dreb" in stat_totals:
                        world_df["reb_sim"] = stat_totals["oreb"][offset] + stat_totals["dreb"][offset]

                    out_path = out_dir / f"world={world_id:04d}.parquet"
                    world_df.to_parquet(out_path, index=False)

            if world_fpts_samples:
                combined_fpts = np.concatenate(world_fpts_samples, axis=None)
                typer.echo(
                    f"[sim_v2] {game_date.date()} dk_fpts_world min/med/max="
                    f"{combined_fpts.min():.2f}/{np.median(combined_fpts):.2f}/{combined_fpts.max():.2f}"
                )
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

            world_fpts_samples: list[np.ndarray] = []
            for world_id in range(n_worlds):
                rng = np.random.default_rng(date_seed + world_id)
                eps = sample_residuals_with_team_factor(
                    date_df,
                    residual_model,
                    rng,
                    dk_fpts_col="dk_fpts_mean",
                    minutes_col=minutes_col,
                    is_starter_col="is_starter",
                    game_id_col="game_id",
                    team_id_col="team_id",
                    team_factor_sigma=team_factor_sigma_eff,
                    alpha_gamma=team_factor_gamma_eff,
                )
                dk_fpts_world = date_df["dk_fpts_mean"].to_numpy() + eps
                base_cols = [
                    "game_date",
                    "game_id",
                    "team_id",
                    "player_id",
                    "is_starter",
                    minutes_col,
                    "dk_fpts_mean",
                ]
                if minutes_col != "minutes_p50" and "minutes_p50" in date_df.columns:
                    base_cols.append("minutes_p50")
                base_cols = list(dict.fromkeys(base_cols))
                world_df = date_df[base_cols + ([c for c in ["dk_fpts_actual"] if c in date_df.columns])].copy()
                world_df["world_id"] = world_id
                world_df["minutes_sim"] = date_df[minutes_col].to_numpy()
                world_df["dk_fpts_world"] = dk_fpts_world
                out_path = out_dir / f"world={world_id:04d}.parquet"
                world_df.to_parquet(out_path, index=False)
                world_fpts_samples.append(dk_fpts_world)

            if world_fpts_samples:
                combined = np.concatenate(world_fpts_samples, axis=None)
                typer.echo(
                    f"[sim_v2] {game_date.date()} dk_fpts_world min/med/max="
                    f"{combined.min():.2f}/{np.median(combined):.2f}/{combined.max():.2f}"
                )


if __name__ == "__main__":
    app()
