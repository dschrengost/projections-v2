"""
Diagnostics: compare simulated stat/FPTS noise vs actual residuals.

Loads worlds_fpts_v2 outputs, joins to actuals and mean predictions, and reports
variance and same-team correlation of residuals (actual vs sim).
"""

from __future__ import annotations

import itertools
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import typer

from projections.fpts_v2.scoring import compute_dk_fpts
from projections.paths import data_path

app = typer.Typer(add_completion=False)

STATS = [
    "fga2",
    "fga3",
    "fta",
    "ast",
    "tov",
    "oreb",
    "dreb",
    "stl",
    "blk",
    "reb",
    "dk_fpts",
]


def _dk_from_components(
    fga2: np.ndarray,
    fga3: np.ndarray,
    fta: np.ndarray,
    ast: np.ndarray,
    tov: np.ndarray,
    oreb: np.ndarray,
    dreb: np.ndarray,
    stl: np.ndarray,
    blk: np.ndarray,
) -> np.ndarray:
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
            "pf": np.zeros_like(pts),
            "plus_minus": np.zeros_like(pts),
        }
    )
    return compute_dk_fpts(df).to_numpy()


def _parse_date(value: str) -> pd.Timestamp:
    try:
        return pd.Timestamp(datetime.fromisoformat(value).date()).normalize()
    except ValueError as exc:
        raise typer.BadParameter(f"Invalid date: {value}") from exc


def _iter_base_partitions(root: Path, start: pd.Timestamp, end: pd.Timestamp) -> List[Path]:
    base = root / "gold" / "fpts_training_base"
    parts: List[Path] = []
    for season_dir in base.glob("season=*"):
        for day_dir in season_dir.glob("game_date=*"):
            try:
                day = pd.Timestamp(day_dir.name.split("=", 1)[1]).normalize()
            except ValueError:
                continue
            if start <= day <= end:
                cand = day_dir / "fpts_training_base.parquet"
                if cand.exists():
                    parts.append(cand)
    return sorted(parts)


def _load_base(root: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    parts = _iter_base_partitions(root, start, end)
    if not parts:
        raise FileNotFoundError("No fpts_training_base partitions found in date range.")
    frames = [pd.read_parquet(p) for p in parts]
    df = pd.concat(frames, ignore_index=True)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.normalize()
    return df


def _load_worlds(root: Path, start: pd.Timestamp, end: pd.Timestamp, n_worlds: Optional[int]) -> pd.DataFrame:
    base = root / "artifacts" / "sim_v2" / "worlds_fpts_v2"
    frames: List[pd.DataFrame] = []
    for day in pd.date_range(start, end, freq="D"):
        day_dir = base / f"game_date={day.date()}"
        if not day_dir.exists():
            continue
        world_files = sorted(day_dir.glob("world=*.parquet"))
        if n_worlds is not None:
            world_files = world_files[:n_worlds]
        for f in world_files:
            try:
                df = pd.read_parquet(f)
            except Exception:
                continue
            frames.append(df)
    if not frames:
        raise FileNotFoundError("No world files found for requested range.")
    worlds = pd.concat(frames, ignore_index=True)
    worlds["game_date"] = pd.to_datetime(worlds["game_date"]).dt.normalize()
    return worlds


def _team_corr(
    residuals: np.ndarray,
    game_ids: np.ndarray,
    team_ids: np.ndarray,
    world_ids: Optional[np.ndarray] = None,
) -> Optional[float]:
    masks = [np.isfinite(residuals), np.isfinite(game_ids), np.isfinite(team_ids)]
    if world_ids is not None:
        masks.append(np.isfinite(world_ids))
    mask = np.logical_and.reduce(masks)
    if mask.sum() < 2:
        return None
    r = residuals[mask]
    g = game_ids[mask].astype(int)
    t = team_ids[mask].astype(int)
    w = world_ids[mask].astype(int) if world_ids is not None else None
    var_all = np.var(r)
    if var_all <= 0:
        return None
    cov_sum = 0.0
    pair_count = 0
    keys = zip(g, t, w) if w is not None else zip(g, t)
    for key in set(keys):
        if w is not None:
            idx = np.where((g == key[0]) & (t == key[1]) & (w == key[2]))[0]
        else:
            idx = np.where((g == key[0]) & (t == key[1]))[0]
        n = idx.size
        if n < 2:
            continue
        vals = r[idx]
        s1 = float(vals.sum())
        s2 = float((vals**2).sum())
        pairs = n * (n - 1)
        cov_pairs = (s1 * s1 - s2) / pairs
        cov_sum += cov_pairs * pairs
        pair_count += pairs
    if pair_count == 0:
        return None
    cov_avg = cov_sum / pair_count
    return cov_avg / var_all


def _prepare_means(base: pd.DataFrame) -> Dict[str, np.ndarray]:
    means: Dict[str, np.ndarray] = {}
    minutes = pd.to_numeric(base.get("minutes_p50"), errors="coerce").to_numpy()
    for stat in STATS:
        if stat == "reb":
            # derive later if oreb/dreb available
            continue
        pred_col = f"pred_{stat}_per_min"
        if pred_col in base.columns:
            rates = pd.to_numeric(base[pred_col], errors="coerce").to_numpy()
            means[stat] = np.clip(rates * minutes, 0.0, None)
    # reb mean if components exist
    if "oreb" in means and "dreb" in means:
        means["reb"] = means["oreb"] + means["dreb"]
    return means


def _compute_actual_dk(base: pd.DataFrame) -> np.ndarray:
    if "dk_fpts_actual" in base.columns:
        return pd.to_numeric(base["dk_fpts_actual"], errors="coerce").to_numpy()
    # Derive using same approximations as the simulator if raw stat totals are available.
    required_parts = ["fga2", "fga3", "fta", "ast", "tov", "oreb", "dreb", "stl", "blk"]
    if set(required_parts).issubset(base.columns):
        fga2 = pd.to_numeric(base["fga2"], errors="coerce").to_numpy()
        fga3 = pd.to_numeric(base["fga3"], errors="coerce").to_numpy()
        fta = pd.to_numeric(base["fta"], errors="coerce").to_numpy()
        ast = pd.to_numeric(base["ast"], errors="coerce").to_numpy()
        tov = pd.to_numeric(base["tov"], errors="coerce").to_numpy()
        oreb = pd.to_numeric(base["oreb"], errors="coerce").to_numpy()
        dreb = pd.to_numeric(base["dreb"], errors="coerce").to_numpy()
        stl = pd.to_numeric(base["stl"], errors="coerce").to_numpy()
        blk = pd.to_numeric(base["blk"], errors="coerce").to_numpy()
        return _dk_from_components(fga2, fga3, fta, ast, tov, oreb, dreb, stl, blk)
    return np.full(len(base), np.nan, dtype=float)


@app.command()
def main(
    start_date: str = typer.Option(..., "--start-date"),
    end_date: str = typer.Option(..., "--end-date"),
    data_root: Optional[Path] = typer.Option(None, "--data-root"),
    n_worlds: Optional[int] = typer.Option(None, "--n-worlds", help="Optional cap on number of worlds to read per date."),
    verbose: bool = typer.Option(False, "--verbose/--no-verbose"),
) -> None:
    root = data_root or data_path()
    root = Path(str(root)).resolve()
    start = _parse_date(start_date)
    end = _parse_date(end_date)

    worlds = _load_worlds(root, start, end, n_worlds)
    base = _load_base(root, start, end)

    # Use first world per player as means source (dk_fpts_mean) if present; fallback to computed means.
    mean_lookup = (
        worlds.sort_values("world_id")
        .drop_duplicates(subset=["game_date", "game_id", "player_id"])
        .set_index(["game_date", "game_id", "player_id"])
    )
    base_idx = base.set_index(["game_date", "game_id", "player_id"])
    merged_mean = base_idx.join(mean_lookup[["dk_fpts_mean"]] if "dk_fpts_mean" in mean_lookup else pd.DataFrame(), how="left")
    mean_from_rates = _prepare_means(base_idx.reset_index())
    if "dk_fpts_mean" not in merged_mean or merged_mean["dk_fpts_mean"].isna().all():
        # Compute dk mean from rate means if available.
        components = ["fga2", "fga3", "fta", "ast", "tov", "oreb", "dreb", "stl", "blk"]
        if all(c in mean_from_rates for c in components):
            merged_mean["dk_fpts_mean"] = _dk_from_components(
                mean_from_rates["fga2"],
                mean_from_rates["fga3"],
                mean_from_rates["fta"],
                mean_from_rates["ast"],
                mean_from_rates["tov"],
                mean_from_rates["oreb"],
                mean_from_rates["dreb"],
                mean_from_rates["stl"],
                mean_from_rates["blk"],
            )

    # Actual DK fpts
    base_idx["dk_fpts_actual"] = _compute_actual_dk(base_idx.reset_index())

    # Merge actuals and means into worlds
    worlds_idx = worlds.set_index(["game_date", "game_id", "player_id"])
    merged = worlds_idx.join(base_idx, how="inner", lsuffix="_world", rsuffix="_base")
    merged["dk_fpts_mean"] = merged.get("dk_fpts_mean")
    merged["dk_fpts_mean"] = merged["dk_fpts_mean"].fillna(merged_mean.get("dk_fpts_mean"))

    results: List[Tuple[str, float, float, Optional[float], Optional[float], int, int]] = []

    for stat in STATS:
        stat_sim_col = f"{stat}_sim" if stat != "dk_fpts" else "dk_fpts_world"
        if stat_sim_col not in merged.columns:
            if verbose:
                typer.echo(f"[warn] missing sim column for {stat}; skipping.")
            continue
        mean_col = None
        actual_col = None
        if stat == "dk_fpts":
            mean_col = "dk_fpts_mean"
            if "dk_fpts_actual" in merged.columns:
                actual_col = "dk_fpts_actual"
            elif "dk_fpts_actual_base" in merged.columns:
                actual_col = "dk_fpts_actual_base"
            else:
                actual_col = "dk_fpts_actual"
        else:
            mean_col = f"mean_{stat}"
            # build mean column from prepared means if available
            if stat in mean_from_rates:
                base_series = pd.Series(mean_from_rates[stat], index=base_idx.index)
                merged[mean_col] = base_series.reindex(merged.index)
            elif stat == "reb" and "oreb" in mean_from_rates and "dreb" in mean_from_rates:
                base_series = pd.Series(mean_from_rates["oreb"] + mean_from_rates["dreb"], index=base_idx.index)
                merged[mean_col] = base_series.reindex(merged.index)
            actual_col = stat if stat in merged.columns else f"{stat}_base"
        if mean_col not in merged.columns or actual_col not in merged.columns:
            if verbose:
                typer.echo(f"[warn] missing mean/actual for {stat}; skipping.")
            continue

        mean_vals = pd.to_numeric(merged[mean_col], errors="coerce")
        actual_vals = pd.to_numeric(merged[actual_col], errors="coerce")
        sim_vals = pd.to_numeric(merged[stat_sim_col], errors="coerce")

        r_actual = actual_vals - mean_vals
        r_sim = sim_vals - mean_vals

        var_actual = float(np.nanvar(r_actual.to_numpy()))
        var_sim = float(np.nanvar(r_sim.to_numpy()))
        count_actual = int(r_actual.notna().sum())
        count_sim = int(r_sim.notna().sum())

        game_ids_arr = merged.index.get_level_values("game_id").to_numpy()
        team_col = "team_id_base" if "team_id_base" in merged.columns else "team_id_world"
        team_ids_arr = pd.to_numeric(merged[team_col], errors="coerce").to_numpy()
        world_ids_arr = pd.to_numeric(merged["world_id"], errors="coerce").to_numpy() if "world_id" in merged.columns else None
        rho_actual = _team_corr(r_actual.to_numpy(), game_ids_arr, team_ids_arr, world_ids=None)
        rho_sim = _team_corr(r_sim.to_numpy(), game_ids_arr, team_ids_arr, world_ids=world_ids_arr)

        results.append((stat, var_actual, var_sim, rho_actual, rho_sim, count_actual, count_sim))

    # Print table
    headers = ["stat", "var_actual", "var_sim", "rho_team_actual", "rho_team_sim", "count_actual", "count_sim"]
    typer.echo(" | ".join(headers))
    for row in sorted(results, key=lambda x: x[0]):
        stat, v_a, v_s, r_a, r_s, c_a, c_s = row
        typer.echo(
            f"{stat} | {v_a:.4f} | {v_s:.4f} | "
            f"{'nan' if r_a is None else f'{r_a:.4f}'} | "
            f"{'nan' if r_s is None else f'{r_s:.4f}'} | "
            f"{c_a} | {c_s}"
        )

    typer.echo(
        f"[done] dates={len(pd.date_range(start, end))} players={merged.index.get_level_values('player_id').nunique()} "
        f"world_rows={len(merged)}"
    )


if __name__ == "__main__":
    app()
