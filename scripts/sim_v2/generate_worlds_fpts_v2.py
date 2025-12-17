"""Generate FPTS v2 worlds using residual model noise."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import typer

from projections.fpts_v2.features import CATEGORICAL_FEATURES_DEFAULT, build_fpts_design_matrix
from projections.fpts_v2.loader import load_fpts_and_residual, load_fpts_bundle
from projections.fpts_v2.scoring import compute_dk_fpts
from projections.paths import data_path, get_project_root
from projections.sim_v2.config import DEFAULT_PROFILES_PATH, UsageSharesConfig, load_sim_v2_profile
from projections.sim_v2.game_script import GameScriptConfig, classify_script, sample_minutes_with_scripts
from projections.sim_v2.minutes_noise import (
    build_sigma_per_player,
    enforce_team_240_minutes,
    load_minutes_noise_params,
    status_bucket_from_raw,
)
from projections.sim_v2.noise import load_rates_noise_params
from projections.sim_v2.residuals import sample_residuals_with_team_factor

app = typer.Typer(add_completion=False)

DEFAULT_MAX_ROTATION_SIZE = 10


def _build_implied_team_points(
    minutes_df: pd.DataFrame,
    schedule_df: pd.DataFrame,
) -> dict[tuple[int, int], float]:
    """Return {(game_id, team_id): implied_points} using total/spread_home + home/away ids."""

    if minutes_df.empty or schedule_df.empty:
        return {}
    required = {"game_id", "total", "spread_home"}
    sched_required = {"game_id", "home_team_id", "away_team_id"}
    if not required.issubset(minutes_df.columns) or not sched_required.issubset(schedule_df.columns):
        return {}

    odds = minutes_df.loc[:, ["game_id", "total", "spread_home"]].dropna(subset=["game_id"]).drop_duplicates("game_id").copy()
    odds["game_id"] = pd.to_numeric(odds["game_id"], errors="coerce").astype("Int64")
    odds["total"] = pd.to_numeric(odds["total"], errors="coerce")
    odds["spread_home"] = pd.to_numeric(odds["spread_home"], errors="coerce")
    odds = odds.dropna(subset=["game_id", "total", "spread_home"]).copy()

    sched = schedule_df.loc[:, ["game_id", "home_team_id", "away_team_id"]].copy()
    sched["game_id"] = pd.to_numeric(sched["game_id"], errors="coerce").astype("Int64")
    for col in ("home_team_id", "away_team_id"):
        sched[col] = pd.to_numeric(sched[col], errors="coerce").astype("Int64")
    sched = sched.dropna(subset=["game_id", "home_team_id", "away_team_id"]).drop_duplicates("game_id").copy()

    merged = odds.merge(sched, on="game_id", how="inner")
    if merged.empty:
        return {}

    implied: dict[tuple[int, int], float] = {}
    for _, row in merged.iterrows():
        gid = int(row["game_id"])
        total = float(row["total"])
        spread_home = float(row["spread_home"])
        home_id = int(row["home_team_id"])
        away_id = int(row["away_team_id"])
        implied_home = total / 2.0 - spread_home / 2.0
        implied_away = total - implied_home
        implied[(gid, home_id)] = float(implied_home)
        implied[(gid, away_id)] = float(implied_away)
    return implied


def _apply_team_points_vegas_anchor(
    pts_worlds: np.ndarray,
    *,
    group_map: dict[tuple[int, int], np.ndarray],
    implied_team_points: dict[tuple[int, int], float],
    drift_pct: float,
) -> np.ndarray:
    """Scale per-team points in-place so team totals fall within implied*(1±drift_pct)."""

    if pts_worlds.size == 0 or not group_map or not implied_team_points:
        return pts_worlds
    drift = float(max(0.0, drift_pct))
    eps = 1e-6
    for key, idxs in group_map.items():
        implied = implied_team_points.get((int(key[0]), int(key[1])))
        if implied is None or not np.isfinite(implied):
            continue
        implied_f = float(implied)
        lo = implied_f * (1.0 - drift)
        hi = implied_f * (1.0 + drift)

        team_pts = pts_worlds[:, idxs].sum(axis=1)
        scale = np.ones_like(team_pts, dtype=float)

        low_mask = (team_pts < lo) & (team_pts > eps)
        high_mask = team_pts > hi
        if low_mask.any():
            scale[low_mask] = lo / team_pts[low_mask]
        if high_mask.any():
            scale[high_mask] = hi / np.maximum(team_pts[high_mask], eps)
        if (scale != 1.0).any():
            pts_worlds[:, idxs] *= scale[:, None]
    return pts_worlds


def _compute_usage_shares(
    log_weights: np.ndarray,
    team_indices: np.ndarray,
    active_mask: np.ndarray,
    temperature: float,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Compute stochastic usage shares within each team via logit noise + softmax.

    Args:
        log_weights: shape (n_worlds, n_players) - log of baseline weights
        team_indices: shape (n_players,) - integer indices mapping players to teams
        active_mask: shape (n_worlds, n_players) - True if player is active
        temperature: softmax temperature (1.0 = standard, <1 = sharper)
        noise_std: std of Gaussian noise to add to log_weights
        rng: random generator

    Returns:
        shares: shape (n_worlds, n_players) - shares summing to 1 within each team per world
    """
    n_worlds, n_players = log_weights.shape
    n_teams = int(team_indices.max()) + 1 if len(team_indices) > 0 else 0

    # Add noise to log weights for active players
    noisy_logw = log_weights.copy()
    if noise_std > 0:
        noise = rng.normal(loc=0.0, scale=noise_std, size=(n_worlds, n_players))
        noisy_logw += noise * active_mask.astype(float)

    # Apply temperature
    scaled_logw = noisy_logw / max(temperature, 1e-6)

    # Set inactive players to -inf so they get share=0
    scaled_logw = np.where(active_mask, scaled_logw, -np.inf)

    # Compute softmax per team
    shares = np.zeros((n_worlds, n_players), dtype=float)
    for t in range(n_teams):
        team_mask = team_indices == t
        if not team_mask.any():
            continue
        team_logits = scaled_logw[:, team_mask]  # (n_worlds, n_team_players)
        # Stable softmax
        max_logits = np.max(team_logits, axis=1, keepdims=True)
        max_logits = np.where(np.isfinite(max_logits), max_logits, 0.0)
        exp_logits = np.exp(team_logits - max_logits)
        sum_exp = exp_logits.sum(axis=1, keepdims=True)
        team_shares = np.where(sum_exp > 0, exp_logits / np.maximum(sum_exp, 1e-12), 0.0)
        shares[:, team_mask] = team_shares

    return shares


def _apply_usage_shares_allocation(
    stat_totals: dict[str, np.ndarray],
    minutes_worlds: np.ndarray,
    rate_arrays: dict[str, np.ndarray],
    group_map: dict[tuple[int, int], np.ndarray],
    usage_cfg: UsageSharesConfig,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """
    Apply stochastic usage share allocation for FGA/FTA/TOV.

    For each target in usage_cfg.targets:
    1. Compute baseline weights w_i = rate_per_min_i * minutes_i
    2. Compute log weights, add noise, apply softmax within each team
    3. Compute team total and redistribute according to shares

    This preserves team totals while introducing within-team coupling.

    Args:
        stat_totals: dict of stat arrays, shape (n_worlds, n_players)
        minutes_worlds: shape (n_worlds, n_players)
        rate_arrays: dict mapping target names to per-minute rates, shape (n_players,)
        group_map: {(game_id, team_id): player_indices}
        usage_cfg: UsageSharesConfig
        rng: random generator

    Returns:
        Updated stat_totals dict
    """
    if not usage_cfg.enabled:
        return stat_totals

    n_worlds, n_players = minutes_worlds.shape
    eps = 1e-9

    # Build team_indices from group_map
    team_indices = np.zeros(n_players, dtype=int)
    team_to_idx = {}
    for key, player_idxs in group_map.items():
        if key not in team_to_idx:
            team_to_idx[key] = len(team_to_idx)
        team_indices[player_idxs] = team_to_idx[key]

    # Active mask: players with minutes >= cutoff
    active_mask = minutes_worlds >= usage_cfg.min_minutes_active_cutoff

    # Process each target
    for target in usage_cfg.targets:
        if target == "fga":
            # FGA = fga2 + fga3
            fga2_rate = rate_arrays.get("fga2_per_min")
            fga3_rate = rate_arrays.get("fga3_per_min")
            if fga2_rate is None or fga3_rate is None:
                continue
            fga_rate = fga2_rate + fga3_rate

            # Compute baseline weights
            weights = np.clip(fga_rate[None, :] * minutes_worlds, eps, None)
            log_weights = np.log(weights)

            # Compute shares with noise
            shares = _compute_usage_shares(
                log_weights,
                team_indices,
                active_mask,
                usage_cfg.share_temperature,
                usage_cfg.share_noise_std,
                rng,
            )

            # Compute team totals from baseline (before reallocation)
            # Team total = sum of original fga2 + fga3 for team
            orig_fga2 = stat_totals.get("fga2")
            orig_fga3 = stat_totals.get("fga3")
            if orig_fga2 is None or orig_fga3 is None:
                continue
            orig_fga = orig_fga2 + orig_fga3

            # Compute team totals per world
            team_totals = np.zeros((n_worlds, len(team_to_idx)), dtype=float)
            for key, player_idxs in group_map.items():
                tidx = team_to_idx[key]
                team_totals[:, tidx] = orig_fga[:, player_idxs].sum(axis=1)

            # Allocate to players based on shares
            new_fga = np.zeros_like(orig_fga)
            for key, player_idxs in group_map.items():
                tidx = team_to_idx[key]
                tt = team_totals[:, tidx : tidx + 1]  # (n_worlds, 1)
                new_fga[:, player_idxs] = shares[:, player_idxs] * tt

            # Split FGA into 2PA/3PA using player prior mix
            # p3_i = fga3_rate_i / (fga2_rate_i + fga3_rate_i)
            denom = fga2_rate + fga3_rate
            p3 = np.where(denom > eps, fga3_rate / denom, 0.0)  # (n_players,)
            new_fga3 = new_fga * p3[None, :]
            new_fga2 = new_fga - new_fga3

            stat_totals["fga2"] = new_fga2
            stat_totals["fga3"] = new_fga3

        elif target == "fta":
            fta_rate = rate_arrays.get("fta_per_min")
            if fta_rate is None:
                continue

            weights = np.clip(fta_rate[None, :] * minutes_worlds, eps, None)
            log_weights = np.log(weights)

            shares = _compute_usage_shares(
                log_weights,
                team_indices,
                active_mask,
                usage_cfg.share_temperature,
                usage_cfg.share_noise_std,
                rng,
            )

            orig_fta = stat_totals.get("fta")
            if orig_fta is None:
                continue

            team_totals = np.zeros((n_worlds, len(team_to_idx)), dtype=float)
            for key, player_idxs in group_map.items():
                tidx = team_to_idx[key]
                team_totals[:, tidx] = orig_fta[:, player_idxs].sum(axis=1)

            new_fta = np.zeros_like(orig_fta)
            for key, player_idxs in group_map.items():
                tidx = team_to_idx[key]
                tt = team_totals[:, tidx : tidx + 1]
                new_fta[:, player_idxs] = shares[:, player_idxs] * tt

            stat_totals["fta"] = new_fta

        elif target == "tov":
            tov_rate = rate_arrays.get("tov_per_min")
            if tov_rate is None:
                continue

            weights = np.clip(tov_rate[None, :] * minutes_worlds, eps, None)
            log_weights = np.log(weights)

            shares = _compute_usage_shares(
                log_weights,
                team_indices,
                active_mask,
                usage_cfg.share_temperature,
                usage_cfg.share_noise_std,
                rng,
            )

            orig_tov = stat_totals.get("tov")
            if orig_tov is None:
                continue

            team_totals = np.zeros((n_worlds, len(team_to_idx)), dtype=float)
            for key, player_idxs in group_map.items():
                tidx = team_to_idx[key]
                team_totals[:, tidx] = orig_tov[:, player_idxs].sum(axis=1)

            new_tov = np.zeros_like(orig_tov)
            for key, player_idxs in group_map.items():
                tidx = team_to_idx[key]
                tt = team_totals[:, tidx : tidx + 1]
                new_tov[:, player_idxs] = shares[:, player_idxs] * tt

            stat_totals["tov"] = new_tov

    return stat_totals


def _load_usage_shares_bundle(
    data_root: Path,
    usage_cfg: UsageSharesConfig,
) -> tuple[any, bool]:
    """
    Load usage shares LGBM residual bundle if configured.
    
    Returns:
        (bundle, success) - bundle is None if loading fails
    """
    if not usage_cfg.enabled or usage_cfg.backend != "lgbm_residual":
        return None, False
    
    try:
        from projections.usage_shares_v1.production import load_bundle, get_current_run_id
        
        # Resolve run_id
        run_id = usage_cfg.run_id
        
        # 1. Check profile-specified run_id
        if run_id is None:
            # 2. Check project config file
            config_path = get_project_root() / "config" / "usage_shares_current_run.json"
            if config_path.exists():
                try:
                    import json
                    cfg = json.loads(config_path.read_text())
                    run_id = cfg.get("run_id")
                except Exception:
                    pass
        
        if run_id is None:
            # 3. Check production config
            run_id = get_current_run_id()
        
        if run_id is None:
            # 4. Try latest decision run
            decision_dir = data_root / "artifacts" / "usage_shares_v1" / "decision"
            if decision_dir.exists():
                runs = sorted(decision_dir.glob("decision_*"))
                if runs:
                    run_id = runs[-1].name
        
        if run_id is None:
            typer.echo("[sim_v2] usage_shares: no run_id found, falling back to rate_weighted", err=True)
            return None, False
        
        typer.echo(f"[sim_v2] usage_shares: resolved run_id={run_id}")
        
        # Load model directly from decision directory if it's a decision run
        if run_id.startswith("decision_"):
            decision_path = data_root / "artifacts" / "usage_shares_v1" / "decision" / run_id
            if decision_path.exists():
                # Create a lightweight bundle for decision runs
                from dataclasses import dataclass
                from typing import Any
                
                @dataclass
                class DecisionBundle:
                    run_id: str
                    meta: dict[str, Any]
                    lgbm_models: dict | None = None
                    feature_cols: list[str] | None = None
                
                # Load results.json for config
                results_path = decision_path / "results.json"
                meta = {"data_root": str(data_root), "run_dir": str(decision_path)}
                feature_cols = None
                if results_path.exists():
                    import json
                    results = json.loads(results_path.read_text())
                    feature_cols = results.get("feature_cols")
                    meta["best_shrink"] = results.get("best_shrink", 0.75)
                
                bundle = DecisionBundle(
                    run_id=run_id,
                    meta=meta,
                    lgbm_models=None,  # Will load on demand
                    feature_cols=feature_cols,
                )
                return bundle, True
        
        # Otherwise use standard bundle loader
        bundle = load_bundle(
            data_root=data_root,
            run_id=run_id,
            backend="lgbm",
        )
        return bundle, True
        
    except Exception as e:
        typer.echo(f"[sim_v2] usage_shares: failed to load bundle: {e}, falling back", err=True)
        return None, False


# Vacancy clipping constants (conservative caps to prevent extreme values)
VAC_MIN_CAP = 240.0  # Max expected missing minutes per team
VAC_FGA_CAP = 100.0  # Max expected missing FGA per team


def _add_vacancy_features_from_minutes_df(
    df: pd.DataFrame,
    group_cols: tuple[str, str] = ("game_id", "team_id"),
    vacancy_mode: str = "game",
) -> pd.DataFrame:
    """
    Compute vacancy features from minutes model outputs.
    
    Vacancy v1: Uses (1 - play_prob) * minutes_pred_p50 as "expected missing minutes"
    per player, then aggregates to team level.
    
    This is leak-safe as it only uses model predictions, not actual outcomes.
    
    Args:
        df: DataFrame with minutes projections (must have play_prob and minutes columns)
        group_cols: Columns to group by for team aggregation
        vacancy_mode: "none" = set all vacancy to 0, "game" = compute from play_prob
        
    Returns:
        DataFrame with vacancy columns added
    """
    df = df.copy()
    
    # Handle vacancy_mode="none" - set all vacancy features to 0
    if vacancy_mode == "none":
        for vac_col in ["vac_min_szn", "vac_fga_szn", "vac_min_guard_szn", "vac_min_wing_szn", "vac_min_big_szn"]:
            df[vac_col] = 0.0
        return df
    
    # Resolve minutes and play_prob columns
    minutes_col = None
    for c in ["minutes_pred_p50", "minutes_p50_cond", "minutes_p50"]:
        if c in df.columns:
            minutes_col = c
            break
    
    prob_col = None
    for c in ["minutes_pred_play_prob", "play_prob"]:
        if c in df.columns:
            prob_col = c
            break
    
    if minutes_col is None:
        # No minutes column available, can't compute vacancy
        return df
    
    # Get minutes values
    minutes = pd.to_numeric(df[minutes_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    
    # Get play probability (default to 1.0 if missing = no expected vacancy)
    if prob_col is not None:
        play_prob = pd.to_numeric(df[prob_col], errors="coerce").fillna(1.0).clip(0.0, 1.0)
    else:
        play_prob = pd.Series(1.0, index=df.index)
    
    # Compute per-player vacancy minutes: expected missing minutes
    # vac_minutes = (1 - p) * m = minutes that player is expected to NOT play
    df["_vac_minutes"] = (1.0 - play_prob) * minutes
    
    # Get season rates for vacancy-weighted stats
    fga_rate = 0.0
    for c in ["season_fga_per_min", "pred_fga2_per_min", "pred_fga3_per_min"]:
        if c in df.columns:
            if c == "season_fga_per_min":
                fga_rate = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
                break
            else:
                fga_rate = fga_rate + pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    if isinstance(fga_rate, float):
        fga_rate = pd.Series(0.0, index=df.index)
    
    df["_vac_fga"] = df["_vac_minutes"] * fga_rate
    
    # Position flags for guard/wing/big classification
    pos_col = None
    for c in ["pos_bucket", "position_primary", "position"]:
        if c in df.columns:
            pos_col = c
            break
    
    if pos_col is not None:
        pos_str = df[pos_col].astype(str).str.upper()
        # Guard = PG or SG
        is_guard = (pos_str.str.contains("PG", na=False) | pos_str.str.contains("SG", na=False)).astype(float)
        # Wing = SF or PF (some overlap with guard/big positions)
        is_wing = (pos_str.str.contains("SF", na=False) | pos_str.str.contains("PF", na=False)).astype(float)
        # Big = C
        is_big = pos_str.str.contains("C", na=False).astype(float)
    else:
        # Check for individual position flags
        is_guard = 0.0
        is_wing = 0.0
        is_big = 0.0
        for flag in ["position_flags_PG", "position_flags_SG"]:
            if flag in df.columns:
                is_guard = is_guard + pd.to_numeric(df[flag], errors="coerce").fillna(0.0)
        for flag in ["position_flags_SF", "position_flags_PF"]:
            if flag in df.columns:
                is_wing = is_wing + pd.to_numeric(df[flag], errors="coerce").fillna(0.0)
        if "position_flags_C" in df.columns:
            is_big = pd.to_numeric(df["position_flags_C"], errors="coerce").fillna(0.0)
        # Convert to binary (if multiple flags, any match = 1)
        is_guard = (is_guard > 0).astype(float) if not isinstance(is_guard, float) else 0.0
        is_wing = (is_wing > 0).astype(float) if not isinstance(is_wing, float) else 0.0
        is_big = (is_big > 0).astype(float) if not isinstance(is_big, float) else 0.0
    
    # Ensure series for consistent indexing
    if isinstance(is_guard, float):
        is_guard = pd.Series(is_guard, index=df.index)
    if isinstance(is_wing, float):
        is_wing = pd.Series(is_wing, index=df.index)
    if isinstance(is_big, float):
        is_big = pd.Series(is_big, index=df.index)
    
    df["_vac_guard"] = df["_vac_minutes"] * is_guard
    df["_vac_wing"] = df["_vac_minutes"] * is_wing
    df["_vac_big"] = df["_vac_minutes"] * is_big
    
    # Aggregate to team level
    team_aggs = df.groupby(list(group_cols)).agg({
        "_vac_minutes": "sum",
        "_vac_fga": "sum",
        "_vac_guard": "sum",
        "_vac_wing": "sum",
        "_vac_big": "sum",
    }).rename(columns={
        "_vac_minutes": "vac_min_szn",
        "_vac_fga": "vac_fga_szn",
        "_vac_guard": "vac_min_guard_szn",
        "_vac_wing": "vac_min_wing_szn",
        "_vac_big": "vac_min_big_szn",
    })
    
    # Track clipping diagnostics before applying clips
    vac_min_max_before = team_aggs["vac_min_szn"].max()
    vac_fga_max_before = team_aggs["vac_fga_szn"].max()
    teams_clipped_min = (team_aggs["vac_min_szn"] > VAC_MIN_CAP).sum()
    teams_clipped_fga = (team_aggs["vac_fga_szn"] > VAC_FGA_CAP).sum()
    
    # Apply conservative clipping to prevent extreme values
    team_aggs["vac_min_szn"] = team_aggs["vac_min_szn"].clip(0, VAC_MIN_CAP)
    team_aggs["vac_fga_szn"] = team_aggs["vac_fga_szn"].clip(0, VAC_FGA_CAP)
    team_aggs["vac_min_guard_szn"] = team_aggs["vac_min_guard_szn"].clip(0, VAC_MIN_CAP)
    team_aggs["vac_min_wing_szn"] = team_aggs["vac_min_wing_szn"].clip(0, VAC_MIN_CAP)
    team_aggs["vac_min_big_szn"] = team_aggs["vac_min_big_szn"].clip(0, VAC_MIN_CAP)
    
    # Store diagnostics in dataframe attrs for later logging
    team_aggs.attrs["vac_min_max_before_clip"] = vac_min_max_before
    team_aggs.attrs["vac_fga_max_before_clip"] = vac_fga_max_before
    team_aggs.attrs["teams_clipped_min"] = teams_clipped_min
    team_aggs.attrs["teams_clipped_fga"] = teams_clipped_fga
    
    # Merge back to player rows
    df = df.merge(team_aggs, on=list(group_cols), how="left", suffixes=("_old", ""))
    
    # Clean up temp columns
    for col in ["_vac_minutes", "_vac_fga", "_vac_guard", "_vac_wing", "_vac_big"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    # Also drop any _old suffix columns if they existed
    for col in list(df.columns):
        if col.endswith("_old"):
            df = df.drop(columns=[col])
    
    return df


def _prepare_live_features_for_usage_shares(
    df: pd.DataFrame,
    group_cols: tuple[str, str] = ("game_id", "team_id"),
) -> pd.DataFrame:
    """
    Prepare live slate dataframe with features required by usage shares model.
    
    Derives features that can be computed from existing columns:
    - Renames: minutes_pred_p50, minutes_pred_play_prob
    - Ranks: minutes_pred_team_rank
    - Team aggregates: minutes_pred_p50_team_scaled, minutes_pred_team_sum_invalid
    - One-hot: position_flags_PG/SG/SF/PF/C
    - Odds: spread_close, total_close, has_odds, odds_lead_time_minutes
    - Season rates: use pred_* columns as proxy for season rates
    """
    df = df.copy()
    
    # 1. Rename columns (minutes predictions)
    for new_col, old_cols in [
        ("minutes_pred_p50", ["minutes_p50_cond", "minutes_p50"]),
        ("minutes_pred_play_prob", ["play_prob"]),
    ]:
        if new_col not in df.columns:
            for old in old_cols:
                if old in df.columns:
                    df[new_col] = df[old]
                    break
            if new_col not in df.columns:
                df[new_col] = 0.0
    
    # 2. Position flags (one-hot from pos_bucket)
    pos_col = None
    for c in ["pos_bucket", "position_primary", "position"]:
        if c in df.columns:
            pos_col = c
            break
    
    for pos in ["PG", "SG", "SF", "PF", "C"]:
        flag_col = f"position_flags_{pos}"
        if flag_col not in df.columns:
            if pos_col:
                df[flag_col] = df[pos_col].astype(str).str.contains(pos, case=False, na=False).astype(float)
            else:
                df[flag_col] = 0.0
    
    # 3. Odds columns
    if "spread_close" not in df.columns:
        if "spread_home" in df.columns:
            df["spread_close"] = df["spread_home"]
        else:
            df["spread_close"] = 0.0
    
    if "total_close" not in df.columns:
        if "total" in df.columns:
            df["total_close"] = df["total"]
        else:
            df["total_close"] = 220.0  # Default NBA total
    
    if "has_odds" not in df.columns:
        df["has_odds"] = (
            df.get("total", pd.Series([0.0])).notna() & 
            df.get("spread_home", pd.Series([0.0])).notna()
        ).astype(float)
    
    # Odds lead time
    if "odds_lead_time_minutes" not in df.columns:
        if "tip_ts" in df.columns and "odds_as_of_ts" in df.columns:
            tip = pd.to_datetime(df["tip_ts"], errors="coerce")
            odds_ts = pd.to_datetime(df["odds_as_of_ts"], errors="coerce")
            df["odds_lead_time_minutes"] = (tip - odds_ts).dt.total_seconds() / 60.0
            df["odds_lead_time_minutes"] = df["odds_lead_time_minutes"].fillna(0.0)
        else:
            df["odds_lead_time_minutes"] = 60.0  # Default 1 hour
    
    # 4. Team-level features (rank, scaled, validity)
    if "minutes_pred_team_rank" not in df.columns:
        df["minutes_pred_team_rank"] = (
            df.groupby(list(group_cols))["minutes_pred_p50"]
            .rank(ascending=False, method="min")
            .astype(float)
        )
    
    if "minutes_pred_p50_team_scaled" not in df.columns or "minutes_pred_team_sum_invalid" not in df.columns:
        team_sums = df.groupby(list(group_cols))["minutes_pred_p50"].transform("sum")
        df["minutes_pred_p50_team_scaled"] = (df["minutes_pred_p50"] / team_sums.clip(lower=1.0)) * 240.0
        df["minutes_pred_team_sum_invalid"] = ((team_sums < 200) | (team_sums > 280)).astype(float)
    
    # 5. Team implied totals (ITT) - derive from total and spread
    if "team_itt" not in df.columns or "opp_itt" not in df.columns:
        total = df.get("total_close", pd.Series([220.0] * len(df)))
        # Simple approximation: use total/2 as proxy since we don't know home/away per player
        df["team_itt"] = total / 2.0
        df["opp_itt"] = total / 2.0
    
    # 6. Season rates - use predicted rates as proxy
    rate_mapping = {
        "season_fga_per_min": ["pred_fga2_per_min", "pred_fga3_per_min"],  # Sum of 2PA + 3PA
        "season_fta_per_min": ["pred_fta_per_min"],
        "season_tov_per_min": ["pred_tov_per_min"],
    }
    for target, source_cols in rate_mapping.items():
        if target not in df.columns:
            val = 0.0
            for src in source_cols:
                if src in df.columns:
                    val = val + pd.to_numeric(df[src], errors="coerce").fillna(0.0)
            df[target] = val
    
    # 7. Vacancy features - pass through if computed upstream, else fallback to 0 for compatibility
    for vac_col in ["vac_min_szn", "vac_fga_szn", "vac_min_guard_szn", "vac_min_wing_szn", "vac_min_big_szn"]:
        if vac_col not in df.columns:
            df[vac_col] = 0.0
    
    # 8. Interaction features
    if "vac_min_szn_x_minutes_rank" not in df.columns:
        df["vac_min_szn_x_minutes_rank"] = df["vac_min_szn"] * df["minutes_pred_team_rank"]
    
    return df


def _apply_learned_fga_shares_allocation(
    stat_totals: dict[str, np.ndarray],
    player_df: pd.DataFrame,
    team_indices: np.ndarray,
    active_mask: np.ndarray,
    minutes_worlds: np.ndarray,
    usage_cfg: UsageSharesConfig,
    bundle: any,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """
    Apply learned FGA shares allocation using LGBM residual model.
    
    Args:
        stat_totals: dict of stat arrays, shape (n_worlds, n_players)
        player_df: DataFrame with player features
        team_indices: shape (n_players,) - integer indices mapping players to teams
        active_mask: shape (n_worlds, n_players) - True if player is active
        minutes_worlds: shape (n_worlds, n_players)
        usage_cfg: UsageSharesConfig
        bundle: Loaded usage shares bundle
        rng: random generator
        
    Returns:
        Updated stat_totals dict
    """
    from projections.usage_shares_v1.metrics import compute_baseline_log_weights
    from projections.usage_shares_v1.features import add_derived_features
    
    target = "fga"
    n_worlds, n_players = minutes_worlds.shape
    eps = 1e-9
    
    # Get original FGA totals
    orig_fga2 = stat_totals.get("fga2")
    orig_fga3 = stat_totals.get("fga3")
    if orig_fga2 is None or orig_fga3 is None:
        return stat_totals
    orig_fga = orig_fga2 + orig_fga3
    
    # Prepare features for prediction
    try:
        # First, prepare live features (renames, ranks, flags, etc.)
        pred_df = _prepare_live_features_for_usage_shares(player_df.copy())
        
        # Then add any additional derived features from usage_shares_v1 module
        pred_df = add_derived_features(pred_df)
        
        # Load model and config
        config = None
        
        # Try to use bundle directly
        if bundle.lgbm_models and target in bundle.lgbm_models:
            model = bundle.lgbm_models[target]
        else:
            # Try decision run structure
            data_root = Path(bundle.meta.get("data_root", "/home/daniel/projections-data"))
            run_id = bundle.run_id
            
            # Check decision directory
            decision_path = data_root / "artifacts" / "usage_shares_v1" / "decision" / run_id
            if (decision_path / f"model_{target}_starterless.txt").exists():
                import lightgbm as lgb
                model = lgb.Booster(model_file=str(decision_path / f"model_{target}_starterless.txt"))
                config_path = decision_path / "results.json"
                if config_path.exists():
                    config = json.loads(config_path.read_text())
            else:
                # Fallback
                typer.echo("[sim_v2] usage_shares: couldn't find model, using rate_weighted", err=True)
                return stat_totals
        
        # Get shrink value
        shrink = usage_cfg.shrink 
        if shrink is None and config:
            shrink = config.get("best_shrink", 0.75)
        if shrink is None:
            shrink = 0.75
        
        # Get feature columns from config
        feature_cols = None
        if config:
            feature_cols = config.get("feature_cols")
        if feature_cols is None and hasattr(bundle, "feature_cols"):
            feature_cols = bundle.feature_cols
        if feature_cols is None:
            # Default starterless features
            feature_cols = [
                "minutes_pred_p50", "minutes_pred_play_prob", "minutes_pred_p50_team_scaled",
                "minutes_pred_team_sum_invalid", "minutes_pred_team_rank",
                "position_flags_PG", "position_flags_SG", "position_flags_SF",
                "position_flags_PF", "position_flags_C",
                "spread_close", "total_close", "team_itt", "opp_itt", "has_odds",
                "odds_lead_time_minutes",
                "vac_min_szn", "vac_fga_szn", "vac_min_guard_szn", "vac_min_wing_szn", "vac_min_big_szn",
                "vac_min_szn_x_minutes_rank",
                "season_fga_per_min", "season_fta_per_min", "season_tov_per_min",
            ]
        
        # Prepare features
        available_cols = [c for c in feature_cols if c in pred_df.columns]
        if len(available_cols) < len(feature_cols) * 0.5:
            typer.echo(f"[sim_v2] usage_shares: insufficient features ({len(available_cols)}/{len(feature_cols)}), using rate_weighted", err=True)
            return stat_totals
        
        X = pred_df[available_cols].copy()
        for col in available_cols:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)
        
        # Predict delta
        delta_pred = model.predict(X.values)
        
        # Compute baseline log-weights
        alpha = 0.5
        baseline_logw = compute_baseline_log_weights(pred_df, target, alpha)
        
        # Compute learned log-weights: baseline + shrink * delta
        learned_logw = baseline_logw + shrink * delta_pred  # (n_players,)
        
    except Exception as e:
        typer.echo(f"[sim_v2] usage_shares: prediction failed: {e}, using rate_weighted", err=True)
        return stat_totals
    
    # Now compute shares per world (with noise if configured)
    n_teams = int(team_indices.max()) + 1 if len(team_indices) > 0 else 0
    
    # Broadcast log-weights to worlds
    log_weights_2d = np.broadcast_to(learned_logw[None, :], (n_worlds, n_players)).copy()
    
    # Add per-world noise if configured
    if usage_cfg.share_noise_std > 0:
        noise = rng.normal(loc=0.0, scale=usage_cfg.share_noise_std, size=(n_worlds, n_players))
        log_weights_2d += noise * active_mask.astype(float)
    
    # Apply temperature
    scaled_logw = log_weights_2d / max(usage_cfg.share_temperature, 1e-6)
    
    # Set inactive players (minutes < cutoff OR not active) to -inf
    min_cutoff_mask = minutes_worlds >= usage_cfg.min_minutes_active_cutoff
    valid_mask = active_mask & min_cutoff_mask
    scaled_logw = np.where(valid_mask, scaled_logw, -np.inf)
    
    # Compute softmax per team
    shares = np.zeros((n_worlds, n_players), dtype=float)
    for t in range(n_teams):
        team_mask = team_indices == t
        if not team_mask.any():
            continue
        team_logits = scaled_logw[:, team_mask]
        max_logits = np.max(team_logits, axis=1, keepdims=True)
        max_logits = np.where(np.isfinite(max_logits), max_logits, 0.0)
        exp_logits = np.exp(team_logits - max_logits)
        sum_exp = exp_logits.sum(axis=1, keepdims=True)
        team_shares = np.where(sum_exp > 0, exp_logits / np.maximum(sum_exp, 1e-12), 0.0)
        shares[:, team_mask] = team_shares
    
    # Compute team totals (from original FGA)
    team_totals = np.zeros((n_worlds, n_teams), dtype=float)
    for idx_list in range(n_players):
        t = team_indices[idx_list]
        team_totals[:, t] += orig_fga[:, idx_list]
    
    # Allocate new FGA
    new_fga = np.zeros_like(orig_fga)
    for idx_list in range(n_players):
        t = team_indices[idx_list]
        new_fga[:, idx_list] = shares[:, idx_list] * team_totals[:, t]
    
    # Split into FGA2/FGA3 using player prior mix
    fga2_prior = orig_fga2.mean(axis=0) + eps
    fga3_prior = orig_fga3.mean(axis=0) + eps
    p3 = fga3_prior / (fga2_prior + fga3_prior)  # (n_players,)
    
    new_fga3 = new_fga * p3[None, :]
    new_fga2 = new_fga - new_fga3
    
    # Clip negatives (numerical safety)
    new_fga2 = np.clip(new_fga2, 0.0, None)
    new_fga3 = np.clip(new_fga3, 0.0, None)
    
    stat_totals["fga2"] = new_fga2
    stat_totals["fga3"] = new_fga3
    
    return stat_totals


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


def _read_latest_run_id(base_dir: Path) -> Optional[str]:
    latest = base_dir / "latest_run.json"
    if not latest.exists():
        return None
    try:
        payload = json.loads(latest.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    run_id = payload.get("run_id") or payload.get("run_as_of_ts")
    return str(run_id) if run_id else None


def _load_minutes_projection(
    root: Path, game_date: pd.Timestamp, *, run_id: Optional[str], minutes_source: str
) -> tuple[pd.DataFrame, Optional[str], Path, str]:
    date_token = pd.Timestamp(game_date).date().isoformat()
    if minutes_source != "minutes_v1":
        raise ValueError(f"Unsupported minutes_source={minutes_source}")

    daily_base = root / "artifacts" / "minutes_v1" / "daily" / date_token
    resolved_run = run_id or _read_latest_run_id(daily_base)

    candidates: list[tuple[Path, Optional[str], str]] = []
    if resolved_run:
        candidates.append(
            (daily_base / f"run={resolved_run}" / "minutes.parquet", resolved_run, "minutes_v1_daily")
        )
    gold_path = root / "gold" / "projections_minutes_v1" / f"game_date={date_token}" / "minutes.parquet"
    candidates.append((gold_path, resolved_run, "projections_minutes_v1"))

    project_root = get_project_root()
    if project_root != root:
        daily_base_project = project_root / "artifacts" / "minutes_v1" / "daily" / date_token
        resolved_project_run = run_id or _read_latest_run_id(daily_base_project)
        if resolved_project_run:
            candidates.append(
                (
                    daily_base_project / f"run={resolved_project_run}" / "minutes.parquet",
                    resolved_project_run,
                    "minutes_v1_daily_project",
                )
            )
        gold_project = (
            project_root / "gold" / "projections_minutes_v1" / f"game_date={date_token}" / "minutes.parquet"
        )
        candidates.append((gold_project, resolved_project_run, "projections_minutes_v1_project"))

    for path, rid, label in candidates:
        if path.exists():
            df = pd.read_parquet(path)
            return df, rid, path, label
    raise FileNotFoundError(f"No minutes_v1 projection found for {date_token} (source={minutes_source}).")


def _load_schedule_for_date(root: Path, game_date: pd.Timestamp) -> pd.DataFrame:
    season = int(game_date.year) if game_date.month >= 8 else int(game_date.year - 1)
    month = int(game_date.month)
    schedule_path = root / "silver" / "schedule" / f"season={season}" / f"month={month:02d}" / "schedule.parquet"
    if not schedule_path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(schedule_path)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.date
    day = pd.Timestamp(game_date).date()
    return df.loc[df["game_date"] == day].copy()


def _load_rates_live_frame(
    root: Path, game_date: pd.Timestamp, *, run_id: Optional[str]
) -> tuple[pd.DataFrame, Optional[str], Path]:
    date_token = pd.Timestamp(game_date).date().isoformat()
    base = root / "gold" / "rates_v1_live" / date_token
    resolved_run = run_id or _read_latest_run_id(base)
    candidate = base / "rates.parquet"
    if resolved_run:
        candidate = base / f"run={resolved_run}" / "rates.parquet"
    if not candidate.exists():
        raise FileNotFoundError(f"No rates_v1_live parquet found at {candidate}")
    df = pd.read_parquet(candidate)
    if "game_date" not in df.columns:
        df["game_date"] = pd.to_datetime(date_token)
    return df, resolved_run, candidate


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


def _compute_fpts_and_boxscore(
    stats: dict[str, np.ndarray],
    efficiency_pct: dict[str, np.ndarray] | None = None,
    use_efficiency: bool = False,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Compute DK FPTS and derived box score totals from simulated stat totals.

    If use_efficiency is True and efficiency_pct contains fg2_pct/fg3_pct/ft_pct,
    makes are derived from attempts * pct. Otherwise, attempts are treated as makes
    with FT at 0.75x.
    Accepts 1D arrays (players) or 2D arrays (worlds, players); output mirrors input shape.
    """

    if not stats:
        return np.array([]), {}
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

    eff = efficiency_pct or {}
    eff_ready = use_efficiency and all(k in eff for k in ("fg2_pct", "fg3_pct", "ft_pct"))
    if eff_ready:
        fg2_pct = np.clip(eff.get("fg2_pct", zeros), 0.3, 0.75)
        fg3_pct = np.clip(eff.get("fg3_pct", zeros), 0.2, 0.55)
        ft_pct = np.clip(eff.get("ft_pct", zeros), 0.5, 0.95)
        fgm2 = fga2 * fg2_pct
        fgm3 = fga3 * fg3_pct
        ftm = fta * ft_pct
    else:
        fgm2 = fga2
        fgm3 = fga3
        ftm = 0.75 * fta

    pts = 2.0 * fgm2 + 3.0 * fgm3 + ftm

    fgm = fgm2 + fgm3
    fga = fga2 + fga3
    fg3m = fgm3
    fg3a = fga3

    shaped_like = sample.shape

    def flat(arr: np.ndarray) -> np.ndarray:
        return arr.reshape(-1)

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
    fpts = fpts_flat.reshape(shaped_like)
    stat_box = {
        "pts": pts,
        "reb": reb,
        "oreb": oreb,
        "dreb": dreb,
        "ast": ast,
        "stl": stl,
        "blk": blk,
        "tov": tov,
        "fga2": fga2,
        "fga3": fga3,
        "fta": fta,
    }
    return fpts, stat_box


def _compute_fpts_from_stats(stats: dict[str, np.ndarray]) -> np.ndarray:
    fpts, _ = _compute_fpts_and_boxscore(stats)
    return fpts


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


def build_rates_mean_fpts(minutes_df: pd.DataFrame, rates_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join minutes_v1 and rates_v1 predictions and compute mean DK FPTS per player.

    Returns a DataFrame keyed by (game_date, game_id, team_id, player_id) with:
      - minutes_mean
      - fpts_mean
      - optional passthrough columns (minutes_p50_cond, minutes_p50, play_prob, is_starter)
    """

    minutes_df = minutes_df.copy()
    rates_df = rates_df.copy()

    if "minutes_mean" not in minutes_df.columns:
        minutes_col = _resolve_minutes_column(minutes_df)
        minutes_df["minutes_mean"] = pd.to_numeric(minutes_df[minutes_col], errors="coerce")
    else:
        minutes_df["minutes_mean"] = pd.to_numeric(minutes_df["minutes_mean"], errors="coerce")

    join_keys = ["game_date", "game_id", "team_id", "player_id"]
    missing_keys = [k for k in join_keys if k not in minutes_df.columns or k not in rates_df.columns]
    if missing_keys:
        raise KeyError(f"Missing join keys for rates->minutes join: {missing_keys}")

    minutes_df["game_date"] = pd.to_datetime(minutes_df["game_date"]).dt.normalize()
    rates_df["game_date"] = pd.to_datetime(rates_df["game_date"]).dt.normalize()
    for key in ("game_id", "team_id", "player_id"):
        if key in minutes_df.columns:
            minutes_df[key] = pd.to_numeric(minutes_df[key], errors="coerce")
        if key in rates_df.columns:
            rates_df[key] = pd.to_numeric(rates_df[key], errors="coerce")

    merged = pd.merge(minutes_df, rates_df, on=join_keys, how="inner", suffixes=("", "_rates"))
    merged = merged[merged["minutes_mean"].notna()]
    if merged.empty:
        return merged.assign(fpts_mean=pd.Series(dtype=float))

    stat_targets = [
        "fga2_per_min",
        "fga3_per_min",
        "fta_per_min",
        "ast_per_min",
        "tov_per_min",
        "oreb_per_min",
        "dreb_per_min",
        "stl_per_min",
        "blk_per_min",
    ]
    efficiency_targets = ["fg2_pct", "fg3_pct", "ft_pct"]
    mapping = _resolve_rate_columns(merged, stat_targets)
    missing_targets = [t for t in stat_targets if t not in mapping]
    if missing_targets:
        raise KeyError(f"Missing rate columns for targets={missing_targets}")
    eff_mapping = _resolve_rate_columns(merged, efficiency_targets)
    use_efficiency = len(eff_mapping) == len(efficiency_targets)
    if not use_efficiency:
        typer.echo("[sim_v2] warning: missing fg% preds; falling back to attempts==makes for mean_fpts", err=True)

    minutes_mean = merged["minutes_mean"].to_numpy(dtype=float)
    stat_totals: dict[str, np.ndarray] = {}
    for target, col in mapping.items():
        base = target.replace("_per_min", "")
        rates_arr = pd.to_numeric(merged[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        stat_totals[base] = np.clip(minutes_mean * rates_arr, 0.0, None)

    eff_arrays: dict[str, np.ndarray] | None = None
    if use_efficiency:
        eff_arrays = {}
        eff_clamp = {"fg2_pct": (0.3, 0.75), "fg3_pct": (0.2, 0.55), "ft_pct": (0.5, 0.95)}
        for target, col in eff_mapping.items():
            lo, hi = eff_clamp[target]
            vals = pd.to_numeric(merged[col], errors="coerce").to_numpy(dtype=float)
            eff_arrays[target] = np.clip(vals, lo, hi)

    fpts_mean, base_stat_box = _compute_fpts_and_boxscore(stat_totals, eff_arrays, use_efficiency=use_efficiency)
    merged["fpts_mean"] = fpts_mean

    base_cols = ["game_date", "game_id", "team_id", "player_id", "minutes_mean", "fpts_mean"]
    if base_stat_box:
        for name, values in base_stat_box.items():
            merged[f"{name}_mean"] = values
    for extra in ("minutes_p50_cond", "minutes_p50", "play_prob", "is_starter"):
        if extra in merged.columns:
            base_cols.append(extra)
    # Passthrough vacancy features for learned usage shares model
    for vac_col in [
        "vac_min_szn", "vac_fga_szn", "vac_min_guard_szn",
        "vac_min_wing_szn", "vac_min_big_szn",
    ]:
        if vac_col in merged.columns:
            base_cols.append(vac_col)
    # Passthrough other features needed by learned model
    for feat in ["pos_bucket", "position_primary", "spread_home", "total", "odds_as_of_ts", "tip_ts"]:
        if feat in merged.columns and feat not in base_cols:
            base_cols.append(feat)
    return merged[base_cols]


def draw_independent_noise(
    mu: np.ndarray,
    n_worlds: int,
    *,
    nu: float,
    k: float,
    rng: np.random.Generator,
    epsilon_dist: str = "student_t",
    sigma_mode: str = "k_times_mu",
) -> np.ndarray:
    """
    Draw independent noise per player and world.

    sigma_mode currently supports k_times_mu: sigma_i = k * mu_i.
    """

    mu_arr = np.asarray(mu, dtype=float).reshape(-1)
    sigma = np.clip(mu_arr, 0.0, None)
    if sigma_mode == "k_times_mu":
        sigma = sigma * k
    else:
        sigma = sigma * k

    if epsilon_dist == "normal":
        eps = rng.standard_normal(size=(mu_arr.shape[0], n_worlds))
    else:
        df = nu if nu is not None else 5.0
        eps = rng.standard_t(df=df, size=(mu_arr.shape[0], n_worlds))
    return eps * sigma[:, None]


def sample_and_apply_game_scripts(
    minutes_worlds: np.ndarray,
    game_ids: np.ndarray,
    team_ids: np.ndarray,
    is_starter: np.ndarray,
    spreads_home: np.ndarray,
    home_team_ids: dict[int, int],  # game_id -> home_team_id
    config: GameScriptConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample game scripts per world and apply minutes adjustments.
    
    Args:
        minutes_worlds: shape (n_worlds, n_players)
        game_ids: shape (n_players,)
        team_ids: shape (n_players,)  
        is_starter: shape (n_players,)
        spreads_home: shape (n_players,) - home team's spread (negative = home favored)
        home_team_ids: mapping game_id -> home_team_id to determine home/away
        config: GameScriptConfig
        rng: random generator
        
    Returns:
        adjusted minutes_worlds
    """
    n_worlds, n_players = minutes_worlds.shape
    
    # Get unique games and compute team-perspective spread
    unique_games = {}  # (game_id, team_id) -> team_spread
    for i in range(n_players):
        gid = int(game_ids[i])
        tid = int(team_ids[i])
        spread_home = spreads_home[i]
        
        if pd.isna(spread_home):
            continue
        
        # Determine if this team is home or away
        home_tid = home_team_ids.get(gid)
        is_home = (tid == home_tid) if home_tid is not None else True
        
        # Convert spread to team's perspective
        # spread_home < 0 means home team is favored
        # Team's spread: home uses as-is, away flips sign
        team_spread = spread_home if is_home else -spread_home
        
        key = (gid, tid)
        if key not in unique_games:
            unique_games[key] = team_spread
    
    if not unique_games:
        return minutes_worlds
    
    # Sample margins for each game-team
    sampled_scripts = {}  # (game_id, team_id, world_id) -> script
    for (gid, tid), team_spread in unique_games.items():
        mean_margin = config.spread_coef * team_spread
        margins = rng.normal(mean_margin, config.margin_std, size=n_worlds)
        for w in range(n_worlds):
            script = classify_script(margins[w], config)
            sampled_scripts[(gid, tid, w)] = script
    
    # Apply adjustments
    adjusted = minutes_worlds.copy()
    for w in range(n_worlds):
        for i in range(n_players):
            gid = int(game_ids[i])
            tid = int(team_ids[i])
            key = (gid, tid, w)
            
            script = sampled_scripts.get(key, "close")
            starter = is_starter[i]
            
            if script in config.adjustments:
                starter_adj, bench_adj = config.adjustments[script]
                mult = starter_adj if starter else bench_adj
                adjusted[w, i] *= mult
    
    return adjusted


@app.command()
def main(
    start_date: str = typer.Option(..., "--start-date"),
    end_date: str = typer.Option(..., "--end-date"),
    n_worlds: Optional[int] = typer.Option(
        None, "--n-worlds", help="Number of worlds to generate (default from profile or 2000)."
    ),
    profile: str = typer.Option(
        "baseline",
        "--profile",
        "--profile-name",
        help="Name of sim_v2 profile to load.",
    ),
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
    team_sigma_scale: Optional[float] = typer.Option(
        None,
        "--team-sigma-scale",
        help="Override team sigma scale for rates noise (otherwise profile).",
    ),
    player_sigma_scale: Optional[float] = typer.Option(
        None,
        "--player-sigma-scale",
        help="Override player sigma scale for rates noise (otherwise profile).",
    ),
    rates_run_id: Optional[str] = typer.Option(
        None,
        "--rates-run-id",
        help="Override rates run id for noise lookup (otherwise profile).",
    ),
    minutes_run_id: Optional[str] = typer.Option(
        None,
        "--minutes-run-id",
        help="Override minutes run id for minutes lookup (otherwise profile).",
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
    use_efficiency_scoring: Optional[bool] = typer.Option(
        None,
        "--use-efficiency-scoring/--no-efficiency-scoring",
        help="Toggle efficiency-based scoring (fg% heads). Defaults to profile setting.",
    ),
    export_attempt_means: bool = typer.Option(
        False,
        "--export-attempt-means",
        help="Export fga2_mean, fga3_mean, fta_mean in projections for diagnostics.",
    ),
) -> None:
    profile_cfg = load_sim_v2_profile(profile=profile, profiles_path=profiles_path)

    def _resolve(value, override, label):
        if override is not None and override != value:
            typer.echo(f"[sim_v2] override {label}: profile={value} -> cli={override}")
            return override
        return value

    mean_source = getattr(profile_cfg, "mean_source", "fpts")
    minutes_source = profile_cfg.minutes_source or "minutes_v1"
    rates_source = profile_cfg.rates_source or "rates_v1_live"

    fpts_run = _resolve(profile_cfg.fpts_run_id, fpts_run_id, "fpts_run_id")
    use_rates_noise_eff = profile_cfg.use_rates_noise if use_rates_noise is None else use_rates_noise
    resolved_rates_run = _resolve(profile_cfg.rates_run_id, rates_run_id, "rates_run_id")
    # For noise, prefer rates_noise_run_id if specified (allows using older residuals with newer model)
    rates_noise_run_id_eff = getattr(profile_cfg, "rates_noise_run_id", None) or resolved_rates_run
    rates_run = rates_noise_run_id_eff if use_rates_noise_eff else None
    rates_split = _resolve(profile_cfg.rates_noise_split, rates_noise_split, "rates_noise_split") if use_rates_noise_eff else None
    rates_sigma_scale = float(getattr(profile_cfg, "rates_sigma_scale", 1.0))
    team_sigma_scale_eff = _resolve(getattr(profile_cfg, "team_sigma_scale", 1.0), team_sigma_scale, "team_sigma_scale")
    player_sigma_scale_eff = _resolve(
        getattr(profile_cfg, "player_sigma_scale", 1.0), player_sigma_scale, "player_sigma_scale"
    )
    use_minutes_noise_eff = profile_cfg.use_minutes_noise if use_minutes_noise is None else use_minutes_noise
    resolved_minutes_run = _resolve(profile_cfg.minutes_run_id, minutes_run_id, "minutes_run_id")
    minutes_run = resolved_minutes_run if use_minutes_noise_eff else None
    if use_minutes_noise_eff and minutes_noise_run_id is not None and minutes_noise_run_id != minutes_run:
        typer.echo(f"[sim_v2] override minutes_noise_run_id: profile={minutes_run} -> cli={minutes_noise_run_id}")
        minutes_run = minutes_noise_run_id
    minutes_sigma_min_eff = (
        _resolve(profile_cfg.minutes_sigma_min, minutes_sigma_min, "minutes_sigma_min")
        if use_minutes_noise_eff
        else profile_cfg.minutes_sigma_min
    )
    seed_eff = seed if seed is not None else (profile_cfg.seed or 1234)
    min_play_prob_eff = min_play_prob if min_play_prob is not None else profile_cfg.min_play_prob
    team_factor_sigma_eff = team_factor_sigma if team_factor_sigma is not None else profile_cfg.team_factor_sigma
    team_factor_gamma_eff = team_factor_gamma if team_factor_gamma is not None else profile_cfg.team_factor_gamma
    worlds_per_chunk = max(1, (profile_cfg.worlds_batch_size or profile_cfg.worlds_per_chunk))
    n_worlds_eff = int(n_worlds) if n_worlds is not None else int(profile_cfg.worlds_n or 2000)
    enforce_team_240 = profile_cfg.enforce_team_240
    use_efficiency_scoring_eff = (
        profile_cfg.use_efficiency_scoring if use_efficiency_scoring is None else use_efficiency_scoring
    )

    root = data_root or data_path()
    start_dt = _parse_date(start_date)
    end_dt = _parse_date(end_date)
    start_ts = pd.Timestamp(start_dt).normalize()
    end_ts = pd.Timestamp(end_dt).normalize()

    typer.echo(
        f"[sim_v2] profile={profile_cfg.name} mean_source={mean_source} config={profiles_path or DEFAULT_PROFILES_PATH} "
        f"worlds={n_worlds_eff} chunk={worlds_per_chunk} seed={seed_eff} efficiency={use_efficiency_scoring_eff}"
    )
    typer.echo(
        f"[sim_v2] fpts_run_id={fpts_run} rates_run_id={resolved_rates_run} minutes_run_id={resolved_minutes_run} "
        f"use_rates_noise={use_rates_noise_eff} split={rates_split} "
        f"use_minutes_noise={use_minutes_noise_eff} sigma_min={minutes_sigma_min_eff} min_play_prob={min_play_prob_eff}"
    )

    if mean_source == "rates":
        noise_cfg = profile_cfg.noise or {}
        nu = float(noise_cfg.get("nu", 5))
        k_default = float(noise_cfg.get("k_default", 0.35))
        epsilon_dist = str(noise_cfg.get("epsilon_dist", "student_t"))
        if rates_source != "rates_v1_live":
            raise ValueError(f"Unsupported rates_source for rates mean: {rates_source}")
        output_base = output_root or (root / "artifacts" / "sim_v2" / "worlds_fpts_v2")

        # Optional: rates residual noise params (team/player shocks) for rates mode.
        # When enabled and available, we use these calibrated sigmas instead of the heuristic k_default noise.
        rates_noise_params = None
        rates_noise_path = None
        if use_rates_noise_eff:
            try:
                rates_noise_params, rates_noise_path = load_rates_noise_params(
                    data_root=root,
                    run_id=rates_run,
                    split=rates_split or "val",
                    sigma_scale=rates_sigma_scale,
                )
                typer.echo(
                    f"[sim_v2] rates_noise enabled (rates path): run_id={rates_run or 'current'} split={rates_split or 'val'} "
                    f"sigma_scale={rates_sigma_scale:.3f} team_sigma_scale={float(team_sigma_scale_eff):.3f} "
                    f"player_sigma_scale={float(player_sigma_scale_eff):.3f} targets={len(rates_noise_params)} path={rates_noise_path}"
                )
            except FileNotFoundError as exc:
                typer.echo(
                    f"[sim_v2] warning: rates noise params not found; falling back to heuristic noise ({exc})",
                    err=True,
                )
                rates_noise_params = None
                rates_noise_path = None

        # Game script config
        use_game_scripts = profile_cfg.use_game_scripts
        game_script_config = None
        if use_game_scripts:
            game_script_config = GameScriptConfig(
                margin_std=profile_cfg.game_script_margin_std,
                spread_coef=profile_cfg.game_script_spread_coef,
                quantile_noise_std=profile_cfg.game_script_quantile_noise_std,
                quantile_targets=profile_cfg.game_script_quantile_targets,
            )
            typer.echo(f"[sim_v2] game_scripts enabled: margin_std={game_script_config.margin_std} spread_coef={game_script_config.spread_coef}")
        minutes_noise_params = None
        if use_minutes_noise_eff:
            try:
                minutes_noise_params = load_minutes_noise_params(data_root=root, minutes_run_id=minutes_run)
                typer.echo(
                    f"[sim_v2] using minutes noise run_id={minutes_noise_params.run_id} "
                    f"sigma_min={minutes_sigma_min_eff:.3f} path={minutes_noise_params.source_path}"
                )
            except FileNotFoundError as exc:
                typer.echo(f"[sim_v2] warning: minutes noise params not found; disabling minutes noise ({exc})", err=True)
                minutes_noise_params = None

        typer.echo(
            f"[sim_v2] rates mean: minutes_source={minutes_source} rates_source={rates_source} "
            f"rates_run_id={resolved_rates_run or 'latest'} minutes_run_id={resolved_minutes_run or 'latest'} "
            f"noise k={k_default} nu={nu} dist={epsilon_dist}"
        )
        for game_date in pd.date_range(start_ts, end_ts, freq="D"):
            try:
                minutes_df, minutes_run_eff, minutes_path, minutes_label = _load_minutes_projection(
                    root, game_date, run_id=resolved_minutes_run, minutes_source=minutes_source
                )
            except FileNotFoundError:
                typer.echo(f"[sim_v2] {pd.Timestamp(game_date).date()} missing minutes ({minutes_source}); skipping.")
                continue

            minutes_df = minutes_df.copy()
            typer.echo(
                f"[sim_v2] {pd.Timestamp(game_date).date()} minutes source={minutes_label} "
                f"run={minutes_run_eff or 'latest'} path={minutes_path}"
            )
            minutes_df["game_date"] = pd.to_datetime(minutes_df["game_date"]).dt.normalize()
            try:
                minutes_col = _resolve_minutes_column(minutes_df)
            except KeyError:
                typer.echo(f"[sim_v2] {pd.Timestamp(game_date).date()} missing minutes columns; skipping.")
                continue
            minutes_df[minutes_col] = pd.to_numeric(minutes_df[minutes_col], errors="coerce")
            minutes_df["is_starter"] = pd.to_numeric(
                minutes_df.get("is_projected_starter", minutes_df.get("starter_flag")), errors="coerce"
            )
            minutes_df["play_prob"] = pd.to_numeric(minutes_df.get("play_prob"), errors="coerce").fillna(1.0)
            minutes_df = _ensure_status_bucket(minutes_df)
            minutes_df = minutes_df[minutes_df[minutes_col].notna()]
            
            # Compute vacancy features BEFORE filtering by min_play_prob
            # Only compute if usage_shares enabled (where vacancy is used by learned model)
            if profile_cfg.usage_shares.enabled and profile_cfg.vacancy_mode != "none":
                minutes_df = _add_vacancy_features_from_minutes_df(
                    minutes_df,
                    group_cols=("game_id", "team_id"),
                    vacancy_mode=profile_cfg.vacancy_mode,
                )
            
            # Now filter by min_play_prob (removes players unlikely to play)
            minutes_df = minutes_df[minutes_df["play_prob"].fillna(0.0) >= min_play_prob_eff]
            if minutes_df.empty:
                continue

            minutes_mean_arr = minutes_df[minutes_col].to_numpy(dtype=float)
            minutes_df["minutes_mean"] = minutes_mean_arr

            rates_df = None
            try:
                rates_df, rates_run_eff, rates_path = _load_rates_live_frame(
                    root, game_date, run_id=resolved_rates_run if rates_source == "rates_v1_live" else resolved_rates_run
                )
            except FileNotFoundError:
                typer.echo(f"[sim_v2] {pd.Timestamp(game_date).date()} missing rates ({rates_source}); skipping.")
                rates_df = None
            if rates_df is None:
                continue
            typer.echo(
                f"[sim_v2] {pd.Timestamp(game_date).date()} rates run={rates_run_eff or 'latest'} path={rates_path}"
            )
            rates_df["game_date"] = pd.to_datetime(game_date).normalize()

            try:
                mu_df = build_rates_mean_fpts(minutes_df, rates_df)
            except KeyError as exc:
                typer.echo(f"[sim_v2] {pd.Timestamp(game_date).date()} {exc}; skipping.")
                continue
            if mu_df.empty:
                typer.echo(f"[sim_v2] {pd.Timestamp(game_date).date()} empty minutes/rates join; skipping.")
                continue

            mu_df = mu_df.reset_index(drop=True)
            mu_df["dk_fpts_mean"] = mu_df["fpts_mean"]
            mu_df["sim_profile"] = profile_cfg.name
            if "play_prob" in mu_df.columns:
                play_prob_arr = (
                    pd.to_numeric(mu_df["play_prob"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
                )
            else:
                play_prob_arr = np.ones(len(mu_df), dtype=float)

            sigma_minutes_mu: np.ndarray | None = None
            if use_minutes_noise_eff and minutes_noise_params is not None:
                try:
                    sigma_raw = build_sigma_per_player(
                        minutes_df,
                        minutes_noise_params,
                        minutes_col=minutes_col,
                        starter_col="is_starter",
                        status_col="status_bucket",
                    )
                    if minutes_sigma_min_eff is not None:
                        sigma_raw = np.maximum(sigma_raw, minutes_sigma_min_eff)
                    sigma_df = minutes_df[["game_date", "game_id", "team_id", "player_id"]].copy()
                    sigma_df["sigma_minutes"] = sigma_raw
                    for key in ("game_id", "team_id", "player_id"):
                        sigma_df[key] = pd.to_numeric(sigma_df[key], errors="coerce")
                        mu_df[key] = pd.to_numeric(mu_df[key], errors="coerce")
                    mu_df = mu_df.merge(sigma_df, on=["game_date", "game_id", "team_id", "player_id"], how="left")
                    sigma_minutes_mu = pd.to_numeric(mu_df["sigma_minutes"], errors="coerce").to_numpy(dtype=float)
                    sigma_fallback = float(minutes_sigma_min_eff or minutes_noise_params.sigma_min or 0.5)
                    sigma_minutes_mu = np.nan_to_num(sigma_minutes_mu, nan=sigma_fallback)
                except Exception as exc:
                    typer.echo(f"[sim_v2] warning: failed to build minutes noise sigma; disabling minutes noise ({exc})", err=True)
                    sigma_minutes_mu = None

            stat_targets = [
                "fga2_per_min",
                "fga3_per_min",
                "fta_per_min",
                "ast_per_min",
                "tov_per_min",
                "oreb_per_min",
                "dreb_per_min",
                "stl_per_min",
                "blk_per_min",
            ]
            efficiency_targets = ["fg2_pct", "fg3_pct", "ft_pct"]
            rates_mapping = _resolve_rate_columns(rates_df, stat_targets)
            if len(rates_mapping) < len(stat_targets):
                missing = [t for t in stat_targets if t not in rates_mapping]
                typer.echo(f"[sim_v2] warning: missing rate columns for {missing}; stats will be NaN.")
            else:
                rate_cols = [rates_mapping[t] for t in stat_targets]
                rates_slice = rates_df[["game_date", "game_id", "team_id", "player_id"] + rate_cols].copy()
                mu_df = mu_df.merge(
                    rates_slice, on=["game_date", "game_id", "team_id", "player_id"], how="left", suffixes=("", "_rates")
                )

            eff_mapping = _resolve_rate_columns(rates_df, efficiency_targets)
            use_efficiency = use_efficiency_scoring_eff and len(eff_mapping) == len(efficiency_targets)
            if use_efficiency:
                eff_cols = [eff_mapping[t] for t in efficiency_targets]
                eff_slice = rates_df[["game_date", "game_id", "team_id", "player_id"] + eff_cols].copy()
                mu_df = mu_df.merge(
                    eff_slice, on=["game_date", "game_id", "team_id", "player_id"], how="left", suffixes=("", "_eff")
                )
            elif use_efficiency_scoring_eff:
                typer.echo("[sim_v2] warning: missing fg% preds; falling back to attempts==makes for worlds.", err=True)

            mu_stats = mu_df["fpts_mean"]
            typer.echo(
                f"[sim_v2] {pd.Timestamp(game_date).date()} rows={len(mu_df)} "
                f"dk_fpts_mean (rates) min/med/max={mu_stats.min():.2f}/{mu_stats.median():.2f}/{mu_stats.max():.2f}"
            )

            out_dir = output_base / f"game_date={pd.Timestamp(game_date).date()}"
            out_dir.mkdir(parents=True, exist_ok=True)

            date_seed = seed_eff + int(pd.Timestamp(game_date).toordinal())
            mu_arr = mu_df["fpts_mean"].to_numpy(dtype=float)
            minutes_sim_base = mu_df["minutes_mean"].to_numpy(dtype=float)
            world_fpts_samples: list[np.ndarray] = []
            minutes_world_samples: list[np.ndarray] = []
            base_cols = ["game_date", "game_id", "team_id", "player_id", "minutes_mean", "dk_fpts_mean", "sim_profile"]
            for extra in ("minutes_p50_cond", "minutes_p50", "play_prob", "is_starter"):
                if extra in mu_df.columns and extra not in base_cols:
                    base_cols.append(extra)
            base_cols = list(dict.fromkeys(base_cols))
            stat_defaults = np.full_like(minutes_sim_base, np.nan, dtype=float)

            # Minutes sampling inputs (for scripts and/or noise)
            gs_game_ids = mu_df["game_id"].to_numpy()
            gs_team_ids = mu_df["team_id"].to_numpy()
            if "is_starter" in mu_df.columns:
                gs_is_starter = (
                    pd.to_numeric(mu_df["is_starter"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                )
            else:
                gs_is_starter = np.zeros(len(mu_df), dtype=float)
            gs_minutes_p50 = minutes_sim_base.copy()

            if "minutes_p10" in minutes_df.columns and "minutes_p90" in minutes_df.columns:
                p10_map = minutes_df.groupby("player_id")["minutes_p10"].first().to_dict()
                p90_map = minutes_df.groupby("player_id")["minutes_p90"].first().to_dict()
                p10_raw = mu_df["player_id"].map(p10_map).to_numpy(dtype=float)
                p90_raw = mu_df["player_id"].map(p90_map).to_numpy(dtype=float)
                gs_minutes_p10 = np.where(np.isnan(p10_raw), gs_minutes_p50 * 0.7, p10_raw)
                gs_minutes_p90 = np.where(np.isnan(p90_raw), gs_minutes_p50 * 1.3, p90_raw)
            else:
                gs_minutes_p10 = gs_minutes_p50 * 0.7
                gs_minutes_p90 = gs_minutes_p50 * 1.3

            if sigma_minutes_mu is not None:
                z90 = 1.2815515655446004
                sigma = np.maximum(sigma_minutes_mu, 0.1)
                gs_minutes_p10 = np.maximum(gs_minutes_p50 - z90 * sigma, 0.0)
                gs_minutes_p90 = np.maximum(gs_minutes_p50 + z90 * sigma, gs_minutes_p10 + 0.01)

            # Spread is optional: if missing, still sample minutes (noise-only scripts).
            gs_spreads_home = np.full(len(mu_df), np.nan, dtype=float)
            spread_col = "spread_home"
            if spread_col in minutes_df.columns:
                spread_map = minutes_df.groupby("game_id")[spread_col].first().to_dict()
                gs_spreads_home = mu_df["game_id"].map(spread_map).to_numpy(dtype=float)

            # Build home_team_ids mapping (best-effort).
            gs_home_team_ids: dict[int, int] = {}
            try:
                sched_path = (
                    root
                    / "silver"
                    / "schedule"
                    / "season=2025"
                    / f"month={pd.Timestamp(game_date).month:02d}"
                    / "schedule.parquet"
                )
                if sched_path.exists():
                    sched = pd.read_parquet(sched_path)
                    date_sched = sched[sched["game_date"] == pd.Timestamp(game_date).normalize()]
                    gs_home_team_ids = dict(zip(date_sched["game_id"], date_sched["home_team_id"]))
            except Exception:
                pass  # Default to treating all as home if schedule unavailable

            team_codes = mu_df["team_id"].astype("category")
            team_indices = team_codes.cat.codes.to_numpy(dtype=int)
            n_teams = int(team_indices.max()) + 1 if len(team_indices) else 0
            if "rotation_prob" in mu_df.columns:
                rot_prob_arr = (
                    pd.to_numeric(mu_df["rotation_prob"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                )
                rotation_mask = (rot_prob_arr >= 0.5) | (gs_is_starter > 0)
                bench_mask = (~rotation_mask) & (gs_minutes_p50 > 0.0)
            else:
                rotation_mask = gs_minutes_p50 >= 12.0
                bench_mask = (~rotation_mask) & (gs_minutes_p50 > 0.0)
            if use_game_scripts and game_script_config is not None:
                typer.echo(f"[sim_v2] game_scripts: {len(gs_home_team_ids)} games, sampling minutes via scripts")

            # Usage shares config
            usage_shares_cfg = profile_cfg.usage_shares
            usage_shares_bundle = None
            use_learned_fga = False
            if usage_shares_cfg.enabled:
                typer.echo(
                    f"[sim_v2] usage_shares enabled: targets={usage_shares_cfg.targets} "
                    f"backend={usage_shares_cfg.backend} noise_std={usage_shares_cfg.share_noise_std} "
                    f"temp={usage_shares_cfg.share_temperature}"
                )
                # Load learned model bundle if backend is lgbm_residual
                if usage_shares_cfg.backend == "lgbm_residual" and "fga" in usage_shares_cfg.targets:
                    usage_shares_bundle, use_learned_fga = _load_usage_shares_bundle(root, usage_shares_cfg)
                    if use_learned_fga:
                        typer.echo(
                            f"[sim_v2] usage_shares: loaded LGBM residual bundle "
                            f"(run_id={usage_shares_bundle.run_id if usage_shares_bundle else 'N/A'}, "
                            f"shrink={usage_shares_cfg.shrink or 0.75})"
                        )
                        # Log vacancy stats for debugging
                        if "vac_min_szn" in minutes_df.columns:
                            vac_per_team = minutes_df.groupby(["game_id", "team_id"])["vac_min_szn"].first()
                            vac_p50 = vac_per_team.median()
                            vac_p90 = vac_per_team.quantile(0.9)
                            vac_max = vac_per_team.max()
                            n_high = (vac_per_team > 20).sum()
                            
                            # Check for FGA clipping (caps are applied in _add_vacancy)
                            fga_per_team = minutes_df.groupby(["game_id", "team_id"])["vac_fga_szn"].first() if "vac_fga_szn" in minutes_df.columns else pd.Series([0])
                            teams_clipped_fga = (fga_per_team >= VAC_FGA_CAP).sum()
                            fga_max = fga_per_team.max()
                            
                            clip_info = f" fga_max={fga_max:.1f} teams_clipped_fga={teams_clipped_fga}" if teams_clipped_fga > 0 else ""
                            typer.echo(
                                f"[sim_v2] vacancy_mode={profile_cfg.vacancy_mode} "
                                f"vac_min_szn p50={vac_p50:.1f} p90={vac_p90:.1f} max={vac_max:.1f} "
                                f"teams_with_vac>20={n_high}{clip_info}"
                            )
                    else:
                        typer.echo(
                            "[sim_v2] usage_shares: could not load LGBM bundle, "
                            f"falling back to {usage_shares_cfg.fallback}",
                            err=True,
                        )

            # Precompute team group indices for team-level residual shocks.
            group_map: dict[tuple[int, int], np.ndarray] = {}
            for idx, key in enumerate(zip(gs_game_ids, gs_team_ids)):
                group_map.setdefault((int(key[0]), int(key[1])), []).append(idx)
            group_map = {k: np.array(v, dtype=int) for k, v in group_map.items()}

            # Vegas implied team points for optional anchoring.
            schedule_df = _load_schedule_for_date(root, pd.Timestamp(game_date))
            implied_team_points = _build_implied_team_points(minutes_df, schedule_df)

            missing_noise_targets: list[str] = []
            if rates_noise_params is not None:
                missing_noise_targets = [t for t in stat_targets if t not in rates_noise_params]
                if missing_noise_targets:
                    typer.echo(
                        f"[sim_v2] warning: missing rates_noise targets for {missing_noise_targets}; "
                        f"those stats will fall back to heuristic noise",
                        err=True,
                    )

            eff_arrays: dict[str, np.ndarray] | None = None
            if use_efficiency:
                eff_arrays = {}
                eff_clamp = {"fg2_pct": (0.3, 0.75), "fg3_pct": (0.2, 0.55), "ft_pct": (0.5, 0.95)}
                for target in efficiency_targets:
                    col = eff_mapping.get(target)
                    if not col or col not in mu_df.columns:
                        eff_arrays = None
                        break
                    lo, hi = eff_clamp[target]
                    vals = pd.to_numeric(mu_df[col], errors="coerce").to_numpy(dtype=float)
                    eff_arrays[target] = np.clip(vals, lo, hi)
                if eff_arrays is None:
                    typer.echo("[sim_v2] warning: partial fg% preds missing; disabling efficiency scoring for this date.", err=True)
                    use_efficiency = False

            # Build rate_arrays for usage shares (per-minute rates for each player)
            usage_rate_arrays: dict[str, np.ndarray] = {}
            if usage_shares_cfg.enabled:
                for target in stat_targets:
                    col = rates_mapping.get(target)
                    if col and col in mu_df.columns:
                        usage_rate_arrays[target] = (
                            pd.to_numeric(mu_df[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                        )

            stat_world_samples: dict[str, list[np.ndarray]] = {}
            active_mask_samples: list[np.ndarray] = []  # Track active masks for conditional aggregation
            for chunk_start in range(0, n_worlds_eff, worlds_per_chunk):
                chunk_size = min(worlds_per_chunk, n_worlds_eff - chunk_start)
                rng = np.random.default_rng(date_seed + chunk_start)

                # 1. Sample availability FIRST (before minutes)
                u_active = rng.random(size=(chunk_size, len(play_prob_arr)))
                active_mask = u_active < play_prob_arr[None, :]
                active_mask_samples.append(active_mask)

                # 2. Sample minutes based on game script (script determines quantile position)
                if use_game_scripts and game_script_config is not None:
                    minutes_worlds = sample_minutes_with_scripts(
                        minutes_p10=gs_minutes_p10,
                        minutes_p50=gs_minutes_p50,
                        minutes_p90=gs_minutes_p90,
                        is_starter=gs_is_starter,
                        game_ids=gs_game_ids,
                        team_ids=gs_team_ids,
                        spreads_home=gs_spreads_home,
                        home_team_ids=gs_home_team_ids,
                        n_worlds=chunk_size,
                        config=game_script_config,
                        rng=rng,
                    )
                else:
                    # Fallback: sample minutes from per-player distribution.
                    z90 = 1.2815515655446004
                    p50 = gs_minutes_p50
                    p10 = np.minimum(gs_minutes_p10, p50)
                    p90 = np.maximum(gs_minutes_p90, p50)
                    sigma_low = np.maximum((p50 - p10) / z90, 0.5)
                    sigma_high = np.maximum((p90 - p50) / z90, 0.5)

                    z = rng.standard_normal(size=(chunk_size, len(gs_minutes_p50)))
                    sigma = np.where(z < 0.0, sigma_low[None, :], sigma_high[None, :])
                    minutes_worlds = np.maximum(p50[None, :] + z * sigma, 0.0)

                # 3. Zero out inactive players' minutes (before reconciliation)
                minutes_worlds = minutes_worlds * active_mask.astype(float)

                # 4. Reconcile to 240 per team per world, considering only active players
                if enforce_team_240 and n_teams > 0:
                    minutes_worlds = enforce_team_240_minutes(
                        minutes_world=minutes_worlds,
                        team_indices=team_indices,
                        rotation_mask=rotation_mask,
                        bench_mask=bench_mask,
                        baseline_minutes=gs_minutes_p50,
                        clamp_scale=(0.7, 1.3),
                        active_mask=active_mask,  # Pass active mask for proper redistribution
                        starter_mask=gs_is_starter > 0,
                        max_rotation_size=DEFAULT_MAX_ROTATION_SIZE,
                        play_prob=play_prob_arr,
                    )
                minutes_world_samples.append(minutes_worlds)

                stat_totals: dict[str, np.ndarray] = {}
                for target in stat_targets:
                    col = rates_mapping.get(target)
                    if not col or col not in mu_df.columns:
                        continue
                    base = target.replace("_per_min", "")
                    rates = pd.to_numeric(mu_df[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                    mu_stat = np.clip(rates[None, :] * minutes_worlds, 0.0, None)
                    if rates_noise_params is not None and target in rates_noise_params:
                        params = rates_noise_params.get(target, {})
                        sigma_team = float(params.get("sigma_team", 0.0) or 0.0) * float(team_sigma_scale_eff)
                        sigma_player = float(params.get("sigma_player", 0.0) or 0.0) * float(player_sigma_scale_eff)
                        team_shock = np.zeros_like(mu_stat)
                        if sigma_team > 0.0:
                            for _, idxs in group_map.items():
                                ts = rng.normal(loc=0.0, scale=sigma_team, size=chunk_size)
                                team_shock[:, idxs] = ts[:, None]
                        player_eps = rng.normal(loc=0.0, scale=sigma_player, size=mu_stat.shape) if sigma_player > 0.0 else 0.0
                        total = np.clip(mu_stat + team_shock + player_eps, 0.0, None)
                    else:
                        # Heuristic independent noise (legacy): relative scale to mean.
                        if epsilon_dist == "normal":
                            eps = rng.standard_normal(size=mu_stat.shape)
                        else:
                            eps = rng.standard_t(df=nu, size=mu_stat.shape)
                        eps = eps * (k_default * np.clip(mu_stat, 0.0, None))
                        total = np.clip(mu_stat + eps, 0.0, None)
                    stat_totals[base] = total

                # Apply usage shares allocation (redistributes FGA/FTA/TOV within teams)
                if usage_shares_cfg.enabled and stat_totals:
                    # If learned FGA backend is available, use it for FGA
                    if use_learned_fga and usage_shares_bundle is not None and "fga" in usage_shares_cfg.targets:
                        # Build team indices array
                        team_indices_arr = np.zeros(len(gs_team_ids), dtype=int)
                        team_to_idx = {}
                        for key, player_idxs in group_map.items():
                            if key not in team_to_idx:
                                team_to_idx[key] = len(team_to_idx)
                            team_indices_arr[player_idxs] = team_to_idx[key]
                        
                        # Apply learned FGA allocation
                        stat_totals = _apply_learned_fga_shares_allocation(
                            stat_totals=stat_totals,
                            player_df=mu_df,
                            team_indices=team_indices_arr,
                            active_mask=active_mask,
                            minutes_worlds=minutes_worlds,
                            usage_cfg=usage_shares_cfg,
                            bundle=usage_shares_bundle,
                            rng=rng,
                        )
                        
                        # Apply rate_weighted for non-FGA targets (FTA, TOV)
                        non_fga_targets = [t for t in usage_shares_cfg.targets if t != "fga"]
                        if non_fga_targets:
                            from copy import copy
                            rate_weighted_cfg = copy(usage_shares_cfg)
                            rate_weighted_cfg = UsageSharesConfig(
                                enabled=True,
                                targets=tuple(non_fga_targets),
                                backend="rate_weighted",
                                share_temperature=usage_shares_cfg.share_temperature,
                                share_noise_std=usage_shares_cfg.share_noise_std,
                                min_minutes_active_cutoff=usage_shares_cfg.min_minutes_active_cutoff,
                                fallback="rate_weighted",
                            )
                            stat_totals = _apply_usage_shares_allocation(
                                stat_totals=stat_totals,
                                minutes_worlds=minutes_worlds,
                                rate_arrays=usage_rate_arrays,
                                group_map=group_map,
                                usage_cfg=rate_weighted_cfg,
                                rng=rng,
                            )
                    else:
                        # Use rate_weighted for all targets
                        stat_totals = _apply_usage_shares_allocation(
                            stat_totals=stat_totals,
                            minutes_worlds=minutes_worlds,
                            rate_arrays=usage_rate_arrays,
                            group_map=group_map,
                            usage_cfg=usage_shares_cfg,
                            rng=rng,
                        )

                if not stat_totals:
                    fpts_chunk = mu_arr[:, None]  # fallback: no stat noise
                    stat_box = {}
                else:
                    fpts_chunk, stat_box = _compute_fpts_and_boxscore(
                        stat_totals, efficiency_pct=eff_arrays, use_efficiency=use_efficiency
                    )

                    # Optional vegas anchoring: keep team points within implied*(1±drift_pct).
                    if (
                        profile_cfg.vegas_points_anchor
                        and "pts" in stat_box
                        and implied_team_points
                        and np.isfinite(profile_cfg.vegas_points_drift_pct)
                    ):
                        pts_before = stat_box["pts"].copy()
                        stat_box["pts"] = _apply_team_points_vegas_anchor(
                            stat_box["pts"],
                            group_map=group_map,
                            implied_team_points=implied_team_points,
                            drift_pct=profile_cfg.vegas_points_drift_pct,
                        )
                        fpts_chunk = fpts_chunk + (stat_box["pts"] - pts_before)
                world_fpts_samples.append(fpts_chunk)
                # Track individual stat worlds for aggregation
                for stat_name in ("pts", "reb", "ast", "stl", "blk", "tov", "fga2", "fga3", "fta"):
                    if stat_name in stat_box:
                        stat_world_samples.setdefault(stat_name, []).append(stat_box[stat_name])

            # Aggregate all worlds in-memory and compute CONDITIONAL quantiles
            # (only count worlds where player is active)
            if world_fpts_samples:
                # fpts_chunk is shape (chunk_size, n_players), stack all chunks
                all_fpts = np.vstack(world_fpts_samples)  # shape: (n_worlds, n_players)
                all_minutes = np.vstack(minutes_world_samples) if minutes_world_samples else None
                all_active = np.vstack(active_mask_samples)  # shape: (n_worlds, n_players)

                # Compute CONDITIONAL statistics (only worlds where player is active)
                # This is what DFS lineup builders want: E[FPTS | plays]
                n_worlds_total, n_players = all_fpts.shape
                active_counts = all_active.sum(axis=0)  # worlds active per player

                # Conditional mean: sum over active worlds / count of active worlds
                fpts_sum = (all_fpts * all_active).sum(axis=0)
                fpts_mean = np.where(active_counts > 0, fpts_sum / active_counts, 0.0)

                # Conditional std: std over active worlds only
                fpts_sq_sum = ((all_fpts ** 2) * all_active).sum(axis=0)
                fpts_var = np.where(
                    active_counts > 1,
                    (fpts_sq_sum / active_counts) - (fpts_mean ** 2),
                    0.0
                )
                fpts_std = np.sqrt(np.maximum(fpts_var, 0.0))

                # Conditional quantiles: compute per-player over active worlds only
                quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
                fpts_quantiles = np.zeros((len(quantiles), n_players))
                for p in range(n_players):
                    active_worlds_p = all_active[:, p]
                    if active_worlds_p.sum() > 0:
                        fpts_active = all_fpts[active_worlds_p, p]
                        fpts_quantiles[:, p] = np.percentile(fpts_active, [q * 100 for q in quantiles])
                    else:
                        fpts_quantiles[:, p] = 0.0

                # Conditional minutes statistics
                if all_minutes is not None:
                    minutes_sum = (all_minutes * all_active).sum(axis=0)
                    minutes_mean = np.where(active_counts > 0, minutes_sum / active_counts, 0.0)
                    minutes_sq_sum = ((all_minutes ** 2) * all_active).sum(axis=0)
                    minutes_var = np.where(
                        active_counts > 1,
                        (minutes_sq_sum / active_counts) - (minutes_mean ** 2),
                        0.0
                    )
                    minutes_std = np.sqrt(np.maximum(minutes_var, 0.0))

                    # Conditional minutes quantiles
                    minutes_quantiles = np.zeros((3, n_players))
                    for p in range(n_players):
                        active_worlds_p = all_active[:, p]
                        if active_worlds_p.sum() > 0:
                            mins_active = all_minutes[active_worlds_p, p]
                            minutes_quantiles[:, p] = np.percentile(mins_active, [10, 50, 90])
                        else:
                            minutes_quantiles[:, p] = 0.0
                else:
                    minutes_mean = minutes_sim_base
                    minutes_std = np.zeros_like(minutes_sim_base)
                    minutes_quantiles = None
                
                # Build output projection DataFrame
                proj_df = mu_df[["game_date", "game_id", "team_id", "player_id"]].copy()
                proj_df["minutes_mean"] = minutes_sim_base
                proj_df["minutes_sim_mean"] = minutes_mean
                proj_df["minutes_sim_std"] = minutes_std
                if minutes_quantiles is not None:
                    proj_df["minutes_sim_p10"] = minutes_quantiles[0]
                    proj_df["minutes_sim_p50"] = minutes_quantiles[1]
                    proj_df["minutes_sim_p90"] = minutes_quantiles[2]
                proj_df["dk_fpts_mean"] = fpts_mean
                proj_df["dk_fpts_std"] = fpts_std
                proj_df["dk_fpts_p05"] = fpts_quantiles[0]
                proj_df["dk_fpts_p10"] = fpts_quantiles[1]
                proj_df["dk_fpts_p25"] = fpts_quantiles[2]
                proj_df["dk_fpts_p50"] = fpts_quantiles[3]
                proj_df["dk_fpts_p75"] = fpts_quantiles[4]
                proj_df["dk_fpts_p90"] = fpts_quantiles[5]
                proj_df["dk_fpts_p95"] = fpts_quantiles[6]
                proj_df["sim_profile"] = profile_cfg.name
                proj_df["n_worlds"] = n_worlds_eff
                
                # Add individual stat means for dashboard diagnostics (CONDITIONAL)
                for stat_name in ("pts", "reb", "ast", "stl", "blk", "tov"):
                    if stat_name in stat_world_samples and stat_world_samples[stat_name]:
                        all_stat = np.vstack(stat_world_samples[stat_name])
                        # Conditional mean: only count worlds where player is active
                        stat_sum = (all_stat * all_active).sum(axis=0)
                        stat_mean = np.where(active_counts > 0, stat_sum / active_counts, 0.0)
                        proj_df[f"{stat_name}_mean"] = stat_mean
                
                # Add optional columns
                for extra in ("is_starter", "play_prob"):
                    if extra in mu_df.columns:
                        proj_df[extra] = mu_df[extra]
                
                # Add attempt means for diagnostics (when --export-attempt-means is set)
                if export_attempt_means:
                    for stat_name in ("fga2", "fga3", "fta"):
                        if stat_name in stat_world_samples and stat_world_samples[stat_name]:
                            all_stat = np.vstack(stat_world_samples[stat_name])
                            stat_sum = (all_stat * all_active).sum(axis=0)
                            stat_mean = np.where(active_counts > 0, stat_sum / active_counts, 0.0)
                            proj_df[f"{stat_name}_mean"] = stat_mean
                    # Also add vacancy cols if present
                    for vac_col in ["vac_min_szn", "vac_fga_szn"]:
                        if vac_col in mu_df.columns:
                            proj_df[vac_col] = mu_df[vac_col]
                
                # Write single projections file
                proj_path = out_dir / "projections.parquet"
                proj_df.to_parquet(proj_path, index=False)
                
                typer.echo(
                    f"[sim_v2] {pd.Timestamp(game_date).date()} dk_fpts_world min/med/max="
                    f"{all_fpts.min():.2f}/{np.median(all_fpts):.2f}/{all_fpts.max():.2f} "
                    f"mean={all_fpts.mean():.2f} -> {proj_path}"
                )
        return

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
    efficiency_targets = ["fg2_pct", "fg3_pct", "ft_pct"]
    if use_rates_noise_eff:
        noise_params, noise_path = load_rates_noise_params(
            data_root=root,
            run_id=rates_run,
            split=rates_split or "val",
            sigma_scale=rates_sigma_scale,
        )
        stat_targets = list(noise_params.keys())
        typer.echo(
            f"[sim_v2] using rates noise run_id={rates_run or 'current'} split={rates_split or 'val'} "
            f"sigma_scale={rates_sigma_scale:.3f} team_sigma_scale={float(team_sigma_scale_eff):.3f} "
            f"player_sigma_scale={float(player_sigma_scale_eff):.3f} targets={len(stat_targets)} path={noise_path}"
        )

    df = _load_base(root, start_ts, end_ts)
    output_base = output_root or (root / "artifacts" / "sim_v2" / "worlds_fpts_v2")

    for game_date, date_df in df.groupby("game_date"):
        date_df = date_df.copy()
        date_df["sim_profile"] = profile_cfg.name
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

            eff_mapping = _resolve_rate_columns(date_df, efficiency_targets)
            use_efficiency = use_efficiency_scoring_eff and len(eff_mapping) == len(efficiency_targets)
            if use_efficiency_scoring_eff and not use_efficiency:
                typer.echo(
                    f"[sim_v2] warning: missing fg% preds for {game_date.date()}; falling back to attempts==makes.",
                    err=True,
                )

            minutes_center = date_df[minutes_col].to_numpy(dtype=float)
            minutes_pred_center = pd.to_numeric(
                date_df.get("minutes_pred_p50", date_df[minutes_col]), errors="coerce"
            ).to_numpy(dtype=float)
            role_low = pd.to_numeric(date_df.get("track_role_is_low_minutes", 0), errors="coerce").fillna(0).to_numpy(dtype=float)
            if "rotation_prob" in date_df.columns:
                rot_prob_arr = (
                    pd.to_numeric(date_df["rotation_prob"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                )
                rotation_mask = (rot_prob_arr >= 0.5) | (pd.to_numeric(date_df.get("is_starter", 0), errors="coerce").fillna(0).to_numpy(dtype=float) > 0)
                bench_mask = (~rotation_mask) & (minutes_pred_center > 0.0)
            else:
                rotation_mask = (minutes_pred_center >= 12.0) | (role_low == 0)
                bench_mask = (~rotation_mask) & (minutes_pred_center > 0.0)
            play_prob_arr = date_df["play_prob"].fillna(1.0).to_numpy(dtype=float)

            team_codes = date_df["team_id"].astype("category")
            team_indices = team_codes.cat.codes.to_numpy()
            n_teams = int(team_indices.max()) + 1 if len(team_indices) else 0
            team_one_hot = np.eye(n_teams, dtype=float)[team_indices] if n_teams > 0 else np.zeros((len(team_indices), 0))

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
            eff_arrays_base: dict[str, np.ndarray] | None = None
            if use_efficiency:
                eff_arrays_base = {}
                eff_clamp = {"fg2_pct": (0.3, 0.75), "fg3_pct": (0.2, 0.55), "ft_pct": (0.5, 0.95)}
                for target, col in eff_mapping.items():
                    lo, hi = eff_clamp[target]
                    vals = pd.to_numeric(date_df[col], errors="coerce").to_numpy(dtype=float)
                    eff_arrays_base[target] = np.clip(vals, lo, hi)

            dk_fpts_mean, base_stat_box = _compute_fpts_and_boxscore(
                base_mu, efficiency_pct=eff_arrays_base, use_efficiency=use_efficiency
            )
            date_df["dk_fpts_mean"] = dk_fpts_mean
            if base_stat_box:
                for name, values in base_stat_box.items():
                    date_df[f"{name}_mean"] = values

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

            schedule_df = _load_schedule_for_date(root, pd.Timestamp(game_date))
            implied_team_points = _build_implied_team_points(date_df, schedule_df)

            rng = np.random.default_rng(date_seed)
            world_fpts_samples: list[np.ndarray] = []
            active_mask_samples: list[np.ndarray] = []  # Track active masks for conditional aggregation
            starter_mask_arr = (
                pd.to_numeric(date_df.get("is_starter", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float) > 0
            )
            stat_defaults = np.full(len(date_df), np.nan, dtype=float)

            # Usage shares config for this path
            usage_shares_cfg_backfill = profile_cfg.usage_shares
            if usage_shares_cfg_backfill.enabled:
                typer.echo(
                    f"[sim_v2] usage_shares enabled (backfill): targets={usage_shares_cfg_backfill.targets} "
                    f"noise_std={usage_shares_cfg_backfill.share_noise_std} temp={usage_shares_cfg_backfill.share_temperature}"
                )

            for chunk_start in range(0, n_worlds_eff, worlds_per_chunk):
                chunk_size = min(worlds_per_chunk, n_worlds_eff - chunk_start)

                # 1. Sample availability FIRST
                u_active = rng.random(size=(chunk_size, len(minutes_center)))
                active_mask = u_active < play_prob_arr[None, :]
                active_mask_samples.append(active_mask)

                # 2. Sample minutes
                if use_minutes_noise_eff and sigma_minutes is not None:
                    eps_minutes = rng.standard_normal(size=(chunk_size, len(minutes_center)))
                    minutes_worlds = np.maximum(minutes_center[None, :] + eps_minutes * sigma_minutes[None, :], 0.0)
                else:
                    minutes_worlds = np.repeat(minutes_center[None, :], chunk_size, axis=0)

                # 3. Zero out inactive players
                minutes_worlds = minutes_worlds * active_mask.astype(float)

                # 4. Reconcile to 240 with active mask
                if enforce_team_240 and n_teams > 0:
                    minutes_worlds = enforce_team_240_minutes(
                        minutes_world=minutes_worlds,
                        team_indices=team_indices,
                        rotation_mask=rotation_mask,
                        bench_mask=bench_mask,
                        baseline_minutes=minutes_center,
                        clamp_scale=(0.7, 1.3),
                        active_mask=active_mask,
                        starter_mask=starter_mask_arr,
                        max_rotation_size=DEFAULT_MAX_ROTATION_SIZE,
                        play_prob=play_prob_arr,
                    )
                    team_sum = minutes_worlds @ team_one_hot  # (W_chunk, T)
                    typer.echo(
                        f"[sim_v2] {game_date.date()} enforce_240 team_minutes min/med/max="
                        f"{float(team_sum.min()):.2f}/{float(np.median(team_sum)):.2f}/{float(team_sum.max()):.2f}"
                    )

                stat_totals: dict[str, np.ndarray] = {}
                for target, rates in rate_arrays.items():
                    params = noise_params.get(target, {})
                    sigma_team = float(params.get("sigma_team", 0.0) or 0.0)
                    sigma_player = float(params.get("sigma_player", 0.0) or 0.0)
                    sigma_team *= float(team_sigma_scale_eff)
                    sigma_player *= float(player_sigma_scale_eff)

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

                # Apply usage shares allocation (redistributes FGA/FTA/TOV within teams)
                if usage_shares_cfg_backfill.enabled and stat_totals:
                    stat_totals = _apply_usage_shares_allocation(
                        stat_totals=stat_totals,
                        minutes_worlds=minutes_worlds,
                        rate_arrays=rate_arrays,
                        group_map=group_map,
                        usage_cfg=usage_shares_cfg_backfill,
                        rng=rng,
                    )

                eff_arrays: dict[str, np.ndarray] | None = None
                if use_efficiency:
                    eff_arrays = {}
                    eff_clamp = {"fg2_pct": (0.3, 0.75), "fg3_pct": (0.2, 0.55), "ft_pct": (0.5, 0.95)}
                    for target, col in eff_mapping.items():
                        lo, hi = eff_clamp[target]
                        vals = pd.to_numeric(date_df[col], errors="coerce").to_numpy(dtype=float)
                        eff_arrays[target] = np.clip(vals, lo, hi)

                dk_fpts_worlds, stat_box = _compute_fpts_and_boxscore(
                    stat_totals, efficiency_pct=eff_arrays, use_efficiency=use_efficiency
                )

                if (
                    profile_cfg.vegas_points_anchor
                    and "pts" in stat_box
                    and implied_team_points
                    and np.isfinite(profile_cfg.vegas_points_drift_pct)
                ):
                    pts_before = stat_box["pts"].copy()
                    stat_box["pts"] = _apply_team_points_vegas_anchor(
                        stat_box["pts"],
                        group_map={ (int(k[0]), int(k[1])): v for k, v in group_map.items() },
                        implied_team_points=implied_team_points,
                        drift_pct=profile_cfg.vegas_points_drift_pct,
                    )
                    dk_fpts_worlds = dk_fpts_worlds + (stat_box["pts"] - pts_before)
                world_fpts_samples.append(dk_fpts_worlds)

                base_cols = [
                    "game_date",
                    "game_id",
                    "team_id",
                    "player_id",
                    "is_starter",
                    minutes_col,
                    "dk_fpts_mean",
                    "sim_profile",
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
                    world_df["pts_world"] = stat_box.get("pts", stat_defaults)[offset] if stat_box else stat_defaults
                    world_df["reb_world"] = stat_box.get("reb", stat_defaults)[offset] if stat_box else stat_defaults
                    world_df["ast_world"] = stat_box.get("ast", stat_defaults)[offset] if stat_box else stat_defaults
                    world_df["stl_world"] = stat_box.get("stl", stat_defaults)[offset] if stat_box else stat_defaults
                    world_df["blk_world"] = stat_box.get("blk", stat_defaults)[offset] if stat_box else stat_defaults
                    world_df["tov_world"] = stat_box.get("tov", stat_defaults)[offset] if stat_box else stat_defaults
                    world_df["oreb_world"] = stat_box.get("oreb", stat_defaults)[offset] if stat_box else stat_defaults
                    world_df["dreb_world"] = stat_box.get("dreb", stat_defaults)[offset] if stat_box else stat_defaults

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
            for world_id in range(n_worlds_eff):
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
                    "sim_profile",
                ]
                if minutes_col != "minutes_p50" and "minutes_p50" in date_df.columns:
                    base_cols.append("minutes_p50")
                base_cols = list(dict.fromkeys(base_cols))
                world_df = date_df[base_cols + ([c for c in ["dk_fpts_actual"] if c in date_df.columns])].copy()
                world_df["world_id"] = world_id
                minutes_world = date_df[minutes_col].to_numpy()
                if enforce_team_240 and n_teams > 0:
                    minutes_world = enforce_team_240_minutes(
                        minutes_world=minutes_world[None, :],
                        team_indices=team_indices,
                        rotation_mask=rotation_mask,
                        bench_mask=bench_mask,
                        baseline_minutes=minutes_world,
                        clamp_scale=(0.7, 1.3),
                        starter_mask=(
                            pd.to_numeric(date_df.get("is_starter", 0.0), errors="coerce")
                            .fillna(0.0)
                            .to_numpy(dtype=float)
                            > 0
                        ),
                        max_rotation_size=DEFAULT_MAX_ROTATION_SIZE,
                        play_prob=date_df["play_prob"].fillna(1.0).to_numpy(dtype=float) if "play_prob" in date_df.columns else None,
                    )[0]
                world_df["minutes_sim"] = minutes_world
                world_df["dk_fpts_world"] = dk_fpts_world
                world_df["pts_world"] = np.full(len(world_df), np.nan, dtype=float)
                world_df["reb_world"] = np.full(len(world_df), np.nan, dtype=float)
                world_df["ast_world"] = np.full(len(world_df), np.nan, dtype=float)
                world_df["stl_world"] = np.full(len(world_df), np.nan, dtype=float)
                world_df["blk_world"] = np.full(len(world_df), np.nan, dtype=float)
                world_df["tov_world"] = np.full(len(world_df), np.nan, dtype=float)
                world_df["oreb_world"] = np.full(len(world_df), np.nan, dtype=float)
                world_df["dreb_world"] = np.full(len(world_df), np.nan, dtype=float)
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
