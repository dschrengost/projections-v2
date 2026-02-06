"""Generate sim_v2 worlds from minutes + rates with stochastic noise."""

from __future__ import annotations

import os
import json
import time
import math
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import typer

from projections.fpts_v2.scoring import compute_dk_fpts
from projections.minutes import PLAY_THRESHOLD_MINUTES, ROTATION_THRESHOLD_MINUTES
from projections.paths import data_path, get_project_root
from projections.sim_v2.bench_zero_mixture import apply_bench_zero_mixture
from projections.sim_v2.config import DEFAULT_PROFILES_PATH, UsageSharesConfig, load_sim_v2_profile
from projections.sim_v2.game_factor import apply_game_factor
from projections.sim_v2.game_script import GameScriptConfig, classify_script, sample_minutes_with_scripts
from projections.sim_v2.minutes_allocator import allocate_team_minutes_matrix
from projections.sim_v2.minutes_noise import (
    build_sigma_per_player,
    load_minutes_noise_params,
    status_bucket_from_raw,
)
from projections.sim_v2.minutes_physics import (
    apply_minutes_availability_policy,
    apply_team_feasibility_gate,
    compute_max_increase_by_depth,
    compute_rotation_lock_mask,
)
from projections.sim_v2.play_prob_policy import apply_play_prob_policy_with_diagnostics
from projections.sim_v2.minutes_stabilization import (
    apply_pre_sim_qp_reconcile,
    recenter_team_minutes_to_conditional_means,
    sample_minutes_noise_per_world,
)
from projections.sim_v2.minutes_worlds_model_space_v1 import (
    MinutesWorldsConfig as ModelSpaceMinutesWorldsConfig,
    sample_minutes_worlds_model_space_v1,
)
from projections.sim_v2.noise import load_rates_noise_params
from projections.sim_v2.worlds_summary import compute_played_mask

app = typer.Typer(add_completion=False)

DEFAULT_MAX_ROTATION_SIZE = 10
TEAM_MINUTES_TARGET = 240.0
MINUTES_CAP_SIM_V3 = 41.0
MIN_TEAM_SIZE_FOR_TEAM_MINUTES_RECONCILE = 5


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


_NORM_PDF_COEF = 1.0 / math.sqrt(2.0 * math.pi)


def _norm_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) * _NORM_PDF_COEF


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    # Abramowitz-Stegun approximation; fast and accurate for our use.
    x_abs = np.abs(x)
    t = 1.0 / (1.0 + 0.2316419 * x_abs)
    d = _norm_pdf(x_abs)
    poly = t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))))
    prob = 1.0 - d * poly
    return np.where(x >= 0.0, prob, 1.0 - prob)


def _assert_inactive_zero_minutes(
    *,
    stage: str,
    minutes_worlds: np.ndarray,
    active_mask: np.ndarray,
    game_date: str,
    player_ids: np.ndarray,
    team_ids: np.ndarray,
    game_ids: np.ndarray,
    policy_reason: np.ndarray | None,
    world_offset: int,
) -> None:
    """Debug assertion: inactive players must have exactly 0 minutes."""
    if minutes_worlds.size == 0:
        return
    bad = (~active_mask) & (minutes_worlds > 0.0)
    if not bad.any():
        return

    idxs = np.argwhere(bad)
    reason_arr = policy_reason if policy_reason is not None else None
    for w_idx, p_idx in idxs:
        pid = player_ids[p_idx] if player_ids.size else str(p_idx)
        tid = team_ids[p_idx] if team_ids.size else "unknown"
        gid = game_ids[p_idx] if game_ids.size else "unknown"
        reason = reason_arr[p_idx] if reason_arr is not None and reason_arr.size == player_ids.size else "n/a"
        typer.echo(
            "[sim_v2][inactive_minutes_violation] "
            f"stage={stage} date={game_date} game_id={gid} team_id={tid} "
            f"world={world_offset + int(w_idx)} player_id={pid} "
            f"minutes={float(minutes_worlds[w_idx, p_idx]):.6f} "
            f"active={bool(active_mask[w_idx, p_idx])} policy_reason={reason}",
            err=True,
        )

    raise AssertionError(
        f"[sim_v2][dev_assert] inactive minutes > 0 detected at stage={stage}: n={len(idxs)}"
    )


def _adjust_mean_for_clip(mu: np.ndarray, sigma: float, max_iter: int = 6) -> np.ndarray:
    """
    Solve for m so that E[max(N(m, sigma), 0)] == mu (mu >= 0).
    """
    sigma_f = float(sigma)
    if sigma_f <= 0.0:
        return mu
    target = np.maximum(mu, 0.0)
    m = target.copy()
    for _ in range(max_iter):
        a = m / sigma_f
        Phi = _norm_cdf(a)
        phi = _norm_pdf(a)
        f = m * Phi + sigma_f * phi
        denom = np.maximum(Phi, 1e-6)
        m = m - (f - target) / denom
    min_m = -8.0 * sigma_f
    return np.maximum(m, min_m)


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
    for c in ["minutes_final", "minutes_pred_p50", "minutes_p50_cond", "minutes_p50"]:
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
        ("minutes_pred_p50", ["minutes_final", "minutes_p50_cond", "minutes_p50"]),
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
    from projections.pipeline.effective_inputs import EFFECTIVE_MINUTES_FILENAME
    from projections.pipeline import control_plane

    date_token = pd.Timestamp(game_date).date().isoformat()
    if minutes_source != "minutes_v1":
        raise ValueError(f"Unsupported minutes_source={minutes_source}")

    daily_base = root / "artifacts" / "minutes_v1" / "daily" / date_token
    gold_base = root / "gold" / "projections_minutes_v1" / f"game_date={date_token}"

    resolved_daily = _read_latest_run_id(daily_base)
    resolved_gold = _read_latest_run_id(gold_base)
    resolved_run = run_id or resolved_daily or resolved_gold

    candidates: list[tuple[Path, Optional[str], str]] = []
    if run_id:
        candidates.append((daily_base / f"run={run_id}" / EFFECTIVE_MINUTES_FILENAME, run_id, "minutes_v1_daily_effective"))
        candidates.append((daily_base / f"run={run_id}" / "minutes.parquet", run_id, "minutes_v1_daily"))
        candidates.append((gold_base / f"run={run_id}" / EFFECTIVE_MINUTES_FILENAME, run_id, "projections_minutes_v1_effective"))
        candidates.append((gold_base / f"run={run_id}" / "minutes.parquet", run_id, "projections_minutes_v1"))
    else:
        if resolved_run:
            candidates.append(
                (daily_base / f"run={resolved_run}" / EFFECTIVE_MINUTES_FILENAME, resolved_run, "minutes_v1_daily_effective")
            )
            candidates.append(
                (daily_base / f"run={resolved_run}" / "minutes.parquet", resolved_run, "minutes_v1_daily")
            )
            candidates.append(
                (gold_base / f"run={resolved_run}" / EFFECTIVE_MINUTES_FILENAME, resolved_run, "projections_minutes_v1_effective")
            )
            candidates.append(
                (gold_base / f"run={resolved_run}" / "minutes.parquet", resolved_run, "projections_minutes_v1")
            )
        allow_legacy_flat = (
            os.environ.get("PROJECTIONS_ALLOW_LEGACY_FLAT_GOLD_READS", "").strip().lower() in {"1", "true", "yes"}
            or bool(os.environ.get("PYTEST_CURRENT_TEST"))
            or control_plane.allow_unpromoted_run_reads()
        )
        if allow_legacy_flat:
            gold_path = gold_base / "minutes.parquet"
            candidates.append((gold_path, resolved_gold, "projections_minutes_v1_flat"))

    project_root = get_project_root()
    if project_root != root:
        daily_base_project = project_root / "artifacts" / "minutes_v1" / "daily" / date_token
        gold_base_project = project_root / "gold" / "projections_minutes_v1" / f"game_date={date_token}"
        resolved_daily_project = _read_latest_run_id(daily_base_project)
        resolved_gold_project = _read_latest_run_id(gold_base_project)

        if run_id:
            candidates.append(
                (
                    daily_base_project / f"run={run_id}" / EFFECTIVE_MINUTES_FILENAME,
                    run_id,
                    "minutes_v1_daily_effective_project",
                )
            )
            candidates.append(
                (
                    daily_base_project / f"run={run_id}" / "minutes.parquet",
                    run_id,
                    "minutes_v1_daily_project",
                )
            )
            candidates.append(
                (
                    gold_base_project / f"run={run_id}" / EFFECTIVE_MINUTES_FILENAME,
                    run_id,
                    "projections_minutes_v1_effective_project",
                )
            )
            candidates.append(
                (
                    gold_base_project / f"run={run_id}" / "minutes.parquet",
                    run_id,
                    "projections_minutes_v1_project",
                )
            )
        else:
            resolved_project_run = resolved_daily_project or resolved_gold_project
            if resolved_project_run:
                candidates.append(
                    (
                        daily_base_project / f"run={resolved_project_run}" / EFFECTIVE_MINUTES_FILENAME,
                        resolved_project_run,
                        "minutes_v1_daily_effective_project",
                    )
                )
                candidates.append(
                    (
                        daily_base_project / f"run={resolved_project_run}" / "minutes.parquet",
                        resolved_project_run,
                        "minutes_v1_daily_project",
                    )
                )
                candidates.append(
                    (
                        gold_base_project / f"run={resolved_project_run}" / EFFECTIVE_MINUTES_FILENAME,
                        resolved_project_run,
                        "projections_minutes_v1_effective_project",
                    )
                )
                candidates.append(
                    (
                        gold_base_project / f"run={resolved_project_run}" / "minutes.parquet",
                        resolved_project_run,
                        "projections_minutes_v1_project",
                    )
                )
            allow_legacy_flat = (
                os.environ.get("PROJECTIONS_ALLOW_LEGACY_FLAT_GOLD_READS", "").strip().lower() in {"1", "true", "yes"}
                or bool(os.environ.get("PYTEST_CURRENT_TEST"))
                or control_plane.allow_unpromoted_run_reads()
            )
            if allow_legacy_flat:
                candidates.append(
                    (
                        gold_base_project / "minutes.parquet",
                        resolved_gold_project,
                        "projections_minutes_v1_flat_project",
                    )
                )

    for path, rid, label in candidates:
        if path.exists():
            df = pd.read_parquet(path)
            if "game_date" not in df.columns:
                if "date" in df.columns:
                    df["game_date"] = pd.to_datetime(df["date"], errors="coerce")
                else:
                    df["game_date"] = pd.to_datetime(date_token)
            return df, rid, path, label
    raise FileNotFoundError(
        f"No minutes_v1 projection found for {date_token} (source={minutes_source}, run_id={run_id!r})."
    )


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
    from projections.pipeline.effective_inputs import EFFECTIVE_RATES_FILENAME

    date_token = pd.Timestamp(game_date).date().isoformat()
    base = root / "gold" / "rates_v1_live" / date_token
    resolved_run = run_id or _read_latest_run_id(base)
    candidate = base / "rates.parquet"
    candidates: list[Path] = [base / EFFECTIVE_RATES_FILENAME, base / "rates.parquet"]
    if resolved_run:
        candidates = [
            base / f"run={resolved_run}" / EFFECTIVE_RATES_FILENAME,
            base / f"run={resolved_run}" / "rates.parquet",
            *candidates,
        ]
        candidate = candidates[0]
    for path in candidates:
        if path.exists():
            df = pd.read_parquet(path)
            candidate = path
            break
    else:
        raise FileNotFoundError(f"No rates_v1_live parquet found under {base}")
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


def _minutes_concentration_metrics(vec: np.ndarray) -> dict[str, float]:
    arr = np.asarray(vec, dtype=float)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    arr = np.clip(arr, 0.0, None)
    total = float(arr.sum())
    if total <= 1e-9:
        return {
            "top8_share": 0.0,
            "entropy": 0.0,
            "count_gt_18": 0.0,
            "count_gt_10": 0.0,
        }

    if arr.size <= 8:
        top8_sum = float(arr.sum())
    else:
        top8_sum = float(np.partition(arr, -8)[-8:].sum())

    p = arr / total
    entropy = float(-(p * np.log(np.maximum(p, 1e-12))).sum())
    return {
        "top8_share": float(top8_sum / total),
        "entropy": entropy,
        "count_gt_18": float((arr > 18.0).sum()),
        "count_gt_10": float((arr > 10.0).sum()),
    }


def _resolve_minutes_column(df: pd.DataFrame) -> str:
    for candidate in ("minutes_final", "minutes_p50_cond", "minutes_p50", "minutes_pred_p50"):
        if candidate in df.columns:
            return candidate
    raise KeyError(
        "Missing minutes column (expected minutes_final/minutes_p50_cond/minutes_p50/minutes_pred_p50)"
    )


def _ensure_status_bucket(df: pd.DataFrame) -> pd.DataFrame:
    # Load any status overrides by player name
    from projections.paths import get_project_root
    
    override_path = get_project_root() / "config" / "status_overrides.json"
    status_overrides = {}
    if override_path.exists():
        try:
            payload = json.loads(override_path.read_text(encoding="utf-8"))
            status_overrides = payload.get("overrides", {})
            if status_overrides:
                typer.echo(f"[sim_v2] Loaded {len(status_overrides)} status overrides: {list(status_overrides.keys())}")
        except (json.JSONDecodeError, Exception) as e:
            typer.echo(f"[sim_v2] warning: failed to load status overrides: {e}", err=True)
    
    if "status_bucket" in df.columns:
        df["status_bucket"] = df["status_bucket"].apply(status_bucket_from_raw)
    else:
        for col in ("status", "injury_status", "availability_status"):
            if col in df.columns:
                df["status_bucket"] = df[col].apply(status_bucket_from_raw)
                break
        else:
            df["status_bucket"] = "healthy"
    
    # Apply status overrides by player_name
    if status_overrides and "player_name" in df.columns:
        for player_name, override_status in status_overrides.items():
            mask = df["player_name"].str.lower() == player_name.lower()
            if mask.any():
                original = df.loc[mask, "status_bucket"].iloc[0]
                df.loc[mask, "status_bucket"] = status_bucket_from_raw(override_status)
                typer.echo(f"[sim_v2] Status override: {player_name}: {original} -> {override_status}")
    
    return df



def build_rates_mean_fpts(minutes_df: pd.DataFrame, rates_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join minutes_v1 and rates_v1 predictions and compute mean DK FPTS per player.

    Returns a DataFrame keyed by (game_date, game_id, team_id, player_id) with:
      - minutes_mean
      - fpts_mean
      - optional passthrough columns (minutes_p50_cond, minutes_p50, play_prob, is_starter, eligible_flag, minutes_alloc_mode)
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
    for extra in (
        "minutes_p50_cond",
        "minutes_p50",
        "minutes_lock_eff",
        "minutes_target_eff",
        "play_prob",
        "status_bucket",
        "is_starter",
        "rotation_prob",
        "eligible_flag",
        "minutes_alloc_mode",
        "p_rot",
        "mu_cond",
        "team_minutes_sum",
    ):
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
    sim_run_id: Optional[str] = typer.Option(
        None,
        "--run-id",
        help="Optional run id to partition outputs under game_date=.../run=...",
    ),
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
    sim_audit = os.environ.get("PROJECTIONS_SIM_AUDIT", "0").strip() == "1"
    dev_asserts = os.environ.get("PROJECTIONS_SIM_DEV_ASSERTS", "0").strip() == "1"

    def _resolve(value, override, label):
        if override is not None and override != value:
            typer.echo(f"[sim_v2] override {label}: profile={value} -> cli={override}")
            return override
        return value

    mean_source = getattr(profile_cfg, "mean_source", "rates")
    minutes_source = profile_cfg.minutes_source or "minutes_v1"
    rates_source = profile_cfg.rates_source or "rates_v1_live"

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
    seed_eff = seed if seed is not None else (profile_cfg.seed if profile_cfg.seed is not None else int(time.time() % 2**31))
    min_play_prob_eff = min_play_prob if min_play_prob is not None else profile_cfg.min_play_prob
    team_factor_sigma_eff = team_factor_sigma if team_factor_sigma is not None else profile_cfg.team_factor_sigma
    worlds_per_chunk = max(1, (profile_cfg.worlds_batch_size or profile_cfg.worlds_per_chunk))
    n_worlds_eff = int(n_worlds) if n_worlds is not None else int(profile_cfg.worlds_n or 2000)
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
        f"[sim_v2] rates_run_id={resolved_rates_run} minutes_run_id={resolved_minutes_run} "
        f"use_rates_noise={use_rates_noise_eff} split={rates_split} "
        f"use_minutes_noise={use_minutes_noise_eff} sigma_min={minutes_sigma_min_eff} min_play_prob={min_play_prob_eff}"
    )
    typer.echo(
        f"[sim_v2] game_scripts={profile_cfg.use_game_scripts} "
        f"play_prob_masking={getattr(profile_cfg, 'use_play_prob_masking', True)} "
        f"team_factor_sigma={team_factor_sigma_eff}"
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
                typer.echo(
                    f"[sim_v2] {pd.Timestamp(game_date).date()} missing minutes ({minutes_source}); skipping."
                )
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
                minutes_df.get(
                    "is_starter",
                    minutes_df.get("is_projected_starter", minutes_df.get("starter_flag")),
                ),
                errors="coerce",
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
            if sim_run_id:
                out_dir = out_dir / f"run={sim_run_id}"
            out_dir.mkdir(parents=True, exist_ok=True)

            date_seed = seed_eff + int(pd.Timestamp(game_date).toordinal())
            mu_arr = mu_df["fpts_mean"].to_numpy(dtype=float)
            minutes_sim_base = mu_df["minutes_mean"].to_numpy(dtype=float)
            world_fpts_samples: list[np.ndarray] = []
            minutes_world_samples: list[np.ndarray] = []
            base_cols = ["game_date", "game_id", "team_id", "player_id", "minutes_mean", "dk_fpts_mean", "sim_profile"]
            for extra in ("minutes_final", "minutes_p50_cond", "minutes_p50", "play_prob", "is_starter"):
                if extra in mu_df.columns and extra not in base_cols:
                    base_cols.append(extra)
            base_cols = list(dict.fromkeys(base_cols))
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
            alloc_mode = "legacy"
            if "minutes_alloc_mode" in mu_df.columns:
                try:
                    alloc_mode = str(mu_df["minutes_alloc_mode"].dropna().astype(str).iloc[0]).strip().lower()
                except Exception:
                    alloc_mode = "legacy"
            env_alloc = os.environ.get("PROJECTIONS_MINUTES_ALLOC_MODE")
            if env_alloc:
                env_value = str(env_alloc).strip().lower()
                if env_value in {"legacy", "lgbm", "minutes_v1"}:
                    alloc_mode = "legacy"
                elif env_value in {"rotalloc", "rotalloc_expk", "rotalloc-expk"}:
                    alloc_mode = "rotalloc_expk"
            minutes_alloc_metrics: dict[str, object] = {"minutes_alloc_mode": alloc_mode}
            eligible_flag_arr: np.ndarray | None = None
            max_rotation_size_eff: int | None = profile_cfg.max_rotation_size or DEFAULT_MAX_ROTATION_SIZE
            if alloc_mode in {"rotalloc_expk", "rotalloc_fringe_alpha", "share_with_rotalloc_elig"} and "eligible_flag" in mu_df.columns:
                eligible_flag_arr = (
                    pd.to_numeric(mu_df["eligible_flag"], errors="coerce")
                    .fillna(0.0)
                    .to_numpy(dtype=float)
                    > 0.5
                )
                try:
                    eligible_sizes = (
                        mu_df.assign(_eligible=eligible_flag_arr.astype(int))
                        .groupby(["game_id", "team_id"])["_eligible"]
                        .sum()
                    )
                    eligible_size_max = int(eligible_sizes.max()) if not eligible_sizes.empty else 0
                    max_rotation_size_eff = max(int(profile_cfg.max_rotation_size or 0), eligible_size_max)
                    minutes_alloc_metrics["eligible_size_p50"] = float(eligible_sizes.quantile(0.5))
                    minutes_alloc_metrics["eligible_size_p90"] = float(eligible_sizes.quantile(0.9))
                except Exception:
                    max_rotation_size_eff = max_rotation_size_eff

                team_sums = mu_df.groupby(["game_id", "team_id"])["minutes_mean"].sum()
                max_dev = float((team_sums - 240.0).abs().max()) if len(team_sums) else 0.0
                minutes_alloc_metrics["minutes_mean_team_sum_dev_max"] = float(max_dev)
                if max_dev > 1e-6:
                    typer.echo(
                        f"[sim_v2] warning: rotalloc minutes_mean sum-to-240 violated (max_dev={max_dev:.6f}); "
                        "treating minutes as legacy for reconciliation.",
                        err=True,
                    )
                    alloc_mode = "legacy"
                    eligible_flag_arr = None
                    max_rotation_size_eff = profile_cfg.max_rotation_size or DEFAULT_MAX_ROTATION_SIZE
                    minutes_alloc_metrics["minutes_alloc_mode"] = alloc_mode
            minutes_alloc_metrics["max_rotation_size_eff"] = int(max_rotation_size_eff or 0)
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

            # ------------------------------------------------------------------
            # Minutes physics knobs (availability policy, feasibility gate, caps)
            # ------------------------------------------------------------------
            hard_cap_minutes = float(MINUTES_CAP_SIM_V3)

            # Optional play_prob transform (sim uses p_eff for active sampling).
            play_prob_raw = np.clip(play_prob_arr.astype(float), 0.0, 1.0)
            play_prob_eff = play_prob_raw
            rotation_lock_mask = None
            policy_reason_arr: np.ndarray | None = None
            play_prob_policy_cfg = getattr(profile_cfg, "play_prob_policy", None)
            if play_prob_policy_cfg is not None and getattr(play_prob_policy_cfg, "enabled", False) and group_map:
                policy_df, policy_diag = apply_play_prob_policy_with_diagnostics(
                    mu_df,
                    play_prob_policy_cfg,
                    asof_ts=None,
                    lock_ts=None,
                )
                # Keep raw play_prob column unchanged for downstream/UI; use p_eff only for sampling.
                play_prob_eff = pd.to_numeric(policy_df["play_prob_eff"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                play_prob_eff = np.clip(play_prob_eff, 0.0, 1.0)
                rotation_lock_mask = policy_df["rotation_lock"].astype(bool).to_numpy()
                policy_reason_arr = policy_df["play_prob_policy_reason"].astype(str).to_numpy(dtype=object)

                minutes_alloc_metrics["play_prob_policy_enabled"] = True
                minutes_alloc_metrics["play_prob_policy"] = {
                    "n_players": int(policy_diag.n_players),
                    "n_rotation_locks": int(policy_diag.n_rotation_locks),
                    "n_changed": int(policy_diag.n_changed),
                    "reasons": dict(policy_diag.reasons),
                    "play_prob_raw": dict(policy_diag.play_prob_raw_summary),
                    "play_prob_eff": dict(policy_diag.play_prob_eff_summary),
                    "max_delta": float(policy_diag.max_delta),
                    "rotation_lock_floor": float(getattr(play_prob_policy_cfg, "rotation_lock_floor", 0.0)),
                    "probable_floor": float(getattr(play_prob_policy_cfg, "probable_floor", 0.0)),
                    "rotation_lock_min_cond_p50": float(getattr(play_prob_policy_cfg, "rotation_lock_min_cond_p50", 0.0)),
                    "rotation_lock_topk": int(getattr(play_prob_policy_cfg, "rotation_lock_topk", 0)),
                }
                if sim_audit:
                    typer.echo(
                        "[sim-physics] play_prob_policy enabled: "
                        f"locks={policy_diag.n_rotation_locks} changed={policy_diag.n_changed} "
                        f"max_delta={policy_diag.max_delta:.3f} reasons={policy_diag.reasons}"
                    )
            else:
                minutes_alloc_metrics["play_prob_policy_enabled"] = False

                # Legacy availability policy (older floor-only policy), if enabled.
                availability_policy_cfg = getattr(profile_cfg, "minutes_availability_policy", None)
                if availability_policy_cfg is not None and getattr(availability_policy_cfg, "enabled", False) and group_map:
                    status_bucket_arr = (
                        mu_df["status_bucket"].astype(str).to_numpy()
                        if "status_bucket" in mu_df.columns
                        else None
                    )
                    rotation_lock_mask, play_prob_eff, policy_diag = apply_minutes_availability_policy(
                        play_prob_raw=play_prob_raw,
                        baseline_minutes=minutes_sim_base,
                        is_starter=(gs_is_starter > 0) if gs_is_starter is not None else None,
                        status_bucket=status_bucket_arr,
                        group_map=group_map,
                        cfg=availability_policy_cfg,
                    )
                    minutes_alloc_metrics["minutes_availability_policy_enabled"] = True
                    minutes_alloc_metrics["minutes_availability_policy"] = {
                        "n_rotation_locks": int(policy_diag.n_rotation_locks),
                        "n_floored": int(policy_diag.n_floored),
                        "max_floor_delta": float(policy_diag.max_floor_delta),
                        "p_floor": float(getattr(availability_policy_cfg, "play_prob_floor", 0.0)),
                    }
                    if sim_audit:
                        typer.echo(
                            "[sim-physics] availability_policy enabled: "
                            f"locks={policy_diag.n_rotation_locks} floored={policy_diag.n_floored} "
                            f"max_delta={policy_diag.max_floor_delta:.3f}"
                        )
                else:
                    minutes_alloc_metrics["minutes_availability_policy_enabled"] = False

            # Increase-only absorption caps: compute per-player max increase minutes (delta_i_max).
            absorption_cfg = getattr(profile_cfg, "minutes_absorption_caps", None)
            max_increase_arr: np.ndarray | None = None
            cap_upper_arr = np.full(len(mu_df), hard_cap_minutes, dtype=float)
            if absorption_cfg is not None and getattr(absorption_cfg, "enabled", False) and group_map:
                max_increase_arr = compute_max_increase_by_depth(
                    baseline_minutes=minutes_sim_base,
                    is_starter=(gs_is_starter > 0) if gs_is_starter is not None else None,
                    group_map=group_map,
                    cfg=absorption_cfg,
                )
                cap_upper_arr = np.minimum(
                    hard_cap_minutes,
                    np.clip(minutes_sim_base, 0.0, None) + np.clip(max_increase_arr, 0.0, None),
                )
                minutes_alloc_metrics["minutes_absorption_caps_enabled"] = True
                minutes_alloc_metrics["minutes_absorption_caps"] = {
                    "core_rank_max": int(getattr(absorption_cfg, "core_rank_max", 0)),
                    "rotation_rank_max": int(getattr(absorption_cfg, "rotation_rank_max", 0)),
                    "bench_rank_max": int(getattr(absorption_cfg, "bench_rank_max", 0)),
                    "fringe_rank_max": int(getattr(absorption_cfg, "fringe_rank_max", 0)),
                    "core_delta_max": float(getattr(absorption_cfg, "core_delta_max", 0.0)),
                    "rotation_delta_max": float(getattr(absorption_cfg, "rotation_delta_max", 0.0)),
                    "bench_delta_max": float(getattr(absorption_cfg, "bench_delta_max", 0.0)),
                    "fringe_delta_max": float(getattr(absorption_cfg, "fringe_delta_max", 0.0)),
                    "deep_delta_max": float(getattr(absorption_cfg, "deep_delta_max", 0.0)),
                }
                if sim_audit:
                    typer.echo(
                        "[sim-physics] absorption_caps enabled: "
                        f"delta_p50={float(np.percentile(max_increase_arr, 50)):.1f} "
                        f"delta_p90={float(np.percentile(max_increase_arr, 90)):.1f}"
                    )
            else:
                minutes_alloc_metrics["minutes_absorption_caps_enabled"] = False

            # Rotation lock mask for feasibility (even if availability policy is disabled).
            feasibility_cfg = getattr(profile_cfg, "minutes_feasibility", None)
            if (
                rotation_lock_mask is None
                and feasibility_cfg is not None
                and getattr(feasibility_cfg, "enabled", False)
                and getattr(feasibility_cfg, "min_rotation_locks_active", None) is not None
                and group_map
            ):
                rotation_lock_mask = compute_rotation_lock_mask(
                    baseline_minutes=minutes_sim_base,
                    is_starter=(gs_is_starter > 0) if gs_is_starter is not None else None,
                    group_map=group_map,
                    top_k=8,
                    minutes_threshold=20.0,
                )

            # Minutes allocator priority signal (deterministic; no retrain).
            # Higher values => more protected from adjustment when projecting team totals to 240.
            priority_base_col = None
            for candidate in ("baseline_minutes_p50", "minutes_p50_cond", "minutes_p50", "minutes_mean"):
                if candidate in mu_df.columns:
                    priority_base_col = candidate
                    break
            if priority_base_col is None:
                priority_base = gs_minutes_p50.astype(float, copy=True)
                priority_base_col = "minutes_mean"
            else:
                priority_base = (
                    pd.to_numeric(mu_df[priority_base_col], errors="coerce")
                    .fillna(0.0)
                    .to_numpy(dtype=float)
                )
            starter_like = np.zeros(len(mu_df), dtype=float)
            for col in ("is_confirmed_starter", "is_projected_starter", "starter_flag", "is_starter"):
                if col in mu_df.columns:
                    starter_like = np.maximum(
                        starter_like,
                        pd.to_numeric(mu_df[col], errors="coerce").fillna(0.0).to_numpy(dtype=float),
                    )
            starter_like = starter_like > 0.5
            starter_bump = 2.0  # small deterministic bump
            minutes_alloc_priority = np.clip(priority_base, 0.0, None) + starter_bump * starter_like.astype(float)
            mu_df["_minutes_alloc_priority"] = minutes_alloc_priority
            minutes_alloc_metrics["minutes_alloc_priority_base_col"] = priority_base_col
            minutes_alloc_metrics["minutes_alloc_priority_starter_bump"] = float(starter_bump)

            # Stage 1A ops overrides: hard minutes locks/targets that must persist through team=240 allocation.
            fixed_mask_arr: np.ndarray | None = None
            fixed_minutes_arr: np.ndarray | None = None
            if "minutes_lock_eff" in mu_df.columns and "minutes_target_eff" in mu_df.columns:
                fixed_mask_arr = (
                    pd.to_numeric(mu_df["minutes_lock_eff"], errors="coerce")
                    .fillna(0.0)
                    .to_numpy(dtype=float)
                    > 0.5
                )
                fixed_minutes_arr = (
                    pd.to_numeric(mu_df["minutes_target_eff"], errors="coerce")
                    .fillna(0.0)
                    .to_numpy(dtype=float)
                )
                fixed_minutes_arr = np.clip(fixed_minutes_arr, 0.0, hard_cap_minutes)

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
            # Track realized availability draw counts (pre bench-zero mixture) for audit/debug.
            # Conditional moments later use minutes > 0 as the definition of "played".
            avail_counts_total = np.zeros(len(mu_df), dtype=np.int64)

            # Pre-sim QP reconciliation: when preserve_input_rotation=True and pre_sim_reconcile.enabled,
            # run QP once before simulation to ensure team totals = 240.
            # This replaces the per-world reconciliation in the sim loop.
            use_pre_sim_reconcile = (
                getattr(profile_cfg, "preserve_input_rotation", False)
                and getattr(profile_cfg, "pre_sim_reconcile", None) is not None
                and getattr(profile_cfg.pre_sim_reconcile, "enabled", False)
            )
            minutes_reconciled_arr: np.ndarray | None = None
            if use_pre_sim_reconcile:
                try:
                    # Apply QP reconciliation to the minutes DataFrame
                    reconcile_cfg = profile_cfg.pre_sim_reconcile
                    reconciled_df = apply_pre_sim_qp_reconcile(
                        mu_df,
                        starter_weight=reconcile_cfg.starter_weight,
                        minutes_weight_scale=reconcile_cfg.minutes_weight_scale,
                    )
                    # Use the reconciled minutes as the base for simulation
                    if "minutes_p50" in reconciled_df.columns:
                        minutes_reconciled_arr = reconciled_df["minutes_p50"].to_numpy(dtype=float)
                        gs_minutes_p50 = minutes_reconciled_arr.copy()
                        mu_df["minutes_mean"] = minutes_reconciled_arr
                        typer.echo(
                            f"[sim_v2] pre_sim_reconcile applied: "
                            f"starter_weight={reconcile_cfg.starter_weight} "
                            f"minutes_weight_scale={reconcile_cfg.minutes_weight_scale}"
                        )
                except Exception as exc:
                    typer.echo(
                        f"[sim_v2] warning: pre_sim_reconcile failed ({exc}); continuing without",
                        err=True,
                    )

            # New structured minutes noise config (when preserve_input_rotation=True and minutes_noise_config.enabled)
            # This replaces the legacy minutes noise + enforce_team_240 logic.
            use_structured_minutes_noise = (
                getattr(profile_cfg, "preserve_input_rotation", False)
                and getattr(profile_cfg, "minutes_noise_config", None) is not None
                and getattr(profile_cfg.minutes_noise_config, "enabled", False)
            )
            if use_structured_minutes_noise:
                mnc = profile_cfg.minutes_noise_config
                extra_bits: list[str] = []
                if getattr(mnc, "min_minutes_for_noise_override", None) is not None:
                    extra_bits.append(
                        f"min_minutes_for_noise_override={getattr(mnc, 'min_minutes_for_noise_override', None)}"
                    )
                if getattr(mnc, "include_tail_in_projection", False):
                    extra_bits.append("include_tail_in_projection=True")
                    extra_bits.append(
                        f"tail_min_adjustable_minutes={getattr(mnc, 'tail_min_adjustable_minutes', 0.0)}"
                    )
                extra = (" " + " ".join(extra_bits)) if extra_bits else ""
                typer.echo(
                    f"[sim_v2] structured minutes_noise enabled: "
                    f"sigma_starter={mnc.sigma_starter} sigma_bench={mnc.sigma_bench} "
                    f"min_minutes={mnc.min_minutes_for_noise} cap_abs={mnc.cap_abs}{extra}"
                )

            # PR5 backend: model-space minutes worlds sampling
            # Check if minutes_worlds config is set to model_space_v1 mode
            minutes_worlds_cfg = getattr(profile_cfg, "minutes_worlds", None)
            use_model_space_minutes = (
                minutes_worlds_cfg is not None
                and getattr(minutes_worlds_cfg, "mode", "legacy") == "model_space_v1"
            )
            if use_model_space_minutes:
                # Validate play_prob is not missing (per spec, PR5 backend must not silently fill with 1.0)
                if "play_prob" not in mu_df.columns or mu_df["play_prob"].isna().all():
                    if getattr(minutes_worlds_cfg, "fail_on_missing_play_prob", True):
                        raise ValueError(
                            "[sim_v2] model_space_v1 backend requires valid play_prob column; "
                            "set fail_on_missing_play_prob=False to degrade to legacy backend"
                        )
                    else:
                        typer.echo(
                            "[sim_v2] warning: play_prob missing, degrading from model_space_v1 to legacy backend",
                            err=True,
                        )
                        use_model_space_minutes = False
                else:
                    typer.echo(
                        f"[sim_v2] model_space_v1 minutes worlds enabled: "
                        f"gate_temperature={getattr(minutes_worlds_cfg, 'gate_temperature', 1.0)}"
                    )
                # Ensure mutual exclusion with other backends
                if use_model_space_minutes:
                    use_structured_minutes_noise = False

            audit_team_sum_errs: list[np.ndarray] = []
            audit_cap_bind_team_worlds = 0
            audit_cap_infeasible_team_worlds = 0
            audit_all_inactive_team_worlds = 0
            audit_total_team_worlds = 0

            # Run-level minutes allocation diagnostics (aggregate; no per-world spam).
            ma_team_worlds = 0
            ma_sum_n_active = 0.0
            ma_sum_n_nonzero = 0.0
            ma_sum_active_players = 0.0
            ma_sum_active_lt1 = 0.0
            ma_sum_off_240 = 0
            ma_top1: list[np.ndarray] = []
            ma_top5: list[np.ndarray] = []

            # Minutes physics diagnostics (team-world level, aggregated).
            phys_team_worlds = 0
            phys_infeasible_pre = 0
            phys_resampled_team_worlds = 0
            phys_resample_attempts_total = 0
            phys_promoted_team_worlds = 0
            phys_promoted_players_total = 0

            # Bench-zero mixture per-player drop probability (deterministic, pre world sampling).
            # This is a second regime on top of play_prob_eff: availability draws happen first, then
            # low-minute players can be dropped to 0 minutes with this probability.
            bz_cfg = getattr(profile_cfg, "bench_zero_mixture", None)
            bench_zero_p_zero = np.zeros(len(mu_df), dtype=float)
            bench_zero_threshold: float | None = None
            if bz_cfg is not None and getattr(bz_cfg, "enabled", False):
                bench_zero_threshold = float(getattr(bz_cfg, "minutes_threshold", 0.0))
                p_zero_base = float(getattr(bz_cfg, "p_zero_base", 0.0))
                p_zero_slope = float(getattr(bz_cfg, "p_zero_slope", 0.0))
                if bench_zero_threshold > 0.0 and p_zero_base >= 0.0:
                    in_bucket = minutes_sim_base < bench_zero_threshold
                    x = np.clip((bench_zero_threshold - minutes_sim_base) / bench_zero_threshold, 0.0, 1.0)
                    p_zero = np.clip(p_zero_base + p_zero_slope * x, 0.0, 0.95)
                    bench_zero_p_zero = np.where(in_bucket, p_zero, 0.0).astype(float)

            for chunk_start in range(0, n_worlds_eff, worlds_per_chunk):
                chunk_size = min(worlds_per_chunk, n_worlds_eff - chunk_start)
                rng = np.random.default_rng(date_seed + chunk_start)

                # 1. Sample availability FIRST (before minutes)
                # When use_play_prob_masking is False, skip Bernoulli sampling but still
                # exclude players with play_prob=0 (OUT/injured players)
                use_play_prob_masking = getattr(profile_cfg, "use_play_prob_masking", True)
                if use_play_prob_masking:
                    u_active = rng.random(size=(chunk_size, len(play_prob_arr)))
                    active_mask = u_active < play_prob_eff[None, :]
                else:
                    # All players with play_prob > 0 are active; OUT players (play_prob=0) stay inactive
                    active_mask = np.broadcast_to(play_prob_eff > 0, (chunk_size, len(play_prob_eff)))
                if eligible_flag_arr is not None and not getattr(profile_cfg, "preserve_input_rotation", False):
                    # Only apply eligible_flag filtering when NOT preserving input rotation.
                    # With preserve_input_rotation=True, the input frame is assumed to already reflect the
                    # desired rotation/eligibility set.
                    active_mask = active_mask & np.broadcast_to(eligible_flag_arr[None, :], active_mask.shape)

                # 1b. Team/world feasibility gate: resample availability draws until constraints are feasible.
                if (
                    use_play_prob_masking
                    and feasibility_cfg is not None
                    and getattr(feasibility_cfg, "enabled", False)
                    and group_map
                ):
                    eligible_mask_for_gate = (
                        eligible_flag_arr
                        if (eligible_flag_arr is not None and not getattr(profile_cfg, "preserve_input_rotation", False))
                        else None
                    )
                    active_mask, gate_diag = apply_team_feasibility_gate(
                        active_mask,
                        play_prob=play_prob_eff,
                        baseline_minutes=minutes_sim_base,
                        cap_upper=cap_upper_arr,
                        group_map=group_map,
                        cfg=feasibility_cfg,
                        rng=rng,
                        eligible_mask=eligible_mask_for_gate,
                        rotation_lock_mask=rotation_lock_mask,
                        target_total=TEAM_MINUTES_TARGET,
                        eps=1e-6,
                    )
                    phys_team_worlds += int(gate_diag.n_team_worlds)
                    phys_infeasible_pre += int(gate_diag.n_infeasible_pre_resample)
                    phys_resampled_team_worlds += int(gate_diag.n_resampled_team_worlds)
                    phys_resample_attempts_total += int(gate_diag.resample_attempts_total)
                    phys_promoted_team_worlds += int(gate_diag.n_promoted_team_worlds)
                    phys_promoted_players_total += int(gate_diag.promoted_players_total)
                    if sim_audit and chunk_start == 0:
                        frac_infeasible = (
                            float(gate_diag.n_infeasible_pre_resample) / float(gate_diag.n_team_worlds)
                            if gate_diag.n_team_worlds
                            else 0.0
                        )
                        frac_resampled = (
                            float(gate_diag.n_resampled_team_worlds) / float(gate_diag.n_team_worlds)
                            if gate_diag.n_team_worlds
                            else 0.0
                        )
                        avg_attempts = (
                            float(gate_diag.resample_attempts_total) / float(gate_diag.n_resampled_team_worlds)
                            if gate_diag.n_resampled_team_worlds
                            else 0.0
                        )
                        typer.echo(
                            "[sim-physics][resample] feasibility_gate: "
                            f"team_worlds={gate_diag.n_team_worlds} "
                            f"frac_infeasible_pre={frac_infeasible:.4f} "
                            f"frac_resampled={frac_resampled:.4f} avg_attempts={avg_attempts:.2f} "
                            f"promoted_worlds={gate_diag.n_promoted_team_worlds} promoted_players={gate_diag.promoted_players_total}"
                        )

                # Snapshot realized availability counts (pre bench-zero mixture and pre minutes allocation).
                # This should roughly match play_prob_eff, except for feasibility-gate promotions.
                avail_counts_total += active_mask.sum(axis=0).astype(np.int64, copy=False)

                # 2. Sample minutes based on backend: model_space_v1 > structured_noise > game_scripts > fallback
                if use_model_space_minutes:
                    # PR5 backend: model-space minutes worlds using transformer aux outputs
                    # Extract gate/share logits if available in mu_df
                    gate_logit_arr = (
                        pd.to_numeric(mu_df["gate_logit"], errors="coerce").to_numpy(dtype=float)
                        if "gate_logit" in mu_df.columns
                        else None
                    )
                    gate_prob_arr = (
                        pd.to_numeric(mu_df["gate_prob"], errors="coerce").to_numpy(dtype=float)
                        if "gate_prob" in mu_df.columns
                        else None
                    )
                    share_logit_arr = (
                        pd.to_numeric(mu_df["share_logit"], errors="coerce").to_numpy(dtype=float)
                        if "share_logit" in mu_df.columns
                        else np.zeros(len(mu_df), dtype=float)  # Fallback: no share variation
                    )

                    # Build model-space config from profile
                    model_space_cfg = ModelSpaceMinutesWorldsConfig(
                        gate_temperature=getattr(minutes_worlds_cfg, "gate_temperature", 1.0),
                        use_bench_zero_mixture=True,  # Default on per spec
                        bench_zero_minutes_threshold=8.0,
                        bench_zero_p_base=0.25,
                        bench_zero_p_slope=0.5,
                    )

                    # Call the PR5 backend (handles active_mask internally)
                    pr5_result = sample_minutes_worlds_model_space_v1(
                        minutes_mean=gs_minutes_p50,
                        gate_logit=gate_logit_arr,
                        gate_prob=gate_prob_arr,
                        share_logit=share_logit_arr,
                        play_prob=play_prob_eff,
                        team_indices=team_indices,
                        n_worlds=chunk_size,
                        rng=rng,
                        config=model_space_cfg,
                    )
                    minutes_worlds = pr5_result.minutes_worlds
                    active_mask = pr5_result.active_mask

                    # Log diagnostics on first chunk
                    if chunk_start == 0:
                        diag = pr5_result.diagnostics
                        typer.echo(
                            f"[sim_v2] model_space_v1 stats: "
                            f"active_rate={diag['active_rate']:.3f} "
                            f"rotation_rate={diag['rotation_rate']:.3f} "
                            f"zero_minutes_rate={diag['zero_minutes_rate']:.3f}"
                        )
                elif use_structured_minutes_noise:
                    # New path: structured minutes noise with cheap team-240 projection.
                    # When preserve_input_rotation=True, we skip the heavy per-world QP
                    # and instead use fast bounded noise + team-240 projection.
                    mnc = profile_cfg.minutes_noise_config
                    minutes_worlds, noise_stats = sample_minutes_noise_per_world(
                        minutes_reconciled=gs_minutes_p50,
                        minutes_p10=gs_minutes_p10,
                        minutes_p90=gs_minutes_p90,
                        is_starter=gs_is_starter > 0,
                        team_indices=team_indices,
                        n_worlds=chunk_size,
                        sigma_starter=mnc.sigma_starter,
                        sigma_bench=mnc.sigma_bench,
                        min_minutes_for_noise=mnc.min_minutes_for_noise,
                        min_minutes_for_noise_override=getattr(mnc, "min_minutes_for_noise_override", None),
                        cap_abs=mnc.cap_abs,
                        use_student_t=mnc.use_student_t,
                        t_df=mnc.t_df,
                        include_tail_in_projection=getattr(mnc, "include_tail_in_projection", False),
                        tail_min_adjustable_minutes=getattr(mnc, "tail_min_adjustable_minutes", 0.0),
                        lo_source=mnc.lo_source,
                        hi_source=mnc.hi_source,
                        lo_pad=mnc.lo_pad,
                        hi_pad=mnc.hi_pad,
                        rng=rng,
                    )
                    # Log diagnostics on first chunk
                    if chunk_start == 0:
                        typer.echo(
                            f"[sim_v2] minutes_noise stats: "
                            f"max_delta_before={noise_stats.max_delta_before_projection:.2f} "
                            f"mean_delta_before={noise_stats.mean_delta_before_projection:.3f} "
                            f"frac_residual_push={noise_stats.frac_teams_residual_push:.4f} "
                            f"sum_240_violations={noise_stats.sum_240_violations}"
                        )
                        if getattr(mnc, "include_tail_in_projection", False):
                            noise_threshold = getattr(mnc, "min_minutes_for_noise_override", None)
                            if noise_threshold is None:
                                noise_threshold = mnc.min_minutes_for_noise
                            noise_threshold = float(max(0.0, noise_threshold))
                            adj_threshold = float(max(0.0, getattr(mnc, "tail_min_adjustable_minutes", 0.0)))

                            noisy_mask = gs_minutes_p50 >= noise_threshold
                            adjustable_mask = gs_minutes_p50 >= adj_threshold

                            for (gid, tid), idxs in group_map.items():
                                base_vec = gs_minutes_p50[idxs]
                                post_vec = minutes_worlds[:, idxs].mean(axis=0)
                                frozen_sum = float(base_vec[~adjustable_mask[idxs]].sum()) if len(idxs) else 0.0

                                base_m = _minutes_concentration_metrics(base_vec)
                                post_m = _minutes_concentration_metrics(post_vec)
                                typer.echo(
                                    "[sim_v2][tail_proj] game_id=%s team_id=%s adj=%d noisy=%d frozen_sum=%.2f "
                                    "top8_share pre=%.3f post=%.3f | gt18 pre=%.0f post=%.0f | gt10 pre=%.0f post=%.0f | "
                                    "entropy pre=%.3f post=%.3f"
                                    % (
                                        int(gid),
                                        int(tid),
                                        int(adjustable_mask[idxs].sum()),
                                        int(noisy_mask[idxs].sum()),
                                        frozen_sum,
                                        base_m["top8_share"],
                                        post_m["top8_share"],
                                        base_m["count_gt_18"],
                                        post_m["count_gt_18"],
                                        base_m["count_gt_10"],
                                        post_m["count_gt_10"],
                                        base_m["entropy"],
                                        post_m["entropy"],
                                    )
                                )
                                if (post_m["top8_share"] - base_m["top8_share"]) > 0.05:
                                    typer.echo(
                                        "[sim_v2][tail_proj] WARNING: more concentrated after projection "
                                        f"(game_id={int(gid)} team_id={int(tid)} "
                                        f"top8_share_delta={post_m['top8_share']-base_m['top8_share']:.3f})",
                                        err=True,
                                    )
                    # Zero out inactive players' minutes (reconciled after masking below).
                    minutes_worlds = minutes_worlds * active_mask.astype(float)
                elif use_game_scripts and game_script_config is not None:
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
                        rotation_p50_threshold=profile_cfg.game_script_rotation_threshold,
                    )
                    # 3. Zero out inactive players' minutes (before reconciliation)
                    minutes_worlds = minutes_worlds * active_mask.astype(float)
                    # NOTE: team-240 is enforced later via the minutes allocator (after masking/bench-zero).
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
                    # NOTE: team-240 is enforced later via the minutes allocator (after masking/bench-zero).
                # Defense-in-depth: ensure inactive players are hard-zero before reconciliation.
                minutes_worlds = minutes_worlds * active_mask.astype(float)
                if dev_asserts:
                    _assert_inactive_zero_minutes(
                        stage="pre_reconcile",
                        minutes_worlds=minutes_worlds,
                        active_mask=active_mask,
                        game_date=str(pd.Timestamp(game_date).date()),
                        player_ids=mu_df["player_id"].astype(str).to_numpy(),
                        team_ids=gs_team_ids,
                        game_ids=gs_game_ids,
                        policy_reason=policy_reason_arr,
                        world_offset=chunk_start,
                    )

                # Optional bench/DNP mass-at-zero mixture: drop low-minute players to 0 with p_zero,
                # then let reconciliation redistribute minutes among remaining active players.
                bz_cfg = getattr(profile_cfg, "bench_zero_mixture", None)
                if bz_cfg is not None and getattr(bz_cfg, "enabled", False) and group_map:
                    min_active_override = (
                        int(getattr(feasibility_cfg, "min_active_players", 0))
                        if (feasibility_cfg is not None and getattr(feasibility_cfg, "enabled", False))
                        else None
                    )
                    stats = apply_bench_zero_mixture(
                        minutes_worlds,
                        active_mask,
                        group_map=group_map,
                        minutes_target=minutes_sim_base,
                        minutes_threshold=float(getattr(bz_cfg, "minutes_threshold", 8.0)),
                        p_zero_base=float(getattr(bz_cfg, "p_zero_base", 0.25)),
                        p_zero_slope=float(getattr(bz_cfg, "p_zero_slope", 0.0)),
                        cap_minutes=hard_cap_minutes,
                        total_minutes=TEAM_MINUTES_TARGET,
                        rng=rng,
                        min_active_needed_override=min_active_override,
                    )
                    minutes_worlds = minutes_worlds * active_mask.astype(float)
                    if sim_audit and chunk_start == 0:
                        typer.echo(
                            f"[sim_v2][audit] bench_zero_mixture: dropped={stats.n_player_worlds_dropped} "
                            f"restored={stats.n_player_worlds_restored_for_feasibility} "
                            f"min_active_needed={stats.min_active_needed}"
                        )
                    if dev_asserts:
                        _assert_inactive_zero_minutes(
                            stage="post_bench_zero",
                            minutes_worlds=minutes_worlds,
                            active_mask=active_mask,
                            game_date=str(pd.Timestamp(game_date).date()),
                            player_ids=mu_df["player_id"].astype(str).to_numpy(),
                            team_ids=gs_team_ids,
                            game_ids=gs_game_ids,
                            policy_reason=policy_reason_arr,
                            world_offset=chunk_start,
                        )

                # After masking, project active-only minutes to TEAM_MINUTES_TARGET per (team, world)
                # with bounds while protecting high-priority players from being squeezed.
                cap_bind_chunk = 0
                cap_infeasible_chunk = 0
                all_inactive_chunk = 0
                for _, idxs in group_map.items():
                    # Guardrail for tests/dev: if the input minutes frame is missing most of a roster
                    # (e.g., only 1-2 players present), scaling them to 240 produces nonsense minutes.
                    # Production minutes inputs always have >=20 rows (pipeline health checks).
                    if len(idxs) < MIN_TEAM_SIZE_FOR_TEAM_MINUTES_RECONCILE:
                        continue
                    idxs_arr = np.asarray(idxs, dtype=int)
                    priority_team = minutes_alloc_priority[idxs_arr]
                    allocated, stats = allocate_team_minutes_matrix(
                        minutes_worlds[:, idxs_arr],
                        active_mask[:, idxs_arr],
                        priority=priority_team,
                        cap=hard_cap_minutes,
                        max_increase=(max_increase_arr[idxs_arr] if max_increase_arr is not None else None),
                        baseline=(minutes_sim_base[idxs_arr] if max_increase_arr is not None else None),
                        fixed_mask=(fixed_mask_arr[idxs_arr] if fixed_mask_arr is not None else None),
                        fixed_minutes=(fixed_minutes_arr[idxs_arr] if fixed_minutes_arr is not None else None),
                        target_total=TEAM_MINUTES_TARGET,
                        k=3.0,
                        eps=1e-6,
                    )
                    minutes_worlds[:, idxs_arr] = allocated
                    cap_bind_chunk += int(stats["n_cap_bind_rows"])
                    cap_infeasible_chunk += int(stats["n_cap_infeasible_rows"])
                    all_inactive_chunk += int(stats["n_all_inactive"])

                    # Allocation diagnostics (team-world level).
                    active_team = active_mask[:, idxs_arr]
                    n_active = active_team.sum(axis=1).astype(float)
                    n_nonzero = (allocated > 1e-9).sum(axis=1).astype(float)
                    ma_team_worlds += int(len(n_active))
                    ma_sum_n_active += float(n_active.sum())
                    ma_sum_n_nonzero += float(n_nonzero.sum())

                    ma_sum_active_players += float(n_active.sum())
                    ma_sum_active_lt1 += float(((allocated < 1.0) & active_team).sum())

                    team_sum = allocated.sum(axis=1)
                    ma_sum_off_240 += int((np.abs(team_sum - TEAM_MINUTES_TARGET) > 1e-3).sum())

                    ma_top1.append(allocated.max(axis=1))
                    if allocated.shape[1] >= 5:
                        top5 = np.partition(allocated, allocated.shape[1] - 5, axis=1)[:, -5:].sum(axis=1)
                    else:
                        top5 = team_sum
                    ma_top5.append(top5)

                # Optional post-reconcile mean preservation (conditional on being active).
                mmr_cfg = getattr(profile_cfg, "minutes_mean_recentering", None)
                if mmr_cfg is not None and getattr(mmr_cfg, "enabled", False) and group_map:
                    max_abs_before = 0.0
                    max_abs_after = 0.0
                    for _, idxs in group_map.items():
                        targets = minutes_sim_base[np.asarray(idxs, dtype=int)]
                        before = minutes_worlds[:, idxs]
                        means_before = np.divide(
                            before.sum(axis=0, dtype=float),
                            active_mask[:, idxs].sum(axis=0).astype(float),
                            out=np.zeros(len(idxs), dtype=float),
                            where=active_mask[:, idxs].sum(axis=0) > 0,
                        )
                        max_abs_before = max(max_abs_before, float(np.max(np.abs(means_before - targets))))

                        corrected, rec_stats = recenter_team_minutes_to_conditional_means(
                            before,
                            active_mask[:, idxs],
                            target_minutes_conditional=targets,
                            total_minutes=TEAM_MINUTES_TARGET,
                            cap_minutes=MINUTES_CAP_SIM_V3,
                            max_iters=int(getattr(mmr_cfg, "max_iters", 10)),
                            step=float(getattr(mmr_cfg, "step", 1.0)),
                            tol=float(getattr(mmr_cfg, "tol", 1e-2)),
                        )
                        minutes_worlds[:, idxs] = corrected

                        means_after = np.divide(
                            corrected.sum(axis=0, dtype=float),
                            active_mask[:, idxs].sum(axis=0).astype(float),
                            out=np.zeros(len(idxs), dtype=float),
                            where=active_mask[:, idxs].sum(axis=0) > 0,
                        )
                        max_abs_after = max(max_abs_after, float(np.max(np.abs(means_after - targets))))

                    if sim_audit and chunk_start == 0:
                        typer.echo(
                            f"[sim_v2][audit] minutes_mean_recentering: max_abs_err_before={max_abs_before:.3f} "
                            f"max_abs_err_after={max_abs_after:.3f}"
                        )
                if dev_asserts:
                    _assert_inactive_zero_minutes(
                        stage="post_reconcile",
                        minutes_worlds=minutes_worlds,
                        active_mask=active_mask,
                        game_date=str(pd.Timestamp(game_date).date()),
                        player_ids=mu_df["player_id"].astype(str).to_numpy(),
                        team_ids=gs_team_ids,
                        game_ids=gs_game_ids,
                        policy_reason=policy_reason_arr,
                        world_offset=chunk_start,
                    )

                # DEV-ONLY integrity asserts: after world generation (minutes sampling + masking + reconcile),
                # enforce that minutes are non-negative and each (team, world) sums to ~240 before audits.
                if dev_asserts and group_map:
                    count_negative_minutes = int((minutes_worlds < 0.0).sum())
                    if count_negative_minutes != 0:
                        raise AssertionError(f"[sim_v2][dev_assert] negative minutes found: n={count_negative_minutes}")

                    team_sums = np.stack(
                        [minutes_worlds[:, idxs].sum(axis=1, dtype=float) for idxs in group_map.values()],
                        axis=1,
                    )  # (W, T)
                    max_abs_team_sum_dev = float(np.max(np.abs(team_sums - TEAM_MINUTES_TARGET)))
                    if max_abs_team_sum_dev >= 1e-4:
                        raise AssertionError(
                            f"[sim_v2][dev_assert] team-world minutes sum deviates from {TEAM_MINUTES_TARGET}: "
                            f"max_abs_dev={max_abs_team_sum_dev:.6g}"
                        )

                audit_cap_bind_team_worlds += cap_bind_chunk
                audit_cap_infeasible_team_worlds += cap_infeasible_chunk
                audit_all_inactive_team_worlds += all_inactive_chunk
                audit_total_team_worlds += int(len(group_map)) * int(chunk_size)

                if sim_audit and group_map:
                    err_chunks: list[np.ndarray] = []
                    for _, idxs in group_map.items():
                        err_chunks.append(np.abs(minutes_worlds[:, idxs].sum(axis=1) - TEAM_MINUTES_TARGET))
                    err_vec = np.concatenate(err_chunks) if err_chunks else np.zeros(0, dtype=float)
                    audit_team_sum_errs.append(err_vec)
                    if chunk_start == 0:
                        max_err = float(err_vec.max()) if err_vec.size else 0.0
                        p99_err = float(np.quantile(err_vec, 0.99)) if err_vec.size else 0.0
                        n_bad = int((err_vec > 1e-3).sum())
                        typer.echo(
                            f"[sim_v2][audit] minutes_team_sum_err: max={max_err:.6g} p99={p99_err:.6g} "
                            f"n_bad_gt_1e-3={n_bad} cap_bind_team_worlds={cap_bind_chunk} "
                            f"cap_infeasible_team_worlds={cap_infeasible_chunk} all_inactive_team_worlds={all_inactive_chunk}"
                        )

                # NOTE: Conditional moments are computed post-hoc using minutes >= PLAY_THRESHOLD_MINUTES.
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
                        mu_world = mu_stat + team_shock
                        if sigma_player > 0.0:
                            mu_adj = _adjust_mean_for_clip(mu_world, sigma_player)
                            total = np.clip(mu_adj + player_eps, 0.0, None)
                        else:
                            total = np.clip(mu_world, 0.0, None)
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

                # Enforce DNP semantics: inactive players must contribute exactly 0 stats/FPTS.
                # Rates noise uses additive shocks; without masking, inactive players can accrue non-zero stats.
                if stat_totals:
                    active_float = active_mask.astype(float)
                    for key, arr in stat_totals.items():
                        stat_totals[key] = arr * active_float

                if not stat_totals:
                    fpts_chunk = mu_arr[:, None]  # fallback: no stat noise
                    stat_box = {}
                else:
                    fpts_chunk, stat_box = _compute_fpts_and_boxscore(
                        stat_totals, efficiency_pct=eff_arrays, use_efficiency=use_efficiency
                    )
                    # Defense-in-depth: ensure inactive worlds are hard-zero in outputs.
                    fpts_chunk = np.where(active_mask, fpts_chunk, 0.0)
                    for stat_name, values in list(stat_box.items()):
                        stat_box[stat_name] = np.where(active_mask, values, 0.0)

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
                        # Preserve DNP=0 after post-processing.
                        fpts_chunk = np.where(active_mask, fpts_chunk, 0.0)

                    # Optional game-level factor to induce cross-team correlation.
                    gf_cfg = getattr(profile_cfg, "game_factor", None)
                    if gf_cfg is not None and getattr(gf_cfg, "enabled", False):
                        sigma = float(getattr(gf_cfg, "sigma", 0.0))
                        if sigma > 0.0:
                            mode = str(getattr(gf_cfg, "mode", "additive"))
                            beta_basis = str(getattr(gf_cfg, "beta_basis", "minutes_share"))
                            if beta_basis == "fpts_share":
                                basis = (
                                    pd.to_numeric(mu_df["dk_fpts_mean"], errors="coerce")
                                    .fillna(0.0)
                                    .to_numpy(dtype=float)
                                )
                            else:
                                basis = minutes_sim_base.astype(float, copy=False)
                            apply_game_factor(
                                fpts_chunk,
                                active_mask,
                                game_ids=gs_game_ids,
                                beta_basis=basis,
                                sigma=sigma,
                                mode=mode,  # type: ignore[arg-type]
                                rng=rng,
                            )
                            fpts_chunk = np.where(active_mask, fpts_chunk, 0.0)
                world_fpts_samples.append(fpts_chunk)
                # Track individual stat worlds for aggregation
                for stat_name in ("pts", "reb", "ast", "stl", "blk", "tov", "fga2", "fga3", "fta"):
                    if stat_name in stat_box:
                        stat_world_samples.setdefault(stat_name, []).append(stat_box[stat_name])

            if sim_audit and audit_total_team_worlds > 0 and audit_team_sum_errs:
                err_all = np.concatenate(audit_team_sum_errs)
                max_err = float(err_all.max()) if err_all.size else 0.0
                p99_err = float(np.quantile(err_all, 0.99)) if err_all.size else 0.0
                n_bad = int((err_all > 1e-3).sum())
                typer.echo(
                    f"[sim_v2][audit] minutes_team_sum_err (all chunks): max={max_err:.6g} p99={p99_err:.6g} "
                    f"n_bad_gt_1e-3={n_bad}/{audit_total_team_worlds} "
                    f"cap_bind_team_worlds={audit_cap_bind_team_worlds}/{audit_total_team_worlds} "
                    f"cap_infeasible_team_worlds={audit_cap_infeasible_team_worlds}/{audit_total_team_worlds} "
                    f"all_inactive_team_worlds={audit_all_inactive_team_worlds}/{audit_total_team_worlds}"
                )

            if audit_total_team_worlds > 0 and audit_cap_infeasible_team_worlds > 0:
                typer.echo(
                    f"[alloc-infeasible] cap_infeasible_team_worlds={audit_cap_infeasible_team_worlds}/{audit_total_team_worlds}",
                    err=True,
                )

            # Minutes physics diagnostics (availability gate / resampling).
            minutes_physics_summary: dict[str, object] = {}
            if feasibility_cfg is not None and getattr(feasibility_cfg, "enabled", False) and phys_team_worlds > 0:
                frac_infeasible_pre = float(phys_infeasible_pre) / float(phys_team_worlds)
                frac_resampled = float(phys_resampled_team_worlds) / float(phys_team_worlds)
                avg_attempts = (
                    float(phys_resample_attempts_total) / float(phys_resampled_team_worlds)
                    if phys_resampled_team_worlds > 0
                    else 0.0
                )
                frac_promoted = float(phys_promoted_team_worlds) / float(phys_team_worlds)
                minutes_physics_summary = {
                    "team_worlds": int(phys_team_worlds),
                    "min_active_players": int(getattr(feasibility_cfg, "min_active_players", 0)),
                    "min_sum_demand": float(getattr(feasibility_cfg, "min_sum_demand", 0.0)),
                    "max_resample_attempts": int(getattr(feasibility_cfg, "max_resample_attempts", 0)),
                    "frac_infeasible_pre_resample": float(frac_infeasible_pre),
                    "frac_resampled": float(frac_resampled),
                    "avg_resample_attempts": float(avg_attempts),
                    "frac_promoted": float(frac_promoted),
                    "promoted_players_total": int(phys_promoted_players_total),
                }
                minutes_alloc_metrics["minutes_physics"] = minutes_physics_summary
                typer.echo(f"[sim-physics] {json.dumps(minutes_physics_summary, separators=(',', ':'))}", err=True)

            # Emit a single compact minutes allocator diagnostic line for this date/profile.
            minutes_alloc_summary: dict[str, object] = {}
            if ma_team_worlds > 0:
                top1_all = np.concatenate(ma_top1) if ma_top1 else np.zeros(0, dtype=float)
                top5_all = np.concatenate(ma_top5) if ma_top5 else np.zeros(0, dtype=float)

                def _dist(arr: np.ndarray) -> dict[str, float]:
                    if arr.size == 0:
                        return {"mean": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0}
                    return {
                        "mean": float(np.mean(arr)),
                        "p10": float(np.percentile(arr, 10)),
                        "p50": float(np.percentile(arr, 50)),
                        "p90": float(np.percentile(arr, 90)),
                        "max": float(np.max(arr)),
                    }

                pct_active_lt1 = (
                    (ma_sum_active_lt1 / ma_sum_active_players) * 100.0
                    if ma_sum_active_players > 0
                    else 0.0
                )
                minutes_alloc_summary = {
                    "team_worlds": int(ma_team_worlds),
                    "n_active_mean": float(ma_sum_n_active / ma_team_worlds),
                    "n_nonzero_mean": float(ma_sum_n_nonzero / ma_team_worlds),
                    "top1_minutes": _dist(top1_all),
                    "top5_minutes_sum": _dist(top5_all),
                    "n_team_worlds_sum_off_gt_1e-3": int(ma_sum_off_240),
                    "pct_active_lt1_min": float(pct_active_lt1),
                }
                typer.echo(f"[minutes-alloc] {json.dumps(minutes_alloc_summary, separators=(',', ':'))}", err=True)

            if not world_fpts_samples:
                raise RuntimeError(
                    f"[sim_v2] No world samples generated for game_date={pd.Timestamp(game_date).date().isoformat()} "
                    f"(rows={len(mu_df)} profile={profile_cfg.name} n_worlds={n_worlds_eff})."
                )

            # Aggregate all worlds in-memory and compute CONDITIONAL quantiles
            # (only count worlds where player is active)
            if world_fpts_samples:
                # fpts_chunk is shape (chunk_size, n_players), stack all chunks
                all_fpts = np.vstack(world_fpts_samples)  # shape: (n_worlds, n_players)
                all_minutes = np.vstack(minutes_world_samples) if minutes_world_samples else None
                n_worlds_total, n_players = all_fpts.shape

                # Define "played" for conditional moments.
                # - When play_prob_masking=True (sim_v3/prod), worlds are unconditional and DNP => 0 minutes.
                #   Use minutes >= PLAY_THRESHOLD_MINUTES as the definition of "played".
                # - When play_prob_masking=False (baseline), worlds are generated conditional-on-playing
                #   except play_prob_eff==0 players; conditional moments should count all worlds.
                use_play_prob_masking = getattr(profile_cfg, "use_play_prob_masking", True)
                if all_minutes is not None:
                    all_active = compute_played_mask(
                        minutes_worlds=all_minutes,
                        play_prob_eff=play_prob_eff,
                        use_play_prob_masking=use_play_prob_masking,
                        play_threshold_minutes=float(PLAY_THRESHOLD_MINUTES),
                    )
                else:
                    # Fallback: treat FPTS > 0 as "played". This is imperfect but only used when minutes
                    # outputs are missing (should not happen in production).
                    all_active = all_fpts > 0.0

                # Sanitize inf/nan values AND physically impossible values that cause instability.
                # We interpret any valid NBA fantasy score as < 2000. Anything higher is numerical noise.
                # This protects against float32 overflow (max ~3.4e38) and variance explosion.
                MAX_VALID_FPTS = 2000.0
                bad_mask = ~np.isfinite(all_fpts) | (np.abs(all_fpts) > MAX_VALID_FPTS)
                if bad_mask.any():
                    n_bad = bad_mask.sum()
                    typer.echo(
                        f"[sim_v2] warning: {n_bad} invalid FPTS values (> {MAX_VALID_FPTS} or inf/nan) detected, marking as inactive",
                        err=True,
                    )
                    all_fpts = np.where(bad_mask, 0.0, all_fpts)
                    if all_minutes is not None:
                        all_minutes = np.where(bad_mask, 0.0, all_minutes)
                    all_active = all_active & ~bad_mask

                # Sanity guardrail: warn about extreme but plausible FPTS values (> 120).
                # NBA all-time single-game record is ~80 DK FPTS. Values > 120 indicate noise issues.
                EXTREME_FPTS_THRESHOLD = 120.0
                extreme_mask = (all_fpts > EXTREME_FPTS_THRESHOLD) & all_active
                if extreme_mask.any():
                    n_extreme = extreme_mask.sum()
                    max_extreme = float(all_fpts[extreme_mask].max())
                    # Find which players have extreme values
                    extreme_per_player = extreme_mask.sum(axis=0)
                    players_with_extreme = int((extreme_per_player > 0).sum())
                    typer.echo(
                        f"[sim_v2] GUARDRAIL: {n_extreme} player-worlds have FPTS > {EXTREME_FPTS_THRESHOLD:.0f} "
                        f"(max={max_extreme:.2f}, {players_with_extreme} players affected)",
                        err=True,
                    )
                    # Optionally fail if env var is set
                    if os.environ.get("PROJECTIONS_SIM_FAIL_ON_EXTREME_FPTS", "").lower() in {"1", "true", "yes"}:
                        raise RuntimeError(
                            f"Extreme FPTS values detected (max={max_extreme:.2f}). "
                            f"Set PROJECTIONS_SIM_FAIL_ON_EXTREME_FPTS=0 to disable this check."
                        )

                # Compute CONDITIONAL statistics (only worlds where player is active)
                # This is what DFS lineup builders want: E[FPTS | plays]
                n_worlds_total, n_players = all_fpts.shape
                active_counts = all_active.sum(axis=0)  # worlds active per player

                # Suppress divide-by-zero warnings for OUT players with active_counts=0
                # np.where handles these correctly (returns 0.0), but numpy still warns
                with np.errstate(divide='ignore', invalid='ignore'):
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
                
                # Compute UNCONDITIONAL statistics (include inactive worlds as 0).
                # This is the required semantics for any decision metric: DNP => 0.
                use_play_prob_masking = getattr(profile_cfg, "use_play_prob_masking", True)
                if (not use_play_prob_masking) and ("play_prob" in mu_df.columns):
                    # When availability masking is disabled, worlds are generated conditional-on-playing.
                    # Still emit unconditional moments by mixing with a point mass at 0 using play_prob.
                    p_play = np.clip(play_prob_arr.astype(float), 0.0, 1.0)
                    active_rate_sim = p_play

                    fpts_mean_uncond = fpts_mean * p_play
                    fpts_second_moment_cond = (fpts_std**2) + (fpts_mean**2)
                    fpts_var_uncond = p_play * fpts_second_moment_cond - (fpts_mean_uncond**2)
                    fpts_std_uncond = np.sqrt(np.maximum(fpts_var_uncond, 0.0))

                    # Unconditional quantiles for a (1-p)*delta_0 + p*F_cond mixture.
                    q_levels = np.asarray(quantiles, dtype=float)
                    q0 = 1.0 - p_play  # mass at 0
                    q_adj = np.where(p_play > 0, (q_levels[:, None] - q0[None, :]) / p_play[None, :], 0.0)
                    q_adj = np.minimum(q_adj, q_levels.max())
                    # If the desired quantile falls into the point mass at 0, the mixture quantile is 0.
                    in_zero = q_levels[:, None] <= q0[None, :]

                    # Linear interpolation of conditional quantiles on the precomputed grid.
                    # Include an explicit zero point to allow interpolation below the minimum conditional quantile.
                    grid = np.concatenate(([0.0], q_levels))
                    grid_q = np.vstack([np.zeros((1, n_players), dtype=float), fpts_quantiles])  # (Q+1, P)
                    idx_hi = np.searchsorted(grid, q_adj, side="left")
                    idx_hi = np.clip(idx_hi, 0, len(grid) - 1)
                    idx_lo = np.clip(idx_hi - 1, 0, len(grid) - 1)
                    x0 = grid[idx_lo]
                    x1 = grid[idx_hi]
                    y0 = np.take_along_axis(grid_q, idx_lo, axis=0)
                    y1 = np.take_along_axis(grid_q, idx_hi, axis=0)
                    t = np.divide(q_adj - x0, x1 - x0, out=np.zeros_like(q_adj), where=(x1 - x0) > 0)
                    fpts_quantiles_uncond = np.where(in_zero, 0.0, y0 + t * (y1 - y0)).astype(float)
                else:
                    fpts_mean_uncond = all_fpts.mean(axis=0, dtype=float)
                    fpts_std_uncond = all_fpts.std(axis=0, ddof=0, dtype=float)
                    fpts_quantiles_uncond = np.percentile(
                        all_fpts, [q * 100 for q in quantiles], axis=0
                    ).astype(float)
                    active_rate_sim = (active_counts / float(max(1, n_worlds_total))).astype(float)

                # Play-prob policy audit (best-effort; aggregate only).
                if (
                    minutes_alloc_metrics.get("play_prob_policy_enabled")
                    and rotation_lock_mask is not None
                    and "status_bucket" in mu_df.columns
                ):
                    sb = mu_df["status_bucket"].astype(str).str.strip().str.lower().to_numpy(dtype=object)
                    not_out_or_q = ~np.isin(sb, np.array(["out", "questionable"], dtype=object))
                    rot_mask = rotation_lock_mask.astype(bool) & not_out_or_q
                    fringe_mask = (~rotation_lock_mask.astype(bool)) & not_out_or_q

                    def _dist(arr: np.ndarray) -> dict[str, float]:
                        v = np.asarray(arr, dtype=float)
                        if v.size == 0:
                            return {"mean": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0}
                        return {
                            "mean": float(np.mean(v)),
                            "p10": float(np.percentile(v, 10)),
                            "p50": float(np.percentile(v, 50)),
                            "p90": float(np.percentile(v, 90)),
                        }

                    sim_p_active = np.asarray(active_rate_sim, dtype=float)
                    p_raw = np.asarray(play_prob_raw, dtype=float)
                    p_eff = np.asarray(play_prob_eff, dtype=float)

                    minutes_alloc_metrics["play_prob_policy_audit"] = {
                        "rotation_lock": {
                            "n": int(rot_mask.sum()),
                            "sim_p_active": _dist(sim_p_active[rot_mask]),
                            "play_prob_raw": _dist(p_raw[rot_mask]),
                            "play_prob_eff": _dist(p_eff[rot_mask]),
                        },
                        "fringe": {
                            "n": int(fringe_mask.sum()),
                            "sim_p_active": _dist(sim_p_active[fringe_mask]),
                            "play_prob_raw": _dist(p_raw[fringe_mask]),
                            "play_prob_eff": _dist(p_eff[fringe_mask]),
                        },
                        "abs_sim_minus_eff": _dist(np.abs(sim_p_active - p_eff)),
                    }

                if all_minutes is not None:
                    if (not use_play_prob_masking) and ("play_prob" in mu_df.columns):
                        p_play = np.clip(play_prob_arr.astype(float), 0.0, 1.0)
                        minutes_mean_uncond = minutes_mean * p_play
                        minutes_second_moment_cond = (minutes_std**2) + (minutes_mean**2)
                        minutes_var_uncond = p_play * minutes_second_moment_cond - (minutes_mean_uncond**2)
                        minutes_std_uncond = np.sqrt(np.maximum(minutes_var_uncond, 0.0))

                        # Mixture quantiles using the conditional 10/50/90 grid.
                        q_levels = np.asarray([0.10, 0.50, 0.90], dtype=float)
                        q0 = 1.0 - p_play
                        q_adj = np.where(p_play > 0, (q_levels[:, None] - q0[None, :]) / p_play[None, :], 0.0)
                        q_adj = np.minimum(q_adj, q_levels.max())
                        in_zero = q_levels[:, None] <= q0[None, :]

                        if minutes_quantiles is None:
                            minutes_quantiles_uncond = np.zeros((3, n_players), dtype=float)
                        else:
                            # Include an explicit zero point to allow interpolation below the minimum conditional quantile.
                            grid = np.concatenate(([0.0], q_levels))
                            grid_q = np.vstack([np.zeros((1, n_players), dtype=float), minutes_quantiles])  # (4, P)
                            idx_hi = np.searchsorted(grid, q_adj, side="left")
                            idx_hi = np.clip(idx_hi, 0, len(grid) - 1)
                            idx_lo = np.clip(idx_hi - 1, 0, len(grid) - 1)
                            x0 = grid[idx_lo]
                            x1 = grid[idx_hi]
                            y0 = np.take_along_axis(grid_q, idx_lo, axis=0)
                            y1 = np.take_along_axis(grid_q, idx_hi, axis=0)
                            t = np.divide(q_adj - x0, x1 - x0, out=np.zeros_like(q_adj), where=(x1 - x0) > 0)
                            minutes_quantiles_uncond = np.where(in_zero, 0.0, y0 + t * (y1 - y0)).astype(float)
                    else:
                        minutes_mean_uncond = all_minutes.mean(axis=0, dtype=float)
                        minutes_std_uncond = all_minutes.std(axis=0, ddof=0, dtype=float)
                        minutes_quantiles_uncond = np.percentile(all_minutes, [10, 50, 90], axis=0).astype(float)
                else:
                    minutes_mean_uncond = None
                    minutes_std_uncond = None
                    minutes_quantiles_uncond = None

                # ------------------------------------------------------------------
                # Coherence / sanity-check invariants (warn loudly; do not silently drift)
                # ------------------------------------------------------------------
                if n_worlds_total > 0 and all_minutes is not None:
                    # Availability draws should roughly match the effective play_prob used for sampling.
                    # Small differences are expected due to feasibility gate promotions.
                    p_avail_realized = (avail_counts_total / float(n_worlds_total)).astype(float)
                    if p_avail_realized.shape == play_prob_eff.shape:
                        diff_avail = np.abs(p_avail_realized - play_prob_eff)
                        worst = float(np.max(diff_avail)) if diff_avail.size else 0.0
                        if worst > 0.10:
                            topk = np.argsort(-diff_avail)[:5]
                            pid = mu_df["player_id"].astype(str).to_numpy()
                            rows = ", ".join(
                                f"{pid[i]}:|p_avail-p_eff|={diff_avail[i]:.3f}" for i in topk if diff_avail[i] > 0
                            )
                            typer.echo(
                                f"[sim_v2][coherence] WARNING: availability mismatch vs play_prob_eff "
                                f"(max_abs={worst:.3f}): {rows}",
                                err=True,
                            )

                    # Played rate should not exceed realized availability.
                    if active_rate_sim.size and p_avail_realized.size and active_rate_sim.shape == p_avail_realized.shape:
                        exceed = active_rate_sim > (p_avail_realized + 0.01)
                        if exceed.any():
                            idx = int(np.argmax(active_rate_sim - p_avail_realized))
                            pid = str(mu_df["player_id"].iloc[idx])
                            typer.echo(
                                "[sim_v2][coherence] WARNING: p_played > p_available "
                                f"(player_id={pid} p_played={float(active_rate_sim[idx]):.3f} "
                                f"p_available={float(p_avail_realized[idx]):.3f})",
                                err=True,
                            )

                    # Unconditional mean minutes must not exceed conditional mean minutes.
                    if minutes_mean_uncond is not None and minutes_mean is not None:
                        bad = minutes_mean_uncond > (minutes_mean + 1e-6)
                        if bad.any():
                            idx = int(np.argmax(minutes_mean_uncond - minutes_mean))
                            pid = str(mu_df["player_id"].iloc[idx])
                            typer.echo(
                                "[sim_v2][coherence] WARNING: minutes_mean_uncond > minutes_mean_cond "
                                f"(player_id={pid} uncond={float(minutes_mean_uncond[idx]):.3f} "
                                f"cond={float(minutes_mean[idx]):.3f})",
                                err=True,
                            )

                    # Non-core players should not be effectively always-active unless play_prob_eff is near 1.
                    if rotation_lock_mask is not None and active_rate_sim.size == play_prob_eff.size:
                        non_core = ~rotation_lock_mask.astype(bool)
                        suspicious = non_core & (play_prob_eff < 0.9) & (active_rate_sim > 0.98)
                        if suspicious.any():
                            idx = int(np.argmax(active_rate_sim * suspicious.astype(float)))
                            pid = str(mu_df["player_id"].iloc[idx])
                            typer.echo(
                                "[sim_v2][coherence] WARNING: non-core player nearly always played despite "
                                f"play_prob_eff<0.9 (player_id={pid} p_eff={float(play_prob_eff[idx]):.3f} "
                                f"p_played={float(active_rate_sim[idx]):.3f})",
                                err=True,
                            )

                # Lightweight worlds integrity report (aggregates only; no heavy output).
                if all_minutes is not None:
                    zero_mask = all_minutes == 0.0
                else:
                    zero_mask = all_fpts == 0.0

                worlds_integrity_payload = {
                    "worlds_shape": [int(n_worlds_total), int(n_players)],
                    "invalid_fpts_values": int(bad_mask.sum()) if "bad_mask" in locals() else 0,
                    "active_rate_sim": {
                        "mean": float(active_rate_sim.mean()) if active_rate_sim.size else 0.0,
                        "p10": float(np.percentile(active_rate_sim, 10)) if active_rate_sim.size else 0.0,
                        "p50": float(np.percentile(active_rate_sim, 50)) if active_rate_sim.size else 0.0,
                        "p90": float(np.percentile(active_rate_sim, 90)) if active_rate_sim.size else 0.0,
                    },
                    "avail_rate_sim": {
                        "mean": float((avail_counts_total / float(max(1, n_worlds_total))).mean())
                        if avail_counts_total.size
                        else 0.0,
                    },
                    "bench_zero": {
                        "enabled": bool(bz_cfg is not None and getattr(bz_cfg, "enabled", False)),
                        "minutes_threshold": float(bench_zero_threshold) if bench_zero_threshold is not None else None,
                        "p_zero_mean": float(np.mean(bench_zero_p_zero)) if bench_zero_p_zero.size else 0.0,
                        "p_zero_p90": float(np.percentile(bench_zero_p_zero, 90)) if bench_zero_p_zero.size else 0.0,
                    },
                    "zero_cells": {
                        "total": int(zero_mask.sum()),
                        "inactive": int((zero_mask & (~all_active)).sum()),
                        "active": int((zero_mask & all_active).sum()),
                    },
                }
                if "play_prob" in mu_df.columns:
                    play_prob_vals = pd.to_numeric(mu_df["play_prob"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                    worlds_integrity_payload["play_prob"] = {
                        "mean": float(np.mean(play_prob_vals)) if play_prob_vals.size else 0.0,
                        "p10": float(np.percentile(play_prob_vals, 10)) if play_prob_vals.size else 0.0,
                        "p50": float(np.percentile(play_prob_vals, 50)) if play_prob_vals.size else 0.0,
                        "p90": float(np.percentile(play_prob_vals, 90)) if play_prob_vals.size else 0.0,
                        "n_zero": int(np.sum(play_prob_vals <= 0.0)),
                        "n_one": int(np.sum(play_prob_vals >= 1.0)),
                    }

                # Rotation (meaningful minutes) rate across worlds.
                if all_minutes is not None:
                    sim_p_rotation = (
                        (all_minutes >= float(ROTATION_THRESHOLD_MINUTES)).mean(axis=0, dtype=float).astype(float)
                    )
                else:
                    sim_p_rotation = np.zeros(n_players, dtype=float)

                # Build output projection DataFrame
                dk_fpts_mean_target = (
                    pd.to_numeric(mu_df["dk_fpts_mean"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                )
                proj_df = mu_df[["game_date", "game_id", "team_id", "player_id"]].copy()
                # Expose play_prob used for world activation so downstream audits can validate
                # unconditional mean targets (E[stat] = E[stat|plays] * play_prob).
                if "play_prob" in mu_df.columns:
                    proj_df["play_prob"] = play_prob_arr
                # Simulation availability input and regime diagnostics.
                proj_df["play_prob_raw"] = play_prob_raw
                proj_df["play_prob_eff"] = play_prob_eff
                proj_df["sim_p_available"] = (avail_counts_total / float(max(1, n_worlds_total))).astype(float)
                proj_df["sim_p_rotation"] = sim_p_rotation
                proj_df["bench_zero_p_zero"] = bench_zero_p_zero
                proj_df["bench_zero_threshold_minutes"] = (
                    float(bench_zero_threshold) if bench_zero_threshold is not None else float("nan")
                )
                if rotation_lock_mask is not None:
                    proj_df["rotation_lock"] = rotation_lock_mask.astype(bool)
                else:
                    proj_df["rotation_lock"] = False
                if policy_reason_arr is not None:
                    proj_df["play_prob_policy_reason"] = policy_reason_arr.astype(str)
                else:
                    proj_df["play_prob_policy_reason"] = "n/a"
                proj_df["minutes_mean"] = minutes_sim_base
                proj_df["minutes_sim_mean"] = minutes_mean
                proj_df["minutes_sim_std"] = minutes_std
                if minutes_quantiles is not None:
                    proj_df["minutes_sim_p10"] = minutes_quantiles[0]
                    proj_df["minutes_sim_p50"] = minutes_quantiles[1]
                    proj_df["minutes_sim_p90"] = minutes_quantiles[2]
                    # PR5 contract: conditional minutes quantiles (given plays)
                    proj_df["minutes_p10_cond"] = minutes_quantiles[0]
                    proj_df["minutes_p50_cond"] = minutes_quantiles[1]
                    proj_df["minutes_p90_cond"] = minutes_quantiles[2]
                # Unconditional minutes summaries (DNP => 0)
                if minutes_mean_uncond is not None and minutes_std_uncond is not None:
                    proj_df["minutes_sim_mean_uncond"] = minutes_mean_uncond
                    proj_df["minutes_sim_std_uncond"] = minutes_std_uncond
                if minutes_quantiles_uncond is not None:
                    proj_df["minutes_sim_p10_uncond"] = minutes_quantiles_uncond[0]
                    proj_df["minutes_sim_p50_uncond"] = minutes_quantiles_uncond[1]
                    proj_df["minutes_sim_p90_uncond"] = minutes_quantiles_uncond[2]
                    # PR5 contract: unconditional minutes quantiles (includes DNP zeros)
                    # These are the primary display columns per spec (decision-relevant)
                    proj_df["minutes_p10"] = minutes_quantiles_uncond[0]
                    proj_df["minutes_p50"] = minutes_quantiles_uncond[1]
                    proj_df["minutes_p90"] = minutes_quantiles_uncond[2]
                proj_df["dk_fpts_mean_target"] = dk_fpts_mean_target
                proj_df["dk_fpts_mean"] = fpts_mean
                proj_df["dk_fpts_std"] = fpts_std
                proj_df["dk_fpts_p05"] = fpts_quantiles[0]
                proj_df["dk_fpts_p10"] = fpts_quantiles[1]
                proj_df["dk_fpts_p25"] = fpts_quantiles[2]
                proj_df["dk_fpts_p50"] = fpts_quantiles[3]
                proj_df["dk_fpts_p75"] = fpts_quantiles[4]
                proj_df["dk_fpts_p90"] = fpts_quantiles[5]
                proj_df["dk_fpts_p95"] = fpts_quantiles[6]
                # Unconditional FPTS summaries (DNP => 0)
                proj_df["dk_fpts_mean_uncond"] = fpts_mean_uncond
                proj_df["dk_fpts_std_uncond"] = fpts_std_uncond
                proj_df["dk_fpts_p05_uncond"] = fpts_quantiles_uncond[0]
                proj_df["dk_fpts_p10_uncond"] = fpts_quantiles_uncond[1]
                proj_df["dk_fpts_p25_uncond"] = fpts_quantiles_uncond[2]
                proj_df["dk_fpts_p50_uncond"] = fpts_quantiles_uncond[3]
                proj_df["dk_fpts_p75_uncond"] = fpts_quantiles_uncond[4]
                proj_df["dk_fpts_p90_uncond"] = fpts_quantiles_uncond[5]
                proj_df["dk_fpts_p95_uncond"] = fpts_quantiles_uncond[6]
                proj_df["sim_p_active"] = active_rate_sim
                proj_df["sim_profile"] = profile_cfg.name
                proj_df["n_worlds"] = n_worlds_eff
                proj_df["minutes_run_id"] = minutes_run_eff
                proj_df["rates_run_id"] = rates_run_eff
                
                # Add individual stat means for dashboard diagnostics (CONDITIONAL)
                with np.errstate(divide='ignore', invalid='ignore'):
                    for stat_name in ("pts", "reb", "ast", "stl", "blk", "tov"):
                        if stat_name in stat_world_samples and stat_world_samples[stat_name]:
                            all_stat = np.vstack(stat_world_samples[stat_name])
                            # Sanitize inf/nan values
                            stat_inf_mask = ~np.isfinite(all_stat)
                            if stat_inf_mask.any():
                                all_stat = np.where(stat_inf_mask, 0.0, all_stat)
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
                    with np.errstate(divide='ignore', invalid='ignore'):
                        for stat_name in ("fga2", "fga3", "fta"):
                            if stat_name in stat_world_samples and stat_world_samples[stat_name]:
                                all_stat = np.vstack(stat_world_samples[stat_name])
                                # Sanitize inf/nan values
                                stat_inf_mask = ~np.isfinite(all_stat)
                                if stat_inf_mask.any():
                                    all_stat = np.where(stat_inf_mask, 0.0, all_stat)
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

                # Persist small run-scoped diagnostics for debugging/regressions.
                try:
                    metrics_payload = {
                        "game_date": pd.Timestamp(game_date).date().isoformat(),
                        "sim_profile": profile_cfg.name,
                        "n_worlds": int(n_worlds_eff),
                        "minutes_run_id": minutes_run_eff,
                        "rates_run_id": rates_run_eff,
                        "minutes_column_used": minutes_col,
                        "preserve_input_rotation": getattr(profile_cfg, 'preserve_input_rotation', False),
                    }
                    metrics_payload.update(minutes_alloc_metrics)
                    if minutes_alloc_summary:
                        metrics_payload["minutes_allocator"] = minutes_alloc_summary
                    if "worlds_integrity_payload" in locals():
                        metrics_payload["worlds_integrity"] = worlds_integrity_payload
                    (out_dir / "metrics.json").write_text(
                        json.dumps(metrics_payload, indent=2),
                        encoding="utf-8",
                    )
                except Exception as exc:
                    typer.echo(f"[sim_v2] warning: failed to write metrics.json ({exc})", err=True)

                # Write sim_manifest.json with full provenance for audit/reproducibility.
                try:
                    import subprocess
                    git_commit = None
                    try:
                        git_commit = subprocess.check_output(
                            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
                        ).strip()[:12]
                    except Exception:
                        pass

                    manifest_payload = {
                        "date": pd.Timestamp(game_date).date().isoformat(),
                        "run_id": sim_run_id or "default",
                        "profile": profile_cfg.name,
                        "seed": int(seed_eff),
                        "n_worlds": int(n_worlds_eff),
                        "chunk_size": int(worlds_per_chunk),
                        "minutes_run_id": minutes_run_eff,
                        "minutes_path": str(minutes_path),
                        "rates_run_id": rates_run_eff,
                        "rates_path": str(rates_path) if rates_path else None,
                        "mean_source": mean_source,
                        "use_rates_noise": use_rates_noise_eff,
                        "use_minutes_noise": use_minutes_noise_eff,
                        "noise": {
                            "epsilon_dist": epsilon_dist,
                            "nu": float(nu),
                            "k_default": float(k_default),
                        },
                        "team_factor_sigma": float(team_factor_sigma_eff),
                        "team_sigma_scale": float(team_sigma_scale_eff),
                        "player_sigma_scale": float(player_sigma_scale_eff),
                        "game_scripts": use_game_scripts,
                        "play_prob_masking": getattr(profile_cfg, 'use_play_prob_masking', True),
                        "preserve_input_rotation": getattr(profile_cfg, 'preserve_input_rotation', False),
                        "git_commit": git_commit,
                    }
                    # Add minutes_noise_config if present
                    if getattr(profile_cfg, 'minutes_noise_config', None) is not None:
                        mnc = profile_cfg.minutes_noise_config
                        manifest_payload["minutes_noise_config"] = {
                            "enabled": getattr(mnc, 'enabled', False),
                            "sigma_starter": getattr(mnc, 'sigma_starter', None),
                            "sigma_bench": getattr(mnc, 'sigma_bench', None),
                            "min_minutes_for_noise": getattr(mnc, 'min_minutes_for_noise', None),
                            "min_minutes_for_noise_override": getattr(mnc, "min_minutes_for_noise_override", None),
                            "cap_abs": getattr(mnc, 'cap_abs', None),
                            "use_student_t": getattr(mnc, "use_student_t", None),
                            "t_df": getattr(mnc, "t_df", None),
                            "include_tail_in_projection": getattr(mnc, "include_tail_in_projection", None),
                            "tail_min_adjustable_minutes": getattr(mnc, "tail_min_adjustable_minutes", None),
                            "lo_source": getattr(mnc, "lo_source", None),
                            "hi_source": getattr(mnc, "hi_source", None),
                            "lo_pad": getattr(mnc, "lo_pad", None),
                            "hi_pad": getattr(mnc, "hi_pad", None),
                        }
                    (out_dir / "sim_manifest.json").write_text(
                        json.dumps(manifest_payload, indent=2),
                        encoding="utf-8",
                    )
                except Exception as exc:
                    typer.echo(f"[sim_v2] warning: failed to write sim_manifest.json ({exc})", err=True)

                # === SIM DIAGNOSTICS: active counts, minute comparisons, guardrails ===
                try:
                    # Compute per-team-game diagnostics
                    team_game_diag = []
                    for (gid, tid), player_idxs in group_map.items():
                        player_idxs_arr = np.array(player_idxs, dtype=int)
                        n_players_team = len(player_idxs_arr)
                        
                        # Active counts per player (worlds where active)
                        team_active_counts = active_counts[player_idxs_arr]
                        min_active_counts = int(team_active_counts.min()) if len(team_active_counts) else 0
                        
                        # Per-world: count active players in team
                        team_active_per_world = all_active[:, player_idxs_arr].sum(axis=1)  # (n_worlds,)
                        pct_worlds_zero_active = float((team_active_per_world == 0).sum()) / n_worlds_total * 100
                        
                        # Minutes sums
                        team_minutes_sim = minutes_mean[player_idxs_arr]
                        team_minutes_input = gs_minutes_p50[player_idxs_arr]
                        
                        team_game_diag.append({
                            "game_id": int(gid),
                            "team_id": int(tid),
                            "n_players": n_players_team,
                            "n_active_mean": float(team_active_per_world.mean()),
                            "sum_minutes_input": float(team_minutes_input.sum()),
                            "sum_minutes_sim_mean": float(team_minutes_sim.sum()),
                            "min_active_counts_player": min_active_counts,
                            "pct_worlds_zero_active": round(pct_worlds_zero_active, 4),
                        })
                    
                    # Global minute comparisons
                    max_minutes_input = float(gs_minutes_p50.max()) if len(gs_minutes_p50) else 0.0
                    max_minutes_sim = float(minutes_mean.max()) if len(minutes_mean) else 0.0
                    p95_minutes_input = float(np.percentile(gs_minutes_p50, 95)) if len(gs_minutes_p50) else 0.0
                    p95_minutes_sim = float(np.percentile(minutes_mean, 95)) if len(minutes_mean) else 0.0
                    p99_minutes_input = float(np.percentile(gs_minutes_p50, 99)) if len(gs_minutes_p50) else 0.0
                    p99_minutes_sim = float(np.percentile(minutes_mean, 99)) if len(minutes_mean) else 0.0
                    
                    # Per-team scale factor
                    team_scale_factors = []
                    for tg in team_game_diag:
                        if tg["sum_minutes_input"] > 0:
                            scale = tg["sum_minutes_sim_mean"] / tg["sum_minutes_input"]
                            team_scale_factors.append(scale)
                    
                    # Guardrail checks
                    any_zero_active = any(tg["pct_worlds_zero_active"] > 0 for tg in team_game_diag)
                    max_minutes_exceeded = max_minutes_sim > max_minutes_input + 2.0
                    p95_exceeded = p95_minutes_sim > p95_minutes_input + 1.0
                    
                    diagnostics_payload = {
                        "game_date": pd.Timestamp(game_date).date().isoformat(),
                        "sim_profile": profile_cfg.name,
                        "n_worlds": int(n_worlds_eff),
                        "minutes_column_used": minutes_col,
                        "preserve_input_rotation": getattr(profile_cfg, 'preserve_input_rotation', False),
                        "global_minutes": {
                            "max_input_p50": round(max_minutes_input, 2),
                            "max_sim_mean": round(max_minutes_sim, 2),
                            "p95_input_p50": round(p95_minutes_input, 2),
                            "p95_sim_mean": round(p95_minutes_sim, 2),
                            "p99_input_p50": round(p99_minutes_input, 2),
                            "p99_sim_mean": round(p99_minutes_sim, 2),
                        },
                        "team_scale_factors": {
                            "min": round(min(team_scale_factors), 4) if team_scale_factors else None,
                            "max": round(max(team_scale_factors), 4) if team_scale_factors else None,
                            "mean": round(float(np.mean(team_scale_factors)), 4) if team_scale_factors else None,
                        },
                        "guardrails": {
                            "any_zero_active_worlds": any_zero_active,
                            "max_minutes_exceeded_threshold": max_minutes_exceeded,
                            "p95_exceeded_threshold": p95_exceeded,
                        },
                        "team_game_details": team_game_diag,
                    }
                    
                    (out_dir / "sim_diagnostics.json").write_text(
                        json.dumps(diagnostics_payload, indent=2),
                        encoding="utf-8",
                    )
                    
                    # Guardrail logging/failure
                    if any_zero_active:
                        fail_hard = os.environ.get("PROJECTIONS_SIM_FAIL_HARD", "0") == "1"
                        teams_with_zero = [tg for tg in team_game_diag if tg["pct_worlds_zero_active"] > 0]
                        msg = f"[sim_v2] WARNING: {len(teams_with_zero)} team-games have worlds with zero active players"
                        typer.echo(msg, err=True)
                        if fail_hard:
                            raise RuntimeError(msg)
                    
                    if max_minutes_exceeded:
                        typer.echo(
                            f"[sim_v2] WARNING: max sim minutes ({max_minutes_sim:.1f}) exceeds input ({max_minutes_input:.1f}) + 2.0",
                            err=True,
                        )
                    
                    if p95_exceeded:
                        typer.echo(
                            f"[sim_v2] WARNING: p95 sim minutes ({p95_minutes_sim:.1f}) exceeds input ({p95_minutes_input:.1f}) + 1.0",
                            err=True,
                        )
                    
                except Exception as exc:
                    typer.echo(f"[sim_v2] warning: failed to write sim_diagnostics.json ({exc})", err=True)

                # Also persist the full per-player worlds matrix for downstream consumers
                # (e.g., contest simulation). This is much smaller than writing one parquet
                # per world and keeps the fast in-memory aggregation path.
                # NOTE: We use the sanitized `all_fpts` which has inf/nan values replaced with 0.0,
                # rather than re-stacking `world_fpts_samples` which contains raw unsanitized values.
                if world_fpts_samples:
                    try:
                        player_ids = mu_df["player_id"].astype(str).tolist()
                        # Use sanitized all_fpts (already stacked and inf-sanitized) instead of raw samples
                        worlds_matrix = all_fpts.astype(np.float32, copy=True)
                        worlds_path = out_dir / "worlds_matrix.parquet"
                        pd.DataFrame(worlds_matrix, columns=player_ids).to_parquet(worlds_path, index=False)
                    except Exception as exc:
                        typer.echo(f"[sim_v2] warning: failed to write worlds_matrix.parquet ({exc})", err=True)

                # Optional: persist minutes worlds matrix for audits/invariant checks.
                # Kept behind an env var to avoid expanding default artifact footprints.
                if all_minutes is not None and os.environ.get("PROJECTIONS_SIM_WRITE_MINUTES_MATRIX", "0").strip().lower() in {
                    "1",
                    "true",
                    "yes",
                }:
                    try:
                        player_ids = mu_df["player_id"].astype(str).tolist()
                        minutes_matrix = all_minutes.astype(np.float32, copy=True)
                        minutes_path = out_dir / "minutes_matrix.parquet"
                        pd.DataFrame(minutes_matrix, columns=player_ids).to_parquet(minutes_path, index=False)
                    except Exception as exc:
                        typer.echo(f"[sim_v2] warning: failed to write minutes_matrix.parquet ({exc})", err=True)
                
                typer.echo(
                    f"[sim_v2] {pd.Timestamp(game_date).date()} dk_fpts_world min/med/max="
                    f"{all_fpts.min():.2f}/{np.median(all_fpts):.2f}/{all_fpts.max():.2f} "
                    f"mean={all_fpts.mean():.2f} -> {proj_path}"
                )
        return

    raise ValueError(
        f"Unsupported sim_v2 mean_source={mean_source!r}; FPTS-only path has been removed."
    )


if __name__ == "__main__":
    app()
