"""Per-team L2 reconciliation for minutes_v1 quantiles."""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Literal

import numpy as np
import pandas as pd
import yaml

from projections.math.qp_solvers import QPProblem, QPSolverError, solve_qp
from projections.models.rotalloc import waterfill_redistribute

LOGGER = logging.getLogger(__name__)

ReconcileMethod = Literal["qp", "weighted"]

# Optional env var overrides (opt-in; safe defaults remain config-driven).
ENV_L2_MIN_ACTIVE = "MINUTES_L2_MIN_ACTIVE"
ENV_L2_MAX_ACTIVE = "MINUTES_L2_MAX_ACTIVE"
ENV_L2_MASS_CUTOFF = "MINUTES_L2_MASS_CUTOFF"


@dataclass(slots=True)
class TeamMinutesConfig:
    target: float = 240.0
    tolerance: float = 0.0


@dataclass(slots=True)
class BoundsConfig:
    starter_floor: float = 16.0
    p90_cap_multiplier: float = 1.10
    max_extra_minutes_above_p50: float = 10.0
    hard_cap: float = 44.0


@dataclass(slots=True)
class WeightsConfig:
    starter_penalty: float = 1.0
    rotation_penalty: float = 0.5
    deep_penalty: float = 0.1
    spread_epsilon: float = 0.5
    scale_with_spread: bool = True


@dataclass(slots=True)
class ReconcileConfig:
    """Top-level configuration for the L2 reconciliation layer."""

    method: ReconcileMethod = "weighted"
    team_minutes: TeamMinutesConfig = field(default_factory=TeamMinutesConfig)
    p_play_min_rotation: float = 0.05
    min_minutes_for_rotation: float = 4.0
    min_rotation_size: int = 8
    max_rotation_size: int | None = 13
    rotation_mass_cutoff: float = 0.995
    bounds: BoundsConfig = field(default_factory=BoundsConfig)
    weights: WeightsConfig = field(default_factory=WeightsConfig)
    clamp_tails: bool = True


@dataclass(slots=True)
class TeamReconcileDebug:
    game_id: int | str | None
    team_id: int | str | None
    pre_total: float
    post_total: float
    top_deltas: list[dict[str, float | int | str | None]]


def load_reconcile_config(path: Path | str) -> ReconcileConfig:
    """Load `ReconcileConfig` from YAML."""

    config_path = Path(path).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"L2 reconcile config missing at {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    root = payload.get("l2_reconcile", payload)
    team_minutes = TeamMinutesConfig(**root.get("team_minutes", {}))
    bounds = BoundsConfig(**root.get("bounds", {}))
    weights = WeightsConfig(**root.get("weights", {}))

    method_raw = str(root.get("method", "weighted")).strip().lower()
    method: ReconcileMethod = "weighted"
    if method_raw == "qp":
        method = "qp"

    min_rotation_size = int(root.get("min_rotation_size", 8))
    if min_rotation_size <= 0:
        min_rotation_size = 8

    if "max_rotation_size" in root:
        raw_max = root.get("max_rotation_size")
        if raw_max in (None, "", "none", "null", 0, False):
            max_rotation_size = None
        else:
            max_rotation_size = int(raw_max)
    else:
        max_rotation_size = 13

    rotation_mass_cutoff = float(root.get("rotation_mass_cutoff", 0.995))
    rotation_mass_cutoff = float(np.clip(rotation_mass_cutoff, 0.0, 1.0))

    # Env var overrides (opt-in only).
    env_min = os.environ.get(ENV_L2_MIN_ACTIVE)
    if env_min:
        min_rotation_size = max(1, int(env_min))
    env_max = os.environ.get(ENV_L2_MAX_ACTIVE)
    if env_max:
        max_rotation_size = int(env_max)
    env_cutoff = os.environ.get(ENV_L2_MASS_CUTOFF)
    if env_cutoff:
        rotation_mass_cutoff = float(np.clip(float(env_cutoff), 0.0, 1.0))

    return ReconcileConfig(
        method=method,
        team_minutes=team_minutes,
        p_play_min_rotation=float(root.get("p_play_min_rotation", 0.05)),
        min_minutes_for_rotation=float(root.get("min_minutes_for_rotation", 4.0)),
        min_rotation_size=int(min_rotation_size),
        max_rotation_size=max_rotation_size,
        rotation_mass_cutoff=float(rotation_mass_cutoff),
        bounds=bounds,
        weights=weights,
        clamp_tails=bool(root.get("clamp_tails", True)),
    )


def _starter_series(df: pd.DataFrame) -> pd.Series:
    for column in ("is_projected_starter", "starter_flag", "starter_flag_label"):
        if column in df.columns:
            values = df[column]
            if pd.api.types.is_bool_dtype(values):
                return values.fillna(False)
            return values.fillna(0).astype(int).astype(bool)
    return pd.Series(False, index=df.index)


def _probability_series(df: pd.DataFrame) -> pd.Series:
    for column in ("p_play", "play_prob", "play_probability"):
        if column in df.columns:
            return pd.to_numeric(df[column], errors="coerce").fillna(0.0)
    # If we don't have a probability head, assume active.
    return pd.Series(1.0, index=df.index, dtype=float)


def _rotation_mask(df: pd.DataFrame, config: ReconcileConfig) -> pd.Series:
    starters = _starter_series(df)
    probs = _probability_series(df)
    minutes = pd.to_numeric(df.get("minutes_p50", 0.0), errors="coerce").fillna(0.0)
    rotation = (probs >= config.p_play_min_rotation) & (
        minutes >= config.min_minutes_for_rotation
    )
    base_mask = rotation | starters
    if not base_mask.any():
        return base_mask

    # If no max cap is configured, preserve all rotation candidates.
    if config.max_rotation_size is None or config.max_rotation_size <= 0:
        return base_mask

    base_mask_arr = base_mask.to_numpy(dtype=bool)
    starter_arr = starters.to_numpy(dtype=bool)
    starter_idx = np.flatnonzero(base_mask_arr & starter_arr)

    max_k = int(config.max_rotation_size)
    min_k = max(int(config.min_rotation_size), int(starter_idx.size))
    min_k = min(min_k, max_k)

    # If we have too few candidates, widen the pool to meet min_k (prefer higher p_play).
    if int(base_mask_arr.sum()) < min_k:
        active = np.asarray(probs.to_numpy(dtype=float) > 0.0, dtype=bool)
        candidate_idx = np.flatnonzero(active & ~base_mask_arr)
        if candidate_idx.size:
            player_id = pd.to_numeric(
                df.get("player_id", pd.Series(np.arange(len(df)), index=df.index)),
                errors="coerce",
            ).fillna(0.0).to_numpy(dtype=float)
            order = np.lexsort((player_id[candidate_idx], -probs.to_numpy(dtype=float)[candidate_idx]))
            for idx in candidate_idx[order]:
                if int(base_mask_arr.sum()) >= min_k:
                    break
                base_mask_arr[int(idx)] = True

    if int(base_mask_arr.sum()) <= max_k and int(base_mask_arr.sum()) >= min_k:
        return pd.Series(base_mask_arr, index=df.index)

    # Adaptive depth selection: choose k in [min_k, max_k] based on a cumulative mass cutoff.
    minutes_arr = minutes.to_numpy(dtype=float)
    weights = np.where(base_mask_arr, np.maximum(minutes_arr, 0.0), 0.0)

    # If minutes are flat/degenerate, prefer a deterministic rotation-rank tie-breaker.
    rank_w = None
    if "team_roll_mean_10_rank" in df.columns:
        rank = pd.to_numeric(df["team_roll_mean_10_rank"], errors="coerce").fillna(np.inf).to_numpy(dtype=float)
        rank_w = np.where(np.isfinite(rank) & (rank > 0.0), 1.0 / rank, 0.0)
    elif "team_roll_mean_10_rank_pct" in df.columns:
        rank_pct = pd.to_numeric(df["team_roll_mean_10_rank_pct"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
        rank_w = np.clip(1.0 - rank_pct, 0.0, 1.0)
    elif "rotation_prob" in df.columns:
        rank_w = pd.to_numeric(df["rotation_prob"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    eligible_weights = weights[base_mask_arr]
    mean_w = float(np.mean(eligible_weights)) if eligible_weights.size else 0.0
    std_w = float(np.std(eligible_weights)) if eligible_weights.size else 0.0
    flat = mean_w <= 1e-9 or (std_w / max(mean_w, 1e-9)) < 0.02
    if flat and rank_w is not None and float(np.sum(rank_w[base_mask_arr])) > 0.0:
        weights = weights * np.maximum(rank_w, 1e-6)

    base_idx = np.flatnonzero(base_mask_arr)
    player_id = pd.to_numeric(
        df.get("player_id", pd.Series(np.arange(len(df)), index=df.index)),
        errors="coerce",
    ).fillna(0.0).to_numpy(dtype=float)
    order = np.lexsort((player_id[base_idx], -weights[base_idx]))
    ordered = base_idx[order]

    total_mass = float(np.sum(weights[ordered]))
    if not math.isfinite(total_mass) or total_mass <= 0.0:
        # Fall back to deterministic top-k by (rotation_prob, minutes).
        if "rotation_prob" in df.columns:
            rotprob = pd.to_numeric(df["rotation_prob"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        else:
            rotprob = np.zeros(len(df), dtype=float)
        composite = rotprob * 1000.0 + minutes_arr
        order = np.lexsort((player_id[base_idx], -composite[base_idx]))
        ordered = base_idx[order]
        total_mass = float(np.sum(np.maximum(composite[ordered], 0.0)))

    cutoff = float(np.clip(config.rotation_mass_cutoff, 0.0, 1.0))
    target_mass = cutoff * total_mass
    cum = np.cumsum(weights[ordered])
    k_mass = int(np.searchsorted(cum, target_mass, side="left") + 1)
    k = max(min_k, min(k_mass, max_k))

    keep = set(starter_idx.tolist())
    slots = k - len(keep)
    for idx in ordered:
        if slots <= 0:
            break
        if int(idx) in keep:
            continue
        keep.add(int(idx))
        slots -= 1

    capped = np.zeros_like(base_mask_arr, dtype=bool)
    capped[list(keep)] = True
    return pd.Series(capped, index=df.index)


def _allocate_team_weighted(
    df_team: pd.DataFrame,
    *,
    config: ReconcileConfig,
) -> np.ndarray:
    """Deterministic minutes allocator that sums to the team target.

    This is a production-safe alternative to the QP reconciler:
      - selects an adaptive rotation set via `_rotation_mask`
      - allocates minutes proportionally to per-player weights (with fallback)
      - enforces an effective cap via water-fill redistribution
    """
    if df_team.empty:
        return np.zeros(0, dtype=float)

    probs = _probability_series(df_team).to_numpy(dtype=float)
    status_upper = (
        df_team["status"].astype(str).str.upper().fillna("")
        if "status" in df_team.columns
        else pd.Series("", index=df_team.index)
    )
    out_mask = (probs <= 0.0) | status_upper.isin({"OUT", "O", "INACTIVE"}).to_numpy(dtype=bool)

    starters = _starter_series(df_team).to_numpy(dtype=bool)
    rotation_mask = _rotation_mask(df_team, config).to_numpy(dtype=bool) & ~out_mask
    if not rotation_mask.any():
        return np.zeros(len(df_team), dtype=float)

    p50_raw = pd.to_numeric(
        df_team.get("minutes_p50_raw", df_team.get("minutes_p50", 0.0)),
        errors="coerce",
    ).fillna(0.0).to_numpy(dtype=float)

    weights = np.where(rotation_mask, np.maximum(p50_raw, 0.0), 0.0)
    active_w = weights[rotation_mask]
    mean_w = float(np.mean(active_w)) if active_w.size else 0.0
    std_w = float(np.std(active_w)) if active_w.size else 0.0
    flat = mean_w <= 1e-9 or (std_w / max(mean_w, 1e-9)) < 0.02

    if float(np.sum(weights)) <= 1e-12 or flat:
        rank_w = None
        if "team_roll_mean_10_rank" in df_team.columns:
            rank = (
                pd.to_numeric(df_team["team_roll_mean_10_rank"], errors="coerce")
                .fillna(np.inf)
                .to_numpy(dtype=float)
            )
            rank_w = np.where(np.isfinite(rank) & (rank > 0.0), 1.0 / rank, 0.0)
        elif "team_roll_mean_10_rank_pct" in df_team.columns:
            rank_pct = (
                pd.to_numeric(df_team["team_roll_mean_10_rank_pct"], errors="coerce")
                .fillna(1.0)
                .to_numpy(dtype=float)
            )
            rank_w = np.clip(1.0 - rank_pct, 0.0, 1.0)
        elif "rotation_prob" in df_team.columns:
            rank_w = pd.to_numeric(df_team["rotation_prob"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

        if rank_w is not None and float(np.sum(rank_w[rotation_mask])) > 0.0:
            weights = np.where(rotation_mask, np.maximum(rank_w, 0.0), 0.0)
    else:
        # If overall weights are informative but a starter/bench subgroup is flat, use
        # rotation-rank as a deterministic within-group tie-breaker.
        rank_w = None
        if "team_roll_mean_10_rank" in df_team.columns:
            rank = (
                pd.to_numeric(df_team["team_roll_mean_10_rank"], errors="coerce")
                .fillna(np.inf)
                .to_numpy(dtype=float)
            )
            rank_w = np.where(np.isfinite(rank) & (rank > 0.0), 1.0 / rank, 0.0)
        elif "team_roll_mean_10_rank_pct" in df_team.columns:
            rank_pct = (
                pd.to_numeric(df_team["team_roll_mean_10_rank_pct"], errors="coerce")
                .fillna(1.0)
                .to_numpy(dtype=float)
            )
            rank_w = np.clip(1.0 - rank_pct, 0.0, 1.0)
        elif "rotation_prob" in df_team.columns:
            rank_w = pd.to_numeric(df_team["rotation_prob"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

        if rank_w is not None:
            # Only apply tie-breakers within the bench group. Starters already carry
            # stronger minute priors and over-tilting them creates unrealistic 44-minute p50s.
            for group_mask in (rotation_mask & ~starters,):
                group_w = weights[group_mask]
                if group_w.size < 2:
                    continue
                mean_g = float(np.mean(group_w))
                std_g = float(np.std(group_w))
                flat_g = mean_g <= 1e-9 or (std_g / max(mean_g, 1e-9)) < 0.02
                if not flat_g:
                    continue
                group_rank = np.maximum(rank_w[group_mask], 1e-6)
                if float(np.sum(group_rank)) <= 0.0:
                    continue
                # Normalize to preserve group-level scale, then clip to avoid extreme distortions.
                rank_mean = float(np.mean(group_rank))
                factor = group_rank / max(rank_mean, 1e-6)
                factor = np.clip(factor, 0.5, 2.0)
                weights[group_mask] = weights[group_mask] * factor

    target = float(config.team_minutes.target)
    cap_max = float(config.bounds.hard_cap) if math.isfinite(config.bounds.hard_cap) else 44.0

    w_sum = float(np.sum(weights))
    if not math.isfinite(w_sum) or w_sum <= 1e-12:
        minutes = np.zeros(len(df_team), dtype=float)
        minutes[rotation_mask] = target / float(int(rotation_mask.sum()))
        return waterfill_redistribute(
            minutes,
            weights,
            rotation_mask,
            cap_max=cap_max,
            target_sum=target,
        )

    minutes = np.zeros(len(df_team), dtype=float)
    minutes[rotation_mask] = target * (weights[rotation_mask] / w_sum)
    minutes = waterfill_redistribute(
        minutes,
        weights,
        rotation_mask,
        cap_max=cap_max,
        target_sum=target,
    )
    minutes[out_mask] = 0.0
    return minutes


def _compute_lower_bounds(
    df: pd.DataFrame,
    config: ReconcileConfig,
) -> np.ndarray:
    starters = _starter_series(df).to_numpy(dtype=bool)
    p50 = pd.to_numeric(df["minutes_p50"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    floors = np.where(
        starters,
        np.minimum(p50, config.bounds.starter_floor),
        0.0,
    )
    return floors


def _compute_upper_bounds(
    df: pd.DataFrame,
    config: ReconcileConfig,
) -> np.ndarray:
    cap_column = None
    for candidate in ("minutes_cap", "minutes_ceiling"):
        if candidate in df.columns:
            cap_column = pd.to_numeric(df[candidate], errors="coerce")
            break
    if cap_column is not None and cap_column.notna().any():
        caps = cap_column.fillna(np.inf).to_numpy(dtype=float)
    else:
        p90 = (
            pd.to_numeric(df.get("minutes_p90", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        )
        p50 = pd.to_numeric(df["minutes_p50"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        caps = np.minimum(
            np.minimum(
                p90 * config.bounds.p90_cap_multiplier,
                p50 + config.bounds.max_extra_minutes_above_p50,
            ),
            config.bounds.hard_cap,
        )
    return np.maximum(0.0, caps)


def _compute_weights(df: pd.DataFrame, config: ReconcileConfig) -> np.ndarray:
    starters = _starter_series(df).to_numpy(dtype=bool)
    rotation_mask = _rotation_mask(df, config).to_numpy(dtype=bool)
    spreads = (
        pd.to_numeric(df.get("minutes_p90", 0.0), errors="coerce")
        - pd.to_numeric(df.get("minutes_p10", 0.0), errors="coerce")
    ).to_numpy(dtype=float)
    spreads = np.maximum(spreads, config.weights.spread_epsilon)
    base = np.where(
        starters,
        config.weights.starter_penalty,
        np.where(rotation_mask, config.weights.rotation_penalty, config.weights.deep_penalty),
    )
    if config.weights.scale_with_spread:
        scaled = base * (1.0 / np.square(spreads))
    else:
        scaled = base
    return np.clip(scaled, 1e-6, None)


def _team_debug_payload(
    df_team: pd.DataFrame,
    reconciled: np.ndarray,
    config: ReconcileConfig,
) -> TeamReconcileDebug | None:
    if df_team.empty:
        return None
    player_ids = df_team.get("player_id")
    player_names = df_team.get("player_name")
    deltas = np.abs(reconciled - df_team["minutes_p50"].to_numpy(dtype=float))
    order = np.argsort(-deltas)
    top_rows: list[dict[str, float | int | str | None]] = []
    for idx in order[:5]:
        top_rows.append(
            {
                "player_id": None if player_ids is None else player_ids.iloc[idx],
                "player_name": None if player_names is None else player_names.iloc[idx],
                "delta": float(deltas[idx]),
                "minutes_before": float(df_team["minutes_p50"].iloc[idx]),
                "minutes_after": float(reconciled[idx]),
            }
        )
    raw_col = "minutes_p50_raw" if "minutes_p50_raw" in df_team else "minutes_p50"
    pre_total = float(df_team[raw_col].sum())
    post_total = float(np.sum(reconciled))
    return TeamReconcileDebug(
        game_id=df_team.get("game_id").iloc[0] if "game_id" in df_team else None,
        team_id=df_team.get("team_id").iloc[0] if "team_id" in df_team else None,
        pre_total=pre_total,
        post_total=post_total,
        top_deltas=top_rows,
    )


def _solve_team_qp(
    df_team: pd.DataFrame,
    *,
    config: ReconcileConfig,
) -> tuple[np.ndarray, bool]:
    rotation_mask = _rotation_mask(df_team, config).to_numpy(dtype=bool)
    if not rotation_mask.any():
        return df_team["minutes_p50"].to_numpy(dtype=float), False
    decision_df = df_team.loc[rotation_mask].copy()
    mu = decision_df["minutes_p50"].to_numpy(dtype=float)
    weights = _compute_weights(decision_df, config)
    lower = _compute_lower_bounds(decision_df, config)
    upper = _compute_upper_bounds(decision_df, config)
    upper = np.maximum(upper, lower)
    total_lb = float(np.sum(lower))
    total_ub = float(np.sum(upper))
    target = config.team_minutes.target
    if target < total_lb - 1e-3 or target > total_ub + 1e-3:
        LOGGER.warning(
            "L2 reconciliation infeasible for team %s (target=%.1f, bounds=[%.1f, %.1f]).",
            df_team.get("team_id").iloc[0] if "team_id" in df_team else "unknown",
            target,
            total_lb,
            total_ub,
        )
        return df_team["minutes_p50"].to_numpy(dtype=float), False

    if config.team_minutes.tolerance <= 0:
        A_eq = np.ones((1, len(mu)))
        b_eq = np.array([target], dtype=float)
        A_ineq = None
        b_ineq = None
    else:
        tol = config.team_minutes.tolerance
        A_eq = None
        b_eq = None
        A_ineq = np.vstack([np.ones((1, len(mu))), -np.ones((1, len(mu)))])
        b_ineq = np.array([target + tol, -(target - tol)], dtype=float)

    Q = 2.0 * np.diag(weights)
    c = -2.0 * weights * mu
    problem = QPProblem(
        Q=Q,
        c=c,
        A_eq=A_eq,
        b_eq=b_eq,
        A_ineq=A_ineq,
        b_ineq=b_ineq,
        lb=lower,
        ub=upper,
    )
    try:
        solution = solve_qp(problem)
    except QPSolverError as exc:
        LOGGER.warning("QP solver failed for team %s: %s", df_team.get("team_id", "unknown"), exc)
        return df_team["minutes_p50"].to_numpy(dtype=float), False

    reconciled = np.zeros(len(df_team), dtype=float)
    reconciled[rotation_mask] = solution
    return reconciled, True


def reconcile_team_minutes_p50(
    df_team: pd.DataFrame,
    config: ReconcileConfig,
) -> pd.Series:
    """Reconcile one team slice."""

    if config.method == "qp":
        reconciled, _ = _solve_team_qp(df_team, config=config)
    else:
        reconciled = _allocate_team_weighted(df_team, config=config)
    return pd.Series(reconciled, index=df_team.index)


def reconcile_minutes_p50_all(
    df: pd.DataFrame,
    config: ReconcileConfig,
    *,
    debug_hook: Callable[[TeamReconcileDebug], None] | None = None,
    group_cols: Iterable[str] = ("game_id", "team_id"),
) -> pd.DataFrame:
    """Apply reconciliation to each team in `df` and return a copy."""

    if df.empty:
        return df
    working = df.copy()
    if "minutes_p50_raw" not in working.columns:
        working["minutes_p50_raw"] = working["minutes_p50"]
    reconciled_values = pd.Series(index=working.index, dtype=float)
    for key, group in working.groupby(list(group_cols), sort=False):
        if config.method == "qp":
            reconciled, updated = _solve_team_qp(group, config=config)
            reconciled_values.loc[group.index] = reconciled
            if debug_hook and updated:
                debug_payload = _team_debug_payload(group, reconciled, config)
                if debug_payload:
                    debug_hook(debug_payload)
        else:
            reconciled = _allocate_team_weighted(group, config=config)
            reconciled_values.loc[group.index] = reconciled
    working["minutes_p50"] = reconciled_values.values
    working["minutes_p50_cond"] = working["minutes_p50"]
    if config.clamp_tails:
        _enforce_monotonic_quantiles(working)
    return working


def _enforce_monotonic_quantiles(df: pd.DataFrame) -> None:
    if {"minutes_p10", "minutes_p50"}.issubset(df.columns):
        df["minutes_p10"] = np.minimum(
            pd.to_numeric(df["minutes_p10"], errors="coerce").fillna(0.0),
            df["minutes_p50"],
        )
        df["minutes_p10_cond"] = df["minutes_p10"]
    if {"minutes_p90", "minutes_p50"}.issubset(df.columns):
        df["minutes_p90"] = np.maximum(
            pd.to_numeric(df["minutes_p90"], errors="coerce").fillna(0.0),
            df["minutes_p50"],
        )
        df["minutes_p90_cond"] = df["minutes_p90"]
