"""Per-team L2 reconciliation for minutes_v1 quantiles."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import yaml

from projections.math.qp_solvers import QPProblem, QPSolverError, solve_qp

LOGGER = logging.getLogger(__name__)


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

    team_minutes: TeamMinutesConfig = field(default_factory=TeamMinutesConfig)
    p_play_min_rotation: float = 0.05
    min_minutes_for_rotation: float = 4.0
    max_rotation_size: int | None = 10
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
    return ReconcileConfig(
        team_minutes=team_minutes,
        p_play_min_rotation=float(root.get("p_play_min_rotation", 0.05)),
        min_minutes_for_rotation=float(root.get("min_minutes_for_rotation", 4.0)),
        max_rotation_size=int(root["max_rotation_size"]) if "max_rotation_size" in root else 10,
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
    return pd.Series(0.0, index=df.index)


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
    if config.max_rotation_size is None or config.max_rotation_size <= 0:
        return base_mask
    # Enforce a hard cap on rotation size: always keep starters, then highest minutes_p50.
    mask = base_mask.to_numpy()
    if mask.sum() <= config.max_rotation_size:
        return base_mask
    # Keep starters, fill remaining slots by descending minutes.
    minutes_arr = minutes.to_numpy()
    if "rotation_prob" in df.columns:
        rotation_prob_series = pd.to_numeric(df["rotation_prob"], errors="coerce").fillna(0.0)
    else:
        rotation_prob_series = pd.Series(0.0, index=df.index, dtype=float)
    rotation_prob_arr = rotation_prob_series.to_numpy(dtype=float)
    starter_idx = np.where(starters.to_numpy())[0]
    keep = set(starter_idx.tolist())
    # Candidates excluding starters
    # Prefer higher rotation_prob when available to reduce churn from deep bench ghosts.
    composite = rotation_prob_arr * 1000.0 + minutes_arr
    candidate_idx = [i for i in np.argsort(-composite, kind="mergesort") if i not in keep]
    slots = config.max_rotation_size - len(keep)
    for idx in candidate_idx:
        if slots <= 0:
            break
        keep.add(int(idx))
        slots -= 1
    capped_mask = np.zeros_like(mask, dtype=bool)
    capped_mask[list(keep)] = True
    return pd.Series(capped_mask, index=df.index)


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

    reconciled, _ = _solve_team_qp(df_team, config=config)
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
        reconciled, updated = _solve_team_qp(group, config=config)
        reconciled_values.loc[group.index] = reconciled
        if debug_hook and updated:
            debug_payload = _team_debug_payload(group, reconciled, config)
            if debug_payload:
                debug_hook(debug_payload)
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
