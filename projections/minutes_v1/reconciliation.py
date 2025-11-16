"""L2 reconciliation utilities for team-level minutes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .constants import ROLE_MINUTES_CAPS


@dataclass
class ReconciliationConfig:
    """Configuration for the L2 projection step."""

    target_total: float = 240.0
    quantile_width_col: str = "quantile_width"
    minutes_col: str = "p50"
    starter_col: str = "starter_flag"
    ramp_col: str = "ramp_flag"
    blowout_index_col: str = "blowout_index"
    alpha: float = 0.1  # starter reduction on blowouts
    beta: float = 0.08  # bench boost on blowouts


def _minutes_caps(row: pd.Series) -> float:
    if bool(row.get("ramp_flag")):
        return ROLE_MINUTES_CAPS["ramp"]
    if bool(row.get("starter_flag")):
        return ROLE_MINUTES_CAPS["starter"]
    return ROLE_MINUTES_CAPS["bench"]


def _weights_from_width(width: np.ndarray) -> np.ndarray:
    variance = np.square(np.maximum(width, 1.0) / 2.0)
    return 1.0 / (1.0 + variance)


def _apply_blowout_adjustment(
    minutes: np.ndarray,
    starter_mask: np.ndarray,
    blowout_index: float | None,
    alpha: float,
    beta: float,
) -> np.ndarray:
    if blowout_index is None or np.isnan(blowout_index) or blowout_index < 1.5:
        return minutes
    intensity = min(blowout_index / 3.0, 1.0)
    adjusted = minutes.copy()
    adjusted[starter_mask] *= 1 - alpha * intensity
    adjusted[~starter_mask] *= 1 + beta * intensity
    return adjusted


def _project_with_bounds(
    m: np.ndarray,
    w: np.ndarray,
    upper: np.ndarray,
    target: float,
) -> np.ndarray:
    lower = np.zeros_like(m)
    x = np.zeros_like(m)
    free = np.ones_like(m, dtype=bool)
    target_remaining = target

    while True:
        if not free.any():
            break
        denom = np.sum(1.0 / w[free])
        if denom == 0:
            break
        delta = (np.sum(m[free]) - target_remaining) / denom
        tentative = m[free] - delta / w[free]
        below = tentative < lower[free]
        above = tentative > upper[free]
        if not below.any() and not above.any():
            x[free] = tentative
            break
        free_indices = np.where(free)[0]
        if below.any():
            for idx in free_indices[below]:
                x[idx] = lower[idx]
                free[idx] = False
                target_remaining -= lower[idx]
        if above.any():
            for idx in free_indices[above]:
                x[idx] = upper[idx]
                free[idx] = False
                target_remaining -= upper[idx]

    if free.any():
        denom = np.sum(1.0 / w[free])
        delta = (np.sum(m[free]) - target_remaining) / denom if denom != 0 else 0.0
        x[free] = m[free] - delta / w[free]
    return x


def reconcile_team_minutes(
    team_df: pd.DataFrame,
    *,
    config: ReconciliationConfig | None = None,
) -> np.ndarray:
    """Reconcile one team-game using L2 projection with caps."""

    cfg = config or ReconciliationConfig()
    if team_df.empty:
        return np.array([])

    minutes = team_df[cfg.minutes_col].to_numpy(dtype=float)
    starter_mask = team_df[cfg.starter_col].astype(bool).to_numpy()
    caps = team_df.apply(_minutes_caps, axis=1).to_numpy(dtype=float)
    blowout_index = team_df[cfg.blowout_index_col].iloc[0] if cfg.blowout_index_col in team_df else None

    if cfg.quantile_width_col in team_df:
        width = team_df[cfg.quantile_width_col].fillna(5.0).to_numpy(dtype=float)
    else:
        width = np.full_like(minutes, 6.0, dtype=float)
    weights = _weights_from_width(width)

    minutes = np.minimum(minutes, caps)
    minutes = _apply_blowout_adjustment(minutes, starter_mask, blowout_index, cfg.alpha, cfg.beta)
    reconciled = _project_with_bounds(minutes, weights, caps, cfg.target_total)
    return reconciled


def reconcile_minutes(
    df: pd.DataFrame,
    *,
    group_cols: Iterable[str] = ("game_id", "team_id"),
    config: ReconciliationConfig | None = None,
    output_column: str = "minutes_reconciled",
) -> pd.DataFrame:
    """Apply reconciliation across the dataframe and append a column with the result."""

    cfg = config or ReconciliationConfig()
    reconciled_frames: list[pd.DataFrame] = []
    for _, group in df.groupby(list(group_cols)):
        group = group.copy()
        group[output_column] = reconcile_team_minutes(group, config=cfg)
        reconciled_frames.append(group)
    return pd.concat(reconciled_frames, ignore_index=True)
