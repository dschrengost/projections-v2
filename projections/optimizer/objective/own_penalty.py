# optimizer/objective/own_penalty.py
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Tuple, Optional
import numpy as np
import pandas as pd

class CurveKind(str, Enum):
    PIECEWISE = "piecewise"
    POWER = "power"
    LOGISTIC = "logistic"

@dataclass(frozen=True)
class OwnershipPenaltyConfig:
    # core penalty
    lambda1: float = 0.35
    pct_scale: float = 1.0
    clip_min: float = 0.0
    clip_max: float = 100.0

    # curve choice
    curve: CurveKind = CurveKind.PIECEWISE

    # --- PIECEWISE params ---
    breaks: Tuple[float, ...] = (10.0, 20.0, 30.0)
    weights: Tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)

    # --- POWER params ---
    # maps own_pct in [0,100] -> weight = (own_pct/100)^gamma * (1+boost)
    power_gamma: float = 1.6         # >1 convex, <1 concave
    power_boost: float = 1.0         # baseline to keep low-own nonzero

    # --- LOGISTIC params ---
    # weight = base + max_bump / (1 + exp(-k*(own - mid)))
    logi_base: float = 0.25
    logi_max_bump: float = 1.75
    logi_k: float = 0.18             # steepness
    logi_mid: float = 22.0           # “knee” in percent

    def validate(self) -> None:
        if self.lambda1 < 0:
            raise ValueError("lambda1 must be >= 0")
        if self.curve == CurveKind.PIECEWISE:
            if len(self.weights) != len(self.breaks) + 1:
                raise ValueError("weights must have len(breaks)+1")
            if sorted(self.breaks) != list(self.breaks):
                raise ValueError("breaks must be ascending")

@dataclass(frozen=True)
class AggroSchedule:
    """
    Controls how aggressive the solver gets across lineup iterations.
    Example: ramp lambda1 from 0.30 to 0.60 between lineups 20 and 80.
    """
    start_lambda: float = 0.30
    max_lambda: float = 0.60
    ramp_start: int = 20
    ramp_end: int = 80

    def lambda_for(self, lineup_idx: int) -> float:
        if lineup_idx < self.ramp_start:
            return self.start_lambda
        if lineup_idx >= self.ramp_end:
            return self.max_lambda
        # linear ramp
        t = (lineup_idx - self.ramp_start) / max(1, (self.ramp_end - self.ramp_start))
        return self.start_lambda + t * (self.max_lambda - self.start_lambda)

def _curve_weights(own_pct: np.ndarray, cfg: OwnershipPenaltyConfig) -> np.ndarray:
    if cfg.curve == CurveKind.PIECEWISE:
        bins = np.digitize(own_pct, bins=np.asarray(cfg.breaks, dtype=float), right=False)
        return np.asarray(cfg.weights, dtype=float)[bins]
    elif cfg.curve == CurveKind.POWER:
        x = np.clip(own_pct / 100.0, 0.0, 1.0)
        return cfg.power_boost + np.power(x, cfg.power_gamma)
    else:  # LOGISTIC
        return cfg.logi_base + cfg.logi_max_bump / (1.0 + np.exp(-cfg.logi_k * (own_pct - cfg.logi_mid)))

def build_ownership_penalty(
    df: pd.DataFrame,
    cfg: OwnershipPenaltyConfig,
    own_col: str = "own",
    lineup_idx: Optional[int] = None,
    aggro: Optional[AggroSchedule] = None,
) -> np.ndarray:
    """
    Returns a FLOAT vector (negative penalty) to add to per-player weights.
    If `aggro` is provided, lambda1 is overridden by aggro.lambda_for(lineup_idx).
    """
    cfg.validate()

    # If ownership column is missing entirely, return zero penalties by contract
    if own_col not in df.columns:
        return np.zeros(len(df), dtype=float)

    own_series = df[own_col].astype(float)
    # Identify missing/NaN per-row and treat as zero-penalty
    mask_missing = ~np.isfinite(own_series.to_numpy())
    own_raw = own_series.fillna(0.0).to_numpy()
    own_pct = np.clip(own_raw * cfg.pct_scale, cfg.clip_min, cfg.clip_max)

    mult = _curve_weights(own_pct, cfg)
    # Zero out weights where ownership is missing
    if mask_missing.any():
        mult = mult.copy()
        mult[mask_missing] = 0.0

    lam = cfg.lambda1
    if aggro is not None and lineup_idx is not None:
        lam = aggro.lambda_for(lineup_idx)

    return -lam * mult
