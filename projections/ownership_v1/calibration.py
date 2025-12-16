"""
Softmax-style calibration for ownership predictions.

Mathematical Foundation
-----------------------
For a slate with N players, each having:
  - s_i: unconstrained score from the model (e.g., predicted ownership %)
  - y_i: actual ownership fraction in [0, 1]

The calibration layer computes:
  w_i = exp(a * s_i + b)
  p_i = w_i / sum_j(w_j)
  ŷ_i = R * p_i

Where:
  - (a, b) are fitted parameters
  - R = slots_per_lineup (8 for DK NBA classic)

This construction guarantees:
  - sum_i(ŷ_i) = R (by construction)
  - Ranking of s_i is preserved (exp is monotonic)
  - a > 0 stretches gaps (higher contrast), a < 0 compresses
  - b shifts the baseline (bias term, typically absorbs mean)

The fitting minimizes MSE: sum_slates sum_i (ŷ_i - y_i)²

Numerical Stability
-------------------
We use log-softmax for numerical stability:
  log_p_i = (a * s_i + b) - logsumexp(a * s + b)
  p_i = exp(log_p_i)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import optimize


@dataclass
class CalibrationParams:
    """Fitted calibration parameters."""
    a: float = 1.0  # Scale factor (stretches/compresses score gaps)
    b: float = 0.0  # Bias term
    R: float = 8.0  # Target sum (slots per lineup)
    
    def to_dict(self) -> dict:
        return {"a": self.a, "b": self.b, "R": self.R}
    
    @classmethod
    def from_dict(cls, d: dict) -> "CalibrationParams":
        return cls(a=d["a"], b=d["b"], R=d.get("R", 8.0))


def _log_softmax(scores: np.ndarray, a: float, b: float) -> np.ndarray:
    """Compute log-softmax with numerical stability via logsumexp trick."""
    z = a * scores + b
    max_z = np.max(z)  # Shift for numerical stability
    log_sum_exp = max_z + np.log(np.sum(np.exp(z - max_z)))
    return z - log_sum_exp


def apply_calibration(
    scores: np.ndarray,
    params: CalibrationParams,
) -> np.ndarray:
    """
    Apply softmax calibration to a slate's scores.
    
    Args:
        scores: Array of model scores s_i for each player in the slate.
        params: Fitted calibration parameters.
    
    Returns:
        Array of calibrated ownership fractions that sum to R.
        (Multiply by 100 to get percent space.)
    """
    if len(scores) == 0:
        return np.array([])
    
    # Compute softmax probabilities
    log_probs = _log_softmax(scores, params.a, params.b)
    probs = np.exp(log_probs)
    
    # Scale by R so sum = R
    calibrated = params.R * probs
    
    return calibrated


def apply_calibration_with_mask(
    scores: np.ndarray,
    mask: np.ndarray,
    params: CalibrationParams,
) -> np.ndarray:
    """
    Apply softmax calibration with structural zero masking.
    
    Players where mask=False are treated as structural zeros:
    - Excluded from softmax normalization
    - Get exactly 0 ownership
    
    The remaining players (mask=True) share the full R allocation.
    
    Args:
        scores: Array of model scores s_i for each player.
        mask: Boolean array where True = include in calibration,
              False = structural zero (set to 0).
        params: Fitted calibration parameters.
    
    Returns:
        Array of calibrated ownership fractions that sum to R.
        Masked-out players have exactly 0.
    """
    if len(scores) == 0:
        return np.array([])
    
    result = np.zeros_like(scores, dtype=float)
    
    if not np.any(mask):
        # All players masked out - return zeros
        return result
    
    # Apply calibration only to non-masked players
    eligible_scores = scores[mask]
    calibrated = apply_calibration(eligible_scores, params)
    
    # Put calibrated values back in result
    result[mask] = calibrated
    
    return result


def _compute_loss(
    params_vec: np.ndarray,
    slate_data: list[tuple[np.ndarray, np.ndarray]],
    R: float,
) -> float:
    """
    Compute MSE loss over all slates.
    
    Args:
        params_vec: [a, b] parameter vector
        slate_data: List of (scores, targets) tuples for each slate
        R: Target sum
    
    Returns:
        Sum of squared errors across all slates and players
    """
    a, b = params_vec
    total_loss = 0.0
    n_total = 0
    
    for scores, targets in slate_data:
        log_probs = _log_softmax(scores, a, b)
        preds = R * np.exp(log_probs)
        total_loss += np.sum((preds - targets) ** 2)
        n_total += len(scores)
    
    # Return mean squared error
    return total_loss / max(n_total, 1)


def fit_calibration(
    df: pd.DataFrame,
    score_col: str = "pred_own_pct",
    target_col: str = "own_pct",
    slate_id_col: str = "slate_id",
    R: float = 8.0,
    initial_a: float = 0.1,
    initial_b: float = 0.0,
    verbose: bool = True,
) -> CalibrationParams:
    """
    Fit calibration parameters on historical data.
    
    Args:
        df: DataFrame with columns [score_col, target_col, slate_id_col].
        score_col: Column name for model scores (s_i).
        target_col: Column name for actual ownership fractions (y_i).
            Expected in percent space (0-100), will be divided by 100.
        slate_id_col: Column to group by for per-slate calibration.
        R: Target sum (slots per lineup, 8 for DK NBA classic).
        initial_a: Starting value for scale parameter.
        initial_b: Starting value for bias parameter.
        verbose: Whether to print fitting progress.
    
    Returns:
        Fitted CalibrationParams.
    """
    # Prepare per-slate data
    slate_data = []
    for slate_id, group in df.groupby(slate_id_col):
        scores = group[score_col].values.astype(float)
        targets = group[target_col].values.astype(float) / 100.0  # Convert to fractions
        
        # Skip slates with missing data
        if np.any(np.isnan(scores)) or np.any(np.isnan(targets)):
            continue
        if len(scores) == 0:
            continue
            
        slate_data.append((scores, targets))
    
    if not slate_data:
        raise ValueError("No valid slates found for fitting")
    
    if verbose:
        print(f"[calibration] Fitting on {len(slate_data)} slates, "
              f"{sum(len(s) for s, _ in slate_data)} total players")
    
    # Initial loss
    initial_params = np.array([initial_a, initial_b])
    initial_loss = _compute_loss(initial_params, slate_data, R)
    
    # Optimize using L-BFGS-B
    result = optimize.minimize(
        _compute_loss,
        initial_params,
        args=(slate_data, R),
        method="L-BFGS-B",
        bounds=[(-10.0, 10.0), (-100.0, 100.0)],  # Reasonable bounds
        options={"maxiter": 1000, "disp": verbose},
    )
    
    a_fit, b_fit = result.x
    final_loss = result.fun
    
    if verbose:
        print(f"[calibration] Fitted: a={a_fit:.4f}, b={b_fit:.4f}")
        print(f"[calibration] MSE: {initial_loss:.6f} -> {final_loss:.6f} "
              f"({(1 - final_loss/initial_loss)*100:.1f}% reduction)")
    
    return CalibrationParams(a=a_fit, b=b_fit, R=R)


@dataclass
class SoftmaxCalibrator:
    """
    Softmax calibrator for ownership predictions.
    
    Combines fitting and application of calibration parameters.
    Supports serialization for production use.
    """
    params: CalibrationParams = field(default_factory=CalibrationParams)
    fitted: bool = False
    
    def fit(
        self,
        df: pd.DataFrame,
        score_col: str = "pred_own_pct",
        target_col: str = "own_pct",
        slate_id_col: str = "slate_id",
        R: float = 8.0,
        verbose: bool = True,
    ) -> "SoftmaxCalibrator":
        """Fit calibration parameters on historical data."""
        self.params = fit_calibration(
            df,
            score_col=score_col,
            target_col=target_col,
            slate_id_col=slate_id_col,
            R=R,
            verbose=verbose,
        )
        self.fitted = True
        return self
    
    def apply(self, scores: np.ndarray) -> np.ndarray:
        """
        Apply calibration to a slate's scores.
        
        Args:
            scores: Model scores for each player (in same units as training).
        
        Returns:
            Calibrated ownership fractions summing to R.
            Multiply by 100 for percent space.
        """
        return apply_calibration(scores, self.params)
    
    def apply_df(
        self,
        df: pd.DataFrame,
        score_col: str = "pred_own_pct",
        output_col: str = "calibrated_own_pct",
    ) -> pd.DataFrame:
        """
        Apply calibration to a DataFrame and return calibrated values.
        
        Assumes the entire DataFrame is a single slate.
        
        Args:
            df: DataFrame with score column.
            score_col: Column name for input scores.
            output_col: Column name for calibrated output (in percent space).
        
        Returns:
            DataFrame with added output column.
        """
        result = df.copy()
        scores = df[score_col].values.astype(float)
        calibrated = self.apply(scores)
        result[output_col] = calibrated * 100.0  # Convert to percent
        return result
    
    def save(self, path: Path | str) -> None:
        """Save calibrator to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "params": self.params.to_dict(),
                "fitted": self.fitted,
            }, f, indent=2)
    
    @classmethod
    def load(cls, path: Path | str) -> "SoftmaxCalibrator":
        """Load calibrator from JSON file."""
        with open(path) as f:
            data = json.load(f)
        calibrator = cls(
            params=CalibrationParams.from_dict(data["params"]),
            fitted=data.get("fitted", True),
        )
        return calibrator
    
    @property
    def a(self) -> float:
        return self.params.a
    
    @property
    def b(self) -> float:
        return self.params.b
    
    @property
    def R(self) -> float:
        return self.params.R


__all__ = [
    "CalibrationParams",
    "SoftmaxCalibrator",
    "apply_calibration",
    "apply_calibration_with_mask",
    "fit_calibration",
]

