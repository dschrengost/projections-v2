from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SigmoidCalibrator:
    """Platt-style sigmoid calibrator on probability logits.

    This class is intentionally defined in an importable module (not a script) so
    it can be safely pickled/unpickled via joblib for production scoring.
    """

    coef: float
    intercept: float
    eps: float = 1e-6

    def transform(self, p_raw: np.ndarray) -> np.ndarray:
        p = np.asarray(p_raw, dtype=np.float64)
        p = np.clip(p, float(self.eps), 1.0 - float(self.eps))
        x = np.log(p / (1.0 - p))
        z = float(self.coef) * x + float(self.intercept)
        return 1.0 / (1.0 + np.exp(-z))


__all__ = ["SigmoidCalibrator"]

