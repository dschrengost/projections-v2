from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ProjectionInfeasibleError(ValueError):
    """Raised when bounded projection cannot satisfy the sum constraint."""

    reason: str
    target_sum: float
    sum_lb: float
    sum_ub: float
    n_items: int

    def __str__(self) -> str:
        return (
            f"Projection infeasible: reason={self.reason} target_sum={self.target_sum:.6f} "
            f"sum_lb={self.sum_lb:.6f} sum_ub={self.sum_ub:.6f} n_items={self.n_items}"
        )

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "reason": self.reason,
            "target_sum": float(self.target_sum),
            "sum_lb": float(self.sum_lb),
            "sum_ub": float(self.sum_ub),
            "n_items": int(self.n_items),
        }


def _as_vector(name: str, values: np.ndarray | list[float] | tuple[float, ...], n: int | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if n is not None and arr.shape != (n,):
        raise ValueError(f"{name} must have shape ({n},), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def project_sum_with_bounds(
    x: np.ndarray | list[float] | tuple[float, ...],
    target_sum: float,
    lb: np.ndarray | list[float] | tuple[float, ...],
    ub: np.ndarray | list[float] | tuple[float, ...],
    weights: np.ndarray | list[float] | tuple[float, ...] | None = None,
    *,
    eps: float = 1e-9,
    max_iters: int = 80,
) -> np.ndarray:
    """Project ``x`` onto ``{y: sum(y)=target_sum, lb<=y<=ub}`` with weighted L2 objective.

    Objective:
      minimize sum_i w_i * (y_i - x_i)^2
      subject to lb_i <= y_i <= ub_i and sum_i y_i = target_sum

    Raises:
      ProjectionInfeasibleError when bounds make the target sum infeasible.
    """

    x_vec = _as_vector("x", x)
    n = int(x_vec.shape[0])
    lb_vec = _as_vector("lb", lb, n=n)
    ub_vec = _as_vector("ub", ub, n=n)

    if n == 0:
        if abs(float(target_sum)) <= eps:
            return np.zeros(0, dtype=float)
        raise ProjectionInfeasibleError(
            reason="empty_vector_nonzero_target",
            target_sum=float(target_sum),
            sum_lb=0.0,
            sum_ub=0.0,
            n_items=0,
        )

    if np.any(lb_vec > ub_vec + eps):
        bad = int(np.flatnonzero(lb_vec > ub_vec + eps)[0])
        raise ValueError(f"lb exceeds ub at index={bad}: lb={lb_vec[bad]:.6f} ub={ub_vec[bad]:.6f}")

    if weights is None:
        w_vec = np.ones(n, dtype=float)
    else:
        w_vec = _as_vector("weights", weights, n=n)
        if np.any(w_vec <= 0.0):
            bad = int(np.flatnonzero(w_vec <= 0.0)[0])
            raise ValueError(f"weights must be > 0; index={bad} value={w_vec[bad]:.6f}")

    target = float(target_sum)
    sum_lb = float(lb_vec.sum())
    sum_ub = float(ub_vec.sum())

    if sum_lb > target + eps:
        raise ProjectionInfeasibleError(
            reason="sum_lb_exceeds_target",
            target_sum=target,
            sum_lb=sum_lb,
            sum_ub=sum_ub,
            n_items=n,
        )
    if sum_ub < target - eps:
        raise ProjectionInfeasibleError(
            reason="sum_ub_below_target",
            target_sum=target,
            sum_lb=sum_lb,
            sum_ub=sum_ub,
            n_items=n,
        )

    if abs(sum_lb - target) <= eps:
        return lb_vec.copy()
    if abs(sum_ub - target) <= eps:
        return ub_vec.copy()

    # Weighted water-filling in lambda space:
    # y_i(lambda) = clip(x_i - lambda / (2 w_i), lb_i, ub_i)
    lam_lo = float(np.min(2.0 * w_vec * (x_vec - ub_vec))) - 1.0
    lam_hi = float(np.max(2.0 * w_vec * (x_vec - lb_vec))) + 1.0

    lo = lam_lo
    hi = lam_hi
    mid = 0.0
    tol_sum = max(eps * n, 1e-8)

    for _ in range(int(max_iters)):
        mid = 0.5 * (lo + hi)
        y_mid = np.clip(x_vec - mid / (2.0 * w_vec), lb_vec, ub_vec)
        s = float(y_mid.sum())
        if abs(s - target) <= tol_sum:
            break
        if s > target:
            lo = mid
        else:
            hi = mid

    y = np.clip(x_vec - mid / (2.0 * w_vec), lb_vec, ub_vec)

    # Residual polish for numerical precision and deterministic outputs.
    residual = target - float(y.sum())
    if abs(residual) > tol_sum:
        if residual > 0.0:
            slack = ub_vec - y
            candidates = np.flatnonzero(slack > eps)
            if candidates.size > 0:
                total = float(slack[candidates].sum())
                if total > eps:
                    add = residual * (slack[candidates] / total)
                    add = np.minimum(add, slack[candidates])
                    y[candidates] = y[candidates] + add
        else:
            slack = y - lb_vec
            candidates = np.flatnonzero(slack > eps)
            if candidates.size > 0:
                total = float(slack[candidates].sum())
                if total > eps:
                    sub = (-residual) * (slack[candidates] / total)
                    sub = np.minimum(sub, slack[candidates])
                    y[candidates] = y[candidates] - sub

    y = np.clip(y, lb_vec, ub_vec)
    final_err = abs(float(y.sum()) - target)
    if final_err > max(tol_sum, 1e-6):
        # Feasibility already checked; this should not happen unless numerics are unstable.
        raise RuntimeError(
            "project_sum_with_bounds failed to converge: "
            f"target={target:.6f} actual={float(y.sum()):.6f} err={final_err:.6f}"
        )

    return y


__all__ = ["project_sum_with_bounds", "ProjectionInfeasibleError"]
