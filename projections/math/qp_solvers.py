"""Small quadratic program helpers powered by cvxpy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cvxpy as cp
import numpy as np


@dataclass(slots=True)
class QPProblem:
    """Quadratic program in standard form.

    Objective: 0.5 * x^T Q x + c^T x
    Subject to:
        A_eq x = b_eq
        A_ineq x <= b_ineq
        lb <= x <= ub
    """

    Q: np.ndarray
    c: np.ndarray
    A_eq: np.ndarray | None = None
    b_eq: np.ndarray | None = None
    A_ineq: np.ndarray | None = None
    b_ineq: np.ndarray | None = None
    lb: np.ndarray | None = None
    ub: np.ndarray | None = None


class QPSolverError(RuntimeError):
    """Raised when the QP solver fails to find a feasible optimum."""


def _validate_shape(array: np.ndarray | None, name: str, expected: tuple[int, ...]) -> None:
    if array is None:
        return
    if array.shape != expected:
        raise ValueError(f"{name} expected shape {expected}, found {array.shape}.")


def solve_qp(
    problem: QPProblem,
    *,
    solver: str = "OSQP",
    solver_opts: dict[str, Any] | None = None,
) -> np.ndarray:
    """Solve `problem` with cvxpy and return the optimal primal vector.

    Raises:
        QPSolverError: if the problem is infeasible or the solver fails to converge.
    """

    Q = np.asarray(problem.Q, dtype=float)
    c = np.asarray(problem.c, dtype=float).reshape(-1)
    n = c.shape[0]
    _validate_shape(Q, "Q", (n, n))

    A_eq = None if problem.A_eq is None else np.asarray(problem.A_eq, dtype=float)
    b_eq = None if problem.b_eq is None else np.asarray(problem.b_eq, dtype=float).reshape(-1)
    if A_eq is not None:
        _validate_shape(A_eq, "A_eq", (A_eq.shape[0], n))
        if b_eq is None:
            raise ValueError("A_eq specified without b_eq.")
        _validate_shape(b_eq, "b_eq", (A_eq.shape[0],))
    elif b_eq is not None:
        raise ValueError("b_eq specified without A_eq.")

    A_ineq = None if problem.A_ineq is None else np.asarray(problem.A_ineq, dtype=float)
    b_ineq = None if problem.b_ineq is None else np.asarray(problem.b_ineq, dtype=float).reshape(-1)
    if A_ineq is not None:
        _validate_shape(A_ineq, "A_ineq", (A_ineq.shape[0], n))
        if b_ineq is None:
            raise ValueError("A_ineq specified without b_ineq.")
        _validate_shape(b_ineq, "b_ineq", (A_ineq.shape[0],))
    elif b_ineq is not None:
        raise ValueError("b_ineq specified without A_ineq.")

    lb = None if problem.lb is None else np.asarray(problem.lb, dtype=float).reshape(-1)
    ub = None if problem.ub is None else np.asarray(problem.ub, dtype=float).reshape(-1)
    if lb is not None:
        _validate_shape(lb, "lb", (n,))
    if ub is not None:
        _validate_shape(ub, "ub", (n,))

    sym_Q = 0.5 * (Q + Q.T)
    x = cp.Variable(n)
    objective = 0.5 * cp.quad_form(x, sym_Q) + c @ x
    constraints: list[cp.Constraint] = []
    if A_eq is not None and b_eq is not None:
        constraints.append(A_eq @ x == b_eq)
    if A_ineq is not None and b_ineq is not None:
        constraints.append(A_ineq @ x <= b_ineq)
    if lb is not None:
        constraints.append(x >= lb)
    if ub is not None:
        constraints.append(x <= ub)

    problem_cvx = cp.Problem(cp.Minimize(objective), constraints)
    opts = {"solver": getattr(cp, solver), "max_iter": 10_000, "warm_start": True}
    if solver_opts:
        opts.update(solver_opts)
    try:
        problem_cvx.solve(**opts)
    except cp.SolverError as exc:  # pragma: no cover - delegated to cvxpy
        raise QPSolverError(f"QP solver failure: {exc}") from exc

    if problem_cvx.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise QPSolverError(f"QP solver returned status {problem_cvx.status}")

    if x.value is None:
        raise QPSolverError("QP solver did not return a solution vector.")
    return np.asarray(x.value).reshape(n)
