from __future__ import annotations

import numpy as np


def _risk(counts: np.ndarray, sigma: np.ndarray) -> float:
    return float(counts @ (sigma @ counts))


def _lineup_var(idxs: np.ndarray, sigma: np.ndarray) -> float:
    return float(np.sum(sigma[np.ix_(idxs, idxs)]))


def test_swap_delta_matches_exact_risk_change() -> None:
    """Property test for the swap delta derivation used in the portfolio optimizer.

    Risk objective:
        R(counts) = countsᵀ Σ counts

    Swap:
        counts' = counts - a_i + a_j
    where a_k is a 0/1 indicator vector for lineup k.

    Closed form (with counts including a_i at evaluation time):
        ΔR = (var_i - 2 * countsᵀΣ a_i) + (var_j + 2 * countsᵀΣ a_j) - 2 * cov_ij
    """

    rng = np.random.default_rng(0)
    W = 500
    P = 50
    K = 200
    N = 20
    lineup_size = 8

    X = rng.normal(size=(W, P)).astype(np.float64)
    sigma = np.cov(X, rowvar=False, bias=False).astype(np.float64)
    sigma = (sigma + sigma.T) / 2.0  # ensure symmetry

    lineups = [np.sort(rng.choice(P, size=lineup_size, replace=False)) for _ in range(K)]

    for _case in range(5):
        selected = rng.choice(K, size=N, replace=False).astype(int)
        selected_set = set(int(x) for x in selected)
        not_selected = [i for i in range(K) if i not in selected_set]

        counts = np.zeros((P,), dtype=np.float64)
        for k in selected:
            counts[lineups[int(k)]] += 1.0

        v = sigma @ counts
        risk = _risk(counts, sigma)

        for _ in range(50):
            i = int(rng.choice(selected))
            j = int(rng.choice(not_selected))

            idxs_i = lineups[i]
            idxs_j = lineups[j]

            counts2 = counts.copy()
            counts2[idxs_i] -= 1.0
            counts2[idxs_j] += 1.0
            risk2 = _risk(counts2, sigma)
            delta_exact = risk2 - risk

            # Fast delta via closed form.
            var_i = _lineup_var(idxs_i, sigma)
            var_j = _lineup_var(idxs_j, sigma)
            penalty_i = float(np.sum(v[idxs_i]))  # a_iᵀ Σ counts == countsᵀ Σ a_i
            penalty_j = float(np.sum(v[idxs_j]))  # a_jᵀ Σ counts == countsᵀ Σ a_j
            cov_ij = float(np.sum(sigma[np.ix_(idxs_i, idxs_j)]))  # a_iᵀ Σ a_j

            delta_fast = (var_i - 2.0 * penalty_i) + (var_j + 2.0 * penalty_j) - 2.0 * cov_ij

            scale = max(1.0, abs(risk), abs(risk2))
            tol = 1e-8 * scale + 1e-10
            assert abs(delta_fast - delta_exact) <= tol

            # v update should match: v' = Σ @ counts' = v - Σ a_i + Σ a_j.
            sigma_ai = np.sum(sigma[:, idxs_i], axis=1)
            sigma_aj = np.sum(sigma[:, idxs_j], axis=1)
            v2_fast = v - sigma_ai + sigma_aj
            v2_exact = sigma @ counts2
            assert np.allclose(v2_fast, v2_exact, rtol=1e-8, atol=1e-10)
