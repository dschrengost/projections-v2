"""Tests for _max_overlap_from_jaccard conversion.

Verifies the Jaccard→overlap cap function correctly computes k_max such that:
    J(k_max) < t  AND  J(k_max + 1) >= t

For equal-size lineups A, B of size n with overlap k:
    J(A,B) = k / (2n - k)
"""

import math
import pytest

from projections.optimizer.quick_build import _max_overlap_from_jaccard


def jaccard(k: int, n: int) -> float:
    """Compute Jaccard similarity for overlap k in lineups of size n."""
    if k <= 0:
        return 0.0
    union = 2 * n - k
    if union <= 0:
        return 1.0
    return k / union


class TestMaxOverlapFromJaccard:
    """Test suite for _max_overlap_from_jaccard."""

    @pytest.mark.parametrize("n,t,expected", [
        # n=8 examples
        (8, 0.60, 5),   # raw = 6.0 exactly → floor(6.0 - eps) = 5
        (8, 0.75, 6),   # raw ≈ 6.857 → 6
        (8, 0.70, 6),   # raw ≈ 6.588 → 6
        (8, 0.50, 5),   # raw ≈ 5.333 → 5
        # n=9 examples
        (9, 0.60, 6),   # raw = 6.75 → 6
        (9, 0.75, 7),   # raw ≈ 7.714 → 7
        (9, 0.70, 7),   # raw ≈ 7.412 → 7
        (9, 0.50, 5),   # raw = 6.0 exactly → 5
        # Edge cases: disable semantics (return n-1)
        (8, 0.0, 7),
        (8, -0.5, 7),
        (8, 1.0, 7),
        (8, 1.5, 7),
        (9, 0.0, 8),
        (9, 1.0, 8),
    ])
    def test_known_values(self, n: int, t: float, expected: int):
        """Verify specific known-good values."""
        result = _max_overlap_from_jaccard(t, n)
        assert result == expected, f"n={n}, t={t}: got {result}, expected {expected}"

    @pytest.mark.parametrize("n", [8, 9])
    @pytest.mark.parametrize("t", [0.50, 0.60, 0.70, 0.75])
    def test_jaccard_inequality_holds(self, n: int, t: float):
        """Verify J(k_max) < t for valid thresholds in (0, 1)."""
        k_max = _max_overlap_from_jaccard(t, n)
        if k_max > 0:
            j_val = jaccard(k_max, n)
            assert j_val < t, f"J({k_max}, {n}) = {j_val} should be < {t}"

    @pytest.mark.parametrize("n", [8, 9])
    @pytest.mark.parametrize("t", [0.50, 0.60, 0.70, 0.75])
    def test_next_overlap_violates(self, n: int, t: float):
        """Verify J(k_max + 1) >= t when k_max + 1 <= n."""
        k_max = _max_overlap_from_jaccard(t, n)
        if k_max + 1 <= n:
            j_next = jaccard(k_max + 1, n)
            assert j_next >= t, f"J({k_max + 1}, {n}) = {j_next} should be >= {t}"

    @pytest.mark.parametrize("n", [8, 9])
    def test_monotonicity(self, n: int):
        """Increasing t should not decrease k_max within valid range (0 < t < 1).
        
        Note: t<=0 and t>=1 have special 'disable' semantics (return n-1),
        so monotonicity only applies to the valid range.
        """
        thresholds = [0.25, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90]
        prev_k = -1
        for t in thresholds:
            k = _max_overlap_from_jaccard(t, n)
            assert k >= prev_k, f"Monotonicity violated at t={t}: {k} < {prev_k}"
            prev_k = k

    def test_boundary_at_integer_raw(self):
        """raw=6.0 exactly should yield k_max=5 for strict < inequality.
        
        n=8, t=0.6 gives raw = (2*8*0.6) / 1.6 = 6.0 exactly.
        J(5, 8) = 5/11 ≈ 0.4545 < 0.6 ✓
        J(6, 8) = 6/10 = 0.6 NOT < 0.6 ✓
        """
        k_max = _max_overlap_from_jaccard(0.6, 8)
        assert k_max == 5, f"At raw=6.0, k_max should be 5, got {k_max}"

    def test_boundary_at_integer_raw_n9(self):
        """Another integer boundary: n=9, t=0.5 gives raw=6.0."""
        k_max = _max_overlap_from_jaccard(0.5, 9)
        assert k_max == 5, f"At raw=6.0 (n=9, t=0.5), k_max should be 5, got {k_max}"
        # Verify: J(5, 9) = 5/(18-5) = 5/13 ≈ 0.385 < 0.5 ✓
        # Verify: J(6, 9) = 6/(18-6) = 6/12 = 0.5 NOT < 0.5 ✓

    def test_nan_returns_disable(self):
        """NaN threshold should return n-1 (disable)."""
        k = _max_overlap_from_jaccard(float("nan"), 8)
        assert k == 7

    def test_invalid_string_returns_disable(self):
        """Non-numeric threshold should return n-1 (disable)."""
        k = _max_overlap_from_jaccard("invalid", 8)  # type: ignore
        assert k == 7

    def test_negative_threshold_disables(self):
        """Negative threshold should return n-1 (disable)."""
        k = _max_overlap_from_jaccard(-0.5, 8)
        assert k == 7

    def test_threshold_above_one_disables(self):
        """Threshold > 1 should return n-1 (disable)."""
        k = _max_overlap_from_jaccard(1.5, 8)
        assert k == 7

    def test_zero_lineup_size(self):
        """Zero lineup size should return 0."""
        k = _max_overlap_from_jaccard(0.75, 0)
        assert k == 0


class TestPropertyBased:
    """Property-based tests using random sampling."""

    def test_random_thresholds(self):
        """For random t in (0,1), verify k_max satisfies J(k_max) < t."""
        import random
        random.seed(42)
        for n in [8, 9]:
            for _ in range(50):
                t = random.uniform(0.01, 0.99)
                k_max = _max_overlap_from_jaccard(t, n)
                # J(k_max) < t
                if k_max > 0:
                    assert jaccard(k_max, n) < t, f"n={n}, t={t}, k_max={k_max}"
                # J(k_max + 1) >= t (when valid)
                if k_max + 1 <= n:
                    assert jaccard(k_max + 1, n) >= t, f"n={n}, t={t}, k_max+1={k_max+1}"
