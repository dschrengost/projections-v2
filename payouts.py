"""Contest payout helpers for lineup scoring."""

from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Sequence, Tuple

import numpy as np

from .scoring_models import PayoutTier

__all__ = [
    "normalize_payout_tiers",
    "compute_expected_user_payouts",
    "PayoutTolerance",
    "PayoutComputationResult",
    "compute_expected_user_payouts_vectorized",
]


_LEGACY_TOLERANCE_SAMPLE_WORLDS = 64


def _configure_blas_threads_for_payouts(workers: int | str | None) -> None:
    """Configure BLAS threading for payout computation to prevent oversubscription.

    This function ensures that when multiple payout workers are used, BLAS
    operations don't oversubscribe the CPU by spawning too many threads.
    """
    # Convert to actual worker count
    if isinstance(workers, str) and workers.lower() == "auto":
        cpu_count = os.cpu_count() or 1
        worker_count = cpu_count
    else:
        worker_count = int(workers) if workers else 1

    # Determine optimal BLAS thread count
    if worker_count > 1:
        # Limit BLAS to 1 thread per process when using multiple workers
        blas_threads = 1
    else:
        # Use single thread for single-worker execution
        blas_threads = 1

    # Set environment variables for all BLAS implementations
    blas_env_vars = {
        "OMP_NUM_THREADS": str(blas_threads),
        "OPENBLAS_NUM_THREADS": str(blas_threads),
        "MKL_NUM_THREADS": str(blas_threads),
        "VECLIB_MAXIMUM_THREADS": str(blas_threads),
        "NUMEXPR_NUM_THREADS": str(blas_threads),
    }

    # Only set if not already set to avoid overriding explicit user configuration
    for env_var, value in blas_env_vars.items():
        if env_var not in os.environ:
            os.environ[env_var] = value


@dataclass(frozen=True)
class _PayoutSpan:
    start: int
    end: int
    payout: float


@dataclass(frozen=True)
class PayoutTolerance:
    payout_dollars: float
    percentage_points: float  # Expressed as 0-1 rate delta, not percentage string.


@dataclass(frozen=True)
class PayoutComputationResult:
    expected_payouts: np.ndarray
    win_rates: np.ndarray
    cash_rates: np.ndarray
    field_expected_payouts: np.ndarray
    field_win_rates: np.ndarray
    field_cash_rates: np.ndarray
    max_delta_payout: float
    max_delta_percent: float


def normalize_payout_tiers(payout_tiers: Sequence[PayoutTier]) -> List[_PayoutSpan]:
    """Validate and sort payout tiers.

    Returns a list of spans with ``start`` ≤ ``end`` and payout per place.
    """

    spans: List[_PayoutSpan] = []
    for tier in payout_tiers:
        if tier.start_place <= 0 or tier.end_place <= 0:
            raise ValueError("Payout tiers must use positive ranks")
        if tier.end_place < tier.start_place:
            raise ValueError("Payout tier end_place must be ≥ start_place")
        spans.append(_PayoutSpan(tier.start_place, tier.end_place, float(tier.payout)))

    spans.sort(key=lambda span: span.start)
    return spans


def _expand_weighted_indices(weights: Sequence[int]) -> List[int]:
    expanded: List[int] = []
    for idx, weight in enumerate(weights):
        if weight <= 0:
            continue
        expanded.extend([idx] * weight)
    return expanded


class _VectorizedPayoutAccumulator:
    """Per-world accumulator mirroring the vectorized payout maths."""

    def __init__(
        self,
        user_weights: np.ndarray,
        field_weights: np.ndarray,
        payout_prefix: np.ndarray,
    ) -> None:
        if user_weights.ndim != 1 or field_weights.ndim != 1:
            raise ValueError("weights must be 1D arrays")
        self.user_weights = user_weights
        self.field_weights = field_weights
        self.payout_prefix = payout_prefix
        user_weights_f = user_weights.astype(np.float64)
        self.user_weights_positive = np.maximum(user_weights_f, 0.0)
        self.user_weights_float = np.maximum(user_weights_f, 1.0)
        self.total_field_weight = float(np.sum(field_weights))
        self.total_entries = int(
            np.sum(np.maximum(user_weights, 0)) + np.sum(field_weights)
        )
        if self.total_entries <= 0:
            raise ValueError("Total entrant weight must be positive")
        self.expected_sum = np.zeros_like(self.user_weights_float)
        self.win_sum = np.zeros_like(self.user_weights_float)
        self.cash_sum = np.zeros_like(self.user_weights_float)
        field_weights_f = field_weights.astype(np.float64)
        self.field_weights_positive = np.maximum(field_weights_f, 0.0)
        self.field_weights_float = np.maximum(field_weights_f, 1.0)
        self.field_expected_sum = np.zeros_like(self.field_weights_float)
        self.field_win_sum = np.zeros_like(self.field_weights_float)
        self.field_cash_sum = np.zeros_like(self.field_weights_float)
        self.world_count = 0

    def spawn_empty(self) -> "_VectorizedPayoutAccumulator":
        clone = self.__class__.__new__(self.__class__)
        clone.user_weights = self.user_weights
        clone.field_weights = self.field_weights
        clone.payout_prefix = self.payout_prefix
        clone.user_weights_positive = self.user_weights_positive
        clone.user_weights_float = self.user_weights_float
        clone.total_field_weight = self.total_field_weight
        clone.total_entries = self.total_entries
        clone.expected_sum = np.zeros_like(self.expected_sum)
        clone.win_sum = np.zeros_like(self.win_sum)
        clone.cash_sum = np.zeros_like(self.cash_sum)
        clone.field_weights_positive = self.field_weights_positive
        clone.field_weights_float = self.field_weights_float
        clone.field_expected_sum = np.zeros_like(self.field_expected_sum)
        clone.field_win_sum = np.zeros_like(self.field_win_sum)
        clone.field_cash_sum = np.zeros_like(self.field_cash_sum)
        clone.world_count = 0
        return clone

    def merge(self, other: "_VectorizedPayoutAccumulator") -> None:
        self.expected_sum += other.expected_sum
        self.win_sum += other.win_sum
        self.cash_sum += other.cash_sum
        self.field_expected_sum += other.field_expected_sum
        self.field_win_sum += other.field_win_sum
        self.field_cash_sum += other.field_cash_sum
        self.world_count += other.world_count

    def update(self, user_scores_w: np.ndarray, field_scores_w: np.ndarray) -> None:
        import numpy as np

        user_scores = np.asarray(user_scores_w, dtype=np.float64)
        field_scores = np.asarray(field_scores_w, dtype=np.float64)
        if user_scores.ndim != 1 or field_scores.ndim != 1:
            raise ValueError("Accumulator update expects 1D score vectors")
        if user_scores.size != self.user_weights.size:
            raise ValueError("User scores length must match user weights")
        if field_scores.size != self.field_weights.size:
            raise ValueError("Field scores length must match field weights")

        self.world_count += 1

        # ---- FIELD: sort once, cumulative weights (O(F log F)) ----
        f_order = np.argsort(field_scores)
        f_sorted_scores = field_scores[f_order]
        f_sorted_weights = self.field_weights[f_order]
        f_cum = np.cumsum(f_sorted_weights, dtype=np.float64)
        f_cum_pad = np.concatenate(([0.0], f_cum))
        total_field_weight = self.total_field_weight

        # ---- USERS: sort once, prefix & tie groups (O(N log N)) ----
        u_scores = user_scores
        u_weights_pos = self.user_weights_positive
        u_order = np.argsort(u_scores)
        u_sorted_scores = u_scores[u_order]
        u_sorted_weights = u_weights_pos[u_order]
        if u_sorted_weights.size:
            u_prefix = np.cumsum(u_sorted_weights, dtype=np.float64)
            total_user_weight = float(u_prefix[-1])
            u_cum_pad = np.concatenate(([0.0], u_prefix))
        else:
            u_prefix = np.array([], dtype=np.float64)
            total_user_weight = 0.0
            u_cum_pad = np.array([0.0], dtype=np.float64)

        if u_sorted_scores.size:
            tie_breaks = np.r_[True, np.abs(np.diff(u_sorted_scores)) > 1e-9]
            group_ids = np.cumsum(tie_breaks) - 1
            G = int(group_ids[-1]) + 1
        else:
            group_ids = np.array([], dtype=np.int64)
            G = 0

        equal_weight_sorted = np.zeros_like(u_sorted_weights, dtype=np.float64)
        greater_weight_sorted = np.zeros_like(u_sorted_weights, dtype=np.float64)
        if G > 0:
            group_starts = np.flatnonzero(tie_breaks)
            group_lengths = np.diff(np.r_[group_starts, u_sorted_scores.size])
            group_sums = np.add.reduceat(u_sorted_weights, group_starts)
            cumulative_groups = np.cumsum(group_sums, dtype=np.float64)
            equal_weight_sorted = np.repeat(group_sums, group_lengths)
            greater_weight_sorted = np.repeat(total_user_weight - cumulative_groups, group_lengths)

        if u_order.size:
            inv_u_order = np.empty_like(u_order)
            inv_u_order[u_order] = np.arange(u_order.size)
            user_equal_weight = equal_weight_sorted[inv_u_order]
            user_greater_weight = greater_weight_sorted[inv_u_order]
        else:
            user_equal_weight = np.array([], dtype=np.float64)
            user_greater_weight = np.array([], dtype=np.float64)

        # ---- FIELD lookups for all users (vectorized) ----
        idx_left = np.searchsorted(f_sorted_scores, u_scores, side="left")
        idx_right = np.searchsorted(f_sorted_scores, u_scores, side="right")
        field_weight_lt = f_cum_pad[idx_left]
        field_weight_le = f_cum_pad[idx_right]
        field_greater_weight = total_field_weight - field_weight_le
        field_equal_weight = field_weight_le - field_weight_lt

        # ---- Entrants ahead / tie weights ----
        entrants_ahead = field_greater_weight + user_greater_weight
        tie_weight_base = field_equal_weight + user_equal_weight
        tie_weight = np.where(
            tie_weight_base > 0.0,
            tie_weight_base,
            self.user_weights_positive,
        )

        # ---- Start/End ranks (floor semantics; 1-indexed; clamped) ----
        total_entries = int(self.total_entries)
        start_rank = np.floor(entrants_ahead).astype(np.int64) + 1
        end_rank = np.floor(entrants_ahead + tie_weight).astype(np.int64)
        np.clip(start_rank, 1, total_entries, out=start_rank)
        end_rank = np.maximum(end_rank, start_rank - 1)
        np.clip(end_rank, 0, total_entries, out=end_rank)

        # ---- Payout split via prefix ----
        seg_sum = self.payout_prefix[end_rank] - self.payout_prefix[start_rank - 1]
        payout_each = np.divide(seg_sum, tie_weight, out=np.zeros_like(seg_sum), where=tie_weight > 0.0)

        # ---- Accumulate ----
        user_w_pos = self.user_weights_positive
        self.expected_sum += payout_each * user_w_pos
        win_each = np.divide(
            user_w_pos,
            tie_weight,
            out=np.zeros_like(user_w_pos),
            where=tie_weight > 0.0,
        )
        self.win_sum += np.where(start_rank == 1, win_each, 0.0)
        self.cash_sum += np.where(payout_each > 0.0, user_w_pos, 0.0)

        # ---- USER lookups for all field lineups (vectorized symmetry) ----
        if f_order.size:
            inv_f_order = np.empty_like(f_order)
            inv_f_order[f_order] = np.arange(f_order.size)
            f_sorted_weights_pos = self.field_weights_positive[f_order]
            if f_sorted_scores.size:
                f_tie_breaks = np.r_[True, np.abs(np.diff(f_sorted_scores)) > 1e-9]
                f_group_starts = np.flatnonzero(f_tie_breaks)
                f_group_lengths = np.diff(np.r_[f_group_starts, f_sorted_scores.size])
                f_group_sums = np.add.reduceat(f_sorted_weights_pos, f_group_starts)
                f_cumulative_groups = np.cumsum(f_group_sums, dtype=np.float64)
                field_equal_weight_sorted = np.repeat(f_group_sums, f_group_lengths)
                field_greater_weight_sorted = np.repeat(
                    total_field_weight - f_cumulative_groups,
                    f_group_lengths,
                )
            else:
                field_equal_weight_sorted = np.zeros_like(f_sorted_weights_pos)
                field_greater_weight_sorted = np.zeros_like(f_sorted_weights_pos)
            field_equal_weight_field = field_equal_weight_sorted[inv_f_order]
            field_greater_weight_field = field_greater_weight_sorted[inv_f_order]
        else:
            field_equal_weight_field = np.zeros_like(field_scores)
            field_greater_weight_field = np.zeros_like(field_scores)

        if u_sorted_scores.size:
            user_idx_left = np.searchsorted(u_sorted_scores, field_scores, side="left")
            user_idx_right = np.searchsorted(u_sorted_scores, field_scores, side="right")
            user_weight_lt_field = u_cum_pad[user_idx_left]
            user_weight_le_field = u_cum_pad[user_idx_right]
            user_greater_weight_field = total_user_weight - user_weight_le_field
            user_equal_weight_field = user_weight_le_field - user_weight_lt_field
        else:
            user_greater_weight_field = np.zeros_like(field_scores)
            user_equal_weight_field = np.zeros_like(field_scores)

        field_tie_weight_base = field_equal_weight_field + user_equal_weight_field
        field_tie_weight = np.where(
            field_tie_weight_base > 0.0,
            field_tie_weight_base,
            self.field_weights_positive,
        )
        field_entrants_ahead = field_greater_weight_field + user_greater_weight_field
        field_start_rank = np.floor(field_entrants_ahead).astype(np.int64) + 1
        field_end_rank = np.floor(field_entrants_ahead + field_tie_weight).astype(np.int64)
        np.clip(field_start_rank, 1, total_entries, out=field_start_rank)
        field_end_rank = np.maximum(field_end_rank, field_start_rank - 1)
        np.clip(field_end_rank, 0, total_entries, out=field_end_rank)

        field_seg_sum = self.payout_prefix[field_end_rank] - self.payout_prefix[field_start_rank - 1]
        field_payout_each = np.divide(
            field_seg_sum,
            field_tie_weight,
            out=np.zeros_like(field_seg_sum),
            where=field_tie_weight > 0.0,
        )

        field_w_pos = self.field_weights_positive
        self.field_expected_sum += field_payout_each * field_w_pos
        field_win_each = np.divide(
            field_w_pos,
            field_tie_weight,
            out=np.zeros_like(field_w_pos),
            where=field_tie_weight > 0.0,
        )
        self.field_win_sum += np.where(field_start_rank == 1, field_win_each, 0.0)
        self.field_cash_sum += np.where(field_payout_each > 0.0, field_w_pos, 0.0)

    def finalize(
        self,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        if self.world_count == 0:
            zeros = np.zeros_like(self.expected_sum)
            field_zeros = np.zeros_like(self.field_expected_sum)
            return zeros, zeros.copy(), zeros.copy(), field_zeros, field_zeros.copy(), field_zeros.copy()

        expected = (self.expected_sum / self.world_count) / self.user_weights_float
        win_rates = (self.win_sum / self.world_count) / self.user_weights_float
        cash_rates = (self.cash_sum / self.world_count) / self.user_weights_float
        field_expected = (self.field_expected_sum / self.world_count) / self.field_weights_float
        field_win = (self.field_win_sum / self.world_count) / self.field_weights_float
        field_cash = (self.field_cash_sum / self.world_count) / self.field_weights_float
        return expected, win_rates, cash_rates, field_expected, field_win, field_cash


class _LegacyPayoutAccumulator:
    """Streaming variant of the legacy payout logic (used for tolerance checks)."""

    def __init__(
        self,
        user_weights: Sequence[int],
        field_weights: Sequence[int],
        payout_tiers: Sequence[PayoutTier],
    ) -> None:
        self.user_expanded = _expand_weighted_indices(user_weights)
        self.field_expanded = _expand_weighted_indices(field_weights)
        self.spans = normalize_payout_tiers(payout_tiers)
        self.expected_sum = np.zeros(len(user_weights), dtype=np.float64)
        self.win_sum = np.zeros(len(user_weights), dtype=np.float64)
        self.cash_sum = np.zeros(len(user_weights), dtype=np.float64)
        self.user_weights = np.array(user_weights, dtype=np.float64)
        self.user_weights[self.user_weights <= 0] = 1.0
        self.world_count = 0

    def update(self, user_scores_w: np.ndarray, field_scores_w: np.ndarray) -> None:
        user_scores = np.asarray(user_scores_w, dtype=np.float64)
        field_scores = np.asarray(field_scores_w, dtype=np.float64)

        combined: List[tuple[float, str, int]] = []
        for idx in self.user_expanded:
            combined.append((float(user_scores[idx]), "user", idx))
        for idx in self.field_expanded:
            combined.append((float(field_scores[idx]), "field", idx))

        combined.sort(key=lambda item: item[0], reverse=True)

        position = 1
        cursor = 0
        while cursor < len(combined):
            score = combined[cursor][0]
            tied: List[tuple[float, str, int]] = []
            while cursor < len(combined) and abs(combined[cursor][0] - score) < 1e-9:
                tied.append(combined[cursor])
                cursor += 1
            start = position
            end = position + len(tied) - 1
            position = end + 1

            payout_total = _payout_for_rank_block(self.spans, start, end)
            payout_each = payout_total / len(tied) if tied else 0.0

            if start == 1:
                for _, kind, owner_idx in tied:
                    if kind == "user":
                        self.win_sum[owner_idx] += 1.0 / len(tied)

            if payout_each > 0.0:
                for _, kind, owner_idx in tied:
                    if kind == "user":
                        self.expected_sum[owner_idx] += payout_each
                        self.cash_sum[owner_idx] += 1.0

        self.world_count += 1

    def finalize(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.world_count == 0:
            zeros = np.zeros_like(self.expected_sum)
            return zeros, zeros.copy(), zeros.copy()

        expected = (self.expected_sum / self.world_count) / self.user_weights
        win_rates = (self.win_sum / self.world_count) / self.user_weights
        cash_rates = (self.cash_sum / self.world_count) / self.user_weights
        return expected, win_rates, cash_rates


def _payout_for_rank_block(spans: List[_PayoutSpan], start: int, end: int) -> float:
    payout = 0.0
    for span in spans:
        overlap_start = max(start, span.start)
        overlap_end = min(end, span.end)
        if overlap_start <= overlap_end:
            payout += (overlap_end - overlap_start + 1) * span.payout
    return payout


def compute_expected_user_payouts(
    user_scores: np.ndarray,
    field_scores: np.ndarray,
    user_weights: Sequence[int],
    field_weights: Sequence[int],
    payout_tiers: Sequence[PayoutTier],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return expected payout, win%, and cash% for each user lineup.

    ``user_scores`` and ``field_scores`` are ``(N, W)`` matrices where rows map
    to lineups and columns to worlds. ``user_weights`` / ``field_weights`` are
    integer multiplicities per lineup representing duplicate entries.
    """

    if user_scores.ndim != 2 or field_scores.ndim != 2:
        raise ValueError("Scores must be 2D arrays")
    if user_scores.shape[1] != field_scores.shape[1]:
        raise ValueError("User and field scores must have matching world counts")

    spans = normalize_payout_tiers(payout_tiers)
    user_expanded = _expand_weighted_indices(user_weights)
    field_expanded = _expand_weighted_indices(field_weights)

    num_worlds = user_scores.shape[1]
    num_users = user_scores.shape[0]

    expected = np.zeros(num_users, dtype=np.float64)
    win_rates = np.zeros(num_users, dtype=np.float64)
    cash_rates = np.zeros(num_users, dtype=np.float64)

    for world_idx in range(num_worlds):
        combined: List[Tuple[float, str, int]] = []
        for idx in user_expanded:
            combined.append((float(user_scores[idx, world_idx]), "user", idx))
        for idx in field_expanded:
            combined.append((float(field_scores[idx, world_idx]), "field", idx))

        combined.sort(key=lambda item: item[0], reverse=True)

        position = 1
        cursor = 0
        while cursor < len(combined):
            score = combined[cursor][0]
            tied: List[Tuple[float, str, int]] = []
            while cursor < len(combined) and abs(combined[cursor][0] - score) < 1e-9:
                tied.append(combined[cursor])
                cursor += 1
            start = position
            end = position + len(tied) - 1
            position = end + 1

            payout_total = _payout_for_rank_block(spans, start, end)
            payout_each = payout_total / len(tied) if tied else 0.0

            if start == 1:
                for _, kind, owner_idx in tied:
                    if kind == "user":
                        win_rates[owner_idx] += 1.0 / len(tied)

            if payout_each > 0.0:
                for _, kind, owner_idx in tied:
                    if kind == "user":
                        expected[owner_idx] += payout_each
                        cash_rates[owner_idx] += 1.0

    world_count = float(num_worlds)
    if world_count == 0:
        return expected, win_rates, cash_rates

    user_weights_arr = np.array(user_weights, dtype=np.float64)
    user_weights_arr[user_weights_arr <= 0] = 1.0

    expected = (expected / world_count) / user_weights_arr
    win_rates = (win_rates / world_count) / user_weights_arr
    cash_rates = (cash_rates / world_count) / user_weights_arr

    return expected, win_rates, cash_rates


def compute_expected_user_payouts_vectorized(
    user_scores: np.ndarray | None,
    field_scores: np.ndarray | None,
    user_weights: Sequence[int],
    field_weights: Sequence[int],
    payout_tiers: Sequence[PayoutTier],
    *,
    tolerance: PayoutTolerance,
    score_iterator: Iterator[tuple[np.ndarray, np.ndarray]] | None = None,
    world_count: int | None = None,
    workers: int | str | None = None,
) -> PayoutComputationResult:
    """Vectorized payout computation with tolerance verification and streaming support.

    Ranks derived from entrant weights use floor semantics before applying payouts.

    Parameters
    ----------
    workers:
        Optional worker hint controlling how many worlds columns are processed
        in parallel when computing payouts. ``None``/``"auto"`` chooses a CPU
        bound default based on the provided world count.
    """

    if tolerance is None:
        raise ValueError("tolerance must be provided")

    # Configure BLAS threading for optimal performance
    _configure_blas_threads_for_payouts(workers)

    spans = normalize_payout_tiers(payout_tiers)
    user_weights_arr = np.array(user_weights, dtype=np.int64)
    field_weights_arr = np.array(field_weights, dtype=np.int64)

    if score_iterator is None:
        if user_scores is None or field_scores is None:
            raise ValueError("user_scores and field_scores are required when score_iterator is not provided")
        if user_scores.ndim != 2 or field_scores.ndim != 2:
            raise ValueError("Scores must be 2D arrays")
        if user_scores.shape[1] != field_scores.shape[1]:
            raise ValueError("User and field scores must have matching world counts")
        if user_weights_arr.size != user_scores.shape[0]:
            raise ValueError("User weights length must match user scores")
        if field_weights_arr.size != field_scores.shape[0]:
            raise ValueError("Field weights length must match field scores")
        world_total = user_scores.shape[1]

        def _array_iterator() -> Iterator[tuple[np.ndarray, np.ndarray]]:
            yield user_scores, field_scores

        chunk_iterator: Iterator[tuple[np.ndarray, np.ndarray]] = _array_iterator()
    else:
        if world_count is None:
            raise ValueError("world_count must be provided when using score_iterator")
        world_total = int(world_count)
        chunk_iterator = score_iterator
        if user_scores is not None or field_scores is not None:
            # Guard against mixed usage that could introduce inconsistencies.
            raise ValueError("Provide either arrays or score_iterator, not both")

        if user_weights_arr.size <= 0 or field_weights_arr.size <= 0:
            raise ValueError("Weights must be non-empty for streaming payouts")

    total_entries = int(np.sum(field_weights_arr) + np.sum(np.maximum(user_weights_arr, 0)))
    if total_entries <= 0:
        raise ValueError("Total entrant weight must be positive")

    payout_per_place = np.zeros(total_entries, dtype=np.float64)
    for span in spans:
        start = min(span.start, total_entries)
        end = min(span.end, total_entries)
        if start > total_entries:
            continue
        payout_per_place[start - 1 : end] = span.payout
    payout_prefix = np.zeros(total_entries + 1, dtype=np.float64)
    payout_prefix[1:] = np.cumsum(payout_per_place)

    vector_acc = _VectorizedPayoutAccumulator(user_weights_arr, field_weights_arr, payout_prefix)
    sample_vector_acc: _VectorizedPayoutAccumulator | None = None
    sample_legacy_acc: _LegacyPayoutAccumulator | None = None
    if tolerance is not None:
        sample_vector_acc = _VectorizedPayoutAccumulator(user_weights_arr, field_weights_arr, payout_prefix)
        sample_legacy_acc = _LegacyPayoutAccumulator(user_weights, field_weights, payout_tiers)

    processed_worlds = 0
    sample_limit = min(_LEGACY_TOLERANCE_SAMPLE_WORLDS, world_total) if world_total else 0

    resolved_workers = _resolve_workers(workers, world_total)
    executor: ThreadPoolExecutor | None = None
    if resolved_workers > 1 and world_total > 1:
        executor = ThreadPoolExecutor(max_workers=resolved_workers, thread_name_prefix="payouts")

    try:
        for chunk_user_scores, chunk_field_scores in chunk_iterator:
            user_chunk = np.asarray(chunk_user_scores)
            field_chunk = np.asarray(chunk_field_scores)

            if user_chunk.ndim == 1:
                user_chunk = user_chunk.reshape(user_chunk.shape[0], 1)
            if field_chunk.ndim == 1:
                field_chunk = field_chunk.reshape(field_chunk.shape[0], 1)

            if user_chunk.ndim != 2 or field_chunk.ndim != 2:
                raise ValueError("Scores must be 2D in each chunk")
            if user_chunk.shape[1] != field_chunk.shape[1]:
                raise ValueError("User and field chunks must share world dimension")
            if user_chunk.shape[0] != user_weights_arr.size:
                raise ValueError("User chunk row count must match user weights")
            if field_chunk.shape[0] != field_weights_arr.size:
                raise ValueError("Field chunk row count must match field weights")

            chunk_worlds = user_chunk.shape[1]

            if executor is not None and chunk_worlds > 1:
                ranges = _iter_ranges(chunk_worlds, resolved_workers)

                def _payout_worker(start: int, stop: int) -> _VectorizedPayoutAccumulator:
                    partial = vector_acc.spawn_empty()
                    for idx in range(start, stop):
                        column_user = user_chunk[:, idx]
                        column_field = field_chunk[:, idx]
                        partial.update(column_user, column_field)
                    return partial

                futures = [executor.submit(_payout_worker, start, stop) for start, stop in ranges if start < stop]
                for future in futures:
                    vector_acc.merge(future.result())
            else:
                for col_idx in range(chunk_worlds):
                    column_user = user_chunk[:, col_idx]
                    column_field = field_chunk[:, col_idx]
                    vector_acc.update(column_user, column_field)

            if sample_vector_acc is not None and sample_legacy_acc is not None:
                for offset in range(chunk_worlds):
                    absolute_idx = processed_worlds + offset
                    if absolute_idx >= sample_limit:
                        break
                    column_user = user_chunk[:, offset]
                    column_field = field_chunk[:, offset]
                    sample_vector_acc.update(column_user, column_field)
                    sample_legacy_acc.update(column_user, column_field)

            processed_worlds += chunk_worlds

    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    (
        expected,
        win_rates,
        cash_rates,
        field_expected,
        field_win_rates,
        field_cash_rates,
    ) = vector_acc.finalize()

    max_delta_payout = 0.0
    max_delta_percent = 0.0

    if sample_vector_acc is not None and sample_legacy_acc is not None and sample_vector_acc.world_count > 0:
        sample_expected, sample_win, sample_cash, *_ = sample_vector_acc.finalize()
        legacy_expected, legacy_win, legacy_cash = sample_legacy_acc.finalize()

        delta_payout = np.abs(sample_expected - legacy_expected)
        delta_win = np.abs(sample_win - legacy_win)
        delta_cash = np.abs(sample_cash - legacy_cash)

        max_delta_payout = float(delta_payout.max()) if delta_payout.size else 0.0
        max_delta_percent = float(np.maximum(delta_win, delta_cash).max()) if delta_win.size else 0.0

        if (
            max_delta_payout > tolerance.payout_dollars
            or max_delta_percent > tolerance.percentage_points
        ):
            raise ValueError(
                "Vectorized payout computation exceeded tolerance: "
                f"payout Δ={max_delta_payout:.6f}, percent Δ={max_delta_percent:.6f}"
            )

    return PayoutComputationResult(
        expected_payouts=expected,
        win_rates=win_rates,
        cash_rates=cash_rates,
        field_expected_payouts=field_expected,
        field_win_rates=field_win_rates,
        field_cash_rates=field_cash_rates,
        max_delta_payout=max_delta_payout,
        max_delta_percent=max_delta_percent,
    )


def _resolve_workers(requested: int | str | None, task_count: int) -> int:
    if isinstance(requested, int):
        return max(1, requested)
    if isinstance(requested, str) and requested.lower() != "auto":
        raise ValueError("worker hint must be 'auto' or a positive integer")
    cpu = os.cpu_count() or 1
    if task_count <= 0:
        return max(1, cpu)
    return max(1, min(cpu, task_count))


def _iter_ranges(total: int, workers: int) -> List[tuple[int, int]]:
    if total <= 0 or workers <= 0:
        return []
    step = max(1, math.ceil(total / workers))
    ranges: List[tuple[int, int]] = []
    for start in range(0, total, step):
        stop = min(start + step, total)
        ranges.append((start, stop))
    return ranges
