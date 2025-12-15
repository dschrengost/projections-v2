"""Contest payout calculation engine.

Adapted from sim-v2/Simulator/worlds/payouts.py with extensions for
positional rate tracking (top 1%, 5%, 10%).
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Iterator, List, Sequence, Tuple

import numpy as np

from .scoring_models import PayoutTier

__all__ = [
    "normalize_payout_tiers",
    "compute_expected_user_payouts_vectorized",
    "PayoutTolerance",
    "PayoutComputationResult",
]


@dataclass(frozen=True)
class _PayoutSpan:
    start: int
    end: int
    payout: float


@dataclass(frozen=True)
class PayoutTolerance:
    """Tolerance for verifying vectorized vs legacy payout calculation."""
    payout_dollars: float
    percentage_points: float


@dataclass(frozen=True)
class PayoutComputationResult:
    """Results from payout computation."""
    expected_payouts: np.ndarray
    win_rates: np.ndarray
    cash_rates: np.ndarray
    top_1pct_rates: np.ndarray
    top_5pct_rates: np.ndarray
    top_10pct_rates: np.ndarray
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


class _VectorizedPayoutAccumulator:
    """Per-world accumulator with positional rate tracking."""

    def __init__(
        self,
        user_weights: np.ndarray,
        field_weights: np.ndarray,
        payout_prefix: np.ndarray,
        total_entries: int,
    ) -> None:
        if user_weights.ndim != 1 or field_weights.ndim != 1:
            raise ValueError("weights must be 1D arrays")
        self.user_weights = user_weights
        self.field_weights = field_weights
        self.payout_prefix = payout_prefix
        self.total_entries = total_entries

        user_weights_f = user_weights.astype(np.float64)
        self.user_weights_positive = np.maximum(user_weights_f, 0.0)
        self.user_weights_float = np.maximum(user_weights_f, 1.0)
        self.total_field_weight = float(np.sum(field_weights))

        # Accumulators
        self.expected_sum = np.zeros_like(self.user_weights_float)
        self.win_sum = np.zeros_like(self.user_weights_float)
        self.cash_sum = np.zeros_like(self.user_weights_float)
        self.top_1pct_sum = np.zeros_like(self.user_weights_float)
        self.top_5pct_sum = np.zeros_like(self.user_weights_float)
        self.top_10pct_sum = np.zeros_like(self.user_weights_float)

        field_weights_f = field_weights.astype(np.float64)
        self.field_weights_positive = np.maximum(field_weights_f, 0.0)
        self.field_weights_float = np.maximum(field_weights_f, 1.0)
        self.field_expected_sum = np.zeros_like(self.field_weights_float)
        self.field_win_sum = np.zeros_like(self.field_weights_float)
        self.field_cash_sum = np.zeros_like(self.field_weights_float)

        self.world_count = 0

        # Positional cutoffs
        self.top_1pct_cutoff = max(1, int(math.ceil(total_entries * 0.01)))
        self.top_5pct_cutoff = max(1, int(math.ceil(total_entries * 0.05)))
        self.top_10pct_cutoff = max(1, int(math.ceil(total_entries * 0.10)))

    def spawn_empty(self) -> "_VectorizedPayoutAccumulator":
        """Create empty accumulator with same config for parallel processing."""
        clone = self.__class__.__new__(self.__class__)
        clone.user_weights = self.user_weights
        clone.field_weights = self.field_weights
        clone.payout_prefix = self.payout_prefix
        clone.total_entries = self.total_entries
        clone.user_weights_positive = self.user_weights_positive
        clone.user_weights_float = self.user_weights_float
        clone.total_field_weight = self.total_field_weight
        clone.expected_sum = np.zeros_like(self.expected_sum)
        clone.win_sum = np.zeros_like(self.win_sum)
        clone.cash_sum = np.zeros_like(self.cash_sum)
        clone.top_1pct_sum = np.zeros_like(self.top_1pct_sum)
        clone.top_5pct_sum = np.zeros_like(self.top_5pct_sum)
        clone.top_10pct_sum = np.zeros_like(self.top_10pct_sum)
        clone.field_weights_positive = self.field_weights_positive
        clone.field_weights_float = self.field_weights_float
        clone.field_expected_sum = np.zeros_like(self.field_expected_sum)
        clone.field_win_sum = np.zeros_like(self.field_win_sum)
        clone.field_cash_sum = np.zeros_like(self.field_cash_sum)
        clone.world_count = 0
        clone.top_1pct_cutoff = self.top_1pct_cutoff
        clone.top_5pct_cutoff = self.top_5pct_cutoff
        clone.top_10pct_cutoff = self.top_10pct_cutoff
        return clone

    def merge(self, other: "_VectorizedPayoutAccumulator") -> None:
        """Merge results from another accumulator."""
        self.expected_sum += other.expected_sum
        self.win_sum += other.win_sum
        self.cash_sum += other.cash_sum
        self.top_1pct_sum += other.top_1pct_sum
        self.top_5pct_sum += other.top_5pct_sum
        self.top_10pct_sum += other.top_10pct_sum
        self.field_expected_sum += other.field_expected_sum
        self.field_win_sum += other.field_win_sum
        self.field_cash_sum += other.field_cash_sum
        self.world_count += other.world_count

    def update(self, user_scores_w: np.ndarray, field_scores_w: np.ndarray) -> None:
        """Update accumulators with scores from one world."""
        user_scores = np.asarray(user_scores_w, dtype=np.float64)
        field_scores = np.asarray(field_scores_w, dtype=np.float64)
        if user_scores.ndim != 1 or field_scores.ndim != 1:
            raise ValueError("Accumulator update expects 1D score vectors")
        if user_scores.size != self.user_weights.size:
            raise ValueError("User scores length must match user weights")
        if field_scores.size != self.field_weights.size:
            raise ValueError("Field scores length must match field weights")

        self.world_count += 1

        # Field: sort once, cumulative weights
        f_order = np.argsort(field_scores)
        f_sorted_scores = field_scores[f_order]
        f_sorted_weights = self.field_weights[f_order]
        f_cum = np.cumsum(f_sorted_weights, dtype=np.float64)
        f_cum_pad = np.concatenate(([0.0], f_cum))
        total_field_weight = self.total_field_weight

        # Users: sort once, prefix & tie groups
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

        # Field lookups for all users
        idx_left = np.searchsorted(f_sorted_scores, u_scores, side="left")
        idx_right = np.searchsorted(f_sorted_scores, u_scores, side="right")
        field_weight_lt = f_cum_pad[idx_left]
        field_weight_le = f_cum_pad[idx_right]
        field_greater_weight = total_field_weight - field_weight_le
        field_equal_weight = field_weight_le - field_weight_lt

        # Entrants ahead / tie weights
        entrants_ahead = field_greater_weight + user_greater_weight
        tie_weight_base = field_equal_weight + user_equal_weight
        tie_weight = np.where(
            tie_weight_base > 0.0,
            tie_weight_base,
            self.user_weights_positive,
        )

        # Start/End ranks
        total_entries = int(self.total_entries)
        start_rank = np.floor(entrants_ahead).astype(np.int64) + 1
        end_rank = np.floor(entrants_ahead + tie_weight).astype(np.int64)
        np.clip(start_rank, 1, total_entries, out=start_rank)
        end_rank = np.maximum(end_rank, start_rank - 1)
        np.clip(end_rank, 0, total_entries, out=end_rank)

        # Payout split via prefix
        seg_sum = self.payout_prefix[end_rank] - self.payout_prefix[start_rank - 1]
        payout_each = np.divide(seg_sum, tie_weight, out=np.zeros_like(seg_sum), where=tie_weight > 0.0)

        # Accumulate
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

        # Positional rates (top 1%, 5%, 10%)
        self.top_1pct_sum += np.where(start_rank <= self.top_1pct_cutoff, user_w_pos, 0.0)
        self.top_5pct_sum += np.where(start_rank <= self.top_5pct_cutoff, user_w_pos, 0.0)
        self.top_10pct_sum += np.where(start_rank <= self.top_10pct_cutoff, user_w_pos, 0.0)

        # Field side (simplified - just track EV, win, cash)
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
    ) -> Tuple[
        np.ndarray,  # expected
        np.ndarray,  # win_rates
        np.ndarray,  # cash_rates
        np.ndarray,  # top_1pct_rates
        np.ndarray,  # top_5pct_rates
        np.ndarray,  # top_10pct_rates
        np.ndarray,  # field_expected
        np.ndarray,  # field_win
        np.ndarray,  # field_cash
    ]:
        """Finalize and return averaged rates."""
        if self.world_count == 0:
            zeros = np.zeros_like(self.expected_sum)
            field_zeros = np.zeros_like(self.field_expected_sum)
            return zeros, zeros.copy(), zeros.copy(), zeros.copy(), zeros.copy(), zeros.copy(), field_zeros, field_zeros.copy(), field_zeros.copy()

        expected = (self.expected_sum / self.world_count) / self.user_weights_float
        win_rates = (self.win_sum / self.world_count) / self.user_weights_float
        cash_rates = (self.cash_sum / self.world_count) / self.user_weights_float
        top_1pct_rates = (self.top_1pct_sum / self.world_count) / self.user_weights_float
        top_5pct_rates = (self.top_5pct_sum / self.world_count) / self.user_weights_float
        top_10pct_rates = (self.top_10pct_sum / self.world_count) / self.user_weights_float
        field_expected = (self.field_expected_sum / self.world_count) / self.field_weights_float
        field_win = (self.field_win_sum / self.world_count) / self.field_weights_float
        field_cash = (self.field_cash_sum / self.world_count) / self.field_weights_float
        return expected, win_rates, cash_rates, top_1pct_rates, top_5pct_rates, top_10pct_rates, field_expected, field_win, field_cash


def _resolve_workers(requested: int | str | None, task_count: int) -> int:
    """Translate a worker request into a concrete pool size."""
    if isinstance(requested, int):
        return max(1, requested)
    if isinstance(requested, str) and requested.lower() != "auto":
        raise ValueError("worker hint must be 'auto' or a positive integer")
    cpu = os.cpu_count() or 1
    if task_count <= 0:
        return max(1, cpu)
    return max(1, min(cpu, task_count))


def _iter_ranges(total: int, workers: int) -> List[Tuple[int, int]]:
    """Generate ranges for parallel processing."""
    if total <= 0 or workers <= 0:
        return []
    step = max(1, math.ceil(total / workers))
    ranges: List[Tuple[int, int]] = []
    for start in range(0, total, step):
        stop = min(start + step, total)
        ranges.append((start, stop))
    return ranges


def compute_expected_user_payouts_vectorized(
    user_scores: np.ndarray | None,
    field_scores: np.ndarray | None,
    user_weights: Sequence[int],
    field_weights: Sequence[int],
    payout_tiers: Sequence[PayoutTier],
    *,
    tolerance: PayoutTolerance | None = None,
    score_iterator: Iterator[Tuple[np.ndarray, np.ndarray]] | None = None,
    world_count: int | None = None,
    workers: int | str | None = None,
) -> PayoutComputationResult:
    """Vectorized payout computation with positional rate tracking.

    Parameters
    ----------
    user_scores : np.ndarray | None
        (n_users, n_worlds) score matrix, or None if using score_iterator
    field_scores : np.ndarray | None
        (n_field, n_worlds) score matrix, or None if using score_iterator
    user_weights : Sequence[int]
        Entry counts per user lineup
    field_weights : Sequence[int]
        Entry counts per field lineup
    payout_tiers : Sequence[PayoutTier]
        Payout structure
    tolerance : PayoutTolerance | None
        Optional tolerance for verification (not used in V1)
    score_iterator : Iterator | None
        Optional streaming iterator yielding (user_block, field_block) per chunk
    world_count : int | None
        Total worlds when using score_iterator
    workers : int | str | None
        Worker count for parallel processing

    Returns
    -------
    PayoutComputationResult
        Expected payouts, win rates, cash rates, positional rates
    """
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

        def _array_iterator() -> Iterator[Tuple[np.ndarray, np.ndarray]]:
            yield user_scores, field_scores

        chunk_iterator: Iterator[Tuple[np.ndarray, np.ndarray]] = _array_iterator()
    else:
        if world_count is None:
            raise ValueError("world_count must be provided when using score_iterator")
        world_total = int(world_count)
        chunk_iterator = score_iterator
        if user_scores is not None or field_scores is not None:
            raise ValueError("Provide either arrays or score_iterator, not both")
        if user_weights_arr.size <= 0 or field_weights_arr.size <= 0:
            raise ValueError("Weights must be non-empty for streaming payouts")

    total_entries = int(np.sum(field_weights_arr) + np.sum(np.maximum(user_weights_arr, 0)))
    if total_entries <= 0:
        raise ValueError("Total entrant weight must be positive")

    # Build payout prefix sum
    payout_per_place = np.zeros(total_entries, dtype=np.float64)
    for span in spans:
        start = min(span.start, total_entries)
        end = min(span.end, total_entries)
        if start > total_entries:
            continue
        payout_per_place[start - 1 : end] = span.payout
    payout_prefix = np.zeros(total_entries + 1, dtype=np.float64)
    payout_prefix[1:] = np.cumsum(payout_per_place)

    vector_acc = _VectorizedPayoutAccumulator(user_weights_arr, field_weights_arr, payout_prefix, total_entries)

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

    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    (
        expected,
        win_rates,
        cash_rates,
        top_1pct_rates,
        top_5pct_rates,
        top_10pct_rates,
        field_expected,
        field_win_rates,
        field_cash_rates,
    ) = vector_acc.finalize()

    return PayoutComputationResult(
        expected_payouts=expected,
        win_rates=win_rates,
        cash_rates=cash_rates,
        top_1pct_rates=top_1pct_rates,
        top_5pct_rates=top_5pct_rates,
        top_10pct_rates=top_10pct_rates,
        field_expected_payouts=field_expected,
        field_win_rates=field_win_rates,
        field_cash_rates=field_cash_rates,
        max_delta_payout=0.0,
        max_delta_percent=0.0,
    )
