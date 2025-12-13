"""Lineup scoring utilities for shared worlds artifacts.

The module provides low-level score matrix generation (:func:`score_lineups`)
as well as higher-level contest scoring helpers (:func:`score_contests`) that
produce the data models defined in ``scoring_models``.
"""

from __future__ import annotations

import math
import os
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Union

import numpy as np
from scipy import sparse

from .api import WorldsReader
from Simulator.lineups import DraftKingsLineup

from .leverage import assign_delta_bucket, assign_percentile_bucket
from .payouts import (
    PayoutTolerance,
    compute_expected_user_payouts,
    compute_expected_user_payouts_vectorized,
)
from .scoring_models import (
    ContestConfig,
    ContestResult,
    DupePenaltyConfig,
    FieldComparison,
    LeverageBuckets,
    LineupMetrics,
    ScoringRequest,
    TelemetrySummary,
)
from .telemetry import TelemetryRecorder

__all__ = [
    "LineupScoringError",
    "UnionScoreHandle",
    "score_lineups",
    "score_union_lineups",
    "score_contests",
]


WORLD_CHUNK = 8192
LINEUP_CHUNK = 4096
_SPARSE_LINEUP_THRESHOLD = 10_000
_OUTPUT_MEMMAP_THRESHOLD = 1_000_000_000  # bytes (~1 GiB)
_DEFAULT_PAYOUT_TOLERANCE = PayoutTolerance(payout_dollars=0.04, percentage_points=0.01)
_MIN_STD = 1e-9
_MIN_MASS = 1e-9
_AUTO_BIAS_SCALE_MIN = 0.1
_AUTO_BIAS_SCALE_MAX = 10.0
_PAYOUT_CHUNK_TARGET_BYTES = 64 * 1024 * 1024  # 64 MiB per field block
_PAYOUT_CHUNK_MAX_WORLDS = 1024


class LineupScoringError(ValueError):
    """Raised when supplied lineups cannot be aligned to the worlds index."""


@dataclass(frozen=True)
class UnionScoreHandle:
    scores: Union[np.ndarray, np.memmap]
    index_map: Dict[tuple[str, ...], int]
    shape: tuple[int, int]
    dtype: np.dtype
    matrix_path: Path | None
    sparse: bool
    world_workers: int = 1

    def __post_init__(self):
        if self.shape != self.scores.shape:
            raise ValueError("shape must match scores array shape")

    @property
    def matrix(self) -> Union[np.ndarray, np.memmap]:
        return self.scores


def _normalize_player_id(raw: object) -> str:
    pid = str(raw).strip()
    if not pid:
        raise LineupScoringError("Lineup contains blank or null player identifier")
    return pid


def _normalize_lineups(lineups: Iterable[Sequence[object]]) -> List[List[str]]:
    normalized: List[List[str]] = []
    for idx, lineup in enumerate(lineups):
        if lineup is None:
            raise LineupScoringError(f"Lineup #{idx} is None; expected an iterable of player IDs")
        try:
            players = [_normalize_player_id(pid) for pid in lineup if str(pid).strip()]
        except LineupScoringError as exc:
            raise LineupScoringError(f"Lineup #{idx}: {exc}") from exc
        if not players:
            raise LineupScoringError(f"Lineup #{idx} has no valid player identifiers")
        normalized.append(players)
    if not normalized:
        raise LineupScoringError("No lineups provided for scoring")
    return normalized


def _canonicalize_lineup(players: Sequence[str]) -> tuple[str, ...]:
    """Return an order-sensitive key for deduping identical lineups."""

    return tuple(players)


def _build_union_lineups(lineups: Iterable[Sequence[object]]) -> tuple[list[list[str]], dict[tuple[str, ...], int]]:
    unique: list[list[str]] = []
    index_map: dict[tuple[str, ...], int] = {}
    for lineup in _normalize_lineups(lineups):
        key = _canonicalize_lineup(lineup)
        if key not in index_map:
            index_map[key] = len(unique)
            unique.append(list(lineup))
    return unique, index_map


def _build_lineup_matrix(
    index_map: dict[str, int],
    lineups: List[List[str]],
    *,
    dtype: Union[str, np.dtype] = np.float32,
    sparse_threshold: int | None = None,
) -> tuple[Union[np.ndarray, sparse.csr_matrix], bool]:
    """Create a player (rows) × lineup (columns) indicator matrix."""

    dtype = np.dtype(dtype)
    num_players = len(index_map)
    num_lineups = len(lineups)
    threshold = _SPARSE_LINEUP_THRESHOLD if sparse_threshold is None else int(sparse_threshold)
    use_sparse = threshold > 0 and num_lineups >= threshold

    if use_sparse:
        rows: List[int] = []
        cols: List[int] = []
        data: List[float] = []
        for col, players in enumerate(lineups):
            seen: set[str] = set()
            for pid in players:
                if pid in seen:
                    raise LineupScoringError(f"Lineup #{col} contains duplicate player_id '{pid}'")
                seen.add(pid)
                try:
                    row = index_map[pid]
                except KeyError as exc:
                    raise LineupScoringError(
                        f"Lineup #{col} references unknown player_id '{pid}'"
                    ) from exc
                rows.append(row)
                cols.append(col)
                data.append(1.0)
        matrix = sparse.csr_matrix((data, (rows, cols)), shape=(num_players, num_lineups), dtype=dtype)
        return matrix, True

    matrix = np.zeros((num_players, num_lineups), dtype=dtype)
    for col, players in enumerate(lineups):
        seen: set[str] = set()
        for pid in players:
            if pid in seen:
                raise LineupScoringError(f"Lineup #{col} contains duplicate player_id '{pid}'")
            seen.add(pid)
            try:
                row = index_map[pid]
            except KeyError as exc:
                raise LineupScoringError(f"Lineup #{col} references unknown player_id '{pid}'") from exc
            matrix[row, col] = matrix[row, col] + 1.0
    return matrix, False


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


def _iter_ranges(total: int, step: int) -> List[tuple[int, int]]:
    """Return deterministic (start, stop) windows covering ``total`` items."""

    if step <= 0:
        raise ValueError("step must be positive")
    if total <= 0:
        return []
    ranges: List[tuple[int, int]] = []
    for start in range(0, total, step):
        stop = min(start + step, total)
        ranges.append((start, stop))
    return ranges


def score_lineups(
    reader: WorldsReader,
    lineups: Iterable[Sequence[object]],
    *,
    chunk_size: int = 8192,
    dtype: Union[str, np.dtype] = np.float32,
    world_workers: int | str | None = None,
    sparse_threshold: int | None = None,
) -> np.ndarray:
    """Score lineups against a worlds tensor.

    Parameters
    ----------
    reader:
        Worlds handle that exposes ``shape()``, ``get_block()``, and
        ``player_index_map()``.
    lineups:
        Iterable of lineups, each a sequence of player identifiers that exist in
        the worlds' index map.
    chunk_size:
        Number of worlds rows to process per chunk. Larger chunks improve
        throughput at the cost of memory. Defaults to 8192.
    dtype:
        Floating dtype for the returned scores. Defaults to ``np.float32``.
    world_workers:
        Optional worker hint controlling how many parallel lineup chunks to
        process per worlds block. ``None``/``"auto"`` uses the available CPU
        count (bounded by the number of lineup chunks); integers force an
        explicit pool size.
    sparse_threshold:
        When positive, switch to a sparse indicator matrix once the lineup
        count meets or exceeds this value. ``0`` or negative disables the
        sparse fallback regardless of lineup volume. Defaults to the module
        constant when unset.

    Returns
    -------
    np.ndarray
        Array with shape ``(num_lineups, W)`` where each row contains the
        per-world fantasy totals for the corresponding lineup.

    Notes
    -----
    Large outputs (estimated > ~1 GiB) may be backed by a temporary
    ``numpy.memmap`` on disk. The returned object still quacks like an
    ``ndarray`` and can be consumed the same way.
    """

    dtype = np.dtype(dtype)
    normalized = _normalize_lineups(lineups)
    index_map = reader.player_index_map()
    lineup_matrix, is_sparse = _build_lineup_matrix(
        index_map,
        normalized,
        dtype=dtype,
        sparse_threshold=sparse_threshold,
    )
    if not is_sparse:
        lineup_matrix = np.ascontiguousarray(lineup_matrix)

    total_worlds, num_players = reader.shape()
    if lineup_matrix.shape[0] != num_players:
        raise LineupScoringError(
            "Lineup matrix player dimension does not match worlds tensor columns"
        )

    world_chunk = WORLD_CHUNK if chunk_size is None else int(max(1, chunk_size))
    world_chunk = min(world_chunk, total_worlds)
    base_lineup_chunk = (
        min(LINEUP_CHUNK, lineup_matrix.shape[1]) if lineup_matrix.shape[1] else LINEUP_CHUNK
    )

    num_lineups = lineup_matrix.shape[1]
    chunk_groups = max(1, math.ceil(num_lineups / base_lineup_chunk)) if num_lineups else 1
    worker_count = _resolve_workers(world_workers, chunk_groups)
    if is_sparse:
        worker_count = 1

    lineup_chunk = base_lineup_chunk
    if worker_count > 1 and num_lineups:
        target = max(1, math.ceil(num_lineups / worker_count))
        lineup_chunk = min(base_lineup_chunk, target)

    bytes_needed = num_lineups * total_worlds * dtype.itemsize
    use_memmap = bytes_needed > _OUTPUT_MEMMAP_THRESHOLD

    if use_memmap:
        tmp_dir = Path(tempfile.mkdtemp(prefix="worlds_scores_"))
        mmap_path = tmp_dir / "scores.dat"
        scores = np.memmap(mmap_path, dtype=dtype, mode="w+", shape=(num_lineups, total_worlds))
    else:
        scores = np.zeros((num_lineups, total_worlds), dtype=dtype)

    executor: ThreadPoolExecutor | None = None
    if worker_count > 1 and num_lineups:
        executor = ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="worlds-score")

    try:
        for world_start in range(0, total_worlds, world_chunk):
            world_stop = min(world_start + world_chunk, total_worlds)
            block = reader.get_block(world_start, world_stop)
            block_array = np.asarray(block, order="C")
            if block_array.dtype != dtype:
                block_array = block_array.astype(dtype, copy=False)
            if not block_array.flags.c_contiguous:
                block_array = np.ascontiguousarray(block_array)
            if block_array.shape[1] != num_players:
                raise LineupScoringError(
                    f"Worlds block returned shape {block_array.shape}, expected (*, {num_players})"
                )

            block_T = block_array.T
            lineup_ranges = _iter_ranges(num_lineups, lineup_chunk)
            if not lineup_ranges:
                continue

            if executor is not None and not is_sparse:

                def _dense_worker(start: int, stop: int) -> None:
                    lineup_block = lineup_matrix[:, start:stop]
                    target = scores[start:stop, world_start:world_stop]
                    out_view = np.empty((stop - start, block_T.shape[1]), dtype=dtype)
                    np.dot(lineup_block.T, block_T, out=out_view)
                    target[:] = out_view

                futures = [executor.submit(_dense_worker, start, stop) for start, stop in lineup_ranges]
                for future in futures:
                    future.result()
            else:
                buffer = None
                if not is_sparse:
                    buffer = np.empty((min(lineup_chunk, num_lineups), block_T.shape[1]), dtype=dtype)
                for lineup_start, lineup_stop in lineup_ranges:
                    target = scores[lineup_start:lineup_stop, world_start:world_stop]
                    if is_sparse:
                        lineup_block = lineup_matrix[:, lineup_start:lineup_stop]
                        chunk_scores = block_array @ lineup_block
                        np.copyto(target, chunk_scores.T, casting="same_kind")
                    else:
                        lineup_block = lineup_matrix[:, lineup_start:lineup_stop]
                        out_view = buffer[: lineup_stop - lineup_start, :]
                        np.dot(lineup_block.T, block_T, out=out_view)
                        target[:] = out_view
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    return scores


def score_union_lineups(
    reader: WorldsReader,
    lineups: Iterable[Sequence[object]],
    *,
    dtype: Union[str, np.dtype] = np.float32,
    world_workers: int | str | None = None,
    chunk_size: int | None = None,
    memmap_dir: Path | None = None,
    lineup_batch: int | None = None,
    sparse_threshold: int | None = None,
) -> UnionScoreHandle:
    """Score the union of lineups once and expose a reusable score matrix.

    Parameters
    ----------
    world_workers:
        Worker hint controlling how many lineup chunks are processed in
        parallel per worlds block. ``None``/``"auto"`` selects a CPU-bound
        default based on available lineup chunks.
    chunk_size:
        Override for worlds block size. ``None`` keeps the module default.
    memmap_dir:
        Optional directory to back the output matrix via ``numpy.memmap`` when
        memory pressure is high.
    lineup_batch:
        Hard cap on the number of lineups processed per batch regardless of
        chunk size, useful when the union includes extreme duplicate counts.
    sparse_threshold:
        Override for the dense→sparse crossover. ``None`` keeps the module
        default; ``0`` or negative forces dense processing.
    """

    dtype = np.dtype(dtype)
    unique_lineups, union_index = _build_union_lineups(lineups)
    if not unique_lineups:
        raise LineupScoringError("No lineups provided for union scoring")

    player_index_map = reader.player_index_map()
    lineup_matrix, is_sparse = _build_lineup_matrix(
        player_index_map,
        unique_lineups,
        dtype=dtype,
        sparse_threshold=sparse_threshold,
    )
    if not is_sparse:
        lineup_matrix = np.ascontiguousarray(lineup_matrix)

    total_worlds, num_players = reader.shape()
    if lineup_matrix.shape[0] != num_players:
        raise LineupScoringError(
            "Lineup matrix player dimension does not match worlds tensor columns"
        )

    world_chunk = WORLD_CHUNK if chunk_size is None else int(max(1, chunk_size))
    world_chunk = min(world_chunk, total_worlds)
    base_lineup_chunk = (
        min(LINEUP_CHUNK, lineup_matrix.shape[1]) if lineup_matrix.shape[1] else LINEUP_CHUNK
    )

    num_lineups = lineup_matrix.shape[1]
    bytes_needed = num_lineups * total_worlds * dtype.itemsize
    use_memmap = memmap_dir is not None or bytes_needed > _OUTPUT_MEMMAP_THRESHOLD

    tmp_dir: Path | None = None
    if use_memmap:
        tmp_dir = Path(
            tempfile.mkdtemp(
                prefix="union_scores_",
                dir=str(memmap_dir) if memmap_dir is not None else None,
            )
        )
        matrix_path = tmp_dir / "scores.dat"
        scores = np.memmap(matrix_path, dtype=dtype, mode="w+", shape=(num_lineups, total_worlds))
    else:
        matrix_path = None
        scores = np.zeros((num_lineups, total_worlds), dtype=dtype)

    batch_size = lineup_batch or num_lineups
    step = base_lineup_chunk
    if batch_size:
        step = min(step, batch_size)
    chunk_groups = max(1, math.ceil(num_lineups / step)) if num_lineups else 1
    worker_count = _resolve_workers(world_workers, chunk_groups)
    if is_sparse:
        worker_count = 1
    if worker_count > 1 and num_lineups:
        target = max(1, math.ceil(num_lineups / worker_count))
        step = min(step, target)

    executor: ThreadPoolExecutor | None = None
    if worker_count > 1 and num_lineups:
        executor = ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="worlds-union")

    try:
        for world_start in range(0, total_worlds, world_chunk):
            world_stop = min(world_start + world_chunk, total_worlds)
            block = reader.get_block(world_start, world_stop)
            block_array = np.asarray(block, order="C")
            if block_array.dtype != dtype:
                block_array = block_array.astype(dtype, copy=False)
            if not block_array.flags.c_contiguous:
                block_array = np.ascontiguousarray(block_array)
            if block_array.shape[1] != num_players:
                raise LineupScoringError(
                    f"Worlds block returned shape {block_array.shape}, expected (*, {num_players})"
                )

            block_T = block_array.T
            lineup_ranges = _iter_ranges(num_lineups, step)
            if not lineup_ranges:
                continue

            if executor is not None and not is_sparse:

                def _dense_worker(start: int, stop: int) -> None:
                    lineup_block = lineup_matrix[:, start:stop]
                    target = scores[start:stop, world_start:world_stop]
                    out_view = np.empty((stop - start, block_T.shape[1]), dtype=dtype)
                    np.dot(lineup_block.T, block_T, out=out_view)
                    target[:] = out_view

                futures = [executor.submit(_dense_worker, start, stop) for start, stop in lineup_ranges]
                for future in futures:
                    future.result()
            else:
                buffer = None
                if not is_sparse:
                    buffer = np.empty((min(step, num_lineups), block_T.shape[1]), dtype=dtype)
                for lineup_start, lineup_stop in lineup_ranges:
                    target = scores[lineup_start:lineup_stop, world_start:world_stop]
                    if is_sparse:
                        lineup_block = lineup_matrix[:, lineup_start:lineup_stop]
                        chunk_scores = block_array @ lineup_block
                        np.copyto(target, chunk_scores.T, casting="same_kind")
                    else:
                        lineup_block = lineup_matrix[:, lineup_start:lineup_stop]
                        out_view = buffer[: lineup_stop - lineup_start, :]
                        np.dot(lineup_block.T, block_T, out=out_view)
                        target[:] = out_view
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    handle = UnionScoreHandle(
        scores=scores,
        index_map=union_index,
        shape=scores.shape,
        dtype=dtype,
        matrix_path=matrix_path,
        sparse=is_sparse,
        world_workers=worker_count,
    )

    return handle


def score_multi_contests(
    reader: WorldsReader,
    request: ScoringRequest,
    *,
    world_workers: int | str | None = None,
    contest_workers: int | str | None = None,
    chunk_size: int | None = None,
    memmap_dir: Path | None = None,
    lineup_batch: int | None = None,
) -> List[ContestResult]:
    if not request.contests:
        raise LineupScoringError("ScoringRequest must include at least one contest")

    dtype_name = request.options.dtype if request.options and request.options.dtype else "float32"
    dtype_np = np.dtype(dtype_name)
    chunk_override = request.options.chunk_size if request.options else chunk_size

    user_lineups = request.user_lineups
    user_sequences = [_lineup_player_ids(lineup) for lineup in user_lineups]
    user_weight_array = np.array([
        _normalize_weight(lineup.weight) for lineup in user_lineups
    ], dtype=np.int64)
    user_exposure = _compute_exposure(user_lineups, weights_override=user_weight_array)

    dupe_penalty_cfg: DupePenaltyConfig | None = None
    if request.options and getattr(request.options, "dupe_penalty", None):
        option_value = request.options.dupe_penalty
        if isinstance(option_value, DupePenaltyConfig):
            dupe_penalty_cfg = option_value
        elif isinstance(option_value, dict):
            dupe_penalty_cfg = DupePenaltyConfig.from_mapping(option_value)
    dupe_penalty_enabled = bool(dupe_penalty_cfg and dupe_penalty_cfg.enabled)

    field_sequences_by_contest: list[list[list[str]]] = []
    all_sequences: list[list[str]] = list(user_sequences)

    for contest in request.contests:
        if not contest.field_lineups:
            raise LineupScoringError(
                f"Contest {contest.contest_id} missing field_lineups; generation not yet implemented"
            )
        field_sequences = [_lineup_player_ids(lineup) for lineup in contest.field_lineups]
        field_sequences_by_contest.append(field_sequences)
        all_sequences.extend(field_sequences)

    sparse_threshold_override = (
        getattr(request.options, "sparse_lineup_threshold", None) if request.options else None
    )

    union_handle = score_union_lineups(
        reader,
        all_sequences,
        dtype=dtype_name,
        world_workers=world_workers,
        chunk_size=chunk_override,
        memmap_dir=memmap_dir,
        lineup_batch=lineup_batch,
        sparse_threshold=sparse_threshold_override,
    )

    scores_matrix = union_handle.matrix
    world_count = scores_matrix.shape[1]

    total_sequences = len(all_sequences)
    union_lineups_count = union_handle.shape[0]
    reuse_ratio = 0.0
    if total_sequences:
        reuse_ratio = 1.0 - float(union_lineups_count) / float(total_sequences)

    resolved_world_workers = int(max(1, union_handle.world_workers))

    resolved_contest_workers = _resolve_workers(contest_workers, max(1, world_count))

    def _row_index(players: Sequence[str]) -> int:
        key = _canonicalize_lineup(players)
        try:
            return union_handle.index_map[key]
        except KeyError as exc:
            raise LineupScoringError("Lineup sequence missing from union handle") from exc

    user_row_indices = [_row_index(seq) for seq in user_sequences]
    user_scores = scores_matrix[user_row_indices, :]
    if world_count > 0:
        user_mean_scores = np.asarray(user_scores.mean(axis=1), dtype=np.float64)
    else:
        user_mean_scores = np.zeros(len(user_lineups), dtype=np.float64)

    results: List[ContestResult] = []

    for contest, field_sequences in zip(request.contests, field_sequences_by_contest):
        field_row_indices = [_row_index(seq) for seq in field_sequences]

        # --- SANITY: empirical top-1 rate for the best user lineup (by mean) ---
        try:
            best_user_idx = int(np.argmax(user_mean_scores))
            u = user_scores[best_user_idx, :]            # (W,)
            F = scores_matrix[field_row_indices, :]      # (F, W)

            # Strict wins: user > everyone
            strict_wins = np.sum(np.all(F < u, axis=0))

            # Ties where user is among the top score
            ties_mask = np.all(F <= u, axis=0) & np.any(np.isclose(F, u, atol=1e-9), axis=0)
            ties = int(np.sum(ties_mask))

            empirical_win_rate = strict_wins / float(world_count)
            print(f"[sanity] contest={contest.contest_id} best_user_idx={best_user_idx} "
                  f"emp_win={empirical_win_rate*100:.3f}%  strict_wins={strict_wins} ties={ties}/{world_count}")
        except Exception as e:
            print(f"[sanity] error: {e!r}")

        field_weight_int = _quantize_field_weights(
            contest.field_lineups,
            field_size=contest.field_size,
        )
        print(f"DEBUG: Contest '{contest.contest_id}'")
        print(
            f"DEBUG: Number of unique field lineups provided: {len(field_row_indices)}"
        )
        print(f"DEBUG: Quantized field weights: {field_weight_int.tolist()}")
        print(
            f"DEBUG: Calculated total field entries: {field_weight_int.sum()}"
        )
        field_exposure = _compute_exposure(
            contest.field_lineups,
            weights_override=field_weight_int,
        )

        field_weight_array = field_weight_int.astype(np.float64, copy=False)
        total_field_entries = float(field_weight_array.sum())
        unique_lineups_count = len(field_row_indices)
        duplicate_mask = field_weight_array > 1.0
        duplicate_lineups = int(np.count_nonzero(duplicate_mask))
        duplicate_entries = (
            float(field_weight_array[duplicate_mask].sum()) if duplicate_lineups else 0.0
        )
        max_entries_per_lineup = (
            float(field_weight_array.max()) if field_weight_array.size else 0.0
        )
        mean_entries_per_lineup = (
            float(total_field_entries / unique_lineups_count)
            if unique_lineups_count
            else 0.0
        )
        duplicate_ratio = (
            float(duplicate_lineups / unique_lineups_count)
            if unique_lineups_count
            else 0.0
        )
        duplicate_entry_share = (
            float(duplicate_entries / total_field_entries)
            if total_field_entries
            else 0.0
        )

        field_sum = np.zeros(len(field_row_indices), dtype=np.float64)
        chunk_counter = 0
        lambda_values: np.ndarray | None = None
        penalty_factors: np.ndarray | None = None
        expected_payout_penalized: np.ndarray | None = None

        def _score_iterator() -> Iterable[tuple[np.ndarray, np.ndarray]]:
            nonlocal chunk_counter
            if world_count <= 0:
                return
            dtype_local = dtype_np
            field_lineup_count = max(1, len(field_row_indices))
            bytes_per_world = field_lineup_count * dtype_local.itemsize
            max_chunk_from_budget = max(
                1, int(_PAYOUT_CHUNK_TARGET_BYTES // max(bytes_per_world, 1))
            )
            payout_chunk = max(
                1,
                min(world_count, max_chunk_from_budget, _PAYOUT_CHUNK_MAX_WORLDS),
            )

            for world_start in range(0, world_count, payout_chunk):
                world_stop = min(world_start + payout_chunk, world_count)
                user_block = np.asarray(
                    scores_matrix[user_row_indices, world_start:world_stop],
                    dtype=dtype_local,
                    order="C",
                )
                field_block = np.asarray(
                    scores_matrix[field_row_indices, world_start:world_stop],
                    dtype=dtype_local,
                    order="C",
                )
                field_sum[:] += field_block.sum(axis=1, dtype=np.float64)
                chunk_counter += world_stop - world_start
                yield user_block, field_block

        with TelemetryRecorder() as recorder:
            recorder.set_extra("world_workers", resolved_world_workers)
            recorder.set_extra("contest_workers", resolved_contest_workers)
            recorder.set_extra("union_lineups", union_lineups_count)
            recorder.set_extra("reuse_ratio", reuse_ratio)
            recorder.set_extra("field_unique_lineups", unique_lineups_count)
            recorder.set_extra("field_duplicate_lineups", duplicate_lineups)
            recorder.set_extra("field_duplicate_entry_share", duplicate_entry_share)
            payout_result = compute_expected_user_payouts_vectorized(
                None,
                None,
                user_weight_array.tolist(),
                field_weight_int.tolist(),
                contest.payout_table,
                tolerance=_DEFAULT_PAYOUT_TOLERANCE,
                score_iterator=_score_iterator(),
                world_count=world_count,
                workers=resolved_contest_workers,
            )
            if isinstance(union_handle.scores, np.memmap):
                recorder.note_memmap_bytes(union_handle.scores.nbytes)
            recorder.set_extra("payout_delta_max_dollars", payout_result.max_delta_payout)
            recorder.set_extra(
                "roi_delta_max_percentage_points", payout_result.max_delta_percent
            )
            expected_payout_penalized = payout_result.expected_payouts
            if dupe_penalty_enabled and dupe_penalty_cfg is not None:
                entrants_for_hazard = total_field_entries
                if entrants_for_hazard <= 0:
                    entrants_for_hazard = float(contest.field_size or 0)
                if entrants_for_hazard <= 0:
                    entrants_for_hazard = float(field_weight_array.sum()) if field_weight_array.size else float(len(contest.field_lineups))
                if world_count > 0:
                    field_mean_array = field_sum / float(world_count)
                else:
                    field_mean_array = np.zeros(len(field_row_indices), dtype=np.float64)
                field_features = _lineup_features(
                    contest.field_lineups,
                    field_exposure,
                    fallback_projection=field_mean_array,
                )
                user_features = _lineup_features(
                    user_lineups,
                    field_exposure,
                    fallback_projection=user_mean_scores,
                )
                field_stats = _field_dup_stats(field_features, dupe_penalty_cfg)
                lambda_values = _dupe_hazard_lambda(
                    user_features,
                    field_stats,
                    entrants_for_hazard,
                    dupe_penalty_cfg,
                )
                if dupe_penalty_cfg.auto_bias:
                    lambda_values = _apply_auto_bias(lambda_values, dupe_penalty_cfg, user_features.weights)
                penalty_factors = _tie_penalty(lambda_values, dupe_penalty_cfg)
                expected_payout_penalized = payout_result.expected_payouts * penalty_factors
                for key, value in _dupe_penalty_stats(lambda_values, penalty_factors, user_features).items():
                    recorder.set_extra(f"dupe_{key}", value)
            if chunk_counter <= 0:
                chunk_counter = world_count
            recorder.set_extra("chunk_iterations", chunk_counter)
        telemetry = recorder.summary()

        if expected_payout_penalized is not None:
            expected_payout = np.asarray(expected_payout_penalized, dtype=np.float64)
        else:
            expected_payout = payout_result.expected_payouts
        win_rate = payout_result.win_rates
        cash_rate = payout_result.cash_rates

        if lambda_values is None:
            lambda_values = np.zeros(len(user_lineups), dtype=np.float64)
        if penalty_factors is None:
            penalty_factors = np.ones(len(user_lineups), dtype=np.float64)

        lineup_metrics: List[LineupMetrics] = []

        if world_count > 0:
            field_means = (field_sum / float(world_count)).tolist()
        else:
            field_means = [0.0] * len(field_row_indices)
        percentile_weights = (
            field_weight_int.tolist() if field_weight_int.size else [1] * len(field_row_indices)
        )

        field_summary = FieldComparison(
            distribution={
                "entrants_total": total_field_entries,
                "unique_lineups": float(unique_lineups_count),
                "duplicate_lineups": float(duplicate_lineups),
                "duplicate_ratio": duplicate_ratio,
                "duplicate_entry_share": duplicate_entry_share,
                "max_entries_per_lineup": max_entries_per_lineup,
                "mean_entries_per_lineup": mean_entries_per_lineup,
            }
        )

        if world_count > 0:
            field_scores_matrix = np.asarray(scores_matrix[field_row_indices, :])
            field_mean_array = field_scores_matrix.mean(axis=1)
            field_std_array = field_scores_matrix.std(axis=1, ddof=0)
            field_p90_array = np.percentile(field_scores_matrix, 90, axis=1)
            field_p95_array = np.percentile(field_scores_matrix, 95, axis=1)
            field_p99_array = np.percentile(field_scores_matrix, 99, axis=1)
        else:
            lineup_count = len(field_row_indices)
            field_scores_matrix = np.zeros((lineup_count, 0), dtype=np.float64)
            field_mean_array = np.zeros(lineup_count, dtype=np.float64)
            field_std_array = np.zeros(lineup_count, dtype=np.float64)
            field_p90_array = np.zeros(lineup_count, dtype=np.float64)
            field_p95_array = np.zeros(lineup_count, dtype=np.float64)
            field_p99_array = np.zeros(lineup_count, dtype=np.float64)

        for idx, lineup in enumerate(user_lineups):
            scores_vector = user_scores[idx]
            mean = float(scores_vector.mean())
            std = float(scores_vector.std(ddof=0))
            ceiling = float(np.percentile(scores_vector, 90))
            p95 = float(np.percentile(scores_vector, 95))
            p99 = float(np.percentile(scores_vector, 99))
            dupe_lambda_value = float(lambda_values[idx]) if idx < len(lambda_values) else 0.0
            dupe_penalty_value = float(penalty_factors[idx]) if idx < len(penalty_factors) else 1.0

            delta_values = []
            chalk_values = []
            for entry in lineup.slots.values():
                pid = entry.player_id
                delta_values.append(user_exposure.get(pid, 0.0) - field_exposure.get(pid, 0.0))
                chalk_values.append(field_exposure.get(pid, 0.0))
            ownership_delta = float(np.mean(delta_values)) if delta_values else 0.0
            chalk_index = float(np.mean(chalk_values)) if chalk_values else 0.0

            percentile = _percentile_rank(field_means, percentile_weights, mean)
            buckets = LeverageBuckets()
            delta_bucket = assign_delta_bucket(ownership_delta)
            percentile_bucket = assign_percentile_bucket(percentile)
            buckets.delta_bucket = delta_bucket.label
            buckets.delta_color = delta_bucket.color
            buckets.pl_bucket = percentile_bucket.label
            buckets.pl_color = percentile_bucket.color

            expected_payout_value = float(expected_payout[idx]) if idx < expected_payout.size else contest.entry_fee
            expected_value = expected_payout_value - contest.entry_fee

            metrics = LineupMetrics(
                lineup_id=lineup.lineup_id,
                slots=dict(lineup.slots),
                mean=mean,
                std=std,
                ceiling=ceiling,
                p95=p95,
                p99=p99,
                expected_payout=expected_payout_value,
                expected_value=expected_value,
                roi=(expected_payout_value - contest.entry_fee) / contest.entry_fee,
                win_rate=float(win_rate[idx]),
                cash_rate=float(cash_rate[idx]),
                ownership_delta=ownership_delta,
                percentile_leverage=percentile,
                cli=ownership_delta * percentile / 100.0,
                chalk_index=chalk_index,
                buckets=buckets,
                metadata=dict(getattr(lineup, "metadata", {})),
                dupe_risk_lambda=dupe_lambda_value,
                dupe_penalty_factor=dupe_penalty_value,
            )
            lineup_metrics.append(metrics)

        field_lineup_metrics: List[LineupMetrics] = []
        if contest.field_lineups:
            fallback_length = len(field_row_indices)
            field_expected = getattr(
                payout_result,
                "field_expected_payouts",
                np.zeros(fallback_length, dtype=np.float64),
            )
            field_win_rates = getattr(
                payout_result,
                "field_win_rates",
                np.zeros(fallback_length, dtype=np.float64),
            )
            field_cash_rates = getattr(
                payout_result,
                "field_cash_rates",
                np.zeros(fallback_length, dtype=np.float64),
            )
            for idx, lineup in enumerate(contest.field_lineups):
                mean = float(field_mean_array[idx]) if idx < field_mean_array.size else 0.0
                std = float(field_std_array[idx]) if idx < field_std_array.size else 0.0
                ceiling = float(field_p90_array[idx]) if idx < field_p90_array.size else 0.0
                p95 = float(field_p95_array[idx]) if idx < field_p95_array.size else 0.0
                p99 = float(field_p99_array[idx]) if idx < field_p99_array.size else 0.0

                delta_values = []
                chalk_values = []
                for entry in lineup.slots.values():
                    pid = entry.player_id
                    delta_values.append(user_exposure.get(pid, 0.0) - field_exposure.get(pid, 0.0))
                    chalk_values.append(field_exposure.get(pid, 0.0))
                ownership_delta = float(np.mean(delta_values)) if delta_values else 0.0
                chalk_index = float(np.mean(chalk_values)) if chalk_values else 0.0

                percentile = _percentile_rank(field_means, percentile_weights, mean)
                buckets = LeverageBuckets()
                delta_bucket = assign_delta_bucket(ownership_delta)
                percentile_bucket = assign_percentile_bucket(percentile)
                buckets.delta_bucket = delta_bucket.label
                buckets.delta_color = delta_bucket.color
                buckets.pl_bucket = percentile_bucket.label
                buckets.pl_color = percentile_bucket.color

                expected_payout_value = float(field_expected[idx]) if idx < field_expected.size else contest.entry_fee
                expected_value = expected_payout_value - contest.entry_fee

                metrics = LineupMetrics(
                    lineup_id=lineup.lineup_id,
                    slots=dict(lineup.slots),
                    mean=mean,
                    std=std,
                    ceiling=ceiling,
                    p95=p95,
                    p99=p99,
                    expected_payout=expected_payout_value,
                    expected_value=expected_value,
                    roi=(expected_payout_value - contest.entry_fee) / contest.entry_fee,
                    win_rate=float(field_win_rates[idx]) if idx < field_win_rates.size else 0.0,
                    cash_rate=float(field_cash_rates[idx]) if idx < field_cash_rates.size else 0.0,
                    ownership_delta=ownership_delta,
                    percentile_leverage=percentile,
                    cli=ownership_delta * percentile / 100.0,
                    chalk_index=chalk_index,
                    buckets=buckets,
                    metadata=dict(getattr(lineup, "metadata", {})),
                    dupe_risk_lambda=0.0,
                    dupe_penalty_factor=1.0,
                )
                field_lineup_metrics.append(metrics)

        contest_result = ContestResult(
            contest_id=contest.contest_id,
            field_summary=field_summary,
            user_lineup_metrics=lineup_metrics,
            field_lineup_metrics=field_lineup_metrics,
            telemetry=telemetry,
        )
        results.append(contest_result)

    _ = contest_workers  # reserved for future concurrency support

    return results


def _lineup_player_ids(lineup: DraftKingsLineup) -> List[str]:
    return [entry.player_id for entry in lineup.slots.values()]


def _compute_exposure(
    lineups: List[DraftKingsLineup],
    *,
    weights_override: Sequence[float] | None = None,
) -> Dict[str, float]:
    totals: Dict[str, float] = defaultdict(float)
    total_weight = 0.0
    if weights_override is not None and len(weights_override) != len(lineups):
        raise ValueError("weights_override length must match lineups")
    for idx, lineup in enumerate(lineups):
        if weights_override is not None:
            weight = float(weights_override[idx])
        else:
            weight = float(lineup.weight or 1.0)
        if weight <= 0:
            continue
        total_weight += weight
        for entry in lineup.slots.values():
            totals[entry.player_id] += weight
    if total_weight <= 0:
        return {pid: 0.0 for pid in totals}
    return {pid: (value / total_weight) * 100.0 for pid, value in totals.items()}


def _quantize_field_weights(
    lineups: List[DraftKingsLineup],
    *,
    field_size: int | None,
) -> np.ndarray:
    raw = np.array(
        [max(0.0, float(getattr(lineup, "weight", 0.0) or 0.0)) for lineup in lineups],
        dtype=np.float64,
    )
    if raw.size == 0:
        return np.zeros(0, dtype=np.int64)

    metadata_counts = np.array([
        _extract_lineup_metadata_count(getattr(lineup, "metadata", {}))
        for lineup in lineups
    ], dtype=np.float64)

    total_raw = float(raw.sum())
    if (field_size or 0) <= 0 and np.any(metadata_counts > 0.0):
        metadata_sum = float(metadata_counts.sum())
        if metadata_sum > total_raw:
            raw = raw.copy()
            for idx, value in enumerate(metadata_counts):
                if value > 0.0:
                    raw[idx] = max(raw[idx], value)
            total_raw = float(raw.sum())

    total_raw = float(raw.sum())
    target_entries = int(field_size or 0)
    if target_entries <= 0:
        if total_raw > 0:
            target_entries = max(int(round(total_raw)), 1)
        else:
            target_entries = raw.size
    target_entries = max(target_entries, 0)
    if target_entries == 0:
        return np.zeros_like(raw, dtype=np.int64)

    if total_raw <= 0:
        base = target_entries // raw.size
        remainder = target_entries - base * raw.size
        weights = np.full(raw.size, base, dtype=np.int64)
        if remainder > 0:
            weights[:remainder] += 1
        return weights

    scaled = raw * (target_entries / total_raw)
    floored = np.floor(scaled).astype(np.int64)
    remainder = target_entries - int(floored.sum())

    if remainder > 0:
        fractional = scaled - floored
        order = np.argsort(-fractional)
        for idx in order[:remainder]:
            floored[idx] += 1
    elif remainder < 0:
        fractional = scaled - floored
        order = np.argsort(fractional)
        deficit = -remainder
        for idx in order:
            if deficit <= 0:
                break
            if floored[idx] <= 0:
                continue
            floored[idx] -= 1
            deficit -= 1

    return floored


_COUNT_METADATA_HINTS: tuple[str, ...] = (
    "entrycount",
    "entries",
    "entrant",
    "entrants",
    "entrytotal",
    "totalentries",
    "numentries",
    "numentry",
    "dupe",
    "dupes",
    "duplication",
    "duplicate",
    "dupcount",
    "dupecount",
)


def _metadata_key_indicates_entry_count(key_norm: str) -> bool:
    if not key_norm:
        return False
    return any(hint in key_norm for hint in _COUNT_METADATA_HINTS)


def _extract_lineup_metadata_count(metadata: Mapping[str, object]) -> float:
    if not metadata:
        return 0.0

    best = 0.0
    for key, value in metadata.items():
        key_norm = _normalize_metadata_key(str(key))
        if not _metadata_key_indicates_entry_count(key_norm):
            continue
        number: float | None
        try:
            number = float(value)
        except (TypeError, ValueError):
            text = str(value).strip()
            if not text:
                continue
            text = text.replace(",", "")
            try:
                number = float(text)
            except (TypeError, ValueError):
                continue
        if not np.isfinite(number):
            continue
        if number > best:
            best = number
    return max(0.0, best)


def _normalize_metadata_key(value: str) -> str:
    if not value:
        return ""
    return "".join(ch for ch in value.lower() if ch.isalnum())


@dataclass
class _LineupFeatureBundle:
    own_sum: np.ndarray
    proj_sum: np.ndarray
    salary_sum: np.ndarray
    chalk_k: np.ndarray
    weights: np.ndarray


def _resolve_player_ownership(entry: PlayerEntry, ownership_lookup: Dict[str, float] | None) -> float:
    if entry.ownership_pct is not None:
        try:
            return float(entry.ownership_pct)
        except (TypeError, ValueError):
            return 0.0
    if ownership_lookup is None:
        return 0.0
    try:
        return float(ownership_lookup.get(entry.player_id, 0.0))
    except (TypeError, ValueError):
        return 0.0


def _resolve_lineup_projection(lineup: DraftKingsLineup, fallback: float | None) -> float:
    metadata = getattr(lineup, "metadata", {}) or {}
    for key in ("ProjPoints", "Proj", "Projection", "ProjPts"):
        value = metadata.get(key)
        if value not in (None, ""):
            try:
                return float(value)
            except (TypeError, ValueError):
                continue

    total = 0.0
    missing_projection = False
    for entry in lineup.slots.values():
        if entry.projection is None:
            missing_projection = True
            continue
        try:
            total += float(entry.projection)
        except (TypeError, ValueError):
            missing_projection = True
    if not missing_projection:
        return total

    if fallback is not None:
        try:
            return float(fallback)
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _lineup_features(
    lineups: Sequence[DraftKingsLineup],
    ownership_lookup: Dict[str, float] | None,
    *,
    fallback_projection: Sequence[float] | None = None,
    topk: int = 3,
) -> _LineupFeatureBundle:
    count = len(lineups)
    own_sum = np.zeros(count, dtype=np.float64)
    proj_sum = np.zeros(count, dtype=np.float64)
    salary_sum = np.zeros(count, dtype=np.float64)
    chalk_k = np.zeros(count, dtype=np.float64)
    weights = np.zeros(count, dtype=np.float64)

    for idx, lineup in enumerate(lineups):
        entries = list(lineup.slots.values())
        owns = np.array(
            [_resolve_player_ownership(entry, ownership_lookup) for entry in entries],
            dtype=np.float64,
        )
        salaries = np.array([float(entry.salary or 0.0) for entry in entries], dtype=np.float64)

        fallback_value = None
        if fallback_projection is not None and idx < len(fallback_projection):
            fallback_value = fallback_projection[idx]
        proj_sum[idx] = _resolve_lineup_projection(lineup, fallback_value)
        own_sum[idx] = owns.sum() if owns.size else 0.0
        salary_sum[idx] = salaries.sum() if salaries.size else 0.0
        if owns.size:
            k = min(topk, owns.size)
            if k > 0:
                partition = np.partition(owns, owns.size - k)
                chalk_k[idx] = partition[-k:].sum()
        weights[idx] = float(lineup.weight or 1.0)

    return _LineupFeatureBundle(own_sum=own_sum, proj_sum=proj_sum, salary_sum=salary_sum, chalk_k=chalk_k, weights=weights)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    total_weight = float(np.sum(weights))
    if total_weight <= 0:
        return float(np.mean(values))
    return float(np.sum(values * weights) / total_weight)


def _weighted_std(values: np.ndarray, weights: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    total_weight = float(np.sum(weights))
    if total_weight <= 0:
        return float(np.std(values))
    mean = np.sum(values * weights) / total_weight
    variance = np.sum(weights * (values - mean) ** 2) / total_weight
    variance = max(0.0, float(variance))
    return float(np.sqrt(variance))


def _field_dup_stats(features: _LineupFeatureBundle, params: DupePenaltyConfig) -> dict[str, float]:
    weights = features.weights if features.weights.size else np.ones_like(features.own_sum)
    qual = features.proj_sum - float(params.lambda_own) * features.own_sum

    qual_mean = _weighted_mean(qual, weights)
    qual_std = max(_weighted_std(qual, weights), _MIN_STD)
    salary_mean = _weighted_mean(features.salary_sum, weights)
    salary_std = max(_weighted_std(features.salary_sum, weights), _MIN_STD)

    log_mu = (
        float(params.alpha) * np.log(np.maximum(features.own_sum, float(params.epsilon)))
        + float(params.beta) * ((qual - qual_mean) / qual_std)
        + float(params.gamma) * ((features.salary_sum - salary_mean) / salary_std)
        + float(params.delta) * (features.chalk_k / 100.0)
        + float(params.bias)
    )
    mu = np.exp(log_mu)
    dup_ref_mass = np.sum(mu * weights)

    return {
        "qual_mean": float(qual_mean),
        "qual_std": float(max(qual_std, _MIN_STD)),
        "salary_mean": float(salary_mean),
        "salary_std": float(max(salary_std, _MIN_STD)),
        "dup_ref_mass": float(max(dup_ref_mass, _MIN_MASS)),
    }


def _dupe_hazard_lambda(
    features: _LineupFeatureBundle,
    field_stats: dict[str, float],
    entrants: float,
    params: DupePenaltyConfig,
) -> np.ndarray:
    if entrants <= 0:
        return np.zeros_like(features.own_sum)

    qual = features.proj_sum - float(params.lambda_own) * features.own_sum
    qual_std = max(float(field_stats.get("qual_std", 0.0)), _MIN_STD)
    salary_std = max(float(field_stats.get("salary_std", 0.0)), _MIN_STD)
    z_qual = (qual - float(field_stats.get("qual_mean", 0.0))) / qual_std
    z_salary = (features.salary_sum - float(field_stats.get("salary_mean", 0.0))) / salary_std

    log_mu = (
        float(params.alpha) * np.log(np.maximum(features.own_sum, float(params.epsilon)))
        + float(params.beta) * z_qual
        + float(params.gamma) * z_salary
        + float(params.delta) * (features.chalk_k / 100.0)
        + float(params.bias)
    )
    mu = np.exp(log_mu)
    ref_mass = max(float(field_stats.get("dup_ref_mass", 0.0)), _MIN_MASS)
    lam = (float(entrants) / ref_mass) * mu

    if params.max_lambda is not None:
        lam = np.minimum(lam, float(params.max_lambda))
    lam = np.clip(lam, 0.0, None)
    return lam


def _apply_auto_bias(lam: np.ndarray, params: DupePenaltyConfig, weights: np.ndarray) -> np.ndarray:
    if not params.auto_bias or lam.size == 0:
        return lam

    total_weight = float(np.sum(weights))
    if total_weight > 0:
        order = np.argsort(lam)
        sorted_lam = lam[order]
        sorted_weights = weights[order]
        cumulative = np.cumsum(sorted_weights)
        median_threshold = total_weight / 2.0
        median_idx = np.searchsorted(cumulative, median_threshold, side="left")
        median_value = float(sorted_lam[min(median_idx, sorted_lam.size - 1)])
    else:
        median_value = float(np.median(lam))

    target = float(params.auto_bias_target or 0.0)
    if median_value <= max(float(params.min_lambda), _MIN_MASS) or target <= 0.0:
        return lam

    scale = target / median_value
    scale = float(np.clip(scale, _AUTO_BIAS_SCALE_MIN, _AUTO_BIAS_SCALE_MAX))
    return lam * scale


def _tie_penalty(lam: np.ndarray, params: DupePenaltyConfig) -> np.ndarray:
    lam = np.clip(lam, 0.0, None)
    tiny = max(float(params.min_lambda), _MIN_MASS)
    penalty = np.ones_like(lam, dtype=np.float64)
    mask = lam > tiny
    if np.any(mask):
        penalty[mask] = (1.0 - np.exp(-lam[mask])) / lam[mask]
    penalty = np.clip(penalty, 0.0, 1.0)
    weight = float(params.weight)
    if not np.isclose(weight, 1.0):
        penalty = np.power(penalty, weight)
    return penalty


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return 0.0
    try:
        corr = np.corrcoef(a, b)[0, 1]
    except FloatingPointError:
        return 0.0
    if np.isnan(corr):
        return 0.0
    return float(corr)


def _dupe_penalty_stats(
    lam: np.ndarray,
    penalty: np.ndarray,
    features: _LineupFeatureBundle,
) -> dict[str, float]:
    if lam.size == 0:
        return {}

    stats: dict[str, float] = {}
    stats["lambda_median"] = float(np.median(lam))
    stats["lambda_p90"] = float(np.percentile(lam, 90))
    stats["lambda_p99"] = float(np.percentile(lam, 99))
    stats["penalty_median"] = float(np.median(penalty))
    stats["penalty_p90"] = float(np.percentile(penalty, 90))
    stats["penalty_p99"] = float(np.percentile(penalty, 99))
    stats["lambda_own_corr"] = _safe_corr(lam, features.own_sum)
    stats["lambda_chalk_corr"] = _safe_corr(lam, features.chalk_k)
    return stats


def _percentile_rank(field_means: List[float], field_weights: List[int], value: float) -> float:
    if not field_means or not field_weights:
        return 0.0
    ordered = sorted(zip(field_means, field_weights), key=lambda pair: pair[0])
    total_weight = float(sum(weight for _, weight in ordered))
    if total_weight <= 0:
        return 0.0
    weight_leq = 0.0
    for mean, weight in ordered:
        if mean <= value + 1e-9:
            weight_leq += weight
    return min(100.0, max(0.0, (weight_leq / total_weight) * 100.0))


def _normalize_weight(weight: float) -> int:
    if weight is None or weight <= 0:
        return 1
    return max(1, int(round(weight)))


def score_contests(reader: WorldsReader, request: ScoringRequest) -> List[ContestResult]:
    return score_multi_contests(reader, request)
