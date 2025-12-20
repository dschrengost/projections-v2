"""Utilities for scaling and filtering contest-entry weights."""

from __future__ import annotations

from typing import List, Sequence, Tuple, TypeVar


T = TypeVar("T")


def scale_integer_weights_to_target(
    weights: Sequence[int],
    target_sum: int,
    *,
    min_weight: int = 0,
) -> List[int]:
    """Scale non-negative integer weights to sum exactly to ``target_sum``.

    Uses a largest-remainder (Hamilton) rounding scheme to preserve proportions.
    """
    if target_sum <= 0:
        raise ValueError("target_sum must be positive")
    if not weights:
        raise ValueError("weights must be non-empty")
    if min_weight < 0:
        raise ValueError("min_weight must be non-negative")

    weights_int = [int(w) for w in weights]
    if any(w < 0 for w in weights_int):
        raise ValueError("weights must be non-negative integers")

    baseline_total = int(min_weight) * len(weights_int)
    if baseline_total > target_sum:
        raise ValueError("min_weight is too large for the requested target_sum")
    if baseline_total == target_sum:
        return [int(min_weight)] * len(weights_int)

    # Remaining mass after minimum allocation.
    target_sum_remaining = target_sum - baseline_total

    total = sum(weights_int)
    if total <= 0:
        # Uniform fallback
        weights_int = [1] * len(weights_int)
        total = len(weights_int)

    scale = target_sum_remaining / total
    raw = [w * scale for w in weights_int]
    floors = [int(x) for x in raw]
    remainders = [x - f for x, f in zip(raw, floors)]

    missing = target_sum_remaining - sum(floors)
    if missing > 0:
        for idx, _ in sorted(enumerate(remainders), key=lambda kv: kv[1], reverse=True)[:missing]:
            floors[idx] += 1
    elif missing < 0:
        to_remove = -missing
        for idx, _ in sorted(enumerate(remainders), key=lambda kv: kv[1]):
            if to_remove <= 0:
                break
            if floors[idx] > 0:
                floors[idx] -= 1
                to_remove -= 1

    # Final correction for any off-by-one from float precision
    delta = target_sum_remaining - sum(floors)
    if delta > 0:
        for idx in range(len(floors)):
            if delta <= 0:
                break
            floors[idx] += 1
            delta -= 1
    elif delta < 0:
        for idx in range(len(floors)):
            if delta >= 0:
                break
            if floors[idx] > 0:
                floors[idx] -= 1
                delta += 1

    if sum(floors) != target_sum_remaining:
        raise RuntimeError("Failed to scale weights to the requested target_sum")

    if min_weight:
        return [int(min_weight) + int(w) for w in floors]
    return floors


def drop_zero_weight_items(items: Sequence[T], weights: Sequence[int]) -> Tuple[List[T], List[int]]:
    """Filter out items whose corresponding weight is zero or negative."""
    if len(items) != len(weights):
        raise ValueError("items and weights must have the same length")
    kept_items: List[T] = []
    kept_weights: List[int] = []
    for item, weight in zip(items, weights):
        w = int(weight)
        if w <= 0:
            continue
        kept_items.append(item)
        kept_weights.append(w)
    return kept_items, kept_weights


def scale_float_weights_to_target(
    weights: Sequence[float],
    target_sum: int,
    *,
    min_weight: int = 0,
) -> List[int]:
    """Scale non-negative float weights to integer weights summing to ``target_sum``."""
    if target_sum <= 0:
        raise ValueError("target_sum must be positive")
    if not weights:
        raise ValueError("weights must be non-empty")
    if min_weight < 0:
        raise ValueError("min_weight must be non-negative")

    weights_f = [float(w) for w in weights]
    if any(w < 0 for w in weights_f):
        raise ValueError("weights must be non-negative")

    baseline_total = int(min_weight) * len(weights_f)
    if baseline_total > target_sum:
        raise ValueError("min_weight is too large for the requested target_sum")
    if baseline_total == target_sum:
        return [int(min_weight)] * len(weights_f)

    remaining = target_sum - baseline_total
    total = float(sum(weights_f))
    if total <= 0:
        # Uniform fallback
        weights_f = [1.0] * len(weights_f)
        total = float(len(weights_f))

    scale = remaining / total
    raw = [w * scale for w in weights_f]
    floors = [int(w) for w in raw]
    remainders = [w - f for w, f in zip(raw, floors)]

    missing = remaining - sum(floors)
    if missing > 0:
        for idx, _ in sorted(enumerate(remainders), key=lambda kv: kv[1], reverse=True)[:missing]:
            floors[idx] += 1
    elif missing < 0:
        to_remove = -missing
        for idx, _ in sorted(enumerate(remainders), key=lambda kv: kv[1]):
            if to_remove <= 0:
                break
            if floors[idx] > 0:
                floors[idx] -= 1
                to_remove -= 1

    delta = remaining - sum(floors)
    if delta > 0:
        for idx in range(len(floors)):
            if delta <= 0:
                break
            floors[idx] += 1
            delta -= 1
    elif delta < 0:
        for idx in range(len(floors)):
            if delta >= 0:
                break
            if floors[idx] > 0:
                floors[idx] -= 1
                delta += 1

    if sum(floors) != remaining:
        raise RuntimeError("Failed to scale float weights to the requested target_sum")

    if min_weight:
        return [int(min_weight) + int(w) for w in floors]
    return floors
