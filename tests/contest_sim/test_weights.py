from __future__ import annotations

from projections.contest_sim.weights import drop_zero_weight_items, scale_integer_weights_to_target


def test_scale_integer_weights_to_target_preserves_sum() -> None:
    scaled = scale_integer_weights_to_target([1, 2, 3], 60)
    assert sum(scaled) == 60


def test_scale_integer_weights_to_target_with_min_weight() -> None:
    scaled = scale_integer_weights_to_target([1, 1, 1], 5, min_weight=1)
    assert sum(scaled) == 5
    assert min(scaled) >= 1


def test_drop_zero_weight_items_filters() -> None:
    items, weights = drop_zero_weight_items(["a", "b", "c"], [1, 0, 2])
    assert items == ["a", "c"]
    assert weights == [1, 2]

