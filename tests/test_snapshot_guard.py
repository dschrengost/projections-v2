from __future__ import annotations

import pandas as pd
import pytest

from projections.etl.snapshot_guard import compute_key_stats, enforce_non_regression


def test_compute_key_stats_counts_unique_keys() -> None:
    frame = pd.DataFrame(
        {
            "game_id": [1, 1, 2, 3],
            "player_id": [11, 11, 22, 33],
            "value": [0, 1, 2, 3],
        }
    )
    stats = compute_key_stats(frame, key_cols=("game_id", "player_id"))
    assert stats.rows == 4
    assert stats.unique_keys == 3


def test_enforce_non_regression_raises_on_key_drop() -> None:
    existing = pd.DataFrame({"game_id": [1, 2, 3]})
    candidate = pd.DataFrame({"game_id": [1, 2]})
    with pytest.raises(RuntimeError, match="key coverage regressed"):
        enforce_non_regression(
            dataset_name="odds",
            existing=existing,
            candidate=candidate,
            key_cols=("game_id",),
            allow_regression=False,
        )


def test_enforce_non_regression_allows_when_flag_enabled() -> None:
    existing = pd.DataFrame({"game_id": [1, 2, 3]})
    candidate = pd.DataFrame({"game_id": [1, 2]})
    enforce_non_regression(
        dataset_name="odds",
        existing=existing,
        candidate=candidate,
        key_cols=("game_id",),
        allow_regression=True,
    )
