from __future__ import annotations

import pandas as pd
import pytest

from projections.eval.walk_forward import assert_fold_integrity


def test_assert_fold_integrity_raises_on_key_overlap() -> None:
    train_df = pd.DataFrame(
        {
            "game_id": [1, 2],
            "player_id": [10, 11],
            "game_date": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")],
        }
    )
    val_df = pd.DataFrame(
        {
            "game_id": [2, 3],
            "player_id": [11, 12],
            "game_date": [pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-05")],
        }
    )
    with pytest.raises(ValueError, match=r"overlap on keys"):
        assert_fold_integrity(
            train_df,
            val_df,
            time_col="game_date",
            train_end_ts=pd.Timestamp("2024-01-03"),
            fold_id="fold_test",
        )


def test_assert_fold_integrity_raises_on_time_overlap() -> None:
    train_df = pd.DataFrame(
        {
            "game_id": [1],
            "player_id": [10],
            "game_date": [pd.Timestamp("2024-01-03")],
        }
    )
    val_df = pd.DataFrame(
        {
            "game_id": [2],
            "player_id": [11],
            "game_date": [pd.Timestamp("2024-01-03")],
        }
    )
    with pytest.raises(ValueError, match=r"val min .* <= train_end_ts"):
        assert_fold_integrity(
            train_df,
            val_df,
            time_col="game_date",
            train_end_ts=pd.Timestamp("2024-01-03"),
            fold_id="fold_test",
        )

