from __future__ import annotations

import pandas as pd

from prefect_flows.live_nba_pipeline_v3 import (
    _left_overlay_from_source_by_keys,
    _sanitize_frame_to_expected_keys,
)


def test_sanitize_frame_to_expected_keys_filters_without_merge_side_effects() -> None:
    expected = pd.DataFrame(
        {
            "game_id": [1, 1, 2],
            "team_id": [10, 11, 20],
            "player_id": [100, 101, 200],
        }
    )
    df = pd.DataFrame(
        {
            "game_id": [1, 1, 3, None, 2],
            "team_id": [10, 99, 30, 10, 20],
            "player_id": [100, 999, 300, 100, 200],
            "value": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )

    out, report = _sanitize_frame_to_expected_keys(
        df,
        expected_keys_df=expected,
        key_cols=("game_id", "team_id", "player_id"),
        label="unit-test",
    )

    assert out[["game_id", "team_id", "player_id"]].values.tolist() == [
        [1, 10, 100],
        [2, 20, 200],
    ]
    assert out["value"].tolist() == [1.0, 5.0]
    assert report["rows_in"] == 5
    assert report["rows_out"] == 2
    assert report["dropped_null_key_rows"] == 1
    assert report["dropped_unexpected_key_rows"] == 2


def test_left_overlay_from_source_by_keys_updates_existing_and_adds_missing_columns() -> None:
    base = pd.DataFrame(
        {
            "game_id": [1, 1, 2],
            "team_id": [10, 11, 20],
            "player_id": [100, 101, 200],
            "dk_fpts_mean": [25.0, 30.0, 15.0],
        }
    )
    source = pd.DataFrame(
        {
            "game_id": [1, 1, 2, 2],
            "team_id": [10, 11, 20, 20],
            "player_id": [100, 101, 200, 200],
            "dk_fpts_mean": [26.0, None, 18.0, 19.0],
            "value": [5.1, 6.0, 3.7, 4.0],
        }
    )

    out = _left_overlay_from_source_by_keys(
        base,
        source_df=source,
        key_cols=("game_id", "team_id", "player_id"),
        value_cols=("dk_fpts_mean", "value"),
        label="unit-test",
    )

    # existing values are overwritten only when source is non-null
    assert out["dk_fpts_mean"].tolist() == [26.0, 30.0, 19.0]
    # new columns are created from source overlay
    assert out["value"].tolist() == [5.1, 6.0, 4.0]
