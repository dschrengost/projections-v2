from __future__ import annotations

import pandas as pd

from scripts.rotation.build_joint_rotation_rates_dataset_v1 import (
    _apply_tracking_context_asof_fallback,
    _load_rates_labels,
)


def test_load_rates_labels_keeps_requested_context_columns(tmp_path) -> None:
    path = tmp_path / "rates_training_base.parquet"
    pd.DataFrame(
        {
            "game_id": [1],
            "team_id": [10],
            "player_id": [100],
            "game_date": ["2025-11-01"],
            "minutes_actual": [30.0],
            "fga2_per_min": [0.2],
            "track_touches_per_min_szn": [2.5],
        }
    ).to_parquet(path, index=False)

    rates_df, meta = _load_rates_labels(
        [path],
        context_cols=["track_touches_per_min_szn", "track_missing_col"],
    )

    assert "track_touches_per_min_szn" in rates_df.columns
    assert "track_missing_col" not in rates_df.columns
    assert meta["context_cols_present"] == ["track_touches_per_min_szn"]
    assert meta["context_cols_missing"] == ["track_missing_col"]


def test_apply_tracking_context_asof_fallback_fills_missing_rows() -> None:
    features_df = pd.DataFrame(
        {
            "season": [2025, 2025, 2025],
            "game_date": ["2025-10-25", "2025-10-27", "2025-10-27"],
            "player_id": [100, 100, 200],
            "track_touches_per_min_szn": [pd.NA, 9.0, pd.NA],
            "track_role_cluster": [pd.NA, 4.0, pd.NA],
        }
    )
    tracking_df = pd.DataFrame(
        {
            "season": [2025, 2025],
            "game_date": ["2025-10-24", "2025-10-26"],
            "player_id": [100, 100],
            "track_touches_per_min_szn": [5.0, 7.0],
            "track_role_cluster": [2.0, 3.0],
        }
    )

    out, meta = _apply_tracking_context_asof_fallback(features_df, tracking_df)

    assert float(out.loc[0, "track_touches_per_min_szn"]) == 5.0
    assert float(out.loc[0, "track_role_cluster"]) == 2.0
    assert float(out.loc[1, "track_touches_per_min_szn"]) == 9.0
    assert float(out.loc[1, "track_role_cluster"]) == 4.0
    assert pd.isna(out.loc[2, "track_touches_per_min_szn"])
    assert pd.isna(out.loc[2, "track_role_cluster"])

    assert int(out.loc[0, "track_touches_per_min_szn_missing"]) == 0
    assert int(out.loc[1, "track_touches_per_min_szn_missing"]) == 0
    assert int(out.loc[2, "track_touches_per_min_szn_missing"]) == 1

    assert meta["rows_filled_any_tracking"] == 1
    assert meta["before_coverage_any_tracking"] < meta["after_coverage_any_tracking"]
