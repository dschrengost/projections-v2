from __future__ import annotations

import pandas as pd

from projections.models.minutes_features import infer_feature_columns


def test_infer_feature_columns_excludes_retrain_only_artifacts() -> None:
    df = pd.DataFrame(
        {
            "game_id": ["g1", "g2"],
            "player_id": [1, 2],
            "team_id": [10, 11],
            "game_date": [pd.Timestamp("2026-02-01"), pd.Timestamp("2026-02-02")],
            "tip_ts": [pd.Timestamp("2026-02-01T00:00:00Z"), pd.Timestamp("2026-02-02T00:00:00Z")],
            "feature_as_of_ts": [pd.Timestamp("2026-02-01T00:00:00Z"), pd.Timestamp("2026-02-02T00:00:00Z")],
            "minutes": [12.0, 0.0],
            "feat_live_ok": [0.1, 0.2],
            "plays_target": [1, 0],
            "weight_recency": [1.0, 0.5],
        }
    )

    feature_columns = infer_feature_columns(df, target_col="minutes")

    assert "feat_live_ok" in feature_columns
    assert "plays_target" not in feature_columns
    assert "weight_recency" not in feature_columns
