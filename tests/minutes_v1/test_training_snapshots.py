from __future__ import annotations

import pandas as pd

from projections.minutes_v1.training_snapshots import merge_features_with_labels


def test_merge_features_with_labels_overrides_minutes_and_keeps_snapshots() -> None:
    features = pd.DataFrame(
        {
            "game_id": [1, 1],
            "player_id": [10, 10],
            "team_id": [100, 100],
            "feature_as_of_ts": ["2025-01-01T18:00:00Z", "2025-01-01T19:00:00Z"],
            "tip_ts": ["2025-01-01T20:00:00Z", "2025-01-01T20:00:00Z"],
            # Live features may include minutes but they are not the target; ensure we override.
            "minutes": [None, None],
            "some_feature": [1.0, 2.0],
        }
    )
    labels = pd.DataFrame(
        {
            "game_id": [1],
            "player_id": [10],
            "team_id": [100],
            "minutes": [33.5],
            "starter_flag_label": [1],
        }
    )

    merged = merge_features_with_labels(features, labels)
    assert len(merged) == 2
    assert merged["minutes"].tolist() == [33.5, 33.5]
    assert merged["starter_flag_label"].tolist() == [1, 1]


def test_merge_features_with_labels_dedupes_labels_by_latest_frozen_ts() -> None:
    features = pd.DataFrame(
        {
            "game_id": [1],
            "player_id": [10],
            "team_id": [100],
            "feature_as_of_ts": ["2025-01-01T19:00:00Z"],
            "tip_ts": ["2025-01-01T20:00:00Z"],
        }
    )
    labels = pd.DataFrame(
        {
            "game_id": [1, 1],
            "player_id": [10, 10],
            "team_id": [100, 100],
            "minutes": [10.0, 20.0],
            "starter_flag_label": [0, 1],
            "label_frozen_ts": ["2025-01-02T00:00:00Z", "2025-01-03T00:00:00Z"],
        }
    )

    merged = merge_features_with_labels(features, labels)
    assert merged["minutes"].tolist() == [20.0]
    assert merged["starter_flag_label"].tolist() == [1]

