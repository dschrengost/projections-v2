from __future__ import annotations

import pandas as pd

from projections.rates_v1.preprocess import (
    TRACKING_FILL_FEATURES,
    apply_tracking_fill_values,
    fit_tracking_fill_values,
    resolve_tracking_fill_values,
)


def test_fit_tracking_fill_values_uses_train_stats() -> None:
    train_df = pd.DataFrame(
        {
            "track_touches_per_min_szn": [0.1, 0.3, None, 0.5],
            "track_pull_up_3pa_share_szn": [0.2, None, 0.4, 0.6],
            "track_role_cluster": [2, 2, None, 3],
            "track_role_is_low_minutes": [1, None, 1, 0],
        }
    )
    feature_cols = [
        "track_touches_per_min_szn",
        "track_pull_up_3pa_share_szn",
        "track_role_cluster",
        "track_role_is_low_minutes",
    ]

    fill_values = fit_tracking_fill_values(train_df, feature_cols)

    assert fill_values["track_touches_per_min_szn"] == 0.3
    assert fill_values["track_pull_up_3pa_share_szn"] == 0.4
    assert fill_values["track_role_cluster"] == 2.0
    assert fill_values["track_role_is_low_minutes"] == 1.0


def test_apply_and_resolve_tracking_fill_values_bundle_meta_and_legacy_fallback() -> None:
    features = pd.DataFrame({"track_touches_per_min_szn": [None], "other": [1.0]})
    feature_cols = ["track_touches_per_min_szn", "track_role_cluster"]

    explicit = resolve_tracking_fill_values(
        {"preprocess": {"tracking_fill_values": {"track_touches_per_min_szn": 0.25, "track_role_cluster": -1}}},
        feature_cols,
    )
    filled = apply_tracking_fill_values(features, explicit)
    assert filled.loc[0, "track_touches_per_min_szn"] == 0.25
    assert int(filled.loc[0, "track_role_cluster"]) == -1

    legacy = resolve_tracking_fill_values({}, list(TRACKING_FILL_FEATURES))
    assert legacy["track_touches_per_min_szn"] == 0.0
    assert legacy["track_role_cluster"] == 0.0

