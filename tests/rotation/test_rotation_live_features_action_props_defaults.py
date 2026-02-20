from __future__ import annotations

import pandas as pd

from projections.rotation.live_features_v1 import build_rotation_set_minutes_v1_features


def test_build_rotation_live_features_fills_action_props_defaults_when_missing() -> None:
    minutes_features = pd.DataFrame(
        {
            "game_id": [22500001, 22500001],
            "team_id": [1610612747, 1610612747],
            "player_id": [1, 2],
            # Required by apply_odds_missing_flags (even if not in feature_columns)
            "spread_home": [-3.5, -3.5],
            "total": [228.5, 228.5],
        }
    )

    feature_columns = [
        "an_has_any_props",
        "an_pts_line",
        "an_pts_p_over",
        "an_has_pts",
    ]

    result = build_rotation_set_minutes_v1_features(
        minutes_features,
        team_priors=pd.DataFrame(),
        player_priors=pd.DataFrame(),
        feature_columns=feature_columns,
    )

    out = result.features
    assert set(feature_columns).issubset(out.columns)
    assert out["an_has_any_props"].astype(float).eq(0.0).all()
    assert out["an_has_pts"].astype(float).eq(0.0).all()
    assert out["an_pts_line"].astype(float).eq(0.0).all()
    assert out["an_pts_p_over"].astype(float).eq(0.5).all()


def test_build_rotation_live_features_preserves_existing_action_props_values() -> None:
    minutes_features = pd.DataFrame(
        {
            "game_id": [22500001, 22500001],
            "team_id": [1610612747, 1610612747],
            "player_id": [1, 2],
            "spread_home": [-3.5, -3.5],
            "total": [228.5, 228.5],
            # One existing prop value; other columns missing.
            "an_pts_line": [21.5, 0.0],
        }
    )

    feature_columns = [
        "an_has_any_props",
        "an_pts_line",
        "an_pts_p_over",
        "an_has_pts",
    ]

    result = build_rotation_set_minutes_v1_features(
        minutes_features,
        team_priors=pd.DataFrame(),
        player_priors=pd.DataFrame(),
        feature_columns=feature_columns,
    )

    out = result.features
    assert float(out.loc[out["player_id"] == 1, "an_pts_line"].iloc[0]) == 21.5
    assert float(out.loc[out["player_id"] == 2, "an_pts_line"].iloc[0]) == 0.0
    assert out["an_pts_p_over"].astype(float).eq(0.5).all()
