from projections.rates_v1.features import get_rates_feature_sets


def test_stage5_includes_tracking_3pa_profile_features() -> None:
    feature_sets = get_rates_feature_sets()
    stage5 = feature_sets["stage5_fta_tracking"]

    assert "track_catch_shoot_fg3a_per_min_szn" in stage5
    assert "track_pull_up_fg3a_per_min_szn" in stage5
    assert "track_pull_up_3pa_share_szn" in stage5
