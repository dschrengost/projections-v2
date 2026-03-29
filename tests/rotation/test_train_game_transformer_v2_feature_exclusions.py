from __future__ import annotations

import pandas as pd
import pytest

from scripts.rotation.train_game_transformer_v2 import (
    _add_efficiency_sidecar_interaction_features,
    _coerce_join_keys,
    _exclude_feature_patterns,
    _infer_feature_columns,
)


def test_infer_feature_columns_excludes_same_game_rotation_leak_fields() -> None:
    features = pd.DataFrame(
        {
            'game_id': [1, 1],
            'team_id': [10, 10],
            'player_id': [101, 102],
            'game_date': ['2026-01-01', '2026-01-01'],
            'minutes_from_stints': [34.0, 12.0],
            'num_stints': [7, 3],
            'max_stint_len_real': [12.0, 5.0],
            'depth_6': [1, 0],
            'starter_pool_minutes': [155.0, 155.0],
            'lineup_available': [1, 1],
            'lineup_starter_announced': [1, 0],
            'minutes_from_stints_prior_20': [30.0, 10.0],
            'prior_play_prob': [0.9, 0.2],
        }
    )
    labels = pd.DataFrame(
        {
            'game_id': [1, 1],
            'team_id': [10, 10],
            'player_id': [101, 102],
            'game_date': ['2026-01-01', '2026-01-01'],
            'minutes_label': [34.0, 12.0],
        }
    )

    cols = _infer_feature_columns(features, labels)

    assert 'minutes_from_stints' not in cols
    assert 'num_stints' not in cols
    assert 'max_stint_len_real' not in cols
    assert 'depth_6' not in cols
    assert 'starter_pool_minutes' not in cols

    assert 'minutes_from_stints_prior_20' in cols
    assert 'lineup_available' in cols
    assert 'lineup_starter_announced' in cols
    assert 'prior_play_prob' in cols


def test_coerce_join_keys_handles_datetime64_game_date_without_pandas_index_error() -> None:
    df = pd.DataFrame(
        {
            "game_id": [1, 1],
            "team_id": [10, 10],
            "player_id": [101, 102],
            "game_date": pd.to_datetime(["2026-01-01 19:00:00", "2026-01-01 21:30:00"]),
        }
    )

    out = _coerce_join_keys(df, name="unit")

    assert out["game_date"].dt.hour.eq(0).all()
    assert out["game_date"].dt.date.nunique() == 1


def test_exclude_feature_patterns_can_drop_raw_context_bucket_priors() -> None:
    cols = [
        "minutes_from_stints_ctx_same_pos_thin_prior_20",
        "started_proxy_rate_ctx_same_pos_normal_prior_10",
        "ctx_same_pos_thin_prior_n_games_20",
        "ctx_minutes_from_stints_prior_20",
        "ctx_started_proxy_rate_prior_20",
        "ctx_prior_n_games_20",
        "ctx_prior_backoff_used_20",
    ]

    kept = _exclude_feature_patterns(
        cols,
        exclude_patterns=[
            r"(^|_)ctx_same_pos_(thin|normal|deep)_prior_",
            r"_ctx_same_pos_(thin|normal|deep)_prior_",
        ],
    )

    assert "minutes_from_stints_ctx_same_pos_thin_prior_20" not in kept
    assert "started_proxy_rate_ctx_same_pos_normal_prior_10" not in kept
    assert "ctx_same_pos_thin_prior_n_games_20" not in kept
    assert "ctx_minutes_from_stints_prior_20" in kept
    assert "ctx_started_proxy_rate_prior_20" in kept
    assert "ctx_prior_n_games_20" in kept
    assert "ctx_prior_backoff_used_20" in kept


def test_add_efficiency_sidecar_interaction_features_builds_matchup_deltas() -> None:
    df = pd.DataFrame(
        {
            "fg2_pct_prior_5": [0.58],
            "opp_fg2_pct_allowed_prior_5": [0.52],
            "fg3_pct_prior_10": [0.39],
            "opp_fg3_pct_allowed_prior_10": [0.35],
            "fta_per_min_prior_20": [0.21],
            "opp_fta_rate_allowed_prior_20": [0.17],
            "three_pa_share_prior_5": [0.44],
            "opp_three_pa_share_allowed_prior_5": [0.36],
            "team_off_rtg_szn": [118.4],
            "opp_def_rtg_szn": [112.1],
        }
    )

    out, derived_cols = _add_efficiency_sidecar_interaction_features(
        df,
        sidecar_feature_columns=[
            "fg2_pct_prior_5",
            "opp_fg2_pct_allowed_prior_5",
            "fg3_pct_prior_10",
            "opp_fg3_pct_allowed_prior_10",
            "fta_per_min_prior_20",
            "opp_fta_rate_allowed_prior_20",
            "three_pa_share_prior_5",
            "opp_three_pa_share_allowed_prior_5",
        ],
    )

    assert "fg2_pct_matchup_delta_5" in derived_cols
    assert "fg3_pct_matchup_delta_10" in derived_cols
    assert "fta_rate_matchup_delta_20" in derived_cols
    assert "three_pa_share_matchup_delta_5" in derived_cols
    assert "team_off_vs_opp_def_delta" in derived_cols
    assert out.loc[0, "fg2_pct_matchup_delta_5"] == pytest.approx(0.06)
    assert out.loc[0, "fg3_pct_matchup_delta_10"] == pytest.approx(0.04)
    assert out.loc[0, "fta_rate_matchup_delta_20"] == pytest.approx(0.04)
    assert out.loc[0, "three_pa_share_matchup_delta_5"] == pytest.approx(0.08)
    assert out.loc[0, "team_off_vs_opp_def_delta"] == pytest.approx(6.3)
