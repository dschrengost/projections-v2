from __future__ import annotations

import pandas as pd

from scripts.rotation.train_game_transformer_v2 import _infer_feature_columns


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
