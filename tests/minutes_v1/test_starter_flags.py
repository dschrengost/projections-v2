from __future__ import annotations

import pandas as pd

from projections.minutes_v1.starter_flags import (
    derive_starter_flag_label,
    normalize_starter_signals,
)


def test_normalize_starter_signals_uses_lineup_role():
    df = pd.DataFrame(
        {
            "lineup_role": ["confirmed_starter", "bench", "projected_starter"],
            "lineup_status": ["Confirmed", "Expected", "Expected"],
            "is_confirmed_starter": [False, False, False],
            "is_projected_starter": [False, False, False],
        }
    )
    normalized = normalize_starter_signals(df.copy())
    assert normalized["is_confirmed_starter"].tolist() == [True, False, False]
    assert normalized["is_projected_starter"].tolist() == [True, False, True]


def test_derive_starter_flag_label_prefers_starter_flag_then_fallbacks():
    df = pd.DataFrame(
        {
            "game_id": [1] * 6 + [2] * 5,
            "team_id": [10] * 6 + [11] * 5,
            "player_id": list(range(11)),
            "starter_flag": [1, 1, 1, 1, 1, 0] + [0] * 5,
            "lineup_role": ["bench"] * 6 + ["confirmed_starter"] * 5,
        }
    )
    normalized = normalize_starter_signals(df.copy())
    result = derive_starter_flag_label(normalized)
    first_team = normalized["game_id"].eq(1)
    second_team = normalized["game_id"].eq(2)
    assert result.values.loc[first_team].sum() == 5
    assert result.values.loc[second_team].sum() == 5
    assert result.source_counts.get("starter_flag") == 5
    assert result.source_counts.get("is_confirmed_starter") == 5


def test_derive_starter_flag_label_records_overflow_when_clipped():
    df = pd.DataFrame(
        {
            "game_id": [3] * 7,
            "team_id": [12] * 7,
            "player_id": list(range(7)),
            "lineup_role": ["confirmed_starter"] * 7,
        }
    )
    normalized = normalize_starter_signals(df.copy())
    result = derive_starter_flag_label(normalized)
    assert result.values.sum() == 5
    assert result.overflow == [(3, 12, 7)]
