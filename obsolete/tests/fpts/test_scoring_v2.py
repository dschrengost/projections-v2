from __future__ import annotations

import pandas as pd
import pytest

from projections.fpts_v2.scoring import compute_dk_fpts


def _base_row(**overrides: float) -> dict:
    row = {
        "pts": 0.0,
        "fgm": 0.0,
        "fga": 0.0,
        "fg3m": 0.0,
        "fg3a": 0.0,
        "ftm": 0.0,
        "fta": 0.0,
        "reb": 0.0,
        "oreb": 0.0,
        "dreb": 0.0,
        "ast": 0.0,
        "stl": 0.0,
        "blk": 0.0,
        "tov": 0.0,
        "pf": 0.0,
        "plus_minus": 0.0,
    }
    row.update(overrides)
    return row


def test_compute_dk_fpts_double_and_triple() -> None:
    df = pd.DataFrame(
        [
            _base_row(pts=20, reb=10, ast=5, stl=1, blk=1, tov=3),  # double-double
            _base_row(pts=10, reb=11, ast=10, stl=0, blk=0, tov=2),  # triple-double
            _base_row(pts=15, reb=4, ast=3, stl=2, blk=1, tov=1),  # no bonus
        ]
    )
    scores = compute_dk_fpts(df)
    # base calculations
    expected0 = 20 + 1.25 * 10 + 1.5 * 5 + 2 * 1 + 2 * 1 - 0.5 * 3 + 1.5
    expected1 = 10 + 1.25 * 11 + 1.5 * 10 - 0.5 * 2 + 3.0
    expected2 = 15 + 1.25 * 4 + 1.5 * 3 + 2 * 2 + 2 * 1 - 0.5 * 1
    assert scores.iloc[0] == pytest.approx(expected0)
    assert scores.iloc[1] == pytest.approx(expected1)
    assert scores.iloc[2] == pytest.approx(expected2)


def test_compute_dk_fpts_missing_column_error() -> None:
    df = pd.DataFrame([{"pts": 10, "reb": 10}])
    with pytest.raises(KeyError):
        compute_dk_fpts(df)
