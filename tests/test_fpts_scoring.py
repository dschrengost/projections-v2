from __future__ import annotations

import pandas as pd

from projections.fpts_v2.scoring import compute_dk_fpts, compute_fd_fpts


def test_compute_dk_fpts_includes_bonus_logic() -> None:
    df = pd.DataFrame(
        [
            {
                "pts": 20,
                "fgm": 8,
                "fga": 16,
                "fg3m": 2,
                "fg3a": 6,
                "ftm": 2,
                "fta": 3,
                "reb": 10,
                "oreb": 3,
                "dreb": 7,
                "ast": 10,
                "stl": 1,
                "blk": 0,
                "tov": 4,
                "pf": 2,
                "plus_minus": 5,
            }
        ]
    )
    # Base:
    # 20 (pts) + 12.5 (reb) + 15 (ast) + 2 (stl) + 0 (blk) - 2 (tov) + 1 (fg3 bonus)
    # = 48.5, plus double-double bonus 2.0 -> 50.5
    out = compute_dk_fpts(df)
    assert out.iloc[0] == 50.5


def test_compute_fd_fpts_uses_fd_weights_without_bonus() -> None:
    df = pd.DataFrame(
        [
            {
                "pts": 20,
                "fgm": 8,
                "fga": 16,
                "fg3m": 2,
                "fg3a": 6,
                "ftm": 2,
                "fta": 3,
                "reb": 10,
                "oreb": 3,
                "dreb": 7,
                "ast": 10,
                "stl": 1,
                "blk": 0,
                "tov": 4,
                "pf": 2,
                "plus_minus": 5,
            }
        ]
    )
    # 20 + 12 + 15 + 3 + 0 - 4 = 46
    out = compute_fd_fpts(df)
    assert out.iloc[0] == 46.0
