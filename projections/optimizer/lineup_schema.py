"""Lightweight helpers to normalize optimizer lineup outputs.

Schema (DraftKings classic for now):
  - lineup_id, site, game_date, contest_type
  - p1_id..p8_id: numeric player_id (int)
  - p1_name..p8_name: player display names (string)
  - mean_fpts: total projected fantasy points across slots
  - total_salary: summed salary across slots
"""

from __future__ import annotations

from typing import Iterable, List

import pandas as pd

LINEUP_ID_COLS = [f"p{i}_id" for i in range(1, 9)]
LINEUP_NAME_COLS = [f"p{i}_name" for i in range(1, 9)]


def normalize_lineups_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure a consistent column order/shape for downstream consumers.

    Expected columns:
      - lineup_id, site, game_date, contest_type
      - p1_id..p8_id (numeric player_id)
      - p1_name..p8_name (string player names)
      - mean_fpts, total_salary
    """

    base_cols: List[str] = [
        "lineup_id",
        "site",
        "game_date",
        "contest_type",
    ] + LINEUP_ID_COLS + LINEUP_NAME_COLS + ["mean_fpts", "total_salary"]

    for col in base_cols:
        if col not in df.columns:
            df[col] = None

    return df[base_cols]
