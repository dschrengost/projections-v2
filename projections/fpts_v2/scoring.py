from __future__ import annotations

import pandas as pd

EXPECTED_COLUMNS = {
    "pts",
    "fgm",
    "fga",
    "fg3m",
    "fg3a",
    "ftm",
    "fta",
    "reb",
    "oreb",
    "dreb",
    "ast",
    "stl",
    "blk",
    "tov",
    "pf",
    "plus_minus",
}


def compute_dk_fpts(df: pd.DataFrame) -> pd.Series:
    """
    Compute DraftKings fantasy points from boxscore-like columns.

    Expected columns (per player-game):
      - pts, fgm, fga, fg3m, fg3a, ftm, fta,
      - reb, oreb, dreb, ast, stl, blk, tov, pf, plus_minus

    Scoring:
      - 1 pt per real point
      - 1.25 per rebound
      - 1.5 per assist
      - 2 per steal
      - 2 per block
      - -0.5 per turnover
      - 1.5 double-double bonus (>=10 in exactly two of PTS/REB/AST/STL/BLK)
      - 3.0 triple-double bonus (>=10 in at least three of those five)
    """
    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns for DK scoring: {sorted(missing)}")

    pts = pd.to_numeric(df["pts"], errors="coerce")
    reb = pd.to_numeric(df["reb"], errors="coerce")
    ast = pd.to_numeric(df["ast"], errors="coerce")
    stl = pd.to_numeric(df["stl"], errors="coerce")
    blk = pd.to_numeric(df["blk"], errors="coerce")
    tov = pd.to_numeric(df["tov"], errors="coerce")

    base = pts + 1.25 * reb + 1.5 * ast + 2.0 * stl + 2.0 * blk - 0.5 * tov

    counting = pd.concat([pts, reb, ast, stl, blk], axis=1)
    qualifying_counts = (counting >= 10).sum(axis=1)
    double_double_bonus = (qualifying_counts == 2).astype(float) * 1.5
    triple_double_bonus = (qualifying_counts >= 3).astype(float) * 3.0

    return base + double_double_bonus + triple_double_bonus


__all__ = ["compute_dk_fpts"]
