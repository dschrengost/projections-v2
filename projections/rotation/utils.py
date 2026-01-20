"""Pure utility functions for rotation model (no torch dependency).

These helpers are extracted from set_model.py to allow importing by feature
builders that don't need the full PyTorch-based model, avoiding scipy/torch
import order conflicts that can cause segfaults.
"""

from __future__ import annotations

import pandas as pd


def zfill_game_id_series(series: pd.Series) -> pd.Series:
    """Normalize NBA game ids to a zero-filled 10-character string."""
    coerced = pd.to_numeric(series, errors="coerce").astype("Int64")
    return coerced.astype("string").str.zfill(10)


def _coerce_int_series(series: pd.Series, *, name: str) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce").astype("Int64")
    if out.isna().any():
        raise ValueError(f"{name} contains missing/invalid values after coercion")
    return out


# Constants shared with set_model
GAME_ID_NORM_COL = "game_id_norm"
KEY_COLS = ("game_id", "team_id", "player_id")
OPPONENT_TEAM_ID_COL = "opponent_team_id"


def normalize_key_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with normalized key columns used for grouping."""
    missing = [c for c in KEY_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required key columns: {missing}")
    out = df.copy()
    out[GAME_ID_NORM_COL] = zfill_game_id_series(out["game_id"])
    if out[GAME_ID_NORM_COL].isna().any():
        raise ValueError("game_id contains missing/invalid values after normalization")
    out["team_id"] = _coerce_int_series(out["team_id"], name="team_id")
    out["player_id"] = _coerce_int_series(out["player_id"], name="player_id")
    return out


__all__ = [
    "zfill_game_id_series",
    "_coerce_int_series",
    "GAME_ID_NORM_COL",
    "KEY_COLS",
    "OPPONENT_TEAM_ID_COL",
    "normalize_key_columns",
]
