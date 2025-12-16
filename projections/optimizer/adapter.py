"""Thin adapter that feeds DataFrame player pools into the optimizer."""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from .cpsat_solver import solve_cpsat_iterative_counts
from .optimizer_types import Constraints

try:
    from . import lineup_schema
except Exception:  # pragma: no cover - optional at import time
    lineup_schema = None  # type: ignore


def _normalize_positions(val: object) -> List[str]:
    if isinstance(val, str):
        parts = [p.strip() for p in val.replace("|", "/").replace(",", "/").split("/") if p.strip()]
        return parts
    if isinstance(val, (list, tuple, set)):
        return [str(p).strip() for p in val if str(p).strip()]
    return []


def _pick_projection_column(df: pd.DataFrame) -> str:
    for col in ("dk_fpts_mean", "proj", "fpts_mean", "fpts", "projection"):
        if col in df.columns:
            return col
    raise ValueError(
        "Player pool missing a projection column (looked for dk_fpts_mean/proj/fpts_mean/fpts/projection)."
    )


def build_lineups_from_player_pool(
    player_pool_df: pd.DataFrame,
    num_lineups: int,
    site: str = "dk",
    game_date: str | None = None,
) -> pd.DataFrame:
    """
    Given a player pool DataFrame, call the optimizer to generate `num_lineups`
    lineups and return them as a normalized DataFrame.

    Required player_pool_df columns (best-effort):
      - player_id
      - salary
      - positions (list or slash/comma-separated string)
      - projection column (dk_fpts_mean/proj/fpts_mean/...)
    """

    required = {"player_id", "salary"}
    missing = [c for c in required if c not in player_pool_df.columns]
    if missing:
        raise ValueError(f"player_pool_df missing required columns: {missing}")

    proj_col = _pick_projection_column(player_pool_df)
    players_payload = []
    for _, row in player_pool_df.iterrows():
        positions = row.get("positions")
        pos_list = _normalize_positions(positions)
        name_val = row.get("name") or row.get("player_name") or row.get("Name") or row["player_id"]
        players_payload.append(
            {
                "player_id": str(row["player_id"]),
                "name": name_val,
                "team": row.get("team", "UNK"),
                "positions": pos_list,
                "salary": int(row["salary"]),
                "proj": float(row[proj_col]),
                "own_proj": row.get("own_proj"),
                "stddev": row.get("stddev"),
                "minutes": row.get("minutes_p50", row.get("minutes")),
                "dk_id": row.get("dk_id"),
            }
        )

    constraints = Constraints(N_lineups=int(num_lineups))
    constraints.validate(site, stddev_available="stddev" in player_pool_df.columns)

    lineups, diagnostics = solve_cpsat_iterative_counts(
        players_payload, constraints, seed=0, site=site
    )
    # TODO: surface diagnostics to caller as needed

    rows = []
    for lineup in lineups:
        row = {
            "lineup_id": lineup.lineup_id,
            "site": site,
            "game_date": game_date,
            "contest_type": "classic",
            "mean_fpts": float(lineup.total_proj),
            "total_salary": int(lineup.total_salary),
        }
        for idx, player in enumerate(lineup.players, start=1):
            try:
                row[f"p{idx}_id"] = int(str(player.player_id))
            except Exception:
                row[f"p{idx}_id"] = player.player_id
            row[f"p{idx}_name"] = player.name
            row[f"p{idx}_pos"] = player.pos
        rows.append(row)

    df = pd.DataFrame(rows)
    if lineup_schema is not None:
        df = lineup_schema.normalize_lineups_df(df)
    return df
