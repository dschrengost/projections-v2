"""Helpers for computing lineup-level distribution stats from sim_v2 worlds.

The key idea: lineup percentiles must be computed on the *sum per world*, not
by summing player-level percentiles.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

logger = logging.getLogger(__name__)

FPTS_COLUMN_CANDIDATES: tuple[str, ...] = ("dk_fpts_world", "dk_fpts_sim", "dk_fpts")
PROFILE_COLUMN_CANDIDATES: tuple[str, ...] = ("sim_profile", "profile")


@dataclass(frozen=True)
class LineupDistributionStats:
    mean: Optional[float]
    p10: Optional[float]
    p50: Optional[float]
    p75: Optional[float]
    p90: Optional[float]
    stdev: Optional[float]
    ceiling_upside: Optional[float]

    def to_dict(self) -> dict[str, Optional[float]]:
        return {
            "mean": self.mean,
            "p10": self.p10,
            "p50": self.p50,
            "p75": self.p75,
            "p90": self.p90,
            "stdev": self.stdev,
            "ceiling_upside": self.ceiling_upside,
        }


def _world_files(worlds_dir: Path) -> list[Path]:
    return sorted(worlds_dir.glob("world=*.parquet"))


def _pick_column(candidates: Sequence[str], available: Iterable[str]) -> Optional[str]:
    available_set = set(available)
    return next((c for c in candidates if c in available_set), None)


def load_world_fpts_matrix(
    *,
    worlds_dir: Path,
    player_ids: Sequence[int],
    sim_profile: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load sim world FPTS into a dense matrix.

    Args:
        worlds_dir: Directory containing `world=XXXX.parquet` files.
        player_ids: Players to include (column order matches this sequence).
        sim_profile: Optional profile filter if the column exists.

    Returns:
        (world_ids, player_ids, fpts_matrix) where fpts_matrix has shape (W, P).
        Missing (world_id, player_id) cells are filled with 0.0.
    """
    unique_player_ids: list[int] = []
    seen: set[int] = set()
    for pid in player_ids:
        pid_int = int(pid)
        if pid_int in seen:
            continue
        unique_player_ids.append(pid_int)
        seen.add(pid_int)

    if not unique_player_ids:
        raise ValueError("player_ids must be non-empty")

    paths = _world_files(worlds_dir)
    if not paths:
        raise FileNotFoundError(f"No world parquet files found under {worlds_dir}")

    dataset = ds.dataset([str(p) for p in paths], format="parquet")
    schema_names = list(dataset.schema.names)

    fpts_col = _pick_column(FPTS_COLUMN_CANDIDATES, schema_names)
    if fpts_col is None:
        raise ValueError(
            f"Worlds dataset missing FPTS column; expected one of {FPTS_COLUMN_CANDIDATES}, found {schema_names}"
        )

    profile_col = _pick_column(PROFILE_COLUMN_CANDIDATES, schema_names)

    filter_expr = ds.field("player_id").isin(unique_player_ids)
    if sim_profile and profile_col:
        filter_expr = filter_expr & (ds.field(profile_col) == sim_profile)

    cols = ["world_id", "player_id", fpts_col]
    table = dataset.to_table(columns=cols, filter=filter_expr)
    df = table.to_pandas()

    if df.empty:
        raise ValueError(
            f"No matching worlds rows found (players={len(unique_player_ids)}, profile={sim_profile}) in {worlds_dir}"
        )

    df["world_id"] = pd.to_numeric(df["world_id"], errors="coerce").astype("Int64")
    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64")
    df[fpts_col] = pd.to_numeric(df[fpts_col], errors="coerce").fillna(0.0).astype(float)
    df = df.dropna(subset=["world_id", "player_id"])

    world_ids = np.sort(df["world_id"].astype(int).unique())
    if world_ids.size == 0:
        raise ValueError(f"No valid world_id values found in {worlds_dir}")

    world_index = pd.Index(world_ids)
    player_index = pd.Index(unique_player_ids)

    world_pos = world_index.get_indexer(df["world_id"].astype(int))
    player_pos = player_index.get_indexer(df["player_id"].astype(int))
    valid = (world_pos >= 0) & (player_pos >= 0)
    if not np.any(valid):
        raise ValueError(f"No rows mapped into matrix for {worlds_dir}")

    values = df.loc[valid, fpts_col].to_numpy(dtype=np.float32, copy=False)
    out = np.zeros((len(world_ids), len(unique_player_ids)), dtype=np.float32)
    out[world_pos[valid], player_pos[valid]] = values
    return world_ids.astype(int), np.asarray(unique_player_ids, dtype=int), out


def compute_lineup_distribution_stats(
    *,
    lineups: Sequence[Sequence[str]],
    world_player_ids: Sequence[int],
    fpts_by_world: np.ndarray,
    batch_size: int = 256,
) -> list[LineupDistributionStats]:
    """Compute lineup totals per world then summarize percentiles.

    Args:
        lineups: List of lineups, each a sequence of player_id strings.
        world_player_ids: Player IDs matching the columns of `fpts_by_world`.
        fpts_by_world: Matrix of shape (W, P) where each row is a world.
        batch_size: Number of lineups to process per batch.
    """
    if not lineups:
        return []

    if fpts_by_world.ndim != 2:
        raise ValueError(f"Expected fpts_by_world to be 2D, got shape={fpts_by_world.shape}")

    n_worlds, n_players = fpts_by_world.shape
    if n_players != len(world_player_ids):
        raise ValueError(
            f"world_player_ids length {len(world_player_ids)} != fpts_by_world columns {n_players}"
        )

    pid_to_col = {int(pid): idx for idx, pid in enumerate(world_player_ids)}

    # Pre-resolve lineup -> column indices (and track missing players).
    lineup_cols: list[list[int]] = []
    lineup_has_missing: list[bool] = []
    for lu in lineups:
        cols: list[int] = []
        missing = False
        for pid_raw in lu:
            try:
                pid = int(str(pid_raw))
            except Exception:
                missing = True
                continue
            col = pid_to_col.get(pid)
            if col is None:
                missing = True
                continue
            cols.append(col)
        lineup_cols.append(cols)
        lineup_has_missing.append(missing)

    results: list[LineupDistributionStats] = []
    for start in range(0, len(lineups), max(1, int(batch_size))):
        end = min(len(lineups), start + max(1, int(batch_size)))
        for i in range(start, end):
            if lineup_has_missing[i]:
                results.append(
                    LineupDistributionStats(
                        mean=None,
                        p10=None,
                        p50=None,
                        p75=None,
                        p90=None,
                        stdev=None,
                        ceiling_upside=None,
                    )
                )
                continue

            cols = lineup_cols[i]
            if not cols:
                results.append(
                    LineupDistributionStats(
                        mean=None,
                        p10=None,
                        p50=None,
                        p75=None,
                        p90=None,
                        stdev=None,
                        ceiling_upside=None,
                    )
                )
                continue

            totals = fpts_by_world[:, cols].sum(axis=1)
            if n_worlds == 0:
                results.append(
                    LineupDistributionStats(
                        mean=None,
                        p10=None,
                        p50=None,
                        p75=None,
                        p90=None,
                        stdev=None,
                        ceiling_upside=None,
                    )
                )
                continue

            mean_val = float(np.mean(totals))
            stdev_val = float(np.std(totals, ddof=0))
            p10_val, p50_val, p75_val, p90_val = np.percentile(
                totals, [10, 50, 75, 90]
            ).astype(float)
            results.append(
                LineupDistributionStats(
                    mean=mean_val,
                    p10=float(p10_val),
                    p50=float(p50_val),
                    p75=float(p75_val),
                    p90=float(p90_val),
                    stdev=stdev_val,
                    ceiling_upside=float(p90_val - mean_val),
                )
            )

    return results

