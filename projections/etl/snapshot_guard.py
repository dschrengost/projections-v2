"""Helpers to prevent destructive snapshot overwrites."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class SnapshotKeyStats:
    """Key-level cardinality summary for a snapshot dataframe."""

    rows: int
    unique_keys: int


def compute_key_stats(df: pd.DataFrame, *, key_cols: Iterable[str]) -> SnapshotKeyStats:
    """Return row count and unique key count for the provided key columns."""
    keys = [col for col in key_cols if col in df.columns]
    if df.empty or not keys:
        return SnapshotKeyStats(rows=int(len(df)), unique_keys=0)

    work = df.loc[:, keys].copy()
    work = work.dropna(subset=keys)
    if work.empty:
        return SnapshotKeyStats(rows=int(len(df)), unique_keys=0)

    unique = int(len(work.drop_duplicates(subset=keys)))
    return SnapshotKeyStats(rows=int(len(df)), unique_keys=unique)


def enforce_non_regression(
    *,
    dataset_name: str,
    existing: pd.DataFrame,
    candidate: pd.DataFrame,
    key_cols: Iterable[str],
    allow_regression: bool = False,
) -> None:
    """Raise when a candidate snapshot regresses key coverage vs existing."""
    if allow_regression:
        return

    existing_stats = compute_key_stats(existing, key_cols=key_cols)
    if existing_stats.unique_keys <= 0:
        return

    candidate_stats = compute_key_stats(candidate, key_cols=key_cols)
    if candidate_stats.unique_keys < existing_stats.unique_keys:
        keys_str = ",".join(key_cols)
        raise RuntimeError(
            f"[{dataset_name}] refusing snapshot overwrite: key coverage regressed "
            f"({keys_str}: {existing_stats.unique_keys} -> {candidate_stats.unique_keys}). "
            "Pass --allow-snapshot-regression only for intentional recovery operations."
        )
