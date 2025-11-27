"""Helpers for normalizing starter signals and deriving starter labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pandas as pd


@dataclass(frozen=True)
class StarterFlagResult:
    """Container for derived starter flags and diagnostics."""

    values: pd.Series
    overflow: list[tuple[int | None, int | None, int]]
    source_counts: dict[str, int]


def _coerce_bool_series(series: pd.Series | Iterable | None, index: pd.Index) -> pd.Series:
    """Return a boolean series aligned to ``index`` with NA filled as False."""

    if series is None:
        return pd.Series(False, index=index, dtype=bool)
    if isinstance(series, pd.Series):
        aligned = series.reindex(index)
    else:
        aligned = pd.Series(series, index=index)
    return aligned.astype("boolean", copy=False).fillna(False)


def normalize_starter_signals(
    df: pd.DataFrame,
    *,
    lineup_role_col: str = "lineup_role",
    lineup_status_col: str = "lineup_status",
) -> pd.DataFrame:
    """Ensure projected/confirmed starter columns incorporate lineup metadata.

    The function updates ``is_projected_starter`` and ``is_confirmed_starter`` in-place
    based on ``lineup_role`` (primary) and ``lineup_status`` (``Expected`` only).
    """

    if df.empty:
        return df
    index = df.index
    role_norm = pd.Series("", index=index, dtype="string")
    if lineup_role_col in df.columns:
        role_norm = (
            df[lineup_role_col]
            .astype("string", copy=False)
            .str.strip()
            .str.lower()
            .fillna("")
        )
    status_norm = pd.Series("", index=index, dtype="string")
    if lineup_status_col in df.columns:
        status_norm = (
            df[lineup_status_col]
            .astype("string", copy=False)
            .str.strip()
            .str.lower()
            .fillna("")
        )

    confirmed_from_role = role_norm.eq("confirmed_starter")
    projected_from_role = role_norm.isin({"confirmed_starter", "projected_starter"})
    projected_from_status = status_norm.eq("expected") & role_norm.eq("")

    confirmed_series = _coerce_bool_series(df.get("is_confirmed_starter"), index)
    projected_series = _coerce_bool_series(df.get("is_projected_starter"), index)

    df["is_confirmed_starter"] = (confirmed_series | confirmed_from_role).astype(bool)
    df["is_projected_starter"] = (
        projected_series | projected_from_role | projected_from_status
    ).astype(bool)
    return df


def derive_starter_flag_label(
    df: pd.DataFrame,
    *,
    prefer_sources: Sequence[str] = ("starter_flag", "is_confirmed_starter", "is_projected_starter"),
    group_cols: Sequence[str] = ("game_id", "team_id"),
    max_starters: int = 5,
) -> StarterFlagResult:
    """Derive a ``starter_flag_label`` series with diagnostics."""

    if df.empty:
        empty = pd.Series(pd.array([], dtype="Int64"), index=df.index)
        return StarterFlagResult(values=empty, overflow=[], source_counts={})

    result = pd.Series(False, index=df.index, dtype=bool)
    assignment = pd.Series("none", index=df.index, dtype="string")
    candidates: dict[str, pd.Series] = {}
    for source in prefer_sources:
        if source in df.columns:
            candidates[source] = _coerce_bool_series(df[source], df.index)
        else:
            candidates[source] = pd.Series(False, index=df.index, dtype=bool)

    if group_cols and all(col in df.columns for col in group_cols):
        groups = df.groupby(list(group_cols)).groups
    else:
        groups = {(): df.index}

    overflow_counts: dict[tuple, int] = {}
    for key, idx in groups.items():
        remaining_slots = max_starters
        group_index = pd.Index(idx)
        for source in prefer_sources:
            candidate = candidates.get(source)
            if candidate is None or not remaining_slots:
                continue
            mask = candidate.loc[group_index]
            if not mask.any():
                continue
            available_mask = (~result.loc[group_index]) & mask
            if not available_mask.any():
                continue
            desired_idx = available_mask[available_mask].index
            requested = len(desired_idx)
            if requested > remaining_slots:
                overflow_counts[key] = max(overflow_counts.get(key, 0), requested)
                desired_idx = desired_idx[:remaining_slots]
            if not len(desired_idx):
                continue
            result.loc[desired_idx] = True
            assignment.loc[desired_idx] = source
            remaining_slots -= len(desired_idx)
            if remaining_slots <= 0:
                break

    value_series = result.astype("Int64")
    overflow: list[tuple[int | None, int | None, int]] = []
    for key, count in overflow_counts.items():
        normalized_key: tuple[int | None, ...]
        if not isinstance(key, tuple):
            key = (key,)
        normalized: list[int | None] = []
        for value in key:
            if pd.isna(value):
                normalized.append(None)
            else:
                try:
                    normalized.append(int(value))
                except Exception:  # pragma: no cover - fallback for non-int keys
                    normalized.append(value)
        while len(normalized) < len(group_cols):
            normalized.append(None)
        normalized_key = tuple(normalized)
        overflow.append(
            (
                normalized_key[0] if normalized_key else None,
                normalized_key[1] if len(normalized_key) > 1 else None,
                int(count),
            )
        )

    source_counts = assignment.value_counts().to_dict()
    source_counts.pop("none", None)
    return StarterFlagResult(
        values=value_series,
        overflow=overflow,
        source_counts=source_counts,
    )


__all__ = ["StarterFlagResult", "derive_starter_flag_label", "normalize_starter_signals"]
