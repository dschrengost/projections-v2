"""Alloc mask construction for rotation set minutes model.

This module provides helper functions to build the alloc_mask (eligibility mask)
used during live inference to determine which players should receive minutes
allocation. This prevents the "minutes smear" issue where 240 minutes are
distributed across the entire roster instead of the expected rotation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_OUT_LIKE_STATUSES: frozenset[str] = frozenset(
    {"OUT", "O", "DOUBTFUL", "D", "INACTIVE"}
)


def build_alloc_mask_from_features(
    df: pd.DataFrame,
    *,
    min_eligible: int = 9,
    max_eligible: int | None = None,
    prior_play_prob_threshold: float = 0.20,
    baseline_minutes_threshold: float = 4.0,
    baseline_minutes_col: str | None = None,
) -> np.ndarray:
    """Build eligibility mask for rotation minutes allocation.

    Determines which players are eligible to receive minutes based on live features.
    This prevents the "minutes smear" issue where 240 minutes are spread across
    ~15-18 roster players instead of the expected ~9-12 rotation players.

    Players are eligible if ANY of the following are true:
    - is_confirmed_starter == 1
    - is_projected_starter == 1
    - prior_play_prob >= prior_play_prob_threshold (default 0.20)
    - baseline_minutes >= baseline_minutes_threshold (if column exists)

    Excludes:
    - Players with is_out == 1
    - Players with status == "OUT" (case-insensitive)

    Guarantees:
    - At least min_eligible players per team-game (backfills from not-out players
      sorted by prior_play_prob or baseline minutes descending)
    - If max_eligible is set, trims to top max_eligible by priority score
      (but always keeps starters)

    Args:
        df: DataFrame with player rows for a single team-game.
            Expected columns: is_out, prior_play_prob, is_confirmed_starter,
            is_projected_starter. Optional: status, baseline minutes column.
        min_eligible: Minimum number of eligible players to ensure (default 9).
        max_eligible: Optional maximum eligible players (None = no limit).
        prior_play_prob_threshold: Threshold for prior_play_prob eligibility.
        baseline_minutes_threshold: Threshold for baseline minutes eligibility.
        baseline_minutes_col: Name of baseline minutes column if available.

    Returns:
        Boolean array of shape (len(df),) where True = eligible for allocation.
    """
    n = len(df)
    if n == 0:
        return np.zeros(0, dtype=bool)

    # Parse is_out (1 = out, 0 = available)
    # When column is missing, assume all players are available
    if "is_out" in df.columns:
        is_out = pd.to_numeric(df["is_out"], errors="coerce").fillna(0).astype(int)
    else:
        is_out = pd.Series(0, index=df.index)
    not_out = (is_out == 0).to_numpy(dtype=bool)

    # Also check status column for out-like strings.
    if "status" in df.columns:
        status_out = (
            df["status"]
            .fillna("")
            .astype(str)
            .str.upper()
            .str.strip()
            .isin(_OUT_LIKE_STATUSES)
        )
        not_out = not_out & ~status_out.to_numpy(dtype=bool)

    # Parse starter flags
    if "is_confirmed_starter" in df.columns:
        is_confirmed = (
            pd.to_numeric(df["is_confirmed_starter"], errors="coerce")
            .fillna(0)
            .astype(int)
            == 1
        ).to_numpy(dtype=bool)
    else:
        is_confirmed = np.zeros(n, dtype=bool)

    if "is_projected_starter" in df.columns:
        is_projected = (
            pd.to_numeric(df["is_projected_starter"], errors="coerce")
            .fillna(0)
            .astype(int)
            == 1
        ).to_numpy(dtype=bool)
    else:
        is_projected = np.zeros(n, dtype=bool)
    is_starter = is_confirmed | is_projected

    # Parse prior_play_prob
    if "prior_play_prob" in df.columns:
        prior_prob = (
            pd.to_numeric(df["prior_play_prob"], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
    else:
        prior_prob = np.zeros(n, dtype=float)
    high_prob = prior_prob >= prior_play_prob_threshold

    # Parse baseline minutes if available
    high_baseline = np.zeros(n, dtype=bool)
    baseline_vals = np.zeros(n, dtype=float)
    if baseline_minutes_col and baseline_minutes_col in df.columns:
        baseline_vals = (
            pd.to_numeric(df[baseline_minutes_col], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        high_baseline = baseline_vals >= baseline_minutes_threshold

    # Build initial eligibility: not_out AND (starter OR high_prob OR high_baseline)
    eligible = not_out & (is_starter | high_prob | high_baseline)

    # Guarantee minimum eligible count
    n_eligible = int(eligible.sum())
    if n_eligible < min_eligible:
        # Backfill from not_out players sorted by priority (prior_prob, then baseline)
        not_out_indices = np.where(not_out & ~eligible)[0]
        if len(not_out_indices) > 0:
            # Sort by prior_prob descending, then baseline descending
            sort_scores = prior_prob[not_out_indices] + baseline_vals[not_out_indices] / 100.0
            order = np.argsort(-sort_scores)
            need = min_eligible - n_eligible
            to_add = not_out_indices[order][:need]
            eligible[to_add] = True

    # Optional: cap at max_eligible (but always keep starters)
    if max_eligible is not None:
        n_eligible = int(eligible.sum())
        if n_eligible > max_eligible:
            # Compute priority scores for all eligible
            eligible_indices = np.where(eligible)[0]
            # Starters get highest priority (1000 + score), others just score
            scores = prior_prob[eligible_indices] + baseline_vals[eligible_indices] / 100.0
            starter_bonus = is_starter[eligible_indices].astype(float) * 1000.0
            scores = scores + starter_bonus
            order = np.argsort(-scores)
            # Keep top max_eligible
            keep_indices = eligible_indices[order][:max_eligible]
            eligible = np.zeros(n, dtype=bool)
            eligible[keep_indices] = True

    return eligible
