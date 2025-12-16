"""Depth and archetype feature helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from projections.minutes_v1.constants import ARCHETYPE_MAP
from projections.minutes_v1.snapshots import ensure_as_of_column
from projections.minutes_v1.starter_flags import normalize_starter_signals

ARCHETYPE_BUCKETS: tuple[str, ...] = ("G", "W", "B")


def attach_depth_features(
    base_df: pd.DataFrame, roster_nightly: pd.DataFrame
) -> pd.DataFrame:
    """Attach roster depth counts and archetype overlap features."""

    if roster_nightly is None or roster_nightly.empty:
        merged = base_df.copy()
        for bucket in ARCHETYPE_BUCKETS:
            col = f"available_{bucket}"
            if col not in merged.columns:
                merged[col] = 0
            merged[col] = merged[col].fillna(0).astype(int)
        for col, default in (
            ("lineup_role", pd.NA),
            ("lineup_status", pd.NA),
            ("lineup_roster_status", pd.NA),
            ("lineup_timestamp", pd.NaT),
            ("is_projected_starter", pd.NA),
            ("is_confirmed_starter", pd.NA),
            ("roster_as_of_ts", pd.NaT),
            ("same_archetype_overlap", 0),
            ("depth_same_pos_active", 0),
        ):
            if col not in merged.columns:
                merged[col] = default
        return normalize_starter_signals(merged)

    roster = ensure_as_of_column(roster_nightly.copy())
    required = {"team_id", "game_date", "player_id", "active_flag", "listed_pos"}
    if not required.issubset(roster.columns):
        return attach_depth_features(base_df, pd.DataFrame())
    roster["game_date"] = pd.to_datetime(roster["game_date"]).dt.normalize()
    roster["active_flag"] = roster["active_flag"].astype(bool)
    roster["listed_pos"] = roster["listed_pos"].fillna("W").str.upper()
    roster["archetype"] = roster["listed_pos"].map(ARCHETYPE_MAP).fillna("W")

    active = roster[roster["active_flag"]]
    archetype_counts = (
        active.groupby(["team_id", "game_date", "archetype"])["player_id"]
        .nunique()
        .unstack(fill_value=0)
    )
    archetype_counts = archetype_counts.rename(
        columns={bucket: f"available_{bucket}" for bucket in archetype_counts.columns}
    ).reset_index()

    merged = base_df.merge(archetype_counts, on=["team_id", "game_date"], how="left")
    for bucket in ARCHETYPE_BUCKETS:
        col = f"available_{bucket}"
        if col not in merged.columns:
            merged[col] = 0
        merged[col] = merged[col].fillna(0).astype(int)

    extra_lineup_cols = [
        col
        for col in (
            "lineup_role",
            "lineup_status",
            "lineup_roster_status",
            "lineup_timestamp",
            "is_projected_starter",
            "is_confirmed_starter",
        )
        if col in roster.columns
    ]
    base_cols = ["team_id", "game_date", "player_id", "archetype", "as_of_ts"] + extra_lineup_cols
    player_positions = (
        roster[base_cols]
        .sort_values("as_of_ts")
        .drop_duplicates(subset=["team_id", "game_date", "player_id"], keep="last")
        .rename(columns={"as_of_ts": "roster_as_of_ts"})
    )
    merged = merged.merge(
        player_positions,
        on=["team_id", "game_date", "player_id"],
        how="left",
    )
    # Fallback: always merge lineup metadata directly on (game_id, team_id, player_id) if any values are missing.
    if {"game_id", "team_id", "player_id"}.issubset(roster.columns):
        alt_cols = [
            col
            for col in (
                "game_id",
                "team_id",
                "player_id",
                "lineup_status",
                "is_projected_starter",
                "is_confirmed_starter",
                "as_of_ts",
            )
            if col in roster.columns
        ]
        alt_positions = roster[alt_cols].drop_duplicates(subset=["game_id", "team_id", "player_id"])
        alt_positions = alt_positions.rename(columns={"as_of_ts": "roster_as_of_ts_alt"})
        merged = merged.merge(
            alt_positions,
            on=["game_id", "team_id", "player_id"],
            how="left",
            suffixes=("", "_alt"),
        )
        # Only apply fallback if we are missing lineup metadata after the merge.
        missing_meta = merged[["lineup_status", "is_projected_starter", "is_confirmed_starter"]].isna().any(axis=None)
        if missing_meta:
            for col in ("lineup_status", "is_projected_starter", "is_confirmed_starter"):
                alt_col = f"{col}_alt"
                if alt_col in merged.columns:
                    merged[col] = merged[col].combine_first(merged[alt_col])
            if "roster_as_of_ts_alt" in merged.columns:
                merged["roster_as_of_ts"] = merged["roster_as_of_ts"].combine_first(merged["roster_as_of_ts_alt"])
        # Clean up alt columns
        for col in ("lineup_status", "is_projected_starter", "is_confirmed_starter", "roster_as_of_ts"):
            alt_col = f"{col}_alt"
            if alt_col in merged.columns:
                merged.drop(columns=[alt_col], inplace=True)
    if "tip_ts" in merged:
        tip_ts = pd.to_datetime(merged["tip_ts"], utc=True, errors="coerce")
        roster_ts = pd.to_datetime(merged["roster_as_of_ts"], utc=True, errors="coerce")
        late_mask = roster_ts.notna() & tip_ts.notna() & (roster_ts > tip_ts)
        if late_mask.any():
            merged.loc[late_mask, "roster_as_of_ts"] = tip_ts[late_mask]

    if "lineup_timestamp" in merged.columns:
        merged["lineup_timestamp"] = pd.to_datetime(
            merged["lineup_timestamp"], utc=True, errors="coerce"
        )
    else:
        merged["lineup_timestamp"] = pd.NaT
    for column in ("lineup_role", "lineup_status", "lineup_roster_status"):
        if column not in merged.columns:
            merged[column] = pd.NA
    merged = normalize_starter_signals(merged)

    archetype = merged["archetype"].fillna("W")
    archetype_counts_arr = np.select(
        [
            archetype == "G",
            archetype == "W",
            archetype == "B",
        ],
        [
            merged["available_G"],
            merged["available_W"],
            merged["available_B"],
        ],
        default=0,
    )
    merged["same_archetype_overlap"] = np.where(archetype_counts_arr > 1, 1, 0)
    depth_counts = np.maximum(archetype_counts_arr - 1, 0)
    merged["depth_same_pos_active"] = depth_counts.astype(int)
    return merged
