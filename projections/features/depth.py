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
            ("active_flag", pd.NA),
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
    for col in ("listed_pos", "archetype", "lineup_role", "lineup_status", "lineup_roster_status"):
        if col in roster.columns:
            roster[col] = roster[col].astype(object)

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
    base_cols = ["team_id", "game_date", "player_id", "archetype", "active_flag", "as_of_ts"] + extra_lineup_cols
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
    # Fallback: overlay lineup metadata by (game_id, team_id, player_id) without
    # DataFrame.merge, which has intermittently crashed in native pandas internals.
    if {"game_id", "team_id", "player_id"}.issubset(roster.columns) and {"game_id", "team_id", "player_id"}.issubset(merged.columns):
        lookup_cols = [
            col
            for col in (
                "active_flag",
                "lineup_status",
                "is_projected_starter",
                "is_confirmed_starter",
                "as_of_ts",
            )
            if col in roster.columns
        ]
        if lookup_cols:
            key_cols = ["game_id", "team_id", "player_id"]
            alt_positions = roster[key_cols + lookup_cols].copy()
            if "as_of_ts" in alt_positions.columns:
                alt_positions = alt_positions.sort_values("as_of_ts")
            alt_positions = alt_positions.drop_duplicates(subset=key_cols, keep="last")
            if not alt_positions.empty:
                alt_indexed = alt_positions.set_index(key_cols)
                merged_indexed = merged.set_index(key_cols, drop=False)
                for source_col, target_col in (
                    ("active_flag", "active_flag"),
                    ("lineup_status", "lineup_status"),
                    ("is_projected_starter", "is_projected_starter"),
                    ("is_confirmed_starter", "is_confirmed_starter"),
                    ("as_of_ts", "roster_as_of_ts"),
                ):
                    if source_col not in alt_indexed.columns:
                        continue
                    aligned = alt_indexed[source_col].reindex(merged_indexed.index)
                    if target_col in merged_indexed.columns:
                        fill_mask = merged_indexed[target_col].isna()
                        if fill_mask.any():
                            merged_indexed.loc[fill_mask, target_col] = aligned.loc[fill_mask]
                    else:
                        merged_indexed[target_col] = aligned
                merged = merged_indexed.reset_index(drop=True)
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
