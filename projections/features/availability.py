"""Availability feature helpers (status priors, injury provenance)."""

from __future__ import annotations

import pandas as pd

from projections.minutes_v1.constants import AvailabilityStatus, STATUS_PRIORS
from projections.minutes_v1.snapshots import ensure_as_of_column

_STATUS_ALIASES: dict[str, AvailabilityStatus] = {
    "OUT": AvailabilityStatus.OUT,
    "O": AvailabilityStatus.OUT,
    "QUESTIONABLE": AvailabilityStatus.QUESTIONABLE,
    "Q": AvailabilityStatus.QUESTIONABLE,
    "PROBABLE": AvailabilityStatus.PROBABLE,
    "PROB": AvailabilityStatus.PROBABLE,
    "AVAIL": AvailabilityStatus.AVAILABLE,
    "AVAILABLE": AvailabilityStatus.AVAILABLE,
    "ACTIVE": AvailabilityStatus.AVAILABLE,
    "A": AvailabilityStatus.AVAILABLE,
}

_INJURY_COLUMNS: tuple[str, ...] = (
    "game_id",
    "player_id",
    "status",
    "restriction_flag",
    "ramp_flag",
    "games_since_return",
    "days_since_return",
    "as_of_ts",
)
_OPTIONAL_INJURY_COLUMNS: tuple[str, ...] = ("snapshot_missing",)


def normalize_status(value: str | None) -> AvailabilityStatus:
    """Map raw injury text into the canonical enum."""

    if value is None:
        return AvailabilityStatus.UNKNOWN
    normalized = value.strip().upper()
    return _STATUS_ALIASES.get(normalized, AvailabilityStatus.UNKNOWN)


def prepare_injuries_snapshot(injuries_snapshot: pd.DataFrame) -> pd.DataFrame:
    """Return a normalized copy of the latest injuries snapshot."""

    prepared = ensure_as_of_column(injuries_snapshot.copy())
    prepared["status"] = prepared["status"].apply(normalize_status)
    return prepared


def _select_latest_injury_snapshot(
    base_df: pd.DataFrame,
    injuries_snapshot: pd.DataFrame,
    *,
    tip_col: str = "tip_ts",
    as_of_col: str = "as_of_ts",
) -> pd.DataFrame:
    """Return at most one injury row per (game_id, player_id).

    The silver injuries snapshot tables can contain multiple rows per
    (game_id, player_id) (e.g. periodic refreshes, legacy partition overlap).
    If we merge those directly we duplicate player-game rows and corrupt any
    shift-based history features.
    """

    if injuries_snapshot.empty:
        return injuries_snapshot.copy()

    working = ensure_as_of_column(injuries_snapshot.copy(), column=as_of_col)
    working[as_of_col] = pd.to_datetime(working[as_of_col], utc=True, errors="coerce")
    working = working.dropna(subset=["game_id", "player_id"])

    tip_lookup: pd.DataFrame | None = None
    if tip_col in base_df.columns and "game_id" in base_df.columns:
        tip_lookup = base_df.loc[:, ["game_id", tip_col]].drop_duplicates().copy()
        tip_lookup[tip_col] = pd.to_datetime(tip_lookup[tip_col], utc=True, errors="coerce")

    group_cols = ["game_id", "player_id"]
    if tip_lookup is not None and not tip_lookup.empty:
        working = working.merge(tip_lookup, on="game_id", how="left")
        valid = working[as_of_col].notna() & working[tip_col].notna() & (working[as_of_col] <= working[tip_col])
        eligible = working.loc[valid].copy()
        if eligible.empty:
            return working.iloc[0:0].drop(columns=[tip_col], errors="ignore")
        idx = eligible.groupby(group_cols)[as_of_col].idxmax()
        return eligible.loc[idx].drop(columns=[tip_col], errors="ignore").reset_index(drop=True)

    eligible = working.dropna(subset=[as_of_col])
    if eligible.empty:
        return working.iloc[0:0].copy()
    idx = eligible.groupby(group_cols)[as_of_col].idxmax()
    return eligible.loc[idx].reset_index(drop=True)


def attach_availability_features(
    base_df: pd.DataFrame,
    injuries_snapshot: pd.DataFrame | None = None,
    *,
    prepared_injuries: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attach status priors and availability indicators to the base label frame.
    
    Key invariant: injury_as_of_ts is broadcast to ALL rows in a game if ANY
    injury data exists for that game. This ensures the injury snapshot timestamp
    is available for provenance even for players without an injury record.
    """

    if prepared_injuries is None:
        if injuries_snapshot is None or injuries_snapshot.empty:
            enriched = base_df.copy()
            enriched["status"] = AvailabilityStatus.UNKNOWN
            enriched["prior_play_prob"] = STATUS_PRIORS[AvailabilityStatus.UNKNOWN]
            enriched["is_out"] = 0
            enriched["is_q"] = 0
            enriched["is_prob"] = 0
            enriched["injury_as_of_ts"] = pd.NaT
            enriched["injury_snapshot_missing"] = 1
            enriched["injury_row_present"] = False
            return enriched
        prepared_injuries = prepare_injuries_snapshot(injuries_snapshot)

    missing_cols: list[str] = [
        col for col in _INJURY_COLUMNS if col not in prepared_injuries.columns
    ]
    if missing_cols:
        raise ValueError(
            f"Injuries snapshot missing required columns: {', '.join(sorted(missing_cols))}"
        )

    prepared_injuries = _select_latest_injury_snapshot(base_df, prepared_injuries)

    # Step 1: Create per-game injury snapshot timestamp table
    # This will be broadcast to ALL rows in a game
    if not prepared_injuries.empty and "as_of_ts" in prepared_injuries.columns:
        game_injury_ts = (
            prepared_injuries.groupby("game_id")["as_of_ts"]
            .max()
            .reset_index()
            .rename(columns={"as_of_ts": "injury_as_of_ts"})
        )
    else:
        game_injury_ts = pd.DataFrame(columns=["game_id", "injury_as_of_ts"])

    # Step 2: Merge per-game injury_as_of_ts to ALL rows by game_id
    merged = base_df.merge(game_injury_ts, on="game_id", how="left")
    
    # Step 3: Player-level injury join for status/is_out
    available_cols = [col for col in _INJURY_COLUMNS if col in prepared_injuries.columns and col != "as_of_ts"]
    optional_cols = [col for col in _OPTIONAL_INJURY_COLUMNS if col in prepared_injuries.columns]
    merge_cols = available_cols + optional_cols
    
    if merge_cols and not prepared_injuries.empty:
        player_injuries = prepared_injuries[merge_cols].copy()
        # Add indicator for player-level join
        player_injuries["injury_row_present"] = True
        merged = merged.merge(
            player_injuries,
            on=["game_id", "player_id"],
            how="left",
        )
    else:
        merged["injury_row_present"] = False
    
    # Fill defaults for players without injury records
    merged["injury_row_present"] = merged["injury_row_present"].fillna(False).astype(bool)
    merged["status"] = merged["status"].fillna(AvailabilityStatus.UNKNOWN) if "status" in merged.columns else AvailabilityStatus.UNKNOWN
    merged["prior_play_prob"] = merged["status"].map(STATUS_PRIORS)
    merged["is_out"] = (merged["status"] == AvailabilityStatus.OUT).astype(int)
    merged["is_q"] = (merged["status"] == AvailabilityStatus.QUESTIONABLE).astype(int)
    merged["is_prob"] = (merged["status"] == AvailabilityStatus.PROBABLE).astype(int)
    
    # Fill missing return/ramp metadata to avoid NaNs at inference.
    for col in ("restriction_flag", "ramp_flag"):
        if col in merged.columns:
            merged[col] = merged[col].fillna(False).astype(bool)
    for col in ("games_since_return", "days_since_return"):
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0).astype("Int64")

    # Step 4: injury_snapshot_missing = 1 only if NO usable snapshot for that GAME
    # (not if player is missing from snapshot - that's injury_row_present=False)
    if "snapshot_missing" in merged.columns:
        merged["injury_snapshot_missing"] = merged["snapshot_missing"].fillna(1).astype(int)
        merged.drop(columns=["snapshot_missing"], inplace=True)
    else:
        merged["injury_snapshot_missing"] = merged["injury_as_of_ts"].isna().astype(int)
    
    return merged

