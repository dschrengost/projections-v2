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


def attach_availability_features(
    base_df: pd.DataFrame,
    injuries_snapshot: pd.DataFrame | None = None,
    *,
    prepared_injuries: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attach status priors and availability indicators to the base label frame."""

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
            return enriched
        prepared_injuries = prepare_injuries_snapshot(injuries_snapshot)

    missing_cols: list[str] = [
        col for col in _INJURY_COLUMNS if col not in prepared_injuries.columns
    ]
    if missing_cols:
        raise ValueError(
            f"Injuries snapshot missing required columns: {', '.join(sorted(missing_cols))}"
        )

    available_cols = [col for col in _INJURY_COLUMNS if col in prepared_injuries.columns]
    optional_cols = [col for col in _OPTIONAL_INJURY_COLUMNS if col in prepared_injuries.columns]
    merge_cols = available_cols + optional_cols
    merged = base_df.merge(
        prepared_injuries[merge_cols],
        on=["game_id", "player_id"],
        how="left",
    )
    merged["status"] = merged["status"].fillna(AvailabilityStatus.UNKNOWN)
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

    merged.rename(columns={"as_of_ts": "injury_as_of_ts"}, inplace=True)
    if "snapshot_missing" in merged.columns:
        # Treat missing snapshot indicator as missing (conservative).
        merged["injury_snapshot_missing"] = merged["snapshot_missing"].fillna(1).astype(int)
        merged.drop(columns=["snapshot_missing"], inplace=True)
    else:
        merged["injury_snapshot_missing"] = merged["injury_as_of_ts"].isna().astype(int)
    return merged
