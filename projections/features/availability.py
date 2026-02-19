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
    "AVA": AvailabilityStatus.AVAILABLE,
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


def normalize_status(value: str | AvailabilityStatus | None) -> AvailabilityStatus:
    """Map raw injury text into the canonical enum."""

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return AvailabilityStatus.UNKNOWN
    if isinstance(value, AvailabilityStatus):
        return value
    if value is pd.NA:
        return AvailabilityStatus.UNKNOWN
    normalized = str(value).strip().upper()
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
        valid = (
            working[as_of_col].notna()
            & working[tip_col].notna()
            & (working[as_of_col] <= working[tip_col])
        )

        selected_frames: list[pd.DataFrame] = []

        eligible = working.loc[valid].copy()
        if not eligible.empty:
            idx = eligible.groupby(group_cols)[as_of_col].idxmax()
            selected_frames.append(eligible.loc[idx])

        # Parity: if no pre-tip snapshot exists for a player-game, keep a single
        # placeholder row (e.g., snapshot_missing=1 / status=UNK) instead of
        # dropping the key entirely.
        if len(selected_frames) < len(working):
            present_keys = set()
            if selected_frames:
                already = pd.concat(selected_frames, ignore_index=True)
                present_keys = set(
                    zip(
                        pd.to_numeric(already["game_id"], errors="coerce").fillna(-1).astype(int),
                        pd.to_numeric(already["player_id"], errors="coerce").fillna(-1).astype(int),
                    )
                )

            all_keys = set(
                zip(
                    pd.to_numeric(working["game_id"], errors="coerce").fillna(-1).astype(int),
                    pd.to_numeric(working["player_id"], errors="coerce").fillna(-1).astype(int),
                )
            )
            missing_keys = all_keys - present_keys
            if missing_keys:
                missing_df = pd.DataFrame(list(missing_keys), columns=["game_id", "player_id"])
                fallback = working.merge(missing_df, on=["game_id", "player_id"], how="inner")

                # Prefer explicit snapshot_missing placeholders when available.
                if "snapshot_missing" in fallback.columns:
                    snap = pd.to_numeric(fallback["snapshot_missing"], errors="coerce").fillna(0).astype(int)
                    fallback = fallback.loc[snap == 1].copy()

                if not fallback.empty:
                    order_cols: list[str] = []
                    if "ingested_ts" in fallback.columns:
                        fallback["ingested_ts"] = pd.to_datetime(fallback["ingested_ts"], utc=True, errors="coerce")
                        order_cols.append("ingested_ts")
                    # As a secondary stable order, keep the original as_of_ts (often NaT).
                    order_cols.append(as_of_col)
                    fallback = fallback.sort_values(order_cols, ascending=True, na_position="last")
                    fallback = fallback.groupby(group_cols, as_index=False).tail(1)
                    selected_frames.append(fallback)

        if not selected_frames:
            return working.iloc[0:0].drop(columns=[tip_col], errors="ignore")
        selected = pd.concat(selected_frames, ignore_index=True)
        selected = selected.drop(columns=[tip_col], errors="ignore")
        selected = selected.drop_duplicates(subset=group_cols, keep="last")
        return selected.reset_index(drop=True)

    eligible = working.dropna(subset=[as_of_col])
    selected_frames: list[pd.DataFrame] = []
    if not eligible.empty:
        idx = eligible.groupby(group_cols)[as_of_col].idxmax()
        selected_frames.append(eligible.loc[idx])

    if len(selected_frames) < len(working):
        present_keys = set()
        if selected_frames:
            already = pd.concat(selected_frames, ignore_index=True)
            present_keys = set(
                zip(
                    pd.to_numeric(already["game_id"], errors="coerce").fillna(-1).astype(int),
                    pd.to_numeric(already["player_id"], errors="coerce").fillna(-1).astype(int),
                )
            )
        all_keys = set(
            zip(
                pd.to_numeric(working["game_id"], errors="coerce").fillna(-1).astype(int),
                pd.to_numeric(working["player_id"], errors="coerce").fillna(-1).astype(int),
            )
        )
        missing_keys = all_keys - present_keys
        if missing_keys:
            missing_df = pd.DataFrame(list(missing_keys), columns=["game_id", "player_id"])
            fallback = working.merge(missing_df, on=["game_id", "player_id"], how="inner")
            if "snapshot_missing" in fallback.columns:
                snap = pd.to_numeric(fallback["snapshot_missing"], errors="coerce").fillna(0).astype(int)
                fallback = fallback.loc[snap == 1].copy()
            if not fallback.empty:
                order_cols: list[str] = []
                if "ingested_ts" in fallback.columns:
                    fallback["ingested_ts"] = pd.to_datetime(fallback["ingested_ts"], utc=True, errors="coerce")
                    order_cols.append("ingested_ts")
                order_cols.append(as_of_col)
                fallback = fallback.sort_values(order_cols, ascending=True, na_position="last")
                fallback = fallback.groupby(group_cols, as_index=False).tail(1)
                selected_frames.append(fallback)

    if not selected_frames:
        return working.iloc[0:0].copy()

    selected = pd.concat(selected_frames, ignore_index=True)
    selected = selected.drop_duplicates(subset=group_cols, keep="last")
    return selected.reset_index(drop=True)


def attach_availability_features(
    base_df: pd.DataFrame,
    injuries_snapshot: pd.DataFrame | None = None,
    *,
    prepared_injuries: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attach status priors and availability indicators to the base label frame.

    Availability invariants (live + training):
    - Rows without an injury-feed record are labeled `status="Ava"` ("no injury row"), with:
        - injury_as_of_ts = NaT
        - injury_snapshot_missing = 1
        - prior_play_prob = STATUS_PRIORS[AvailabilityStatus.AVAILABLE]
    - Rows with a record but missing the pre-tip snapshot (e.g. snapshot_missing=1) keep
      `status="UNK"`, with:
        - injury_as_of_ts = NaT
        - injury_snapshot_missing = 1
        - prior_play_prob = 0.82

    Rows with a valid pre-tip snapshot keep canonical status (OUT/Q/PROB/AVAIL) and
    injury_snapshot_missing=0.
    """

    if prepared_injuries is None:
        if injuries_snapshot is None or injuries_snapshot.empty:
            enriched = base_df.copy()
            enriched["status"] = "Ava"
            enriched["prior_play_prob"] = float(STATUS_PRIORS[AvailabilityStatus.AVAILABLE])
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

    # Player-level injury join for status/is_out.
    merged = base_df.copy()
    optional_cols = [col for col in _OPTIONAL_INJURY_COLUMNS if col in prepared_injuries.columns]
    merge_cols = list(_INJURY_COLUMNS) + list(optional_cols)

    if not prepared_injuries.empty:
        player_injuries = prepared_injuries.loc[:, [c for c in merge_cols if c in prepared_injuries.columns]].copy()
        # Rename the snapshot timestamp onto the canonical column name.
        if "as_of_ts" in player_injuries.columns:
            player_injuries = player_injuries.rename(columns={"as_of_ts": "injury_as_of_ts"})
        player_injuries["injury_row_present"] = True
        merged = merged.merge(player_injuries, on=["game_id", "player_id"], how="left")
    else:
        merged["injury_row_present"] = False

    # Avoid pandas FutureWarning about silent downcasting on fillna for object dtype.
    merged["injury_row_present"] = (
        merged["injury_row_present"].astype("boolean", copy=False).fillna(False).astype(bool)
    )
    merged["injury_as_of_ts"] = pd.to_datetime(merged.get("injury_as_of_ts"), utc=True, errors="coerce")

    # Note: injury_row_present indicates that a player-game key exists in the snapshot,
    # even when the pre-tip snapshot is missing (injury_as_of_ts is null).
    has_row_mask = merged["injury_row_present"].astype(bool)

    # Missing injury rows become status="Ava" with an explicit available prior.
    no_row = ~has_row_mask
    status = merged.get("status", pd.Series(pd.NA, index=merged.index))
    status = status.where(~no_row, "Ava")
    # Normalize statuses only for rows that have an injury-feed record.
    normalized = status.copy()
    if has_row_mask.any():
        normalized.loc[has_row_mask] = normalized.loc[has_row_mask].apply(normalize_status)

    # Convert enum -> string tokens for modeling / parquet compatibility.
    status_out = pd.Series("Ava", index=merged.index, dtype="string[pyarrow]")
    if has_row_mask.any():
        # normalized values are AvailabilityStatus for has_row rows, but may be strings/NA in edge cases.
        norm_vals = normalized.loc[has_row_mask]
        norm_enum = norm_vals.apply(normalize_status)
        status_out.loc[has_row_mask] = norm_enum.map(lambda v: v.value).astype("string[pyarrow]")
    merged["status"] = status_out

    prior = pd.Series(
        float(STATUS_PRIORS[AvailabilityStatus.AVAILABLE]),
        index=merged.index,
        dtype="float64",
    )
    if has_row_mask.any():
        norm_enum = normalized.loc[has_row_mask].apply(normalize_status)
        prior.loc[has_row_mask] = norm_enum.map(STATUS_PRIORS).astype("float64")
    merged["prior_play_prob"] = prior

    merged["is_out"] = ((merged["status"] == AvailabilityStatus.OUT.value) & has_row_mask).astype(int)
    merged["is_q"] = ((merged["status"] == AvailabilityStatus.QUESTIONABLE.value) & has_row_mask).astype(int)
    merged["is_prob"] = ((merged["status"] == AvailabilityStatus.PROBABLE.value) & has_row_mask).astype(int)
    
    # Fill missing return/ramp metadata to avoid NaNs at inference.
    for col in ("restriction_flag", "ramp_flag"):
        if col in merged.columns:
            merged[col] = merged[col].fillna(False).astype(bool)
    for col in ("games_since_return", "days_since_return"):
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0).astype("Int64")

    # injury_snapshot_missing: row-level missingness (parity with training).
    merged["injury_snapshot_missing"] = 1
    if "snapshot_missing" in merged.columns:
        snapshot_missing = pd.to_numeric(merged["snapshot_missing"], errors="coerce").fillna(0).astype(int)
        merged.loc[has_row_mask, "injury_snapshot_missing"] = snapshot_missing.loc[has_row_mask]
        merged.drop(columns=["snapshot_missing"], inplace=True)
    else:
        # If we have a row and a timestamp, treat as observed. Otherwise keep missing=1.
        observed = has_row_mask & merged["injury_as_of_ts"].notna()
        merged.loc[observed, "injury_snapshot_missing"] = 0
    
    return merged
