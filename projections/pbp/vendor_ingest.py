from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from projections.pbp.identity import IdentityResolutionResult, resolve_player_ids


VENDOR_LINEUP_COLS_AWAY = [f"a{i}" for i in range(1, 6)]
VENDOR_LINEUP_COLS_HOME = [f"h{i}" for i in range(1, 6)]
VENDOR_PLAYER_NAME_COLS = [
    "player",
    "assist",
    "block",
    "steal",
    "entered",
    "left",
    "away",
    "home",
    "opponent",
    "outof",
]


CANON_PBP_EVENTS_COLS = [
    "schema_version",
    "season_id",
    "game_id",
    "game_date",
    "data_set",
    "period",
    "play_id",
    "event_type",
    "team",
    "away_score",
    "home_score",
    "remaining_time",
    "clock_sec",
    "elapsed",
    "period_elapsed_sec",
    "play_length",
    "play_length_sec",
    "description",
    # identity-resolved players
    "player_id",
    "assist_player_id",
    "block_player_id",
    "steal_player_id",
    "entered_player_id",
    "left_player_id",
    "away_player_id",
    "home_player_id",
    "opponent_player_id",
    "outof_player_id",
    # on-court state
    "away_p1",
    "away_p2",
    "away_p3",
    "away_p4",
    "away_p5",
    "home_p1",
    "home_p2",
    "home_p3",
    "home_p4",
    "home_p5",
    "away_lineup_key",
    "home_lineup_key",
    # deterministic ordering
    "event_index",
]


def _sha256_file(path: Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_bytes)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def canonical_game_id(vendor_game_id: str) -> str:
    s = str(vendor_game_id).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        raise ValueError("Missing vendor game_id")
    # Vendor example: "22400061" (no leading zeros). Canonical: 10-digit.
    return s.zfill(10)


def parse_hms_to_seconds(value: str) -> int:
    s = str(value).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return 0
    parts = s.split(":")
    if len(parts) == 2:
        mm, ss = parts
        hh = "0"
    elif len(parts) == 3:
        hh, mm, ss = parts
    else:
        raise ValueError(f"Unparseable time string: {value!r}")
    return int(hh) * 3600 + int(mm) * 60 + int(ss)


def period_length_sec(period: int) -> int:
    return 720 if period <= 4 else 300


def sort_lineup_ids(arr: np.ndarray) -> np.ndarray:
    """Sort on-court player_id columns for stable (unordered) set comparison."""
    return np.sort(arr.astype(np.int64), axis=1)


def _require_columns(df: pd.DataFrame, required: list[str], *, context: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{context}: missing required columns: {missing}. Got={list(df.columns)}")


@dataclass
class IngestResult:
    pbp_events: pd.DataFrame
    identity: IdentityResolutionResult
    input_sha256: str


class VendorLineupMissingError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        num_rows_with_lineup_na_before_fill: int,
        num_rows_with_lineup_na_after_fill: int,
        bad_counts: dict[str, int] | None = None,
        sample_rows: list[dict] | None = None,
    ) -> None:
        super().__init__(message)
        self.num_rows_with_lineup_na_before_fill = num_rows_with_lineup_na_before_fill
        self.num_rows_with_lineup_na_after_fill = num_rows_with_lineup_na_after_fill
        self.bad_counts = bad_counts or {}
        self.sample_rows = sample_rows or []


def _num_rows_with_any_lineup_na(df: pd.DataFrame, lineup_cols: list[str]) -> int:
    return int(df[lineup_cols].isna().any(axis=1).sum())


def ingest_vendor_game_csv(
    csv_path: Path,
    *,
    season_id: str,
    schema_version: str,
    prev_players_dim: Optional[pd.DataFrame],
) -> IngestResult:
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=True)
    _require_columns(
        df,
        [
            "game_id",
            "date",
            "data_set",
            "period",
            "play_id",
            "remaining_time",
            "elapsed",
            "play_length",
            "event_type",
        ]
        + VENDOR_LINEUP_COLS_AWAY
        + VENDOR_LINEUP_COLS_HOME,
        context=f"vendor csv {csv_path.name}",
    )

    lineup_cols = VENDOR_LINEUP_COLS_AWAY + VENDOR_LINEUP_COLS_HOME

    # Lineup "state fill" to repair sparse NA rows (timeouts/subs at same clock, etc.).
    # Sort deterministically first: period asc, clock_sec desc, play_id asc.
    df[lineup_cols] = df[lineup_cols].replace(r"^\\s*$", np.nan, regex=True)
    df["_period_int"] = pd.to_numeric(df["period"], errors="coerce").astype("Int64")
    df["_play_id_int"] = pd.to_numeric(df["play_id"], errors="coerce").astype("Int64")
    df["_clock_sec"] = df["remaining_time"].map(parse_hms_to_seconds).astype(int)
    df = df.sort_values(
        ["game_id", "_period_int", "_clock_sec", "_play_id_int"],
        ascending=[True, True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    num_rows_with_lineup_na_before_fill = _num_rows_with_any_lineup_na(df, lineup_cols)
    df[lineup_cols] = df.groupby(["game_id", "_period_int"], sort=False)[lineup_cols].ffill().bfill()
    num_rows_with_lineup_na_after_fill = _num_rows_with_any_lineup_na(df, lineup_cols)

    if num_rows_with_lineup_na_after_fill:
        bad_mask = df[lineup_cols].isna()
        bad_counts = {c: int(bad_mask[c].sum()) for c in lineup_cols if int(bad_mask[c].sum())}
        sample_rows = df.loc[
            bad_mask.any(axis=1), ["period", "play_id", "remaining_time"] + lineup_cols
        ].head(10)
        raise VendorLineupMissingError(
            "Vendor lineup columns contain missing values after state-fill.",
            num_rows_with_lineup_na_before_fill=num_rows_with_lineup_na_before_fill,
            num_rows_with_lineup_na_after_fill=num_rows_with_lineup_na_after_fill,
            bad_counts=bad_counts,
            sample_rows=sample_rows.to_dict(orient="records"),
        )

    # Resolve identity across lineups + common player-name columns.
    all_name_cols = (
        VENDOR_LINEUP_COLS_AWAY + VENDOR_LINEUP_COLS_HOME + [c for c in VENDOR_PLAYER_NAME_COLS if c in df.columns]
    )
    vendor_names = (
        pd.concat([df[c] for c in all_name_cols], ignore_index=True)
        .dropna()
        .astype(str)
        .tolist()
    )
    identity = resolve_player_ids(vendor_names, season_id=season_id, prev_players_dim=prev_players_dim)
    name_to_id = identity.name_to_player_id

    out = pd.DataFrame()
    out["schema_version"] = schema_version
    out["season_id"] = season_id
    out["game_id"] = df["game_id"].map(canonical_game_id)
    out["game_date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
    out["data_set"] = df["data_set"].astype("string")

    out["period"] = df["period"].astype(int)
    out["play_id"] = df["play_id"].astype(int)

    out["event_type"] = df["event_type"].astype("string")
    out["team"] = df["team"].astype("string") if "team" in df.columns else pd.Series([pd.NA] * len(df), dtype="string")
    if "away_score" in df.columns:
        out["away_score"] = pd.to_numeric(df["away_score"], errors="coerce").fillna(0).astype(int)
    else:
        out["away_score"] = 0
    if "home_score" in df.columns:
        out["home_score"] = pd.to_numeric(df["home_score"], errors="coerce").fillna(0).astype(int)
    else:
        out["home_score"] = 0

    out["remaining_time"] = df["remaining_time"].astype("string")
    out["clock_sec"] = df["remaining_time"].map(parse_hms_to_seconds).astype(int)
    out["elapsed"] = df["elapsed"].astype("string")
    out["period_elapsed_sec"] = df["elapsed"].map(parse_hms_to_seconds).astype(int)
    out["play_length"] = df["play_length"].astype("string")
    out["play_length_sec"] = df["play_length"].map(parse_hms_to_seconds).astype(int)
    out["description"] = df["description"].astype("string") if "description" in df.columns else pd.Series([pd.NA] * len(df), dtype="string")

    def _id_col(col: str) -> pd.Series:
        if col not in df.columns:
            return pd.Series([pd.NA] * len(df), dtype="Int64")
        return df[col].map(lambda x: name_to_id.get(str(x).strip()) if pd.notna(x) and str(x).strip() != "" else pd.NA).astype("Int64")

    out["player_id"] = _id_col("player")
    out["assist_player_id"] = _id_col("assist")
    out["block_player_id"] = _id_col("block")
    out["steal_player_id"] = _id_col("steal")
    out["entered_player_id"] = _id_col("entered")
    out["left_player_id"] = _id_col("left")
    out["away_player_id"] = _id_col("away")
    out["home_player_id"] = _id_col("home")
    out["opponent_player_id"] = _id_col("opponent")
    out["outof_player_id"] = _id_col("outof")

    # On-court lineups as stable (sorted) player_id columns.
    away_ids_df = pd.DataFrame(
        {c: df[c].astype(str).str.strip().map(name_to_id) for c in VENDOR_LINEUP_COLS_AWAY}
    )
    home_ids_df = pd.DataFrame(
        {c: df[c].astype(str).str.strip().map(name_to_id) for c in VENDOR_LINEUP_COLS_HOME}
    )
    if away_ids_df.isna().any().any() or home_ids_df.isna().any().any():
        missing = {
            "away_missing": int(away_ids_df.isna().sum().sum()),
            "home_missing": int(home_ids_df.isna().sum().sum()),
        }
        raise ValueError(f"Identity mapping produced null on-court player_id values: {missing}")

    away_sorted = sort_lineup_ids(away_ids_df.to_numpy(dtype=np.int64))
    home_sorted = sort_lineup_ids(home_ids_df.to_numpy(dtype=np.int64))

    for i in range(5):
        out[f"away_p{i+1}"] = away_sorted[:, i].astype(np.int64)
        out[f"home_p{i+1}"] = home_sorted[:, i].astype(np.int64)

    out["away_lineup_key"] = (
        out[[f"away_p{i}" for i in range(1, 6)]]
        .astype(str)
        .agg("|".join, axis=1)
        .astype("string")
    )
    out["home_lineup_key"] = (
        out[[f"home_p{i}" for i in range(1, 6)]]
        .astype(str)
        .agg("|".join, axis=1)
        .astype("string")
    )

    # Deterministic ordering + stable event_index.
    out = out.sort_values(["game_id", "period", "period_elapsed_sec", "play_id"], kind="mergesort").reset_index(drop=True)
    out["event_index"] = (
        out.groupby("game_id").cumcount().astype(int)
    )

    # Enforce strict column order + presence (internal schema contract).
    missing = [c for c in CANON_PBP_EVENTS_COLS if c not in out.columns]
    if missing:
        raise ValueError(f"Internal schema bug: missing canonical columns: {missing}")
    out = out[CANON_PBP_EVENTS_COLS].copy()

    return IngestResult(
        pbp_events=out,
        identity=identity,
        input_sha256=_sha256_file(csv_path),
    )


def save_input_hashes(input_hashes: dict[str, str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"files": input_hashes}
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
