from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from projections import paths

MANUAL_AVAILABILITY_FILENAME = "manual_overrides.parquet"
MANUAL_OVERRIDE_TYPES: tuple[str, ...] = ("force_out", "force_in")
MANUAL_OVERRIDE_COLUMNS: tuple[str, ...] = (
    "override_id",
    "game_date",
    "game_id",
    "player_id",
    "player_name",
    "team_id",
    "team_tricode",
    "override_type",
    "reason_code",
    "reason_text",
    "source_label",
    "entered_by",
    "created_ts",
    "effective_ts",
    "expires_ts",
    "active",
    "cleared_ts",
    "cleared_by",
)

_STRING_COLUMNS = {
    "override_id",
    "game_date",
    "game_id",
    "player_id",
    "player_name",
    "team_tricode",
    "override_type",
    "reason_code",
    "reason_text",
    "source_label",
    "entered_by",
    "cleared_by",
}
_TIMESTAMP_COLUMNS = {"created_ts", "effective_ts", "expires_ts", "cleared_ts"}

FRAME_METADATA_COLUMNS: tuple[str, ...] = (
    "manual_override_id",
    "manual_override_type",
    "manual_override_reason_code",
    "manual_override_reason_text",
    "manual_override_source_label",
    "manual_override_entered_by",
    "manual_override_created_ts",
    "manual_override_effective_ts",
    "manual_override_expires_ts",
    "manual_override_active",
    "manual_override_used",
)


def _utc_now_ts() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(tz=UTC).replace(microsecond=0))


def _coerce_utc_ts(value: Any) -> pd.Timestamp:
    if value is None or value == "":
        return pd.NaT
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    return ts if not pd.isna(ts) else pd.NaT


def _normalize_id_str(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _normalize_id_str_series(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series([], index=series.index, dtype="string")
    out = series.astype("string", copy=False).fillna("")
    numeric = pd.to_numeric(series, errors="coerce")
    int_like = numeric.notna() & (numeric % 1 == 0)
    if int_like.any():
        out = out.where(~int_like, numeric.where(int_like).astype("Int64").astype("string"))
    return out.str.replace(r"\.0$", "", regex=True)


def _manual_override_dir(data_root: Path, game_date: date) -> Path:
    return data_root / "live" / "manual_overrides" / f"game_date={game_date.isoformat()}"


def manual_overrides_path(game_date: date, *, data_root: Path | None = None) -> Path:
    root = data_root or paths.data_path()
    return _manual_override_dir(root, game_date) / MANUAL_AVAILABILITY_FILENAME


def _empty_overrides_df() -> pd.DataFrame:
    df = pd.DataFrame(columns=MANUAL_OVERRIDE_COLUMNS)
    for col in _STRING_COLUMNS:
        df[col] = df[col].astype("string")
    for col in _TIMESTAMP_COLUMNS:
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    df["team_id"] = df["team_id"].astype("Int64")
    df["active"] = df["active"].astype("boolean")
    return df


def _normalize_override_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return _empty_overrides_df()

    work = df.copy()
    for col in MANUAL_OVERRIDE_COLUMNS:
        if col not in work.columns:
            work[col] = pd.NA

    for col in _STRING_COLUMNS:
        work[col] = work[col].astype("string")
    for col in _TIMESTAMP_COLUMNS:
        work[col] = pd.to_datetime(work[col], utc=True, errors="coerce")
    work["team_id"] = pd.to_numeric(work["team_id"], errors="coerce").astype("Int64")
    work["active"] = work["active"].astype("boolean").fillna(False)
    work["game_date"] = work["game_date"].fillna("").astype("string")
    work["game_id"] = _normalize_id_str_series(work["game_id"])
    work["player_id"] = _normalize_id_str_series(work["player_id"])
    work["override_id"] = work["override_id"].fillna("").astype("string")
    work["override_type"] = work["override_type"].fillna("").astype("string").str.strip().str.lower()

    work = work.loc[
        work["override_id"].ne("")
        & work["game_id"].ne("")
        & work["player_id"].ne("")
        & work["override_type"].isin(MANUAL_OVERRIDE_TYPES)
    ].copy()
    if work.empty:
        return _empty_overrides_df()

    return work.loc[:, list(MANUAL_OVERRIDE_COLUMNS)].sort_values(
        ["created_ts", "override_id"], kind="mergesort"
    )


def load_manual_overrides_df(game_date: date, *, data_root: Path | None = None) -> pd.DataFrame:
    path = manual_overrides_path(game_date, data_root=data_root)
    if not path.exists():
        return _empty_overrides_df()
    try:
        df = pd.read_parquet(path)
    except Exception:
        return _empty_overrides_df()
    return _normalize_override_frame(df)


def _active_mask(df: pd.DataFrame, *, as_of_ts: Any | None = None) -> pd.Series:
    if df.empty:
        return pd.Series([], index=df.index, dtype=bool)
    effective_ts = (
        _coerce_utc_ts(as_of_ts)
        if as_of_ts is not None
        else _utc_now_ts()
    )
    active = df["active"].fillna(False).astype(bool)
    not_cleared = df["cleared_ts"].isna()
    effective = df["effective_ts"].isna() | (df["effective_ts"] <= effective_ts)
    not_expired = df["expires_ts"].isna() | (df["expires_ts"] > effective_ts)
    return active & not_cleared & effective & not_expired


def list_manual_overrides(
    game_date: date,
    *,
    data_root: Path | None = None,
    active_only: bool = False,
    as_of_ts: Any | None = None,
) -> list[dict[str, Any]]:
    df = load_manual_overrides_df(game_date, data_root=data_root)
    if active_only:
        df = df.loc[_active_mask(df, as_of_ts=as_of_ts)].copy()
    return [_record_to_json(row) for row in df.to_dict(orient="records")]


def _record_to_json(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: (
            None
            if pd.isna(value)
            else value.isoformat().replace("+00:00", "Z")
            if isinstance(value, pd.Timestamp)
            else value.item()
            if hasattr(value, "item")
            else value
        )
        for key, value in row.items()
    }


def _write_override_frame(df: pd.DataFrame, *, game_date: date, data_root: Path | None = None) -> Path:
    path = manual_overrides_path(game_date, data_root=data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _normalize_override_frame(df)
    tmp = path.with_suffix(f".tmp.{datetime.now(tz=UTC).strftime('%Y%m%dT%H%M%SZ')}.parquet")
    payload.to_parquet(tmp, index=False)
    tmp.replace(path)
    return path


def upsert_manual_override(
    game_date: date,
    *,
    game_id: Any,
    player_id: Any,
    player_name: str | None,
    team_id: Any,
    team_tricode: str | None,
    override_type: str,
    entered_by: str,
    reason_code: str | None = None,
    reason_text: str | None = None,
    source_label: str | None = None,
    expires_ts: Any | None = None,
    effective_ts: Any | None = None,
    data_root: Path | None = None,
) -> dict[str, Any]:
    override_type_norm = str(override_type or "").strip().lower()
    if override_type_norm not in MANUAL_OVERRIDE_TYPES:
        raise ValueError(f"override_type must be one of {MANUAL_OVERRIDE_TYPES}")
    game_id_norm = _normalize_id_str(game_id)
    player_id_norm = _normalize_id_str(player_id)
    entered_by_norm = str(entered_by or "").strip()
    if not game_id_norm:
        raise ValueError("game_id is required")
    if not player_id_norm:
        raise ValueError("player_id is required")
    if not entered_by_norm:
        raise ValueError("entered_by is required")

    now = _utc_now_ts()
    eff_ts = _coerce_utc_ts(effective_ts)
    exp_ts = _coerce_utc_ts(expires_ts)
    if pd.isna(eff_ts):
        eff_ts = now
    if not pd.isna(exp_ts) and exp_ts <= eff_ts:
        raise ValueError("expires_ts must be after effective_ts")

    existing = load_manual_overrides_df(game_date, data_root=data_root)
    if not existing.empty:
        same_player_game = (
            existing["game_id"].eq(game_id_norm)
            & existing["player_id"].eq(player_id_norm)
            & _active_mask(existing, as_of_ts=now)
        )
        if same_player_game.any():
            existing.loc[same_player_game, "active"] = False
            existing.loc[same_player_game, "cleared_ts"] = now
            existing.loc[same_player_game, "cleared_by"] = entered_by_norm

    record = {
        "override_id": str(uuid4()),
        "game_date": game_date.isoformat(),
        "game_id": game_id_norm,
        "player_id": player_id_norm,
        "player_name": None if player_name is None else str(player_name),
        "team_id": pd.to_numeric(pd.Series([team_id]), errors="coerce").astype("Int64").iloc[0],
        "team_tricode": None if team_tricode is None else str(team_tricode).strip().upper(),
        "override_type": override_type_norm,
        "reason_code": None if reason_code is None else str(reason_code),
        "reason_text": None if reason_text is None else str(reason_text),
        "source_label": None if source_label is None else str(source_label),
        "entered_by": entered_by_norm,
        "created_ts": now,
        "effective_ts": eff_ts,
        "expires_ts": exp_ts,
        "active": True,
        "cleared_ts": pd.NaT,
        "cleared_by": pd.NA,
    }
    new_row = _normalize_override_frame(pd.DataFrame([record]))
    if existing.empty:
        updated = new_row
    else:
        updated = pd.concat(
            [existing, new_row],
            ignore_index=True,
        )
    _write_override_frame(updated, game_date=game_date, data_root=data_root)
    return _record_to_json(record)


def clear_manual_override(
    game_date: date,
    *,
    override_id: str,
    cleared_by: str,
    data_root: Path | None = None,
) -> dict[str, Any] | None:
    override_id_norm = str(override_id or "").strip()
    cleared_by_norm = str(cleared_by or "").strip()
    if not override_id_norm:
        raise ValueError("override_id is required")
    if not cleared_by_norm:
        raise ValueError("cleared_by is required")

    df = load_manual_overrides_df(game_date, data_root=data_root)
    if df.empty:
        return None
    mask = df["override_id"].eq(override_id_norm)
    if not mask.any():
        return None
    now = _utc_now_ts()
    df.loc[mask, "active"] = False
    df.loc[mask, "cleared_ts"] = now
    df.loc[mask, "cleared_by"] = cleared_by_norm
    _write_override_frame(df, game_date=game_date, data_root=data_root)
    match_row = df.loc[mask].sort_values(["created_ts", "override_id"], kind="mergesort").iloc[-1].to_dict()
    return _record_to_json(match_row)


def manual_override_report(
    game_date: date,
    *,
    data_root: Path | None = None,
    as_of_ts: Any | None = None,
) -> dict[str, Any]:
    active = load_manual_overrides_df(game_date, data_root=data_root)
    active = active.loc[_active_mask(active, as_of_ts=as_of_ts)].copy()
    if active.empty:
        return {
            "active_override_count": 0,
            "affected_game_ids": [],
            "override_digest": None,
            "latest_effective_ts": None,
            "per_game": {},
        }

    digest_cols = [
        "override_id",
        "game_id",
        "player_id",
        "override_type",
        "reason_code",
        "reason_text",
        "source_label",
        "entered_by",
        "effective_ts",
        "expires_ts",
    ]
    digest_df = active.loc[:, digest_cols].copy().sort_values(
        ["game_id", "player_id", "override_id"], kind="mergesort"
    )
    for col in ("effective_ts", "expires_ts"):
        digest_df[col] = digest_df[col].map(
            lambda value: None
            if pd.isna(value)
            else pd.Timestamp(value).isoformat().replace("+00:00", "Z")
        )
    digest_payload = digest_df.where(pd.notna(digest_df), None).to_dict(orient="records")
    digest = hashlib.sha256(
        json.dumps(digest_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]

    per_game: dict[str, dict[str, Any]] = {}
    for game_id, game_df in active.groupby("game_id", sort=False):
        game_payload = game_df.loc[:, digest_cols].copy().sort_values(
            ["player_id", "override_id"], kind="mergesort"
        )
        for col in ("effective_ts", "expires_ts"):
            game_payload[col] = game_payload[col].map(
                lambda value: None
                if pd.isna(value)
                else pd.Timestamp(value).isoformat().replace("+00:00", "Z")
            )
        game_digest = hashlib.sha256(
            json.dumps(
                game_payload.where(pd.notna(game_payload), None).to_dict(orient="records"),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()[:16]
        latest_effective = game_df["effective_ts"].dropna().max()
        per_game[str(game_id)] = {
            "source_used": "manual_override",
            "latest_as_of_ts": None
            if pd.isna(latest_effective)
            else pd.Timestamp(latest_effective).isoformat(),
            "content_digest": game_digest,
            "active_override_count": int(len(game_df)),
        }

    latest_effective = active["effective_ts"].dropna().max()
    return {
        "active_override_count": int(len(active)),
        "affected_game_ids": sorted(
            pd.to_numeric(active["game_id"], errors="coerce").dropna().astype(int).unique().tolist()
        ),
        "override_digest": digest,
        "latest_effective_ts": None
        if pd.isna(latest_effective)
        else pd.Timestamp(latest_effective).isoformat(),
        "per_game": per_game,
    }


def apply_manual_overrides_to_frame(
    df: pd.DataFrame,
    *,
    overrides_df: pd.DataFrame,
    as_of_ts: Any | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    work = df.copy()
    for col in FRAME_METADATA_COLUMNS:
        if col not in work.columns:
            if col in {"manual_override_active", "manual_override_used"}:
                work[col] = False
            else:
                work[col] = pd.NA

    if work.empty:
        return work, {"matched_override_count": 0, "unmatched_override_count": 0, "unmatched_override_ids": []}

    active = _normalize_override_frame(overrides_df)
    active = active.loc[_active_mask(active, as_of_ts=as_of_ts)].copy()
    if active.empty:
        return work, {"matched_override_count": 0, "unmatched_override_count": 0, "unmatched_override_ids": []}

    work["_manual_key"] = _normalize_id_str_series(work["game_id"]) + "|" + _normalize_id_str_series(work["player_id"])
    active["_manual_key"] = active["game_id"] + "|" + active["player_id"]
    merge_cols = [
        "_manual_key",
        "override_id",
        "override_type",
        "reason_code",
        "reason_text",
        "source_label",
        "entered_by",
        "created_ts",
        "effective_ts",
        "expires_ts",
    ]
    merged = work.merge(active.loc[:, merge_cols], on="_manual_key", how="left")
    override_present = merged["override_type"].notna()

    merged["manual_override_id"] = merged["override_id"].where(override_present, pd.NA)
    merged["manual_override_type"] = merged["override_type"].where(override_present, pd.NA)
    merged["manual_override_reason_code"] = merged["reason_code"].where(override_present, pd.NA)
    merged["manual_override_reason_text"] = merged["reason_text"].where(override_present, pd.NA)
    merged["manual_override_source_label"] = merged["source_label"].where(override_present, pd.NA)
    merged["manual_override_entered_by"] = merged["entered_by"].where(override_present, pd.NA)
    merged["manual_override_created_ts"] = merged["created_ts"].where(override_present, pd.NaT)
    merged["manual_override_effective_ts"] = merged["effective_ts"].where(override_present, pd.NaT)
    merged["manual_override_expires_ts"] = merged["expires_ts"].where(override_present, pd.NaT)
    merged["manual_override_active"] = override_present
    merged["manual_override_used"] = override_present

    force_out = merged["override_type"].eq("force_out")
    force_in = merged["override_type"].eq("force_in")

    if "status" in merged.columns:
        merged.loc[force_out, "status"] = "OUT"
        merged.loc[force_in, "status"] = "ACTIVE"
    if "is_out" in merged.columns:
        merged.loc[force_out, "is_out"] = 1
        merged.loc[force_in, "is_out"] = 0
    if "is_q" in merged.columns:
        merged.loc[force_in, "is_q"] = 0
    if "is_prob" in merged.columns:
        merged.loc[force_in, "is_prob"] = 0
    if "play_prob" in merged.columns:
        merged.loc[force_out, "play_prob"] = 0.0
        merged.loc[force_in, "play_prob"] = 1.0
    if "prior_play_prob" in merged.columns:
        merged.loc[force_out, "prior_play_prob"] = 0.0
        merged.loc[force_in, "prior_play_prob"] = 1.0
    # Sim-level activity probability columns: keep consistent with play_prob so the
    # optimizer delta calculation uses p_active=1.0 for forced-in players.
    for p_active_col in ("minutes_sim_p_active", "p_play_eff", "sim_p_active"):
        if p_active_col in merged.columns:
            merged.loc[force_out, p_active_col] = 0.0
            merged.loc[force_in, p_active_col] = 1.0
    if "lineup_role" in merged.columns:
        merged.loc[force_out, "lineup_role"] = "out"
        merged.loc[force_in & merged["lineup_role"].astype("string").str.lower().eq("out"), "lineup_role"] = pd.NA
    for starter_col in ("is_projected_starter", "is_confirmed_starter", "starter_flag"):
        if starter_col in merged.columns:
            merged.loc[force_out, starter_col] = False if starter_col != "starter_flag" else 0
    for minutes_col in (
        "minutes_final",
        "minutes_p10",
        "minutes_p50",
        "minutes_p90",
        "minutes_p10_cond",
        "minutes_p50_cond",
        "minutes_p90_cond",
        "effective_minutes",
        "proj_minutes",
    ):
        if minutes_col in merged.columns:
            merged.loc[force_out, minutes_col] = 0.0

    matched_ids = set(merged.loc[override_present, "override_id"].dropna().astype(str).tolist())
    unmatched_ids = sorted(
        set(active["override_id"].astype(str).tolist()) - matched_ids
    )
    merged = merged.drop(
        columns=[
            "_manual_key",
            "override_id",
            "override_type",
            "reason_code",
            "reason_text",
            "source_label",
            "entered_by",
            "created_ts",
            "effective_ts",
            "expires_ts",
        ],
        errors="ignore",
    )
    return merged, {
        "matched_override_count": len(matched_ids),
        "unmatched_override_count": len(unmatched_ids),
        "unmatched_override_ids": unmatched_ids,
    }
