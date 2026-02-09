from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from projections.minutes.depth_chart_prior import (
    _assign_team_ids_from_name,
    _load_snapshot_for_asof,
    load_depth_chart_prior_config,
)
from projections.pbp.identity import normalize_name

logger = logging.getLogger(__name__)

_DEFAULT_OVERRIDE_REL = Path("bronze/realgm/player_id_crosswalk_overrides.csv")


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table type: {path}")


def _atomic_write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        tmp = path.with_suffix(f".tmp.{pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%SZ')}.csv")
        tmp.write_text(df.to_csv(index=False), encoding="utf-8")
        tmp.replace(path)
        return
    tmp = path.with_suffix(f".tmp.{pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%SZ')}.parquet")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def _empty_crosswalk_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "realgm_player_id",
            "player_id",
            "updated_at",
            "match_method",
            "source_snapshot_ts",
            "note",
        ]
    )


def _load_overrides(path: Path) -> pd.DataFrame:
    if not path.exists() or not path.is_file():
        return _empty_crosswalk_frame().iloc[0:0].copy()

    raw = _read_table(path)
    required = {"realgm_player_id", "player_id"}
    missing = [c for c in sorted(required) if c not in raw.columns]
    if missing:
        raise ValueError(f"Crosswalk overrides missing columns {missing}: {path}")

    out = raw.copy()
    out["realgm_player_id"] = pd.to_numeric(out["realgm_player_id"], errors="coerce").astype("Int64")
    out["player_id"] = pd.to_numeric(out["player_id"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["realgm_player_id", "player_id"]).copy()
    if out.empty:
        return _empty_crosswalk_frame().iloc[0:0].copy()
    out["realgm_player_id"] = out["realgm_player_id"].astype(int)
    out["player_id"] = out["player_id"].astype(int)

    if "updated_at" in out.columns:
        out["updated_at"] = pd.to_datetime(out["updated_at"], utc=True, errors="coerce")
    else:
        out["updated_at"] = pd.NaT
    out["updated_at"] = out["updated_at"].where(
        out["updated_at"].notna(),
        pd.Timestamp.now(tz="UTC"),
    )
    out["match_method"] = "override"
    out["source_snapshot_ts"] = pd.NaT
    out["note"] = out.get("note", pd.Series("", index=out.index)).astype("string").fillna("")
    return out[_empty_crosswalk_frame().columns].copy()


def _load_existing_crosswalk(path: Path) -> pd.DataFrame:
    if not path.exists() or not path.is_file():
        return _empty_crosswalk_frame().iloc[0:0].copy()
    raw = _read_table(path)
    required = {"realgm_player_id", "player_id"}
    missing = [c for c in sorted(required) if c not in raw.columns]
    if missing:
        logger.warning(
            "[dc-crosswalk] existing crosswalk missing columns %s at %s; ignoring existing",
            missing,
            path,
        )
        return _empty_crosswalk_frame().iloc[0:0].copy()
    out = raw.copy()
    out["realgm_player_id"] = pd.to_numeric(out["realgm_player_id"], errors="coerce").astype("Int64")
    out["player_id"] = pd.to_numeric(out["player_id"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["realgm_player_id", "player_id"]).copy()
    if out.empty:
        return _empty_crosswalk_frame().iloc[0:0].copy()
    out["realgm_player_id"] = out["realgm_player_id"].astype(int)
    out["player_id"] = out["player_id"].astype(int)
    out["updated_at"] = pd.to_datetime(out.get("updated_at"), utc=True, errors="coerce")
    out["updated_at"] = out["updated_at"].fillna(pd.Timestamp("1970-01-01T00:00:00Z"))
    out["match_method"] = out.get("match_method", pd.Series("history", index=out.index)).astype("string").fillna("history")
    out["source_snapshot_ts"] = pd.to_datetime(out.get("source_snapshot_ts"), utc=True, errors="coerce")
    out["note"] = out.get("note", pd.Series("", index=out.index)).astype("string").fillna("")
    return out[_empty_crosswalk_frame().columns].copy()


def _build_team_name_matches(
    *,
    snapshot_df: pd.DataFrame,
    minutes_df: pd.DataFrame,
    selected_snapshot_ts: pd.Timestamp | None,
) -> tuple[pd.DataFrame, int]:
    if snapshot_df.empty or minutes_df.empty:
        return _empty_crosswalk_frame().iloc[0:0].copy(), 0

    if "player_id" not in minutes_df.columns:
        return _empty_crosswalk_frame().iloc[0:0].copy(), int(snapshot_df["realgm_player_id"].nunique())

    depth = snapshot_df.copy()
    depth = _assign_team_ids_from_name(depth, minutes_df)
    depth["team_id"] = pd.to_numeric(depth.get("team_id"), errors="coerce").astype("Int64")
    depth = depth.dropna(subset=["realgm_player_id", "team_id", "player_name"]).copy()
    if depth.empty:
        return _empty_crosswalk_frame().iloc[0:0].copy(), int(snapshot_df["realgm_player_id"].nunique())
    depth["team_id"] = depth["team_id"].astype(int)
    depth["_name_key"] = depth["player_name"].map(normalize_name)
    depth = depth[depth["_name_key"] != ""].copy()
    if depth.empty:
        return _empty_crosswalk_frame().iloc[0:0].copy(), int(snapshot_df["realgm_player_id"].nunique())

    minutes = minutes_df.copy()
    minutes["player_id"] = pd.to_numeric(minutes["player_id"], errors="coerce").astype("Int64")
    minutes["team_id"] = pd.to_numeric(minutes.get("team_id"), errors="coerce").astype("Int64")
    minutes = minutes.dropna(subset=["player_id", "team_id", "player_name"]).copy()
    if minutes.empty:
        return _empty_crosswalk_frame().iloc[0:0].copy(), int(snapshot_df["realgm_player_id"].nunique())
    minutes["player_id"] = minutes["player_id"].astype(int)
    minutes["team_id"] = minutes["team_id"].astype(int)
    minutes["_name_key"] = minutes["player_name"].map(normalize_name)
    minutes = minutes[minutes["_name_key"] != ""].copy()
    if minutes.empty:
        return _empty_crosswalk_frame().iloc[0:0].copy(), int(snapshot_df["realgm_player_id"].nunique())

    d_counts = depth.groupby(["team_id", "_name_key"], dropna=False)["realgm_player_id"].nunique().reset_index(name="_n_d")
    m_counts = minutes.groupby(["team_id", "_name_key"], dropna=False)["player_id"].nunique().reset_index(name="_n_m")

    depth = depth.merge(d_counts, on=["team_id", "_name_key"], how="left")
    minutes = minutes.merge(m_counts, on=["team_id", "_name_key"], how="left")
    depth = depth[depth["_n_d"] == 1].copy()
    minutes = minutes[minutes["_n_m"] == 1].copy()

    if depth.empty or minutes.empty:
        return _empty_crosswalk_frame().iloc[0:0].copy(), int(snapshot_df["realgm_player_id"].nunique())

    merged = depth.merge(
        minutes[["team_id", "_name_key", "player_id", "player_name"]],
        on=["team_id", "_name_key"],
        how="inner",
        suffixes=("_realgm", "_model"),
    )
    if merged.empty:
        return _empty_crosswalk_frame().iloc[0:0].copy(), int(snapshot_df["realgm_player_id"].nunique())

    now = pd.Timestamp.now(tz="UTC")
    out = pd.DataFrame(
        {
            "realgm_player_id": pd.to_numeric(merged["realgm_player_id"], errors="coerce").astype("Int64"),
            "player_id": pd.to_numeric(merged["player_id"], errors="coerce").astype("Int64"),
            "updated_at": selected_snapshot_ts if selected_snapshot_ts is not None else now,
            "match_method": "team_name",
            "source_snapshot_ts": selected_snapshot_ts if selected_snapshot_ts is not None else pd.NaT,
            "note": "",
        }
    )
    out = out.dropna(subset=["realgm_player_id", "player_id"]).copy()
    out["realgm_player_id"] = out["realgm_player_id"].astype(int)
    out["player_id"] = out["player_id"].astype(int)
    out = out.drop_duplicates(subset=["realgm_player_id"], keep="last").reset_index(drop=True)
    unmatched = int(snapshot_df["realgm_player_id"].nunique() - out["realgm_player_id"].nunique())
    return out, max(0, unmatched)


def refresh_realgm_player_crosswalk_from_minutes(
    minutes_df: pd.DataFrame,
    *,
    data_root: Path,
    as_of_ts: pd.Timestamp | None,
) -> dict[str, Any]:
    """Build/update RealGM->canonical player crosswalk using current minutes frame.

    Deterministic behavior:
    - Uses latest depth snapshot with `scraped_at <= as_of_ts`.
    - Uses one-to-one `(team_id, normalized_name)` matches only.
    - Applies optional manual overrides last.
    """
    try:
        cfg, _cfg_path = load_depth_chart_prior_config(data_root=data_root)
        crosswalk_path = Path(str(cfg.get("crosswalk_path"))).expanduser()
        overrides_path_raw = cfg.get("crosswalk_overrides_path")
        overrides_path = (
            Path(str(overrides_path_raw)).expanduser()
            if overrides_path_raw
            else (Path(data_root) / _DEFAULT_OVERRIDE_REL)
        )

        snapshot_df, snapshot_ts, snapshot_source = _load_snapshot_for_asof(
            data_root=data_root,
            cfg=cfg,
            as_of_ts=as_of_ts,
        )
        snapshot_unique_players = int(snapshot_df["realgm_player_id"].nunique()) if not snapshot_df.empty else 0
        if snapshot_df.empty:
            return {
                "applied": False,
                "reason": "no_snapshot",
                "snapshot_source": snapshot_source,
                "crosswalk_path": str(crosswalk_path),
                "snapshot_unique_players": snapshot_unique_players,
            }

        matched_rows, unmatched_snapshot = _build_team_name_matches(
            snapshot_df=snapshot_df,
            minutes_df=minutes_df,
            selected_snapshot_ts=snapshot_ts,
        )
        overrides = _load_overrides(overrides_path)
        existing = _load_existing_crosswalk(crosswalk_path)

        frames = [f for f in (existing, matched_rows, overrides) if not f.empty]
        if not frames:
            return {
                "applied": False,
                "reason": "no_matches",
                "snapshot_source": snapshot_source,
                "crosswalk_path": str(crosswalk_path),
                "snapshot_unique_players": snapshot_unique_players,
            }
        records: list[dict[str, Any]] = []
        for frame in frames:
            records.extend(frame.to_dict(orient="records"))
        combined = pd.DataFrame.from_records(records, columns=_empty_crosswalk_frame().columns)

        combined["updated_at"] = pd.to_datetime(combined["updated_at"], utc=True, errors="coerce").fillna(
            pd.Timestamp("1970-01-01T00:00:00Z")
        )
        combined["_priority"] = combined["match_method"].astype(str).str.lower().eq("override").astype(int)
        combined = combined.sort_values(
            ["realgm_player_id", "_priority", "updated_at"],
            ascending=[True, True, True],
            kind="mergesort",
        )
        combined = combined.drop_duplicates(subset=["realgm_player_id"], keep="last").reset_index(drop=True)
        combined = combined.drop(columns=["_priority"], errors="ignore")
        _atomic_write_table(combined, crosswalk_path)

        match_rate = (
            float(len(matched_rows) / snapshot_unique_players)
            if snapshot_unique_players > 0
            else 0.0
        )
        return {
            "applied": True,
            "reason": "ok",
            "snapshot_source": snapshot_source,
            "snapshot_ts": snapshot_ts.isoformat().replace("+00:00", "Z") if snapshot_ts is not None else None,
            "crosswalk_path": str(crosswalk_path),
            "rows_written": int(len(combined)),
            "existing_rows": int(len(existing)),
            "matched_rows": int(len(matched_rows)),
            "match_rate": match_rate,
            "snapshot_unique_players": snapshot_unique_players,
            "override_rows": int(len(overrides)),
            "unmatched_snapshot_rows": int(unmatched_snapshot),
            "overrides_path": str(overrides_path),
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("[dc-crosswalk] refresh failed: %s", exc)
        return {
            "applied": False,
            "reason": "error",
            "error": str(exc),
        }


def summarize_crosswalk_json(diag: dict[str, Any]) -> str:
    return json.dumps(diag, sort_keys=True, default=str)


__all__ = [
    "refresh_realgm_player_crosswalk_from_minutes",
    "summarize_crosswalk_json",
]
