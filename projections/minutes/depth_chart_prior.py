from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from projections.paths import get_project_root
from projections.pbp.identity import normalize_name

logger = logging.getLogger(__name__)


ROLE_VALUES: tuple[str, ...] = ("starter", "rotation", "limited", "not_listed")
ROLE_PRIORITY: dict[str, int] = {
    "starter": 0,
    "rotation": 1,
    "limited": 2,
    "not_listed": 3,
}

_DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": True,
    "seed_rotation_prob_from_play_prob": True,
    "apply_to_play_prob": True,
    "play_prob_scale": 0.35,
    "k_role": {
        "starter": 0.70,
        "rotation": 0.20,
        "limited": -0.70,
        "not_listed": -1.10,
    },
    "k_ahead": -0.12,
    "cap_p90": {
        "starter": 42.0,
        "rotation": 34.0,
        "limited": 22.0,
        "not_listed": 12.0,
    },
    "cap_p95": {
        "starter": 44.0,
        "rotation": 37.0,
        "limited": 24.0,
        "not_listed": 14.0,
    },
    "spread_mult": {
        "starter": 1.00,
        "rotation": 0.90,
        "limited": 0.65,
        "not_listed": 0.50,
    },
    "vacancy_col": "vac_min_szn",
    "vacancy_threshold": 48.0,
    "vacancy_slope": 0.010,
    "vacancy_max_relax": 0.35,
    "use_name_fallback": True,
    "top_n_debug": 15,
    "warn_min_match_rate": 0.25,
    "warn_max_snapshot_age_minutes": 360.0,
    "dnp_guardrail_enabled": True,
    "dnp_streak_threshold": 2.0,
    "dnp_rate_threshold": 0.30,
    "dnp_inactive_streak_threshold": 1.0,
    "dnp_k_streak": -0.35,
    "dnp_k_rate": -2.00,
    "dnp_k_inactive_streak": -0.10,
    "dnp_rotation_scale": 1.10,
    "dnp_penalty_min": -1.60,
    "dnp_guardrail_max_p50": 26.0,
    "dnp_require_non_starter": True,
    "dnp_severe_streak_threshold": 6.0,
    "dnp_severe_rate_threshold": 0.60,
    "dnp_severe_max_p50_eligible": 22.0,
    "dnp_severe_cap_p50": 14.0,
    "dnp_severe_cap_p90": 24.0,
    "dnp_severe_cap_p95": 28.0,
    "snapshot_path": None,
    "snapshots_root": None,
    "crosswalk_path": None,
}


_TEAM_ALIASES: dict[str, str] = {
    # Canonical NBA team names.
    "atlanta hawks": "atlanta hawks",
    "boston celtics": "boston celtics",
    "brooklyn nets": "brooklyn nets",
    "charlotte hornets": "charlotte hornets",
    "chicago bulls": "chicago bulls",
    "cleveland cavaliers": "cleveland cavaliers",
    "dallas mavericks": "dallas mavericks",
    "denver nuggets": "denver nuggets",
    "detroit pistons": "detroit pistons",
    "golden state warriors": "golden state warriors",
    "houston rockets": "houston rockets",
    "indiana pacers": "indiana pacers",
    "la clippers": "los angeles clippers",
    "los angeles clippers": "los angeles clippers",
    "los angeles lakers": "los angeles lakers",
    "memphis grizzlies": "memphis grizzlies",
    "miami heat": "miami heat",
    "milwaukee bucks": "milwaukee bucks",
    "minnesota timberwolves": "minnesota timberwolves",
    "new orleans pelicans": "new orleans pelicans",
    "new york knicks": "new york knicks",
    "oklahoma city thunder": "oklahoma city thunder",
    "orlando magic": "orlando magic",
    "philadelphia sixers": "philadelphia sixers",
    "phoenix suns": "phoenix suns",
    "portland trail blazers": "portland trail blazers",
    "sacramento kings": "sacramento kings",
    "san antonio spurs": "san antonio spurs",
    "toronto raptors": "toronto raptors",
    "utah jazz": "utah jazz",
    "washington wizards": "washington wizards",
    # Common shortened names used by minutes pipeline team_name.
    "hawks": "atlanta hawks",
    "celtics": "boston celtics",
    "nets": "brooklyn nets",
    "hornets": "charlotte hornets",
    "bulls": "chicago bulls",
    "cavaliers": "cleveland cavaliers",
    "cavs": "cleveland cavaliers",
    "mavericks": "dallas mavericks",
    "nuggets": "denver nuggets",
    "pistons": "detroit pistons",
    "warriors": "golden state warriors",
    "rockets": "houston rockets",
    "pacers": "indiana pacers",
    "clippers": "los angeles clippers",
    "lakers": "los angeles lakers",
    "grizzlies": "memphis grizzlies",
    "heat": "miami heat",
    "bucks": "milwaukee bucks",
    "timberwolves": "minnesota timberwolves",
    "wolves": "minnesota timberwolves",
    "pelicans": "new orleans pelicans",
    "knicks": "new york knicks",
    "thunder": "oklahoma city thunder",
    "magic": "orlando magic",
    "76ers": "philadelphia sixers",
    "sixers": "philadelphia sixers",
    "suns": "phoenix suns",
    "trail blazers": "portland trail blazers",
    "blazers": "portland trail blazers",
    "kings": "sacramento kings",
    "spurs": "san antonio spurs",
    "raptors": "toronto raptors",
    "jazz": "utah jazz",
    "wizards": "washington wizards",
    # Keep explicit legacy aliases.
    "philadelphia 76ers": "philadelphia sixers",
}


@dataclass(frozen=True, slots=True)
class DepthChartPriorResult:
    frame: pd.DataFrame
    diagnostics: dict[str, Any]


def _normalize_text_key(value: object) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    return _TEAM_ALIASES.get(text, text)


def _role_norm(value: object) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"starter", "starter(s)"}:
        return "starter"
    if raw in {"rotation", "rot", "bench_rotation"}:
        return "rotation"
    if raw in {"limited", "lim", "limited_pt", "limited_pts", "deep_bench"}:
        return "limited"
    return "not_listed"


def _to_numeric_series(series: pd.Series, *, fill: float | None = None) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    if fill is not None:
        out = out.fillna(float(fill))
    return out


def _merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            nested = dict(out[key])
            nested.update(value)
            out[key] = nested
        else:
            out[key] = value
    return out


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Depth chart prior config must be a JSON object: {path}")
    return payload


def load_depth_chart_prior_config(*, data_root: Path) -> tuple[dict[str, Any], Path | None]:
    cfg_path_raw = Path(
        str(
            (
                __import__("os").environ.get("PROJECTIONS_DEPTH_CHART_PRIOR_CONFIG")
                or (get_project_root() / "config" / "depth_chart_prior.json")
            )
        )
    ).expanduser()

    cfg = dict(_DEFAULT_CONFIG)
    loaded_from: Path | None = None
    if cfg_path_raw.exists():
        try:
            cfg = _merge_config(cfg, _load_json(cfg_path_raw))
            loaded_from = cfg_path_raw
        except Exception as exc:  # noqa: BLE001
            logger.warning("[dc-prior] failed to parse config %s (%s); using defaults", cfg_path_raw, exc)

    # Resolve default data paths relative to PROJECTIONS_DATA_ROOT.
    if not cfg.get("snapshots_root"):
        cfg["snapshots_root"] = str((data_root / "bronze" / "realgm" / "depth_charts").resolve())
    if not cfg.get("crosswalk_path"):
        cfg["crosswalk_path"] = str((data_root / "bronze" / "realgm" / "player_id_crosswalk.parquet").resolve())

    cfg["top_n_debug"] = int(max(1, int(cfg.get("top_n_debug", 15))))
    cfg["play_prob_scale"] = float(cfg.get("play_prob_scale", 0.35))
    cfg["k_ahead"] = float(cfg.get("k_ahead", -0.12))
    cfg["vacancy_threshold"] = float(cfg.get("vacancy_threshold", 48.0))
    cfg["vacancy_slope"] = float(cfg.get("vacancy_slope", 0.01))
    cfg["vacancy_max_relax"] = float(cfg.get("vacancy_max_relax", 0.35))
    cfg["warn_min_match_rate"] = float(cfg.get("warn_min_match_rate", 0.25))
    cfg["warn_max_snapshot_age_minutes"] = float(cfg.get("warn_max_snapshot_age_minutes", 360.0))
    cfg["dnp_streak_threshold"] = float(cfg.get("dnp_streak_threshold", 2.0))
    cfg["dnp_rate_threshold"] = float(cfg.get("dnp_rate_threshold", 0.30))
    cfg["dnp_inactive_streak_threshold"] = float(cfg.get("dnp_inactive_streak_threshold", 1.0))
    cfg["dnp_k_streak"] = float(cfg.get("dnp_k_streak", -0.35))
    cfg["dnp_k_rate"] = float(cfg.get("dnp_k_rate", -2.00))
    cfg["dnp_k_inactive_streak"] = float(cfg.get("dnp_k_inactive_streak", -0.10))
    cfg["dnp_rotation_scale"] = float(cfg.get("dnp_rotation_scale", 1.10))
    cfg["dnp_penalty_min"] = float(cfg.get("dnp_penalty_min", -1.60))
    cfg["dnp_guardrail_max_p50"] = float(cfg.get("dnp_guardrail_max_p50", 26.0))
    cfg["dnp_severe_streak_threshold"] = float(cfg.get("dnp_severe_streak_threshold", 6.0))
    cfg["dnp_severe_rate_threshold"] = float(cfg.get("dnp_severe_rate_threshold", 0.60))
    cfg["dnp_severe_max_p50_eligible"] = float(cfg.get("dnp_severe_max_p50_eligible", 22.0))
    cfg["dnp_severe_cap_p50"] = float(cfg.get("dnp_severe_cap_p50", 14.0))
    cfg["dnp_severe_cap_p90"] = float(cfg.get("dnp_severe_cap_p90", 24.0))
    cfg["dnp_severe_cap_p95"] = float(cfg.get("dnp_severe_cap_p95", 28.0))

    return cfg, loaded_from


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table type for depth chart prior: {path}")


def _parse_run_ts_from_name(name: str) -> pd.Timestamp | None:
    token = str(name).strip()
    if token.startswith("run_ts="):
        token = token.split("=", 1)[1].strip()
    if not token:
        return None
    ts = pd.to_datetime(token, utc=True, errors="coerce")
    if ts is None or pd.isna(ts):
        return None
    return pd.Timestamp(ts)


def _extract_run_ts_from_path(path: Path) -> pd.Timestamp | None:
    for part in (path, *path.parents):
        name = part.name
        if not name.startswith("run_ts="):
            continue
        parsed = _parse_run_ts_from_name(name)
        if parsed is not None:
            return parsed
    return None


def _history_snapshot_file_for_asof(
    *,
    data_root: Path,
    cfg: dict[str, Any],
    as_of_ts: pd.Timestamp | None,
) -> Path | None:
    if as_of_ts is None:
        return None
    cutoff = pd.Timestamp(as_of_ts)
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    else:
        cutoff = cutoff.tz_convert("UTC")

    snapshots_root = Path(str(cfg.get("snapshots_root") or "")).expanduser()
    if not snapshots_root.exists() or not snapshots_root.is_dir():
        snapshots_root = data_root / "bronze" / "realgm" / "depth_charts"
    if not snapshots_root.exists() or not snapshots_root.is_dir():
        return None

    best_path: Path | None = None
    best_ts: pd.Timestamp | None = None
    for candidate in snapshots_root.rglob("depth_charts.parquet"):
        if not candidate.parent.name.startswith("run_ts="):
            continue
        run_ts = _extract_run_ts_from_path(candidate)
        if run_ts is None or run_ts > cutoff:
            continue
        if best_ts is None or run_ts > best_ts:
            best_ts = run_ts
            best_path = candidate
    return best_path


def _candidate_snapshot_files(*, data_root: Path, cfg: dict[str, Any]) -> list[Path]:
    candidates: list[Path] = []

    snapshot_path_raw = cfg.get("snapshot_path")
    if snapshot_path_raw:
        p = Path(str(snapshot_path_raw)).expanduser()
        if p.is_file():
            candidates.append(p)
        elif p.is_dir():
            candidates.extend(sorted(p.rglob("*.parquet")))
            candidates.extend(sorted(p.rglob("*.csv")))

    if candidates:
        return candidates

    default_files = [
        data_root / "bronze" / "realgm" / "depth_charts_latest.parquet",
        data_root / "bronze" / "realgm" / "depth_charts.parquet",
        data_root / "bronze" / "realgm" / "depth_charts_latest.csv",
        data_root / "bronze" / "realgm" / "depth_charts.csv",
    ]
    for p in default_files:
        if p.exists() and p.is_file():
            candidates.append(p)

    if candidates:
        return candidates

    snapshots_root = Path(str(cfg.get("snapshots_root") or "")).expanduser()
    if snapshots_root.exists() and snapshots_root.is_dir():
        candidates.extend(sorted(snapshots_root.rglob("*.parquet")))
        candidates.extend(sorted(snapshots_root.rglob("*.csv")))

    return candidates


def _load_snapshot_for_asof(*, data_root: Path, cfg: dict[str, Any], as_of_ts: pd.Timestamp | None) -> tuple[pd.DataFrame, pd.Timestamp | None, str | None]:
    files = _candidate_snapshot_files(data_root=data_root, cfg=cfg)
    if not files:
        return pd.DataFrame(), None, None

    frames: list[pd.DataFrame] = []
    source_path: str | None = None
    for p in files:
        try:
            frame = _read_table(p)
            if frame.empty:
                continue
            frame = frame.copy()
            frame["_dc_source_path"] = str(p)
            frames.append(frame)
            if source_path is None:
                source_path = str(p)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[dc-prior] failed reading snapshot file %s (%s)", p, exc)

    if not frames:
        return pd.DataFrame(), None, source_path

    depth = pd.concat(frames, ignore_index=True)

    required = {"team_name", "player_name", "realgm_player_id", "depth_role", "depth_order", "scraped_at"}
    missing = [c for c in sorted(required) if c not in depth.columns]
    if missing:
        logger.warning("[dc-prior] depth snapshot missing required columns %s", missing)
        return pd.DataFrame(), None, source_path

    depth["scraped_at"] = pd.to_datetime(depth["scraped_at"], utc=True, errors="coerce")
    depth = depth.dropna(subset=["scraped_at"]).copy()
    if depth.empty:
        return pd.DataFrame(), None, source_path

    if as_of_ts is not None:
        filtered = depth.loc[depth["scraped_at"] <= as_of_ts].copy()
        if filtered.empty:
            history_path = _history_snapshot_file_for_asof(
                data_root=data_root,
                cfg=cfg,
                as_of_ts=as_of_ts,
            )
            if history_path is None:
                return pd.DataFrame(), None, source_path
            try:
                history = _read_table(history_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[dc-prior] failed reading history snapshot %s (%s)", history_path, exc)
                return pd.DataFrame(), None, source_path
            if history.empty:
                return pd.DataFrame(), None, source_path
            history = history.copy()
            history["scraped_at"] = pd.to_datetime(history["scraped_at"], utc=True, errors="coerce")
            history = history.dropna(subset=["scraped_at"]).copy()
            history = history.loc[history["scraped_at"] <= as_of_ts].copy()
            if history.empty:
                return pd.DataFrame(), None, source_path
            history["_dc_source_path"] = str(history_path)
            depth = history
            source_path = str(history_path)
        else:
            depth = filtered

    selected_ts = pd.to_datetime(depth["scraped_at"], utc=True, errors="coerce").max()
    if pd.isna(selected_ts):
        return pd.DataFrame(), None, source_path

    snapshot = depth.loc[depth["scraped_at"] == selected_ts].copy()
    snapshot["realgm_player_id"] = pd.to_numeric(snapshot["realgm_player_id"], errors="coerce").astype("Int64")
    snapshot = snapshot.dropna(subset=["realgm_player_id"]).copy()
    snapshot["depth_role"] = snapshot["depth_role"].map(_role_norm)
    snapshot["depth_order"] = pd.to_numeric(snapshot["depth_order"], errors="coerce")
    snapshot["depth_order"] = snapshot["depth_order"].fillna(0).astype(int)
    snapshot["_team_key"] = snapshot["team_name"].map(_normalize_text_key)

    # Deterministic row ordering for stable tie-breaking.
    snapshot["_role_priority"] = snapshot["depth_role"].map(ROLE_PRIORITY).fillna(ROLE_PRIORITY["not_listed"]).astype(int)
    snapshot = (
        snapshot.sort_values(
            ["realgm_player_id", "_role_priority", "depth_order", "player_name"],
            kind="mergesort",
        )
        .drop_duplicates(subset=["realgm_player_id"], keep="first")
        .reset_index(drop=True)
    )

    return snapshot, pd.Timestamp(selected_ts), source_path


def _load_crosswalk(*, data_root: Path, cfg: dict[str, Any]) -> tuple[pd.DataFrame, str | None]:
    path_raw = cfg.get("crosswalk_path")
    if not path_raw:
        return pd.DataFrame(columns=["realgm_player_id", "player_id"]), None

    path = Path(str(path_raw)).expanduser()
    if not path.exists() or not path.is_file():
        return pd.DataFrame(columns=["realgm_player_id", "player_id"]), str(path)

    try:
        cross = _read_table(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[dc-prior] failed reading crosswalk %s (%s)", path, exc)
        return pd.DataFrame(columns=["realgm_player_id", "player_id"]), str(path)

    required = {"realgm_player_id", "player_id"}
    missing = [c for c in sorted(required) if c not in cross.columns]
    if missing:
        logger.warning("[dc-prior] crosswalk missing required columns %s", missing)
        return pd.DataFrame(columns=["realgm_player_id", "player_id"]), str(path)

    cross = cross.copy()
    cross["realgm_player_id"] = pd.to_numeric(cross["realgm_player_id"], errors="coerce").astype("Int64")
    cross["player_id"] = pd.to_numeric(cross["player_id"], errors="coerce").astype("Int64")
    cross = cross.dropna(subset=["realgm_player_id", "player_id"]).copy()

    if "updated_at" in cross.columns:
        cross["updated_at"] = pd.to_datetime(cross["updated_at"], utc=True, errors="coerce")
        cross = cross.sort_values(["realgm_player_id", "updated_at"], kind="mergesort")
    else:
        cross = cross.sort_values(["realgm_player_id"], kind="mergesort")

    cross = cross.drop_duplicates(subset=["realgm_player_id"], keep="last")
    return cross[["realgm_player_id", "player_id"]].copy(), str(path)


def _build_team_lookup(minutes_df: pd.DataFrame) -> dict[str, int]:
    if "team_name" not in minutes_df.columns or "team_id" not in minutes_df.columns:
        return {}
    lookup_df = minutes_df[["team_id", "team_name"]].dropna(subset=["team_id", "team_name"]).drop_duplicates()
    if lookup_df.empty:
        return {}

    out: dict[str, int] = {}
    for row in lookup_df.itertuples(index=False):
        key = _normalize_text_key(getattr(row, "team_name"))
        if not key:
            continue
        try:
            out[key] = int(getattr(row, "team_id"))
        except Exception:
            continue
    return out


def _assign_team_ids_from_name(depth_df: pd.DataFrame, minutes_df: pd.DataFrame) -> pd.DataFrame:
    if depth_df.empty:
        return depth_df
    out = depth_df.copy()
    if "team_id" in out.columns:
        out["team_id"] = pd.to_numeric(out["team_id"], errors="coerce").astype("Int64")

    lookup = _build_team_lookup(minutes_df)
    if not lookup:
        if "team_id" not in out.columns:
            out["team_id"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
        return out

    mapped = out.get("_team_key", out.get("team_name", pd.Series("", index=out.index))).map(lambda x: lookup.get(_normalize_text_key(x)))
    mapped = pd.to_numeric(mapped, errors="coerce").astype("Int64")

    if "team_id" in out.columns:
        out["team_id"] = out["team_id"].fillna(mapped)
    else:
        out["team_id"] = mapped
    return out


def _eligible_name_fallback(depth_df: pd.DataFrame, minutes_df: pd.DataFrame) -> pd.DataFrame:
    required_depth = {"team_id", "player_name", "depth_role", "depth_order", "scraped_at"}
    required_minutes = {"team_id", "player_name", "player_id"}
    if not required_depth.issubset(depth_df.columns) or not required_minutes.issubset(minutes_df.columns):
        return pd.DataFrame(columns=["player_id", "depth_role", "depth_order", "team_id", "scraped_at", "_dc_join_source"])

    d = depth_df.copy()
    m = minutes_df.copy()

    d["team_id"] = pd.to_numeric(d["team_id"], errors="coerce").astype("Int64")
    m["team_id"] = pd.to_numeric(m["team_id"], errors="coerce").astype("Int64")
    m["player_id"] = pd.to_numeric(m["player_id"], errors="coerce").astype("Int64")

    d = d.dropna(subset=["team_id", "player_name"]).copy()
    m = m.dropna(subset=["team_id", "player_name", "player_id"]).copy()
    if d.empty or m.empty:
        return pd.DataFrame(columns=["player_id", "depth_role", "depth_order", "team_id", "scraped_at", "_dc_join_source"])

    d["_name_key"] = d["player_name"].map(normalize_name)
    m["_name_key"] = m["player_name"].map(normalize_name)
    d = d[d["_name_key"] != ""].copy()
    m = m[m["_name_key"] != ""].copy()
    if d.empty or m.empty:
        return pd.DataFrame(columns=["player_id", "depth_role", "depth_order", "team_id", "scraped_at", "_dc_join_source"])

    d_counts = d.groupby(["team_id", "_name_key"], dropna=False).size().rename("_n_d").reset_index()
    m_counts = m.groupby(["team_id", "_name_key"], dropna=False).size().rename("_n_m").reset_index()

    d = d.merge(d_counts, on=["team_id", "_name_key"], how="left")
    m = m.merge(m_counts, on=["team_id", "_name_key"], how="left")
    d = d[d["_n_d"] == 1].copy()
    m = m[m["_n_m"] == 1].copy()

    if d.empty or m.empty:
        return pd.DataFrame(columns=["player_id", "depth_role", "depth_order", "team_id", "scraped_at", "_dc_join_source"])

    merged = m.merge(
        d[["team_id", "_name_key", "depth_role", "depth_order", "scraped_at"]],
        on=["team_id", "_name_key"],
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame(columns=["player_id", "depth_role", "depth_order", "team_id", "scraped_at", "_dc_join_source"])

    merged = merged[["player_id", "team_id", "depth_role", "depth_order", "scraped_at"]].copy()
    merged["_dc_join_source"] = "name_fallback"
    return merged


def _attach_depth_view(minutes_df: pd.DataFrame, snapshot_df: pd.DataFrame, crosswalk_df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    work = minutes_df.copy()
    if "player_id" not in work.columns:
        return work, {"applied": False, "reason": "missing_player_id"}

    work["player_id"] = pd.to_numeric(work["player_id"], errors="coerce").astype("Int64")
    work["team_id"] = pd.to_numeric(work.get("team_id"), errors="coerce").astype("Int64") if "team_id" in work.columns else pd.Series(pd.NA, index=work.index, dtype="Int64")

    if snapshot_df.empty or crosswalk_df.empty:
        # Populate defaults for downstream no-op consistency.
        work["dc_present"] = False
        work["dc_role"] = "not_listed"
        work["dc_role_priority"] = int(ROLE_PRIORITY["not_listed"])
        work["dc_order_in_role"] = pd.Series(pd.NA, index=work.index, dtype="Int64")
        work["dc_ahead_global"] = 0
        work["dc_is_primary_backup"] = False
        return work, {
            "applied": False,
            "reason": "missing_snapshot_or_crosswalk",
            "matched_id": 0,
            "matched_name_fallback": 0,
            "unmatched": int(len(work)),
        }

    depth = snapshot_df.copy()
    depth = depth.merge(crosswalk_df, on="realgm_player_id", how="inner", suffixes=("", "_cw"))
    if depth.empty:
        work["dc_present"] = False
        work["dc_role"] = "not_listed"
        work["dc_role_priority"] = int(ROLE_PRIORITY["not_listed"])
        work["dc_order_in_role"] = pd.Series(pd.NA, index=work.index, dtype="Int64")
        work["dc_ahead_global"] = 0
        work["dc_is_primary_backup"] = False
        return work, {
            "applied": False,
            "reason": "no_crosswalk_matches",
            "matched_id": 0,
            "matched_name_fallback": 0,
            "unmatched": int(len(work)),
        }

    depth["player_id"] = pd.to_numeric(depth["player_id"], errors="coerce").astype("Int64")
    depth = depth.dropna(subset=["player_id"]).copy()
    depth = _assign_team_ids_from_name(depth, work)

    depth["dc_role"] = depth["depth_role"].map(_role_norm)
    depth["dc_role_priority"] = depth["dc_role"].map(ROLE_PRIORITY).fillna(ROLE_PRIORITY["not_listed"]).astype(int)
    depth["dc_order_in_role"] = pd.to_numeric(depth["depth_order"], errors="coerce").astype("Int64")

    depth = (
        depth.sort_values(["player_id", "dc_role_priority", "dc_order_in_role", "player_name"], kind="mergesort")
        .drop_duplicates(subset=["player_id"], keep="first")
        .reset_index(drop=True)
    )
    depth["_dc_join_source"] = "id"

    keep_cols = ["player_id", "team_id", "dc_role", "dc_role_priority", "dc_order_in_role", "scraped_at", "_dc_join_source"]
    merged = work.merge(depth[keep_cols], on="player_id", how="left", suffixes=("", "_dc"))

    # Keep only team-consistent matches when team_id is known in both sources.
    team_mismatch = (
        merged["team_id_dc"].notna()
        & merged["team_id"].notna()
        & (merged["team_id_dc"].astype("Int64") != merged["team_id"].astype("Int64"))
    )
    if team_mismatch.any():
        cols_reset = ["dc_role", "dc_role_priority", "dc_order_in_role", "scraped_at", "_dc_join_source"]
        for c in cols_reset:
            if c in merged.columns:
                merged.loc[team_mismatch, c] = pd.NA

    matched_id_mask = merged["_dc_join_source"].astype("string").fillna("").eq("id")

    matched_name_fallback = 0
    if bool(cfg.get("use_name_fallback", True)):
        fallback = _eligible_name_fallback(snapshot_df, merged)
        if not fallback.empty:
            fallback = fallback.rename(columns={"team_id": "team_id_fb"})
            merged = merged.merge(fallback, on="player_id", how="left", suffixes=("", "_fb"))

            needs_fb = merged["_dc_join_source"].isna()
            if "team_id_fb" in merged.columns and "team_id" in merged.columns:
                same_team = (
                    merged["team_id_fb"].isna()
                    | merged["team_id"].isna()
                    | (merged["team_id_fb"].astype("Int64") == merged["team_id"].astype("Int64"))
                )
                needs_fb = needs_fb & same_team

            if needs_fb.any():
                merged.loc[needs_fb, "dc_role"] = merged.loc[needs_fb, "depth_role"].map(_role_norm)
                merged.loc[needs_fb, "dc_role_priority"] = (
                    merged.loc[needs_fb, "dc_role"].map(ROLE_PRIORITY).fillna(ROLE_PRIORITY["not_listed"]).astype(int)
                )
                merged.loc[needs_fb, "dc_order_in_role"] = pd.to_numeric(
                    merged.loc[needs_fb, "depth_order"], errors="coerce"
                ).astype("Int64")
                merged.loc[needs_fb, "scraped_at"] = merged.loc[needs_fb, "scraped_at_fb"]
                merged.loc[needs_fb, "_dc_join_source"] = "name_fallback"
                matched_name_fallback = int(needs_fb.sum())

            merged = merged.drop(
                columns=[c for c in ("team_id_fb", "depth_role", "depth_order", "scraped_at_fb") if c in merged.columns],
                errors="ignore",
            )

    merged["dc_present"] = merged["_dc_join_source"].notna()
    merged["dc_role"] = merged["dc_role"].map(_role_norm)
    merged.loc[~merged["dc_present"], "dc_role"] = "not_listed"
    merged["dc_role_priority"] = (
        merged["dc_role_priority"].where(merged["dc_present"], ROLE_PRIORITY["not_listed"]).fillna(ROLE_PRIORITY["not_listed"]).astype(int)
    )
    merged["dc_order_in_role"] = pd.to_numeric(merged["dc_order_in_role"], errors="coerce").astype("Int64")

    # Positionless buriedness count by team: sort listed players by role/order.
    merged["dc_ahead_global"] = 0
    if {"game_id", "team_id"}.issubset(merged.columns):
        for _, idx in merged.groupby(["game_id", "team_id"], dropna=False, sort=False).groups.items():
            idx_arr = np.array(list(idx), dtype=int)
            if idx_arr.size == 0:
                continue
            g = merged.iloc[idx_arr].copy()
            listed = g["dc_present"].astype(bool).to_numpy()
            n_listed = int(listed.sum())
            ahead = np.full(len(g), n_listed, dtype=int)
            if n_listed > 0:
                listed_idx = np.flatnonzero(listed)
                listed_df = g.iloc[listed_idx].copy()
                listed_df["_ord"] = pd.to_numeric(listed_df["dc_order_in_role"], errors="coerce").fillna(999).astype(int)
                listed_df = listed_df.sort_values(["dc_role_priority", "_ord", "player_id"], kind="mergesort")
                for rank, ridx in enumerate(listed_df.index.tolist()):
                    local_pos = int(np.where(g.index.to_numpy() == ridx)[0][0])
                    ahead[local_pos] = rank
            merged.loc[g.index, "dc_ahead_global"] = ahead

    merged["dc_is_primary_backup"] = (
        merged["dc_present"].astype(bool)
        & merged["dc_role"].eq("rotation")
        & pd.to_numeric(merged["dc_order_in_role"], errors="coerce").fillna(-1).eq(0)
    )

    unmatched = int((~merged["dc_present"].astype(bool)).sum())
    out_diag = {
        "applied": True,
        "reason": "ok",
        "matched_id": int(matched_id_mask.sum()),
        "matched_name_fallback": int(matched_name_fallback),
        "unmatched": unmatched,
        "team_mismatch_dropped": int(team_mismatch.sum()) if "team_mismatch" in locals() else 0,
    }
    return merged, out_diag


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: np.ndarray, *, eps: float = 1e-6) -> np.ndarray:
    p2 = np.clip(p, eps, 1.0 - eps)
    return np.log(p2 / (1.0 - p2))


def _vacancy_relax_factor(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)

    vac_col = str(cfg.get("vacancy_col") or "").strip()
    if vac_col and vac_col in df.columns:
        vac = pd.to_numeric(df[vac_col], errors="coerce").fillna(0.0)
    else:
        status = df.get("status", pd.Series("", index=df.index)).astype(str).str.strip().str.lower()
        out_like = status.eq("out") | status.str.contains("inactive") | status.str.contains("susp")
        if {"game_id", "team_id"}.issubset(df.columns):
            counts = out_like.groupby([df["game_id"], df["team_id"]], sort=False).transform("sum")
            vac = counts.astype(float) * 24.0
        else:
            vac = out_like.astype(float) * 24.0

    threshold = float(cfg.get("vacancy_threshold", 48.0))
    slope = float(cfg.get("vacancy_slope", 0.01))
    max_relax = float(cfg.get("vacancy_max_relax", 0.35))
    relax_add = np.clip((vac.to_numpy(dtype=float) - threshold) * slope, 0.0, max_relax)
    return pd.Series(1.0 + relax_add, index=df.index, dtype=float)


def _forced_inactive_mask(df: pd.DataFrame) -> pd.Series:
    if df.empty or "status" not in df.columns:
        return pd.Series(False, index=df.index)
    status = df["status"].astype(str).str.strip().str.lower()
    return (
        status.eq("out")
        | status.eq("inactive")
        | status.str.contains("dnp", regex=False)
        | status.str.contains("susp", regex=False)
    )


def _apply_tail_shaping(df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = df.copy()
    if "minutes_p50" not in out.columns:
        return out, {"cap_hits_by_role": {}, "largest_reductions": []}

    p50 = pd.to_numeric(out["minutes_p50"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    role = out["dc_role"].astype(str).str.strip().str.lower()
    relax = _vacancy_relax_factor(out, cfg).to_numpy(dtype=float)

    spread_mult_map = {k: float(v) for k, v in dict(cfg.get("spread_mult") or {}).items()}
    cap_p90_map = {k: float(v) for k, v in dict(cfg.get("cap_p90") or {}).items()}
    cap_p95_map = {k: float(v) for k, v in dict(cfg.get("cap_p95") or {}).items()}

    cap_hits_by_role: dict[str, int] = {r: 0 for r in ROLE_VALUES}
    largest_reductions: list[dict[str, Any]] = []

    def _shape_col(col: str, cap_map: dict[str, float]) -> None:
        nonlocal largest_reductions
        if col not in out.columns:
            return
        raw = pd.to_numeric(out[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        spread0 = np.maximum(raw - p50, 0.0)
        mult = np.array([spread_mult_map.get(r, spread_mult_map.get("not_listed", 0.5)) for r in role], dtype=float)
        spread1 = spread0 * np.clip(mult, 0.0, 2.0)
        candidate = p50 + spread1

        cap = np.array([cap_map.get(r, cap_map.get("not_listed", 12.0)) for r in role], dtype=float)
        cap_eff = cap * relax
        shaped = np.minimum(candidate, cap_eff)
        shaped = np.maximum(shaped, p50)
        shaped = np.clip(shaped, 0.0, 48.0)

        hit_mask = candidate > (cap_eff + 1e-9)
        if hit_mask.any():
            role_vals = role.to_numpy(dtype=object)
            for r in ROLE_VALUES:
                cap_hits_by_role[r] = int(cap_hits_by_role.get(r, 0) + np.sum(hit_mask & (role_vals == r)))

        reduction = np.maximum(raw - shaped, 0.0)
        if np.any(reduction > 0):
            idx_top = np.argsort(-reduction)[: int(cfg.get("top_n_debug", 15))]
            for idx in idx_top:
                if reduction[idx] <= 0:
                    continue
                largest_reductions.append(
                    {
                        "player_id": int(pd.to_numeric(out.iloc[idx].get("player_id"), errors="coerce") or -1),
                        "player_name": str(out.iloc[idx].get("player_name") or ""),
                        "team_id": int(pd.to_numeric(out.iloc[idx].get("team_id"), errors="coerce") or -1),
                        "col": col,
                        "before": float(raw[idx]),
                        "after": float(shaped[idx]),
                        "reduction": float(reduction[idx]),
                        "dc_role": str(role.iloc[idx]),
                    }
                )

        out[col] = shaped

    _shape_col("minutes_p90", cap_p90_map)
    _shape_col("minutes_p95", cap_p95_map)

    if "minutes_p10" in out.columns:
        q10 = pd.to_numeric(out["minutes_p10"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        q10 = np.minimum(q10, p50)
        out["minutes_p10"] = np.clip(q10, 0.0, 48.0)

    # Monotone repair.
    q50 = pd.to_numeric(out["minutes_p50"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if "minutes_p10" in out.columns:
        q10 = pd.to_numeric(out["minutes_p10"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        q10 = np.minimum(q10, q50)
        q50 = np.maximum(q50, q10)
        out["minutes_p10"] = q10
        out["minutes_p50"] = q50

    if "minutes_p90" in out.columns:
        q90 = pd.to_numeric(out["minutes_p90"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        q90 = np.maximum(q90, q50)
        out["minutes_p90"] = np.clip(q90, 0.0, 48.0)

    if "minutes_p95" in out.columns:
        q95 = pd.to_numeric(out["minutes_p95"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if "minutes_p90" in out.columns:
            q90 = pd.to_numeric(out["minutes_p90"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            q95 = np.maximum(q95, q90)
        q95 = np.maximum(q95, q50)
        out["minutes_p95"] = np.clip(q95, 0.0, 48.0)

    # Keep conditional aliases in sync when present.
    for base in ("minutes_p10", "minutes_p50", "minutes_p90", "minutes_p95"):
        cond = f"{base}_cond"
        if base in out.columns and cond in out.columns:
            out[cond] = out[base]

    largest_reductions = sorted(largest_reductions, key=lambda x: x.get("reduction", 0.0), reverse=True)
    largest_reductions = largest_reductions[: int(cfg.get("top_n_debug", 15))]

    return out, {
        "cap_hits_by_role": {k: int(v) for k, v in cap_hits_by_role.items() if int(v) > 0},
        "largest_reductions": largest_reductions,
    }


def _top_probability_deltas(df: pd.DataFrame, *, col_pre: str, col_post: str, top_n: int) -> list[dict[str, Any]]:
    if col_pre not in df.columns or col_post not in df.columns:
        return []

    pre = pd.to_numeric(df[col_pre], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    post = pd.to_numeric(df[col_post], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    delta = post - pre
    if not np.any(np.abs(delta) > 1e-12):
        return []

    idx_top = np.argsort(-np.abs(delta))[:top_n]
    out: list[dict[str, Any]] = []
    for idx in idx_top:
        out.append(
            {
                "player_id": int(pd.to_numeric(df.iloc[idx].get("player_id"), errors="coerce") or -1),
                "player_name": str(df.iloc[idx].get("player_name") or ""),
                "team_id": int(pd.to_numeric(df.iloc[idx].get("team_id"), errors="coerce") or -1),
                "dc_role": str(df.iloc[idx].get("dc_role") or "not_listed"),
                "dc_ahead_global": int(pd.to_numeric(df.iloc[idx].get("dc_ahead_global"), errors="coerce") or 0),
                "before": float(pre[idx]),
                "after": float(post[idx]),
                "delta": float(delta[idx]),
            }
        )
    return out


def _apply_dnp_guardrail(df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = df.copy()
    if out.empty:
        return out, {"enabled": bool(cfg.get("dnp_guardrail_enabled", True)), "applied": False, "reason": "empty"}
    if not bool(cfg.get("dnp_guardrail_enabled", True)):
        return out, {"enabled": False, "applied": False, "reason": "disabled"}

    def _as_numeric_feature(name: str) -> np.ndarray:
        if name in out.columns:
            series = pd.to_numeric(out[name], errors="coerce").fillna(0.0)
        else:
            series = pd.Series(0.0, index=out.index, dtype=float)
        return series.to_numpy(dtype=float)

    streak = _as_numeric_feature("consecutive_active_dnp")
    dnp_rate = _as_numeric_feature("active_but_dnp_rate_last10")
    inactive_streak = _as_numeric_feature("inactive_streak_len")

    streak_signal = np.maximum(streak - float(cfg.get("dnp_streak_threshold", 2.0)), 0.0)
    rate_signal = np.maximum(dnp_rate - float(cfg.get("dnp_rate_threshold", 0.30)), 0.0)
    inactive_signal = np.maximum(
        inactive_streak - float(cfg.get("dnp_inactive_streak_threshold", 1.0)),
        0.0,
    )
    penalty = (
        float(cfg.get("dnp_k_streak", -0.35)) * streak_signal
        + float(cfg.get("dnp_k_rate", -2.00)) * rate_signal
        + float(cfg.get("dnp_k_inactive_streak", -0.10)) * inactive_signal
    )
    penalty = np.maximum(penalty, float(cfg.get("dnp_penalty_min", -1.60)))
    forced_inactive = _forced_inactive_mask(out).to_numpy(dtype=bool)
    target = (streak_signal > 0.0) | (rate_signal > 0.0) | (inactive_signal > 0.0)

    p50_for_gate = pd.to_numeric(out.get("minutes_p50"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    gate_by_minutes = p50_for_gate <= float(cfg.get("dnp_guardrail_max_p50", 26.0))
    starter_mask = np.zeros(len(out), dtype=bool)
    for col in ("is_confirmed_starter", "is_projected_starter", "is_starter", "starter_flag"):
        if col in out.columns:
            v = pd.to_numeric(out[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            starter_mask = starter_mask | (v >= 0.5)
    if bool(cfg.get("dnp_require_non_starter", True)):
        gate_by_role = ~starter_mask
    else:
        gate_by_role = np.ones(len(out), dtype=bool)

    target = target & (~forced_inactive) & gate_by_minutes & gate_by_role

    diag: dict[str, Any] = {
        "enabled": True,
        "applied": bool(np.any(target)),
        "n_flagged": int(np.sum(target)),
        "n_adjusted_play_prob": 0,
        "n_adjusted_rotation_prob": 0,
        "n_severe_capped": 0,
        "top_play_prob_deltas": [],
        "top_rotation_prob_deltas": [],
    }
    if not np.any(target):
        diag["reason"] = "no_dnp_signal"
        return out, diag

    if "play_prob" in out.columns:
        pre = pd.to_numeric(out["play_prob"], errors="coerce").fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float)
        post = pre.copy()
        post[target] = _sigmoid(_logit(pre[target]) + penalty[target])
        post = np.clip(post, 0.0, 1.0)
        out["play_prob"] = post
        out["play_prob_pre_dnp"] = pre
        diag["n_adjusted_play_prob"] = int(np.sum(np.abs(post - pre) > 1e-12))

    if "rotation_prob" in out.columns:
        rot_pre = pd.to_numeric(out["rotation_prob"], errors="coerce").fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float)
        rot_post = rot_pre.copy()
        rot_penalty = penalty * float(cfg.get("dnp_rotation_scale", 1.10))
        rot_post[target] = _sigmoid(_logit(rot_pre[target]) + rot_penalty[target])
        rot_post = np.clip(rot_post, 0.0, 1.0)
        out["rotation_prob"] = rot_post
        out["rotation_prob_pre_dnp"] = rot_pre
        diag["n_adjusted_rotation_prob"] = int(np.sum(np.abs(rot_post - rot_pre) > 1e-12))

    severe_mask = (
        (streak >= float(cfg.get("dnp_severe_streak_threshold", 6.0)))
        | (dnp_rate >= float(cfg.get("dnp_severe_rate_threshold", 0.60)))
    ) & target & (p50_for_gate <= float(cfg.get("dnp_severe_max_p50_eligible", 22.0)))
    diag["n_severe_capped"] = int(np.sum(severe_mask))
    if np.any(severe_mask):
        cap_p50 = float(cfg.get("dnp_severe_cap_p50", 14.0))
        cap_p90 = float(cfg.get("dnp_severe_cap_p90", 24.0))
        cap_p95 = float(cfg.get("dnp_severe_cap_p95", 28.0))

        if "minutes_p50" in out.columns:
            q50 = pd.to_numeric(out["minutes_p50"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            q50[severe_mask] = np.minimum(q50[severe_mask], cap_p50)
            out["minutes_p50"] = np.clip(q50, 0.0, 48.0)
        if "minutes_p90" in out.columns:
            q90 = pd.to_numeric(out["minutes_p90"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            q90[severe_mask] = np.minimum(q90[severe_mask], cap_p90)
            out["minutes_p90"] = np.clip(q90, 0.0, 48.0)
        if "minutes_p95" in out.columns:
            q95 = pd.to_numeric(out["minutes_p95"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            q95[severe_mask] = np.minimum(q95[severe_mask], cap_p95)
            out["minutes_p95"] = np.clip(q95, 0.0, 48.0)

        # Keep basic quantile monotonicity for severe-capped rows.
        if {"minutes_p10", "minutes_p50"}.issubset(out.columns):
            q10 = pd.to_numeric(out["minutes_p10"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            q50 = pd.to_numeric(out["minutes_p50"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            q10 = np.minimum(q10, q50)
            out["minutes_p10"] = np.clip(q10, 0.0, 48.0)
        if {"minutes_p50", "minutes_p90"}.issubset(out.columns):
            q50 = pd.to_numeric(out["minutes_p50"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            q90 = pd.to_numeric(out["minutes_p90"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            q90 = np.maximum(q90, q50)
            out["minutes_p90"] = np.clip(q90, 0.0, 48.0)
        if {"minutes_p90", "minutes_p95"}.issubset(out.columns):
            q90 = pd.to_numeric(out["minutes_p90"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            q95 = pd.to_numeric(out["minutes_p95"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            q95 = np.maximum(q95, q90)
            out["minutes_p95"] = np.clip(q95, 0.0, 48.0)

        for base in ("minutes_p10", "minutes_p50", "minutes_p90", "minutes_p95"):
            cond = f"{base}_cond"
            if base in out.columns and cond in out.columns:
                out[cond] = out[base]

    if "play_prob_pre_dnp" in out.columns:
        diag["top_play_prob_deltas"] = _top_probability_deltas(
            out,
            col_pre="play_prob_pre_dnp",
            col_post="play_prob",
            top_n=int(cfg.get("top_n_debug", 15)),
        )
    if "rotation_prob_pre_dnp" in out.columns:
        diag["top_rotation_prob_deltas"] = _top_probability_deltas(
            out,
            col_pre="rotation_prob_pre_dnp",
            col_post="rotation_prob",
            top_n=int(cfg.get("top_n_debug", 15)),
        )

    out = out.drop(columns=["play_prob_pre_dnp", "rotation_prob_pre_dnp"], errors="ignore")
    return out, diag


def _disagreement_rows(df: pd.DataFrame, *, top_n: int) -> list[dict[str, Any]]:
    if "rotation_prob_pre" not in df.columns:
        return []

    expected = {
        "starter": 0.95,
        "rotation": 0.70,
        "limited": 0.30,
        "not_listed": 0.08,
    }
    model = pd.to_numeric(df["rotation_prob_pre"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    role = df["dc_role"].astype(str).str.strip().str.lower().to_numpy(dtype=object)
    exp = np.array([expected.get(str(r), expected["not_listed"]) for r in role], dtype=float)
    score = np.abs(model - exp)
    if not np.any(score > 0.0):
        return []

    idx_top = np.argsort(-score)[:top_n]
    rows: list[dict[str, Any]] = []
    for idx in idx_top:
        rows.append(
            {
                "player_id": int(pd.to_numeric(df.iloc[idx].get("player_id"), errors="coerce") or -1),
                "player_name": str(df.iloc[idx].get("player_name") or ""),
                "team_id": int(pd.to_numeric(df.iloc[idx].get("team_id"), errors="coerce") or -1),
                "dc_role": str(df.iloc[idx].get("dc_role") or "not_listed"),
                "rotation_prob_model": float(model[idx]),
                "rotation_prob_expected_from_role": float(exp[idx]),
                "abs_gap": float(score[idx]),
            }
        )
    return rows


def apply_depth_chart_prior_from_realgm(
    minutes_df: pd.DataFrame,
    *,
    data_root: Path,
    as_of_ts: pd.Timestamp | None,
) -> DepthChartPriorResult:
    """Apply RealGM depth-chart priors to membership and tail support.

    This function is inference-only and deterministic. If any required input
    is missing it degrades to a no-op and returns diagnostics describing why.
    """

    if minutes_df.empty:
        return DepthChartPriorResult(
            frame=minutes_df,
            diagnostics={
                "enabled": False,
                "applied": False,
                "reason": "empty_minutes_frame",
            },
        )

    cfg, cfg_path = load_depth_chart_prior_config(data_root=data_root)
    as_of_norm: pd.Timestamp | None = None
    if as_of_ts is not None:
        as_of_norm = pd.Timestamp(as_of_ts)
        if as_of_norm.tzinfo is None:
            as_of_norm = as_of_norm.tz_localize("UTC")
        else:
            as_of_norm = as_of_norm.tz_convert("UTC")
    if not bool(cfg.get("enabled", False)):
        return DepthChartPriorResult(
            frame=minutes_df,
            diagnostics={
                "enabled": False,
                "applied": False,
                "reason": "disabled",
                "config_path": str(cfg_path) if cfg_path is not None else None,
            },
        )

    snapshot_df, selected_ts, snapshot_source = _load_snapshot_for_asof(
        data_root=data_root,
        cfg=cfg,
        as_of_ts=as_of_ts,
    )
    crosswalk_df, crosswalk_source = _load_crosswalk(data_root=data_root, cfg=cfg)

    joined, attach_diag = _attach_depth_view(minutes_df, snapshot_df, crosswalk_df, cfg)
    joined, dnp_diag = _apply_dnp_guardrail(joined, cfg)

    diagnostics: dict[str, Any] = {
        "enabled": True,
        "applied": bool(attach_diag.get("applied", False)),
        "reason": str(attach_diag.get("reason", "unknown")),
        "config_path": str(cfg_path) if cfg_path is not None else None,
        "snapshot_source": snapshot_source,
        "crosswalk_source": crosswalk_source,
        "dc_snapshot_ts": selected_ts.isoformat().replace("+00:00", "Z") if selected_ts is not None else None,
        "matched_id": int(attach_diag.get("matched_id", 0)),
        "matched_name_fallback": int(attach_diag.get("matched_name_fallback", 0)),
        "unmatched": int(attach_diag.get("unmatched", len(joined))),
        "team_mismatch_dropped": int(attach_diag.get("team_mismatch_dropped", 0)),
        "dnp_guardrail": dnp_diag,
    }
    players_total = int(len(joined))
    matched_total = int(diagnostics["matched_id"]) + int(diagnostics["matched_name_fallback"])
    matched_rate = float(matched_total / players_total) if players_total > 0 else 0.0
    diagnostics["players_total"] = players_total
    diagnostics["matched_total"] = matched_total
    diagnostics["matched_rate"] = matched_rate
    diagnostics["warn_min_match_rate"] = float(cfg.get("warn_min_match_rate", 0.25))
    diagnostics["warn_max_snapshot_age_minutes"] = float(cfg.get("warn_max_snapshot_age_minutes", 360.0))
    if selected_ts is not None and as_of_norm is not None:
        age_minutes = float((as_of_norm - selected_ts).total_seconds() / 60.0)
        diagnostics["snapshot_age_minutes"] = max(0.0, age_minutes)
    else:
        diagnostics["snapshot_age_minutes"] = None

    if not bool(attach_diag.get("applied", False)):
        # Ensure derived fields exist for contract stability.
        out = joined.copy()
        if "dc_present" not in out.columns:
            out["dc_present"] = False
        if "dc_role" not in out.columns:
            out["dc_role"] = "not_listed"
        if "dc_role_priority" not in out.columns:
            out["dc_role_priority"] = ROLE_PRIORITY["not_listed"]
        if "dc_order_in_role" not in out.columns:
            out["dc_order_in_role"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
        if "dc_ahead_global" not in out.columns:
            out["dc_ahead_global"] = 0
        if "dc_is_primary_backup" not in out.columns:
            out["dc_is_primary_backup"] = False
        if "dc_snapshot_ts" not in out.columns:
            out["dc_snapshot_ts"] = pd.NA
        alerts = ["prior_not_applied"]
        if float(diagnostics.get("matched_rate", 0.0)) < float(diagnostics.get("warn_min_match_rate", 0.25)):
            alerts.append("low_match_rate")
        snap_age = diagnostics.get("snapshot_age_minutes")
        if snap_age is not None and float(snap_age) > float(diagnostics.get("warn_max_snapshot_age_minutes", 360.0)):
            alerts.append("stale_snapshot")
        diagnostics["alert_flags"] = alerts
        diagnostics["has_alerts"] = bool(alerts)
        return DepthChartPriorResult(frame=out, diagnostics=diagnostics)

    out = joined.copy()
    out["dc_snapshot_ts"] = selected_ts if selected_ts is not None else pd.NaT

    # Seed rotation_prob if unavailable in the selected scorer path.
    if "rotation_prob" not in out.columns and bool(cfg.get("seed_rotation_prob_from_play_prob", True)) and "play_prob" in out.columns:
        out["rotation_prob"] = pd.to_numeric(out["play_prob"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
        diagnostics["rotation_prob_seeded_from_play_prob"] = True
    else:
        diagnostics["rotation_prob_seeded_from_play_prob"] = False

    k_role = {k: float(v) for k, v in dict(cfg.get("k_role") or {}).items()}
    k_ahead = float(cfg.get("k_ahead", -0.12))

    dc_role = out["dc_role"].astype(str).str.strip().str.lower()
    dc_ahead = pd.to_numeric(out.get("dc_ahead_global", 0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    role_shift = np.array([k_role.get(r, k_role.get("not_listed", -1.10)) for r in dc_role], dtype=float)
    delta = role_shift + k_ahead * dc_ahead
    forced_inactive = _forced_inactive_mask(out).to_numpy(dtype=bool)
    mutable = ~forced_inactive

    if "rotation_prob" in out.columns:
        rot_pre = pd.to_numeric(out["rotation_prob"], errors="coerce").fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float)
        rot_post = rot_pre.copy()
        if np.any(mutable):
            rot_post[mutable] = _sigmoid(_logit(rot_pre[mutable]) + delta[mutable])
        rot_post = np.clip(rot_post, 0.0, 1.0)
        if np.any(forced_inactive):
            rot_post[forced_inactive] = 0.0
        out["rotation_prob_pre"] = rot_pre
        out["rotation_prob"] = rot_post

    if "play_prob" in out.columns and bool(cfg.get("apply_to_play_prob", True)):
        play_scale = float(cfg.get("play_prob_scale", 0.35))
        p_pre = pd.to_numeric(out["play_prob"], errors="coerce").fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float)
        p_post = p_pre.copy()
        if np.any(mutable):
            p_post[mutable] = _sigmoid(_logit(p_pre[mutable]) + play_scale * delta[mutable])
        p_post = np.clip(p_post, 0.0, 1.0)
        if np.any(forced_inactive):
            p_post[forced_inactive] = 0.0
        out["play_prob_pre"] = p_pre
        out["play_prob"] = p_post

    out, tail_diag = _apply_tail_shaping(out, cfg)

    if np.any(forced_inactive):
        for col in (
            "minutes_p10",
            "minutes_p50",
            "minutes_p90",
            "minutes_p95",
            "minutes_p10_cond",
            "minutes_p50_cond",
            "minutes_p90_cond",
            "minutes_p95_cond",
        ):
            if col in out.columns:
                out.loc[forced_inactive, col] = 0.0

    # Final monotonicity check for core minutes columns.
    if {"minutes_p10", "minutes_p50"}.issubset(out.columns):
        out["minutes_p10"] = np.minimum(
            pd.to_numeric(out["minutes_p10"], errors="coerce").fillna(0.0),
            pd.to_numeric(out["minutes_p50"], errors="coerce").fillna(0.0),
        )
    if {"minutes_p90", "minutes_p50"}.issubset(out.columns):
        out["minutes_p90"] = np.maximum(
            pd.to_numeric(out["minutes_p90"], errors="coerce").fillna(0.0),
            pd.to_numeric(out["minutes_p50"], errors="coerce").fillna(0.0),
        )

    # Preserve conditional aliases.
    for base in ("minutes_p10", "minutes_p50", "minutes_p90"):
        cond = f"{base}_cond"
        if base in out.columns and cond in out.columns:
            out[cond] = out[base]

    role_dist = (
        out["dc_role"].astype(str).value_counts(dropna=False).to_dict()
        if "dc_role" in out.columns
        else {}
    )
    diagnostics["role_distribution"] = {str(k): int(v) for k, v in role_dist.items()}
    diagnostics["top_rotation_prob_deltas"] = _top_probability_deltas(
        out,
        col_pre="rotation_prob_pre",
        col_post="rotation_prob",
        top_n=int(cfg.get("top_n_debug", 15)),
    )
    diagnostics["top_play_prob_deltas"] = _top_probability_deltas(
        out,
        col_pre="play_prob_pre",
        col_post="play_prob",
        top_n=int(cfg.get("top_n_debug", 15)),
    )
    diagnostics["cap_hits_by_role"] = tail_diag.get("cap_hits_by_role", {})
    diagnostics["largest_q_reductions"] = tail_diag.get("largest_reductions", [])
    diagnostics["model_vs_depth_disagreements"] = _disagreement_rows(
        out,
        top_n=int(cfg.get("top_n_debug", 15)),
    )
    alerts: list[str] = []
    if not bool(diagnostics.get("applied", False)):
        alerts.append("prior_not_applied")
    if float(diagnostics.get("matched_rate", 0.0)) < float(diagnostics.get("warn_min_match_rate", 0.25)):
        alerts.append("low_match_rate")
    snap_age = diagnostics.get("snapshot_age_minutes")
    if snap_age is not None and float(snap_age) > float(diagnostics.get("warn_max_snapshot_age_minutes", 360.0)):
        alerts.append("stale_snapshot")
    diagnostics["alert_flags"] = alerts
    diagnostics["has_alerts"] = bool(alerts)

    return DepthChartPriorResult(frame=out, diagnostics=diagnostics)


__all__ = [
    "DepthChartPriorResult",
    "apply_depth_chart_prior_from_realgm",
    "load_depth_chart_prior_config",
    "ROLE_PRIORITY",
    "ROLE_VALUES",
]
