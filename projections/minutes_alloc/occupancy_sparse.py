"""Occupancy-aware sparse minutes allocation for production scoring and eval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from projections.models.rotalloc import (
    allocate_adaptive_depth,
    build_eligible_mask,
    compute_bench_share_prior,
)


DEFAULT_OCCUPANCY_P_CUTOFF = 0.10
DEFAULT_OCCUPANCY_K_MIN = 8
DEFAULT_OCCUPANCY_K_MAX = 11
DEFAULT_OCCUPANCY_CAP_MAX = 40.0
DEFAULT_OCCUPANCY_FRINGE_CAP_MAX = 14.0
DEFAULT_OCCUPANCY_SCALE = 8.0
DEFAULT_OCCUPANCY_STARTER_FLOOR = 0.85
DEFAULT_OCCUPANCY_DYNAMIC_K_BOUNDS_ENABLED = True
DEFAULT_OCCUPANCY_DYNAMIC_K_MAX_CAP = 13
DEFAULT_OCCUPANCY_DYNAMIC_K_MIN_FLOOR = 7
DEFAULT_OCCUPANCY_DYNAMIC_K_WINDOW = 3
DEFAULT_OCCUPANCY_DYNAMIC_DEPTH_PROB_FLOOR = 0.06
DEFAULT_OCCUPANCY_DYNAMIC_DEPTH_MINUTES_FLOOR = 4.0
DEFAULT_OCCUPANCY_DYNAMIC_BENCH_SHARE_MIDPOINT = 0.18
DEFAULT_OCCUPANCY_DYNAMIC_BENCH_SHARE_SCALE = 25.0
DEFAULT_OCCUPANCY_DNP_SUPPRESSION_ENABLED = True
DEFAULT_OCCUPANCY_DNP_RATE_THRESHOLD = 0.35
DEFAULT_OCCUPANCY_DNP_PRIOR_PLAY_PROB_MAX = 0.50
DEFAULT_OCCUPANCY_DNP_INACTIVE_STREAK_THRESHOLD = 3
DEFAULT_OCCUPANCY_DNP_CONSECUTIVE_ACTIVE_DNP_THRESHOLD = 2
DEFAULT_OCCUPANCY_DNP_SUPPRESSION_RELAX_IN_INJURY_REGIME = True
DEFAULT_OCCUPANCY_DNP_INJURY_REGIME_OUT_COUNT_THRESHOLD = 2
DEFAULT_OCCUPANCY_DNP_INJURY_REGIME_OUT_STARTERS_THRESHOLD = 1
DEFAULT_OCCUPANCY_DNP_INJURY_REGIME_MIN_BENCH_SHARE_PRED = 0.22
DEFAULT_OCCUPANCY_ARCHETYPE_SHORTAGE_ENABLED = True
DEFAULT_OCCUPANCY_ARCHETYPE_SOURCE_COL = "pos_bucket"
DEFAULT_OCCUPANCY_ARCHETYPE_OUT_COUNT_THRESHOLD = 1
DEFAULT_OCCUPANCY_ARCHETYPE_OUT_INACTIVE_STREAK_MAX = 10
DEFAULT_OCCUPANCY_ARCHETYPE_DNP_RATE_RELAX_ADD = 0.10
DEFAULT_OCCUPANCY_ARCHETYPE_DNP_INACTIVE_RELAX_ADD = 1
DEFAULT_OCCUPANCY_ARCHETYPE_DNP_CONSECUTIVE_RELAX_ADD = 1
DEFAULT_OCCUPANCY_ARCHETYPE_PLAY_PROB_FLOOR = 0.08
DEFAULT_OCCUPANCY_ARCHETYPE_SEED_P90_MIN = 8.0
DEFAULT_OCCUPANCY_ARCHETYPE_SEED_MINUTES_MIN = 4.0
DEFAULT_OCCUPANCY_ARCHETYPE_SEED_MINUTES_MAX = 10.0
DEFAULT_OCCUPANCY_ARCHETYPE_SEED_MINUTES_P90_MULT = 0.50
DEFAULT_OCCUPANCY_ARCHETYPE_SEED_MAX_PLAYERS = 2

_OUT_STATUS_PREFIXES = ("OUT", "INACTIVE", "DNP")
_STARTER_ROLE_VALUES = {"PROJECTED_STARTER", "CONFIRMED_STARTER"}
_ARCHETYPE_KNOWN = ("guard", "wing", "big")


@dataclass(frozen=True)
class OccupancySparseConfig:
    p_cutoff: float = DEFAULT_OCCUPANCY_P_CUTOFF
    k_min: int = DEFAULT_OCCUPANCY_K_MIN
    k_max: int = DEFAULT_OCCUPANCY_K_MAX
    cap_max: float = DEFAULT_OCCUPANCY_CAP_MAX
    fringe_cap_max: float = DEFAULT_OCCUPANCY_FRINGE_CAP_MAX
    occupancy_scale: float = DEFAULT_OCCUPANCY_SCALE
    starter_floor: float = DEFAULT_OCCUPANCY_STARTER_FLOOR
    dynamic_k_bounds_enabled: bool = DEFAULT_OCCUPANCY_DYNAMIC_K_BOUNDS_ENABLED
    dynamic_k_max_cap: int = DEFAULT_OCCUPANCY_DYNAMIC_K_MAX_CAP
    dynamic_k_min_floor: int = DEFAULT_OCCUPANCY_DYNAMIC_K_MIN_FLOOR
    dynamic_k_window: int = DEFAULT_OCCUPANCY_DYNAMIC_K_WINDOW
    dynamic_depth_prob_floor: float = DEFAULT_OCCUPANCY_DYNAMIC_DEPTH_PROB_FLOOR
    dynamic_depth_minutes_floor: float = DEFAULT_OCCUPANCY_DYNAMIC_DEPTH_MINUTES_FLOOR
    dynamic_bench_share_midpoint: float = DEFAULT_OCCUPANCY_DYNAMIC_BENCH_SHARE_MIDPOINT
    dynamic_bench_share_scale: float = DEFAULT_OCCUPANCY_DYNAMIC_BENCH_SHARE_SCALE
    dnp_suppression_enabled: bool = DEFAULT_OCCUPANCY_DNP_SUPPRESSION_ENABLED
    dnp_rate_threshold: float = DEFAULT_OCCUPANCY_DNP_RATE_THRESHOLD
    dnp_prior_play_prob_max: float = DEFAULT_OCCUPANCY_DNP_PRIOR_PLAY_PROB_MAX
    dnp_inactive_streak_threshold: int = DEFAULT_OCCUPANCY_DNP_INACTIVE_STREAK_THRESHOLD
    dnp_consecutive_active_dnp_threshold: int = (
        DEFAULT_OCCUPANCY_DNP_CONSECUTIVE_ACTIVE_DNP_THRESHOLD
    )
    dnp_suppression_relax_in_injury_regime: bool = (
        DEFAULT_OCCUPANCY_DNP_SUPPRESSION_RELAX_IN_INJURY_REGIME
    )
    dnp_injury_regime_out_count_threshold: int = (
        DEFAULT_OCCUPANCY_DNP_INJURY_REGIME_OUT_COUNT_THRESHOLD
    )
    dnp_injury_regime_out_starters_threshold: int = (
        DEFAULT_OCCUPANCY_DNP_INJURY_REGIME_OUT_STARTERS_THRESHOLD
    )
    dnp_injury_regime_min_bench_share_pred: float = (
        DEFAULT_OCCUPANCY_DNP_INJURY_REGIME_MIN_BENCH_SHARE_PRED
    )
    archetype_shortage_enabled: bool = DEFAULT_OCCUPANCY_ARCHETYPE_SHORTAGE_ENABLED
    archetype_source_col: str = DEFAULT_OCCUPANCY_ARCHETYPE_SOURCE_COL
    archetype_out_count_threshold: int = DEFAULT_OCCUPANCY_ARCHETYPE_OUT_COUNT_THRESHOLD
    archetype_out_inactive_streak_max: int = DEFAULT_OCCUPANCY_ARCHETYPE_OUT_INACTIVE_STREAK_MAX
    archetype_dnp_rate_relax_add: float = DEFAULT_OCCUPANCY_ARCHETYPE_DNP_RATE_RELAX_ADD
    archetype_dnp_inactive_relax_add: int = DEFAULT_OCCUPANCY_ARCHETYPE_DNP_INACTIVE_RELAX_ADD
    archetype_dnp_consecutive_relax_add: int = DEFAULT_OCCUPANCY_ARCHETYPE_DNP_CONSECUTIVE_RELAX_ADD
    archetype_play_prob_floor: float = DEFAULT_OCCUPANCY_ARCHETYPE_PLAY_PROB_FLOOR
    archetype_seed_p90_min: float = DEFAULT_OCCUPANCY_ARCHETYPE_SEED_P90_MIN
    archetype_seed_minutes_min: float = DEFAULT_OCCUPANCY_ARCHETYPE_SEED_MINUTES_MIN
    archetype_seed_minutes_max: float = DEFAULT_OCCUPANCY_ARCHETYPE_SEED_MINUTES_MAX
    archetype_seed_minutes_p90_mult: float = DEFAULT_OCCUPANCY_ARCHETYPE_SEED_MINUTES_P90_MULT
    archetype_seed_max_players: int = DEFAULT_OCCUPANCY_ARCHETYPE_SEED_MAX_PLAYERS

    @classmethod
    def from_payload(cls, payload: dict[str, Any] | None) -> OccupancySparseConfig:
        raw = payload or {}
        k_min = int(raw.get("k_min", cls.k_min))
        k_max = int(raw.get("k_max", cls.k_max))
        if k_min <= 0:
            k_min = cls.k_min
        if k_max <= 0:
            k_max = cls.k_max
        if k_max < k_min:
            k_max = k_min
        cap_max = float(raw.get("cap_max", cls.cap_max))
        if cap_max <= 0.0:
            cap_max = cls.cap_max
        fringe_cap_max = float(raw.get("fringe_cap_max", cls.fringe_cap_max))
        if fringe_cap_max <= 0.0:
            fringe_cap_max = cls.fringe_cap_max
        p_cutoff = float(raw.get("p_cutoff", cls.p_cutoff))
        if not np.isfinite(p_cutoff):
            p_cutoff = cls.p_cutoff
        p_cutoff = float(np.clip(p_cutoff, 0.0, 1.0))
        occupancy_scale = float(raw.get("occupancy_scale", raw.get("scale", cls.occupancy_scale)))
        if occupancy_scale <= 0.0 or not np.isfinite(occupancy_scale):
            occupancy_scale = cls.occupancy_scale
        starter_floor = float(raw.get("starter_floor", cls.starter_floor))
        if not np.isfinite(starter_floor):
            starter_floor = cls.starter_floor
        starter_floor = float(np.clip(starter_floor, 0.0, 1.0))
        dynamic_k_bounds_enabled = _coerce_bool(
            raw.get("dynamic_k_bounds_enabled", raw.get("dynamic_k_bounds", cls.dynamic_k_bounds_enabled)),
            default=cls.dynamic_k_bounds_enabled,
        )
        dynamic_k_max_cap = int(raw.get("dynamic_k_max_cap", cls.dynamic_k_max_cap))
        if dynamic_k_max_cap <= 0:
            dynamic_k_max_cap = cls.dynamic_k_max_cap
        dynamic_k_min_floor = int(raw.get("dynamic_k_min_floor", cls.dynamic_k_min_floor))
        if dynamic_k_min_floor <= 0:
            dynamic_k_min_floor = cls.dynamic_k_min_floor
        dynamic_k_window = int(raw.get("dynamic_k_window", cls.dynamic_k_window))
        if dynamic_k_window <= 0:
            dynamic_k_window = cls.dynamic_k_window
        dynamic_depth_prob_floor = float(raw.get("dynamic_depth_prob_floor", cls.dynamic_depth_prob_floor))
        if not np.isfinite(dynamic_depth_prob_floor):
            dynamic_depth_prob_floor = cls.dynamic_depth_prob_floor
        dynamic_depth_prob_floor = float(np.clip(dynamic_depth_prob_floor, 0.0, 1.0))
        dynamic_depth_minutes_floor = float(raw.get("dynamic_depth_minutes_floor", cls.dynamic_depth_minutes_floor))
        if not np.isfinite(dynamic_depth_minutes_floor) or dynamic_depth_minutes_floor < 0.0:
            dynamic_depth_minutes_floor = cls.dynamic_depth_minutes_floor
        dynamic_bench_share_midpoint = float(
            raw.get("dynamic_bench_share_midpoint", cls.dynamic_bench_share_midpoint)
        )
        if not np.isfinite(dynamic_bench_share_midpoint):
            dynamic_bench_share_midpoint = cls.dynamic_bench_share_midpoint
        dynamic_bench_share_midpoint = float(np.clip(dynamic_bench_share_midpoint, 0.0, 1.0))
        dynamic_bench_share_scale = float(raw.get("dynamic_bench_share_scale", cls.dynamic_bench_share_scale))
        if not np.isfinite(dynamic_bench_share_scale) or dynamic_bench_share_scale <= 0.0:
            dynamic_bench_share_scale = cls.dynamic_bench_share_scale
        dnp_suppression_enabled = _coerce_bool(
            raw.get("dnp_suppression_enabled", cls.dnp_suppression_enabled),
            default=cls.dnp_suppression_enabled,
        )
        dnp_rate_threshold = float(raw.get("dnp_rate_threshold", cls.dnp_rate_threshold))
        if not np.isfinite(dnp_rate_threshold):
            dnp_rate_threshold = cls.dnp_rate_threshold
        dnp_rate_threshold = float(np.clip(dnp_rate_threshold, 0.0, 1.0))
        dnp_prior_play_prob_max = float(raw.get("dnp_prior_play_prob_max", cls.dnp_prior_play_prob_max))
        if not np.isfinite(dnp_prior_play_prob_max):
            dnp_prior_play_prob_max = cls.dnp_prior_play_prob_max
        dnp_prior_play_prob_max = float(np.clip(dnp_prior_play_prob_max, 0.0, 1.0))
        dnp_inactive_streak_threshold = int(
            raw.get("dnp_inactive_streak_threshold", cls.dnp_inactive_streak_threshold)
        )
        if dnp_inactive_streak_threshold <= 0:
            dnp_inactive_streak_threshold = cls.dnp_inactive_streak_threshold
        dnp_consecutive_active_dnp_threshold = int(
            raw.get(
                "dnp_consecutive_active_dnp_threshold",
                cls.dnp_consecutive_active_dnp_threshold,
            )
        )
        if dnp_consecutive_active_dnp_threshold <= 0:
            dnp_consecutive_active_dnp_threshold = cls.dnp_consecutive_active_dnp_threshold
        dnp_suppression_relax_in_injury_regime = _coerce_bool(
            raw.get(
                "dnp_suppression_relax_in_injury_regime",
                cls.dnp_suppression_relax_in_injury_regime,
            ),
            default=cls.dnp_suppression_relax_in_injury_regime,
        )
        dnp_injury_regime_out_count_threshold = int(
            raw.get(
                "dnp_injury_regime_out_count_threshold",
                cls.dnp_injury_regime_out_count_threshold,
            )
        )
        if dnp_injury_regime_out_count_threshold <= 0:
            dnp_injury_regime_out_count_threshold = cls.dnp_injury_regime_out_count_threshold
        dnp_injury_regime_out_starters_threshold = int(
            raw.get(
                "dnp_injury_regime_out_starters_threshold",
                cls.dnp_injury_regime_out_starters_threshold,
            )
        )
        if dnp_injury_regime_out_starters_threshold <= 0:
            dnp_injury_regime_out_starters_threshold = cls.dnp_injury_regime_out_starters_threshold
        dnp_injury_regime_min_bench_share_pred = float(
            raw.get(
                "dnp_injury_regime_min_bench_share_pred",
                cls.dnp_injury_regime_min_bench_share_pred,
            )
        )
        if not np.isfinite(dnp_injury_regime_min_bench_share_pred):
            dnp_injury_regime_min_bench_share_pred = cls.dnp_injury_regime_min_bench_share_pred
        dnp_injury_regime_min_bench_share_pred = float(
            np.clip(dnp_injury_regime_min_bench_share_pred, 0.0, 1.0)
        )
        archetype_shortage_enabled = _coerce_bool(
            raw.get("archetype_shortage_enabled", cls.archetype_shortage_enabled),
            default=cls.archetype_shortage_enabled,
        )
        archetype_source_col = str(raw.get("archetype_source_col", cls.archetype_source_col) or cls.archetype_source_col).strip()
        if not archetype_source_col:
            archetype_source_col = cls.archetype_source_col
        archetype_out_count_threshold = int(
            raw.get("archetype_out_count_threshold", cls.archetype_out_count_threshold)
        )
        if archetype_out_count_threshold <= 0:
            archetype_out_count_threshold = cls.archetype_out_count_threshold
        archetype_out_inactive_streak_max = int(
            raw.get("archetype_out_inactive_streak_max", cls.archetype_out_inactive_streak_max)
        )
        if archetype_out_inactive_streak_max < 0:
            archetype_out_inactive_streak_max = cls.archetype_out_inactive_streak_max
        archetype_dnp_rate_relax_add = float(
            raw.get("archetype_dnp_rate_relax_add", cls.archetype_dnp_rate_relax_add)
        )
        if not np.isfinite(archetype_dnp_rate_relax_add):
            archetype_dnp_rate_relax_add = cls.archetype_dnp_rate_relax_add
        archetype_dnp_rate_relax_add = float(np.clip(archetype_dnp_rate_relax_add, 0.0, 1.0))
        archetype_dnp_inactive_relax_add = int(
            raw.get("archetype_dnp_inactive_relax_add", cls.archetype_dnp_inactive_relax_add)
        )
        if archetype_dnp_inactive_relax_add < 0:
            archetype_dnp_inactive_relax_add = cls.archetype_dnp_inactive_relax_add
        archetype_dnp_consecutive_relax_add = int(
            raw.get("archetype_dnp_consecutive_relax_add", cls.archetype_dnp_consecutive_relax_add)
        )
        if archetype_dnp_consecutive_relax_add < 0:
            archetype_dnp_consecutive_relax_add = cls.archetype_dnp_consecutive_relax_add
        archetype_play_prob_floor = float(
            raw.get("archetype_play_prob_floor", cls.archetype_play_prob_floor)
        )
        if not np.isfinite(archetype_play_prob_floor):
            archetype_play_prob_floor = cls.archetype_play_prob_floor
        archetype_play_prob_floor = float(np.clip(archetype_play_prob_floor, 0.0, 1.0))
        archetype_seed_p90_min = float(raw.get("archetype_seed_p90_min", cls.archetype_seed_p90_min))
        if not np.isfinite(archetype_seed_p90_min) or archetype_seed_p90_min < 0.0:
            archetype_seed_p90_min = cls.archetype_seed_p90_min
        archetype_seed_minutes_min = float(raw.get("archetype_seed_minutes_min", cls.archetype_seed_minutes_min))
        if not np.isfinite(archetype_seed_minutes_min) or archetype_seed_minutes_min < 0.0:
            archetype_seed_minutes_min = cls.archetype_seed_minutes_min
        archetype_seed_minutes_max = float(raw.get("archetype_seed_minutes_max", cls.archetype_seed_minutes_max))
        if not np.isfinite(archetype_seed_minutes_max) or archetype_seed_minutes_max < 0.0:
            archetype_seed_minutes_max = cls.archetype_seed_minutes_max
        if archetype_seed_minutes_max < archetype_seed_minutes_min:
            archetype_seed_minutes_max = archetype_seed_minutes_min
        archetype_seed_minutes_p90_mult = float(
            raw.get("archetype_seed_minutes_p90_mult", cls.archetype_seed_minutes_p90_mult)
        )
        if not np.isfinite(archetype_seed_minutes_p90_mult) or archetype_seed_minutes_p90_mult < 0.0:
            archetype_seed_minutes_p90_mult = cls.archetype_seed_minutes_p90_mult
        archetype_seed_max_players = int(raw.get("archetype_seed_max_players", cls.archetype_seed_max_players))
        if archetype_seed_max_players <= 0:
            archetype_seed_max_players = cls.archetype_seed_max_players
        return cls(
            p_cutoff=p_cutoff,
            k_min=k_min,
            k_max=k_max,
            cap_max=cap_max,
            fringe_cap_max=fringe_cap_max,
            occupancy_scale=occupancy_scale,
            starter_floor=starter_floor,
            dynamic_k_bounds_enabled=dynamic_k_bounds_enabled,
            dynamic_k_max_cap=dynamic_k_max_cap,
            dynamic_k_min_floor=dynamic_k_min_floor,
            dynamic_k_window=dynamic_k_window,
            dynamic_depth_prob_floor=dynamic_depth_prob_floor,
            dynamic_depth_minutes_floor=dynamic_depth_minutes_floor,
            dynamic_bench_share_midpoint=dynamic_bench_share_midpoint,
            dynamic_bench_share_scale=dynamic_bench_share_scale,
            dnp_suppression_enabled=dnp_suppression_enabled,
            dnp_rate_threshold=dnp_rate_threshold,
            dnp_prior_play_prob_max=dnp_prior_play_prob_max,
            dnp_inactive_streak_threshold=dnp_inactive_streak_threshold,
            dnp_consecutive_active_dnp_threshold=dnp_consecutive_active_dnp_threshold,
            dnp_suppression_relax_in_injury_regime=dnp_suppression_relax_in_injury_regime,
            dnp_injury_regime_out_count_threshold=dnp_injury_regime_out_count_threshold,
            dnp_injury_regime_out_starters_threshold=dnp_injury_regime_out_starters_threshold,
            dnp_injury_regime_min_bench_share_pred=dnp_injury_regime_min_bench_share_pred,
            archetype_shortage_enabled=archetype_shortage_enabled,
            archetype_source_col=archetype_source_col,
            archetype_out_count_threshold=archetype_out_count_threshold,
            archetype_out_inactive_streak_max=archetype_out_inactive_streak_max,
            archetype_dnp_rate_relax_add=archetype_dnp_rate_relax_add,
            archetype_dnp_inactive_relax_add=archetype_dnp_inactive_relax_add,
            archetype_dnp_consecutive_relax_add=archetype_dnp_consecutive_relax_add,
            archetype_play_prob_floor=archetype_play_prob_floor,
            archetype_seed_p90_min=archetype_seed_p90_min,
            archetype_seed_minutes_min=archetype_seed_minutes_min,
            archetype_seed_minutes_max=archetype_seed_minutes_max,
            archetype_seed_minutes_p90_mult=archetype_seed_minutes_p90_mult,
            archetype_seed_max_players=archetype_seed_max_players,
        )


def _safe_status_upper(series: pd.Series | None, *, index: pd.Index) -> pd.Series:
    if series is None:
        return pd.Series("", index=index, dtype="string")
    return series.astype("string", copy=False).fillna("").str.upper()


def _coerce_bool_series(series: pd.Series | None, *, index: pd.Index) -> pd.Series:
    if series is None:
        return pd.Series(False, index=index, dtype=bool)
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return numeric.fillna(0.0).astype(float).ne(0.0)

    text = series.astype("string", copy=False).str.strip().str.lower()
    return text.fillna("").isin({"1", "true", "t", "yes", "y"})


def _coerce_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    return bool(default)


def _coerce_numeric_series(
    frame: pd.DataFrame,
    column: str,
    *,
    default: float = 0.0,
) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _normalize_archetype_value(value: Any) -> str:
    token = str(value).strip().upper()
    if not token or token in {"NAN", "NONE", "<NA>"}:
        return "unknown"
    if "BIG" in token:
        return "big"
    if token in {"C", "PF", "FC", "CF", "C-F", "F-C", "PF/C", "C/PF"}:
        return "big"
    if "GUARD" in token:
        return "guard"
    if token in {"G", "PG", "SG", "G-F", "F-G", "PG/SG", "SG/PG"}:
        return "guard"
    if "WING" in token:
        return "wing"
    if "FORWARD" in token:
        return "wing"
    if token in {"W", "F", "SF", "SF/PF", "PF/SF"}:
        return "wing"
    if "/" in token or "-" in token:
        parts = token.replace("-", "/").split("/")
        for part in parts:
            mapped = _normalize_archetype_value(part)
            if mapped != "unknown":
                return mapped
    return "unknown"


def _resolve_archetype_series(
    frame: pd.DataFrame,
    *,
    index: pd.Index,
    source_col: str,
) -> pd.Series:
    candidates = [source_col, "pos_bucket", "position", "pos", "position_group"]
    seen: set[str] = set()
    selected_col: str | None = None
    for col in candidates:
        if not col or col in seen:
            continue
        seen.add(col)
        if col in frame.columns:
            selected_col = col
            break
    if selected_col is None:
        return pd.Series("unknown", index=index, dtype="string")
    raw = frame[selected_col].astype("string", copy=False).fillna("")
    return raw.map(_normalize_archetype_value).astype("string")


def apply_occupancy_sparse_allocation(
    frame: pd.DataFrame,
    *,
    config: OccupancySparseConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply occupancy+sparse reallocation on canonical minutes prediction columns.

    Expected input columns:
      - keys: game_id, team_id, player_id
      - prediction: play_prob, minutes_p50 (+ optional minutes_p10/minutes_p90)
      - optional context: status, is_out, lineup_role, starter flags, spread_home, total, home_flag
    """
    required_cols = {"game_id", "team_id", "player_id", "play_prob", "minutes_p50"}
    missing = sorted(required_cols - set(frame.columns))
    if missing:
        raise ValueError(f"apply_occupancy_sparse_allocation missing required columns: {missing}")
    if frame.empty:
        return frame.copy(), pd.DataFrame()

    working = frame.copy()
    p10_base = pd.to_numeric(working.get("minutes_p10"), errors="coerce").fillna(
        pd.to_numeric(working["minutes_p50"], errors="coerce").fillna(0.0)
    )
    p50_base = pd.to_numeric(working["minutes_p50"], errors="coerce").fillna(0.0)
    p90_base = pd.to_numeric(working.get("minutes_p90"), errors="coerce").fillna(p50_base)

    p10_base = np.maximum(0.0, p10_base)
    p50_base = np.maximum(0.0, p50_base)
    p90_base = np.maximum(p10_base, p90_base)

    lower_width = np.maximum(0.0, p50_base - p10_base).to_numpy(dtype=float)
    upper_width = np.maximum(0.0, p90_base - p50_base).to_numpy(dtype=float)
    p90_pred = p90_base.to_numpy(dtype=float)
    p_rot = (
        pd.to_numeric(working["play_prob"], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0, upper=1.0)
        .to_numpy(dtype=float)
    )
    mu_pred = pd.to_numeric(working["minutes_p50"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    mu_pred = np.maximum(mu_pred, 0.0)

    status_upper = _safe_status_upper(working.get("status"), index=working.index)
    role_upper = _safe_status_upper(working.get("lineup_role"), index=working.index)

    out_mask = pd.Series(False, index=working.index, dtype=bool)
    if "is_out" in working.columns:
        out_mask = out_mask | pd.to_numeric(working["is_out"], errors="coerce").fillna(0).astype(int).eq(1)
    out_mask = out_mask | status_upper.str.startswith(_OUT_STATUS_PREFIXES)
    out_mask = out_mask | status_upper.str.contains("SUSP", na=False)
    out_mask = out_mask | role_upper.eq("OUT")

    starter_mask_raw = pd.Series(False, index=working.index, dtype=bool)
    for col in ("starter_flag", "is_projected_starter", "is_confirmed_starter", "is_starter"):
        starter_mask_raw = starter_mask_raw | _coerce_bool_series(working.get(col), index=working.index)
    starter_mask_raw = starter_mask_raw | role_upper.isin(_STARTER_ROLE_VALUES)
    starter_mask = starter_mask_raw & ~out_mask

    active_but_dnp_rate = (
        _coerce_numeric_series(working, "active_but_dnp_rate_last10", default=0.0)
        .fillna(0.0)
        .clip(lower=0.0, upper=1.0)
        .to_numpy(dtype=float)
    )
    consecutive_active_dnp = (
        _coerce_numeric_series(working, "consecutive_active_dnp", default=0.0)
        .fillna(0)
        .clip(lower=0)
        .to_numpy(dtype=float)
    )
    inactive_streak_len = (
        _coerce_numeric_series(working, "inactive_streak_len", default=0.0)
        .fillna(0)
        .clip(lower=0)
        .to_numpy(dtype=float)
    )
    prior_play_prob = _coerce_numeric_series(
        working, "prior_play_prob", default=np.nan
    ).to_numpy(dtype=float)
    archetype_arr = _resolve_archetype_series(
        working,
        index=working.index,
        source_col=config.archetype_source_col,
    ).to_numpy(dtype=object)

    home_flag_series = (
        pd.to_numeric(working.get("home_flag"), errors="coerce").fillna(0.0)
        if "home_flag" in working.columns
        else pd.Series(0.0, index=working.index, dtype=float)
    )

    minutes_occ = np.zeros(len(working), dtype=float)
    eligible_flags = np.zeros(len(working), dtype=bool)
    bench_share_pred_arr = np.zeros(len(working), dtype=float)
    diagnostic_rows: list[dict[str, Any]] = []

    for (game_id, team_id), group in working.groupby(["game_id", "team_id"], sort=False):
        idx = group.index.to_numpy()
        if idx.size == 0:
            continue

        p_g = p_rot[idx]
        mu_g = mu_pred[idx]
        p90_g = p90_pred[idx]
        out_g = out_mask.iloc[idx].to_numpy(dtype=bool)
        starter_g = starter_mask.iloc[idx].to_numpy(dtype=bool)
        starter_raw_g = starter_mask_raw.iloc[idx].to_numpy(dtype=bool)
        archetype_g = archetype_arr[idx]
        dnp_rate_g = active_but_dnp_rate[idx]
        consec_dnp_g = consecutive_active_dnp[idx]
        inactive_streak_g = inactive_streak_len[idx]
        prior_play_g = prior_play_prob[idx]
        prior_is_low_or_missing = np.isnan(prior_play_g) | (
            prior_play_g <= float(config.dnp_prior_play_prob_max)
        )

        dnp_risk_g = np.zeros(len(idx), dtype=bool)
        injury_regime_active = False
        out_starters_count = int((out_g & starter_raw_g).sum())
        injury_count_trigger = False
        rate_threshold_used = float(config.dnp_rate_threshold)
        inactive_threshold_used = int(config.dnp_inactive_streak_threshold)
        consecutive_threshold_used = int(config.dnp_consecutive_active_dnp_threshold)
        n_archetype_seeded = 0
        n_archetype_dnp_rescued = 0
        archetype_seed_mask = np.zeros(len(idx), dtype=bool)
        archetype_out_counts: dict[str, int] = {k: 0 for k in _ARCHETYPE_KNOWN}
        archetype_shortage_mask = np.zeros(len(idx), dtype=bool)
        archetype_shortage_active = False

        out_recent_or_starter = out_g & (
            starter_raw_g
            | (
                inactive_streak_g
                <= int(config.archetype_out_inactive_streak_max)
            )
        )
        out_count_context = (
            int(out_recent_or_starter.sum())
            if bool(config.archetype_shortage_enabled)
            else int(out_g.sum())
        )
        injury_count_trigger = (
            int(out_count_context) >= int(config.dnp_injury_regime_out_count_threshold)
        ) or (
            out_starters_count
            >= int(config.dnp_injury_regime_out_starters_threshold)
        )
        if bool(config.archetype_shortage_enabled):
            for bucket in _ARCHETYPE_KNOWN:
                archetype_out_counts[bucket] = int(
                    np.sum(out_recent_or_starter & (archetype_g == bucket))
                )
            shortage_by_archetype = {
                bucket: count
                for bucket, count in archetype_out_counts.items()
                if count >= int(config.archetype_out_count_threshold)
            }
            archetype_shortage_active = bool(shortage_by_archetype) and bool(injury_count_trigger)
            if shortage_by_archetype:
                archetype_shortage_mask = np.array(
                    [shortage_by_archetype.get(str(bucket), 0) > 0 for bucket in archetype_g],
                    dtype=bool,
                )

        spread_val: float | None = None
        if "spread_home" in group.columns:
            spread_raw = pd.to_numeric(group["spread_home"], errors="coerce").iloc[0]
            if pd.notna(spread_raw):
                is_home = bool(home_flag_series.iloc[idx[0]] > 0.0)
                spread_val = float(spread_raw) if is_home else -float(spread_raw)

        total_val: float | None = None
        if "total" in group.columns:
            total_raw = pd.to_numeric(group["total"], errors="coerce").iloc[0]
            if pd.notna(total_raw):
                total_val = float(total_raw)

        bench_share = compute_bench_share_prior(
            team_bench_share_avg=None,
            spread=spread_val,
            total=total_val,
            out_count=int(out_count_context),
        )
        if config.dnp_suppression_enabled:
            dnp_risk_base = (
                (
                    (dnp_rate_g >= float(config.dnp_rate_threshold))
                    & prior_is_low_or_missing
                )
                | (inactive_streak_g >= int(config.dnp_inactive_streak_threshold))
                | (
                    consec_dnp_g
                    >= int(config.dnp_consecutive_active_dnp_threshold)
                )
            )
            injury_regime_active = (
                injury_count_trigger
                and float(bench_share)
                >= float(config.dnp_injury_regime_min_bench_share_pred)
            )
            if (
                injury_regime_active
                and config.dnp_suppression_relax_in_injury_regime
            ):
                relaxed_rate_threshold = float(
                    min(1.0, float(config.dnp_rate_threshold) + 0.15)
                )
                relaxed_inactive_threshold = int(
                    max(
                        int(config.dnp_inactive_streak_threshold),
                        int(config.dnp_inactive_streak_threshold) + 2,
                    )
                )
                relaxed_consecutive_threshold = int(
                    max(
                        int(config.dnp_consecutive_active_dnp_threshold),
                        int(config.dnp_consecutive_active_dnp_threshold) + 1,
                    )
                )
                rate_threshold_used = relaxed_rate_threshold
                inactive_threshold_used = relaxed_inactive_threshold
                consecutive_threshold_used = relaxed_consecutive_threshold
                dnp_risk_g = (
                    (
                        (dnp_rate_g >= relaxed_rate_threshold)
                        & prior_is_low_or_missing
                    )
                    | (inactive_streak_g >= relaxed_inactive_threshold)
                    | (consec_dnp_g >= relaxed_consecutive_threshold)
                )
            else:
                dnp_risk_g = dnp_risk_base
            dnp_risk_g = dnp_risk_g & (~starter_g) & (~out_g)
            if bool(config.archetype_shortage_enabled) and archetype_shortage_active:
                shortage_target = archetype_shortage_mask & (~starter_g) & (~out_g)
                if shortage_target.any():
                    archetype_rate_threshold = float(
                        min(1.0, rate_threshold_used + float(config.archetype_dnp_rate_relax_add))
                    )
                    archetype_inactive_threshold = int(
                        inactive_threshold_used + int(config.archetype_dnp_inactive_relax_add)
                    )
                    archetype_consecutive_threshold = int(
                        consecutive_threshold_used + int(config.archetype_dnp_consecutive_relax_add)
                    )
                    archetype_risk = (
                        (
                            (dnp_rate_g >= archetype_rate_threshold)
                            & prior_is_low_or_missing
                        )
                        | (inactive_streak_g >= archetype_inactive_threshold)
                        | (consec_dnp_g >= archetype_consecutive_threshold)
                    )
                    dnp_risk_g[shortage_target] = archetype_risk[shortage_target]

        p_for_mask = p_g.copy()
        mu_for_mask = mu_g.copy()
        if bool(config.archetype_shortage_enabled) and archetype_shortage_active:
            shortage_active = archetype_shortage_mask & (~out_g)
            if shortage_active.any():
                p_for_mask[shortage_active] = np.maximum(
                    p_for_mask[shortage_active],
                    float(config.archetype_play_prob_floor),
                )
            dnp_risk_before_seed = dnp_risk_g.copy()
            seed_candidates = (
                shortage_active
                & (p90_g >= float(config.archetype_seed_p90_min))
                & ((mu_for_mask <= 0.0) | dnp_risk_g)
            )
            if seed_candidates.any():
                seed_scores = np.where(seed_candidates, p90_g, -np.inf)
                order = np.argsort(-seed_scores)
                take_n = min(int(config.archetype_seed_max_players), int(seed_candidates.sum()))
                selected = [int(i) for i in order if seed_candidates[int(i)]][:take_n]
                if selected:
                    sel_idx = np.asarray(selected, dtype=int)
                    seed_zero_mu = mu_for_mask[sel_idx] <= 0.0
                    if np.any(seed_zero_mu):
                        seed_idx = sel_idx[seed_zero_mu]
                        seed_minutes = np.clip(
                            p90_g[seed_idx] * float(config.archetype_seed_minutes_p90_mult),
                            float(config.archetype_seed_minutes_min),
                            float(config.archetype_seed_minutes_max),
                        )
                        mu_for_mask[seed_idx] = np.maximum(mu_for_mask[seed_idx], seed_minutes)
                    dnp_risk_g[sel_idx] = False
                    p_for_mask[sel_idx] = np.maximum(
                        p_for_mask[sel_idx],
                        float(config.archetype_play_prob_floor),
                    )
                    archetype_seed_mask[sel_idx] = True
                    n_archetype_seeded = int(sel_idx.size)
                    n_archetype_dnp_rescued = int(dnp_risk_before_seed[sel_idx].sum())

        candidate_mask = (
            (~out_g)
            & np.isfinite(mu_for_mask)
            & np.isfinite(p_for_mask)
            & (mu_for_mask > 0.0)
            & (p_for_mask > 0.0)
        )
        candidate_mask = candidate_mask & (~dnp_risk_g)
        if not candidate_mask.any():
            starter_candidates = starter_g & (~out_g)
            if starter_candidates.any():
                candidate_mask = starter_candidates
            elif (~out_g).any():
                fallback_pool = (~out_g) & (~dnp_risk_g)
                if not fallback_pool.any():
                    fallback_pool = ~out_g
                best_idx = int(np.argmax(np.where(fallback_pool, mu_g, -np.inf)))
                candidate_mask = np.zeros_like(out_g, dtype=bool)
                candidate_mask[best_idx] = True

        bench_share_pred_arr[idx] = bench_share

        active_count = int((~out_g).sum())
        k_min_eff = int(config.k_min)
        k_max_eff = int(config.k_max)
        depth_signal_count = 0
        bench_depth_boost = 0
        if config.dynamic_k_bounds_enabled and active_count > 0:
            depth_mask = (~out_g) & (mu_g >= float(config.dynamic_depth_minutes_floor))
            prob_floor = float(np.clip(config.dynamic_depth_prob_floor, 0.0, 1.0))
            if prob_floor > 0.0:
                depth_mask &= p_g >= prob_floor
            depth_mask &= ~dnp_risk_g
            depth_signal_count = int(depth_mask.sum())

            bench_term = (float(bench_share) - float(config.dynamic_bench_share_midpoint)) * float(
                config.dynamic_bench_share_scale
            )
            if np.isfinite(bench_term):
                bench_depth_boost = int(np.clip(np.round(bench_term), 0.0, 2.0))
            k_max_cap = max(int(config.k_max), int(config.dynamic_k_max_cap))
            k_max_candidate = max(int(config.k_max), depth_signal_count, int(config.k_max) + bench_depth_boost)
            k_max_eff = min(active_count, max(1, min(k_max_candidate, k_max_cap)))

            k_window = max(1, int(config.dynamic_k_window))
            k_min_floor = max(1, int(config.dynamic_k_min_floor))
            k_min_from_window = max(k_min_floor, k_max_eff - k_window)
            k_min_eff = min(k_max_eff, max(int(config.k_min), k_min_from_window))
        elif active_count > 0:
            k_max_eff = min(active_count, max(1, int(config.k_max)))
            k_min_eff = min(k_max_eff, max(1, int(config.k_min)))

        eligible_g = build_eligible_mask(
            p_for_mask,
            mu_for_mask,
            candidate_mask,
            a=1.0,
            mu_power=1.0,
            p_cutoff=float(config.p_cutoff),
            k_min=k_min_eff,
            k_max=k_max_eff,
            use_expected_k=True,
        )
        eligible_g = eligible_g | (starter_g & (~out_g))
        eligible_g = eligible_g | (archetype_seed_mask & (~out_g))
        eligible_g = eligible_g & (~dnp_risk_g)
        eligible_g = eligible_g & (~out_g)
        if not eligible_g.any() and (~out_g).any():
            fallback_pool = (~out_g) & (~dnp_risk_g)
            if not fallback_pool.any():
                fallback_pool = ~out_g
            best_idx = int(np.argmax(np.where(fallback_pool, mu_g, -np.inf)))
            eligible_g = np.zeros_like(out_g, dtype=bool)
            eligible_g[best_idx] = True

        team_minutes = np.zeros(len(idx), dtype=float)
        depth_diag: dict[str, Any] = {}
        if eligible_g.any():
            team_minutes, depth_diag = allocate_adaptive_depth(
                p_for_mask,
                mu_for_mask,
                eligible_g,
                a=1.0,
                mu_power=1.0,
                bench_share_pred=float(bench_share),
                core_k_min=k_min_eff,
                core_k_max=k_max_eff,
                fringe_cap_max=float(config.fringe_cap_max),
                cap_max=float(config.cap_max),
            )
        team_minutes[out_g] = 0.0
        minutes_occ[idx] = np.maximum(team_minutes, 0.0)
        eligible_flags[idx] = eligible_g

        team_sum = float(team_minutes.sum())
        team_sum_dev = abs(team_sum - 240.0) if active_count > 0 else 0.0
        diagnostic_rows.append(
            {
                "game_id": game_id,
                "team_id": team_id,
                "rows": int(len(idx)),
                "n_out": int(out_g.sum()),
                "n_out_context": int(out_count_context),
                "n_starters": int(starter_g.sum()),
                "n_eligible": int(eligible_g.sum()),
                "n_dnp_suppressed": int(dnp_risk_g.sum()),
                "active_count": active_count,
                "n_out_starters": out_starters_count,
                "injury_regime_active": bool(injury_regime_active),
                "archetype_shortage_active": bool(archetype_shortage_active),
                "archetype_out_guard": int(archetype_out_counts["guard"]),
                "archetype_out_wing": int(archetype_out_counts["wing"]),
                "archetype_out_big": int(archetype_out_counts["big"]),
                "n_archetype_shortage_players": int((archetype_shortage_mask & (~out_g)).sum()),
                "n_archetype_seeded": int(n_archetype_seeded),
                "n_archetype_dnp_rescued": int(n_archetype_dnp_rescued),
                "k_min_eff": int(k_min_eff),
                "k_max_eff": int(k_max_eff),
                "depth_signal_count": int(depth_signal_count),
                "bench_depth_boost": int(bench_depth_boost),
                "dynamic_k_bounds_enabled": bool(config.dynamic_k_bounds_enabled),
                "dnp_suppression_enabled": bool(config.dnp_suppression_enabled),
                "team_minutes_sum": team_sum,
                "team_minutes_sum_dev": team_sum_dev,
                "bench_share_pred": float(bench_share),
                "bench_share_actual": float(depth_diag.get("bench_share_actual", 0.0)),
                "core_k": int(depth_diag.get("core_k", 0)),
                "fringe_minutes_sum": float(depth_diag.get("fringe_minutes_sum", 0.0)),
            }
        )

    scale = max(float(config.occupancy_scale), 1e-6)
    play_prob_occ = 1.0 - np.exp(-minutes_occ / scale)
    play_prob_occ = np.clip(play_prob_occ, 0.0, 1.0)
    out_arr = out_mask.to_numpy(dtype=bool)
    starter_arr = starter_mask.to_numpy(dtype=bool)
    play_prob_occ[out_arr] = 0.0
    starter_active = starter_arr & (~out_arr) & (minutes_occ > 0.0)
    play_prob_occ[starter_active] = np.maximum(play_prob_occ[starter_active], float(config.starter_floor))

    p10_occ = np.maximum(0.0, minutes_occ - lower_width)
    p90_occ = np.maximum(p10_occ, minutes_occ + upper_width)
    p10_occ[out_arr] = 0.0
    p90_occ[out_arr] = 0.0

    working["minutes_occ"] = minutes_occ
    working["play_prob_occ"] = play_prob_occ
    working["minutes_p10_occ"] = p10_occ
    working["minutes_p90_occ"] = p90_occ
    working["eligible_flag_occ"] = eligible_flags.astype(int)
    working["bench_share_pred_occ"] = bench_share_pred_arr
    working["out_flag_occ"] = out_mask.astype(int)
    working["starter_flag_occ"] = starter_mask.astype(int)

    diagnostics = pd.DataFrame(diagnostic_rows)
    return working, diagnostics
