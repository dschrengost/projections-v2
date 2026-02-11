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

_OUT_STATUS_PREFIXES = ("OUT", "INACTIVE", "DNP")
_STARTER_ROLE_VALUES = {"PROJECTED_STARTER", "CONFIRMED_STARTER"}


@dataclass(frozen=True)
class OccupancySparseConfig:
    p_cutoff: float = DEFAULT_OCCUPANCY_P_CUTOFF
    k_min: int = DEFAULT_OCCUPANCY_K_MIN
    k_max: int = DEFAULT_OCCUPANCY_K_MAX
    cap_max: float = DEFAULT_OCCUPANCY_CAP_MAX
    fringe_cap_max: float = DEFAULT_OCCUPANCY_FRINGE_CAP_MAX
    occupancy_scale: float = DEFAULT_OCCUPANCY_SCALE
    starter_floor: float = DEFAULT_OCCUPANCY_STARTER_FLOOR

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
        return cls(
            p_cutoff=p_cutoff,
            k_min=k_min,
            k_max=k_max,
            cap_max=cap_max,
            fringe_cap_max=fringe_cap_max,
            occupancy_scale=occupancy_scale,
            starter_floor=starter_floor,
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

    starter_mask = pd.Series(False, index=working.index, dtype=bool)
    for col in ("starter_flag", "is_projected_starter", "is_confirmed_starter", "is_starter"):
        starter_mask = starter_mask | _coerce_bool_series(working.get(col), index=working.index)
    starter_mask = starter_mask | role_upper.isin(_STARTER_ROLE_VALUES)
    starter_mask = starter_mask & ~out_mask

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
        out_g = out_mask.iloc[idx].to_numpy(dtype=bool)
        starter_g = starter_mask.iloc[idx].to_numpy(dtype=bool)

        candidate_mask = (~out_g) & np.isfinite(mu_g) & np.isfinite(p_g) & (mu_g > 0.0) & (p_g > 0.0)
        if not candidate_mask.any():
            starter_candidates = starter_g & (~out_g)
            if starter_candidates.any():
                candidate_mask = starter_candidates
            elif (~out_g).any():
                best_idx = int(np.argmax(np.where(~out_g, mu_g, -np.inf)))
                candidate_mask = np.zeros_like(out_g, dtype=bool)
                candidate_mask[best_idx] = True

        eligible_g = build_eligible_mask(
            p_g,
            mu_g,
            candidate_mask,
            a=1.0,
            mu_power=1.0,
            p_cutoff=float(config.p_cutoff),
            k_min=int(config.k_min),
            k_max=int(config.k_max),
            use_expected_k=True,
        )
        eligible_g = eligible_g | (starter_g & (~out_g))
        eligible_g = eligible_g & (~out_g)
        if not eligible_g.any() and (~out_g).any():
            best_idx = int(np.argmax(np.where(~out_g, mu_g, -np.inf)))
            eligible_g = np.zeros_like(out_g, dtype=bool)
            eligible_g[best_idx] = True

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
            out_count=int(out_g.sum()),
        )
        bench_share_pred_arr[idx] = bench_share

        team_minutes = np.zeros(len(idx), dtype=float)
        depth_diag: dict[str, Any] = {}
        if eligible_g.any():
            team_minutes, depth_diag = allocate_adaptive_depth(
                p_g,
                mu_g,
                eligible_g,
                a=1.0,
                mu_power=1.0,
                bench_share_pred=float(bench_share),
                core_k_min=int(config.k_min),
                core_k_max=int(config.k_max),
                fringe_cap_max=float(config.fringe_cap_max),
                cap_max=float(config.cap_max),
            )
        team_minutes[out_g] = 0.0
        minutes_occ[idx] = np.maximum(team_minutes, 0.0)
        eligible_flags[idx] = eligible_g

        active_count = int((~out_g).sum())
        team_sum = float(team_minutes.sum())
        team_sum_dev = abs(team_sum - 240.0) if active_count > 0 else 0.0
        diagnostic_rows.append(
            {
                "game_id": game_id,
                "team_id": team_id,
                "rows": int(len(idx)),
                "n_out": int(out_g.sum()),
                "n_starters": int(starter_g.sum()),
                "n_eligible": int(eligible_g.sum()),
                "active_count": active_count,
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
