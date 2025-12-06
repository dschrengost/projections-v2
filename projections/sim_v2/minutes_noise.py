from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from projections.features.availability import normalize_status
from projections.minutes_v1.constants import AvailabilityStatus
from projections.minutes_v1.production import resolve_production_run_dir
from projections.paths import data_path

MINUTES_P50_BIN_EDGES: tuple[float, ...] = (0.0, 8.0, 16.0, 24.0, 32.0, 40.0, float("inf"))


def _coerce_edges(edges: Sequence[float] | np.ndarray | None) -> tuple[float, ...]:
    if edges is None:
        return MINUTES_P50_BIN_EDGES
    arr = np.asarray(edges, dtype=float)
    if arr.ndim != 1 or arr.size < 2:
        return MINUTES_P50_BIN_EDGES
    return tuple(float(x) for x in arr)


def minutes_bin_indices(values: pd.Series | np.ndarray, *, edges: Sequence[float] | None = None) -> np.ndarray:
    edge_tuple = _coerce_edges(edges)
    edge_arr = np.asarray(edge_tuple, dtype=float)
    idx = np.digitize(values, edge_arr, right=False) - 1
    idx = np.clip(idx, 0, len(edge_arr) - 2)
    return idx.astype(int)


def bin_bounds(idx: int, *, edges: Sequence[float] | None = None) -> tuple[float, float]:
    edge_tuple = _coerce_edges(edges)
    if idx < 0:
        idx = 0
    if idx >= len(edge_tuple) - 1:
        idx = len(edge_tuple) - 2
    return float(edge_tuple[idx]), float(edge_tuple[idx + 1])


def status_bucket_from_raw(value: object) -> str:
    if value is None:
        return "healthy"
    if isinstance(value, AvailabilityStatus):
        enum_val = value
    else:
        try:
            enum_val = normalize_status(str(value))
        except Exception:
            enum_val = None
    if enum_val is not None:
        if enum_val == AvailabilityStatus.OUT:
            return "out"
        if enum_val == AvailabilityStatus.QUESTIONABLE:
            return "questionable"
        if enum_val == AvailabilityStatus.PROBABLE:
            return "probable"
        if enum_val == AvailabilityStatus.AVAILABLE:
            return "healthy"
        if enum_val == AvailabilityStatus.UNKNOWN:
            # fall through to text heuristics
            pass
    text = str(value).strip().lower()
    if not text:
        return "healthy"
    if "out" in text:
        return "out"
    if "prob" in text:
        return "probable"
    if any(token in text for token in ("q", "question", "doubt", "gtd")):
        return "questionable"
    if any(token in text for token in ("available", "healthy", "active", "cleared")):
        return "healthy"
    return "other"


def _resolve_minutes_run_id(explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit
    _, run_id = resolve_production_run_dir()
    if not run_id:
        raise RuntimeError("Unable to resolve minutes run_id from config/minutes_current_run.json")
    return str(run_id)


@dataclasses.dataclass
class MinutesNoiseParams:
    run_id: str
    sigma_min: float
    buckets: Dict[Tuple[int, str, int], float]
    bin_edges: tuple[float, ...] = dataclasses.field(default_factory=lambda: MINUTES_P50_BIN_EDGES)
    source_path: Path | None = None


def load_minutes_noise_params(
    data_root: Optional[Path] = None,
    minutes_run_id: Optional[str] = None,
) -> MinutesNoiseParams:
    root = data_root or data_path()
    run_id = _resolve_minutes_run_id(minutes_run_id)
    path = root / "artifacts" / "sim_v2" / "minutes_noise" / f"{run_id}_minutes_noise.json"
    if not path.exists():
        raise FileNotFoundError(f"Minutes noise params not found at {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    sigma_min = float(payload.get("sigma_min", 0.0) or 0.0)
    edges = _coerce_edges(payload.get("bin_edges"))
    buckets_raw = payload.get("buckets") or []
    buckets: Dict[Tuple[int, str, int], float] = {}
    for entry in buckets_raw:
        starter = int(entry.get("starter_flag", 0))
        status = str(entry.get("status_bucket", "unknown"))
        bin_idx = entry.get("p50_bin_idx")
        if bin_idx is None and "p50_bin" in entry:
            bounds = entry["p50_bin"]
            try:
                lo, hi = float(bounds[0]), float(bounds[1])
                # match bin edges if possible
                for i in range(len(edges) - 1):
                    if np.isclose(edges[i], lo) and np.isclose(edges[i + 1], hi):
                        bin_idx = i
                        break
            except Exception:
                bin_idx = None
        if bin_idx is None:
            bin_idx = 0
        buckets[(starter, status, int(bin_idx))] = float(entry.get("sigma", sigma_min))

    return MinutesNoiseParams(
        run_id=run_id,
        sigma_min=sigma_min,
        buckets=buckets,
        bin_edges=edges,
        source_path=path,
    )


def _resolve_minutes_series(df: pd.DataFrame, minutes_col: str) -> pd.Series:
    candidates = [minutes_col, "minutes_p50_cond", "minutes_p50", "minutes_pred_p50"]
    for col in candidates:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce")
            if not series.isna().all():
                return series
    raise KeyError(f"None of the minutes columns found: {candidates}")


def _status_series(df: pd.DataFrame, preferred: str | None = None) -> pd.Series:
    if preferred and preferred in df.columns:
        return df[preferred]
    for col in ("status_bucket", "status", "injury_status", "availability_status"):
        if col in df.columns:
            return df[col]
    return pd.Series("healthy", index=df.index, dtype=object)


def build_sigma_per_player(
    df: pd.DataFrame,
    params: MinutesNoiseParams,
    minutes_col: str = "minutes_p50",
    starter_col: str = "starter_flag",
    status_col: str = "status_bucket",
) -> np.ndarray:
    if df.empty:
        return np.array([], dtype=float)

    minutes_series = _resolve_minutes_series(df, minutes_col)
    p50_bin_idx = minutes_bin_indices(minutes_series.to_numpy(dtype=float), edges=params.bin_edges)

    starter_series = pd.to_numeric(df.get(starter_col, 0), errors="coerce").fillna(0).astype(int)
    status_series_raw = _status_series(df, preferred=status_col)
    status_bucket = status_series_raw.apply(status_bucket_from_raw).astype(str)

    keys_df = pd.DataFrame(
        {
            "starter_flag": starter_series.to_numpy(dtype=int),
            "status_bucket": status_bucket.to_numpy(dtype=object),
            "p50_bin_idx": p50_bin_idx,
        }
    )

    if params.buckets:
        bucket_df = pd.DataFrame(
            [
                {
                    "starter_flag": sf,
                    "status_bucket": sb,
                    "p50_bin_idx": idx,
                    "sigma": sig,
                }
                for (sf, sb, idx), sig in params.buckets.items()
            ]
        )
    else:
        bucket_df = pd.DataFrame(columns=["starter_flag", "status_bucket", "p50_bin_idx", "sigma"])

    merged = keys_df.merge(bucket_df, how="left", on=["starter_flag", "status_bucket", "p50_bin_idx"])

    missing = merged["sigma"].isna()
    if missing.any():
        healthy_df = bucket_df[bucket_df["status_bucket"] == "healthy"]
        if not healthy_df.empty:
            fallback = keys_df.loc[missing, ["starter_flag", "p50_bin_idx"]].merge(
                healthy_df,
                how="left",
                on=["starter_flag", "p50_bin_idx"],
            )
            merged.loc[missing, "sigma"] = fallback.get("sigma")

    still_missing = merged["sigma"].isna()
    if still_missing.any():
        healthy_by_starter: dict[int, pd.DataFrame] = {}
        for starter_value, group in bucket_df[bucket_df["status_bucket"] == "healthy"].groupby("starter_flag"):
            healthy_by_starter[int(starter_value)] = group
        for idx in np.flatnonzero(still_missing.to_numpy()):
            starter_value = int(keys_df.iloc[idx]["starter_flag"])
            bin_idx_value = int(keys_df.iloc[idx]["p50_bin_idx"])
            fallback_frame = healthy_by_starter.get(starter_value)
            if fallback_frame is None or fallback_frame.empty:
                continue
            distances = np.abs(fallback_frame["p50_bin_idx"].to_numpy(dtype=int) - bin_idx_value)
            nearest_pos = int(np.argmin(distances))
            merged.iat[idx, merged.columns.get_loc("sigma")] = float(fallback_frame.iloc[nearest_pos]["sigma"])

    merged["sigma"] = merged["sigma"].fillna(params.sigma_min)
    return merged["sigma"].to_numpy(dtype=float)


__all__ = [
    "MINUTES_P50_BIN_EDGES",
    "MinutesNoiseParams",
    "bin_bounds",
    "build_sigma_per_player",
    "enforce_team_240_minutes",
    "load_minutes_noise_params",
    "minutes_bin_indices",
    "status_bucket_from_raw",
]


def enforce_team_240_minutes(
    minutes_world: np.ndarray,
    team_indices: np.ndarray,
    rotation_mask: np.ndarray,
    bench_mask: np.ndarray,
    clamp_scale: tuple[float, float] = (0.7, 1.3),
) -> np.ndarray:
    """
    Rescale rotation players per team per world so totals approach 240 minutes.

    minutes_world: (W, P) non-negative minutes after noise
    team_indices:  (P,) int codes 0..T-1
    rotation_mask: (P,) bool for rotation players
    bench_mask:    (P,) bool for deep bench players
    clamp_scale:   allowable scaling range
    """

    if minutes_world.size == 0:
        return minutes_world

    mins = np.maximum(minutes_world, 0.0)
    team_idx = team_indices.astype(int)
    n_teams = int(team_idx.max()) + 1 if team_idx.size else 0
    if n_teams == 0:
        return mins

    team_one_hot = np.eye(n_teams, dtype=float)[team_idx]  # (P, T)
    rot_one_hot = team_one_hot * rotation_mask[:, None]
    bench_one_hot = team_one_hot * bench_mask[:, None]

    bench_sum = mins @ bench_one_hot  # (W, T)
    rot_sum = mins @ rot_one_hot      # (W, T)

    target_rot = np.clip(240.0 - bench_sum, a_min=0.0, a_max=None)
    scale = np.ones_like(target_rot)
    mask_nonzero = rot_sum > 1e-6
    scale[mask_nonzero] = target_rot[mask_nonzero] / rot_sum[mask_nonzero]
    scale = np.clip(scale, clamp_scale[0], clamp_scale[1])

    scale_per_player = scale[:, team_idx]  # (W, P)
    mins_scaled = mins * np.where(rotation_mask[None, :], scale_per_player, 1.0)
    return mins_scaled
