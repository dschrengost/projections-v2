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
    baseline_minutes: np.ndarray | None = None,
    clamp_scale: tuple[float, float] = (0.7, 1.3),
    active_mask: np.ndarray | None = None,
    starter_mask: np.ndarray | None = None,
    max_rotation_size: int | None = None,
) -> np.ndarray:
    """
    Rescale rotation players per team per world so totals approach 240 minutes.

    minutes_world: (W, P) non-negative minutes after noise
    team_indices:  (P,) int codes 0..T-1
    rotation_mask: (P,) bool for rotation players (based on projected minutes >= 12)
    bench_mask:    (P,) bool for deep bench players
    baseline_minutes: (P,) optional baseline minutes used to pick a stable
                     rotation when max_rotation_size is set. When provided,
                     the capped rotation uses baseline ordering instead of
                     sampled per-world minutes (reduces churn into the tail).
    clamp_scale:   allowable scaling range
    active_mask:   (W, P) bool indicating which players are active in each world.
                   If provided, only active players participate in reconciliation,
                   and inactive players' minutes flow to active teammates.
    starter_mask:  (P,) bool indicating starters (optional). Used to avoid
                   nerfing starter minutes when teams are oversubscribed.
    max_rotation_size: Optional hard cap on the number of players that can
                       receive minutes per team per world. When set, players
                       outside the capped rotation are zeroed out before scaling.
    """

    if minutes_world.size == 0:
        return minutes_world

    mins = np.maximum(minutes_world, 0.0)
    team_idx = team_indices.astype(int)
    n_teams = int(team_idx.max()) + 1 if team_idx.size else 0
    if n_teams == 0:
        return mins

    n_worlds, n_players = mins.shape

    if active_mask is not None:
        mins = mins * active_mask.astype(float)

    starter = (
        np.asarray(starter_mask, dtype=bool)
        if starter_mask is not None
        else np.zeros(n_players, dtype=bool)
    )

    if max_rotation_size is not None and max_rotation_size > 0:
        # Rotation-capped reconciliation (per-team, per-world):
        # 1) Keep all active starters + top-N non-starters by sampled minutes
        #    (or by baseline minutes if provided)
        # 2) Zero out minutes for players outside the capped rotation
        # 3) If team is oversubscribed (>240), scale non-starters down to fit while
        #    leaving starters unchanged (prevents starter nerf)
        # 4) If team is undersubscribed (<240), scale everyone up (clamped)

        out = mins.copy()
        team_to_players = [np.flatnonzero(team_idx == t) for t in range(n_teams)]
        full_active = active_mask is None

        for team_players in team_to_players:
            if team_players.size == 0:
                continue
            starter_local = starter[team_players]
            baseline_local = None
            if baseline_minutes is not None:
                baseline_local = np.asarray(baseline_minutes, dtype=float)[team_players]
            for w in range(n_worlds):
                if full_active:
                    active_local = np.ones(team_players.size, dtype=bool)
                else:
                    active_local = active_mask[w, team_players].astype(bool)
                    if not active_local.any():
                        out[w, team_players] = 0.0
                        continue

                desired = out[w, team_players]
                if baseline_local is None:
                    candidate_local = active_local & ((desired > 0.0) | starter_local)
                else:
                    candidate_local = active_local & ((baseline_local > 0.0) | starter_local)
                if not candidate_local.any():
                    out[w, team_players] = 0.0
                    continue

                starter_keep = candidate_local & starter_local
                keep_local = starter_keep.copy()
                starter_count = int(starter_keep.sum())
                slots_left = max_rotation_size - starter_count
                if slots_left > 0:
                    nonstarter_candidates = np.flatnonzero(candidate_local & ~starter_local)
                    if nonstarter_candidates.size:
                        scores = (
                            baseline_local
                            if baseline_local is not None
                            else desired
                        )
                        order = nonstarter_candidates[np.argsort(-scores[nonstarter_candidates], kind="mergesort")]
                        if baseline_local is None:
                            keep_local[order[:slots_left]] = True
                        else:
                            # Prefer bench players with positive sampled minutes to avoid
                            # creating short team totals due to clamp_scale when a top
                            # baseline bench player's sample hits 0.0 in a given world.
                            positive_mask = desired[order] > 1e-9
                            take = order[positive_mask][:slots_left]
                            if take.size < slots_left:
                                take = np.concatenate([take, order[~positive_mask][: slots_left - take.size]])
                            keep_local[take] = True
                # Never drop starters due to the cap: if starters exceed max_rotation_size,
                # keep them all and only cap the non-starters.

                # Zero out active players not in the kept rotation.
                drop_local = active_local & ~keep_local
                if drop_local.any():
                    out[w, team_players[drop_local]] = 0.0
                    desired = out[w, team_players]  # refresh view after drops

                kept = keep_local & active_local
                if not kept.any():
                    continue

                starter_kept = kept & starter_local
                bench_kept = kept & ~starter_local
                sum_starters = float(desired[starter_kept].sum())
                sum_bench = float(desired[bench_kept].sum())
                sum_total = sum_starters + sum_bench
                if sum_total <= 1e-6:
                    continue

                if sum_total >= 240.0:
                    if sum_bench > 1e-6 and sum_starters < 240.0:
                        bench_scale = (240.0 - sum_starters) / sum_bench
                        bench_scale = max(0.0, float(bench_scale))
                        out[w, team_players[bench_kept]] = desired[bench_kept] * bench_scale
                        # Starters unchanged when oversubscribed.
                    else:
                        # No bench to shrink (or starters already exceed 240): scale starters down.
                        starter_scale = 240.0 / max(sum_starters, 1e-6)
                        starter_scale = float(np.clip(starter_scale, clamp_scale[0], 1.0))
                        out[w, team_players[starter_kept]] = desired[starter_kept] * starter_scale
                        out[w, team_players[bench_kept]] = 0.0
                else:
                    scale = 240.0 / sum_total
                    scale = float(np.clip(scale, 1.0, clamp_scale[1]))
                    out[w, team_players[kept]] = desired[kept] * scale

        # Ensure inactive players stay at 0 if active_mask was provided.
        if active_mask is not None:
            out = out * active_mask.astype(float)
        return out

    # Legacy behavior: scale rotation minutes proportionally, leaving bench fixed.
    # If active_mask provided, determine rotation/bench per-world based on who is active.
    if active_mask is not None:
        per_world_rotation = active_mask & (rotation_mask[None, :] | (mins >= 12.0))
        per_world_bench = active_mask & ~per_world_rotation & (mins > 0.0)
    else:
        per_world_rotation = np.broadcast_to(rotation_mask[None, :], (n_worlds, n_players))
        per_world_bench = np.broadcast_to(bench_mask[None, :], (n_worlds, n_players))

    team_one_hot = np.eye(n_teams, dtype=float)[team_idx]  # (P, T)

    # Compute per-world, per-team sums for rotation and bench
    # rot_one_hot_world: (W, P, T) - which players count as rotation per world
    # We need to sum mins where player is rotation for each team

    # For efficiency, compute team totals using matrix ops
    # bench_sum[w, t] = sum of mins[w, p] where per_world_bench[w, p] and team[p] == t
    # rot_sum[w, t] = sum of mins[w, p] where per_world_rotation[w, p] and team[p] == t

    bench_mins = mins * per_world_bench.astype(float)  # (W, P)
    rot_mins = mins * per_world_rotation.astype(float)  # (W, P)

    bench_sum = bench_mins @ team_one_hot  # (W, T)
    rot_sum = rot_mins @ team_one_hot      # (W, T)

    target_rot = np.clip(240.0 - bench_sum, a_min=0.0, a_max=None)
    scale = np.ones_like(target_rot)
    mask_nonzero = rot_sum > 1e-6
    scale[mask_nonzero] = target_rot[mask_nonzero] / rot_sum[mask_nonzero]
    scale = np.clip(scale, clamp_scale[0], clamp_scale[1])

    scale_per_player = scale[:, team_idx]  # (W, P)

    # Only scale rotation players (per-world determination)
    mins_scaled = mins * np.where(per_world_rotation, scale_per_player, 1.0)

    # Ensure inactive players stay at 0 if active_mask was provided
    if active_mask is not None:
        mins_scaled = mins_scaled * active_mask.astype(float)

    return mins_scaled
