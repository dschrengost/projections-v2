from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from projections.sim_v2.minutes_physics import compute_rotation_lock_mask


@dataclass(frozen=True)
class PlayProbPolicyDiagnostics:
    enabled: bool
    n_players: int
    n_rotation_locks: int
    n_changed: int
    reasons: dict[str, int]
    play_prob_raw_summary: dict[str, float]
    play_prob_eff_summary: dict[str, float]
    max_delta: float


def _first_present(columns: Iterable[str], available: set[str]) -> str | None:
    for col in columns:
        if col in available:
            return col
    return None


def _safe_numeric(series: pd.Series, *, default: float = 0.0) -> np.ndarray:
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    arr = np.nan_to_num(arr, nan=float(default))
    return arr


def _summary(values: np.ndarray) -> dict[str, float]:
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return {"mean": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0}
    return {
        "mean": float(np.mean(v)),
        "p10": float(np.percentile(v, 10)),
        "p50": float(np.percentile(v, 50)),
        "p90": float(np.percentile(v, 90)),
    }


def _normalize_text(values: pd.Series | None) -> np.ndarray:
    if values is None:
        return np.array([], dtype=object)
    s = values.astype(str)
    s = s.where(~values.isna(), "")
    return s.str.strip().str.lower().to_numpy(dtype=object)


def _build_group_map(df: pd.DataFrame) -> dict[Any, np.ndarray]:
    """Return deterministic group_map keyed by (game_id, team_id) or team_id."""

    if df.empty:
        return {}
    cols = set(df.columns)
    if {"game_id", "team_id"}.issubset(cols):
        game_id = pd.to_numeric(df["game_id"], errors="coerce").fillna(-1).astype(int).to_numpy()
        team_id = pd.to_numeric(df["team_id"], errors="coerce").fillna(-1).astype(int).to_numpy()
        keys = list(zip(game_id.tolist(), team_id.tolist()))
    elif "team_id" in cols:
        team_id = pd.to_numeric(df["team_id"], errors="coerce").fillna(-1).astype(int).to_numpy()
        keys = team_id.tolist()
    else:
        # Single group fallback.
        return {"all": np.arange(len(df), dtype=int)}

    group_map: dict[Any, list[int]] = {}
    for idx, key in enumerate(keys):
        group_map.setdefault(key, []).append(idx)
    # Sort keys for determinism.
    return {k: np.asarray(group_map[k], dtype=int) for k in sorted(group_map.keys())}


def _resolve_cond_p50(df: pd.DataFrame) -> tuple[str | None, np.ndarray]:
    cols = set(df.columns)
    col = _first_present(
        (
            "cond_minutes_p50",
            "minutes_p50_cond",
            "baseline_minutes_p50",
            "minutes_p50",
            "minutes_mean",
        ),
        cols,
    )
    if col is None:
        return None, np.zeros(len(df), dtype=float)
    return col, _safe_numeric(df[col], default=0.0)


def _resolve_starter_flag(df: pd.DataFrame) -> tuple[str | None, np.ndarray]:
    cols = set(df.columns)
    col = _first_present(
        (
            "starter_flag",
            "is_confirmed_starter",
            "is_projected_starter",
            "is_starter",
        ),
        cols,
    )
    if col is None:
        return None, np.zeros(len(df), dtype=bool)
    arr = _safe_numeric(df[col], default=0.0)
    return col, (arr > 0.5)


def _resolve_rotation_prob(df: pd.DataFrame) -> tuple[str | None, np.ndarray]:
    cols = set(df.columns)
    col = _first_present(("rotation_prob", "p_rot"), cols)
    if col is None:
        return None, np.zeros(len(df), dtype=float)
    arr = _safe_numeric(df[col], default=0.0)
    return col, np.clip(arr, 0.0, 1.0)


def _numeric_feature(df: pd.DataFrame, name: str) -> np.ndarray:
    if name not in df.columns:
        return np.zeros(len(df), dtype=float)
    return _safe_numeric(df[name], default=0.0)


def _apply_bounded_floor(
    *,
    p_eff: np.ndarray,
    p_raw: np.ndarray,
    mask: np.ndarray,
    floor: float,
    max_floor_delta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply `max(p_eff, floor)` on mask with an optional per-player max uplift cap."""

    if p_eff.size == 0 or not np.any(mask):
        return p_eff, np.zeros_like(mask, dtype=bool)

    out = p_eff.copy()
    target = np.maximum(out[mask], float(floor))
    if float(max_floor_delta) >= 0.0:
        target = np.minimum(target, p_raw[mask] + float(max_floor_delta))
    target = np.clip(target, 0.0, 1.0)
    changed = target > (out[mask] + 1e-12)
    out[mask] = target
    changed_full = np.zeros_like(mask, dtype=bool)
    changed_full[np.where(mask)[0]] = changed
    return out, changed_full


def apply_play_prob_policy_with_diagnostics(
    players_df: pd.DataFrame,
    config: Any,
    asof_ts: object | None = None,
    lock_ts: object | None = None,
) -> tuple[pd.DataFrame, PlayProbPolicyDiagnostics]:
    """Apply a play_prob policy layer for simulation availability sampling.

    Adds columns:
      - play_prob_raw: original play_prob (clipped to [0, 1])
      - play_prob_eff: effective play_prob used by sim
      - rotation_lock: heuristic rotation-lock boolean
      - play_prob_policy_reason: auditing string (single best-effort reason)

    Notes:
    - This is intentionally deterministic (no RNG).
    - Freshness gating is optional and best-effort: if the necessary timestamp
      columns are missing, the "fresh snapshot" requirement will not pass.
    """

    df = players_df.copy()
    cols = set(df.columns)

    enabled = bool(getattr(config, "enabled", False))
    if "play_prob" in cols:
        p_raw = _safe_numeric(df["play_prob"], default=0.0)
    elif "play_prob_raw" in cols:
        p_raw = _safe_numeric(df["play_prob_raw"], default=0.0)
    else:
        p_raw = np.ones(len(df), dtype=float)
    p_raw = np.clip(p_raw, 0.0, 1.0)

    if not enabled or df.empty:
        df["play_prob_raw"] = p_raw
        df["play_prob_eff"] = p_raw
        df["rotation_lock"] = False
        df["play_prob_policy_reason"] = "disabled" if not enabled else "empty"
        return (
            df,
            PlayProbPolicyDiagnostics(
                enabled=enabled,
                n_players=int(len(df)),
                n_rotation_locks=0,
                n_changed=0,
                reasons={},
                play_prob_raw_summary=_summary(p_raw),
                play_prob_eff_summary=_summary(p_raw),
                max_delta=0.0,
            ),
        )

    # Rotation lock heuristic uses only sim-input-available columns.
    cond_col, cond_p50 = _resolve_cond_p50(df)
    _starter_col, is_starter = _resolve_starter_flag(df)
    group_map = _build_group_map(df)
    rotation_lock = compute_rotation_lock_mask(
        baseline_minutes=cond_p50,
        is_starter=is_starter if is_starter.size else None,
        group_map=group_map,
        top_k=int(getattr(config, "rotation_lock_topk", 8)),
        minutes_threshold=float(getattr(config, "rotation_lock_min_cond_p50", 18.0)),
    )

    status_bucket = None
    if "status_bucket" in cols:
        status_bucket = _normalize_text(df["status_bucket"])
    else:
        # Best-effort: try common raw status columns.
        raw_col = _first_present(("status", "injury_status", "availability_status"), cols)
        if raw_col is not None:
            status_bucket = _normalize_text(df[raw_col])

    raw_status_col = _first_present(("injury_status", "status", "availability_status"), cols)
    if raw_status_col is not None:
        status_text = _normalize_text(df[raw_status_col])
    else:
        # Conservative fallback: reuse bucket text so we do not accidentally treat all rows
        # as "not listed" just because the raw status column was not passed through.
        status_text = status_bucket if status_bucket is not None and status_bucket.size else np.array([""] * len(df), dtype=object)

    # Identify buckets / flags.
    out_like = np.zeros(len(df), dtype=bool)
    probable_like = np.zeros(len(df), dtype=bool)
    if status_bucket is not None and status_bucket.size:
        out_like |= status_bucket == "out"
        probable_like |= status_bucket == "probable"
        # "questionable" is intentionally not treated as OUT; it should keep DNP risk.

    # Text heuristics for other "no-play" statuses.
    out_like |= np.char.find(status_text.astype(str), "inactive") >= 0
    out_like |= np.char.find(status_text.astype(str), "susp") >= 0

    # "Not listed" semantics: in our pipeline this is usually represented as "Ava"/"AVAIL".
    # We treat these as healthy (eligible for rotation-lock flooring).
    if raw_status_col is None:
        # With no raw status, "healthy" is our best approximation for "not listed".
        not_listed_like = (
            (status_bucket == "healthy")
            if status_bucket is not None and status_bucket.size
            else np.zeros(len(df), dtype=bool)
        )
    else:
        not_listed_like = np.isin(
            status_text,
            np.array(["", "ava", "avail", "available", "active", "cleared"], dtype=object),
        )
        # Also recognize explicit "not listed" strings when present.
        not_listed_like |= np.char.find(status_text.astype(str), "not listed") >= 0
        not_listed_like |= np.char.find(status_text.astype(str), "not_listed") >= 0

    # Optional freshness gating (best-effort; off by default).
    fresh_ok = np.ones(len(df), dtype=bool)
    if bool(getattr(config, "require_fresh_injury_snapshot", False)):
        fresh_ok[:] = False
        freshness_minutes = float(getattr(config, "freshness_minutes", 90.0))
        freshness_ns = int(max(0.0, freshness_minutes) * 60.0 * 1e9)

        # Prefer per-row tip_ts when present; otherwise fall back to passed lock_ts/asof_ts.
        ref = None
        if "tip_ts" in cols:
            ref = pd.to_datetime(df["tip_ts"], utc=True, errors="coerce")
        elif lock_ts is not None:
            ref = pd.Series([pd.to_datetime(lock_ts, utc=True, errors="coerce")] * len(df))
        elif asof_ts is not None:
            ref = pd.Series([pd.to_datetime(asof_ts, utc=True, errors="coerce")] * len(df))

        snap_col = _first_present(("injury_as_of_ts", "as_of_ts"), cols)
        if ref is not None and snap_col is not None:
            snap = pd.to_datetime(df[snap_col], utc=True, errors="coerce")
            # Ref and snapshot must exist and be close enough.
            delta = (ref - snap).dt.total_seconds().to_numpy(dtype=float) * 1e9
            fresh_ok = np.isfinite(delta) & (delta >= 0.0) & (delta <= float(freshness_ns))

    p_eff = p_raw.copy()
    reasons = np.full(len(df), "raw", dtype=object)

    # Rule (a): OUT/INACTIVE/SUSPENDED => 0.0
    p_eff = np.where(out_like, 0.0, p_eff)
    reasons = np.where(out_like, "out_like", reasons)

    policy_mode = str(getattr(config, "mode", "guarded_v2")).strip().lower()
    if policy_mode not in {"legacy", "guarded_v2"}:
        policy_mode = "guarded_v2"
    # "legacy" is intentionally treated as guarded_v2 to prevent broad rotation-lock
    # flooring from forcing fringe players into near-certain availability.
    if policy_mode == "legacy":
        policy_mode = "guarded_v2"

    # Guarded v2:
    # - protect starters/core locks from under-confident p_play
    # - never floor players with strong depth/DNP risk
    # - bound floor uplift by max_floor_delta to avoid p_play jumps
    _rotation_col, rotation_prob = _resolve_rotation_prob(df)
    raw_min = float(np.clip(float(getattr(config, "min_raw_play_prob_for_floor", 0.35)), 0.0, 1.0))
    rot_min = float(np.clip(float(getattr(config, "min_rotation_prob_for_floor", 0.65)), 0.0, 1.0))
    max_floor_delta = float(getattr(config, "max_floor_delta", 0.25))

    dc_role = (
        _normalize_text(df["dc_role"])
        if "dc_role" in cols
        else np.full(len(df), "", dtype=object)
    )
    depth_block_roles_raw = getattr(config, "depth_block_roles", ("limited", "not_listed"))
    depth_block_roles = np.array(
        [str(v).strip().lower() for v in depth_block_roles_raw if str(v).strip()],
        dtype=object,
    )
    depth_block = (
        np.isin(dc_role, depth_block_roles)
        if depth_block_roles.size > 0
        else np.zeros(len(df), dtype=bool)
    )
    ahead = _numeric_feature(df, "dc_ahead_global")
    depth_block |= ahead >= float(getattr(config, "depth_block_min_ahead_global", 7))

    streak = _numeric_feature(df, "consecutive_active_dnp")
    dnp_rate = _numeric_feature(df, "active_but_dnp_rate_last10")
    inactive_streak = _numeric_feature(df, "inactive_streak_len")
    dnp_block = streak >= float(getattr(config, "dnp_block_streak_threshold", 3.0))
    dnp_block |= dnp_rate >= float(getattr(config, "dnp_block_rate_threshold", 0.50))
    dnp_block |= inactive_streak >= float(getattr(config, "dnp_block_inactive_streak_threshold", 2.0))

    risk_block = depth_block | dnp_block
    base_eligible = not_listed_like & fresh_ok & ~out_like & ~risk_block & (p_raw >= raw_min)
    rot_ok = rotation_prob >= rot_min

    # Starter lock floor (keeps unconditionals sane for clear starters).
    starter_floor = float(np.clip(float(getattr(config, "starter_floor", 0.995)), 0.0, 1.0))
    starter_mask = base_eligible & is_starter
    p_eff, changed_starter = _apply_bounded_floor(
        p_eff=p_eff,
        p_raw=p_raw,
        mask=starter_mask,
        floor=starter_floor,
        max_floor_delta=max_floor_delta,
    )
    reasons = np.where(changed_starter, "starter_floor_guarded_v2", reasons)

    # Core-lock floor (non-starters only; stricter than legacy top-8 heuristic).
    core_lock = compute_rotation_lock_mask(
        baseline_minutes=cond_p50,
        is_starter=is_starter if is_starter.size else None,
        group_map=group_map,
        top_k=int(getattr(config, "core_lock_topk", 5)),
        minutes_threshold=float(getattr(config, "core_lock_min_cond_p50", 24.0)),
    )
    core_floor = float(np.clip(float(getattr(config, "core_floor", 0.90)), 0.0, 1.0))
    core_mask = base_eligible & (~is_starter) & core_lock & rot_ok
    p_eff, changed_core = _apply_bounded_floor(
        p_eff=p_eff,
        p_raw=p_raw,
        mask=core_mask,
        floor=core_floor,
        max_floor_delta=max_floor_delta,
    )
    reasons = np.where(changed_core, "core_floor_guarded_v2", reasons)

    blocked_mask = rotation_lock & not_listed_like & fresh_ok & ~out_like & risk_block
    reasons = np.where((reasons == "raw") & blocked_mask, "raw_blocked_depth_dnp", reasons)

    # Rule (c): PROBABLE => floor, but keep guarded blockers (depth/DNP/freshness).
    prob_floor = float(np.clip(float(getattr(config, "probable_floor", 0.90)), 0.0, 1.0))
    prob_mask = probable_like & ~out_like & fresh_ok & ~risk_block
    before = p_eff.copy()
    p_eff = np.where(prob_mask, np.maximum(p_eff, prob_floor), p_eff)
    bumped_prob = (p_eff > before) & prob_mask
    reasons = np.where(bumped_prob, "probable_floor", reasons)

    df["play_prob_raw"] = p_raw
    df["play_prob_eff"] = p_eff
    df["rotation_lock"] = rotation_lock
    df["play_prob_policy_reason"] = reasons.astype(str)

    # Summarize reason counts and deltas.
    unique, counts = np.unique(reasons.astype(str), return_counts=True)
    reason_counts = {str(k): int(v) for k, v in zip(unique.tolist(), counts.tolist())}

    delta = p_eff - p_raw
    n_changed = int((np.abs(delta) > 1e-12).sum())

    return (
        df,
        PlayProbPolicyDiagnostics(
            enabled=True,
            n_players=int(len(df)),
            n_rotation_locks=int(rotation_lock.sum()),
            n_changed=n_changed,
            reasons=reason_counts,
            play_prob_raw_summary=_summary(p_raw),
            play_prob_eff_summary=_summary(p_eff),
            max_delta=float(np.max(delta)) if delta.size else 0.0,
        ),
    )


def apply_play_prob_policy(
    players_df: pd.DataFrame,
    config: Any,
    asof_ts: object | None = None,
    lock_ts: object | None = None,
) -> pd.DataFrame:
    """Apply policy and return the mutated frame (no diagnostics)."""

    out, _diag = apply_play_prob_policy_with_diagnostics(
        players_df=players_df,
        config=config,
        asof_ts=asof_ts,
        lock_ts=lock_ts,
    )
    return out


__all__ = [
    "PlayProbPolicyDiagnostics",
    "apply_play_prob_policy",
    "apply_play_prob_policy_with_diagnostics",
]
