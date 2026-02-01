from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HumilityConfig:
    enabled: bool = True
    minutes_p50_fringe_max: float = 8.0
    minutes_p50_bench_max: float = 14.0
    cap_p_ge5_fringe: float = 0.35
    cap_p_ge5_bench: float = 0.65
    cap_play_prob_fringe: float = 0.60
    min_p_eq0_fringe: float = 0.25
    top_n_lock: int = 8
    protect_starters: bool = True
    protect_top_n: bool = True
    seed: int = 0


def _as_float_series(df: pd.DataFrame, col: str, *, default: float) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.full(len(df), default, dtype=np.float64), index=df.index, dtype=np.float64)
    return pd.to_numeric(df[col], errors="coerce").astype(np.float64).fillna(default)


def _as_bool_series(df: pd.DataFrame, col: str, *, default: bool) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.full(len(df), bool(default), dtype=bool), index=df.index, dtype=bool)
    s = df[col]
    if s.dtype == bool:
        return s.fillna(default).astype(bool)
    if str(s.dtype).startswith("boolean"):
        return s.fillna(default).astype(bool)
    return (
        s.fillna(default)
        .astype("string")
        .str.strip()
        .str.lower()
        .isin(["1", "true", "t", "yes", "y"])
        .astype(bool)
    )


def _ensure_quantile_order(p10: float, p50: float, p90: float) -> tuple[float, float, float]:
    a = float(p10)
    b = float(p50)
    c = float(p90)
    lo = min(a, b, c)
    hi = max(a, b, c)
    mid = float(np.clip(b, lo, hi))
    return float(max(lo, 0.0)), float(max(mid, 0.0)), float(max(hi, 0.0))


def _implied_p_ge5_from_p10_p90(*, p10: float, p50: float, p90: float) -> float:
    """Deterministic proxy for P(minutes>=5) using (p10,p90) with linear interpolation.

    Assumptions:
    - The CDF between p10 and p90 is approximately linear from 0.1 to 0.9.
    - We clamp into [0.1, 0.9] outside that interval.
    """
    p10, p50, p90 = _ensure_quantile_order(p10, p50, p90)
    eps = 1e-9
    if p90 <= p10 + eps:
        return 1.0 if p50 >= 5.0 else 0.0
    if 5.0 <= p10:
        return 0.9
    if 5.0 >= p90:
        return 0.1
    cdf_5 = 0.1 + 0.8 * ((5.0 - p10) / (p90 - p10))
    return float(np.clip(1.0 - cdf_5, 0.0, 1.0))


def _max_scale_for_p_ge5_cap(
    *,
    p10: float,
    p50: float,
    p90: float,
    cap: float,
    scale_upper: float,
    enabled: bool,
) -> float:
    if not enabled:
        return float(scale_upper)
    cap = float(np.clip(cap, 0.0, 1.0))
    if cap >= 1.0:
        return float(scale_upper)
    if scale_upper <= 0.0:
        return 0.0

    def implied(scale: float) -> float:
        return _implied_p_ge5_from_p10_p90(p10=float(scale) * p10, p50=float(scale) * p50, p90=float(scale) * p90)

    if implied(float(scale_upper)) <= cap:
        return float(scale_upper)

    lo = 0.0
    hi = float(scale_upper)
    for _ in range(28):
        mid = (lo + hi) / 2.0
        if implied(mid) <= cap:
            lo = mid
        else:
            hi = mid
    return float(lo)


def load_humility_config_json(path: Path, *, base: HumilityConfig | None = None) -> HumilityConfig:
    """Load a JSON object and apply as overrides on top of `base` (default: HumilityConfig())."""
    base_cfg = base or HumilityConfig()
    overrides = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(overrides, dict):
        raise ValueError("Humility config JSON must be an object")
    allowed = set(base_cfg.__dataclass_fields__.keys())
    unknown = sorted(set(overrides.keys()) - allowed)
    if unknown:
        raise ValueError(f"Unknown humility config keys: {unknown}. Allowed: {sorted(allowed)}")
    return replace(base_cfg, **overrides)


def apply_prior_humility(df_priors: pd.DataFrame, cfg: HumilityConfig) -> pd.DataFrame:
    """Transform raw priors into guardrailed priors for rotation generation.

    Adds (new, additive) columns:
    - minutes_prior_adj, minutes_p10_adj, minutes_p90_adj, play_prob_adj
    - humility_tier in {starter, top_n, core, bench, fringe}
    - humility_reason (short string codes; '|' delimited)
    """
    required = {"game_id", "team_id", "player_id", "minutes_prior"}
    missing = sorted([c for c in required if c not in df_priors.columns])
    if missing:
        raise ValueError(f"df_priors missing required columns: {missing}")

    has_minutes_p10_p90 = ("minutes_p10" in df_priors.columns) and ("minutes_p90" in df_priors.columns)

    df = df_priors.copy()
    df["game_id"] = df["game_id"].astype("string")
    df["team_id"] = pd.to_numeric(df["team_id"], errors="coerce").astype("Int64")
    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["game_id", "team_id", "player_id"]).copy()
    df["team_id"] = df["team_id"].astype(int)
    df["player_id"] = df["player_id"].astype(int)

    minutes_p50 = _as_float_series(df, "minutes_p50", default=np.nan)
    minutes_prior = _as_float_series(df, "minutes_prior", default=0.0).clip(lower=0.0)
    minutes_p50 = minutes_p50.fillna(minutes_prior).astype(np.float64).clip(lower=0.0)

    minutes_p10 = _as_float_series(df, "minutes_p10", default=np.nan)
    minutes_p90 = _as_float_series(df, "minutes_p90", default=np.nan)
    minutes_p10 = minutes_p10.fillna(minutes_prior).astype(np.float64).clip(lower=0.0)
    minutes_p90 = minutes_p90.fillna(minutes_prior).astype(np.float64).clip(lower=0.0)

    play_prob = _as_float_series(df, "play_prob", default=1.0).clip(0.0, 1.0)

    n = len(df)
    tier = np.full(n, "core", dtype=object)
    reason_parts: list[list[str]] = [[] for _ in range(n)]

    # Tiering: starters
    starter_mask = np.zeros(n, dtype=bool)
    if bool(cfg.protect_starters) and "starter_candidate" in df.columns:
        starter_mask = _as_bool_series(df, "starter_candidate", default=False).to_numpy(dtype=bool)
        tier[starter_mask] = "starter"
        for i in np.where(starter_mask)[0].tolist():
            reason_parts[i].append("protect_starter")

    # Tiering: top-N by minutes_p50 within (game_id, team_id).
    if bool(cfg.protect_top_n) and int(cfg.top_n_lock) > 0:
        tmp = df[["game_id", "team_id", "player_id"]].copy()
        tmp["_minutes_p50"] = minutes_p50.to_numpy(dtype=np.float64)
        tmp = tmp.sort_values(
            ["game_id", "team_id", "_minutes_p50", "player_id"],
            ascending=[True, True, False, True],
            kind="mergesort",
        )
        tmp["_rank"] = tmp.groupby(["game_id", "team_id"], sort=False).cumcount() + 1
        top_idx = tmp.index[(tmp["_rank"] <= int(cfg.top_n_lock)).to_numpy(dtype=bool)]
        # Honor starter protection first (else-if semantics).
        top_idx = [int(i) for i in top_idx.tolist() if not bool(starter_mask[int(i)])]
        if top_idx:
            tier[np.asarray(top_idx, dtype=int)] = "top_n"
            for i in top_idx:
                reason_parts[int(i)].append("protect_top_n")

    # Remaining tiers by minutes_p50.
    remaining = ~np.isin(tier, ["starter", "top_n"])
    p50_arr = minutes_p50.to_numpy(dtype=np.float64)
    core_mask = remaining & (p50_arr >= float(cfg.minutes_p50_bench_max))
    bench_mask = remaining & (p50_arr >= float(cfg.minutes_p50_fringe_max)) & (p50_arr < float(cfg.minutes_p50_bench_max))
    fringe_mask = remaining & (p50_arr < float(cfg.minutes_p50_fringe_max))
    tier[core_mask] = "core"
    tier[bench_mask] = "bench"
    tier[fringe_mask] = "fringe"

    # Prepare adjusted outputs.
    p10_adj = np.zeros(n, dtype=np.float64)
    p50_adj = np.zeros(n, dtype=np.float64)
    p90_adj = np.zeros(n, dtype=np.float64)
    play_prob_adj = play_prob.to_numpy(dtype=np.float64).copy()

    p10_arr = minutes_p10.to_numpy(dtype=np.float64)
    p50_base_arr = minutes_p50.to_numpy(dtype=np.float64)
    p90_arr = minutes_p90.to_numpy(dtype=np.float64)

    for i in range(n):
        p10_i, p50_i, p90_i = _ensure_quantile_order(p10_arr[i], p50_base_arr[i], p90_arr[i])
        t = str(tier[i])

        if not bool(cfg.enabled):
            p10_adj[i], p50_adj[i], p90_adj[i] = p10_i, p50_i, p90_i
            continue

        if t in {"starter", "top_n", "core"}:
            p10_adj[i], p50_adj[i], p90_adj[i] = p10_i, p50_i, p90_i
            continue

        if t == "bench":
            shrink = 0.85
            max_p50 = float(cfg.minutes_p50_bench_max)
            cap = float(cfg.cap_p_ge5_bench)
        else:
            shrink = 0.60
            max_p50 = float(cfg.minutes_p50_fringe_max)
            cap = float(cfg.cap_p_ge5_fringe)

        scale_upper = float(shrink)
        if p50_i > 0.0:
            scale_upper = min(scale_upper, max_p50 / p50_i)
        else:
            scale_upper = min(scale_upper, 1.0)
        scale_upper = float(np.clip(scale_upper, 0.0, 1.0))

        scale = scale_upper
        if has_minutes_p10_p90:
            scale = _max_scale_for_p_ge5_cap(
                p10=p10_i,
                p50=p50_i,
                p90=p90_i,
                cap=cap,
                scale_upper=scale_upper,
                enabled=True,
            )
            if scale < scale_upper - 1e-9:
                reason_parts[i].append("cap_ge5")

        if scale < 1.0 - 1e-9:
            reason_parts[i].append("shrink")

        p10_j, p50_j, p90_j = _ensure_quantile_order(scale * p10_i, scale * p50_i, scale * p90_i)
        p10_adj[i], p50_adj[i], p90_adj[i] = p10_j, p50_j, p90_j

        if t == "fringe":
            pp = float(play_prob_adj[i])
            pp_cap = min(float(cfg.cap_play_prob_fringe), 1.0 - float(cfg.min_p_eq0_fringe))
            pp_new = float(min(pp, pp_cap))
            if pp_new < pp - 1e-12:
                play_prob_adj[i] = pp_new
                if pp_new <= float(cfg.cap_play_prob_fringe) + 1e-12:
                    reason_parts[i].append("cap_play_prob")
                if pp_new <= (1.0 - float(cfg.min_p_eq0_fringe)) + 1e-12:
                    reason_parts[i].append("min_p_eq0")

    humility_reason = ["|".join(parts) if parts else "" for parts in reason_parts]

    df["minutes_prior_adj"] = p50_adj.astype(np.float64)
    df["minutes_p10_adj"] = p10_adj.astype(np.float64)
    df["minutes_p90_adj"] = p90_adj.astype(np.float64)
    df["play_prob_adj"] = np.clip(play_prob_adj.astype(np.float64), 0.0, 1.0)
    df["humility_tier"] = pd.Series(tier, index=df.index, dtype="string")
    df["humility_reason"] = pd.Series(humility_reason, index=df.index, dtype="string")

    return df


def humility_config_as_dict(cfg: HumilityConfig) -> dict[str, Any]:
    # Stable for JSON/manifest.
    out = asdict(cfg)
    out["top_n_lock"] = int(out["top_n_lock"])
    out["seed"] = int(out["seed"])
    return out

