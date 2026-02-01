from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GateConfig:
    """Non-structural gating layer to suppress fringe bench minute explosions under chaos.

    This layer is intentionally conservative:
    - Protect starters and top-N by minutes prior
    - Trust p_ge15 more than p_ge5 for "real rotation" protection
    - Cap low-probability fringe player minutes/play_prob (never excludes players)
    """

    enabled: bool = False

    protect_starters: bool = True
    protect_top_n: bool = True
    top_n_lock: int = 8

    core_ge15_min: float = 0.35
    core_minutes_prior_min: float = 20.0

    bench_ge5_min: float = 0.35
    bench_minutes_prior_min: float = 10.0

    bench_minutes_cap: float = 14.0
    fringe_minutes_cap: float = 6.0
    fringe_play_prob_cap: float = 0.70

    # Deprecated/ignored: gate is non-structural (no exclusions).
    hard_exclude_ge5: float = 0.05
    hard_exclude_minutes_prior: float = 4.0
    min_team_minutes_prior_sum: float = 200.0
    seed: int = 0


def gate_config_as_dict(cfg: GateConfig) -> dict[str, Any]:
    return asdict(cfg)


def load_gate_config_json(path: Path, *, base: GateConfig | None = None) -> GateConfig:
    base_cfg = base or GateConfig()
    overrides = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(overrides, dict):
        raise ValueError("Gate config JSON must be an object")
    allowed = set(base_cfg.__dataclass_fields__.keys())
    unknown = sorted(set(overrides.keys()) - allowed)
    if unknown:
        raise ValueError(f"Unknown gate config keys: {unknown}. Allowed: {sorted(allowed)}")
    return replace(base_cfg, **overrides)


def _as_float_series(df: pd.DataFrame, col: str, *, default: float) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.full(len(df), default, dtype=np.float64), index=df.index, dtype=np.float64)
    return pd.to_numeric(df[col], errors="coerce").astype(np.float64).fillna(default)


def apply_rotation_gate(
    df_priors: pd.DataFrame,
    preds_df: pd.DataFrame,
    *,
    starters_set: set[int],
    cfg: GateConfig,
    seed: int,
) -> pd.DataFrame:
    """Apply rotation-gate diagnostics + caps for a single (game_id, team_id).

    Inputs:
    - df_priors: one team-game slice with at least columns:
        game_id(str), team_id(int), player_id(int internal), minutes_prior(float), play_prob(float)
      Optional: minutes_p10/minutes_p90.
    - preds_df: per-player probabilities keyed by the same ids:
        game_id, team_id, player_id, p_ge5_pred, p_ge15_pred

    Adds columns:
    - p_ge5_pred, p_ge15_pred
    - gate_tier, gate_reason, gate_missing_pred, gate_excluded
    - gate_minutes_cap, gate_play_prob_cap, p_ge5_used, p_ge15_used
    - minutes_prior_adj, minutes_p10_adj, minutes_p90_adj, play_prob_adj
    """
    required = {"game_id", "team_id", "player_id", "minutes_prior", "play_prob"}
    missing = sorted([c for c in required if c not in df_priors.columns])
    if missing:
        raise ValueError(f"df_priors missing required columns: {missing}")

    if df_priors.empty:
        out = df_priors.copy()
        out["p_ge5_pred"] = pd.Series(dtype=np.float64)
        out["p_ge15_pred"] = pd.Series(dtype=np.float64)
        out["gate_tier"] = pd.Series(dtype="string")
        out["gate_reason"] = pd.Series(dtype="string")
        out["gate_missing_pred"] = pd.Series(dtype=bool)
        out["gate_excluded"] = pd.Series(dtype=bool)
        out["gate_minutes_cap"] = pd.Series(dtype=np.float64)
        out["gate_play_prob_cap"] = pd.Series(dtype=np.float64)
        out["p_ge5_used"] = pd.Series(dtype=np.float64)
        out["p_ge15_used"] = pd.Series(dtype=np.float64)
        out["minutes_prior_adj"] = pd.Series(dtype=np.float64)
        out["minutes_p10_adj"] = pd.Series(dtype=np.float64)
        out["minutes_p90_adj"] = pd.Series(dtype=np.float64)
        out["play_prob_adj"] = pd.Series(dtype=np.float64)
        return out

    df = df_priors.copy()
    in_player_ids = set(
        pd.to_numeric(df["player_id"], errors="coerce").astype("Int64").dropna().astype(int).unique().tolist()
    )
    df["game_id"] = df["game_id"].astype("string")
    df["team_id"] = pd.to_numeric(df["team_id"], errors="coerce").astype("Int64")
    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["game_id", "team_id", "player_id"]).copy()
    df["team_id"] = df["team_id"].astype(int)
    df["player_id"] = df["player_id"].astype(int)

    game_ids = [str(v) for v in df["game_id"].dropna().unique().tolist() if str(v)]
    team_ids = [int(v) for v in df["team_id"].dropna().unique().tolist()]
    if len(game_ids) != 1 or len(team_ids) != 1:
        raise ValueError(
            "apply_rotation_gate expects a single (game_id, team_id) slice. "
            f"Got game_ids={game_ids[:5]} team_ids={team_ids[:5]}"
        )

    minutes_prior_orig = _as_float_series(df, "minutes_prior", default=0.0).clip(lower=0.0)
    minutes_p10_orig = _as_float_series(df, "minutes_p10", default=np.nan)
    minutes_p90_orig = _as_float_series(df, "minutes_p90", default=np.nan)
    play_prob_orig = _as_float_series(df, "play_prob", default=1.0).clip(0.0, 1.0)

    # Use filled values for tiering/capping logic, but preserve strict no-op for missing preds.
    minutes_prior = minutes_prior_orig
    minutes_p10 = minutes_p10_orig.fillna(minutes_prior_orig).clip(lower=0.0)
    minutes_p90 = minutes_p90_orig.fillna(minutes_prior_orig).clip(lower=0.0)
    play_prob = play_prob_orig

    preds = preds_df.copy() if preds_df is not None else pd.DataFrame()
    if not preds.empty:
        expected = {"game_id", "team_id", "player_id", "p_ge5_pred", "p_ge15_pred"}
        miss2 = sorted([c for c in expected if c not in preds.columns])
        if miss2:
            raise ValueError(f"preds_df missing required columns: {miss2}")
        preds = preds[["game_id", "team_id", "player_id", "p_ge5_pred", "p_ge15_pred"]].copy()
        preds["game_id"] = preds["game_id"].astype("string")
        preds["team_id"] = pd.to_numeric(preds["team_id"], errors="coerce").astype("Int64")
        preds["player_id"] = pd.to_numeric(preds["player_id"], errors="coerce").astype("Int64")
        preds["p_ge5_pred"] = pd.to_numeric(preds["p_ge5_pred"], errors="coerce").astype(np.float64).clip(0.0, 1.0)
        preds["p_ge15_pred"] = pd.to_numeric(preds["p_ge15_pred"], errors="coerce").astype(np.float64).clip(0.0, 1.0)
        preds = preds.dropna(subset=["game_id", "team_id", "player_id"]).copy()
        preds["team_id"] = preds["team_id"].astype(int)
        preds["player_id"] = preds["player_id"].astype(int)
        preds = preds.drop_duplicates(subset=["game_id", "team_id", "player_id"], keep="last").copy()

        df = df.merge(preds, on=["game_id", "team_id", "player_id"], how="left", validate="many_to_one")
    else:
        df["p_ge5_pred"] = np.nan
        df["p_ge15_pred"] = np.nan

    p_ge5 = pd.to_numeric(df["p_ge5_pred"], errors="coerce").astype(np.float64)
    p_ge15 = pd.to_numeric(df["p_ge15_pred"], errors="coerce").astype(np.float64)
    missing_pred = p_ge5.isna() | p_ge15.isna()
    missing_pred_arr = missing_pred.to_numpy(dtype=bool)

    n = len(df)
    tier = np.full(n, "fringe", dtype=object)
    reason_parts: list[list[str]] = [[] for _ in range(n)]

    protected = np.zeros(n, dtype=bool)
    if bool(cfg.protect_starters) and starters_set:
        starter_mask = (df["player_id"].isin({int(v) for v in starters_set}).to_numpy(dtype=bool)) & ~missing_pred_arr
        if starter_mask.any():
            protected |= starter_mask
            tier[starter_mask] = "starter"
            for i in np.where(starter_mask)[0].tolist():
                reason_parts[i].append("protect_starter")

    if bool(cfg.protect_top_n) and int(cfg.top_n_lock) > 0:
        tmp = df[["player_id"]].copy()
        tmp["_minutes_prior"] = minutes_prior.to_numpy(dtype=np.float64)
        tmp = tmp.sort_values(["_minutes_prior", "player_id"], ascending=[False, True], kind="mergesort")
        tmp["_rank"] = np.arange(1, len(tmp) + 1, dtype=np.int64)
        top_idx = tmp.index[(tmp["_rank"] <= int(cfg.top_n_lock)).to_numpy(dtype=bool)]
        top_idx = [int(i) for i in top_idx.tolist() if not bool(protected[int(i)])]
        top_idx = [int(i) for i in top_idx if not bool(missing_pred_arr[int(i)])]
        if top_idx:
            protected[np.asarray(top_idx, dtype=int)] = True
            tier[np.asarray(top_idx, dtype=int)] = "top_n"
            for i in top_idx:
                reason_parts[int(i)].append("protect_top_n")

    # Missing predictions: strict fail-open (no gating, no caps).
    if missing_pred_arr.any():
        miss_idx = np.where(missing_pred_arr)[0].tolist()
        tier[np.asarray(miss_idx, dtype=int)] = "unknown"
        for i in miss_idx:
            reason_parts[int(i)] = ["missing_pred"]

    remaining = ~protected & ~missing_pred_arr

    p_ge15_arr = p_ge15.fillna(0.0).to_numpy(dtype=np.float64)
    p_ge5_arr = p_ge5.fillna(0.0).to_numpy(dtype=np.float64)
    m_arr = minutes_prior.to_numpy(dtype=np.float64)

    core_mask = remaining & ((p_ge15_arr >= float(cfg.core_ge15_min)) | (m_arr >= float(cfg.core_minutes_prior_min)))
    bench_mask = remaining & ~core_mask & (
        (p_ge5_arr >= float(cfg.bench_ge5_min)) | (m_arr >= float(cfg.bench_minutes_prior_min))
    )
    fringe_mask = remaining & ~core_mask & ~bench_mask

    tier[core_mask] = "core"
    tier[bench_mask] = "bench"
    tier[fringe_mask] = "fringe"

    # Base capping (before hard exclusion).
    minutes_prior_capped = m_arr.copy()
    minutes_p10_capped = minutes_p10.to_numpy(dtype=np.float64).copy()
    minutes_p90_capped = minutes_p90.to_numpy(dtype=np.float64).copy()
    play_prob_capped = play_prob.to_numpy(dtype=np.float64).copy()

    minutes_cap = np.full(n, np.nan, dtype=np.float64)
    play_prob_cap = np.full(n, np.nan, dtype=np.float64)

    if bench_mask.any():
        cap = float(cfg.bench_minutes_cap)
        minutes_prior_capped[bench_mask] = np.minimum(minutes_prior_capped[bench_mask], cap)
        minutes_p10_capped[bench_mask] = np.minimum(minutes_p10_capped[bench_mask], cap)
        minutes_p90_capped[bench_mask] = np.minimum(minutes_p90_capped[bench_mask], cap)
        minutes_cap[bench_mask] = cap
        for i in np.where(bench_mask)[0].tolist():
            reason_parts[int(i)].append("bench_cap")

    if fringe_mask.any():
        cap = float(cfg.fringe_minutes_cap)
        minutes_prior_capped[fringe_mask] = np.minimum(minutes_prior_capped[fringe_mask], cap)
        minutes_p10_capped[fringe_mask] = np.minimum(minutes_p10_capped[fringe_mask], cap)
        minutes_p90_capped[fringe_mask] = np.minimum(minutes_p90_capped[fringe_mask], cap)
        pp_cap = float(cfg.fringe_play_prob_cap)
        play_prob_capped[fringe_mask] = np.minimum(play_prob_capped[fringe_mask], pp_cap)
        minutes_cap[fringe_mask] = cap
        play_prob_cap[fringe_mask] = pp_cap
        for i in np.where(fringe_mask)[0].tolist():
            reason_parts[int(i)].append("fringe_cap")

    minutes_prior_adj = minutes_prior_capped.copy()
    minutes_p10_adj = minutes_p10_capped.copy()
    minutes_p90_adj = minutes_p90_capped.copy()
    play_prob_adj = play_prob_capped.copy()

    # Strict fail-open no-op: missing predictions must not change any priors/caps.
    if missing_pred_arr.any():
        minutes_prior_adj[missing_pred_arr] = minutes_prior_orig.to_numpy(dtype=np.float64)[missing_pred_arr]
        minutes_p10_adj[missing_pred_arr] = minutes_p10_orig.to_numpy(dtype=np.float64)[missing_pred_arr]
        minutes_p90_adj[missing_pred_arr] = minutes_p90_orig.to_numpy(dtype=np.float64)[missing_pred_arr]
        play_prob_adj[missing_pred_arr] = play_prob_orig.to_numpy(dtype=np.float64)[missing_pred_arr]

    # Final outputs.
    df["gate_tier"] = pd.Series([str(x) for x in tier.tolist()], index=df.index, dtype="string")
    df["gate_reason"] = pd.Series(["|".join(parts) for parts in reason_parts], index=df.index, dtype="string")
    df["gate_missing_pred"] = pd.Series(missing_pred_arr, index=df.index, dtype=bool)
    df["gate_excluded"] = pd.Series(np.zeros(n, dtype=bool), index=df.index, dtype=bool)
    df["gate_minutes_cap"] = pd.Series(minutes_cap, index=df.index, dtype=np.float64)
    df["gate_play_prob_cap"] = pd.Series(play_prob_cap, index=df.index, dtype=np.float64)
    df["p_ge5_used"] = pd.Series(np.where(missing_pred_arr, np.nan, p_ge5.to_numpy(dtype=np.float64)), index=df.index, dtype=np.float64)
    df["p_ge15_used"] = pd.Series(np.where(missing_pred_arr, np.nan, p_ge15.to_numpy(dtype=np.float64)), index=df.index, dtype=np.float64)
    df["minutes_prior_adj"] = pd.Series(minutes_prior_adj, index=df.index, dtype=np.float64).clip(lower=0.0)
    df["minutes_p10_adj"] = pd.Series(minutes_p10_adj, index=df.index, dtype=np.float64)
    df["minutes_p90_adj"] = pd.Series(minutes_p90_adj, index=df.index, dtype=np.float64)
    df["play_prob_adj"] = pd.Series(play_prob_adj, index=df.index, dtype=np.float64).clip(0.0, 1.0)

    # Keep deterministic row order stable.
    df = df.sort_values(["player_id"], kind="mergesort").reset_index(drop=True)

    out_player_ids = set(df["player_id"].astype(int).unique().tolist())
    if out_player_ids != in_player_ids:
        raise ValueError(
            "apply_rotation_gate must preserve input player ids (non-structural invariant). "
            f"in={sorted(in_player_ids)[:10]} out={sorted(out_player_ids)[:10]}"
        )
    return df
