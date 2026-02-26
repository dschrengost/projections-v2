"""Initial world generation path for GameTransformerV2 using inverse flow sampling."""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from projections.rotation.possession_backbone import FTA_POSS_COEFF
from torch.utils.data import DataLoader

from projections.minutes import PLAY_THRESHOLD_MINUTES, ROTATION_THRESHOLD_MINUTES
from projections.projections_bundle import add_canonical_projection_fields
from projections import paths
from projections.rotation.game_transformer_v2 import (
    FLOW_TARGET_COLUMNS_V1,
    GameLevelDataset,
    GameTransformerV2Config,
    build_game_level_examples,
    build_game_transformer_v2,
    collate_game_level_examples,
)
from projections.rotation.set_model import zfill_game_id_series

logger = logging.getLogger(__name__)

JOIN_KEYS = ["game_id", "team_id", "player_id", "game_date"]


def _utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _coerce_join_keys(df: pd.DataFrame, *, name: str) -> pd.DataFrame:
    out = df.copy()
    for col in ["game_id", "team_id", "player_id"]:
        if col not in out.columns:
            raise ValueError(f"{name} missing key column: {col}")
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
    if "game_date" not in out.columns:
        raise ValueError(f"{name} missing key column: game_date")
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce").dt.normalize()

    invalid = out[["game_id", "team_id", "player_id", "game_date"]].isna().any(axis=1)
    if invalid.any():
        raise ValueError(f"{name} has invalid key rows: {int(invalid.sum())}")
    return out


def _resolve_dataset_dir(value: str | None) -> Path:
    root = paths.get_data_root() / "training" / "datasets"
    if value:
        p = Path(value).expanduser()
        if p.exists():
            return p.resolve()
        p2 = root / value
        if p2.exists():
            return p2.resolve()
        raise FileNotFoundError(f"Dataset directory not found: {value}")

    candidates = sorted(root.glob("joint_rotation_rates_v1*"))
    if not candidates:
        raise FileNotFoundError(f"No joint_rotation_rates_v1* datasets found under {root}")
    return candidates[-1].resolve()


def _resolve_run_dir(value: str) -> Path:
    p = Path(value).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"Run directory not found: {value}")
    if not (p / "config.json").exists() or not (p / "model.pt").exists():
        raise FileNotFoundError(f"Run directory missing config.json/model.pt: {p}")
    return p.resolve()


def _split_val(df: pd.DataFrame, *, val_days: int) -> pd.DataFrame:
    days = sorted(pd.to_datetime(df["game_date"]).dropna().dt.normalize().unique().tolist())
    vd = max(1, int(val_days))
    val_dates = set(days[-vd:])
    return df.loc[pd.to_datetime(df["game_date"]).dt.normalize().isin(val_dates)].copy()


def _flow_idx(flow_target_columns: list[str], name: str) -> int:
    try:
        return int(flow_target_columns.index(name))
    except ValueError as exc:
        raise KeyError(f"Missing flow target column: {name}") from exc


def project_flow_stats_to_contract(flow_values: torch.Tensor, *, flow_target_columns: list[str]) -> torch.Tensor:
    """Apply minimal numeric cleanup for v1 count-stat contracts."""

    y = torch.clamp(flow_values, min=0.0)
    fg2m_idx = _flow_idx(flow_target_columns, "fg2m")
    fga2_idx = _flow_idx(flow_target_columns, "fga2")
    fg3m_idx = _flow_idx(flow_target_columns, "fg3m")
    fga3_idx = _flow_idx(flow_target_columns, "fga3")
    ftm_idx = _flow_idx(flow_target_columns, "ftm")
    fta_idx = _flow_idx(flow_target_columns, "fta")

    y = y.clone()
    y[..., fg2m_idx] = torch.minimum(y[..., fg2m_idx], y[..., fga2_idx])
    y[..., fg3m_idx] = torch.minimum(y[..., fg3m_idx], y[..., fga3_idx])
    y[..., ftm_idx] = torch.minimum(y[..., ftm_idx], y[..., fta_idx])
    return y


def check_world_contracts(
    *,
    minutes: torch.Tensor,
    flow_values: torch.Tensor,
    valid_mask: torch.Tensor,
    team_index: torch.Tensor,
    flow_target_columns: list[str],
    active_mask: torch.Tensor | None = None,
    tol: float = 1e-4,
) -> dict[str, int]:
    if minutes.ndim != 2 or valid_mask.shape != minutes.shape or team_index.shape != minutes.shape:
        raise ValueError("minutes/valid_mask/team_index must have shape (N,30)")
    if flow_values.ndim != 3 or flow_values.shape[:2] != minutes.shape:
        raise ValueError("flow_values must have shape (N,30,S)")
    if active_mask is not None and active_mask.shape != minutes.shape:
        raise ValueError("active_mask must have shape (N,30)")

    valid = valid_mask.to(dtype=torch.bool)
    minutes_valid = minutes * valid.to(dtype=minutes.dtype)
    flow_valid = flow_values * valid.unsqueeze(-1).to(dtype=flow_values.dtype)

    out: dict[str, int] = {}
    out["minutes_negative"] = int(((minutes_valid < -float(tol))).sum().item())
    out["minutes_over_48"] = int(((minutes_valid > (48.0 + float(tol)))).sum().item())
    out["invalid_player_nonzero_minutes"] = int(
        ((~valid) & (minutes.abs() > float(tol))).sum().item()
    )

    minutes_team_sum_violations = 0
    for side in (0, 1):
        mask = valid & (team_index == side)
        team_sum = (minutes * mask.to(dtype=minutes.dtype)).sum(dim=1)
        minutes_team_sum_violations += int((team_sum - 240.0).abs().gt(float(tol)).sum().item())
    out["team_minutes_not_240"] = int(minutes_team_sum_violations)

    out["negative_stats"] = int((flow_valid < -float(tol)).sum().item())
    fg2m_idx = _flow_idx(flow_target_columns, "fg2m")
    fga2_idx = _flow_idx(flow_target_columns, "fga2")
    fg3m_idx = _flow_idx(flow_target_columns, "fg3m")
    fga3_idx = _flow_idx(flow_target_columns, "fga3")
    ftm_idx = _flow_idx(flow_target_columns, "ftm")
    fta_idx = _flow_idx(flow_target_columns, "fta")

    out["fg2m_gt_fga2"] = int((flow_valid[..., fg2m_idx] > flow_valid[..., fga2_idx] + float(tol)).sum().item())
    out["fg3m_gt_fga3"] = int((flow_valid[..., fg3m_idx] > flow_valid[..., fga3_idx] + float(tol)).sum().item())
    out["ftm_gt_fta"] = int((flow_valid[..., ftm_idx] > flow_valid[..., fta_idx] + float(tol)).sum().item())
    if active_mask is not None:
        inactive = (~active_mask.to(dtype=torch.bool)) & valid
        inactive_nonzero_stats = (
            flow_values.abs() * inactive.unsqueeze(-1).to(dtype=flow_values.dtype)
        ) > float(tol)
        out["inactive_nonzero_stats"] = int(inactive_nonzero_stats.sum().item())
        out["inactive_nonzero_fpts_proxy"] = int(
            (flow_values.sum(dim=-1).abs() * inactive.to(dtype=flow_values.dtype) > float(tol)).sum().item()
        )
    out["total_violations"] = int(sum(out.values()))
    return out


def check_possession_symmetry(
    *,
    flow_values: torch.Tensor,
    valid_mask: torch.Tensor,
    team_index: torch.Tensor,
    flow_target_columns: list[str],
) -> dict[str, float]:
    """Compute possession symmetry diagnostics from sampled world stats.

    Returns a dict with:
      - poss_home_mean, poss_away_mean: average team possessions
      - poss_delta_abs_mean: mean |home - away| possession gap
      - poss_delta_abs_p95: p95 |home - away| possession gap
      - poss_delta_abs_max: max |home - away| possession gap
    """
    if flow_values.ndim != 3:
        raise ValueError("flow_values must have shape (N, 30, S)")

    valid = valid_mask.to(dtype=torch.bool)
    fga2_idx = _flow_idx(flow_target_columns, "fga2")
    fga3_idx = _flow_idx(flow_target_columns, "fga3")
    fta_idx = _flow_idx(flow_target_columns, "fta")
    oreb_idx = _flow_idx(flow_target_columns, "oreb")
    tov_idx = _flow_idx(flow_target_columns, "tov")

    poss_list: list[torch.Tensor] = []
    for side in (0, 1):
        mask = valid & (team_index == side)
        mask_f = mask.unsqueeze(-1).to(dtype=flow_values.dtype)
        fv = flow_values * mask_f
        fga = fv[:, :, fga2_idx].sum(dim=1) + fv[:, :, fga3_idx].sum(dim=1)
        fta = fv[:, :, fta_idx].sum(dim=1)
        oreb = fv[:, :, oreb_idx].sum(dim=1)
        tov = fv[:, :, tov_idx].sum(dim=1)
        poss = fga - oreb + tov + FTA_POSS_COEFF * fta
        poss_list.append(poss)

    poss_home, poss_away = poss_list
    delta = (poss_home - poss_away).abs()

    return {
        "poss_home_mean": float(poss_home.mean().item()),
        "poss_away_mean": float(poss_away.mean().item()),
        "poss_delta_abs_mean": float(delta.mean().item()),
        "poss_delta_abs_p95": float(torch.quantile(delta, 0.95).item()) if delta.numel() > 0 else 0.0,
        "poss_delta_abs_max": float(delta.max().item()) if delta.numel() > 0 else 0.0,
    }


def _compute_dk_fpts(
    *,
    pts: torch.Tensor,
    reb: torch.Tensor,
    ast: torch.Tensor,
    stl: torch.Tensor,
    blk: torch.Tensor,
    tov: torch.Tensor,
) -> torch.Tensor:
    base = pts + 1.25 * reb + 1.5 * ast + 2.0 * stl + 2.0 * blk - 0.5 * tov
    qualifying = torch.stack([pts, reb, ast, stl, blk], dim=-1).ge(10.0).sum(dim=-1)
    dd_bonus = (qualifying == 2).to(dtype=base.dtype) * 1.5
    td_bonus = (qualifying >= 3).to(dtype=base.dtype) * 3.0
    return base + dd_bonus + td_bonus


def _build_world_rows(
    *,
    batch: dict[str, torch.Tensor | list[str]],
    world_offset: int,
    minutes: torch.Tensor,
    active_mask: torch.Tensor,
    flow_values: torch.Tensor,
    flow_target_columns: list[str],
) -> list[dict[str, Any]]:
    bsz = int(batch["player_features"].shape[0])  # type: ignore[index]
    n_worlds = int(minutes.shape[1])
    valid = batch["player_valid_mask"].cpu().numpy().astype(bool)
    player_ids = batch["player_ids"].cpu().numpy().astype(np.int64)
    team_ids = batch["team_ids"].cpu().numpy().astype(np.int64)
    game_ids = [str(v) for v in batch["game_id_norm"]]  # type: ignore[index]
    game_dates = [str(v) for v in batch["game_date"]]  # type: ignore[index]

    mins_np = minutes.cpu().numpy()
    active_np = active_mask.cpu().numpy().astype(bool)
    flow_np = flow_values.cpu().numpy()

    idx = {name: _flow_idx(flow_target_columns, name) for name in FLOW_TARGET_COLUMNS_V1}
    has_pf = "pf" in flow_target_columns
    pf_idx = flow_target_columns.index("pf") if has_pf else None

    rows: list[dict[str, Any]] = []
    for b_idx in range(bsz):
        valid_flat = np.concatenate([valid[b_idx, 0], valid[b_idx, 1]], axis=0)
        player_flat = np.concatenate([player_ids[b_idx, 0], player_ids[b_idx, 1]], axis=0)
        team_flat = np.concatenate(
            [
                np.full((15,), int(team_ids[b_idx, 0]), dtype=np.int64),
                np.full((15,), int(team_ids[b_idx, 1]), dtype=np.int64),
            ],
            axis=0,
        )
        for w_idx in range(n_worlds):
            flow_world = flow_np[b_idx, w_idx]
            fga2 = flow_world[:, idx["fga2"]]
            fg2m = flow_world[:, idx["fg2m"]]
            fga3 = flow_world[:, idx["fga3"]]
            fg3m = flow_world[:, idx["fg3m"]]
            fta = flow_world[:, idx["fta"]]
            ftm = flow_world[:, idx["ftm"]]
            oreb = flow_world[:, idx["oreb"]]
            dreb = flow_world[:, idx["dreb"]]
            ast = flow_world[:, idx["ast"]]
            stl = flow_world[:, idx["stl"]]
            blk = flow_world[:, idx["blk"]]
            tov = flow_world[:, idx["tov"]]
            pf = flow_world[:, int(pf_idx)] if pf_idx is not None else np.zeros_like(fga2)
            fga = fga2 + fga3
            fgm = fg2m + fg3m
            pts = 2.0 * fg2m + 3.0 * fg3m + ftm
            reb = oreb + dreb

            dk = _compute_dk_fpts(
                pts=torch.from_numpy(pts),
                reb=torch.from_numpy(reb),
                ast=torch.from_numpy(ast),
                stl=torch.from_numpy(stl),
                blk=torch.from_numpy(blk),
                tov=torch.from_numpy(tov),
            ).numpy()
            for p_idx in np.where(valid_flat)[0]:
                rows.append(
                    {
                        "world_idx": int(world_offset + w_idx),
                        "game_id": int(game_ids[b_idx]),
                        "game_id_norm": str(game_ids[b_idx]),
                        "game_date": str(game_dates[b_idx]),
                        "team_id": int(team_flat[p_idx]),
                        "player_id": int(player_flat[p_idx]),
                        "active": int(bool(active_np[b_idx, w_idx, p_idx])),
                        "minutes": float(mins_np[b_idx, w_idx, p_idx]),
                        "fga2": float(fga2[p_idx]),
                        "fg2m": float(fg2m[p_idx]),
                        "fga3": float(fga3[p_idx]),
                        "fg3m": float(fg3m[p_idx]),
                        "fta": float(fta[p_idx]),
                        "ftm": float(ftm[p_idx]),
                        "oreb": float(oreb[p_idx]),
                        "dreb": float(dreb[p_idx]),
                        "ast": float(ast[p_idx]),
                        "stl": float(stl[p_idx]),
                        "blk": float(blk[p_idx]),
                        "tov": float(tov[p_idx]),
                        "pf": float(pf[p_idx]),
                        "fga": float(fga[p_idx]),
                        "fgm": float(fgm[p_idx]),
                        "fg3a": float(fga3[p_idx]),
                        "pts": float(pts[p_idx]),
                        "reb": float(reb[p_idx]),
                        "plus_minus": 0.0,
                        "dk_fpts": float(dk[p_idx]),
                    }
                )
    return rows


def _q(values: np.ndarray, q: float) -> float:
    if values.size <= 0:
        return 0.0
    return float(np.quantile(values, float(q)))


def summarize_worlds_to_projections(
    worlds_df: pd.DataFrame,
    *,
    sim_profile: str,
    play_threshold_minutes: float = float(PLAY_THRESHOLD_MINUTES),
    rotation_threshold_minutes: float = float(ROTATION_THRESHOLD_MINUTES),
) -> pd.DataFrame:
    required = {"world_idx", "game_date", "game_id", "team_id", "player_id", "active", "minutes", "dk_fpts"}
    missing = sorted(required - set(worlds_df.columns))
    if missing:
        raise ValueError(f"worlds_df missing required columns: {missing}")
    if worlds_df.empty:
        return worlds_df.copy()

    df = worlds_df.copy()
    df["active"] = pd.to_numeric(df["active"], errors="coerce").fillna(0).astype(int) > 0
    df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce").fillna(0.0)
    df["dk_fpts"] = pd.to_numeric(df["dk_fpts"], errors="coerce").fillna(0.0)
    for c in ("pts", "reb", "ast", "stl", "blk", "tov"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    key_cols = ["game_date", "game_id", "team_id", "player_id"]
    rows: list[dict[str, Any]] = []
    for keys, grp in df.groupby(key_cols, sort=False):
        game_date, game_id, team_id, player_id = keys
        minutes = grp["minutes"].to_numpy(dtype=float, copy=False)
        fpts = grp["dk_fpts"].to_numpy(dtype=float, copy=False)
        active_mask = grp["active"].to_numpy(dtype=bool, copy=False)
        n_worlds = int(max(1, grp["world_idx"].nunique()))

        # Spec semantics: conditional moments are over active worlds.
        cond_mask = active_mask
        if not bool(cond_mask.any()):
            cond_mask = minutes >= float(play_threshold_minutes)
        fpts_cond = fpts[cond_mask]
        mins_cond = minutes[cond_mask]

        payload: dict[str, Any] = {
            "game_date": str(game_date),
            "game_id": int(game_id),
            "team_id": int(team_id),
            "player_id": int(player_id),
            "play_prob": float(active_mask.mean()),
            "play_prob_raw": float(active_mask.mean()),
            "play_prob_eff": float(active_mask.mean()),
            "sim_p_available": float(active_mask.mean()),
            "sim_p_active": float(active_mask.mean()),
            "sim_p_rotation": float((minutes >= float(rotation_threshold_minutes)).mean()),
            "rotation_lock": False,
            "play_prob_policy_reason": "n/a",
            "bench_zero_p_zero": 0.0,
            "bench_zero_threshold_minutes": float("nan"),
            "minutes_mean": float(minutes.mean()),
            "minutes_sim_mean": float(mins_cond.mean()) if mins_cond.size > 0 else 0.0,
            "minutes_sim_std": float(mins_cond.std(ddof=0)) if mins_cond.size > 0 else 0.0,
            "minutes_sim_p10": _q(mins_cond, 0.10),
            "minutes_sim_p50": _q(mins_cond, 0.50),
            "minutes_sim_p90": _q(mins_cond, 0.90),
            "minutes_p10_cond": _q(mins_cond, 0.10),
            "minutes_p50_cond": _q(mins_cond, 0.50),
            "minutes_p90_cond": _q(mins_cond, 0.90),
            "minutes_sim_mean_uncond": float(minutes.mean()),
            "minutes_sim_std_uncond": float(minutes.std(ddof=0)),
            "minutes_sim_p10_uncond": _q(minutes, 0.10),
            "minutes_sim_p50_uncond": _q(minutes, 0.50),
            "minutes_sim_p90_uncond": _q(minutes, 0.90),
            "minutes_p10": _q(minutes, 0.10),
            "minutes_p50": _q(minutes, 0.50),
            "minutes_p90": _q(minutes, 0.90),
            "dk_fpts_mean_target": float(fpts.mean()),
            "dk_fpts_mean": float(fpts_cond.mean()) if fpts_cond.size > 0 else 0.0,
            "dk_fpts_std": float(fpts_cond.std(ddof=0)) if fpts_cond.size > 0 else 0.0,
            "dk_fpts_p05": _q(fpts_cond, 0.05),
            "dk_fpts_p10": _q(fpts_cond, 0.10),
            "dk_fpts_p25": _q(fpts_cond, 0.25),
            "dk_fpts_p50": _q(fpts_cond, 0.50),
            "dk_fpts_p75": _q(fpts_cond, 0.75),
            "dk_fpts_p90": _q(fpts_cond, 0.90),
            "dk_fpts_p95": _q(fpts_cond, 0.95),
            "dk_fpts_mean_uncond": float(fpts.mean()),
            "dk_fpts_std_uncond": float(fpts.std(ddof=0)),
            "dk_fpts_p05_uncond": _q(fpts, 0.05),
            "dk_fpts_p10_uncond": _q(fpts, 0.10),
            "dk_fpts_p25_uncond": _q(fpts, 0.25),
            "dk_fpts_p50_uncond": _q(fpts, 0.50),
            "dk_fpts_p75_uncond": _q(fpts, 0.75),
            "dk_fpts_p90_uncond": _q(fpts, 0.90),
            "dk_fpts_p95_uncond": _q(fpts, 0.95),
            "sim_profile": str(sim_profile),
            "n_worlds": int(n_worlds),
            # Keep legacy-prefixed aliases used by some API code paths.
            "sim_dk_fpts_mean": float(fpts_cond.mean()) if fpts_cond.size > 0 else 0.0,
            "sim_dk_fpts_std": float(fpts_cond.std(ddof=0)) if fpts_cond.size > 0 else 0.0,
            "sim_dk_fpts_p05": _q(fpts_cond, 0.05),
            "sim_dk_fpts_p10": _q(fpts_cond, 0.10),
            "sim_dk_fpts_p25": _q(fpts_cond, 0.25),
            "sim_dk_fpts_p50": _q(fpts_cond, 0.50),
            "sim_dk_fpts_p75": _q(fpts_cond, 0.75),
            "sim_dk_fpts_p90": _q(fpts_cond, 0.90),
            "sim_dk_fpts_p95": _q(fpts_cond, 0.95),
            "sim_minutes_sim_mean": float(mins_cond.mean()) if mins_cond.size > 0 else 0.0,
            "sim_minutes_sim_p10": _q(mins_cond, 0.10),
            "sim_minutes_sim_p50": _q(mins_cond, 0.50),
            "sim_minutes_sim_p90": _q(mins_cond, 0.90),
            "sim_minutes_sim_std": float(mins_cond.std(ddof=0)) if mins_cond.size > 0 else 0.0,
            "sim_minutes_sim_mean_uncond": float(minutes.mean()),
            "sim_minutes_sim_p10_uncond": _q(minutes, 0.10),
            "sim_minutes_sim_p50_uncond": _q(minutes, 0.50),
            "sim_minutes_sim_p90_uncond": _q(minutes, 0.90),
            "sim_minutes_sim_std_uncond": float(minutes.std(ddof=0)),
        }
        for stat_name in ("pts", "reb", "ast", "stl", "blk", "tov"):
            if stat_name in grp.columns:
                vals = grp[stat_name].to_numpy(dtype=float, copy=False)
                payload[f"{stat_name}_mean"] = float(vals[cond_mask].mean()) if cond_mask.any() else 0.0
                payload[f"{stat_name}_mean_uncond"] = float(vals.mean())
                payload[f"sim_{stat_name}_mean_uncond"] = float(vals.mean())

        rows.append(payload)

    out = pd.DataFrame(rows)
    out = add_canonical_projection_fields(out)
    sort_cols = [c for c in ("game_date", "game_id", "team_id", "player_id") if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    return out


def sample_worlds_for_batch(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor | list[str]],
    *,
    device: torch.device,
    num_worlds: int,
    chunk_size: int,
    active_temperature: float,
    strict_contracts: bool,
    poss_symmetry_gate: float | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    if not hasattr(model, "flow_head") or model.flow_head is None:  # type: ignore[attr-defined]
        raise RuntimeError("Model does not expose flow_head for inverse flow sampling")

    flow_target_columns = list(model.flow_target_columns)  # type: ignore[attr-defined]
    bsz = int(batch["player_features"].shape[0])  # type: ignore[index]

    player_features = batch["player_features"].to(device=device)  # type: ignore[index]
    player_valid_mask = batch["player_valid_mask"].to(device=device)  # type: ignore[index]
    game_features = batch["game_features"].to(device=device)  # type: ignore[index]
    team_features = batch["team_features"].to(device=device)  # type: ignore[index]

    frames: list[pd.DataFrame] = []
    contract_counter: Counter[str] = Counter()
    poss_sym_flow_parts: list[torch.Tensor] = []
    poss_sym_valid_parts: list[torch.Tensor] = []
    poss_sym_team_parts: list[torch.Tensor] = []
    total_worlds = max(1, int(num_worlds))
    chunk = max(1, int(chunk_size))
    for world_offset in range(0, total_worlds, chunk):
        n_worlds_chunk = min(chunk, total_worlds - world_offset)
        rep_player_features = player_features.repeat_interleave(n_worlds_chunk, dim=0)
        rep_player_valid_mask = player_valid_mask.repeat_interleave(n_worlds_chunk, dim=0)
        rep_game_features = game_features.repeat_interleave(n_worlds_chunk, dim=0)
        rep_team_features = team_features.repeat_interleave(n_worlds_chunk, dim=0)

        with torch.no_grad():
            # Enable backbone sampling when possession backbone is present
            has_backbone = getattr(model, "enable_possession_backbone", False)
            out = model(
                rep_player_features,
                rep_player_valid_mask,
                game_features=rep_game_features,
                team_features=rep_team_features,
                sample_active=True,
                active_temperature=float(active_temperature),
                run_flow=False,
                sample_backbone=bool(has_backbone),
            )
            z = torch.randn(
                (rep_player_features.shape[0], out.player_states.shape[1], len(flow_target_columns)),
                device=device,
                dtype=out.player_states.dtype,
            )
            flow_raw = model.flow_head.sample(  # type: ignore[attr-defined]
                z,
                player_states=out.player_states,
                team_states=out.team_states,
                game_state=out.game_state,
                player_team_index=out.player_team_index,
                valid_mask=out.player_valid_mask,
            )
            flow_projected = project_flow_stats_to_contract(
                flow_raw,
                flow_target_columns=flow_target_columns,
            )
            # Enforce DNP semantics: inactive players contribute zero counting stats.
            flow_projected = flow_projected * out.active.active_mask.unsqueeze(-1).to(dtype=flow_projected.dtype)

        minutes = out.minutes.minutes.reshape(bsz, n_worlds_chunk, -1)
        active = out.active.active_mask.reshape(bsz, n_worlds_chunk, -1)
        flow_vals = flow_projected.reshape(bsz, n_worlds_chunk, out.player_states.shape[1], len(flow_target_columns))
        valid_flat = out.player_valid_mask.reshape(bsz, n_worlds_chunk, -1)
        team_flat = out.player_team_index.reshape(bsz, n_worlds_chunk, -1)

        checks = check_world_contracts(
            minutes=minutes.reshape(-1, minutes.shape[-1]),
            flow_values=flow_vals.reshape(-1, flow_vals.shape[-2], flow_vals.shape[-1]),
            valid_mask=valid_flat.reshape(-1, valid_flat.shape[-1]),
            team_index=team_flat.reshape(-1, team_flat.shape[-1]),
            flow_target_columns=flow_target_columns,
            active_mask=active.reshape(-1, active.shape[-1]),
        )
        contract_counter.update(checks)
        if strict_contracts and int(checks.get("total_violations", 0)) > 0:
            raise RuntimeError(f"World contract check failed: {checks}")

        # Accumulate tensors for possession symmetry check (lightweight CPU copies)
        if has_backbone:
            poss_sym_flow_parts.append(flow_vals.reshape(-1, flow_vals.shape[-2], flow_vals.shape[-1]).cpu())
            poss_sym_valid_parts.append(valid_flat.reshape(-1, valid_flat.shape[-1]).cpu())
            poss_sym_team_parts.append(team_flat.reshape(-1, team_flat.shape[-1]).cpu())

        chunk_rows = _build_world_rows(
            batch=batch,
            world_offset=int(world_offset),
            minutes=minutes,
            active_mask=active,
            flow_values=flow_vals,
            flow_target_columns=flow_target_columns,
        )
        if chunk_rows:
            frames.append(pd.DataFrame.from_records(chunk_rows))

    # -- Possession symmetry diagnostics (when backbone is enabled) --
    if has_backbone and poss_sym_flow_parts:
        all_flow = torch.cat(poss_sym_flow_parts, dim=0)
        all_valid = torch.cat(poss_sym_valid_parts, dim=0)
        all_team = torch.cat(poss_sym_team_parts, dim=0)
        poss_diag = check_possession_symmetry(
            flow_values=all_flow,
            valid_mask=all_valid,
            team_index=all_team,
            flow_target_columns=flow_target_columns,
        )
        logger.info(
            "possession symmetry: home=%.1f  away=%.1f  |delta| mean=%.2f  p95=%.2f  max=%.2f",
            poss_diag["poss_home_mean"],
            poss_diag["poss_away_mean"],
            poss_diag["poss_delta_abs_mean"],
            poss_diag["poss_delta_abs_p95"],
            poss_diag["poss_delta_abs_max"],
        )
        # Add possession symmetry diagnostics to contract counter for upstream visibility
        for k, v in poss_diag.items():
            contract_counter[k] = v  # type: ignore[assignment]

        # Hard validation gate: p95(|Poss_home - Poss_away|) must be within threshold
        gate = poss_symmetry_gate
        if gate is not None and poss_diag["poss_delta_abs_p95"] > gate:
            msg = (
                f"Possession symmetry gate FAILED: "
                f"p95(|delta|)={poss_diag['poss_delta_abs_p95']:.2f} > {gate:.1f}"
            )
            logger.warning(msg)
            if strict_contracts:
                raise RuntimeError(msg)

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return out, dict(contract_counter)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--dataset-dir", type=str, default=None)
    parser.add_argument("--val-days", type=int, default=14)
    parser.add_argument("--num-games", type=int, default=4)
    parser.add_argument("--num-worlds", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--active-temperature", type=float, default=1.0)
    parser.add_argument(
        "--flow-mean-ctx-weight-override",
        type=float,
        default=None,
        help="Inference-only override for mean-pooled flow conditioner context weight.",
    )
    parser.add_argument(
        "--flow-scale-clip-override",
        type=float,
        default=None,
        help="Inference-only override for flow coupling scale_clip (default 2.0). "
        "Higher values allow more extreme scale factors in affine coupling blocks. "
        "Also respects GT_FLOW_SCALE_CLIP env var if CLI not set.",
    )
    parser.add_argument("--strict-contracts", action="store_true")
    parser.add_argument(
        "--poss-symmetry-gate",
        type=float,
        default=None,
        help="Hard gate for p95(|Poss_home - Poss_away|). Warn if exceeded; fail if --strict-contracts.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out-parquet", type=str, default=None)
    parser.add_argument("--out-summary-json", type=str, default=None)
    parser.add_argument("--out-projections-parquet", type=str, default=None)
    parser.add_argument("--sim-profile-name", type=str, default="game_transformer_v2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    run_dir = _resolve_run_dir(args.run_dir)
    dataset_dir = _resolve_dataset_dir(args.dataset_dir)

    config = GameTransformerV2Config.load(run_dir / "config.json")
    model = build_game_transformer_v2(config)
    state = torch.load(run_dir / "model.pt", map_location="cpu")
    model.load_state_dict(state)
    if args.flow_mean_ctx_weight_override is not None:
        if not hasattr(model, "flow_head") or model.flow_head is None:  # type: ignore[attr-defined]
            raise RuntimeError("Model does not expose flow_head for mean context override")
        model.flow_head.set_mean_ctx_weight(float(args.flow_mean_ctx_weight_override))  # type: ignore[attr-defined]

    # flow_scale_clip override: CLI takes precedence over env var
    scale_clip_override = args.flow_scale_clip_override
    if scale_clip_override is None:
        env_clip = os.environ.get("GT_FLOW_SCALE_CLIP")
        if env_clip is not None:
            scale_clip_override = float(env_clip)
    if scale_clip_override is not None:
        if not hasattr(model, "flow_head") or model.flow_head is None:  # type: ignore[attr-defined]
            raise RuntimeError("Model does not expose flow_head for scale_clip override")
        print(f"[scale_clip override] setting flow_head.scale_clip = {scale_clip_override}")
        model.flow_head.set_scale_clip(float(scale_clip_override))  # type: ignore[attr-defined]

    device = torch.device(str(args.device))
    model = model.to(device=device)
    model.eval()

    features_df = _coerce_join_keys(pd.read_parquet(dataset_dir / "features.parquet"), name="features")
    labels_minutes_df = _coerce_join_keys(pd.read_parquet(dataset_dir / "labels_minutes.parquet"), name="labels_minutes")

    label_overlap = [c for c in labels_minutes_df.columns if c in features_df.columns and c not in JOIN_KEYS]
    labels_for_merge = labels_minutes_df.drop(columns=label_overlap)
    merged = features_df.merge(labels_for_merge, on=JOIN_KEYS, how="left", validate="one_to_one")
    merged["game_id_norm"] = zfill_game_id_series(merged["game_id"])
    val_df = _split_val(merged, val_days=int(args.val_days))

    examples = build_game_level_examples(
        val_df,
        feature_columns=list(config.feature_columns),
        feature_mean=np.asarray(config.feature_mean, dtype=np.float32),
        feature_std=np.asarray(config.feature_std, dtype=np.float32),
        game_feature_columns=list(config.game_feature_columns),
        team_feature_columns=list(config.team_feature_columns),
        minutes_label_col="minutes_label" if "minutes_label" in val_df.columns else "minutes",
        overflow_protected_prior_play_prob_floor=float(config.overflow_protected_prior_play_prob_floor),
        overflow_protected_prior_minutes_floor=float(config.overflow_protected_prior_minutes_floor),
        overflow_risk_weight_consecutive_active_dnp=float(config.overflow_risk_weight_consecutive_active_dnp),
        overflow_risk_weight_active_but_dnp_rate_last10=float(config.overflow_risk_weight_active_but_dnp_rate_last10),
        overflow_risk_weight_inactive_streak_len=float(config.overflow_risk_weight_inactive_streak_len),
        overflow_keep_weight_prior_play_prob=float(config.overflow_keep_weight_prior_play_prob),
        overflow_keep_weight_prior_minutes=float(config.overflow_keep_weight_prior_minutes),
    )
    max_games = max(1, int(args.num_games))
    examples = examples[:max_games]
    loader = DataLoader(
        GameLevelDataset(examples),
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        num_workers=max(0, int(args.num_workers)),
        collate_fn=collate_game_level_examples,
    )

    frames: list[pd.DataFrame] = []
    contract_counter: Counter[str] = Counter()
    for batch in loader:
        df_batch, checks = sample_worlds_for_batch(
            model,
            batch,
            device=device,
            num_worlds=int(args.num_worlds),
            chunk_size=int(args.chunk_size),
            active_temperature=float(args.active_temperature),
            strict_contracts=bool(args.strict_contracts),
            poss_symmetry_gate=getattr(args, "poss_symmetry_gate", None),
        )
        frames.append(df_batch)
        contract_counter.update(checks)

    worlds_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out_parquet = Path(args.out_parquet).expanduser() if args.out_parquet else (
        run_dir / f"sample_worlds_v2_{_utc_now_compact()}.parquet"
    )
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    worlds_df.to_parquet(out_parquet, index=False)

    projections_df = summarize_worlds_to_projections(
        worlds_df,
        sim_profile=str(args.sim_profile_name),
    )
    out_projections = Path(args.out_projections_parquet).expanduser() if args.out_projections_parquet else (
        run_dir / "projections.parquet"
    )
    out_projections.parent.mkdir(parents=True, exist_ok=True)
    projections_df.to_parquet(out_projections, index=False)

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "dataset_dir": str(dataset_dir),
        "num_games": int(max_games),
        "num_worlds_per_game": int(args.num_worlds),
        "flow_mean_ctx_weight_effective": float(
            args.flow_mean_ctx_weight_override
            if args.flow_mean_ctx_weight_override is not None
            else getattr(model.flow_head, "mean_ctx_weight", float("nan"))  # type: ignore[attr-defined]
        ),
        "rows": int(len(worlds_df)),
        "flow_target_columns": list(model.flow_target_columns),  # type: ignore[attr-defined]
        "contract_checks": dict(contract_counter),
        "out_parquet": str(out_parquet),
        "out_projections_parquet": str(out_projections),
        "projection_rows": int(len(projections_df)),
    }
    out_summary = Path(args.out_summary_json).expanduser() if args.out_summary_json else (
        run_dir / f"sample_worlds_v2_summary_{_utc_now_compact()}.json"
    )
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
