"""Projected-starter boost for rotation_set_minutes inference.

Motivation
----------
The rotation-set minutes model outputs:
- `gate_logit` / `gate_prob`: in-rotation likelihood (healthy scratch / DNP-ish),
- `share_logit`: within-team share.

In some "next man up" situations, a player can be marked as a projected starter
but have a low `gate_prob` due to weak historical starter priors. This helper
optionally boosts those projected starters into the rotation and recomputes
minutes using the model's allocation math (including entmax/softmax selection
and optional prior-head base weights).

This must be done carefully: `gate_prob` is a sigmoid, so it is almost always
< 1.0. Boosting on `(gate_prob < 1.0)` will trigger for essentially every
projected starter and can flatten starter minutes in production.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from projections.rotation.alloc_mask import build_alloc_mask_from_features
from projections.rotation.set_model import minutes_from_gate_and_share_logits


@dataclass(frozen=True)
class ProjectedStarterBoostStats:
    enabled: bool
    gate_prob_threshold: float
    gate_logit_boost: float
    boosted_players: int
    recomputed_team_games: int
    skipped_reason: str | None = None


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def apply_projected_starter_boost(
    rot_pred: pd.DataFrame,
    *,
    rot_features: pd.DataFrame,
    model_dir: Path,
    alloc_mask_mode: str,
    alloc_min_eligible: int,
    alloc_prior_play_prob_threshold: float,
    alloc_baseline_minutes_threshold: float,
    gate_prob_threshold: float,
    gate_logit_boost: float,
    key_cols: tuple[str, str, str] = ("game_id", "team_id", "player_id"),
    minutes_col: str = "rotation_minutes_p50",
    gate_prob_col: str = "gate_prob",
    gate_logit_col: str = "gate_logit",
    share_logit_col: str = "share_logit",
) -> tuple[pd.DataFrame, ProjectedStarterBoostStats]:
    """Boost low-gate projected starters and recompute minutes for affected team-games.

    Returns an updated copy of rot_pred and diagnostics stats.
    """
    if not (0.0 < float(gate_prob_threshold) < 1.0):
        return (
            rot_pred,
            ProjectedStarterBoostStats(
                enabled=False,
                gate_prob_threshold=float(gate_prob_threshold),
                gate_logit_boost=float(gate_logit_boost),
                boosted_players=0,
                recomputed_team_games=0,
                skipped_reason="invalid_gate_prob_threshold",
            ),
        )
    if rot_pred.empty:
        return (
            rot_pred,
            ProjectedStarterBoostStats(
                enabled=False,
                gate_prob_threshold=float(gate_prob_threshold),
                gate_logit_boost=float(gate_logit_boost),
                boosted_players=0,
                recomputed_team_games=0,
                skipped_reason="rot_pred_empty",
            ),
        )
    missing_keys = [c for c in key_cols if c not in rot_pred.columns]
    if missing_keys:
        return (
            rot_pred,
            ProjectedStarterBoostStats(
                enabled=False,
                gate_prob_threshold=float(gate_prob_threshold),
                gate_logit_boost=float(gate_logit_boost),
                boosted_players=0,
                recomputed_team_games=0,
                skipped_reason=f"missing_keys:{missing_keys}",
            ),
        )

    if "is_projected_starter" not in rot_features.columns:
        return (
            rot_pred,
            ProjectedStarterBoostStats(
                enabled=False,
                gate_prob_threshold=float(gate_prob_threshold),
                gate_logit_boost=float(gate_logit_boost),
                boosted_players=0,
                recomputed_team_games=0,
                skipped_reason="missing_is_projected_starter",
            ),
        )
    if gate_prob_col not in rot_pred.columns:
        return (
            rot_pred,
            ProjectedStarterBoostStats(
                enabled=False,
                gate_prob_threshold=float(gate_prob_threshold),
                gate_logit_boost=float(gate_logit_boost),
                boosted_players=0,
                recomputed_team_games=0,
                skipped_reason="missing_gate_prob",
            ),
        )
    if gate_logit_col not in rot_pred.columns or share_logit_col not in rot_pred.columns:
        return (
            rot_pred,
            ProjectedStarterBoostStats(
                enabled=False,
                gate_prob_threshold=float(gate_prob_threshold),
                gate_logit_boost=float(gate_logit_boost),
                boosted_players=0,
                recomputed_team_games=0,
                skipped_reason="missing_gate_or_share_logit",
            ),
        )

    alloc_mode = str(alloc_mask_mode or "").strip().lower()
    if alloc_mode not in {"strict", "not_out"}:
        alloc_mode = "strict"

    # Align projected-starter mask to rot_pred rows.
    pred_index = rot_pred.loc[:, list(key_cols)].copy()
    pred_index["_row_id"] = rot_pred.index
    starter_frame = (
        rot_features.loc[:, list(key_cols) + ["is_projected_starter"]]
        .copy()
        .drop_duplicates(subset=list(key_cols), keep="last")
    )
    starter_join = pred_index.merge(starter_frame, on=list(key_cols), how="left")
    projected_mask = (
        pd.to_numeric(starter_join["is_projected_starter"], errors="coerce")
        .fillna(0)
        .astype(int)
        .to_numpy(dtype=int)
        > 0
    )

    gate_prob_arr = pd.to_numeric(rot_pred[gate_prob_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    needs_boost = projected_mask & (gate_prob_arr < float(gate_prob_threshold))
    n_boosted = int(needs_boost.sum())
    if n_boosted == 0:
        return (
            rot_pred,
            ProjectedStarterBoostStats(
                enabled=True,
                gate_prob_threshold=float(gate_prob_threshold),
                gate_logit_boost=float(gate_logit_boost),
                boosted_players=0,
                recomputed_team_games=0,
            ),
        )

    model_cfg_path = Path(model_dir).expanduser() / "config.json"
    try:
        model_cfg = json.loads(model_cfg_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return (
            rot_pred,
            ProjectedStarterBoostStats(
                enabled=False,
                gate_prob_threshold=float(gate_prob_threshold),
                gate_logit_boost=float(gate_logit_boost),
                boosted_players=0,
                recomputed_team_games=0,
                skipped_reason=f"model_config_read_failed:{exc}",
            ),
        )

    alloc_activation = str(model_cfg.get("alloc_activation", "entmax"))
    entmax_alpha = float(model_cfg.get("entmax_alpha", 1.5))
    share_temperature = float(model_cfg.get("share_temperature", 1.0))
    total_minutes = float(model_cfg.get("total_minutes", 240.0))
    eps = float(model_cfg.get("eps", 1e-6))
    use_prior_head = bool(model_cfg.get("use_prior_head", False))
    prior_weight_col = str(model_cfg.get("prior_weight_col", "minutes_from_stints_prior_20"))
    prior_weight_floor = float(model_cfg.get("prior_weight_floor", 0.0))

    # Join minimal columns needed for alloc_mask + prior weights.
    join_cols = [
        *key_cols,
        "is_out",
        "prior_play_prob",
        "is_confirmed_starter",
        "is_projected_starter",
    ]
    if use_prior_head:
        join_cols.append(prior_weight_col)
    join_cols = [c for c in join_cols if c in rot_features.columns]

    out = rot_pred.copy()
    work = out.copy()
    work["_row_id"] = work.index
    work = work.merge(
        rot_features.loc[:, join_cols].drop_duplicates(subset=list(key_cols), keep="last"),
        on=list(key_cols),
        how="left",
    )
    if "is_out" in work.columns:
        work["is_out"] = pd.to_numeric(work["is_out"], errors="coerce").fillna(0).astype(int)
    else:
        work["is_out"] = 0

    needs_boost_series = pd.Series(needs_boost, index=out.index, dtype=bool)

    recomputed_team_games = 0
    for (_game_id, _team_id), grp in work.groupby(["game_id", "team_id"], sort=False):
        row_ids = grp["_row_id"].to_numpy()
        local_boost = needs_boost_series.reindex(row_ids).fillna(False).to_numpy(dtype=bool)
        if not local_boost.any():
            continue

        if alloc_mode == "not_out":
            alloc_mask = grp["is_out"].to_numpy(dtype=int) == 0
        else:
            baseline_col = None
            for cand in ("minutes_p50", "baseline_p50", prior_weight_col):
                if cand in grp.columns:
                    baseline_col = cand
                    break
            alloc_mask = build_alloc_mask_from_features(
                grp,
                min_eligible=int(alloc_min_eligible),
                prior_play_prob_threshold=float(alloc_prior_play_prob_threshold),
                baseline_minutes_threshold=float(alloc_baseline_minutes_threshold),
                baseline_minutes_col=baseline_col,
            )

        gate_logits = (
            pd.to_numeric(grp[gate_logit_col], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
        )
        share_logits = (
            pd.to_numeric(grp[share_logit_col], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
        )

        gate_logits = gate_logits.copy()
        gate_logits[local_boost] = float(gate_logit_boost)

        base_weights_t = None
        if use_prior_head and prior_weight_col in grp.columns:
            bw = (
                pd.to_numeric(grp[prior_weight_col], errors="coerce")
                .fillna(0.0)
                .to_numpy(dtype=np.float32)
            )
            base_weights_t = torch.as_tensor(bw[None, :], dtype=torch.float32)

        minutes_t = minutes_from_gate_and_share_logits(
            torch.as_tensor(gate_logits[None, :], dtype=torch.float32),
            torch.as_tensor(share_logits[None, :], dtype=torch.float32),
            torch.as_tensor(alloc_mask[None, :], dtype=torch.bool),
            total_minutes=total_minutes,
            eps=eps,
            base_weights=base_weights_t,
            base_floor=prior_weight_floor,
            alloc_activation=alloc_activation,
            entmax_alpha=entmax_alpha,
            share_temperature=share_temperature,
        )
        minutes_new = minutes_t.detach().cpu().numpy()[0].astype(float)

        out.loc[row_ids, minutes_col] = minutes_new
        out.loc[row_ids, gate_logit_col] = gate_logits.astype(float)
        out.loc[row_ids, gate_prob_col] = _sigmoid(gate_logits).astype(float)
        recomputed_team_games += 1

    return (
        out,
        ProjectedStarterBoostStats(
            enabled=True,
            gate_prob_threshold=float(gate_prob_threshold),
            gate_logit_boost=float(gate_logit_boost),
            boosted_players=n_boosted,
            recomputed_team_games=recomputed_team_games,
        ),
    )

