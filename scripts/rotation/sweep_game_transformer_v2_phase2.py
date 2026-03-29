#!/usr/bin/env python3
"""Targeted Phase 2 sweep for GameTransformerV2 anchor-parity recovery.

Supports two operation modes:
- anchor-recovery sweep (legacy/default)
- optimizer-quality sweep with optional multi-seed confirmation
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from projections import paths
from projections.pipeline.gtv2_inference_runtime import (
    load_gtv2_model,
    resolve_torch_device,
    score_gtv2_features_df,
)

PYTHON_EXE = sys.executable


@dataclass(frozen=True)
class Trial:
    name: str
    params: dict[str, Any]


@dataclass(frozen=True)
class EvalMetrics:
    minutes_mae_lineup0: float
    minutes_mae_lineup1: float
    minutes_mae_gap_abs: float
    active_acc_lineup0: float
    active_acc_lineup1: float
    active_count_mae: float
    possessions_proxy_mae: float


REALISM_GATE_DEFAULTS: dict[str, float] = {
    "realism_max_pts_mae": 17.0,
    "realism_max_abs_pts_bias": 8.0,
    "realism_max_abs_star_bias": 12.0,
    "realism_max_abs_elite_bias": 22.0,
    "realism_max_spread_mae_vs_vegas": 7.5,
    "realism_min_spread_span_ratio": 0.5,
    "realism_min_spread_corr_vs_vegas": 0.2,
    "realism_max_total_mae_vs_vegas": 10.0,
    "realism_min_total_span_ratio": 0.5,
    "realism_max_p90_calib_error": 0.03,
    "realism_max_p95_calib_error": 0.03,
    "realism_max_top1_share_bias": 0.05,
    "realism_max_top2_share_bias": 0.05,
}


def _utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _slugify(text: str) -> str:
    out: list[str] = []
    for ch in text.lower():
        if ch.isalnum() or ch in {"-", "_"}:
            out.append(ch)
        elif ch in {" ", "."}:
            out.append("_")
    return "".join(out).strip("_") or "trial"


def _default_trials() -> list[Trial]:
    rows = [
        (
            "anchor90_warm8_flow025_count060",
            {
                "phase2_anchor_end_weight": 0.90,
                "phase2_flow_warmup_epochs": 8,
                "w_count": 0.60,
                "w_member": 0.60,
                "w_minutes_nll": 0.75,
                "w_flow_nll": 0.25,
            },
        ),
        (
            "anchor90_warm10_flow020_count065",
            {
                "phase2_anchor_end_weight": 0.90,
                "phase2_flow_warmup_epochs": 10,
                "w_count": 0.65,
                "w_member": 0.65,
                "w_minutes_nll": 0.75,
                "w_flow_nll": 0.20,
            },
        ),
        (
            "anchor85_warm8_flow025_count060",
            {
                "phase2_anchor_end_weight": 0.85,
                "phase2_flow_warmup_epochs": 8,
                "w_count": 0.60,
                "w_member": 0.60,
                "w_minutes_nll": 1.00,
                "w_flow_nll": 0.25,
            },
        ),
        (
            "anchor85_warm6_flow030_count055",
            {
                "phase2_anchor_end_weight": 0.85,
                "phase2_flow_warmup_epochs": 6,
                "w_count": 0.55,
                "w_member": 0.55,
                "w_minutes_nll": 1.00,
                "w_flow_nll": 0.30,
            },
        ),
        (
            "anchor80_warm8_flow020_count050",
            {
                "phase2_anchor_end_weight": 0.80,
                "phase2_flow_warmup_epochs": 8,
                "w_count": 0.50,
                "w_member": 0.50,
                "w_minutes_nll": 1.00,
                "w_flow_nll": 0.20,
            },
        ),
        (
            "anchor80_warm6_flow035_count050",
            {
                "phase2_anchor_end_weight": 0.80,
                "phase2_flow_warmup_epochs": 6,
                "w_count": 0.50,
                "w_member": 0.50,
                "w_minutes_nll": 1.00,
                "w_flow_nll": 0.35,
            },
        ),
    ]
    return [Trial(name=_slugify(name), params=params) for name, params in rows]


def _default_optimizer_trials() -> list[Trial]:
    rows = [
        (
            "opt_lr5e4_wd5e5_bs32_clip075_flow4",
            {
                "lr": 5e-4,
                "weight_decay": 5e-5,
                "batch_size": 32,
                "backbone_grad_clip_norm": 0.75,
                "flow_grad_clip_norm": 3.0,
                "flow_num_blocks": 4,
                "flow_scale_clip": 2.0,
                "phase2_anchor_end_weight": 0.95,
                "phase2_flow_warmup_epochs": 12,
                "w_count": 0.50,
                "w_member": 0.50,
                "w_minutes_nll": 0.50,
                "w_flow_nll": 0.10,
            },
        ),
        (
            "opt_lr3e4_wd1e4_bs32_clip075_flow4_scale18",
            {
                "lr": 3e-4,
                "weight_decay": 1e-4,
                "batch_size": 32,
                "backbone_grad_clip_norm": 0.75,
                "flow_grad_clip_norm": 3.0,
                "flow_num_blocks": 4,
                "flow_scale_clip": 1.8,
                "phase2_anchor_end_weight": 0.95,
                "phase2_flow_warmup_epochs": 12,
                "w_count": 0.50,
                "w_member": 0.50,
                "w_minutes_nll": 0.50,
                "w_flow_nll": 0.10,
            },
        ),
        (
            "opt_lr7e4_wd1e5_bs24_clip10_flow5_warm10",
            {
                "lr": 7e-4,
                "weight_decay": 1e-5,
                "batch_size": 24,
                "backbone_grad_clip_norm": 1.0,
                "flow_grad_clip_norm": 4.0,
                "flow_num_blocks": 5,
                "flow_scale_clip": 2.0,
                "phase2_anchor_end_weight": 0.95,
                "phase2_flow_warmup_epochs": 10,
                "w_count": 0.50,
                "w_member": 0.50,
                "w_minutes_nll": 0.60,
                "w_flow_nll": 0.08,
            },
        ),
        (
            "opt_lr4e4_wd5e5_bs24_clip075_flow5_anchor96_w14",
            {
                "lr": 4e-4,
                "weight_decay": 5e-5,
                "batch_size": 24,
                "backbone_grad_clip_norm": 0.75,
                "flow_grad_clip_norm": 3.5,
                "flow_num_blocks": 5,
                "flow_scale_clip": 1.8,
                "phase2_anchor_end_weight": 0.96,
                "phase2_flow_warmup_epochs": 14,
                "w_count": 0.55,
                "w_member": 0.55,
                "w_minutes_nll": 0.55,
                "w_flow_nll": 0.08,
            },
        ),
    ]
    return [Trial(name=_slugify(name), params=params) for name, params in rows]


def _default_all_losses_trials() -> list[Trial]:
    """Broader search preset that includes all exposed training loss weights."""
    base: dict[str, Any] = {
        # Core phase2 scheduling
        "phase2_flow_delay_epochs": 6,
        "phase2_flow_warmup_epochs": 8,
        "phase2_anchor_end_weight": 0.75,
        # Ensure all relevant heads are active in this preset
        "enable_possession_backbone": True,
        "enable_three_pa_share": True,
        "enable_efficiency_head": True,
        "enable_usage_share_head": True,
        # Loss defaults from recent stable staged runs
        "w_minutes": 1.0,
        "w_minutes_nll": 1.0,
        "w_count": 0.5,
        "w_member": 0.5,
        "w_flow_nll": 0.10,
        "w_poss_nll": 0.20,
        "w_backbone_nll": 0.10,
        "w_three_pa_nll": 0.05,
        "w_poss_regression": 2.0,
        "w_efficiency_nll": 0.50,
        "w_usage_share_nll": 0.25,
        "w_emergent_share_aux": 0.00,
        "w_ast_share_aux": 0.00,
        "w_reb_share_aux": 0.00,
        "w_ast_team_rate_aux": 0.00,
        "w_reb_opportunity_rate_aux": 0.00,
        # Optimization stability defaults
        "encoder_lr_scale": 0.05,
        "backbone_head_lr_scale": 1.5,
        "backbone_grad_clip_norm": 1.0,
    }
    rows: list[tuple[str, dict[str, Any]]] = [
        ("allloss_baseline", dict(base)),
        ("allloss_eff_up", {**base, "w_efficiency_nll": 0.80}),
        ("allloss_eff_down", {**base, "w_efficiency_nll": 0.30}),
        ("allloss_usage_up", {**base, "w_usage_share_nll": 0.40}),
        ("allloss_usage_down", {**base, "w_usage_share_nll": 0.15}),
        ("allloss_flow_up", {**base, "w_flow_nll": 0.20}),
        ("allloss_flow_down", {**base, "w_flow_nll": 0.05}),
        ("allloss_possreg_up", {**base, "w_poss_regression": 3.0}),
        ("allloss_possreg_down", {**base, "w_poss_regression": 1.0}),
        ("allloss_anchor_stronger", {**base, "phase2_anchor_end_weight": 0.85}),
        (
            "allloss_with_ast_reb_aux",
            {
                **base,
                "w_ast_share_aux": 0.05,
                "w_reb_share_aux": 0.05,
                "w_ast_team_rate_aux": 0.05,
                "w_reb_opportunity_rate_aux": 0.05,
            },
        ),
        (
            "allloss_conservative_flow_with_aux",
            {
                **base,
                "phase2_flow_delay_epochs": 8,
                "phase2_flow_warmup_epochs": 10,
                "w_flow_nll": 0.08,
                "w_efficiency_nll": 0.60,
                "w_usage_share_nll": 0.30,
                "w_emergent_share_aux": 0.10,
            },
        ),
    ]
    return [Trial(name=_slugify(name), params=params) for name, params in rows]


def _default_sparse_recall_trials() -> list[Trial]:
    """Focused preset for sparse-prior / starter recall in minutes allocation."""
    base: dict[str, Any] = {
        "phase2_flow_delay_epochs": 4,
        "phase2_flow_warmup_epochs": 8,
        "phase2_anchor_end_weight": 0.85,
        "enable_possession_backbone": True,
        "enable_three_pa_share": True,
        "enable_efficiency_head": True,
        "enable_usage_share_head": True,
        "w_minutes": 1.0,
        "w_minutes_nll": 1.0,
        "w_count": 0.60,
        "w_member": 0.80,
        "active_positive_weight": 2.0,
        "lineup_available_sample_weight": 3.0,
        "w_flow_nll": 0.10,
        "w_poss_nll": 0.20,
        "w_backbone_nll": 0.10,
        "w_three_pa_nll": 0.05,
        "w_poss_regression": 2.0,
        "w_efficiency_nll": 0.50,
        "w_usage_share_nll": 0.25,
        "encoder_lr_scale": 0.05,
        "backbone_head_lr_scale": 1.5,
        "backbone_grad_clip_norm": 1.0,
    }
    rows: list[tuple[str, dict[str, Any]]] = [
        ("sparse_recall_baseline", dict(base)),
        (
            "sparse_recall_strong_member",
            {
                **base,
                "w_member": 1.00,
                "active_positive_weight": 3.0,
                "lineup_available_sample_weight": 4.0,
                "w_count": 0.70,
            },
        ),
        (
            "sparse_recall_conservative",
            {
                **base,
                "w_member": 0.70,
                "active_positive_weight": 1.5,
                "lineup_available_sample_weight": 2.0,
                "w_count": 0.55,
                "phase2_anchor_end_weight": 0.90,
            },
        ),
    ]
    return [Trial(name=_slugify(name), params=params) for name, params in rows]


def _default_sparse_hurdle_trials() -> list[Trial]:
    """Sparse-prior preset that compares baseline vs hurdle minutes head variants."""
    base: dict[str, Any] = {
        "phase2_flow_delay_epochs": 4,
        "phase2_flow_warmup_epochs": 8,
        "phase2_anchor_end_weight": 0.85,
        "enable_possession_backbone": True,
        "enable_three_pa_share": True,
        "enable_efficiency_head": True,
        "enable_usage_share_head": True,
        "w_minutes": 1.0,
        "w_minutes_nll": 1.0,
        "w_count": 0.60,
        "w_member": 0.80,
        "active_positive_weight": 2.0,
        "lineup_available_sample_weight": 3.0,
        "w_flow_nll": 0.10,
        "w_poss_nll": 0.20,
        "w_backbone_nll": 0.10,
        "w_three_pa_nll": 0.05,
        "w_poss_regression": 2.0,
        "w_efficiency_nll": 0.50,
        "w_usage_share_nll": 0.25,
        "encoder_lr_scale": 0.05,
        "backbone_head_lr_scale": 1.5,
        "backbone_grad_clip_norm": 1.0,
        "minutes_hurdle_zero_threshold": 0.5,
    }
    rows: list[tuple[str, dict[str, Any]]] = [
        ("sparse_hurdle_baseline", dict(base)),
        (
            "sparse_hurdle_moderate",
            {
                **base,
                "enable_minutes_hurdle_head": True,
                "minutes_hurdle_hidden": 64,
                "minutes_hurdle_sigma_floor": 0.7,
                "w_minutes_hurdle_nll": 0.10,
            },
        ),
        (
            "sparse_hurdle_strong",
            {
                **base,
                "enable_minutes_hurdle_head": True,
                "minutes_hurdle_hidden": 128,
                "minutes_hurdle_sigma_floor": 0.7,
                "w_minutes_hurdle_nll": 0.25,
                "w_member": 0.90,
                "active_positive_weight": 2.5,
            },
        ),
    ]
    return [Trial(name=_slugify(name), params=params) for name, params in rows]


def _read_trials_file(path: Path) -> list[Trial]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("trials JSON must be a list")

    out: list[Trial] = []
    for idx, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"trial[{idx}] must be an object")
        params = item.get("params", {k: v for k, v in item.items() if k != "name"})
        if not isinstance(params, dict):
            raise ValueError(f"trial[{idx}] params must be object")
        name = str(item.get("name", f"trial_{idx:03d}"))
        out.append(Trial(name=_slugify(name), params={str(k): v for k, v in params.items()}))
    return out


def _to_cli_args(params: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for key, value in params.items():
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                out.append(flag)
            continue
        out.extend([flag, str(value)])
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
        raise FileNotFoundError(f"dataset directory not found: {value}")
    candidates = sorted(root.glob("joint_rotation_rates_v1*"))
    if not candidates:
        raise FileNotFoundError(f"no joint_rotation_rates_v1* datasets found under {root}")
    return candidates[-1].resolve()


def _load_snapshot_features(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "game_date" not in df.columns:
        try:
            inferred_game_date = str(path.parents[1].name)
        except Exception as exc:  # pragma: no cover - defensive fallback
            raise ValueError(f"snapshot features missing game_date: {path}") from exc
        df["game_date"] = inferred_game_date

    for col in ("game_id", "team_id", "player_id"):
        if col not in df.columns:
            raise ValueError(f"snapshot features missing required column: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isna().any():
            raise ValueError(f"snapshot features has invalid {col} rows")
        df[col] = df[col].astype(int)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.date.astype(str)
    return df


def _deterministic_snapshot_metrics(
    *,
    bundle_dir: Path,
    snapshot_features_df: pd.DataFrame,
    device: torch.device,
) -> dict[str, float]:
    config, model = load_gtv2_model(bundle_dir, device=device)
    game_date = str(pd.to_datetime(snapshot_features_df["game_date"], errors="coerce").dt.date.astype(str).iloc[0])
    scores = score_gtv2_features_df(
        features_df=snapshot_features_df,
        game_date=game_date,
        config=config,
        model=model,
        device=device,
        batch_size=4,
    )
    meta_cols = [
        "game_date",
        "game_id",
        "team_id",
        "player_id",
        "an_pts_line",
        "an_implied_minutes",
    ]
    use_meta_cols = [c for c in meta_cols if c in snapshot_features_df.columns]
    if use_meta_cols:
        meta = snapshot_features_df.loc[:, use_meta_cols].drop_duplicates(
            subset=["game_date", "game_id", "team_id", "player_id"]
        )
        score_meta = scores.merge(
            meta,
            on=["game_date", "game_id", "team_id", "player_id"],
            how="left",
        )
    else:
        score_meta = scores

    star_mask = pd.Series(False, index=score_meta.index)
    if "an_pts_line" in score_meta.columns:
        star_mask |= pd.to_numeric(score_meta["an_pts_line"], errors="coerce").fillna(0.0).ge(20.0)
    if "an_implied_minutes" in score_meta.columns:
        star_mask |= pd.to_numeric(score_meta["an_implied_minutes"], errors="coerce").fillna(0.0).ge(30.0)

    if "minutes_deterministic" not in score_meta.columns or "active_prob_proxy" not in score_meta.columns:
        raise RuntimeError("score_gtv2_features_df missing minutes_deterministic/active_prob_proxy columns")
    det_minutes = pd.to_numeric(score_meta["minutes_deterministic"], errors="coerce")
    det_active_prob = pd.to_numeric(score_meta["active_prob_proxy"], errors="coerce")
    return {
        "det_minutes_mean": float(det_minutes.mean()),
        "det_active_prob_mean": float(det_active_prob.mean()),
        "det_prop_star_minutes_mean": float(det_minutes.loc[star_mask].mean()) if bool(star_mask.any()) else float("nan"),
        "det_n_players": float(len(score_meta)),
        "det_n_prop_stars": float(int(star_mask.sum())),
    }


def _meets_snapshot_guard(
    *,
    candidate: dict[str, float],
    baseline: dict[str, float],
    max_det_active_prob_delta: float,
    min_det_prop_star_minutes_delta: float,
) -> tuple[bool, list[str], dict[str, float]]:
    delta_active_prob = float(candidate.get("det_active_prob_mean", float("nan"))) - float(
        baseline.get("det_active_prob_mean", float("nan"))
    )
    delta_star_minutes = float(candidate.get("det_prop_star_minutes_mean", float("nan"))) - float(
        baseline.get("det_prop_star_minutes_mean", float("nan"))
    )
    deltas = {
        "delta_det_active_prob_mean": float(delta_active_prob),
        "delta_det_prop_star_minutes_mean": float(delta_star_minutes),
    }
    failures: list[str] = []
    if not math.isfinite(delta_active_prob) or delta_active_prob > float(max_det_active_prob_delta):
        failures.append(f"delta_det_active_prob_mean>{float(max_det_active_prob_delta):.6f}")
    if not math.isfinite(delta_star_minutes) or delta_star_minutes < float(min_det_prop_star_minutes_delta):
        failures.append(f"delta_det_prop_star_minutes_mean<{float(min_det_prop_star_minutes_delta):.6f}")
    return len(failures) == 0, failures, deltas


def _load_eval_metrics(path: Path) -> EvalMetrics:
    payload = json.loads(path.read_text(encoding="utf-8"))
    parity = payload.get("lineup_state_parity", {}) or {}
    volume = payload.get("game_volume_calibration", {}) or {}
    active_count = (volume.get("active_count", {}) or {})
    poss = (volume.get("possessions_proxy", {}) or {})
    return EvalMetrics(
        minutes_mae_lineup0=_float_or_nan((parity.get("lineup_available_0", {}) or {}).get("minutes_mae")),
        minutes_mae_lineup1=_float_or_nan((parity.get("lineup_available_1", {}) or {}).get("minutes_mae")),
        minutes_mae_gap_abs=_float_or_nan(parity.get("minutes_mae_gap_abs")),
        active_acc_lineup0=_float_or_nan((parity.get("lineup_available_0", {}) or {}).get("active_acc")),
        active_acc_lineup1=_float_or_nan((parity.get("lineup_available_1", {}) or {}).get("active_acc")),
        active_count_mae=_float_or_nan(active_count.get("mae")),
        possessions_proxy_mae=_float_or_nan(poss.get("mae")),
    )


def _diff_metrics(candidate: EvalMetrics, baseline: EvalMetrics) -> dict[str, float]:
    return {
        "delta_minutes_mae_lineup0": float(candidate.minutes_mae_lineup0 - baseline.minutes_mae_lineup0),
        "delta_minutes_mae_lineup1": float(candidate.minutes_mae_lineup1 - baseline.minutes_mae_lineup1),
        "delta_minutes_mae_gap_abs": float(candidate.minutes_mae_gap_abs - baseline.minutes_mae_gap_abs),
        "delta_active_acc_lineup0": float(candidate.active_acc_lineup0 - baseline.active_acc_lineup0),
        "delta_active_acc_lineup1": float(candidate.active_acc_lineup1 - baseline.active_acc_lineup1),
        "delta_active_count_mae": float(candidate.active_count_mae - baseline.active_count_mae),
        "delta_possessions_proxy_mae": float(candidate.possessions_proxy_mae - baseline.possessions_proxy_mae),
    }


def _composite_score(deltas: dict[str, float], *, promotion_gate_mode: str) -> float:
    d0 = max(0.0, float(deltas["delta_minutes_mae_lineup0"]))
    d1 = max(0.0, float(deltas["delta_minutes_mae_lineup1"]))
    dgap = max(0.0, float(deltas["delta_minutes_mae_gap_abs"]))
    dacc0 = max(0.0, -float(deltas["delta_active_acc_lineup0"]))
    dacc1 = max(0.0, -float(deltas["delta_active_acc_lineup1"]))
    dactive = max(0.0, float(deltas["delta_active_count_mae"]))
    dposs = max(0.0, float(deltas["delta_possessions_proxy_mae"]))
    if str(promotion_gate_mode) == "prod_like":
        return float(0.25 * d0 + 1.5 * d1 + 0.25 * dgap + 0.5 * dacc0 + 2.0 * dacc1 + 1.5 * dactive + 0.25 * dposs)
    return float(1.0 * d0 + 1.0 * d1 + 2.0 * dgap + 1.5 * dactive + 0.25 * dposs)


def _is_finite_eval(m: EvalMetrics, *, require_active_acc: bool) -> bool:
    values = [
        m.minutes_mae_lineup0,
        m.minutes_mae_lineup1,
        m.minutes_mae_gap_abs,
        m.active_count_mae,
        m.possessions_proxy_mae,
    ]
    if require_active_acc:
        values.extend([m.active_acc_lineup0, m.active_acc_lineup1])
    return all(math.isfinite(v) for v in values)


def _meets_promotion_gate(
    *,
    deltas: dict[str, float],
    rollback_triggered: bool,
    promotion_gate_mode: str,
    max_delta_minutes_mae_lineup0: float,
    max_delta_minutes_mae_lineup1: float,
    max_delta_minutes_gap_abs: float,
    min_delta_active_acc_lineup1: float,
    max_delta_active_count_mae: float,
) -> bool:
    if rollback_triggered:
        return False
    if str(promotion_gate_mode) == "prod_like":
        return bool(
            float(deltas["delta_minutes_mae_lineup1"]) <= float(max_delta_minutes_mae_lineup1)
            and float(deltas["delta_active_acc_lineup1"]) >= float(min_delta_active_acc_lineup1)
            and float(deltas["delta_active_count_mae"]) <= float(max_delta_active_count_mae)
        )
    return bool(
        float(deltas["delta_minutes_mae_lineup0"]) <= float(max_delta_minutes_mae_lineup0)
        and float(deltas["delta_minutes_mae_lineup1"]) <= float(max_delta_minutes_mae_lineup1)
        and float(deltas["delta_minutes_mae_gap_abs"]) <= float(max_delta_minutes_gap_abs)
        and float(deltas["delta_active_count_mae"]) <= float(max_delta_active_count_mae)
    )


def _parse_seed_list(value: str | None, *, base_seed: int, min_seeds: int) -> list[int]:
    seeds: list[int] = []
    if value:
        for token in str(value).split(","):
            tok = token.strip()
            if not tok:
                continue
            seeds.append(int(tok))

    if not seeds:
        seeds = [int(base_seed), int(base_seed) + 17, int(base_seed) + 29]

    seeds = [int(base_seed), *[int(s) for s in seeds if int(s) != int(base_seed)]]

    ordered: list[int] = []
    seen: set[int] = set()
    for seed in seeds:
        if int(seed) in seen:
            continue
        seen.add(int(seed))
        ordered.append(int(seed))

    while len(ordered) < int(min_seeds):
        ordered.append(int(ordered[-1]) + 17)

    return ordered


def _safe_mean(values: list[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def _mean_deltas(seed_rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = [
        "delta_minutes_mae_lineup0",
        "delta_minutes_mae_lineup1",
        "delta_minutes_mae_gap_abs",
        "delta_active_acc_lineup0",
        "delta_active_acc_lineup1",
        "delta_active_count_mae",
        "delta_possessions_proxy_mae",
    ]
    out: dict[str, float] = {}
    for key in keys:
        out[key] = _safe_mean([
            _float_or_nan((r.get("deltas_vs_baseline", {}) or {}).get(key)) for r in seed_rows
        ])
    return out


def _meets_multi_seed_promotion_gate(
    *,
    seed_rows: list[dict[str, Any]],
    min_required: int,
    require_all_pass: bool,
    require_mean_gains: bool,
    promotion_gate_mode: str,
    max_mean_delta_minutes_mae_lineup1: float,
    max_mean_delta_minutes_gap_abs: float,
    min_mean_delta_active_acc_lineup1: float,
) -> bool:
    if not seed_rows:
        return False

    ok_rows = [r for r in seed_rows if str(r.get("status")) == "ok"]
    if len(ok_rows) < int(min_required):
        return False

    pass_rows = [r for r in ok_rows if bool(r.get("promotion_gate_pass", False))]
    if len(pass_rows) < int(min_required):
        return False
    if bool(require_all_pass) and len(pass_rows) < len(seed_rows):
        return False

    mean_deltas = _mean_deltas(pass_rows)
    if not math.isfinite(float(mean_deltas["delta_minutes_mae_lineup1"])):
        return False
    if str(promotion_gate_mode) == "prod_like":
        if not math.isfinite(float(mean_deltas["delta_active_acc_lineup1"])):
            return False
    elif not math.isfinite(float(mean_deltas["delta_minutes_mae_gap_abs"])):
        return False
    if float(mean_deltas["delta_minutes_mae_lineup1"]) > float(max_mean_delta_minutes_mae_lineup1):
        return False
    if str(promotion_gate_mode) == "prod_like":
        if float(mean_deltas["delta_active_acc_lineup1"]) < float(min_mean_delta_active_acc_lineup1):
            return False
    elif float(mean_deltas["delta_minutes_mae_gap_abs"]) > float(max_mean_delta_minutes_gap_abs):
        return False

    if bool(require_mean_gains):
        if float(mean_deltas["delta_minutes_mae_lineup0"]) > 0.0:
            return False
        if float(mean_deltas["delta_active_count_mae"]) > 0.0:
            return False

    return True


def _run(cmd: list[str], *, log_path: Path) -> subprocess.CompletedProcess[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    proc = subprocess.run(
        cmd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    output = proc.stdout or ""
    log_path.write_text(output, encoding="utf-8")
    return proc


def _print_cmd(prefix: str, cmd: list[str]) -> None:
    print(f"{prefix}: {' '.join(shlex.quote(c) for c in cmd)}", flush=True)


def _select_trials(args: argparse.Namespace) -> list[Trial]:
    if args.trials_json:
        return _read_trials_file(Path(args.trials_json).expanduser().resolve())
    if str(args.trial_preset) == "all_losses":
        return _default_all_losses_trials()
    if str(args.trial_preset) == "sparse_recall":
        return _default_sparse_recall_trials()
    if str(args.trial_preset) == "sparse_hurdle":
        return _default_sparse_hurdle_trials()
    if str(args.trial_preset) == "optimizer_quality":
        return _default_optimizer_trials()
    return _default_trials()


def _build_train_cmd(
    *,
    args: argparse.Namespace,
    dataset_dir: Path,
    run_dir: Path,
    seed: int,
    params: dict[str, Any],
) -> list[str]:
    params_effective = dict(params)

    def _as_float(key: str, default: float = 0.0) -> float:
        try:
            return float(params_effective.get(key, default))
        except Exception:
            return float(default)

    # Auto-enable heads when corresponding losses are non-zero so trial JSONs can
    # focus on loss weights without repeating every structural flag.
    if (
        _as_float("w_poss_nll") > 0.0
        or _as_float("w_backbone_nll") > 0.0
        or _as_float("w_three_pa_nll") > 0.0
        or _as_float("w_poss_regression") > 0.0
    ):
        params_effective.setdefault("enable_possession_backbone", True)
    if _as_float("w_three_pa_nll") > 0.0:
        params_effective.setdefault("enable_three_pa_share", True)
    if _as_float("w_efficiency_nll") > 0.0:
        params_effective.setdefault("enable_efficiency_head", True)
    if _as_float("w_usage_share_nll") > 0.0:
        params_effective.setdefault("enable_usage_share_head", True)
    if _as_float("w_minutes_hurdle_nll") > 0.0:
        params_effective.setdefault("enable_minutes_hurdle_head", True)

    train_cmd = [
        PYTHON_EXE,
        "-m",
        "scripts.rotation.train_game_transformer_v2",
        "--dataset-dir",
        str(dataset_dir),
        "--out-dir",
        str(run_dir),
        "--epochs",
        str(int(args.epochs)),
        "--val-days",
        str(int(args.train_val_days)),
        "--batch-size",
        str(int(args.batch_size)),
        "--num-workers",
        str(int(args.num_workers)),
        "--device",
        str(args.device),
        "--seed",
        str(int(seed)),
        "--w-minutes",
        str(float(args.w_minutes)),
        "--phase2-anchor-start-weight",
        str(float(args.phase2_anchor_start_weight)),
        "--enable-phase2-flow",
        "--phase2-nll-guard-ratio",
        str(float(args.phase2_nll_guard_ratio)),
        "--phase2-nll-guard-abs",
        str(float(args.phase2_nll_guard_abs)),
        "--phase2-nll-guard-ema-alpha",
        str(float(args.phase2_nll_guard_ema_alpha)),
        "--phase2-nll-guard-consecutive-batches",
        str(int(args.phase2_nll_guard_consecutive_batches)),
        "--phase2-max-backoffs-before-rollback",
        str(int(args.phase2_max_backoffs_before_rollback)),
        "--phase2-min-a2-scale",
        str(float(args.phase2_min_a2_scale)),
    ]
    if args.init_model_pt:
        train_cmd.extend(["--init-model-pt", str(Path(args.init_model_pt).expanduser().resolve())])
    train_cmd.extend(_to_cli_args(params_effective))
    return train_cmd


def _build_eval_cmd(
    *,
    args: argparse.Namespace,
    dataset_dir: Path,
    run_dir: Path,
    eval_json: Path,
    params: dict[str, Any],
) -> list[str]:
    eval_batch_size = int(params.get("batch_size", int(args.batch_size)))
    eval_device = str(args.eval_device or args.device)
    eval_cmd = [
        PYTHON_EXE,
        "-m",
        "scripts.rotation.eval_game_transformer_v2",
        "--run-dir",
        str(run_dir),
        "--dataset-dir",
        str(dataset_dir),
        "--val-days",
        str(int(args.eval_val_days)),
        "--batch-size",
        str(int(eval_batch_size)),
        "--num-workers",
        str(int(args.num_workers)),
        "--device",
        str(eval_device),
        "--active-threshold",
        "4.0",
        "--out-json",
        str(eval_json),
    ]
    return eval_cmd


def _build_world_cmd(
    *,
    args: argparse.Namespace,
    dataset_dir: Path,
    run_dir: Path,
    world_summary: Path,
    worlds_parquet: Path,
) -> list[str]:
    cmd = [
        PYTHON_EXE,
        "-m",
        "scripts.rotation.generate_worlds_game_transformer_v2",
        "--run-dir",
        str(run_dir),
        "--dataset-dir",
        str(dataset_dir),
        "--val-days",
        str(int(args.eval_val_days)),
        "--num-games",
        str(int(args.world_num_games)),
        "--num-worlds",
        str(int(args.world_num_worlds)),
        "--batch-size",
        "1",
        "--num-workers",
        "0",
        "--device",
        str(args.device),
        "--active-temperature",
        str(float(args.world_active_temperature)),
        "--make-model",
        str(args.world_make_model),
        "--allocation-source",
        str(args.world_allocation_source),
        "--strict-contracts",
        "--out-parquet",
        str(worlds_parquet),
        "--out-summary-json",
        str(world_summary),
    ]
    if str(args.world_allocation_source) == "blend":
        cmd.extend(
            [
                "--allocation-blend-alpha",
                str(float(args.world_allocation_blend_alpha)),
            ]
        )
    return cmd


def _build_realism_cmd(
    *,
    dataset_dir: Path,
    worlds_parquet: Path,
    out_json: Path,
    name: str,
) -> list[str]:
    return [
        PYTHON_EXE,
        "-m",
        "scripts.rotation.eval_make_rate_calibration",
        "--dataset-dir",
        str(dataset_dir),
        "--worlds-parquet",
        str(worlds_parquet),
        "--name",
        str(name),
        "--out-json",
        str(out_json),
    ]


def _meets_realism_gate(metrics: dict[str, Any], *, args: argparse.Namespace) -> tuple[bool, list[str]]:
    failures: list[str] = []

    def _val(key: str) -> float:
        try:
            return float(metrics.get(key, float("nan")))
        except Exception:
            return float("nan")

    def _check_max(key: str, threshold: float | None) -> None:
        if threshold is None:
            return
        v = _val(key)
        if not math.isfinite(v) or v > float(threshold):
            failures.append(f"{key}>{threshold}")

    def _check_min(key: str, threshold: float | None) -> None:
        if threshold is None:
            return
        v = _val(key)
        if not math.isfinite(v) or v < float(threshold):
            failures.append(f"{key}<{threshold}")

    def _check_abs_max(key: str, threshold: float | None) -> None:
        if threshold is None:
            return
        v = _val(key)
        if not math.isfinite(v) or abs(v) > float(threshold):
            failures.append(f"abs({key})>{threshold}")

    _check_max("pts_mae", args.realism_max_pts_mae)
    _check_abs_max("pts_bias_mean", args.realism_max_abs_pts_bias)
    _check_abs_max("star_bias_pts_25_34", args.realism_max_abs_star_bias)
    _check_abs_max("elite_bias_pts_35plus", args.realism_max_abs_elite_bias)
    _check_max("spread_mae_vs_vegas", args.realism_max_spread_mae_vs_vegas)
    _check_min("spread_span_ratio", args.realism_min_spread_span_ratio)
    _check_min("spread_corr_vs_vegas", args.realism_min_spread_corr_vs_vegas)
    _check_max("total_mae_vs_vegas", args.realism_max_total_mae_vs_vegas)
    _check_min("total_span_ratio", args.realism_min_total_span_ratio)
    _check_max("p90_calibration_error_abs", args.realism_max_p90_calib_error)
    _check_max("p95_calibration_error_abs", args.realism_max_p95_calib_error)
    _check_abs_max("top1_share_bias_pts", args.realism_max_top1_share_bias)
    _check_abs_max("top2_share_bias_pts", args.realism_max_top2_share_bias)

    return len(failures) == 0, failures


def _run_trial_once(
    *,
    args: argparse.Namespace,
    dataset_dir: Path,
    baseline: EvalMetrics,
    trial_name: str,
    params: dict[str, Any],
    run_root: Path,
    seed: int,
    step_prefix: str,
    require_world_check: bool,
    dry_run: bool,
    snapshot_guard: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run_dir = run_root / "run"
    eval_json = run_root / f"eval_slices_{int(args.eval_val_days)}d.json"
    worlds_parquet = run_root / "worlds.parquet"

    result: dict[str, Any] = {
        "trial_name": str(trial_name),
        "params": dict(params),
        "seed": int(seed),
        "run_dir": str(run_dir),
        "eval_json": str(eval_json),
        "worlds_parquet": str(worlds_parquet),
        "status": "planned",
    }

    train_cmd = _build_train_cmd(
        args=args,
        dataset_dir=dataset_dir,
        run_dir=run_dir,
        seed=int(seed),
        params=params,
    )
    eval_cmd = _build_eval_cmd(
        args=args,
        dataset_dir=dataset_dir,
        run_dir=run_dir,
        eval_json=eval_json,
        params=params,
    )

    _print_cmd(f"{step_prefix} train", train_cmd)
    if dry_run:
        result["status"] = "dry_run"
        return result

    train_proc = _run(train_cmd, log_path=run_root / "train.log")
    result["train_rc"] = int(train_proc.returncode)
    if train_proc.returncode != 0:
        result["status"] = "train_failed"
        return result

    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        result["status"] = "missing_summary"
        return result

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    stability = summary.get("phase2_stability", {}) or {}
    rollback = bool(stability.get("rollback_triggered", False))
    result["rollback_triggered"] = rollback
    result["phase2_backoff_count"] = int(stability.get("backoff_count", 0))
    result["phase2_final_a2_scale"] = _float_or_nan(stability.get("final_a2_scale"))

    _print_cmd(f"{step_prefix} eval", eval_cmd)
    eval_proc = _run(eval_cmd, log_path=run_root / "eval.log")
    result["eval_rc"] = int(eval_proc.returncode)
    result["eval_device"] = str(args.eval_device or args.device)
    if (
        eval_proc.returncode != 0
        and bool(args.eval_retry_cpu_on_failure)
        and str(args.eval_device or args.device).startswith("cuda")
    ):
        fallback_eval_json = run_root / f"eval_slices_{int(args.eval_val_days)}d_cpu_fallback.json"
        fallback_cmd = _build_eval_cmd(
            args=argparse.Namespace(**{**vars(args), "eval_device": "cpu"}),
            dataset_dir=dataset_dir,
            run_dir=run_dir,
            eval_json=fallback_eval_json,
            params=params,
        )
        _print_cmd(f"{step_prefix} eval-cpu-fallback", fallback_cmd)
        fallback_proc = _run(fallback_cmd, log_path=run_root / "eval_cpu_fallback.log")
        result["eval_cpu_fallback_rc"] = int(fallback_proc.returncode)
        result["eval_cpu_fallback_json"] = str(fallback_eval_json)
        if fallback_proc.returncode == 0 and fallback_eval_json.exists():
            eval_json = fallback_eval_json
            result["eval_json"] = str(eval_json)
            result["eval_device"] = "cpu"
            result["eval_rc"] = 0
    if int(result.get("eval_rc", eval_proc.returncode)) != 0 or not eval_json.exists():
        result["status"] = "eval_failed"
        return result

    eval_payload = json.loads(eval_json.read_text(encoding="utf-8"))
    metrics = _load_eval_metrics(eval_json)
    if not _is_finite_eval(metrics, require_active_acc=bool(str(args.promotion_gate_mode) == "prod_like")):
        result["status"] = "eval_nonfinite"
        result["metrics"] = metrics.__dict__
        return result

    deltas = _diff_metrics(metrics, baseline)
    score = _composite_score(deltas, promotion_gate_mode=str(args.promotion_gate_mode))
    single_gate_pass = _meets_promotion_gate(
        deltas=deltas,
        rollback_triggered=rollback,
        promotion_gate_mode=str(args.promotion_gate_mode),
        max_delta_minutes_mae_lineup0=float(args.max_delta_minutes_mae_lineup0),
        max_delta_minutes_mae_lineup1=float(args.max_delta_minutes_mae_lineup1),
        max_delta_minutes_gap_abs=float(args.max_delta_minutes_gap_abs),
        min_delta_active_acc_lineup1=float(args.min_delta_active_acc_lineup1),
        max_delta_active_count_mae=float(args.max_delta_active_count_mae),
    )

    world_ok = True
    realism_ok = True
    realism_failures: list[str] = []
    if bool(require_world_check):
        world_summary = run_root / "world_summary.json"
        world_cmd = _build_world_cmd(
            args=args,
            dataset_dir=dataset_dir,
            run_dir=run_dir,
            world_summary=world_summary,
            worlds_parquet=worlds_parquet,
        )
        _print_cmd(f"{step_prefix} world", world_cmd)
        world_proc = _run(world_cmd, log_path=run_root / "world.log")
        result["world_check_rc"] = int(world_proc.returncode)
        result["world_check_summary_json"] = str(world_summary)
        world_ok = bool(world_proc.returncode == 0)
        result["world_contract_pass"] = world_ok
        if world_summary.exists():
            try:
                result["world_check"] = json.loads(world_summary.read_text(encoding="utf-8"))
            except Exception:
                result["world_check"] = None
    else:
        result["world_contract_pass"] = True

    if bool(args.realism_gate):
        realism_json = run_root / "realism_eval.json"
        result["realism_eval_json"] = str(realism_json)
        if not world_ok or not worlds_parquet.exists():
            realism_ok = False
            realism_failures.append("missing_worlds_parquet")
        else:
            realism_name = f"{trial_name}_seed{int(seed)}"
            realism_cmd = _build_realism_cmd(
                dataset_dir=dataset_dir,
                worlds_parquet=worlds_parquet,
                out_json=realism_json,
                name=str(realism_name),
            )
            _print_cmd(f"{step_prefix} realism", realism_cmd)
            realism_proc = _run(realism_cmd, log_path=run_root / "realism.log")
            result["realism_eval_rc"] = int(realism_proc.returncode)
            if realism_proc.returncode != 0 or not realism_json.exists():
                realism_ok = False
                realism_failures.append("realism_eval_failed")
            else:
                payload = json.loads(realism_json.read_text(encoding="utf-8"))
                realism_metrics = payload.get("metrics", payload)
                result["realism_metrics"] = realism_metrics
                realism_ok, realism_failures = _meets_realism_gate(realism_metrics, args=args)
        result["realism_gate_pass"] = bool(realism_ok)
        result["realism_gate_failures"] = list(realism_failures)

    snapshot_ok = True
    if snapshot_guard is not None:
        snapshot_metrics = _deterministic_snapshot_metrics(
            bundle_dir=run_dir,
            snapshot_features_df=snapshot_guard["features_df"],
            device=snapshot_guard["device"],
        )
        snapshot_ok, snapshot_failures, snapshot_deltas = _meets_snapshot_guard(
            candidate=snapshot_metrics,
            baseline=snapshot_guard["baseline_metrics"],
            max_det_active_prob_delta=float(snapshot_guard["max_det_active_prob_delta"]),
            min_det_prop_star_minutes_delta=float(snapshot_guard["min_det_prop_star_minutes_delta"]),
        )
        result["snapshot_metrics"] = snapshot_metrics
        result["snapshot_deltas_vs_baseline"] = snapshot_deltas
        result["snapshot_guard_pass"] = bool(snapshot_ok)
        result["snapshot_guard_failures"] = list(snapshot_failures)
    else:
        result["snapshot_guard_pass"] = True

    result["metrics"] = metrics.__dict__
    result["sparse_rotation_diagnostics"] = eval_payload.get("sparse_rotation_diagnostics", {})
    result["deltas_vs_baseline"] = deltas
    result["composite_score"] = float(score)
    result["single_run_gate_pass"] = bool(single_gate_pass)
    result["promotion_gate_pass"] = bool(single_gate_pass and world_ok and realism_ok and snapshot_ok)
    result["status"] = "ok"
    return result


def _write_leaderboard(rows: list[dict[str, Any]], *, csv_path: Path, md_path: Path, title: str) -> None:
    if not rows:
        return

    leaderboard = pd.DataFrame(rows)
    leaderboard.to_csv(csv_path, index=False)

    cols = list(leaderboard.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = [
        "| " + " | ".join(str(row[c]) for c in cols) + " |"
        for row in leaderboard.to_dict(orient="records")
    ]
    md_lines = [
        f"# {title}",
        "",
        f"Generated at: {datetime.now(timezone.utc).isoformat()}",
        "",
        header,
        sep,
        *body,
        "",
    ]
    md_path.write_text("\n".join(md_lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=str, default=None)
    parser.add_argument("--baseline-eval-json", type=str, required=True)
    parser.add_argument("--init-model-pt", type=str, default=None)
    parser.add_argument("--trials-json", type=str, default=None)
    parser.add_argument(
        "--trial-preset",
        type=str,
        default="anchor_recovery",
        choices=["anchor_recovery", "optimizer_quality", "all_losses", "sparse_recall", "sparse_hurdle"],
        help="Default trial grid to use when --trials-json is not provided.",
    )
    parser.add_argument("--sweep-root", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train-val-days", type=int, default=60)
    parser.add_argument("--eval-val-days", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval-device", type=str, default=None)
    parser.add_argument("--eval-retry-cpu-on-failure", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--phase2-anchor-start-weight", type=float, default=1.0)
    parser.add_argument("--w-minutes", type=float, default=1.0)
    parser.add_argument("--phase2-nll-guard-ratio", type=float, default=3.0)
    parser.add_argument("--phase2-nll-guard-abs", type=float, default=25.0)
    parser.add_argument("--phase2-nll-guard-ema-alpha", type=float, default=0.1)
    parser.add_argument("--phase2-nll-guard-consecutive-batches", type=int, default=2)
    parser.add_argument("--phase2-max-backoffs-before-rollback", type=int, default=3)
    parser.add_argument("--phase2-min-a2-scale", type=float, default=0.125)
    parser.add_argument(
        "--promotion-gate-mode",
        type=str,
        default="prod_like",
        choices=["prod_like", "parity_gap"],
        help="prod_like hard-gates the lineup_available=1 slice; parity_gap preserves the legacy raw parity-gap blocker.",
    )
    parser.add_argument("--max-delta-minutes-mae-lineup0", type=float, default=0.12)
    parser.add_argument("--max-delta-minutes-mae-lineup1", type=float, default=0.15)
    parser.add_argument("--max-delta-minutes-gap-abs", type=float, default=0.05)
    parser.add_argument("--min-delta-active-acc-lineup1", type=float, default=-0.01)
    parser.add_argument("--max-delta-active-count-mae", type=float, default=0.10)
    parser.add_argument("--skip-world-contract-check", action="store_true")
    parser.add_argument(
        "--require-world-contract-check-all",
        action="store_true",
        help="Run strict world contract checks for every candidate (and for each multi-seed run).",
    )
    parser.add_argument(
        "--realism-gate",
        action="store_true",
        help="Run additional realism diagnostics (spread, tails, concentration) and gate promotions.",
    )
    parser.add_argument("--realism-max-pts-mae", type=float, default=None)
    parser.add_argument("--realism-max-abs-pts-bias", type=float, default=None)
    parser.add_argument("--realism-max-abs-star-bias", type=float, default=None)
    parser.add_argument("--realism-max-abs-elite-bias", type=float, default=None)
    parser.add_argument("--realism-max-spread-mae-vs-vegas", type=float, default=None)
    parser.add_argument("--realism-min-spread-span-ratio", type=float, default=None)
    parser.add_argument("--realism-min-spread-corr-vs-vegas", type=float, default=None)
    parser.add_argument("--realism-max-total-mae-vs-vegas", type=float, default=None)
    parser.add_argument("--realism-min-total-span-ratio", type=float, default=None)
    parser.add_argument("--realism-max-p90-calib-error", type=float, default=None)
    parser.add_argument("--realism-max-p95-calib-error", type=float, default=None)
    parser.add_argument("--realism-max-top1-share-bias", type=float, default=None)
    parser.add_argument("--realism-max-top2-share-bias", type=float, default=None)
    parser.add_argument("--world-num-games", type=int, default=1)
    parser.add_argument("--world-num-worlds", type=int, default=64)
    parser.add_argument("--world-active-temperature", type=float, default=1.0)
    parser.add_argument(
        "--world-make-model",
        type=str,
        default="legacy",
        choices=["legacy", "beta_binomial_ft", "beta_binomial_fg", "beta_binomial_all"],
    )
    parser.add_argument(
        "--world-allocation-source",
        type=str,
        default="emergent",
        choices=["emergent", "usage_head", "blend"],
    )
    parser.add_argument("--world-allocation-blend-alpha", type=float, default=0.5)
    parser.add_argument("--multi-seed-top-k", type=int, default=0)
    parser.add_argument(
        "--multi-seed-list",
        type=str,
        default="",
        help="Comma-separated seeds for confirmation. Base seed is auto-included if missing.",
    )
    parser.add_argument("--multi-seed-min-seeds", type=int, default=3)
    parser.add_argument("--multi-seed-require-all-pass", action="store_true")
    parser.add_argument("--multi-seed-require-mean-gains", action="store_true")
    parser.add_argument("--multi-seed-max-mean-delta-minutes-mae-lineup1", type=float, default=0.05)
    parser.add_argument("--multi-seed-max-mean-delta-minutes-gap-abs", type=float, default=0.05)
    parser.add_argument("--multi-seed-min-mean-delta-active-acc-lineup1", type=float, default=-0.005)
    parser.add_argument("--auto-promote", action="store_true")
    parser.add_argument(
        "--snapshot-features-parquet",
        type=str,
        default=None,
        help="Optional live snapshot features parquet for deterministic same-snapshot guard checks.",
    )
    parser.add_argument(
        "--snapshot-baseline-bundle-dir",
        type=str,
        default=None,
        help="Baseline bundle/run dir used as the deterministic same-snapshot reference.",
    )
    parser.add_argument(
        "--snapshot-max-det-active-prob-delta",
        type=float,
        default=0.02,
        help="Maximum allowed candidate - baseline delta for det_active_prob_mean.",
    )
    parser.add_argument(
        "--snapshot-min-det-prop-star-minutes-delta",
        type=float,
        default=-0.50,
        help="Minimum allowed candidate - baseline delta for det_prop_star_minutes_mean.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if bool(args.realism_gate):
        if bool(args.skip_world_contract_check):
            print(
                "[realism_gate] overriding --skip-world-contract-check; realism gate requires world sampling",
                flush=True,
            )
            args.skip_world_contract_check = False
        for key, default in REALISM_GATE_DEFAULTS.items():
            if getattr(args, key) is None:
                setattr(args, key, float(default))

    dataset_dir = _resolve_dataset_dir(args.dataset_dir)
    baseline_eval_path = Path(args.baseline_eval_json).expanduser().resolve()
    if not baseline_eval_path.exists():
        raise FileNotFoundError(f"baseline eval json not found: {baseline_eval_path}")
    baseline = _load_eval_metrics(baseline_eval_path)
    if not _is_finite_eval(baseline, require_active_acc=bool(str(args.promotion_gate_mode) == "prod_like")):
        raise ValueError("baseline eval has non-finite metrics")

    snapshot_guard: dict[str, Any] | None = None
    if args.snapshot_features_parquet or args.snapshot_baseline_bundle_dir:
        if not args.snapshot_features_parquet or not args.snapshot_baseline_bundle_dir:
            raise ValueError(
                "--snapshot-features-parquet and --snapshot-baseline-bundle-dir must be provided together"
            )
        snapshot_features_path = Path(args.snapshot_features_parquet).expanduser().resolve()
        if not snapshot_features_path.exists():
            raise FileNotFoundError(f"snapshot features parquet not found: {snapshot_features_path}")
        snapshot_baseline_dir = Path(args.snapshot_baseline_bundle_dir).expanduser().resolve()
        if not snapshot_baseline_dir.exists():
            raise FileNotFoundError(f"snapshot baseline bundle dir not found: {snapshot_baseline_dir}")
        snapshot_features_df = _load_snapshot_features(snapshot_features_path)
        snapshot_device = resolve_torch_device(str(args.eval_device or args.device))
        snapshot_baseline_metrics = _deterministic_snapshot_metrics(
            bundle_dir=snapshot_baseline_dir,
            snapshot_features_df=snapshot_features_df,
            device=snapshot_device,
        )
        snapshot_guard = {
            "features_path": str(snapshot_features_path),
            "baseline_bundle_dir": str(snapshot_baseline_dir),
            "features_df": snapshot_features_df,
            "device": snapshot_device,
            "baseline_metrics": snapshot_baseline_metrics,
            "max_det_active_prob_delta": float(args.snapshot_max_det_active_prob_delta),
            "min_det_prop_star_minutes_delta": float(args.snapshot_min_det_prop_star_minutes_delta),
        }
        print(
            (
                "[snapshot_guard] enabled "
                f"det_active_prob_mean_baseline={snapshot_baseline_metrics['det_active_prob_mean']:.6f} "
                f"det_prop_star_minutes_mean_baseline={snapshot_baseline_metrics['det_prop_star_minutes_mean']:.6f}"
            ),
            flush=True,
        )

    root_default = paths.get_data_root() / "training" / "runs" / f"game_transformer_v2_phase2_sweep_{_utc_now_compact()}"
    sweep_root = Path(args.sweep_root).expanduser().resolve() if args.sweep_root else root_default
    sweep_root.mkdir(parents=True, exist_ok=True)
    trials_dir = sweep_root / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)

    trials = _select_trials(args)
    if not trials:
        raise ValueError("no trials resolved")

    require_world_check_all = bool(args.require_world_contract_check_all)
    if not require_world_check_all and str(args.trial_preset) == "optimizer_quality":
        # Quality pass defaults to contract verification on every candidate.
        require_world_check_all = True
    if bool(args.realism_gate):
        require_world_check_all = True

    multi_seed_enabled = int(args.multi_seed_top_k) > 0
    multi_seed_list = _parse_seed_list(
        args.multi_seed_list,
        base_seed=int(args.seed),
        min_seeds=max(1, int(args.multi_seed_min_seeds)),
    )

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(dataset_dir),
        "baseline_eval_json": str(baseline_eval_path),
        "baseline_metrics": baseline.__dict__,
        "sweep_root": str(sweep_root),
        "trial_preset": str(args.trial_preset),
        "epochs": int(args.epochs),
        "train_val_days": int(args.train_val_days),
        "eval_val_days": int(args.eval_val_days),
        "batch_size": int(args.batch_size),
        "device": str(args.device),
        "seed": int(args.seed),
        "init_model_pt": str(Path(args.init_model_pt).expanduser().resolve()) if args.init_model_pt else None,
        "promotion_gate": {
            "mode": str(args.promotion_gate_mode),
            "max_delta_minutes_mae_lineup0": float(args.max_delta_minutes_mae_lineup0),
            "max_delta_minutes_mae_lineup1": float(args.max_delta_minutes_mae_lineup1),
            "max_delta_minutes_gap_abs": float(args.max_delta_minutes_gap_abs),
            "min_delta_active_acc_lineup1": float(args.min_delta_active_acc_lineup1),
            "max_delta_active_count_mae": float(args.max_delta_active_count_mae),
        },
        "realism_gate": {
            "enabled": bool(args.realism_gate),
            "thresholds": {
                "realism_max_pts_mae": args.realism_max_pts_mae,
                "realism_max_abs_pts_bias": args.realism_max_abs_pts_bias,
                "realism_max_abs_star_bias": args.realism_max_abs_star_bias,
                "realism_max_abs_elite_bias": args.realism_max_abs_elite_bias,
                "realism_max_spread_mae_vs_vegas": args.realism_max_spread_mae_vs_vegas,
                "realism_min_spread_span_ratio": args.realism_min_spread_span_ratio,
                "realism_min_spread_corr_vs_vegas": args.realism_min_spread_corr_vs_vegas,
                "realism_max_total_mae_vs_vegas": args.realism_max_total_mae_vs_vegas,
                "realism_min_total_span_ratio": args.realism_min_total_span_ratio,
                "realism_max_p90_calib_error": args.realism_max_p90_calib_error,
                "realism_max_p95_calib_error": args.realism_max_p95_calib_error,
                "realism_max_top1_share_bias": args.realism_max_top1_share_bias,
                "realism_max_top2_share_bias": args.realism_max_top2_share_bias,
            },
        },
        "world_check": {
            "skip_world_contract_check": bool(args.skip_world_contract_check),
            "require_world_contract_check_all": bool(require_world_check_all),
            "world_num_games": int(args.world_num_games),
            "world_num_worlds": int(args.world_num_worlds),
            "world_active_temperature": float(args.world_active_temperature),
            "world_make_model": str(args.world_make_model),
            "world_allocation_source": str(args.world_allocation_source),
            "world_allocation_blend_alpha": float(args.world_allocation_blend_alpha),
        },
        "multi_seed": {
            "enabled": bool(multi_seed_enabled),
            "promotion_gate_mode": str(args.promotion_gate_mode),
            "top_k": int(args.multi_seed_top_k),
            "seed_list": list(multi_seed_list),
            "min_seeds": int(args.multi_seed_min_seeds),
            "require_all_pass": bool(args.multi_seed_require_all_pass),
            "require_mean_gains": bool(args.multi_seed_require_mean_gains),
            "max_mean_delta_minutes_mae_lineup1": float(args.multi_seed_max_mean_delta_minutes_mae_lineup1),
            "max_mean_delta_minutes_gap_abs": float(args.multi_seed_max_mean_delta_minutes_gap_abs),
            "min_mean_delta_active_acc_lineup1": float(args.multi_seed_min_mean_delta_active_acc_lineup1),
        },
        "snapshot_guard": {
            "enabled": bool(snapshot_guard is not None),
            "features_path": None if snapshot_guard is None else str(snapshot_guard["features_path"]),
            "baseline_bundle_dir": None if snapshot_guard is None else str(snapshot_guard["baseline_bundle_dir"]),
            "max_det_active_prob_delta": float(args.snapshot_max_det_active_prob_delta),
            "min_det_prop_star_minutes_delta": float(args.snapshot_min_det_prop_star_minutes_delta),
            "baseline_metrics": None if snapshot_guard is None else snapshot_guard["baseline_metrics"],
        },
        "trials": [{"name": t.name, "params": t.params} for t in trials],
    }
    (sweep_root / "sweep_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    results: list[dict[str, Any]] = []
    trials_by_name = {t.name: t for t in trials}

    for idx, trial in enumerate(trials, start=1):
        trial_root = trials_dir / trial.name
        trial_result = _run_trial_once(
            args=args,
            dataset_dir=dataset_dir,
            baseline=baseline,
            trial_name=trial.name,
            params=trial.params,
            run_root=trial_root,
            seed=int(args.seed),
            step_prefix=f"[phase2_sweep] trial {idx}/{len(trials)} {trial.name}",
            require_world_check=bool(require_world_check_all and not args.skip_world_contract_check),
            dry_run=bool(args.dry_run),
            snapshot_guard=snapshot_guard,
        )
        trial_result["trial_index"] = idx
        results.append(trial_result)

    (sweep_root / "trial_results.json").write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")

    ok_rows = [r for r in results if r.get("status") == "ok"]
    ranked: list[dict[str, Any]] = sorted(ok_rows, key=lambda r: (float(r.get("composite_score", float("inf"))), str(r.get("trial_name", ""))))

    if ranked:
        leaderboard_rows = [
            {
                "trial_name": str(r.get("trial_name")),
                "seed": int(r.get("seed", int(args.seed))),
                "composite_score": float(r.get("composite_score", float("nan"))),
                "single_run_gate_pass": bool(r.get("single_run_gate_pass", False)),
                "world_contract_pass": bool(r.get("world_contract_pass", False)),
                "realism_gate_pass": bool(r.get("realism_gate_pass", False)) if args.realism_gate else True,
                "snapshot_guard_pass": bool(r.get("snapshot_guard_pass", True)),
                "promotion_gate_pass": bool(r.get("promotion_gate_pass", False)),
                "minutes_mae_lineup0": _float_or_nan((r.get("metrics", {}) or {}).get("minutes_mae_lineup0")),
                "minutes_mae_lineup1": _float_or_nan((r.get("metrics", {}) or {}).get("minutes_mae_lineup1")),
                "minutes_mae_gap_abs": _float_or_nan((r.get("metrics", {}) or {}).get("minutes_mae_gap_abs")),
                "active_acc_lineup0": _float_or_nan((r.get("metrics", {}) or {}).get("active_acc_lineup0")),
                "active_acc_lineup1": _float_or_nan((r.get("metrics", {}) or {}).get("active_acc_lineup1")),
                "active_count_mae": _float_or_nan((r.get("metrics", {}) or {}).get("active_count_mae")),
                "delta_minutes_mae_lineup0": _float_or_nan((r.get("deltas_vs_baseline", {}) or {}).get("delta_minutes_mae_lineup0")),
                "delta_minutes_mae_lineup1": _float_or_nan((r.get("deltas_vs_baseline", {}) or {}).get("delta_minutes_mae_lineup1")),
                "delta_minutes_mae_gap_abs": _float_or_nan((r.get("deltas_vs_baseline", {}) or {}).get("delta_minutes_mae_gap_abs")),
                "delta_active_acc_lineup0": _float_or_nan((r.get("deltas_vs_baseline", {}) or {}).get("delta_active_acc_lineup0")),
                "delta_active_acc_lineup1": _float_or_nan((r.get("deltas_vs_baseline", {}) or {}).get("delta_active_acc_lineup1")),
                "delta_active_count_mae": _float_or_nan((r.get("deltas_vs_baseline", {}) or {}).get("delta_active_count_mae")),
                "delta_det_active_prob_mean": _float_or_nan(
                    (r.get("snapshot_deltas_vs_baseline", {}) or {}).get("delta_det_active_prob_mean")
                ),
                "delta_det_prop_star_minutes_mean": _float_or_nan(
                    (r.get("snapshot_deltas_vs_baseline", {}) or {}).get("delta_det_prop_star_minutes_mean")
                ),
                "rollback_triggered": bool(r.get("rollback_triggered", False)),
                "run_dir": str(r.get("run_dir", "")),
                "eval_json": str(r.get("eval_json", "")),
            }
            for r in ranked
        ]
        _write_leaderboard(
            leaderboard_rows,
            csv_path=sweep_root / "leaderboard.csv",
            md_path=sweep_root / "leaderboard.md",
            title="GameTransformerV2 Phase 2 Sweep Leaderboard",
        )

    promoted: dict[str, Any] | None = None

    multi_seed_results: list[dict[str, Any]] = []
    if multi_seed_enabled and not args.dry_run:
        passable_rows = [r for r in ranked if bool(r.get("promotion_gate_pass", False))]
        top_rows = passable_rows[: max(0, int(args.multi_seed_top_k))]

        for cand in top_rows:
            trial_name = str(cand["trial_name"])
            trial = trials_by_name[trial_name]
            trial_root = trials_dir / trial_name

            seed_rows: list[dict[str, Any]] = []
            for seed in multi_seed_list:
                if int(seed) == int(args.seed):
                    seed_row = dict(cand)
                    seed_row["seed"] = int(seed)
                    seed_row["reused_primary_seed_run"] = True
                    seed_rows.append(seed_row)
                    continue

                seed_root = trial_root / "multiseed" / f"seed_{int(seed)}"
                seed_row = _run_trial_once(
                    args=args,
                    dataset_dir=dataset_dir,
                    baseline=baseline,
                    trial_name=trial_name,
                    params=trial.params,
                    run_root=seed_root,
                    seed=int(seed),
                    step_prefix=f"[phase2_sweep][multiseed] {trial_name} seed={int(seed)}",
                    require_world_check=bool(not args.skip_world_contract_check),
                    dry_run=False,
                    snapshot_guard=snapshot_guard,
                )
                seed_rows.append(seed_row)

            passing_seed_rows = [r for r in seed_rows if str(r.get("status")) == "ok" and bool(r.get("promotion_gate_pass", False))]
            mean_deltas = _mean_deltas(passing_seed_rows) if passing_seed_rows else _mean_deltas(seed_rows)
            mean_composite = _safe_mean([_float_or_nan(r.get("composite_score")) for r in passing_seed_rows])

            multi_pass = _meets_multi_seed_promotion_gate(
                seed_rows=seed_rows,
                min_required=max(1, int(args.multi_seed_min_seeds)),
                require_all_pass=bool(args.multi_seed_require_all_pass),
                require_mean_gains=bool(args.multi_seed_require_mean_gains),
                promotion_gate_mode=str(args.promotion_gate_mode),
                max_mean_delta_minutes_mae_lineup1=float(args.multi_seed_max_mean_delta_minutes_mae_lineup1),
                max_mean_delta_minutes_gap_abs=float(args.multi_seed_max_mean_delta_minutes_gap_abs),
                min_mean_delta_active_acc_lineup1=float(args.multi_seed_min_mean_delta_active_acc_lineup1),
            )

            record = {
                "trial_name": trial_name,
                "params": trial.params,
                "seed_list": list(multi_seed_list),
                "seed_runs": seed_rows,
                "num_seed_runs": int(len(seed_rows)),
                "num_seed_ok": int(len([r for r in seed_rows if str(r.get("status")) == "ok"])),
                "num_seed_promotion_pass": int(len(passing_seed_rows)),
                "mean_deltas_vs_baseline": mean_deltas,
                "mean_composite_score": float(mean_composite),
                "multi_seed_promotion_pass": bool(multi_pass),
            }
            multi_seed_results.append(record)

        (sweep_root / "multiseed_results.json").write_text(
            json.dumps(multi_seed_results, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        if multi_seed_results:
            ms_rows = sorted(
                multi_seed_results,
                key=lambda r: (float(r.get("mean_composite_score", float("inf"))), str(r.get("trial_name", ""))),
            )
            ms_lb_rows = [
                {
                    "trial_name": str(r.get("trial_name", "")),
                    "mean_composite_score": _float_or_nan(r.get("mean_composite_score")),
                    "multi_seed_promotion_pass": bool(r.get("multi_seed_promotion_pass", False)),
                    "num_seed_runs": int(r.get("num_seed_runs", 0)),
                    "num_seed_ok": int(r.get("num_seed_ok", 0)),
                    "num_seed_promotion_pass": int(r.get("num_seed_promotion_pass", 0)),
                    "mean_delta_minutes_mae_lineup0": _float_or_nan((r.get("mean_deltas_vs_baseline", {}) or {}).get("delta_minutes_mae_lineup0")),
                    "mean_delta_minutes_mae_lineup1": _float_or_nan((r.get("mean_deltas_vs_baseline", {}) or {}).get("delta_minutes_mae_lineup1")),
                    "mean_delta_minutes_mae_gap_abs": _float_or_nan((r.get("mean_deltas_vs_baseline", {}) or {}).get("delta_minutes_mae_gap_abs")),
                    "mean_delta_active_acc_lineup1": _float_or_nan((r.get("mean_deltas_vs_baseline", {}) or {}).get("delta_active_acc_lineup1")),
                    "mean_delta_active_count_mae": _float_or_nan((r.get("mean_deltas_vs_baseline", {}) or {}).get("delta_active_count_mae")),
                }
                for r in ms_rows
            ]
            _write_leaderboard(
                ms_lb_rows,
                csv_path=sweep_root / "multiseed_leaderboard.csv",
                md_path=sweep_root / "multiseed_leaderboard.md",
                title="GameTransformerV2 Phase 2 Multi-Seed Leaderboard",
            )

    if args.auto_promote and not args.dry_run:
        if multi_seed_results:
            passing_ms = [r for r in multi_seed_results if bool(r.get("multi_seed_promotion_pass", False))]
            if passing_ms:
                best_cfg = sorted(
                    passing_ms,
                    key=lambda r: (float(r.get("mean_composite_score", float("inf"))), str(r.get("trial_name", ""))),
                )[0]
                pass_seeds = [
                    r
                    for r in best_cfg.get("seed_runs", [])
                    if str(r.get("status")) == "ok" and bool(r.get("promotion_gate_pass", False))
                ]
                best_seed_run = sorted(
                    pass_seeds,
                    key=lambda r: (float(r.get("composite_score", float("inf"))), int(r.get("seed", 10**9))),
                )[0]
                promoted = {
                    "promotion_mode": "multi_seed",
                    "trial_name": str(best_cfg.get("trial_name", "")),
                    "mean_composite_score": float(best_cfg.get("mean_composite_score", float("nan"))),
                    "mean_deltas_vs_baseline": best_cfg.get("mean_deltas_vs_baseline", {}),
                    "num_seed_runs": int(best_cfg.get("num_seed_runs", 0)),
                    "num_seed_promotion_pass": int(best_cfg.get("num_seed_promotion_pass", 0)),
                    "selected_seed": int(best_seed_run.get("seed", int(args.seed))),
                    "run_dir": str(best_seed_run.get("run_dir", "")),
                    "eval_json": str(best_seed_run.get("eval_json", "")),
                    "metrics": best_seed_run.get("metrics", {}),
                    "deltas_vs_baseline": best_seed_run.get("deltas_vs_baseline", {}),
                    "composite_score": _float_or_nan(best_seed_run.get("composite_score")),
                    "world_check_summary_json": best_seed_run.get("world_check_summary_json"),
                    "realism_eval_json": best_seed_run.get("realism_eval_json"),
                    "realism_gate_pass": bool(best_seed_run.get("realism_gate_pass", False)),
                }
        else:
            passing = [r for r in ranked if bool(r.get("promotion_gate_pass", False))]
            if passing:
                best = passing[0]
                promoted = {
                    "promotion_mode": "single_seed",
                    "trial_name": str(best.get("trial_name", "")),
                    "selected_seed": int(best.get("seed", int(args.seed))),
                    "run_dir": str(best.get("run_dir", "")),
                    "eval_json": str(best.get("eval_json", "")),
                    "metrics": best.get("metrics", {}),
                    "deltas_vs_baseline": best.get("deltas_vs_baseline", {}),
                    "composite_score": _float_or_nan(best.get("composite_score")),
                    "world_check_summary_json": best.get("world_check_summary_json"),
                    "realism_eval_json": best.get("realism_eval_json"),
                    "realism_gate_pass": bool(best.get("realism_gate_pass", False)),
                }

    if promoted is not None:
        (sweep_root / "promoted_phase2.json").write_text(
            json.dumps(promoted, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(dataset_dir),
        "baseline_eval_json": str(baseline_eval_path),
        "baseline_metrics": baseline.__dict__,
        "sweep_root": str(sweep_root),
        "trial_preset": str(args.trial_preset),
        "promotion_gate_mode": str(args.promotion_gate_mode),
        "num_trials": int(len(trials)),
        "num_completed": int(len([r for r in results if r.get("status") == "ok"])),
        "num_promotion_pass": int(len([r for r in results if bool(r.get("promotion_gate_pass"))])),
        "num_snapshot_guard_pass": int(len([r for r in results if bool(r.get("snapshot_guard_pass", True))])),
        "num_realism_pass": int(len([r for r in results if bool(r.get("realism_gate_pass"))]))
        if bool(args.realism_gate)
        else int(len([r for r in results if r.get("status") == "ok"])),
        "realism_gate_enabled": bool(args.realism_gate),
        "snapshot_guard_enabled": bool(snapshot_guard is not None),
        "world_check_all_candidates": bool(require_world_check_all and not args.skip_world_contract_check),
        "multi_seed": {
            "enabled": bool(multi_seed_enabled),
            "top_k": int(args.multi_seed_top_k),
            "seed_list": list(multi_seed_list),
            "num_configs_checked": int(len(multi_seed_results)),
            "num_configs_pass": int(len([r for r in multi_seed_results if bool(r.get("multi_seed_promotion_pass", False))])),
        },
        "promoted": promoted,
    }
    (sweep_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
