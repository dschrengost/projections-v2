from __future__ import annotations

from dataclasses import dataclass
import itertools
import json
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from projections.sim_v2.play_prob_policy import apply_play_prob_policy_with_diagnostics


@dataclass(frozen=True)
class PolicyObjectiveWeights:
    """Weights for scalar objective used to rank policy candidates."""

    brier: float = 1.0
    false_active_p90: float = 0.25
    fringe_false_active_p90: float = 0.50
    starter_under95: float = 0.20


def binary_log_loss(y_true: np.ndarray, p_pred: np.ndarray, *, eps: float = 1e-6) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p_pred, dtype=float)
    if y.size == 0:
        return float("nan")
    p = np.clip(p, float(eps), 1.0 - float(eps))
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def expected_calibration_error(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    *,
    bins: int = 10,
) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p_pred, dtype=float)
    if y.size == 0:
        return float("nan")
    n_bins = max(int(bins), 1)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = float(len(y))
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
        if not np.any(mask):
            continue
        w = float(mask.sum()) / total
        ece += w * abs(float(np.mean(p[mask])) - float(np.mean(y[mask])))
    return float(ece)


def _safe_rate(mask: np.ndarray) -> float:
    if mask.size == 0:
        return 0.0
    return float(np.mean(mask))


def _as_bool(values: pd.Series | np.ndarray | None, *, n_rows: int) -> np.ndarray:
    if values is None:
        return np.zeros(n_rows, dtype=bool)
    arr = pd.to_numeric(values, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    return arr > 0.5


def compute_policy_metrics(
    policy_df: pd.DataFrame,
    *,
    plays_col: str = "plays_target",
    bins: int = 10,
) -> dict[str, float]:
    if policy_df.empty:
        return {
            "rows": 0.0,
            "plays_rate": 0.0,
            "brier_raw": float("nan"),
            "brier_eff": float("nan"),
            "brier_delta_eff_minus_raw": float("nan"),
            "logloss_raw": float("nan"),
            "logloss_eff": float("nan"),
            "logloss_delta_eff_minus_raw": float("nan"),
            "ece_raw": float("nan"),
            "ece_eff": float("nan"),
            "ece_delta_eff_minus_raw": float("nan"),
            "false_active_p90_raw": 0.0,
            "false_active_p90_eff": 0.0,
            "fringe_false_active_p90_raw": 0.0,
            "fringe_false_active_p90_eff": 0.0,
            "starter_under95_raw": 0.0,
            "starter_under95_eff": 0.0,
            "starter_mean_p_raw": float("nan"),
            "starter_mean_p_eff": float("nan"),
            "changed_rate": 0.0,
        }

    if plays_col not in policy_df.columns:
        raise KeyError(f"Missing required column: {plays_col}")

    y = pd.to_numeric(policy_df[plays_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    y = np.clip(y, 0.0, 1.0)

    if "play_prob_raw" in policy_df.columns:
        p_raw = pd.to_numeric(policy_df["play_prob_raw"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    elif "play_prob" in policy_df.columns:
        p_raw = pd.to_numeric(policy_df["play_prob"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    else:
        p_raw = np.ones(len(policy_df), dtype=float)
    p_raw = np.clip(p_raw, 0.0, 1.0)

    if "play_prob_eff" in policy_df.columns:
        p_eff = pd.to_numeric(policy_df["play_prob_eff"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    else:
        p_eff = p_raw.copy()
    p_eff = np.clip(p_eff, 0.0, 1.0)

    starter = _as_bool(
        policy_df["is_starter"] if "is_starter" in policy_df.columns else policy_df.get("starter_flag"),
        n_rows=len(policy_df),
    )
    non_starter = ~starter
    not_played = y < 0.5
    played = ~not_played

    starter_played = starter & played
    if np.any(starter):
        starter_mean_p_raw = float(np.mean(p_raw[starter]))
        starter_mean_p_eff = float(np.mean(p_eff[starter]))
    else:
        starter_mean_p_raw = float("nan")
        starter_mean_p_eff = float("nan")

    changed = np.abs(p_eff - p_raw) > 1e-12

    metrics: dict[str, float] = {
        "rows": float(len(policy_df)),
        "plays_rate": float(np.mean(y)),
        "brier_raw": float(np.mean((y - p_raw) ** 2)),
        "brier_eff": float(np.mean((y - p_eff) ** 2)),
        "logloss_raw": binary_log_loss(y, p_raw),
        "logloss_eff": binary_log_loss(y, p_eff),
        "ece_raw": expected_calibration_error(y, p_raw, bins=bins),
        "ece_eff": expected_calibration_error(y, p_eff, bins=bins),
        "false_active_p90_raw": _safe_rate((p_raw >= 0.90) & not_played),
        "false_active_p90_eff": _safe_rate((p_eff >= 0.90) & not_played),
        "fringe_false_active_p90_raw": _safe_rate((p_raw >= 0.90) & not_played & non_starter),
        "fringe_false_active_p90_eff": _safe_rate((p_eff >= 0.90) & not_played & non_starter),
        "starter_under95_raw": _safe_rate((p_raw < 0.95) & starter_played),
        "starter_under95_eff": _safe_rate((p_eff < 0.95) & starter_played),
        "starter_mean_p_raw": starter_mean_p_raw,
        "starter_mean_p_eff": starter_mean_p_eff,
        "changed_rate": _safe_rate(changed),
    }
    metrics["brier_delta_eff_minus_raw"] = float(metrics["brier_eff"] - metrics["brier_raw"])
    metrics["logloss_delta_eff_minus_raw"] = float(metrics["logloss_eff"] - metrics["logloss_raw"])
    metrics["ece_delta_eff_minus_raw"] = float(metrics["ece_eff"] - metrics["ece_raw"])
    return metrics


def compute_policy_objective(
    metrics: Mapping[str, float],
    *,
    weights: PolicyObjectiveWeights | None = None,
) -> float:
    cfg = weights or PolicyObjectiveWeights()
    return float(
        cfg.brier * float(metrics.get("brier_eff", 0.0))
        + cfg.false_active_p90 * float(metrics.get("false_active_p90_eff", 0.0))
        + cfg.fringe_false_active_p90 * float(metrics.get("fringe_false_active_p90_eff", 0.0))
        + cfg.starter_under95 * float(metrics.get("starter_under95_eff", 0.0))
    )


def evaluate_policy_candidate(
    eval_df: pd.DataFrame,
    policy_config: Any,
    *,
    bins: int = 10,
    weights: PolicyObjectiveWeights | None = None,
) -> dict[str, Any]:
    policy_df, diag = apply_play_prob_policy_with_diagnostics(eval_df, policy_config)
    metrics = compute_policy_metrics(policy_df, bins=int(bins))
    metrics["objective"] = compute_policy_objective(metrics, weights=weights)
    metrics["n_rotation_locks"] = float(diag.n_rotation_locks)
    metrics["n_changed"] = float(diag.n_changed)
    metrics["max_delta"] = float(diag.max_delta)
    metrics["reason_counts_json"] = json.dumps(diag.reasons, sort_keys=True)
    return metrics


def build_policy_grid_overrides(
    base_config: Any,
    *,
    grids: Mapping[str, Sequence[Any] | None],
    include_baseline: bool = True,
) -> list[dict[str, Any]]:
    """Build deterministic override dictionaries from a per-knob grid mapping."""

    keys = sorted(grids.keys())
    value_lists: list[list[Any]] = []
    for key in keys:
        raw_values = list(grids.get(key) or [])
        if not raw_values:
            raw_values = [getattr(base_config, key)]
        normalized: list[Any] = []
        for value in raw_values:
            if isinstance(value, list):
                normalized.append(tuple(value))
            else:
                normalized.append(value)
        deduped = list(dict.fromkeys(normalized))
        value_lists.append(deduped)

    candidates: list[dict[str, Any]] = []
    if include_baseline:
        baseline = {k: getattr(base_config, k) for k in keys}
        candidates.append(baseline)

    for combo in itertools.product(*value_lists):
        cand = {k: v for k, v in zip(keys, combo, strict=True)}
        candidates.append(cand)

    # Dedupe in stable order.
    out: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for cand in candidates:
        sig = tuple((k, cand[k]) for k in keys)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(cand)
    return out


__all__ = [
    "PolicyObjectiveWeights",
    "binary_log_loss",
    "build_policy_grid_overrides",
    "compute_policy_metrics",
    "compute_policy_objective",
    "evaluate_policy_candidate",
    "expected_calibration_error",
]
