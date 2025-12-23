"""Production scoring for RotAlloc (rotation+minutes+allocation) minutes.

This loader is designed to be used by the live minutes scoring pipeline as an
optional allocation mode that replaces per-player minutes with a team-sum-to-240
allocator restricted to an eligible set.

Kill switch:
  - Set `PROJECTIONS_MINUTES_ALLOC_MODE=legacy` to force legacy behavior.
  - Set `PROJECTIONS_MINUTES_ALLOC_MODE=rotalloc_expk` to force RotAlloc.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from projections.models.rotalloc import allocate_team_minutes, build_eligible_mask


ENV_MINUTES_ALLOC_MODE = "PROJECTIONS_MINUTES_ALLOC_MODE"


def _normalize_alloc_mode(raw: str | None) -> str:
    if not raw:
        return "legacy"
    value = str(raw).strip().lower()
    if value in {"legacy", "lgbm", "minutes_v1"}:
        return "legacy"
    if value in {"rotalloc", "rotalloc_expk", "rotalloc-expk", "rotalloc_expected_k"}:
        return "rotalloc_expk"
    return value


def resolve_minutes_alloc_mode(config_path: Path | None) -> str:
    """Resolve allocation mode from env override then config (defaults to legacy)."""
    env = os.environ.get(ENV_MINUTES_ALLOC_MODE)
    if env:
        return _normalize_alloc_mode(env)

    if config_path is None:
        return "legacy"
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return "legacy"
    except json.JSONDecodeError:
        return "legacy"

    return _normalize_alloc_mode(payload.get("minutes_alloc_mode") or payload.get("minutes_alloc") or "legacy")


def resolve_rotalloc_bundle_dir(config_path: Path | None) -> Path | None:
    if config_path is None:
        return None
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None

    raw = payload.get("rotalloc_bundle_dir") or payload.get("rotalloc_dir")
    if not raw:
        return None
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def ensure_infer_feature_columns(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    indicator_suffix: str = "_is_nan",
) -> pd.DataFrame:
    df = df.copy()
    for col in feature_cols:
        if col.endswith(indicator_suffix):
            base = col[: -len(indicator_suffix)]
            if base not in df.columns:
                df[base] = 0.0
        elif col not in df.columns:
            df[col] = 0.0
    for col in feature_cols:
        if not col.endswith(indicator_suffix):
            continue
        base = col[: -len(indicator_suffix)]
        if base in df.columns:
            df[col] = df[base].isna().astype(np.float32)
        else:
            df[col] = 1.0
    return df


def fill_missing_values(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in feature_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


def _load_feature_columns(models_dir: Path) -> list[str]:
    path = models_dir / "feature_columns.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        cols = payload
    elif isinstance(payload, dict):
        cols = payload.get("columns", [])
    else:
        cols = []
    cols = [str(c) for c in cols]
    if not cols:
        raise ValueError(f"Empty feature columns in {path}")
    return cols


@dataclass(frozen=True)
class RotAllocAllocatorConfig:
    a: float
    mu_power: float
    p_cutoff: float | None
    use_expected_k: bool
    k_min: int
    k_max: int
    cap_max: float


@dataclass(frozen=True)
class RotAllocDiagnostics:
    cutoff_empty_events: int
    fallback_top1_used: int
    eligible_size_p50: float
    eligible_size_p90: float
    team_sum_dev_max: float
    minutes_below_cutoff_p90: float


def _safe_status_upper(series: pd.Series) -> pd.Series:
    return series.astype(str).str.upper().fillna("")


def _eligibility_pre_fallback(
    p_rot: np.ndarray,
    mu: np.ndarray,
    mask: np.ndarray,
    *,
    a: float,
    mu_power: float,
    p_cutoff: float | None,
    use_expected_k: bool,
    k_min: int,
    k_max: int,
) -> np.ndarray:
    p = np.clip(np.asarray(p_rot, dtype=np.float64), 0.0, 1.0)
    m = np.maximum(np.asarray(mu, dtype=np.float64), 0.0)
    mask_bool = np.asarray(mask, dtype=bool)
    proxy = np.power(p, float(a)) * np.power(m, float(mu_power))
    proxy = np.where(np.isfinite(proxy), proxy, 0.0)

    eligible = mask_bool.copy()
    if p_cutoff is not None:
        eligible &= p >= float(p_cutoff)

    if use_expected_k:
        expected_k = int(np.round(p[eligible].sum()))
        k = max(int(k_min), min(int(k_max), expected_k))
        ranked_proxy = np.where(eligible, proxy, -np.inf)
        order = np.argsort(-ranked_proxy, kind="mergesort")
        k_eff = min(int(k), int(eligible.sum()))
        top_idx = order[:k_eff]
        eligible = np.zeros_like(mask_bool, dtype=bool)
        eligible[top_idx] = True
        eligible &= mask_bool

    return eligible


def score_rotalloc_minutes(
    features: pd.DataFrame,
    *,
    bundle_dir: Path,
    group_cols: tuple[str, str] = ("game_id", "team_id"),
) -> tuple[pd.DataFrame, RotAllocAllocatorConfig, RotAllocDiagnostics]:
    """Score RotAlloc minutes for a feature slice (one slate)."""
    promote_path = bundle_dir / "promote_config.json"
    payload = json.loads(promote_path.read_text(encoding="utf-8"))
    allocator_payload = payload.get("allocator", {}) if isinstance(payload, dict) else {}
    p_cutoff_raw = allocator_payload.get("p_cutoff")
    if p_cutoff_raw is None:
        # Safe default: require some rotation probability to be eligible. This matches
        # the intent of the rotation model and avoids allocating minutes to everyone.
        p_cutoff_raw = 0.15
        message = "[rotalloc] warning: allocator.p_cutoff missing in promote_config.json; defaulting to 0.15"
        print(message)
        if os.environ.get("CI"):
            raise ValueError(
                "rotalloc requires allocator.p_cutoff in promote_config.json (CI strict). "
                "Set allocator.p_cutoff explicitly."
            )
    allocator = RotAllocAllocatorConfig(
        a=float(allocator_payload.get("a", 1.5)),
        mu_power=float(allocator_payload.get("mu_power", 1.5)),
        p_cutoff=(None if p_cutoff_raw is None else float(p_cutoff_raw)),
        use_expected_k=bool(allocator_payload.get("use_expected_k", True)),
        k_min=int(allocator_payload.get("k_min", 6)),
        k_max=int(allocator_payload.get("k_max", 11)),
        cap_max=float(allocator_payload.get("cap_max", 48.0)),
    )

    models_dir = bundle_dir / "models"
    if not models_dir.exists() and isinstance(payload, dict):
        rotalloc_dir = payload.get("rotalloc_dir")
        if rotalloc_dir:
            candidate = Path(str(rotalloc_dir)).expanduser().resolve() / "models"
            if candidate.exists():
                models_dir = candidate
    if not models_dir.exists():
        raise FileNotFoundError(f"RotAlloc models directory not found under {bundle_dir}")

    feature_cols = _load_feature_columns(models_dir)
    clf = joblib.load(models_dir / "rot8_classifier.joblib")
    reg = joblib.load(models_dir / "minutes_regressor.joblib")
    calibrator = None
    for name in ("rot8_calibrator_sigmoid.joblib", "rot8_calibrator_isotonic.joblib"):
        calibrator_path = models_dir / name
        if calibrator_path.exists():
            calibrator = joblib.load(calibrator_path)
            break

    df = features.copy()
    df = ensure_infer_feature_columns(df, feature_cols)
    df = fill_missing_values(df, feature_cols)

    X = df[feature_cols]
    p_raw = clf.predict_proba(X)[:, 1]
    p_rot = calibrator.transform(p_raw) if calibrator is not None else p_raw
    p_rot = np.clip(np.asarray(p_rot, dtype=np.float64), 0.0, 1.0)
    mu = np.asarray(reg.predict(X), dtype=np.float64)
    mu = np.maximum(mu, 0.0)
    # Guardrail: classifier output should be a smooth probability distribution.
    # If it collapses to a handful of discrete values, RotAlloc eligibility won't
    # separate rotation from bench players.
    unique_probs = np.unique(np.round(p_rot, 3))
    if unique_probs.size <= 5:
        message = (
            "[rotalloc] warning: rotation classifier outputs are highly discrete "
            f"(unique_probs={unique_probs.size}, values={unique_probs.tolist()}). "
            "Re-train the rotation classifier or update calibration."
        )
        print(message)
        if os.environ.get("CI"):
            raise ValueError(
                "RotAlloc rotation classifier outputs are too discrete; "
                "retrain required (CI strict)."
            )

    # Mask candidates: padded rows are not expected here; exclude inactive rows if present.
    mask = np.ones(len(df), dtype=bool)
    if "status" in df.columns:
        status_upper = _safe_status_upper(df["status"])
        mask &= status_upper.to_numpy() != "OUT"
    if "play_prob" in df.columns:
        play_prob = pd.to_numeric(df["play_prob"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
        mask &= play_prob > 0.0

    minutes = np.zeros(len(df), dtype=np.float64)
    eligible_flags = np.zeros(len(df), dtype=bool)
    team_sum = np.zeros(len(df), dtype=np.float64)

    cutoff_empty_events = 0
    fallback_top1_used = 0
    eligible_sizes: list[int] = []
    below_cutoff_team_minutes: list[float] = []
    max_dev = 0.0

    for _, g in df.groupby(list(group_cols), sort=False):
        idx = g.index.to_numpy()
        if idx.size == 0:
            continue
        mask_g = mask[idx]
        if not mask_g.any():
            game_id = g["game_id"].iloc[0] if "game_id" in g.columns else "?"
            team_id = g["team_id"].iloc[0] if "team_id" in g.columns else "?"
            raise ValueError(f"RotAlloc mask empty for team-game: game_id={game_id} team_id={team_id}")
        p_g = p_rot[idx]
        mu_g = mu[idx]

        eligible_pre = _eligibility_pre_fallback(
            p_g,
            mu_g,
            mask_g,
            a=allocator.a,
            mu_power=allocator.mu_power,
            p_cutoff=allocator.p_cutoff,
            use_expected_k=allocator.use_expected_k,
            k_min=allocator.k_min,
            k_max=allocator.k_max,
        )
        if allocator.p_cutoff is not None and not (mask_g & (p_g >= float(allocator.p_cutoff))).any():
            cutoff_empty_events += 1
        if not eligible_pre.any():
            fallback_top1_used += 1

        eligible = build_eligible_mask(
            p_g,
            mu_g,
            mask_g,
            a=allocator.a,
            mu_power=allocator.mu_power,
            p_cutoff=allocator.p_cutoff,
            use_expected_k=allocator.use_expected_k,
            k_min=allocator.k_min,
            k_max=allocator.k_max,
        )
        eligible_flags[idx] = eligible
        eligible_sizes.append(int(eligible.sum()))

        m = allocate_team_minutes(
            p_g,
            mu_g,
            mask_g,
            a=allocator.a,
            mu_power=allocator.mu_power,
            cap_max=allocator.cap_max,
            p_cutoff=allocator.p_cutoff,
            use_expected_k=allocator.use_expected_k,
            k_min=allocator.k_min,
            k_max=allocator.k_max,
        )
        minutes[idx] = m
        total = float(m.sum())
        team_sum[idx] = total
        dev = abs(total - 240.0)
        if dev > max_dev:
            max_dev = dev
        if allocator.p_cutoff is not None:
            below_cutoff_team_minutes.append(float(m[p_g < float(allocator.p_cutoff)].sum()))

    # Build output frame aligned to input rows.
    out = df.loc[:, ["game_id", "team_id", "player_id"]].copy()
    out["minutes_mean"] = minutes
    out["p_rot"] = p_rot
    out["mu_cond"] = mu
    out["eligible_flag"] = eligible_flags.astype(int)
    out["team_minutes_sum"] = team_sum
    out["minutes_alloc_mode"] = "rotalloc_expk"

    eligible_series = pd.Series(eligible_sizes, dtype=float)
    eligible_p50 = float(eligible_series.quantile(0.5)) if not eligible_series.empty else float("nan")
    eligible_p90 = float(eligible_series.quantile(0.9)) if not eligible_series.empty else float("nan")
    below_cutoff_p90 = float("nan")
    if allocator.p_cutoff is not None and below_cutoff_team_minutes:
        below_cutoff_p90 = float(pd.Series(below_cutoff_team_minutes, dtype=float).quantile(0.9))

    diag = RotAllocDiagnostics(
        cutoff_empty_events=int(cutoff_empty_events),
        fallback_top1_used=int(fallback_top1_used),
        eligible_size_p50=eligible_p50,
        eligible_size_p90=eligible_p90,
        team_sum_dev_max=float(max_dev),
        minutes_below_cutoff_p90=float(below_cutoff_p90),
    )
    if diag.team_sum_dev_max > 1e-6:
        raise ValueError(f"RotAlloc sum-to-240 violation (max_dev={diag.team_sum_dev_max:.6f})")
    return out, allocator, diag
