"""Production scoring for RotAlloc (rotation+minutes+allocation) minutes.

This loader is designed to be used by the live minutes scoring pipeline as an
optional allocation mode that replaces per-player minutes with a team-sum-to-240
allocator restricted to an eligible set.

Kill switch:
  - Set `PROJECTIONS_MINUTES_ALLOC_MODE=legacy` to force legacy behavior.
  - Set `PROJECTIONS_MINUTES_ALLOC_MODE=rotalloc_expk` to force RotAlloc.
  
Debug / fail-hard:
  - Set `PROJECTIONS_ROTALLOC_FAIL_HARD=1` to raise on RotAlloc errors instead of fallback.

Configuration:
  - Production knobs are versioned in config/rotalloc_production.json (NOT experiment artifacts).
  - Env overrides: ROTALLOC_P_CUTOFF, ROTALLOC_K_MIN, ROTALLOC_K_MAX, ROTALLOC_CAP_MAX
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from projections.models.rotalloc import (
    allocate_adaptive_depth,
    allocate_team_minutes,
    allocate_two_tier,
    apply_regulation_cap,
    build_eligible_mask,
    compute_bench_share_prior,
    compute_sixth_man_crush_metric,
)


ENV_MINUTES_ALLOC_MODE = "PROJECTIONS_MINUTES_ALLOC_MODE"
ENV_ROTALLOC_FAIL_HARD = "PROJECTIONS_ROTALLOC_FAIL_HARD"

# Explicit env var overrides for allocator params (opt-in, logged, must be intentional)
ENV_ROTALLOC_P_CUTOFF = "ROTALLOC_P_CUTOFF"
ENV_ROTALLOC_K_MIN = "ROTALLOC_K_MIN"
ENV_ROTALLOC_K_MAX = "ROTALLOC_K_MAX"
ENV_ROTALLOC_CAP_MAX = "ROTALLOC_CAP_MAX"
ENV_ROTALLOC_BENCH_SHARE = "ROTALLOC_BENCH_SHARE"
ENV_ROTALLOC_ADAPTIVE_DEPTH = "ROTALLOC_ADAPTIVE_DEPTH"

# Versioned production config path (relative to repo root)
VERSIONED_PROD_CONFIG = Path(__file__).parent.parent.parent / "config" / "rotalloc_production.json"

# Guardrail thresholds (fallback triggers if exceeded and not FAIL_HARD)
GUARDRAIL_FRAC_P90_AT_CAP_MAX = 0.15
GUARDRAIL_MAX_P50_MAX = 41.0
GUARDRAIL_P95_MINUTES_P50_MAX = 40.0
GUARDRAIL_ELIGIBLE_SIZE_P50_MIN = 9



def _normalize_alloc_mode(raw: str | None) -> str:
    if not raw:
        return "legacy"
    value = str(raw).strip().lower()
    if value in {"legacy", "lgbm", "minutes_v1"}:
        return "legacy"
    if value in {"rotalloc", "rotalloc_expk", "rotalloc-expk", "rotalloc_expected_k"}:
        return "rotalloc_expk"
    # Allocator C: share model scaled within RotAlloc eligibility
    if value in {"share_with_rotalloc_elig", "share-with-rotalloc-elig", "share_rotalloc_elig", "allocator_c"}:
        return "share_with_rotalloc_elig"
    # Allocator E: core/fringe blend of share + rotalloc proxy weights
    if value in {
        "rotalloc_fringe_alpha",
        "rotalloc-fringe-alpha",
        "fringe_only_alpha",
        "fringe-only-alpha",
        "allocator_e",
    }:
        return "rotalloc_fringe_alpha"
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
    # Post-allocation regulation cap (applied to p50 only, with core redistribution)
    cap_p50: float = 48.0
    k_core: int = 8
    # Two-tier allocation (core/fringe split to prevent scrub inflation)
    enable_two_tier: bool = False
    core_minutes_floor: float = 200.0
    fringe_cap_max: float = 15.0
    # Adaptive depth allocation (replaces fixed k_core with cumulative proxy mass)
    enable_adaptive_depth: bool = False
    bench_share_default: float = 0.15
    bench_share_min: float = 0.05
    bench_share_max: float = 0.35
    core_k_min: int | None = None
    core_k_max: int | None = None
    # Depth prior regression coefficients
    spread_coef: float = -0.002
    total_coef: float = 0.0003
    out_coef: float = 0.015
    team_prior_weight: float = 0.6
    # Track env var overrides that were applied (for summary.json diagnostics)
    overrides_applied: dict | None = None


@dataclass(frozen=True)
class RotAllocDiagnostics:
    cutoff_empty_events: int
    fallback_top1_used: int
    eligible_size_p50: float
    eligible_size_p90: float
    team_sum_dev_max: float
    minutes_below_cutoff_p90: float
    # Adaptive depth diagnostics
    bench_share_pred_mean: float = float("nan")
    bench_share_pred_p50: float = float("nan")
    bench_share_pred_p90: float = float("nan")
    bench_share_actual_mean: float = float("nan")
    bench_share_actual_p50: float = float("nan")
    bench_share_actual_p90: float = float("nan")
    sixth_man_crush_mean: float = float("nan")
    sixth_man_crush_p10: float = float("nan")
    p95_minutes_p50: float = float("nan")
    p99_minutes_p50: float = float("nan")
    frac_p50_ge_42: float = float("nan")
    frac_p50_ge_44: float = float("nan")
    frac_p90_at_cap: float = float("nan")
    core_k_mean: float = float("nan")
    core_k_p50: float = float("nan")


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
    """Score RotAlloc minutes for a feature slice (one slate).
    
    Config loading priority:
      1. Versioned production config (config/rotalloc_production.json)
      2. Bundle's promote_config.json (experiment artifacts)
      3. Safe defaults (p_cutoff=0.2, k_min=8, k_max=11)
      
    Allocation modes (priority: adaptive_depth > two_tier > legacy):
      - adaptive_depth: Uses cumulative proxy mass for core/fringe split
      - two_tier: Fixed k_core split with core_minutes_floor
      - legacy: allocate_team_minutes + regulation cap
    """
    # Load versioned production config (source of truth for prod knobs)
    allocator_payload: dict = {}
    versioned_config: dict = {}
    if VERSIONED_PROD_CONFIG.exists():
        try:
            versioned_config = json.loads(VERSIONED_PROD_CONFIG.read_text(encoding="utf-8"))
            allocator_payload = versioned_config.get("allocator", {}) if isinstance(versioned_config, dict) else {}
            print(f"[rotalloc] loaded versioned config from {VERSIONED_PROD_CONFIG}")
        except (json.JSONDecodeError, OSError) as e:
            print(f"[rotalloc] warning: failed to load versioned config: {e}")
    
    # Fall back to bundle's promote_config.json for any missing values
    promote_path = bundle_dir / "promote_config.json"
    if promote_path.exists():
        try:
            bundle_config = json.loads(promote_path.read_text(encoding="utf-8"))
            bundle_allocator = bundle_config.get("allocator", {}) if isinstance(bundle_config, dict) else {}
            # Only backfill missing keys
            for key, value in bundle_allocator.items():
                if key not in allocator_payload:
                    allocator_payload[key] = value
        except (json.JSONDecodeError, OSError):
            pass
    
    # Safe defaults (match versioned config)
    p_cutoff_raw = allocator_payload.get("p_cutoff", 0.2)
    a_val = float(allocator_payload.get("a", 1.5))
    mu_power_val = float(allocator_payload.get("mu_power", 1.5))
    p_cutoff_val = float(p_cutoff_raw) if p_cutoff_raw is not None else 0.2
    use_expected_k_val = bool(allocator_payload.get("use_expected_k", True))
    k_min_val = int(allocator_payload.get("k_min", 8))
    k_max_val = int(allocator_payload.get("k_max", 11))
    cap_max_val = float(allocator_payload.get("cap_max", 48.0))
    
    # Check for explicit env var overrides (opt-in only)
    overrides_applied: dict = {}
    
    env_p_cutoff = os.environ.get(ENV_ROTALLOC_P_CUTOFF)
    if env_p_cutoff is not None:
        old_val = p_cutoff_val
        p_cutoff_val = float(env_p_cutoff)
        overrides_applied["p_cutoff"] = {"from": old_val, "to": p_cutoff_val, "source": ENV_ROTALLOC_P_CUTOFF}
        print(f"[rotalloc] OVERRIDE: p_cutoff {old_val} -> {p_cutoff_val} (from {ENV_ROTALLOC_P_CUTOFF})")
    
    env_k_min = os.environ.get(ENV_ROTALLOC_K_MIN)
    if env_k_min is not None:
        old_val = k_min_val
        k_min_val = int(env_k_min)
        overrides_applied["k_min"] = {"from": old_val, "to": k_min_val, "source": ENV_ROTALLOC_K_MIN}
        print(f"[rotalloc] OVERRIDE: k_min {old_val} -> {k_min_val} (from {ENV_ROTALLOC_K_MIN})")
    
    env_k_max = os.environ.get(ENV_ROTALLOC_K_MAX)
    if env_k_max is not None:
        old_val = k_max_val
        k_max_val = int(env_k_max)
        overrides_applied["k_max"] = {"from": old_val, "to": k_max_val, "source": ENV_ROTALLOC_K_MAX}
        print(f"[rotalloc] OVERRIDE: k_max {old_val} -> {k_max_val} (from {ENV_ROTALLOC_K_MAX})")
    
    env_cap_max = os.environ.get(ENV_ROTALLOC_CAP_MAX)
    if env_cap_max is not None:
        old_val = cap_max_val
        cap_max_val = float(env_cap_max)
        overrides_applied["cap_max"] = {"from": old_val, "to": cap_max_val, "source": ENV_ROTALLOC_CAP_MAX}
        print(f"[rotalloc] OVERRIDE: cap_max {old_val} -> {cap_max_val} (from {ENV_ROTALLOC_CAP_MAX})")
    
    # Load regulation cap settings (post-allocation cap with core redistribution)
    regulation_payload = versioned_config.get("regulation_cap", {}) if isinstance(versioned_config, dict) else {}
    two_tier_payload = versioned_config.get("two_tier", {}) if isinstance(versioned_config, dict) else {}
    adaptive_depth_payload = versioned_config.get("adaptive_depth", {}) if isinstance(versioned_config, dict) else {}
    
    cap_p50_val = float(regulation_payload.get("cap_p50", cap_max_val))
    k_core_val = int(regulation_payload.get("k_core", 8))
    
    # Two-tier allocation params (core/fringe split)
    enable_two_tier_val = bool(two_tier_payload.get("enable", False))
    if "k_core" in two_tier_payload:
        k_core_val = int(two_tier_payload["k_core"])
    core_minutes_floor_val = float(two_tier_payload.get("core_minutes_floor", 200.0))
    fringe_cap_max_val = float(two_tier_payload.get("fringe_cap_max", 15.0))
    
    # Adaptive depth allocation params (replaces fixed k_core with cumulative proxy mass)
    enable_adaptive_depth_val = bool(adaptive_depth_payload.get("enable", False))
    bench_share_default_val = float(adaptive_depth_payload.get("bench_share_default", 0.15))
    bench_share_min_val = float(adaptive_depth_payload.get("bench_share_min", 0.05))
    bench_share_max_val = float(adaptive_depth_payload.get("bench_share_max", 0.35))
    core_k_min_raw = adaptive_depth_payload.get("core_k_min")
    core_k_max_raw = adaptive_depth_payload.get("core_k_max")
    core_k_min_val = int(core_k_min_raw) if core_k_min_raw is not None else None
    core_k_max_val = int(core_k_max_raw) if core_k_max_raw is not None else None
    spread_coef_val = float(adaptive_depth_payload.get("spread_coef", -0.002))
    total_coef_val = float(adaptive_depth_payload.get("total_coef", 0.0003))
    out_coef_val = float(adaptive_depth_payload.get("out_coef", 0.015))
    team_prior_weight_val = float(adaptive_depth_payload.get("team_prior_weight", 0.6))
    
    # Check env override for adaptive depth
    env_adaptive = os.environ.get(ENV_ROTALLOC_ADAPTIVE_DEPTH)
    if env_adaptive is not None:
        old_val = enable_adaptive_depth_val
        enable_adaptive_depth_val = env_adaptive.lower() in ("1", "true", "yes")
        overrides_applied["enable_adaptive_depth"] = {"from": old_val, "to": enable_adaptive_depth_val, "source": ENV_ROTALLOC_ADAPTIVE_DEPTH}
        print(f"[rotalloc] OVERRIDE: enable_adaptive_depth {old_val} -> {enable_adaptive_depth_val}")
    
    env_bench_share = os.environ.get(ENV_ROTALLOC_BENCH_SHARE)
    if env_bench_share is not None:
        old_val = bench_share_default_val
        bench_share_default_val = float(env_bench_share)
        overrides_applied["bench_share_default"] = {"from": old_val, "to": bench_share_default_val, "source": ENV_ROTALLOC_BENCH_SHARE}
        print(f"[rotalloc] OVERRIDE: bench_share_default {old_val} -> {bench_share_default_val}")
    
    # Use fringe_cap_max from two_tier if adaptive_depth doesn't specify
    if "fringe_cap_max" not in adaptive_depth_payload:
        adaptive_fringe_cap = fringe_cap_max_val
    else:
        adaptive_fringe_cap = float(adaptive_depth_payload["fringe_cap_max"])
    
    allocator = RotAllocAllocatorConfig(
        a=a_val,
        mu_power=mu_power_val,
        p_cutoff=p_cutoff_val,
        use_expected_k=use_expected_k_val,
        k_min=k_min_val,
        k_max=k_max_val,
        cap_max=cap_max_val,
        cap_p50=cap_p50_val,
        k_core=k_core_val,
        enable_two_tier=enable_two_tier_val,
        core_minutes_floor=core_minutes_floor_val,
        fringe_cap_max=fringe_cap_max_val,
        enable_adaptive_depth=enable_adaptive_depth_val,
        bench_share_default=bench_share_default_val,
        bench_share_min=bench_share_min_val,
        bench_share_max=bench_share_max_val,
        core_k_min=core_k_min_val,
        core_k_max=core_k_max_val,
        spread_coef=spread_coef_val,
        total_coef=total_coef_val,
        out_coef=out_coef_val,
        team_prior_weight=team_prior_weight_val,
        overrides_applied=overrides_applied if overrides_applied else None,
    )

    models_dir = bundle_dir / "models"
    if not models_dir.exists():
        # Check if bundle_dir itself contains models
        if (bundle_dir / "rot8_classifier.joblib").exists():
            models_dir = bundle_dir
        else:
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

    # Mask candidates: exclude inactive rows
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
    bench_share_pred_arr = np.zeros(len(df), dtype=np.float64)

    cutoff_empty_events = 0
    fallback_top1_used = 0
    eligible_sizes: list[int] = []
    below_cutoff_team_minutes: list[float] = []
    max_dev = 0.0
    
    # Adaptive depth diagnostics
    bench_share_preds: list[float] = []
    bench_share_actuals: list[float] = []
    sixth_man_crush_values: list[float] = []
    core_k_values: list[int] = []
    all_minutes_p50: list[float] = []

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

        if allocator.enable_adaptive_depth:
            # Adaptive depth allocation: uses cumulative proxy mass for core/fringe split
            # Extract context features for depth prior
            spread_val = None
            total_val = None
            out_count = 0
            team_bench_share_avg = None  # TODO: Load from rolling team averages
            
            if "spread_home" in g.columns:
                spread_home = pd.to_numeric(g["spread_home"], errors="coerce").iloc[0]
                is_home = g["home_flag"].iloc[0] if "home_flag" in g.columns else False
                if pd.notna(spread_home):
                    spread_val = float(spread_home) if is_home else -float(spread_home)
            if "total" in g.columns:
                total_raw = pd.to_numeric(g["total"], errors="coerce").iloc[0]
                if pd.notna(total_raw):
                    total_val = float(total_raw)
            
            # Count OUT players (from mask)
            out_count = int((~mask_g).sum())
            
            # Compute bench share prior
            bench_share = compute_bench_share_prior(
                team_bench_share_avg=team_bench_share_avg,
                spread=spread_val,
                total=total_val,
                out_count=out_count,
                league_bench_share=allocator.bench_share_default,
                spread_coef=allocator.spread_coef,
                total_coef=allocator.total_coef,
                out_coef=allocator.out_coef,
                team_prior_weight=allocator.team_prior_weight,
            )
            bench_share_preds.append(bench_share)
            bench_share_pred_arr[idx] = bench_share
            
            m, depth_diag = allocate_adaptive_depth(
                p_g,
                mu_g,
                eligible,
                a=allocator.a,
                mu_power=allocator.mu_power,
                bench_share_pred=bench_share,
                bench_share_min=allocator.bench_share_min,
                bench_share_max=allocator.bench_share_max,
                core_k_min=allocator.core_k_min,
                core_k_max=allocator.core_k_max,
                fringe_cap_max=adaptive_fringe_cap,
                cap_max=allocator.cap_max,
            )
            
            bench_share_actuals.append(depth_diag.get("bench_share_actual", 0.0))
            core_k_values.append(depth_diag.get("core_k", 0))
            
            # Compute 6th-man crush metric
            proxy_g = np.power(p_g, allocator.a) * np.power(mu_g, allocator.mu_power)
            proxy_g = np.where(np.isfinite(proxy_g), proxy_g, 0.0)
            crush = compute_sixth_man_crush_metric(m, mu_g, proxy_g, eligible)
            if np.isfinite(crush):
                sixth_man_crush_values.append(crush)
                
        elif allocator.enable_two_tier:
            # Two-tier allocation: core/fringe split with fixed k_core
            m, _two_tier_diag = allocate_two_tier(
                p_g,
                mu_g,
                eligible,
                a=allocator.a,
                mu_power=allocator.mu_power,
                k_core=allocator.k_core,
                core_minutes_floor=allocator.core_minutes_floor,
                fringe_cap_max=allocator.fringe_cap_max,
                cap_max=allocator.cap_max,
            )
            core_k_values.append(_two_tier_diag.get("core_k", allocator.k_core))
        else:
            # Legacy path: allocate_team_minutes + regulation cap
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
            
            # Apply post-allocation regulation cap with core rotation redistribution
            weights_g = np.power(p_g, allocator.a) * np.power(mu_g, allocator.mu_power)
            weights_g = np.where(np.isfinite(weights_g), weights_g, 0.0)
            m, _reg_diag = apply_regulation_cap(
                m,
                weights_g,
                eligible,
                cap_p50=allocator.cap_p50,
                k_core=allocator.k_core,
            )
        
        minutes[idx] = m
        all_minutes_p50.extend(m[eligible].tolist())
        total = float(m.sum())
        team_sum[idx] = total
        dev = abs(total - 240.0)
        if dev > max_dev:
            max_dev = dev
        if allocator.p_cutoff is not None:
            below_cutoff_team_minutes.append(float(m[p_g < float(allocator.p_cutoff)].sum()))

    # Build output frame aligned to input rows
    out = df.loc[:, ["game_id", "team_id", "player_id"]].copy()
    out["minutes_mean"] = minutes
    out["p_rot"] = p_rot
    out["mu_cond"] = mu
    out["eligible_flag"] = eligible_flags.astype(int)
    out["team_minutes_sum"] = team_sum
    out["minutes_alloc_mode"] = "rotalloc_adaptive" if allocator.enable_adaptive_depth else "rotalloc_expk"
    if allocator.enable_adaptive_depth:
        out["bench_share_pred"] = bench_share_pred_arr

    eligible_series = pd.Series(eligible_sizes, dtype=float)
    eligible_p50 = float(eligible_series.quantile(0.5)) if not eligible_series.empty else float("nan")
    eligible_p90 = float(eligible_series.quantile(0.9)) if not eligible_series.empty else float("nan")
    below_cutoff_p90 = float("nan")
    if allocator.p_cutoff is not None and below_cutoff_team_minutes:
        below_cutoff_p90 = float(pd.Series(below_cutoff_team_minutes, dtype=float).quantile(0.9))

    # Compute extended diagnostics
    minutes_series = pd.Series(all_minutes_p50, dtype=float)
    p95_minutes = float(minutes_series.quantile(0.95)) if not minutes_series.empty else float("nan")
    p99_minutes = float(minutes_series.quantile(0.99)) if not minutes_series.empty else float("nan")
    frac_ge_42 = float((minutes_series >= 42.0).mean()) if not minutes_series.empty else float("nan")
    frac_ge_44 = float((minutes_series >= 44.0).mean()) if not minutes_series.empty else float("nan")
    frac_at_cap = float((minutes_series >= allocator.cap_max - 0.1).mean()) if not minutes_series.empty else float("nan")
    
    # Adaptive depth diagnostics
    bench_pred_series = pd.Series(bench_share_preds, dtype=float)
    bench_actual_series = pd.Series(bench_share_actuals, dtype=float)
    crush_series = pd.Series(sixth_man_crush_values, dtype=float)
    core_k_series = pd.Series(core_k_values, dtype=float)
    
    diag = RotAllocDiagnostics(
        cutoff_empty_events=int(cutoff_empty_events),
        fallback_top1_used=int(fallback_top1_used),
        eligible_size_p50=eligible_p50,
        eligible_size_p90=eligible_p90,
        team_sum_dev_max=float(max_dev),
        minutes_below_cutoff_p90=float(below_cutoff_p90),
        bench_share_pred_mean=float(bench_pred_series.mean()) if not bench_pred_series.empty else float("nan"),
        bench_share_pred_p50=float(bench_pred_series.quantile(0.5)) if not bench_pred_series.empty else float("nan"),
        bench_share_pred_p90=float(bench_pred_series.quantile(0.9)) if not bench_pred_series.empty else float("nan"),
        bench_share_actual_mean=float(bench_actual_series.mean()) if not bench_actual_series.empty else float("nan"),
        bench_share_actual_p50=float(bench_actual_series.quantile(0.5)) if not bench_actual_series.empty else float("nan"),
        bench_share_actual_p90=float(bench_actual_series.quantile(0.9)) if not bench_actual_series.empty else float("nan"),
        sixth_man_crush_mean=float(crush_series.mean()) if not crush_series.empty else float("nan"),
        sixth_man_crush_p10=float(crush_series.quantile(0.1)) if not crush_series.empty else float("nan"),
        p95_minutes_p50=p95_minutes,
        p99_minutes_p50=p99_minutes,
        frac_p50_ge_42=frac_ge_42,
        frac_p50_ge_44=frac_ge_44,
        frac_p90_at_cap=frac_at_cap,
        core_k_mean=float(core_k_series.mean()) if not core_k_series.empty else float("nan"),
        core_k_p50=float(core_k_series.quantile(0.5)) if not core_k_series.empty else float("nan"),
    )
    if diag.team_sum_dev_max > 1e-6:
        raise ValueError(f"RotAlloc sum-to-240 violation (max_dev={diag.team_sum_dev_max:.6f})")
    return out, allocator, diag
