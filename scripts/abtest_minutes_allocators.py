#!/usr/bin/env python
"""A/B/C/D test harness comparing minutes allocation methods.

Allocator A: SCALE_SHARES
    - Uses predicted minute_share and scales to 240 per team
    - Simple: minutes_i = 240 * share_i / sum(share_eligible)

Allocator B: ROTALLOC
    - Uses production RotAlloc allocator with rotation classifier + conditional means
    - Complex: eligible set pruning, waterfill redistribution, adaptive depth

Allocator C: SHARE_WITH_ROTALLOC_ELIGIBILITY
    - Uses share model predictions scaled WITHIN RotAlloc's eligible set
    - Hybrid: shares gated by RotAlloc eligibility, then scaled to 240

Allocator D: BLEND_WITHIN_ELIGIBLE
    - Blends share weights with RotAlloc proxy weights within eligible set
    - w = alpha * w_share + (1-alpha) * w_rot
    - minutes = 240 * w / sum(w)
    - Combines realism from share model with bench ranking from RotAlloc

Usage (single slate):
    uv run python scripts/abtest_minutes_allocators.py single \\
        --game-date 2025-12-23

Usage (multi-slate):
    uv run python scripts/abtest_minutes_allocators.py multi \\
        --start-date 2025-11-01 \\
        --end-date 2025-12-31
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import typer

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from projections.eval.minutes_alloc_abtest import (
    AllocatorComparisonMetrics,
    ScaleSharesDiagnostics,
    build_players_report,
    build_team_summary,
    compare_allocators,
    scale_shares_to_240,
)
from projections.eval.minutes_alloc_abtest_aggregate import (
    MISSING_FEATURE_CLEAN_THRESHOLD,
    MISSING_FEATURE_SKIP_THRESHOLD,
    run_aggregation,
    save_aggregates,
)
from projections.minutes_alloc.rotalloc_production import (
    resolve_rotalloc_bundle_dir,
    score_rotalloc_minutes,
)
from projections.minutes_v1.minute_share import (
    MinuteShareArtifacts,
    predict_minutes as predict_minute_shares,
)
from projections.models.minute_share_mixture import (
    MixtureBundle,
    predict_expected_minutes as predict_mixture_minutes,
)
from projections.cli.score_minutes_v1 import (
    _load_bundle as load_prod_bundle,
    _score_rows as score_prod_rows,
    _prepare_features as prepare_prod_features,
)
from projections.paths import data_path

app = typer.Typer(help=__doc__)

# Default paths
DEFAULT_CONFIG_PATH = Path("config/minutes_current_run.json")
DEFAULT_FEATURES_ROOT = data_path("gold", "features_minutes_v1")
DEFAULT_LIVE_FEATURES_ROOT = data_path("artifacts", "minutes_v1", "daily")
DEFAULT_LABELS_ROOT = data_path("gold", "labels_daily")
DEFAULT_OUT_ROOT = Path("runs/abtest_minutes_alloc")
DEFAULT_SHARE_MODEL_ROOT = Path("artifacts/minute_share")
DEFAULT_PROD_MINUTES_BUNDLE = Path("artifacts/minutes_lgbm/minutes_v1_safe_starter_20251214")

# Candidate column names for shares prediction
SHARES_COL_CANDIDATES = [
    "minute_share_pred",
    "share_pred",
    "predicted_share",
    "normalized_share",
    "raw_share",
    "share",
]

# Proxy columns that can be used as fallback shares
PROXY_COL_CANDIDATES = [
    "roll_mean_5",
    "min_last3",
    "min_last5",
    "roll_mean_10",
]

# Integrity thresholds
MIN_TEAMS_PER_SLATE = 2  # At least 1 game
MIN_PLAYERS_PER_TEAM = 8  # Minimum roster


def _validate_live_features_file(path: Path, *, target_date: date) -> bool:
    """Return True when a live features parquet matches the requested slate date."""
    try:
        df = pd.read_parquet(path, columns=["game_date", "tip_ts", "game_id"])
    except Exception:
        return False
    if df.empty:
        return False

    game_date = pd.to_datetime(df.get("game_date"), errors="coerce")
    unique_days = game_date.dt.date.dropna().unique()
    if unique_days.size != 1 or unique_days[0] != target_date:
        return False

    tip_ts = pd.to_datetime(df.get("tip_ts"), utc=True, errors="coerce")
    if tip_ts.isna().all():
        # Older feature slices may omit tip_ts; accept if game_date matches.
        return True

    tip_days = tip_ts.dt.date.dropna().unique()
    if tip_days.size == 0:
        return True

    allowed = {
        target_date,
        target_date - timedelta(days=1),
        target_date + timedelta(days=1),
    }
    return any(day in allowed for day in tip_days.tolist())


def _resolve_features_path(game_date: str, features_root: Path | None) -> Path | None:
    """Resolve path to features for a given game date.
    
    If features_root is explicitly provided, only look there (for rebuilt features).
    Otherwise, search default locations.
    """
    date_obj = datetime.strptime(game_date, "%Y-%m-%d").date()

    # If explicit features_root provided, only look there
    if features_root and features_root.exists():
        date_dir = features_root / game_date
        if date_dir.exists():
            # Prefer _contract suffix runs (rebuilt with contract enforcement)
            run_dirs = sorted([d for d in date_dir.iterdir() if d.is_dir()], reverse=True)
            # Prioritize contract-enforced runs
            for sub in run_dirs:
                if "_contract" in sub.name:
                    for name in ["features.parquet", "dedup_features.parquet"]:
                        candidate = sub / name
                        if candidate.exists():
                            return candidate
            # Fall back to any run
            for sub in run_dirs:
                for name in ["features.parquet", "dedup_features.parquet"]:
                    candidate = sub / name
                    if candidate.exists():
                        return candidate
        return None  # Don't fall through if explicit root was provided

    # Candidate live feature directories (try multiple locations)
    live_candidates = [
        DEFAULT_LIVE_FEATURES_ROOT / game_date,
        Path.home() / "projections-data" / "live" / "features_minutes_v1" / game_date,
        data_path("live", "features_minutes_v1") / game_date,
    ]

    for live_path in live_candidates:
        if not live_path.exists():
            continue
        # Look for latest run
        latest_pointer = live_path / "latest_run.json"
        if latest_pointer.exists():
            try:
                pointer = json.loads(latest_pointer.read_text())
                run_id = pointer.get("run_id")
                if run_id:
                    features_file = live_path / f"run={run_id}" / "features.parquet"
                    if features_file.exists() and _validate_live_features_file(features_file, target_date=date_obj):
                        return features_file
            except Exception:
                pass
        # Fall back to newest run directory with features.parquet
        run_dirs = sorted([d for d in live_path.iterdir() if d.is_dir()], reverse=True)
        candidates: list[Path] = []
        for sub in run_dirs:
            for name in ["features.parquet", "dedup_features.parquet"]:
                candidate = sub / name
                if candidate.exists():
                    candidates.append(candidate)
                    break
        if not candidates:
            continue

        # Prefer contract-enforced runs, but validate against slate date + tip timestamps.
        candidates_sorted = sorted(candidates, key=lambda p: p.parent.name, reverse=True)
        candidates_sorted = sorted(candidates_sorted, key=lambda p: "_contract" not in p.parent.name)
        for candidate in candidates_sorted:
            if _validate_live_features_file(candidate, target_date=date_obj):
                return candidate

        # Last resort: return the newest candidate even if validation failed.
        return candidates[0]

    return None


def _load_actual_minutes(game_date: str) -> pd.DataFrame | None:
    """Try to load actual minutes labels for evaluation."""
    date_obj = datetime.strptime(game_date, "%Y-%m-%d").date()
    season = date_obj.year if date_obj.month >= 8 else date_obj.year - 1

    label_paths = [
        Path.home() / "projections-data" / "labels" / f"season={season}" / "boxscore_labels.parquet",
        Path.home() / "projections-data" / "gold" / "labels_minutes_v1" / f"season={season}" / f"game_date={game_date}" / "labels.parquet",
        DEFAULT_LABELS_ROOT / f"season={season}" / "labels.parquet",
        data_path("labels", "nba_boxscores"),
    ]

    for labels_path in label_paths:
        if not labels_path.exists():
            continue
        try:
            labels = pd.read_parquet(labels_path)
            if "game_date" in labels.columns:
                labels["game_date"] = pd.to_datetime(labels["game_date"]).dt.date
                filtered = labels[labels["game_date"] == date_obj]
                if len(filtered) > 0:
                    return filtered
        except Exception:
            continue

    return None


def _load_share_model_with_columns() -> tuple[MinuteShareArtifacts | None, list[str], Path | None]:
    """Load share model and its expected feature columns.
    
    Returns:
        Tuple of (model, expected_columns, model_path)
    """
    share_model_root = DEFAULT_SHARE_MODEL_ROOT
    if not share_model_root.exists():
        return None, [], None

    # Look for minute_share_wfcv2_fold_* (newest first)
    candidates = sorted(
        [d for d in share_model_root.iterdir() 
         if d.is_dir() and ("wfcv2_fold" in d.name or "full_history_v2" in d.name)],
        key=lambda x: x.name,
        reverse=True,
    )

    for candidate in candidates:
        model_path = candidate / "minute_share_model.joblib"
        cols_path = candidate / "feature_columns.json"
        
        if model_path.exists():
            model = joblib.load(model_path)
            expected_cols = []
            
            if cols_path.exists():
                try:
                    cols_data = json.loads(cols_path.read_text())
                    expected_cols = cols_data.get("columns", [])
                except Exception:
                    pass
            
            return model, expected_cols, candidate
    
    return None, [], None


def _ensure_feature_columns(
    df: pd.DataFrame,
    expected_columns: list[str],
) -> tuple[pd.DataFrame, int, int, list[str]]:
    """Ensure all expected feature columns exist, filling missing with 0.
    
    Returns:
        Tuple of (df, n_expected, n_missing, missing_cols)
    """
    if not expected_columns:
        return df, 0, 0, []
    
    df = df.copy()
    missing_cols = []
    
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0.0
            missing_cols.append(col)
        else:
            # Fill NaNs with 0 for feature columns
            if df[col].isna().any():
                df[col] = df[col].fillna(0.0)
    
    return df, len(expected_columns), len(missing_cols), missing_cols


def _check_slate_integrity(
    df: pd.DataFrame,
) -> tuple[bool, str | None, dict[str, int]]:
    """Check if slate data has valid integrity.
    
    Returns:
        Tuple of (is_valid, skip_reason, counts)
    """
    n_players = len(df)
    n_games = df["game_id"].nunique() if "game_id" in df.columns else 0
    n_teams = df.groupby(["game_id", "team_id"]).ngroups if "game_id" in df.columns and "team_id" in df.columns else 0
    
    counts = {
        "n_players": n_players,
        "n_games": n_games,
        "n_teams": n_teams,
    }
    
    # Check basic existence
    if n_games == 0 or n_teams == 0:
        return False, "no_games_or_teams", counts
    
    # Check team/game ratio - should be exactly 2 teams per game
    if n_teams < 2 * n_games:
        return False, f"missing_teams (expected {2 * n_games}, got {n_teams})", counts
    
    # Check minimum players per team
    players_per_team = df.groupby(["game_id", "team_id"]).size()
    if players_per_team.min() < MIN_PLAYERS_PER_TEAM:
        return False, f"incomplete_rosters (min {players_per_team.min()} players)", counts
    
    return True, None, counts


def _predict_shares(
    features: pd.DataFrame,
    model: MinuteShareArtifacts,
) -> pd.DataFrame:
    """Use minute share model to predict shares on features."""
    # Build is_out mask
    is_out = pd.Series(0, index=features.index, dtype=int)
    if "is_out" in features.columns:
        is_out = is_out | features["is_out"].fillna(0).astype(int)
    if "status" in features.columns:
        status_out = features["status"].astype(str).str.upper() == "OUT"
        is_out = is_out | status_out.astype(int)

    # Predict
    predictions = predict_minute_shares(
        model,
        features,
        game_ids=features["game_id"],
        team_ids=features["team_id"],
        is_out=is_out,
        sharpen_exponent=2.0,
    )

    result = features.copy()
    result["raw_share"] = predictions["raw_share"]
    result["normalized_share"] = predictions["normalized_share"]
    result["predicted_minutes_share_model"] = predictions["predicted_minutes"]

    return result


def _compute_starter_realism_metrics(
    df: pd.DataFrame,
    minutes_col: str,
    group_cols: tuple[str, str] = ("game_id", "team_id"),
) -> dict[str, float]:
    """Compute starter realism metrics: top5 sum and max minutes per team."""
    if minutes_col not in df.columns:
        return {
            "top5_sum_mean": 0.0,
            "top5_sum_std": 0.0,
            "max_minutes_mean": 0.0,
            "max_minutes_std": 0.0,
        }

    top5_sums = []
    max_mins = []

    for _, g in df.groupby(list(group_cols), sort=False):
        mins = pd.to_numeric(g[minutes_col], errors="coerce").fillna(0.0).to_numpy()
        sorted_mins = np.sort(mins)[::-1]
        top5_sums.append(sorted_mins[:5].sum())
        max_mins.append(sorted_mins[0] if len(sorted_mins) > 0 else 0.0)

    return {
        "top5_sum_mean": float(np.mean(top5_sums)) if top5_sums else 0.0,
        "top5_sum_std": float(np.std(top5_sums)) if len(top5_sums) > 1 else 0.0,
        "max_minutes_mean": float(np.mean(max_mins)) if max_mins else 0.0,
        "max_minutes_std": float(np.std(max_mins)) if len(max_mins) > 1 else 0.0,
    }


@dataclass
class BlendDiagnostics:
    """Diagnostics from Allocator D (BLEND_WITHIN_ELIGIBLE)."""

    n_teams: int = 0
    n_players: int = 0
    n_eligible: int = 0
    alpha: float = 0.0
    gamma: float = 1.0
    team_sum_dev_max: float = 0.0
    cap_applied_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)


@dataclass
class FringeOnlyAlphaDiagnostics:
    """Diagnostics from Allocator E (FRINGE_ONLY_ALPHA)."""

    n_teams: int = 0
    n_players: int = 0
    n_eligible: int = 0
    k_core: int = 8
    alpha_core: float = 0.8
    alpha_fringe: float = 0.3
    gamma: float = 1.0
    mean_core_size: float = 0.0
    core_size_min: int = 0
    core_size_max: int = 0
    fallback_used_count: int = 0
    cap_applied_count: int = 0
    max_redistribution_rounds: int = 0
    team_sum_dev_max: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)


@dataclass
class PowerPostprocessDiagnostics:
    """Diagnostics from Allocator F (POWER_POSTPROCESS)."""

    n_teams: int = 0
    n_players: int = 0
    p_value: float = 1.2
    cap_applied_count: int = 0
    ineligible_nonzero_count: int = 0  # Should always be 0
    fallback_used_count: int = 0  # When all m_B=0
    max_redistribution_rounds: int = 0
    team_sum_dev_max: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)


def blend_within_eligible(
    features: pd.DataFrame,
    df_rotalloc: pd.DataFrame,
    shares_col: str,
    *,
    alpha: float = 0.7,
    gamma: float = 1.0,
    cap_max: float = 48.0,
    group_cols: tuple[str, str] = ("game_id", "team_id"),
) -> tuple[pd.DataFrame, BlendDiagnostics]:
    """Allocator D: Blend share weights with RotAlloc proxy weights within eligible set.

    Steps per (game_id, team_id):
        1. Use RotAlloc's eligible_flag, p_rot, mu_cond
        2. Compute w_share = share_pred ** gamma (within eligible)
        3. Compute w_rot = (p_rot ** a) * (mu_cond ** mu_power) (from RotAlloc config)
        4. Blend: w = alpha * w_share + (1-alpha) * w_rot
        5. Allocate: minutes = 240 * w / sum(w)
        6. Apply cap with redistribution

    Args:
        features: Input features with shares column
        df_rotalloc: Output from score_rotalloc_minutes (has p_rot, mu_cond, eligible_flag)
        shares_col: Column name for share predictions
        alpha: Blend weight (1.0 = pure shares, 0.0 = pure rotalloc proxy)
        gamma: Exponent for share weights
        cap_max: Hard cap on minutes per player
        group_cols: Team grouping columns

    Returns:
        Tuple of (output_df, diagnostics)
    """
    diag = BlendDiagnostics(alpha=alpha, gamma=gamma)

    # Merge features with rotalloc outputs
    merge_cols = ["player_id", "game_id", "team_id"]
    working = features.merge(
        df_rotalloc[merge_cols + ["p_rot", "mu_cond", "eligible_flag"]],
        on=merge_cols,
        how="left",
    )

    # Get shares
    shares = pd.to_numeric(working[shares_col], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy()

    # Get rotalloc outputs
    p_rot = pd.to_numeric(working["p_rot"], errors="coerce").fillna(0.0).clip(0.0, 1.0).to_numpy()
    mu_cond = pd.to_numeric(working["mu_cond"], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy()
    eligible = working["eligible_flag"].fillna(0).astype(int).to_numpy()

    # RotAlloc config defaults (from production config)
    a = 1.5
    mu_power = 1.5

    diag.n_players = len(working)
    diag.n_eligible = int(eligible.sum())

    # Allocate per team
    minutes_out = np.zeros(len(working), dtype=np.float64)
    team_sum_out = np.zeros(len(working), dtype=np.float64)
    max_dev = 0.0
    cap_count = 0

    for keys, g in working.groupby(list(group_cols), sort=False):
        idx = g.index.to_numpy()
        shares_g = shares[idx]
        p_rot_g = p_rot[idx]
        mu_cond_g = mu_cond[idx]
        eligible_g = eligible[idx].astype(bool)

        if not eligible_g.any():
            # No eligible players - fallback to top-1 by share
            top_idx_local = np.argmax(shares_g)
            minutes_g = np.zeros(len(idx), dtype=np.float64)
            minutes_g[top_idx_local] = 240.0
        else:
            # Compute share weights (within eligible only)
            w_share = np.where(eligible_g, np.power(shares_g + 1e-9, gamma), 0.0)

            # Compute rotalloc proxy weights
            w_rot = np.power(p_rot_g, a) * np.power(mu_cond_g, mu_power)
            w_rot = np.where(eligible_g & np.isfinite(w_rot), w_rot, 0.0)

            # Normalize each set of weights before blending
            w_share_sum = w_share.sum()
            w_rot_sum = w_rot.sum()

            if w_share_sum > 1e-12:
                w_share_norm = w_share / w_share_sum
            else:
                w_share_norm = w_share

            if w_rot_sum > 1e-12:
                w_rot_norm = w_rot / w_rot_sum
            else:
                w_rot_norm = w_rot

            # Blend
            w = alpha * w_share_norm + (1.0 - alpha) * w_rot_norm
            w = np.where(eligible_g, w, 0.0)

            # Allocate
            w_sum = w.sum()
            if w_sum > 1e-12:
                minutes_g = 240.0 * w / w_sum
            else:
                # Uniform among eligible
                n_elig = eligible_g.sum()
                minutes_g = np.where(eligible_g, 240.0 / n_elig, 0.0)

            # Apply cap with redistribution
            for _ in range(10):
                over_cap = minutes_g > cap_max
                if not over_cap.any():
                    break
                excess = (minutes_g - cap_max)[over_cap].sum()
                if excess < 1e-9:
                    break
                minutes_g = np.where(over_cap, cap_max, minutes_g)
                can_receive = eligible_g & (minutes_g < cap_max - 0.1)
                receive_weights = np.where(can_receive, w, 0.0)
                receive_sum = receive_weights.sum()
                if receive_sum > 1e-12:
                    minutes_g = np.where(
                        can_receive,
                        minutes_g + excess * receive_weights / receive_sum,
                        minutes_g,
                    )
                else:
                    break

            minutes_g = np.minimum(minutes_g, cap_max)
            cap_count += int((minutes_g >= cap_max - 0.1).sum())

        # Validate sum
        team_total = float(minutes_g.sum())
        dev = abs(team_total - 240.0)
        if dev > max_dev:
            max_dev = dev

        minutes_out[idx] = minutes_g
        team_sum_out[idx] = team_total

    # Build output
    out_cols = list(group_cols) + ["player_id"]
    out = working[out_cols].copy()
    out["minutes_mean_D"] = minutes_out
    out["alpha"] = alpha
    out["eligible_flag"] = eligible
    out["team_minutes_sum"] = team_sum_out

    diag.n_teams = working.groupby(list(group_cols), sort=False).ngroups
    diag.team_sum_dev_max = max_dev
    diag.cap_applied_count = cap_count

    return out, diag


def fringe_only_alpha_within_eligible(
    features: pd.DataFrame,
    df_rotalloc: pd.DataFrame,
    shares_col: str,
    *,
    k_core: int = 8,
    alpha_core: float = 0.8,
    alpha_fringe: float = 0.3,
    gamma: float = 1.0,
    cap_max: float = 48.0,
    group_cols: tuple[str, str] = ("game_id", "team_id"),
) -> tuple[pd.DataFrame, FringeOnlyAlphaDiagnostics]:
    """Allocator E: Two-tier alpha blend within eligible set.

    Core players (top k_core by w_rot proxy) use alpha_core (lean shares).
    Fringe players (remaining eligible) use alpha_fringe (lean RotAlloc proxy).

    This is designed to improve MAE in 10-30 minute buckets where fringe
    rotation players benefit more from RotAlloc's proxy ordering.

    Steps per (game_id, team_id):
        1. Use RotAlloc's eligible_flag, p_rot, mu_cond
        2. Compute w_rot = (p_rot^a) * (mu_cond^mu_power) for eligible
        3. Stable-sort eligible by w_rot descending -> top k_core are "core"
        4. For each eligible player i:
           - If i in core: w_i = alpha_core * w_share + (1-alpha_core) * w_rot
           - Else (fringe): w_i = alpha_fringe * w_share + (1-alpha_fringe) * w_rot
        5. Allocate: minutes = 240 * w / sum(w)
        6. Apply cap with redistribution
        7. Validate sum-to-240

    Args:
        features: Input features with shares column
        df_rotalloc: Output from score_rotalloc_minutes (has p_rot, mu_cond, eligible_flag)
        shares_col: Column name for share predictions
        k_core: Number of top players by w_rot to treat as "core" (default 8)
        alpha_core: Blend weight for core players (default 0.8 = lean shares)
        alpha_fringe: Blend weight for fringe players (default 0.3 = lean RotAlloc)
        gamma: Exponent for share weights (default 1.0)
        cap_max: Hard cap on minutes per player (default 48.0)
        group_cols: Team grouping columns

    Returns:
        Tuple of (output_df, diagnostics)
    """
    diag = FringeOnlyAlphaDiagnostics(
        k_core=k_core,
        alpha_core=alpha_core,
        alpha_fringe=alpha_fringe,
        gamma=gamma,
    )

    # Merge features with rotalloc outputs
    merge_cols = ["player_id", "game_id", "team_id"]
    working = features.merge(
        df_rotalloc[merge_cols + ["p_rot", "mu_cond", "eligible_flag"]],
        on=merge_cols,
        how="left",
    )

    # Get shares
    shares = pd.to_numeric(working[shares_col], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy()

    # Get rotalloc outputs
    p_rot = pd.to_numeric(working["p_rot"], errors="coerce").fillna(0.0).clip(0.0, 1.0).to_numpy()
    mu_cond = pd.to_numeric(working["mu_cond"], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy()
    eligible = working["eligible_flag"].fillna(0).astype(int).to_numpy()

    # RotAlloc config defaults (from production config)
    a = 1.5
    mu_power = 1.5

    diag.n_players = len(working)
    diag.n_eligible = int(eligible.sum())

    # Allocate per team
    minutes_out = np.zeros(len(working), dtype=np.float64)
    team_sum_out = np.zeros(len(working), dtype=np.float64)
    is_core_out = np.zeros(len(working), dtype=np.int32)
    max_dev = 0.0
    cap_count = 0
    fallback_count = 0
    max_redist_rounds = 0
    core_sizes: list[int] = []

    for keys, g in working.groupby(list(group_cols), sort=False):
        idx = g.index.to_numpy()
        shares_g = shares[idx]
        p_rot_g = p_rot[idx]
        mu_cond_g = mu_cond[idx]
        eligible_g = eligible[idx].astype(bool)

        if not eligible_g.any():
            # No eligible players - fallback to top-1 by share
            top_idx_local = np.argmax(shares_g)
            minutes_g = np.zeros(len(idx), dtype=np.float64)
            minutes_g[top_idx_local] = 240.0
            is_core_g = np.zeros(len(idx), dtype=np.int32)
            fallback_count += 1
            core_sizes.append(0)
        else:
            # Compute w_rot for all eligible
            w_rot_raw = np.power(p_rot_g, a) * np.power(mu_cond_g, mu_power)
            w_rot_raw = np.where(eligible_g & np.isfinite(w_rot_raw), w_rot_raw, 0.0)

            # Determine core vs fringe within eligible
            # Get indices of eligible players, sorted by w_rot descending (stable sort)
            eligible_indices = np.where(eligible_g)[0]
            w_rot_eligible = w_rot_raw[eligible_indices]
            # Stable sort: use argsort with kind='stable' on negated values
            sorted_order = np.argsort(-w_rot_eligible, kind="stable")
            sorted_eligible_indices = eligible_indices[sorted_order]

            # Top k_core are "core", rest are "fringe"
            actual_core_size = min(k_core, len(sorted_eligible_indices))
            core_indices = set(sorted_eligible_indices[:actual_core_size])
            core_sizes.append(actual_core_size)

            # Build is_core mask for this team
            is_core_g = np.array([1 if i in core_indices else 0 for i in range(len(idx))], dtype=np.int32)

            # Compute share weights (within eligible only)
            w_share = np.where(eligible_g, np.power(shares_g + 1e-9, gamma), 0.0)

            # Normalize each set of weights before blending
            w_share_sum = w_share.sum()
            w_rot_sum = w_rot_raw.sum()

            if w_share_sum > 1e-12:
                w_share_norm = w_share / w_share_sum
            else:
                w_share_norm = np.zeros_like(w_share)
                fallback_count += 1

            if w_rot_sum > 1e-12:
                w_rot_norm = w_rot_raw / w_rot_sum
            else:
                w_rot_norm = np.zeros_like(w_rot_raw)

            # Blend with two-tier alpha
            w = np.zeros(len(idx), dtype=np.float64)
            for i in range(len(idx)):
                if eligible_g[i]:
                    if is_core_g[i]:
                        w[i] = alpha_core * w_share_norm[i] + (1.0 - alpha_core) * w_rot_norm[i]
                    else:
                        w[i] = alpha_fringe * w_share_norm[i] + (1.0 - alpha_fringe) * w_rot_norm[i]

            # Allocate
            w_sum = w.sum()
            if w_sum > 1e-12:
                minutes_g = 240.0 * w / w_sum
            else:
                # Fallback: uniform among eligible
                n_elig = eligible_g.sum()
                minutes_g = np.where(eligible_g, 240.0 / n_elig, 0.0)
                fallback_count += 1

            # Apply cap with redistribution
            redist_rounds = 0
            for round_i in range(10):
                over_cap = minutes_g > cap_max
                if not over_cap.any():
                    break
                excess = (minutes_g - cap_max)[over_cap].sum()
                if excess < 1e-9:
                    break
                minutes_g = np.where(over_cap, cap_max, minutes_g)
                can_receive = eligible_g & (minutes_g < cap_max - 0.1)
                receive_weights = np.where(can_receive, w, 0.0)
                receive_sum = receive_weights.sum()
                if receive_sum > 1e-12:
                    minutes_g = np.where(
                        can_receive,
                        minutes_g + excess * receive_weights / receive_sum,
                        minutes_g,
                    )
                    redist_rounds = round_i + 1
                else:
                    break

            minutes_g = np.minimum(minutes_g, cap_max)
            cap_count += int((minutes_g >= cap_max - 0.1).sum())
            if redist_rounds > max_redist_rounds:
                max_redist_rounds = redist_rounds

        # Validate sum
        team_total = float(minutes_g.sum())
        dev = abs(team_total - 240.0)
        if dev > max_dev:
            max_dev = dev

        minutes_out[idx] = minutes_g
        team_sum_out[idx] = team_total
        is_core_out[idx] = is_core_g

    # Build output
    out_cols = list(group_cols) + ["player_id"]
    out = working[out_cols].copy()
    out["minutes_mean_E"] = minutes_out
    out["is_core"] = is_core_out
    out["eligible_flag"] = eligible
    out["team_minutes_sum"] = team_sum_out

    # Finalize diagnostics
    diag.n_teams = working.groupby(list(group_cols), sort=False).ngroups
    diag.team_sum_dev_max = max_dev
    diag.cap_applied_count = cap_count
    diag.fallback_used_count = fallback_count
    diag.max_redistribution_rounds = max_redist_rounds
    if core_sizes:
        diag.mean_core_size = float(np.mean(core_sizes))
        diag.core_size_min = int(np.min(core_sizes))
        diag.core_size_max = int(np.max(core_sizes))

    return out, diag


def postprocess_power_minutes(
    df_rotalloc: pd.DataFrame,
    *,
    p: float = 1.2,
    cap_max: float = 48.0,
    group_cols: tuple[str, str] = ("game_id", "team_id"),
) -> tuple[pd.DataFrame, PowerPostprocessDiagnostics]:
    """
    Allocator F: Apply power transform to RotAlloc (B) minutes to fix flat-top.
    
    Per team:
    1. Take m_B from RotAlloc (already sums to 240)
    2. Apply exponent p >= 1: m_raw = m_B ** p
    3. Renormalize: m = 240 * m_raw / sum(m_raw)
    4. Apply cap_max + redistribution
    5. Validate sum-to-240
    
    Higher p => more top-heavy distribution (fixes flat top pathology).
    
    Args:
        df_rotalloc: RotAlloc output containing minutes_mean, eligible_flag, player_id
        p: Power exponent (default 1.2). Higher = more concentration.
        cap_max: Maximum minutes per player (default 48.0)
        group_cols: Columns for team grouping
    
    Returns:
        DataFrame with minutes_mean_F column and original columns
        PowerPostprocessDiagnostics with execution stats
    """
    if df_rotalloc.empty:
        out = df_rotalloc.copy()
        out["minutes_mean_F"] = 0.0
        return out, PowerPostprocessDiagnostics()
    
    diag = PowerPostprocessDiagnostics(p_value=p)
    
    # Ensure required columns
    required = ["player_id", "minutes_mean"]
    for col in required:
        if col not in df_rotalloc.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Add eligible_flag if missing (treat all as eligible if not present)
    if "eligible_flag" not in df_rotalloc.columns:
        df_rotalloc = df_rotalloc.copy()
        df_rotalloc["eligible_flag"] = 1
    
    # Build output
    out = df_rotalloc.copy()
    out["minutes_mean_F"] = 0.0
    
    # Group by team
    gcols = list(group_cols)
    for gcol in gcols:
        if gcol not in out.columns:
            # Single-team mode
            gcols = []
            break
    
    if gcols:
        groups = out.groupby(gcols, sort=False)
    else:
        # Single team for testing
        groups = [(None, out)]
    
    team_sum_devs = []
    cap_applied_total = 0
    ineligible_nonzero_total = 0
    fallback_count = 0
    max_redist_rounds = 0
    
    for group_key, group in groups:
        diag.n_teams += 1
        idx = group.index
        n = len(idx)
        diag.n_players += n
        
        m_B = group["minutes_mean"].values.astype(np.float64)
        eligible = group["eligible_flag"].values.astype(np.int32)
        
        # Only process players with m_B > 0 (which should match eligible)
        active_mask = m_B > 0
        
        if not active_mask.any():
            # All zeros - fallback to original (shouldn't happen normally)
            out.loc[idx, "minutes_mean_F"] = m_B
            fallback_count += 1
            team_sum_devs.append(abs(m_B.sum() - 240.0))
            continue
        
        # Apply power transform
        m_raw = np.zeros(n, dtype=np.float64)
        m_raw[active_mask] = np.power(m_B[active_mask], p)
        
        # Renormalize to 240
        total_raw = m_raw.sum()
        if total_raw > 0:
            m_F = 240.0 * m_raw / total_raw
        else:
            # Fallback - shouldn't happen if active_mask check passed
            m_F = m_B.copy()
            fallback_count += 1
        
        # Apply cap with redistribution
        cap_rounds = 0
        for _ in range(20):  # Max redistribution rounds
            over_cap = m_F > cap_max
            if not over_cap.any():
                break
            
            cap_rounds += 1
            cap_applied_total += over_cap.sum()
            
            # Calculate excess
            excess = (m_F[over_cap] - cap_max).sum()
            m_F[over_cap] = cap_max
            
            # Redistribute to uncapped active players
            uncapped_active = active_mask & ~over_cap & (m_F < cap_max)
            if uncapped_active.any():
                uncapped_sum = m_F[uncapped_active].sum()
                if uncapped_sum > 0:
                    m_F[uncapped_active] += excess * m_F[uncapped_active] / uncapped_sum
        
        max_redist_rounds = max(max_redist_rounds, cap_rounds)
        
        # Ensure non-active (ineligible) players stay at 0
        m_F[~active_mask] = 0.0
        
        # Check for ineligible with nonzero (should never happen)
        ineligible_with_minutes = ((eligible == 0) & (m_F > 0)).sum()
        ineligible_nonzero_total += ineligible_with_minutes
        
        # Final normalization to exactly 240
        active_sum = m_F[active_mask].sum()
        if active_sum > 0:
            m_F[active_mask] *= 240.0 / active_sum
        
        out.loc[idx, "minutes_mean_F"] = m_F
        team_sum_devs.append(abs(m_F.sum() - 240.0))
    
    # Populate diagnostics
    diag.cap_applied_count = cap_applied_total
    diag.ineligible_nonzero_count = ineligible_nonzero_total
    diag.fallback_used_count = fallback_count
    diag.max_redistribution_rounds = max_redist_rounds
    if team_sum_devs:
        diag.team_sum_dev_max = float(max(team_sum_devs))
    
    return out, diag


@dataclass
class BenchpoolDiagnostics:
    """Diagnostics from Allocator N (MIXTURE_BENCHPOOL)."""

    n_teams: int = 0
    n_players: int = 0
    bench_pool: float = 80.0
    core_k: int = 6
    cap_applied_count: int = 0
    fallback_count: int = 0
    max_redistribution_rounds: int = 0
    team_sum_dev_max: float = 0.0
    mean_core_size: float = 0.0
    core_size_min: int = 0
    core_size_max: int = 0

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)


def allocate_mixture_benchpool(
    expected_minutes: np.ndarray,
    mask_inactive: np.ndarray,
    team_ids: np.ndarray,
    *,
    bench_pool: float = 80.0,
    core_k: int = 6,
    cap_max: float = 48.0,
) -> tuple[np.ndarray, BenchpoolDiagnostics]:
    """Allocator N: Mixture-based allocation with core/bench pool split.

    Steps per team:
        1. Use E_minutes from mixture model (already provided as expected_minutes)
        2. Mask out inactive players (OUT/is_out)
        3. Determine core set: top core_k players by E_minutes (masked excluded)
        4. Allocate core_pool = 240 - bench_pool minutes among core set proportional to E_minutes
        5. Allocate bench_pool minutes among remaining eligible proportional to E_minutes
        6. Apply cap_max + redistribution
        7. Validate sum-to-240

    Args:
        expected_minutes: E[minutes] from mixture model (shape: n_players)
        mask_inactive: Boolean mask where True = inactive (OUT/is_out), should get 0 minutes
        team_ids: Array of team IDs for grouping
        bench_pool: Minutes to allocate to bench (non-core) players (default 80)
        core_k: Number of top players by E_minutes to treat as "core" (default 6)
        cap_max: Maximum minutes per player (default 48.0)

    Returns:
        Tuple of (minutes_array, diagnostics)
        minutes_array has same length as expected_minutes, sums to 240 per team
    """
    n_players = len(expected_minutes)
    minutes_out = np.zeros(n_players, dtype=np.float64)

    diag = BenchpoolDiagnostics(bench_pool=bench_pool, core_k=core_k)
    diag.n_players = n_players

    # Get unique teams
    unique_teams = np.unique(team_ids)
    diag.n_teams = len(unique_teams)

    core_pool = 240.0 - bench_pool
    max_dev = 0.0
    cap_count = 0
    fallback_count = 0
    max_redist_rounds = 0
    core_sizes: list[int] = []

    for team in unique_teams:
        team_mask = team_ids == team
        idx = np.where(team_mask)[0]

        E_min_team = expected_minutes[idx]
        inactive_team = mask_inactive[idx]

        # Active = not inactive
        active_mask = ~inactive_team
        E_min_active = np.where(active_mask, E_min_team, 0.0)

        n_active = active_mask.sum()
        if n_active == 0:
            # No active players - fallback to top-1 by E_minutes (ignoring mask)
            top_idx_local = np.argmax(E_min_team)
            minutes_team = np.zeros(len(idx), dtype=np.float64)
            minutes_team[top_idx_local] = 240.0
            fallback_count += 1
            core_sizes.append(0)
            minutes_out[idx] = minutes_team
            continue

        # Determine core set: top core_k active players by E_minutes
        active_indices = np.where(active_mask)[0]
        E_min_for_sort = E_min_active[active_indices]
        # Sort descending by E_minutes (stable sort)
        sorted_order = np.argsort(-E_min_for_sort, kind="stable")
        sorted_active_indices = active_indices[sorted_order]

        actual_core_k = min(core_k, len(sorted_active_indices))
        core_indices_local = set(sorted_active_indices[:actual_core_k])
        bench_indices_local = set(sorted_active_indices[actual_core_k:])
        core_sizes.append(actual_core_k)

        # Compute shares within core and bench
        E_core = np.array([E_min_team[i] if i in core_indices_local else 0.0 for i in range(len(idx))])
        E_bench = np.array([E_min_team[i] if i in bench_indices_local else 0.0 for i in range(len(idx))])

        core_sum = E_core.sum()
        bench_sum = E_bench.sum()

        # Allocate core_pool to core players proportionally
        if core_sum > 1e-9:
            minutes_core = core_pool * E_core / core_sum
        else:
            # Uniform among core if no E_minutes signal
            minutes_core = np.zeros(len(idx))
            if actual_core_k > 0:
                for i in core_indices_local:
                    minutes_core[i] = core_pool / actual_core_k

        # Allocate bench_pool to bench players proportionally
        if bench_sum > 1e-9:
            minutes_bench = bench_pool * E_bench / bench_sum
        else:
            # Uniform among bench if no E_minutes signal
            n_bench = len(bench_indices_local)
            minutes_bench = np.zeros(len(idx))
            if n_bench > 0:
                for i in bench_indices_local:
                    minutes_bench[i] = bench_pool / n_bench

        # Combine
        minutes_team = minutes_core + minutes_bench

        # Ensure inactive players get 0
        minutes_team[inactive_team] = 0.0

        # Apply cap with redistribution
        for round_num in range(20):
            over_cap = minutes_team > cap_max
            if not over_cap.any():
                break

            cap_count += int(over_cap.sum())
            excess = (minutes_team[over_cap] - cap_max).sum()
            minutes_team[over_cap] = cap_max

            # Redistribute to uncapped active players
            uncapped_active = active_mask & ~over_cap & (minutes_team < cap_max)
            if uncapped_active.any():
                uncapped_sum = minutes_team[uncapped_active].sum()
                if uncapped_sum > 0:
                    minutes_team[uncapped_active] += excess * minutes_team[uncapped_active] / uncapped_sum

            max_redist_rounds = max(max_redist_rounds, round_num + 1)

        # Final normalization to exactly 240
        active_sum = minutes_team[active_mask].sum()
        if active_sum > 0:
            minutes_team[active_mask] *= 240.0 / active_sum

        # Ensure inactive stay at 0
        minutes_team[~active_mask] = 0.0

        dev = abs(minutes_team.sum() - 240.0)
        max_dev = max(max_dev, dev)

        minutes_out[idx] = minutes_team

    # Populate diagnostics
    diag.cap_applied_count = cap_count
    diag.fallback_count = fallback_count
    diag.max_redistribution_rounds = max_redist_rounds
    diag.team_sum_dev_max = max_dev

    if core_sizes:
        diag.mean_core_size = float(np.mean(core_sizes))
        diag.core_size_min = int(np.min(core_sizes))
        diag.core_size_max = int(np.max(core_sizes))

    return minutes_out, diag


def _compute_ordering_accuracy_10_30(
    pred_minutes: np.ndarray,
    actual_minutes: np.ndarray,
    team_ids: np.ndarray,
) -> float | None:
    """Compute pairwise ordering accuracy for players with true minutes in [10, 30].
    
    For each team, among players with true minutes in [10, 30]:
    - Count pairs where sign(pred_i - pred_j) == sign(true_i - true_j)
    - Return mean accuracy across teams
    
    This directly measures whether the model learned mid-range ordering.
    
    Args:
        pred_minutes: Predicted minutes array
        actual_minutes: Actual minutes array
        team_ids: Team ID array
        
    Returns:
        Mean pairwise ordering accuracy, or None if insufficient data
    """
    # Build dataframe for grouping
    df = pd.DataFrame({
        "pred": pred_minutes,
        "actual": actual_minutes,
        "team_id": team_ids,
    })
    
    # Filter to [10, 30] range
    mask_10_30 = (df["actual"] >= 10) & (df["actual"] <= 30)
    df_filtered = df[mask_10_30]
    
    if len(df_filtered) < 2:
        return None
    
    team_accuracies = []
    
    for team_id, group in df_filtered.groupby("team_id"):
        n = len(group)
        if n < 2:
            continue
        
        pred_vals = group["pred"].values
        actual_vals = group["actual"].values
        
        # Count concordant pairs
        concordant = 0
        total_pairs = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                # Skip ties in actual
                if actual_vals[i] == actual_vals[j]:
                    continue
                
                total_pairs += 1
                pred_sign = np.sign(pred_vals[i] - pred_vals[j])
                actual_sign = np.sign(actual_vals[i] - actual_vals[j])
                
                if pred_sign == actual_sign:
                    concordant += 1
        
        if total_pairs > 0:
            team_accuracies.append(concordant / total_pairs)
    
    if not team_accuracies:
        return None
    
    return float(np.mean(team_accuracies))


def _run_single_slate(
    game_date: str,
    bundle_dir: Path,
    out_dir: Path,
    cap_max: float,
    k_core: int,
    use_share_model: bool,
    share_model: MinuteShareArtifacts | None,
    expected_columns: list[str],
    features_root: Path | None = None,
    alpha_grid: list[float] | None = None,
    alpha_core: float = 0.8,
    alpha_fringe: float = 0.3,
    p_grid: list[float] | None = None,
    mixture_bundle: MixtureBundle | None = None,
    debug_mixture: bool = False,
    bench_pool_grid: list[float] | None = None,
    core_k_grid: list[int] | None = None,
    prod_bundle: dict | None = None,
    prod_bundle_dir: Path | None = None,
) -> dict[str, Any]:
    """Run single slate A/B/C/D/E/F/M/N/P test.

    Args:
        alpha_grid: List of alpha values for Allocator D grid search.
                   Default is [0.0, 0.3, 0.5, 0.7, 0.9, 1.0].
        alpha_core: Blend weight for core players in Allocator E (default 0.8).
        alpha_fringe: Blend weight for fringe players in Allocator E (default 0.3).
        p_grid: List of power exponents for Allocator F grid search.
                Default is [1.0, 1.1, 1.2, 1.3]. Default p for metrics is 1.2.
        mixture_bundle: Optional trained MixtureBundle for Allocator M/N.
                       If None, Allocator M and N are skipped.
        debug_mixture: If True, emit detailed debug logs for Allocator M/N.
        bench_pool_grid: List of bench_pool values for Allocator N grid search.
                        Default is [70, 80, 90]. Default for metrics is 80.
        core_k_grid: List of core_k values for Allocator N grid search.
                    Default is [5, 6]. Default for metrics is 6.
        prod_bundle: Optional production minutes model bundle for Allocator P.
                    If None, Allocator P is skipped.
        prod_bundle_dir: Path to production bundle directory (for metadata).

    Returns a dict with status, quality tier, and metrics.
    """
    if alpha_grid is None:
        alpha_grid = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]
    if p_grid is None:
        p_grid = [1.0, 1.1, 1.2, 1.3]
    if bench_pool_grid is None:
        bench_pool_grid = [70.0, 80.0, 90.0]
    if core_k_grid is None:
        core_k_grid = [5, 6]
    # Default params for N metrics
    default_N_bench_pool = 80.0
    default_N_core_k = 6
    result = {
        "game_date": game_date,
        "status": "unknown",
        "quality_tier": "skipped",
        "skip_reason": None,
        "missing_feature_frac": 0.0,
        "n_expected_features": len(expected_columns),
        "n_missing_features": 0,
        "integrity_counts": {},
    }
    
    def _save_skip_summary():
        """Save a stub summary.json for skipped slates."""
        try:
            summary = {
                "game_date": game_date,
                "timestamp": datetime.now().isoformat(),
                "quality_tier": "skipped",
                "skip_reason": result.get("skip_reason"),
                "missing_feature_frac": result.get("missing_feature_frac", 0.0),
                "n_expected_features": result.get("n_expected_features", 0),
                "n_missing_features": result.get("n_missing_features", 0),
                "integrity_counts": result.get("integrity_counts", {}),
                "metrics": {},
            }
            (slate_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        except Exception:
            pass  # Ignore save errors for skip summaries
    
    try:
        # Create slate-specific output dir
        slate_dir = out_dir / game_date
        slate_dir.mkdir(parents=True, exist_ok=True)
        
        # Resolve features path (prefer explicit features_root if provided)
        features_path = _resolve_features_path(game_date, features_root)
        if features_path is None:
            result["status"] = "skipped"
            result["skip_reason"] = "features_not_found"
            _save_skip_summary()
            return result
        
        # Load features
        features = pd.read_parquet(features_path)
        if "game_date" in features.columns:
            features["game_date"] = pd.to_datetime(features["game_date"]).dt.date
            target_date = datetime.strptime(game_date, "%Y-%m-%d").date()
            features = features[features["game_date"] == target_date]
        
        if features.empty:
            result["status"] = "skipped"
            result["skip_reason"] = "no_features_for_date"
            _save_skip_summary()
            return result
        
        # Check slate integrity
        is_valid, integrity_reason, integrity_counts = _check_slate_integrity(features)
        result["integrity_counts"] = integrity_counts
        
        if not is_valid:
            result["status"] = "skipped"
            result["skip_reason"] = f"integrity_failed: {integrity_reason}"
            _save_skip_summary()
            return result
        
        # Ensure feature columns for share model
        if use_share_model and expected_columns:
            features, n_expected, n_missing, missing_cols = _ensure_feature_columns(
                features, expected_columns
            )
            result["n_expected_features"] = n_expected
            result["n_missing_features"] = n_missing
            
            missing_frac = n_missing / n_expected if n_expected > 0 else 0.0
            result["missing_feature_frac"] = missing_frac
            
            # Skip if too many features missing
            if missing_frac > MISSING_FEATURE_SKIP_THRESHOLD:
                result["status"] = "skipped"
                result["skip_reason"] = f"too_many_missing_features ({n_missing}/{n_expected})"
                _save_skip_summary()
                return result
            
            # Determine quality tier
            if missing_frac <= MISSING_FEATURE_CLEAN_THRESHOLD:
                result["quality_tier"] = "clean"
            else:
                result["quality_tier"] = "degraded"
        else:
            result["quality_tier"] = "clean"
        
        # Predict shares if model available
        if use_share_model and share_model is not None:
            features = _predict_shares(features, share_model)
            shares_col = "normalized_share"
        else:
            # Fall back to proxy
            for col in PROXY_COL_CANDIDATES:
                if col in features.columns:
                    shares_col = col
                    break
            else:
                result["status"] = "skipped"
                result["skip_reason"] = "no_shares_column"
                return result
        
        # --- Run Allocator B: ROTALLOC (first to get eligible_flag) ---
        df_B, rotalloc_config, rotalloc_diag = score_rotalloc_minutes(
            features,
            bundle_dir=bundle_dir,
        )
        
        # --- Run Allocator A: SCALE_SHARES (all players) ---
        df_A, diag_A = scale_shares_to_240(
            features,
            shares_col=shares_col,
            cap_max=cap_max,
            require_positive_share=False,
            redistribute_after_cap=True,
        )
        
        # --- Run Allocator C: SHARE_WITH_ROTALLOC_ELIGIBILITY ---
        # Scale shares ONLY within eligible set
        if "eligible_flag" in df_B.columns:
            merge_cols = ["player_id", "game_id", "team_id"]
            features_c = features.merge(
                df_B[merge_cols + ["eligible_flag"]].rename(columns={"eligible_flag": "_eligible"}),
                on=merge_cols,
                how="left",
            )
            features_c["_eligible_share"] = np.where(
                features_c["_eligible"].fillna(0) == 1,
                features_c[shares_col],
                0.0,
            )
            df_C, diag_C = scale_shares_to_240(
                features_c,
                shares_col="_eligible_share",
                cap_max=cap_max,
                require_positive_share=False,
                redistribute_after_cap=True,
            )
            # Rename output column
            df_C = df_C.rename(columns={"minutes_mean_A": "minutes_mean_C"})
        else:
            # No eligibility info, C = A
            df_C = df_A.copy()
            df_C = df_C.rename(columns={"minutes_mean_A": "minutes_mean_C"})
            diag_C = diag_A

        # --- Run Allocator D: BLEND_WITHIN_ELIGIBLE (alpha grid) ---
        # Each alpha produces a separate allocator variant: D_alpha_0.7, etc.
        allocator_d_results: dict[float, tuple[pd.DataFrame, BlendDiagnostics]] = {}
        for alpha in alpha_grid:
            df_D_alpha, diag_D_alpha = blend_within_eligible(
                features,
                df_B,
                shares_col,
                alpha=alpha,
                gamma=1.0,
                cap_max=cap_max,
            )
            allocator_d_results[alpha] = (df_D_alpha, diag_D_alpha)

        # --- Run Allocator E: FRINGE_ONLY_ALPHA ---
        # Two-tier alpha: core players use alpha_core, fringe use alpha_fringe
        df_E, diag_E = fringe_only_alpha_within_eligible(
            features,
            df_B,
            shares_col,
            k_core=k_core,
            alpha_core=alpha_core,
            alpha_fringe=alpha_fringe,
            gamma=1.0,
            cap_max=cap_max,
        )

        # --- Run Allocator F: POWER_POSTPROCESS ---
        # Apply power transform to RotAlloc minutes to fix flat-top pathology
        # Grid search over p values, track default (p=1.2) and best
        allocator_f_results: dict[float, tuple[pd.DataFrame, PowerPostprocessDiagnostics]] = {}
        default_F_p = 1.2  # Default p for F metrics
        
        for p in p_grid:
            df_F_p, diag_F_p = postprocess_power_minutes(
                df_B,
                p=p,
                cap_max=cap_max,
            )
            allocator_f_results[p] = (df_F_p, diag_F_p)
        
        # Get default p result (use closest available if exact not in grid)
        if default_F_p in allocator_f_results:
            df_F, diag_F = allocator_f_results[default_F_p]
        else:
            # Use closest p to default
            closest_p = min(p_grid, key=lambda x: abs(x - default_F_p))
            df_F, diag_F = allocator_f_results[closest_p]
            default_F_p = closest_p

        # --- Run Allocator M: MIXTURE_SHARES_SCALED ---
        # Use mixture model to predict expected minutes, convert to shares, scale to 240
        df_M = None
        allocator_M_status = "skipped"
        allocator_M_skip_reason: str | None = None

        # Group columns for team-level operations
        gcols = ["game_id", "team_id"]

        # Debug helper
        def _dbg(msg: str) -> None:
            if debug_mixture:
                import sys
                print(f"[DEBUG-M] {msg}", file=sys.stderr)

        if mixture_bundle is None:
            allocator_M_skip_reason = "mixture_bundle_not_provided"
            _dbg("mixture_bundle is None - skipped")
        else:
            _dbg(f"mixture_bundle loaded with {len(mixture_bundle.feature_columns)} features")

            # Prepare features for mixture model
            mixture_features_cols = mixture_bundle.feature_columns
            missing_cols = [c for c in mixture_features_cols if c not in features.columns]
            present_cols = [c for c in mixture_features_cols if c in features.columns]

            _dbg(f"Feature check: {len(present_cols)}/{len(mixture_features_cols)} present, {len(missing_cols)} missing")
            if missing_cols and debug_mixture:
                _dbg(f"Missing cols (first 10): {missing_cols[:10]}")

            if missing_cols:
                allocator_M_skip_reason = f"missing_{len(missing_cols)}_of_{len(mixture_features_cols)}_features"
                _dbg(f"SKIP: {allocator_M_skip_reason}")
            else:
                X_mix = features[mixture_features_cols].copy()
                _dbg(f"X_mix shape: {X_mix.shape}, NaN count: {X_mix.isna().sum().sum()}")

                try:
                    # Predict expected minutes
                    expected_minutes_M = predict_mixture_minutes(X_mix, mixture_bundle)
                    _dbg(f"expected_minutes_M: min={expected_minutes_M.min():.2f}, max={expected_minutes_M.max():.2f}, mean={expected_minutes_M.mean():.2f}")

                    # Mask inactive: status OUT or is_out=1
                    mask_inactive = np.zeros(len(features), dtype=bool)
                    n_status_out = 0
                    n_is_out = 0
                    if "status" in features.columns:
                        status_mask = features["status"].str.upper().isin(["OUT", "OFS", "NWT"])
                        mask_inactive |= status_mask
                        n_status_out = status_mask.sum()
                    if "is_out" in features.columns:
                        is_out_mask = features["is_out"].fillna(0).astype(bool)
                        mask_inactive |= is_out_mask
                        n_is_out = is_out_mask.sum()

                    _dbg(f"Mask: {mask_inactive.sum()} inactive (status_OUT={n_status_out}, is_out={n_is_out}), {(~mask_inactive).sum()} active")

                    expected_minutes_M[mask_inactive] = 0.0
                    expected_minutes_M = np.clip(expected_minutes_M, 0, cap_max)
                    _dbg(f"After mask+clip: min={expected_minutes_M.min():.2f}, max={expected_minutes_M.max():.2f}, mean={expected_minutes_M.mean():.2f}")

                    # Convert to shares within team
                    df_M_temp = features[gcols + ["player_id"]].copy()
                    df_M_temp["expected_min_M"] = expected_minutes_M

                    team_sum = df_M_temp.groupby(gcols)["expected_min_M"].transform("sum")
                    _dbg(f"Team sums before normalize: min={team_sum.min():.2f}, max={team_sum.max():.2f}, mean={team_sum.mean():.2f}")

                    team_sum = team_sum.replace(0, 1e-8)  # Avoid divide by zero
                    df_M_temp["share_M"] = df_M_temp["expected_min_M"] / team_sum

                    # Scale to 240
                    df_M_temp["minutes_mean_M"] = 240.0 * df_M_temp["share_M"]

                    # Apply cap and redistribute
                    for group_key, group_idx in df_M_temp.groupby(gcols).groups.items():
                        group = df_M_temp.loc[group_idx]
                        minutes = group["minutes_mean_M"].values.copy()

                        # Iterative cap/redistribute
                        for _ in range(10):
                            over_cap = minutes > cap_max
                            if not over_cap.any():
                                break
                            excess = (minutes - cap_max).clip(min=0).sum()
                            minutes[over_cap] = cap_max
                            under_cap = (minutes < cap_max) & (minutes > 0)
                            if under_cap.any():
                                headroom = (cap_max - minutes[under_cap]).sum()
                                if headroom > 0:
                                    redist = min(excess, headroom)
                                    minutes[under_cap] += redist * (cap_max - minutes[under_cap]) / headroom

                        # Renormalize to exactly 240
                        if minutes.sum() > 0:
                            minutes = 240.0 * minutes / minutes.sum()

                        df_M_temp.loc[group_idx, "minutes_mean_M"] = minutes

                    # Build final df_M
                    df_M = features.copy()
                    # Drop 'minutes' column if present to avoid suffix collision when merging with actual_df
                    if "minutes" in df_M.columns:
                        df_M = df_M.drop(columns=["minutes"])
                    df_M["minutes_mean_M"] = df_M_temp["minutes_mean_M"].values

                    final_min = df_M["minutes_mean_M"].min()
                    final_max = df_M["minutes_mean_M"].max()
                    final_mean = df_M["minutes_mean_M"].mean()
                    final_sum = df_M.groupby(gcols)["minutes_mean_M"].sum()
                    _dbg(f"Final minutes_mean_M: min={final_min:.2f}, max={final_max:.2f}, mean={final_mean:.2f}")
                    _dbg(f"Team sums: min={final_sum.min():.2f}, max={final_sum.max():.2f}")

                    allocator_M_status = "success"
                    _dbg("Allocator M completed successfully")

                except Exception as e:
                    allocator_M_status = "error"
                    allocator_M_skip_reason = f"predict_error: {e}"
                    _dbg(f"ERROR during prediction: {e}")
                    df_M = None

        # --- Run Allocator N: MIXTURE_BENCHPOOL ---
        # Uses mixture E_minutes with core/bench pool split
        df_N = None
        allocator_N_status = "skipped"
        allocator_N_skip_reason: str | None = None
        allocator_N_results: dict[tuple[float, int], tuple[pd.DataFrame, BenchpoolDiagnostics]] = {}
        diag_N_default = None

        if mixture_bundle is None:
            allocator_N_skip_reason = "mixture_bundle_not_provided"
            _dbg("Allocator N skipped: mixture_bundle is None")
        elif allocator_M_status != "success":
            # N depends on the same E_minutes as M
            allocator_N_skip_reason = f"allocator_M_failed: {allocator_M_skip_reason}"
            _dbg(f"Allocator N skipped: M failed with {allocator_M_skip_reason}")
        else:
            try:
                # We already have expected_minutes_M and mask_inactive from M computation
                # Get team_ids array
                team_ids_arr = features["team_id"].values

                # Grid search over bench_pool and core_k
                for bp in bench_pool_grid:
                    for ck in core_k_grid:
                        minutes_N, diag_N = allocate_mixture_benchpool(
                            expected_minutes_M,
                            mask_inactive,
                            team_ids_arr,
                            bench_pool=bp,
                            core_k=ck,
                            cap_max=cap_max,
                        )

                        # Build DataFrame for this combo
                        df_N_combo = features.copy()
                        if "minutes" in df_N_combo.columns:
                            df_N_combo = df_N_combo.drop(columns=["minutes"])
                        df_N_combo["minutes_mean_N"] = minutes_N

                        allocator_N_results[(bp, ck)] = (df_N_combo, diag_N)

                        # Track default
                        if bp == default_N_bench_pool and ck == default_N_core_k:
                            df_N = df_N_combo
                            diag_N_default = diag_N

                # If default not in grid, use first available
                if df_N is None and allocator_N_results:
                    first_key = next(iter(allocator_N_results))
                    df_N, diag_N_default = allocator_N_results[first_key]
                    _dbg(f"Default N params not in grid, using {first_key}")

                if df_N is not None:
                    allocator_N_status = "success"
                    _dbg(f"Allocator N completed: {len(allocator_N_results)} combos evaluated")
                else:
                    allocator_N_status = "error"
                    allocator_N_skip_reason = "no_valid_combos"

            except Exception as e:
                import traceback
                allocator_N_status = "error"
                allocator_N_skip_reason = f"allocate_error: {e}"
                _dbg(f"ERROR during Allocator N: {e}")
                _dbg(traceback.format_exc())

        # --- Run Allocator P: PRODUCTION BASELINE ---
        # Uses the production minutes model for comparison
        df_P = None
        allocator_P_status = "skipped"
        allocator_P_skip_reason: str | None = None

        if prod_bundle is None:
            allocator_P_skip_reason = "prod_bundle_not_provided"
            _dbg("Allocator P skipped: prod_bundle is None")
        else:
            try:
                # Score using production model
                # Prepare features the same way production does
                features_for_P = prepare_prod_features(features.copy(), mode="historical")
                
                if features_for_P.empty:
                    allocator_P_status = "error"
                    allocator_P_skip_reason = "prepared_features_empty"
                    _dbg("Allocator P skipped: prepared features empty")
                else:
                    # Score rows using production model
                    scored_P = score_prod_rows(
                        features_for_P,
                        prod_bundle,
                        enable_play_prob_head=True,
                        enable_play_prob_mixing=False,
                    )
                    
                    # Extract minutes_p50 as the canonical prediction
                    if "minutes_p50" in scored_P.columns:
                        df_P = features.copy()
                        if "minutes" in df_P.columns:
                            df_P = df_P.drop(columns=["minutes"])
                        
                        # Match indices properly
                        df_P["minutes_mean_P"] = 0.0
                        for idx in df_P.index:
                            if idx in scored_P.index:
                                df_P.loc[idx, "minutes_mean_P"] = scored_P.loc[idx, "minutes_p50"]
                        
                        allocator_P_status = "success"
                        _dbg(f"Allocator P completed: {len(df_P)} players scored")
                    else:
                        allocator_P_status = "error"
                        allocator_P_skip_reason = "minutes_p50_not_in_output"
                        _dbg("Allocator P error: minutes_p50 not in scored output")

            except Exception as e:
                import traceback
                allocator_P_status = "error"
                allocator_P_skip_reason = f"scoring_error: {e}"
                _dbg(f"ERROR during Allocator P: {e}")
                _dbg(traceback.format_exc())

        # Check sum-to-240 guardrails for all D variants
        for alpha, (df_D_alpha, diag_D_alpha) in allocator_d_results.items():
            if diag_D_alpha.team_sum_dev_max > 1e-6:
                result["status"] = "skipped"
                result["skip_reason"] = f"sum_violation_D_alpha_{alpha}: {diag_D_alpha.team_sum_dev_max:.2f}"
                result["quality_tier"] = "skipped"
                _save_skip_summary()
                return result

        # Check sum-to-240 guardrails for E
        if diag_E.team_sum_dev_max > 1e-6:
            result["status"] = "skipped"
            result["skip_reason"] = f"sum_violation_E: {diag_E.team_sum_dev_max:.2f}"
            result["quality_tier"] = "skipped"
            _save_skip_summary()
            return result

        # Check sum-to-240 guardrails for all F variants
        for p, (df_F_p, diag_F_p) in allocator_f_results.items():
            if diag_F_p.team_sum_dev_max > 1e-6:
                result["status"] = "skipped"
                result["skip_reason"] = f"sum_violation_F_p_{p}: {diag_F_p.team_sum_dev_max:.2f}"
                result["quality_tier"] = "skipped"
                _save_skip_summary()
                return result

        # Check sum-to-240 guardrails
        if diag_A.team_sum_dev_max > 1e-6:
            result["status"] = "skipped"
            result["skip_reason"] = f"sum_violation_A: {diag_A.team_sum_dev_max:.2f}"
            result["quality_tier"] = "skipped"
            _save_skip_summary()
            return result
        if diag_C.team_sum_dev_max > 1e-6:
            result["status"] = "skipped"
            result["skip_reason"] = f"sum_violation_C: {diag_C.team_sum_dev_max:.2f}"
            result["quality_tier"] = "skipped"
            _save_skip_summary()
            return result
        if rotalloc_diag.team_sum_dev_max > 1e-6:
            result["status"] = "skipped"
            result["skip_reason"] = f"sum_violation_B: {rotalloc_diag.team_sum_dev_max:.2f}"
            result["quality_tier"] = "skipped"
            _save_skip_summary()
            return result
        
        # Compute starter realism for A, B, C
        starter_A = _compute_starter_realism_metrics(df_A, "minutes_mean_A")
        starter_B = _compute_starter_realism_metrics(df_B, "minutes_mean")
        starter_C = _compute_starter_realism_metrics(df_C, "minutes_mean_C")

        # Compute starter realism for each D variant
        starter_D_by_alpha: dict[float, dict[str, float]] = {}
        for alpha, (df_D_alpha, _) in allocator_d_results.items():
            starter_D_by_alpha[alpha] = _compute_starter_realism_metrics(df_D_alpha, "minutes_mean_D")

        # Compute starter realism for E
        starter_E = _compute_starter_realism_metrics(df_E, "minutes_mean_E")

        # Compute starter realism for each F variant
        starter_F_by_p: dict[float, dict[str, float]] = {}
        for p, (df_F_p, _) in allocator_f_results.items():
            starter_F_by_p[p] = _compute_starter_realism_metrics(df_F_p, "minutes_mean_F")
        
        # Get default F starter realism
        starter_F = starter_F_by_p.get(default_F_p, {})

        # Load actual minutes
        actual_df = _load_actual_minutes(game_date)
        
        # Compare A vs B
        metrics_AB = compare_allocators(
            df_A, df_B,
            actual_minutes_df=actual_df,
            minutes_col_A="minutes_mean_A",
            minutes_col_B="minutes_mean",
            actual_minutes_col="minutes" if actual_df is not None else "minutes",
        )
        
        # Compare C vs B
        metrics_CB = compare_allocators(
            df_C, df_B,
            actual_minutes_df=actual_df,
            minutes_col_A="minutes_mean_C",
            minutes_col_B="minutes_mean",
            actual_minutes_col="minutes" if actual_df is not None else "minutes",
        )

        # Compare D (each alpha) vs B
        metrics_D_by_alpha: dict[float, AllocatorComparisonMetrics] = {}
        for alpha, (df_D_alpha, _) in allocator_d_results.items():
            metrics_DB = compare_allocators(
                df_D_alpha, df_B,
                actual_minutes_df=actual_df,
                minutes_col_A="minutes_mean_D",
                minutes_col_B="minutes_mean",
                actual_minutes_col="minutes" if actual_df is not None else "minutes",
            )
            metrics_D_by_alpha[alpha] = metrics_DB

        # Compare E vs B
        metrics_EB = compare_allocators(
            df_E, df_B,
            actual_minutes_df=actual_df,
            minutes_col_A="minutes_mean_E",
            minutes_col_B="minutes_mean",
            actual_minutes_col="minutes" if actual_df is not None else "minutes",
        )

        # Compare F vs B for each p value
        metrics_F_by_p: dict[float, AllocatorComparisonMetrics] = {}
        for p, (df_F_p, _) in allocator_f_results.items():
            # Drop minutes_mean from df_F_p to avoid column collision during merge
            # (df_F_p inherits minutes_mean from df_B copy)
            df_F_compare = df_F_p.drop(columns=["minutes_mean"], errors="ignore")
            metrics_FB_p = compare_allocators(
                df_F_compare, df_B,
                actual_minutes_df=actual_df,
                minutes_col_A="minutes_mean_F",
                minutes_col_B="minutes_mean",
                actual_minutes_col="minutes" if actual_df is not None else "minutes",
            )
            metrics_F_by_p[p] = metrics_FB_p

        # Compare M vs B (if M was computed)
        metrics_MB = None
        if df_M is not None:
            metrics_MB = compare_allocators(
                df_M, df_B,
                actual_minutes_df=actual_df,
                minutes_col_A="minutes_mean_M",
                minutes_col_B="minutes_mean",
                actual_minutes_col="minutes" if actual_df is not None else "minutes",
            )

        # Compare N (each combo) vs B - for grid search
        metrics_N_by_combo: dict[tuple[float, int], AllocatorComparisonMetrics] = {}
        metrics_NB = None  # Default combo
        for (bp, ck), (df_N_combo, _) in allocator_N_results.items():
            metrics_NB_combo = compare_allocators(
                df_N_combo, df_B,
                actual_minutes_df=actual_df,
                minutes_col_A="minutes_mean_N",
                minutes_col_B="minutes_mean",
                actual_minutes_col="minutes" if actual_df is not None else "minutes",
            )
            metrics_N_by_combo[(bp, ck)] = metrics_NB_combo
            if bp == default_N_bench_pool and ck == default_N_core_k:
                metrics_NB = metrics_NB_combo

        # If default combo not in results, use first
        if metrics_NB is None and metrics_N_by_combo:
            metrics_NB = next(iter(metrics_N_by_combo.values()))

        # Compare P vs B (if P was computed)
        metrics_PB = None
        if df_P is not None:
            metrics_PB = compare_allocators(
                df_P, df_B,
                actual_minutes_df=actual_df,
                minutes_col_A="minutes_mean_P",
                minutes_col_B="minutes_mean",
                actual_minutes_col="minutes" if actual_df is not None else "minutes",
            )

        # Compute ordering accuracy for [10, 30] range (only if we have labels)
        ordering_10_30 = {}
        if actual_df is not None and len(actual_df) > 0:
            # Need to merge actual minutes with predictions
            merge_cols = ["player_id", "game_id"]
            actual_merge = actual_df[merge_cols + ["minutes"]].copy()

            # A
            df_A_merged = df_A.merge(actual_merge, on=merge_cols, how="inner")
            if len(df_A_merged) > 0 and "team_id" in df_A_merged.columns:
                ordering_10_30["A"] = _compute_ordering_accuracy_10_30(
                    df_A_merged["minutes_mean_A"].values,
                    df_A_merged["minutes"].values,
                    df_A_merged["team_id"].values,
                )

            # B
            df_B_merged = df_B.merge(actual_merge, on=merge_cols, how="inner")
            if len(df_B_merged) > 0 and "team_id" in df_B_merged.columns:
                ordering_10_30["B"] = _compute_ordering_accuracy_10_30(
                    df_B_merged["minutes_mean"].values,
                    df_B_merged["minutes"].values,
                    df_B_merged["team_id"].values,
                )

            # C
            df_C_merged = df_C.merge(actual_merge, on=merge_cols, how="inner")
            if len(df_C_merged) > 0 and "team_id" in df_C_merged.columns:
                ordering_10_30["C"] = _compute_ordering_accuracy_10_30(
                    df_C_merged["minutes_mean_C"].values,
                    df_C_merged["minutes"].values,
                    df_C_merged["team_id"].values,
                )

            # D (default alpha=0.7)
            if 0.7 in allocator_d_results:
                df_D_default, _ = allocator_d_results[0.7]
                df_D_merged = df_D_default.merge(actual_merge, on=merge_cols, how="inner")
                if len(df_D_merged) > 0 and "team_id" in df_D_merged.columns:
                    ordering_10_30["D"] = _compute_ordering_accuracy_10_30(
                        df_D_merged["minutes_mean_D"].values,
                        df_D_merged["minutes"].values,
                        df_D_merged["team_id"].values,
                    )

            # E
            df_E_merged = df_E.merge(actual_merge, on=merge_cols, how="inner")
            if len(df_E_merged) > 0 and "team_id" in df_E_merged.columns:
                ordering_10_30["E"] = _compute_ordering_accuracy_10_30(
                    df_E_merged["minutes_mean_E"].values,
                    df_E_merged["minutes"].values,
                    df_E_merged["team_id"].values,
                )

            # F (default p)
            if default_F_p in allocator_f_results:
                df_F_default, _ = allocator_f_results[default_F_p]
                df_F_merged = df_F_default.merge(actual_merge, on=merge_cols, how="inner")
                if len(df_F_merged) > 0 and "team_id" in df_F_merged.columns:
                    ordering_10_30["F"] = _compute_ordering_accuracy_10_30(
                        df_F_merged["minutes_mean_F"].values,
                        df_F_merged["minutes"].values,
                        df_F_merged["team_id"].values,
                    )

            # M
            if df_M is not None:
                df_M_merged = df_M.merge(actual_merge, on=merge_cols, how="inner")
                if len(df_M_merged) > 0 and "team_id" in df_M_merged.columns:
                    ordering_10_30["M"] = _compute_ordering_accuracy_10_30(
                        df_M_merged["minutes_mean_M"].values,
                        df_M_merged["minutes"].values,
                        df_M_merged["team_id"].values,
                    )

            # N (default combo)
            if df_N is not None:
                df_N_merged = df_N.merge(actual_merge, on=merge_cols, how="inner")
                if len(df_N_merged) > 0 and "team_id" in df_N_merged.columns:
                    ordering_10_30["N"] = _compute_ordering_accuracy_10_30(
                        df_N_merged["minutes_mean_N"].values,
                        df_N_merged["minutes"].values,
                        df_N_merged["team_id"].values,
                    )

            # P (production baseline)
            if df_P is not None:
                df_P_merged = df_P.merge(actual_merge, on=merge_cols, how="inner")
                if len(df_P_merged) > 0 and "team_id" in df_P_merged.columns:
                    ordering_10_30["P"] = _compute_ordering_accuracy_10_30(
                        df_P_merged["minutes_mean_P"].values,
                        df_P_merged["minutes"].values,
                        df_P_merged["team_id"].values,
                    )

        # Build combined metrics dict
        metrics_dict = metrics_AB.to_dict()
        # Rename A metrics
        metrics_dict["mae_A"] = metrics_AB.mae_A
        metrics_dict["rmse_A"] = metrics_AB.rmse_A
        metrics_dict["mae_by_bucket_A"] = metrics_AB.mae_by_bucket_A
        metrics_dict["team_sum_error_max_A"] = metrics_AB.team_sum_error_max_A
        metrics_dict["team_sum_error_mean_A"] = metrics_AB.team_sum_error_mean_A
        metrics_dict["gini_mean_A"] = metrics_AB.gini_mean_A
        metrics_dict["roster_size_mean_A"] = metrics_AB.roster_size_mean_A
        metrics_dict["bench_max_mean_A"] = metrics_AB.bench_max_mean_A
        metrics_dict["hhi_mean_A"] = metrics_AB.hhi_mean_A
        metrics_dict["top6_share_mean_A"] = metrics_AB.top6_share_mean_A
        metrics_dict["top8_share_mean_A"] = metrics_AB.top8_share_mean_A
        metrics_dict["sixth_man_mae_A"] = metrics_AB.sixth_man_mae_A
        metrics_dict["sixth_man_mae_B"] = metrics_AB.sixth_man_mae_B
        
        # Add C metrics
        metrics_dict["mae_C"] = metrics_CB.mae_A
        metrics_dict["rmse_C"] = metrics_CB.rmse_A
        metrics_dict["mae_by_bucket_C"] = metrics_CB.mae_by_bucket_A
        metrics_dict["team_sum_error_max_C"] = metrics_CB.team_sum_error_max_A
        metrics_dict["team_sum_error_mean_C"] = metrics_CB.team_sum_error_mean_A
        metrics_dict["gini_mean_C"] = metrics_CB.gini_mean_A
        metrics_dict["roster_size_mean_C"] = metrics_CB.roster_size_mean_A
        metrics_dict["bench_max_mean_C"] = metrics_CB.bench_max_mean_A
        metrics_dict["hhi_mean_C"] = metrics_CB.hhi_mean_A
        metrics_dict["top6_share_mean_C"] = metrics_CB.top6_share_mean_A
        metrics_dict["top8_share_mean_C"] = metrics_CB.top8_share_mean_A
        metrics_dict["sixth_man_mae_C"] = metrics_CB.sixth_man_mae_A

        # Add D metrics for each alpha
        metrics_dict["allocator_D"] = {}
        for alpha, metrics_DB in metrics_D_by_alpha.items():
            alpha_key = f"alpha_{alpha:.1f}"
            metrics_dict["allocator_D"][alpha_key] = {
                "alpha": alpha,
                "mae": metrics_DB.mae_A,
                "rmse": metrics_DB.rmse_A,
                "mae_by_bucket": metrics_DB.mae_by_bucket_A,
                "team_sum_error_max": metrics_DB.team_sum_error_max_A,
                "team_sum_error_mean": metrics_DB.team_sum_error_mean_A,
                "gini_mean": metrics_DB.gini_mean_A,
                "hhi_mean": metrics_DB.hhi_mean_A,
                "top6_share_mean": metrics_DB.top6_share_mean_A,
                "top8_share_mean": metrics_DB.top8_share_mean_A,
                "roster_size_mean": metrics_DB.roster_size_mean_A,
                "bench_max_mean": metrics_DB.bench_max_mean_A,
                "sixth_man_mae": metrics_DB.sixth_man_mae_A,
            }

        # Find best D variant by MAE (lowest wins)
        best_alpha = None
        best_mae = float("inf")
        for alpha, metrics_DB in metrics_D_by_alpha.items():
            if metrics_DB.mae_A is not None and metrics_DB.mae_A < best_mae:
                best_mae = metrics_DB.mae_A
                best_alpha = alpha
        metrics_dict["best_D_alpha"] = best_alpha
        metrics_dict["mae_D_best"] = best_mae if best_alpha is not None else None

        # Also add D with default alpha=0.7 for backward compatibility
        if 0.7 in metrics_D_by_alpha:
            metrics_dict["mae_D"] = metrics_D_by_alpha[0.7].mae_A
            metrics_dict["rmse_D"] = metrics_D_by_alpha[0.7].rmse_A
            metrics_dict["mae_by_bucket_D"] = metrics_D_by_alpha[0.7].mae_by_bucket_A
            metrics_dict["gini_mean_D"] = metrics_D_by_alpha[0.7].gini_mean_A
            metrics_dict["hhi_mean_D"] = metrics_D_by_alpha[0.7].hhi_mean_A
            metrics_dict["top6_share_mean_D"] = metrics_D_by_alpha[0.7].top6_share_mean_A
            metrics_dict["top8_share_mean_D"] = metrics_D_by_alpha[0.7].top8_share_mean_A
            metrics_dict["roster_size_mean_D"] = metrics_D_by_alpha[0.7].roster_size_mean_A
            metrics_dict["bench_max_mean_D"] = metrics_D_by_alpha[0.7].bench_max_mean_A
            metrics_dict["sixth_man_mae_D"] = metrics_D_by_alpha[0.7].sixth_man_mae_A

        # Add E metrics
        metrics_dict["mae_E"] = metrics_EB.mae_A
        metrics_dict["rmse_E"] = metrics_EB.rmse_A
        metrics_dict["mae_by_bucket_E"] = metrics_EB.mae_by_bucket_A
        metrics_dict["team_sum_error_max_E"] = metrics_EB.team_sum_error_max_A
        metrics_dict["team_sum_error_mean_E"] = metrics_EB.team_sum_error_mean_A
        metrics_dict["gini_mean_E"] = metrics_EB.gini_mean_A
        metrics_dict["hhi_mean_E"] = metrics_EB.hhi_mean_A
        metrics_dict["top6_share_mean_E"] = metrics_EB.top6_share_mean_A
        metrics_dict["top8_share_mean_E"] = metrics_EB.top8_share_mean_A
        metrics_dict["roster_size_mean_E"] = metrics_EB.roster_size_mean_A
        metrics_dict["bench_max_mean_E"] = metrics_EB.bench_max_mean_A
        metrics_dict["sixth_man_mae_E"] = metrics_EB.sixth_man_mae_A

        # Add F metrics for each p value
        metrics_dict["allocator_F"] = {}
        for p, metrics_FB in metrics_F_by_p.items():
            p_key = f"p_{p:.1f}"
            metrics_dict["allocator_F"][p_key] = {
                "p": p,
                "mae": metrics_FB.mae_A,
                "rmse": metrics_FB.rmse_A,
                "mae_by_bucket": metrics_FB.mae_by_bucket_A,
                "team_sum_error_max": metrics_FB.team_sum_error_max_A,
                "team_sum_error_mean": metrics_FB.team_sum_error_mean_A,
                "gini_mean": metrics_FB.gini_mean_A,
                "hhi_mean": metrics_FB.hhi_mean_A,
                "top6_share_mean": metrics_FB.top6_share_mean_A,
                "top8_share_mean": metrics_FB.top8_share_mean_A,
                "roster_size_mean": metrics_FB.roster_size_mean_A,
                "bench_max_mean": metrics_FB.bench_max_mean_A,
                "sixth_man_mae": metrics_FB.sixth_man_mae_A,
            }

        # Find best F variant by MAE (lowest wins)
        best_F_p = None
        best_F_mae = float("inf")
        for p, metrics_FB in metrics_F_by_p.items():
            if metrics_FB.mae_A is not None and metrics_FB.mae_A < best_F_mae:
                best_F_mae = metrics_FB.mae_A
                best_F_p = p
        metrics_dict["best_F_p"] = best_F_p
        metrics_dict["mae_F_best"] = best_F_mae if best_F_p is not None else None

        # Add F with default p for main metrics
        if default_F_p in metrics_F_by_p:
            metrics_FB_default = metrics_F_by_p[default_F_p]
            metrics_dict["mae_F"] = metrics_FB_default.mae_A
            metrics_dict["rmse_F"] = metrics_FB_default.rmse_A
            metrics_dict["mae_by_bucket_F"] = metrics_FB_default.mae_by_bucket_A
            metrics_dict["gini_mean_F"] = metrics_FB_default.gini_mean_A
            metrics_dict["hhi_mean_F"] = metrics_FB_default.hhi_mean_A
            metrics_dict["top6_share_mean_F"] = metrics_FB_default.top6_share_mean_A
            metrics_dict["top8_share_mean_F"] = metrics_FB_default.top8_share_mean_A
            metrics_dict["roster_size_mean_F"] = metrics_FB_default.roster_size_mean_A
            metrics_dict["bench_max_mean_F"] = metrics_FB_default.bench_max_mean_A
            metrics_dict["sixth_man_mae_F"] = metrics_FB_default.sixth_man_mae_A
        metrics_dict["default_F_p"] = default_F_p

        # Add M metrics (if M was computed)
        if metrics_MB is not None:
            metrics_dict["mae_M"] = metrics_MB.mae_A
            metrics_dict["rmse_M"] = metrics_MB.rmse_A
            metrics_dict["mae_by_bucket_M"] = metrics_MB.mae_by_bucket_A
            metrics_dict["team_sum_error_max_M"] = metrics_MB.team_sum_error_max_A
            metrics_dict["team_sum_error_mean_M"] = metrics_MB.team_sum_error_mean_A
            metrics_dict["gini_mean_M"] = metrics_MB.gini_mean_A
            metrics_dict["hhi_mean_M"] = metrics_MB.hhi_mean_A
            metrics_dict["top6_share_mean_M"] = metrics_MB.top6_share_mean_A
            metrics_dict["top8_share_mean_M"] = metrics_MB.top8_share_mean_A
            metrics_dict["roster_size_mean_M"] = metrics_MB.roster_size_mean_A
            metrics_dict["bench_max_mean_M"] = metrics_MB.bench_max_mean_A
            metrics_dict["sixth_man_mae_M"] = metrics_MB.sixth_man_mae_A

        # Add N metrics for each (bench_pool, core_k) combo
        metrics_dict["allocator_N"] = {}
        for (bp, ck), metrics_N_combo in metrics_N_by_combo.items():
            combo_key = f"bp_{bp:.0f}_ck_{ck}"
            metrics_dict["allocator_N"][combo_key] = {
                "bench_pool": bp,
                "core_k": ck,
                "mae": metrics_N_combo.mae_A,
                "rmse": metrics_N_combo.rmse_A,
                "mae_by_bucket": metrics_N_combo.mae_by_bucket_A,
                "team_sum_error_max": metrics_N_combo.team_sum_error_max_A,
                "team_sum_error_mean": metrics_N_combo.team_sum_error_mean_A,
                "gini_mean": metrics_N_combo.gini_mean_A,
                "hhi_mean": metrics_N_combo.hhi_mean_A,
                "top6_share_mean": metrics_N_combo.top6_share_mean_A,
                "top8_share_mean": metrics_N_combo.top8_share_mean_A,
                "roster_size_mean": metrics_N_combo.roster_size_mean_A,
                "bench_max_mean": metrics_N_combo.bench_max_mean_A,
                "sixth_man_mae": metrics_N_combo.sixth_man_mae_A,
            }

        # Find best N variant by MAE (lowest wins)
        best_N_params = None
        best_N_mae = float("inf")
        for (bp, ck), metrics_N_combo in metrics_N_by_combo.items():
            if metrics_N_combo.mae_A is not None and metrics_N_combo.mae_A < best_N_mae:
                best_N_mae = metrics_N_combo.mae_A
                best_N_params = (bp, ck)
        if best_N_params is not None:
            metrics_dict["best_N_bench_pool"] = best_N_params[0]
            metrics_dict["best_N_core_k"] = best_N_params[1]
            metrics_dict["mae_N_best"] = best_N_mae
        else:
            metrics_dict["best_N_bench_pool"] = None
            metrics_dict["best_N_core_k"] = None
            metrics_dict["mae_N_best"] = None

        # Add N with default params for main metrics
        if metrics_NB is not None:
            metrics_dict["mae_N"] = metrics_NB.mae_A
            metrics_dict["rmse_N"] = metrics_NB.rmse_A
            metrics_dict["mae_by_bucket_N"] = metrics_NB.mae_by_bucket_A
            metrics_dict["team_sum_error_max_N"] = metrics_NB.team_sum_error_max_A
            metrics_dict["team_sum_error_mean_N"] = metrics_NB.team_sum_error_mean_A
            metrics_dict["gini_mean_N"] = metrics_NB.gini_mean_A
            metrics_dict["hhi_mean_N"] = metrics_NB.hhi_mean_A
            metrics_dict["top6_share_mean_N"] = metrics_NB.top6_share_mean_A
            metrics_dict["top8_share_mean_N"] = metrics_NB.top8_share_mean_A
            metrics_dict["roster_size_mean_N"] = metrics_NB.roster_size_mean_A
            metrics_dict["bench_max_mean_N"] = metrics_NB.bench_max_mean_A
            metrics_dict["sixth_man_mae_N"] = metrics_NB.sixth_man_mae_A
        metrics_dict["default_N_bench_pool"] = default_N_bench_pool
        metrics_dict["default_N_core_k"] = default_N_core_k

        # Add P metrics (if P was computed)
        if metrics_PB is not None:
            metrics_dict["mae_P"] = metrics_PB.mae_A
            metrics_dict["rmse_P"] = metrics_PB.rmse_A
            metrics_dict["mae_by_bucket_P"] = metrics_PB.mae_by_bucket_A
            metrics_dict["team_sum_error_max_P"] = metrics_PB.team_sum_error_max_A
            metrics_dict["team_sum_error_mean_P"] = metrics_PB.team_sum_error_mean_A
            metrics_dict["gini_mean_P"] = metrics_PB.gini_mean_A
            metrics_dict["hhi_mean_P"] = metrics_PB.hhi_mean_A
            metrics_dict["top6_share_mean_P"] = metrics_PB.top6_share_mean_A
            metrics_dict["top8_share_mean_P"] = metrics_PB.top8_share_mean_A
            metrics_dict["roster_size_mean_P"] = metrics_PB.roster_size_mean_A
            metrics_dict["bench_max_mean_P"] = metrics_PB.bench_max_mean_A
            metrics_dict["sixth_man_mae_P"] = metrics_PB.sixth_man_mae_A

        # Add ordering_10_30 metrics
        metrics_dict["ordering_10_30"] = ordering_10_30

        # Compute starter realism for M
        starter_M = None
        if df_M is not None:
            starter_M = _compute_starter_realism_metrics(df_M, "minutes_mean_M")

        # Compute starter realism for N (default combo)
        starter_N = None
        if df_N is not None:
            starter_N = _compute_starter_realism_metrics(df_N, "minutes_mean_N")

        # Compute starter realism for P (production baseline)
        starter_P = None
        if df_P is not None:
            starter_P = _compute_starter_realism_metrics(df_P, "minutes_mean_P")

        # Save outputs
        summary = {
            "game_date": game_date,
            "timestamp": datetime.now().isoformat(),
            "quality_tier": result["quality_tier"],
            "missing_feature_frac": result["missing_feature_frac"],
            "n_expected_features": result["n_expected_features"],
            "n_missing_features": result["n_missing_features"],
            "integrity_counts": integrity_counts,
            "config": {
                "bundle_dir": str(bundle_dir),
                "features_path": str(features_path),
                "shares_col": shares_col,
                "cap_max": cap_max,
                "k_core": k_core,
                "alpha_core": alpha_core,
                "alpha_fringe": alpha_fringe,
                "use_share_model": use_share_model,
                "alpha_grid": alpha_grid,
                "p_grid": p_grid,
                "default_F_p": default_F_p,
                "bench_pool_grid": bench_pool_grid,
                "core_k_grid": core_k_grid,
                "default_N_bench_pool": default_N_bench_pool,
                "default_N_core_k": default_N_core_k,
            },
            "metrics": metrics_dict,
            "starter_realism_A": starter_A,
            "starter_realism_B": starter_B,
            "starter_realism_C": starter_C,
            "starter_realism_D": starter_D_by_alpha,
            "starter_realism_E": starter_E,
            "starter_realism_F": starter_F_by_p,
            "starter_realism_M": starter_M,
            "starter_realism_N": starter_N,
            "starter_realism_P": starter_P,
            "allocator_M_status": allocator_M_status,
            "allocator_M_skip_reason": allocator_M_skip_reason,
            "allocator_N_status": allocator_N_status,
            "allocator_N_skip_reason": allocator_N_skip_reason,
            "allocator_P_status": allocator_P_status,
            "allocator_P_skip_reason": allocator_P_skip_reason,
            "prod_bundle_dir": str(prod_bundle_dir) if prod_bundle_dir else None,
            "prod_bundle_name": prod_bundle_dir.name if prod_bundle_dir else None,
            "diagnostics_A": diag_A.to_dict(),
            "diagnostics_C": diag_C.to_dict(),
            "diagnostics_B": {
                "eligible_size_p50": rotalloc_diag.eligible_size_p50,
                "eligible_size_p90": rotalloc_diag.eligible_size_p90,
                "fallback_top1_used": rotalloc_diag.fallback_top1_used,
                "cutoff_empty_events": rotalloc_diag.cutoff_empty_events,
                "team_sum_dev_max": rotalloc_diag.team_sum_dev_max,
            },
            "diagnostics_D": {
                alpha: diag.to_dict()
                for alpha, (_, diag) in allocator_d_results.items()
            },
            "diagnostics_E": diag_E.to_dict(),
            "diagnostics_F": {
                p: diag.to_dict()
                for p, (_, diag) in allocator_f_results.items()
            },
            "diagnostics_N": {
                f"bp_{bp:.0f}_ck_{ck}": diag.to_dict()
                for (bp, ck), (_, diag) in allocator_N_results.items()
            },
        }
        (slate_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        
        # Save player reports (optional, can be large)
        try:
            players_report = build_players_report(
                df_A, df_B, features,
                minutes_col_A="minutes_mean_A",
                minutes_col_B="minutes_mean",
            )
            players_report.to_parquet(slate_dir / "players.parquet", index=False)
        except Exception:
            pass
        
        result["status"] = "success"
        result["n_players"] = metrics_dict.get("n_players", len(features))
        result["n_teams"] = metrics_dict.get("n_teams", integrity_counts.get("n_teams", 0))
        result["has_labels"] = actual_df is not None and len(actual_df) > 0
        result["mae_A"] = metrics_dict.get("mae_A")
        result["mae_B"] = metrics_dict.get("mae_B")
        result["mae_C"] = metrics_dict.get("mae_C")
        result["mae_D"] = metrics_dict.get("mae_D")
        result["mae_D_best"] = metrics_dict.get("mae_D_best")
        result["best_D_alpha"] = metrics_dict.get("best_D_alpha")
        result["mae_E"] = metrics_dict.get("mae_E")
        result["mae_F"] = metrics_dict.get("mae_F")
        result["mae_F_best"] = metrics_dict.get("mae_F_best")
        result["best_F_p"] = metrics_dict.get("best_F_p")
        result["mae_M"] = metrics_dict.get("mae_M")
        result["mae_N"] = metrics_dict.get("mae_N")
        result["mae_N_best"] = metrics_dict.get("mae_N_best")
        result["best_N_bench_pool"] = metrics_dict.get("best_N_bench_pool")
        result["best_N_core_k"] = metrics_dict.get("best_N_core_k")
        result["mae_P"] = metrics_dict.get("mae_P")
        result["ordering_10_30"] = ordering_10_30

        return result
        
    except Exception as e:
        import traceback
        result["status"] = "error"
        result["skip_reason"] = str(e)
        result["quality_tier"] = "skipped"
        result["traceback"] = traceback.format_exc()
        # Try to save skip summary if slate_dir exists
        try:
            slate_dir = out_dir / game_date
            slate_dir.mkdir(parents=True, exist_ok=True)
            summary = {
                "game_date": game_date,
                "timestamp": datetime.now().isoformat(),
                "quality_tier": "skipped",
                "skip_reason": str(e),
                "traceback": traceback.format_exc(),
                "metrics": {},
            }
            (slate_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        except Exception:
            pass
        return result


@app.command("single")
def run_single(
    game_date: str = typer.Option(
        ...,
        "--game-date",
        help="Target slate date (YYYY-MM-DD format)",
    ),
    bundle_dir: Path = typer.Option(
        None,
        "--bundle-dir",
        help="RotAlloc bundle directory",
    ),
    out_dir: Path = typer.Option(
        None,
        "--out-dir",
        help="Output directory",
    ),
    cap_max: float = typer.Option(48.0, "--cap-max"),
    k_core: int = typer.Option(8, "--k-core"),
    use_share_model: bool = typer.Option(True, "--use-share-model/--no-share-model"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    mixture_bundle_path: Path | None = typer.Option(
        None,
        "--mixture-bundle",
        help="Path to trained mixture model bundle for Allocator M. If not provided, M is skipped.",
    ),
    debug_mixture: bool = typer.Option(
        False,
        "--debug-mixture",
        help="Emit detailed debug logs for Allocator M",
    ),
) -> None:
    """Run A/B/C/D/E/F/M test for a single slate."""
    allocator_list = "A/B/C/D/E/F" + ("/M" if mixture_bundle_path else "")
    typer.echo(f"[single] Starting {allocator_list} test for {game_date}", err=True)

    # Resolve bundle
    if bundle_dir is None:
        bundle_dir = resolve_rotalloc_bundle_dir(DEFAULT_CONFIG_PATH)
        if bundle_dir is None:
            raise typer.BadParameter("RotAlloc bundle not found")
    bundle_dir = bundle_dir.expanduser().resolve()

    # Load share model
    share_model, expected_cols, model_path = _load_share_model_with_columns()
    if share_model:
        typer.echo(f"[share-model] Loaded from {model_path}", err=True)

    # Load mixture bundle for Allocator M (optional)
    mixture_bundle = None
    if mixture_bundle_path is not None:
        mixture_bundle_path = mixture_bundle_path.expanduser().resolve()
        if mixture_bundle_path.exists():
            mixture_bundle = MixtureBundle.load(mixture_bundle_path)
            typer.echo(f"[mixture-bundle] Loaded from {mixture_bundle_path}", err=True)
        else:
            typer.echo(f"[mixture-bundle] WARNING: Path not found: {mixture_bundle_path}", err=True)
    else:
        typer.echo("[mixture-bundle] Not provided, Allocator M skipped", err=True)

    # Output dir
    if out_dir is None:
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        out_dir = DEFAULT_OUT_ROOT / f"{timestamp}_{game_date}"
    out_dir.mkdir(parents=True, exist_ok=True)

    result = _run_single_slate(
        game_date,
        bundle_dir,
        out_dir,
        cap_max,
        k_core,
        use_share_model,
        share_model,
        expected_cols,
        mixture_bundle=mixture_bundle,
        debug_mixture=debug_mixture,
    )

    # Print result
    print(f"\n{'='*70}")
    print(f"SLATE: {game_date}")
    print(f"Status: {result['status']}")
    print(f"Quality: {result['quality_tier']}")
    if result.get("skip_reason"):
        print(f"Skip reason: {result['skip_reason']}")
    if result.get("mae_A"):
        mae_d = result.get("mae_D")
        mae_d_str = f", D: {mae_d:.2f}" if mae_d else ""
        mae_m = result.get("mae_M")
        mae_m_str = f", M: {mae_m:.2f}" if mae_m else ""
        print(f"\nMAE - A: {result['mae_A']:.2f}, B: {result['mae_B']:.2f}, C: {result['mae_C']:.2f}{mae_d_str}{mae_m_str}")
        if result.get("best_D_alpha") is not None:
            print(f"Best D: alpha={result['best_D_alpha']:.1f}, MAE={result['mae_D_best']:.2f}")
    print(f"{'='*70}\n")


@app.command("multi")
def run_multi(
    start_date: str = typer.Option(..., "--start-date", help="Start date (YYYY-MM-DD)"),
    end_date: str = typer.Option(..., "--end-date", help="End date (YYYY-MM-DD)"),
    bundle_dir: Path = typer.Option(None, "--bundle-dir"),
    out_dir: Path = typer.Option(None, "--out-dir"),
    features_root: Path = typer.Option(
        None,
        "--features-root",
        help="Explicit features root directory. If provided, load from <features_root>/<date>/run=*/features.parquet",
    ),
    cap_max: float = typer.Option(48.0, "--cap-max"),
    k_core: int = typer.Option(8, "--k-core"),
    alpha_core: float = typer.Option(0.8, "--alpha-core", help="Allocator E: blend weight for core players"),
    alpha_fringe: float = typer.Option(0.3, "--alpha-fringe", help="Allocator E: blend weight for fringe players"),
    use_share_model: bool = typer.Option(True, "--use-share-model/--no-share-model"),
    parallel: int = typer.Option(1, "--parallel", help="Number of parallel workers"),
    alpha_grid: str = typer.Option(
        "0.0,0.3,0.5,0.7,0.9,1.0",
        "--alpha-grid",
        help="Comma-separated alpha values for Allocator D grid search",
    ),
    p_grid: str = typer.Option(
        "1.0,1.1,1.2,1.3",
        "--p-grid",
        help="Comma-separated p exponents for Allocator F power postprocess",
    ),
    mixture_bundle_path: Path | None = typer.Option(
        None,
        "--mixture-bundle",
        help="Path to trained mixture model bundle for Allocator M. If not provided, M is skipped.",
    ),
    debug_mixture: bool = typer.Option(
        False,
        "--debug-mixture",
        help="Emit detailed debug logs for Allocator M",
    ),
    bench_pool_grid: str = typer.Option(
        "70,80,90",
        "--bench-pool-grid",
        help="Comma-separated bench_pool values for Allocator N grid search",
    ),
    core_k_grid: str = typer.Option(
        "5,6",
        "--core-k-grid",
        help="Comma-separated core_k values for Allocator N grid search",
    ),
    prod_minutes_bundle: Path | None = typer.Option(
        None,
        "--prod-minutes-bundle",
        help=f"Path to production minutes bundle for Allocator P baseline. Default: {DEFAULT_PROD_MINUTES_BUNDLE}",
    ),
    include_prod_baseline: bool = typer.Option(
        True,
        "--include-prod-baseline/--no-prod-baseline",
        help="Include production baseline (P) in comparison",
    ),
) -> None:
    """Run A/B/C/D/E/F/M test across multiple slates and aggregate results."""
    allocator_list = "A/B/C/D/E/F" + ("/M" if mixture_bundle_path else "")
    typer.echo(f"[multi] Starting {allocator_list} test: {start_date} to {end_date}", err=True)

    # Parse alpha grid
    alpha_values = [float(a.strip()) for a in alpha_grid.split(",")]
    typer.echo(f"[multi] Alpha grid for D: {alpha_values}", err=True)
    
    # Parse p grid
    p_values = [float(p.strip()) for p in p_grid.split(",")]
    typer.echo(f"[multi] P grid for F: {p_values}", err=True)
    
    # Parse bench_pool and core_k grids for Allocator N
    bench_pool_values = [float(bp.strip()) for bp in bench_pool_grid.split(",")]
    core_k_values = [int(ck.strip()) for ck in core_k_grid.split(",")]
    typer.echo(f"[multi] Bench pool grid for N: {bench_pool_values}", err=True)
    typer.echo(f"[multi] Core K grid for N: {core_k_values}", err=True)
    
    # Resolve bundle
    if bundle_dir is None:
        bundle_dir = resolve_rotalloc_bundle_dir(DEFAULT_CONFIG_PATH)
        if bundle_dir is None:
            raise typer.BadParameter("RotAlloc bundle not found")
    bundle_dir = bundle_dir.expanduser().resolve()
    typer.echo(f"[multi] RotAlloc bundle: {bundle_dir}", err=True)
    
    if features_root:
        features_root = features_root.expanduser().resolve()
        typer.echo(f"[multi] Features root: {features_root}", err=True)
    
    # Load share model once
    share_model, expected_cols, model_path = _load_share_model_with_columns()
    if share_model:
        typer.echo(f"[share-model] Loaded from {model_path} ({len(expected_cols)} features)", err=True)
    else:
        typer.echo("[share-model] Not found, using proxy columns", err=True)

    # Load mixture bundle for Allocator M (optional)
    mixture_bundle = None
    if mixture_bundle_path is not None:
        mixture_bundle_path = mixture_bundle_path.expanduser().resolve()
        if mixture_bundle_path.exists():
            mixture_bundle = MixtureBundle.load(mixture_bundle_path)
            typer.echo(f"[mixture-bundle] Loaded from {mixture_bundle_path}", err=True)
        else:
            typer.echo(f"[mixture-bundle] WARNING: Path not found: {mixture_bundle_path}", err=True)
    else:
        typer.echo("[mixture-bundle] Not provided, Allocator M skipped", err=True)

    # Load production minutes bundle for Allocator P (optional)
    prod_bundle = None
    prod_bundle_dir_resolved = None
    if include_prod_baseline:
        # Use provided path or default
        if prod_minutes_bundle is not None:
            prod_bundle_dir_resolved = prod_minutes_bundle.expanduser().resolve()
        else:
            prod_bundle_dir_resolved = DEFAULT_PROD_MINUTES_BUNDLE.expanduser().resolve()
        
        if prod_bundle_dir_resolved.exists():
            try:
                prod_bundle = load_prod_bundle(prod_bundle_dir_resolved)
                typer.echo(f"[prod-bundle] Loaded from {prod_bundle_dir_resolved}", err=True)
            except Exception as e:
                typer.echo(f"[prod-bundle] WARNING: Failed to load: {e}", err=True)
                prod_bundle = None
        else:
            typer.echo(f"[prod-bundle] WARNING: Path not found: {prod_bundle_dir_resolved}", err=True)
    else:
        typer.echo("[prod-bundle] Production baseline disabled", err=True)

    # Output directory
    if out_dir is None:
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        out_dir = Path(f"runs/abtest_minutes_alloc_multi/{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    typer.echo(f"[multi] Output: {out_dir}", err=True)
    
    # Generate date range
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    
    typer.echo(f"[multi] Processing {len(dates)} dates", err=True)
    
    # Run all slates
    results = []
    counts = {"success": 0, "clean": 0, "degraded": 0, "skipped": 0}
    skip_reasons: dict[str, int] = {}
    
    for date in dates:
        result = _run_single_slate(
            date,
            bundle_dir,
            out_dir,
            cap_max,
            k_core,
            use_share_model,
            share_model,
            expected_cols,
            features_root=features_root,
            alpha_grid=alpha_values,
            alpha_core=alpha_core,
            alpha_fringe=alpha_fringe,
            p_grid=p_values,
            mixture_bundle=mixture_bundle,
            debug_mixture=debug_mixture,
            bench_pool_grid=bench_pool_values,
            core_k_grid=core_k_values,
            prod_bundle=prod_bundle,
            prod_bundle_dir=prod_bundle_dir_resolved,
        )
        results.append(result)
        
        status = result["status"]
        tier = result["quality_tier"]
        
        if status == "success":
            counts["success"] += 1
            if tier == "clean":
                counts["clean"] += 1
                typer.echo(f"  ✓ {date} [clean]", err=True)
            else:
                counts["degraded"] += 1
                typer.echo(f"  ~ {date} [degraded]", err=True)
        else:
            counts["skipped"] += 1
            reason = result.get("skip_reason", "unknown")
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            typer.echo(f"  ⊘ {date} ({reason})", err=True)
    
    # Aggregate
    typer.echo("[multi] Aggregating results...", err=True)
    agg_all, agg_clean, slate_df, bucket_df, skips_df = run_aggregation(out_dir)
    save_aggregates(agg_all, agg_clean, slate_df, bucket_df, skips_df, out_dir)
    
    # Print summary
    print("\n" + "=" * 70)
    print("MULTI-SLATE A/B/C/D/E/F TEST SUMMARY")
    print("=" * 70)
    
    print(f"\n📊 Coverage:")
    print(f"  Processed: {counts['success']} (clean: {counts['clean']}, degraded: {counts['degraded']})")
    print(f"  Skipped: {counts['skipped']}")
    
    if skip_reasons:
        print("\n  Skip reasons:")
        for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1])[:5]:
            print(f"    - {reason}: {count}")
    
    print(f"\n📈 Accuracy (clean slates, mean MAE):")
    print(f"  A: {agg_clean.mae_A_mean:.2f}" if not np.isnan(agg_clean.mae_A_mean) else "  A: N/A")
    print(f"  B: {agg_clean.mae_B_mean:.2f}" if not np.isnan(agg_clean.mae_B_mean) else "  B: N/A")
    print(f"  C: {agg_clean.mae_C_mean:.2f}" if not np.isnan(agg_clean.mae_C_mean) else "  C: N/A")
    print(f"  D: {agg_clean.mae_D_mean:.2f}" if not np.isnan(agg_clean.mae_D_mean) else "  D: N/A")
    if hasattr(agg_clean, 'mae_E_mean') and not np.isnan(agg_clean.mae_E_mean):
        print(f"  E: {agg_clean.mae_E_mean:.2f}")
    if hasattr(agg_clean, 'mae_F_mean') and not np.isnan(agg_clean.mae_F_mean):
        print(f"  F: {agg_clean.mae_F_mean:.2f}")
    
    print(f"\n⭐ Starter Realism (clean slates, top5 sum):")
    print(f"  A: {agg_clean.top5_sum_A_mean:.1f}" if not np.isnan(agg_clean.top5_sum_A_mean) else "  A: N/A")
    print(f"  B: {agg_clean.top5_sum_B_mean:.1f}" if not np.isnan(agg_clean.top5_sum_B_mean) else "  B: N/A")
    print(f"  C: {agg_clean.top5_sum_C_mean:.1f}" if not np.isnan(agg_clean.top5_sum_C_mean) else "  C: N/A")
    print(f"  D: {agg_clean.top5_sum_D_mean:.1f}" if not np.isnan(agg_clean.top5_sum_D_mean) else "  D: N/A")
    if hasattr(agg_clean, 'top5_sum_E_mean') and not np.isnan(agg_clean.top5_sum_E_mean):
        print(f"  E: {agg_clean.top5_sum_E_mean:.1f}")
    if hasattr(agg_clean, 'top5_sum_F_mean') and not np.isnan(agg_clean.top5_sum_F_mean):
        print(f"  F: {agg_clean.top5_sum_F_mean:.1f}")
    
    print(f"\n🔄 Win Rates (vs B, clean slates):")
    if not np.isnan(agg_clean.pct_slates_A_wins_mae):
        print(f"  A wins MAE: {agg_clean.pct_slates_A_wins_mae * 100:.1f}%")
    if not np.isnan(agg_clean.pct_slates_C_wins_mae):
        print(f"  C wins MAE: {agg_clean.pct_slates_C_wins_mae * 100:.1f}%")
    if not np.isnan(agg_clean.pct_slates_D_wins_mae):
        print(f"  D wins MAE: {agg_clean.pct_slates_D_wins_mae * 100:.1f}%")
    if hasattr(agg_clean, 'pct_slates_E_wins_mae') and not np.isnan(agg_clean.pct_slates_E_wins_mae):
        print(f"  E wins MAE: {agg_clean.pct_slates_E_wins_mae * 100:.1f}%")
    if hasattr(agg_clean, 'pct_slates_F_wins_mae') and not np.isnan(agg_clean.pct_slates_F_wins_mae):
        print(f"  F wins MAE: {agg_clean.pct_slates_F_wins_mae * 100:.1f}%")
    
    print(f"\n📁 Outputs:")
    print(f"  {out_dir / 'README.md'}")
    print(f"  {out_dir / 'aggregate_summary_clean.json'}")
    
    print("\n" + "=" * 70)
    typer.echo("\n[multi] Done! ✓", err=True)


if __name__ == "__main__":
    app()
