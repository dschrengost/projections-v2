"""V3 live pipeline scaffold for GameTransformerV2 integration.

This flow intentionally keeps a minimal critical path and strict gate boundaries:
- scrape core inputs
- freeze run manifest
- build features
- preflight parity validation
- score model
- generate worlds
- finalize projections
- postflight contract validation
- atomic pointer publish

The current implementation supports placeholder mode for end-to-end dev plumbing
while strict preflight/postflight gates and parity checks are enforced.
"""

from __future__ import annotations

# ruff: noqa: E402

import json
import hashlib
import logging
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from prefect import flow, get_run_logger, task
from zoneinfo import ZoneInfo

from projections import model_selectors, ownership_selector, paths
from projections.etl import storage as bronze_storage
from projections.names import normalize_player_name
from projections.pipeline import control_plane, writer_guard
from projections.pipeline.parity_manifest import (
    build_parity_manifest,
    hash_paths,
    load_parity_manifest,
    resolve_parity_manifest_path,
    stable_json_sha256,
    write_parity_manifest,
)
from projections.pipeline.triton_inference_client import (
    TritonEndpointConfig,
    TritonInferenceError,
    check_triton_health,
    infer_json_action,
)
from projections.pipeline.v3_postflight import run_postflight_gate
from projections.pipeline.v3_preflight import run_preflight_gate
from projections.ops.manual_availability import list_manual_overrides, manual_override_report
from projections.rotation.gtv2_promotion_hybrid import (
    PromotionHybridConfig,
    SparseEmergencyGateConfig,
    SparseEmergencyHybridConfig,
    assert_promotion_hybrid_compatible,
)
from projections.rotation.sample_worlds_v2 import _coerce_join_keys
from projections.rotation.tree_rate_bundle import (
    score_tree_rate_bundle_features_to_csv,
)
from projections.runtime_stamp import (
    enforce_clean_tree,
    enforce_prod_sanity,
    log_runtime_stamp,
)


PROJECT_ROOT = paths.get_project_root()

FEATURES_ROOT = "features_gtv2_v1"
SCORES_ROOT = "gtv2_scores"
WORLDS_ROOT = "gtv2_worlds"

PLACEHOLDER_PROJECTION_COLUMNS = [
    "game_date",
    "game_id",
    "team_id",
    "player_id",
    "minutes_sim_mean",
    "minutes_sim_p50",
    "dk_fpts_mean",
    "dk_fpts_p50",
    "sim_p_active",
    "n_worlds",
    "sim_profile",
]

_PROPS_TEAM_ABBR_TO_NBA: dict[str, str] = {
    "PHO": "PHX",
    "GS": "GSW",
    "NO": "NOP",
    "SA": "SAS",
    "NY": "NYK",
}

_ET = ZoneInfo("America/New_York")
_LOCK_WINDOW_THRESHOLDS = (
    {
        "window": "last_30",
        "max_minutes_to_tip": 30.0,
        "injuries_max_age_minutes": 30.0,
        "lineups_max_age_minutes": 30.0,
    },
    {
        "window": "last_60",
        "max_minutes_to_tip": 60.0,
        "injuries_max_age_minutes": 60.0,
        "lineups_max_age_minutes": 60.0,
    },
)
_REPORT_WINDOWS = (
    {
        "label": "nba_injury_report_1pm_et",
        "hour": 13,
        "minute": 0,
        "pre_minutes": 10,
        "post_minutes": 15,
    },
    {
        "label": "nba_injury_report_230pm_et",
        "hour": 14,
        "minute": 30,
        "pre_minutes": 10,
        "post_minutes": 15,
    },
    {
        "label": "nba_injury_report_5pm_et",
        "hour": 17,
        "minute": 0,
        "pre_minutes": 10,
        "post_minutes": 15,
    },
)
_REPORT_WINDOW_WAIT_TIMEOUT_SECONDS = 300
_REPORT_WINDOW_WAIT_INTERVAL_SECONDS = 30
_STALE_INPUT_TOLERANCE_SECONDS = 30
_ODDS_MATERIALITY_MAX_MINUTES_TO_TIP = 180.0
_PROPS_PLAYER_SET_EXPANSION_MAX_MINUTES_TO_TIP = 360.0
_WORLD_CONTRACT_TOL = 1e-4
_WORLD_REALISM_SHORT_MINUTES_DK_THRESHOLD = 35.0
_WORLD_REALISM_GAME_PTS_MAX_THRESHOLD = 340.0
_WORLD_REALISM_GAME_PTS_MIN_THRESHOLD = 110.0
_WORLD_BASE_STAT_CAPS: dict[str, float] = {
    "minutes": 60.0,
    "fga2": 60.0,
    "fg2m": 60.0,
    "fga3": 45.0,
    "fg3m": 30.0,
    "fta": 45.0,
    "ftm": 45.0,
    "oreb": 25.0,
    "dreb": 30.0,
    "ast": 30.0,
    "stl": 15.0,
    "blk": 15.0,
    "tov": 20.0,
    "pf": 10.0,
}
_WORLD_DERIVED_STAT_CAPS: dict[str, float] = {
    "fga": 90.0,
    "fgm": 70.0,
    "pts": 120.0,
    "reb": 45.0,
    "dk_fpts": 150.0,
}
_TEAM_IMPLIED_UNCOVERED_ADD_MIN_MINUTES_MEAN = 12.0
_TEAM_IMPLIED_UNCOVERED_ADD_MIN_PRIOR_PLAY_PROB = 0.35
_TEAM_IMPLIED_UNCOVERED_MAX_DEPTH_RANK = 9
_RETRYABLE_SUBPROCESS_EXIT_CODES = frozenset({-11, -7, -6, 134, 135, 139})
_SUBPROCESS_CRASH_RETRY_ATTEMPTS = max(
    1,
    int(os.environ.get("PROJECTIONS_SUBPROCESS_CRASH_RETRY_ATTEMPTS", "5")),
)
_SUBPROCESS_CRASH_RETRY_DELAY_SECONDS = max(
    0,
    int(os.environ.get("PROJECTIONS_SUBPROCESS_CRASH_RETRY_DELAY_SECONDS", "3")),
)
_TORCH_RUNTIME_CONFIGURED = False


def _gtv2_inference_runtime():
    # Lazy import to avoid importing torch during Prefect flow load.
    from projections.pipeline import gtv2_inference_runtime as runtime

    return runtime


def _gtv2_worlds_runtime():
    # Lazy import to avoid importing torch during Prefect flow load.
    from projections.rotation import sample_worlds_v2 as runtime

    return runtime


def _utc_now_iso() -> str:
    return (
        datetime.now(tz=UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )


def _cli_compatible_ts(ts_value: str) -> str:
    ts = pd.to_datetime(ts_value, utc=True, errors="coerce")
    if pd.isna(ts):
        raise RuntimeError(f"invalid timestamp: {ts_value}")
    ts_utc = pd.Timestamp(ts).tz_convert("UTC").tz_localize(None)
    return ts_utc.strftime("%Y-%m-%dT%H:%M:%S")


def _subprocess_python() -> str:
    # Use the exact interpreter running the flow to avoid `uv run` syncing
    # and mutating site-packages while the worker is active.
    override = os.environ.get("PROJECTIONS_SUBPROCESS_PYTHON")
    if override:
        resolved = (
            shutil.which(override) if Path(override).name == override else override
        )
        if resolved and Path(resolved).exists():
            return str(resolved)
        raise FileNotFoundError(
            f"PROJECTIONS_SUBPROCESS_PYTHON={override} does not exist"
        )
    return sys.executable


def _run_python_module(
    module: str,
    args: list[str],
    *,
    data_root: Path,
    timeout_s: int,
) -> None:
    env = os.environ.copy()
    env["PROJECTIONS_DATA_ROOT"] = str(data_root)
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    cmd = [_subprocess_python(), "-m", module, *args]
    last_result: subprocess.CompletedProcess[str] | None = None
    for attempt in range(1, _SUBPROCESS_CRASH_RETRY_ATTEMPTS + 1):
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        if result.stdout:
            print(result.stdout.rstrip())
        if result.stderr:
            print(result.stderr.rstrip(), file=sys.stderr)
        if result.returncode == 0:
            return
        last_result = result
        if (
            result.returncode not in _RETRYABLE_SUBPROCESS_EXIT_CODES
            or attempt >= _SUBPROCESS_CRASH_RETRY_ATTEMPTS
        ):
            break
        print(
            f"[subprocess-retry] {module} exited with {result.returncode}; "
            f"retrying attempt {attempt + 1}/{_SUBPROCESS_CRASH_RETRY_ATTEMPTS}",
            file=sys.stderr,
        )
        time.sleep(_SUBPROCESS_CRASH_RETRY_DELAY_SECONDS)
    raise RuntimeError(
        f"{module} failed with exit_code={last_result.returncode if last_result else 'unknown'}"
    )


def _resolve_game_date(game_date: str | None) -> str:
    if game_date is not None:
        return str(game_date)
    et = ZoneInfo("America/New_York")
    return datetime.now(tz=et).date().isoformat()


def _resolve_season_month(game_date: str) -> tuple[int, int]:
    ts = pd.Timestamp(game_date)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    season = int(ts.year) if int(ts.month) >= 8 else int(ts.year) - 1
    return season, int(ts.month)


def _load_gtv2_inference_current_config(
    config_path: Path | None = None,
) -> dict[str, Any]:
    resolved_config_path = (
        config_path
        if config_path is not None
        else PROJECT_ROOT / "config" / "gtv2_inference_current.json"
    )
    cfg: dict[str, Any] = {
        "bundle_dir": None,
        "model_version": None,
        "promoted_at": None,
        "bundle_hash": None,
        "promotion_hybrid_enabled": False,
        "promotion_expert_run_dir": None,
        "promotion_prior_minutes_max": 12.0,
        "promotion_hist_start_rate_max": 0.20,
        "promotion_blend_mode": "uplift_only",
        "promotion_force_active_candidates": False,
        "sparse_hybrid_enabled": False,
        "sparse_expert_run_dir": None,
        "sparse_prior_minutes_max": 12.0,
        "sparse_prior_play_prob_max": 0.50,
        "sparse_blend_mode": "uplift_only",
        "sparse_blend_alpha": 1.0,
        "sparse_require_no_props": False,
        "sparse_gate_artifact": None,
        "tree_rate_bundle_dir": None,
        "tree_rate_predictions_csv": None,
        "tree_rate_blend_alpha": 0.0,
        "tree_rate_oreb_share_override_enabled": False,
        "minutes_uncertainty_enabled": False,
        "minutes_uncertainty_mode": "gaussian",
        "minutes_uncertainty_gaussian_scale": 1.0,
        "minutes_uncertainty_min_sigma": 0.75,
        "minutes_uncertainty_max_sigma": 6.0,
        "minutes_uncertainty_fallback_sigma": 1.5,
        "minutes_uncertainty_use_hurdle_sigma": True,
        "minutes_uncertainty_use_prior_std": True,
        "minutes_uncertainty_preserve_top_k_per_team": 3,
        "minutes_uncertainty_full_sigma_at_minutes_or_below": 24.0,
        "minutes_uncertainty_zero_sigma_at_minutes_or_above": 32.0,
        "minutes_uncertainty_apply_minutes_taper": True,
        "minutes_uncertainty_dirichlet_base_concentration": 24.0,
        "minutes_uncertainty_lookup_artifact": None,
        "minutes_uncertainty_empirical_blend_alpha": 1.0,
    }
    if resolved_config_path.exists():
        payload = json.loads(resolved_config_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError(
                f"invalid inference current config payload: {resolved_config_path}"
            )
        cfg.update(payload)
    return cfg


def _resolve_bundle_dir(
    *,
    data_root: Path,
    gtv2_bundle_dir: str | None,
    current_config_payload: dict[str, Any] | None = None,
) -> Path:
    if gtv2_bundle_dir:
        return Path(gtv2_bundle_dir).expanduser().resolve()
    env = os.environ.get("PROJECTIONS_GTV2_BUNDLE_DIR")
    if env:
        return Path(env).expanduser().resolve()
    current_bundle_dir = str(
        (current_config_payload or {}).get("bundle_dir") or ""
    ).strip()
    if current_bundle_dir:
        return Path(current_bundle_dir).expanduser().resolve()
    return (
        data_root / "artifacts" / "game_transformer_v2" / "bundle_current"
    ).resolve()


def _load_gtv2_inference_server_config() -> dict[str, Any]:
    config_path = PROJECT_ROOT / "config" / "gtv2_inference_server.json"
    cfg: dict[str, Any] = {
        "enabled": False,
        "backend": "local",
        "triton_endpoint": "localhost:8000",
        "model_name": "gtv2_scorer",
        "model_version": None,
        "timeout_seconds": 90.0,
        "healthcheck_timeout_seconds": 3.0,
    }
    if config_path.exists():
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError(f"invalid inference server config payload: {config_path}")
        cfg.update(payload)
    if cfg.get("triton_url") and not cfg.get("triton_endpoint"):
        cfg["triton_endpoint"] = cfg.get("triton_url")
    return cfg


def _normalize_tree_world_game_date(df: pd.DataFrame, *, col: str = "game_date") -> pd.DataFrame:
    out = df.copy()
    out[col] = pd.to_datetime(out[col], errors="coerce").dt.date.astype(str)
    return out


def _compute_tree_world_dk_fpts(df: pd.DataFrame) -> pd.Series:
    pts = pd.to_numeric(df["pts"], errors="coerce").fillna(0.0)
    reb = pd.to_numeric(df["reb"], errors="coerce").fillna(0.0)
    ast = pd.to_numeric(df["ast"], errors="coerce").fillna(0.0)
    stl = pd.to_numeric(df["stl"], errors="coerce").fillna(0.0)
    blk = pd.to_numeric(df["blk"], errors="coerce").fillna(0.0)
    tov = pd.to_numeric(df["tov"], errors="coerce").fillna(0.0)
    base = pts + 1.25 * reb + 1.5 * ast + 2.0 * stl + 2.0 * blk - 0.5 * tov
    qualifiers = pd.concat(
        [
            (pts >= 10.0).astype(int),
            (reb >= 10.0).astype(int),
            (ast >= 10.0).astype(int),
            (stl >= 10.0).astype(int),
            (blk >= 10.0).astype(int),
        ],
        axis=1,
    ).sum(axis=1)
    return base + np.where(qualifiers == 2, 1.5, 0.0) + np.where(qualifiers >= 3, 3.0, 0.0)


def _rescale_tree_world_stat_to_target_mean(
    work: pd.DataFrame,
    *,
    stat_col: str,
    target_mean_col: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = work.copy()
    player_keys = ["game_date", "game_id", "team_id", "player_id"]
    current_means = (
        out.groupby(player_keys, as_index=False)[stat_col]
        .mean()
        .rename(columns={stat_col: f"{stat_col}_current_mean"})
    )
    out = out.merge(current_means, on=player_keys, how="left")
    target_means = pd.to_numeric(out[target_mean_col], errors="coerce").fillna(0.0)
    current_means_arr = pd.to_numeric(out[f"{stat_col}_current_mean"], errors="coerce").fillna(0.0)
    current_vals = pd.to_numeric(out[stat_col], errors="coerce").fillna(0.0)
    scale = np.where(current_means_arr > 1e-9, target_means / current_means_arr, np.nan)
    scaled_vals = current_vals * scale
    fallback_mask = ~np.isfinite(scale)
    fallback_groups = 0
    if bool(fallback_mask.any()):
        fallback_groups = int(out.loc[fallback_mask, player_keys].drop_duplicates().shape[0])
        minutes = pd.to_numeric(out["minutes"], errors="coerce").fillna(0.0)
        active_world = (minutes > 0.0).astype(float)
        fallback_weight = np.where(minutes > 0.0, minutes, active_world)
        out["_fallback_weight"] = fallback_weight
        group_weight_sum = out.groupby(player_keys)["_fallback_weight"].transform("sum")
        group_world_count = out.groupby(player_keys)[stat_col].transform("size")
        fallback_target_total = target_means * pd.to_numeric(group_world_count, errors="coerce").fillna(0.0)
        fallback_vals = np.where(
            group_weight_sum > 1e-9,
            fallback_target_total * out["_fallback_weight"] / group_weight_sum,
            target_means,
        )
        scaled_vals = np.where(fallback_mask, fallback_vals, scaled_vals)
        out = out.drop(columns=["_fallback_weight"])
    out[stat_col] = np.clip(np.asarray(scaled_vals, dtype=float), 0.0, None)
    out = out.drop(columns=[f"{stat_col}_current_mean"])
    return out, {"stat": stat_col, "fallback_group_count": int(fallback_groups)}


def _override_tree_rebound_share_to_target_mean(
    work: pd.DataFrame,
    *,
    stat_col: str,
    target_mean_col: str,
    blend_alpha: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = work.copy()
    team_keys = ["game_date", "game_id", "team_id"]
    player_keys = ["game_date", "game_id", "team_id", "player_id"]
    world_team_keys = ["game_date", "game_id", "team_id", "world_idx"]

    target_player = out.loc[:, player_keys + [target_mean_col]].drop_duplicates(player_keys).copy()
    target_player[target_mean_col] = pd.to_numeric(target_player[target_mean_col], errors="coerce").fillna(0.0)
    target_team = (
        target_player.groupby(team_keys, as_index=False)[target_mean_col]
        .sum()
        .rename(columns={target_mean_col: f"{stat_col}_target_team_mean"})
    )
    target_player = target_player.merge(target_team, on=team_keys, how="left")
    target_player[f"{stat_col}_target_share"] = np.where(
        pd.to_numeric(target_player[f"{stat_col}_target_team_mean"], errors="coerce").fillna(0.0) > 1e-9,
        pd.to_numeric(target_player[target_mean_col], errors="coerce").fillna(0.0)
        / pd.to_numeric(target_player[f"{stat_col}_target_team_mean"], errors="coerce").fillna(1.0),
        0.0,
    )
    out = out.merge(
        target_player.loc[:, player_keys + [f"{stat_col}_target_share"]],
        on=player_keys,
        how="left",
        validate="many_to_one",
    )
    team_world_total = out.groupby(world_team_keys)[stat_col].transform("sum")
    current_vals = pd.to_numeric(out[stat_col], errors="coerce").fillna(0.0)
    active_mask = pd.to_numeric(out.get("minutes", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float) > 1e-9
    target_share = pd.to_numeric(out[f"{stat_col}_target_share"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    target_share = np.where(active_mask, target_share, 0.0)
    current_share = np.where(active_mask & (team_world_total > 1e-9), current_vals / team_world_total, 0.0)
    alpha = float(np.clip(blend_alpha, 0.0, 1.0))
    out[f"{stat_col}_share_blended"] = np.clip((1.0 - alpha) * current_share + alpha * target_share, 0.0, None)
    share_sum = out.groupby(world_team_keys)[f"{stat_col}_share_blended"].transform("sum")
    out[f"{stat_col}_share_final"] = np.where(share_sum > 1e-9, out[f"{stat_col}_share_blended"] / share_sum, 0.0)
    out[stat_col] = np.where(
        team_world_total > 1e-9,
        team_world_total * pd.to_numeric(out[f"{stat_col}_share_final"], errors="coerce").fillna(0.0),
        current_vals,
    )
    out = out.drop(columns=[c for c in [f"{stat_col}_target_share", f"{stat_col}_share_blended", f"{stat_col}_share_final"] if c in out.columns])
    current_means = (
        out.groupby(player_keys, as_index=False)[stat_col]
        .mean()
        .rename(columns={stat_col: f"{stat_col}_post_mean"})
    )
    compare = current_means.merge(target_player.loc[:, player_keys + [target_mean_col]], on=player_keys, how="left")
    err = pd.to_numeric(compare[f"{stat_col}_post_mean"], errors="coerce").fillna(0.0) - pd.to_numeric(compare[target_mean_col], errors="coerce").fillna(0.0)
    return out, {
        "stat": stat_col,
        "mode": "team_budget_share_override",
        "post_minus_target_mean_abs_mean": float(np.abs(err).mean()) if len(err) else 0.0,
        "post_minus_target_mean_bias": float(err.mean()) if len(err) else 0.0,
    }


def _apply_tree_rate_mean_override_to_worlds(
    worlds: pd.DataFrame,
    *,
    predictions_csv: Path,
    blend_alpha: float,
    oreb_share_override_enabled: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not predictions_csv.exists():
        raise FileNotFoundError(f"tree-rate predictions csv not found: {predictions_csv}")
    pred_df = _coerce_join_keys(pd.read_csv(predictions_csv), name="tree_rate_predictions")
    stat_to_rate_col = {"ast": "pred_ast_per_min", "oreb": "pred_oreb_per_min", "dreb": "pred_dreb_per_min"}
    available_stats = [stat for stat, rate_col in stat_to_rate_col.items() if rate_col in pred_df.columns]
    if not available_stats:
        raise ValueError(f"tree-rate predictions must include at least one of {sorted(stat_to_rate_col.values())}: {predictions_csv}")
    pred_df = _normalize_tree_world_game_date(pred_df)
    keep_pred_cols = [stat_to_rate_col[stat] for stat in available_stats]
    pred_df = pred_df.loc[:, ["game_date", "game_id", "team_id", "player_id", *keep_pred_cols]].drop_duplicates(["game_date", "game_id", "team_id", "player_id"])
    work = worlds.merge(pred_df, on=["game_date", "game_id", "team_id", "player_id"], how="left", validate="many_to_one")
    override_keys = pd.Series(False, index=work.index)
    for rate_col in keep_pred_cols:
        override_keys = override_keys | work[rate_col].notna()
    matched_player_count = int(
        work.loc[override_keys, ["game_date", "game_id", "team_id", "player_id"]]
        .drop_duplicates()
        .shape[0]
    )
    if matched_player_count <= 0:
        return worlds.copy(), {
            "applied": False,
            "predictions_csv": str(predictions_csv),
            "blend_alpha": float(np.clip(blend_alpha, 0.0, 1.0)),
            "available_stats": available_stats,
            "player_count_with_predictions": 0,
            "stat_reports": [],
            "oreb_share_override_enabled": bool(oreb_share_override_enabled),
            "skip_reason": "no_matching_players",
        }
    minutes_mean = (
        work.groupby(["game_date", "game_id", "team_id", "player_id"], as_index=False)["minutes"]
        .mean()
        .rename(columns={"minutes": "variant_minutes_mean"})
    )
    work = work.merge(minutes_mean, on=["game_date", "game_id", "team_id", "player_id"], how="left")
    alpha = float(np.clip(blend_alpha, 0.0, 1.0))
    for stat_col in available_stats:
        per_min_col = stat_to_rate_col[stat_col]
        current_mean = (
            work.groupby(["game_date", "game_id", "team_id", "player_id"], as_index=False)[stat_col]
            .mean()
            .rename(columns={stat_col: f"{stat_col}_current_mean"})
        )
        work = work.merge(current_mean, on=["game_date", "game_id", "team_id", "player_id"], how="left")
        tree_target_mean = pd.to_numeric(work["variant_minutes_mean"], errors="coerce").fillna(0.0) * pd.to_numeric(work[per_min_col], errors="coerce").fillna(0.0)
        current_mean_arr = pd.to_numeric(work[f"{stat_col}_current_mean"], errors="coerce").fillna(0.0)
        work[f"{stat_col}_target_mean"] = (1.0 - alpha) * current_mean_arr + alpha * tree_target_mean
        work = work.drop(columns=[f"{stat_col}_current_mean"])
    stat_reports: list[dict[str, Any]] = []
    for stat_col in available_stats:
        if stat_col == "oreb" and oreb_share_override_enabled:
            work, stat_report = _override_tree_rebound_share_to_target_mean(work, stat_col=stat_col, target_mean_col=f"{stat_col}_target_mean", blend_alpha=alpha)
        elif stat_col == "dreb":
            work, stat_report = _override_tree_rebound_share_to_target_mean(work, stat_col=stat_col, target_mean_col=f"{stat_col}_target_mean", blend_alpha=alpha)
        else:
            work, stat_report = _rescale_tree_world_stat_to_target_mean(work, stat_col=stat_col, target_mean_col=f"{stat_col}_target_mean")
        stat_reports.append(stat_report)
    if {"oreb", "dreb"} & set(available_stats):
        work["reb"] = pd.to_numeric(work["oreb"], errors="coerce").fillna(0.0) + pd.to_numeric(work["dreb"], errors="coerce").fillna(0.0)
    work["dk_fpts"] = _compute_tree_world_dk_fpts(work)
    report = {
        "applied": True,
        "predictions_csv": str(predictions_csv),
        "blend_alpha": alpha,
        "available_stats": available_stats,
        "player_count_with_predictions": matched_player_count,
        "stat_reports": stat_reports,
        "oreb_share_override_enabled": bool(oreb_share_override_enabled),
    }
    drop_cols = ["pred_ast_per_min", "pred_oreb_per_min", "pred_dreb_per_min", "variant_minutes_mean", "ast_target_mean", "oreb_target_mean", "dreb_target_mean"]
    work = work.drop(columns=[col for col in drop_cols if col in work.columns])
    return work, report


def _resolve_gtv2_inference_backend(
    *,
    requested: str | None,
    config_payload: dict[str, Any],
) -> str:
    mode = str(requested or "auto").strip().lower()
    if mode not in {"auto", "local", "triton"}:
        raise RuntimeError(
            "gtv2_inference_backend must be one of: auto, local, triton"
        )
    if mode != "auto":
        return mode
    cfg_backend = str(config_payload.get("backend", "local")).strip().lower()
    if bool(config_payload.get("enabled")) and cfg_backend == "triton":
        return "triton"
    return "local"


def _resolve_triton_endpoint_config(
    *,
    config_payload: dict[str, Any],
    endpoint_override: str | None,
    model_name_override: str | None,
    model_version_override: str | None,
    timeout_seconds_override: float | None,
) -> TritonEndpointConfig:
    endpoint = (
        endpoint_override
        or str(config_payload.get("triton_endpoint") or "").strip()
        or str(config_payload.get("triton_url") or "").strip()
        or "localhost:8000"
    )
    model_name = (
        model_name_override
        or str(config_payload.get("model_name") or "").strip()
        or "gtv2_scorer"
    )
    model_version = (
        model_version_override
        or str(config_payload.get("model_version") or "").strip()
        or None
    )
    timeout_seconds = float(
        timeout_seconds_override
        if timeout_seconds_override is not None
        else config_payload.get("timeout_seconds", 90.0)
    )
    return TritonEndpointConfig(
        endpoint=endpoint,
        model_name=model_name,
        model_version=model_version,
        timeout_seconds=timeout_seconds,
    )


def _placeholder_feature_frame(*, game_date: str, as_of_ts: str) -> pd.DataFrame:
    game_id = 900001
    team_a = 100
    team_b = 200
    rows: list[dict[str, Any]] = []
    for idx in range(20):
        team_id = team_a if idx < 10 else team_b
        local_idx = idx if idx < 10 else idx - 10
        rows.append(
            {
                "game_date": game_date,
                "game_id": game_id,
                "team_id": team_id,
                "player_id": 1000 + idx,
                "lineup_available": 1,
                "lineup_starter_announced": 1 if local_idx < 5 else 0,
                "vegas_total": 228.5,
                "vegas_spread": -2.5,
                "estimated_possessions": 99.4,
                "minutes_prior": 26.0 - float(local_idx),
                "usage_prior": 0.22 - 0.005 * float(local_idx),
                "as_of_ts": as_of_ts,
            }
        )
    return pd.DataFrame(rows)


def _ensure_placeholder_bundle(
    *,
    bundle_dir: Path,
    features_df: pd.DataFrame,
    transform_manifest: dict[str, Any],
    integrity: dict[str, Any],
) -> Path:
    # Placeholder mode must never mutate production bundles. Callers should pass
    # a run-scoped scratch directory (not the promoted bundle directory).
    bundle_dir.mkdir(parents=True, exist_ok=True)
    stub = bundle_dir / "bundle_stub.txt"
    if not stub.exists():
        stub.write_text("placeholder bundle", encoding="utf-8")

    manifest_path = resolve_parity_manifest_path(bundle_dir)
    manifest_payload = build_parity_manifest(
        model_id="game_transformer_v2_placeholder",
        features_df=features_df,
        feature_columns=list(features_df.columns),
        missing_value_policy={"disallow_null_columns": list(features_df.columns)},
        transform_manifest=transform_manifest,
        output_manifest={
            "projection_columns": list(PLACEHOLDER_PROJECTION_COLUMNS),
            "semantics": {
                "dk_fpts_mean": "conditional_on_active",
                "minutes_sim_mean": "conditional_on_active",
            },
        },
        integrity=integrity,
    )
    write_parity_manifest(manifest_path, manifest_payload)
    return manifest_path


def _bundle_artifact_hash(bundle_dir: Path) -> str:
    if not bundle_dir.exists():
        return stable_json_sha256([])
    files = [p for p in bundle_dir.rglob("*") if p.is_file()]
    return hash_paths(files)


def _normalize_gtv2_projection_surface_semantics(df: pd.DataFrame) -> pd.DataFrame:
    """Expose unconditional GTv2 summaries as the default live-facing columns."""

    if df.empty:
        return df

    out = df.copy()
    updates: dict[str, pd.Series] = {}

    def _num(col: str) -> pd.Series | None:
        if col in updates:
            return updates[col]
        if col not in out.columns:
            return None
        return pd.to_numeric(out[col], errors="coerce")

    def _promote_uncond(*, cond_col: str, uncond_col: str, cond_alias: str) -> None:
        cond_series = _num(cond_col)
        if cond_series is not None and cond_alias not in out.columns and cond_alias not in updates:
            updates[cond_alias] = cond_series
        uncond_series = _num(uncond_col)
        if uncond_series is not None:
            updates[cond_col] = uncond_series

    sim_family_specs = {
        "minutes_sim": ("mean", "std", "p10", "p50", "p90"),
        "dk_fpts": ("mean", "std", "p05", "p10", "p25", "p50", "p75", "p90", "p95"),
    }
    for family, suffixes in sim_family_specs.items():
        for suffix in suffixes:
            cond_col = f"{family}_{suffix}"
            uncond_col = f"{family}_{suffix}_uncond"
            cond_alias = f"{family}_{suffix}_cond"
            _promote_uncond(
                cond_col=cond_col,
                uncond_col=uncond_col,
                cond_alias=cond_alias,
            )

            pref_cond_col = f"sim_{family}_{suffix}"
            pref_uncond_col = f"sim_{family}_{suffix}_uncond"
            pref_cond_alias = f"sim_{family}_{suffix}_cond"
            if pref_uncond_col not in out.columns and pref_uncond_col not in updates and uncond_col in out.columns:
                base_uncond = _num(uncond_col)
                if base_uncond is not None:
                    updates[pref_uncond_col] = base_uncond
            _promote_uncond(
                cond_col=pref_cond_col,
                uncond_col=pref_uncond_col,
                cond_alias=pref_cond_alias,
            )

    for stat in ("pts", "reb", "ast", "stl", "blk", "tov"):
        cond_col = f"{stat}_mean"
        uncond_col = f"{stat}_mean_uncond"
        cond_alias = f"{stat}_mean_cond"
        _promote_uncond(
            cond_col=cond_col,
            uncond_col=uncond_col,
            cond_alias=cond_alias,
        )

        pref_cond_col = f"sim_{stat}_mean"
        pref_uncond_col = f"sim_{stat}_mean_uncond"
        pref_cond_alias = f"sim_{stat}_mean_cond"
        if pref_uncond_col not in out.columns and pref_uncond_col not in updates and uncond_col in out.columns:
            base_uncond = _num(uncond_col)
            if base_uncond is not None:
                updates[pref_uncond_col] = base_uncond
        _promote_uncond(
            cond_col=pref_cond_col,
            uncond_col=pref_uncond_col,
            cond_alias=pref_cond_alias,
        )

    if updates:
        out = out.assign(**updates)
    return out


def _set_inference_seed(seed: int) -> None:
    runtime = _gtv2_inference_runtime()
    runtime.set_inference_seed(int(seed))


def _configure_torch_runtime_for_inference() -> None:
    """Apply conservative torch runtime settings for long live inference tasks.

    We intentionally default to single-threaded CPU execution and disabled MKLDNN
    to reduce intermittent native crashes observed in long-running world generation.
    Operators can override defaults via environment variables:
      - PROJECTIONS_TORCH_NUM_THREADS
      - PROJECTIONS_TORCH_NUM_INTEROP_THREADS
      - PROJECTIONS_TORCH_DISABLE_MKLDNN
    """
    global _TORCH_RUNTIME_CONFIGURED
    if _TORCH_RUNTIME_CONFIGURED:
        return

    num_threads = int(os.environ.get("PROJECTIONS_TORCH_NUM_THREADS", "1"))
    interop_threads = int(
        os.environ.get("PROJECTIONS_TORCH_NUM_INTEROP_THREADS", "1")
    )
    disable_mkldnn = (
        str(os.environ.get("PROJECTIONS_TORCH_DISABLE_MKLDNN", "1"))
        .strip()
        .lower()
        in {"1", "true", "yes"}
    )

    try:
        import torch

        torch.set_num_threads(max(1, int(num_threads)))
    except Exception:
        pass
    try:
        import torch

        torch.set_num_interop_threads(max(1, int(interop_threads)))
    except Exception:
        pass
    if disable_mkldnn:
        try:
            import torch

            torch.backends.mkldnn.enabled = False
        except Exception:
            pass

    _TORCH_RUNTIME_CONFIGURED = True


def _resolve_torch_device(device: str | None) -> Any:
    runtime = _gtv2_inference_runtime()
    return runtime.resolve_torch_device(device)


def _load_gtv2_model(
    bundle_dir: Path,
    *,
    device: Any,
    flow_scale_clip_override: float | None = None,
) -> tuple[Any, Any]:
    runtime = _gtv2_inference_runtime()
    return runtime.load_gtv2_model(
        bundle_dir=Path(bundle_dir),
        device=device,
        flow_scale_clip_override=flow_scale_clip_override,
    )


def _coerce_frame_to_manifest_schema(
    features_df: pd.DataFrame, manifest: dict[str, Any]
) -> pd.DataFrame:
    schema = manifest.get("feature_schema")
    if not isinstance(schema, list) or len(schema) <= 0:
        raise RuntimeError("parity manifest missing feature_schema")

    out = features_df.copy()
    ordered_cols: list[str] = []
    for row in schema:
        col = str(row.get("name"))
        dtype = str(row.get("dtype"))
        nullable = bool(row.get("nullable", True))
        if col not in out.columns:
            raise RuntimeError(f"feature frame missing manifest column: {col}")
        series = out[col]
        try:
            if dtype in {
                "int64",
                "int32",
                "int16",
                "int8",
                "Int64",
                "Int32",
                "Int16",
                "Int8",
            }:
                series = pd.to_numeric(series, errors="coerce").astype(dtype)
            elif dtype in {"float64", "float32", "float16"}:
                series = pd.to_numeric(series, errors="coerce").astype(dtype)
            elif dtype in {"bool", "boolean"}:
                if dtype == "bool":
                    series = series.fillna(False).astype(bool)
                else:
                    series = series.astype("boolean")
            elif dtype.startswith("datetime64"):
                utc = "UTC" in dtype
                series = pd.to_datetime(series, errors="coerce", utc=utc)
                if not utc:
                    series = series.dt.tz_localize(None)
            else:
                series = series.astype(dtype)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"failed to coerce feature column '{col}' to dtype '{dtype}': {exc}"
            ) from exc

        if not nullable and bool(series.isna().any()):
            raise RuntimeError(
                f"non-nullable feature column has nulls after coercion: {col}"
            )
        out[col] = series
        ordered_cols.append(col)

    return out.loc[:, ordered_cols].copy()


def _build_gtv2_inference_examples(
    *,
    features_df: pd.DataFrame,
    game_date: str,
    config: Any,
) -> list[Any]:
    runtime = _gtv2_inference_runtime()
    return runtime.build_gtv2_inference_examples(
        features_df=features_df,
        game_date=game_date,
        config=config,
    )


def _attach_gtv2_force_active_worlds(
    features_df: pd.DataFrame,
    *,
    game_date: str,
    data_root: Path,
    as_of_ts: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = features_df.copy()
    if out.empty:
        out["force_active_worlds"] = pd.Series(dtype="int8")
        return out, {"starter_rows": 0, "manual_force_in_rows": 0, "total_force_active_rows": 0}

    starter_mask = np.zeros(len(out), dtype=bool)
    for col in ("lineup_starter_announced", "is_projected_starter", "is_confirmed_starter"):
        if col in out.columns:
            starter_mask |= (
                pd.to_numeric(out[col], errors="coerce")
                .fillna(0.0)
                .to_numpy(dtype=float)
                >= 0.5
            )

    manual_force_in_mask = np.zeros(len(out), dtype=bool)
    try:
        overrides = list_manual_overrides(
            pd.Timestamp(game_date).date(),
            data_root=data_root,
            active_only=True,
            as_of_ts=as_of_ts,
        )
    except Exception:
        overrides = []
    force_in_keys: set[str] = set()
    for row in overrides:
        if str(row.get("override_type", "")).strip().lower() != "force_in":
            continue
        game_id_raw = pd.to_numeric(pd.Series([row.get("game_id")]), errors="coerce").iloc[0]
        player_id_raw = pd.to_numeric(pd.Series([row.get("player_id")]), errors="coerce").iloc[0]
        if pd.isna(game_id_raw) or pd.isna(player_id_raw):
            continue
        force_in_keys.add(f"{int(game_id_raw)}|{int(player_id_raw)}")
    if force_in_keys and {"game_id", "player_id"}.issubset(out.columns):
        game_ids = pd.to_numeric(out["game_id"], errors="coerce").astype("Int64")
        player_ids = pd.to_numeric(out["player_id"], errors="coerce").astype("Int64")
        keys = (game_ids.astype("string") + "|" + player_ids.astype("string")).fillna("")
        manual_force_in_mask = keys.isin(force_in_keys).to_numpy(dtype=bool)

    force_active_mask = starter_mask | manual_force_in_mask
    out["force_active_worlds"] = force_active_mask.astype("int8")
    return out, {
        "starter_rows": int(starter_mask.sum()),
        "manual_force_in_rows": int(manual_force_in_mask.sum()),
        "total_force_active_rows": int(force_active_mask.sum()),
    }


def _attach_gtv2_score_surface(
    features_df: pd.DataFrame,
    *,
    scores_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = features_df.copy()
    if out.empty:
        out["gtv2_score_minutes_deterministic"] = pd.Series(dtype="float32")
        out["gtv2_score_active_deterministic"] = pd.Series(dtype="float32")
        out["gtv2_score_active_prob_proxy"] = pd.Series(dtype="float32")
        return out, {"rows": 0, "matched_rows": 0, "missing_rows": 0}
    if scores_df.empty:
        out["gtv2_score_minutes_deterministic"] = np.nan
        out["gtv2_score_active_deterministic"] = np.nan
        out["gtv2_score_active_prob_proxy"] = np.nan
        return out, {"rows": int(len(out)), "matched_rows": 0, "missing_rows": int(len(out))}

    keys = ["game_id", "team_id", "player_id"]
    score_cols = [
        "minutes_deterministic",
        "active_deterministic",
        "active_prob_proxy",
    ]
    scored = scores_df.copy()
    for col in keys:
        if col not in scored.columns:
            raise ValueError(f"scores_df missing key column: {col}")
        scored[col] = pd.to_numeric(scored[col], errors="coerce").astype("Int64")
    for col in score_cols:
        if col not in scored.columns:
            scored[col] = np.nan
    keep_cols = keys + score_cols
    scored = scored[keep_cols].drop_duplicates(subset=keys, keep="last")
    scored = scored.rename(
        columns={
            "minutes_deterministic": "gtv2_score_minutes_deterministic",
            "active_deterministic": "gtv2_score_active_deterministic",
            "active_prob_proxy": "gtv2_score_active_prob_proxy",
        }
    )

    base = out.copy()
    for col in keys:
        if col not in base.columns:
            raise ValueError(f"features_df missing key column: {col}")
        base[col] = pd.to_numeric(base[col], errors="coerce").astype("Int64")
    out = base.merge(scored, on=keys, how="left")
    matched = int(out["gtv2_score_minutes_deterministic"].notna().sum()) if "gtv2_score_minutes_deterministic" in out.columns else 0
    return out, {
        "rows": int(len(out)),
        "matched_rows": matched,
        "missing_rows": int(len(out) - matched),
    }


def _load_minutes_uncertainty_lookup_artifact(path: str | None) -> dict[str, Any]:
    resolved = str(path or "").strip()
    if not resolved:
        return {"applied": False, "reason": "disabled", "bin_edges": [], "sigma_by_bin": []}
    artifact_path = Path(resolved).expanduser().resolve()
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid minutes uncertainty artifact payload: {artifact_path}")
    bin_edges = [float(x) for x in payload.get("bin_edges", [])]
    sigma_by_bin = [float(x) for x in payload.get("sigma_by_bin", [])]
    if len(bin_edges) < 2 or len(sigma_by_bin) != len(bin_edges) - 1:
        raise RuntimeError(
            f"invalid minutes uncertainty artifact bins: path={artifact_path} "
            f"len(bin_edges)={len(bin_edges)} len(sigma_by_bin)={len(sigma_by_bin)}"
        )
    return {
        "applied": True,
        "artifact_path": str(artifact_path),
        "sigma_source": payload.get("sigma_source"),
        "row_count": int(payload.get("row_count", 0)),
        "bin_edges": bin_edges,
        "sigma_by_bin": sigma_by_bin,
    }


def _selected_props_source_from_checklist(checklist: dict[str, Any]) -> str | None:
    checks = checklist.get("checks")
    if not isinstance(checks, list):
        return None
    for entry in checks:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("name")) != "props_source_policy_satisfied":
            continue
        details = entry.get("details")
        if isinstance(details, dict):
            val = details.get("selected_source")
            return None if val is None else str(val)
    return None


def _read_parquet_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _filter_slate_rows(df: pd.DataFrame, game_ids: list[int]) -> pd.DataFrame:
    if df.empty or not game_ids or "game_id" not in df.columns:
        return df.iloc[0:0].copy()
    gids = pd.to_numeric(df["game_id"], errors="coerce").astype("Int64")
    return df.loc[gids.isin(game_ids)].copy()


def _latest_ts(df: pd.DataFrame, *, time_col: str = "as_of_ts") -> pd.Timestamp | None:
    if df.empty or time_col not in df.columns:
        return None
    ts = pd.to_datetime(df[time_col], utc=True, errors="coerce").dropna()
    if ts.empty:
        return None
    return pd.Timestamp(ts.max())


def _normalize_props_team_abbr(value: object) -> str:
    raw = str(value or "").strip().upper()
    if not raw:
        return ""
    return _PROPS_TEAM_ABBR_TO_NBA.get(raw, raw)


def _ts_to_iso(value: pd.Timestamp | None) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat()


def _age_minutes(run_ts: pd.Timestamp, source_ts: pd.Timestamp | None) -> float | None:
    if source_ts is None or pd.isna(source_ts):
        return None
    return float((run_ts - source_ts).total_seconds() / 60.0)


def _latest_ts_by_game(
    df: pd.DataFrame, *, time_col: str = "as_of_ts"
) -> dict[int, pd.Timestamp]:
    if df.empty or "game_id" not in df.columns or time_col not in df.columns:
        return {}
    working = df.loc[:, ["game_id", time_col]].copy()
    working["game_id"] = pd.to_numeric(working["game_id"], errors="coerce").astype(
        "Int64"
    )
    working[time_col] = pd.to_datetime(working[time_col], utc=True, errors="coerce")
    working = working.dropna(subset=["game_id", time_col])
    if working.empty:
        return {}
    latest = working.groupby("game_id", sort=False)[time_col].max()
    return {int(game_id): pd.Timestamp(ts) for game_id, ts in latest.items()}


def _latest_ts_by_game_from_teams(
    slate_df: pd.DataFrame,
    source_df: pd.DataFrame,
    *,
    time_col: str,
) -> dict[int, pd.Timestamp]:
    if slate_df.empty or source_df.empty or time_col not in source_df.columns:
        return {}
    team_col = next(
        (
            candidate
            for candidate in ("team_tricode", "team_abbreviation", "team")
            if candidate in source_df.columns
        ),
        None,
    )
    if team_col is None:
        return {}
    working = source_df.loc[:, [team_col, time_col]].copy()
    working["team_tricode"] = working[team_col].map(_normalize_props_team_abbr)
    working[time_col] = pd.to_datetime(working[time_col], utc=True, errors="coerce")
    working = working.dropna(subset=["team_tricode", time_col])
    if working.empty:
        return {}
    per_team = working.groupby("team_tricode", sort=False)[time_col].max().to_dict()
    out: dict[int, pd.Timestamp] = {}
    for row in slate_df.itertuples(index=False):
        game_id = pd.to_numeric(getattr(row, "game_id", None), errors="coerce")
        if pd.isna(game_id):
            continue
        ts_values: list[pd.Timestamp] = []
        for attr in ("home_team_tricode", "away_team_tricode"):
            team = _normalize_props_team_abbr(getattr(row, attr, None))
            if not team:
                continue
            ts = per_team.get(team)
            if ts is not None and not pd.isna(ts):
                ts_values.append(pd.Timestamp(ts))
        if ts_values:
            out[int(game_id)] = max(ts_values)
    return out


def _probe_rotowire_props_snapshot_summary(
    *,
    rotowire_props_root: Path,
    game_date: pd.Timestamp,
    data_root: Path,
    run_as_of_ts: pd.Timestamp | None = None,
    timeout_s: int = 180,
) -> dict[str, Any]:
    probe_code = """
import json
import hashlib
import sys
from pathlib import Path

import pandas as pd

from projections.features.action_props import (
    build_action_props_feature_snapshots,
    load_rotowire_props_long_from_bronze,
)

root = Path(sys.argv[1])
day = pd.Timestamp(sys.argv[2])
run_as_of = pd.to_datetime(sys.argv[3], utc=True, errors="coerce")
frames = []
for offset in (0, 1):
    current_day = day + pd.Timedelta(days=offset)
    long_df = load_rotowire_props_long_from_bronze(
        rotowire_props_root=root,
        game_date=current_day,
    )
    if not pd.isna(run_as_of):
        asof = pd.to_datetime(long_df.get("action_props_as_of_ts"), utc=True, errors="coerce")
        long_df = long_df.loc[asof.notna() & (asof <= run_as_of)].copy()
    snap_df = build_action_props_feature_snapshots(long_df)
    if not snap_df.empty:
        keep_cols = [
            c
            for c in ("team_tricode", "action_props_as_of_ts", "player_name_norm", "an_has_any_props")
            if c in snap_df.columns
        ]
        frames.append(snap_df.loc[:, keep_cols].copy())

if frames:
    combined = pd.concat(frames, ignore_index=True)
    combined["team_tricode"] = combined["team_tricode"].astype(str).str.strip().str.upper()
    combined["action_props_as_of_ts"] = pd.to_datetime(
        combined["action_props_as_of_ts"], utc=True, errors="coerce"
    )
    combined = combined.dropna(subset=["team_tricode", "action_props_as_of_ts", "player_name_norm"])
    if "an_has_any_props" in combined.columns:
        combined = combined.loc[
            pd.to_numeric(combined["an_has_any_props"], errors="coerce").fillna(0.0)
            > 0.0
        ].copy()
else:
    combined = pd.DataFrame(
        columns=["team_tricode", "action_props_as_of_ts", "player_name_norm"]
    )

team_latest = (
    combined.groupby("team_tricode", sort=False)["action_props_as_of_ts"].max().to_dict()
    if not combined.empty
    else {}
)
team_player_digest = {}
team_player_count = {}
if not combined.empty:
    players_by_team = combined.groupby("team_tricode", sort=False)["player_name_norm"]
    for team, players in players_by_team:
        names = sorted({str(v).strip() for v in players if str(v).strip()})
        payload = json.dumps(names, separators=(",", ":"), ensure_ascii=True)
        team_player_digest[str(team)] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        team_player_count[str(team)] = int(len(names))
latest = combined["action_props_as_of_ts"].max() if not combined.empty else None
payload = {
    "parsed_rows": int(len(combined)),
    "latest_action_props_as_of_ts": (
        None if latest is None or pd.isna(latest) else pd.Timestamp(latest).isoformat()
    ),
    "teams": sorted(team_latest.keys()),
    "team_latest_as_of_ts": {
        str(team): pd.Timestamp(ts).isoformat()
        for team, ts in team_latest.items()
        if ts is not None and not pd.isna(ts)
    },
    "team_player_digest": team_player_digest,
    "team_player_count": team_player_count,
}
print(json.dumps(payload))
""".strip()
    env = os.environ.copy()
    env["PROJECTIONS_DATA_ROOT"] = str(data_root)
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    cmd = [
        _subprocess_python(),
        "-c",
        probe_code,
        str(rotowire_props_root),
        pd.Timestamp(game_date).normalize().date().isoformat(),
        (
            pd.Timestamp(run_as_of_ts).isoformat()
            if run_as_of_ts is not None and not pd.isna(run_as_of_ts)
            else ""
        ),
    ]
    last_error = "rotowire props probe did not run"
    for attempt in range(1, _SUBPROCESS_CRASH_RETRY_ATTEMPTS + 1):
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        if result.stderr:
            print(result.stderr.rstrip(), file=sys.stderr)
        if result.returncode == 0:
            stdout = result.stdout.strip()
            if not stdout:
                return {
                    "parsed_rows": 0,
                    "latest_action_props_as_of_ts": None,
                    "teams": [],
                    "team_latest_as_of_ts": {},
                    "team_player_digest": {},
                    "team_player_count": {},
                    "parse_error": "rotowire props probe returned empty stdout",
                }
            try:
                payload = json.loads(stdout)
            except json.JSONDecodeError as exc:
                last_error = f"rotowire props probe invalid json: {exc}"
            else:
                payload["parse_error"] = None
                return payload
        else:
            last_error = f"rotowire props probe exited with code {result.returncode}"
            if (
                result.returncode in _RETRYABLE_SUBPROCESS_EXIT_CODES
                and attempt < _SUBPROCESS_CRASH_RETRY_ATTEMPTS
            ):
                print(
                    "[subprocess-retry] rotowire props probe exited with "
                    f"{result.returncode}; retrying attempt "
                    f"{attempt + 1}/{_SUBPROCESS_CRASH_RETRY_ATTEMPTS}",
                    file=sys.stderr,
                )
                time.sleep(_SUBPROCESS_CRASH_RETRY_DELAY_SECONDS)
                continue
        break
    return {
        "parsed_rows": 0,
        "latest_action_props_as_of_ts": None,
        "teams": [],
        "team_latest_as_of_ts": {},
        "team_player_digest": {},
        "team_player_count": {},
        "parse_error": last_error,
    }


def _content_digest_by_game_from_teams(
    slate_df: pd.DataFrame,
    source_df: pd.DataFrame,
    *,
    exclude_columns: set[str] | None = None,
) -> dict[int, str | None]:
    if slate_df.empty or source_df.empty:
        return {}
    team_col = next(
        (
            candidate
            for candidate in ("team_tricode", "team_abbreviation", "team")
            if candidate in source_df.columns
        ),
        None,
    )
    if team_col is None:
        return {}
    working = source_df.copy()
    working["_team_tricode"] = working[team_col].map(_normalize_props_team_abbr)
    working = working.loc[working["_team_tricode"].astype(str).str.len() > 0].copy()
    if working.empty:
        return {}
    out: dict[int, str | None] = {}
    for row in slate_df.itertuples(index=False):
        game_id = pd.to_numeric(getattr(row, "game_id", None), errors="coerce")
        if pd.isna(game_id):
            continue
        teams = {
            _normalize_props_team_abbr(getattr(row, attr, None))
            for attr in ("home_team_tricode", "away_team_tricode")
        }
        teams = {team for team in teams if team}
        game_df = working.loc[working["_team_tricode"].isin(teams)].copy()
        out[int(game_id)] = _frame_content_digest(
            game_df, exclude_columns=set(exclude_columns or set()) | {"_team_tricode"}
        )
    return out


def _report_window_status(
    *,
    run_ts: pd.Timestamp,
    per_game_freshness: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    run_ts_et = pd.Timestamp(run_ts).tz_convert(_ET)
    live_games = [
        game for game in per_game_freshness.values() if bool(game.get("is_live_game"))
    ]
    for window in _REPORT_WINDOWS:
        boundary_et = run_ts_et.normalize() + pd.Timedelta(
            hours=int(window["hour"]), minutes=int(window["minute"])
        )
        window_start = boundary_et - pd.Timedelta(minutes=float(window["pre_minutes"]))
        window_end = boundary_et + pd.Timedelta(minutes=float(window["post_minutes"]))
        if not (window_start <= run_ts_et <= window_end):
            continue
        blocking_games: list[dict[str, Any]] = []
        for game in live_games:
            tip_ts = pd.to_datetime(game.get("tip_ts"), utc=True, errors="coerce")
            if pd.isna(tip_ts):
                continue
            if pd.Timestamp(tip_ts).tz_convert(_ET) < boundary_et:
                continue
            injuries = dict(game.get("sources", {}).get("injuries", {}))
            latest = pd.to_datetime(
                injuries.get("latest_as_of_ts"), utc=True, errors="coerce"
            )
            if pd.isna(latest) or pd.Timestamp(latest).tz_convert(_ET) < boundary_et:
                blocking_games.append(
                    {
                        "game_id": int(game["game_id"]),
                        "tip_ts": game.get("tip_ts"),
                        "latest_injuries_ts": injuries.get("latest_as_of_ts"),
                        "injuries_source_used": injuries.get("source_used"),
                    }
                )
        return {
            "active": True,
            "label": str(window["label"]),
            "boundary_ts": boundary_et.tz_convert("UTC").isoformat(),
            "window_start_ts": window_start.tz_convert("UTC").isoformat(),
            "window_end_ts": window_end.tz_convert("UTC").isoformat(),
            "wait_timeout_seconds": int(_REPORT_WINDOW_WAIT_TIMEOUT_SECONDS),
            "wait_interval_seconds": int(_REPORT_WINDOW_WAIT_INTERVAL_SECONDS),
            "needs_wait": bool(blocking_games),
            "blocking_games": blocking_games,
        }
    return {
        "active": False,
        "label": None,
        "boundary_ts": None,
        "window_start_ts": None,
        "window_end_ts": None,
        "wait_timeout_seconds": int(_REPORT_WINDOW_WAIT_TIMEOUT_SECONDS),
        "wait_interval_seconds": int(_REPORT_WINDOW_WAIT_INTERVAL_SECONDS),
        "needs_wait": False,
        "blocking_games": [],
    }


def _lock_window_gate_status(
    *,
    per_game_freshness: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    checked_games = 0
    for game in per_game_freshness.values():
        if not bool(game.get("is_live_game")):
            continue
        minutes_to_tip = game.get("minutes_to_tip")
        if not isinstance(minutes_to_tip, (int, float)):
            continue
        threshold = next(
            (
                item
                for item in _LOCK_WINDOW_THRESHOLDS
                if float(minutes_to_tip) <= float(item["max_minutes_to_tip"])
            ),
            None,
        )
        if threshold is None:
            continue
        checked_games += 1
        sources = dict(game.get("sources", {}))
        injuries = dict(sources.get("injuries", {}))
        lineups = dict(sources.get("lineups", {}))
        game_failures: list[str] = []
        injury_age = injuries.get("age_minutes")
        if injury_age is None or float(injury_age) > float(
            threshold["injuries_max_age_minutes"]
        ):
            game_failures.append(
                "injuries age="
                f"{'missing' if injury_age is None else f'{float(injury_age):.1f}m'} "
                f"> {float(threshold['injuries_max_age_minutes']):.1f}m"
            )
        lineup_age = lineups.get("age_minutes")
        if lineup_age is None or float(lineup_age) > float(
            threshold["lineups_max_age_minutes"]
        ):
            game_failures.append(
                "lineups age="
                f"{'missing' if lineup_age is None else f'{float(lineup_age):.1f}m'} "
                f"> {float(threshold['lineups_max_age_minutes']):.1f}m"
            )
        if game_failures:
            failures.append(
                {
                    "game_id": int(game["game_id"]),
                    "window": str(threshold["window"]),
                    "minutes_to_tip": float(minutes_to_tip),
                    "failures": game_failures,
                    "sources": {
                        "injuries": injuries,
                        "lineups": lineups,
                    },
                }
            )
    return {
        "ok": len(failures) == 0,
        "checked_games": int(checked_games),
        "failures": failures,
        "thresholds": list(_LOCK_WINDOW_THRESHOLDS),
    }


def _detect_stale_authoritative_inputs(
    *,
    frozen_source_freshness: dict[str, Any] | None,
    current_source_freshness: dict[str, Any] | None,
    as_of_ts: str,
) -> dict[str, Any]:
    frozen_games = dict((frozen_source_freshness or {}).get("per_game", {}))
    current_games = dict((current_source_freshness or {}).get("per_game", {}))
    stale_games: list[dict[str, Any]] = []
    tolerance = pd.Timedelta(seconds=_STALE_INPUT_TOLERANCE_SECONDS)
    for game_id, frozen in frozen_games.items():
        current = current_games.get(str(game_id)) or current_games.get(game_id)
        if not isinstance(current, dict):
            continue
        if not bool(current.get("is_live_game")):
            continue
        sources_out: dict[str, dict[str, str | None]] = {}
        for source_name in ("injuries", "lineups", "manual_overrides"):
            frozen_source = dict(frozen.get("sources", {}).get(source_name, {}))
            current_source = dict(current.get("sources", {}).get(source_name, {}))
            frozen_ts = pd.to_datetime(
                frozen_source.get("latest_as_of_ts"), utc=True, errors="coerce"
            )
            current_ts = pd.to_datetime(
                current_source.get("latest_as_of_ts"), utc=True, errors="coerce"
            )
            if pd.isna(current_ts):
                continue
            if (
                pd.isna(frozen_ts)
                or pd.Timestamp(current_ts) > pd.Timestamp(frozen_ts) + tolerance
            ):
                sources_out[source_name] = {
                    "frozen_ts": None
                    if pd.isna(frozen_ts)
                    else pd.Timestamp(frozen_ts).isoformat(),
                    "current_ts": pd.Timestamp(current_ts).isoformat(),
                    "frozen_source_used": frozen_source.get("source_used"),
                    "current_source_used": current_source.get("source_used"),
                }
        if sources_out:
            stale_games.append(
                {
                    "game_id": int(current.get("game_id", game_id)),
                    "tip_ts": current.get("tip_ts"),
                    "minutes_to_tip": current.get("minutes_to_tip"),
                    "sources": sources_out,
                }
            )
    return {
        "checked_at": as_of_ts,
        "stale": len(stale_games) > 0,
        "stale_games": stale_games,
        "tolerance_seconds": int(_STALE_INPUT_TOLERANCE_SECONDS),
    }


def _stable_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _source_digest_payload(source_payload: dict[str, Any]) -> dict[str, Any]:
    source = dict(source_payload)
    content_digest = source.get("content_digest")
    latest_as_of_ts = source.get("latest_as_of_ts")
    if content_digest is not None:
        latest_as_of_ts = None
    return {
        "latest_as_of_ts": latest_as_of_ts,
        "source_used": source.get("source_used"),
        "content_digest": content_digest,
    }


def _compute_per_game_input_digests(
    source_freshness: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    per_game = dict((source_freshness or {}).get("per_game", {}))
    digests: dict[str, dict[str, Any]] = {}
    for game_id, payload in per_game.items():
        game = dict(payload)
        sources = dict(game.get("sources", {}))
        digest_payload = {
            "game_id": int(game.get("game_id", game_id)),
            "tip_ts": game.get("tip_ts"),
            "is_live_game": bool(game.get("is_live_game")),
            "injuries": _source_digest_payload(dict(sources.get("injuries", {}))),
            "lineups": _source_digest_payload(dict(sources.get("lineups", {}))),
            "odds": _source_digest_payload(dict(sources.get("odds", {}))),
            "props": {
                "latest_as_of_ts": dict(sources.get("props", {})).get(
                    "latest_as_of_ts"
                ),
                "source_used": dict(sources.get("props", {})).get("source_used"),
                "player_set_digest": dict(sources.get("props", {})).get(
                    "player_set_digest"
                ),
                "player_set_count": dict(sources.get("props", {})).get(
                    "player_set_count"
                ),
            },
            "roster": _source_digest_payload(dict(sources.get("roster", {}))),
            "manual_overrides": _source_digest_payload(
                dict(sources.get("manual_overrides", {}))
            ),
        }
        digests[str(game_id)] = {
            "digest_sha256": _stable_digest(digest_payload),
            "payload": digest_payload,
        }
    return digests


def _normalize_game_ids(values: list[int] | list[str] | None) -> list[int]:
    if not values:
        return []
    out: list[int] = []
    seen: set[int] = set()
    for value in values:
        num = pd.to_numeric(value, errors="coerce")
        if pd.isna(num):
            continue
        game_id = int(num)
        if game_id in seen:
            continue
        seen.add(game_id)
        out.append(game_id)
    return out


def _frame_content_digest(
    df: pd.DataFrame,
    *,
    exclude_columns: set[str] | None = None,
) -> str | None:
    if df.empty:
        return None
    exclude = set(exclude_columns or set())
    cols = sorted(c for c in df.columns if c not in exclude)
    if not cols:
        return None
    work = df.loc[:, cols].copy()
    for col in cols:
        series = work[col]
        if pd.api.types.is_datetime64_any_dtype(
            series
        ) or pd.api.types.is_datetime64tz_dtype(series):
            work[col] = pd.to_datetime(series, utc=True, errors="coerce").astype(
                "string"
            )
        else:
            work[col] = series.astype("string")
    work = work.sort_values(by=cols, kind="stable", na_position="last").reset_index(
        drop=True
    )
    payload = {
        "columns": cols,
        "rows": work.where(pd.notna(work), None).to_dict(orient="records"),
    }
    return _stable_digest(payload)


def _content_digest_by_game(
    df: pd.DataFrame,
    game_ids: list[int],
    *,
    exclude_columns: set[str] | None = None,
) -> dict[int, str | None]:
    out: dict[int, str | None] = {}
    if not game_ids:
        return out
    if df.empty or "game_id" not in df.columns:
        return {int(gid): None for gid in game_ids}
    gids = pd.to_numeric(df["game_id"], errors="coerce").astype("Int64")
    for game_id in game_ids:
        game_df = df.loc[gids == int(game_id)].copy()
        out[int(game_id)] = _frame_content_digest(
            game_df, exclude_columns=exclude_columns
        )
    return out


def _load_promoted_manifest_payload(
    *, data_root: Path, game_date: str
) -> dict[str, Any] | None:
    pointer_path = (
        data_root
        / "artifacts"
        / "projections"
        / game_date
        / control_plane.LATEST_DIRNAME
        / control_plane.CURRENT_POINTER_NAME
    )
    if not pointer_path.exists():
        return None
    try:
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    manifest_path = pointer.get("manifest_path")
    if not manifest_path:
        return None
    path = Path(str(manifest_path))
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _load_promoted_pointer_payload(*, dataset_dir: Path) -> dict[str, Any] | None:
    for candidate in (
        dataset_dir / control_plane.LATEST_DIRNAME / control_plane.CURRENT_POINTER_NAME,
        dataset_dir / control_plane.LEGACY_POINTER_NAME,
    ):
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _build_publish_superseded_report(
    *,
    run_id: str,
    manifest_path: Path,
    dataset_dir: Path,
) -> dict[str, Any]:
    current_pointer = _load_promoted_pointer_payload(dataset_dir=dataset_dir)
    current_run_id = None if current_pointer is None else current_pointer.get("run_id")
    current_as_of_ts = None if current_pointer is None else current_pointer.get("as_of_ts")
    try:
        manifest_payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        manifest_payload = {}
    candidate_as_of_ts = manifest_payload.get("as_of_ts")

    superseded = False
    reason: str | None = None
    current_ts = pd.to_datetime(current_as_of_ts, utc=True, errors="coerce")
    candidate_ts = pd.to_datetime(candidate_as_of_ts, utc=True, errors="coerce")
    if current_pointer and str(current_run_id or "") != str(run_id):
        if not pd.isna(current_ts) and not pd.isna(candidate_ts):
            if pd.Timestamp(current_ts) > pd.Timestamp(candidate_ts):
                superseded = True
                reason = "newer_pointer_as_of_ts"
            elif pd.Timestamp(current_ts) == pd.Timestamp(candidate_ts):
                superseded = True
                reason = "equal_as_of_ts_other_run_already_published"
        else:
            superseded = True
            reason = "existing_pointer_present_unknown_order"

    return {
        "checked_at": _utc_now_iso(),
        "superseded": bool(superseded),
        "reason": reason,
        "candidate": {
            "run_id": str(run_id),
            "manifest_path": str(manifest_path),
            "as_of_ts": candidate_as_of_ts,
        },
        "current_pointer": current_pointer,
    }


def _build_input_change_set(
    *,
    game_date: str,
    current_source_freshness: dict[str, Any] | None,
    previous_manifest_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    current_digests = _compute_per_game_input_digests(current_source_freshness)
    previous_source_freshness = {}
    previous_run_id = None
    previous_digests: dict[str, dict[str, Any]] = {}
    if isinstance(previous_manifest_payload, dict):
        previous_run_id = previous_manifest_payload.get("run_id")
        previous_source_freshness = dict(
            previous_manifest_payload.get("source_freshness", {})
        )
        previous_digests = dict(
            previous_manifest_payload.get("input_change_set", {}).get(
                "per_game_digests", {}
            )
        )
        if not previous_digests and previous_source_freshness:
            previous_digests = _compute_per_game_input_digests(
                previous_source_freshness
            )

    current_games = dict((current_source_freshness or {}).get("per_game", {}))
    previous_games = dict(previous_source_freshness.get("per_game", {}))

    changed_games: list[dict[str, Any]] = []
    unchanged_games: list[int] = []
    new_games: list[int] = []
    removed_games: list[int] = []

    for game_id, current in current_digests.items():
        previous = previous_digests.get(str(game_id))
        if previous is None:
            new_games.append(int(game_id))
            continue
        if str(previous.get("digest_sha256")) == str(current.get("digest_sha256")):
            unchanged_games.append(int(game_id))
            continue
        current_payload = dict(current.get("payload", {}))
        previous_payload = dict(previous.get("payload", {}))
        changed_sources: list[str] = []
        source_deltas: dict[str, dict[str, Any]] = {}
        for source_name in (
            "injuries",
            "lineups",
            "odds",
            "props",
            "roster",
            "manual_overrides",
        ):
            current_source = dict(current_payload.get(source_name, {}))
            previous_source = dict(previous_payload.get(source_name, {}))
            current_digest = current_source.get("content_digest")
            previous_digest = previous_source.get("content_digest")
            source_used_changed = current_source.get(
                "source_used"
            ) != previous_source.get("source_used")
            content_changed = False
            if current_digest is not None or previous_digest is not None:
                content_changed = str(current_digest) != str(previous_digest)
            else:
                content_changed = current_source.get(
                    "latest_as_of_ts"
                ) != previous_source.get("latest_as_of_ts")
            if source_used_changed or content_changed:
                changed_sources.append(source_name)
                source_deltas[source_name] = {
                    "previous": previous_source,
                    "current": current_source,
                }
        changed_games.append(
            {
                "game_id": int(game_id),
                "changed_sources": changed_sources,
                "current_digest_sha256": current.get("digest_sha256"),
                "previous_digest_sha256": previous.get("digest_sha256"),
                "tip_ts": dict(current_games.get(str(game_id), {})).get("tip_ts")
                or dict(previous_games.get(str(game_id), {})).get("tip_ts"),
                "source_deltas": source_deltas,
            }
        )

    for game_id in previous_digests:
        if str(game_id) not in current_digests:
            removed_games.append(int(game_id))

    return {
        "version": 1,
        "game_date": game_date,
        "previous_run_id": previous_run_id,
        "current_game_count": int(len(current_digests)),
        "changed_game_ids": sorted(item["game_id"] for item in changed_games),
        "unchanged_game_ids": sorted(unchanged_games),
        "new_game_ids": sorted(new_games),
        "removed_game_ids": sorted(removed_games),
        "changed_games": changed_games,
        "per_game_digests": current_digests,
    }


def _build_rerun_plan(
    *,
    game_date: str,
    input_change_set: dict[str, Any],
    current_source_freshness: dict[str, Any] | None,
    previous_manifest_payload: dict[str, Any] | None,
    current_bundle_hash: str,
    current_minutes_selector_path: Path,
    current_rates_selector_path: Path,
    current_ownership_selector_path: Path,
    current_gtv2_inference_config_path: Path | None = None,
    current_gtv2_inference_config_hash: str | None = None,
    manual_target_game_ids: list[int] | None = None,
) -> dict[str, Any]:
    current_games = dict((current_source_freshness or {}).get("per_game", {}))
    current_game_ids = sorted(
        int(v)
        for v in pd.to_numeric(list(current_games.keys()), errors="coerce")
        if not pd.isna(v)
    )
    if previous_manifest_payload is None:
        return {
            "policy_version": 1,
            "game_date": game_date,
            "mode": "full_slate",
            "reason": "no_previous_published_run",
            "target_game_ids": current_game_ids,
            "ignored_changes": [],
        }

    previous_v3 = dict(previous_manifest_payload.get("v3", {}))
    if str(previous_v3.get("bundle_hash") or "") != str(current_bundle_hash):
        return {
            "policy_version": 1,
            "game_date": game_date,
            "mode": "full_slate",
            "reason": "bundle_hash_changed",
            "target_game_ids": current_game_ids,
            "ignored_changes": [],
        }
    if (
        Path(
            str(previous_manifest_payload.get("minutes_current_run_path", ""))
        ).resolve()
        != current_minutes_selector_path.resolve()
    ):
        return {
            "policy_version": 1,
            "game_date": game_date,
            "mode": "full_slate",
            "reason": "minutes_selector_changed",
            "target_game_ids": current_game_ids,
            "ignored_changes": [],
        }
    if (
        Path(str(previous_manifest_payload.get("rates_current_run_path", ""))).resolve()
        != current_rates_selector_path.resolve()
    ):
        return {
            "policy_version": 1,
            "game_date": game_date,
            "mode": "full_slate",
            "reason": "rates_selector_changed",
            "target_game_ids": current_game_ids,
            "ignored_changes": [],
        }
    if (
        Path(
            str(previous_manifest_payload.get("ownership_current_run_path", ""))
        ).resolve()
        != current_ownership_selector_path.resolve()
    ):
        return {
            "policy_version": 1,
            "game_date": game_date,
            "mode": "full_slate",
            "reason": "ownership_selector_changed",
            "target_game_ids": current_game_ids,
            "ignored_changes": [],
        }
    if current_gtv2_inference_config_path is not None:
        previous_gtv2_inference_config_path_raw = str(
            previous_manifest_payload.get("gtv2_inference_current_path", "")
        ).strip()
        previous_gtv2_inference_config_hash = str(
            previous_manifest_payload.get("v3", {}).get(
                "gtv2_inference_current_hash", ""
            )
            or ""
        )
        resolved_previous_gtv2_inference_config_path = (
            Path(previous_gtv2_inference_config_path_raw).resolve()
            if previous_gtv2_inference_config_path_raw
            else None
        )
        if (
            resolved_previous_gtv2_inference_config_path is None
            or resolved_previous_gtv2_inference_config_path
            != current_gtv2_inference_config_path.resolve()
        ):
            return {
                "policy_version": 1,
                "game_date": game_date,
                "mode": "full_slate",
                "reason": "gtv2_inference_config_path_changed",
                "target_game_ids": current_game_ids,
                "ignored_changes": [],
            }
        if (
            not previous_gtv2_inference_config_hash
            or previous_gtv2_inference_config_hash
            != str(current_gtv2_inference_config_hash or "")
        ):
            return {
                "policy_version": 1,
                "game_date": game_date,
                "mode": "full_slate",
                "reason": "gtv2_inference_config_changed",
                "target_game_ids": current_game_ids,
                "ignored_changes": [],
            }

    if input_change_set.get("new_game_ids") or input_change_set.get("removed_game_ids"):
        return {
            "policy_version": 1,
            "game_date": game_date,
            "mode": "full_slate",
            "reason": "slate_composition_changed",
            "target_game_ids": current_game_ids,
            "ignored_changes": [],
        }

    requested_manual_targets = _normalize_game_ids(manual_target_game_ids)
    if requested_manual_targets:
        current_game_set = set(current_game_ids)
        applied_manual_targets = sorted(
            game_id for game_id in requested_manual_targets if game_id in current_game_set
        )
        invalid_manual_targets = sorted(
            game_id for game_id in requested_manual_targets if game_id not in current_game_set
        )
        if not applied_manual_targets:
            return {
                "policy_version": 1,
                "game_date": game_date,
                "mode": "skip",
                "reason": "manual_targets_not_on_slate",
                "target_game_ids": [],
                "ignored_changes": [],
                "manual_trigger": {
                    "requested_game_ids": requested_manual_targets,
                    "applied_game_ids": [],
                    "invalid_game_ids": invalid_manual_targets,
                    "source": "operator",
                },
            }
        manual_mode = (
            "full_slate"
            if len(applied_manual_targets) >= len(current_game_ids)
            else "game_scoped"
        )
        return {
            "policy_version": 1,
            "game_date": game_date,
            "mode": manual_mode,
            "reason": "manual_operator_trigger",
            "target_game_ids": (
                current_game_ids if manual_mode == "full_slate" else applied_manual_targets
            ),
            "ignored_changes": [],
            "manual_trigger": {
                "requested_game_ids": requested_manual_targets,
                "applied_game_ids": applied_manual_targets,
                "invalid_game_ids": invalid_manual_targets,
                "source": "operator",
            },
        }

    changed_games = list(input_change_set.get("changed_games", []))
    if not changed_games:
        return {
            "policy_version": 1,
            "game_date": game_date,
            "mode": "skip",
            "reason": "no_changed_games",
            "target_game_ids": [],
            "ignored_changes": [],
        }

    material_targets: list[int] = []
    ignored_changes: list[dict[str, Any]] = []
    for change in changed_games:
        game_id = int(change.get("game_id"))
        current_game = dict(current_games.get(str(game_id), {}))
        minutes_to_tip = current_game.get("minutes_to_tip")
        changed_sources = [str(v) for v in change.get("changed_sources", [])]
        if not isinstance(minutes_to_tip, (int, float)) or float(minutes_to_tip) <= 0.0:
            ignored_changes.append(
                {
                    "game_id": game_id,
                    "changed_sources": changed_sources,
                    "reason": "game_not_pre_tip",
                }
            )
            continue
        material = False
        material_reason: str | None = None
        if any(
            source in {"injuries", "lineups", "roster"} for source in changed_sources
        ):
            material = True
            material_reason = "always_material_source_changed"
        elif "odds" in changed_sources and float(minutes_to_tip) <= float(
            _ODDS_MATERIALITY_MAX_MINUTES_TO_TIP
        ):
            material = True
            material_reason = "odds_change_within_tip_window"
        elif "props" in changed_sources and float(minutes_to_tip) <= float(
            _PROPS_PLAYER_SET_EXPANSION_MAX_MINUTES_TO_TIP
        ):
            props_delta = dict(change.get("source_deltas", {})).get("props", {})
            props_previous = dict(props_delta.get("previous", {}))
            props_current = dict(props_delta.get("current", {}))
            prev_digest = str(props_previous.get("player_set_digest") or "").strip()
            curr_digest = str(props_current.get("player_set_digest") or "").strip()
            prev_count_num = pd.to_numeric(
                props_previous.get("player_set_count"), errors="coerce"
            )
            curr_count_num = pd.to_numeric(
                props_current.get("player_set_count"), errors="coerce"
            )
            prev_count = int(prev_count_num) if pd.notna(prev_count_num) else 0
            curr_count = int(curr_count_num) if pd.notna(curr_count_num) else 0
            if curr_digest and curr_digest != prev_digest and curr_count > prev_count:
                material = True
                material_reason = "props_player_set_expanded"
        if material:
            if material_reason is not None:
                change["material_reason"] = material_reason
            material_targets.append(game_id)
        else:
            ignored_changes.append(
                {
                    "game_id": game_id,
                    "changed_sources": changed_sources,
                    "reason": "changes_below_materiality_policy",
                    "minutes_to_tip": minutes_to_tip,
                }
            )

    material_targets = sorted(set(material_targets))
    if not material_targets:
        return {
            "policy_version": 1,
            "game_date": game_date,
            "mode": "skip",
            "reason": "no_material_game_changes",
            "target_game_ids": [],
            "ignored_changes": ignored_changes,
            "materiality_policy": {
                "always_material_sources": ["injuries", "lineups", "roster"],
                "odds_materiality_max_minutes_to_tip": float(
                    _ODDS_MATERIALITY_MAX_MINUTES_TO_TIP
                ),
                "props_auto_trigger_enabled": False,
                "props_player_set_expansion_enabled": True,
                "props_player_set_expansion_max_minutes_to_tip": float(
                    _PROPS_PLAYER_SET_EXPANSION_MAX_MINUTES_TO_TIP
                ),
            },
        }
    if len(material_targets) >= len(current_game_ids):
        mode = "full_slate"
        target_game_ids = current_game_ids
        reason = "all_games_material"
    else:
        mode = "game_scoped"
        target_game_ids = material_targets
        reason = "material_game_subset"
    return {
        "policy_version": 1,
        "game_date": game_date,
        "mode": mode,
        "reason": reason,
        "target_game_ids": target_game_ids,
        "ignored_changes": ignored_changes,
        "materiality_policy": {
            "always_material_sources": ["injuries", "lineups", "roster"],
            "odds_materiality_max_minutes_to_tip": float(
                _ODDS_MATERIALITY_MAX_MINUTES_TO_TIP
            ),
            "props_auto_trigger_enabled": False,
            "props_player_set_expansion_enabled": True,
            "props_player_set_expansion_max_minutes_to_tip": float(
                _PROPS_PLAYER_SET_EXPANSION_MAX_MINUTES_TO_TIP
            ),
        },
    }


def _resolve_report_window_wait_policy(
    *,
    as_of_ts_override: str | None,
    replay_mode: bool,
    manual_target_game_ids: list[int] | None,
) -> tuple[bool, str]:
    """Decide whether report-window freshness waits are allowed for this run."""
    if as_of_ts_override is not None:
        return False, "as_of_ts_override"
    if bool(replay_mode):
        return False, "replay_mode"
    manual_targets = _normalize_game_ids(manual_target_game_ids)
    if manual_targets:
        return False, "manual_target_game_rerun"
    return True, "eligible"


def _filter_to_target_games(
    df: pd.DataFrame, target_game_ids: list[int] | None
) -> pd.DataFrame:
    if df.empty or not target_game_ids or "game_id" not in df.columns:
        return df.copy()
    gids = pd.to_numeric(df["game_id"], errors="coerce").astype("Int64")
    return df.loc[gids.isin(target_game_ids)].copy()


def _split_frame_by_game(df: pd.DataFrame) -> list[tuple[int, pd.DataFrame]]:
    if df.empty or "game_id" not in df.columns:
        return []
    gids = pd.to_numeric(df["game_id"], errors="coerce").astype("Int64")
    game_ids = _normalize_game_ids(gids.dropna().astype(int).tolist())
    out: list[tuple[int, pd.DataFrame]] = []
    for game_id in game_ids:
        game_df = df.loc[gids == int(game_id)].copy()
        if game_df.empty:
            continue
        out.append((int(game_id), game_df))
    return out


def _per_game_request_seed(base_seed: int, game_id: int) -> int:
    """
    Derive a deterministic per-game seed so per-game Triton requests do not
    replay identical RNG streams across different games.
    """
    mixed = (int(base_seed) * 1_000_003 + int(game_id) * 97_531) & 0x7FFFFFFF
    if mixed == 0:
        return max(1, int(base_seed))
    return int(mixed)


def _sort_for_stable_write(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    preferred = [
        c
        for c in ("world_idx", "game_date", "game_id", "team_id", "player_id")
        if c in df.columns
    ]
    if not preferred:
        return df.reset_index(drop=True)
    return df.sort_values(preferred, kind="stable").reset_index(drop=True)


def _stream_validate_parquet(
    path: Path,
    *,
    expected_rows: int | None = None,
    required_cols: tuple[str, ...] = (),
) -> dict[str, Any]:
    def _is_retryable_validation_error(exc: Exception) -> bool:
        text = str(exc).lower()
        return (
            "unexpected end of stream" in text
            or "end of stream" in text
            or "corrupt snappy compressed data" in text
            or "corrupt data page" in text
            or "invalid number of bytes" in text
            or "page was smaller than expected" in text
        )

    max_attempts = 3
    columns: tuple[str, ...] = ()
    for attempt in range(1, max_attempts + 1):
        try:
            parquet_file = pq.ParquetFile(path)
            columns = tuple(str(name) for name in parquet_file.schema_arrow.names)
            break
        except Exception as exc:
            if attempt < max_attempts and _is_retryable_validation_error(exc):
                time.sleep(0.5 * attempt)
                continue
            raise RuntimeError(f"failed to open parquet for validation: {path}") from exc

    missing = [col for col in required_cols if col not in columns]
    if missing:
        raise RuntimeError(
            f"validated parquet missing required columns {missing}: {path}"
        )

    row_count = 0
    for attempt in range(1, max_attempts + 1):
        row_count = 0
        try:
            parquet_file = pq.ParquetFile(path)
            for batch in parquet_file.iter_batches(batch_size=65536, use_threads=False):
                row_count += int(batch.num_rows)
            break
        except Exception as exc:
            if attempt < max_attempts and _is_retryable_validation_error(exc):
                time.sleep(0.5 * attempt)
                continue
            raise RuntimeError(f"failed to stream-validate parquet contents: {path}") from exc

    if expected_rows is not None and row_count != int(expected_rows):
        raise RuntimeError(
            f"validated parquet row count mismatch for {path}: "
            f"expected={expected_rows} actual={row_count}"
        )

    return {
        "path": str(path),
        "rows": int(row_count),
        "columns": list(columns),
    }


def _atomic_write_validated_parquet(
    df: pd.DataFrame,
    path: Path,
    *,
    required_cols: tuple[str, ...] = (),
    compression: str | None = "snappy",
    row_group_size: int | None = None,
) -> dict[str, Any]:
    def _is_retryable_validation_error(exc: Exception) -> bool:
        text = str(exc).lower()
        return (
            "corrupt snappy compressed data" in text
            or "failed to stream-validate parquet contents" in text
            or "failed to open parquet for validation" in text
            or "unexpected end of stream" in text
            or "end of stream" in text
        )

    compression_schedule: list[str | None] = [compression]
    if compression == "snappy":
        # Snappy corruption has appeared intermittently on large world-matrix writes
        # in long-lived worker processes. Fall back to safer codecs before failing.
        compression_schedule.extend(["zstd", None])
    max_attempts = max(1, len(compression_schedule))
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, max_attempts + 1):
        current_compression = compression_schedule[min(attempt - 1, len(compression_schedule) - 1)]
        tmp = path.with_suffix(
            f".tmp.{control_plane.canonical_run_id()}.{os.getpid()}.{attempt}.parquet"
        )
        try:
            write_kwargs: dict[str, Any] = {"index": False, "compression": current_compression}
            if row_group_size is not None:
                write_kwargs["row_group_size"] = int(row_group_size)
            df.to_parquet(tmp, **write_kwargs)
            validation = _stream_validate_parquet(
                tmp,
                expected_rows=int(len(df)),
                required_cols=required_cols,
            )
            tmp.replace(path)
            return validation
        except Exception as exc:
            retryable = _is_retryable_validation_error(exc)
            is_last_attempt = attempt >= max_attempts
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass
            if (not retryable) or is_last_attempt:
                raise
            if current_compression != compression_schedule[min(attempt, len(compression_schedule) - 1)]:
                print(
                    "[parquet-write] retrying with fallback compression "
                    f"{compression_schedule[min(attempt, len(compression_schedule) - 1)]} "
                    f"for path={path}"
                )
            time.sleep(0.2 * attempt)

    # Unreachable, loop either returns on success or raises on terminal failure.
    raise RuntimeError(f"unreachable atomic parquet write state for {path}")


def _distinct_game_count(df: pd.DataFrame) -> int:
    if "game_id" not in df.columns:
        return 0
    gids = pd.to_numeric(df["game_id"], errors="coerce")
    return int(gids.dropna().nunique())


def _coerce_world_game_date(
    worlds_df: pd.DataFrame,
    *,
    game_date: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Force a canonical slate date on generated world rows."""
    try:
        canonical_game_date = pd.Timestamp(str(game_date)).date().isoformat()
    except Exception:
        canonical_game_date = str(game_date)

    if worlds_df.empty:
        return worlds_df, {
            "applied": False,
            "reason": "empty_worlds",
            "canonical_game_date": canonical_game_date,
            "rows": 0,
            "normalized_rows": 0,
        }

    out = worlds_df
    row_count = int(len(out))
    had_game_date_column = "game_date" in out.columns
    sample_noncanonical_values: list[str] = []
    unique_input_values = 0
    normalized_rows = row_count

    if had_game_date_column:
        game_date_series = out["game_date"]
        canonical_mask = game_date_series.eq(canonical_game_date)
        null_mask = game_date_series.isna()
        noncanonical_mask = (~canonical_mask) | null_mask
        normalized_rows = int(noncanonical_mask.sum())
        unique_input_values = int(game_date_series.nunique(dropna=False))
        if normalized_rows > 0:
            sample_noncanonical_values = [
                str(value)
                for value in game_date_series.loc[noncanonical_mask]
                .head(5)
                .tolist()
            ]
    out["game_date"] = canonical_game_date

    return out, {
        "applied": bool((not had_game_date_column) or normalized_rows > 0),
        "canonical_game_date": canonical_game_date,
        "rows": row_count,
        "had_game_date_column": bool(had_game_date_column),
        "normalized_rows": int(normalized_rows),
        "unique_input_values": int(unique_input_values),
        "sample_noncanonical_values": sample_noncanonical_values,
    }


def _sanitize_frame_to_expected_keys(
    df: pd.DataFrame,
    *,
    expected_keys_df: pd.DataFrame,
    key_cols: Sequence[str],
    label: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    key_cols = tuple(str(col) for col in key_cols)
    key_cols_list = list(key_cols)
    if df.empty:
        return (
            df.iloc[0:0].reset_index(drop=True),
            {
                "label": str(label),
                "rows_in": 0,
                "rows_out": 0,
                "dropped_null_key_rows": 0,
                "dropped_unexpected_key_rows": 0,
                "expected_distinct_keys": 0,
            },
        )

    missing_df = [col for col in key_cols if col not in df.columns]
    if missing_df:
        raise RuntimeError(f"{label} missing required key columns: {missing_df}")
    missing_expected = [col for col in key_cols if col not in expected_keys_df.columns]
    if missing_expected:
        raise RuntimeError(
            f"{label} expected-keys frame missing required columns: {missing_expected}"
        )

    # Avoid deep-copying large heterogeneous frames; pandas block-level copies
    # have crashed intermittently in production workers.
    work_keys = pd.DataFrame(index=df.index)
    for col in key_cols:
        work_keys[col] = pd.to_numeric(df[col], errors="coerce")

    expected = expected_keys_df.loc[:, key_cols_list].copy()
    for col in key_cols:
        expected[col] = pd.to_numeric(expected[col], errors="coerce")

    rows_in = int(len(df))
    null_mask = work_keys.loc[:, key_cols_list].isna().any(axis=1)
    dropped_null_key_rows = int(null_mask.sum())
    keep_non_null = ~null_mask.to_numpy(dtype=bool, copy=False)
    # NOTE: Avoid DataFrame.take on very large mixed-type world frames.
    # In production this has intermittently produced silent column corruption
    # (implausible scientific-notation spikes post-sanitize).
    non_null_df = df.loc[keep_non_null]
    work_keys = work_keys.loc[keep_non_null]
    for col in key_cols:
        work_keys[col] = work_keys[col].astype("int64", copy=False)

    expected = (
        expected.dropna(subset=key_cols_list)
        .drop_duplicates(ignore_index=True)
        .reset_index(drop=True)
    )
    for col in key_cols:
        expected[col] = expected[col].astype("int64", copy=False)
    expected_distinct_keys = int(len(expected))

    if expected.empty:
        return (
            df.iloc[0:0].reset_index(drop=True),
            {
                "label": str(label),
                "rows_in": rows_in,
                "rows_out": 0,
                "dropped_null_key_rows": dropped_null_key_rows,
                "dropped_unexpected_key_rows": int(len(work_keys)),
                "expected_distinct_keys": 0,
            },
        )

    # NOTE: structured-dtype + np.isin triggers an IndexError in numpy's in1d
    # fast-path when the test_elements array is small (e.g. 7 expected keys) —
    # it tries to use raw int64 NBA IDs as array indices.  Use a left merge with
    # indicator instead: O(n log n), vectorised, no native crashes.
    work_key_df = pd.DataFrame(
        work_keys.loc[:, key_cols_list].to_numpy(dtype=np.int64, copy=False),
        columns=key_cols_list,
    )
    expected_key_df = pd.DataFrame(
        expected.loc[:, key_cols_list].to_numpy(dtype=np.int64, copy=False),
        columns=key_cols_list,
    ).drop_duplicates()
    merged_indicator = work_key_df.merge(
        expected_key_df, on=key_cols_list, how="left", indicator=True
    )
    keep_mask = (merged_indicator["_merge"] == "both").to_numpy(dtype=bool)
    dropped_unexpected_key_rows = int(np.count_nonzero(~keep_mask))
    merged = non_null_df.loc[keep_mask].reset_index(drop=True)
    merged_keys = (
        work_keys.loc[keep_mask, key_cols_list].reset_index(drop=True)
    )
    for col in key_cols:
        merged[col] = merged_keys[col].astype("int64", copy=False)

    return (
        merged,
        {
            "label": str(label),
            "rows_in": rows_in,
            "rows_out": int(len(merged)),
            "dropped_null_key_rows": dropped_null_key_rows,
            "dropped_unexpected_key_rows": dropped_unexpected_key_rows,
            "expected_distinct_keys": expected_distinct_keys,
        },
    )


def _concat_frames_without_pandas_concat(
    frames: Sequence[pd.DataFrame],
) -> pd.DataFrame:
    """Concatenate frames via NumPy arrays to avoid pandas concat segfaults."""
    if not frames:
        return pd.DataFrame()
    if len(frames) == 1:
        return frames[0].reset_index(drop=True)

    all_cols: list[str] = []
    seen_cols: set[str] = set()
    row_counts: list[int] = []
    for frame in frames:
        row_counts.append(int(len(frame)))
        for col in frame.columns:
            col_name = str(col)
            if col_name not in seen_cols:
                seen_cols.add(col_name)
                all_cols.append(col_name)

    out_data: dict[str, np.ndarray] = {}
    for col in all_cols:
        pieces: list[np.ndarray] = []
        for frame, row_count in zip(frames, row_counts, strict=False):
            if col in frame.columns:
                pieces.append(np.asarray(frame[col].to_numpy(copy=False)))
            else:
                pieces.append(np.full(row_count, np.nan, dtype=object))
        try:
            out_data[col] = np.concatenate(pieces, axis=0)
        except Exception:
            out_data[col] = np.concatenate(
                [np.asarray(piece, dtype=object) for piece in pieces],
                axis=0,
            )

    return pd.DataFrame(out_data, columns=all_cols)


def _left_overlay_from_source_by_keys(
    base_df: pd.DataFrame,
    *,
    source_df: pd.DataFrame,
    key_cols: Sequence[str],
    value_cols: Sequence[str],
    label: str,
    copy_base: bool = True,
) -> pd.DataFrame:
    key_cols = tuple(str(col) for col in key_cols)
    value_cols = [str(col) for col in value_cols if str(col) in source_df.columns]
    if base_df.empty or source_df.empty or not value_cols:
        return base_df

    missing_base = [col for col in key_cols if col not in base_df.columns]
    missing_source = [col for col in key_cols if col not in source_df.columns]
    if missing_base or missing_source:
        raise RuntimeError(
            f"{label} missing join columns; "
            f"base_missing={missing_base} source_missing={missing_source}"
        )

    base = base_df.copy() if bool(copy_base) else base_df
    source = source_df.loc[:, list(key_cols) + value_cols].copy()
    for col in key_cols:
        base[col] = pd.to_numeric(base[col], errors="coerce")
        source[col] = pd.to_numeric(source[col], errors="coerce")

    source = source.dropna(subset=list(key_cols))
    if source.empty:
        return base
    source = source.drop_duplicates(subset=list(key_cols), keep="last").reset_index(
        drop=True
    )
    for col in key_cols:
        source[col] = source[col].astype("int64", copy=False)

    base_valid_mask = ~base.loc[:, list(key_cols)].isna().any(axis=1)
    if not bool(base_valid_mask.any()):
        return base
    base_valid_positions = np.flatnonzero(base_valid_mask.to_numpy())
    base_keys_valid = base.loc[base_valid_mask, list(key_cols)].copy()
    for col in key_cols:
        base_keys_valid[col] = base_keys_valid[col].astype("int64", copy=False)

    source_key_arrays = [
        source[col].to_numpy(dtype=np.int64, copy=False)
        for col in key_cols
    ]
    base_key_arrays = [
        base_keys_valid[col].to_numpy(dtype=np.int64, copy=False)
        for col in key_cols
    ]
    n_source = int(len(source))
    combined_key_arrays = [
        np.concatenate([source_arr, base_arr])
        for source_arr, base_arr in zip(source_key_arrays, base_key_arrays, strict=False)
    ]
    combined_codes, _ = _factorize_int_key_arrays_preserve_order(*combined_key_arrays)
    source_codes = combined_codes[:n_source]
    base_codes = combined_codes[n_source:]
    if source_codes.size <= 0 or base_codes.size <= 0:
        return base
    max_code = int(max(int(source_codes.max(initial=-1)), int(base_codes.max(initial=-1))))
    if max_code < 0:
        return base
    source_pos_by_code = np.full(max_code + 1, -1, dtype=np.int64)
    source_pos_by_code[source_codes] = np.arange(n_source, dtype=np.int64)
    source_positions_for_base = source_pos_by_code[base_codes]
    hit_mask = source_positions_for_base >= 0
    if not bool(hit_mask.any()):
        return base

    hit_base_positions = base_valid_positions[hit_mask]
    hit_source_positions = source_positions_for_base[hit_mask]

    for col in value_cols:
        if col not in base.columns:
            base[col] = pd.NA
        source_values = source[col].to_numpy(copy=False)[hit_source_positions]
        source_notna = pd.notna(source_values)
        if not bool(source_notna.any()):
            continue
        col_idx = base.columns.get_loc(col)
        base.iloc[hit_base_positions[source_notna], col_idx] = source_values[source_notna]

    return base


def _validate_parquet_key_contract(
    path: Path,
    *,
    expected_keys_df: pd.DataFrame,
    key_cols: Sequence[str],
    label: str,
) -> dict[str, Any]:
    key_cols = tuple(str(col) for col in key_cols)
    df = pd.read_parquet(path, columns=list(key_cols))
    _, report = _sanitize_frame_to_expected_keys(
        df,
        expected_keys_df=expected_keys_df,
        key_cols=key_cols,
        label=label,
    )
    if report["dropped_null_key_rows"] > 0 or report["dropped_unexpected_key_rows"] > 0:
        raise RuntimeError(
            f"{label} key contract failed for {path}: "
            f"null_key_rows={report['dropped_null_key_rows']} "
            f"unexpected_key_rows={report['dropped_unexpected_key_rows']}"
        )
    report["path"] = str(path)
    return report


def _group_mean_by_keys_without_pandas_groupby(
    df: pd.DataFrame,
    *,
    key_cols: Sequence[str],
    value_cols: Sequence[str],
    label: str,
) -> pd.DataFrame:
    """
    Compute grouped means with NumPy to avoid pandas cython groupby crashes on
    very large frames in long-running worker processes.
    """
    key_cols = tuple(str(col) for col in key_cols)
    value_cols = [str(col) for col in value_cols]
    missing_keys = [col for col in key_cols if col not in df.columns]
    missing_values = [col for col in value_cols if col not in df.columns]
    if missing_keys or missing_values:
        raise RuntimeError(
            f"{label} missing columns: key_missing={missing_keys} "
            f"value_missing={missing_values}"
        )

    if df.empty:
        return pd.DataFrame(columns=list(key_cols) + value_cols)

    work = df.loc[:, list(key_cols) + value_cols].copy()
    for col in key_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=list(key_cols))
    if work.empty:
        return pd.DataFrame(columns=list(key_cols) + value_cols)
    for col in key_cols:
        work[col] = work[col].astype("int64", copy=False)

    key_arrays = [
        work[col].to_numpy(dtype=np.int64, copy=False)
        for col in key_cols
    ]
    codes, unique_key_arrays = _factorize_int_key_arrays_preserve_order(*key_arrays)
    out = pd.DataFrame(
        {
            col: unique_key_arrays[idx]
            for idx, col in enumerate(key_cols)
        }
    )
    group_count = int(len(out))
    for col in value_cols:
        values = pd.to_numeric(work[col], errors="coerce").to_numpy(dtype=float, copy=False)
        valid = ~np.isnan(values)
        sums = np.bincount(
            codes[valid],
            weights=values[valid],
            minlength=group_count,
        )
        counts = np.bincount(codes[valid], minlength=group_count)
        means = np.divide(
            sums,
            counts,
            out=np.full(group_count, np.nan, dtype=float),
            where=counts > 0,
        )
        out[col] = means
    return out.reset_index(drop=True)


def _factorize_int_key_arrays_preserve_order(
    *key_arrays: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Factorize integer key arrays without pandas MultiIndex operations.

    Pandas MultiIndex factorization has shown instability on very large live
    frames in long-running workers. This helper avoids that path.
    """
    if len(key_arrays) <= 0:
        return np.array([], dtype=np.int64), []
    row_count = int(len(key_arrays[0]))
    if row_count <= 0:
        empty_codes = np.array([], dtype=np.int64)
        empty_uniques = [np.array([], dtype=np.int64) for _ in key_arrays]
        return empty_codes, empty_uniques

    arrays: list[np.ndarray] = []
    for arr in key_arrays:
        arrays.append(np.asarray(arr, dtype=np.int64))
    for arr in arrays[1:]:
        if len(arr) != row_count:
            raise RuntimeError("key array lengths must match for factorization")

    # Build group ids from a stable lexicographic sort instead of relying on
    # np.unique(..., return_inverse=True) for structured dtypes. That inverse
    # path has produced corrupted indices intermittently on live worker data.
    sort_order = np.lexsort(tuple(arrays[::-1]))
    sorted_arrays = [arr[sort_order] for arr in arrays]

    unique_mask = np.ones(row_count, dtype=bool)
    for arr in sorted_arrays:
        unique_mask[1:] &= arr[1:] == arr[:-1]
    unique_mask = ~unique_mask
    unique_mask[0] = True

    group_starts = np.flatnonzero(unique_mask)
    if len(group_starts) <= 0:
        empty_codes = np.array([], dtype=np.int64)
        empty_uniques = [np.array([], dtype=np.int64) for _ in key_arrays]
        return empty_codes, empty_uniques

    group_ends = np.r_[group_starts[1:], row_count]
    first_positions = np.minimum.reduceat(sort_order, group_starts)
    first_seen_order = np.argsort(first_positions, kind="mergesort")

    inverse = np.empty(row_count, dtype=np.int64)
    for new_code, sorted_group_idx in enumerate(first_seen_order):
        start = int(group_starts[sorted_group_idx])
        end = int(group_ends[sorted_group_idx])
        inverse[sort_order[start:end]] = int(new_code)

    unique_key_arrays = [
        sorted_arrays[idx][group_starts][first_seen_order].astype(np.int64, copy=False)
        for idx in range(len(arrays))
    ]
    return inverse.astype(np.int64, copy=False), unique_key_arrays


_INT64_FLOAT_MIN = float(np.iinfo(np.int64).min)
_INT64_FLOAT_MAX = float(np.iinfo(np.int64).max)


def _valid_int64_numeric_mask(values: np.ndarray) -> np.ndarray:
    """Return rows that are finite, int64-bounded, and near-integral."""
    valid = np.isfinite(values)
    if not bool(np.any(valid)):
        return valid
    valid &= values >= _INT64_FLOAT_MIN
    valid &= values <= _INT64_FLOAT_MAX
    if not bool(np.any(valid)):
        return valid
    valid_positions = np.flatnonzero(valid)
    rounded = np.rint(values[valid_positions])
    integral = np.abs(values[valid_positions] - rounded) <= _WORLD_CONTRACT_TOL
    if bool(np.all(integral)):
        return valid
    valid[valid_positions[~integral]] = False
    return valid


def _team_minutes_sums_without_pandas_groupby(
    *,
    world_idx_col: pd.Series,
    game_id_col: pd.Series,
    team_id_col: pd.Series,
    minutes_col: pd.Series,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-(world_idx, game_id, team_id) minute sums without pandas groupby.
    """
    world_raw = pd.to_numeric(world_idx_col, errors="coerce").to_numpy(dtype=float, copy=False)
    game_raw = pd.to_numeric(game_id_col, errors="coerce").to_numpy(dtype=float, copy=False)
    team_raw = pd.to_numeric(team_id_col, errors="coerce").to_numpy(dtype=float, copy=False)
    minutes = pd.to_numeric(minutes_col, errors="coerce").fillna(0.0).to_numpy(dtype=float, copy=False)
    valid = (
        _valid_int64_numeric_mask(world_raw)
        & _valid_int64_numeric_mask(game_raw)
        & _valid_int64_numeric_mask(team_raw)
    )
    if not bool(np.any(valid)):
        empty_int = np.array([], dtype=np.int64)
        empty_float = np.array([], dtype=float)
        return empty_int, empty_int, empty_int, empty_float

    world_vals = np.rint(world_raw[valid]).astype(np.int64, copy=False)
    game_vals = np.rint(game_raw[valid]).astype(np.int64, copy=False)
    team_vals = np.rint(team_raw[valid]).astype(np.int64, copy=False)
    minute_vals = minutes[valid]
    group_codes, unique_key_arrays = _factorize_int_key_arrays_preserve_order(
        world_vals,
        game_vals,
        team_vals,
    )
    if len(unique_key_arrays) < 3 or len(unique_key_arrays[0]) <= 0:
        empty_int = np.array([], dtype=np.int64)
        empty_float = np.array([], dtype=float)
        return empty_int, empty_int, empty_int, empty_float

    group_count = int(len(unique_key_arrays[0]))
    minute_sums = np.bincount(
        group_codes,
        weights=minute_vals,
        minlength=group_count,
    ).astype(float, copy=False)
    uniq_world = unique_key_arrays[0]
    uniq_game = unique_key_arrays[1]
    uniq_team = unique_key_arrays[2]
    return uniq_world, uniq_game, uniq_team, minute_sums


def _load_fallback_merge_baseline(
    *,
    current_path: Path,
    failed_previous_path: Path,
) -> tuple[pd.DataFrame, Path] | None:
    dataset_dir = current_path.parent.parent
    current_run_dir = current_path.parent.name
    filename = current_path.name
    candidates: list[tuple[int, str, Path]] = []

    for run_dir in sorted(dataset_dir.glob("run=*"), reverse=True):
        if not run_dir.is_dir() or run_dir.name >= current_run_dir:
            continue
        candidate_path = run_dir / filename
        if candidate_path == failed_previous_path or not candidate_path.exists():
            continue
        try:
            probe = pd.read_parquet(candidate_path, columns=["game_id"])
        except Exception:
            continue
        candidates.append((_distinct_game_count(probe), run_dir.name, candidate_path))

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    for game_count, _, candidate_path in candidates:
        try:
            fallback_df = pd.read_parquet(candidate_path)
        except Exception:
            continue
        print(
            "[materialize] promoted baseline unreadable; "
            f"falling back from {failed_previous_path} to {candidate_path} "
            f"(distinct_games={game_count})"
        )
        return fallback_df, candidate_path
    return None


def _merge_parquet_for_target_games(
    *,
    current_path: Path,
    previous_path: Path | None,
    target_game_ids: list[int],
) -> pd.DataFrame:
    current_df = pd.DataFrame()
    fallback_loaded_current = False
    allow_stale_merge_fallback = (
        str(os.environ.get("PROJECTIONS_ALLOW_STALE_MERGE_FALLBACK", "0"))
        .strip()
        .lower()
        in {"1", "true", "yes", "on"}
    )
    if current_path.exists():
        try:
            current_df = pd.read_parquet(current_path)
        except Exception as exc:
            fallback = _load_fallback_merge_baseline(
                current_path=current_path,
                failed_previous_path=current_path,
            )
            if fallback is None:
                raise
            fallback_df, fallback_path = fallback
            if not allow_stale_merge_fallback:
                raise RuntimeError(
                    "[materialize] current run parquet unreadable: "
                    f"{current_path}; fallback candidate={fallback_path}. "
                    "Refusing implicit stale fallback (fail-closed). "
                    "Set PROJECTIONS_ALLOW_STALE_MERGE_FALLBACK=1 for "
                    "emergency stale fallback mode."
                ) from exc
            current_df = fallback_df
            fallback_loaded_current = True
            print(
                "[materialize][stale-fallback] current run parquet unreadable; "
                f"using fallback {fallback_path} instead of {current_path}"
            )
    if previous_path is None or not previous_path.exists():
        merged = current_df
    elif fallback_loaded_current:
        # When current_path is unreadable, the fallback frame already comes from a
        # historical baseline run. Skip extra merge against previous_path to avoid
        # duplicate key rows and keep output publishable.
        merged = current_df
    else:
        try:
            previous_df = pd.read_parquet(previous_path)
        except Exception:
            fallback = _load_fallback_merge_baseline(
                current_path=current_path,
                failed_previous_path=previous_path,
            )
            if fallback is None:
                raise
            previous_df, previous_path = fallback
        previous_keep = previous_df
        if "game_id" in previous_df.columns and target_game_ids:
            gids = pd.to_numeric(previous_df["game_id"], errors="coerce").astype(
                "Int64"
            )
            previous_keep = previous_df.loc[~gids.isin(target_game_ids)].copy()
        merged = _concat_frames_without_pandas_concat([previous_keep, current_df])
    merged = _sort_for_stable_write(merged)
    required_cols = (
        ("game_id", "team_id", "player_id")
        if {"game_id", "team_id", "player_id"}.issubset(merged.columns)
        else tuple()
    )
    _atomic_write_validated_parquet(
        merged,
        current_path,
        required_cols=required_cols,
    )
    return merged


def _summarize_world_contracts_from_frame(worlds_df: pd.DataFrame) -> dict[str, Any]:
    if worlds_df.empty:
        return {
            "team_minutes_not_240": 0,
            "team_minutes_total_checks": 0,
            "team_minutes_max_abs_drift": 0.0,
            "minutes_negative": 0,
            "minutes_over_48": 0,
            "negative_stats": 0,
            "fg2m_gt_fga2": 0,
            "fg3m_gt_fga3": 0,
            "ftm_gt_fta": 0,
            "inactive_nonzero_stats": 0,
            "inactive_nonzero_fpts_proxy": 0,
            "max_abs_stat_value": 0.0,
            "extreme_stat_rows_over_1e6": 0,
        }
    df = worlds_df.copy()
    numeric_cols = [
        "minutes",
        "fga2",
        "fg2m",
        "fga3",
        "fg3m",
        "fta",
        "ftm",
        "oreb",
        "dreb",
        "ast",
        "stl",
        "blk",
        "tov",
        "pf",
        "pts",
        "reb",
        "dk_fpts",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    present_numeric = [col for col in numeric_cols if col in df.columns]
    if present_numeric:
        numeric_arrays = [
            np.abs(pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float, copy=False))
            for col in present_numeric
        ]
        if numeric_arrays:
            stacked_numeric = np.column_stack(numeric_arrays)
            row_max = stacked_numeric.max(axis=1) if stacked_numeric.size > 0 else np.array([], dtype=float)
            max_abs_stat_value = float(row_max.max()) if row_max.size > 0 else 0.0
            extreme_stat_rows_over_1e6 = int(np.count_nonzero(row_max > 1e6))
        else:
            max_abs_stat_value = 0.0
            extreme_stat_rows_over_1e6 = 0
    else:
        max_abs_stat_value = 0.0
        extreme_stat_rows_over_1e6 = 0
    if {"world_idx", "game_id", "team_id", "minutes"}.issubset(df.columns):
        _, _, _, minute_sums = _team_minutes_sums_without_pandas_groupby(
            world_idx_col=df["world_idx"],
            game_id_col=df["game_id"],
            team_id_col=df["team_id"],
            minutes_col=df["minutes"],
        )
        if minute_sums.size > 0:
            team_minute_delta = np.abs(minute_sums - 240.0)
            team_minutes_not_240 = int(np.count_nonzero(team_minute_delta > _WORLD_CONTRACT_TOL))
            team_minutes_total_checks = int(minute_sums.size)
            team_minutes_max_abs_drift = float(np.max(team_minute_delta))
        else:
            team_minutes_not_240 = 0
            team_minutes_total_checks = 0
            team_minutes_max_abs_drift = 0.0
    else:
        team_minutes_not_240 = 0
        team_minutes_total_checks = 0
        team_minutes_max_abs_drift = 0.0
    negative_stats = 0
    for col in ("pts", "reb", "ast", "stl", "blk", "tov"):
        if col in df.columns:
            negative_stats += int((df[col] < -_WORLD_CONTRACT_TOL).sum())
    if "active" in df.columns:
        inactive_mask = (
            pd.to_numeric(df["active"], errors="coerce").fillna(0).to_numpy(dtype=int, copy=False) <= 0
        )
        stat_cols = [
            c
            for c in (
                "pts",
                "reb",
                "ast",
                "stl",
                "blk",
                "tov",
                "fga2",
                "fg2m",
                "fga3",
                "fg3m",
                "fta",
                "ftm",
            )
            if c in df.columns
        ]
        if stat_cols:
            stat_arrays = [
                np.abs(pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float, copy=False))
                for col in stat_cols
            ]
            stacked_stats = np.column_stack(stat_arrays)
            nonzero_stats = (
                stacked_stats.sum(axis=1) > _WORLD_CONTRACT_TOL
                if stacked_stats.size > 0
                else np.zeros(len(df), dtype=bool)
            )
            inactive_nonzero_stats = int(np.count_nonzero(inactive_mask & nonzero_stats))
        else:
            inactive_nonzero_stats = 0
        dk_nonzero = (
            pd.to_numeric(df.get("dk_fpts", 0), errors="coerce").fillna(0.0).abs()
            > _WORLD_CONTRACT_TOL
        ) | (
            pd.to_numeric(df.get("minutes", 0), errors="coerce").fillna(0.0).abs()
            > _WORLD_CONTRACT_TOL
        )
        inactive_nonzero_fpts_proxy = int(
            np.count_nonzero(inactive_mask & np.asarray(dk_nonzero, dtype=bool))
        )
    else:
        inactive_nonzero_stats = 0
        inactive_nonzero_fpts_proxy = 0
    return {
        "team_minutes_not_240": team_minutes_not_240,
        "team_minutes_total_checks": team_minutes_total_checks,
        "team_minutes_max_abs_drift": team_minutes_max_abs_drift,
        "minutes_negative": int(
            (df.get("minutes", pd.Series(dtype=float)) < -_WORLD_CONTRACT_TOL).sum()
        ),
        "minutes_over_48": int(
            (
                df.get("minutes", pd.Series(dtype=float))
                > 48.0 + _WORLD_CONTRACT_TOL
            ).sum()
        ),
        "negative_stats": int(negative_stats),
        "fg2m_gt_fga2": int(
            ((df.get("fg2m", 0) - df.get("fga2", 0)) > _WORLD_CONTRACT_TOL).sum()
        ),
        "fg3m_gt_fga3": int(
            ((df.get("fg3m", 0) - df.get("fga3", 0)) > _WORLD_CONTRACT_TOL).sum()
        ),
        "ftm_gt_fta": int(
            ((df.get("ftm", 0) - df.get("fta", 0)) > _WORLD_CONTRACT_TOL).sum()
        ),
        "inactive_nonzero_stats": inactive_nonzero_stats,
        "inactive_nonzero_fpts_proxy": inactive_nonzero_fpts_proxy,
        "max_abs_stat_value": max_abs_stat_value,
        "extreme_stat_rows_over_1e6": extreme_stat_rows_over_1e6,
    }


def _repair_world_frame_contract_fields(
    worlds_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Repair known contract-field corruption patterns in sampled worlds."""
    if worlds_df.empty:
        return worlds_df, {"applied": False, "reason": "empty_worlds"}

    # Mutate in place to avoid an extra full-frame copy on large live slates.
    out = worlds_df
    report: dict[str, Any] = {
        "applied": False,
        "game_id_from_norm_rows": 0,
        "zero_minute_rows_deactivated": 0,
        "zero_minute_or_inactive_rows_zeroed": 0,
        "nonfinite_stat_values_replaced": 0,
        "base_stat_cap_clipped_rows": 0,
        "base_stat_cap_clipped_rows_by_col": {},
        "derived_stat_cap_clipped_rows": 0,
        "derived_stat_cap_clipped_rows_by_col": {},
        "fg2m_clipped_to_fga2_rows": 0,
        "fg3m_clipped_to_fga3_rows": 0,
        "ftm_clipped_to_fta_rows": 0,
        "dropped_bad_world_game_pairs": 0,
        "dropped_bad_world_rows": 0,
    }
    stat_repair_mask = np.zeros(len(out), dtype=bool)

    if "game_id" in out.columns and "game_id_norm" in out.columns:
        game_id = pd.to_numeric(out["game_id"], errors="coerce").astype("Int64")
        game_id_norm = pd.to_numeric(out["game_id_norm"], errors="coerce").astype("Int64")
        replace_mask = game_id_norm.notna() & game_id.ne(game_id_norm)
        replaced = int(replace_mask.sum())
        if replaced > 0:
            out["game_id"] = game_id.where(~replace_mask, game_id_norm)
            report["game_id_from_norm_rows"] = replaced
            report["applied"] = True

    minutes_zero_like_mask = np.zeros(len(out), dtype=bool)
    if "minutes" in out.columns:
        minutes_vals = pd.to_numeric(out["minutes"], errors="coerce").fillna(0.0)
        minutes_zero_like_mask = (
            pd.to_numeric(minutes_vals, errors="coerce")
            .fillna(0.0)
            .le(float(_WORLD_CONTRACT_TOL))
            .to_numpy(dtype=bool)
        )

    inactive_mask = np.zeros(len(out), dtype=bool)
    if "active" in out.columns:
        active_vals = pd.to_numeric(out["active"], errors="coerce").fillna(0.0)
        inactive_mask = active_vals.le(0.0).to_numpy(dtype=bool)
        deactivate_rows = int(np.count_nonzero(minutes_zero_like_mask & (~inactive_mask)))
        if deactivate_rows > 0:
            out.loc[minutes_zero_like_mask, "active"] = 0
            inactive_mask = pd.to_numeric(out["active"], errors="coerce").fillna(0.0).le(0.0).to_numpy(
                dtype=bool
            )
            report["applied"] = True
            report["zero_minute_rows_deactivated"] = deactivate_rows

    zero_contract_mask = minutes_zero_like_mask | inactive_mask
    if bool(np.any(zero_contract_mask)):
        stat_zero_cols = [
            col
            for col in (
                "fga2",
                "fg2m",
                "fga3",
                "fg3m",
                "fta",
                "ftm",
                "oreb",
                "dreb",
                "ast",
                "stl",
                "blk",
                "tov",
                "pf",
                "fga",
                "fgm",
                "fg3a",
                "pts",
                "reb",
                "dk_fpts",
            )
            if col in out.columns
        ]
        if stat_zero_cols:
            stat_arrays = [
                np.abs(pd.to_numeric(out[col], errors="coerce").fillna(0.0).to_numpy(dtype=float, copy=False))
                for col in stat_zero_cols
            ]
            stat_nonzero = (
                np.column_stack(stat_arrays).sum(axis=1) > float(_WORLD_CONTRACT_TOL)
                if stat_arrays
                else np.zeros(len(out), dtype=bool)
            )
            rows_to_zero = zero_contract_mask & stat_nonzero
            rows_to_zero_count = int(np.count_nonzero(rows_to_zero))
            if rows_to_zero_count > 0:
                for col in stat_zero_cols:
                    out.loc[rows_to_zero, col] = 0.0
                stat_repair_mask[:] = stat_repair_mask | rows_to_zero
                report["applied"] = True
                report["zero_minute_or_inactive_rows_zeroed"] = rows_to_zero_count

    def _clip_numeric_with_cap(
        col: str,
        cap: float,
        *,
        report_key_total: str,
        report_key_by_col: str,
    ) -> None:
        if col not in out.columns:
            return
        raw = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=float)
        finite_mask = np.isfinite(raw)
        nonfinite_rows = int(np.count_nonzero(~finite_mask))
        safe = np.nan_to_num(raw, nan=0.0, posinf=float(cap), neginf=0.0)
        clipped = np.clip(safe, a_min=0.0, a_max=float(cap))
        changed_mask = (~finite_mask) | (
            np.abs(clipped - raw) > _WORLD_CONTRACT_TOL
        )
        changed_rows = int(np.count_nonzero(changed_mask))
        if changed_rows > 0:
            out[col] = clipped
            report["applied"] = True
            report[report_key_total] = int(report.get(report_key_total, 0)) + changed_rows
            per_col = dict(report.get(report_key_by_col) or {})
            per_col[col] = int(per_col.get(col, 0)) + changed_rows
            report[report_key_by_col] = per_col
            stat_repair_mask[:] = stat_repair_mask | changed_mask
        if nonfinite_rows > 0:
            report["nonfinite_stat_values_replaced"] = int(
                report.get("nonfinite_stat_values_replaced", 0)
            ) + nonfinite_rows

    for _col, _cap in _WORLD_BASE_STAT_CAPS.items():
        if _col == "minutes":
            continue
        _clip_numeric_with_cap(
            _col,
            _cap,
            report_key_total="base_stat_cap_clipped_rows",
            report_key_by_col="base_stat_cap_clipped_rows_by_col",
        )

    def _clip_makes_to_attempts(
        attempts_col: str,
        makes_col: str,
        report_key: str,
    ) -> None:
        if attempts_col not in out.columns or makes_col not in out.columns:
            return
        attempts_raw = (
            pd.to_numeric(out[attempts_col], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        makes_raw = (
            pd.to_numeric(out[makes_col], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        attempts = np.clip(attempts_raw, a_min=0.0, a_max=None)
        makes = np.clip(makes_raw, a_min=0.0, a_max=None)
        over_mask = makes > (attempts + _WORLD_CONTRACT_TOL)
        clipped = int(np.count_nonzero(over_mask))
        if clipped > 0:
            makes = np.minimum(makes, attempts)
            report["applied"] = True
            report[report_key] = clipped
        changed_mask = (np.abs(attempts - attempts_raw) > _WORLD_CONTRACT_TOL) | (
            np.abs(makes - makes_raw) > _WORLD_CONTRACT_TOL
        )
        if bool(np.any(changed_mask)):
            report["applied"] = True
            stat_repair_mask[:] = stat_repair_mask | changed_mask
        out[attempts_col] = attempts
        out[makes_col] = makes

    _clip_makes_to_attempts("fga2", "fg2m", "fg2m_clipped_to_fga2_rows")
    _clip_makes_to_attempts("fga3", "fg3m", "fg3m_clipped_to_fga3_rows")
    _clip_makes_to_attempts("fta", "ftm", "ftm_clipped_to_fta_rows")

    # Recompute derived aggregates only for rows that were stat-repaired.
    # This preserves intentional post-model overlays (e.g. props uplift on pts/reb)
    # when no shot-attempt repair was needed.
    if bool(np.any(stat_repair_mask)):
        if {"fga2", "fga3"}.issubset(out.columns):
            out.loc[stat_repair_mask, "fga"] = (
                pd.to_numeric(out.loc[stat_repair_mask, "fga2"], errors="coerce").fillna(0.0)
                + pd.to_numeric(out.loc[stat_repair_mask, "fga3"], errors="coerce").fillna(0.0)
            )
        if {"fg2m", "fg3m"}.issubset(out.columns):
            out.loc[stat_repair_mask, "fgm"] = (
                pd.to_numeric(out.loc[stat_repair_mask, "fg2m"], errors="coerce").fillna(0.0)
                + pd.to_numeric(out.loc[stat_repair_mask, "fg3m"], errors="coerce").fillna(0.0)
            )
        if {"fg2m", "fg3m", "ftm"}.issubset(out.columns):
            out.loc[stat_repair_mask, "pts"] = (
                2.0
                * pd.to_numeric(out.loc[stat_repair_mask, "fg2m"], errors="coerce").fillna(0.0)
                + 3.0
                * pd.to_numeric(out.loc[stat_repair_mask, "fg3m"], errors="coerce").fillna(0.0)
                + pd.to_numeric(out.loc[stat_repair_mask, "ftm"], errors="coerce").fillna(0.0)
            )
        if {"oreb", "dreb"}.issubset(out.columns):
            out.loc[stat_repair_mask, "reb"] = (
                pd.to_numeric(out.loc[stat_repair_mask, "oreb"], errors="coerce").fillna(0.0)
                + pd.to_numeric(out.loc[stat_repair_mask, "dreb"], errors="coerce").fillna(0.0)
            )
        if {"pts", "reb", "ast", "stl", "blk", "tov"}.issubset(out.columns):
            out.loc[stat_repair_mask, "dk_fpts"] = _recompute_dk_fpts(
                out.loc[stat_repair_mask]
            ).to_numpy(dtype=float)

    for _col, _cap in _WORLD_DERIVED_STAT_CAPS.items():
        _clip_numeric_with_cap(
            _col,
            _cap,
            report_key_total="derived_stat_cap_clipped_rows",
            report_key_by_col="derived_stat_cap_clipped_rows_by_col",
        )

    if {"world_idx", "game_id", "team_id", "minutes"}.issubset(out.columns):
        uniq_world, uniq_game, _, minute_sums = _team_minutes_sums_without_pandas_groupby(
            world_idx_col=out["world_idx"],
            game_id_col=out["game_id"],
            team_id_col=out["team_id"],
            minutes_col=out["minutes"],
        )
        bad_team_mask = np.abs(minute_sums - 240.0) > _WORLD_CONTRACT_TOL
        if bool(np.any(bad_team_mask)):
            pair_dtype = np.dtype([("world_idx", np.int64), ("game_id", np.int64)])
            bad_pairs = np.empty(int(np.count_nonzero(bad_team_mask)), dtype=pair_dtype)
            bad_pairs["world_idx"] = uniq_world[bad_team_mask]
            bad_pairs["game_id"] = uniq_game[bad_team_mask]
            bad_pairs = np.unique(bad_pairs)

            row_world_raw = pd.to_numeric(out["world_idx"], errors="coerce").to_numpy(
                dtype=float, copy=False
            )
            row_game_raw = pd.to_numeric(out["game_id"], errors="coerce").to_numpy(
                dtype=float, copy=False
            )
            valid_row_pairs = _valid_int64_numeric_mask(row_world_raw) & _valid_int64_numeric_mask(
                row_game_raw
            )
            drop_mask = np.zeros(len(out), dtype=bool)
            if bool(np.any(valid_row_pairs)):
                row_pairs = np.empty(int(np.count_nonzero(valid_row_pairs)), dtype=pair_dtype)
                row_pairs["world_idx"] = np.rint(row_world_raw[valid_row_pairs]).astype(
                    np.int64, copy=False
                )
                row_pairs["game_id"] = np.rint(row_game_raw[valid_row_pairs]).astype(
                    np.int64, copy=False
                )
                drop_mask[valid_row_pairs] = np.isin(row_pairs, bad_pairs)
            if drop_mask.shape != (len(out),):
                raise RuntimeError("world repair drop mask shape mismatch")
            dropped_rows = int(np.count_nonzero(drop_mask))
            if 0 < dropped_rows < len(out):
                # Avoid DataFrame.take on mixed-type world frames.
                out = out.loc[~drop_mask].reset_index(drop=True).copy()
                report["applied"] = True
                report["dropped_bad_world_game_pairs"] = int(len(bad_pairs))
                report["dropped_bad_world_rows"] = dropped_rows

    return out, report


def _recompute_dk_fpts(worlds_df: pd.DataFrame) -> pd.Series:
    pts = pd.to_numeric(worlds_df.get("pts", 0.0), errors="coerce").fillna(0.0)
    reb = pd.to_numeric(worlds_df.get("reb", 0.0), errors="coerce").fillna(0.0)
    ast = pd.to_numeric(worlds_df.get("ast", 0.0), errors="coerce").fillna(0.0)
    stl = pd.to_numeric(worlds_df.get("stl", 0.0), errors="coerce").fillna(0.0)
    blk = pd.to_numeric(worlds_df.get("blk", 0.0), errors="coerce").fillna(0.0)
    tov = pd.to_numeric(worlds_df.get("tov", 0.0), errors="coerce").fillna(0.0)
    base = pts + 1.25 * reb + 1.5 * ast + 2.0 * stl + 2.0 * blk - 0.5 * tov
    qualifying = pd.concat([pts, reb, ast, stl, blk], axis=1).ge(10.0).sum(axis=1)
    bonus_dd = qualifying.eq(2).astype(float) * 1.5
    bonus_td = qualifying.ge(3).astype(float) * 3.0
    return (base + bonus_dd + bonus_td).clip(lower=0.0)


def _apply_low_minutes_tail_damping_to_worlds(
    worlds_df: pd.DataFrame,
    *,
    minutes_threshold: float = 12.0,
    min_scale: float = 0.55,
    target_game_ids: set[int] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Shrink low-minute tail residuals toward each player's world mean."""
    if worlds_df.empty:
        return worlds_df, {"applied": False, "reason": "empty_worlds"}
    required = {"game_id", "team_id", "player_id", "minutes", "pts", "reb", "ast", "dk_fpts"}
    if not required.issubset(worlds_df.columns):
        return worlds_df, {
            "applied": False,
            "reason": "missing_required_columns",
            "missing_columns": sorted(required - set(worlds_df.columns)),
        }
    if minutes_threshold <= 0.0:
        return worlds_df, {
            "applied": False,
            "reason": "invalid_minutes_threshold",
            "minutes_threshold": float(minutes_threshold),
        }

    low = float(minutes_threshold)
    floor_scale = float(np.clip(min_scale, 0.0, 1.0))
    out = worlds_df.copy()
    minutes = pd.to_numeric(out["minutes"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    ramp = np.clip((low - minutes) / low, 0.0, 1.0)
    scale = 1.0 - (1.0 - floor_scale) * ramp
    damp_mask = (minutes > 0.0) & (minutes < low)

    if target_game_ids:
        game_ids = pd.to_numeric(out["game_id"], errors="coerce").astype("Int64")
        damp_mask = damp_mask & game_ids.isin(sorted(target_game_ids)).to_numpy(dtype=bool)

    if not bool(damp_mask.any()):
        return out, {
            "applied": False,
            "reason": "no_low_minutes_rows",
            "minutes_threshold": low,
            "min_scale": floor_scale,
            "target_game_count": int(len(target_game_ids or set())),
        }

    key_cols = ["game_id", "team_id", "player_id"]
    stat_cols = [c for c in ("pts", "reb", "ast", "stl", "blk", "tov") if c in out.columns]
    for col in stat_cols:
        x = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
        mu = x.groupby([out[k] for k in key_cols], dropna=False).transform("mean")
        new_vals = (mu + pd.Series(scale, index=out.index) * (x - mu)).clip(lower=0.0)
        out[col] = np.where(damp_mask, new_vals.to_numpy(dtype=float), x.to_numpy(dtype=float))

    if {"oreb", "dreb", "reb"}.issubset(out.columns):
        oreb = pd.to_numeric(out["oreb"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        dreb = pd.to_numeric(out["dreb"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        reb = pd.to_numeric(out["reb"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        split_sum = np.maximum(oreb + dreb, 1e-6)
        oreb_share = np.divide(oreb, split_sum)
        oreb_new = reb * oreb_share
        dreb_new = reb * (1.0 - oreb_share)
        out["oreb"] = np.where(damp_mask, oreb_new, oreb)
        out["dreb"] = np.where(damp_mask, dreb_new, dreb)

    out["dk_fpts"] = _recompute_dk_fpts(out)
    report = {
        "applied": True,
        "minutes_threshold": low,
        "min_scale": floor_scale,
        "target_game_count": int(len(target_game_ids or set())),
        "affected_rows": int(np.count_nonzero(damp_mask)),
        "affected_players": int(
            out.loc[damp_mask, key_cols].drop_duplicates().shape[0]
            if np.count_nonzero(damp_mask) > 0
            else 0
        ),
        "scale_mean": float(np.mean(scale[damp_mask])) if np.count_nonzero(damp_mask) > 0 else 1.0,
        "scale_p10": float(np.quantile(scale[damp_mask], 0.10)) if np.count_nonzero(damp_mask) > 0 else 1.0,
        "scale_p90": float(np.quantile(scale[damp_mask], 0.90)) if np.count_nonzero(damp_mask) > 0 else 1.0,
    }
    return out, report


def _resample_extreme_game_worlds(
    worlds_df: pd.DataFrame,
    *,
    random_seed: int,
    max_passes: int = 1,
    short_minutes_threshold: float = 12.0,
    short_minutes_dk_threshold: float = _WORLD_REALISM_SHORT_MINUTES_DK_THRESHOLD,
    game_pts_max: float = _WORLD_REALISM_GAME_PTS_MAX_THRESHOLD,
    game_pts_min: float = _WORLD_REALISM_GAME_PTS_MIN_THRESHOLD,
    target_game_ids: set[int] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Replace extreme game-world pairs with sampled in-game donor worlds."""
    if worlds_df.empty:
        return worlds_df, {"applied": False, "reason": "empty_worlds"}
    required = {"world_idx", "game_id", "team_id", "player_id", "minutes", "pts", "dk_fpts"}
    if not required.issubset(worlds_df.columns):
        return worlds_df, {
            "applied": False,
            "reason": "missing_required_columns",
            "missing_columns": sorted(required - set(worlds_df.columns)),
        }

    out = worlds_df.copy()
    max_iter = max(0, int(max_passes))
    if max_iter == 0:
        return out, {"applied": False, "reason": "disabled_max_passes"}

    rng = np.random.default_rng(int(random_seed))
    pass_reports: list[dict[str, Any]] = []
    total_replaced = 0
    target_games = (
        np.asarray(sorted(target_game_ids), dtype=np.int64)
        if target_game_ids
        else None
    )

    world_idx_arr = (
        pd.to_numeric(out["world_idx"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64, copy=False)
    )
    game_id_arr = (
        pd.to_numeric(out["game_id"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64, copy=False)
    )
    team_id_arr = (
        pd.to_numeric(out["team_id"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64, copy=False)
    )
    player_id_arr = (
        pd.to_numeric(out["player_id"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64, copy=False)
    )

    # Avoid pandas MultiIndex factorization on large live frames; this has
    # intermittently segfaulted on long-running workers.
    pair_codes, pair_unique_arrays = _factorize_int_key_arrays_preserve_order(
        world_idx_arr,
        game_id_arr,
    )
    n_pairs = int(len(pair_unique_arrays[0])) if pair_unique_arrays else 0
    if n_pairs <= 0:
        return out, {"applied": False, "reason": "no_pairs"}

    pair_game_id = (
        np.asarray(pair_unique_arrays[1], dtype=np.int64)
        if len(pair_unique_arrays) >= 2
        else np.array([], dtype=np.int64)
    )
    pair_in_scope = (
        np.isin(pair_game_id, target_games)
        if target_games is not None
        else np.ones(n_pairs, dtype=bool)
    )

    row_order = np.lexsort((player_id_arr, team_id_arr, world_idx_arr, game_id_arr))
    sorted_pair_codes = pair_codes[row_order]
    pair_starts = np.full(n_pairs, -1, dtype=np.int64)
    pair_ends = np.full(n_pairs, -1, dtype=np.int64)
    starts = np.flatnonzero(np.r_[True, sorted_pair_codes[1:] != sorted_pair_codes[:-1]])
    ends = np.r_[starts[1:], len(sorted_pair_codes)]
    pair_starts[sorted_pair_codes[starts]] = starts
    pair_ends[sorted_pair_codes[starts]] = ends
    team_sorted = team_id_arr[row_order]
    player_sorted = player_id_arr[row_order]

    game_codes, _ = _factorize_int_key_arrays_preserve_order(pair_game_id)
    game_pair_order = np.argsort(game_codes, kind="mergesort")
    sorted_game_codes = game_codes[game_pair_order]
    game_starts = np.flatnonzero(np.r_[True, sorted_game_codes[1:] != sorted_game_codes[:-1]])
    game_ends = np.r_[game_starts[1:], len(sorted_game_codes)]
    for pass_idx in range(max_iter):
        minutes = (
            pd.to_numeric(out["minutes"], errors="coerce").fillna(0.0).to_numpy(dtype=float, copy=False)
        )
        dk = (
            pd.to_numeric(out["dk_fpts"], errors="coerce").fillna(0.0).to_numpy(dtype=float, copy=False)
        )
        pts = (
            pd.to_numeric(out["pts"], errors="coerce").fillna(0.0).to_numpy(dtype=float, copy=False)
        )

        row_spike = (minutes < float(short_minutes_threshold)) & (
            dk > float(short_minutes_dk_threshold)
        )
        if target_games is not None:
            row_spike &= np.isin(game_id_arr, target_games)

        pair_short = (
            np.bincount(pair_codes, weights=row_spike.astype(np.int8), minlength=n_pairs) > 0
        )
        pair_game_pts = np.bincount(pair_codes, weights=pts, minlength=n_pairs)
        pair_hi = pair_in_scope & (pair_game_pts > float(game_pts_max))
        pair_lo = pair_in_scope & (pair_game_pts < float(game_pts_min))
        pair_bad = pair_short | pair_hi | pair_lo

        bad_pairs = np.flatnonzero(pair_bad)
        if bad_pairs.size == 0:
            break

        replacements: list[tuple[int, int, int, int]] = []
        skipped_no_donor = 0
        skipped_key_mismatch = 0
        for game_start, game_end in zip(game_starts, game_ends, strict=False):
            pair_idx = game_pair_order[game_start:game_end]
            bad_pair_idx = pair_idx[pair_bad[pair_idx]]
            if bad_pair_idx.size == 0:
                continue
            good_pair_idx = pair_idx[~pair_bad[pair_idx]]
            if good_pair_idx.size == 0:
                skipped_no_donor += int(bad_pair_idx.size)
                continue
            donor_pair_idx = rng.choice(good_pair_idx, size=bad_pair_idx.size, replace=True)
            for target_pair, donor_pair in zip(bad_pair_idx, donor_pair_idx, strict=False):
                target_start = int(pair_starts[target_pair])
                target_end = int(pair_ends[target_pair])
                donor_start = int(pair_starts[int(donor_pair)])
                donor_end = int(pair_ends[int(donor_pair)])
                if target_start < 0 or donor_start < 0:
                    skipped_no_donor += 1
                    continue
                if (target_end - target_start) != (donor_end - donor_start):
                    skipped_key_mismatch += 1
                    continue
                if not np.array_equal(
                    team_sorted[target_start:target_end],
                    team_sorted[donor_start:donor_end],
                ) or not np.array_equal(
                    player_sorted[target_start:target_end],
                    player_sorted[donor_start:donor_end],
                ):
                    skipped_key_mismatch += 1
                    continue
                replacements.append((target_start, target_end, donor_start, donor_end))

        if replacements:
            row_sources = np.arange(len(out), dtype=np.int64)
            for target_start, target_end, donor_start, donor_end in replacements:
                row_sources[row_order[target_start:target_end]] = row_order[
                    donor_start:donor_end
                ]
            world_idx_original = out["world_idx"].to_numpy(copy=False)
            # Avoid DataFrame.take on mixed-type world frames.
            out = out.iloc[row_sources].reset_index(drop=True).copy()
            out["world_idx"] = world_idx_original

        replaced_this_pass = len(replacements)

        pass_reports.append(
            {
                "pass_idx": int(pass_idx + 1),
                "bad_pair_count": int(bad_pairs.size),
                "bad_short_spike_count": int(pair_short.sum()),
                "bad_game_hi_count": int(pair_hi.sum()),
                "bad_game_lo_count": int(pair_lo.sum()),
                "replaced_pair_count": int(replaced_this_pass),
                "skipped_no_donor": int(skipped_no_donor),
                "skipped_key_mismatch": int(skipped_key_mismatch),
            }
        )
        total_replaced += int(replaced_this_pass)
        if replaced_this_pass == 0:
            break

    report = {
        "applied": bool(total_replaced > 0),
        "random_seed": int(random_seed),
        "max_passes": int(max_iter),
        "target_game_count": int(len(target_game_ids or set())),
        "short_minutes_threshold": float(short_minutes_threshold),
        "short_minutes_dk_threshold": float(short_minutes_dk_threshold),
        "game_pts_max": float(game_pts_max),
        "game_pts_min": float(game_pts_min),
        "total_replaced_pairs": int(total_replaced),
        "passes": pass_reports,
    }
    if total_replaced == 0 and not pass_reports:
        report["applied"] = False
        report["reason"] = "no_outlier_pairs"
    return out, report


def _apply_world_realism_controls_to_worlds(
    worlds_df: pd.DataFrame,
    *,
    enabled: bool,
    random_seed: int,
    low_minutes_tail_damping_enabled: bool,
    low_minutes_tail_minutes_threshold: float,
    low_minutes_tail_min_scale: float,
    outlier_resample_enabled: bool,
    outlier_resample_max_passes: int,
    target_game_ids: set[int] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not enabled:
        return worlds_df, {"applied": False, "reason": "disabled"}

    out = worlds_df.copy()
    report: dict[str, Any] = {"applied": False}
    if low_minutes_tail_damping_enabled:
        out, damp_report = _apply_low_minutes_tail_damping_to_worlds(
            out,
            minutes_threshold=float(low_minutes_tail_minutes_threshold),
            min_scale=float(low_minutes_tail_min_scale),
            target_game_ids=target_game_ids,
        )
    else:
        damp_report = {"applied": False, "reason": "disabled"}
    report["low_minutes_tail_damping"] = damp_report

    if outlier_resample_enabled:
        out, resample_report = _resample_extreme_game_worlds(
            out,
            random_seed=int(random_seed),
            max_passes=int(outlier_resample_max_passes),
            target_game_ids=target_game_ids,
        )
    else:
        resample_report = {"applied": False, "reason": "disabled"}
    report["outlier_resample"] = resample_report
    report["applied"] = bool(
        bool((damp_report or {}).get("applied"))
        or bool((resample_report or {}).get("applied"))
    )
    return out, report


def _apply_props_uplift_calibration_to_worlds(
    worlds_df: pd.DataFrame,
    *,
    features_df: pd.DataFrame,
    scope: str = "stars_only",
    confidence_weighted: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply one-sided stat uplifts with tail broadening for undercalled prop-heavy players."""
    resolved_scope = str(scope).strip().lower() or "stars_only"
    if resolved_scope not in {"stars_only", "all_players"}:
        return worlds_df, {
            "applied": False,
            "reason": "invalid_scope",
            "scope": resolved_scope,
            "valid_scopes": ["stars_only", "all_players"],
        }
    if worlds_df.empty:
        return worlds_df, {"applied": False, "reason": "empty_worlds"}

    required_world_cols = {
        "game_id",
        "team_id",
        "player_id",
        "pts",
        "reb",
        "ast",
        "stl",
        "blk",
        "tov",
        "dk_fpts",
    }
    if not required_world_cols.issubset(worlds_df.columns):
        return worlds_df, {
            "applied": False,
            "reason": "missing_world_cols",
            "missing_world_cols": sorted(required_world_cols - set(worlds_df.columns)),
        }

    required_feature_cols = {"game_id", "team_id", "player_id"}
    if not required_feature_cols.issubset(features_df.columns):
        return worlds_df, {
            "applied": False,
            "reason": "missing_feature_keys",
            "missing_feature_cols": sorted(required_feature_cols - set(features_df.columns)),
        }

    stat_cfg: dict[str, dict[str, float | str]] = {
        "pts": {
            "line_col": "an_pts_line",
            "has_col": "an_has_pts",
            "books_col": "an_pts_books",
            "min_line": 20.0,
            "min_gap": 2.5,
            "min_line_all": 8.0,
            "min_gap_all": 1.0,
            "weight": 0.88,
            "max_scale": 2.0,
            "var_weight": 0.40,
            "max_var_scale": 1.50,
            "line_anchor_min_line": 28.0,
            "line_anchor_frac": 0.93,
            "min_line_down": 12.0,
            "min_gap_down": 2.5,
            "min_line_down_all": 6.0,
            "min_gap_down_all": 1.0,
            "weight_down": 0.45,
            "min_scale_down": 0.70,
            "var_weight_down": 0.15,
            "min_var_scale_down": 0.80,
            "line_quality_min": 8.0,
            "line_quality_span": 18.0,
        },
        "reb": {
            "line_col": "an_reb_line",
            "has_col": "an_has_reb",
            "books_col": "an_reb_books",
            "min_line": 7.0,
            "min_gap": 1.5,
            "min_line_all": 3.0,
            "min_gap_all": 0.8,
            "weight": 0.92,
            "max_scale": 2.2,
            "var_weight": 0.45,
            "max_var_scale": 1.60,
            "line_anchor_min_line": 10.0,
            "line_anchor_frac": 0.92,
            "min_line_down": 3.0,
            "min_gap_down": 1.3,
            "min_line_down_all": 2.0,
            "min_gap_down_all": 0.8,
            "weight_down": 0.60,
            "min_scale_down": 0.55,
            "var_weight_down": 0.25,
            "min_var_scale_down": 0.75,
            "line_quality_min": 3.0,
            "line_quality_span": 7.0,
        },
        "ast": {
            "line_col": "an_ast_line",
            "has_col": "an_has_ast",
            "books_col": "an_ast_books",
            "min_line": 5.5,
            "min_gap": 1.0,
            "min_line_all": 2.0,
            "min_gap_all": 0.75,
            "weight": 0.92,
            "max_scale": 2.2,
            "var_weight": 0.50,
            "max_var_scale": 1.65,
            "line_anchor_min_line": 8.0,
            "line_anchor_frac": 0.92,
            "min_line_down": 1.5,
            "min_gap_down": 1.0,
            "min_line_down_all": 1.0,
            "min_gap_down_all": 0.75,
            "weight_down": 0.65,
            "min_scale_down": 0.50,
            "var_weight_down": 0.25,
            "min_var_scale_down": 0.72,
            "line_quality_min": 2.0,
            "line_quality_span": 6.0,
        },
        "stl": {
            "line_col": "an_stl_line",
            "has_col": "an_has_stl",
            "books_col": "an_stl_books",
            "min_line": 1.0,
            "min_gap": 0.40,
            "min_line_all": 0.5,
            "min_gap_all": 0.25,
            "weight": 0.55,
            "max_scale": 1.75,
            "var_weight": 0.20,
            "max_var_scale": 1.25,
            "line_anchor_min_line": 1.5,
            "line_anchor_frac": 0.90,
            "min_line_down": 0.75,
            "min_gap_down": 0.35,
            "min_line_down_all": 0.5,
            "min_gap_down_all": 0.25,
            "weight_down": 0.35,
            "min_scale_down": 0.70,
            "var_weight_down": 0.10,
            "min_var_scale_down": 0.85,
            "line_quality_min": 0.5,
            "line_quality_span": 1.5,
        },
        "blk": {
            "line_col": "an_blk_line",
            "has_col": "an_has_blk",
            "books_col": "an_blk_books",
            "min_line": 1.0,
            "min_gap": 0.40,
            "min_line_all": 0.5,
            "min_gap_all": 0.25,
            "weight": 0.60,
            "max_scale": 1.85,
            "var_weight": 0.22,
            "max_var_scale": 1.28,
            "line_anchor_min_line": 1.5,
            "line_anchor_frac": 0.90,
            "min_line_down": 0.75,
            "min_gap_down": 0.35,
            "min_line_down_all": 0.5,
            "min_gap_down_all": 0.25,
            "weight_down": 0.38,
            "min_scale_down": 0.68,
            "var_weight_down": 0.12,
            "min_var_scale_down": 0.84,
            "line_quality_min": 0.5,
            "line_quality_span": 1.5,
        },
    }
    key_cols = ["game_id", "team_id", "player_id"]

    player_means = _group_mean_by_keys_without_pandas_groupby(
        worlds_df,
        key_cols=key_cols,
        value_cols=("pts", "reb", "ast", "stl", "blk"),
        label="props_uplift/player_means",
    ).rename(
        columns={
            "pts": "pts_mean",
            "reb": "reb_mean",
            "ast": "ast_mean",
            "stl": "stl_mean",
            "blk": "blk_mean",
        }
    )

    feat_cols = list(key_cols)
    if "an_props_market_count" in features_df.columns:
        feat_cols.append("an_props_market_count")
    for cfg in stat_cfg.values():
        line_col = str(cfg["line_col"])
        has_col = str(cfg["has_col"])
        books_col = str(cfg["books_col"])
        if line_col in features_df.columns:
            feat_cols.append(line_col)
        if has_col in features_df.columns:
            feat_cols.append(has_col)
        if books_col in features_df.columns:
            feat_cols.append(books_col)
    feat = features_df.loc[:, sorted(set(feat_cols), key=feat_cols.index)].copy()

    agg_dict: dict[str, str] = {}
    for col in feat.columns:
        if col in key_cols:
            continue
        agg_dict[col] = "max" if col.startswith("an_has_") else "first"
    feat = feat.groupby(key_cols, dropna=False, as_index=False).agg(agg_dict)

    meta = player_means.merge(feat, on=key_cols, how="left")
    if "player_name" in features_df.columns:
        names = (
            features_df.loc[:, key_cols + ["player_name"]]
            .drop_duplicates(subset=key_cols, keep="last")
            .copy()
        )
        meta = meta.merge(names, on=key_cols, how="left")
    for cfg in stat_cfg.values():
        line_col = str(cfg["line_col"])
        has_col = str(cfg["has_col"])
        books_col = str(cfg["books_col"])
        if line_col in meta.columns:
            meta[line_col] = pd.to_numeric(meta[line_col], errors="coerce")
        if has_col in meta.columns:
            meta[has_col] = pd.to_numeric(meta[has_col], errors="coerce").fillna(0.0)
        if books_col in meta.columns:
            meta[books_col] = pd.to_numeric(meta[books_col], errors="coerce")
    if "an_props_market_count" in meta.columns:
        meta["an_props_market_count"] = pd.to_numeric(meta["an_props_market_count"], errors="coerce")

    out = worlds_df.copy()
    report: dict[str, Any] = {
        "applied": True,
        "scope": resolved_scope,
        "confidence_weighted": bool(confidence_weighted),
        "stats": {},
    }
    stat_sanitize_report: dict[str, int] = {}

    def _sanitize_world_stat_column(*, frame: pd.DataFrame, col: str, cap: float) -> int:
        if col not in frame.columns:
            return 0
        raw = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float, copy=False)
        finite_mask = np.isfinite(raw)
        safe = np.nan_to_num(raw, nan=0.0, posinf=float(cap), neginf=0.0)
        clipped = np.clip(safe, a_min=0.0, a_max=float(cap))
        changed = int((~finite_mask).sum())
        if clipped.size > 0:
            changed += int(np.count_nonzero(np.abs(clipped - safe) > _WORLD_CONTRACT_TOL))
        frame[col] = clipped
        return changed

    for col_name, cap in {**_WORLD_BASE_STAT_CAPS, **_WORLD_DERIVED_STAT_CAPS}.items():
        changed_rows = _sanitize_world_stat_column(frame=out, col=str(col_name), cap=float(cap))
        if changed_rows > 0:
            stat_sanitize_report[str(col_name)] = int(changed_rows)
    if stat_sanitize_report:
        report["pre_uplift_stat_sanitize"] = stat_sanitize_report

    adjusted_key_frames: list[pd.DataFrame] = []

    for stat_name, cfg in stat_cfg.items():
        line_col = str(cfg["line_col"])
        has_col = str(cfg["has_col"])
        mean_col = f"{stat_name}_mean"
        if line_col not in meta.columns or mean_col not in meta.columns:
            report["stats"][stat_name] = {
                "applied_player_count": 0,
                "reason": "missing_line_or_mean_column",
            }
            continue

        line = pd.to_numeric(meta[line_col], errors="coerce")
        mean = pd.to_numeric(meta[mean_col], errors="coerce")
        gap = line - mean
        denom = line.clip(lower=1.0)
        min_line = float(cfg["min_line"] if resolved_scope == "stars_only" else cfg["min_line_all"])
        min_gap = float(cfg["min_gap"] if resolved_scope == "stars_only" else cfg["min_gap_all"])
        min_line_down = float(cfg["min_line_down"] if resolved_scope == "stars_only" else cfg["min_line_down_all"])
        min_gap_down = float(cfg["min_gap_down"] if resolved_scope == "stars_only" else cfg["min_gap_down_all"])
        has_market = pd.Series(True, index=meta.index, dtype=bool)
        if has_col in meta.columns:
            has_market = pd.to_numeric(meta[has_col], errors="coerce").fillna(0.0).ge(0.5)
        mask_up = line.ge(min_line) & gap.ge(min_gap) & mean.gt(0.0) & has_market
        over_gap = mean - line
        mask_down = (
            line.ge(min_line_down)
            & over_gap.ge(min_gap_down)
            & mean.gt(0.0)
            & has_market
        )

        confidence = pd.Series(1.0, index=meta.index, dtype=float)
        if bool(confidence_weighted) and resolved_scope == "all_players":
            line_quality = (
                (line - float(cfg["line_quality_min"])) / float(cfg["line_quality_span"])
            ).clip(lower=0.0, upper=1.0)
            books_col = str(cfg["books_col"])
            if books_col in meta.columns:
                books = pd.to_numeric(meta[books_col], errors="coerce")
                books_quality = ((books - 1.0) / 2.0).clip(lower=0.0, upper=1.0).fillna(0.5)
            else:
                books_quality = pd.Series(0.5, index=meta.index, dtype=float)
            if "an_props_market_count" in meta.columns:
                market_count = pd.to_numeric(meta["an_props_market_count"], errors="coerce")
                market_quality = ((market_count - 1.0) / 6.0).clip(lower=0.0, upper=1.0).fillna(0.5)
            else:
                market_quality = pd.Series(0.5, index=meta.index, dtype=float)
            confidence = (
                0.55 * pd.to_numeric(line_quality, errors="coerce").fillna(0.0)
                + 0.25 * pd.to_numeric(books_quality, errors="coerce").fillna(0.5)
                + 0.20 * pd.to_numeric(market_quality, errors="coerce").fillna(0.5)
            ).clip(lower=0.10, upper=1.0)

        weight_up = float(cfg["weight"]) * confidence
        weight_down = float(cfg["weight_down"]) * confidence
        var_weight_up = float(cfg["var_weight"]) * confidence
        var_weight_down = float(cfg["var_weight_down"]) * confidence
        max_scale_eff = 1.0 + (float(cfg["max_scale"]) - 1.0) * confidence
        max_var_scale_eff = 1.0 + (float(cfg["max_var_scale"]) - 1.0) * confidence
        min_scale_down_eff = 1.0 - (1.0 - float(cfg["min_scale_down"])) * confidence
        min_var_scale_down_eff = 1.0 - (1.0 - float(cfg["min_var_scale_down"])) * confidence

        target_up = mean + weight_up * gap
        target_up = target_up.where(
            line.lt(float(cfg["line_anchor_min_line"])),
            np.maximum(
                pd.to_numeric(target_up, errors="coerce").to_numpy(dtype=float),
                float(cfg["line_anchor_frac"]) * pd.to_numeric(line, errors="coerce").to_numpy(dtype=float),
            ),
        )
        scale_up = (target_up / mean).clip(lower=1.0, upper=max_scale_eff)
        var_scale_up = (
            1.0 + var_weight_up * (gap / denom).clip(lower=0.0)
        ).clip(lower=1.0, upper=max_var_scale_eff)
        target_down = mean - weight_down * over_gap
        scale_down = (target_down / mean).clip(
            lower=min_scale_down_eff,
            upper=1.0,
        )
        var_scale_down = (
            1.0 - var_weight_down * (over_gap / denom).clip(lower=0.0)
        ).clip(lower=min_var_scale_down_eff, upper=1.0)

        up_df = meta.loc[mask_up, key_cols].copy()
        down_df = meta.loc[mask_down, key_cols].copy()
        if "player_name" in meta.columns:
            up_df["player_name"] = meta.loc[mask_up, "player_name"].astype(str).values
            down_df["player_name"] = meta.loc[mask_down, "player_name"].astype(str).values
        up_df["mu"] = mean.loc[mask_up].astype(float).values
        up_df["sf_mean"] = scale_up.loc[mask_up].astype(float).values
        up_df["sf_var"] = var_scale_up.loc[mask_up].astype(float).values
        up_df["line_gap"] = gap.loc[mask_up].astype(float).values
        up_df["confidence"] = confidence.loc[mask_up].astype(float).values
        up_df["direction"] = "up"

        down_df["mu"] = mean.loc[mask_down].astype(float).values
        down_df["sf_mean"] = scale_down.loc[mask_down].astype(float).values
        down_df["sf_var"] = var_scale_down.loc[mask_down].astype(float).values
        down_df["line_gap"] = gap.loc[mask_down].astype(float).values
        down_df["confidence"] = confidence.loc[mask_down].astype(float).values
        down_df["direction"] = "down"

        scale_df = pd.concat([up_df, down_df], ignore_index=True)

        if scale_df.empty:
            report["stats"][stat_name] = {
                "applied_player_count": 0,
                "applied_player_count_up": 0,
                "applied_player_count_down": 0,
                "mean_gap_pre": float((mean - line).mean()) if (mean - line).notna().any() else float("nan"),
                "mean_gap_post": float((mean - line).mean()) if (mean - line).notna().any() else float("nan"),
            }
            continue

        adjusted_key_frames.append(scale_df.loc[:, key_cols].copy())
        # Keep report-only fields (e.g. direction/line_gap/player_name) out of the
        # simulation frame to avoid suffix collisions across per-stat passes.
        scale_apply = scale_df.loc[:, key_cols + ["mu", "sf_mean", "sf_var"]]
        out = _left_overlay_from_source_by_keys(
            out,
            source_df=scale_apply,
            key_cols=key_cols,
            value_cols=("mu", "sf_mean", "sf_var"),
            label=f"props_uplift/{stat_name}_scale_overlay",
            copy_base=False,
        )
        stat_cap = float(_WORLD_BASE_STAT_CAPS.get(stat_name, 1_000.0))
        mu = pd.to_numeric(out["mu"], errors="coerce").to_numpy(dtype=float, copy=False)
        mu = np.clip(np.nan_to_num(mu, nan=0.0, posinf=stat_cap, neginf=0.0), 0.0, stat_cap)
        sf_mean = pd.to_numeric(out["sf_mean"], errors="coerce").to_numpy(dtype=float, copy=False)
        sf_mean = np.clip(np.nan_to_num(sf_mean, nan=1.0, posinf=1.0, neginf=1.0), 0.0, 3.0)
        sf_var = pd.to_numeric(out["sf_var"], errors="coerce").to_numpy(dtype=float, copy=False)
        sf_var = np.clip(np.nan_to_num(sf_var, nan=1.0, posinf=1.0, neginf=1.0), 0.0, 3.0)
        target_mu = np.clip(mu * sf_mean, 0.0, stat_cap)
        if "minutes" in out.columns:
            minutes_vals = pd.to_numeric(out["minutes"], errors="coerce").to_numpy(dtype=float, copy=False)
            minutes_vals = np.clip(
                np.nan_to_num(
                    minutes_vals,
                    nan=0.0,
                    posinf=float(_WORLD_BASE_STAT_CAPS["minutes"]),
                    neginf=0.0,
                ),
                0.0,
                float(_WORLD_BASE_STAT_CAPS["minutes"]),
            )
            active_mask = minutes_vals > 0.0
        else:
            fpts_vals = pd.to_numeric(out["dk_fpts"], errors="coerce").to_numpy(dtype=float, copy=False)
            fpts_vals = np.clip(
                np.nan_to_num(
                    fpts_vals,
                    nan=0.0,
                    posinf=float(_WORLD_DERIVED_STAT_CAPS["dk_fpts"]),
                    neginf=0.0,
                ),
                0.0,
                float(_WORLD_DERIVED_STAT_CAPS["dk_fpts"]),
            )
            active_mask = fpts_vals > 0.0
        if stat_name == "pts":
            x = pd.to_numeric(out["pts"], errors="coerce").to_numpy(dtype=float, copy=False)
            x = np.clip(np.nan_to_num(x, nan=0.0, posinf=stat_cap, neginf=0.0), 0.0, stat_cap)
            pts_raw = target_mu + sf_var * (x - mu)
            pts_new = np.clip(np.nan_to_num(pts_raw, nan=0.0, posinf=stat_cap, neginf=0.0), 0.0, stat_cap)
            out["pts"] = np.where(active_mask, pts_new, x)
        elif stat_name == "reb":
            x = pd.to_numeric(out["reb"], errors="coerce").to_numpy(dtype=float, copy=False)
            x = np.clip(np.nan_to_num(x, nan=0.0, posinf=stat_cap, neginf=0.0), 0.0, stat_cap)
            reb_raw = target_mu + sf_var * (x - mu)
            reb_new = np.clip(np.nan_to_num(reb_raw, nan=0.0, posinf=stat_cap, neginf=0.0), 0.0, stat_cap)
            reb_new = np.where(active_mask, reb_new, x)
            if "oreb" in out.columns and "dreb" in out.columns:
                oreb_cap = float(_WORLD_BASE_STAT_CAPS.get("oreb", 25.0))
                dreb_cap = float(_WORLD_BASE_STAT_CAPS.get("dreb", 30.0))
                oreb = pd.to_numeric(out["oreb"], errors="coerce").to_numpy(dtype=float, copy=False)
                oreb = np.clip(np.nan_to_num(oreb, nan=0.0, posinf=oreb_cap, neginf=0.0), 0.0, oreb_cap)
                dreb = pd.to_numeric(out["dreb"], errors="coerce").to_numpy(dtype=float, copy=False)
                dreb = np.clip(np.nan_to_num(dreb, nan=0.0, posinf=dreb_cap, neginf=0.0), 0.0, dreb_cap)
                reb_split_sum = np.maximum(oreb + dreb, 1e-6)
                oreb_share = np.divide(oreb, reb_split_sum)
                oreb_new = np.clip(reb_new * oreb_share, 0.0, oreb_cap)
                dreb_new = np.clip(reb_new * (1.0 - oreb_share), 0.0, dreb_cap)
                out["oreb"] = np.where(active_mask, oreb_new, oreb)
                out["dreb"] = np.where(active_mask, dreb_new, dreb)
            out["reb"] = reb_new
        elif stat_name == "ast":
            x = pd.to_numeric(out["ast"], errors="coerce").to_numpy(dtype=float, copy=False)
            x = np.clip(np.nan_to_num(x, nan=0.0, posinf=stat_cap, neginf=0.0), 0.0, stat_cap)
            ast_raw = target_mu + sf_var * (x - mu)
            ast_new = np.clip(np.nan_to_num(ast_raw, nan=0.0, posinf=stat_cap, neginf=0.0), 0.0, stat_cap)
            out["ast"] = np.where(active_mask, ast_new, x)
        elif stat_name == "stl":
            x = pd.to_numeric(out["stl"], errors="coerce").to_numpy(dtype=float, copy=False)
            x = np.clip(np.nan_to_num(x, nan=0.0, posinf=stat_cap, neginf=0.0), 0.0, stat_cap)
            stl_raw = target_mu + sf_var * (x - mu)
            stl_new = np.clip(np.nan_to_num(stl_raw, nan=0.0, posinf=stat_cap, neginf=0.0), 0.0, stat_cap)
            out["stl"] = np.where(active_mask, stl_new, x)
        elif stat_name == "blk":
            x = pd.to_numeric(out["blk"], errors="coerce").to_numpy(dtype=float, copy=False)
            x = np.clip(np.nan_to_num(x, nan=0.0, posinf=stat_cap, neginf=0.0), 0.0, stat_cap)
            blk_raw = target_mu + sf_var * (x - mu)
            blk_new = np.clip(np.nan_to_num(blk_raw, nan=0.0, posinf=stat_cap, neginf=0.0), 0.0, stat_cap)
            out["blk"] = np.where(active_mask, blk_new, x)
        out = out.drop(columns=["mu", "sf_mean", "sf_var", "line_gap", "player_name"], errors="ignore")

        post_mean_col = f"{stat_name}_mean_post"
        post_means = meta.loc[:, key_cols + [mean_col]].copy().rename(
            columns={mean_col: post_mean_col}
        )
        adjusted_keys = scale_df.loc[:, key_cols].drop_duplicates(ignore_index=True)
        if not adjusted_keys.empty:
            adjusted_out = out.loc[:, key_cols + [stat_name]].copy()
            adjusted_out, _ = _sanitize_frame_to_expected_keys(
                adjusted_out,
                expected_keys_df=adjusted_keys,
                key_cols=tuple(key_cols),
                label=f"props_uplift/{stat_name}_adjusted_subset",
            )
            if not adjusted_out.empty:
                post_delta = _group_mean_by_keys_without_pandas_groupby(
                    adjusted_out,
                    key_cols=key_cols,
                    value_cols=(stat_name,),
                    label=f"props_uplift/{stat_name}_post_means",
                ).rename(columns={stat_name: post_mean_col})
                post_means = _left_overlay_from_source_by_keys(
                    post_means,
                    source_df=post_delta.loc[:, key_cols + [post_mean_col]],
                    key_cols=key_cols,
                    value_cols=(post_mean_col,),
                    label=f"props_uplift/{stat_name}_post_mean_overlay",
                )

        merged_gap = meta.loc[:, key_cols + [mean_col, line_col]].copy()
        merged_gap = _left_overlay_from_source_by_keys(
            merged_gap,
            source_df=post_means.loc[:, key_cols + [post_mean_col]],
            key_cols=key_cols,
            value_cols=(post_mean_col,),
            label=f"props_uplift/{stat_name}_gap_overlay",
        )
        gap_pre = pd.to_numeric(merged_gap[mean_col], errors="coerce") - pd.to_numeric(merged_gap[line_col], errors="coerce")
        gap_post = pd.to_numeric(merged_gap[post_mean_col], errors="coerce") - pd.to_numeric(
            merged_gap[line_col], errors="coerce"
        )
        report["stats"][stat_name] = {
            "applied_player_count": int(len(scale_df)),
            "applied_player_count_up": int(len(up_df)),
            "applied_player_count_down": int(len(down_df)),
            "mean_gap_pre": float(gap_pre.mean()) if gap_pre.notna().any() else float("nan"),
            "mean_gap_post": float(gap_post.mean()) if gap_post.notna().any() else float("nan"),
            "median_gap_pre": float(gap_pre.median()) if gap_pre.notna().any() else float("nan"),
                "median_gap_post": float(gap_post.median()) if gap_post.notna().any() else float("nan"),
                "mean_scale_mean": float(scale_df["sf_mean"].mean()),
                "mean_scale_p90": float(scale_df["sf_mean"].quantile(0.90)),
                "var_scale_mean": float(scale_df["sf_var"].mean()),
                "confidence_mean": float(pd.to_numeric(scale_df["confidence"], errors="coerce").mean()),
                "confidence_p10": float(pd.to_numeric(scale_df["confidence"], errors="coerce").quantile(0.10)),
                "confidence_p90": float(pd.to_numeric(scale_df["confidence"], errors="coerce").quantile(0.90)),
            }
        top_cols = [
            c
            for c in ["player_name", "player_id", "direction", "line_gap", "sf_mean", "sf_var", "confidence"]
            if c in scale_df.columns
        ]
        top_rows = (
            scale_df.loc[:, top_cols]
            .assign(abs_line_gap=lambda d: pd.to_numeric(d["line_gap"], errors="coerce").abs())
            .sort_values("abs_line_gap", ascending=False)
            .head(8)
            .drop(columns=["abs_line_gap"], errors="ignore")
            .replace([np.inf, -np.inf], np.nan)
            .fillna("")
        )
        report["stats"][stat_name]["top_adjustments"] = top_rows.to_dict(orient="records")

    dk_cap = float(_WORLD_DERIVED_STAT_CAPS.get("dk_fpts", 150.0))
    dk_fpts = _recompute_dk_fpts(out).to_numpy(dtype=float, copy=False)
    out["dk_fpts"] = np.clip(np.nan_to_num(dk_fpts, nan=0.0, posinf=dk_cap, neginf=0.0), 0.0, dk_cap)

    report["total_adjustment_events"] = int(
        sum(int((report["stats"].get(s) or {}).get("applied_player_count", 0)) for s in stat_cfg)
    )
    if adjusted_key_frames:
        report["total_adjusted_players"] = int(
            len(pd.concat(adjusted_key_frames, ignore_index=True).drop_duplicates(subset=key_cols))
        )
    else:
        report["total_adjusted_players"] = 0
    return out, report


def _apply_propless_tail_calibration_to_worlds(
    worlds_df: pd.DataFrame,
    *,
    features_df: pd.DataFrame,
    enabled: bool = True,
    min_minutes_mean: float = 21.0,
    min_dk_mean: float = 16.0,
    tail_boost: float = 0.14,
    max_tail_scale: float = 1.22,
    target_game_ids: set[int] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Broaden upper tails for propless players with stable minutes/fpts baselines."""
    if not enabled:
        return worlds_df, {"applied": False, "reason": "disabled"}
    if worlds_df.empty:
        return worlds_df, {"applied": False, "reason": "empty_worlds"}
    if features_df.empty:
        return worlds_df, {"applied": False, "reason": "empty_features"}

    key_cols = ["game_id", "team_id", "player_id"]
    required_world_cols = {"game_id", "team_id", "player_id", "minutes", "dk_fpts"}
    missing_world_cols = sorted(required_world_cols - set(worlds_df.columns))
    if missing_world_cols:
        return worlds_df, {
            "applied": False,
            "reason": "missing_world_cols",
            "missing_world_cols": missing_world_cols,
        }
    missing_feature_cols = sorted(set(key_cols) - set(features_df.columns))
    if missing_feature_cols:
        return worlds_df, {
            "applied": False,
            "reason": "missing_feature_keys",
            "missing_feature_cols": missing_feature_cols,
        }

    indicator_cols: list[str] = []
    if "an_props_market_count" in features_df.columns:
        indicator_cols.append("an_props_market_count")
    has_cols = sorted([col for col in features_df.columns if str(col).startswith("an_has_")])
    line_cols = sorted(
        [
            col
            for col in features_df.columns
            if str(col).startswith("an_") and str(col).endswith("_line")
        ]
    )
    indicator_cols.extend(has_cols)
    indicator_cols.extend(line_cols)
    if not indicator_cols:
        return worlds_df, {
            "applied": False,
            "reason": "missing_props_indicator_columns",
        }

    tail_boost_clipped = float(np.clip(float(tail_boost), 0.0, 0.50))
    max_tail_scale_clipped = float(max(1.0, float(max_tail_scale)))
    min_minutes_mean_f = float(max(0.0, float(min_minutes_mean)))
    min_dk_mean_f = float(max(0.0, float(min_dk_mean)))
    if tail_boost_clipped <= 0.0:
        return worlds_df, {
            "applied": False,
            "reason": "tail_boost_zero",
            "tail_boost": tail_boost_clipped,
        }

    feat_cols = key_cols + [col for col in indicator_cols if col in features_df.columns]
    feat = features_df.loc[:, feat_cols].copy()
    agg_dict: dict[str, str] = {}
    for col in feat.columns:
        if col in key_cols:
            continue
        if str(col).startswith("an_has_"):
            agg_dict[col] = "max"
        elif str(col) == "an_props_market_count":
            agg_dict[col] = "max"
        else:
            agg_dict[col] = "first"
    feat = feat.groupby(key_cols, dropna=False, as_index=False).agg(agg_dict)

    has_any_props = np.zeros(len(feat), dtype=bool)
    has_explicit_props_indicator = False
    if "an_has_any_props" in feat.columns:
        has_explicit_props_indicator = True
        has_any_props |= (
            pd.to_numeric(feat["an_has_any_props"], errors="coerce")
            .fillna(0.0)
            .ge(0.5)
            .to_numpy(dtype=bool)
        )
    if "an_props_market_count" in feat.columns:
        has_explicit_props_indicator = True
        has_any_props |= (
            pd.to_numeric(feat["an_props_market_count"], errors="coerce")
            .fillna(0.0)
            .ge(1.0)
            .to_numpy(dtype=bool)
        )
    for col in has_cols:
        if col in feat.columns:
            has_explicit_props_indicator = True
            has_any_props |= (
                pd.to_numeric(feat[col], errors="coerce")
                .fillna(0.0)
                .ge(0.5)
                .to_numpy(dtype=bool)
            )
    # Some live frames default-fill *_line columns (often 0.0) for all rows.
    # Only use line columns as a fallback signal when explicit has/market
    # indicators are unavailable.
    if not has_explicit_props_indicator:
        for col in line_cols:
            if col in feat.columns:
                has_any_props |= (
                    pd.to_numeric(feat[col], errors="coerce")
                    .fillna(0.0)
                    .abs()
                    .gt(float(_WORLD_CONTRACT_TOL))
                    .to_numpy(dtype=bool)
                )
    feat["has_any_props"] = has_any_props.astype(np.int8)

    stat_cols = [col for col in ("pts", "reb", "ast", "stl", "blk") if col in worlds_df.columns]
    if not stat_cols:
        return worlds_df, {"applied": False, "reason": "missing_tail_stat_columns"}
    mean_cols = ["minutes", "dk_fpts", *stat_cols]
    player_means = _group_mean_by_keys_without_pandas_groupby(
        worlds_df,
        key_cols=key_cols,
        value_cols=mean_cols,
        label="propless_tail/player_means",
    )
    if player_means.empty:
        return worlds_df, {"applied": False, "reason": "empty_player_means"}
    player_means = player_means.rename(
        columns={
            "minutes": "minutes_mean",
            "dk_fpts": "dk_fpts_mean",
            **{col: f"{col}_mean" for col in stat_cols},
        }
    )
    meta = _left_overlay_from_source_by_keys(
        player_means,
        source_df=feat.loc[:, key_cols + ["has_any_props"]],
        key_cols=key_cols,
        value_cols=("has_any_props",),
        label="propless_tail/feature_overlay",
    )
    has_props = (
        pd.to_numeric(meta.get("has_any_props", 0.0), errors="coerce")
        .fillna(0.0)
        .ge(0.5)
        .to_numpy(dtype=bool)
    )
    minutes_mean = pd.to_numeric(meta["minutes_mean"], errors="coerce").fillna(0.0)
    dk_mean = pd.to_numeric(meta["dk_fpts_mean"], errors="coerce").fillna(0.0)
    propless = ~has_props
    minutes_strength = (
        (minutes_mean - min_minutes_mean_f) / max(1.0, 30.0 - min_minutes_mean_f)
    ).clip(lower=0.0, upper=1.0)
    dk_strength = ((dk_mean - min_dk_mean_f) / max(1.0, 35.0 - min_dk_mean_f)).clip(
        lower=0.0, upper=1.0
    )
    confidence = (0.6 * minutes_strength + 0.4 * dk_strength).clip(lower=0.0, upper=1.0)
    eligible = propless & confidence.gt(0.0)
    if not bool(eligible.any()):
        return worlds_df, {
            "applied": False,
            "reason": "no_eligible_propless_players",
            "propless_player_count": int(np.count_nonzero(propless)),
        }

    tail_scale = (
        1.0
        + tail_boost_clipped
        * pd.to_numeric(confidence, errors="coerce").fillna(0.0)
    ).clip(lower=1.0, upper=max_tail_scale_clipped)
    eligible_players = meta.loc[eligible, key_cols].copy()
    eligible_players["propless_tail_scale"] = tail_scale.loc[eligible].astype(float).values
    eligible_players["propless_tail_confidence"] = confidence.loc[eligible].astype(float).values
    if eligible_players.empty:
        return worlds_df, {"applied": False, "reason": "no_eligible_player_rows"}

    out = worlds_df.copy()
    out = _left_overlay_from_source_by_keys(
        out,
        source_df=eligible_players,
        key_cols=key_cols,
        value_cols=("propless_tail_scale", "propless_tail_confidence"),
        label="propless_tail/scale_overlay",
        copy_base=False,
    )
    row_scale = (
        pd.to_numeric(out.get("propless_tail_scale", 1.0), errors="coerce")
        .fillna(1.0)
        .clip(lower=1.0, upper=max_tail_scale_clipped)
        .to_numpy(dtype=float, copy=False)
    )
    row_eligible = row_scale > 1.0 + _WORLD_CONTRACT_TOL
    if target_game_ids:
        game_ids = pd.to_numeric(out["game_id"], errors="coerce").astype("Int64")
        row_eligible &= game_ids.isin(sorted(target_game_ids)).to_numpy(dtype=bool)
    if not bool(np.any(row_eligible)):
        out = out.drop(columns=["propless_tail_scale", "propless_tail_confidence"], errors="ignore")
        return out, {
            "applied": False,
            "reason": "no_eligible_world_rows",
            "eligible_player_count": int(len(eligible_players)),
            "target_game_count": int(len(target_game_ids or set())),
        }

    mean_overlay = player_means.loc[:, key_cols + [f"{col}_mean" for col in stat_cols]]
    out = _left_overlay_from_source_by_keys(
        out,
        source_df=mean_overlay,
        key_cols=key_cols,
        value_cols=[f"{col}_mean" for col in stat_cols],
        label="propless_tail/mean_overlay",
        copy_base=False,
    )

    for col in stat_cols:
        mean_col = f"{col}_mean"
        cap = float(_WORLD_BASE_STAT_CAPS.get(col, 1_000.0))
        x = pd.to_numeric(out[col], errors="coerce").fillna(0.0).to_numpy(dtype=float, copy=False)
        mu = (
            pd.to_numeric(out.get(mean_col, 0.0), errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=float, copy=False)
        )
        resid = x - mu
        resid_pos = np.maximum(resid, 0.0)
        resid_neg = np.minimum(resid, 0.0)
        x_new = np.clip(
            np.nan_to_num(mu + resid_neg + row_scale * resid_pos, nan=0.0, posinf=cap, neginf=0.0),
            0.0,
            cap,
        )
        out[col] = np.where(row_eligible, x_new, x)
        out = out.drop(columns=[mean_col], errors="ignore")

    if {"oreb", "dreb", "reb"}.issubset(out.columns):
        oreb_cap = float(_WORLD_BASE_STAT_CAPS.get("oreb", 25.0))
        dreb_cap = float(_WORLD_BASE_STAT_CAPS.get("dreb", 30.0))
        oreb = pd.to_numeric(out["oreb"], errors="coerce").fillna(0.0).to_numpy(dtype=float, copy=False)
        dreb = pd.to_numeric(out["dreb"], errors="coerce").fillna(0.0).to_numpy(dtype=float, copy=False)
        reb = pd.to_numeric(out["reb"], errors="coerce").fillna(0.0).to_numpy(dtype=float, copy=False)
        split_sum = np.maximum(oreb + dreb, 1e-6)
        oreb_share = np.divide(oreb, split_sum)
        oreb_new = np.clip(reb * oreb_share, 0.0, oreb_cap)
        dreb_new = np.clip(reb * (1.0 - oreb_share), 0.0, dreb_cap)
        out["oreb"] = np.where(row_eligible, oreb_new, oreb)
        out["dreb"] = np.where(row_eligible, dreb_new, dreb)

    dk_cap = float(_WORLD_DERIVED_STAT_CAPS.get("dk_fpts", 150.0))
    dk = _recompute_dk_fpts(out).to_numpy(dtype=float, copy=False)
    out["dk_fpts"] = np.clip(np.nan_to_num(dk, nan=0.0, posinf=dk_cap, neginf=0.0), 0.0, dk_cap)
    out = out.drop(columns=["propless_tail_scale", "propless_tail_confidence"], errors="ignore")

    report = {
        "applied": True,
        "eligible_player_count": int(len(eligible_players)),
        "eligible_world_row_count": int(np.count_nonzero(row_eligible)),
        "target_game_count": int(len(target_game_ids or set())),
        "min_minutes_mean": min_minutes_mean_f,
        "min_dk_mean": min_dk_mean_f,
        "tail_boost": tail_boost_clipped,
        "max_tail_scale": max_tail_scale_clipped,
        "scale_mean": float(np.mean(row_scale[row_eligible])) if bool(np.any(row_eligible)) else 1.0,
        "scale_p90": float(np.quantile(row_scale[row_eligible], 0.90)) if bool(np.any(row_eligible)) else 1.0,
    }
    return out, report


def _build_pre_calibration_points_anchor(
    worlds_df: pd.DataFrame,
    *,
    label: str,
) -> pd.DataFrame:
    key_cols = ["game_id", "team_id", "player_id"]
    required_cols = set(key_cols + ["pts"])
    if worlds_df.empty or not required_cols.issubset(worlds_df.columns):
        return pd.DataFrame(columns=key_cols + ["pts_pre_calibration_mean"])
    anchor = _group_mean_by_keys_without_pandas_groupby(
        worlds_df.loc[:, key_cols + ["pts"]],
        key_cols=key_cols,
        value_cols=("pts",),
        label=label,
    )
    return anchor.rename(columns={"pts": "pts_pre_calibration_mean"})


def _allocate_bounded_budget(
    *,
    budget: float,
    weights: np.ndarray,
    capacity: np.ndarray,
    max_iter: int = 16,
) -> np.ndarray:
    alloc = np.zeros_like(capacity, dtype=float)
    remaining = float(max(0.0, budget))
    if remaining <= 0.0:
        return alloc
    tol = 1e-9
    weight_arr = np.clip(
        np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0),
        0.0,
        np.inf,
    )
    cap_arr = np.clip(
        np.nan_to_num(capacity, nan=0.0, posinf=0.0, neginf=0.0),
        0.0,
        np.inf,
    )
    for _ in range(max_iter):
        spare = np.clip(cap_arr - alloc, 0.0, np.inf)
        active = spare > tol
        if remaining <= tol or not bool(np.any(active)):
            break
        active_weights = weight_arr[active]
        if float(active_weights.sum()) > tol:
            proposal = remaining * active_weights / float(active_weights.sum())
        else:
            proposal = remaining * spare[active] / float(spare[active].sum())
        delta = np.minimum(proposal, spare[active])
        if float(delta.sum()) <= tol:
            break
        alloc[active] += delta
        remaining = float(max(0.0, remaining - float(delta.sum())))
    return np.clip(alloc, 0.0, cap_arr)


def _apply_team_implied_points_reconcile_to_worlds(
    worlds_df: pd.DataFrame,
    *,
    features_df: pd.DataFrame,
    pre_calibration_pts_anchor: pd.DataFrame | None,
    enabled: bool = False,
    alpha: float = 0.75,
    deadband_points: float = 2.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not enabled:
        return worlds_df, {"applied": False, "reason": "disabled"}
    if worlds_df.empty:
        return worlds_df, {"applied": False, "reason": "empty_worlds"}
    if features_df.empty:
        return worlds_df, {"applied": False, "reason": "empty_features"}

    key_cols = ["game_id", "team_id", "player_id"]
    required_world_cols = set(key_cols + ["minutes", "pts", "dk_fpts"])
    missing_world_cols = sorted(required_world_cols - set(worlds_df.columns))
    if missing_world_cols:
        return worlds_df, {
            "applied": False,
            "reason": "missing_world_cols",
            "missing_world_cols": missing_world_cols,
        }
    if "team_implied_total" not in features_df.columns:
        return worlds_df, {
            "applied": False,
            "reason": "missing_team_implied_total",
        }

    alpha_clipped = float(np.clip(float(alpha), 0.0, 1.0))
    deadband_clipped = float(max(0.0, deadband_points))
    if alpha_clipped <= 0.0:
        return worlds_df, {
            "applied": False,
            "reason": "alpha_zero",
            "alpha": alpha_clipped,
            "deadband_points": deadband_clipped,
        }

    current_means = _group_mean_by_keys_without_pandas_groupby(
        worlds_df.loc[:, key_cols + ["minutes", "pts"]],
        key_cols=key_cols,
        value_cols=("minutes", "pts"),
        label="team_implied_points_reconcile/player_means",
    ).rename(columns={"minutes": "minutes_mean", "pts": "pts_mean"})

    feat_cols = key_cols + ["team_implied_total"]
    indicator_cols: list[str] = []
    if "an_props_market_count" in features_df.columns:
        indicator_cols.append("an_props_market_count")
    has_cols = sorted(
        [col for col in features_df.columns if str(col).startswith("an_has_")]
    )
    line_cols = sorted(
        [
            col
            for col in features_df.columns
            if str(col).startswith("an_") and str(col).endswith("_line")
        ]
    )
    indicator_cols.extend(has_cols)
    indicator_cols.extend(line_cols)
    for optional_col in (
        "player_name",
        "prior_play_prob",
        "lineup_role",
        *indicator_cols,
    ):
        if optional_col in features_df.columns:
            feat_cols.append(optional_col)
    feat = features_df.loc[:, feat_cols].copy()
    agg_dict: dict[str, str] = {}
    for col in feat.columns:
        if col in key_cols:
            continue
        if str(col).startswith("an_has_") or col == "an_props_market_count":
            agg_dict[col] = "max"
        else:
            agg_dict[col] = "first"
    feat = feat.groupby(key_cols, dropna=False, as_index=False).agg(agg_dict)

    has_any_props = np.zeros(len(feat), dtype=bool)
    has_explicit_props_indicator = False
    if "an_has_any_props" in feat.columns:
        has_explicit_props_indicator = True
        has_any_props |= (
            pd.to_numeric(feat["an_has_any_props"], errors="coerce")
            .fillna(0.0)
            .ge(0.5)
            .to_numpy(dtype=bool)
        )
    if "an_props_market_count" in feat.columns:
        has_explicit_props_indicator = True
        has_any_props |= (
            pd.to_numeric(feat["an_props_market_count"], errors="coerce")
            .fillna(0.0)
            .ge(1.0)
            .to_numpy(dtype=bool)
        )
    for col in has_cols:
        if col in feat.columns:
            has_explicit_props_indicator = True
            has_any_props |= (
                pd.to_numeric(feat[col], errors="coerce")
                .fillna(0.0)
                .ge(0.5)
                .to_numpy(dtype=bool)
            )
    if not has_explicit_props_indicator:
        for col in line_cols:
            if col in feat.columns:
                has_any_props |= (
                    pd.to_numeric(feat[col], errors="coerce")
                    .fillna(0.0)
                    .abs()
                    .gt(float(_WORLD_CONTRACT_TOL))
                    .to_numpy(dtype=bool)
                )
    feat["has_any_props"] = has_any_props.astype(np.int8)

    meta = current_means.merge(feat, on=key_cols, how="left")
    meta["pts_mean"] = pd.to_numeric(meta["pts_mean"], errors="coerce").fillna(0.0)
    meta["minutes_mean"] = pd.to_numeric(meta["minutes_mean"], errors="coerce").fillna(
        0.0
    )
    meta["team_implied_total"] = pd.to_numeric(
        meta["team_implied_total"], errors="coerce"
    )
    meta["has_any_props"] = (
        pd.to_numeric(meta.get("has_any_props", 0.0), errors="coerce")
        .fillna(0.0)
        .ge(0.5)
    )
    prior_play_prob_source = (
        meta["prior_play_prob"]
        if "prior_play_prob" in meta.columns
        else pd.Series(1.0, index=meta.index, dtype=float)
    )
    prior_play_prob = (
        pd.to_numeric(prior_play_prob_source, errors="coerce")
        .fillna(1.0)
        .clip(lower=0.0, upper=1.0)
    )
    role_series = meta.get("lineup_role", "")
    role_norm = (
        role_series.astype(str).str.lower().str.strip()
        if hasattr(role_series, "astype")
        else pd.Series("", index=meta.index, dtype=str)
    )
    role_block = role_norm.isin(
        {"out", "inactive", "dnp", "dnp-cd", "g_league", "two_way", "two-way"}
    )
    depth_rank = (
        meta.assign(_minutes_mean_rank=meta["minutes_mean"].astype(float))
        .groupby(["game_id", "team_id"], dropna=False)["_minutes_mean_rank"]
        .rank(method="first", ascending=False)
    )
    minutes_strength = (
        (
            meta["minutes_mean"].astype(float)
            - float(_TEAM_IMPLIED_UNCOVERED_ADD_MIN_MINUTES_MEAN)
        )
        / max(1.0, 30.0 - float(_TEAM_IMPLIED_UNCOVERED_ADD_MIN_MINUTES_MEAN))
    ).clip(lower=0.0, upper=1.0)
    depth_strength = (
        1.0
        - (
            (
                pd.to_numeric(depth_rank, errors="coerce")
                .fillna(float(_TEAM_IMPLIED_UNCOVERED_MAX_DEPTH_RANK + 1))
                - 1.0
            )
            / max(1.0, float(_TEAM_IMPLIED_UNCOVERED_MAX_DEPTH_RANK - 1))
        )
    ).clip(lower=0.0, upper=1.0)
    upside_score = (
        0.55 * pd.to_numeric(minutes_strength, errors="coerce").fillna(0.0)
        + 0.30 * pd.to_numeric(prior_play_prob, errors="coerce").fillna(0.0)
        + 0.15 * pd.to_numeric(depth_strength, errors="coerce").fillna(0.0)
    ).clip(lower=0.0, upper=1.0)

    meta["is_covered"] = meta["has_any_props"].astype(bool)
    meta["movable_uncovered"] = (
        (~meta["is_covered"]) & meta["minutes_mean"].gt(0.0) & (~role_block)
    )
    meta["add_eligible_uncovered"] = (
        meta["movable_uncovered"]
        & meta["minutes_mean"].ge(float(_TEAM_IMPLIED_UNCOVERED_ADD_MIN_MINUTES_MEAN))
        & prior_play_prob.ge(float(_TEAM_IMPLIED_UNCOVERED_ADD_MIN_PRIOR_PLAY_PROB))
        & depth_rank.le(float(_TEAM_IMPLIED_UNCOVERED_MAX_DEPTH_RANK))
    )
    meta["upside_score"] = pd.to_numeric(upside_score, errors="coerce").fillna(0.0)

    team_rows: list[dict[str, Any]] = []
    adjustments: list[pd.DataFrame] = []
    team_count_with_uncovered_gap = 0
    team_count_adjusted = 0
    total_mean_delta = 0.0
    total_unresolved_team_gap_mean = 0.0

    team_keys = (
        meta.loc[:, ["game_id", "team_id", "team_implied_total"]]
        .dropna(subset=["game_id", "team_id"])
        .drop_duplicates(subset=["game_id", "team_id"], keep="last")
    )
    for row in team_keys.itertuples(index=False):
        game_id = int(float(row.game_id))
        team_id = int(float(row.team_id))
        target = pd.to_numeric(row.team_implied_total, errors="coerce")
        if pd.isna(target):
            continue
        team_mask = meta["game_id"].eq(game_id) & meta["team_id"].eq(team_id)
        team_current = float(meta.loc[team_mask, "pts_mean"].sum())
        team_gap_pre = float(team_current - float(target))
        covered_sum = float(meta.loc[team_mask & meta["is_covered"], "pts_mean"].sum())
        uncovered_locked_sum = float(
            meta.loc[
                team_mask & (~meta["is_covered"]) & (~meta["movable_uncovered"]),
                "pts_mean",
            ].sum()
        )
        movable_mask = team_mask & meta["movable_uncovered"]
        movable_current = float(meta.loc[movable_mask, "pts_mean"].sum())
        target_movable_total = float(
            max(
                0.0,
                float(target) - covered_sum - uncovered_locked_sum - deadband_clipped,
            )
        )
        movable_delta_needed = float(target_movable_total - movable_current)
        unresolved_gap = float(team_gap_pre)
        moved_mean = 0.0

        if abs(movable_delta_needed) > 1e-9 and bool(movable_mask.any()):
            team_count_with_uncovered_gap += 1
            movable = meta.loc[
                movable_mask,
                key_cols + ["pts_mean", "upside_score", "add_eligible_uncovered"],
            ].copy()
            current_pts = (
                pd.to_numeric(movable["pts_mean"], errors="coerce")
                .fillna(0.0)
                .to_numpy(dtype=float, copy=False)
            )
            upside = (
                pd.to_numeric(movable["upside_score"], errors="coerce")
                .fillna(0.0)
                .to_numpy(dtype=float, copy=False)
            )
            target_pts = current_pts.copy()

            if movable_delta_needed < 0.0:
                remove_budget = float(-movable_delta_needed * alpha_clipped)
                floor_frac = np.clip(0.05 + 0.15 * upside, 0.05, 0.20)
                remove_capacity = np.clip(current_pts * (1.0 - floor_frac), 0.0, np.inf)
                remove_weights = np.clip(current_pts * (1.35 - upside), 0.0, np.inf)
                removed = _allocate_bounded_budget(
                    budget=remove_budget,
                    weights=remove_weights,
                    capacity=remove_capacity,
                )
                target_pts = np.clip(current_pts - removed, 0.0, np.inf)
                moved_mean = float(-removed.sum())
            else:
                add_budget = float(movable_delta_needed * alpha_clipped)
                add_eligible = (
                    pd.to_numeric(
                        movable["add_eligible_uncovered"], errors="coerce"
                    )
                    .fillna(0.0)
                    .ge(0.5)
                    .to_numpy(dtype=bool)
                )
                add_capacity = np.zeros_like(current_pts, dtype=float)
                add_weights = np.zeros_like(current_pts, dtype=float)
                add_capacity[add_eligible] = current_pts[add_eligible] * (
                    0.50 + 1.00 * upside[add_eligible]
                )
                add_weights[add_eligible] = current_pts[add_eligible] * (
                    0.25 + upside[add_eligible]
                )
                added = _allocate_bounded_budget(
                    budget=add_budget,
                    weights=add_weights,
                    capacity=add_capacity,
                )
                target_pts = np.clip(current_pts + added, 0.0, np.inf)
                moved_mean = float(added.sum())

            delta_pts = target_pts - current_pts
            if bool(np.any(np.abs(delta_pts) > 1e-9)):
                team_count_adjusted += 1
                total_mean_delta += float(np.abs(delta_pts).sum())
                scale = np.where(current_pts > 1e-9, target_pts / current_pts, 1.0)
                movable["pts_reconcile_scale"] = np.clip(scale, 0.0, np.inf)
                movable["pts_reconcile_delta_mean"] = delta_pts
                movable["pts_reconcile_direction"] = np.where(
                    delta_pts > 1e-9,
                    "add",
                    np.where(delta_pts < -1e-9, "remove", "flat"),
                )
                adjustments.append(
                    movable.loc[
                        np.abs(
                            pd.to_numeric(
                                movable["pts_reconcile_delta_mean"],
                                errors="coerce",
                            ).to_numpy(dtype=float, copy=False)
                        )
                        > 1e-9,
                        key_cols
                        + [
                            "pts_reconcile_scale",
                            "pts_reconcile_delta_mean",
                            "pts_reconcile_direction",
                        ],
                    ].copy()
                )
                unresolved_gap = float(
                    covered_sum
                    + uncovered_locked_sum
                    + float(target_pts.sum())
                    + deadband_clipped
                    - float(target)
                )

        total_unresolved_team_gap_mean += unresolved_gap
        team_rows.append(
            {
                "game_id": game_id,
                "team_id": team_id,
                "team_pts_mean_pre": float(round(team_current, 6)),
                "team_implied_total": float(round(float(target), 6)),
                "covered_pts_mean": float(round(covered_sum, 6)),
                "uncovered_locked_pts_mean": float(round(uncovered_locked_sum, 6)),
                "uncovered_movable_pts_mean_pre": float(round(movable_current, 6)),
                "uncovered_target_pts_mean": float(round(target_movable_total, 6)),
                "team_gap_pre": float(round(team_gap_pre, 6)),
                "uncovered_delta_needed_mean": float(round(movable_delta_needed, 6)),
                "uncovered_moved_mean": float(round(moved_mean, 6)),
                "unresolved_team_gap_mean": float(round(unresolved_gap, 6)),
            }
        )

    if not adjustments:
        return worlds_df, {
            "applied": False,
            "reason": "no_adjustable_uncovered_pool",
            "mode": "uncovered_residual_allocator",
            "alpha": alpha_clipped,
            "deadband_points": deadband_clipped,
            "team_count_with_uncovered_gap": int(team_count_with_uncovered_gap),
            "team_count_adjusted": 0,
            "player_count_adjusted": 0,
            "total_mean_delta": 0.0,
            "total_unresolved_team_gap_mean": float(
                round(total_unresolved_team_gap_mean, 6)
            ),
            "teams": team_rows,
        }

    adjustment_df = pd.concat(adjustments, ignore_index=True)
    adjustment_df = (
        adjustment_df.groupby(key_cols, dropna=False, as_index=False)
        .agg(
            {
                "pts_reconcile_scale": "prod",
                "pts_reconcile_delta_mean": "sum",
                "pts_reconcile_direction": "last",
            }
        )
    )

    out = _left_overlay_from_source_by_keys(
        worlds_df.copy(),
        source_df=adjustment_df,
        key_cols=key_cols,
        value_cols=("pts_reconcile_scale",),
        label="team_implied_points_reconcile/scale_overlay",
    )
    pts_cap = float(_WORLD_BASE_STAT_CAPS.get("pts", 90.0))
    pts_vals = pd.to_numeric(out["pts"], errors="coerce").to_numpy(dtype=float, copy=False)
    pts_vals = np.clip(
        np.nan_to_num(pts_vals, nan=0.0, posinf=pts_cap, neginf=0.0),
        0.0,
        pts_cap,
    )
    scale_vals = (
        pd.to_numeric(out.get("pts_reconcile_scale", 1.0), errors="coerce")
        .fillna(1.0)
        .to_numpy(dtype=float, copy=False)
    )
    scale_vals = np.clip(scale_vals, 0.0, np.inf)
    minutes_vals = pd.to_numeric(out["minutes"], errors="coerce").to_numpy(
        dtype=float, copy=False
    )
    active_mask = np.nan_to_num(minutes_vals, nan=0.0, posinf=0.0, neginf=0.0) > 0.0
    out["pts"] = np.where(
        active_mask,
        np.clip(pts_vals * scale_vals, 0.0, pts_cap),
        pts_vals,
    )
    dk_cap = float(_WORLD_DERIVED_STAT_CAPS.get("dk_fpts", 150.0))
    dk_fpts = _recompute_dk_fpts(out).to_numpy(dtype=float, copy=False)
    out["dk_fpts"] = np.clip(
        np.nan_to_num(dk_fpts, nan=0.0, posinf=dk_cap, neginf=0.0),
        0.0,
        dk_cap,
    )
    out = out.drop(columns=["pts_reconcile_scale"], errors="ignore")

    top_adjustments = adjustment_df.merge(
        feat.loc[:, [c for c in key_cols + ["player_name"] if c in feat.columns]],
        on=key_cols,
        how="left",
    )
    top_adjustments = (
        top_adjustments.assign(
            abs_pts_reconcile_delta_mean=lambda d: pd.to_numeric(
                d["pts_reconcile_delta_mean"], errors="coerce"
            ).abs()
        )
        .sort_values("abs_pts_reconcile_delta_mean", ascending=False)
        .head(12)
        .drop(columns=["abs_pts_reconcile_delta_mean"], errors="ignore")
        .replace([np.inf, -np.inf], np.nan)
        .fillna("")
    )
    report = {
        "applied": True,
        "mode": "uncovered_residual_allocator",
        "alpha": alpha_clipped,
        "deadband_points": deadband_clipped,
        "team_count_with_uncovered_gap": int(team_count_with_uncovered_gap),
        "team_count_adjusted": int(team_count_adjusted),
        "player_count_adjusted": int(len(adjustment_df)),
        "total_mean_delta": float(round(total_mean_delta, 6)),
        "total_unresolved_team_gap_mean": float(
            round(total_unresolved_team_gap_mean, 6)
        ),
        "eligible_rules": {
            "add_min_minutes_mean": float(_TEAM_IMPLIED_UNCOVERED_ADD_MIN_MINUTES_MEAN),
            "add_min_prior_play_prob": float(
                _TEAM_IMPLIED_UNCOVERED_ADD_MIN_PRIOR_PLAY_PROB
            ),
            "max_depth_rank": int(_TEAM_IMPLIED_UNCOVERED_MAX_DEPTH_RANK),
        },
        "teams": team_rows,
        "top_adjustments": top_adjustments.to_dict(orient="records"),
    }
    return out, report


def _apply_team_dk_fpts_correlation_overlay_to_worlds(
    worlds_df: pd.DataFrame,
    *,
    enabled: bool,
    alpha: float,
    min_minutes: float = 0.0,
    weight_power: float = 1.0,
    target_game_ids: set[int] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Increase same-team dk_fpts covariance while preserving team world totals."""
    if not enabled:
        return worlds_df, {"applied": False, "reason": "disabled"}
    if worlds_df.empty:
        return worlds_df, {"applied": False, "reason": "empty_worlds"}

    required = {"world_idx", "game_id", "team_id", "player_id", "dk_fpts"}
    missing = sorted(required - set(worlds_df.columns))
    if missing:
        return worlds_df, {
            "applied": False,
            "reason": f"missing_columns:{','.join(missing)}",
        }

    alpha_clipped = float(np.clip(float(alpha), 0.0, 1.0))
    if alpha_clipped <= 0.0:
        return worlds_df, {"applied": False, "reason": "alpha_zero"}

    out = worlds_df.copy()
    dk = pd.to_numeric(out["dk_fpts"], errors="coerce").fillna(0.0)
    minutes = (
        pd.to_numeric(out["minutes"], errors="coerce").fillna(0.0)
        if "minutes" in out.columns
        else pd.Series(1.0, index=out.index, dtype=float)
    )
    active = (
        pd.to_numeric(out["active"], errors="coerce").fillna(0.0).gt(0.0)
        if "active" in out.columns
        else pd.Series(True, index=out.index, dtype=bool)
    )
    target_mask = pd.Series(True, index=out.index, dtype=bool)
    if target_game_ids is not None:
        normalized_target_ids = _normalize_game_ids(target_game_ids)
        if not normalized_target_ids:
            return worlds_df, {"applied": False, "reason": "no_target_games"}
        if "game_id" not in out.columns:
            return worlds_df, {"applied": False, "reason": "missing_game_id"}
        target_mask = pd.to_numeric(out["game_id"], errors="coerce").astype("Int64").isin(
            normalized_target_ids
        )
        if not bool(target_mask.any()):
            return worlds_df, {"applied": False, "reason": "no_target_rows"}
    eligible = (
        target_mask
        & active
        & minutes.gt(float(min_minutes))
        & dk.gt(float(_WORLD_CONTRACT_TOL))
    )
    eligible_count = int(eligible.sum())
    if eligible_count <= 0:
        return worlds_df, {"applied": False, "reason": "no_eligible_rows"}

    team_player_keys = ["game_id", "team_id", "player_id"]
    world_team_keys = ["world_idx", "game_id", "team_id"]

    player_mean = dk.groupby([out[k] for k in team_player_keys], sort=False).transform(
        "mean"
    )
    weight_base = np.power(
        np.clip(
            player_mean.to_numpy(dtype=float, copy=False),
            a_min=0.0,
            a_max=None,
        ),
        float(weight_power),
    )
    weight_base = np.where(np.isfinite(weight_base), weight_base, 0.0)
    weight_base = np.where(weight_base > float(_WORLD_CONTRACT_TOL), weight_base, 0.0)

    eligible_np = eligible.to_numpy(dtype=bool, copy=False)
    eligible_float = eligible_np.astype(float, copy=False)
    dk_np = dk.to_numpy(dtype=float, copy=False)
    player_mean_np = player_mean.to_numpy(dtype=float, copy=False)

    active_weight = weight_base * eligible_float
    active_mean_component = player_mean_np * eligible_float
    active_dk_component = dk_np * eligible_float

    active_weight_sum = (
        pd.Series(active_weight, index=out.index)
        .groupby([out[k] for k in world_team_keys], sort=False)
        .transform("sum")
        .to_numpy(dtype=float, copy=False)
    )
    world_weight = np.divide(
        active_weight,
        active_weight_sum,
        out=np.zeros_like(active_weight),
        where=active_weight_sum > float(_WORLD_CONTRACT_TOL),
    )

    active_mean_total = (
        pd.Series(active_mean_component, index=out.index)
        .groupby([out[k] for k in world_team_keys], sort=False)
        .transform("sum")
        .to_numpy(dtype=float, copy=False)
    )
    active_dk_total = (
        pd.Series(active_dk_component, index=out.index)
        .groupby([out[k] for k in world_team_keys], sort=False)
        .transform("sum")
        .to_numpy(dtype=float, copy=False)
    )
    team_total = (
        dk.groupby([out[k] for k in world_team_keys], sort=False)
        .transform("sum")
        .to_numpy(dtype=float, copy=False)
    )
    active_residual = active_dk_total - active_mean_total
    common_component = world_weight * active_residual
    idio_component = dk_np - player_mean_np - common_component

    updated = dk_np.copy()
    updated[eligible_np] = (
        player_mean_np[eligible_np]
        + common_component[eligible_np]
        + (1.0 - alpha_clipped) * idio_component[eligible_np]
    )
    provisional_negative_rows = int(np.count_nonzero(eligible_np & (updated < 0.0)))
    updated = np.where(eligible_np, np.maximum(updated, 0.0), updated)

    updated_total = (
        pd.Series(updated, index=out.index)
        .groupby([out[k] for k in world_team_keys], sort=False)
        .transform("sum")
        .to_numpy(dtype=float, copy=False)
    )
    updated = updated + (world_weight * (team_total - updated_total))
    updated = np.where(eligible_np, np.maximum(updated, 0.0), updated)

    updated_total_final = (
        pd.Series(updated, index=out.index)
        .groupby([out[k] for k in world_team_keys], sort=False)
        .transform("sum")
        .to_numpy(dtype=float, copy=False)
    )
    updated = updated + (world_weight * (team_total - updated_total_final))
    updated = np.where(eligible_np, updated, dk_np)
    out["dk_fpts"] = updated

    updated_player_mean = (
        pd.Series(updated, index=out.index)
        .groupby([out[k] for k in team_player_keys], sort=False)
        .transform("mean")
        .to_numpy(dtype=float, copy=False)
    )
    mean_shift = updated_player_mean - player_mean_np
    team_total_after = (
        pd.Series(updated, index=out.index)
        .groupby([out[k] for k in world_team_keys], sort=False)
        .transform("sum")
        .to_numpy(dtype=float, copy=False)
    )

    report = {
        "applied": True,
        "alpha": alpha_clipped,
        "min_minutes": float(min_minutes),
        "weight_power": float(weight_power),
        "eligible_rows": eligible_count,
        "eligible_player_count": int(out.loc[eligible, "player_id"].nunique()),
        "eligible_team_count": int(
            out.loc[eligible, ["game_id", "team_id"]].drop_duplicates().shape[0]
        ),
        "provisional_negative_rows": provisional_negative_rows,
        "player_mean_max_abs_shift": float(np.max(np.abs(mean_shift)))
        if len(mean_shift)
        else 0.0,
        "player_mean_mean_abs_shift": float(np.mean(np.abs(mean_shift)))
        if len(mean_shift)
        else 0.0,
        "team_total_max_abs_drift": float(np.max(np.abs(team_total_after - team_total)))
        if len(team_total_after)
        else 0.0,
        "team_total_mean_abs_drift": float(
            np.mean(np.abs(team_total_after - team_total))
        )
        if len(team_total_after)
        else 0.0,
    }
    if target_game_ids is not None:
        report["target_game_ids"] = sorted(int(gid) for gid in _normalize_game_ids(target_game_ids))
    return out, report


def _apply_mid_minutes_tail_calibration_to_worlds(
    worlds_df: pd.DataFrame,
    *,
    enabled: bool = True,
    min_minutes: float = 12.0,
    max_minutes: float = 20.0,
    tail_boost: float = 0.14,
    target_game_ids: set[int] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Targeted upper-tail lift for 12-20 minute rows while preserving lower tails."""
    if not enabled:
        return worlds_df, {"applied": False, "reason": "disabled"}
    if worlds_df.empty:
        return worlds_df, {"applied": False, "reason": "empty_worlds"}

    required = {"game_id", "team_id", "player_id", "minutes", "dk_fpts"}
    missing = sorted(required - set(worlds_df.columns))
    if missing:
        return worlds_df, {
            "applied": False,
            "reason": "missing_required_columns",
            "missing_columns": missing,
        }

    lo = float(max(0.0, min(float(min_minutes), float(max_minutes))))
    hi = float(max(float(min_minutes), float(max_minutes)))
    if hi <= lo:
        return worlds_df, {
            "applied": False,
            "reason": "invalid_minutes_window",
            "min_minutes": lo,
            "max_minutes": hi,
        }
    boost = float(np.clip(float(tail_boost), 0.0, 0.40))
    if boost <= 0.0:
        return worlds_df, {"applied": False, "reason": "tail_boost_zero", "tail_boost": boost}

    out = worlds_df.copy()
    minutes = pd.to_numeric(out["minutes"], errors="coerce").fillna(0.0).to_numpy(dtype=float, copy=False)
    bucket_mask = (minutes >= lo) & (minutes <= hi)
    if target_game_ids:
        game_ids = pd.to_numeric(out["game_id"], errors="coerce").astype("Int64")
        bucket_mask &= game_ids.isin(sorted(target_game_ids)).to_numpy(dtype=bool)
    if not bool(np.any(bucket_mask)):
        return out, {
            "applied": False,
            "reason": "no_rows_in_minutes_bucket",
            "min_minutes": lo,
            "max_minutes": hi,
            "target_game_count": int(len(target_game_ids or set())),
        }

    key_cols = ["game_id", "team_id", "player_id"]
    stat_cols = [col for col in ("pts", "reb", "ast", "stl", "blk") if col in out.columns]
    if not stat_cols:
        return out, {"applied": False, "reason": "missing_tail_stat_columns"}

    player_means = _group_mean_by_keys_without_pandas_groupby(
        out,
        key_cols=key_cols,
        value_cols=stat_cols,
        label="mid_minutes_tail/player_means",
    ).rename(columns={col: f"{col}_mean" for col in stat_cols})
    out = _left_overlay_from_source_by_keys(
        out,
        source_df=player_means,
        key_cols=key_cols,
        value_cols=[f"{col}_mean" for col in stat_cols],
        label="mid_minutes_tail/mean_overlay",
        copy_base=False,
    )

    center = 0.5 * (lo + hi)
    half_span = max((hi - lo) * 0.5, 1e-6)
    shape = np.clip(1.0 - np.abs(minutes - center) / half_span, 0.0, 1.0)
    row_scale = 1.0 + boost * shape

    for col in stat_cols:
        mean_col = f"{col}_mean"
        cap = float(_WORLD_BASE_STAT_CAPS.get(col, 1_000.0))
        x = pd.to_numeric(out[col], errors="coerce").fillna(0.0).to_numpy(dtype=float, copy=False)
        mu = (
            pd.to_numeric(out.get(mean_col, 0.0), errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=float, copy=False)
        )
        resid = x - mu
        resid_pos = np.maximum(resid, 0.0)
        resid_neg = np.minimum(resid, 0.0)
        x_new = np.clip(
            np.nan_to_num(mu + resid_neg + row_scale * resid_pos, nan=0.0, posinf=cap, neginf=0.0),
            0.0,
            cap,
        )
        out[col] = np.where(bucket_mask, x_new, x)
        out = out.drop(columns=[mean_col], errors="ignore")

    if {"oreb", "dreb", "reb"}.issubset(out.columns):
        oreb_cap = float(_WORLD_BASE_STAT_CAPS.get("oreb", 25.0))
        dreb_cap = float(_WORLD_BASE_STAT_CAPS.get("dreb", 30.0))
        oreb = pd.to_numeric(out["oreb"], errors="coerce").fillna(0.0).to_numpy(dtype=float, copy=False)
        dreb = pd.to_numeric(out["dreb"], errors="coerce").fillna(0.0).to_numpy(dtype=float, copy=False)
        reb = pd.to_numeric(out["reb"], errors="coerce").fillna(0.0).to_numpy(dtype=float, copy=False)
        split_sum = np.maximum(oreb + dreb, 1e-6)
        oreb_share = np.divide(oreb, split_sum)
        oreb_new = np.clip(reb * oreb_share, 0.0, oreb_cap)
        dreb_new = np.clip(reb * (1.0 - oreb_share), 0.0, dreb_cap)
        out["oreb"] = np.where(bucket_mask, oreb_new, oreb)
        out["dreb"] = np.where(bucket_mask, dreb_new, dreb)

    dk_cap = float(_WORLD_DERIVED_STAT_CAPS.get("dk_fpts", 150.0))
    dk = _recompute_dk_fpts(out).to_numpy(dtype=float, copy=False)
    out["dk_fpts"] = np.clip(np.nan_to_num(dk, nan=0.0, posinf=dk_cap, neginf=0.0), 0.0, dk_cap)
    report = {
        "applied": True,
        "min_minutes": lo,
        "max_minutes": hi,
        "tail_boost": boost,
        "target_game_count": int(len(target_game_ids or set())),
        "affected_rows": int(np.count_nonzero(bucket_mask)),
        "affected_players": int(
            out.loc[bucket_mask, key_cols].drop_duplicates().shape[0]
            if bool(np.any(bucket_mask))
            else 0
        ),
        "scale_mean": float(np.mean(row_scale[bucket_mask])) if bool(np.any(bucket_mask)) else 1.0,
        "scale_p90": float(np.quantile(row_scale[bucket_mask], 0.90)) if bool(np.any(bucket_mask)) else 1.0,
    }
    return out, report


def _resolve_previous_run_file(*, dataset_dir: Path, filename: str) -> Path | None:
    run_id = control_plane.read_promoted_run_id(dataset_dir)
    if not run_id:
        return None
    candidate = dataset_dir / f"run={run_id}" / filename
    return candidate if candidate.exists() else None


def _build_feature_input_checklist(
    *,
    game_date: str,
    run_as_of_ts: str,
    data_root: Path,
    allow_priors_fallback: bool,
    allow_rotowire_props_fallback: bool = False,
    require_action_props: bool = True,
) -> dict[str, Any]:
    day = pd.Timestamp(game_date).normalize()
    run_ts = pd.to_datetime(run_as_of_ts, utc=True, errors="coerce")
    if pd.isna(run_ts):
        raise RuntimeError(f"invalid run_as_of_ts: {run_as_of_ts}")

    season, month = _resolve_season_month(game_date)
    schedule_path = (
        data_root
        / "silver"
        / "schedule"
        / f"season={season}"
        / f"month={month:02d}"
        / "schedule.parquet"
    )
    roster_path = (
        data_root
        / "silver"
        / "roster_nightly"
        / f"season={season}"
        / f"month={month:02d}"
        / "roster.parquet"
    )
    odds_path = (
        data_root
        / "silver"
        / "odds_snapshot"
        / f"season={season}"
        / f"month={month:02d}"
        / "odds_snapshot.parquet"
    )
    injuries_silver_path = (
        data_root
        / "silver"
        / "injuries_snapshot"
        / f"season={season}"
        / f"month={month:02d}"
        / "injuries_snapshot.parquet"
    )
    rotowire_path = (
        data_root
        / "silver"
        / "rotowire_lineups"
        / f"date={day.date()}"
        / "lineups.parquet"
    )
    labels_gold_root = data_root / "gold" / "labels_minutes_v1" / f"season={season}"
    labels_legacy_path = (
        data_root / "labels" / f"season={season}" / "boxscore_labels.parquet"
    )
    priors_team_root = (
        data_root
        / "silver"
        / "rotation_priors_v1"
        / "team_game_priors"
        / f"season={season}"
    )
    priors_player_root = (
        data_root
        / "silver"
        / "rotation_priors_v1"
        / "player_game_priors"
        / f"season={season}"
    )

    checks: list[dict[str, Any]] = []

    schedule_df = _read_parquet_if_exists(schedule_path)
    schedule_days = (
        pd.to_datetime(schedule_df.get("game_date"), errors="coerce").dt.normalize()
        if not schedule_df.empty
        else pd.Series(dtype="datetime64[ns]")
    )
    slate_df = (
        schedule_df.loc[schedule_days == day].copy()
        if not schedule_df.empty
        else pd.DataFrame()
    )
    slate_game_ids = (
        pd.to_numeric(slate_df.get("game_id"), errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
        if not slate_df.empty
        else []
    )
    expected_props_teams: set[str] = set()
    if not slate_df.empty:
        for team_col in ("home_team_tricode", "away_team_tricode"):
            if team_col not in slate_df.columns:
                continue
            vals = (
                slate_df[team_col].dropna().astype(str).str.strip().str.upper().tolist()
            )
            expected_props_teams.update(
                _normalize_props_team_abbr(v) for v in vals if str(v).strip()
            )
    checks.append(
        {
            "name": "schedule_slate_rows",
            "required": True,
            "ok": bool(not slate_df.empty and len(slate_game_ids) > 0),
            "details": {
                "path": str(schedule_path),
                "rows_total": int(len(schedule_df)),
                "rows_slate": int(len(slate_df)),
                "games_slate": int(len(slate_game_ids)),
            },
        }
    )

    def _snapshot_check(
        name: str, path: Path, *, required: bool = True
    ) -> pd.DataFrame:
        df = _read_parquet_if_exists(path)
        slate_rows = _filter_slate_rows(df, slate_game_ids)
        latest = _latest_ts(slate_rows)
        age_minutes = None
        if latest is not None:
            age_minutes = float((run_ts - latest).total_seconds() / 60.0)
        checks.append(
            {
                "name": name,
                "required": required,
                "ok": bool(not slate_rows.empty),
                "details": {
                    "path": str(path),
                    "rows_total": int(len(df)),
                    "rows_slate": int(len(slate_rows)),
                    "latest_as_of_ts": None if latest is None else latest.isoformat(),
                    "age_minutes": age_minutes,
                },
            }
        )
        return slate_rows

    roster_slate = _snapshot_check(
        "roster_snapshot_slate_rows", roster_path, required=True
    )
    odds_slate = _snapshot_check("odds_snapshot_slate_rows", odds_path, required=True)
    injuries_silver_slate = _snapshot_check(
        "injuries_snapshot_silver_slate_rows", injuries_silver_path, required=False
    )

    # Bronze injuries are preferred by build_minutes_live; verify at least one injuries source has slate rows.
    bronze_frames: list[pd.DataFrame] = []
    for offset in (-1, 0, 1):
        day_i = (day + pd.Timedelta(days=offset)).date()
        frame = bronze_storage.read_bronze_day(
            "injuries_raw",
            data_root,
            season,
            day_i,
            include_runs=False,
            prefer_history=True,
        )
        if not frame.empty:
            bronze_frames.append(frame)
    injuries_bronze = (
        pd.concat(bronze_frames, ignore_index=True) if bronze_frames else pd.DataFrame()
    )
    injuries_bronze_slate = _filter_slate_rows(injuries_bronze, slate_game_ids)
    injury_rows_ok = bool(
        not injuries_bronze_slate.empty or not injuries_silver_slate.empty
    )
    checks.append(
        {
            "name": "injuries_any_source_slate_rows",
            "required": True,
            "ok": injury_rows_ok,
            "details": {
                "bronze_rows_window": int(len(injuries_bronze)),
                "bronze_rows_slate": int(len(injuries_bronze_slate)),
                "silver_rows_slate": int(len(injuries_silver_slate)),
            },
        }
    )

    rotowire_df = _read_parquet_if_exists(rotowire_path)
    rotowire_slate = _filter_slate_rows(rotowire_df, slate_game_ids)
    checks.append(
        {
            "name": "rotowire_lineups_file",
            "required": False,
            "ok": bool(not rotowire_df.empty),
            "details": {
                "path": str(rotowire_path),
                "rows": int(len(rotowire_df)),
            },
        }
    )

    gold_exists = labels_gold_root.exists() and any(labels_gold_root.rglob("*.parquet"))
    legacy_exists = labels_legacy_path.exists()
    checks.append(
        {
            "name": "labels_source_available",
            "required": True,
            "ok": bool(gold_exists or legacy_exists),
            "details": {
                "gold_root": str(labels_gold_root),
                "gold_exists": bool(gold_exists),
                "legacy_path": str(labels_legacy_path),
                "legacy_exists": bool(legacy_exists),
            },
        }
    )

    rotowire_props_root = data_root / "bronze" / "props"
    action_props_day = day.date().isoformat()
    action_props_next_day = (day + pd.Timedelta(days=1)).date().isoformat()
    rotowire_raw_files = sorted(
        (rotowire_props_root / f"game_date={action_props_day}").glob("*.parquet")
    ) + sorted(
        (rotowire_props_root / f"game_date={action_props_next_day}").glob("*.parquet")
    )
    rotowire_props_summary = {
        "parsed_rows": 0,
        "latest_action_props_as_of_ts": None,
        "teams": [],
        "team_latest_as_of_ts": {},
        "parse_error": None,
    }
    if rotowire_props_root.exists():
        rotowire_props_summary = _probe_rotowire_props_snapshot_summary(
            rotowire_props_root=rotowire_props_root,
            game_date=day,
            data_root=data_root,
            run_as_of_ts=run_ts,
        )
    rotowire_parse_error = rotowire_props_summary.get("parse_error")

    checks.append(
        {
            "name": "rotowire_props_raw_files",
            "required": False,
            "ok": bool(len(rotowire_raw_files) > 0),
            "details": {
                "root": str(rotowire_props_root),
                "day_partition": str(
                    rotowire_props_root / f"game_date={action_props_day}"
                ),
                "next_day_partition": str(
                    rotowire_props_root / f"game_date={action_props_next_day}"
                ),
                "raw_file_count": int(len(rotowire_raw_files)),
            },
        }
    )
    latest_rotowire_props_ts = pd.to_datetime(
        rotowire_props_summary.get("latest_action_props_as_of_ts"),
        utc=True,
        errors="coerce",
    )
    checks.append(
        {
            "name": "rotowire_props_parsed_snapshots",
            "required": False,
            "ok": bool(
                int(rotowire_props_summary.get("parsed_rows", 0)) > 0
                and rotowire_parse_error is None
            ),
            "details": {
                "parsed_rows": int(rotowire_props_summary.get("parsed_rows", 0)),
                "latest_action_props_as_of_ts": None
                if pd.isna(latest_rotowire_props_ts)
                else latest_rotowire_props_ts.isoformat(),
                "parse_error": rotowire_parse_error,
            },
        }
    )
    rotowire_props_teams = {
        _normalize_props_team_abbr(team)
        for team in rotowire_props_summary.get("teams", [])
        if str(team).strip()
    }
    rotowire_props_team_latest = pd.DataFrame(
        {
            "team_tricode": list(
                (rotowire_props_summary.get("team_latest_as_of_ts") or {}).keys()
            ),
            "action_props_as_of_ts": list(
                (rotowire_props_summary.get("team_latest_as_of_ts") or {}).values()
            ),
        }
    )
    rotowire_props_team_overlap = rotowire_props_teams.intersection(
        expected_props_teams
    )
    rotowire_overlap_ok = bool(
        (not expected_props_teams) or rotowire_props_team_overlap
    )
    checks.append(
        {
            "name": "rotowire_props_team_overlap",
            "required": False,
            "ok": bool(rotowire_overlap_ok),
            "details": {
                "expected_slate_team_count": int(len(expected_props_teams)),
                "snapshot_team_count": int(len(rotowire_props_teams)),
                "overlap_team_count": int(len(rotowire_props_team_overlap)),
                "expected_slate_teams": sorted(expected_props_teams),
                "overlap_teams": sorted(rotowire_props_team_overlap),
            },
        }
    )
    rotowire_ok = bool(
        int(rotowire_props_summary.get("parsed_rows", 0)) > 0
        and rotowire_parse_error is None
        and rotowire_overlap_ok
    )
    policy_ok = (not require_action_props) or rotowire_ok
    checks.append(
        {
            "name": "props_source_policy_satisfied",
            "required": True,
            "ok": bool(policy_ok),
            "details": {
                "require_action_props": bool(require_action_props),
                "allow_rotowire_props_fallback": bool(allow_rotowire_props_fallback),
                "live_props_source": "rotowire",
                "rotowire_ok": bool(rotowire_ok),
                "selected_source": "rotowire" if rotowire_ok else "none",
            },
        }
    )

    team_partitions = (
        list(priors_team_root.glob("game_id=*.parquet"))
        if priors_team_root.exists()
        else []
    )
    player_partitions = (
        list(priors_player_root.glob("game_id=*.parquet"))
        if priors_player_root.exists()
        else []
    )
    checks.append(
        {
            "name": "rotation_priors_roots_nonempty",
            "required": True,
            "ok": bool(team_partitions and player_partitions),
            "details": {
                "team_root": str(priors_team_root),
                "player_root": str(priors_player_root),
                "team_partition_count": int(len(team_partitions)),
                "player_partition_count": int(len(player_partitions)),
            },
        }
    )

    checks.append(
        {
            "name": "rotation_priors_mode_explicit",
            "required": True,
            "ok": True,
            "details": {
                "allow_priors_fallback": bool(allow_priors_fallback),
                "mode": "game_id_partitions_or_latest_by_entity_fallback",
                "explanation": (
                    "Live slates commonly have no same-day game_id priors partitions pre-tip. "
                    "Fallback uses latest completed-game priors by team/player (not future info)."
                ),
            },
        }
    )

    missing_team: list[str] = []
    missing_player: list[str] = []
    for gid in slate_game_ids:
        gid_norm = str(int(gid)).zfill(10)
        if not (priors_team_root / f"game_id={gid_norm}.parquet").exists():
            missing_team.append(gid_norm)
        if not (priors_player_root / f"game_id={gid_norm}.parquet").exists():
            missing_player.append(gid_norm)
    all_gameid_missing = (
        bool(slate_game_ids)
        and len(missing_team) == len(slate_game_ids)
        and len(missing_player) == len(slate_game_ids)
    )
    checks.append(
        {
            "name": "rotation_priors_gameid_partition_coverage",
            "required": False,
            "ok": bool(not all_gameid_missing),
            "details": {
                "slate_games": int(len(slate_game_ids)),
                "present_team_partitions": int(len(slate_game_ids) - len(missing_team)),
                "present_player_partitions": int(
                    len(slate_game_ids) - len(missing_player)
                ),
                "missing_team_partitions": int(len(missing_team)),
                "missing_player_partitions": int(len(missing_player)),
            },
        }
    )
    checks.append(
        {
            "name": "rotation_priors_policy_allows_current_coverage",
            "required": True,
            "ok": bool((not all_gameid_missing) or allow_priors_fallback),
            "details": {
                "allow_priors_fallback": bool(allow_priors_fallback),
                "slate_games": int(len(slate_game_ids)),
                "missing_team_partitions": int(len(missing_team)),
                "missing_player_partitions": int(len(missing_player)),
                "all_gameid_partitions_missing": bool(all_gameid_missing),
                "note": (
                    "If all game_id partitions are missing and fallback is disabled, this fails closed "
                    "to avoid ambiguous priors behavior."
                ),
            },
        }
    )

    selected_props_source = "rotowire" if rotowire_ok else "none"
    manual_override_summary = manual_override_report(
        date.fromisoformat(game_date),
        data_root=data_root,
        as_of_ts=run_ts,
    )
    schedule_tip_by_game = _latest_ts_by_game(slate_df, time_col="tip_ts")
    odds_latest_by_game = _latest_ts_by_game(odds_slate, time_col="as_of_ts")
    roster_latest_by_game = _latest_ts_by_game(roster_slate, time_col="as_of_ts")
    roster_digest_by_game = _content_digest_by_game(
        roster_slate,
        slate_game_ids,
        exclude_columns={"as_of_ts", "game_date", "created_at", "updated_at"},
    )
    odds_digest_by_game = _content_digest_by_game(
        odds_slate,
        slate_game_ids,
        exclude_columns={"as_of_ts", "created_at", "updated_at", "snapshot_ts"},
    )
    injuries_bronze_latest_by_game = _latest_ts_by_game(
        injuries_bronze_slate, time_col="as_of_ts"
    )
    injuries_bronze_digest_by_game = _content_digest_by_game(
        injuries_bronze_slate,
        slate_game_ids,
        exclude_columns={"as_of_ts", "ingested_ts", "created_at", "updated_at"},
    )
    injuries_silver_latest_by_game = _latest_ts_by_game(
        injuries_silver_slate, time_col="as_of_ts"
    )
    injuries_silver_digest_by_game = _content_digest_by_game(
        injuries_silver_slate,
        slate_game_ids,
        exclude_columns={"as_of_ts", "ingested_ts", "created_at", "updated_at"},
    )
    rotowire_latest_by_game = (
        _latest_ts_by_game(rotowire_slate, time_col="ingested_ts")
        if not rotowire_slate.empty
        else _latest_ts_by_game_from_teams(
            slate_df, rotowire_df, time_col="ingested_ts"
        )
    )
    rotowire_digest_by_game = (
        _content_digest_by_game(
            rotowire_slate,
            slate_game_ids,
            exclude_columns={
                "ingested_ts",
                "lineup_timestamp",
                "created_at",
                "updated_at",
            },
        )
        if not rotowire_slate.empty
        else _content_digest_by_game_from_teams(
            slate_df,
            rotowire_df,
            exclude_columns={
                "ingested_ts",
                "lineup_timestamp",
                "created_at",
                "updated_at",
            },
        )
    )
    rotowire_props_latest_by_game = _latest_ts_by_game_from_teams(
        slate_df,
        rotowire_props_team_latest,
        time_col="action_props_as_of_ts",
    )
    rotowire_props_team_player_digest = {
        _normalize_props_team_abbr(team): str(digest)
        for team, digest in (
            dict(rotowire_props_summary.get("team_player_digest", {})).items()
        )
        if str(team).strip() and str(digest).strip()
    }
    rotowire_props_team_player_count = {
        _normalize_props_team_abbr(team): int(num)
        for team, count in (
            dict(rotowire_props_summary.get("team_player_count", {})).items()
        )
        if str(team).strip()
        and pd.notna(num := pd.to_numeric(count, errors="coerce"))
    }
    slate_teams_by_game: dict[int, list[str]] = {}
    for row in slate_df.itertuples(index=False):
        gid_num = pd.to_numeric(getattr(row, "game_id", None), errors="coerce")
        if pd.isna(gid_num):
            continue
        teams = []
        for attr in ("home_team_tricode", "away_team_tricode"):
            team = _normalize_props_team_abbr(getattr(row, attr, None))
            if team:
                teams.append(team)
        if teams:
            slate_teams_by_game[int(gid_num)] = sorted(set(teams))
    per_game_freshness: dict[str, dict[str, Any]] = {}
    for gid in slate_game_ids:
        tip_ts = schedule_tip_by_game.get(int(gid))
        minutes_to_tip = None
        is_live_game = False
        if tip_ts is not None:
            minutes_to_tip = float((tip_ts - run_ts).total_seconds() / 60.0)
            is_live_game = bool(minutes_to_tip > 0.0)
        injuries_bronze_ts = injuries_bronze_latest_by_game.get(int(gid))
        injuries_silver_ts = injuries_silver_latest_by_game.get(int(gid))
        if injuries_bronze_ts is not None:
            injuries_source_used = "bronze"
            injuries_latest = injuries_bronze_ts
            injuries_digest = injuries_bronze_digest_by_game.get(int(gid))
        elif injuries_silver_ts is not None:
            injuries_source_used = "silver"
            injuries_latest = injuries_silver_ts
            injuries_digest = injuries_silver_digest_by_game.get(int(gid))
        else:
            injuries_source_used = "none"
            injuries_latest = None
            injuries_digest = None
        rotowire_props_ts = rotowire_props_latest_by_game.get(int(gid))
        props_latest = rotowire_props_ts
        props_player_payload: list[dict[str, Any]] = []
        props_player_count = 0
        for team in slate_teams_by_game.get(int(gid), []):
            team_digest = rotowire_props_team_player_digest.get(team)
            if not team_digest:
                continue
            team_count = int(rotowire_props_team_player_count.get(team, 0))
            props_player_payload.append(
                {
                    "team_tricode": team,
                    "player_set_digest": team_digest,
                    "player_set_count": team_count,
                }
            )
            props_player_count += team_count
        props_player_set_digest = (
            _stable_digest(props_player_payload) if props_player_payload else None
        )
        per_game_freshness[str(int(gid))] = {
            "game_id": int(gid),
            "tip_ts": _ts_to_iso(tip_ts),
            "minutes_to_tip": minutes_to_tip,
            "is_live_game": bool(is_live_game),
            "sources": {
                "roster": {
                    "source_used": "silver",
                    "latest_as_of_ts": _ts_to_iso(roster_latest_by_game.get(int(gid))),
                    "age_minutes": _age_minutes(
                        run_ts, roster_latest_by_game.get(int(gid))
                    ),
                    "content_digest": roster_digest_by_game.get(int(gid)),
                },
                "odds": {
                    "source_used": "silver",
                    "latest_as_of_ts": _ts_to_iso(odds_latest_by_game.get(int(gid))),
                    "age_minutes": _age_minutes(
                        run_ts, odds_latest_by_game.get(int(gid))
                    ),
                    "content_digest": odds_digest_by_game.get(int(gid)),
                },
                "injuries": {
                    "source_used": injuries_source_used,
                    "latest_as_of_ts": _ts_to_iso(injuries_latest),
                    "age_minutes": _age_minutes(run_ts, injuries_latest),
                    "content_digest": injuries_digest,
                    "bronze_latest_as_of_ts": _ts_to_iso(injuries_bronze_ts),
                    "bronze_content_digest": injuries_bronze_digest_by_game.get(
                        int(gid)
                    ),
                    "silver_latest_as_of_ts": _ts_to_iso(injuries_silver_ts),
                    "silver_content_digest": injuries_silver_digest_by_game.get(
                        int(gid)
                    ),
                },
                "lineups": {
                    "source_used": "rotowire",
                    "latest_as_of_ts": _ts_to_iso(
                        rotowire_latest_by_game.get(int(gid))
                    ),
                    "age_minutes": _age_minutes(
                        run_ts, rotowire_latest_by_game.get(int(gid))
                    ),
                    "content_digest": rotowire_digest_by_game.get(int(gid)),
                },
                "props": {
                    "source_used": selected_props_source,
                    "latest_as_of_ts": _ts_to_iso(props_latest),
                    "age_minutes": _age_minutes(run_ts, props_latest),
                    "rotowire_latest_as_of_ts": _ts_to_iso(rotowire_props_ts),
                    "player_set_digest": props_player_set_digest,
                    "player_set_count": int(props_player_count),
                },
                "manual_overrides": dict(
                    manual_override_summary.get("per_game", {}).get(str(int(gid)), {})
                )
                or {
                    "source_used": "none",
                    "latest_as_of_ts": None,
                    "content_digest": None,
                    "active_override_count": 0,
                },
            },
        }
    report_window = _report_window_status(
        run_ts=run_ts, per_game_freshness=per_game_freshness
    )
    lock_window = _lock_window_gate_status(per_game_freshness=per_game_freshness)
    source_freshness = {
        "summary": {
            "run_as_of_ts": str(run_ts.isoformat()),
            "slate_game_count": int(len(slate_game_ids)),
            "live_game_count": int(
                sum(
                    1
                    for game in per_game_freshness.values()
                    if bool(game.get("is_live_game"))
                )
            ),
            "selected_props_source": selected_props_source,
            "manual_override_count": int(
                manual_override_summary.get("active_override_count", 0)
            ),
            "manual_override_games": list(
                manual_override_summary.get("affected_game_ids", [])
            ),
            "manual_override_digest": manual_override_summary.get("override_digest"),
        },
        "per_game": per_game_freshness,
    }

    failed_required = [
        c["name"] for c in checks if bool(c.get("required")) and not bool(c.get("ok"))
    ]
    return {
        "builder_input_checklist_version": 1,
        "game_date": game_date,
        "season": int(season),
        "month": int(month),
        "run_as_of_ts": str(run_ts.isoformat()),
        "checks": checks,
        "source_freshness": source_freshness,
        "freshness_gates": {
            "lock_window": lock_window,
            "report_window": report_window,
        },
        "failed_required_checks": failed_required,
    }


@task(name="scrape-core-inputs", retries=1, retry_delay_seconds=30)
def scrape_core_inputs_task(
    *,
    game_date: str,
    data_root: Path,
    placeholder_mode: bool,
    require_action_props: bool,
    allow_rotowire_props_fallback: bool,
    replay_mode: bool = False,
) -> Path:
    marker = (
        data_root
        / "bronze"
        / "v3_core_inputs"
        / f"date={game_date}"
        / "core_inputs_ready.json"
    )
    marker.parent.mkdir(parents=True, exist_ok=True)

    if placeholder_mode:
        payload = {
            "game_date": game_date,
            "placeholder_mode": True,
            "completed_at": _utc_now_iso(),
        }
        marker.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        return marker

    if replay_mode:
        payload = {
            "game_date": game_date,
            "placeholder_mode": False,
            "replay_mode": True,
            "completed_at": _utc_now_iso(),
            "props_required": bool(require_action_props),
            "live_props_source": "rotowire",
            "allow_rotowire_props_fallback": bool(allow_rotowire_props_fallback),
            "note": "scrape step skipped in replay_mode; existing snapshots are used",
        }
        marker.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        return marker

    season, month = _resolve_season_month(game_date)
    schedule_path = (
        data_root
        / "silver"
        / "schedule"
        / f"season={season}"
        / f"month={month:02d}"
        / "schedule.parquet"
    )
    args = [
        "--start",
        game_date,
        "--end",
        game_date,
        "--season",
        str(season),
        "--month",
        str(month),
        "--data-root",
        str(data_root),
    ]
    if schedule_path.exists():
        args.extend(["--schedule", str(schedule_path)])
    _run_python_module(
        "projections.cli.live_pipeline",
        args,
        data_root=data_root,
        timeout_s=900,
    )

    props_status: dict[str, str] = {
        "scrape_props_cli": "not_run",
    }
    try:
        _run_python_module(
            "projections.cli.scrape_props",
            ["scrape", "--date", game_date],
            data_root=data_root,
            timeout_s=300,
        )
        props_status["scrape_props_cli"] = "ok"
    except Exception as exc:  # noqa: BLE001
        props_status["scrape_props_cli"] = f"failed: {exc}"
        if require_action_props:
            raise RuntimeError(
                "live props scrape failed while require_action_props=True: " f"{exc}"
            ) from exc
    props_dir = data_root / "bronze" / "props"
    day = pd.Timestamp(game_date).normalize()
    raw_props_files = sorted(
        (props_dir / f"game_date={day.date().isoformat()}").glob("*.parquet")
    )
    _run_python_module(
        "scripts.dk.run_daily_salaries",
        ["--game-date", game_date],
        data_root=data_root,
        timeout_s=600,
    )
    payload = {
        "game_date": game_date,
        "placeholder_mode": False,
        "completed_at": _utc_now_iso(),
        "props_required": bool(require_action_props),
        "live_props_source": "rotowire",
        "allow_rotowire_props_fallback": bool(allow_rotowire_props_fallback),
        "props_status": props_status,
        "rotowire_props_raw_file_count": int(len(raw_props_files)),
    }
    marker.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return marker


@task(name="score-ownership", retries=2, retry_delay_seconds=120)
def score_ownership_task(
    *,
    game_date: str,
    run_id: str,
    data_root: Path,
    placeholder_mode: bool,
    source: str,
    model_family: str | None,
    model_run: str | None,
    gtv2_features_path: str | None,
    fallback_source: str | None,
    fallback_model_family: str | None,
    fallback_model_run: str | None,
    fallback_gtv2_features_path: str | None,
) -> Path:
    out_dir = data_root / "silver" / "ownership_predictions" / game_date / f"run={run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if placeholder_mode:
        features_path = (
            data_root
            / "live"
            / FEATURES_ROOT
            / game_date
            / f"run={run_id}"
            / "features.parquet"
        )
        if features_path.exists():
            features_df = pd.read_parquet(features_path)
            placeholder_df = features_df.copy()
            if "player_id" not in placeholder_df.columns:
                placeholder_df["player_id"] = np.arange(len(placeholder_df)) + 1
            if "player_name" not in placeholder_df.columns:
                placeholder_df["player_name"] = placeholder_df["player_id"].map(
                    lambda value: f"Player {value}"
                )
            keep_cols = [
                column
                for column in ["player_id", "player_name", "team_id", "game_id"]
                if column in placeholder_df.columns
            ]
            placeholder_df = (
                placeholder_df.loc[:, keep_cols]
                .drop_duplicates(subset=["player_id"], keep="last")
                .reset_index(drop=True)
            )
        else:
            placeholder_df = pd.DataFrame(
                {
                    "player_id": list(range(1, 21)),
                    "player_name": [f"Player {idx}" for idx in range(1, 21)],
                }
            )
        placeholder_df["pred_own_pct"] = 0.05
        placeholder_df["source"] = source
        placeholder_df["model_family"] = model_family or "ownership_v1"
        placeholder_df["model_run"] = (
            model_run
            if model_run
            else f"{source}_{model_family or 'ownership_v1'}_placeholder"
        )
        _atomic_write_validated_parquet(
            placeholder_df,
            out_dir / "123.parquet",
            required_cols=("player_id",),
        )
        (out_dir / "slates.json").write_text(
            json.dumps(
                {
                    "123": {
                        "player_count": int(len(placeholder_df)),
                        "teams": [],
                        "first_game_time": None,
                        "is_locked": False,
                        "source": source,
                    }
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return out_dir

    logger = get_run_logger()

    def _module_and_args(
        *,
        source: str,
        model_family: str | None,
        model_run: str | None,
        gtv2_features_path: str | None,
    ) -> tuple[str, list[str]]:
        if source == "linestar":
            return (
                "projections.cli.score_ownership_linestar",
                [
                    "--date",
                    game_date,
                    "--run-id",
                    run_id,
                    "--data-root",
                    str(data_root),
                ],
            )
        if source != "internal":
            raise RuntimeError(f"Unsupported ownership source: {source}")
        family = str(model_family or "ownership_v1")
        args = [
            "--date",
            game_date,
            "--run-id",
            run_id,
            "--data-root",
            str(data_root),
            "--model-family",
            family,
        ]
        if model_run:
            args.extend(["--model-run", str(model_run)])
        if gtv2_features_path:
            args.extend(["--gtv2-features-path", str(gtv2_features_path)])
        return ("projections.cli.score_ownership_live", args)

    attempts: list[dict[str, str | None]] = [
        {
            "source": source,
            "model_family": model_family,
            "model_run": model_run,
            "gtv2_features_path": gtv2_features_path,
        }
    ]
    if fallback_source:
        attempts.append(
            {
                "source": fallback_source,
                "model_family": fallback_model_family,
                "model_run": fallback_model_run,
                "gtv2_features_path": fallback_gtv2_features_path,
            }
        )

    for idx, attempt in enumerate(attempts):
        module, args = _module_and_args(
            source=str(attempt["source"] or ""),
            model_family=(
                str(attempt["model_family"])
                if attempt.get("model_family")
                else None
            ),
            model_run=str(attempt["model_run"]) if attempt.get("model_run") else None,
            gtv2_features_path=(
                str(attempt["gtv2_features_path"])
                if attempt.get("gtv2_features_path")
                else None
            ),
        )
        try:
            logger.info(
                "[ownership] scoring source=%s model_family=%s model_run=%s attempt=%s/%s",
                attempt.get("source"),
                attempt.get("model_family"),
                attempt.get("model_run"),
                idx + 1,
                len(attempts),
            )
            _run_python_module(
                module,
                args,
                data_root=data_root,
                timeout_s=1200,
            )
            break
        except Exception as exc:
            if idx + 1 >= len(attempts):
                raise
            logger.warning(
                "[ownership] primary scoring failed (%s). Falling back to source=%s model_family=%s model_run=%s",
                exc,
                attempts[idx + 1].get("source"),
                attempts[idx + 1].get("model_family"),
                attempts[idx + 1].get("model_run"),
            )
    return out_dir


@task(name="freeze-run-inputs", retries=0)
def freeze_run_inputs_task(
    *,
    game_date: str,
    run_id: str,
    as_of_ts: str,
    bundle_dir: Path,
    data_root: Path,
    ownership_selector_path: Path,
    gtv2_inference_current_path: Path | None = None,
    gtv2_inference_current_hash: str | None = None,
    source_freshness: dict[str, Any] | None = None,
    freshness_gates: dict[str, Any] | None = None,
    bounded_wait: dict[str, Any] | None = None,
    input_change_set: dict[str, Any] | None = None,
) -> Path:
    minutes_selector_path = model_selectors.active_minutes_selector_path(
        data_root=data_root,
        project_root=PROJECT_ROOT,
    )
    rates_selector_path = model_selectors.active_rates_selector_path(
        data_root=data_root,
        project_root=PROJECT_ROOT,
    )
    manifest_path = control_plane.write_run_manifest_start(
        data_root=data_root,
        game_date=game_date,
        run_id=run_id,
        as_of_ts=as_of_ts,
        sim_profile="game_transformer_v2",
        entrypoint="prefect-v3",
        minutes_current_run_path=minutes_selector_path,
        rates_current_run_path=rates_selector_path,
        ownership_current_run_path=ownership_selector_path,
        slate={},
    )

    bundle_hash = _bundle_artifact_hash(bundle_dir)
    control_plane.atomic_update_json(
        manifest_path,
        {
            "gtv2_inference_current_path": (
                str(gtv2_inference_current_path)
                if gtv2_inference_current_path is not None
                else ""
            ),
            "source_freshness": source_freshness or {},
            "freshness_gates": freshness_gates or {},
            "bounded_wait": bounded_wait or {},
            "input_change_set": input_change_set or {},
            "v3": {
                "bundle_dir": str(bundle_dir),
                "bundle_hash": bundle_hash,
                "gtv2_inference_current_hash": str(
                    gtv2_inference_current_hash or ""
                ),
                "parity_manifest_path": str(resolve_parity_manifest_path(bundle_dir)),
            },
        },
    )
    return manifest_path


@task(name="build-features-gtv2-live", retries=0)
def build_features_gtv2_live_task(
    *,
    game_date: str,
    run_id: str,
    run_as_of_ts: str,
    data_root: Path,
    bundle_dir: Path,
    manifest_path: Path,
    placeholder_mode: bool,
    require_action_props: bool,
    allow_rotowire_props_fallback: bool,
    target_game_ids: list[int] | None = None,
) -> Path:
    run_dir = data_root / "live" / FEATURES_ROOT / game_date / f"run={run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "features.parquet"
    runtime_manifest_path = run_dir / "feature_runtime_manifest.json"
    input_checklist_path = run_dir / "feature_input_checklist.json"

    manifest_payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    v3_meta = dict(manifest_payload.get("v3", {}))
    if placeholder_mode:
        features_df = _placeholder_feature_frame(
            game_date=game_date, as_of_ts=run_as_of_ts
        )
        features_df = _filter_to_target_games(features_df, target_game_ids)
        _atomic_write_validated_parquet(
            features_df,
            out_path,
            required_cols=("game_id", "team_id", "player_id"),
        )
        # Keep publish-stage contracts stable even in placeholder mode by also
        # writing the canonical minutes features artifact.
        base_minutes_run_dir = (
            data_root / "live" / "features_minutes_v1" / game_date / f"run={run_id}"
        )
        base_minutes_run_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_validated_parquet(
            features_df,
            base_minutes_run_dir / "features.parquet",
            required_cols=("game_id", "team_id", "player_id"),
        )

        transform_manifest = {
            "feature_builder": "placeholder_gtv2_live_v1",
            "scaling": "none",
            "encoding": "none",
        }
        integrity = {
            "git_sha": str(manifest_payload.get("git_sha")),
            "config_hash": str(v3_meta.get("bundle_hash")),
            "artifact_hash": str(v3_meta.get("bundle_hash")),
        }
        parity_path = _ensure_placeholder_bundle(
            bundle_dir=run_dir / "_placeholder_bundle",
            features_df=features_df,
            transform_manifest=transform_manifest,
            integrity=integrity,
        )
        diagnostics: dict[str, Any] = {
            "placeholder_mode": True,
            "rows": int(len(features_df)),
        }
    else:
        parity_path = resolve_parity_manifest_path(bundle_dir)
        if not parity_path.exists():
            raise RuntimeError(
                f"bundle parity manifest missing (fail-closed): {parity_path}. "
                "Create and ship parity_manifest.json with the promoted bundle."
            )
        parity_payload = load_parity_manifest(parity_path)
        expected_transform = dict(parity_payload.get("transform_manifest", {}))
        expected_priors = dict(expected_transform.get("priors", {}))
        expected_dnp = dict(expected_transform.get("dnp_history", {}))

        allow_priors_fallback = bool(expected_priors.get("allow_priors_fallback", True))
        dnp_mode = str(expected_dnp.get("mode", "bounded_lookback")).strip().lower()
        dnp_lookback_days: int | None
        if dnp_mode in {"full_prior_history", "full-history", "full"}:
            dnp_lookback_days = None
        else:
            raw_lookback = expected_dnp.get("lookback_days", 120)
            dnp_lookback_days = int(raw_lookback) if raw_lookback is not None else None

        allow_rotowire_fallback_cfg = bool(allow_rotowire_props_fallback)

        checklist = _build_feature_input_checklist(
            game_date=game_date,
            run_as_of_ts=run_as_of_ts,
            data_root=data_root,
            allow_priors_fallback=allow_priors_fallback,
            allow_rotowire_props_fallback=allow_rotowire_fallback_cfg,
            require_action_props=bool(require_action_props),
        )
        input_checklist_path.write_text(
            json.dumps(checklist, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        failed = checklist.get("failed_required_checks", [])
        if failed:
            raise RuntimeError(
                "live feature input checklist failed required checks: "
                f"{failed}. See {input_checklist_path}"
            )
        selected_props_source = _selected_props_source_from_checklist(checklist)
        props_source_report_path = run_dir / "props_source_report.json"
        props_source_report_path.write_text(
            json.dumps(
                {
                    "game_date": game_date,
                    "run_id": run_id,
                    "run_as_of_ts": run_as_of_ts,
                    "selected_source": selected_props_source,
                    "live_props_source": "rotowire",
                    "require_action_props": bool(require_action_props),
                    "allow_rotowire_props_fallback": bool(allow_rotowire_fallback_cfg),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        base_minutes_path = (
            data_root
            / "live"
            / "features_minutes_v1"
            / game_date
            / f"run={run_id}"
            / "features.parquet"
        )
        minutes_build_source = "fresh_build"
        minutes_build_fallback_path: str | None = None
        # Build canonical live minutes features first, then project to GTV2 model contract.
        run_as_of_ts_cli = _cli_compatible_ts(run_as_of_ts)
        try:
            _run_python_module(
                "projections.cli.build_minutes_live",
                [
                    "--date",
                    game_date,
                    "--run-id",
                    run_id,
                    "--run-as-of-ts",
                    run_as_of_ts_cli,
                    "--data-root",
                    str(data_root),
                    *(
                        ["--allow-rotowire-props-fallback"]
                        if allow_rotowire_fallback_cfg
                        else []
                    ),
                ],
                data_root=data_root,
                timeout_s=1200,
            )
        except RuntimeError as exc:
            error_text = str(exc)
            if "exit_code=-11" not in error_text:
                raise
            fallback_minutes_path = _find_latest_readable_minutes_features_path(
                data_root=data_root,
                game_date=game_date,
                exclude_run_id=run_id,
            )
            if fallback_minutes_path is None:
                raise
            logger = get_run_logger()
            logger.warning(
                "build_minutes_live crashed (exit_code=-11); using fallback minutes features: %s",
                fallback_minutes_path,
            )
            fallback_df = pd.read_parquet(fallback_minutes_path)
            fallback_df = _filter_to_target_games(fallback_df, target_game_ids)
            if fallback_df.empty:
                raise RuntimeError(
                    "fallback minutes features are empty after applying target_game_ids: "
                    f"{target_game_ids} (source={fallback_minutes_path})"
                ) from exc
            if "run_id" in fallback_df.columns:
                fallback_df["run_id"] = str(run_id)
            _atomic_write_validated_parquet(
                fallback_df,
                base_minutes_path,
                required_cols=("game_id", "team_id", "player_id"),
            )
            minutes_build_source = "fallback_previous_minutes_run"
            minutes_build_fallback_path = str(fallback_minutes_path)
        if not base_minutes_path.exists():
            raise RuntimeError(f"base minutes features not found: {base_minutes_path}")

        from projections.pipeline.gtv2_live_features import (
            build_gtv2_live_features,
            load_gtv2_feature_spec,
        )

        spec = load_gtv2_feature_spec(bundle_dir)
        base_df = pd.read_parquet(base_minutes_path)
        base_df = _filter_to_target_games(base_df, target_game_ids)
        if base_df.empty:
            raise RuntimeError(
                "base minutes features are empty after applying target_game_ids: "
                f"{target_game_ids}"
            )

        built = build_gtv2_live_features(
            minutes_features=base_df,
            spec=spec,
            data_root=data_root,
            game_date=game_date,
            allow_priors_fallback=allow_priors_fallback,
            dnp_lookback_days=dnp_lookback_days,
        )
        transform_manifest = dict(built.transform_manifest)
        if stable_json_sha256(transform_manifest) != stable_json_sha256(
            expected_transform
        ):
            raise RuntimeError(
                "observed transform manifest does not match bundle parity manifest "
                "(fail-closed transform parity gate)"
            )

        features_df = _coerce_frame_to_manifest_schema(built.features, parity_payload)
        _atomic_write_validated_parquet(
            features_df,
            out_path,
            required_cols=("game_id", "team_id", "player_id"),
        )

        integrity_src = dict(parity_payload.get("integrity", {}))
        integrity = {
            "git_sha": integrity_src.get("git_sha"),
            "config_hash": integrity_src.get("config_hash"),
            "artifact_hash": integrity_src.get("artifact_hash"),
        }
        diagnostics = dict(built.diagnostics)
        diagnostics["placeholder_mode"] = False
        diagnostics["base_minutes_features_path"] = str(base_minutes_path)
        diagnostics["feature_input_checklist_path"] = str(input_checklist_path)
        diagnostics["props_source_report_path"] = str(props_source_report_path)
        diagnostics["props_source_selected"] = selected_props_source
        diagnostics["dnp_history_mode"] = (
            "full_prior_history" if dnp_lookback_days is None else "bounded_lookback"
        )
        diagnostics["dnp_lookback_days"] = (
            None if dnp_lookback_days is None else int(dnp_lookback_days)
        )
        diagnostics["allow_rotowire_props_fallback"] = bool(allow_rotowire_fallback_cfg)
        diagnostics["target_game_ids"] = _normalize_game_ids(target_game_ids)
        diagnostics["minutes_build_source"] = str(minutes_build_source)
        diagnostics["minutes_build_fallback_path"] = minutes_build_fallback_path

    runtime_manifest_path.write_text(
        json.dumps(
            {
                "transform_manifest": transform_manifest,
                "integrity": integrity,
                "parity_manifest_path": str(parity_path),
                "diagnostics": diagnostics,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return out_path


def _find_latest_readable_minutes_features_path(
    *,
    data_root: Path,
    game_date: str,
    exclude_run_id: str | None = None,
) -> Path | None:
    """Find the newest readable minutes features parquet for a slate date."""
    features_date_dir = data_root / "live" / "features_minutes_v1" / str(game_date)
    if not features_date_dir.exists():
        return None

    for run_dir in sorted(features_date_dir.glob("run=*"), reverse=True):
        if not run_dir.is_dir():
            continue
        run_token = str(run_dir.name).split("run=", 1)[-1]
        if exclude_run_id and run_token == str(exclude_run_id):
            continue
        candidate = run_dir / "features.parquet"
        if not candidate.exists():
            continue
        try:
            _stream_validate_parquet(
                candidate,
                required_cols=("game_id", "team_id", "player_id"),
            )
            sample = pd.read_parquet(candidate, columns=["game_id", "team_id", "player_id"])
        except Exception:  # noqa: BLE001
            continue
        if sample.empty:
            continue
        return candidate
    return None


@task(name="v3-preflight", retries=0)
def preflight_gate_task(
    *,
    as_of_ts: str,
    manifest_path: Path,
    required_inputs: dict[str, Path],
    run_dirs: list[Path],
    features_path: Path,
    parity_manifest_path: Path,
    runtime_manifest_path: Path,
    input_max_age_minutes: float,
    bundle_config_path: Path | None = None,
) -> dict[str, Any]:
    runtime_payload = json.loads(runtime_manifest_path.read_text(encoding="utf-8"))
    manifest_payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    return run_preflight_gate(
        as_of_ts=as_of_ts,
        required_inputs=required_inputs,
        run_dirs=run_dirs,
        features_path=features_path,
        parity_manifest_path=parity_manifest_path,
        observed_transform_manifest=dict(runtime_payload.get("transform_manifest", {})),
        observed_integrity=dict(runtime_payload.get("integrity", {})),
        input_max_age_minutes=float(input_max_age_minutes),
        bundle_config_path=bundle_config_path,
        frozen_source_freshness=dict(manifest_payload.get("source_freshness", {})),
        frozen_freshness_gates=dict(manifest_payload.get("freshness_gates", {})),
    )


@task(name="score-gtv2-live", retries=0)
def score_gtv2_live_task(
    *,
    game_date: str,
    run_id: str,
    features_path: Path,
    bundle_dir: Path,
    data_root: Path,
    placeholder_mode: bool,
    inference_backend: str = "local",
    triton_endpoint: str | None = None,
    triton_model_name: str = "gtv2_scorer",
    triton_model_version: str | None = None,
    triton_timeout_seconds: float = 90.0,
    triton_healthcheck_timeout_seconds: float = 3.0,
    gtv2_device: str | None = None,
    random_seed: int = 42,
) -> Path:
    run_dir = (
        data_root
        / "artifacts"
        / SCORES_ROOT
        / f"game_date={game_date}"
        / f"run={run_id}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "scores.parquet"
    summary_path = run_dir / "score_summary.json"

    if placeholder_mode:
        features = pd.read_parquet(features_path)
        scores = features[["game_date", "game_id", "team_id", "player_id"]].copy()
        scores["minutes_mean"] = pd.to_numeric(
            features["minutes_prior"], errors="coerce"
        ).fillna(0.0)
        scores["play_prob"] = 0.95
        scores["dk_rate"] = (
            pd.to_numeric(features["usage_prior"], errors="coerce").fillna(0.0) * 100.0
        )
        _atomic_write_validated_parquet(
            scores,
            out_path,
            required_cols=("game_date", "game_id", "team_id", "player_id"),
        )
        summary_path.write_text(
            json.dumps(
                {
                    "placeholder_mode": True,
                    "rows": int(len(scores)),
                    "created_at": _utc_now_iso(),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return out_path

    backend = str(inference_backend).strip().lower()
    if backend not in {"local", "triton"}:
        raise RuntimeError(f"unsupported inference backend for score task: {backend}")

    features_df = pd.read_parquet(features_path)
    triton_request_count = 0
    if backend == "triton":
        if not triton_endpoint:
            raise RuntimeError("triton backend selected but triton_endpoint is empty")
        ready, detail = check_triton_health(
            triton_endpoint,
            timeout_seconds=float(triton_healthcheck_timeout_seconds),
        )
        if not ready:
            raise RuntimeError(
                f"triton readiness check failed for score task: endpoint={triton_endpoint} detail={detail}"
            )
        endpoint_cfg = TritonEndpointConfig(
            endpoint=str(triton_endpoint),
            model_name=str(triton_model_name),
            model_version=(
                str(triton_model_version).strip() if triton_model_version else None
            ),
            timeout_seconds=float(triton_timeout_seconds),
        )
        game_frames = _split_frame_by_game(features_df)
        if not game_frames:
            raise RuntimeError("triton scoring received no game rows")
        score_frames: list[pd.DataFrame] = []
        last_response: dict[str, Any] = {}
        for game_id_value, game_features_df in game_frames:
            game_seed = _per_game_request_seed(int(random_seed), int(game_id_value))
            game_suffix = f"game_{int(game_id_value)}"
            game_features_path = run_dir / f"features_score_{game_suffix}.parquet"
            game_scores_path = run_dir / f"scores_{game_suffix}.parquet"
            _atomic_write_validated_parquet(
                game_features_df,
                game_features_path,
                required_cols=("game_id", "team_id", "player_id"),
            )
            triton_request_count += 1
            try:
                response = infer_json_action(
                    cfg=endpoint_cfg,
                    request_payload={
                        "action": "score",
                        "game_date": str(game_date),
                        "features_path": str(game_features_path),
                        "out_path": str(game_scores_path),
                        "bundle_dir": str(bundle_dir),
                        "random_seed": int(game_seed),
                        "device": str(gtv2_device or ""),
                        "batch_size": 4,
                    },
                )
            except TritonInferenceError as exc:
                raise RuntimeError(
                    f"triton score request failed for game_id={int(game_id_value)}: {exc}"
                ) from exc
            if not bool(response.get("ok")):
                raise RuntimeError(
                    "triton score response indicated failure for "
                    f"game_id={int(game_id_value)}: {response}"
                )
            if not game_scores_path.exists():
                raise RuntimeError(
                    "triton score response ok but output parquet missing for "
                    f"game_id={int(game_id_value)}: {game_scores_path}"
                )
            game_scores = pd.read_parquet(game_scores_path)
            if game_scores.empty:
                raise RuntimeError(
                    f"triton scoring produced zero rows for game_id={int(game_id_value)}"
                )
            score_frames.append(game_scores)
            last_response = dict(response)
        if not score_frames:
            raise RuntimeError("triton scoring produced zero game outputs")
        scores = pd.concat(score_frames, ignore_index=True)
        scores = scores.sort_values(
            ["game_date", "game_id", "team_id", "player_id"]
        ).reset_index(drop=True)
        device_for_summary = str(
            last_response.get("device") or gtv2_device or "triton"
        )
    else:
        _set_inference_seed(int(random_seed))
        device = _resolve_torch_device(gtv2_device)
        config, model = _load_gtv2_model(bundle_dir, device=device)
        runtime = _gtv2_inference_runtime()
        game_frames = _split_frame_by_game(features_df)
        if not game_frames:
            raise RuntimeError("local scoring received no game rows")
        score_frames: list[pd.DataFrame] = []
        for _game_id_value, game_features_df in game_frames:
            game_scores = runtime.score_gtv2_features_df(
                features_df=game_features_df,
                game_date=game_date,
                config=config,
                model=model,
                device=device,
                batch_size=1,
            )
            score_frames.append(game_scores)
        scores = pd.concat(score_frames, ignore_index=True)
        scores = scores.sort_values(
            ["game_date", "game_id", "team_id", "player_id"]
        ).reset_index(drop=True)
        device_for_summary = str(device)

    _atomic_write_validated_parquet(
        scores,
        out_path,
        required_cols=("game_date", "game_id", "team_id", "player_id"),
    )
    summary_path.write_text(
        json.dumps(
            {
                "placeholder_mode": False,
                "rows": int(len(scores)),
                "games": int(scores["game_id"].nunique()),
                "players": int(scores["player_id"].nunique()),
                "device": device_for_summary,
                "inference_backend": backend,
                "bundle_dir": str(bundle_dir),
                "triton_request_count": int(triton_request_count),
                "created_at": _utc_now_iso(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return out_path


@task(name="generate-worlds-gtv2-live", retries=0)
def generate_worlds_gtv2_live_task(
    *,
    game_date: str,
    run_id: str,
    run_as_of_ts: str | None = None,
    features_path: Path,
    scores_path: Path,
    bundle_dir: Path,
    data_root: Path,
    sim_worlds: int,
    placeholder_mode: bool,
    inference_backend: str = "local",
    triton_endpoint: str | None = None,
    triton_model_name: str = "gtv2_scorer",
    triton_model_version: str | None = None,
    triton_timeout_seconds: float = 90.0,
    triton_healthcheck_timeout_seconds: float = 3.0,
    gtv2_device: str | None = None,
    world_chunk_size: int = 64,
    active_temperature: float = 1.0,
    random_seed: int = 42,
    strict_world_contracts: bool = False,
    flow_scale_clip_override: float | None = None,
    make_model_mode: str = "beta_binomial_all",
    make_model_use_learned_efficiency: bool = True,
    allocation_top_usage_top1_scale: float = 1.0,
    allocation_top_usage_top2_scale: float = 1.0,
    apply_props_uplift: bool = True,
    props_uplift_scope: str = "all_players",
    props_uplift_confidence_weighted: bool = True,
    apply_propless_tail_calibration: bool = True,
    propless_tail_min_minutes_mean: float = 21.0,
    propless_tail_min_dk_mean: float = 16.0,
    propless_tail_boost: float = 0.14,
    propless_tail_max_scale: float = 1.22,
    apply_mid_minutes_tail_calibration: bool = True,
    mid_minutes_tail_min_minutes: float = 12.0,
    mid_minutes_tail_max_minutes: float = 20.0,
    mid_minutes_tail_boost: float = 0.14,
    apply_team_implied_points_reconcile: bool = False,
    team_implied_points_reconcile_alpha: float = 0.75,
    team_implied_points_reconcile_deadband_points: float = 2.0,
    apply_world_realism_controls: bool = True,
    world_realism_low_minutes_tail_damping_enabled: bool = True,
    world_realism_low_minutes_threshold: float = 12.0,
    world_realism_low_minutes_min_scale: float = 0.55,
    world_realism_outlier_resample_enabled: bool = True,
    world_realism_outlier_resample_max_passes: int = 1,
    apply_team_dk_fpts_correlation_overlay: bool = False,
    team_dk_fpts_correlation_overlay_alpha: float = 0.0,
    team_dk_fpts_correlation_overlay_min_minutes: float = 0.0,
    team_dk_fpts_correlation_overlay_weight_power: float = 1.0,
    promotion_hybrid_enabled: bool = False,
    promotion_expert_run_dir: str | None = None,
    promotion_prior_minutes_max: float = 12.0,
    promotion_hist_start_rate_max: float = 0.20,
    promotion_blend_mode: str = "uplift_only",
    promotion_force_active_candidates: bool = False,
    sparse_hybrid_enabled: bool = False,
    sparse_expert_run_dir: str | None = None,
    sparse_prior_minutes_max: float = 12.0,
    sparse_prior_play_prob_max: float = 0.50,
    sparse_blend_mode: str = "uplift_only",
    sparse_blend_alpha: float = 1.0,
    sparse_require_no_props: bool = False,
    sparse_gate_artifact: str | None = None,
    tree_rate_bundle_dir: str | None = None,
    tree_rate_predictions_csv: str | None = None,
    tree_rate_blend_alpha: float = 0.0,
    tree_rate_oreb_share_override_enabled: bool = False,
    minutes_uncertainty_enabled: bool = False,
    minutes_uncertainty_mode: str = "gaussian",
    minutes_uncertainty_gaussian_scale: float = 1.0,
    minutes_uncertainty_min_sigma: float = 0.75,
    minutes_uncertainty_max_sigma: float = 6.0,
    minutes_uncertainty_fallback_sigma: float = 1.5,
    minutes_uncertainty_use_hurdle_sigma: bool = True,
    minutes_uncertainty_use_prior_std: bool = True,
    minutes_uncertainty_preserve_top_k_per_team: int = 3,
    minutes_uncertainty_full_sigma_at_minutes_or_below: float = 24.0,
    minutes_uncertainty_zero_sigma_at_minutes_or_above: float = 32.0,
    minutes_uncertainty_apply_minutes_taper: bool = True,
    minutes_uncertainty_dirichlet_base_concentration: float = 24.0,
    minutes_uncertainty_lookup_artifact: str | None = None,
    minutes_uncertainty_empirical_blend_alpha: float = 1.0,
) -> dict[str, str]:
    run_dir = (
        data_root
        / "artifacts"
        / WORLDS_ROOT
        / f"game_date={game_date}"
        / f"run={run_id}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    worlds_path = run_dir / "worlds.parquet"
    projections_path = run_dir / "projections.parquet"
    worlds_summary_path = run_dir / "world_contracts_summary.json"

    if placeholder_mode:
        scores = pd.read_parquet(scores_path)
        projections = scores[["game_date", "game_id", "team_id", "player_id"]].copy()
        projections["minutes_sim_mean"] = scores["minutes_mean"].astype(float)
        projections["minutes_sim_p50"] = scores["minutes_mean"].astype(float)
        projections["dk_fpts_mean"] = (
            scores["minutes_mean"].astype(float)
            * scores["dk_rate"].astype(float)
            / 60.0
        ).round(4)
        projections["dk_fpts_p50"] = projections["dk_fpts_mean"]
        projections["sim_p_active"] = scores["play_prob"].astype(float)
        projections["n_worlds"] = int(sim_worlds)
        projections["sim_profile"] = "game_transformer_v2"
        projections = projections[PLACEHOLDER_PROJECTION_COLUMNS]
        _atomic_write_validated_parquet(
            projections,
            projections_path,
            required_cols=("game_date", "game_id", "team_id", "player_id"),
        )
        _atomic_write_validated_parquet(
            pd.DataFrame(
                columns=["world_idx", "game_id", "team_id", "player_id"]
            ),
            worlds_path,
            required_cols=("world_idx",),
        )
        contract_summary = {
            "contract_checks": {
                "team_minutes_not_240": 0,
                "team_minutes_total_checks": 0,
                "team_minutes_max_abs_drift": 0.0,
                "minutes_negative": 0,
                "minutes_over_48": 0,
                "negative_stats": 0,
                "fg2m_gt_fga2": 0,
                "fg3m_gt_fga3": 0,
                "ftm_gt_fta": 0,
                "inactive_nonzero_stats": 0,
                "inactive_nonzero_fpts_proxy": 0,
                "total_violations": 0,
            },
            "world_realism_controls": {
                "applied": False,
                "reason": "placeholder_mode",
            },
            "propless_tail_calibration": {
                "applied": False,
                "reason": "placeholder_mode",
            },
            "mid_minutes_tail_calibration": {
                "applied": False,
                "reason": "placeholder_mode",
            },
            "team_dk_fpts_correlation_overlay": {
                "applied": False,
                "reason": "placeholder_mode",
            },
            "placeholder_mode": True,
        }
    else:
        backend = str(inference_backend).strip().lower()
        if backend not in {"local", "triton"}:
            raise RuntimeError(
                f"unsupported inference backend for worlds task: {backend}"
            )
        if bool(promotion_hybrid_enabled) and backend != "local":
            raise RuntimeError(
                "promotion hybrid currently supports only local GTv2 worlds inference"
            )
        if bool(sparse_hybrid_enabled) and backend != "local":
            raise RuntimeError(
                "sparse hybrid currently supports only local GTv2 worlds inference"
            )
        logger = get_run_logger()
        _set_inference_seed(int(random_seed))
        device = _resolve_torch_device(gtv2_device)
        device_for_summary = str(device)
        worlds_runtime = _gtv2_worlds_runtime()
        minutes_uncertainty_lookup = _load_minutes_uncertainty_lookup_artifact(
            minutes_uncertainty_lookup_artifact
        )
        make_model_cfg = worlds_runtime.MakeModelConfig(
            mode=str(make_model_mode),
            use_learned_efficiency=bool(make_model_use_learned_efficiency),
        )

        # Warn loudly if using scale_clip override (experimental mode)
        if flow_scale_clip_override is not None:
            logger.warning("=" * 80)
            logger.warning(
                "EXPERIMENTAL: flow_scale_clip_override = %.2f",
                flow_scale_clip_override,
            )
            logger.warning("This is a non-default setting for H1 hypothesis testing.")
            logger.warning("Production runs should use the trained default (2.0).")
            logger.warning("=" * 80)
        features_df_raw = pd.read_parquet(features_path)
        scores_df = pd.read_parquet(scores_path)
        features_df_raw, score_surface_diag = _attach_gtv2_score_surface(
            features_df_raw,
            scores_df=scores_df,
        )
        logger.info("Attached persisted GTv2 score surface: %s", score_surface_diag)
        features_df, force_active_diag = _attach_gtv2_force_active_worlds(
            features_df_raw,
            game_date=game_date,
            data_root=data_root,
            as_of_ts=run_as_of_ts,
        )
        logger.info("Applied force-active world guardrails: %s", force_active_diag)
        tree_rate_predictions_report: dict[str, Any]
        resolved_tree_rate_predictions_csv = (
            str(tree_rate_predictions_csv).strip() if tree_rate_predictions_csv else ""
        )
        if (
            not resolved_tree_rate_predictions_csv
            and tree_rate_bundle_dir
            and float(tree_rate_blend_alpha) > 0.0
        ):
            tree_rate_output_csv = run_dir / "tree_rate_predictions.csv"
            tree_rate_predictions_report = score_tree_rate_bundle_features_to_csv(
                features_df=features_df_raw,
                bundle_dir=Path(str(tree_rate_bundle_dir)).expanduser().resolve(),
                output_csv=tree_rate_output_csv,
                include_extra_cols=["player_name"],
                game_date=game_date,
            )
            resolved_tree_rate_predictions_csv = str(tree_rate_output_csv)
            logger.info("Scored tree rate bundle for live override: %s", tree_rate_predictions_report)
        elif resolved_tree_rate_predictions_csv:
            tree_rate_predictions_report = {
                "applied": False,
                "reason": "external_predictions_csv",
                "output_csv": resolved_tree_rate_predictions_csv,
            }
        else:
            tree_rate_predictions_report = {"applied": False, "reason": "disabled"}
        contract_checks_seed: dict[str, Any]
        triton_request_count = 0
        raw_worlds_path: Path | None = None
        if backend == "triton":
            if not triton_endpoint:
                raise RuntimeError(
                    "triton backend selected but triton_endpoint is empty"
                )
            ready, detail = check_triton_health(
                triton_endpoint,
                timeout_seconds=float(triton_healthcheck_timeout_seconds),
            )
            if not ready:
                raise RuntimeError(
                    "triton readiness check failed for worlds task: "
                    f"endpoint={triton_endpoint} detail={detail}"
                )

            raw_worlds_path = run_dir / "worlds_raw.parquet"
            endpoint_cfg = TritonEndpointConfig(
                endpoint=str(triton_endpoint),
                model_name=str(triton_model_name),
                model_version=(
                    str(triton_model_version).strip()
                    if triton_model_version
                    else None
                ),
                timeout_seconds=float(triton_timeout_seconds),
            )
            game_frames = _split_frame_by_game(features_df)
            if not game_frames:
                raise RuntimeError("triton worlds generation received no game rows")
            world_frames: list[pd.DataFrame] = []
            contract_counter: Counter[str] = Counter()
            last_response: dict[str, Any] = {}
            for game_id_value, game_features_df in game_frames:
                game_seed = _per_game_request_seed(int(random_seed), int(game_id_value))
                game_suffix = f"game_{int(game_id_value)}"
                game_features_path = (
                    run_dir / f"features_for_worlds_{game_suffix}.parquet"
                )
                game_worlds_path = run_dir / f"worlds_raw_{game_suffix}.parquet"
                _atomic_write_validated_parquet(
                    game_features_df,
                    game_features_path,
                    required_cols=("game_id", "team_id", "player_id"),
                )
                request_payload = {
                    "action": "worlds",
                    "game_date": str(game_date),
                    "features_path": str(game_features_path),
                    "out_path": str(game_worlds_path),
                    "bundle_dir": str(bundle_dir),
                    "random_seed": int(game_seed),
                    "device": str(gtv2_device or ""),
                    "num_worlds": int(sim_worlds),
                    "world_chunk_size": int(world_chunk_size),
                    "active_temperature": float(active_temperature),
                    "strict_world_contracts": bool(strict_world_contracts),
                    "flow_scale_clip_override": (
                        float(flow_scale_clip_override)
                        if flow_scale_clip_override is not None
                        else None
                    ),
                    "make_model_mode": str(make_model_mode),
                    "make_model_use_learned_efficiency": bool(
                        make_model_use_learned_efficiency
                    ),
                    "allocation_top_usage_top1_scale": float(
                        allocation_top_usage_top1_scale
                    ),
                    "allocation_top_usage_top2_scale": float(
                        allocation_top_usage_top2_scale
                    ),
                    "minutes_uncertainty_enabled": bool(minutes_uncertainty_enabled),
                    "minutes_uncertainty_mode": str(minutes_uncertainty_mode),
                    "minutes_uncertainty_gaussian_scale": float(minutes_uncertainty_gaussian_scale),
                    "minutes_uncertainty_min_sigma": float(minutes_uncertainty_min_sigma),
                    "minutes_uncertainty_max_sigma": float(minutes_uncertainty_max_sigma),
                    "minutes_uncertainty_fallback_sigma": float(minutes_uncertainty_fallback_sigma),
                    "minutes_uncertainty_use_hurdle_sigma": bool(minutes_uncertainty_use_hurdle_sigma),
                    "minutes_uncertainty_use_prior_std": bool(minutes_uncertainty_use_prior_std),
                    "minutes_uncertainty_preserve_top_k_per_team": int(minutes_uncertainty_preserve_top_k_per_team),
                    "minutes_uncertainty_full_sigma_at_minutes_or_below": float(
                        minutes_uncertainty_full_sigma_at_minutes_or_below
                    ),
                    "minutes_uncertainty_zero_sigma_at_minutes_or_above": float(
                        minutes_uncertainty_zero_sigma_at_minutes_or_above
                    ),
                    "minutes_uncertainty_apply_minutes_taper": bool(minutes_uncertainty_apply_minutes_taper),
                    "minutes_uncertainty_dirichlet_base_concentration": float(
                        minutes_uncertainty_dirichlet_base_concentration
                    ),
                    "minutes_uncertainty_empirical_bin_edges": list(
                        minutes_uncertainty_lookup.get("bin_edges", [])
                    ),
                    "minutes_uncertainty_empirical_sigma_by_bin": list(
                        minutes_uncertainty_lookup.get("sigma_by_bin", [])
                    ),
                    "minutes_uncertainty_empirical_blend_alpha": float(
                        minutes_uncertainty_empirical_blend_alpha
                    ),
                }
                response: dict[str, Any] | None = None
                game_worlds_df: pd.DataFrame | None = None
                read_error: Exception | None = None
                max_world_attempts = 3
                for world_attempt in range(1, max_world_attempts + 1):
                    triton_request_count += 1
                    try:
                        response = infer_json_action(
                            cfg=endpoint_cfg,
                            request_payload=request_payload,
                        )
                    except TritonInferenceError as exc:
                        raise RuntimeError(
                            f"triton worlds request failed for game_id={int(game_id_value)}: {exc}"
                        ) from exc
                    if not bool(response.get("ok")):
                        raise RuntimeError(
                            "triton worlds response indicated failure for "
                            f"game_id={int(game_id_value)}: {response}"
                        )
                    if not game_worlds_path.exists():
                        raise RuntimeError(
                            "triton worlds response ok but output parquet missing for "
                            f"game_id={int(game_id_value)}: {game_worlds_path}"
                        )
                    try:
                        game_worlds_df = pd.read_parquet(game_worlds_path)
                        read_error = None
                        break
                    except (OSError, pa.ArrowInvalid) as exc:
                        read_error = exc
                        if world_attempt >= max_world_attempts:
                            break
                        logger.warning(
                            "Retrying triton worlds generation after unreadable parquet: "
                            "game_id=%s attempt=%s/%s path=%s error=%s",
                            int(game_id_value),
                            world_attempt,
                            max_world_attempts,
                            game_worlds_path,
                            exc,
                        )
                        game_worlds_path.unlink(missing_ok=True)
                        time.sleep(0.2)
                if game_worlds_df is None:
                    raise RuntimeError(
                        "triton worlds parquet unreadable after retries for "
                        f"game_id={int(game_id_value)} path={game_worlds_path}: {read_error}"
                    ) from read_error
                if game_worlds_df.empty:
                    raise RuntimeError(
                        f"triton worlds generation produced zero rows for game_id={int(game_id_value)}"
                    )
                world_frames.append(game_worlds_df)
                raw_checks = (response or {}).get("contract_checks")
                if isinstance(raw_checks, dict):
                    for key, value in raw_checks.items():
                        try:
                            contract_counter[str(key)] += int(value)
                        except Exception:  # noqa: BLE001
                            continue
                last_response = dict(response)
            worlds_df = _concat_frames_without_pandas_concat(world_frames)
            contract_checks_seed = dict(contract_counter)
            device_for_summary = str(
                last_response.get("device")
                or last_response.get("effective_device")
                or device
            )
        else:
            from projections.rotation.game_transformer_v2 import (
                GameLevelDataset,
                collate_game_level_examples,
            )
            from torch.utils.data import DataLoader

            config, model = _load_gtv2_model(
                bundle_dir,
                device=device,
                flow_scale_clip_override=flow_scale_clip_override,
            )
            promotion_expert_model = None
            promotion_hybrid_config: PromotionHybridConfig | None = None
            sparse_expert_model = None
            sparse_hybrid_config: SparseEmergencyHybridConfig | None = None
            sparse_gate_config: SparseEmergencyGateConfig | None = None
            if bool(promotion_hybrid_enabled):
                if not promotion_expert_run_dir:
                    raise RuntimeError(
                        "promotion_hybrid_enabled requires promotion_expert_run_dir"
                    )
                promotion_expert_path = (
                    Path(promotion_expert_run_dir).expanduser().resolve()
                )
                promotion_expert_cfg, promotion_expert_model = _load_gtv2_model(
                    promotion_expert_path,
                    device=device,
                    flow_scale_clip_override=flow_scale_clip_override,
                )
                assert_promotion_hybrid_compatible(config, promotion_expert_cfg)
                blend_mode = str(promotion_blend_mode).strip().lower()
                if blend_mode not in {"uplift_only", "replace"}:
                    raise RuntimeError(
                        "promotion_blend_mode must be one of: uplift_only, replace"
                    )
                promotion_hybrid_config = PromotionHybridConfig.from_model_config(
                    config,
                    prior_minutes_max=float(promotion_prior_minutes_max),
                    hist_start_rate_max=float(promotion_hist_start_rate_max),
                    uplift_only=(blend_mode == "uplift_only"),
                    force_active_candidates=bool(promotion_force_active_candidates),
                )
            if bool(sparse_hybrid_enabled):
                if not sparse_expert_run_dir:
                    raise RuntimeError(
                        "sparse_hybrid_enabled requires sparse_expert_run_dir"
                    )
                sparse_expert_path = Path(sparse_expert_run_dir).expanduser().resolve()
                sparse_expert_cfg, sparse_expert_model = _load_gtv2_model(
                    sparse_expert_path,
                    device=device,
                    flow_scale_clip_override=flow_scale_clip_override,
                )
                assert_promotion_hybrid_compatible(config, sparse_expert_cfg)
                sparse_mode = str(sparse_blend_mode).strip().lower()
                if sparse_mode not in {"uplift_only", "replace"}:
                    raise RuntimeError(
                        "sparse_blend_mode must be one of: uplift_only, replace"
                    )
                sparse_hybrid_config = SparseEmergencyHybridConfig.from_model_config(
                    config,
                    prior_minutes_max=float(sparse_prior_minutes_max),
                    prior_play_prob_max=float(sparse_prior_play_prob_max),
                    uplift_only=(sparse_mode == "uplift_only"),
                    force_active_candidates=False,
                    blend_alpha=float(sparse_blend_alpha),
                    require_no_props=bool(sparse_require_no_props),
                )
                if sparse_gate_artifact:
                    sparse_gate_config = SparseEmergencyGateConfig.from_artifact(
                        config,
                        str(sparse_gate_artifact),
                    )
            examples = _build_gtv2_inference_examples(
                features_df=features_df,
                game_date=game_date,
                config=config,
            )
            loader = DataLoader(
                GameLevelDataset(examples),
                batch_size=1,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_game_level_examples,
            )
            minutes_uncertainty_config = worlds_runtime.MinutesUncertaintyConfig(
                enabled=bool(minutes_uncertainty_enabled),
                mode=str(minutes_uncertainty_mode),
                gaussian_scale=float(minutes_uncertainty_gaussian_scale),
                min_sigma=float(minutes_uncertainty_min_sigma),
                max_sigma=float(minutes_uncertainty_max_sigma),
                fallback_sigma=float(minutes_uncertainty_fallback_sigma),
                use_hurdle_sigma=bool(minutes_uncertainty_use_hurdle_sigma),
                use_prior_std=bool(minutes_uncertainty_use_prior_std),
                preserve_top_k_per_team=int(minutes_uncertainty_preserve_top_k_per_team),
                full_sigma_at_minutes_or_below=float(
                    minutes_uncertainty_full_sigma_at_minutes_or_below
                ),
                zero_sigma_at_minutes_or_above=float(
                    minutes_uncertainty_zero_sigma_at_minutes_or_above
                ),
                apply_minutes_taper=bool(minutes_uncertainty_apply_minutes_taper),
                dirichlet_base_concentration=float(
                    minutes_uncertainty_dirichlet_base_concentration
                ),
                empirical_minutes_bin_edges=tuple(
                    float(x) for x in minutes_uncertainty_lookup.get("bin_edges", [])
                ),
                empirical_sigma_by_bin=tuple(
                    float(x) for x in minutes_uncertainty_lookup.get("sigma_by_bin", [])
                ),
                empirical_blend_alpha=float(minutes_uncertainty_empirical_blend_alpha),
            )
            world_frames: list[pd.DataFrame] = []
            contract_counter: Counter[str] = Counter()
            for batch in loader:
                df_batch, checks = worlds_runtime.sample_worlds_for_batch(
                    model,
                    batch,
                    device=device,
                    num_worlds=int(sim_worlds),
                    chunk_size=max(1, int(world_chunk_size)),
                    active_temperature=float(active_temperature),
                    strict_contracts=bool(strict_world_contracts),
                    make_model_config=make_model_cfg,
                    allocation_top_usage_top1_scale=float(
                        allocation_top_usage_top1_scale
                    ),
                    allocation_top_usage_top2_scale=float(
                        allocation_top_usage_top2_scale
                    ),
                    promotion_expert_model=promotion_expert_model,
                    promotion_hybrid_config=promotion_hybrid_config,
                    sparse_expert_model=sparse_expert_model,
                    sparse_hybrid_config=sparse_hybrid_config,
                    sparse_gate_config=sparse_gate_config,
                    minutes_uncertainty_config=minutes_uncertainty_config,
                )
                world_frames.append(df_batch)
                contract_counter.update(checks)
            worlds_df = _concat_frames_without_pandas_concat(world_frames)
            contract_checks_seed = dict(contract_counter)

        if worlds_df.empty:
            raise RuntimeError("GTV2 worlds generation produced zero rows")
        worlds_df, game_date_normalization_report = _coerce_world_game_date(
            worlds_df,
            game_date=game_date,
        )
        if bool(game_date_normalization_report.get("applied")):
            logger.warning(
                "Normalized generated world game_date values: %s",
                game_date_normalization_report,
            )
        if raw_worlds_path is not None:
            try:
                _atomic_write_validated_parquet(
                    worlds_df,
                    raw_worlds_path,
                    required_cols=("world_idx", "game_id", "team_id", "player_id"),
                    compression="zstd",
                    row_group_size=500_000,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to persist validated raw worlds parquet; continuing: path=%s error=%s",
                    raw_worlds_path,
                    exc,
                )
        worlds_df, world_key_report = _sanitize_frame_to_expected_keys(
            worlds_df,
            expected_keys_df=features_df,
            key_cols=("game_id", "team_id", "player_id"),
            label="generated worlds",
        )
        if worlds_df.empty:
            raise RuntimeError("GTV2 worlds generation produced zero valid rows after key sanitization")
        if (
            world_key_report["dropped_null_key_rows"] > 0
            or world_key_report["dropped_unexpected_key_rows"] > 0
        ):
            logger.warning(
                "Dropped invalid world rows before publish: %s",
                world_key_report,
            )
        tree_rate_override_report: dict[str, Any]
        if resolved_tree_rate_predictions_csv and float(tree_rate_blend_alpha) > 0.0:
            worlds_df, tree_rate_override_report = _apply_tree_rate_mean_override_to_worlds(
                worlds_df,
                predictions_csv=Path(str(resolved_tree_rate_predictions_csv)).expanduser().resolve(),
                blend_alpha=float(tree_rate_blend_alpha),
                oreb_share_override_enabled=bool(tree_rate_oreb_share_override_enabled),
            )
            if bool(tree_rate_override_report.get("applied")):
                logger.info("Applied tree rate world override: %s", tree_rate_override_report)
        else:
            tree_rate_override_report = {"applied": False, "reason": "disabled"}
        pre_calibration_pts_anchor = (
            _build_pre_calibration_points_anchor(
                worlds_df,
                label="generate_worlds_gtv2_live_task/pre_calibration_points_anchor",
            )
            if bool(apply_team_implied_points_reconcile)
            else pd.DataFrame(
                columns=["game_id", "team_id", "player_id", "pts_pre_calibration_mean"]
            )
        )
        props_uplift_report: dict[str, Any]
        if bool(apply_props_uplift):
            worlds_df, props_uplift_report = _apply_props_uplift_calibration_to_worlds(
                worlds_df,
                features_df=features_df,
                scope=str(props_uplift_scope),
                confidence_weighted=bool(props_uplift_confidence_weighted),
            )
            if bool(props_uplift_report.get("applied")):
                logger.info("Applied props uplift calibration: %s", props_uplift_report)
        else:
            props_uplift_report = {"applied": False, "reason": "disabled"}
        worlds_df, propless_tail_report = _apply_propless_tail_calibration_to_worlds(
            worlds_df,
            features_df=features_df,
            enabled=bool(apply_propless_tail_calibration),
            min_minutes_mean=float(propless_tail_min_minutes_mean),
            min_dk_mean=float(propless_tail_min_dk_mean),
            tail_boost=float(propless_tail_boost),
            max_tail_scale=float(propless_tail_max_scale),
            target_game_ids=None,
        )
        if bool(propless_tail_report.get("applied")):
            logger.info("Applied propless tail calibration: %s", propless_tail_report)
        worlds_df, mid_minutes_tail_report = _apply_mid_minutes_tail_calibration_to_worlds(
            worlds_df,
            enabled=bool(apply_mid_minutes_tail_calibration),
            min_minutes=float(mid_minutes_tail_min_minutes),
            max_minutes=float(mid_minutes_tail_max_minutes),
            tail_boost=float(mid_minutes_tail_boost),
            target_game_ids=None,
        )
        if bool(mid_minutes_tail_report.get("applied")):
            logger.info(
                "Applied mid-minutes tail calibration: %s",
                mid_minutes_tail_report,
            )
        worlds_df, team_implied_points_reconcile_report = (
            _apply_team_implied_points_reconcile_to_worlds(
                worlds_df,
                features_df=features_df,
                pre_calibration_pts_anchor=pre_calibration_pts_anchor,
                enabled=bool(apply_team_implied_points_reconcile),
                alpha=float(team_implied_points_reconcile_alpha),
                deadband_points=float(team_implied_points_reconcile_deadband_points),
            )
        )
        if bool(team_implied_points_reconcile_report.get("applied")):
            logger.info(
                "Applied team implied points reconcile: %s",
                team_implied_points_reconcile_report,
            )
        worlds_df, world_realism_report = _apply_world_realism_controls_to_worlds(
            worlds_df,
            enabled=bool(apply_world_realism_controls),
            random_seed=int(random_seed),
            low_minutes_tail_damping_enabled=bool(
                world_realism_low_minutes_tail_damping_enabled
            ),
            low_minutes_tail_minutes_threshold=float(
                world_realism_low_minutes_threshold
            ),
            low_minutes_tail_min_scale=float(world_realism_low_minutes_min_scale),
            outlier_resample_enabled=bool(world_realism_outlier_resample_enabled),
            outlier_resample_max_passes=int(
                world_realism_outlier_resample_max_passes
            ),
            target_game_ids=None,
        )
        if bool(world_realism_report.get("applied")):
            logger.info("Applied world realism controls: %s", world_realism_report)
        worlds_df, world_contract_repair_report = _repair_world_frame_contract_fields(
            worlds_df
        )
        if bool(world_contract_repair_report.get("applied")):
            logger.warning(
                "Applied world contract field repair before publish: %s",
                world_contract_repair_report,
            )
        worlds_df, world_key_report_post = _sanitize_frame_to_expected_keys(
            worlds_df,
            expected_keys_df=features_df,
            key_cols=("game_id", "team_id", "player_id"),
            label="generated worlds post-repair",
        )
        if worlds_df.empty:
            raise RuntimeError(
                "GTV2 worlds generation produced zero valid rows after final key sanitization"
            )
        if (
            world_key_report_post["dropped_null_key_rows"] > 0
            or world_key_report_post["dropped_unexpected_key_rows"] > 0
        ):
            logger.warning(
                "Dropped invalid world rows after repairs before publish: %s",
                world_key_report_post,
            )
        worlds_df, world_contract_repair_report_post_sanitize = (
            _repair_world_frame_contract_fields(worlds_df)
        )
        if bool(world_contract_repair_report_post_sanitize.get("applied")):
            logger.warning(
                "Applied post-sanitize world contract repair safety pass before publish: %s",
                world_contract_repair_report_post_sanitize,
            )
        worlds_df, team_dk_fpts_correlation_overlay_report = (
            _apply_team_dk_fpts_correlation_overlay_to_worlds(
                worlds_df,
                enabled=bool(apply_team_dk_fpts_correlation_overlay),
                alpha=float(team_dk_fpts_correlation_overlay_alpha),
                min_minutes=float(team_dk_fpts_correlation_overlay_min_minutes),
                weight_power=float(team_dk_fpts_correlation_overlay_weight_power),
            )
        )
        if bool(team_dk_fpts_correlation_overlay_report.get("applied")):
            logger.info(
                "Applied team dk_fpts correlation overlay: %s",
                team_dk_fpts_correlation_overlay_report,
            )
        _atomic_write_validated_parquet(
            worlds_df,
            worlds_path,
            required_cols=("world_idx", "game_id", "team_id", "player_id"),
            compression="zstd",
            row_group_size=500_000,
        )

        projections = worlds_runtime.summarize_worlds_to_projections(
            worlds_df,
            sim_profile="game_transformer_v2",
        )
        projections = _normalize_gtv2_projection_surface_semantics(projections)
        projections, projection_key_report = _sanitize_frame_to_expected_keys(
            projections,
            expected_keys_df=features_df,
            key_cols=("game_id", "team_id", "player_id"),
            label="generated world projections",
        )
        _atomic_write_validated_parquet(
            projections,
            projections_path,
            required_cols=("game_date", "game_id", "team_id", "player_id"),
        )
        contract_checks = dict(contract_checks_seed)
        contract_checks.update(_summarize_world_contracts_from_frame(worlds_df))
        contract_summary = {
            "contract_checks": contract_checks,
            "placeholder_mode": False,
            "world_rows": int(len(worlds_df)),
            "projection_rows": int(len(projections)),
            "bundle_dir": str(bundle_dir),
            "device": device_for_summary,
            "inference_backend": backend,
            "key_sanitization": {
                "worlds_pre_transforms": world_key_report,
                "worlds_post_transforms": world_key_report_post,
                "projections": projection_key_report,
            },
            "make_model": {
                "mode": str(make_model_cfg.mode),
                "use_learned_efficiency": bool(make_model_cfg.use_learned_efficiency),
            },
            "triton_request_count": int(triton_request_count),
            "force_active_guardrails": force_active_diag,
            "game_date_normalization": game_date_normalization_report,
            "tree_rate_predictions": tree_rate_predictions_report,
            "tree_rate_override": tree_rate_override_report,
            "props_uplift_calibration": props_uplift_report,
            "propless_tail_calibration": propless_tail_report,
            "mid_minutes_tail_calibration": mid_minutes_tail_report,
            "team_dk_fpts_correlation_overlay": team_dk_fpts_correlation_overlay_report,
            "world_realism_controls": world_realism_report,
            "world_contract_field_repair": world_contract_repair_report,
            "world_contract_field_repair_post_sanitize": (
                world_contract_repair_report_post_sanitize
            ),
            "promotion_hybrid": {
                "enabled": bool(promotion_hybrid_enabled),
                "expert_run_dir": (
                    str(Path(promotion_expert_run_dir).expanduser().resolve())
                    if promotion_expert_run_dir
                    else None
                ),
                "prior_minutes_max": (
                    float(promotion_prior_minutes_max)
                    if bool(promotion_hybrid_enabled)
                    else None
                ),
                "hist_start_rate_max": (
                    float(promotion_hist_start_rate_max)
                    if bool(promotion_hybrid_enabled)
                    else None
                ),
                "blend_mode": (
                    str(promotion_blend_mode)
                    if bool(promotion_hybrid_enabled)
                    else None
                ),
                "force_active_candidates": (
                    bool(promotion_force_active_candidates)
                    if bool(promotion_hybrid_enabled)
                    else None
                ),
            },
            "sparse_hybrid": {
                "enabled": bool(sparse_hybrid_enabled),
                "expert_run_dir": (
                    str(Path(sparse_expert_run_dir).expanduser().resolve())
                    if sparse_expert_run_dir
                    else None
                ),
                "prior_minutes_max": (
                    float(sparse_prior_minutes_max) if bool(sparse_hybrid_enabled) else None
                ),
                "prior_play_prob_max": (
                    float(sparse_prior_play_prob_max) if bool(sparse_hybrid_enabled) else None
                ),
                "blend_mode": (
                    str(sparse_blend_mode) if bool(sparse_hybrid_enabled) else None
                ),
                "blend_alpha": (
                    float(sparse_blend_alpha) if bool(sparse_hybrid_enabled) else None
                ),
                "require_no_props": (
                    bool(sparse_require_no_props) if bool(sparse_hybrid_enabled) else None
                ),
                "gate_artifact": (
                    str(Path(sparse_gate_artifact).expanduser().resolve())
                    if sparse_gate_artifact
                    else None
                ),
            },
            "created_at": _utc_now_iso(),
        }
        if backend == "triton":
            contract_summary["triton"] = {
                "endpoint": str(triton_endpoint),
                "model_name": str(triton_model_name),
                "model_version": (
                    str(triton_model_version).strip()
                    if triton_model_version
                    else None
                ),
                "timeout_seconds": float(triton_timeout_seconds),
                "request_count": int(triton_request_count),
            }

    worlds_summary_path.write_text(
        json.dumps(contract_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return {
        "worlds_dir": str(run_dir),
        "worlds_path": str(worlds_path),
        "projections_path": str(projections_path),
        "world_contract_summary_path": str(worlds_summary_path),
    }


def _merge_live_ownership_into_projections(
    df: pd.DataFrame,
    *,
    game_date: str,
    run_id: str,
    data_root: Path,
) -> pd.DataFrame:
    own_dir = data_root / "silver" / "ownership_predictions" / game_date / f"run={run_id}"
    if not own_dir.exists():
        return df

    slate_files = [
        path
        for path in own_dir.glob("*.parquet")
        if not path.name.endswith("_locked.parquet")
    ]
    if not slate_files:
        return df

    own_path = max(slate_files, key=lambda path: path.stat().st_size)
    own_df = pd.read_parquet(own_path)
    if own_df.empty or "player_name" not in own_df.columns:
        return df

    merged = df.copy()
    own = own_df.copy()
    if "player_name" in merged.columns and "player_name" in own.columns:
        merged["_join_name"] = merged["player_name"].apply(normalize_player_name)
        own["_join_name"] = own["player_name"].apply(normalize_player_name)

        join_cols = ["_join_name"]
        if "team_tricode" in merged.columns and "team" in own.columns:
            merged["_join_team"] = merged["team_tricode"].astype(str).str.upper()
            own["_join_team"] = own["team"].astype(str).str.upper()
            join_cols.append("_join_team")
    elif "player_id" in merged.columns and "player_id" in own.columns:
        merged["player_id"] = pd.to_numeric(merged["player_id"], errors="coerce").astype(
            "Int64"
        )
        own["player_id"] = pd.to_numeric(own["player_id"], errors="coerce").astype(
            "Int64"
        )
        join_cols = ["player_id"]
    else:
        return df

    own_cols = join_cols + [
        col for col in ("salary", "pred_own_pct", "draft_group_id") if col in own.columns
    ]
    if len(own_cols) == len(join_cols):
        return df

    merged = merged.merge(
        own[own_cols].drop_duplicates(subset=join_cols, keep="last"),
        on=join_cols,
        how="left",
        suffixes=("", "__own"),
    )
    for col in ("salary", "pred_own_pct", "draft_group_id"):
        own_col = f"{col}__own"
        if own_col not in merged.columns:
            continue
        merged[col] = merged[col].where(pd.notna(merged[col]), merged[own_col])
        merged = merged.drop(columns=[own_col])

    return merged.drop(columns=["_join_name", "_join_team"], errors="ignore")


@task(name="finalize-projections-live", retries=0)
def finalize_projections_live_task(
    *,
    game_date: str,
    run_id: str,
    worlds_projections_path: Path,
    data_root: Path,
    placeholder_mode: bool,
    target_game_ids: list[int] | None = None,
) -> Path:
    out_dir = data_root / "artifacts" / "projections" / game_date / f"run={run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "projections.parquet"

    df = pd.read_parquet(worlds_projections_path)
    if df.empty:
        raise RuntimeError(f"world projections are empty: {worlds_projections_path}")
    df = _filter_to_target_games(df, target_game_ids)
    if df.empty:
        raise RuntimeError(
            "world projections are empty after applying target_game_ids: "
            f"{target_game_ids}"
        )
    df = _normalize_gtv2_projection_surface_semantics(df)

    # Enrich run-scoped projections with display + vegas context fields so the
    # dashboard can render a read-only game view without additional joins.
    display_src = (
        data_root
        / "live"
        / "features_minutes_v1"
        / game_date
        / f"run={run_id}"
        / "features.parquet"
    )
    if display_src.exists():
        src_df = pd.read_parquet(display_src)
        join_keys = ["game_id", "team_id", "player_id"]
        needed = [
            "player_name",
            "team_name",
            "team_tricode",
            "opponent_team_id",
            "opponent_team_name",
            "opponent_team_tricode",
            "status",
            "is_out",
            "tip_ts",
            "is_projected_starter",
            "is_confirmed_starter",
            "team_implied_total",
            "opponent_implied_total",
            "total",
            "spread_home",
        ]
        present = [c for c in join_keys + needed if c in src_df.columns]
        if all(k in present for k in join_keys):
            enrich = src_df.loc[:, present].copy()
            for key in join_keys:
                enrich[key] = pd.to_numeric(enrich[key], errors="coerce").astype(
                    "Int64"
                )
                df[key] = pd.to_numeric(df[key], errors="coerce").astype("Int64")
            enrich = enrich.dropna(subset=join_keys).drop_duplicates(
                subset=join_keys, keep="last"
            )
            df = df.merge(enrich, on=join_keys, how="left", suffixes=("", "__src"))
            for col in needed:
                src_col = f"{col}__src"
                if src_col not in df.columns:
                    continue
                if col in df.columns:
                    df[col] = df[col].where(pd.notna(df[col]), df[src_col])
                else:
                    df[col] = df[src_col]
            df = df.drop(
                columns=[c for c in df.columns if c.endswith("__src")], errors="ignore"
            )

    status_series = (
        df["status"].fillna("").astype(str).str.upper().str.strip()
        if "status" in df.columns
        else pd.Series("", index=df.index, dtype="string")
    )
    status_out_mask = (
        status_series.isin({"OUT", "O", "INACTIVE", "D", "DOUBTFUL", "SUSPENDED"})
        | status_series.str.contains("DOUBT", na=False)
    )
    is_out_series = (
        pd.to_numeric(df["is_out"], errors="coerce").fillna(0).astype(int).eq(1)
        if "is_out" in df.columns
        else pd.Series(False, index=df.index)
    )
    out_mask = status_out_mask | is_out_series
    if bool(out_mask.any()):
        df["is_out"] = out_mask.astype(int)
        if "status" in df.columns:
            df.loc[out_mask, "status"] = "OUT"
        else:
            df["status"] = np.where(out_mask, "OUT", "")

        zero_prefixes = (
            "minutes",
            "sim_minutes",
            "dk_fpts",
            "sim_dk_fpts",
            "fpts_sim",
            "pts_",
            "reb_",
            "ast_",
            "stl_",
            "blk_",
            "tov_",
            "sim_pts_",
            "sim_reb_",
            "sim_ast_",
            "sim_stl_",
            "sim_blk_",
            "sim_tov_",
            "p_play",
        )
        zero_exact = {
            "value",
            "play_prob",
            "pred_own_pct",
            "own_proj",
            "minutes_sim_p_active",
        }
        id_like_cols = {
            "game_id",
            "team_id",
            "player_id",
            "opponent_team_id",
            "n_worlds",
            "season",
        }
        zero_cols: list[str] = []
        for col in df.columns:
            if col in id_like_cols:
                continue
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            if col in zero_exact or col.startswith(zero_prefixes):
                zero_cols.append(col)
        if zero_cols:
            df.loc[out_mask, zero_cols] = 0.0

    df = _merge_live_ownership_into_projections(
        df,
        game_date=game_date,
        run_id=run_id,
        data_root=data_root,
    )
    if "dk_fpts_mean" in df.columns and "salary" in df.columns:
        salary = pd.to_numeric(df["salary"], errors="coerce")
        df["value"] = (
            pd.to_numeric(df["dk_fpts_mean"], errors="coerce")
            .div(salary.where(salary > 0))
            .mul(1000)
            .round(2)
        )

    _atomic_write_validated_parquet(
        df,
        out_path,
        required_cols=("game_date", "game_id", "team_id", "player_id"),
    )
    return out_dir


def _postprocess_target_world_slice_for_game_scoped_merge(
    *,
    worlds_df: pd.DataFrame,
    features_df: pd.DataFrame,
    target_game_ids: list[int],
    apply_props_uplift: bool,
    props_uplift_scope: str,
    props_uplift_confidence_weighted: bool,
    apply_propless_tail_calibration: bool,
    propless_tail_min_minutes_mean: float,
    propless_tail_min_dk_mean: float,
    propless_tail_boost: float,
    propless_tail_max_scale: float,
    apply_mid_minutes_tail_calibration: bool,
    mid_minutes_tail_min_minutes: float,
    mid_minutes_tail_max_minutes: float,
    mid_minutes_tail_boost: float,
    apply_team_implied_points_reconcile: bool,
    team_implied_points_reconcile_alpha: float,
    team_implied_points_reconcile_deadband_points: float,
    apply_world_realism_controls: bool,
    world_realism_low_minutes_tail_damping_enabled: bool,
    world_realism_low_minutes_threshold: float,
    world_realism_low_minutes_min_scale: float,
    world_realism_outlier_resample_enabled: bool,
    world_realism_outlier_resample_max_passes: int,
    random_seed: int,
) -> tuple[
    pd.DataFrame,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    """Post-process only target games when materializing game-scoped merges."""
    target_ids = _normalize_game_ids(target_game_ids)
    scope_report = {
        "target_game_ids": target_ids,
        "target_row_count_before": 0,
        "target_row_count_after": 0,
    }
    if worlds_df.empty:
        base = {"applied": False, "reason": "empty_worlds", **scope_report}
        return worlds_df, dict(base), dict(base), dict(base), dict(base), dict(base)
    if not target_ids:
        base = {"applied": False, "reason": "no_target_games", **scope_report}
        return worlds_df, dict(base), dict(base), dict(base), dict(base), dict(base)
    if "game_id" not in worlds_df.columns:
        base = {
            "applied": False,
            "reason": "missing_world_game_id",
            **scope_report,
        }
        return worlds_df, dict(base), dict(base), dict(base), dict(base), dict(base)

    world_game_ids = pd.to_numeric(worlds_df["game_id"], errors="coerce").astype("Int64")
    target_mask = world_game_ids.isin(target_ids).to_numpy(dtype=bool)
    target_rows_before = int(np.count_nonzero(target_mask))
    scope_report["target_row_count_before"] = target_rows_before
    if target_rows_before <= 0:
        base = {"applied": False, "reason": "no_target_world_rows", **scope_report}
        return worlds_df, dict(base), dict(base), dict(base), dict(base), dict(base)

    untouched_worlds = worlds_df.loc[~target_mask].copy()
    target_worlds = worlds_df.loc[target_mask].reset_index(drop=True).copy()
    target_features = _filter_to_target_games(features_df, target_ids)
    pre_calibration_pts_anchor = (
        _build_pre_calibration_points_anchor(
            target_worlds,
            label="_postprocess_target_world_slice_for_game_scoped_merge/pre_calibration_points_anchor",
        )
        if bool(apply_team_implied_points_reconcile)
        else pd.DataFrame(
            columns=["game_id", "team_id", "player_id", "pts_pre_calibration_mean"]
        )
    )

    if bool(apply_props_uplift):
        target_worlds, props_uplift_report = _apply_props_uplift_calibration_to_worlds(
            target_worlds,
            features_df=target_features,
            scope=str(props_uplift_scope),
            confidence_weighted=bool(props_uplift_confidence_weighted),
        )
    else:
        props_uplift_report = {"applied": False, "reason": "disabled", **scope_report}
    target_worlds, propless_tail_report = _apply_propless_tail_calibration_to_worlds(
        target_worlds,
        features_df=target_features,
        enabled=bool(apply_propless_tail_calibration),
        min_minutes_mean=float(propless_tail_min_minutes_mean),
        min_dk_mean=float(propless_tail_min_dk_mean),
        tail_boost=float(propless_tail_boost),
        max_tail_scale=float(propless_tail_max_scale),
        target_game_ids=None,
    )
    target_worlds, mid_minutes_tail_report = _apply_mid_minutes_tail_calibration_to_worlds(
        target_worlds,
        enabled=bool(apply_mid_minutes_tail_calibration),
        min_minutes=float(mid_minutes_tail_min_minutes),
        max_minutes=float(mid_minutes_tail_max_minutes),
        tail_boost=float(mid_minutes_tail_boost),
        target_game_ids=None,
    )
    target_worlds, team_implied_points_reconcile_report = (
        _apply_team_implied_points_reconcile_to_worlds(
            target_worlds,
            features_df=target_features,
            pre_calibration_pts_anchor=pre_calibration_pts_anchor,
            enabled=bool(apply_team_implied_points_reconcile),
            alpha=float(team_implied_points_reconcile_alpha),
            deadband_points=float(team_implied_points_reconcile_deadband_points),
        )
    )
    target_worlds, world_realism_report = _apply_world_realism_controls_to_worlds(
        target_worlds,
        enabled=bool(apply_world_realism_controls),
        random_seed=int(random_seed),
        low_minutes_tail_damping_enabled=bool(
            world_realism_low_minutes_tail_damping_enabled
        ),
        low_minutes_tail_minutes_threshold=float(world_realism_low_minutes_threshold),
        low_minutes_tail_min_scale=float(world_realism_low_minutes_min_scale),
        outlier_resample_enabled=bool(world_realism_outlier_resample_enabled),
        outlier_resample_max_passes=int(world_realism_outlier_resample_max_passes),
        target_game_ids=None,
    )
    target_worlds, world_contract_repair_report = _repair_world_frame_contract_fields(
        target_worlds
    )
    scope_report["target_row_count_after"] = int(len(target_worlds))

    merged_worlds = _concat_frames_without_pandas_concat(
        [untouched_worlds, target_worlds]
    )
    merged_worlds = _sort_for_stable_write(merged_worlds)

    def _annotate_scope(report: dict[str, Any]) -> dict[str, Any]:
        out_report = dict(report or {})
        out_report["target_game_ids"] = target_ids
        out_report["target_row_count_before"] = int(
            scope_report["target_row_count_before"]
        )
        out_report["target_row_count_after"] = int(scope_report["target_row_count_after"])
        return out_report

    return (
        merged_worlds,
        _annotate_scope(props_uplift_report),
        _annotate_scope(propless_tail_report),
        _annotate_scope(mid_minutes_tail_report),
        _annotate_scope(world_realism_report),
        _annotate_scope(world_contract_repair_report),
    )


@task(name="materialize-unified-run-artifacts", retries=0)
def materialize_unified_run_artifacts_task(
    *,
    game_date: str,
    run_id: str,
    data_root: Path,
    target_game_ids: list[int],
    apply_props_uplift: bool = False,
    props_uplift_scope: str = "all_players",
    props_uplift_confidence_weighted: bool = True,
    apply_propless_tail_calibration: bool = True,
    propless_tail_min_minutes_mean: float = 21.0,
    propless_tail_min_dk_mean: float = 16.0,
    propless_tail_boost: float = 0.14,
    propless_tail_max_scale: float = 1.22,
    apply_mid_minutes_tail_calibration: bool = True,
    mid_minutes_tail_min_minutes: float = 12.0,
    mid_minutes_tail_max_minutes: float = 20.0,
    mid_minutes_tail_boost: float = 0.14,
    apply_team_implied_points_reconcile: bool = False,
    team_implied_points_reconcile_alpha: float = 0.75,
    team_implied_points_reconcile_deadband_points: float = 2.0,
    apply_world_realism_controls: bool = True,
    world_realism_low_minutes_tail_damping_enabled: bool = True,
    world_realism_low_minutes_threshold: float = 12.0,
    world_realism_low_minutes_min_scale: float = 0.55,
    world_realism_outlier_resample_enabled: bool = True,
    world_realism_outlier_resample_max_passes: int = 1,
    apply_team_dk_fpts_correlation_overlay: bool = False,
    team_dk_fpts_correlation_overlay_alpha: float = 0.0,
    team_dk_fpts_correlation_overlay_min_minutes: float = 0.0,
    team_dk_fpts_correlation_overlay_weight_power: float = 1.0,
    random_seed: int = 42,
) -> dict[str, Any]:
    target_ids = _normalize_game_ids(target_game_ids)
    if not target_ids:
        return {"mode": "no_target_games"}
    try:
        logger = get_run_logger()
    except Exception:
        logger = logging.getLogger(__name__)
    stage_start = time.perf_counter()

    features_dir = data_root / "live" / FEATURES_ROOT / game_date
    scores_dir = data_root / "artifacts" / SCORES_ROOT / f"game_date={game_date}"
    worlds_dir = data_root / "artifacts" / WORLDS_ROOT / f"game_date={game_date}"
    projections_dir = data_root / "artifacts" / "projections" / game_date

    merged_features = _merge_parquet_for_target_games(
        current_path=features_dir / f"run={run_id}" / "features.parquet",
        previous_path=_resolve_previous_run_file(
            dataset_dir=features_dir, filename="features.parquet"
        ),
        target_game_ids=target_ids,
    )
    merged_scores = _merge_parquet_for_target_games(
        current_path=scores_dir / f"run={run_id}" / "scores.parquet",
        previous_path=_resolve_previous_run_file(
            dataset_dir=scores_dir, filename="scores.parquet"
        ),
        target_game_ids=target_ids,
    )
    merged_worlds = _merge_parquet_for_target_games(
        current_path=worlds_dir / f"run={run_id}" / "worlds.parquet",
        previous_path=_resolve_previous_run_file(
            dataset_dir=worlds_dir, filename="worlds.parquet"
        ),
        target_game_ids=target_ids,
    )
    merged_final = _merge_parquet_for_target_games(
        current_path=projections_dir / f"run={run_id}" / "projections.parquet",
        previous_path=_resolve_previous_run_file(
            dataset_dir=projections_dir, filename="projections.parquet"
        ),
        target_game_ids=target_ids,
    )
    logger.info(
        "materialize stage complete: merge_inputs elapsed_sec=%.2f "
        "features_rows=%d scores_rows=%d worlds_rows=%d final_rows=%d target_games=%d",
        time.perf_counter() - stage_start,
        int(len(merged_features)),
        int(len(merged_scores)),
        int(len(merged_worlds)),
        int(len(merged_final)),
        int(len(target_ids)),
    )
    stage_start = time.perf_counter()

    # Use the current run's features as expected keys — the same file publish-atomic
    # validates against. Using merged_features (which includes previous-run players)
    # was too permissive: carry-forward players (e.g. a forced-in DNP) would survive
    # the materialize sanitize but fail the publish-atomic key contract check.
    expected_feature_keys = pd.read_parquet(
        features_dir / f"run={run_id}" / "features.parquet",
        columns=["game_id", "team_id", "player_id"],
    )

    merged_scores, score_key_report = _sanitize_frame_to_expected_keys(
        merged_scores,
        expected_keys_df=expected_feature_keys,
        key_cols=("game_id", "team_id", "player_id"),
        label="merged scores",
    )
    _atomic_write_validated_parquet(
        merged_scores,
        scores_dir / f"run={run_id}" / "scores.parquet",
        required_cols=("game_date", "game_id", "team_id", "player_id"),
    )
    logger.info(
        "materialize stage complete: sanitize_scores elapsed_sec=%.2f rows=%d",
        time.perf_counter() - stage_start,
        int(len(merged_scores)),
    )
    stage_start = time.perf_counter()

    merged_worlds, world_key_report = _sanitize_frame_to_expected_keys(
        merged_worlds,
        expected_keys_df=expected_feature_keys,
        key_cols=("game_id", "team_id", "player_id"),
        label="merged worlds",
    )
    (
        merged_worlds,
        props_uplift_report,
        propless_tail_report,
        mid_minutes_tail_report,
        world_realism_report,
        world_contract_repair_report,
    ) = (
        _postprocess_target_world_slice_for_game_scoped_merge(
            worlds_df=merged_worlds,
            features_df=merged_features,
            target_game_ids=target_ids,
            apply_props_uplift=bool(apply_props_uplift),
            props_uplift_scope=str(props_uplift_scope),
            props_uplift_confidence_weighted=bool(props_uplift_confidence_weighted),
            apply_propless_tail_calibration=bool(apply_propless_tail_calibration),
            propless_tail_min_minutes_mean=float(propless_tail_min_minutes_mean),
            propless_tail_min_dk_mean=float(propless_tail_min_dk_mean),
            propless_tail_boost=float(propless_tail_boost),
            propless_tail_max_scale=float(propless_tail_max_scale),
            apply_mid_minutes_tail_calibration=bool(apply_mid_minutes_tail_calibration),
            mid_minutes_tail_min_minutes=float(mid_minutes_tail_min_minutes),
            mid_minutes_tail_max_minutes=float(mid_minutes_tail_max_minutes),
            mid_minutes_tail_boost=float(mid_minutes_tail_boost),
            apply_team_implied_points_reconcile=bool(apply_team_implied_points_reconcile),
            team_implied_points_reconcile_alpha=float(team_implied_points_reconcile_alpha),
            team_implied_points_reconcile_deadband_points=float(
                team_implied_points_reconcile_deadband_points
            ),
            apply_world_realism_controls=bool(apply_world_realism_controls),
            world_realism_low_minutes_tail_damping_enabled=bool(
                world_realism_low_minutes_tail_damping_enabled
            ),
            world_realism_low_minutes_threshold=float(world_realism_low_minutes_threshold),
            world_realism_low_minutes_min_scale=float(world_realism_low_minutes_min_scale),
            world_realism_outlier_resample_enabled=bool(world_realism_outlier_resample_enabled),
            world_realism_outlier_resample_max_passes=int(
                world_realism_outlier_resample_max_passes
            ),
            random_seed=int(random_seed),
        )
    )
    merged_worlds, world_key_report_postprocess = _sanitize_frame_to_expected_keys(
        merged_worlds,
        expected_keys_df=expected_feature_keys,
        key_cols=("game_id", "team_id", "player_id"),
        label="merged worlds postprocess",
    )
    if (
        world_key_report_postprocess["dropped_null_key_rows"] > 0
        or world_key_report_postprocess["dropped_unexpected_key_rows"] > 0
    ):
        logger.warning(
            "materialize postprocess world key sanitize dropped rows: "
            "null_key_rows=%d unexpected_key_rows=%d",
            int(world_key_report_postprocess["dropped_null_key_rows"]),
            int(world_key_report_postprocess["dropped_unexpected_key_rows"]),
        )
    # Safety pass: sanitize may operate on very large mixed-type world frames.
    # Keep repair as the final transform before write/summary so any
    # post-sanitize numeric spikes are clipped out deterministically.
    merged_worlds, world_contract_repair_report_post_sanitize = _repair_world_frame_contract_fields(
        merged_worlds
    )
    if bool(world_contract_repair_report_post_sanitize.get("applied")):
        logger.warning(
            "Applied post-sanitize world contract repair safety pass in materialize: %s",
            world_contract_repair_report_post_sanitize,
        )
    merged_worlds, team_dk_fpts_correlation_overlay_report = (
        _apply_team_dk_fpts_correlation_overlay_to_worlds(
            merged_worlds,
            enabled=bool(apply_team_dk_fpts_correlation_overlay),
            alpha=float(team_dk_fpts_correlation_overlay_alpha),
            min_minutes=float(team_dk_fpts_correlation_overlay_min_minutes),
            weight_power=float(team_dk_fpts_correlation_overlay_weight_power),
            target_game_ids=target_ids,
        )
    )
    if bool(team_dk_fpts_correlation_overlay_report.get("applied")):
        logger.info(
            "Applied team dk_fpts correlation overlay in materialize: %s",
            team_dk_fpts_correlation_overlay_report,
        )
    _atomic_write_validated_parquet(
        merged_worlds,
        worlds_dir / f"run={run_id}" / "worlds.parquet",
        required_cols=("world_idx", "game_id", "team_id", "player_id"),
    )
    logger.info(
        "materialize stage complete: sanitize_worlds elapsed_sec=%.2f rows=%d",
        time.perf_counter() - stage_start,
        int(len(merged_worlds)),
    )
    stage_start = time.perf_counter()

    target_worlds_for_projection = _filter_to_target_games(merged_worlds, target_ids)
    if target_worlds_for_projection.empty:
        raise RuntimeError(
            "materialize target-world projection recompute received empty target slice "
            f"for target_game_ids={target_ids}"
        )
    worlds_runtime = _gtv2_worlds_runtime()
    target_world_projections = worlds_runtime.summarize_worlds_to_projections(
        target_worlds_for_projection,
        sim_profile="game_transformer_v2",
    )
    target_world_projections = _normalize_gtv2_projection_surface_semantics(
        target_world_projections
    )
    merged_world_projections = _merge_parquet_for_target_games(
        current_path=worlds_dir / f"run={run_id}" / "projections.parquet",
        previous_path=_resolve_previous_run_file(
            dataset_dir=worlds_dir, filename="projections.parquet"
        ),
        target_game_ids=target_ids,
    )
    projection_join_keys = ["game_id", "team_id", "player_id"]
    target_projection_value_cols = [
        col
        for col in target_world_projections.columns
        if col not in {"game_date", "game_id", "team_id", "player_id"}
    ]
    merged_world_projections = _left_overlay_from_source_by_keys(
        merged_world_projections,
        source_df=target_world_projections.loc[
            :, projection_join_keys + target_projection_value_cols
        ],
        key_cols=projection_join_keys,
        value_cols=target_projection_value_cols,
        label="materialize_unified_run_artifacts_task/world_projection_target_overlay",
    )
    merged_world_projections = _normalize_gtv2_projection_surface_semantics(
        merged_world_projections
    )
    merged_world_projections, world_projection_key_report = _sanitize_frame_to_expected_keys(
        merged_world_projections,
        expected_keys_df=expected_feature_keys,
        key_cols=("game_id", "team_id", "player_id"),
        label="merged world projections",
    )
    _atomic_write_validated_parquet(
        merged_world_projections,
        worlds_dir / f"run={run_id}" / "projections.parquet",
        required_cols=("game_date", "game_id", "team_id", "player_id"),
    )
    logger.info(
        "materialize stage complete: world_projections elapsed_sec=%.2f "
        "target_world_rows=%d target_projection_rows=%d merged_projection_rows=%d",
        time.perf_counter() - stage_start,
        int(len(target_worlds_for_projection)),
        int(len(target_world_projections)),
        int(len(merged_world_projections)),
    )
    stage_start = time.perf_counter()

    projection_value_cols = [
        col
        for col in merged_world_projections.columns
        if col not in {"game_date", "game_id", "team_id", "player_id"}
    ]
    merged_final = _left_overlay_from_source_by_keys(
        merged_final,
        source_df=merged_world_projections.loc[
            :, projection_join_keys + projection_value_cols
        ],
        key_cols=projection_join_keys,
        value_cols=projection_value_cols,
        label="materialize_unified_run_artifacts_task/world_projection_overlay",
    )
    merged_final = _normalize_gtv2_projection_surface_semantics(merged_final)
    if "dk_fpts_mean" in merged_final.columns and "salary" in merged_final.columns:
        salary = pd.to_numeric(merged_final["salary"], errors="coerce")
        merged_final["value"] = (
            pd.to_numeric(merged_final["dk_fpts_mean"], errors="coerce")
            .div(salary.where(salary > 0))
            .mul(1000)
            .round(2)
        )

    merged_final, final_projection_key_report = _sanitize_frame_to_expected_keys(
        merged_final,
        expected_keys_df=expected_feature_keys,
        key_cols=("game_id", "team_id", "player_id"),
        label="merged unified projections",
    )
    _atomic_write_validated_parquet(
        merged_final,
        projections_dir / f"run={run_id}" / "projections.parquet",
        required_cols=("game_date", "game_id", "team_id", "player_id"),
    )
    logger.info(
        "materialize stage complete: finalize_overlay elapsed_sec=%.2f final_rows=%d",
        time.perf_counter() - stage_start,
        int(len(merged_final)),
    )
    stage_start = time.perf_counter()

    world_summary_path = worlds_dir / f"run={run_id}" / "world_contracts_summary.json"
    world_summary_payload = {
        "contract_checks": _summarize_world_contracts_from_frame(merged_worlds),
        "merged_from_previous": True,
        "key_sanitization": {
            "scores": score_key_report,
            "worlds": world_key_report,
            "worlds_postprocess": world_key_report_postprocess,
            "world_projections": world_projection_key_report,
            "unified_projections": final_projection_key_report,
        },
        "target_game_ids": target_ids,
        "rows": int(len(merged_worlds)),
        "projection_rows": int(len(merged_world_projections)),
        "props_uplift_calibration": props_uplift_report,
        "propless_tail_calibration": propless_tail_report,
        "mid_minutes_tail_calibration": mid_minutes_tail_report,
        "team_dk_fpts_correlation_overlay": team_dk_fpts_correlation_overlay_report,
        "world_realism_controls": world_realism_report,
        "world_contract_field_repair": world_contract_repair_report,
        "world_contract_field_repair_post_sanitize": (
            world_contract_repair_report_post_sanitize
        ),
        "created_at": _utc_now_iso(),
    }
    world_summary_path.write_text(
        json.dumps(world_summary_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    logger.info(
        "materialize stage complete: write_summary elapsed_sec=%.2f summary_path=%s",
        time.perf_counter() - stage_start,
        str(world_summary_path),
    )
    return {
        "mode": "merged",
        "target_game_ids": target_ids,
        "features_rows": int(len(merged_features)),
        "scores_rows": int(len(merged_scores)),
        "world_rows": int(len(merged_worlds)),
        "projection_rows": int(len(merged_final)),
        "world_contract_summary_path": str(world_summary_path),
    }


def _validate_publishable_run_artifacts(
    *,
    game_date: str,
    run_id: str,
    data_root: Path,
) -> dict[str, Any]:
    stage_reports: dict[str, Any] = {}
    single_file_targets = {
        "features_minutes_v1": (
            data_root
            / "live"
            / "features_minutes_v1"
            / game_date
            / f"run={run_id}"
            / "features.parquet",
            ("game_id", "team_id", "player_id"),
        ),
        "features_gtv2_v1": (
            data_root / "live" / FEATURES_ROOT / game_date / f"run={run_id}" / "features.parquet",
            ("game_id", "team_id", "player_id"),
        ),
        "scores_gtv2": (
            data_root / "artifacts" / SCORES_ROOT / f"game_date={game_date}" / f"run={run_id}" / "scores.parquet",
            ("game_date", "game_id", "team_id", "player_id"),
        ),
        "worlds_gtv2/worlds": (
            data_root / "artifacts" / WORLDS_ROOT / f"game_date={game_date}" / f"run={run_id}" / "worlds.parquet",
            ("world_idx",),
        ),
        "worlds_gtv2/projections": (
            data_root / "artifacts" / WORLDS_ROOT / f"game_date={game_date}" / f"run={run_id}" / "projections.parquet",
            ("game_date", "game_id", "team_id", "player_id"),
        ),
        "unified_projections": (
            data_root / "artifacts" / "projections" / game_date / f"run={run_id}" / "projections.parquet",
            ("game_date", "game_id", "team_id", "player_id"),
        ),
    }
    for stage, (path, required_cols) in single_file_targets.items():
        if not path.exists():
            raise RuntimeError(f"publish validation missing required parquet: {path}")
        stage_reports[stage] = _stream_validate_parquet(
            path,
            required_cols=required_cols,
        )

    feature_keys = pd.read_parquet(
        single_file_targets["features_gtv2_v1"][0],
        columns=["game_id", "team_id", "player_id"],
    )
    stage_reports["semantic_key_contracts"] = {
        "scores_gtv2": _validate_parquet_key_contract(
            single_file_targets["scores_gtv2"][0],
            expected_keys_df=feature_keys,
            key_cols=("game_id", "team_id", "player_id"),
            label="scores_gtv2",
        ),
        "worlds_gtv2/worlds": _validate_parquet_key_contract(
            single_file_targets["worlds_gtv2/worlds"][0],
            expected_keys_df=feature_keys,
            key_cols=("game_id", "team_id", "player_id"),
            label="worlds_gtv2/worlds",
        ),
        "worlds_gtv2/projections": _validate_parquet_key_contract(
            single_file_targets["worlds_gtv2/projections"][0],
            expected_keys_df=feature_keys,
            key_cols=("game_id", "team_id", "player_id"),
            label="worlds_gtv2/projections",
        ),
        "unified_projections": _validate_parquet_key_contract(
            single_file_targets["unified_projections"][0],
            expected_keys_df=feature_keys,
            key_cols=("game_id", "team_id", "player_id"),
            label="unified_projections",
        ),
    }

    ownership_dir = (
        data_root / "silver" / "ownership_predictions" / game_date / f"run={run_id}"
    )
    ownership_files = sorted(
        path for path in ownership_dir.glob("*.parquet") if path.is_file()
    )
    if not ownership_files:
        raise RuntimeError(
            f"publish validation found no ownership parquet files under {ownership_dir}"
        )
    ownership_reports = []
    for path in ownership_files:
        ownership_reports.append(
            _stream_validate_parquet(path, required_cols=("player_id",))
        )
    stage_reports["ownership_predictions"] = {
        "dir": str(ownership_dir),
        "file_count": int(len(ownership_reports)),
        "files": ownership_reports,
    }
    return stage_reports


@task(name="v3-postflight", retries=0)
def postflight_gate_task(
    *,
    projections_path: Path,
    parity_manifest_path: Path,
    world_contract_summary_path: Path,
) -> dict[str, Any]:
    return run_postflight_gate(
        projections_path=projections_path,
        parity_manifest_path=parity_manifest_path,
        world_contract_summary_path=world_contract_summary_path,
        key_columns=("game_id", "team_id", "player_id"),
        min_rows=20,
    )


@task(name="publish-atomic", retries=0)
def publish_atomic_task(
    *,
    game_date: str,
    run_id: str,
    manifest_path: Path,
    data_root: Path,
) -> dict[str, str]:
    manifest_payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    freshness_summary = dict(
        manifest_payload.get("source_freshness", {}).get("summary", {})
    )
    validation_report = _validate_publishable_run_artifacts(
        game_date=game_date,
        run_id=run_id,
        data_root=data_root,
    )
    manifest_payload["publish_validation"] = {
        "validated_at": _utc_now_iso(),
        "stages": validation_report,
    }
    control_plane.atomic_write_json(Path(manifest_path), manifest_payload)
    pointers: dict[str, str] = {}
    targets = {
        "features_minutes_v1": data_root / "live" / "features_minutes_v1" / game_date,
        "features_gtv2_v1": data_root / "live" / FEATURES_ROOT / game_date,
        "scores_gtv2": data_root / "artifacts" / SCORES_ROOT / f"game_date={game_date}",
        "worlds_gtv2": data_root / "artifacts" / WORLDS_ROOT / f"game_date={game_date}",
        "ownership_predictions": data_root / "silver" / "ownership_predictions" / game_date,
        "unified_projections": data_root / "artifacts" / "projections" / game_date,
    }
    for stage, dataset_dir in targets.items():
        pointer = control_plane.promote_run_pointer(
            dataset_dir=dataset_dir,
            run_id=run_id,
            manifest_path=manifest_path,
            extra={
                "entrypoint": "prefect-v3",
                "stage": stage,
                "as_of_ts": manifest_payload.get("as_of_ts"),
                "source_freshness_summary": freshness_summary,
            },
        )
        pointers[stage] = str(pointer)
    return pointers


@flow(name="nba-live-pipeline-v3", log_prints=True)
def nba_live_pipeline_v3_flow(
    *,
    game_date: str | None = None,
    manual_target_game_ids: list[int] | None = None,
    sim_worlds: int = 25000,
    run_id_override: str | None = None,
    promote_pointers: bool = True,
    placeholder_mode: bool = True,
    replay_mode: bool = False,
    as_of_ts_override: str | None = None,
    gtv2_bundle_dir: str | None = None,
    gtv2_device: str | None = None,
    gtv2_inference_backend: str = "auto",
    gtv2_triton_endpoint: str | None = None,
    gtv2_triton_model_name: str | None = None,
    gtv2_triton_model_version: str | None = None,
    gtv2_triton_timeout_seconds: float | None = None,
    gtv2_triton_healthcheck_timeout_seconds: float | None = None,
    gtv2_world_chunk_size: int = 5000,
    gtv2_active_temperature: float = 1.0,
    gtv2_seed: int = 42,
    gtv2_strict_world_contracts: bool = False,
    gtv2_flow_scale_clip_override: float | None = None,
    gtv2_make_model_mode: str = "beta_binomial_all",
    gtv2_make_model_use_learned_efficiency: bool = True,
    gtv2_apply_props_uplift: bool = False,
    gtv2_props_uplift_scope: str = "all_players",
    gtv2_props_uplift_confidence_weighted: bool = True,
    gtv2_apply_propless_tail_calibration: bool = True,
    gtv2_propless_tail_min_minutes_mean: float = 21.0,
    gtv2_propless_tail_min_dk_mean: float = 16.0,
    gtv2_propless_tail_boost: float = 0.14,
    gtv2_propless_tail_max_scale: float = 1.22,
    gtv2_apply_mid_minutes_tail_calibration: bool = True,
    gtv2_mid_minutes_tail_min_minutes: float = 12.0,
    gtv2_mid_minutes_tail_max_minutes: float = 20.0,
    gtv2_mid_minutes_tail_boost: float = 0.14,
    gtv2_apply_team_implied_points_reconcile: bool = False,
    gtv2_team_implied_points_reconcile_alpha: float = 0.75,
    gtv2_team_implied_points_reconcile_deadband_points: float = 2.0,
    gtv2_apply_world_realism_controls: bool = True,
    gtv2_world_realism_low_minutes_tail_damping_enabled: bool = True,
    gtv2_world_realism_low_minutes_threshold: float = 12.0,
    gtv2_world_realism_low_minutes_min_scale: float = 0.55,
    gtv2_world_realism_outlier_resample_enabled: bool = True,
    gtv2_world_realism_outlier_resample_max_passes: int = 1,
    input_max_age_minutes: float = 360.0,
    require_action_props: bool = True,
    allow_rotowire_props_fallback: bool = True,
) -> dict[str, str]:
    logger = get_run_logger()
    data_root = paths.get_data_root()
    resolved_game_date = _resolve_game_date(game_date)
    run_id = run_id_override or control_plane.canonical_run_id()
    minutes_selector_path = model_selectors.active_minutes_selector_path(
        data_root=data_root,
        project_root=PROJECT_ROOT,
    )
    rates_selector_path = model_selectors.active_rates_selector_path(
        data_root=data_root,
        project_root=PROJECT_ROOT,
    )
    ownership_selector_path = model_selectors.active_ownership_selector_path(
        data_root=data_root,
        project_root=PROJECT_ROOT,
    )
    ownership_cfg = ownership_selector.load_ownership_selector(
        config_path=ownership_selector_path,
        data_root=data_root,
        project_root=PROJECT_ROOT,
    )
    gtv2_inference_current_path = model_selectors.active_gtv2_selector_path(
        data_root=data_root,
        project_root=PROJECT_ROOT,
    )
    gtv2_current_cfg = _load_gtv2_inference_current_config(
        config_path=gtv2_inference_current_path
    )
    gtv2_inference_current_hash = _stable_digest(gtv2_current_cfg)
    bundle_dir = _resolve_bundle_dir(
        data_root=data_root,
        gtv2_bundle_dir=gtv2_bundle_dir,
        current_config_payload=gtv2_current_cfg,
    )
    bundle_hash = _bundle_artifact_hash(bundle_dir)
    inference_server_cfg = _load_gtv2_inference_server_config()
    resolved_inference_backend = _resolve_gtv2_inference_backend(
        requested=gtv2_inference_backend,
        config_payload=inference_server_cfg,
    )
    resolved_triton_cfg = _resolve_triton_endpoint_config(
        config_payload=inference_server_cfg,
        endpoint_override=gtv2_triton_endpoint,
        model_name_override=gtv2_triton_model_name,
        model_version_override=gtv2_triton_model_version,
        timeout_seconds_override=gtv2_triton_timeout_seconds,
    )
    resolved_triton_healthcheck_timeout_seconds = float(
        gtv2_triton_healthcheck_timeout_seconds
        if gtv2_triton_healthcheck_timeout_seconds is not None
        else inference_server_cfg.get("healthcheck_timeout_seconds", 3.0)
    )
    if resolved_inference_backend == "triton":
        logger.info(
            "Using triton inference backend: endpoint=%s model=%s version=%s timeout=%.1fs",
            resolved_triton_cfg.endpoint,
            resolved_triton_cfg.model_name,
            resolved_triton_cfg.model_version,
            resolved_triton_cfg.timeout_seconds,
        )
    resolved_allow_rotowire_props_fallback = bool(allow_rotowire_props_fallback)
    allow_nonpublishing_replay = os.environ.get(
        "PROJECTIONS_ALLOW_NONPUBLISHING_REPLAY", ""
    ).strip().lower() in {"1", "true", "yes"}
    if replay_mode and (not promote_pointers) and (not allow_nonpublishing_replay):
        raise RuntimeError(
            "Refusing replay_mode run with promote_pointers=False: this computes artifacts "
            "but does not publish atomic pointers. Set PROJECTIONS_ALLOW_NONPUBLISHING_REPLAY=1 "
            "only for intentional dry-run debugging."
        )
    rotation_cfg_path = PROJECT_ROOT / "config" / "rotation_set_minutes_live.json"
    if rotation_cfg_path.exists():
        try:
            rotation_cfg = json.loads(rotation_cfg_path.read_text(encoding="utf-8"))
            if "allow_rotowire_props_fallback" in rotation_cfg:
                resolved_allow_rotowire_props_fallback = bool(
                    rotation_cfg.get("allow_rotowire_props_fallback")
                )
        except Exception:
            pass

    # Resolve flow_scale_clip override: CLI param > env var > None
    resolved_flow_scale_clip_override = gtv2_flow_scale_clip_override
    if resolved_flow_scale_clip_override is None:
        env_clip = os.environ.get("GT_FLOW_SCALE_CLIP")
        if env_clip is not None:
            resolved_flow_scale_clip_override = float(env_clip)
            logger.warning(
                "GT_FLOW_SCALE_CLIP env var set to %.2f — using experimental scale_clip override",
                resolved_flow_scale_clip_override,
            )

    # Append suffix to run_id if using experimental scale_clip (avoids overwriting production)
    if resolved_flow_scale_clip_override is not None:
        clip_suffix = f"_clip{resolved_flow_scale_clip_override:.1f}".replace(".", "p")
        run_id = run_id + clip_suffix
        logger.info("Experimental run_id with clip suffix: %s", run_id)

    # Runtime stamp for reproducibility and incident triage.
    enforce_clean_tree()
    enforce_prod_sanity()
    runtime_config_paths: dict[str, Path] = {
        "minutes_current_run": minutes_selector_path,
        "rates_current_run": rates_selector_path,
        "ownership_current_run": ownership_selector_path,
        "gtv2_bundle_dir": bundle_dir,
    }
    inference_server_cfg_path = PROJECT_ROOT / "config" / "gtv2_inference_server.json"
    if inference_server_cfg_path.exists():
        runtime_config_paths["gtv2_inference_server"] = inference_server_cfg_path
    log_runtime_stamp(
        entrypoint="prefect:nba-live-pipeline-v3",
        config_paths=runtime_config_paths,
        project_root=PROJECT_ROOT,
        logger=logger,
    )

    v3_run_dir = (
        data_root
        / "artifacts"
        / "runs"
        / "nba_live_v3"
        / f"game_date={resolved_game_date}"
        / f"run={run_id}"
    )
    v3_run_dir.mkdir(parents=True, exist_ok=True)

    # Fail-closed storage guard to prevent runaway writes after incidents.
    try:
        from projections.storage_retention.config import load_storage_retention_policy
        from projections.storage_retention.guard import ensure_storage_headroom_or_raise

        storage_policy = load_storage_retention_policy()
        guard_payload = ensure_storage_headroom_or_raise(
            hot_root=data_root,
            guard_policy=storage_policy.guard,
        )
        (v3_run_dir / "storage_guard.json").write_text(
            json.dumps(guard_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        if (
            storage_policy.reduced_persistence.enabled
            and storage_policy.reduced_persistence.gtv2_max_worlds is not None
            and int(sim_worlds) > int(storage_policy.reduced_persistence.gtv2_max_worlds)
        ):
            capped = int(storage_policy.reduced_persistence.gtv2_max_worlds)
            logger.warning(
                "Reduced persistence is enabled: capping sim_worlds from %s to %s",
                int(sim_worlds),
                capped,
            )
            sim_worlds = capped
    except Exception as exc:  # noqa: BLE001
        # Storage guard should hard-stop only when thresholds are violated.
        # If policy loading/writes fail, keep the run going but record the incident.
        (v3_run_dir / "storage_guard_error.txt").write_text(
            f"{type(exc).__name__}: {exc}\n",
            encoding="utf-8",
        )

    try:
        writer_lock = writer_guard.PipelineWriterLock(data_root=data_root, run_id=run_id)
        writer_lock.__enter__()
    except RuntimeError as exc:
        if "Another writer is active" not in str(exc):
            raise
        duplicate_report = {
            "checked_at": _utc_now_iso(),
            "status": "skipped_due_to_active_writer",
            "reason": str(exc),
            "run_id": run_id,
            "game_date": resolved_game_date,
        }
        (v3_run_dir / "duplicate_run_report.json").write_text(
            json.dumps(duplicate_report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return {
            "run_id": run_id,
            "game_date": resolved_game_date,
            "manifest_path": "",
            "features_path": "",
            "projections_path": "",
            "bundle_dir": str(bundle_dir),
            "pointer_count": "0",
            "rerun_mode": "",
            "rerun_reason": "",
            "publish_status": "skipped_active_writer",
        }

    try:
        os.environ["PROJECTIONS_SKIP_POINTER_WRITES"] = "1"

        scrape_marker = scrape_core_inputs_task(
            game_date=resolved_game_date,
            data_root=data_root,
            placeholder_mode=bool(placeholder_mode),
            require_action_props=bool(require_action_props),
            allow_rotowire_props_fallback=bool(resolved_allow_rotowire_props_fallback),
            replay_mode=bool(replay_mode),
        )

        as_of_ts = str(as_of_ts_override) if as_of_ts_override else _utc_now_iso()
        frozen_checklist: dict[str, Any] = {}
        previous_manifest_payload = _load_promoted_manifest_payload(
            data_root=data_root,
            game_date=resolved_game_date,
        )
        bounded_wait_report: dict[str, Any] = {
            "performed": False,
            "reason": "placeholder_mode" if placeholder_mode else "not_needed",
        }
        if not placeholder_mode:
            frozen_checklist = _build_feature_input_checklist(
                game_date=resolved_game_date,
                run_as_of_ts=as_of_ts,
                data_root=data_root,
                allow_priors_fallback=True,
                allow_rotowire_props_fallback=bool(
                    resolved_allow_rotowire_props_fallback
                ),
                require_action_props=bool(require_action_props),
            )
            report_window = dict(
                frozen_checklist.get("freshness_gates", {}).get("report_window", {})
            )
            bounded_wait_report = {
                "performed": False,
                "reason": "not_needed",
                "report_window": report_window,
                "attempts": 0,
                "timed_out": False,
            }
            wait_allowed, wait_skip_reason = _resolve_report_window_wait_policy(
                as_of_ts_override=as_of_ts_override,
                replay_mode=bool(replay_mode),
                manual_target_game_ids=manual_target_game_ids,
            )
            if (
                bool(report_window.get("active"))
                and bool(report_window.get("needs_wait"))
                and wait_allowed
            ):
                bounded_wait_report["performed"] = True
                bounded_wait_report["reason"] = "report_window_wait"
                deadline = time.monotonic() + float(
                    report_window.get(
                        "wait_timeout_seconds", _REPORT_WINDOW_WAIT_TIMEOUT_SECONDS
                    )
                )
                while (
                    bool(report_window.get("needs_wait"))
                    and time.monotonic() < deadline
                ):
                    sleep_s = min(
                        float(
                            report_window.get(
                                "wait_interval_seconds",
                                _REPORT_WINDOW_WAIT_INTERVAL_SECONDS,
                            )
                        ),
                        max(1.0, deadline - time.monotonic()),
                    )
                    logger.warning(
                        "Freshness wait active for %s; sleeping %.0fs before re-scrape. blocking_games=%s",
                        report_window.get("label"),
                        sleep_s,
                        [
                            game.get("game_id")
                            for game in report_window.get("blocking_games", [])
                        ],
                    )
                    time.sleep(sleep_s)
                    bounded_wait_report["attempts"] = (
                        int(bounded_wait_report.get("attempts", 0)) + 1
                    )
                    scrape_marker = scrape_core_inputs_task(
                        game_date=resolved_game_date,
                        data_root=data_root,
                        placeholder_mode=False,
                        require_action_props=bool(require_action_props),
                        allow_rotowire_props_fallback=bool(
                            resolved_allow_rotowire_props_fallback
                        ),
                        replay_mode=False,
                    )
                    as_of_ts = _utc_now_iso()
                    frozen_checklist = _build_feature_input_checklist(
                        game_date=resolved_game_date,
                        run_as_of_ts=as_of_ts,
                        data_root=data_root,
                        allow_priors_fallback=True,
                        allow_rotowire_props_fallback=bool(
                            resolved_allow_rotowire_props_fallback
                        ),
                        require_action_props=bool(require_action_props),
                    )
                    report_window = dict(
                        frozen_checklist.get("freshness_gates", {}).get(
                            "report_window", {}
                        )
                    )
                    bounded_wait_report["report_window"] = report_window
                if bool(report_window.get("needs_wait")):
                    bounded_wait_report["timed_out"] = True
                    bounded_wait_report["reason"] = "report_window_wait_timed_out"
                    logger.warning(
                        "Freshness wait timed out for %s; continuing with explicit diagnostics.",
                        report_window.get("label"),
                    )
            elif bool(report_window.get("needs_wait")) and not wait_allowed:
                bounded_wait_report["reason"] = f"wait_skipped_{wait_skip_reason}"
                logger.warning(
                    "Skipping freshness wait because %s is active; report_window=%s manual_target_game_ids=%s",
                    wait_skip_reason,
                    report_window.get("label"),
                    _normalize_game_ids(manual_target_game_ids),
                )

        input_change_set = _build_input_change_set(
            game_date=resolved_game_date,
            current_source_freshness=dict(frozen_checklist.get("source_freshness", {})),
            previous_manifest_payload=previous_manifest_payload,
        )
        rerun_plan = _build_rerun_plan(
            game_date=resolved_game_date,
            input_change_set=input_change_set,
            current_source_freshness=dict(frozen_checklist.get("source_freshness", {})),
            previous_manifest_payload=previous_manifest_payload,
            current_bundle_hash=bundle_hash,
            current_minutes_selector_path=minutes_selector_path,
            current_rates_selector_path=rates_selector_path,
            current_ownership_selector_path=ownership_selector_path,
            current_gtv2_inference_config_path=gtv2_inference_current_path,
            current_gtv2_inference_config_hash=gtv2_inference_current_hash,
            manual_target_game_ids=manual_target_game_ids,
        )
        target_game_ids = _normalize_game_ids(rerun_plan.get("target_game_ids"))

        manifest_path = freeze_run_inputs_task(
            game_date=resolved_game_date,
            run_id=run_id,
            as_of_ts=as_of_ts,
            bundle_dir=bundle_dir,
            data_root=data_root,
            ownership_selector_path=ownership_selector_path,
            gtv2_inference_current_path=gtv2_inference_current_path,
            gtv2_inference_current_hash=gtv2_inference_current_hash,
            source_freshness=dict(frozen_checklist.get("source_freshness", {})),
            freshness_gates=dict(frozen_checklist.get("freshness_gates", {})),
            bounded_wait=bounded_wait_report,
            input_change_set={**input_change_set, "rerun_plan": rerun_plan},
        )
        (v3_run_dir / "input_change_set.json").write_text(
            json.dumps(
                {**input_change_set, "rerun_plan": rerun_plan}, indent=2, sort_keys=True
            ),
            encoding="utf-8",
        )
        control_plane.atomic_update_json(manifest_path, {"rerun_plan": rerun_plan})

        if rerun_plan.get("mode") == "skip":
            skip_report = {
                "mode": "skip",
                "reason": rerun_plan.get("reason"),
                "target_game_ids": target_game_ids,
                "previous_run_id": input_change_set.get("previous_run_id"),
            }
            (v3_run_dir / "skip_report.json").write_text(
                json.dumps(skip_report, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            return {
                "run_id": run_id,
                "game_date": resolved_game_date,
                "manifest_path": str(manifest_path),
                "features_path": "",
                "projections_path": "",
                "bundle_dir": str(bundle_dir),
                "pointer_count": "0",
                "rerun_mode": str(rerun_plan.get("mode")),
                "rerun_reason": str(rerun_plan.get("reason")),
                "publish_status": "not_requested",
            }

        features_path = build_features_gtv2_live_task(
            game_date=resolved_game_date,
            run_id=run_id,
            run_as_of_ts=as_of_ts,
            data_root=data_root,
            bundle_dir=bundle_dir,
            manifest_path=manifest_path,
            placeholder_mode=bool(placeholder_mode),
            require_action_props=bool(require_action_props),
            allow_rotowire_props_fallback=bool(resolved_allow_rotowire_props_fallback),
            target_game_ids=(
                None if rerun_plan.get("mode") == "full_slate" else target_game_ids
            ),
        )
        runtime_manifest_path = features_path.parent / "feature_runtime_manifest.json"
        runtime_payload = json.loads(
            runtime_manifest_path.read_text(encoding="utf-8")
        )
        parity_manifest_path = Path(
            str(runtime_payload.get("parity_manifest_path", ""))
        ).expanduser()
        if not parity_manifest_path.exists():
            raise RuntimeError(
                "runtime parity manifest path missing after feature build: "
                f"{parity_manifest_path}"
            )

        score_run_dir = (
            data_root
            / "artifacts"
            / SCORES_ROOT
            / f"game_date={resolved_game_date}"
            / f"run={run_id}"
        )
        worlds_run_dir = (
            data_root
            / "artifacts"
            / WORLDS_ROOT
            / f"game_date={resolved_game_date}"
            / f"run={run_id}"
        )
        projections_run_dir = (
            data_root
            / "artifacts"
            / "projections"
            / resolved_game_date
            / f"run={run_id}"
        )

        preflight_report = preflight_gate_task(
            as_of_ts=as_of_ts,
            manifest_path=manifest_path,
            required_inputs={
                "core_inputs_marker": scrape_marker,
                "features": features_path,
            },
            run_dirs=[score_run_dir, worlds_run_dir, projections_run_dir],
            features_path=features_path,
            parity_manifest_path=parity_manifest_path,
            runtime_manifest_path=runtime_manifest_path,
            input_max_age_minutes=float(input_max_age_minutes),
            bundle_config_path=(bundle_dir / "config.json"),
        )
        (v3_run_dir / "preflight_report.json").write_text(
            json.dumps(preflight_report, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        scores_path = score_gtv2_live_task(
            game_date=resolved_game_date,
            run_id=run_id,
            features_path=features_path,
            bundle_dir=bundle_dir,
            data_root=data_root,
            placeholder_mode=bool(placeholder_mode),
            inference_backend=str(resolved_inference_backend),
            triton_endpoint=str(resolved_triton_cfg.endpoint),
            triton_model_name=str(resolved_triton_cfg.model_name),
            triton_model_version=resolved_triton_cfg.model_version,
            triton_timeout_seconds=float(resolved_triton_cfg.timeout_seconds),
            triton_healthcheck_timeout_seconds=float(
                resolved_triton_healthcheck_timeout_seconds
            ),
            gtv2_device=gtv2_device,
            random_seed=int(gtv2_seed),
        )

        worlds_inference_backend = (
            "local"
            if bool(gtv2_current_cfg.get("sparse_hybrid_enabled", False))
            else str(resolved_inference_backend)
        )
        worlds_outputs = generate_worlds_gtv2_live_task(
            game_date=resolved_game_date,
            run_id=run_id,
            run_as_of_ts=as_of_ts,
            features_path=features_path,
            scores_path=scores_path,
            bundle_dir=bundle_dir,
            data_root=data_root,
            sim_worlds=int(sim_worlds),
            placeholder_mode=bool(placeholder_mode),
            inference_backend=str(worlds_inference_backend),
            triton_endpoint=str(resolved_triton_cfg.endpoint),
            triton_model_name=str(resolved_triton_cfg.model_name),
            triton_model_version=resolved_triton_cfg.model_version,
            triton_timeout_seconds=float(resolved_triton_cfg.timeout_seconds),
            triton_healthcheck_timeout_seconds=float(
                resolved_triton_healthcheck_timeout_seconds
            ),
            gtv2_device=gtv2_device,
            world_chunk_size=int(gtv2_world_chunk_size),
            active_temperature=float(gtv2_active_temperature),
            random_seed=int(gtv2_seed),
            strict_world_contracts=bool(gtv2_strict_world_contracts),
            flow_scale_clip_override=resolved_flow_scale_clip_override,
            make_model_mode=str(gtv2_make_model_mode),
            make_model_use_learned_efficiency=bool(
                gtv2_make_model_use_learned_efficiency
            ),
            apply_props_uplift=bool(gtv2_apply_props_uplift),
            props_uplift_scope=str(gtv2_props_uplift_scope),
            props_uplift_confidence_weighted=bool(
                gtv2_props_uplift_confidence_weighted
            ),
            apply_propless_tail_calibration=bool(
                gtv2_apply_propless_tail_calibration
                and rerun_plan.get("mode") == "full_slate"
            ),
            propless_tail_min_minutes_mean=float(
                gtv2_propless_tail_min_minutes_mean
            ),
            propless_tail_min_dk_mean=float(gtv2_propless_tail_min_dk_mean),
            propless_tail_boost=float(gtv2_propless_tail_boost),
            propless_tail_max_scale=float(gtv2_propless_tail_max_scale),
            apply_mid_minutes_tail_calibration=bool(
                gtv2_apply_mid_minutes_tail_calibration
                and rerun_plan.get("mode") == "full_slate"
            ),
            mid_minutes_tail_min_minutes=float(gtv2_mid_minutes_tail_min_minutes),
            mid_minutes_tail_max_minutes=float(gtv2_mid_minutes_tail_max_minutes),
            mid_minutes_tail_boost=float(gtv2_mid_minutes_tail_boost),
            apply_team_implied_points_reconcile=bool(
                gtv2_apply_team_implied_points_reconcile
                and rerun_plan.get("mode") == "full_slate"
            ),
            team_implied_points_reconcile_alpha=float(
                gtv2_team_implied_points_reconcile_alpha
            ),
            team_implied_points_reconcile_deadband_points=float(
                gtv2_team_implied_points_reconcile_deadband_points
            ),
            apply_world_realism_controls=bool(gtv2_apply_world_realism_controls),
            world_realism_low_minutes_tail_damping_enabled=bool(
                gtv2_world_realism_low_minutes_tail_damping_enabled
            ),
            world_realism_low_minutes_threshold=float(
                gtv2_world_realism_low_minutes_threshold
            ),
            world_realism_low_minutes_min_scale=float(
                gtv2_world_realism_low_minutes_min_scale
            ),
            world_realism_outlier_resample_enabled=bool(
                gtv2_world_realism_outlier_resample_enabled
            ),
            world_realism_outlier_resample_max_passes=int(
                gtv2_world_realism_outlier_resample_max_passes
            ),
            apply_team_dk_fpts_correlation_overlay=bool(
                gtv2_current_cfg.get("team_dk_fpts_correlation_overlay_enabled", False)
            ),
            team_dk_fpts_correlation_overlay_alpha=float(
                gtv2_current_cfg.get("team_dk_fpts_correlation_overlay_alpha", 0.0)
            ),
            team_dk_fpts_correlation_overlay_min_minutes=float(
                gtv2_current_cfg.get("team_dk_fpts_correlation_overlay_min_minutes", 0.0)
            ),
            team_dk_fpts_correlation_overlay_weight_power=float(
                gtv2_current_cfg.get("team_dk_fpts_correlation_overlay_weight_power", 1.0)
            ),
            promotion_hybrid_enabled=bool(
                gtv2_current_cfg.get("promotion_hybrid_enabled", False)
            ),
            promotion_expert_run_dir=(
                str(gtv2_current_cfg.get("promotion_expert_run_dir") or "").strip()
                or None
            ),
            promotion_prior_minutes_max=float(
                gtv2_current_cfg.get("promotion_prior_minutes_max", 12.0)
            ),
            promotion_hist_start_rate_max=float(
                gtv2_current_cfg.get("promotion_hist_start_rate_max", 0.20)
            ),
            promotion_blend_mode=str(
                gtv2_current_cfg.get("promotion_blend_mode", "uplift_only")
            ),
            promotion_force_active_candidates=bool(
                gtv2_current_cfg.get("promotion_force_active_candidates", False)
            ),
            sparse_hybrid_enabled=bool(
                gtv2_current_cfg.get("sparse_hybrid_enabled", False)
            ),
            sparse_expert_run_dir=(
                str(gtv2_current_cfg.get("sparse_expert_run_dir") or "").strip()
                or None
            ),
            sparse_prior_minutes_max=float(
                gtv2_current_cfg.get("sparse_prior_minutes_max", 12.0)
            ),
            sparse_prior_play_prob_max=float(
                gtv2_current_cfg.get("sparse_prior_play_prob_max", 0.50)
            ),
            sparse_blend_mode=str(
                gtv2_current_cfg.get("sparse_blend_mode", "uplift_only")
            ),
            sparse_blend_alpha=float(
                gtv2_current_cfg.get("sparse_blend_alpha", 1.0)
            ),
            sparse_require_no_props=bool(
                gtv2_current_cfg.get("sparse_require_no_props", False)
            ),
            sparse_gate_artifact=(
                str(gtv2_current_cfg.get("sparse_gate_artifact") or "").strip()
                or None
            ),
            tree_rate_bundle_dir=(
                str(gtv2_current_cfg.get("tree_rate_bundle_dir") or "").strip()
                or None
            ),
            tree_rate_predictions_csv=(
                str(gtv2_current_cfg.get("tree_rate_predictions_csv") or "").strip()
                or None
            ),
            tree_rate_blend_alpha=float(
                gtv2_current_cfg.get("tree_rate_blend_alpha", 0.0)
            ),
            tree_rate_oreb_share_override_enabled=bool(
                gtv2_current_cfg.get("tree_rate_oreb_share_override_enabled", False)
            ),
            minutes_uncertainty_enabled=bool(
                gtv2_current_cfg.get("minutes_uncertainty_enabled", False)
            ),
            minutes_uncertainty_mode=str(
                gtv2_current_cfg.get("minutes_uncertainty_mode", "gaussian")
            ),
            minutes_uncertainty_gaussian_scale=float(
                gtv2_current_cfg.get("minutes_uncertainty_gaussian_scale", 1.0)
            ),
            minutes_uncertainty_min_sigma=float(
                gtv2_current_cfg.get("minutes_uncertainty_min_sigma", 0.75)
            ),
            minutes_uncertainty_max_sigma=float(
                gtv2_current_cfg.get("minutes_uncertainty_max_sigma", 6.0)
            ),
            minutes_uncertainty_fallback_sigma=float(
                gtv2_current_cfg.get("minutes_uncertainty_fallback_sigma", 1.5)
            ),
            minutes_uncertainty_use_hurdle_sigma=bool(
                gtv2_current_cfg.get("minutes_uncertainty_use_hurdle_sigma", True)
            ),
            minutes_uncertainty_use_prior_std=bool(
                gtv2_current_cfg.get("minutes_uncertainty_use_prior_std", True)
            ),
            minutes_uncertainty_preserve_top_k_per_team=int(
                gtv2_current_cfg.get("minutes_uncertainty_preserve_top_k_per_team", 3)
            ),
            minutes_uncertainty_full_sigma_at_minutes_or_below=float(
                gtv2_current_cfg.get("minutes_uncertainty_full_sigma_at_minutes_or_below", 24.0)
            ),
            minutes_uncertainty_zero_sigma_at_minutes_or_above=float(
                gtv2_current_cfg.get("minutes_uncertainty_zero_sigma_at_minutes_or_above", 32.0)
            ),
            minutes_uncertainty_apply_minutes_taper=bool(
                gtv2_current_cfg.get("minutes_uncertainty_apply_minutes_taper", True)
            ),
            minutes_uncertainty_dirichlet_base_concentration=float(
                gtv2_current_cfg.get("minutes_uncertainty_dirichlet_base_concentration", 24.0)
            ),
            minutes_uncertainty_lookup_artifact=(
                str(gtv2_current_cfg.get("minutes_uncertainty_lookup_artifact") or "").strip()
                or None
            ),
            minutes_uncertainty_empirical_blend_alpha=float(
                gtv2_current_cfg.get("minutes_uncertainty_empirical_blend_alpha", 1.0)
            ),
        )

        ownership_dir = score_ownership_task(
            game_date=resolved_game_date,
            run_id=run_id,
            data_root=data_root,
            placeholder_mode=bool(placeholder_mode),
            source=ownership_cfg.source,
            model_family=ownership_cfg.model_family,
            model_run=ownership_cfg.model_run,
            gtv2_features_path=ownership_cfg.gtv2_features_path,
            fallback_source=ownership_cfg.fallback_source,
            fallback_model_family=ownership_cfg.fallback_model_family,
            fallback_model_run=ownership_cfg.fallback_model_run,
            fallback_gtv2_features_path=ownership_cfg.fallback_gtv2_features_path,
        )
        if ownership_dir.exists():
            control_plane.copy_manifest_to_dir(manifest_path, ownership_dir)
        control_plane.atomic_update_json(
            manifest_path,
            {
                "ownership": {
                    "selector_path": str(ownership_selector_path),
                    "selected_source": ownership_cfg.source,
                    "selected_model_family": ownership_cfg.model_family,
                    "selected_model_run": ownership_cfg.model_run,
                    "fallback_source": ownership_cfg.fallback_source,
                    "fallback_model_family": ownership_cfg.fallback_model_family,
                    "fallback_model_run": ownership_cfg.fallback_model_run,
                    "placeholder_mode": bool(placeholder_mode),
                }
            },
        )

        projections_dir = finalize_projections_live_task(
            game_date=resolved_game_date,
            run_id=run_id,
            worlds_projections_path=Path(worlds_outputs["projections_path"]),
            data_root=data_root,
            placeholder_mode=bool(placeholder_mode),
            target_game_ids=(
                None if rerun_plan.get("mode") == "full_slate" else target_game_ids
            ),
        )
        if rerun_plan.get("mode") == "game_scoped":
            unified_report = materialize_unified_run_artifacts_task(
                game_date=resolved_game_date,
                run_id=run_id,
                data_root=data_root,
                target_game_ids=target_game_ids,
                apply_props_uplift=bool(gtv2_apply_props_uplift),
                props_uplift_scope=str(gtv2_props_uplift_scope),
                props_uplift_confidence_weighted=bool(
                    gtv2_props_uplift_confidence_weighted
                ),
                apply_propless_tail_calibration=bool(
                    gtv2_apply_propless_tail_calibration
                ),
                propless_tail_min_minutes_mean=float(
                    gtv2_propless_tail_min_minutes_mean
                ),
                propless_tail_min_dk_mean=float(gtv2_propless_tail_min_dk_mean),
                propless_tail_boost=float(gtv2_propless_tail_boost),
                propless_tail_max_scale=float(gtv2_propless_tail_max_scale),
                apply_mid_minutes_tail_calibration=bool(
                    gtv2_apply_mid_minutes_tail_calibration
                ),
                mid_minutes_tail_min_minutes=float(
                    gtv2_mid_minutes_tail_min_minutes
                ),
                mid_minutes_tail_max_minutes=float(
                    gtv2_mid_minutes_tail_max_minutes
                ),
                mid_minutes_tail_boost=float(gtv2_mid_minutes_tail_boost),
                apply_team_implied_points_reconcile=bool(
                    gtv2_apply_team_implied_points_reconcile
                ),
                team_implied_points_reconcile_alpha=float(
                    gtv2_team_implied_points_reconcile_alpha
                ),
                team_implied_points_reconcile_deadband_points=float(
                    gtv2_team_implied_points_reconcile_deadband_points
                ),
                apply_world_realism_controls=bool(gtv2_apply_world_realism_controls),
                world_realism_low_minutes_tail_damping_enabled=bool(
                    gtv2_world_realism_low_minutes_tail_damping_enabled
                ),
                world_realism_low_minutes_threshold=float(
                    gtv2_world_realism_low_minutes_threshold
                ),
                world_realism_low_minutes_min_scale=float(
                    gtv2_world_realism_low_minutes_min_scale
                ),
                world_realism_outlier_resample_enabled=bool(
                    gtv2_world_realism_outlier_resample_enabled
                ),
                world_realism_outlier_resample_max_passes=int(
                    gtv2_world_realism_outlier_resample_max_passes
                ),
                apply_team_dk_fpts_correlation_overlay=bool(
                    gtv2_current_cfg.get(
                        "team_dk_fpts_correlation_overlay_enabled", False
                    )
                ),
                team_dk_fpts_correlation_overlay_alpha=float(
                    gtv2_current_cfg.get(
                        "team_dk_fpts_correlation_overlay_alpha", 0.0
                    )
                ),
                team_dk_fpts_correlation_overlay_min_minutes=float(
                    gtv2_current_cfg.get(
                        "team_dk_fpts_correlation_overlay_min_minutes", 0.0
                    )
                ),
                team_dk_fpts_correlation_overlay_weight_power=float(
                    gtv2_current_cfg.get(
                        "team_dk_fpts_correlation_overlay_weight_power", 1.0
                    )
                ),
                random_seed=int(gtv2_seed),
            )
            (v3_run_dir / "unified_artifacts_report.json").write_text(
                json.dumps(unified_report, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            worlds_outputs["world_contract_summary_path"] = str(
                unified_report["world_contract_summary_path"]
            )

        # Validate output schema contract against parity manifest before publish.
        _ = load_parity_manifest(parity_manifest_path)
        postflight_report = postflight_gate_task(
            projections_path=projections_dir / "projections.parquet",
            parity_manifest_path=parity_manifest_path,
            world_contract_summary_path=Path(
                worlds_outputs["world_contract_summary_path"]
            ),
        )
        (v3_run_dir / "postflight_report.json").write_text(
            json.dumps(postflight_report, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        pointer_payload: dict[str, str] = {}
        publish_status = "not_requested" if not promote_pointers else "pending"
        if promote_pointers:
            if placeholder_mode:
                stale_publish_report: dict[str, Any] = {
                    "checked_at": _utc_now_iso(),
                    "stale": False,
                    "stale_games": [],
                    "skipped": "placeholder_mode",
                }
            else:
                publish_checklist = _build_feature_input_checklist(
                    game_date=resolved_game_date,
                    run_as_of_ts=_utc_now_iso(),
                    data_root=data_root,
                    allow_priors_fallback=True,
                    allow_rotowire_props_fallback=bool(
                        resolved_allow_rotowire_props_fallback
                    ),
                    require_action_props=bool(require_action_props),
                )
                stale_publish_report = _detect_stale_authoritative_inputs(
                    frozen_source_freshness=dict(
                        frozen_checklist.get("source_freshness", {})
                    ),
                    current_source_freshness=dict(
                        publish_checklist.get("source_freshness", {})
                    ),
                    as_of_ts=str(
                        publish_checklist.get("run_as_of_ts") or _utc_now_iso()
                    ),
                )
                control_plane.atomic_update_json(
                    manifest_path,
                    {
                        "publish_precheck": stale_publish_report,
                    },
                )
                (v3_run_dir / "stale_publish_report.json").write_text(
                    json.dumps(stale_publish_report, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                if bool(stale_publish_report.get("stale")):
                    raise RuntimeError(
                        "stale publish blocked: newer authoritative injuries/lineups arrived after freeze. "
                        f"See {v3_run_dir / 'stale_publish_report.json'}"
                    )
            superseded_report = _build_publish_superseded_report(
                run_id=run_id,
                manifest_path=manifest_path,
                dataset_dir=data_root / "artifacts" / "projections" / resolved_game_date,
            )
            control_plane.atomic_update_json(
                manifest_path,
                {
                    "publish_superseded": superseded_report,
                },
            )
            (v3_run_dir / "publish_superseded_report.json").write_text(
                json.dumps(superseded_report, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            if bool(superseded_report.get("superseded")):
                logger.warning(
                    "Skipping publish for %s because a newer run is already published. reason=%s current_run_id=%s",
                    run_id,
                    superseded_report.get("reason"),
                    dict(superseded_report.get("current_pointer") or {}).get("run_id"),
                )
                publish_status = "superseded"
            else:
                pointer_payload = publish_atomic_task(
                    game_date=resolved_game_date,
                    run_id=run_id,
                    manifest_path=manifest_path,
                    data_root=data_root,
                )
                publish_status = "published"
            if placeholder_mode:
                (v3_run_dir / "stale_publish_report.json").write_text(
                    json.dumps(stale_publish_report, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
        return {
            "run_id": run_id,
            "game_date": resolved_game_date,
            "manifest_path": str(manifest_path),
            "features_path": str(features_path),
            "projections_path": str(projections_dir / "projections.parquet"),
            "bundle_dir": str(bundle_dir),
            "inference_backend": str(resolved_inference_backend),
            "triton_endpoint": (
                str(resolved_triton_cfg.endpoint)
                if resolved_inference_backend == "triton"
                else ""
            ),
            "triton_model_name": (
                str(resolved_triton_cfg.model_name)
                if resolved_inference_backend == "triton"
                else ""
            ),
            "pointer_count": str(len(pointer_payload)),
            "rerun_mode": str(rerun_plan.get("mode")),
            "rerun_reason": str(rerun_plan.get("reason")),
            "publish_status": publish_status,
        }
    finally:
        writer_lock.__exit__(None, None, None)
