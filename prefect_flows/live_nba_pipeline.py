"""Canonical Prefect flow for the NBA live pipeline.

This is intended to be the *only* production orchestrator entrypoint.
Each stage is an explicit Prefect task and the flow enforces:
- single-writer guard for pointer publishes
- run_id-scoped outputs
- run manifest written at start
"""

from __future__ import annotations

# ruff: noqa: E402

import faulthandler
import os
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

_STABILITY_ENV_DEFAULTS: dict[str, str] = {
    # Reduce thread/BLAS/Arrow concurrency to avoid occasional hard crashes (SIGSEGV / GPF)
    # observed under heavy parquet I/O + ML workloads.
    "ARROW_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def _apply_stability_env(env: dict[str, str]) -> None:
    for k, v in _STABILITY_ENV_DEFAULTS.items():
        env.setdefault(k, v)


_apply_stability_env(os.environ)  # must run before importing pandas/pyarrow
faulthandler.enable(all_threads=True)

import pandas as pd
from prefect import flow, get_run_logger, task
from zoneinfo import ZoneInfo

from projections import paths
from projections.builders import build_shared_features
from projections.cli import build_minutes_live as live_minutes_builder
from projections.features.action_props import (
    attach_action_props_features,
    load_action_props_feature_snapshots_for_date,
)
from projections import model_selectors
from projections.minutes_v1.datasets import KEY_COLUMNS, deduplicate_latest
from projections.pipeline import control_plane, writer_guard
from projections.pipeline import effective_inputs, health
from projections.pipeline import minutes_models
from projections.runtime_stamp import (
    log_runtime_stamp,
    enforce_clean_tree,
    enforce_prod_sanity,
)


import shutil

PROJECT_ROOT = paths.get_project_root()

# Default absolute path for uv (works under systemd where PATH may not include ~/.local/bin)
_DEFAULT_UV_PATH = Path("/home/daniel/.local/bin/uv")


def _uv_bin() -> str:
    """Resolve path to uv binary.

    Resolution order:
    1. UV_BIN environment variable (explicit override)
    2. Absolute path at /home/daniel/.local/bin/uv (works under systemd)
    3. shutil.which("uv") (finds uv in PATH)
    4. Raises FileNotFoundError with instructions
    """
    # 1. Check env override
    env_uv = os.environ.get("UV_BIN")
    if env_uv:
        if Path(env_uv).exists():
            return env_uv
        raise FileNotFoundError(f"UV_BIN={env_uv} specified but file does not exist")

    # 2. Check default absolute path
    if _DEFAULT_UV_PATH.exists():
        return str(_DEFAULT_UV_PATH)

    # 3. Fall back to shutil.which
    which_uv = shutil.which("uv")
    if which_uv:
        return which_uv

    # 4. Not found - raise with instructions
    raise FileNotFoundError(
        "Could not find 'uv' executable. Either:\n"
        "  1. Set UV_BIN=/path/to/uv environment variable, or\n"
        "  2. Install uv and ensure it's in PATH, or\n"
        f"  3. Install uv to {_DEFAULT_UV_PATH}"
    )


def _run_python_module(
    module: str,
    args: list[str],
    *,
    data_root: Path,
    timeout_s: int,
) -> None:
    env = os.environ.copy()
    env["PROJECTIONS_DATA_ROOT"] = str(data_root)
    _apply_stability_env(env)
    # Resolve uv path (handles systemd PATH issues)
    uv_path = _uv_bin()
    cmd = [uv_path, "run", "python", "-m", module, *args]
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
    if result.returncode != 0:
        raise RuntimeError(f"{module} failed with exit_code={result.returncode}")


def _resolve_season_month(game_date: str) -> tuple[int, int]:
    """Resolve NBA season year and calendar month for a game date.

    NBA seasons span October through June. The season year is the year in which
    the season *starts* (e.g., 2025-26 season -> season=2025).
    - Oct-Dec (month >= 8): season = calendar year
    - Jan-Jul (month < 8): season = calendar year - 1
    """
    ts = pd.Timestamp(game_date)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    season = int(ts.year) if ts.month >= 8 else int(ts.year) - 1
    return season, int(ts.month)


@task(name="scrape", retries=2, retry_delay_seconds=60)
def scrape_task(*, game_date: str, data_root: Path) -> None:
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
        "projections.cli.live_pipeline", args, data_root=data_root, timeout_s=900
    )


@task(name="dk-salaries", retries=2, retry_delay_seconds=120)
def dk_salaries_task(*, game_date: str, data_root: Path) -> None:
    base = data_root / "gold" / "dk_salaries" / "site=dk" / f"game_date={game_date}"
    if base.exists() and any(base.glob("draft_group_id=*/salaries.parquet")):
        return
    _run_python_module(
        "scripts.dk.run_daily_salaries",
        ["--game-date", game_date],
        data_root=data_root,
        timeout_s=600,
    )


@task(name="scrape-props", retries=1, retry_delay_seconds=60)
def scrape_props_task(*, game_date: str, data_root: Path) -> None:
    """Scrape player props for sidecar consumers and Action feature inputs.

    This is a non-blocking task - props failures should not fail the pipeline.
    Props may be unavailable early in the day or during off-season.
    """
    logger = get_run_logger()
    try:
        _run_python_module(
            "projections.cli.scrape_props",
            ["scrape", "--date", game_date],
            data_root=data_root,
            timeout_s=300,
        )
    except Exception as e:
        logger.warning(f"Props scrape failed (continuing): {e}")
    try:
        _run_python_module(
            "scrapers.action_network.props_backfill",
            ["--start-date", game_date, "--end-date", game_date, "--workers", "40"],
            data_root=data_root,
            timeout_s=1200,
        )
    except Exception as e:
        logger.warning(f"Action Network props scrape failed (continuing): {e}")


@task(name="build-minutes-features", retries=1, retry_delay_seconds=60)
def build_minutes_features_task(
    *, game_date: str, run_id: str, run_as_of_ts: str, data_root: Path
) -> Path:
    logger = get_run_logger()
    target_day = pd.Timestamp(game_date).normalize()
    run_ts = pd.Timestamp(run_as_of_ts)
    if run_ts.tzinfo is None:
        run_ts = run_ts.tz_localize("UTC")
    else:
        run_ts = run_ts.tz_convert("UTC")
    season, _ = _resolve_season_month(game_date)

    warnings: list[str] = []
    labels_source_df, label_source = live_minutes_builder._load_label_sources(
        data_root=data_root,
        season_value=season,
        override_path=None,
        warnings=warnings,
    )
    history_labels = live_minutes_builder._load_label_history(
        labels_source_df,
        target_day=target_day,
        history_days=None,
        run_as_of_ts=run_ts,
        label_source=label_source,
    )

    roster_default = data_root / "silver" / "roster_nightly" / f"season={season}"
    roster_df = live_minutes_builder._load_table(roster_default, None)
    roster_slice, roster_source_day, _ = live_minutes_builder._select_roster_slice(
        roster_df,
        target_day=target_day,
        run_as_of_ts=run_ts,
        fallback_days=0,
        max_age_hours=18,
    )
    if roster_slice.empty:
        raise RuntimeError(
            f"Roster snapshot does not include rows for {target_day.date()} and no fallback within 0 day(s) was found."
        )
    if roster_source_day is not None and roster_source_day != target_day:
        warnings.append(
            f"Roster fallback: using snapshot from {roster_source_day.date()} for {target_day.date()} (max 0d)."
        )
    live_labels = live_minutes_builder._build_live_labels(
        roster_slice,
        target_day=target_day,
        season_label=live_minutes_builder._season_label(season),
    )

    labels = pd.concat([history_labels, live_labels], ignore_index=True, sort=False)
    live_game_ids = (
        pd.to_numeric(live_labels["game_id"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    if not live_game_ids:
        raise RuntimeError(
            "No live game_ids available after building roster-based labels."
        )

    schedule_default = data_root / "silver" / "schedule" / f"season={season}"
    schedule_df = live_minutes_builder._load_table(schedule_default, None)

    result = build_shared_features(
        data_root=data_root,
        season=season,
        target_day=target_day.date(),
        as_of_ts=run_ts,
        labels=labels,
        schedule=schedule_df,
        game_ids=live_game_ids,
        backfill_mode=False,
    )
    if warnings:
        for warning in warnings:
            logger.warning(warning)
    if result.warnings:
        for warning in result.warnings:
            logger.warning(warning)

    features = result.features.copy()
    features["game_date"] = pd.to_datetime(features["game_date"]).dt.normalize()
    live_slice = features.loc[features["game_date"] == target_day].copy()
    live_slice = live_slice[
        pd.to_numeric(live_slice["game_id"], errors="coerce")
        .astype("Int64")
        .isin(live_game_ids)
    ].copy()
    if {"feature_as_of_ts", *KEY_COLUMNS}.issubset(live_slice.columns):
        live_slice = deduplicate_latest(
            live_slice,
            key_cols=KEY_COLUMNS,
            order_cols=["feature_as_of_ts"],
        )
        live_slice = live_slice.drop_duplicates(subset=list(KEY_COLUMNS), keep="last")

    action_props_dir = data_root / "bronze" / "action_network" / "props"
    action_props_snapshot_rows = 0
    action_props_matched_rows = 0
    action_props_snapshots = pd.DataFrame()
    if action_props_dir.exists():
        try:
            # Load snapshots for both game_date and game_date+1 (UTC/ET day boundary).
            # AN stores games by UTC start_time, so a 7pm ET game on Feb-19 appears
            # as a Feb-20 UTC file. Loading next-day covers this offset.
            snapshot_frames: list[pd.DataFrame] = []
            for offset in (0, 1):
                snap = load_action_props_feature_snapshots_for_date(
                    props_dir=action_props_dir,
                    game_date=target_day + pd.Timedelta(days=offset),
                )
                if not snap.empty:
                    snapshot_frames.append(snap)
            action_props_snapshots = (
                pd.concat(snapshot_frames, ignore_index=True)
                if snapshot_frames
                else pd.DataFrame()
            )
            action_props_snapshot_rows = int(len(action_props_snapshots))
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Action props load failed: {exc}")

    live_slice = attach_action_props_features(
        live_slice,
        action_props_snapshots,
        strict_asof=True,
        as_of_col="feature_as_of_ts",
        tip_col="tip_ts",
        game_date_offsets=(0, -1),
        clamp_late_asof_to_game_date=True,
    )
    if "an_has_any_props" in live_slice.columns:
        action_props_matched_rows = int(
            pd.to_numeric(live_slice["an_has_any_props"], errors="coerce")
            .fillna(0.0)
            .gt(0.0)
            .sum()
        )
    logger.info(
        "[minutes-features] Action props: snapshots=%s matched_rows=%s total_rows=%s",
        action_props_snapshot_rows,
        action_props_matched_rows,
        len(live_slice),
    )

    run_dir = data_root / "live" / "features_minutes_v1" / game_date / f"run={run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "features.parquet"
    live_slice.to_parquet(out_path, index=False)
    health.require_file(out_path, label="minutes features.parquet")
    df = pd.read_parquet(out_path)
    health.require_row_count(df, min_rows=20, max_rows=5000, label="minutes features")
    health.require_columns(
        df, required=["player_id", "game_id", "team_id"], label="minutes features"
    )
    health.require_non_null(
        df, cols=["player_id", "game_id", "team_id"], label="minutes features"
    )
    health.require_game_date(df, game_date=game_date, label="minutes features")
    health.require_freshness(
        df,
        ts_cols=["feature_as_of_ts", "odds_as_of_ts", "injuries_as_of_ts", "as_of_ts"],
        reference_ts=run_as_of_ts,
        max_age_minutes=240.0,
        label="minutes features",
    )
    return out_path


def _is_rmh_enabled() -> tuple[bool, dict]:
    """Check if RMH scoring is enabled via config/rmh_current_run.json."""
    rmh_config_path = PROJECT_ROOT / "config" / "rmh_current_run.json"
    if not rmh_config_path.exists():
        return False, {}
    try:
        config = json.loads(rmh_config_path.read_text(encoding="utf-8"))
        return bool(config.get("enabled", False)), config
    except Exception:
        return False, {}


def _is_rotation_set_enabled() -> bool:
    """Check if rotation_set scoring is enabled via config/rotation_set_minutes_live.json."""
    rotset_config_path = PROJECT_ROOT / "config" / "rotation_set_minutes_live.json"
    if not rotset_config_path.exists():
        return False
    try:
        config = json.loads(rotset_config_path.read_text(encoding="utf-8"))
        return bool(config.get("enabled", False))
    except Exception:
        return False


@task(name="score-minutes", retries=1, retry_delay_seconds=120)
def score_minutes_task(
    *,
    game_date: str,
    run_id: str,
    reconcile_mode: str,
    data_root: Path,
    minutes_bundle_dir: str | None = None,
    rotshare_quantiles_mode: str | None = None,
    rotshare_n_worlds: int | None = None,
    rotshare_concentration: float | None = None,
    rotshare_seed: int | None = None,
    rotshare_min_active_players: int | None = None,
    rotshare_mc_center: str | None = None,
) -> Path:
    """Score minutes using RMH v1, rotation_set overlay, or legacy scorer.

    Scorer selection:
    1. If config/rmh_current_run.json has enabled=true AND PROJECTIONS_ALLOW_RMH_PRIMARY=1:
       - Uses score_minutes_rmh_v1 for RMH-based minutes prediction
    2. Else if config/rotation_set_minutes_live.json has enabled=true:
       - Uses score_minutes_rotation_set_v1 which builds rotation features + predicts
    3. Otherwise (legacy mode):
       - Uses score_minutes_v1 (baseline minutes_v1 only, no rotation overlay)
    """
    logger = get_run_logger()

    # Check if RMH is enabled
    rmh_enabled, rmh_config = _is_rmh_enabled()
    allow_rmh_primary = str(os.environ.get("PROJECTIONS_ALLOW_RMH_PRIMARY", "")).strip().lower() in {
        "1",
        "true",
        "yes",
    }
    if rmh_enabled and not allow_rmh_primary:
        logger.warning(
            "[score-minutes] RMH config enabled=true but PROJECTIONS_ALLOW_RMH_PRIMARY not set; "
            "checking rotation_set config next"
        )
        rmh_enabled = False

    # Check if rotation_set is enabled
    rotation_set_enabled = _is_rotation_set_enabled()

    if rmh_enabled:
        logger.info(
            f"[score-minutes] SELECTED: RMH v1 scorer "
            f"(bundle={rmh_config.get('bundle_dir', 'N/A')}, "
            f"threshold={rmh_config.get('in_rotation_threshold_min', 5.0)})"
        )
        args = [
            "--date",
            game_date,
            "--run-id",
            run_id,
            "--data-root",
            str(data_root),
        ]
        _run_python_module(
            "projections.cli.score_minutes_rmh_v1",
            args,
            data_root=data_root,
            timeout_s=1200,
        )
    elif rotation_set_enabled:
        logger.info("[score-minutes] SELECTED: rotation_set_v1 scorer (rotation_set enabled=true)")
        args = [
            "--date",
            game_date,
            "--run-id",
            run_id,
            "--reconcile-team-minutes",
            reconcile_mode,
            "--minutes-output",
            "conditional",
        ]
        if minutes_bundle_dir:
            args.extend(["--bundle-dir", minutes_bundle_dir])
        if rotshare_quantiles_mode:
            args.extend(["--rotshare-quantiles-mode", rotshare_quantiles_mode])
        if rotshare_n_worlds is not None:
            args.extend(["--rotshare-n-worlds", str(int(rotshare_n_worlds))])
        if rotshare_concentration is not None:
            args.extend(["--rotshare-concentration", str(float(rotshare_concentration))])
        if rotshare_seed is not None:
            args.extend(["--rotshare-seed", str(int(rotshare_seed))])
        if rotshare_min_active_players is not None:
            args.extend(
                ["--rotshare-min-active-players", str(int(rotshare_min_active_players))]
            )
        if rotshare_mc_center:
            args.extend(["--rotshare-mc-center", rotshare_mc_center])
        _run_python_module(
            "projections.cli.score_minutes_rotation_set_v1",
            args,
            data_root=data_root,
            timeout_s=1200,
        )
    else:
        logger.info("[score-minutes] SELECTED: legacy scorer (rotation_set enabled=false)")
        args = [
            "--date",
            game_date,
            "--run-id",
            run_id,
            "--mode",
            "live",
            "--reconcile-team-minutes",
            reconcile_mode,
            "--minutes-output",
            "conditional",
        ]
        if minutes_bundle_dir:
            args.extend(["--bundle-dir", minutes_bundle_dir])
        if rotshare_quantiles_mode:
            args.extend(["--rotshare-quantiles-mode", rotshare_quantiles_mode])
        if rotshare_n_worlds is not None:
            args.extend(["--rotshare-n-worlds", str(int(rotshare_n_worlds))])
        if rotshare_concentration is not None:
            args.extend(["--rotshare-concentration", str(float(rotshare_concentration))])
        if rotshare_seed is not None:
            args.extend(["--rotshare-seed", str(int(rotshare_seed))])
        if rotshare_min_active_players is not None:
            args.extend(
                ["--rotshare-min-active-players", str(int(rotshare_min_active_players))]
            )
        if rotshare_mc_center:
            args.extend(["--rotshare-mc-center", rotshare_mc_center])
        _run_python_module(
            "projections.cli.score_minutes_v1",
            args,
            data_root=data_root,
            timeout_s=1200,
        )
    out_path = (
        data_root
        / "artifacts"
        / "minutes_v1"
        / "daily"
        / game_date
        / f"run={run_id}"
        / "minutes.parquet"
    )
    health.require_file(out_path, label="minutes.parquet")
    df = pd.read_parquet(out_path)
    health.require_row_count(df, min_rows=20, max_rows=800, label="minutes")
    health.require_columns(
        df, required=["player_id", "game_id", "team_id"], label="minutes"
    )
    health.require_non_null(
        df, cols=["player_id", "game_id", "team_id"], label="minutes"
    )
    health.require_game_date(df, game_date=game_date, label="minutes")
    return out_path


@task(name="effective-minutes", retries=0)
def effective_minutes_task(
    *, game_date: str, run_id: str, minutes_path: Path, data_root: Path
) -> Path:
    run_dir = minutes_path.parent
    result = effective_inputs.write_effective_minutes_layer(
        game_date=pd.Timestamp(game_date).date(),
        minutes_path=minutes_path,
        out_dir=run_dir,
        data_root=data_root,
        source="gameview",
    )
    health.require_file(
        result.effective_minutes_path, label="effective_minutes.parquet"
    )
    eff = pd.read_parquet(result.effective_minutes_path)
    health.require_row_count(eff, min_rows=20, max_rows=800, label="effective minutes")
    health.require_game_date(eff, game_date=game_date, label="effective minutes")
    health.require_minutes_sanity(eff, label="effective minutes")
    return result.effective_minutes_path


@task(name="publish-minutes-gold")
def publish_minutes_gold_task(
    *,
    game_date: str,
    run_id: str,
    effective_minutes_path: Path,
    effective_summary_path: Path,
    data_root: Path,
) -> Path:
    out_day = data_root / "gold" / "projections_minutes_v1" / f"game_date={game_date}"
    out_run = out_day / f"run={run_id}"
    out_run.mkdir(parents=True, exist_ok=True)
    out_minutes = out_run / "minutes.parquet"
    out_effective = out_run / effective_inputs.EFFECTIVE_MINUTES_FILENAME
    out_summary = out_run / effective_inputs.EFFECTIVE_INPUTS_SUMMARY

    df = pd.read_parquet(effective_minutes_path)
    df.to_parquet(out_minutes, index=False)
    df.to_parquet(out_effective, index=False)
    out_summary.write_text(
        effective_summary_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    return out_day


@task(name="rmh-shadow-minutes", retries=0)
def rmh_shadow_minutes_task(
    *,
    game_date: str,
    run_id: str,
    run_as_of_ts: str,
    features_path: Path,
    data_root: Path,
) -> Path | None:
    """Run RMH_v1.1 inference as a non-blocking shadow branch.

    Safety:
    - Gated by RMH_SHADOW_ENABLED=1 (default off)
    - Never raises: errors are logged with "[rmh-shadow]" and the pipeline continues.
    """
    logger = get_run_logger()
    if os.environ.get("RMH_SHADOW_ENABLED") != "1":
        logger.info("[rmh-shadow] skipped: RMH_SHADOW_ENABLED != 1")
        return None

    artifact_dir_raw = os.environ.get("RMH_ARTIFACT_DIR")
    if not artifact_dir_raw:
        logger.warning(
            "[rmh-shadow] skipped: RMH_ARTIFACT_DIR is not set; skipping RMH shadow inference"
        )
        return None

    model_label = os.environ.get("RMH_MODEL_LABEL") or "RMH v1.1"
    artifact_dir = Path(artifact_dir_raw).expanduser().resolve()

    try:
        if not artifact_dir.exists():
            raise FileNotFoundError(f"RMH_ARTIFACT_DIR does not exist: {artifact_dir}")

        # Import lazily to avoid torch import cost when RMH shadow is disabled.
        from projections.models.rotation_minutes_hurdle_v1 import (
            load_bundle,
            predict_frame,
        )

        bundle = load_bundle(artifact_dir)
        logger.info(
            f"[rmh-shadow] RMH shadow loaded delta_out={bundle.model.delta_out}"
        )
        features = pd.read_parquet(features_path)

        # RMH bundles are trained on a superset of minutes features + rotation priors.
        # If we run inference on minutes features alone, missing priors columns get
        # imputed to 0.0 (standardized to large negatives) and can blow up minutes.
        required_cont = set(bundle.feature_spec.continuous)
        required_cat = set(bundle.feature_spec.categorical)
        required = required_cont | required_cat

        missing_cont_before = sorted(required_cont.difference(features.columns))
        missing_before = sorted(required.difference(features.columns))
        missing_frac_before = len(missing_cont_before) / max(len(required_cont), 1)
        if missing_before:
            logger.info(
                "[rmh-shadow] feature_coverage before_priors "
                f"missing_feature_count={len(missing_cont_before)} "
                f"total_required_continuous={len(required_cont)} "
                f"missing_frac={missing_frac_before:.3f} "
                f"missing_sample={missing_cont_before[:10]}"
            )

        # Join rotation priors if the bundle expects any *_prior_* columns.
        # (Rotation set live features uses the same priors source and fallback logic.)
        if any("_prior_" in col for col in missing_before):
            try:
                from projections.rotation.live_features_v1 import (
                    load_rotation_priors_for_live_inference,
                )
                from projections.rotation.rotation_set_minutes_features_v1 import (
                    apply_odds_missing_flags,
                    fill_numeric_missing_with_zero,
                    join_rotation_priors,
                )

                # Mirror the season boundary used across the repo (Aug–Jul season).
                day = pd.Timestamp(game_date).normalize()
                season = int(day.year) if int(day.month) >= 8 else int(day.year) - 1

                allow_priors_fallback = True
                try:
                    cfg = json.loads(
                        (
                            PROJECT_ROOT / "config/rotation_set_minutes_live.json"
                        ).read_text(encoding="utf-8")
                    )
                    allow_priors_fallback = bool(cfg.get("allow_priors_fallback", True))
                except Exception:
                    # Safe default: allow fallback, but never fail RMH shadow on config read.
                    allow_priors_fallback = True

                priors = load_rotation_priors_for_live_inference(
                    data_root=data_root,
                    season=season,
                    game_date=game_date,
                    game_ids=features["game_id"].astype(str).unique().tolist()
                    if "game_id" in features.columns
                    else [],
                    team_ids=(
                        pd.to_numeric(features["team_id"], errors="coerce")
                        .dropna()
                        .astype(int)
                        .unique()
                        .tolist()
                        if "team_id" in features.columns
                        else []
                    ),
                    player_ids=(
                        pd.to_numeric(features["player_id"], errors="coerce")
                        .dropna()
                        .astype(int)
                        .unique()
                        .tolist()
                        if "player_id" in features.columns
                        else []
                    ),
                    allow_priors_fallback=allow_priors_fallback,
                )
                if priors.warning_message:
                    logger.warning(
                        f"[rmh-shadow] priors warning: {priors.warning_message}"
                    )
                logger.info(
                    "[rmh-shadow] priors loaded "
                    f"used_latest_fallback={priors.used_latest_fallback} "
                    f"teams_found={priors.teams_found} teams_missing={priors.teams_missing} "
                    f"players_found={priors.players_found} players_missing={priors.players_missing}"
                )

                work = features.copy()
                if {"spread_home", "total"}.issubset(work.columns):
                    work = apply_odds_missing_flags(work)
                work = fill_numeric_missing_with_zero(work)
                work = join_rotation_priors(
                    work,
                    team_priors=priors.team_priors,
                    player_priors=priors.player_priors,
                )
                features = work
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[rmh-shadow] failed to join rotation priors: {exc}")

        missing_cont_after = sorted(required_cont.difference(features.columns))
        missing_after = sorted(required.difference(features.columns))
        missing_frac = len(missing_cont_after) / max(len(required_cont), 1)
        if missing_after:
            logger.info(
                "[rmh-shadow] feature_coverage after_priors "
                f"missing_feature_count={len(missing_cont_after)} "
                f"total_required_continuous={len(required_cont)} "
                f"missing_frac={missing_frac:.3f} "
                f"missing_sample={missing_cont_after[:10]}"
            )

        max_missing_frac = float(os.environ.get("RMH_MAX_MISSING_FEATURE_FRAC", "0.25"))
        if missing_frac > max_missing_frac:
            logger.warning(
                "[rmh-shadow] skipped: insufficient feature coverage after priors join "
                f"missing_frac={missing_frac:.3f} > RMH_MAX_MISSING_FEATURE_FRAC={max_missing_frac:.3f}"
            )
            return None

        preds = predict_frame(features, bundle=bundle)

        key_cols = [
            c for c in ("game_id", "player_id", "team_id") if c in preds.columns
        ]
        if not key_cols:
            raise RuntimeError(
                "RMH features frame missing required id columns (game_id/player_id/team_id)."
            )

        out = preds.loc[:, key_cols].copy()
        out["snapshot_ts"] = str(run_as_of_ts)

        # RMH semantics: play_threshold=5.0 means "in rotation" (not "played").
        out["p_in_rotation"] = pd.to_numeric(preds["p_play"], errors="coerce").astype(
            float
        )

        # Unconditional moments/quantiles (v1.1)
        out["minutes_mean_uncond"] = pd.to_numeric(
            preds["minutes_mean_uncond"], errors="coerce"
        ).astype(float)
        for q in ("05", "10", "25", "50", "75", "90", "95"):
            col = f"minutes_q{q}_uncond"
            if col in preds.columns:
                out[col] = pd.to_numeric(preds[col], errors="coerce").astype(float)

        # Dashboard compatibility: map RMH unconditional quantiles to minutes_pXX fields.
        if "minutes_q10_uncond" in out.columns:
            out["minutes_p10"] = out["minutes_q10_uncond"]
        if "minutes_q50_uncond" in out.columns:
            out["minutes_p50"] = out["minutes_q50_uncond"]
        if "minutes_q90_uncond" in out.columns:
            out["minutes_p90"] = out["minutes_q90_uncond"]

        play_threshold = float(bundle.config.get("play_threshold", 5.0))
        model_meta = {
            "play_threshold": play_threshold,
            "quantiles": bundle.config.get(
                "quantiles", [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
            ),
            "rmh_artifact_dir": str(artifact_dir),
            "rmh_schema_hash": bundle.schema_hash,
        }
        out_path = minutes_models.write_minutes_model_outputs(
            data_root=data_root,
            game_date=game_date,
            run_id=run_id,
            model_id=minutes_models.MODEL_ID_RMH_V1_1,
            model_label=model_label,
            df=out,
            run_as_of_ts=run_as_of_ts,
            model_meta=model_meta,
        )
        logger.info(f"[rmh-shadow] wrote {len(out)} rows to {out_path}")
        return out_path
    except Exception as exc:  # noqa: BLE001
        logger.exception(f"[rmh-shadow] failed: {exc}")
        return None


@task(name="build-rates-features", retries=1, retry_delay_seconds=60)
def build_rates_features_task(*, game_date: str, run_id: str, data_root: Path) -> Path:
    args = [
        "--date",
        game_date,
        "--run-id",
        run_id,
        "--data-root",
        str(data_root),
        "--no-strict",
    ]
    _run_python_module(
        "projections.cli.build_rates_features_live",
        args,
        data_root=data_root,
        timeout_s=1200,
    )
    out_path = (
        data_root
        / "live"
        / "features_rates_v1"
        / game_date
        / f"run={run_id}"
        / "features.parquet"
    )
    health.require_file(out_path, label="rates features.parquet")
    df = pd.read_parquet(out_path)
    health.require_row_count(df, min_rows=20, max_rows=5000, label="rates features")
    health.require_columns(
        df, required=["player_id", "game_id", "team_id"], label="rates features"
    )
    health.require_non_null(
        df, cols=["player_id", "game_id", "team_id"], label="rates features"
    )
    health.require_game_date(df, game_date=game_date, label="rates features")
    return out_path


@task(name="score-rates", retries=1, retry_delay_seconds=120)
def score_rates_task(*, game_date: str, run_id: str, data_root: Path) -> Path:
    args = [
        "--date",
        game_date,
        "--run-id",
        run_id,
        "--features-root",
        str(data_root / "live" / "features_rates_v1"),
        "--out-root",
        str(data_root / "gold" / "rates_v1_live"),
        "--no-strict",
    ]
    _run_python_module(
        "projections.cli.score_rates_live", args, data_root=data_root, timeout_s=1200
    )
    out_path = (
        data_root
        / "gold"
        / "rates_v1_live"
        / game_date
        / f"run={run_id}"
        / "rates.parquet"
    )
    health.require_file(out_path, label="rates.parquet")
    df = pd.read_parquet(out_path)
    health.require_row_count(df, min_rows=20, max_rows=800, label="rates")
    health.require_columns(
        df, required=["player_id", "game_id", "team_id"], label="rates"
    )
    health.require_non_null(df, cols=["player_id", "game_id", "team_id"], label="rates")
    health.require_game_date(df, game_date=game_date, label="rates")
    return out_path


@task(name="effective-rates", retries=0)
def effective_rates_task(
    *, game_date: str, run_id: str, rates_path: Path, data_root: Path
) -> Path:
    run_dir = rates_path.parent
    eff_path = effective_inputs.write_effective_rates_layer(
        game_date=pd.Timestamp(game_date).date(),
        rates_path=rates_path,
        out_dir=run_dir,
        data_root=data_root,
        source="gameview",
    )
    health.require_file(eff_path, label="effective_rates.parquet")
    eff = pd.read_parquet(eff_path)
    health.require_row_count(eff, min_rows=20, max_rows=1200, label="effective rates")
    health.require_game_date(eff, game_date=game_date, label="effective rates")
    return eff_path


@task(name="run-sim", retries=1, retry_delay_seconds=120)
def run_sim_task(
    *,
    game_date: str,
    run_id: str,
    run_as_of_ts: str,
    data_root: Path,
    sim_worlds: int,
    sim_profile: str,
    minutes_override_mode: str = "v2",
    override_infeasible: str = "error",
) -> Path:
    args = [
        "--run-date",
        game_date,
        "--profile-name",
        sim_profile,
        "--num-worlds",
        str(sim_worlds),
        "--data-root",
        str(data_root),
        "--run-id",
        run_id,
        "--run-as-of-ts",
        run_as_of_ts,
        "--minutes-run-id",
        run_id,
        "--rates-run-id",
        run_id,
        "--minutes-override-mode",
        str(minutes_override_mode),
        "--override-infeasible",
        str(override_infeasible),
    ]
    _run_python_module(
        "scripts.sim_v2.run_sim_live", args, data_root=data_root, timeout_s=2400
    )
    out_dir = (
        data_root
        / "artifacts"
        / "sim_v2"
        / "worlds_fpts_v2"
        / f"game_date={game_date}"
        / f"run={run_id}"
    )
    health.require_file(
        out_dir / "projections.parquet", label="sim projections.parquet"
    )
    sim_df = pd.read_parquet(out_dir / "projections.parquet")
    health.require_row_count(sim_df, min_rows=20, max_rows=800, label="sim projections")
    health.require_columns(sim_df, required=["player_id"], label="sim projections")
    return out_dir


@task(name="score-ownership", retries=2, retry_delay_seconds=120)
def score_ownership_task(*, game_date: str, run_id: str, data_root: Path) -> None:
    """Score ownership using LineStar commercial projections.

    LineStar provides projected ownership % for all players on DK slates.
    The CLI handles auto-refresh of session if premium session expires.
    """
    args = [
        "--date",
        game_date,
        "--run-id",
        run_id,
        "--data-root",
        str(data_root),
    ]
    _run_python_module(
        "projections.cli.score_ownership_linestar",
        args,
        data_root=data_root,
        timeout_s=1200,
    )


@task(name="select-main-draft-group")
def select_main_draft_group_task(*, game_date: str, data_root: Path) -> str:
    base = data_root / "gold" / "dk_salaries" / "site=dk" / f"game_date={game_date}"
    if not base.exists():
        raise RuntimeError(f"DK salaries missing at {base}")

    slates: list[tuple[int, int, int]] = []
    for dg_dir in base.glob("draft_group_id=*"):
        parquet_path = dg_dir / "salaries.parquet"
        if parquet_path.exists():
            dg_id = int(dg_dir.name.split("=", 1)[1])
            try:
                df = pd.read_parquet(parquet_path, columns=["salary"])
            except Exception:
                df = pd.read_parquet(parquet_path)
            n_rows = int(len(df))
            n_salary = int(df["salary"].notna().sum()) if "salary" in df.columns else 0
            slates.append((dg_id, n_rows, n_salary))

    if not slates:
        raise RuntimeError(f"No DK slates found under {base}")

    has_salaries = any(n_salary > 0 for _, _, n_salary in slates)
    candidates = [s for s in slates if (s[2] > 0)] if has_salaries else slates
    candidates.sort(key=lambda x: (-x[2], -x[1], x[0]))
    return str(candidates[0][0])


@task(name="finalize-projections", retries=1, retry_delay_seconds=120)
def finalize_projections_task(
    *,
    game_date: str,
    run_id: str,
    draft_group_id: str,
    data_root: Path,
) -> Path:
    args = [
        "--date",
        game_date,
        "--run-id",
        run_id,
        "--minutes-run-id",
        run_id,
        "--sim-run-id",
        run_id,
        "--ownership-run-id",
        run_id,
        "--draft-group-id",
        str(draft_group_id),
        "--data-root",
        str(data_root),
    ]
    _run_python_module(
        "projections.cli.finalize_projections",
        args,
        data_root=data_root,
        timeout_s=1200,
    )
    out_dir = data_root / "artifacts" / "projections" / game_date / f"run={run_id}"
    health.require_file(
        out_dir / "projections.parquet", label="unified projections.parquet"
    )
    df = pd.read_parquet(out_dir / "projections.parquet")
    health.require_row_count(
        df, min_rows=20, max_rows=1200, label="unified projections"
    )
    health.require_columns(df, required=["player_id"], label="unified projections")
    return out_dir


@flow(name="nba-live-pipeline", log_prints=True)
def nba_live_pipeline_flow(
    *,
    game_date: str | None = None,
    sim_worlds: int = 25000,
    run_id_override: str | None = None,
    promote_pointers: bool = True,
    minutes_override_mode: str = "v2",
    override_infeasible: str = "error",
    # Minutes model overrides (shadow runs).
    minutes_bundle_dir: str | None = None,
    rotshare_quantiles_mode: str | None = None,
    rotshare_n_worlds: int = 25_000,
    rotshare_concentration: float = 60.0,
    rotshare_seed: int = 42,
    rotshare_min_active_players: int = 5,
    rotshare_mc_center: str = "mean",
) -> dict[str, str]:
    logger = get_run_logger()
    data_root = paths.get_data_root()

    minutes_override_mode_eff = str(minutes_override_mode).strip().lower()
    if minutes_override_mode_eff not in {"legacy", "v2"}:
        raise ValueError(
            f"minutes_override_mode must be one of legacy|v2, got {minutes_override_mode!r}"
        )
    override_infeasible_eff = str(override_infeasible).strip().lower()
    if override_infeasible_eff not in {"error", "relax", "ignore"}:
        raise ValueError(
            f"override_infeasible must be one of error|relax|ignore, got {override_infeasible!r}"
        )

    logger.info(
        "[sim] minutes_override_mode=%s override_infeasible=%s",
        minutes_override_mode_eff,
        override_infeasible_eff,
    )

    minutes_selector_path = model_selectors.active_minutes_selector_path(
        data_root=data_root,
        project_root=PROJECT_ROOT,
    )
    rates_selector_path = model_selectors.active_rates_selector_path(
        data_root=data_root,
        project_root=PROJECT_ROOT,
    )

    # Runtime stamp - log what code/config is running at flow start
    enforce_clean_tree()  # Fail-fast if dirty tree in prod (set PROJECTIONS_ALLOW_DIRTY=1 to bypass)
    enforce_prod_sanity()  # Strict PROD checks - no bypass allowed in PROD
    log_runtime_stamp(
        entrypoint="prefect:nba-live-pipeline",
        config_paths={
            "minutes_current_run": minutes_selector_path,
            "rates_current_run": rates_selector_path,
            "rotation_set_minutes_live": PROJECT_ROOT
            / "config/rotation_set_minutes_live.json",
            "sim_v2_profiles": PROJECT_ROOT / "config/sim_v2_profiles.json",
        },
        project_root=PROJECT_ROOT,
        logger=logger,
    )

    if game_date is None:
        # Use Eastern Time for NBA slate date (games are scheduled for ET dates).
        # A 10pm ET game on Jan 2 should still be game_date=2026-01-02.
        et = ZoneInfo("America/New_York")
        game_date = datetime.now(tz=et).date().isoformat()

    run_id = run_id_override or control_plane.canonical_run_id()

    with writer_guard.PipelineWriterLock(data_root=data_root, run_id=run_id):
        # Ensure subprocess CLIs never publish pointers mid-run; this flow promotes atomically at the end.
        os.environ["PROJECTIONS_SKIP_POINTER_WRITES"] = "1"

        # as_of_ts is the *input cutoff* used by downstream steps; set it after scrape completes.
        as_of_ts = _utc_now_iso()
        manifest_path = control_plane.write_run_manifest_start(
            data_root=data_root,
            game_date=game_date,
            run_id=run_id,
            as_of_ts=as_of_ts,
            sim_profile="sim_v3",
            entrypoint="prefect",
            minutes_current_run_path=minutes_selector_path,
            rates_current_run_path=rates_selector_path,
            slate={},
        )
        if minutes_bundle_dir or rotshare_quantiles_mode:
            control_plane.atomic_update_json(
                manifest_path,
                {
                    "minutes_override": {
                        "minutes_bundle_dir": minutes_bundle_dir,
                        "rotshare_quantiles_mode": rotshare_quantiles_mode,
                        "rotshare_n_worlds": int(rotshare_n_worlds),
                        "rotshare_concentration": float(rotshare_concentration),
                        "rotshare_seed": int(rotshare_seed),
                        "rotshare_min_active_players": int(rotshare_min_active_players),
                        "rotshare_mc_center": str(rotshare_mc_center),
                    }
                },
            )
        logger.info(f"[run-manifest] run_id={run_id} manifest={manifest_path}")

        # Stage 1: scrape
        logger.info(f"[inputs] scraping game_date={game_date}")
        scrape_task(game_date=game_date, data_root=data_root)

        # Refresh as_of_ts after scrape so features don't filter out newly-scraped inputs.
        as_of_ts = _utc_now_iso()
        control_plane.atomic_update_json(manifest_path, {"as_of_ts": as_of_ts})
        logger.info(f"[run-manifest] updated as_of_ts={as_of_ts}")

        # Stage 1.25: DK salaries (required for finalize)
        dk_salaries_task(game_date=game_date, data_root=data_root)

        # Stage 1.5: Scrape props (for props analysis tab)
        scrape_props_task(game_date=game_date, data_root=data_root)

        # Stage 2: build minutes features
        feat_minutes = build_minutes_features_task(
            game_date=game_date,
            run_id=run_id,
            run_as_of_ts=as_of_ts,
            data_root=data_root,
        )
        control_plane.copy_manifest_to_dir(manifest_path, feat_minutes.parent)

        # Stage 2.5: RMH shadow minutes (optional, non-blocking)
        rmh_minutes_path = rmh_shadow_minutes_task(
            game_date=game_date,
            run_id=run_id,
            run_as_of_ts=as_of_ts,
            features_path=feat_minutes,
            data_root=data_root,
        )
        if rmh_minutes_path is not None:
            control_plane.copy_manifest_to_dir(manifest_path, rmh_minutes_path.parent)

        # Stage 3: score minutes
        minutes_path = score_minutes_task(
            game_date=game_date,
            run_id=run_id,
            reconcile_mode=_read_minutes_reconcile_mode(),
            data_root=data_root,
            minutes_bundle_dir=minutes_bundle_dir,
            rotshare_quantiles_mode=rotshare_quantiles_mode,
            rotshare_n_worlds=rotshare_n_worlds,
            rotshare_concentration=rotshare_concentration,
            rotshare_seed=rotshare_seed,
            rotshare_min_active_players=rotshare_min_active_players,
            rotshare_mc_center=rotshare_mc_center,
        )
        control_plane.copy_manifest_to_dir(manifest_path, minutes_path.parent)

        # Stage 3.5: effective minutes inputs (overrides from GameView only)
        eff_minutes_path = effective_minutes_task(
            game_date=game_date,
            run_id=run_id,
            minutes_path=minutes_path,
            data_root=data_root,
        )

        # Publish minutes to gold run dir (no flat file writes).
        minutes_gold_day = publish_minutes_gold_task(
            game_date=game_date,
            run_id=run_id,
            effective_minutes_path=eff_minutes_path,
            effective_summary_path=(
                minutes_path.parent / effective_inputs.EFFECTIVE_INPUTS_SUMMARY
            ),
            data_root=data_root,
        )
        control_plane.copy_manifest_to_dir(
            manifest_path, minutes_gold_day / f"run={run_id}"
        )

        # Stage 4: build rates features
        feat_rates = build_rates_features_task(
            game_date=game_date, run_id=run_id, data_root=data_root
        )
        control_plane.copy_manifest_to_dir(manifest_path, feat_rates.parent)

        # Stage 5: score rates
        rates_path = score_rates_task(
            game_date=game_date, run_id=run_id, data_root=data_root
        )
        control_plane.copy_manifest_to_dir(manifest_path, rates_path.parent)

        # Stage 5.5: effective rates inputs (overrides from GameView only)
        _ = effective_rates_task(
            game_date=game_date,
            run_id=run_id,
            rates_path=rates_path,
            data_root=data_root,
        )

        # Stage 6: sim worlds (sim_v3 only)
        sim_dir = run_sim_task(
            game_date=game_date,
            run_id=run_id,
            run_as_of_ts=as_of_ts,
            data_root=data_root,
            sim_worlds=sim_worlds,
            sim_profile="sim_v3",
            minutes_override_mode=minutes_override_mode_eff,
            override_infeasible=override_infeasible_eff,
        )
        control_plane.copy_manifest_to_dir(manifest_path, sim_dir)

        # Stage 7: ownership
        score_ownership_task(game_date=game_date, run_id=run_id, data_root=data_root)
        ownership_run_dir = (
            data_root / "silver" / "ownership_predictions" / game_date / f"run={run_id}"
        )
        if ownership_run_dir.exists():
            control_plane.copy_manifest_to_dir(manifest_path, ownership_run_dir)

        # Stage 8: finalize + publish unified projections
        draft_group_id = select_main_draft_group_task(
            game_date=game_date, data_root=data_root
        )
        control_plane.atomic_update_json(
            manifest_path, {"slate": {"draft_group_id": draft_group_id}}
        )
        projections_dir = finalize_projections_task(
            game_date=game_date,
            run_id=run_id,
            draft_group_id=draft_group_id,
            data_root=data_root,
        )
        control_plane.copy_manifest_to_dir(manifest_path, projections_dir)

        if promote_pointers:
            # Promote pointers (single-writer guarded).
            control_plane.promote_run_pointer(
                dataset_dir=(data_root / "live" / "features_minutes_v1" / game_date),
                run_id=run_id,
                manifest_path=manifest_path,
                extra={"entrypoint": "prefect", "stage": "features_minutes_v1"},
            )
            control_plane.promote_run_pointer(
                dataset_dir=(
                    data_root / "artifacts" / "minutes_v1" / "daily" / game_date
                ),
                run_id=run_id,
                manifest_path=manifest_path,
                extra={"entrypoint": "prefect", "stage": "minutes_v1_daily"},
            )
            control_plane.promote_run_pointer(
                dataset_dir=(
                    data_root
                    / "gold"
                    / "projections_minutes_v1"
                    / f"game_date={game_date}"
                ),
                run_id=run_id,
                manifest_path=manifest_path,
                extra={"entrypoint": "prefect", "stage": "projections_minutes_v1"},
            )
            control_plane.promote_run_pointer(
                dataset_dir=(data_root / "live" / "features_rates_v1" / game_date),
                run_id=run_id,
                manifest_path=manifest_path,
                extra={"entrypoint": "prefect", "stage": "features_rates_v1"},
            )
            control_plane.promote_run_pointer(
                dataset_dir=(data_root / "gold" / "rates_v1_live" / game_date),
                run_id=run_id,
                manifest_path=manifest_path,
                extra={"entrypoint": "prefect", "stage": "rates_v1_live"},
            )
            control_plane.promote_run_pointer(
                dataset_dir=(
                    data_root / "silver" / "ownership_predictions" / game_date
                ),
                run_id=run_id,
                manifest_path=manifest_path,
                extra={"entrypoint": "prefect", "stage": "ownership_predictions"},
            )
            control_plane.promote_run_pointer(
                dataset_dir=(
                    data_root
                    / "artifacts"
                    / "sim_v2"
                    / "worlds_fpts_v2"
                    / f"game_date={game_date}"
                ),
                run_id=run_id,
                manifest_path=manifest_path,
                extra={"entrypoint": "prefect", "stage": "sim_v2_worlds_fpts_v2"},
            )
            control_plane.promote_run_pointer(
                dataset_dir=(data_root / "artifacts" / "projections" / game_date),
                run_id=run_id,
                manifest_path=manifest_path,
                extra={"entrypoint": "prefect", "stage": "unified_projections"},
            )
            if _clear_stale_ops_pin_pointer(
                data_root=data_root,
                game_date=game_date,
                keep_run_id=run_id,
            ):
                logger.info(
                    "[projections] cleared stale ops pinned_run.json after promoting run=%s",
                    run_id,
                )
            if rmh_minutes_path is not None:
                try:
                    control_plane.promote_run_pointer(
                        dataset_dir=minutes_models.model_day_dir(
                            data_root=data_root,
                            model_id=minutes_models.MODEL_ID_RMH_V1_1,
                            game_date=game_date,
                        ),
                        run_id=run_id,
                        manifest_path=manifest_path,
                        extra={
                            "entrypoint": "prefect",
                            "stage": "minutes_shadow",
                            "model_id": "rmh_v1_1",
                        },
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        f"[rmh-shadow] failed to promote pointer (continuing): {exc}"
                    )
        else:
            logger.info(
                "[shadow] promote_pointers=False; skipping latest_run.json updates"
            )

        logger.info(f"[outputs] projections_dir={projections_dir}")
        return {
            "run_id": run_id,
            "game_date": game_date,
            "manifest_path": str(manifest_path),
            "projections_dir": str(projections_dir),
            "rates_path": str(rates_path),
            "minutes_path": str(minutes_path),
        }


def _utc_now_iso() -> str:
    # Keep sub-second precision so as_of_ts ordering stays consistent with
    # upstream snapshot timestamps (which are often microsecond-precision).
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")


def _read_minutes_reconcile_mode() -> str:
    config_path = model_selectors.active_minutes_selector_path()
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "none"
    mode = payload.get("reconcile_team_minutes")
    return str(mode) if mode else "none"


def _clear_stale_ops_pin_pointer(*, data_root: Path, game_date: str, keep_run_id: str) -> bool:
    """Clear stale ops-created pinned run pointer after canonical scheduled publish."""
    pinned_path = data_root / "artifacts" / "projections" / game_date / "pinned_run.json"
    if not pinned_path.exists():
        return False
    try:
        payload = json.loads(pinned_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False
    source = str(payload.get("source") or "").strip().lower()
    pinned_run_id = str(payload.get("run_id") or "").strip()
    if source != "ops_patch_game":
        return False
    if not pinned_run_id or pinned_run_id == str(keep_run_id):
        return False
    try:
        pinned_path.unlink(missing_ok=True)
    except Exception:
        return False
    return True
