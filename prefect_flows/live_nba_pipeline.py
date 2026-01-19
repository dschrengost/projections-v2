"""Canonical Prefect flow for the NBA live pipeline.

This is intended to be the *only* production orchestrator entrypoint.
Each stage is an explicit Prefect task and the flow enforces:
- single-writer guard for pointer publishes
- run_id-scoped outputs
- run manifest written at start
"""

from __future__ import annotations

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
from projections.minutes_v1.datasets import KEY_COLUMNS, deduplicate_latest
from projections.pipeline import control_plane, writer_guard
from projections.pipeline import effective_inputs, health


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
        raise FileNotFoundError(
            f"UV_BIN={env_uv} specified but file does not exist"
        )
    
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
    _run_python_module("projections.cli.live_pipeline", args, data_root=data_root, timeout_s=900)


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
    """Scrape player props from RotoWire for the props analysis tab.

    This is a non-blocking task - props failures should not fail the pipeline.
    Props may be unavailable early in the day or during off-season.
    """
    try:
        _run_python_module(
            "projections.cli.scrape_props",
            ["scrape", "--date", game_date],
            data_root=data_root,
            timeout_s=300,
        )
    except Exception as e:
        logger = get_run_logger()
        logger.warning(f"Props scrape failed (continuing): {e}")


@task(name="build-minutes-features", retries=1, retry_delay_seconds=60)
def build_minutes_features_task(*, game_date: str, run_id: str, run_as_of_ts: str, data_root: Path) -> Path:
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
        raise RuntimeError("No live game_ids available after building roster-based labels.")

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
        pd.to_numeric(live_slice["game_id"], errors="coerce").astype("Int64").isin(live_game_ids)
    ].copy()
    if {"feature_as_of_ts", *KEY_COLUMNS}.issubset(live_slice.columns):
        live_slice = deduplicate_latest(
            live_slice,
            key_cols=KEY_COLUMNS,
            order_cols=["feature_as_of_ts"],
        )
        live_slice = live_slice.drop_duplicates(subset=list(KEY_COLUMNS), keep="last")

    run_dir = data_root / "live" / "features_minutes_v1" / game_date / f"run={run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "features.parquet"
    live_slice.to_parquet(out_path, index=False)
    health.require_file(out_path, label="minutes features.parquet")
    df = pd.read_parquet(out_path)
    health.require_row_count(df, min_rows=20, max_rows=5000, label="minutes features")
    health.require_columns(df, required=["player_id", "game_id", "team_id"], label="minutes features")
    health.require_non_null(df, cols=["player_id", "game_id", "team_id"], label="minutes features")
    health.require_game_date(df, game_date=game_date, label="minutes features")
    health.require_freshness(
        df,
        ts_cols=["feature_as_of_ts", "odds_as_of_ts", "injuries_as_of_ts", "as_of_ts"],
        reference_ts=run_as_of_ts,
        max_age_minutes=240.0,
        label="minutes features",
    )
    return out_path


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
    """Score minutes with rotation set overlay (if enabled).

    Uses score_minutes_rotation_set_v1 which:
    1. Runs baseline minutes_v1 scoring
    2. If rotation_set_minutes_live.json is enabled, builds rotation features + predicts
    3. Applies guardrails and blends rotation predictions with baseline
    """
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
        args.extend(["--rotshare-min-active-players", str(int(rotshare_min_active_players))])
    if rotshare_mc_center:
        args.extend(["--rotshare-mc-center", rotshare_mc_center])
    # Use rotation_set_v1 scorer which runs baseline + rotation overlay if enabled
    _run_python_module("projections.cli.score_minutes_rotation_set_v1", args, data_root=data_root, timeout_s=1200)
    out_path = data_root / "artifacts" / "minutes_v1" / "daily" / game_date / f"run={run_id}" / "minutes.parquet"
    health.require_file(out_path, label="minutes.parquet")
    df = pd.read_parquet(out_path)
    health.require_row_count(df, min_rows=20, max_rows=800, label="minutes")
    health.require_columns(df, required=["player_id", "game_id", "team_id"], label="minutes")
    health.require_non_null(df, cols=["player_id", "game_id", "team_id"], label="minutes")
    health.require_game_date(df, game_date=game_date, label="minutes")
    return out_path


@task(name="effective-minutes", retries=0)
def effective_minutes_task(*, game_date: str, run_id: str, minutes_path: Path, data_root: Path) -> Path:
    run_dir = minutes_path.parent
    result = effective_inputs.write_effective_minutes_layer(
        game_date=pd.Timestamp(game_date).date(),
        minutes_path=minutes_path,
        out_dir=run_dir,
        data_root=data_root,
        source="gameview",
    )
    health.require_file(result.effective_minutes_path, label="effective_minutes.parquet")
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
    out_summary.write_text(effective_summary_path.read_text(encoding="utf-8"), encoding="utf-8")
    return out_day


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
    _run_python_module("projections.cli.build_rates_features_live", args, data_root=data_root, timeout_s=1200)
    out_path = data_root / "live" / "features_rates_v1" / game_date / f"run={run_id}" / "features.parquet"
    health.require_file(out_path, label="rates features.parquet")
    df = pd.read_parquet(out_path)
    health.require_row_count(df, min_rows=20, max_rows=5000, label="rates features")
    health.require_columns(df, required=["player_id", "game_id", "team_id"], label="rates features")
    health.require_non_null(df, cols=["player_id", "game_id", "team_id"], label="rates features")
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
    _run_python_module("projections.cli.score_rates_live", args, data_root=data_root, timeout_s=1200)
    out_path = data_root / "gold" / "rates_v1_live" / game_date / f"run={run_id}" / "rates.parquet"
    health.require_file(out_path, label="rates.parquet")
    df = pd.read_parquet(out_path)
    health.require_row_count(df, min_rows=20, max_rows=800, label="rates")
    health.require_columns(df, required=["player_id", "game_id", "team_id"], label="rates")
    health.require_non_null(df, cols=["player_id", "game_id", "team_id"], label="rates")
    health.require_game_date(df, game_date=game_date, label="rates")
    return out_path


@task(name="effective-rates", retries=0)
def effective_rates_task(*, game_date: str, run_id: str, rates_path: Path, data_root: Path) -> Path:
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
    ]
    _run_python_module("scripts.sim_v2.run_sim_live", args, data_root=data_root, timeout_s=2400)
    out_dir = data_root / "artifacts" / "sim_v2" / "worlds_fpts_v2" / f"game_date={game_date}" / f"run={run_id}"
    health.require_file(out_dir / "projections.parquet", label="sim projections.parquet")
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
    _run_python_module("projections.cli.score_ownership_linestar", args, data_root=data_root, timeout_s=1200)


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
    _run_python_module("projections.cli.finalize_projections", args, data_root=data_root, timeout_s=1200)
    out_dir = data_root / "artifacts" / "projections" / game_date / f"run={run_id}"
    health.require_file(out_dir / "projections.parquet", label="unified projections.parquet")
    df = pd.read_parquet(out_dir / "projections.parquet")
    health.require_row_count(df, min_rows=20, max_rows=1200, label="unified projections")
    health.require_columns(df, required=["player_id"], label="unified projections")
    return out_dir


@flow(name="nba-live-pipeline", log_prints=True)
def nba_live_pipeline_flow(
    *,
    game_date: str | None = None,
    sim_worlds: int = 25000,
    run_id_override: str | None = None,
    promote_pointers: bool = True,
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
            minutes_current_run_path=Path("config/minutes_current_run.json"),
            rates_current_run_path=Path("config/rates_current_run.json"),
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
            game_date=game_date, run_id=run_id, run_as_of_ts=as_of_ts, data_root=data_root
        )
        control_plane.copy_manifest_to_dir(manifest_path, feat_minutes.parent)

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
            game_date=game_date, run_id=run_id, minutes_path=minutes_path, data_root=data_root
        )

        # Publish minutes to gold run dir (no flat file writes).
        minutes_gold_day = publish_minutes_gold_task(
            game_date=game_date,
            run_id=run_id,
            effective_minutes_path=eff_minutes_path,
            effective_summary_path=(minutes_path.parent / effective_inputs.EFFECTIVE_INPUTS_SUMMARY),
            data_root=data_root,
        )
        control_plane.copy_manifest_to_dir(manifest_path, minutes_gold_day / f"run={run_id}")

        # Stage 4: build rates features
        feat_rates = build_rates_features_task(game_date=game_date, run_id=run_id, data_root=data_root)
        control_plane.copy_manifest_to_dir(manifest_path, feat_rates.parent)

        # Stage 5: score rates
        rates_path = score_rates_task(game_date=game_date, run_id=run_id, data_root=data_root)
        control_plane.copy_manifest_to_dir(manifest_path, rates_path.parent)

        # Stage 5.5: effective rates inputs (overrides from GameView only)
        _ = effective_rates_task(game_date=game_date, run_id=run_id, rates_path=rates_path, data_root=data_root)

        # Stage 6: sim worlds (sim_v3 only)
        sim_dir = run_sim_task(
            game_date=game_date,
            run_id=run_id,
            run_as_of_ts=as_of_ts,
            data_root=data_root,
            sim_worlds=sim_worlds,
            sim_profile="sim_v3",
        )
        control_plane.copy_manifest_to_dir(manifest_path, sim_dir)

        # Stage 7: ownership
        score_ownership_task(game_date=game_date, run_id=run_id, data_root=data_root)
        ownership_run_dir = data_root / "silver" / "ownership_predictions" / game_date / f"run={run_id}"
        if ownership_run_dir.exists():
            control_plane.copy_manifest_to_dir(manifest_path, ownership_run_dir)

        # Stage 8: finalize + publish unified projections
        draft_group_id = select_main_draft_group_task(game_date=game_date, data_root=data_root)
        control_plane.atomic_update_json(manifest_path, {"slate": {"draft_group_id": draft_group_id}})
        projections_dir = finalize_projections_task(
            game_date=game_date, run_id=run_id, draft_group_id=draft_group_id, data_root=data_root
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
                dataset_dir=(data_root / "artifacts" / "minutes_v1" / "daily" / game_date),
                run_id=run_id,
                manifest_path=manifest_path,
                extra={"entrypoint": "prefect", "stage": "minutes_v1_daily"},
            )
            control_plane.promote_run_pointer(
                dataset_dir=(data_root / "gold" / "projections_minutes_v1" / f"game_date={game_date}"),
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
                dataset_dir=(data_root / "silver" / "ownership_predictions" / game_date),
                run_id=run_id,
                manifest_path=manifest_path,
                extra={"entrypoint": "prefect", "stage": "ownership_predictions"},
            )
            control_plane.promote_run_pointer(
                dataset_dir=(data_root / "artifacts" / "sim_v2" / "worlds_fpts_v2" / f"game_date={game_date}"),
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
        else:
            logger.info("[shadow] promote_pointers=False; skipping latest_run.json updates")

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
    return datetime.now(tz=UTC).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%S")


def _read_minutes_reconcile_mode() -> str:
    config_path = Path("config/minutes_current_run.json")
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "none"
    mode = payload.get("reconcile_team_minutes")
    return str(mode) if mode else "none"
