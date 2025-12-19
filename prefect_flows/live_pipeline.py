"""
Prefect flows wrapping the existing live-scrape and live-score shell scripts.

Phase 1 of migration: treat scripts as black boxes, capture logs, emit manifests.
These flows run the same shell scripts that systemd timers currently invoke.
"""

import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from prefect import flow, task, get_run_logger
from prefect.artifacts import create_markdown_artifact


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path("/home/daniel/projects/projections-v2")
DATA_ROOT = Path("/home/daniel/projections-data")
MANIFESTS_ROOT = DATA_ROOT / "manifests"
UV_BIN = "/home/daniel/.local/bin/uv"


# ──────────────────────────────────────────────────────────────────────────────
# Manifest helpers
# ──────────────────────────────────────────────────────────────────────────────

def emit_manifest(
    task_name: str,
    start_ts: datetime,
    end_ts: datetime,
    exit_code: int,
    output_paths: list[str] | None = None,
    error_snippet: str | None = None,
    extra: dict | None = None,
) -> Path:
    """
    Write a run manifest JSON to the manifests directory.
    
    Manifests provide durable, append-only records of task runs independent
    of Prefect's database (per TARGET_ARCHITECTURE.md).
    """
    manifest = {
        "task_name": task_name,
        "start_ts": start_ts.isoformat(),
        "end_ts": end_ts.isoformat(),
        "duration_s": (end_ts - start_ts).total_seconds(),
        "exit_code": exit_code,
        "success": exit_code == 0,
        "output_paths": output_paths or [],
        "error_snippet": error_snippet,
        **(extra or {}),
    }
    
    # Organize by date and task
    date_str = start_ts.strftime("%Y-%m-%d")
    manifest_dir = MANIFESTS_ROOT / date_str / task_name
    manifest_dir.mkdir(parents=True, exist_ok=True)
    
    # Use start timestamp as unique ID
    ts_id = start_ts.strftime("%Y%m%dT%H%M%SZ")
    manifest_path = manifest_dir / f"{ts_id}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    
    return manifest_path


def run_shell_script(
    script_path: Path,
    env_overrides: dict[str, str] | None = None,
    timeout_seconds: int = 3600,
) -> tuple[int, str, str]:
    """
    Run a shell script, capturing stdout and stderr.
    
    Returns (exit_code, stdout, stderr).
    """
    import os
    
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    
    result = subprocess.run(
        ["bash", str(script_path)],
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    
    return result.returncode, result.stdout, result.stderr


# ──────────────────────────────────────────────────────────────────────────────
# Tasks
# ──────────────────────────────────────────────────────────────────────────────

@task(
    name="run-live-scrape",
    retries=2,
    retry_delay_seconds=60,
    log_prints=True,
)
def run_live_scrape_task(
    game_date: str | None = None,
    season: str | None = None,
    month: str | None = None,
) -> dict:
    """
    Run the live scraping pipeline (injuries, odds, schedule).
    
    Wraps scripts/run_live_scrape.sh as a black-box subprocess.
    """
    logger = get_run_logger()
    start_ts = datetime.now(timezone.utc)
    
    # Build environment overrides
    env = {}
    if game_date:
        env["LIVE_START_DATE"] = game_date
        env["LIVE_END_DATE"] = game_date
    if season:
        env["LIVE_SEASON"] = season
    if month:
        env["LIVE_MONTH"] = month
    
    script_path = PROJECT_ROOT / "scripts" / "run_live_scrape.sh"
    logger.info(f"Running {script_path.name} for date={game_date or 'today'}")
    
    try:
        exit_code, stdout, stderr = run_shell_script(script_path, env)
    except subprocess.TimeoutExpired:
        exit_code, stdout, stderr = -1, "", "Script timed out after 3600s"
    
    end_ts = datetime.now(timezone.utc)
    
    # Log output
    if stdout:
        for line in stdout.strip().split("\n"):
            logger.info(line)
    if stderr:
        for line in stderr.strip().split("\n"):
            logger.warning(line)
    
    # Emit manifest
    manifest_path = emit_manifest(
        task_name="live-scrape",
        start_ts=start_ts,
        end_ts=end_ts,
        exit_code=exit_code,
        output_paths=[
            str(DATA_ROOT / "bronze" / "injuries"),
            str(DATA_ROOT / "bronze" / "odds"),
        ],
        error_snippet=stderr[:500] if exit_code != 0 and stderr else None,
        extra={"game_date": game_date or datetime.now().strftime("%Y-%m-%d")},
    )
    logger.info(f"Manifest written to {manifest_path}")
    
    if exit_code != 0:
        raise RuntimeError(f"live-scrape failed with exit code {exit_code}")
    
    return {
        "exit_code": exit_code,
        "duration_s": (end_ts - start_ts).total_seconds(),
        "manifest_path": str(manifest_path),
    }


@task(
    name="run-live-score",
    retries=1,  # Less aggressive retries for long-running job
    retry_delay_seconds=120,
    log_prints=True,
)
def run_live_score_task(
    game_date: str | None = None,
    season: str | None = None,
    month: str | None = None,
    skip_scrape: bool = False,
    run_sim: bool = True,
    sim_profile: str = "baseline",
    sim_worlds: int = 10000,
    disable_tip_window: bool = False,
) -> dict:
    """
    Run the full live scoring pipeline (features, scoring, sim, ownership).
    
    Wraps scripts/run_live_score.sh as a black-box subprocess.
    This is the main pipeline that runs every 5 minutes during slate windows.
    """
    logger = get_run_logger()
    start_ts = datetime.now(timezone.utc)
    
    # Build environment overrides
    env = {
        "LIVE_SKIP_SCRAPE": "1" if skip_scrape else "0",
        "LIVE_RUN_SIM": "1" if run_sim else "0",
        "LIVE_SIM_PROFILE": sim_profile,
        "LIVE_SIM_WORLDS": str(sim_worlds),
        "LIVE_DISABLE_TIP_WINDOW": "1" if disable_tip_window else "0",
    }
    if game_date:
        env["LIVE_START_DATE"] = game_date
        env["LIVE_END_DATE"] = game_date
    if season:
        env["LIVE_SEASON"] = season
    if month:
        env["LIVE_MONTH"] = month
    
    script_path = PROJECT_ROOT / "scripts" / "run_live_score.sh"
    logger.info(f"Running {script_path.name} for date={game_date or 'today'}")
    logger.info(f"Options: skip_scrape={skip_scrape}, run_sim={run_sim}, profile={sim_profile}")
    
    try:
        # Give scoring pipeline up to 30 minutes
        exit_code, stdout, stderr = run_shell_script(script_path, env, timeout_seconds=1800)
    except subprocess.TimeoutExpired:
        exit_code, stdout, stderr = -1, "", "Script timed out after 1800s"
    
    end_ts = datetime.now(timezone.utc)
    
    # Log output
    if stdout:
        for line in stdout.strip().split("\n"):
            logger.info(line)
    if stderr:
        for line in stderr.strip().split("\n"):
            logger.warning(line)
    
    # Determine output paths
    effective_date = game_date or datetime.now().strftime("%Y-%m-%d")
    output_paths = [
        str(DATA_ROOT / "gold" / "projections_minutes_v1" / f"game_date={effective_date}"),
        str(DATA_ROOT / "gold" / "rates_v1_live" / f"game_date={effective_date}"),
        str(DATA_ROOT / "gold" / "sim_v2" / f"game_date={effective_date}"),
    ]
    
    # Emit manifest
    manifest_path = emit_manifest(
        task_name="live-score",
        start_ts=start_ts,
        end_ts=end_ts,
        exit_code=exit_code,
        output_paths=output_paths,
        error_snippet=stderr[:500] if exit_code != 0 and stderr else None,
        extra={
            "game_date": effective_date,
            "skip_scrape": skip_scrape,
            "run_sim": run_sim,
            "sim_profile": sim_profile,
        },
    )
    logger.info(f"Manifest written to {manifest_path}")
    
    # Create Prefect artifact with summary
    create_markdown_artifact(
        key="live-score-summary",
        markdown=f"""
## Live Score Run Summary

| Metric | Value |
|--------|-------|
| Date | {effective_date} |
| Duration | {(end_ts - start_ts).total_seconds():.1f}s |
| Exit Code | {exit_code} |
| Sim Worlds | {sim_worlds} |
| Profile | {sim_profile} |

### Output Paths
{chr(10).join(f"- `{p}`" for p in output_paths)}
""",
        description=f"Live scoring run for {effective_date}",
    )
    
    if exit_code != 0:
        raise RuntimeError(f"live-score failed with exit code {exit_code}")
    
    return {
        "exit_code": exit_code,
        "duration_s": (end_ts - start_ts).total_seconds(),
        "manifest_path": str(manifest_path),
        "output_paths": output_paths,
    }


@task(
    name="run-dk-salaries",
    retries=2,
    retry_delay_seconds=120,
    log_prints=True,
)
def run_dk_salaries_task(
    game_date: str | None = None,
    force_refresh: bool = False,
) -> dict:
    """
    Fetch DraftKings salaries for all slates on a given date.
    
    Calls: python -m scripts.dk.run_daily_salaries
    """
    import os
    
    logger = get_run_logger()
    start_ts = datetime.now(timezone.utc)
    
    # Build command
    cmd = [
        UV_BIN, "run", "python", "-m", "scripts.dk.run_daily_salaries",
    ]
    if game_date:
        cmd.extend(["--game-date", game_date])
    if force_refresh:
        cmd.append("--force-refresh")
    
    effective_date = game_date or datetime.now().strftime("%Y-%m-%d")
    logger.info(f"Running DK salaries fetch for {effective_date}")
    logger.info(f"Command: {' '.join(cmd)}")
    
    env = os.environ.copy()
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )
        exit_code, stdout, stderr = result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        exit_code, stdout, stderr = -1, "", "Script timed out after 600s"
    
    end_ts = datetime.now(timezone.utc)
    
    # Log output
    if stdout:
        for line in stdout.strip().split("\n"):
            logger.info(line)
    if stderr:
        for line in stderr.strip().split("\n"):
            logger.warning(line)
    
    # Emit manifest
    manifest_path = emit_manifest(
        task_name="dk-salaries",
        start_ts=start_ts,
        end_ts=end_ts,
        exit_code=exit_code,
        output_paths=[
            str(DATA_ROOT / "gold" / "dk_salaries" / "site=dk" / f"game_date={effective_date}"),
        ],
        error_snippet=stderr[:500] if exit_code != 0 and stderr else None,
        extra={"game_date": effective_date, "force_refresh": force_refresh},
    )
    logger.info(f"Manifest written to {manifest_path}")
    
    if exit_code != 0:
        raise RuntimeError(f"dk-salaries failed with exit code {exit_code}")
    
    return {
        "exit_code": exit_code,
        "duration_s": (end_ts - start_ts).total_seconds(),
        "manifest_path": str(manifest_path),
    }

# ──────────────────────────────────────────────────────────────────────────────
# Flows
# ──────────────────────────────────────────────────────────────────────────────

@flow(
    name="live-scrape",
    description="Scrape live data (injuries, odds, schedule) for today's games.",
    log_prints=True,
)
def live_scrape_flow(
    game_date: str | None = None,
    season: str | None = None,
    month: str | None = None,
) -> dict:
    """
    Flow for live scraping pipeline.
    
    Can be scheduled independently or called as part of live-score.
    """
    return run_live_scrape_task(
        game_date=game_date,
        season=season,
        month=month,
    )


@flow(
    name="live-score",
    description="Full live scoring pipeline: features → scoring → sim → ownership.",
    log_prints=True,
)
def live_score_flow(
    game_date: str | None = None,
    season: str | None = None,
    month: str | None = None,
    skip_scrape: bool = False,
    run_sim: bool = True,
    sim_profile: str = "baseline",
    sim_worlds: int = 10000,
    disable_tip_window: bool = False,
) -> dict:
    """
    Flow for full live scoring pipeline.
    
    This is the main pipeline that produces projections for the dashboard/API.
    During slate windows, this typically runs every 5 minutes.
    """
    return run_live_score_task(
        game_date=game_date,
        season=season,
        month=month,
        skip_scrape=skip_scrape,
        run_sim=run_sim,
        sim_profile=sim_profile,
        sim_worlds=sim_worlds,
        disable_tip_window=disable_tip_window,
    )


@flow(
    name="live-pipeline-full",
    description="Combined scrape + score pipeline (typical production flow).",
    log_prints=True,
)
def live_pipeline_full_flow(
    game_date: str | None = None,
    season: str | None = None,
    month: str | None = None,
    run_sim: bool = True,
    sim_profile: str = "baseline",
    sim_worlds: int = 10000,
) -> dict:
    """
    Combined flow that runs scrape first, then score.
    
    Use this for scheduled runs where you want both steps in sequence.
    """
    logger = get_run_logger()
    
    # Step 1: Scrape
    logger.info("Starting live scrape...")
    scrape_result = run_live_scrape_task(
        game_date=game_date,
        season=season,
        month=month,
    )
    
    # Step 2: Score (skip internal scrape since we just did it)
    logger.info("Starting live score (skip_scrape=True)...")
    score_result = run_live_score_task(
        game_date=game_date,
        season=season,
        month=month,
        skip_scrape=True,
        run_sim=run_sim,
        sim_profile=sim_profile,
        sim_worlds=sim_worlds,
    )
    
    return {
        "scrape": scrape_result,
        "score": score_result,
    }


@flow(
    name="dk-salaries",
    description="Fetch DraftKings salaries for all slates on a given date.",
    log_prints=True,
)
def dk_salaries_flow(
    game_date: str | None = None,
    force_refresh: bool = False,
) -> dict:
    """
    Flow for fetching DraftKings salary data.
    
    Runs daily at 8am to pull salary data for all available slates.
    Can also be triggered manually with --force-refresh to re-fetch.
    """
    return run_dk_salaries_task(
        game_date=game_date,
        force_refresh=force_refresh,
    )


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point (for local testing)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run live pipeline flows locally")
    parser.add_argument("flow", choices=["scrape", "score", "full", "dk-salaries"], help="Which flow to run")
    parser.add_argument("--date", help="Game date (YYYY-MM-DD), default=today")
    parser.add_argument("--no-sim", action="store_true", help="Skip simulation")
    parser.add_argument("--force-refresh", action="store_true", help="Force refresh (dk-salaries)")
    args = parser.parse_args()
    
    if args.flow == "scrape":
        live_scrape_flow(game_date=args.date)
    elif args.flow == "score":
        live_score_flow(game_date=args.date, run_sim=not args.no_sim)
    elif args.flow == "dk-salaries":
        dk_salaries_flow(game_date=args.date, force_refresh=args.force_refresh)
    else:
        live_pipeline_full_flow(game_date=args.date, run_sim=not args.no_sim)
