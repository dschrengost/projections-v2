"""
Live Pipeline Orchestrator

Replaces run_live_score.sh with pure Python for better error handling and debugging.
This is the single source of truth for the live scoring pipeline.

Usage:
    from projections.pipeline.live_orchestrator import run_live_pipeline
    result = run_live_pipeline(game_date="2025-12-27")
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from projections import paths

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for a live pipeline run."""

    game_date: str
    data_root: Path = field(default_factory=paths.get_data_root)
    season: int | None = None
    month: int | None = None

    # Run identification
    run_id: str | None = None
    run_as_of_ts: str | None = None

    # Feature flags
    run_scrape: bool = True
    run_sim: bool = True
    sim_profile: str = "sim_v3"
    sim_worlds: int = 20000

    # Scoring options
    reconcile_mode: str = "none"
    minutes_output_mode: str = "conditional"
    lock_buffer_minutes: int = 0

    # Ownership source: "internal" (LightGBM model) or "linestar" (commercial)
    ownership_source: str = "linestar"

    # Skip options
    skip_digest_check: bool = False
    skip_validation: bool = False
    force_run: bool = False

    def __post_init__(self):
        self.data_root = Path(self.data_root)
        if self.run_id is None:
            self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        if self.run_as_of_ts is None:
            self.run_as_of_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        if self.season is None:
            self.season = int(pd.Timestamp(self.game_date).year)
        if self.month is None:
            self.month = int(pd.Timestamp(self.game_date).month)

    @property
    def schedule_path(self) -> Path:
        return (
            self.data_root
            / "silver"
            / "schedule"
            / f"season={self.season}"
            / f"month={self.month:02d}"
            / "schedule.parquet"
        )


@dataclass
class StepResult:
    """Result of a pipeline step."""

    name: str
    success: bool
    duration_s: float
    rows_written: int = 0
    output_path: str | None = None
    error: str | None = None
    skipped: bool = False


@dataclass
class PipelineResult:
    """Result of the full pipeline run."""

    run_id: str
    game_date: str
    success: bool
    steps: list[StepResult] = field(default_factory=list)
    start_ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_ts: datetime | None = None

    @property
    def duration_s(self) -> float:
        if self.end_ts is None:
            return 0.0
        return (self.end_ts - self.start_ts).total_seconds()

    def add_step(self, step: StepResult):
        self.steps.append(step)
        if not step.success and not step.skipped:
            self.success = False


def _run_python_module(
    module: str,
    args: list[str],
    data_root: Path,
    timeout: int = 600,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """Run a Python module as subprocess with proper environment."""
    import os

    env = os.environ.copy()
    env["PROJECTIONS_DATA_ROOT"] = str(data_root)

    # Prefer uv-managed environment (systemd/PREFECT runners may not have deps on sys.executable).
    uv_bin = os.environ.get("UV_BIN")
    if uv_bin and not Path(uv_bin).exists():
        uv_bin = None
    uv_bin = uv_bin or (str(Path("/home/daniel/.local/bin/uv")) if Path("/home/daniel/.local/bin/uv").exists() else None)
    uv_bin = uv_bin or shutil.which("uv")

    cmd = [uv_bin, "run", "python", "-m", module, *args] if uv_bin else [sys.executable, "-m", module, *args]
    logger.info(f"Running: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(Path(__file__).parent.parent.parent),  # project root
    )

    if result.stdout:
        for line in result.stdout.strip().split("\n"):
            logger.info(f"  {line}")
    if result.stderr:
        for line in result.stderr.strip().split("\n"):
            if "warning" in line.lower():
                logger.warning(f"  {line}")
            else:
                logger.error(f"  {line}")

    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, cmd, result.stdout, result.stderr
        )

    return result


def _all_games_locked(cfg: PipelineConfig, buffer_minutes: int = 5) -> bool:
    """Check if all games for the date have already tipped (plus buffer).

    Returns True if there are no games left to score, False otherwise.
    """
    schedule_path = cfg.data_root / "silver" / "schedule" / f"season={cfg.season}"
    if not schedule_path.exists():
        return False  # Can't determine, assume not locked

    try:
        schedule = pd.read_parquet(schedule_path)
        today_games = schedule[schedule["game_date"] == cfg.game_date]
        if today_games.empty:
            return True  # No games scheduled

        tips = pd.to_datetime(today_games["tip_ts"], utc=True, errors="coerce")
        now = pd.Timestamp.now(tz="UTC")
        cutoff = now - pd.Timedelta(minutes=buffer_minutes)

        # All games locked if every tip_ts is before the cutoff
        unlocked = tips.isna() | (tips >= cutoff)
        return not unlocked.any()
    except Exception as e:
        logger.warning(f"Could not check game lock status: {e}")
        return False  # Assume not locked on error


def step_scrape(cfg: PipelineConfig) -> StepResult:
    """Step 1: Scrape live data (injuries, odds, roster, lineups)."""
    start = datetime.now(timezone.utc)
    name = "scrape"

    if not cfg.run_scrape:
        return StepResult(name=name, success=True, duration_s=0, skipped=True)

    try:
        args = [
            "--start", cfg.game_date,
            "--end", cfg.game_date,
            "--season", str(cfg.season),
            "--month", str(cfg.month),
            "--data-root", str(cfg.data_root),
            "--schedule", str(cfg.schedule_path),
        ]
        _run_python_module("projections.cli.live_pipeline", args, cfg.data_root)

        duration = (datetime.now(timezone.utc) - start).total_seconds()
        return StepResult(name=name, success=True, duration_s=duration)

    except Exception as e:
        duration = (datetime.now(timezone.utc) - start).total_seconds()
        logger.error(f"Scrape failed: {e}")
        return StepResult(name=name, success=False, duration_s=duration, error=str(e))


def step_dk_salaries(cfg: PipelineConfig) -> StepResult:
    """Step 1b: Fetch DraftKings salaries."""
    start = datetime.now(timezone.utc)
    name = "dk_salaries"

    try:
        args = ["--game-date", cfg.game_date]
        _run_python_module("scripts.dk.run_daily_salaries", args, cfg.data_root, check=False)

        duration = (datetime.now(timezone.utc) - start).total_seconds()
        return StepResult(name=name, success=True, duration_s=duration)

    except Exception as e:
        duration = (datetime.now(timezone.utc) - start).total_seconds()
        logger.warning(f"DK salaries failed (continuing): {e}")
        return StepResult(name=name, success=True, duration_s=duration, error=str(e))


def step_scrape_props(cfg: PipelineConfig) -> StepResult:
    """Step 1c: Scrape player props from RotoWire (non-blocking).

    Archives daily props to bronze layer for +EV analysis and historical tracking.
    """
    start = datetime.now(timezone.utc)
    name = "scrape_props"

    try:
        args = ["scrape", "--date", cfg.game_date, "--save", "--no-sample"]
        _run_python_module("projections.cli.scrape_props", args, cfg.data_root, check=False)

        duration = (datetime.now(timezone.utc) - start).total_seconds()
        return StepResult(name=name, success=True, duration_s=duration)

    except Exception as e:
        duration = (datetime.now(timezone.utc) - start).total_seconds()
        logger.warning(f"Props scrape failed (continuing): {e}")
        return StepResult(name=name, success=True, duration_s=duration, error=str(e))


def step_build_minutes_features(cfg: PipelineConfig) -> StepResult:
    """Step 2: Build minutes features from scraped data."""
    start = datetime.now(timezone.utc)
    name = "build_minutes_features"

    try:
        # Important: treat run_as_of_ts as the effective input cutoff *after* scraping completes.
        # The scrape step often writes inputs (Rotowire lineups, ESPN injuries, etc.) during the run,
        # so using a run_as_of_ts captured at pipeline start can incorrectly filter out fresh inputs.
        cfg.run_as_of_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        args = [
            "--date", cfg.game_date,
            "--run-as-of-ts", cfg.run_as_of_ts,
            "--run-id", cfg.run_id,
            "--lock-buffer-minutes", str(cfg.lock_buffer_minutes),
        ]
        _run_python_module("projections.cli.build_minutes_live", args, cfg.data_root)

        # Count output rows
        output_path = (
            cfg.data_root / "live" / "features_minutes_v1" / cfg.game_date
            / f"run={cfg.run_id}" / "features.parquet"
        )
        rows = len(pd.read_parquet(output_path)) if output_path.exists() else 0

        duration = (datetime.now(timezone.utc) - start).total_seconds()
        return StepResult(
            name=name, success=True, duration_s=duration,
            rows_written=rows, output_path=str(output_path)
        )

    except Exception as e:
        duration = (datetime.now(timezone.utc) - start).total_seconds()
        logger.error(f"Build minutes features failed: {e}")
        return StepResult(name=name, success=False, duration_s=duration, error=str(e))


def step_score_minutes(cfg: PipelineConfig) -> StepResult:
    """Step 3: Score minutes model to get predictions."""
    start = datetime.now(timezone.utc)
    name = "score_minutes"

    try:
        args = [
            "--date", cfg.game_date,
            "--mode", "live",
            "--run-id", cfg.run_id,
            "--reconcile-team-minutes", cfg.reconcile_mode,
            "--minutes-output", cfg.minutes_output_mode,
        ]
        _run_python_module("projections.cli.score_minutes_v1", args, cfg.data_root)

        # Copy to gold
        src = (
            cfg.data_root / "artifacts" / "minutes_v1" / "daily" / cfg.game_date
            / f"run={cfg.run_id}" / "minutes.parquet"
        )
        dst_dir = cfg.data_root / "gold" / "projections_minutes_v1" / f"game_date={cfg.game_date}"
        dst_run_dir = dst_dir / f"run={cfg.run_id}"

        rows = 0
        if src.exists():
            dst_run_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst_run_dir / "minutes.parquet")
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst_dir / "minutes.parquet")

            # Write latest_run.json
            (dst_dir / "latest_run.json").write_text(json.dumps({
                "run_id": cfg.run_id,
                "run_as_of_ts": cfg.run_as_of_ts,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }))

            rows = len(pd.read_parquet(src))

        duration = (datetime.now(timezone.utc) - start).total_seconds()
        return StepResult(
            name=name, success=True, duration_s=duration,
            rows_written=rows, output_path=str(dst_dir)
        )

    except Exception as e:
        duration = (datetime.now(timezone.utc) - start).total_seconds()
        logger.error(f"Score minutes failed: {e}")
        return StepResult(name=name, success=False, duration_s=duration, error=str(e))


def step_build_rates_features(cfg: PipelineConfig) -> StepResult:
    """Step 4: Build rates features from minutes features."""
    start = datetime.now(timezone.utc)
    name = "build_rates_features"

    try:
        args = [
            "--date", cfg.game_date,
            "--run-id", cfg.run_id,
            "--data-root", str(cfg.data_root),
            "--no-strict",
        ]
        _run_python_module("projections.cli.build_rates_features_live", args, cfg.data_root)

        duration = (datetime.now(timezone.utc) - start).total_seconds()
        return StepResult(name=name, success=True, duration_s=duration)

    except Exception as e:
        duration = (datetime.now(timezone.utc) - start).total_seconds()
        logger.error(f"Build rates features failed: {e}")
        return StepResult(name=name, success=False, duration_s=duration, error=str(e))


def step_score_rates(cfg: PipelineConfig) -> StepResult:
    """Step 5: Score rates model."""
    start = datetime.now(timezone.utc)
    name = "score_rates"

    try:
        args = [
            "--date", cfg.game_date,
            "--run-id", cfg.run_id,
            "--features-root", str(cfg.data_root / "live" / "features_rates_v1"),
            "--out-root", str(cfg.data_root / "gold" / "rates_v1_live"),
            "--no-strict",
        ]
        _run_python_module("projections.cli.score_rates_live", args, cfg.data_root)

        duration = (datetime.now(timezone.utc) - start).total_seconds()
        return StepResult(name=name, success=True, duration_s=duration)

    except Exception as e:
        duration = (datetime.now(timezone.utc) - start).total_seconds()
        logger.error(f"Score rates failed: {e}")
        return StepResult(name=name, success=False, duration_s=duration, error=str(e))


def step_run_sim(cfg: PipelineConfig) -> StepResult:
    """Step 6: Run Monte Carlo simulation for FPTS distributions."""
    start = datetime.now(timezone.utc)
    name = "run_sim"

    if not cfg.run_sim:
        return StepResult(name=name, success=True, duration_s=0, skipped=True)

    try:
        args = [
            "--run-date", cfg.game_date,
            "--profile-name", cfg.sim_profile,
            "--num-worlds", str(cfg.sim_worlds),
            "--data-root", str(cfg.data_root),
            "--run-id", cfg.run_id,
            "--run-as-of-ts", cfg.run_as_of_ts,
            "--minutes-run-id", cfg.run_id,
            "--rates-run-id", cfg.run_id,
        ]
        _run_python_module(
            "scripts.sim_v2.run_sim_live", args, cfg.data_root,
            timeout=1200, check=False  # sim can take a while, don't fail pipeline if it errors
        )

        duration = (datetime.now(timezone.utc) - start).total_seconds()
        return StepResult(name=name, success=True, duration_s=duration)

    except Exception as e:
        duration = (datetime.now(timezone.utc) - start).total_seconds()
        logger.warning(f"Sim failed (continuing): {e}")
        return StepResult(name=name, success=True, duration_s=duration, error=str(e))


def step_score_ownership(cfg: PipelineConfig) -> StepResult:
    """Step 7: Score ownership predictions.

    Supports two ownership sources:
    - "linestar": Fetch projected ownership from LineStar (commercial source)
    - "internal": Run internal LightGBM ownership model
    """
    start = datetime.now(timezone.utc)
    name = "score_ownership"

    try:
        args = [
            "--date", cfg.game_date,
            "--run-id", cfg.run_id,
            "--data-root", str(cfg.data_root),
        ]

        if cfg.ownership_source == "linestar":
            logger.info("Using LineStar for ownership projections")
            _run_python_module(
                "projections.cli.score_ownership_linestar", args, cfg.data_root,
                check=False  # ownership is optional
            )
        else:
            logger.info("Using internal model for ownership projections")
            _run_python_module(
                "projections.cli.score_ownership_live", args, cfg.data_root,
                check=False  # ownership is optional
            )

        duration = (datetime.now(timezone.utc) - start).total_seconds()
        return StepResult(name=name, success=True, duration_s=duration)

    except Exception as e:
        duration = (datetime.now(timezone.utc) - start).total_seconds()
        logger.warning(f"Ownership scoring failed (continuing): {e}")
        return StepResult(name=name, success=True, duration_s=duration, error=str(e))


def step_finalize_projections(cfg: PipelineConfig) -> StepResult:
    """Step 8: Merge all projections into unified artifact."""
    start = datetime.now(timezone.utc)
    name = "finalize_projections"

    try:
        # Find main draft group (largest slate)
        salaries_base = (
            cfg.data_root / "gold" / "dk_salaries" / "site=dk"
            / f"game_date={cfg.game_date}"
        )

        main_draft_group = None
        if salaries_base.exists():
            slates = []
            for dg_dir in salaries_base.glob("draft_group_id=*"):
                parquet_path = dg_dir / "salaries.parquet"
                if parquet_path.exists():
                    df = pd.read_parquet(parquet_path)
                    dg_id = int(dg_dir.name.split("=")[1])
                    slates.append((dg_id, len(df)))
            if slates:
                slates.sort(key=lambda x: (-x[1], x[0]))
                main_draft_group = str(slates[0][0])

        if not main_draft_group:
            logger.warning("No draft group found, skipping finalize")
            duration = (datetime.now(timezone.utc) - start).total_seconds()
            return StepResult(name=name, success=True, duration_s=duration, skipped=True)

        args = [
            "--date", cfg.game_date,
            "--run-id", cfg.run_id,
            "--minutes-run-id", cfg.run_id,
            "--sim-run-id", cfg.run_id,
            "--ownership-run-id", cfg.run_id,
            "--merge-locked-games",
            "--draft-group-id", main_draft_group,
            "--data-root", str(cfg.data_root),
        ]
        _run_python_module(
            "projections.cli.finalize_projections", args, cfg.data_root,
            check=False
        )

        duration = (datetime.now(timezone.utc) - start).total_seconds()
        return StepResult(name=name, success=True, duration_s=duration)

    except Exception as e:
        duration = (datetime.now(timezone.utc) - start).total_seconds()
        logger.warning(f"Finalize projections failed (continuing): {e}")
        return StepResult(name=name, success=True, duration_s=duration, error=str(e))


def step_validate_and_promote(cfg: PipelineConfig) -> StepResult:
    """Step 9: Validate run and promote to blessed if passing."""
    start = datetime.now(timezone.utc)
    name = "validate_promote"

    try:
        args = [
            "--run-id", cfg.run_id,
            "--game-date", cfg.game_date,
            "--data-root", str(cfg.data_root),
            "--promote",
        ]
        result = _run_python_module(
            "projections.pipeline.validation", args, cfg.data_root,
            check=False
        )

        promoted = "SUCCESS" in result.stdout
        duration = (datetime.now(timezone.utc) - start).total_seconds()
        return StepResult(
            name=name, success=True, duration_s=duration,
            error=None if promoted else "Validation failed, not promoted"
        )

    except Exception as e:
        duration = (datetime.now(timezone.utc) - start).total_seconds()
        logger.warning(f"Validation failed (continuing): {e}")
        return StepResult(name=name, success=True, duration_s=duration, error=str(e))


def run_live_pipeline(
    game_date: str | None = None,
    data_root: Path | str | None = None,
    run_scrape: bool = True,
    run_sim: bool = True,
    sim_profile: str = "sim_v3",
    sim_worlds: int = 20000,
    **kwargs,
) -> PipelineResult:
    """
    Run the complete live pipeline.

    This is the main entry point that orchestrates all steps.

    Args:
        game_date: Target game date (YYYY-MM-DD), defaults to today
        data_root: Data root path, defaults to PROJECTIONS_DATA_ROOT
        run_scrape: Whether to run scrape step
        run_sim: Whether to run simulation
        sim_profile: Simulation profile name
        sim_worlds: Number of simulation worlds
        **kwargs: Additional config options

    Returns:
        PipelineResult with success status and step details
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    # Build config
    if game_date is None:
        game_date = datetime.now().strftime("%Y-%m-%d")
    if data_root is None:
        data_root = paths.get_data_root()

    cfg = PipelineConfig(
        game_date=game_date,
        data_root=Path(data_root),
        run_scrape=run_scrape,
        run_sim=run_sim,
        sim_profile=sim_profile,
        sim_worlds=sim_worlds,
        **kwargs,
    )

    logger.info(f"Starting live pipeline for {cfg.game_date} (run_id={cfg.run_id})")

    # Early exit if all games have already tipped
    if _all_games_locked(cfg):
        logger.info("All games have tipped - nothing left to score. Skipping pipeline.")
        result = PipelineResult(run_id=cfg.run_id, game_date=cfg.game_date, success=True)
        result.end_ts = datetime.now(timezone.utc)
        return result

    result = PipelineResult(run_id=cfg.run_id, game_date=cfg.game_date, success=True)

    # Step 0a: DK Salaries (non-blocking)
    logger.info("=" * 60)
    logger.info("STEP 0a: DK Salaries")
    result.add_step(step_dk_salaries(cfg))

    # Step 0b: Scrape Props (non-blocking)
    logger.info("=" * 60)
    logger.info("STEP 0b: Scrape Props")
    result.add_step(step_scrape_props(cfg))

    # Step 1: Scrape
    logger.info("=" * 60)
    logger.info("STEP 1: Scrape")
    step = step_scrape(cfg)
    result.add_step(step)
    if not step.success:
        logger.error("Scrape failed, aborting pipeline")
        result.end_ts = datetime.now(timezone.utc)
        return result

    # Advance run_as_of_ts to reflect completed scraping. This timestamp is used as the ceiling for
    # "as-of" selection in later steps and should not predate the inputs we just wrote.
    cfg.run_as_of_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    # Step 2: Build minutes features
    logger.info("=" * 60)
    logger.info("STEP 2: Build Minutes Features")
    step = step_build_minutes_features(cfg)
    result.add_step(step)
    if not step.success:
        logger.error("Build minutes features failed, aborting pipeline")
        result.end_ts = datetime.now(timezone.utc)
        return result

    # Step 3: Score minutes
    logger.info("=" * 60)
    logger.info("STEP 3: Score Minutes")
    step = step_score_minutes(cfg)
    result.add_step(step)
    if not step.success:
        logger.error("Score minutes failed, aborting pipeline")
        result.end_ts = datetime.now(timezone.utc)
        return result

    # Step 4: Build rates features
    logger.info("=" * 60)
    logger.info("STEP 4: Build Rates Features")
    step = step_build_rates_features(cfg)
    result.add_step(step)
    if not step.success:
        logger.error("Build rates features failed, aborting pipeline")
        result.end_ts = datetime.now(timezone.utc)
        return result

    # Step 5: Score rates
    logger.info("=" * 60)
    logger.info("STEP 5: Score Rates")
    step = step_score_rates(cfg)
    result.add_step(step)
    if not step.success:
        logger.error("Score rates failed, aborting pipeline")
        result.end_ts = datetime.now(timezone.utc)
        return result

    # Step 6: Run simulation (non-blocking)
    logger.info("=" * 60)
    logger.info("STEP 6: Run Simulation")
    result.add_step(step_run_sim(cfg))

    # Step 7: Score ownership (non-blocking)
    logger.info("=" * 60)
    logger.info("STEP 7: Score Ownership")
    result.add_step(step_score_ownership(cfg))

    # Step 8: Finalize projections (non-blocking)
    logger.info("=" * 60)
    logger.info("STEP 8: Finalize Projections")
    result.add_step(step_finalize_projections(cfg))

    # Step 9: Validate and promote to blessed
    logger.info("=" * 60)
    logger.info("STEP 9: Validate & Promote")
    result.add_step(step_validate_and_promote(cfg))

    result.end_ts = datetime.now(timezone.utc)
    logger.info("=" * 60)
    logger.info(f"Pipeline {'SUCCEEDED' if result.success else 'FAILED'} in {result.duration_s:.1f}s")

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run live pipeline")
    parser.add_argument("--date", help="Game date (YYYY-MM-DD)")
    parser.add_argument("--no-scrape", action="store_true", help="Skip scrape step")
    parser.add_argument("--no-sim", action="store_true", help="Skip simulation")
    parser.add_argument("--sim-profile", default="sim_v3", help="Simulation profile")
    parser.add_argument("--sim-worlds", type=int, default=20000, help="Number of sim worlds")
    parser.add_argument(
        "--ownership-source",
        choices=["linestar", "internal"],
        default="linestar",
        help="Ownership source: 'linestar' (commercial) or 'internal' (LightGBM model)",
    )
    args = parser.parse_args()

    result = run_live_pipeline(
        game_date=args.date,
        run_scrape=not args.no_scrape,
        run_sim=not args.no_sim,
        sim_profile=args.sim_profile,
        sim_worlds=args.sim_worlds,
        ownership_source=args.ownership_source,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    for step in result.steps:
        status = "SKIP" if step.skipped else ("OK" if step.success else "FAIL")
        print(f"  [{status:4s}] {step.name}: {step.duration_s:.1f}s")
        if step.error:
            print(f"         Error: {step.error[:80]}")

    sys.exit(0 if result.success else 1)
