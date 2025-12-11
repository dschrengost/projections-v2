"""QuickBuild optimizer service layer.

This module provides the service functions that:
1. Load and merge player pools from projections + DK salaries
2. Execute QuickBuild jobs with progress tracking
3. Manage job lifecycle (create, poll, retrieve results)
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import numpy as np
import yaml

from projections.dk.salaries_schema import dk_salaries_gold_path, normalize_positions
from projections.dk.slates import list_draft_groups_for_date
from projections.optimizer.quick_build import (
    QuickBuildConfig,
    QuickBuildResult,
    quick_build_pool,
)
from projections.optimizer.optimizer_types import OwnershipPenaltySettings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "optimizer.yaml"
_config_cache: Optional[Dict[str, Any]] = None


def load_optimizer_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load optimizer configuration from YAML."""
    global _config_cache
    if _config_cache is not None and path is None:
        return _config_cache

    config_path = path or _CONFIG_PATH
    if not config_path.exists():
        logger.warning("Optimizer config not found at %s; using defaults", config_path)
        return {}

    with open(config_path) as f:
        config = yaml.safe_load(f) or {}

    if path is None:
        _config_cache = config
    return config


def get_data_root() -> Path:
    """Get the projections data root directory."""
    return Path(os.environ.get("PROJECTIONS_DATA_ROOT", "/home/daniel/projections-data"))


# ---------------------------------------------------------------------------
# Player Pool Building
# ---------------------------------------------------------------------------


def load_projections_for_date(
    game_date: str,
    run_id: Optional[str] = None,
    data_root: Optional[Path] = None,
) -> pd.DataFrame:
    """Load projections from unified projections artifact or gold layer.

    Also merges sim_v2 FPTS projections if available.

    Returns DataFrame with columns:
        player_id, player_name, team_tricode, sim_dk_fpts_mean, pred_own_pct, etc.
    """
    root = data_root or get_data_root()
    df = None

    # Try unified projections artifact first (has sim + ownership)
    unified_dir = root / "artifacts" / "unified_projections" / f"game_date={game_date}"
    if unified_dir.exists():
        runs = sorted(unified_dir.glob("run=*"))
        if run_id:
            target_run = unified_dir / f"run={run_id}"
            if target_run.exists():
                runs = [target_run]
        if runs:
            parquet_files = list(runs[-1].glob("*.parquet"))
            if parquet_files:
                df = pd.read_parquet(parquet_files[0])
                logger.info(
                    "Loaded unified projections for %s from %s (%d rows)",
                    game_date,
                    runs[-1].name,
                    len(df),
                )

    # Fall back to gold projections_minutes_v1
    if df is None:
        gold_dir = root / "gold" / "projections_minutes_v1" / f"game_date={game_date}"
        if gold_dir.exists():
            parquet_files = list(gold_dir.glob("*.parquet"))
            if parquet_files:
                frames = [pd.read_parquet(p) for p in parquet_files]
                df = pd.concat(frames, ignore_index=True)
                logger.info("Loaded gold projections for %s (%d rows)", game_date, len(df))

    if df is None:
        raise FileNotFoundError(f"No projections found for {game_date}")

    # Check if we have FPTS data, if not try to merge from sim_v2
    fpts_cols = ["sim_dk_fpts_mean", "dk_fpts_mean", "proj_fpts", "fpts_mean"]
    has_fpts = any(c in df.columns and df[c].notna().any() for c in fpts_cols)

    if not has_fpts:
        sim_df = _load_sim_projections(game_date, root)
        if sim_df is not None and not sim_df.empty:
            # Merge sim projections
            join_keys = ["player_id"]
            if "game_id" in df.columns and "game_id" in sim_df.columns:
                join_keys.append("game_id")

            # Rename sim columns
            rename_map = {}
            for col in sim_df.columns:
                if col in join_keys:
                    continue
                if col == "dk_fpts_mean":
                    rename_map[col] = "sim_dk_fpts_mean"
                elif col == "dk_fpts_std":
                    rename_map[col] = "sim_dk_fpts_std"
                elif not col.startswith("sim_"):
                    rename_map[col] = f"sim_{col}"

            sim_df = sim_df.rename(columns=rename_map)

            # Merge
            df = df.merge(sim_df, on=join_keys, how="left", suffixes=("", "_sim"))
            logger.info(
                "Merged sim_v2 projections for %s (%d players with FPTS)",
                game_date,
                df["sim_dk_fpts_mean"].notna().sum() if "sim_dk_fpts_mean" in df.columns else 0,
            )

    return df


def _load_sim_projections(game_date: str, root: Path) -> Optional[pd.DataFrame]:
    """Load sim_v2 FPTS projections for a date."""
    sim_candidates = [
        root / "artifacts" / "sim_v2" / "projections" / f"game_date={game_date}" / "projections.parquet",
        root / "artifacts" / "sim_v2" / "projections" / f"date={game_date}" / "projections.parquet",
    ]

    for path in sim_candidates:
        if path.exists():
            try:
                return pd.read_parquet(path)
            except Exception as exc:
                logger.warning("Failed to load sim projections from %s: %s", path, exc)
                continue

    return None


def load_salaries_for_date(
    game_date: str,
    draft_group_id: int,
    site: str = "dk",
    data_root: Optional[Path] = None,
) -> pd.DataFrame:
    """Load DK salaries from gold layer.

    Returns DataFrame with columns:
        dk_player_id, display_name, positions, salary, team_abbrev, status
    """
    root = data_root or get_data_root()
    salaries_path = dk_salaries_gold_path(root, site, game_date, draft_group_id)

    if not salaries_path.exists():
        raise FileNotFoundError(f"Salaries not found: {salaries_path}")

    df = pd.read_parquet(salaries_path)
    logger.info(
        "Loaded salaries for %s draft_group=%d (%d players)",
        game_date,
        draft_group_id,
        len(df),
    )
    return df


def _normalize_name(val: object) -> str:
    """Normalize player name for fuzzy matching."""
    import unicodedata

    if val is None:
        return ""
    s = str(val).strip()
    # Strip accents
    nfkd = unicodedata.normalize("NFKD", s)
    ascii_only = nfkd.encode("ascii", "ignore").decode("ascii")
    return ascii_only.lower()


def _normalize_team(val: object) -> str:
    """Normalize team abbreviation."""
    if val is None:
        return ""
    return str(val).strip().upper()


def build_player_pool(
    game_date: str,
    draft_group_id: int,
    site: str = "dk",
    run_id: Optional[str] = None,
    data_root: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Build optimizer-ready player pool by merging projections with salaries.

    Returns list of player dicts with required QuickBuild fields:
        player_id, name, team, positions, salary, proj, own_proj, stddev, dk_id
    """
    root = data_root or get_data_root()

    # Load data sources
    proj_df = load_projections_for_date(game_date, run_id=run_id, data_root=root)
    sal_df = load_salaries_for_date(game_date, draft_group_id, site=site, data_root=root)

    # Prepare join keys
    proj_df = proj_df.copy()
    sal_df = sal_df.copy()

    # Normalize names for fuzzy join
    proj_name_col = next(
        (c for c in ["player_name", "name", "display_name"] if c in proj_df.columns),
        None,
    )
    sal_name_col = next(
        (c for c in ["display_name", "name", "player_name"] if c in sal_df.columns),
        None,
    )
    proj_team_col = next(
        (c for c in ["team_tricode", "team_abbrev", "team"] if c in proj_df.columns),
        None,
    )
    sal_team_col = next(
        (c for c in ["team_abbrev", "team_tricode", "team"] if c in sal_df.columns),
        None,
    )

    if not all([proj_name_col, sal_name_col, proj_team_col, sal_team_col]):
        raise ValueError(
            f"Missing required columns for join. "
            f"proj cols: {proj_df.columns.tolist()}, sal cols: {sal_df.columns.tolist()}"
        )

    proj_df["__join_name"] = proj_df[proj_name_col].apply(_normalize_name)
    proj_df["__join_team"] = proj_df[proj_team_col].apply(_normalize_team)
    sal_df["__join_name"] = sal_df[sal_name_col].apply(_normalize_name)
    sal_df["__join_team"] = sal_df[sal_team_col].apply(_normalize_team)

    # Merge on name + team
    merged = proj_df.merge(
        sal_df,
        on=["__join_name", "__join_team"],
        how="inner",
        suffixes=("", "_sal"),
    )

    logger.info(
        "Player pool merge: %d projections x %d salaries → %d matched",
        len(proj_df),
        len(sal_df),
        len(merged),
    )

    if len(merged) == 0:
        raise ValueError("No players matched between projections and salaries")

    # Build player pool list
    pool: List[Dict[str, Any]] = []

    # Identify projection column
    proj_col = next(
        (
            c
            for c in ["sim_dk_fpts_mean", "dk_fpts_mean", "proj_fpts", "fpts_mean", "proj"]
            if c in merged.columns
        ),
        None,
    )
    if not proj_col:
        raise ValueError("No projection column found in merged data")

    # Identify ownership column
    own_col = next(
        (c for c in ["pred_own_pct", "own_proj", "ownership"] if c in merged.columns),
        None,
    )

    # Identify stddev column
    stddev_col = next(
        (c for c in ["sim_dk_fpts_std", "stddev", "fpts_std"] if c in merged.columns),
        None,
    )

    for _, row in merged.iterrows():
        # Get player_id (prefer projection's player_id, fall back to dk_player_id)
        player_id = row.get("player_id") or row.get("dk_player_id") or row.get("dk_player_id_sal")

        # Get positions (from salaries - prefer _sal suffix if present from merge)
        positions_raw = row.get("positions_sal") if "positions_sal" in row else row.get("positions")
        if positions_raw is None:
            positions_raw = []
        
        if isinstance(positions_raw, np.ndarray):
            positions_raw = positions_raw.tolist()
        
        if isinstance(positions_raw, str):
            positions = normalize_positions(positions_raw)
        elif hasattr(positions_raw, "__iter__"):
            positions = normalize_positions(list(positions_raw))
        else:
            positions = []

        if not positions:
            logger.warning("Player %s has no positions, skipping", player_id)
            continue

        # Get salary
        salary = row.get("salary") or row.get("salary_sal") or 0
        if pd.isna(salary) or salary <= 0:
            logger.warning("Player %s has invalid salary %s, skipping", player_id, salary)
            continue

        # Get projection
        proj = row.get(proj_col)
        if pd.isna(proj) or proj <= 0:
            logger.debug("Player %s has no projection, skipping", player_id)
            continue

        player = {
            "player_id": str(player_id),
            "name": row.get(proj_name_col) or row.get(sal_name_col) or str(player_id),
            "team": row.get(proj_team_col) or row.get(sal_team_col) or "UNK",
            "positions": positions,
            "salary": int(salary),
            "proj": float(proj),
            "dk_id": str(row.get("dk_player_id") or row.get("dk_player_id_sal") or ""),
        }

        # Optional fields
        if own_col and pd.notna(row.get(own_col)):
            player["own_proj"] = float(row[own_col])
        if stddev_col and pd.notna(row.get(stddev_col)):
            player["stddev"] = float(row[stddev_col])

        pool.append(player)

    logger.info("Built player pool with %d optimizer-ready players", len(pool))
    return pool


# ---------------------------------------------------------------------------
# Job Management
# ---------------------------------------------------------------------------


@dataclass
class OptimizerJob:
    """Tracks state of a QuickBuild job."""

    job_id: str
    status: str  # pending, running, completed, failed
    created_at: datetime
    game_date: str
    draft_group_id: int
    site: str
    config: Dict[str, Any]

    # Progress tracking
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: int = 0
    target: int = 0

    # Results
    lineups: List[tuple] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "game_date": self.game_date,
            "draft_group_id": self.draft_group_id,
            "site": self.site,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "progress": self.progress,
            "target": self.target,
            "lineups_count": len(self.lineups),
            "wall_time_sec": (
                (self.completed_at - self.started_at).total_seconds()
                if self.completed_at and self.started_at
                else None
            ),
            "error": self.error,
        }


class JobStore:
    """Thread-safe in-memory job store."""

    def __init__(self, max_jobs: int = 100):
        self._jobs: Dict[str, OptimizerJob] = {}
        self._lock = threading.Lock()
        self._max_jobs = max_jobs

    def create(
        self,
        game_date: str,
        draft_group_id: int,
        site: str,
        config: Dict[str, Any],
        target: int,
    ) -> OptimizerJob:
        job = OptimizerJob(
            job_id=str(uuid.uuid4()),
            status="pending",
            created_at=datetime.utcnow(),
            game_date=game_date,
            draft_group_id=draft_group_id,
            site=site,
            config=config,
            target=target,
        )
        with self._lock:
            # Evict old jobs if at capacity
            if len(self._jobs) >= self._max_jobs:
                oldest = min(self._jobs.values(), key=lambda j: j.created_at)
                del self._jobs[oldest.job_id]
            self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> Optional[OptimizerJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **kwargs) -> Optional[OptimizerJob]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                for k, v in kwargs.items():
                    if hasattr(job, k):
                        setattr(job, k, v)
            return job

    def list_jobs(self, limit: int = 20) -> List[OptimizerJob]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
            return jobs[:limit]


# Global job store
_job_store = JobStore()


def get_job_store() -> JobStore:
    return _job_store


# ---------------------------------------------------------------------------
# QuickBuild Execution
# ---------------------------------------------------------------------------


def _build_qb_config(config: Dict[str, Any], defaults: Dict[str, Any]) -> QuickBuildConfig:
    """Build QuickBuildConfig from request config merged with defaults."""
    pool_defaults = defaults.get("pool", {})
    solver_defaults = defaults.get("solver", {})

    return QuickBuildConfig(
        builds=config.get("builds", pool_defaults.get("builds", 4)),
        per_build=config.get("per_build", pool_defaults.get("per_build", 6000)),
        max_pool=config.get("max_pool", pool_defaults.get("max_pool", 20000)),
        min_uniq=config.get("min_uniq", pool_defaults.get("min_uniq", 1)),
        jitter=config.get("jitter", pool_defaults.get("jitter", 5e-4)),
        near_dup_jaccard=config.get("near_dup_jaccard", pool_defaults.get("near_dup_jaccard", 0.0)),
        enum_enable=config.get("enum_enable", pool_defaults.get("enum_enable", True)),
        enum_time=config.get("enum_time", pool_defaults.get("enum_time", 20.0)),
        enum_warm_time=config.get("enum_warm_time", pool_defaults.get("enum_warm_time", 5.0)),
        timeout=config.get("timeout", solver_defaults.get("timeout", 0.6)),
        threads=config.get("threads", solver_defaults.get("threads", 1)),
        nogood_rate=config.get("nogood_rate", solver_defaults.get("nogood_rate", 20)),
        lineup_size=8 if config.get("site", "dk") == "dk" else 9,
    )


def _build_constraints(
    config: Dict[str, Any],
    site: str,
    defaults: Dict[str, Any],
) -> Dict[str, Any]:
    """Build constraints dict for QuickBuild."""
    site_defaults = defaults.get("constraints", {}).get(site, {})
    ownership_defaults = defaults.get("ownership_penalty", {})

    constraints = {
        "min_salary": config.get("min_salary", site_defaults.get("min_salary")),
        "max_salary": config.get("max_salary", site_defaults.get("max_salary")),
        "global_team_limit": config.get("global_team_limit", site_defaults.get("global_team_limit", 4)),
        "team_limits": config.get("team_limits", {}),
        "lock_ids": config.get("lock_ids", []),
        "ban_ids": config.get("ban_ids", []),
        "unique_players": config.get("unique_players", 1),
        "N_lineups": 1,  # QuickBuild handles this differently
    }

    # Ownership penalty
    if config.get("ownership_penalty_enabled", ownership_defaults.get("enabled", False)):
        constraints["ownership_penalty"] = {
            "enabled": True,
            "mode": config.get("ownership_mode", ownership_defaults.get("mode", "by_percent")),
            "weight_lambda": config.get("ownership_lambda", ownership_defaults.get("weight_lambda", 1.0)),
            "curve_type": config.get("ownership_curve", ownership_defaults.get("curve_type", "sigmoid")),
            "pivot_p0": ownership_defaults.get("pivot_p0", 0.20),
            "curve_alpha": ownership_defaults.get("curve_alpha", 2.0),
            "clamp_min": ownership_defaults.get("clamp_min", 0.01),
            "clamp_max": ownership_defaults.get("clamp_max", 0.80),
            "shrink_gamma": ownership_defaults.get("shrink_gamma", 1.0),
        }

    # Randomness
    if config.get("randomness_pct"):
        constraints["randomness_pct"] = config["randomness_pct"]

    return constraints


def run_quick_build(
    job: OptimizerJob,
    player_pool: List[Dict[str, Any]],
    on_progress: Optional[Callable[[int], None]] = None,
) -> QuickBuildResult:
    """Execute QuickBuild and update job state.

    This runs in a background thread.
    """
    store = get_job_store()
    defaults = load_optimizer_config()

    try:
        store.update(job.job_id, status="running", started_at=datetime.utcnow())

        qb_cfg = _build_qb_config(job.config, defaults)
        constraints = _build_constraints(job.config, job.site, defaults)

        logger.info(
            "Starting QuickBuild job %s: max_pool=%d, builds=%d",
            job.job_id,
            qb_cfg.max_pool,
            qb_cfg.worker_count,
        )

        result = quick_build_pool(
            slate=player_pool,
            site=job.site,
            constraints=constraints,
            qb_cfg=qb_cfg,
            run_id=job.job_id[:8],
        )

        store.update(
            job.job_id,
            status="completed",
            completed_at=datetime.utcnow(),
            lineups=result.lineups,
            stats=result.stats.to_dict(),
            progress=len(result.lineups),
        )

        # Auto-save build to disk
        try:
            save_build(job, result.lineups, result.stats.to_dict())
        except Exception as save_exc:
            logger.warning("Failed to save build %s to disk: %s", job.job_id, save_exc)

        logger.info(
            "QuickBuild job %s completed: %d lineups in %.1fs",
            job.job_id,
            len(result.lineups),
            result.stats.wall_time_s,
        )

        return result

    except Exception as exc:
        logger.exception("QuickBuild job %s failed: %s", job.job_id, exc)
        store.update(
            job.job_id,
            status="failed",
            completed_at=datetime.utcnow(),
            error=str(exc),
        )
        raise


def get_slates_for_date(game_date: str, slate_type: str = "all") -> List[Dict[str, Any]]:
    """Get available draft groups for a date.
    
    First tries the live DK API, then falls back to disk-based discovery
    from scraped gold salaries.
    """
    # Try live API first
    try:
        df = list_draft_groups_for_date(game_date, slate_type=slate_type)  # type: ignore
        if not df.empty:
            return df.to_dict(orient="records")
    except Exception as exc:
        logger.warning("Failed to fetch live slates for %s: %s", game_date, exc)

    # Fall back to discovering from gold salaries directory
    return _discover_slates_from_disk(game_date, slate_type)


def _discover_slates_from_disk(game_date: str, slate_type: str = "all") -> List[Dict[str, Any]]:
    """Discover available slates from gold dk_salaries directory."""
    root = get_data_root()
    salaries_base = root / "gold" / "dk_salaries" / "site=dk" / f"game_date={game_date}"
    
    if not salaries_base.exists():
        logger.debug("No gold salaries directory for %s", game_date)
        return []
    
    slates: List[Dict[str, Any]] = []
    
    for dg_dir in salaries_base.iterdir():
        if not dg_dir.is_dir() or not dg_dir.name.startswith("draft_group_id="):
            continue
        
        try:
            dg_id = int(dg_dir.name.split("=")[1])
        except (ValueError, IndexError):
            continue
        
        salaries_file = dg_dir / "salaries.parquet"
        if not salaries_file.exists():
            continue
        
        # Try to infer slate type from the bronze draftables if available
        inferred_type = "main"  # default
        bronze_path = root / "bronze" / "dk" / "draftables" / f"draftables_raw_{dg_id}.json"
        if bronze_path.exists():
            try:
                import json
                with open(bronze_path) as f:
                    payload = json.load(f)
                # Try to get contest name from draftables
                contests = payload.get("Contests", [])
                if contests:
                    name = contests[0].get("n", contests[0].get("ContestName", ""))
                    if name:
                        inferred_type = _infer_slate_type(name)
            except Exception:
                pass
        
        slate = {
            "game_date": game_date,
            "slate_type": inferred_type,
            "draft_group_id": dg_id,
            "n_contests": 0,  # Unknown from disk
            "earliest_start": None,
            "latest_start": None,
            "example_contest_name": f"Draft Group {dg_id}",
        }
        
        if slate_type == "all" or inferred_type == slate_type:
            slates.append(slate)
    
    logger.info("Discovered %d slates from disk for %s", len(slates), game_date)
    return sorted(slates, key=lambda s: s["draft_group_id"])


def _infer_slate_type(name: str) -> str:
    """Infer slate type from contest name."""
    name_lower = name.lower()
    if "turbo" in name_lower:
        return "turbo"
    if "late" in name_lower or "night" in name_lower:
        return "night"
    if "early" in name_lower:
        return "early"
    if "showdown" in name_lower or "single game" in name_lower:
        return "showdown"
    return "main"


# ─────────────────────────────────────────────────────────────────────────────
# Build Persistence - Save to projections-data/builds/optimizer
# ─────────────────────────────────────────────────────────────────────────────

def _builds_dir() -> Path:
    """Get the builds directory under projections-data."""
    return get_data_root() / "builds" / "optimizer"


def save_build(job: OptimizerJob, lineups: List[List[str]], stats: Dict[str, Any]) -> Path:
    """Save a completed build to disk.
    
    Saves to: projections-data/builds/optimizer/{game_date}/{job_id}.json
    """
    import json
    
    builds_root = _builds_dir() / job.game_date
    builds_root.mkdir(parents=True, exist_ok=True)
    
    build_file = builds_root / f"{job.job_id}.json"
    
    # Get config from job.config dict
    cfg = job.config or {}
    
    build_data = {
        "job_id": job.job_id,
        "game_date": job.game_date,
        "draft_group_id": job.draft_group_id,
        "site": job.site,
        "created_at": job.created_at.isoformat(),
        "completed_at": datetime.utcnow().isoformat(),
        "lineups_count": len(lineups),
        "config": cfg,
        "stats": stats,
        "lineups": [{"lineup_id": i, "player_ids": lu} for i, lu in enumerate(lineups)],
    }
    
    with open(build_file, "w") as f:
        json.dump(build_data, f, indent=2)
    
    logger.info("Saved build %s to %s (%d lineups)", job.job_id, build_file, len(lineups))
    return build_file


def list_saved_builds(game_date: str, draft_group_id: int | None = None) -> List[Dict[str, Any]]:
    """List saved builds for a game date.
    
    Returns summary info (no lineups) for each build.
    """
    import json
    
    builds_root = _builds_dir() / game_date
    if not builds_root.exists():
        return []
    
    builds = []
    for build_file in sorted(builds_root.glob("*.json"), reverse=True):
        try:
            with open(build_file, "r") as f:
                data = json.load(f)
            
            # Filter by draft_group_id if specified
            if draft_group_id is not None and data.get("draft_group_id") != draft_group_id:
                continue
            
            # Return summary without full lineups
            builds.append({
                "job_id": data["job_id"],
                "game_date": data["game_date"],
                "draft_group_id": data["draft_group_id"],
                "site": data["site"],
                "created_at": data["created_at"],
                "completed_at": data.get("completed_at"),
                "lineups_count": data["lineups_count"],
                "config": data.get("config", {}),
                "stats": data.get("stats", {}),
            })
        except Exception as e:
            logger.warning("Failed to read build file %s: %s", build_file, e)
            continue
    
    return builds


def load_saved_build(game_date: str, job_id: str) -> Dict[str, Any] | None:
    """Load a saved build including lineups."""
    import json
    
    build_file = _builds_dir() / game_date / f"{job_id}.json"
    if not build_file.exists():
        return None
    
    with open(build_file, "r") as f:
        return json.load(f)


def delete_saved_build(game_date: str, job_id: str) -> bool:
    """Delete a saved build."""
    build_file = _builds_dir() / game_date / f"{job_id}.json"
    if build_file.exists():
        build_file.unlink()
        logger.info("Deleted build %s", build_file)
        return True
    return False
