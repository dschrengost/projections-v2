"""Run manifest for lineage tracking.

Each projection run generates a manifest.json with full provenance:
- Input artifacts (paths, timestamps, row counts)
- Pipeline config (model bundles, sim profile, seed)
- Output artifacts (paths, row counts)
- Timing information

This enables debugging "why did projections change?" by comparing
manifests between runs.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class ArtifactInfo:
    """Metadata for an input or output artifact."""
    path: str
    exists: bool = True
    row_count: int | None = None
    as_of_ts: str | None = None
    mtime: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class PipelineConfig:
    """Configuration used for the pipeline run."""
    minutes_bundle: str | None = None
    rotalloc_config_version: str | None = None
    sim_profile: str | None = None
    sim_worlds: int | None = None
    seed: int | None = None
    minutes_alloc_mode: str | None = None
    rates_run_id: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class TimingInfo:
    """Timing information for the run."""
    started_at: str | None = None
    completed_at: str | None = None
    duration_seconds: float | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class ValidationResult:
    """Result of validation checks."""
    passed: bool = False
    checks: dict[str, bool] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunManifest:
    """Full manifest for a projection run."""
    
    # Identifiers
    run_id: str
    game_date: str
    projections_run_id: str | None = None
    
    # Input digest (from Phase 2)
    input_digest: str | None = None
    
    # Input artifacts
    inputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    
    # Pipeline configuration
    config: dict[str, Any] = field(default_factory=dict)
    
    # Output artifacts
    outputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    
    # Timing
    timing: dict[str, Any] = field(default_factory=dict)
    
    # Validation (for blessed run workflow)
    validation: dict[str, Any] = field(default_factory=dict)
    
    # Run state
    state: str = "completed"  # "running", "completed", "failed", "blessed"
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "game_date": self.game_date,
            "projections_run_id": self.projections_run_id,
            "input_digest": self.input_digest,
            "inputs": self.inputs,
            "config": self.config,
            "outputs": self.outputs,
            "timing": self.timing,
            "validation": self.validation,
            "state": self.state,
        }
    
    def save(self, path: Path) -> None:
        """Save manifest to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "RunManifest | None":
        """Load manifest from JSON file."""
        if not path.exists():
            return None
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            return cls(
                run_id=data["run_id"],
                game_date=data["game_date"],
                projections_run_id=data.get("projections_run_id"),
                input_digest=data.get("input_digest"),
                inputs=data.get("inputs", {}),
                config=data.get("config", {}),
                outputs=data.get("outputs", {}),
                timing=data.get("timing", {}),
                validation=data.get("validation", {}),
                state=data.get("state", "completed"),
            )
        except (json.JSONDecodeError, KeyError, TypeError):
            return None


def get_artifact_info(path: Path, time_col: str = "as_of_ts") -> ArtifactInfo:
    """Get metadata for a parquet file or directory."""
    if not path.exists():
        return ArtifactInfo(path=str(path), exists=False)
    
    try:
        if path.is_dir():
            parquet_files = list(path.glob("**/*.parquet"))
            if not parquet_files:
                return ArtifactInfo(path=str(path), exists=False)
            newest = max(parquet_files, key=lambda p: p.stat().st_mtime)
            df = pd.read_parquet(newest)
            mtime = datetime.fromtimestamp(newest.stat().st_mtime, tz=timezone.utc).isoformat()
        else:
            df = pd.read_parquet(path)
            mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
        
        as_of_ts = None
        if time_col in df.columns:
            ts_series = pd.to_datetime(df[time_col], utc=True, errors="coerce").dropna()
            if not ts_series.empty:
                as_of_ts = ts_series.max().isoformat()
        
        return ArtifactInfo(
            path=str(path),
            exists=True,
            row_count=len(df),
            as_of_ts=as_of_ts,
            mtime=mtime,
        )
    except Exception:
        return ArtifactInfo(path=str(path), exists=False)


def build_manifest_inputs(
    game_date: str,
    data_root: Path,
    season: int,
    run_id: str,
) -> dict[str, dict[str, Any]]:
    """Build inputs section of manifest."""
    inputs = {}
    
    # Injuries snapshot
    injuries_path = data_root / "silver" / "injuries_snapshot" / f"season={season}"
    inputs["injuries_snapshot"] = get_artifact_info(injuries_path).to_dict()
    
    # ESPN injuries
    espn_path = data_root / "silver" / "espn_injuries" / f"date={game_date}"
    inputs["espn_injuries"] = get_artifact_info(espn_path).to_dict()
    
    # Roster
    roster_path = data_root / "silver" / "roster_nightly" / f"season={season}"
    inputs["roster_nightly"] = get_artifact_info(roster_path).to_dict()
    
    # Odds
    odds_path = data_root / "silver" / "odds_snapshot" / f"season={season}"
    inputs["odds_snapshot"] = get_artifact_info(odds_path).to_dict()
    
    # Schedule
    month = int(game_date.split("-")[1])
    schedule_path = data_root / "silver" / "schedule" / f"season={season}" / f"month={month:02d}"
    inputs["schedule"] = get_artifact_info(schedule_path).to_dict()
    
    # DK salaries
    salaries_path = data_root / "gold" / "dk_salaries" / "site=dk" / f"game_date={game_date}"
    inputs["dk_salaries"] = get_artifact_info(salaries_path).to_dict()
    
    # Features (intermediate)
    features_path = data_root / "live" / "features_minutes_v1" / game_date / f"run={run_id}"
    inputs["features_minutes"] = get_artifact_info(features_path).to_dict()
    
    return inputs


def build_manifest_outputs(
    game_date: str,
    data_root: Path,
    run_id: str,
) -> dict[str, dict[str, Any]]:
    """Build outputs section of manifest."""
    outputs = {}
    
    # Minutes predictions
    minutes_path = data_root / "artifacts" / "minutes_v1" / "daily" / game_date / f"run={run_id}" / "minutes.parquet"
    outputs["minutes"] = get_artifact_info(minutes_path).to_dict()
    
    # Rates predictions
    rates_path = data_root / "gold" / "rates_v1_live" / game_date / f"run={run_id}" / "rates.parquet"
    outputs["rates"] = get_artifact_info(rates_path).to_dict()
    
    # Sim projections
    sim_path = data_root / "artifacts" / "sim_v2" / "worlds_fpts_v2" / f"game_date={game_date}" / f"run={run_id}" / "projections.parquet"
    outputs["sim_projections"] = get_artifact_info(sim_path).to_dict()
    
    # Unified projections
    final_path = data_root / "artifacts" / "projections" / game_date / f"run={run_id}" / "projections.parquet"
    outputs["projections"] = get_artifact_info(final_path).to_dict()
    
    return outputs


def build_manifest_config(
    config_path: Path | None = None,
    sim_profile: str = "today",
    sim_worlds: int = 25000,
) -> dict[str, Any]:
    """Build config section of manifest."""
    config = {
        "sim_profile": sim_profile,
        "sim_worlds": sim_worlds,
    }
    
    # Load minutes config
    if config_path is None:
        config_path = Path("config/minutes_current_run.json")
    
    if config_path.exists():
        try:
            with open(config_path, encoding="utf-8") as f:
                minutes_config = json.load(f)
            config["minutes_bundle"] = minutes_config.get("bundle_dir")
            config["minutes_alloc_mode"] = minutes_config.get("minutes_alloc_mode")
            config["rotalloc_bundle_dir"] = minutes_config.get("rotalloc_bundle_dir")
        except Exception:
            pass
    
    # Load rotalloc config version
    rotalloc_config_path = Path("config/rotalloc_production.json")
    if rotalloc_config_path.exists():
        try:
            with open(rotalloc_config_path, encoding="utf-8") as f:
                rotalloc_config = json.load(f)
            config["rotalloc_config_version"] = rotalloc_config.get("version")
        except Exception:
            pass
    
    # Load sim profile seed
    sim_profiles_path = Path("config/sim_v2_profiles.json")
    if sim_profiles_path.exists():
        try:
            with open(sim_profiles_path, encoding="utf-8") as f:
                profiles = json.load(f)
            profile_data = profiles.get("profiles", {}).get(sim_profile, {})
            config["seed"] = profile_data.get("seed")
        except Exception:
            pass
    
    return config


def create_run_manifest(
    run_id: str,
    game_date: str,
    data_root: Path,
    season: int,
    *,
    input_digest: str | None = None,
    started_at: str | None = None,
    completed_at: str | None = None,
    sim_profile: str = "today",
    sim_worlds: int = 25000,
) -> RunManifest:
    """Create a complete run manifest."""
    
    # Calculate duration if both times provided
    duration = None
    if started_at and completed_at:
        try:
            start = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
            end = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
            duration = (end - start).total_seconds()
        except Exception:
            pass
    
    manifest = RunManifest(
        run_id=run_id,
        game_date=game_date,
        projections_run_id=run_id,
        input_digest=input_digest,
        inputs=build_manifest_inputs(game_date, data_root, season, run_id),
        config=build_manifest_config(sim_profile=sim_profile, sim_worlds=sim_worlds),
        outputs=build_manifest_outputs(game_date, data_root, run_id),
        timing={
            "started_at": started_at,
            "completed_at": completed_at or datetime.now(timezone.utc).isoformat(),
            "duration_seconds": duration,
        },
        state="completed",
    )
    
    return manifest


def diff_manifests(a: RunManifest, b: RunManifest) -> dict[str, Any]:
    """Compare two manifests and return differences."""
    diff = {
        "run_a": a.run_id,
        "run_b": b.run_id,
        "input_changes": [],
        "config_changes": [],
        "output_changes": [],
    }
    
    # Compare inputs
    for key in set(a.inputs.keys()) | set(b.inputs.keys()):
        a_val = a.inputs.get(key, {})
        b_val = b.inputs.get(key, {})
        
        if a_val.get("as_of_ts") != b_val.get("as_of_ts"):
            diff["input_changes"].append({
                "artifact": key,
                "field": "as_of_ts",
                "old": a_val.get("as_of_ts"),
                "new": b_val.get("as_of_ts"),
            })
        elif a_val.get("row_count") != b_val.get("row_count"):
            diff["input_changes"].append({
                "artifact": key,
                "field": "row_count",
                "old": a_val.get("row_count"),
                "new": b_val.get("row_count"),
            })
    
    # Compare config
    for key in set(a.config.keys()) | set(b.config.keys()):
        if a.config.get(key) != b.config.get(key):
            diff["config_changes"].append({
                "field": key,
                "old": a.config.get(key),
                "new": b.config.get(key),
            })
    
    # Compare outputs
    for key in set(a.outputs.keys()) | set(b.outputs.keys()):
        a_val = a.outputs.get(key, {})
        b_val = b.outputs.get(key, {})
        
        if a_val.get("row_count") != b_val.get("row_count"):
            diff["output_changes"].append({
                "artifact": key,
                "field": "row_count",
                "old": a_val.get("row_count"),
                "new": b_val.get("row_count"),
            })
    
    return diff


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate or view run manifest")
    parser.add_argument("--run-id", required=True, help="Run ID")
    parser.add_argument("--game-date", required=True, help="Game date (YYYY-MM-DD)")
    parser.add_argument("--data-root", default="/home/daniel/projections-data")
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--save", action="store_true", help="Save manifest to run directory")
    parser.add_argument("--diff", type=str, help="Compare with another run_id")
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    
    if args.diff:
        # Load and compare manifests
        manifest_a_path = data_root / "artifacts" / "projections" / args.game_date / f"run={args.run_id}" / "manifest.json"
        manifest_b_path = data_root / "artifacts" / "projections" / args.game_date / f"run={args.diff}" / "manifest.json"
        
        a = RunManifest.load(manifest_a_path)
        b = RunManifest.load(manifest_b_path)
        
        if not a or not b:
            print(f"Could not load manifests: {manifest_a_path.exists()=}, {manifest_b_path.exists()=}")
        else:
            diff = diff_manifests(a, b)
            print(json.dumps(diff, indent=2))
    else:
        # Create manifest
        manifest = create_run_manifest(
            args.run_id,
            args.game_date,
            data_root,
            args.season,
        )
        
        if args.save:
            save_path = data_root / "artifacts" / "projections" / args.game_date / f"run={args.run_id}" / "manifest.json"
            manifest.save(save_path)
            print(f"Saved to {save_path}")
        else:
            print(json.dumps(manifest.to_dict(), indent=2))
