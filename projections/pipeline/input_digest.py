"""Input digest system for run deduplication.

Computes a SHA256 hash of input artifact states to detect when inputs have changed.
If digest is unchanged from the last run, the pipeline can skip re-running.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class InputState:
    """Captures the state of an input artifact."""
    
    path: str
    exists: bool
    row_count: int | None = None
    as_of_ts: str | None = None
    mtime: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class InputDigest:
    """Full digest of all input artifacts for a run."""
    
    game_date: str
    computed_at: str
    digest_sha256: str
    inputs: dict[str, dict[str, Any]]
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
    
    def save(self, path: Path) -> None:
        """Save digest to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "InputDigest | None":
        """Load digest from JSON file, returns None if not found or invalid."""
        if not path.exists():
            return None
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            return cls(
                game_date=data["game_date"],
                computed_at=data["computed_at"],
                digest_sha256=data["digest_sha256"],
                inputs=data["inputs"],
            )
        except (json.JSONDecodeError, KeyError, TypeError):
            return None


def _get_parquet_state(path: Path, time_col: str = "as_of_ts") -> InputState:
    """Get state of a parquet file/directory."""
    if not path.exists():
        return InputState(path=str(path), exists=False)
    
    try:
        # Handle both single files and partitioned directories
        if path.is_dir():
            parquet_files = list(path.glob("**/*.parquet"))
            if not parquet_files:
                return InputState(path=str(path), exists=False)
            # Use the most recently modified parquet file
            newest = max(parquet_files, key=lambda p: p.stat().st_mtime)
            df = pd.read_parquet(newest)
            mtime = datetime.fromtimestamp(newest.stat().st_mtime, tz=timezone.utc).isoformat()
        else:
            df = pd.read_parquet(path)
            mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
        
        # Get the latest as_of_ts if available
        as_of_ts = None
        if time_col in df.columns:
            ts_series = pd.to_datetime(df[time_col], utc=True, errors="coerce").dropna()
            if not ts_series.empty:
                as_of_ts = ts_series.max().isoformat()
        
        return InputState(
            path=str(path),
            exists=True,
            row_count=len(df),
            as_of_ts=as_of_ts,
            mtime=mtime,
        )
    except Exception:
        return InputState(path=str(path), exists=False)


def compute_input_digest(
    game_date: str,
    data_root: Path,
    season: int,
) -> InputDigest:
    """
    Compute digest of all input artifacts for a given game date.
    
    Tracks:
    - injuries_snapshot (silver)
    - espn_injuries (silver)
    - roster_nightly (silver)
    - odds_snapshot (silver)
    - schedule (silver)
    - dk_salaries (gold)
    """
    inputs: dict[str, dict[str, Any]] = {}
    
    # Injuries snapshot
    injuries_path = data_root / "silver" / "injuries_snapshot" / f"season={season}"
    inputs["injuries_snapshot"] = _get_parquet_state(injuries_path).to_dict()
    
    # ESPN injuries (date-partitioned)
    espn_path = data_root / "silver" / "espn_injuries" / f"date={game_date}"
    inputs["espn_injuries"] = _get_parquet_state(espn_path).to_dict()
    
    # Roster nightly
    roster_path = data_root / "silver" / "roster_nightly" / f"season={season}"
    inputs["roster_nightly"] = _get_parquet_state(roster_path).to_dict()
    
    # Odds snapshot
    odds_path = data_root / "silver" / "odds_snapshot" / f"season={season}"
    inputs["odds_snapshot"] = _get_parquet_state(odds_path).to_dict()
    
    # Schedule
    month = int(game_date.split("-")[1])
    schedule_path = data_root / "silver" / "schedule" / f"season={season}" / f"month={month:02d}"
    inputs["schedule"] = _get_parquet_state(schedule_path).to_dict()
    
    # DK salaries
    salaries_path = data_root / "gold" / "dk_salaries" / "site=dk" / f"game_date={game_date}"
    inputs["dk_salaries"] = _get_parquet_state(salaries_path).to_dict()
    
    # Compute stable hash from inputs
    # We use as_of_ts when available (captures data changes), fall back to mtime
    hash_input = json.dumps(inputs, sort_keys=True)
    digest = hashlib.sha256(hash_input.encode()).hexdigest()[:16]
    
    return InputDigest(
        game_date=game_date,
        computed_at=datetime.now(timezone.utc).isoformat(),
        digest_sha256=digest,
        inputs=inputs,
    )


def should_skip_run(
    game_date: str,
    data_root: Path,
    season: int,
    digest_cache_dir: Path | None = None,
) -> tuple[bool, str, InputDigest]:
    """
    Check if the pipeline should skip this run due to unchanged inputs.
    
    Returns:
        (should_skip, reason, current_digest)
    """
    current = compute_input_digest(game_date, data_root, season)
    
    # Determine cache location
    if digest_cache_dir is None:
        digest_cache_dir = data_root / "artifacts" / "digests"
    
    cache_path = digest_cache_dir / f"{game_date}_last_digest.json"
    previous = InputDigest.load(cache_path)
    
    if previous is None:
        return False, "no_previous_digest", current
    
    if previous.digest_sha256 == current.digest_sha256:
        return True, f"inputs_unchanged (digest={current.digest_sha256})", current
    
    # Find what changed
    changed = []
    for key in current.inputs:
        curr = current.inputs.get(key, {})
        prev = previous.inputs.get(key, {})
        
        # Compare as_of_ts first (most reliable indicator of data change)
        if curr.get("as_of_ts") != prev.get("as_of_ts"):
            changed.append(f"{key}:as_of_ts")
        elif curr.get("row_count") != prev.get("row_count"):
            changed.append(f"{key}:rows")
        elif curr.get("mtime") != prev.get("mtime"):
            changed.append(f"{key}:mtime")
    
    reason = f"inputs_changed ({', '.join(changed)})" if changed else "digest_mismatch"
    return False, reason, current


def save_digest_after_run(
    digest: InputDigest,
    data_root: Path,
    digest_cache_dir: Path | None = None,
) -> Path:
    """Save the digest after a successful run."""
    if digest_cache_dir is None:
        digest_cache_dir = data_root / "artifacts" / "digests"
    
    cache_path = digest_cache_dir / f"{digest.game_date}_last_digest.json"
    digest.save(cache_path)
    return cache_path


# CLI interface for shell scripts
if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Check if pipeline inputs have changed")
    parser.add_argument("--game-date", required=True, help="Game date (YYYY-MM-DD)")
    parser.add_argument("--data-root", required=True, help="Data root directory")
    parser.add_argument("--season", type=int, required=True, help="Season year")
    parser.add_argument("--save-on-change", action="store_true", 
                        help="Save digest if inputs changed (for post-run)")
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    should_skip, reason, digest = should_skip_run(
        args.game_date, data_root, args.season
    )
    
    if should_skip:
        print(f"SKIP:{reason}")
        sys.exit(0)
    else:
        print(f"RUN:{reason}")
        if args.save_on_change:
            save_path = save_digest_after_run(digest, data_root)
            print(f"SAVED:{save_path}")
        sys.exit(1)  # Exit 1 = should run pipeline
