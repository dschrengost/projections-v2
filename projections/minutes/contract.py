"""Minutes eligibility contract and provenance stamps.

This module defines the canonical thresholds and contract semantics for minutes
allocation across all models and pipeline stages. Import from here to ensure
consistency.

Contract definitions:
- PLAY_THRESHOLD_MINUTES: Minutes >= this means "played at all" (not OUT/DNP)
- ROTATION_THRESHOLD_MINUTES: Minutes >= this means "in rotation" (meaningful PT)

Bump MINUTES_CONTRACT_VERSION when changing any semantic definition.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Contract thresholds (canonical definitions)
# ---------------------------------------------------------------------------

PLAY_THRESHOLD_MINUTES: float = 1.0
"""Minimum minutes to count as "played" (not OUT/DNP)."""

ROTATION_THRESHOLD_MINUTES: float = 5.0
"""Minimum minutes to count as "in rotation" (meaningful playing time)."""

# Legacy alias for backward compatibility
IN_ROTATION_THRESHOLD_MIN = ROTATION_THRESHOLD_MINUTES

# Contract versioning
MINUTES_CONTRACT_VERSION: int = 1
"""Bump when changing semantics of minutes artifacts."""


def minutes_contract_hash(*, version: int = MINUTES_CONTRACT_VERSION) -> str:
    """Stable hash for the minutes contract.

    Encodes both the version and threshold values so artifacts can be validated.
    """
    payload = {
        "minutes_contract_version": int(version),
        "play_threshold_minutes": float(PLAY_THRESHOLD_MINUTES),
        "rotation_threshold_minutes": float(ROTATION_THRESHOLD_MINUTES),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return digest


# ---------------------------------------------------------------------------
# Allocation mode enum
# ---------------------------------------------------------------------------

MinutesAllocMode = Literal[
    "rotation_set",  # Transformer set model (canonical)
    "rmh",  # Rotation Minutes Hurdle model
    "lgbm",  # LightGBM minutes_v1 model
    "baseline",  # Fallback/baseline (usually lgbm)
    "unknown",
]

FallbackMode = Literal[
    "fail_closed",  # Fail fast on missing inputs or pathology (prefer for production correctness)
    "degrade_loudly",  # Degrade gracefully but loudly (set degraded=True, emit warnings)
]


# ---------------------------------------------------------------------------
# Provenance stamp dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MinutesProvenance:
    """Provenance stamp for minutes artifacts.

    Captures everything needed to answer "what produced this artifact?" for
    incident response and debugging.
    """

    # Allocation metadata
    minutes_alloc_mode: MinutesAllocMode
    minutes_model_dir: str | None
    minutes_run_id: str
    game_date: str

    # Code version
    git_sha: str
    git_branch: str
    git_dirty: bool

    # Contract info
    contract_version: int
    contract_hash: str
    play_threshold_minutes: float
    rotation_threshold_minutes: float

    # Degradation info (non-empty if fallback was used)
    degraded: bool = False
    degraded_reason: str = ""

    # Timestamps
    generated_at: str = field(default_factory=lambda: _utc_now_iso())

    # Optional extras (e.g., blend_weight, guardrail summary)
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        d = asdict(self)
        # Ensure all values are JSON-serializable
        d["extras"] = dict(d.get("extras") or {})
        return d

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MinutesProvenance:
        """Reconstruct from dict (e.g., loaded from summary.json)."""
        # Handle missing fields gracefully
        return cls(
            minutes_alloc_mode=d.get("minutes_alloc_mode", "unknown"),
            minutes_model_dir=d.get("minutes_model_dir"),
            minutes_run_id=d.get("minutes_run_id", ""),
            game_date=d.get("game_date", ""),
            git_sha=d.get("git_sha", "unknown"),
            git_branch=d.get("git_branch", "unknown"),
            git_dirty=d.get("git_dirty", False),
            contract_version=d.get("contract_version", MINUTES_CONTRACT_VERSION),
            contract_hash=d.get("contract_hash", ""),
            play_threshold_minutes=d.get("play_threshold_minutes", PLAY_THRESHOLD_MINUTES),
            rotation_threshold_minutes=d.get("rotation_threshold_minutes", ROTATION_THRESHOLD_MINUTES),
            degraded=d.get("degraded", False),
            degraded_reason=d.get("degraded_reason", ""),
            generated_at=d.get("generated_at", ""),
            extras=d.get("extras") or {},
        )


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _git_info() -> tuple[str, str, bool]:
    """Get git SHA, branch, and dirty state from runtime_stamp if available."""
    try:
        from projections.runtime_stamp import _find_git_repo_root, _git_branch, _git_dirty, _git_sha

        repo_root = _find_git_repo_root()
        if repo_root:
            return _git_sha(repo_root), _git_branch(repo_root), _git_dirty(repo_root)
    except Exception:
        pass
    return "unknown", "unknown", False


def build_provenance(
    *,
    alloc_mode: MinutesAllocMode,
    model_dir: str | Path | None,
    run_id: str,
    game_date: str,
    degraded: bool = False,
    degraded_reason: str = "",
    extras: dict[str, Any] | None = None,
) -> MinutesProvenance:
    """Build a MinutesProvenance stamp with current environment info."""
    git_sha, git_branch, git_dirty = _git_info()

    return MinutesProvenance(
        minutes_alloc_mode=alloc_mode,
        minutes_model_dir=str(model_dir) if model_dir else None,
        minutes_run_id=run_id,
        game_date=game_date,
        git_sha=git_sha,
        git_branch=git_branch,
        git_dirty=git_dirty,
        contract_version=MINUTES_CONTRACT_VERSION,
        contract_hash=minutes_contract_hash(),
        play_threshold_minutes=PLAY_THRESHOLD_MINUTES,
        rotation_threshold_minutes=ROTATION_THRESHOLD_MINUTES,
        degraded=degraded,
        degraded_reason=degraded_reason,
        extras=extras or {},
    )


# ---------------------------------------------------------------------------
# Summary.json helpers
# ---------------------------------------------------------------------------


def load_provenance_from_summary(summary_path: Path) -> MinutesProvenance | None:
    """Load provenance from a summary.json file if present."""
    if not summary_path.exists():
        return None
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        provenance_data = data.get("provenance")
        if provenance_data:
            return MinutesProvenance.from_dict(provenance_data)
    except (json.JSONDecodeError, KeyError, TypeError):
        pass
    return None


def inject_provenance_into_summary(
    summary_path: Path,
    provenance: MinutesProvenance,
    *,
    merge: bool = True,
) -> None:
    """Inject provenance into an existing summary.json file.

    Args:
        summary_path: Path to summary.json
        provenance: Provenance stamp to inject
        merge: If True, merge with existing content; if False, overwrite
    """
    if summary_path.exists() and merge:
        try:
            existing = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
    else:
        existing = {}

    existing["provenance"] = provenance.to_dict()

    # Also inject top-level convenience fields for quick inspection
    existing["minutes_alloc_mode"] = provenance.minutes_alloc_mode
    existing["minutes_contract_version"] = provenance.contract_version
    existing["minutes_contract_hash"] = provenance.contract_hash
    if provenance.degraded:
        existing["minutes_degraded"] = True
        existing["minutes_degraded_reason"] = provenance.degraded_reason

    summary_path.write_text(json.dumps(existing, indent=2, sort_keys=True), encoding="utf-8")


# ---------------------------------------------------------------------------
# Artifact discovery helpers
# ---------------------------------------------------------------------------


def find_minutes_run_dirs(
    artifact_root: Path,
    game_date: str,
) -> list[Path]:
    """Find all run directories for a given date under the artifact root."""
    day_dir = artifact_root / game_date
    if not day_dir.exists():
        return []
    return sorted([p for p in day_dir.iterdir() if p.is_dir() and p.name.startswith("run=")])


def get_latest_run_dir(artifact_root: Path, game_date: str) -> Path | None:
    """Get the latest run directory for a date (by lexicographic sort of run_id)."""
    runs = find_minutes_run_dirs(artifact_root, game_date)
    return runs[-1] if runs else None


__all__ = [
    "PLAY_THRESHOLD_MINUTES",
    "ROTATION_THRESHOLD_MINUTES",
    "IN_ROTATION_THRESHOLD_MIN",
    "MINUTES_CONTRACT_VERSION",
    "minutes_contract_hash",
    "MinutesAllocMode",
    "FallbackMode",
    "MinutesProvenance",
    "build_provenance",
    "load_provenance_from_summary",
    "inject_provenance_into_summary",
    "find_minutes_run_dirs",
    "get_latest_run_dir",
]
