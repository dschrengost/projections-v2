"""Runtime stamp utility for identifying what code/config is running.

This module provides a self-identifying stamp for each Prefect run, CLI invocation,
or API startup. It helps diagnose "which code is actually running" issues by capturing:
- Git commit SHA and dirty state
- Python executable, cwd, hostname, user
- Config file paths and content hash
- Extracted run IDs from key config files

Environment variables:
- PROJECTIONS_ALLOW_DIRTY=1: Allow running with dirty git tree
- PROJECTIONS_RUNTIME_STAMP_STRICT=0: Disable all fail-fast checks (dev only)
"""

from __future__ import annotations

import getpass
import hashlib
import json
import logging
import os
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Default config files to track (relative to project root)
DEFAULT_CONFIG_PATHS: dict[str, str] = {
    "minutes_current_run": "config/minutes_current_run.json",
    "rates_current_run": "config/rates_current_run.json",
    "rotation_set_minutes_live": "config/rotation_set_minutes_live.json",
    "sim_v2_profiles": "config/sim_v2_profiles.json",
}

# PROD repository path - Prefect runs exclusively from here
PROD_REPO_PATH = Path("/home/daniel/prod/projections-v2")


@dataclass
class RuntimeStamp:
    """Immutable snapshot of runtime environment at startup."""

    git_sha: str
    git_dirty: bool
    git_repo_root: str | None
    git_branch: str
    python_executable: str
    python_version: str
    cwd: str
    hostname: str
    user: str
    config_paths: dict[str, str]
    config_hash: str
    run_ids: dict[str, str | None]
    timestamp: str
    entrypoint: str
    pid: int
    env_flags: dict[str, str] = field(default_factory=dict)

    def to_json_line(self) -> str:
        """Single-line JSON for structured logging."""
        return json.dumps(asdict(self), separators=(",", ":"))

    def to_pretty_block(self) -> str:
        """Multi-line formatted block for human readability."""
        lines = [
            "=" * 60,
            "RUNTIME STAMP",
            "=" * 60,
            f"  Timestamp:    {self.timestamp}",
            f"  Entrypoint:   {self.entrypoint}",
            f"  Git SHA:      {self.git_sha}",
            f"  Git Branch:   {self.git_branch}",
            f"  Git Dirty:    {self.git_dirty}",
            f"  Repo Root:    {self.git_repo_root or 'N/A'}",
            f"  Python:       {self.python_executable}",
            f"  Python Ver:   {self.python_version}",
            f"  CWD:          {self.cwd}",
            f"  Hostname:     {self.hostname}",
            f"  User:         {self.user}",
            f"  PID:          {self.pid}",
            f"  Config Hash:  {self.config_hash[:16]}...",
            "  Config Paths:",
        ]
        for name, path in self.config_paths.items():
            lines.append(f"    {name}: {path}")
        lines.append("  Run IDs:")
        for key, val in self.run_ids.items():
            lines.append(f"    {key}: {val or 'N/A'}")
        if self.env_flags:
            lines.append("  Env Flags:")
            for key, val in self.env_flags.items():
                lines.append(f"    {key}: {val}")
        lines.append("=" * 60)
        return "\n".join(lines)


def _find_git_repo_root(start: Path | None = None) -> Path | None:
    """Find git repository root by walking up from start path."""
    if start is None:
        start = Path.cwd()
    current = start.resolve()
    for _ in range(100):  # Safety limit
        if (current / ".git").exists():
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent
    return None


def _git_sha(repo_root: Path) -> str:
    """Get current git commit SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]
    except Exception:
        pass
    return "unknown"


def _git_branch(repo_root: Path) -> str:
    """Get current git branch name."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _git_dirty(repo_root: Path) -> bool:
    """Check if git working tree has uncommitted changes."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return bool(result.stdout.strip())
    except Exception:
        pass
    return False


def _compute_config_hash(config_paths: dict[str, Path]) -> str:
    """Compute SHA-256 hash over sorted config file contents."""
    hasher = hashlib.sha256()
    for name in sorted(config_paths.keys()):
        path = config_paths[name]
        hasher.update(f"{name}:".encode())
        if path.exists():
            try:
                content = path.read_text(encoding="utf-8")
                # Normalize JSON to handle formatting differences
                try:
                    parsed = json.loads(content)
                    normalized = json.dumps(parsed, sort_keys=True, separators=(",", ":"))
                    hasher.update(normalized.encode())
                except json.JSONDecodeError:
                    hasher.update(content.encode())
            except Exception:
                hasher.update(b"<unreadable>")
        else:
            hasher.update(b"<missing>")
        hasher.update(b"\n")
    return hasher.hexdigest()


def _extract_run_ids(config_paths: dict[str, Path]) -> dict[str, str | None]:
    """Extract run_id values from known config files."""
    run_ids: dict[str, str | None] = {}

    # minutes_current_run.json
    minutes_path = config_paths.get("minutes_current_run")
    if minutes_path and minutes_path.exists():
        try:
            data = json.loads(minutes_path.read_text(encoding="utf-8"))
            run_ids["minutes_run_id"] = data.get("run_id")
            run_ids["minutes_bundle_dir"] = data.get("bundle_dir")
        except Exception:
            pass

    # rates_current_run.json
    rates_path = config_paths.get("rates_current_run")
    if rates_path and rates_path.exists():
        try:
            data = json.loads(rates_path.read_text(encoding="utf-8"))
            run_ids["rates_run_id"] = data.get("run_id")
        except Exception:
            pass

    # rotation_set_minutes_live.json
    rotation_path = config_paths.get("rotation_set_minutes_live")
    if rotation_path and rotation_path.exists():
        try:
            data = json.loads(rotation_path.read_text(encoding="utf-8"))
            run_ids["rotation_set_enabled"] = str(data.get("enabled", False))
            run_ids["rotation_set_model_dir"] = data.get("model_dir")
        except Exception:
            pass

    return run_ids


def collect_runtime_stamp(
    *,
    entrypoint: str,
    config_paths: dict[str, Path | str] | None = None,
    project_root: Path | None = None,
) -> RuntimeStamp:
    """Collect runtime stamp with all environment info.

    Args:
        entrypoint: Identifier for the entrypoint (e.g., "prefect", "cli:score_minutes_v1", "api")
        config_paths: Optional dict of config name -> path. If None, uses defaults.
        project_root: Optional project root path. If None, detected from git.

    Returns:
        RuntimeStamp with all captured info.
    """
    # Find git repo root
    repo_root = _find_git_repo_root(project_root)

    # Resolve config paths
    resolved_configs: dict[str, Path] = {}
    if config_paths is not None:
        for name, path in config_paths.items():
            resolved_configs[name] = Path(path) if isinstance(path, str) else path
    elif repo_root:
        for name, rel_path in DEFAULT_CONFIG_PATHS.items():
            resolved_configs[name] = repo_root / rel_path

    # Collect env flags
    env_flags: dict[str, str] = {}
    for key in ("PROJECTIONS_ALLOW_DIRTY", "PROJECTIONS_RUNTIME_STAMP_STRICT", "PROJECTIONS_DATA_ROOT"):
        val = os.environ.get(key)
        if val is not None:
            env_flags[key] = val

    return RuntimeStamp(
        git_sha=_git_sha(repo_root) if repo_root else "unknown",
        git_dirty=_git_dirty(repo_root) if repo_root else False,
        git_repo_root=str(repo_root) if repo_root else None,
        git_branch=_git_branch(repo_root) if repo_root else "unknown",
        python_executable=sys.executable,
        python_version=sys.version.split()[0],
        cwd=os.getcwd(),
        hostname=socket.gethostname(),
        user=getpass.getuser(),
        config_paths={name: str(path) for name, path in resolved_configs.items()},
        config_hash=_compute_config_hash(resolved_configs),
        run_ids=_extract_run_ids(resolved_configs),
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        entrypoint=entrypoint,
        pid=os.getpid(),
        env_flags=env_flags,
    )


def enforce_clean_tree(*, repo_root: Path | None = None) -> None:
    """Fail-fast if git working tree is dirty and strict mode is enabled.

    Raises:
        RuntimeError: If tree is dirty and PROJECTIONS_ALLOW_DIRTY is not set.

    Does nothing if:
        - PROJECTIONS_ALLOW_DIRTY=1 is set
        - PROJECTIONS_RUNTIME_STAMP_STRICT=0 is set
        - Not in a git repository
    """
    # Check if strict mode is disabled
    if os.environ.get("PROJECTIONS_RUNTIME_STAMP_STRICT", "1") == "0":
        return

    # Check if dirty is explicitly allowed
    if os.environ.get("PROJECTIONS_ALLOW_DIRTY", "0") == "1":
        return

    # Find repo root
    root = _find_git_repo_root(repo_root)
    if root is None:
        return  # Not in a git repo, skip check

    if _git_dirty(root):
        raise RuntimeError(
            "Git working tree is dirty. Production runs must use committed code.\n"
            "Options:\n"
            "  1. Commit your changes: git add -A && git commit -m 'wip'\n"
            "  2. Set PROJECTIONS_ALLOW_DIRTY=1 to bypass (dev only)\n"
            "  3. Set PROJECTIONS_RUNTIME_STAMP_STRICT=0 to disable all checks"
        )


def is_prod_execution(repo_root: Path | None = None) -> bool:
    """Check if we're executing from the PROD repository.
    
    Returns True if the current execution context is the PROD checkout
    at /home/daniel/prod/projections-v2.
    """
    if repo_root is None:
        repo_root = _find_git_repo_root()
    if repo_root is None:
        # No git repo found - check CWD
        try:
            return Path.cwd().is_relative_to(PROD_REPO_PATH) or Path.cwd() == PROD_REPO_PATH
        except (ValueError, OSError):
            return False
    try:
        return repo_root == PROD_REPO_PATH or repo_root.is_relative_to(PROD_REPO_PATH)
    except (ValueError, OSError):
        return str(repo_root) == str(PROD_REPO_PATH)


def enforce_prod_sanity(*, repo_root: Path | None = None) -> None:
    """Enforce strict sanity checks for PROD execution.
    
    In PROD (detected via CWD matching PROD_REPO_PATH):
    - .deploy_info must exist (proof of proper deployment)
    - If git repo exists somehow, it must NOT be dirty
    
    PROD intentionally does NOT have a .git directory (excluded by rsync),
    so we validate deployment via .deploy_info instead of git.
    
    DEV execution is not affected by this function.
    
    Raises:
        RuntimeError: If any PROD invariant is violated.
    """
    # Check if we're in PROD by CWD (since PROD has no .git)
    try:
        cwd = Path.cwd()
        in_prod = cwd == PROD_REPO_PATH or cwd.is_relative_to(PROD_REPO_PATH)
    except (ValueError, OSError):
        in_prod = False
    
    if not in_prod:
        return  # DEV execution - no strict checks
    
    # PROD-specific checks (cannot be bypassed)
    deploy_info = PROD_REPO_PATH / ".deploy_info"
    if not deploy_info.exists():
        raise RuntimeError(
            "PROD execution requires .deploy_info file (proof of deployment).\n"
            f"Expected at: {deploy_info}\n"
            "Run scripts/deploy/deploy_live.sh from DEV to deploy properly."
        )
    
    # If there's somehow a git repo in PROD, ensure it's not dirty
    # (this shouldn't happen since rsync excludes .git, but safety check)
    root = _find_git_repo_root()
    if root is not None and _git_dirty(root):
        raise RuntimeError(
            "PROD tree is dirty - this should never happen.\n"
            f"PROD repo: {root}\n"
            "The PROD directory should only be updated via deploy_live.sh.\n"
            "Re-run the deploy script from DEV to restore clean state."
        )


def validate_config_paths(
    config_paths: dict[str, Path | str],
    *,
    required: list[str] | None = None,
) -> list[str]:
    """Validate that config files exist and are readable.

    Args:
        config_paths: Dict of config name -> path
        required: List of config names that must exist. If None, all are required.

    Returns:
        List of warning messages for missing/unreadable configs.

    Raises:
        RuntimeError: In strict mode, if required configs are missing.
    """
    strict = os.environ.get("PROJECTIONS_RUNTIME_STAMP_STRICT", "1") != "0"
    warnings: list[str] = []
    required_set = set(required) if required else set(config_paths.keys())

    for name, path in config_paths.items():
        path = Path(path)
        expects_directory = str(name).endswith("_dir") or str(name).endswith("_root")
        if not path.exists():
            msg = f"Config file missing: {name} at {path}"
            warnings.append(msg)
            if strict and name in required_set:
                raise RuntimeError(
                    f"{msg}\n"
                    "Set PROJECTIONS_RUNTIME_STAMP_STRICT=0 to disable this check."
                )
        elif expects_directory:
            if not path.is_dir():
                msg = f"Config path is not a directory: {name} at {path}"
                warnings.append(msg)
                if strict and name in required_set:
                    raise RuntimeError(
                        f"{msg}\n"
                        "Set PROJECTIONS_RUNTIME_STAMP_STRICT=0 to disable this check."
                    )
        elif not path.is_file():
            msg = f"Config path is not a file: {name} at {path}"
            warnings.append(msg)
        else:
            try:
                path.read_text(encoding="utf-8")
            except Exception as e:
                msg = f"Config file unreadable: {name} at {path}: {e}"
                warnings.append(msg)
                if strict and name in required_set:
                    raise RuntimeError(
                        f"{msg}\n"
                        "Set PROJECTIONS_RUNTIME_STAMP_STRICT=0 to disable this check."
                    )

    return warnings


def log_runtime_stamp(
    *,
    entrypoint: str,
    config_paths: dict[str, Path | str] | None = None,
    project_root: Path | None = None,
    logger: logging.Logger | Any | None = None,
    validate_configs: bool = True,
    required_configs: list[str] | None = None,
) -> RuntimeStamp:
    """Collect and log runtime stamp.

    This is the main entry point for wiring into flows/CLIs.

    Args:
        entrypoint: Identifier for the entrypoint
        config_paths: Optional config paths to track
        project_root: Optional project root
        logger: Logger instance (or Prefect logger with .info method)
        validate_configs: Whether to validate config files exist
        required_configs: List of required config names (default: ["minutes_current_run", "rates_current_run"])

    Returns:
        The collected RuntimeStamp
    """
    stamp = collect_runtime_stamp(
        entrypoint=entrypoint,
        config_paths=config_paths,
        project_root=project_root,
    )

    # Validate configs if requested
    if validate_configs and stamp.config_paths:
        resolved = {name: Path(path) for name, path in stamp.config_paths.items()}
        if required_configs is None:
            required_configs = ["minutes_current_run", "rates_current_run"]
        warnings = validate_config_paths(resolved, required=required_configs)
        for warning in warnings:
            if logger:
                logger.warning(warning)

    # Log the stamp
    if logger:
        # Single-line JSON for structured logging
        logger.info(f"[runtime-stamp] {stamp.to_json_line()}")
        # Pretty block for human readability
        logger.info(stamp.to_pretty_block())
    else:
        # Fall back to print
        print(f"[runtime-stamp] {stamp.to_json_line()}", file=sys.stderr)
        print(stamp.to_pretty_block(), file=sys.stderr)

    return stamp
