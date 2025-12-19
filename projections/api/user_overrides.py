"""User overrides for manual projection intervention.

This module provides:
1. Data models for player overrides (minutes, fpts, ownership, out status)
2. Atomic persistence with backup
3. Centralized apply_overrides() function for all consumers
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 1
DEFAULT_FPPM = 1.0
FPPM_MIN = 0.3
FPPM_MAX = 2.5
MIN_MINUTES_FOR_FPPM = 4.0
MIN_FPTS_FOR_FPPM = 1.0
DK_LINEUP_SIZE = 8  # For ownership normalization


def get_overrides_root() -> Path:
    """Get the root directory for user overrides."""
    data_root = Path(os.environ.get("PROJECTIONS_DATA_ROOT", "/home/daniel/projections-data"))
    return data_root / "user_overrides"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class PlayerOverride:
    """Override values for a single player."""
    
    player_id: str
    minutes: Optional[float] = None
    fpts: Optional[float] = None
    own: Optional[float] = None
    is_out: bool = False
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def has_any_override(self) -> bool:
        """Check if any override value is set."""
        return (
            self.minutes is not None
            or self.fpts is not None
            or self.own is not None
            or self.is_out
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "player_id": self.player_id,
            "minutes": self.minutes,
            "fpts": self.fpts,
            "own": self.own,
            "is_out": self.is_out,
            "updated_at": self.updated_at,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlayerOverride":
        return cls(
            player_id=str(data["player_id"]),
            minutes=data.get("minutes"),
            fpts=data.get("fpts"),
            own=data.get("own"),
            is_out=data.get("is_out", False),
            updated_at=data.get("updated_at", datetime.utcnow().isoformat()),
        )


@dataclass
class SlateOverrides:
    """All overrides for a single slate (date + draft_group)."""
    
    game_date: str
    draft_group_id: int
    overrides: Dict[str, PlayerOverride] = field(default_factory=dict)
    client_revision: int = 0
    schema_version: int = SCHEMA_VERSION
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def get_override(self, player_id: str) -> Optional[PlayerOverride]:
        """Get override for a player, or None if not set."""
        return self.overrides.get(str(player_id))
    
    def set_override(self, override: PlayerOverride) -> None:
        """Set or update an override for a player."""
        self.overrides[override.player_id] = override
        self.updated_at = datetime.utcnow().isoformat()
        self.client_revision += 1
    
    def remove_override(self, player_id: str) -> bool:
        """Remove an override. Returns True if it existed."""
        if str(player_id) in self.overrides:
            del self.overrides[str(player_id)]
            self.updated_at = datetime.utcnow().isoformat()
            self.client_revision += 1
            return True
        return False
    
    def clear_all(self) -> None:
        """Remove all overrides."""
        self.overrides.clear()
        self.updated_at = datetime.utcnow().isoformat()
        self.client_revision += 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "game_date": self.game_date,
            "draft_group_id": self.draft_group_id,
            "schema_version": self.schema_version,
            "client_revision": self.client_revision,
            "updated_at": self.updated_at,
            "overrides": {
                pid: override.to_dict()
                for pid, override in self.overrides.items()
            },
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SlateOverrides":
        overrides = {}
        for pid, override_data in data.get("overrides", {}).items():
            overrides[str(pid)] = PlayerOverride.from_dict(override_data)
        
        return cls(
            game_date=data["game_date"],
            draft_group_id=int(data["draft_group_id"]),
            overrides=overrides,
            client_revision=data.get("client_revision", 0),
            schema_version=data.get("schema_version", SCHEMA_VERSION),
            updated_at=data.get("updated_at", datetime.utcnow().isoformat()),
        )


# ---------------------------------------------------------------------------
# Persistence (Atomic JSON with Backup)
# ---------------------------------------------------------------------------


def _get_override_path(game_date: str, draft_group_id: int) -> Path:
    """Get path to override file for a slate."""
    root = get_overrides_root()
    return root / game_date / f"dg_{draft_group_id}.json"


def _get_backup_path(path: Path) -> Path:
    """Get backup path for an override file."""
    return path.with_suffix(".json.bak")


def load_slate_overrides(
    game_date: str,
    draft_group_id: int,
) -> SlateOverrides:
    """Load overrides for a slate, with fallback to backup on corruption."""
    path = _get_override_path(game_date, draft_group_id)
    backup_path = _get_backup_path(path)
    
    # Try primary file first
    if path.exists():
        try:
            with open(path) as f:
                data = json.load(f)
            return SlateOverrides.from_dict(data)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning("Failed to load overrides from %s: %s", path, e)
    
    # Try backup
    if backup_path.exists():
        try:
            with open(backup_path) as f:
                data = json.load(f)
            logger.info("Loaded overrides from backup: %s", backup_path)
            return SlateOverrides.from_dict(data)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning("Failed to load backup overrides: %s", e)
    
    # Return empty overrides
    return SlateOverrides(game_date=game_date, draft_group_id=draft_group_id)


def save_slate_overrides(
    overrides: SlateOverrides,
    expected_revision: Optional[int] = None,
) -> bool:
    """
    Save overrides atomically with backup.
    
    Args:
        overrides: The overrides to save
        expected_revision: If set, reject if current revision doesn't match
        
    Returns:
        True if saved successfully, False if revision conflict
    """
    path = _get_override_path(overrides.game_date, overrides.draft_group_id)
    backup_path = _get_backup_path(path)
    
    # Check for revision conflict
    if expected_revision is not None:
        current = load_slate_overrides(overrides.game_date, overrides.draft_group_id)
        if current.client_revision != expected_revision:
            logger.warning(
                "Revision conflict: expected %d, got %d",
                expected_revision,
                current.client_revision,
            )
            return False
    
    # Ensure directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Backup existing file if present
    if path.exists():
        try:
            shutil.copy2(path, backup_path)
        except OSError as e:
            logger.warning("Failed to create backup: %s", e)
    
    # Write atomically via temp file + rename
    try:
        fd, tmp_path = tempfile.mkstemp(
            dir=path.parent,
            prefix=".override_",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(overrides.to_dict(), f, indent=2)
            os.replace(tmp_path, path)
            logger.debug("Saved overrides to %s (rev=%d)", path, overrides.client_revision)
            return True
        except Exception:
            # Clean up temp file on error
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    except Exception as e:
        logger.error("Failed to save overrides: %s", e)
        raise


def delete_slate_overrides(game_date: str, draft_group_id: int) -> bool:
    """Delete overrides for a slate. Returns True if file existed."""
    path = _get_override_path(game_date, draft_group_id)
    if path.exists():
        path.unlink()
        return True
    return False


# ---------------------------------------------------------------------------
# FPPM Calculation
# ---------------------------------------------------------------------------


def compute_fppm(
    model_fpts: float,
    model_minutes: float,
    historical_fppm: Optional[float] = None,
    position_default: Optional[float] = None,
) -> tuple[float, bool]:
    """
    Compute FPPM (fantasy points per minute) for a player.
    
    Returns:
        (fppm_value, used_fallback) tuple
    """
    # Try model-based FPPM if we have enough data
    if model_minutes >= MIN_MINUTES_FOR_FPPM and model_fpts >= MIN_FPTS_FOR_FPPM:
        raw_fppm = model_fpts / model_minutes
        bounded_fppm = max(FPPM_MIN, min(FPPM_MAX, raw_fppm))
        return bounded_fppm, (bounded_fppm != raw_fppm)
    
    # Fallback chain
    if historical_fppm is not None and FPPM_MIN <= historical_fppm <= FPPM_MAX:
        return historical_fppm, True
    
    if position_default is not None:
        return position_default, True
    
    return DEFAULT_FPPM, True


def compute_effective_fpts(
    override_minutes: float,
    model_fpts: float,
    model_minutes: float,
    override_fpts: Optional[float] = None,
    historical_fppm: Optional[float] = None,
) -> tuple[float, bool]:
    """
    Compute effective FPTS given an override.
    
    If override_fpts is set, use it directly.
    Otherwise, compute from override_minutes * FPPM.
    
    Returns:
        (effective_fpts, used_fppm_fallback) tuple
    """
    # Direct FPTS override takes precedence
    if override_fpts is not None:
        return override_fpts, False
    
    # Compute from minutes override
    fppm, is_fallback = compute_fppm(model_fpts, model_minutes, historical_fppm)
    effective_fpts = override_minutes * fppm
    return effective_fpts, is_fallback


# ---------------------------------------------------------------------------
# Centralized Override Application
# ---------------------------------------------------------------------------


def apply_overrides(
    player_df: pd.DataFrame,
    overrides: SlateOverrides,
    ownership_mode: Literal["raw", "renormalize"] = "renormalize",
    lineup_size: int = DK_LINEUP_SIZE,
) -> pd.DataFrame:
    """
    Apply user overrides to a player DataFrame.
    
    This is the SINGLE point of override application. All consumers
    (optimizer, contest_sim, etc.) must call this function.
    
    Args:
        player_df: DataFrame with player projections (must have player_id column)
        overrides: SlateOverrides with user adjustments
        ownership_mode: "raw" keeps user ownership as-is, "renormalize" rescales
        lineup_size: For ownership normalization target (default 8 for DK)
    
    Returns:
        DataFrame with added columns:
        - model_minutes, model_fpts, model_own (original values)
        - effective_minutes, effective_fpts, effective_own (post-override)
        - override_minutes, override_fpts, override_own (user values or None)
        - has_override, used_fppm_fallback, is_active, fppm
    """
    df = player_df.copy()
    
    # Identify source columns
    fpts_col = _find_col(df, ["sim_dk_fpts_mean", "dk_fpts_mean", "proj_fpts", "fpts_mean", "proj"])
    minutes_col = _find_col(
        df,
        [
            "sim_minutes_sim_mean",
            "minutes_sim_mean",
            "sim_minutes_sim_p50",
            "minutes_sim_p50",
            "minutes_p50",
            "minutes",
            "minutes_pred",
        ],
    )
    own_col = _find_col(df, ["pred_own_pct", "own_proj", "ownership", "own"])
    
    if fpts_col is None:
        raise ValueError("No FPTS column found in DataFrame")
    
    # Store model values
    df["model_fpts"] = df[fpts_col].fillna(0)
    df["model_minutes"] = df[minutes_col].fillna(0) if minutes_col else 0
    df["model_own"] = df[own_col].fillna(0) if own_col else 0
    
    # Initialize effective values from model
    df["effective_fpts"] = df["model_fpts"].copy()
    df["effective_minutes"] = df["model_minutes"].copy()
    df["effective_own"] = df["model_own"].copy()
    
    # Initialize override columns
    df["override_minutes"] = None
    df["override_fpts"] = None
    df["override_own"] = None
    df["is_out"] = False
    df["has_override"] = False
    df["used_fppm_fallback"] = False
    df["is_active"] = True
    
    # Apply overrides
    for player_id, override in overrides.overrides.items():
        mask = df["player_id"].astype(str) == str(player_id)
        if not mask.any():
            continue
        
        idx = df.index[mask][0]
        
        # Mark player as out
        if override.is_out:
            df.loc[idx, "is_out"] = True
            df.loc[idx, "is_active"] = False
            df.loc[idx, "effective_fpts"] = 0
            df.loc[idx, "effective_minutes"] = 0
            df.loc[idx, "effective_own"] = 0
            df.loc[idx, "has_override"] = True
            continue
        
        df.loc[idx, "has_override"] = override.has_any_override()
        
        # Minutes override
        if override.minutes is not None:
            df.loc[idx, "override_minutes"] = override.minutes
            df.loc[idx, "effective_minutes"] = override.minutes
            
            # Compute effective FPTS from minutes
            model_fpts = df.loc[idx, "model_fpts"]
            model_minutes = df.loc[idx, "model_minutes"]
            
            eff_fpts, used_fallback = compute_effective_fpts(
                override_minutes=override.minutes,
                model_fpts=model_fpts,
                model_minutes=model_minutes,
                override_fpts=override.fpts,  # Direct fpts override takes precedence
            )
            df.loc[idx, "effective_fpts"] = eff_fpts
            df.loc[idx, "used_fppm_fallback"] = used_fallback
        
        # Direct FPTS override (independent of minutes)
        if override.fpts is not None:
            df.loc[idx, "override_fpts"] = override.fpts
            # Only override effective_fpts if not already computed from minutes
            if override.minutes is None:
                df.loc[idx, "effective_fpts"] = override.fpts
        
        # Ownership override
        if override.own is not None:
            df.loc[idx, "override_own"] = override.own
            df.loc[idx, "effective_own"] = override.own
    
    # Ownership renormalization
    if ownership_mode == "renormalize":
        df = _renormalize_ownership(df, lineup_size)
    
    # Compute FPPM for each player
    df["fppm"] = df.apply(
        lambda row: row["effective_fpts"] / row["effective_minutes"]
        if row["effective_minutes"] > 0
        else DEFAULT_FPPM,
        axis=1,
    )
    
    return df


def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Find first matching column from candidates."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _renormalize_ownership(
    df: pd.DataFrame,
    lineup_size: int = DK_LINEUP_SIZE,
) -> pd.DataFrame:
    """
    Renormalize ownership to sum to lineup_size * 100%.
    
    Fixed (overridden) ownership is preserved, remaining players are scaled.
    """
    target_total = lineup_size * 100.0  # 800% for DK
    
    # Active players only
    active = df["is_active"]
    
    # Separate fixed (overridden) and flexible ownership
    has_own_override = df["override_own"].notna()
    fixed_mask = active & has_own_override
    flex_mask = active & ~has_own_override
    
    fixed_total = df.loc[fixed_mask, "effective_own"].sum()
    flex_total = df.loc[flex_mask, "effective_own"].sum()
    
    if flex_total <= 0:
        return df  # Nothing to scale
    
    remaining_target = max(0, target_total - fixed_total)
    scale = remaining_target / flex_total
    
    df.loc[flex_mask, "effective_own"] = df.loc[flex_mask, "effective_own"] * scale
    
    logger.debug(
        "Ownership renormalized: fixed=%.1f%%, flex=%.1f%% → scale=%.3f",
        fixed_total,
        flex_total,
        scale,
    )
    
    return df
