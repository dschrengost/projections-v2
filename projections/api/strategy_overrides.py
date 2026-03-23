"""Persistent downstream strategy overrides for optimizer and contest sim.

This layer is slate-scoped and survives model reruns. It never mutates
canonical live outputs; instead, it adjusts downstream effective summaries and
private worlds used by optimizer/contest sim when explicitly enabled.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2
DEFAULT_FPPM = 1.0
FPPM_MIN = 0.3
FPPM_MAX = 2.5
MIN_MINUTES_FOR_FPPM = 4.0
MIN_FPTS_FOR_FPPM = 1.0
DK_LINEUP_SIZE = 8
FLOOR_MINUTES = 1.0
FLOOR_FPTS = 1.0
OWNERSHIP_REBALANCE_MIN_RATIO = 0.15
OWNERSHIP_REBALANCE_MAX_RATIO = 6.0
OWNERSHIP_REBALANCE_EXPONENT = 1.0


def _normalize_player_id(value: object) -> str:
    text = str(value).strip()
    if not text:
        return ""
    numeric = pd.to_numeric(text, errors="coerce")
    if pd.notna(numeric):
        try:
            as_float = float(numeric)
            as_int = int(round(as_float))
            if np.isfinite(as_float) and abs(as_float - as_int) < 1e-9:
                return str(as_int)
        except (TypeError, ValueError, OverflowError):
            pass
    return text


def get_overrides_root() -> Path:
    data_root = Path(os.environ.get("PROJECTIONS_DATA_ROOT", "/home/daniel/projections-data"))
    return data_root / "user_overrides"


@dataclass
class PlayerStrategyOverride:
    """Operator strategy adjustment for a single player."""

    player_id: str
    minutes_delta: Optional[float] = None
    fpts_delta: Optional[float] = None
    # Legacy absolute-target fields kept for backwards compatibility only.
    minutes: Optional[float] = None
    fpts: Optional[float] = None
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def has_any_override(self) -> bool:
        return any(
            value is not None
            for value in (
                self.minutes_delta,
                self.fpts_delta,
                self.minutes,
                self.fpts,
            )
        )

    def resolved_minutes_delta(self, *, model_minutes: float | None) -> Optional[float]:
        if self.minutes_delta is not None:
            return float(self.minutes_delta)
        if self.minutes is not None and model_minutes is not None and np.isfinite(model_minutes):
            return float(self.minutes) - float(model_minutes)
        return None

    def resolved_fpts_delta(self, *, model_fpts: float | None) -> Optional[float]:
        if self.fpts_delta is not None:
            return float(self.fpts_delta)
        if self.fpts is not None and model_fpts is not None and np.isfinite(model_fpts):
            return float(self.fpts) - float(model_fpts)
        return None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "player_id": self.player_id,
            "minutes_delta": self.minutes_delta,
            "fpts_delta": self.fpts_delta,
            "updated_at": self.updated_at,
        }
        if self.minutes is not None:
            payload["minutes"] = self.minutes
        if self.fpts is not None:
            payload["fpts"] = self.fpts
        return payload

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlayerStrategyOverride":
        return cls(
            player_id=_normalize_player_id(data["player_id"]),
            minutes_delta=data.get("minutes_delta"),
            fpts_delta=data.get("fpts_delta"),
            minutes=data.get("minutes"),
            fpts=data.get("fpts"),
            updated_at=data.get("updated_at", datetime.utcnow().isoformat()),
        )


@dataclass
class SlateStrategyOverrides:
    """All strategy overrides for a single slate."""

    game_date: str
    draft_group_id: int
    overrides: Dict[str, PlayerStrategyOverride] = field(default_factory=dict)
    client_revision: int = 0
    schema_version: int = SCHEMA_VERSION
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def get_override(self, player_id: str) -> Optional[PlayerStrategyOverride]:
        return self.overrides.get(_normalize_player_id(player_id))

    def set_override(self, override: PlayerStrategyOverride) -> None:
        normalized_pid = _normalize_player_id(override.player_id)
        override.player_id = normalized_pid
        self.overrides[normalized_pid] = override
        self.updated_at = datetime.utcnow().isoformat()
        self.client_revision += 1

    def remove_override(self, player_id: str) -> bool:
        normalized_pid = _normalize_player_id(player_id)
        if normalized_pid in self.overrides:
            del self.overrides[normalized_pid]
            self.updated_at = datetime.utcnow().isoformat()
            self.client_revision += 1
            return True
        return False

    def clear_all(self) -> None:
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
            "overrides": {pid: override.to_dict() for pid, override in self.overrides.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SlateStrategyOverrides":
        overrides = {
            _normalize_player_id(pid): PlayerStrategyOverride.from_dict(payload)
            for pid, payload in data.get("overrides", {}).items()
        }
        return cls(
            game_date=data["game_date"],
            draft_group_id=int(data["draft_group_id"]),
            overrides=overrides,
            client_revision=data.get("client_revision", 0),
            schema_version=data.get("schema_version", SCHEMA_VERSION),
            updated_at=data.get("updated_at", datetime.utcnow().isoformat()),
        )


def _get_override_path(game_date: str, draft_group_id: int) -> Path:
    return get_overrides_root() / game_date / f"dg_{draft_group_id}.json"


def _get_backup_path(path: Path) -> Path:
    return path.with_suffix(".json.bak")


def load_slate_strategy_overrides(game_date: str, draft_group_id: int) -> SlateStrategyOverrides:
    path = _get_override_path(game_date, draft_group_id)
    backup_path = _get_backup_path(path)

    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            return SlateStrategyOverrides.from_dict(data)
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("Failed to load strategy overrides from %s: %s", path, exc)

    if backup_path.exists():
        try:
            with open(backup_path, encoding="utf-8") as f:
                data = json.load(f)
            logger.info("Loaded strategy overrides from backup: %s", backup_path)
            return SlateStrategyOverrides.from_dict(data)
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("Failed to load strategy override backup: %s", exc)

    return SlateStrategyOverrides(game_date=game_date, draft_group_id=draft_group_id)


def save_slate_strategy_overrides(
    overrides: SlateStrategyOverrides,
    expected_revision: Optional[int] = None,
) -> bool:
    path = _get_override_path(overrides.game_date, overrides.draft_group_id)
    backup_path = _get_backup_path(path)

    if expected_revision is not None:
        current = load_slate_strategy_overrides(overrides.game_date, overrides.draft_group_id)
        if current.client_revision != expected_revision:
            logger.warning(
                "Strategy override revision conflict: expected %d, got %d",
                expected_revision,
                current.client_revision,
            )
            return False

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        shutil.copy2(path, backup_path)

    tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=".tmp_strategy_", suffix=".json")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(overrides.to_dict(), f, indent=2, sort_keys=True)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    return True


def _find_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def compute_fppm(
    model_fpts: float,
    model_minutes: float,
    historical_fppm: Optional[float] = None,
) -> tuple[float, bool]:
    if (
        model_minutes is not None
        and model_fpts is not None
        and np.isfinite(model_minutes)
        and np.isfinite(model_fpts)
        and model_minutes >= MIN_MINUTES_FOR_FPPM
        and model_fpts >= MIN_FPTS_FOR_FPPM
    ):
        fppm = float(model_fpts) / max(float(model_minutes), FLOOR_MINUTES)
        return float(np.clip(fppm, FPPM_MIN, FPPM_MAX)), False

    if historical_fppm is not None and np.isfinite(historical_fppm):
        return float(np.clip(float(historical_fppm), FPPM_MIN, FPPM_MAX)), False

    return DEFAULT_FPPM, True


def _rebalance_effective_ownership(
    df: pd.DataFrame,
    *,
    lineup_size: int = DK_LINEUP_SIZE,
) -> pd.DataFrame:
    """Shift ownership by effective-vs-model FPTS while preserving slate mass."""
    del lineup_size  # Preserve observed slate mass instead of forcing an external target.

    model_own = pd.to_numeric(df.get("model_own"), errors="coerce").fillna(0.0).clip(lower=0.0)
    model_total = float(model_own.sum())
    if model_total <= 0.0:
        return df

    model_fpts = pd.to_numeric(df.get("model_fpts"), errors="coerce").fillna(0.0).clip(lower=0.0)
    effective_fpts = pd.to_numeric(df.get("effective_fpts"), errors="coerce").fillna(0.0).clip(lower=0.0)

    ratio = np.ones(len(df), dtype=np.float64)
    model_vals = model_fpts.to_numpy(dtype=np.float64)
    effective_vals = effective_fpts.to_numpy(dtype=np.float64)

    stable_mask = model_vals > FLOOR_FPTS
    ratio[stable_mask] = effective_vals[stable_mask] / model_vals[stable_mask]

    # When baseline FPTS is near zero, treat effective_fpts as an absolute lift proxy.
    low_baseline_mask = (~stable_mask) & (effective_vals > 0.0)
    ratio[low_baseline_mask] = effective_vals[low_baseline_mask] / FLOOR_FPTS

    ratio = np.clip(ratio, OWNERSHIP_REBALANCE_MIN_RATIO, OWNERSHIP_REBALANCE_MAX_RATIO)
    if OWNERSHIP_REBALANCE_EXPONENT != 1.0:
        ratio = np.power(ratio, OWNERSHIP_REBALANCE_EXPONENT)

    weighted_own = model_own.to_numpy(dtype=np.float64) * ratio
    weighted_total = float(np.nansum(weighted_own))
    if not np.isfinite(weighted_total) or weighted_total <= 0.0:
        return df

    scale = model_total / weighted_total
    df["effective_own"] = weighted_own * scale
    return df


def summarize_worlds(
    *,
    fpts_matrix: np.ndarray,
    player_index: Dict[str, int],
    minutes_matrix: np.ndarray | None = None,
) -> Dict[str, Dict[str, float]]:
    summaries: Dict[str, Dict[str, float]] = {}
    if fpts_matrix.size == 0:
        return summaries

    for pid, idx in player_index.items():
        fpts = np.asarray(fpts_matrix[:, idx], dtype=np.float64)
        mins = np.asarray(minutes_matrix[:, idx], dtype=np.float64) if minutes_matrix is not None else None
        summary = {
            "effective_fpts_mean": float(np.mean(fpts)),
            "effective_fpts_std": float(np.std(fpts)),
            "effective_fpts_p90": float(np.percentile(fpts, 90)),
            "p_active": float(np.mean(fpts > 0)),
        }
        if mins is not None:
            summary["effective_minutes_mean"] = float(np.mean(mins))
        summaries[str(pid)] = summary
    return summaries


def apply_strategy_overrides_to_worlds(
    *,
    fpts_matrix: np.ndarray,
    player_index: Dict[str, int],
    overrides: SlateStrategyOverrides,
    minutes_matrix: np.ndarray | None = None,
    model_minutes_by_player: Optional[Dict[str, float]] = None,
    model_fpts_by_player: Optional[Dict[str, float]] = None,
) -> tuple[np.ndarray, np.ndarray | None, Dict[str, Any]]:
    adjusted_fpts = np.asarray(fpts_matrix, dtype=np.float64).copy()
    adjusted_minutes = None if minutes_matrix is None else np.asarray(minutes_matrix, dtype=np.float64).copy()
    diagnostics: Dict[str, Any] = {
        "matched_override_count": 0,
        "applied_minutes_delta_count": 0,
        "applied_fpts_delta_count": 0,
        "used_minutes_fallback_count": 0,
        "applied_player_ids": [],
    }

    if adjusted_fpts.size == 0 or not overrides.overrides:
        return adjusted_fpts, adjusted_minutes, diagnostics

    normalized_player_index = {_normalize_player_id(pid): idx for pid, idx in player_index.items()}
    normalized_model_minutes = (
        {_normalize_player_id(pid): value for pid, value in model_minutes_by_player.items()}
        if model_minutes_by_player is not None
        else None
    )
    normalized_model_fpts = (
        {_normalize_player_id(pid): value for pid, value in model_fpts_by_player.items()}
        if model_fpts_by_player is not None
        else None
    )

    for pid, override in overrides.overrides.items():
        normalized_pid = _normalize_player_id(pid)
        col_idx = normalized_player_index.get(normalized_pid)
        if col_idx is None:
            continue

        model_minutes = None
        if normalized_model_minutes is not None:
            model_minutes = normalized_model_minutes.get(normalized_pid)
        model_fpts = None
        if normalized_model_fpts is not None:
            model_fpts = normalized_model_fpts.get(normalized_pid)

        minutes_delta = override.resolved_minutes_delta(model_minutes=model_minutes)
        fpts_delta = override.resolved_fpts_delta(model_fpts=model_fpts)
        if minutes_delta is None and fpts_delta is None:
            continue

        diagnostics["matched_override_count"] = int(diagnostics["matched_override_count"]) + 1
        diagnostics["applied_player_ids"].append(normalized_pid)

        fpts_col = adjusted_fpts[:, col_idx]
        if adjusted_minutes is not None:
            minutes_col = adjusted_minutes[:, col_idx]
            active_mask = (fpts_col > 0.0) | (minutes_col > 0.0)
        else:
            minutes_col = None
            active_mask = fpts_col > 0.0

        if not np.any(active_mask):
            continue

        if minutes_delta is not None:
            if minutes_col is not None:
                active_minutes = minutes_col[active_mask]
                new_minutes = np.maximum(0.0, active_minutes + float(minutes_delta))
                scale = new_minutes / np.maximum(active_minutes, FLOOR_MINUTES)
                adjusted_minutes[active_mask, col_idx] = new_minutes
                adjusted_fpts[active_mask, col_idx] = np.maximum(0.0, fpts_col[active_mask] * scale)
            else:
                diagnostics["used_minutes_fallback_count"] = int(diagnostics["used_minutes_fallback_count"]) + 1
                baseline_minutes = float(model_minutes) if model_minutes is not None else 0.0
                target_minutes = max(0.0, baseline_minutes + float(minutes_delta))
                scale = target_minutes / max(baseline_minutes, FLOOR_MINUTES)
                adjusted_fpts[active_mask, col_idx] = np.maximum(0.0, fpts_col[active_mask] * scale)
            diagnostics["applied_minutes_delta_count"] = int(diagnostics["applied_minutes_delta_count"]) + 1
            fpts_col = adjusted_fpts[:, col_idx]

        if fpts_delta is not None:
            active_fpts = fpts_col[active_mask]
            mu_active = float(np.mean(active_fpts)) if active_fpts.size else 0.0
            target_mu = max(0.0, mu_active + float(fpts_delta))
            scale = target_mu / max(mu_active, FLOOR_FPTS)
            adjusted_fpts[active_mask, col_idx] = np.maximum(0.0, active_fpts * scale)
            diagnostics["applied_fpts_delta_count"] = int(diagnostics["applied_fpts_delta_count"]) + 1

    return adjusted_fpts, adjusted_minutes, diagnostics


def apply_strategy_overrides(
    player_df: pd.DataFrame,
    overrides: SlateStrategyOverrides,
    *,
    ownership_mode: Literal["raw", "renormalize"] = "renormalize",
    adjusted_summaries: Optional[Dict[str, Dict[str, float]]] = None,
    lineup_size: int = DK_LINEUP_SIZE,
) -> pd.DataFrame:
    """Apply strategy overrides to a player DataFrame.

    `adjusted_summaries` should come from adjusted worlds when available.
    Without worlds, a mean-based fallback is used.
    """
    df = player_df.copy()

    fpts_col = _find_col(
        df,
        [
            "fpts_sim_uncond_mean",
            "dk_fpts_mean_uncond",
            "fpts_sim_cond_mean",
            "dk_fpts_mean",
            "sim_dk_fpts_mean",
            "proj_fpts",
            "fpts_mean",
            "proj",
        ],
    )
    minutes_col = _find_col(
        df,
        [
            "minutes_sim_uncond_mean",
            "minutes_sim_mean_uncond",
            "minutes_sim_cond_mean",
            "minutes_sim_mean",
            "minutes_sim_uncond_p50",
            "minutes_sim_p50_uncond",
            "minutes_sim_cond_p50",
            "minutes_sim_p50",
            "sim_minutes_sim_mean",
            "sim_minutes_sim_p50",
            "minutes_p50",
            "minutes",
            "minutes_pred",
        ],
    )
    own_col = _find_col(df, ["pred_own_pct", "own_proj", "ownership", "own"])
    p_active_col = _find_col(df, ["minutes_sim_p_active", "p_play_eff", "play_prob"])
    std_col = _find_col(
        df,
        [
            "fpts_sim_uncond_std",
            "sim_dk_fpts_std_uncond",
            "dk_fpts_std_uncond",
            "fpts_sim_cond_std",
            "sim_dk_fpts_std",
            "stddev",
            "fpts_std",
        ],
    )
    p90_col = _find_col(
        df,
        [
            "fpts_sim_uncond_p90",
            "sim_dk_fpts_p90_uncond",
            "dk_fpts_p90_uncond",
            "fpts_sim_cond_p90",
            "sim_dk_fpts_p90",
            "dk_fpts_p90",
            "fpts_p90",
        ],
    )

    if fpts_col is None:
        raise ValueError("No FPTS column found in DataFrame")

    df["model_fpts"] = pd.to_numeric(df[fpts_col], errors="coerce").fillna(0.0)
    df["model_minutes"] = pd.to_numeric(df[minutes_col], errors="coerce").fillna(0.0) if minutes_col else 0.0
    df["model_own"] = pd.to_numeric(df[own_col], errors="coerce").fillna(0.0) if own_col else 0.0
    df["effective_fpts"] = df["model_fpts"].copy()
    df["effective_minutes"] = df["model_minutes"].copy()
    df["effective_own"] = df["model_own"].copy()
    if std_col:
        df["effective_fpts_std"] = pd.to_numeric(df[std_col], errors="coerce").fillna(0.0)
    else:
        df["effective_fpts_std"] = 0.0
    if p90_col:
        df["effective_fpts_p90"] = pd.to_numeric(df[p90_col], errors="coerce").fillna(0.0)
    else:
        df["effective_fpts_p90"] = 0.0

    df["override_minutes_delta"] = None
    df["override_fpts_delta"] = None
    df["has_override"] = False
    df["used_fppm_fallback"] = False
    df["strategy_override_source"] = "model"

    player_id_norm = df["player_id"].map(_normalize_player_id)
    for pid, override in overrides.overrides.items():
        normalized_pid = _normalize_player_id(pid)
        mask = player_id_norm == normalized_pid
        if not mask.any():
            continue
        idx = df.index[mask][0]
        model_minutes = float(df.loc[idx, "model_minutes"])
        model_fpts = float(df.loc[idx, "model_fpts"])
        p_active = float(df.loc[idx, p_active_col]) if p_active_col and pd.notna(df.loc[idx, p_active_col]) else 1.0
        p_active = float(np.clip(p_active, 0.0, 1.0))

        minutes_delta = override.resolved_minutes_delta(model_minutes=model_minutes)
        fpts_delta = override.resolved_fpts_delta(model_fpts=model_fpts)
        if minutes_delta is None and fpts_delta is None:
            continue

        df.loc[idx, "has_override"] = True
        df.loc[idx, "override_minutes_delta"] = minutes_delta
        df.loc[idx, "override_fpts_delta"] = fpts_delta

        summary = adjusted_summaries.get(str(pid)) if adjusted_summaries else None
        if summary:
            df.loc[idx, "effective_fpts"] = float(summary["effective_fpts_mean"])
            if "effective_minutes_mean" in summary:
                df.loc[idx, "effective_minutes"] = float(summary["effective_minutes_mean"])
            df.loc[idx, "effective_fpts_std"] = float(summary.get("effective_fpts_std", df.loc[idx, "effective_fpts_std"]))
            df.loc[idx, "effective_fpts_p90"] = float(summary.get("effective_fpts_p90", df.loc[idx, "effective_fpts_p90"]))
            df.loc[idx, "strategy_override_source"] = "worlds"
            continue

        if minutes_delta is not None:
            df.loc[idx, "effective_minutes"] = max(0.0, model_minutes + p_active * float(minutes_delta))

        fppm, used_fallback = compute_fppm(model_fpts=model_fpts, model_minutes=model_minutes)
        effective_fpts = model_fpts
        if minutes_delta is not None:
            effective_fpts = effective_fpts + p_active * float(minutes_delta) * fppm
        if fpts_delta is not None:
            effective_fpts = effective_fpts + p_active * float(fpts_delta)
        df.loc[idx, "effective_fpts"] = max(0.0, float(effective_fpts))
        df.loc[idx, "used_fppm_fallback"] = bool(minutes_delta is not None and used_fallback)
        df.loc[idx, "strategy_override_source"] = "fallback"

    ownership_mode_n = str(ownership_mode or "renormalize").strip().lower()
    if ownership_mode_n not in {"raw", "renormalize"}:
        ownership_mode_n = "renormalize"
    if ownership_mode_n == "renormalize" and bool(df["has_override"].any()):
        df = _rebalance_effective_ownership(df, lineup_size=lineup_size)

    df["fppm"] = df.apply(
        lambda row: row["effective_fpts"] / row["effective_minutes"] if float(row["effective_minutes"]) > 0 else DEFAULT_FPPM,
        axis=1,
    )
    return df
