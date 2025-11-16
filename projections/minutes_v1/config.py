"""Pydantic models for YAML-driven minutes workflows."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class MinutesTrainingConfig(BaseModel):
    """Overrides for the LightGBM trainer."""

    model_config = ConfigDict(extra="ignore")

    run_id: str | None = None
    train_start: datetime | None = None
    train_end: datetime | None = None
    cal_start: datetime | None = None
    cal_end: datetime | None = None
    val_start: datetime | None = None
    val_end: datetime | None = None
    cal_days: int | None = Field(
        default=None, description="If set, derive calibration window as N days prior to val_start."
    )
    data_root: Path | None = None
    season: int | None = Field(
        default=None, description="Season year for default feature path resolution (e.g., 2024)."
    )
    month: int | None = Field(
        default=None, description="Month partition (1-12) used when deriving default features path."
    )
    features: Path | None = Field(
        default=None,
        description="Explicit feature parquet path. Overrides season/month derived path when provided.",
    )
    artifact_root: Path | None = None
    target_col: str | None = None
    random_state: int | None = None
    conformal_buckets: Literal["starter,p50bins", "starter,p50bins,injury_snapshot", "none"] | None = None
    conformal_k: int | None = None
    conformal_mode: Literal["two-sided", "tail-deltas", "center-width"] | None = None
    tolerance_relaxed: bool | None = None
    winkler_baseline: float | None = None
    allow_guard_failure: bool | None = None
    fold_id: str | None = None
    playable_min_p50: float | None = None
    playable_winkler_baseline: float | None = None
    playable_winkler_tolerance: float | None = None


class MinutesScoringConfig(BaseModel):
    """Overrides for the scoring CLI."""

    model_config = ConfigDict(extra="ignore")
    date: datetime | None = Field(default=None, description="Primary slate date (YYYY-MM-DD).")
    end_date: datetime | None = Field(
        default=None, description="Optional inclusive end date when running multi-day batches."
    )
    features_root: Path | None = None
    features_path: Path | None = None
    bundle_dir: Path | None = None
    bundle_config: Path | None = None
    artifact_root: Path | None = None
    injuries_root: Path | None = None
    schedule_root: Path | None = None
    limit_rows: int | None = Field(default=None, ge=1)
    mode: Literal["historical", "live"] | None = None
    run_id: str | None = None
    live_features_root: Path | None = None
    minutes_output: Literal["conditional", "unconditional", "both"] | None = None
    starter_priors_path: Path | None = None
    starter_history_path: Path | None = None
    promotion_config: Path | None = None
    promotion_prior_enabled: bool | None = None
    promotion_prior_debug: bool | None = None
    reconcile_team_minutes: Literal["none", "p50", "p50_and_tails"] | None = None
    reconcile_config: Path | None = None
    reconcile_debug: bool | None = None


def _load_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file missing at {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config payload in {path} must be a mapping")
    return payload


def _resolve_section(payload: dict[str, Any], key: str) -> dict[str, Any]:
    """Allow nested sections for shared config files."""

    section = payload.get(key)
    if section is None:
        return payload
    if not isinstance(section, dict):
        raise ValueError(f"{key} section inside config must be a mapping")
    return section


def load_training_config(path: Path) -> MinutesTrainingConfig:
    """Load a YAML config describing a minutes training run."""

    payload = _load_payload(path)
    return MinutesTrainingConfig.model_validate(_resolve_section(payload, "minutes_training"))


def load_scoring_config(path: Path) -> MinutesScoringConfig:
    """Load a YAML config describing a minutes inference/scoring run."""

    payload = _load_payload(path)
    return MinutesScoringConfig.model_validate(_resolve_section(payload, "minutes_scoring"))
