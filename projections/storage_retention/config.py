"""Configuration loader for storage retention policy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from projections import paths

DEFAULT_CONFIG_PATH = paths.get_project_root() / "config" / "storage_retention.yaml"


@dataclass(frozen=True)
class GuardPolicy:
    hot_warn_free_gb: float = 150.0
    hot_warn_free_pct: float = 15.0
    hot_hard_free_gb: float = 100.0
    hot_hard_free_pct: float = 10.0
    root_hard_free_gb: float = 50.0


@dataclass(frozen=True)
class RetentionPolicy:
    lead_time_minutes: int = 2
    start_time_bucket_minutes: int = 30
    keep_latest_debug_runs: int = 1
    protect_current_day: bool = True


@dataclass(frozen=True)
class ReducedPersistencePolicy:
    enabled: bool = False
    sim_write_worlds_matrix: bool = True
    sim_write_minutes_matrix: bool = False
    gtv2_max_worlds: int | None = None


@dataclass(frozen=True)
class StorageRetentionPolicy:
    version: int = 1
    guard: GuardPolicy = GuardPolicy()
    retention: RetentionPolicy = RetentionPolicy()
    reduced_persistence: ReducedPersistencePolicy = ReducedPersistencePolicy()
    families: tuple[str, ...] = ("gtv2_worlds", "sim_v2_worlds_fpts_v2")


def _to_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        return {}
    return payload


def load_storage_retention_policy(config_path: Path | None = None) -> StorageRetentionPolicy:
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    payload = _load_yaml(path)

    guard_raw = payload.get("guard") or {}
    retention_raw = payload.get("retention") or {}
    rp_raw = payload.get("reduced_persistence") or {}
    families_raw = payload.get("families") or ["gtv2_worlds", "sim_v2_worlds_fpts_v2"]
    families: tuple[str, ...]
    if isinstance(families_raw, list):
        families = tuple(str(v).strip() for v in families_raw if str(v).strip())
    else:
        families = ("gtv2_worlds", "sim_v2_worlds_fpts_v2")
    if not families:
        families = ("gtv2_worlds", "sim_v2_worlds_fpts_v2")

    guard = GuardPolicy(
        hot_warn_free_gb=float(guard_raw.get("hot_warn_free_gb", 150.0)),
        hot_warn_free_pct=float(guard_raw.get("hot_warn_free_pct", 15.0)),
        hot_hard_free_gb=float(guard_raw.get("hot_hard_free_gb", 100.0)),
        hot_hard_free_pct=float(guard_raw.get("hot_hard_free_pct", 10.0)),
        root_hard_free_gb=float(guard_raw.get("root_hard_free_gb", 50.0)),
    )
    retention = RetentionPolicy(
        lead_time_minutes=int(retention_raw.get("lead_time_minutes", 2)),
        start_time_bucket_minutes=int(retention_raw.get("start_time_bucket_minutes", 30)),
        keep_latest_debug_runs=max(0, int(retention_raw.get("keep_latest_debug_runs", 1))),
        protect_current_day=_to_bool(retention_raw.get("protect_current_day"), True),
    )
    gtv2_max_worlds_raw = rp_raw.get("gtv2_max_worlds")
    gtv2_max_worlds = (
        int(gtv2_max_worlds_raw)
        if gtv2_max_worlds_raw is not None and str(gtv2_max_worlds_raw).strip() != ""
        else None
    )
    reduced = ReducedPersistencePolicy(
        enabled=_to_bool(rp_raw.get("enabled"), False),
        sim_write_worlds_matrix=_to_bool(rp_raw.get("sim_write_worlds_matrix"), True),
        sim_write_minutes_matrix=_to_bool(rp_raw.get("sim_write_minutes_matrix"), False),
        gtv2_max_worlds=gtv2_max_worlds,
    )

    version_raw = payload.get("version")
    version = int(version_raw) if version_raw is not None else 1
    return StorageRetentionPolicy(
        version=version,
        guard=guard,
        retention=retention,
        reduced_persistence=reduced,
        families=families,
    )
