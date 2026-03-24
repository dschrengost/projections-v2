"""Minutes V1 quick-start helpers.

Keep package import side-effects minimal. Several runtime entrypoints import
submodules from ``projections.minutes_v1`` for schemas/snapshots only; eager
imports here can pull in heavy native stacks (e.g., LightGBM) unnecessarily.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "ensure_as_of_column",
    "latest_pre_tip_snapshot",
    "freeze_boxscore_labels",
    "load_frozen_labels",
    "MinutesFeatureBuilder",
    "train_minutes_quickstart_models",
    "predict_minutes",
    "QuickstartModelArtifacts",
    "ConformalIntervalCalibrator",
    "compute_monitoring_snapshot",
    "sample_anti_leak_check",
    "hash_season_labels",
    "validate_label_hashes",
    "reconciliation_sanity_check",
    "ReconciliationReport",
    "calibration",
]


_LAZY_EXPORTS: dict[str, tuple[str, str | None]] = {
    "ensure_as_of_column": ("projections.minutes_v1.snapshots", "ensure_as_of_column"),
    "latest_pre_tip_snapshot": (
        "projections.minutes_v1.snapshots",
        "latest_pre_tip_snapshot",
    ),
    "freeze_boxscore_labels": ("projections.minutes_v1.labels", "freeze_boxscore_labels"),
    "load_frozen_labels": ("projections.minutes_v1.labels", "load_frozen_labels"),
    "MinutesFeatureBuilder": ("projections.minutes_v1.features", "MinutesFeatureBuilder"),
    "train_minutes_quickstart_models": (
        "projections.minutes_v1.modeling",
        "train_minutes_quickstart_models",
    ),
    "predict_minutes": ("projections.minutes_v1.modeling", "predict_minutes"),
    "QuickstartModelArtifacts": (
        "projections.minutes_v1.modeling",
        "QuickstartModelArtifacts",
    ),
    "ConformalIntervalCalibrator": (
        "projections.minutes_v1.modeling",
        "ConformalIntervalCalibrator",
    ),
    "compute_monitoring_snapshot": (
        "projections.minutes_v1.monitoring",
        "compute_monitoring_snapshot",
    ),
    "sample_anti_leak_check": (
        "projections.minutes_v1.validation",
        "sample_anti_leak_check",
    ),
    "hash_season_labels": ("projections.minutes_v1.validation", "hash_season_labels"),
    "validate_label_hashes": ("projections.minutes_v1.validation", "validate_label_hashes"),
    "reconciliation_sanity_check": (
        "projections.minutes_v1.validation",
        "reconciliation_sanity_check",
    ),
    "ReconciliationReport": ("projections.minutes_v1.validation", "ReconciliationReport"),
    "calibration": ("projections.minutes_v1.calibration", None),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    module = import_module(module_name)
    value = module if attr_name is None else getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
