"""Minutes V1 quick-start helpers."""

from . import calibration
from .features import MinutesFeatureBuilder
from .labels import freeze_boxscore_labels, load_frozen_labels
from .modeling import (
    ConformalIntervalCalibrator,
    QuickstartModelArtifacts,
    predict_minutes,
    train_minutes_quickstart_models,
)
from .monitoring import compute_monitoring_snapshot
try:  # pragma: no cover - legacy shim may be absent in streamlined builds
    from .reconciliation import ReconciliationConfig, reconcile_minutes, reconcile_team_minutes
except ImportError:  # pragma: no cover
    ReconciliationConfig = None
    reconcile_minutes = None
    reconcile_team_minutes = None
from .snapshots import ensure_as_of_column, latest_pre_tip_snapshot
from .validation import (
    ReconciliationReport,
    hash_season_labels,
    reconciliation_sanity_check,
    sample_anti_leak_check,
    validate_label_hashes,
)

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
    "ReconciliationConfig",
    "reconcile_team_minutes",
    "reconcile_minutes",
    "compute_monitoring_snapshot",
    "sample_anti_leak_check",
    "hash_season_labels",
    "validate_label_hashes",
    "reconciliation_sanity_check",
    "ReconciliationReport",
    "calibration",
]
