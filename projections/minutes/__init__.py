"""Shared minutes pipeline utilities (cross-model)."""

from __future__ import annotations

from projections.minutes.contract import (
    IN_ROTATION_THRESHOLD_MIN,
    MINUTES_CONTRACT_VERSION,
    PLAY_THRESHOLD_MINUTES,
    ROTATION_THRESHOLD_MINUTES,
    MinutesAllocMode,
    MinutesProvenance,
    build_provenance,
    find_minutes_run_dirs,
    get_latest_run_dir,
    inject_provenance_into_summary,
    load_provenance_from_summary,
    minutes_contract_hash,
)
from projections.minutes.parity import (
    ColumnParityIssue,
    ParityReport,
    compute_parity_report,
    inject_parity_into_summary,
    load_feature_columns,
    write_parity_report,
)

__all__ = [
    # Contract
    "PLAY_THRESHOLD_MINUTES",
    "ROTATION_THRESHOLD_MINUTES",
    "IN_ROTATION_THRESHOLD_MIN",
    "MINUTES_CONTRACT_VERSION",
    "minutes_contract_hash",
    "MinutesAllocMode",
    "MinutesProvenance",
    "build_provenance",
    "load_provenance_from_summary",
    "inject_provenance_into_summary",
    "find_minutes_run_dirs",
    "get_latest_run_dir",
    # Parity
    "ColumnParityIssue",
    "ParityReport",
    "compute_parity_report",
    "load_feature_columns",
    "write_parity_report",
    "inject_parity_into_summary",
]
