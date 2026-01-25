"""Training↔Inference parity audit for minutes models.

This module provides tools to detect drift between training features and
inference features, including:
- Missing required columns
- Dtype mismatches
- Join missingness patterns
- Value range checks

The parity report is written as a JSON artifact next to minutes outputs
for incident response and debugging.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ColumnParityIssue:
    """A single column-level parity issue."""

    column: str
    issue_type: str  # "missing", "dtype_mismatch", "high_nan_rate", "range_violation"
    severity: str  # "error", "warning", "info"
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ParityReport:
    """Complete parity audit report for a scoring run."""

    # Metadata
    model_dir: str
    game_date: str
    run_id: str
    generated_at: str

    # Feature counts
    expected_features: int
    present_features: int
    missing_features: list[str]
    extra_features: list[str]

    # Issues by severity
    errors: list[ColumnParityIssue] = field(default_factory=list)
    warnings: list[ColumnParityIssue] = field(default_factory=list)
    info: list[ColumnParityIssue] = field(default_factory=list)

    # Summary stats
    nan_rate_by_column: dict[str, float] = field(default_factory=dict)
    value_range_by_column: dict[str, dict[str, float]] = field(default_factory=dict)

    # Overall pass/fail
    parity_passed: bool = True
    failure_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Convert ColumnParityIssue objects to dicts
        d["errors"] = [e.to_dict() if hasattr(e, "to_dict") else e for e in self.errors]
        d["warnings"] = [w.to_dict() if hasattr(w, "to_dict") else w for w in self.warnings]
        d["info"] = [i.to_dict() if hasattr(i, "to_dict") else i for i in self.info]
        return d

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @property
    def n_errors(self) -> int:
        return len(self.errors)

    @property
    def n_warnings(self) -> int:
        return len(self.warnings)

    def summary_line(self) -> str:
        status = "PASS" if self.parity_passed else "FAIL"
        return (
            f"[parity] {status}: expected={self.expected_features} present={self.present_features} "
            f"missing={len(self.missing_features)} errors={self.n_errors} warnings={self.n_warnings}"
        )


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_training_stats(model_dir: Path) -> dict[str, Any] | None:
    """Load training feature statistics if available.

    Looks for `training_stats.json` in model_dir which contains:
    - per-column: mean, std, p01, p50, p99, nan_rate
    """
    stats_path = model_dir / "training_stats.json"
    if not stats_path.exists():
        return None
    try:
        return json.loads(stats_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def load_feature_columns(model_dir: Path) -> list[str]:
    """Load expected feature columns from model manifest."""
    path = model_dir / "feature_columns.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing feature_columns.json at {model_dir}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    cols = payload.get("columns")
    if not isinstance(cols, list) or not cols:
        raise ValueError(f"Invalid feature_columns.json at {path}")
    return [str(c) for c in cols]


def compute_parity_report(
    df: pd.DataFrame,
    *,
    model_dir: Path,
    game_date: str,
    run_id: str,
    expected_features: list[str] | None = None,
    nan_rate_warning_threshold: float = 0.5,
    nan_rate_error_threshold: float = 0.9,
    training_stats: dict[str, Any] | None = None,
) -> ParityReport:
    """Compute a parity report comparing inference features to training expectations.

    Args:
        df: Inference feature DataFrame
        model_dir: Path to model directory
        game_date: Slate date
        run_id: Run identifier
        expected_features: List of expected feature columns (loaded from model if None)
        nan_rate_warning_threshold: NaN rate above this triggers a warning
        nan_rate_error_threshold: NaN rate above this triggers an error
        training_stats: Optional training statistics for drift detection

    Returns:
        ParityReport with all detected issues
    """
    model_dir = Path(model_dir).expanduser().resolve()

    if expected_features is None:
        expected_features = load_feature_columns(model_dir)

    if training_stats is None:
        training_stats = load_training_stats(model_dir)

    report = ParityReport(
        model_dir=str(model_dir),
        game_date=game_date,
        run_id=run_id,
        generated_at=_utc_now_iso(),
        expected_features=len(expected_features),
        present_features=0,
        missing_features=[],
        extra_features=[],
    )

    # Check for missing features
    present = set(df.columns)
    expected_set = set(expected_features)
    missing = sorted(expected_set - present)
    extra = sorted(present - expected_set - {"game_id", "team_id", "player_id", "opponent_team_id", "minutes_features_row_missing", "injury_snapshot_missing"})

    report.missing_features = missing
    report.extra_features = extra
    report.present_features = len(expected_set & present)

    # Missing features are errors
    for col in missing:
        issue = ColumnParityIssue(
            column=col,
            issue_type="missing",
            severity="error",
            details={"expected": True, "found": False},
        )
        report.errors.append(issue)

    # Analyze present features
    for col in expected_features:
        if col not in df.columns:
            continue

        series = df[col]
        numeric_series = pd.to_numeric(series, errors="coerce")

        # NaN rate
        nan_rate = float(numeric_series.isna().mean()) if len(numeric_series) > 0 else 0.0
        report.nan_rate_by_column[col] = nan_rate

        if nan_rate >= nan_rate_error_threshold:
            issue = ColumnParityIssue(
                column=col,
                issue_type="high_nan_rate",
                severity="error",
                details={"nan_rate": nan_rate, "threshold": nan_rate_error_threshold},
            )
            report.errors.append(issue)
        elif nan_rate >= nan_rate_warning_threshold:
            issue = ColumnParityIssue(
                column=col,
                issue_type="high_nan_rate",
                severity="warning",
                details={"nan_rate": nan_rate, "threshold": nan_rate_warning_threshold},
            )
            report.warnings.append(issue)

        # Value range (for non-null values)
        valid = numeric_series.dropna()
        if len(valid) > 0:
            report.value_range_by_column[col] = {
                "min": float(valid.min()),
                "max": float(valid.max()),
                "mean": float(valid.mean()),
                "std": float(valid.std()) if len(valid) > 1 else 0.0,
                "p01": float(np.percentile(valid, 1)),
                "p50": float(np.percentile(valid, 50)),
                "p99": float(np.percentile(valid, 99)),
            }

            # Check against training stats if available
            if training_stats and col in training_stats:
                train_col = training_stats[col]
                train_p01 = train_col.get("p01", float("-inf"))
                train_p99 = train_col.get("p99", float("inf"))

                # Check if inference values are outside training range
                inf_min = report.value_range_by_column[col]["min"]
                inf_max = report.value_range_by_column[col]["max"]

                if inf_min < train_p01 * 0.5 or inf_max > train_p99 * 2.0:
                    issue = ColumnParityIssue(
                        column=col,
                        issue_type="range_violation",
                        severity="warning",
                        details={
                            "inference_min": inf_min,
                            "inference_max": inf_max,
                            "training_p01": train_p01,
                            "training_p99": train_p99,
                        },
                    )
                    report.warnings.append(issue)

    # Determine overall pass/fail
    if report.missing_features:
        report.parity_passed = False
        report.failure_reason = f"Missing {len(report.missing_features)} required features: {report.missing_features[:5]}"
    elif report.n_errors > 0:
        report.parity_passed = False
        report.failure_reason = f"{report.n_errors} parity errors detected"

    return report


def write_parity_report(report: ParityReport, output_dir: Path) -> Path:
    """Write parity report to a JSON file.

    Args:
        report: ParityReport to write
        output_dir: Directory to write to (usually the run directory)

    Returns:
        Path to written file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "parity_report.json"
    output_path.write_text(report.to_json(), encoding="utf-8")
    return output_path


def inject_parity_into_summary(
    summary_path: Path,
    report: ParityReport,
) -> None:
    """Inject parity report summary into an existing summary.json file."""
    if summary_path.exists():
        try:
            existing = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
    else:
        existing = {}

    existing["parity"] = {
        "passed": report.parity_passed,
        "failure_reason": report.failure_reason,
        "expected_features": report.expected_features,
        "present_features": report.present_features,
        "missing_features": report.missing_features,
        "n_errors": report.n_errors,
        "n_warnings": report.n_warnings,
    }

    summary_path.write_text(json.dumps(existing, indent=2, sort_keys=True), encoding="utf-8")


__all__ = [
    "ColumnParityIssue",
    "ParityReport",
    "compute_parity_report",
    "load_feature_columns",
    "load_training_stats",
    "write_parity_report",
    "inject_parity_into_summary",
]
