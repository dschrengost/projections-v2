"""Feature contract enforcement for the minute share model.

This module ensures that features used for share model prediction satisfy
the expected schema, regardless of when the features were built.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# Critical features that strongly impact model quality
# If these are missing, the slate should likely be skipped
CRITICAL_FEATURES: set[str] = {
    "roll_mean_5",
    "roll_mean_10",
    "min_last3",
    "min_last5",
    "starter_flag",
    "is_starter",
    "games_played_szn",
    "minutes_per_game_szn",
}

# Severity levels for reporting
SEVERITY_CRITICAL = "critical"
SEVERITY_IMPORTANT = "important"
SEVERITY_MINOR = "minor"


@dataclass
class ContractReport:
    """Report on feature contract enforcement."""
    n_expected: int = 0
    n_missing: int = 0
    n_present: int = 0
    missing_cols: list[str] = None
    missing_feature_frac: float = 0.0
    nan_indicators_created: int = 0
    cols_filled_with_zero: int = 0
    
    # Severity breakdown
    n_critical_missing: int = 0
    n_important_missing: int = 0
    critical_missing: list[str] = None
    severity: str = SEVERITY_MINOR  # Overall severity level
    
    # Track columns where base was entirely missing (indicator set to 1.0)
    base_missing_cols: list[str] = None
    
    def __post_init__(self):
        if self.missing_cols is None:
            self.missing_cols = []
        if self.critical_missing is None:
            self.critical_missing = []
        if self.base_missing_cols is None:
            self.base_missing_cols = []
    
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Truncate lists for JSON output
        d["missing_cols"] = d["missing_cols"][:25]
        d["critical_missing"] = d["critical_missing"][:10]
        d["base_missing_cols"] = d["base_missing_cols"][:10]
        return d


def load_expected_share_features(bundle_dir: str | Path) -> list[str]:
    """Load expected feature columns from share model bundle.
    
    Args:
        bundle_dir: Path to the share model bundle directory
        
    Returns:
        List of expected feature column names
    """
    bundle_path = Path(bundle_dir)
    
    # Try feature_columns.json
    cols_path = bundle_path / "feature_columns.json"
    if cols_path.exists():
        data = json.loads(cols_path.read_text())
        if isinstance(data, dict):
            return data.get("columns", [])
        elif isinstance(data, list):
            return data
    
    # Fallback: try to infer from model if available
    model_path = bundle_path / "minute_share_model.joblib"
    if model_path.exists():
        try:
            import joblib
            model = joblib.load(model_path)
            # Try common model attributes
            if hasattr(model, "feature_name_"):
                return list(model.feature_name_)
            if hasattr(model, "feature_names_in_"):
                return list(model.feature_names_in_)
        except Exception:
            pass
    
    return []


def load_critical_features(bundle_dir: str | Path | None = None) -> set[str]:
    """Load critical feature set.
    
    Tries to load from bundle's critical_features.json, falls back to defaults.
    
    Args:
        bundle_dir: Optional path to share model bundle
        
    Returns:
        Set of critical feature names
    """
    if bundle_dir:
        critical_path = Path(bundle_dir) / "critical_features.json"
        if critical_path.exists():
            try:
                data = json.loads(critical_path.read_text())
                if isinstance(data, list):
                    return set(data)
                elif isinstance(data, dict):
                    return set(data.get("features", []))
            except Exception:
                pass
    return CRITICAL_FEATURES.copy()


def enforce_share_feature_contract(
    df: pd.DataFrame,
    expected_cols: list[str],
    *,
    indicator_suffix: str = "_is_nan",
    fill_value: float = 0.0,
    critical_features: set[str] | None = None,
) -> tuple[pd.DataFrame, ContractReport]:
    """Ensure DataFrame satisfies the share model feature contract.
    
    This function:
    1. For columns with `{indicator_suffix}` indicators: create indicator FIRST
       (before filling NaNs in base) so we capture original NaN status
    2. If base column is MISSING, set indicator to 1.0 for all rows (data was missing)
    3. Adds missing numeric columns with fill_value (default 0.0)
    4. Coerces expected columns to numeric
    5. Fills NaN values with fill_value
    6. Tracks critical feature missingness for severity reporting
    
    Args:
        df: Input DataFrame
        expected_cols: List of expected column names
        indicator_suffix: Suffix for NaN indicator columns (default "_is_nan")
        fill_value: Value to fill for missing/NaN values (default 0.0)
        critical_features: Set of critical feature names (uses defaults if None)
        
    Returns:
        Tuple of (fixed_df, contract_report)
    """
    df = df.copy()
    report = ContractReport(n_expected=len(expected_cols))
    
    if not expected_cols:
        return df, report
    
    if critical_features is None:
        critical_features = CRITICAL_FEATURES
    
    missing_cols = []
    base_missing_cols = []
    critical_missing = []
    nan_indicators_created = 0
    cols_filled = 0
    
    # Identify indicator columns and their bases
    indicator_cols = {col for col in expected_cols if col.endswith(indicator_suffix)}
    base_for_indicator = {
        col: col[: -len(indicator_suffix)] for col in indicator_cols
    }
    
    # Track which columns are originally missing in the input df
    # This is important for correctly tracking base_missing_cols
    originally_missing = {col for col in expected_cols if col not in df.columns}
    
    # FIRST PASS: Create indicators from existing bases BEFORE we fill their NaNs
    for ind_col in indicator_cols:
        if ind_col not in df.columns:
            base_col = base_for_indicator[ind_col]
            if base_col in df.columns:
                # Create indicator from base's current NaN status (before we fill NaNs)
                df[ind_col] = df[base_col].isna().astype(float)
                nan_indicators_created += 1
                missing_cols.append(ind_col)
            # If base doesn't exist, we'll handle it in second pass
    
    # SECOND PASS: Fill missing columns and NaN values
    for col in expected_cols:
        if col not in df.columns:
            if col not in missing_cols:  # Don't re-add indicators we just created
                missing_cols.append(col)
            
            # Check if this is a critical feature
            # Strip _is_nan suffix for comparison
            base_name = col[:-len(indicator_suffix)] if col.endswith(indicator_suffix) else col
            if base_name in critical_features or col in critical_features:
                if col not in critical_missing:
                    critical_missing.append(col)
            
            # Check if this is an indicator column (already handled above if base existed)
            if col in indicator_cols:
                if col not in df.columns:  # Base didn't exist originally
                    base_col = base_for_indicator[col]
                    # Check if base was ORIGINALLY missing (use our pre-computed set)
                    base_was_originally_missing = base_col in originally_missing
                    # Base doesn't exist, create it
                    if base_col not in df.columns:
                        df[base_col] = fill_value
                    # Track that base was missing from original data
                    if base_was_originally_missing and base_col not in base_missing_cols:
                        base_missing_cols.append(base_col)
                    # Set indicator to 1.0 when base was entirely missing (footgun prevention!)
                    df[col] = 1.0 if base_was_originally_missing else 0.0
                    nan_indicators_created += 1
            else:
                # Regular missing column - fill with default
                df[col] = fill_value
                cols_filled += 1
        else:
            # Column exists - ensure numeric and fill NaNs
            if df[col].dtype == object:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            if df[col].isna().any():
                df[col] = df[col].fillna(fill_value)
                cols_filled += 1
    
    # Calculate severity
    n_critical = len(critical_missing)
    n_important = len([c for c in missing_cols if c not in critical_missing and not c.endswith(indicator_suffix)])
    
    if n_critical >= 3:
        severity = SEVERITY_CRITICAL
    elif n_critical >= 1 or n_important >= 5:
        severity = SEVERITY_IMPORTANT
    else:
        severity = SEVERITY_MINOR
    
    # Populate report
    report.n_missing = len(missing_cols)
    report.n_present = report.n_expected - report.n_missing
    report.missing_cols = missing_cols
    report.missing_feature_frac = len(missing_cols) / len(expected_cols) if expected_cols else 0.0
    report.nan_indicators_created = nan_indicators_created
    report.cols_filled_with_zero = cols_filled
    report.n_critical_missing = n_critical
    report.n_important_missing = n_important
    report.critical_missing = critical_missing
    report.severity = severity
    report.base_missing_cols = base_missing_cols
    
    return df, report


def get_missing_feature_summary(reports: list[ContractReport]) -> dict[str, int]:
    """Aggregate missing columns across multiple reports.
    
    Args:
        reports: List of ContractReport objects
        
    Returns:
        Dict mapping column name to count of times it was missing
    """
    missing_counts: dict[str, int] = {}
    for report in reports:
        for col in report.missing_cols:
            missing_counts[col] = missing_counts.get(col, 0) + 1
    return dict(sorted(missing_counts.items(), key=lambda x: -x[1]))


def get_severity_summary(reports: list[ContractReport]) -> dict[str, int]:
    """Aggregate severity levels across reports.
    
    Args:
        reports: List of ContractReport objects
        
    Returns:
        Dict mapping severity level to count
    """
    counts = {SEVERITY_CRITICAL: 0, SEVERITY_IMPORTANT: 0, SEVERITY_MINOR: 0}
    for report in reports:
        counts[report.severity] = counts.get(report.severity, 0) + 1
    return counts


__all__ = [
    "ContractReport",
    "CRITICAL_FEATURES",
    "SEVERITY_CRITICAL",
    "SEVERITY_IMPORTANT", 
    "SEVERITY_MINOR",
    "load_expected_share_features",
    "load_critical_features",
    "enforce_share_feature_contract",
    "get_missing_feature_summary",
    "get_severity_summary",
]
