"""Tests for minutes parity audit module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from projections.minutes.parity import (
    ColumnParityIssue,
    ParityReport,
    compute_parity_report,
    inject_parity_into_summary,
    write_parity_report,
)


@pytest.fixture
def mock_model_dir(tmp_path: Path) -> Path:
    """Create a mock model directory with feature_columns.json."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    feature_cols = {
        "columns": [
            "feature_a",
            "feature_b",
            "feature_c",
            "feature_d",
        ]
    }
    (model_dir / "feature_columns.json").write_text(
        json.dumps(feature_cols), encoding="utf-8"
    )
    return model_dir


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create a sample feature DataFrame."""
    return pd.DataFrame({
        "game_id": [1, 1, 2, 2],
        "team_id": [100, 100, 200, 200],
        "player_id": [1001, 1002, 2001, 2002],
        "feature_a": [1.0, 2.0, 3.0, 4.0],
        "feature_b": [0.5, 0.6, 0.7, 0.8],
        "feature_c": [10, 20, 30, 40],
        "feature_d": [0.1, 0.2, 0.3, 0.4],
    })


class TestComputeParityReport:
    """Tests for compute_parity_report function."""

    def test_all_features_present_passes(
        self, mock_model_dir: Path, sample_df: pd.DataFrame
    ) -> None:
        """Parity passes when all expected features are present."""
        report = compute_parity_report(
            sample_df,
            model_dir=mock_model_dir,
            game_date="2026-01-24",
            run_id="test-run",
        )

        assert report.parity_passed is True
        assert report.failure_reason == ""
        assert len(report.missing_features) == 0
        assert report.n_errors == 0

    def test_missing_feature_triggers_error(
        self, mock_model_dir: Path, sample_df: pd.DataFrame
    ) -> None:
        """Missing required feature triggers a parity error."""
        # Remove feature_c
        df_missing = sample_df.drop(columns=["feature_c"])

        report = compute_parity_report(
            df_missing,
            model_dir=mock_model_dir,
            game_date="2026-01-24",
            run_id="test-run",
        )

        assert report.parity_passed is False
        assert "feature_c" in report.missing_features
        assert report.n_errors >= 1
        assert "Missing" in report.failure_reason

    def test_multiple_missing_features(
        self, mock_model_dir: Path
    ) -> None:
        """Multiple missing features are all reported."""
        df = pd.DataFrame({
            "game_id": [1],
            "team_id": [100],
            "player_id": [1001],
            "feature_a": [1.0],
            # feature_b, feature_c, feature_d all missing
        })

        report = compute_parity_report(
            df,
            model_dir=mock_model_dir,
            game_date="2026-01-24",
            run_id="test-run",
        )

        assert report.parity_passed is False
        assert set(report.missing_features) == {"feature_b", "feature_c", "feature_d"}
        assert report.n_errors == 3

    def test_high_nan_rate_warning(
        self, mock_model_dir: Path
    ) -> None:
        """High NaN rate triggers a warning."""
        df = pd.DataFrame({
            "game_id": [1, 2, 3, 4, 5],
            "team_id": [100, 100, 100, 100, 100],
            "player_id": [1001, 1002, 1003, 1004, 1005],
            "feature_a": [1.0, None, None, None, 5.0],  # 60% NaN
            "feature_b": [0.5, 0.6, 0.7, 0.8, 0.9],
            "feature_c": [10, 20, 30, 40, 50],
            "feature_d": [0.1, 0.2, 0.3, 0.4, 0.5],
        })

        report = compute_parity_report(
            df,
            model_dir=mock_model_dir,
            game_date="2026-01-24",
            run_id="test-run",
            nan_rate_warning_threshold=0.5,
        )

        # Still passes but has warnings
        assert report.parity_passed is True
        assert report.n_warnings >= 1
        warning_cols = [w.column for w in report.warnings]
        assert "feature_a" in warning_cols

    def test_high_nan_rate_error(
        self, mock_model_dir: Path
    ) -> None:
        """Very high NaN rate triggers an error."""
        df = pd.DataFrame({
            "game_id": [1, 2, 3, 4, 5],
            "team_id": [100, 100, 100, 100, 100],
            "player_id": [1001, 1002, 1003, 1004, 1005],
            "feature_a": [None, None, None, None, 5.0],  # 80% NaN
            "feature_b": [0.5, 0.6, 0.7, 0.8, 0.9],
            "feature_c": [10, 20, 30, 40, 50],
            "feature_d": [0.1, 0.2, 0.3, 0.4, 0.5],
        })

        report = compute_parity_report(
            df,
            model_dir=mock_model_dir,
            game_date="2026-01-24",
            run_id="test-run",
            nan_rate_error_threshold=0.75,
        )

        # Should fail due to high NaN rate
        assert report.parity_passed is False
        assert report.n_errors >= 1

    def test_value_ranges_computed(
        self, mock_model_dir: Path, sample_df: pd.DataFrame
    ) -> None:
        """Value ranges are computed for all features."""
        report = compute_parity_report(
            sample_df,
            model_dir=mock_model_dir,
            game_date="2026-01-24",
            run_id="test-run",
        )

        assert "feature_a" in report.value_range_by_column
        ranges = report.value_range_by_column["feature_a"]
        assert "min" in ranges
        assert "max" in ranges
        assert "mean" in ranges
        assert ranges["min"] == 1.0
        assert ranges["max"] == 4.0

    def test_summary_line_format(
        self, mock_model_dir: Path, sample_df: pd.DataFrame
    ) -> None:
        """Summary line has expected format."""
        report = compute_parity_report(
            sample_df,
            model_dir=mock_model_dir,
            game_date="2026-01-24",
            run_id="test-run",
        )

        summary = report.summary_line()
        assert "[parity]" in summary
        assert "PASS" in summary or "FAIL" in summary


class TestParityReportSerialization:
    """Tests for ParityReport serialization."""

    def test_to_dict(self, mock_model_dir: Path, sample_df: pd.DataFrame) -> None:
        """Report can be converted to dict."""
        report = compute_parity_report(
            sample_df,
            model_dir=mock_model_dir,
            game_date="2026-01-24",
            run_id="test-run",
        )

        d = report.to_dict()
        assert isinstance(d, dict)
        assert d["game_date"] == "2026-01-24"
        assert d["run_id"] == "test-run"
        assert d["parity_passed"] is True

    def test_to_json(self, mock_model_dir: Path, sample_df: pd.DataFrame) -> None:
        """Report can be serialized to JSON."""
        report = compute_parity_report(
            sample_df,
            model_dir=mock_model_dir,
            game_date="2026-01-24",
            run_id="test-run",
        )

        json_str = report.to_json()
        parsed = json.loads(json_str)
        assert parsed["game_date"] == "2026-01-24"


class TestWriteParityReport:
    """Tests for write_parity_report function."""

    def test_writes_json_file(
        self, mock_model_dir: Path, sample_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        """Parity report is written to a JSON file."""
        report = compute_parity_report(
            sample_df,
            model_dir=mock_model_dir,
            game_date="2026-01-24",
            run_id="test-run",
        )

        output_dir = tmp_path / "output"
        output_path = write_parity_report(report, output_dir)

        assert output_path.exists()
        assert output_path.name == "parity_report.json"

        content = json.loads(output_path.read_text(encoding="utf-8"))
        assert content["parity_passed"] is True


class TestInjectParityIntoSummary:
    """Tests for inject_parity_into_summary function."""

    def test_injects_into_existing_summary(
        self, mock_model_dir: Path, sample_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        """Parity is injected into existing summary.json."""
        summary_path = tmp_path / "summary.json"
        summary_path.write_text(
            json.dumps({"existing_key": "value"}),
            encoding="utf-8",
        )

        report = compute_parity_report(
            sample_df,
            model_dir=mock_model_dir,
            game_date="2026-01-24",
            run_id="test-run",
        )

        inject_parity_into_summary(summary_path, report)

        content = json.loads(summary_path.read_text(encoding="utf-8"))
        assert "existing_key" in content
        assert "parity" in content
        assert content["parity"]["passed"] is True

    def test_creates_summary_if_missing(
        self, mock_model_dir: Path, sample_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        """Summary is created if it doesn't exist."""
        summary_path = tmp_path / "new_summary.json"
        assert not summary_path.exists()

        report = compute_parity_report(
            sample_df,
            model_dir=mock_model_dir,
            game_date="2026-01-24",
            run_id="test-run",
        )

        inject_parity_into_summary(summary_path, report)

        assert summary_path.exists()
        content = json.loads(summary_path.read_text(encoding="utf-8"))
        assert "parity" in content


class TestColumnParityIssue:
    """Tests for ColumnParityIssue dataclass."""

    def test_to_dict(self) -> None:
        """Issue can be converted to dict."""
        issue = ColumnParityIssue(
            column="test_col",
            issue_type="missing",
            severity="error",
            details={"expected": True, "found": False},
        )

        d = issue.to_dict()
        assert d["column"] == "test_col"
        assert d["issue_type"] == "missing"
        assert d["severity"] == "error"
        assert d["details"] == {"expected": True, "found": False}
