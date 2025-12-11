from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import shutil
import pandas as pd
import pytest
from typer.testing import CliRunner

from projections.fpts_v1.datasets import FptsDatasetBuilder
from projections.models import fpts_lgbm

from tests.fpts.conftest import (
    TEST_MINUTES_RUN_ID,
    _write_boxscores,
    _write_features,
    _write_legacy_prediction_logs,
    _write_prediction_logs,
)


def test_dataset_builder_merges_predictions(sample_data_root: Path) -> None:
    builder = FptsDatasetBuilder(
        data_root=sample_data_root,
        history_days=1,
        minutes_source="predicted",
        minutes_run_id=TEST_MINUTES_RUN_ID,
    )
    frame = builder.build(datetime(2024, 10, 1), datetime(2024, 10, 1))
    assert not frame.empty
    assert (frame["actual_minutes"] > 0).all()
    assert {"minutes_p50_pred", "fpts_per_min_prior_5", "teammate_out_count"}.issubset(frame.columns)
    assert frame["minutes_p50_pred"].iloc[0] == pytest.approx(33.0)
    assert (frame["teammate_out_count"] >= 0).all()


def test_dataset_builder_legacy_prediction_logs(tmp_path: Path) -> None:
    data_root = tmp_path / "legacy"
    _write_features(data_root, "2024-10-02")
    _write_boxscores(data_root, "2024-10-02")
    _write_legacy_prediction_logs(data_root, "2024-10-02")

    builder = FptsDatasetBuilder(
        data_root=data_root,
        history_days=1,
        minutes_source="predicted",
        minutes_run_id=TEST_MINUTES_RUN_ID,
    )
    frame = builder.build(datetime(2024, 10, 2), datetime(2024, 10, 2))
    assert not frame.empty


def test_dataset_builder_actual_minutes_without_logs(tmp_path: Path) -> None:
    data_root = tmp_path / "actual_only"
    _write_features(data_root, "2024-10-03")
    _write_boxscores(data_root, "2024-10-03")
    builder = FptsDatasetBuilder(
        data_root=data_root,
        history_days=1,
        minutes_source="actual",
    )
    frame = builder.build(datetime(2024, 10, 3), datetime(2024, 10, 3))
    assert not frame.empty
    assert (frame["minutes_p50_pred"] == frame["actual_minutes"]).all()
    assert (frame["play_prob_pred"] == 1.0).all()


def test_dataset_builder_predicted_minutes_missing_logs(tmp_path: Path) -> None:
    data_root = tmp_path / "missing_logs"
    _write_features(data_root, "2024-10-04")
    _write_boxscores(data_root, "2024-10-04")
    builder = FptsDatasetBuilder(
        data_root=data_root,
        history_days=1,
        minutes_source="predicted",
        minutes_run_id=TEST_MINUTES_RUN_ID,
    )
    with pytest.raises(FileNotFoundError):
        builder.build(datetime(2024, 10, 4), datetime(2024, 10, 4))


def test_training_cli_writes_artifacts(sample_data_root: Path, tmp_path: Path) -> None:
    for legacy in (
        sample_data_root / "gold" / "prediction_logs_minutes",
        sample_data_root / "gold" / "prediction_logs_minutes_v1",
    ):
        if legacy.exists():
            shutil.rmtree(legacy)
    runner = CliRunner()
    result = runner.invoke(
        fpts_lgbm.app,
        [
            "--run-id",
            "testrun",
            "--data-root",
            str(sample_data_root),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--train-start",
            "2024-10-01",
            "--train-end",
            "2024-10-01",
            "--cal-start",
            "2024-10-01",
            "--cal-end",
            "2024-10-01",
            "--minutes-source",
            "actual",
            "--minutes-run-id",
            TEST_MINUTES_RUN_ID,
        ],
    )
    assert result.exit_code == 0, result.output
    run_dir = tmp_path / "artifacts" / "testrun"
    assert run_dir.exists()
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert "train" in metrics and "cal" in metrics
    train_metrics = metrics["train"]
    assert "baseline_mae_per_min" in train_metrics
    assert "model_mae_per_min" in train_metrics
    assert train_metrics["baseline_mae_fpts"] >= 0
    assert train_metrics["model_mae_fpts"] >= 0
    assert "buckets" in train_metrics and "starters" in train_metrics["buckets"]
    assert (run_dir / "model.joblib").exists()
    report = (run_dir / "report.md").read_text()
    assert "Model vs Baseline" in report


def test_feature_selection_excludes_leakage(sample_data_root: Path) -> None:
    builder = FptsDatasetBuilder(
        data_root=sample_data_root,
        history_days=1,
        minutes_source="predicted",
        minutes_run_id=TEST_MINUTES_RUN_ID,
    )
    frame = builder.build(datetime(2024, 10, 1), datetime(2024, 10, 1))
    columns = fpts_lgbm._infer_safe_feature_columns(
        frame,
        target_col="fpts_per_min_actual",
        extra_excluded={"actual_minutes", "actual_fpts"},
    )
    assert columns
    for col in columns:
        lowered = col.lower()
        assert "_game" not in lowered
        assert not lowered.startswith("actual_")
