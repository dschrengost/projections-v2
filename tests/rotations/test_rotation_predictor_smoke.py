"""Smoke test for rotation predictor v1 CLI."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

# The artifact we know exists
ROTATION_SET_ARTIFACT = Path(
    "artifacts/rotation_set_minutes/rotation_set_minutes_v1_settransformer_20260126T024705Z"
)


@pytest.fixture
def temp_artifact_dir():
    """Create a temporary directory for test artifacts."""
    tmpdir = tempfile.mkdtemp(prefix="rot_pred_test_")
    yield Path(tmpdir)
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.skipif(
    not ROTATION_SET_ARTIFACT.exists(),
    reason=f"Rotation-set artifact not found: {ROTATION_SET_ARTIFACT}",
)
def test_rotation_predictor_cli_smoke(temp_artifact_dir: Path):
    """Smoke test: run CLI with --max-rows 5000 and verify outputs."""
    run_id = "smoke_test"

    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "projections.cli.train_rotation_predictor_v1",
        "--rotation-set-artifact",
        str(ROTATION_SET_ARTIFACT),
        "--run-id",
        run_id,
        "--artifact-root",
        str(temp_artifact_dir),
        "--max-rows",
        "5000",
        "--overwrite",
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=600,  # 10 minute timeout
        cwd=Path(__file__).parent.parent.parent,  # repo root
    )

    # Print output for debugging
    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

    assert result.returncode == 0, f"CLI failed with: {result.stderr}"

    # Verify outputs exist
    run_dir = temp_artifact_dir / run_id

    assert run_dir.exists(), f"Run directory not created: {run_dir}"
    assert (run_dir / "report.md").exists(), "report.md not created"
    assert (run_dir / "predictions_all.parquet").exists(), "predictions_all.parquet not created"
    assert (run_dir / "predictions_test.parquet").exists(), "predictions_test.parquet not created"
    assert (run_dir / "model_ge5.lgb").exists(), "model_ge5.lgb not created"
    assert (run_dir / "model_ge15.lgb").exists(), "model_ge15.lgb not created"
    assert (run_dir / "meta.json").exists(), "meta.json not created"

    # Verify report.md has content
    report_content = (run_dir / "report.md").read_text()
    assert "# Rotation Predictor v1 Report" in report_content
    assert "Brier score" in report_content

    # Verify predictions parquet is readable
    import pandas as pd

    preds = pd.read_parquet(run_dir / "predictions_test.parquet")
    assert len(preds) > 0, "predictions_test.parquet is empty"
    assert "p_ge5" in preds.columns
    assert "p_ge15" in preds.columns
    assert "y_ge5" in preds.columns
    assert "y_ge15" in preds.columns
    assert "minutes_actual" in preds.columns

    # Probabilities should be in [0, 1]
    assert (preds["p_ge5"] >= 0).all() and (preds["p_ge5"] <= 1).all()
    assert (preds["p_ge15"] >= 0).all() and (preds["p_ge15"] <= 1).all()
