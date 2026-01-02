"""Tests for promotion gates."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestCheckPromotionReadiness:
    """Tests for check_promotion_readiness function."""

    @pytest.fixture
    def mock_client(self):
        """Create a mocked MLflow client."""
        with patch("projections.registry.promotion_gates.MlflowClient") as MockClient:
            client = MagicMock()
            MockClient.return_value = client
            yield client

    def test_passes_with_all_requirements(self, mock_client):
        """Test that check passes when all requirements are met."""
        # Mock run
        mock_run = MagicMock()
        mock_run.info.status = "FINISHED"
        mock_run.data.metrics = {"val_mae": 3.5, "val_rmse": 5.2}
        mock_client.get_run.return_value = mock_run

        # Mock artifacts
        mock_artifacts = [
            MagicMock(path="dataset_manifest.json"),
            MagicMock(path="feature_schema.json"),
            MagicMock(path="model.joblib"),
        ]
        mock_client.list_artifacts.return_value = mock_artifacts

        from projections.registry.promotion_gates import check_promotion_readiness

        passed, issues = check_promotion_readiness("run_123")

        assert passed is True
        assert issues == []

    def test_fails_without_metrics(self, mock_client):
        """Test that check fails when metrics are missing."""
        mock_run = MagicMock()
        mock_run.info.status = "FINISHED"
        mock_run.data.metrics = {}  # No metrics
        mock_client.get_run.return_value = mock_run

        mock_artifacts = [
            MagicMock(path="dataset_manifest.json"),
            MagicMock(path="feature_schema.json"),
            MagicMock(path="model.joblib"),
        ]
        mock_client.list_artifacts.return_value = mock_artifacts

        from projections.registry.promotion_gates import check_promotion_readiness

        passed, issues = check_promotion_readiness("run_123")

        assert passed is False
        assert any("metrics" in issue.lower() for issue in issues)

    def test_fails_without_dataset_manifest(self, mock_client):
        """Test that check fails when dataset manifest is missing."""
        mock_run = MagicMock()
        mock_run.info.status = "FINISHED"
        mock_run.data.metrics = {"mae": 3.5}
        mock_client.get_run.return_value = mock_run

        # Missing dataset_manifest.json
        mock_artifacts = [
            MagicMock(path="feature_schema.json"),
            MagicMock(path="model.joblib"),
        ]
        mock_client.list_artifacts.return_value = mock_artifacts

        from projections.registry.promotion_gates import check_promotion_readiness

        passed, issues = check_promotion_readiness("run_123")

        assert passed is False
        assert any("dataset_manifest.json" in issue for issue in issues)

    def test_fails_without_feature_schema(self, mock_client):
        """Test that check fails when feature schema is missing."""
        mock_run = MagicMock()
        mock_run.info.status = "FINISHED"
        mock_run.data.metrics = {"rmse": 5.2}
        mock_client.get_run.return_value = mock_run

        # Missing feature_schema.json
        mock_artifacts = [
            MagicMock(path="dataset_manifest.json"),
            MagicMock(path="model.joblib"),
        ]
        mock_client.list_artifacts.return_value = mock_artifacts

        from projections.registry.promotion_gates import check_promotion_readiness

        passed, issues = check_promotion_readiness("run_123")

        assert passed is False
        assert any("feature_schema.json" in issue for issue in issues)

    def test_fails_without_model_artifact(self, mock_client):
        """Test that check fails when model artifact is missing."""
        mock_run = MagicMock()
        mock_run.info.status = "FINISHED"
        mock_run.data.metrics = {"mae": 3.5}
        mock_client.get_run.return_value = mock_run

        # No model artifact
        mock_artifacts = [
            MagicMock(path="dataset_manifest.json"),
            MagicMock(path="feature_schema.json"),
        ]
        mock_client.list_artifacts.return_value = mock_artifacts

        from projections.registry.promotion_gates import check_promotion_readiness

        passed, issues = check_promotion_readiness("run_123")

        assert passed is False
        assert any("model" in issue.lower() for issue in issues)

    def test_fails_for_running_run(self, mock_client):
        """Test that check fails when run is not finished."""
        mock_run = MagicMock()
        mock_run.info.status = "RUNNING"
        mock_run.data.metrics = {"mae": 3.5}
        mock_client.get_run.return_value = mock_run

        mock_artifacts = [
            MagicMock(path="dataset_manifest.json"),
            MagicMock(path="feature_schema.json"),
            MagicMock(path="model.joblib"),
        ]
        mock_client.list_artifacts.return_value = mock_artifacts

        from projections.registry.promotion_gates import check_promotion_readiness

        passed, issues = check_promotion_readiness("run_123")

        assert passed is False
        assert any("RUNNING" in issue for issue in issues)


class TestGetRunMetadata:
    """Tests for get_run_metadata function."""

    def test_extracts_metadata(self):
        """Test that metadata is correctly extracted from run."""
        with patch("projections.registry.promotion_gates.MlflowClient") as MockClient:
            client = MagicMock()
            MockClient.return_value = client

            mock_run = MagicMock()
            mock_run.data.tags = {
                "mlflow.runName": "my_run",
                "git_sha": "abc123",
                "custom_tag": "value",
            }
            mock_run.data.params = {"lr": "0.01", "epochs": "100"}
            mock_run.data.metrics = {"mae": 3.5, "rmse": 5.2}
            mock_run.info.experiment_id = "exp_1"
            mock_run.info.status = "FINISHED"
            mock_run.info.start_time = 1234567890
            mock_run.info.end_time = 1234567900
            client.get_run.return_value = mock_run

            from projections.registry.promotion_gates import get_run_metadata

            metadata = get_run_metadata("run_xyz")

            assert metadata["run_id"] == "run_xyz"
            assert metadata["run_name"] == "my_run"
            assert metadata["git_sha"] == "abc123"
            assert metadata["metrics"] == {"mae": 3.5, "rmse": 5.2}
            assert metadata["status"] == "FINISHED"
            assert "custom_tag" in metadata["tags"]
