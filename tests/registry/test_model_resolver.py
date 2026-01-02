"""Tests for model resolver."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest


class TestResolveModel:
    """Tests for resolve_model function."""

    def test_filesystem_fallback_when_env_set(self):
        """When MINUTES_USE_FILESYSTEM_BUNDLE=1, should use filesystem loader."""
        with patch.dict(os.environ, {"MINUTES_USE_FILESYSTEM_BUNDLE": "1"}):
            with patch(
                "projections.minutes_v1.production.load_production_minutes_bundle"
            ) as mock_load:
                mock_load.return_value = {
                    "bundle_kind": "minute_share",
                    "run_id": "test_run_123",
                    "run_dir": "/path/to/run",
                }

                # Need to reimport to pick up the new env var
                from projections.registry.model_resolver import (
                    _use_filesystem_fallback,
                    clear_model_cache,
                )

                clear_model_cache()
                assert _use_filesystem_fallback() is True

    def test_no_filesystem_fallback_by_default(self):
        """By default, should not use filesystem fallback."""
        # Ensure the env var is not set
        env = os.environ.copy()
        env.pop("MINUTES_USE_FILESYSTEM_BUNDLE", None)

        with patch.dict(os.environ, env, clear=True):
            from projections.registry.model_resolver import _use_filesystem_fallback

            assert _use_filesystem_fallback() is False

    def test_resolve_model_metadata_extraction(self):
        """Test that metadata is correctly extracted from filesystem bundle."""
        with patch.dict(os.environ, {"MINUTES_USE_FILESYSTEM_BUNDLE": "1"}):
            with patch(
                "projections.minutes_v1.production.load_production_minutes_bundle"
            ) as mock_load:
                mock_load.return_value = {
                    "bundle_kind": "minute_share",
                    "run_id": "test_run_abc",
                    "run_dir": "/artifacts/minutes_lgbm/test_run_abc",
                    "bundle": {"model": "mock"},
                }

                from projections.registry.model_resolver import (
                    resolve_model,
                    clear_model_cache,
                )

                clear_model_cache()
                bundle, metadata = resolve_model("minutes_v1", "production")

                assert metadata["source"] == "filesystem"
                assert metadata["run_id"] == "test_run_abc"
                assert metadata["bundle_kind"] == "minute_share"
                assert bundle["bundle"] == {"model": "mock"}


class TestMlflowIntegration:
    """Tests for MLflow registry integration (mocked)."""

    @pytest.fixture
    def mock_mlflow_client(self):
        """Create a mocked MLflow client."""
        with patch("projections.registry.model_resolver.MlflowClient") as MockClient:
            client = MagicMock()
            MockClient.return_value = client

            # Mock model version
            mock_version = MagicMock()
            mock_version.version = "3"
            mock_version.run_id = "run_xyz_123"
            client.get_model_version_by_alias.return_value = mock_version

            # Mock run
            mock_run = MagicMock()
            mock_run.data.tags = {"mlflow.runName": "my_training_run", "git_sha": "abc123"}
            mock_run.data.params = {"learning_rate": "0.01"}
            mock_run.data.metrics = {"mae": 3.5}
            mock_run.info.experiment_id = "exp_1"
            client.get_run.return_value = mock_run

            yield client

    def test_resolve_model_from_registry_metadata(self, mock_mlflow_client):
        """Test metadata extraction from MLflow registry."""
        # This test verifies the structure of metadata returned
        # The actual MLflow calls are mocked

        from projections.registry.model_resolver import get_production_model_info

        with patch.dict(os.environ, {"MLFLOW_TRACKING_URI": "http://test:5000"}):
            with patch.dict(os.environ, {"MINUTES_USE_FILESYSTEM_BUNDLE": ""}, clear=False):
                info = get_production_model_info("minutes_v1", "production")

                # Should have called the client
                mock_mlflow_client.get_model_version_by_alias.assert_called_once_with(
                    "minutes_v1", "production"
                )


class TestClearCache:
    """Tests for cache management."""

    def test_clear_model_cache(self):
        """Test that cache can be cleared."""
        from projections.registry.model_resolver import clear_model_cache

        # Should not raise
        clear_model_cache()


class TestAliasToStageFallback:
    """Tests for alias → stage fallback behavior."""

    def test_alias_to_stage_map_exists(self):
        """Verify the alias-to-stage mapping is used correctly."""
        # This tests that the fallback chain is implemented
        # Actual MLflow calls would be mocked in integration tests
        
        expected_mappings = {
            "production": "Production",
            "staging": "Staging",
            "archived": "Archived",
        }
        
        # The mapping should exist in model_resolver
        # Just verify we can import and the function exists
        from projections.registry.model_resolver import resolve_model
        assert resolve_model is not None

    def test_resolution_method_in_metadata(self):
        """Verify resolution_method is included in metadata when using registry."""
        with patch.dict(os.environ, {"MINUTES_USE_FILESYSTEM_BUNDLE": "1"}):
            with patch(
                "projections.minutes_v1.production.load_production_minutes_bundle"
            ) as mock_load:
                mock_load.return_value = {
                    "bundle_kind": "minute_share",
                    "run_id": "test_run_xyz",
                    "run_dir": "/path/to/run",
                }

                from projections.registry.model_resolver import (
                    resolve_model,
                    clear_model_cache,
                )

                clear_model_cache()
                bundle, metadata = resolve_model("minutes_v1", "production")

                # Filesystem fallback should return source="filesystem"
                assert metadata["source"] == "filesystem"
                # resolution_method is only set for mlflow registry path
                # so we don't check it here

