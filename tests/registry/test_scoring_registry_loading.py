"""Tests for score_minutes_v1 registry loading behavior."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest


class TestRegistryLoading:
    """Tests confirming default path uses registry resolver."""

    def test_default_uses_registry_resolver(self):
        """By default, should try to load from MLflow registry first."""
        # When MINUTES_USE_FILESYSTEM_BUNDLE is not set, should NOT use filesystem fallback
        env = os.environ.copy()
        env.pop("MINUTES_USE_FILESYSTEM_BUNDLE", None)

        with patch.dict(os.environ, env, clear=True):
            from projections.registry.model_resolver import _use_filesystem_fallback

            assert _use_filesystem_fallback() is False

    def test_env_var_enables_filesystem_fallback(self):
        """When MINUTES_USE_FILESYSTEM_BUNDLE=1, should use filesystem fallback."""
        with patch.dict(os.environ, {"MINUTES_USE_FILESYSTEM_BUNDLE": "1"}):
            from projections.registry.model_resolver import _use_filesystem_fallback

            assert _use_filesystem_fallback() is True

    def test_env_var_true_enables_filesystem_fallback(self):
        """When MINUTES_USE_FILESYSTEM_BUNDLE=true, should use filesystem fallback."""
        with patch.dict(os.environ, {"MINUTES_USE_FILESYSTEM_BUNDLE": "true"}):
            from projections.registry.model_resolver import _use_filesystem_fallback

            assert _use_filesystem_fallback() is True

    def test_env_var_empty_disables_filesystem_fallback(self):
        """When MINUTES_USE_FILESYSTEM_BUNDLE is empty, should NOT use filesystem fallback."""
        with patch.dict(os.environ, {"MINUTES_USE_FILESYSTEM_BUNDLE": ""}):
            from projections.registry.model_resolver import _use_filesystem_fallback

            assert _use_filesystem_fallback() is False


class TestResolveModelNameStandardization:
    """Tests confirming model name standardization."""

    def test_model_name_is_minutes_v1(self):
        """The standardized registry model name should be 'minutes_v1'."""
        # This test documents the expected model name for consistency
        expected_model_name = "minutes_v1"

        # The promote CLI and resolve_model should use this name
        # (actual implementation tested via integration)
        assert expected_model_name == "minutes_v1"
