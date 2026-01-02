"""Tests for bundle resolver."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest


class TestResolveRatesBundle:
    """Tests for resolve_rates_bundle function."""

    def test_default_uses_registry(self):
        """By default, should try to load from MLflow registry first."""
        # When RATES_USE_FILESYSTEM_BUNDLE is not set, should NOT use filesystem fallback
        env = os.environ.copy()
        env.pop("RATES_USE_FILESYSTEM_BUNDLE", None)

        with patch.dict(os.environ, env, clear=True):
            from projections.registry.bundle_resolver import _use_rates_filesystem_fallback

            assert _use_rates_filesystem_fallback() is False

    def test_env_var_enables_filesystem_fallback(self):
        """When RATES_USE_FILESYSTEM_BUNDLE=1, should use filesystem fallback."""
        with patch.dict(os.environ, {"RATES_USE_FILESYSTEM_BUNDLE": "1"}):
            from projections.registry.bundle_resolver import _use_rates_filesystem_fallback

            assert _use_rates_filesystem_fallback() is True

    def test_env_var_true_enables_filesystem_fallback(self):
        """When RATES_USE_FILESYSTEM_BUNDLE=true, should use filesystem fallback."""
        with patch.dict(os.environ, {"RATES_USE_FILESYSTEM_BUNDLE": "true"}):
            from projections.registry.bundle_resolver import _use_rates_filesystem_fallback

            assert _use_rates_filesystem_fallback() is True

    def test_env_var_empty_disables_filesystem_fallback(self):
        """When RATES_USE_FILESYSTEM_BUNDLE is empty, should NOT use filesystem fallback."""
        with patch.dict(os.environ, {"RATES_USE_FILESYSTEM_BUNDLE": ""}):
            from projections.registry.bundle_resolver import _use_rates_filesystem_fallback

            assert _use_rates_filesystem_fallback() is False


class TestResolveMinutesBundle:
    """Tests for resolve_minutes_bundle function."""

    def test_default_uses_registry(self):
        """Minutes resolver should also try registry by default."""
        env = os.environ.copy()
        env.pop("MINUTES_USE_FILESYSTEM_BUNDLE", None)

        with patch.dict(os.environ, env, clear=True):
            from projections.registry.bundle_resolver import _use_minutes_filesystem_fallback

            assert _use_minutes_filesystem_fallback() is False


class TestBuildProvenance:
    """Tests for build_provenance function."""

    def test_builds_complete_provenance(self):
        """Should build provenance dict with all expected keys."""
        from projections.registry.bundle_resolver import build_provenance

        minutes_meta = {
            "source": "mlflow_registry",
            "run_id": "minutes_run_abc123",
            "alias": "production",
        }
        rates_meta = {
            "source": "filesystem",
            "run_id": "rates_run_xyz789",
            "alias": "production",
        }

        provenance = build_provenance(
            minutes_meta=minutes_meta,
            rates_meta=rates_meta,
            sim_profile="baseline",
            seed=42,
            n_worlds=1000,
        )

        # Check minutes_v1 block
        assert "minutes_v1" in provenance
        assert provenance["minutes_v1"]["alias"] == "production"
        assert provenance["minutes_v1"]["run_id"] == "minutes_run_abc123"
        assert provenance["minutes_v1"]["source"] == "mlflow_registry"

        # Check rates_v1 block
        assert "rates_v1" in provenance
        assert provenance["rates_v1"]["alias"] == "production"
        assert provenance["rates_v1"]["run_id"] == "rates_run_xyz789"
        assert provenance["rates_v1"]["source"] == "filesystem"

        # Check other fields
        assert provenance["sim_profile"] == "baseline"
        assert provenance["seed"] == 42
        assert provenance["n_worlds"] == 1000

    def test_handles_missing_metadata(self):
        """Should handle missing metadata gracefully."""
        from projections.registry.bundle_resolver import build_provenance

        provenance = build_provenance(
            minutes_meta=None,
            rates_meta=None,
            sim_profile="test",
        )

        assert "minutes_v1" not in provenance
        assert "rates_v1" not in provenance
        assert provenance["sim_profile"] == "test"
        assert provenance["seed"] == "random"

    def test_seed_defaults_to_random(self):
        """When seed is None, should record 'random'."""
        from projections.registry.bundle_resolver import build_provenance

        provenance = build_provenance(seed=None)
        assert provenance["seed"] == "random"


class TestClearBundleCaches:
    """Tests for cache management."""

    def test_clear_bundle_caches(self):
        """Should clear caches without error."""
        from projections.registry.bundle_resolver import clear_bundle_caches

        # Should not raise
        clear_bundle_caches()
