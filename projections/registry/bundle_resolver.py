"""Bundle resolvers for sim_v2 inputs.

Provides registry-first resolution for minutes and rates bundles, with
filesystem fallback for development and backward compatibility.

Environment variables:
    MINUTES_USE_FILESYSTEM_BUNDLE=1  - Force filesystem loading for minutes
    RATES_USE_FILESYSTEM_BUNDLE=1    - Force filesystem loading for rates
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

try:
    from mlflow.tracking import MlflowClient

    MLFLOW_AVAILABLE = True
except ImportError:
    MlflowClient = None  # type: ignore
    MLFLOW_AVAILABLE = False


def _get_git_sha() -> str | None:
    """Get current git SHA for provenance."""
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]
    except Exception:
        pass
    return None


def _use_minutes_filesystem_fallback() -> bool:
    """Check if minutes filesystem fallback is enabled."""
    val = os.environ.get("MINUTES_USE_FILESYSTEM_BUNDLE", "").strip().lower()
    return val in {"1", "true", "yes"}


def _use_rates_filesystem_fallback() -> bool:
    """Check if rates filesystem fallback is enabled."""
    val = os.environ.get("RATES_USE_FILESYSTEM_BUNDLE", "").strip().lower()
    return val in {"1", "true", "yes"}


def _get_tracking_uri() -> str | None:
    """Get MLflow tracking URI from environment or mlflow_utils."""
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        return uri
    try:
        from projections.mlflow_utils import get_tracking_uri

        return get_tracking_uri()
    except ImportError:
        return None


@lru_cache(maxsize=4)
def resolve_minutes_bundle(
    model: str = "minutes_v1",
    alias: str = "production",
    *,
    tracking_uri: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Resolve minutes quantile bundle artifacts.

    Returns:
        Tuple of (bundle_data, metadata) where metadata includes:
        - source: "mlflow_registry" or "filesystem"
        - run_id: the resolved run ID
        - alias: the requested alias
        - model_name: the model name
    """
    # Check filesystem fallback
    if _use_minutes_filesystem_fallback():
        return _load_minutes_from_filesystem()

    # Try MLflow registry
    uri = tracking_uri or _get_tracking_uri()
    if uri and MLFLOW_AVAILABLE:
        try:
            from projections.registry.model_resolver import resolve_model

            bundle, meta = resolve_model(model, alias, tracking_uri=uri)
            meta["model_name"] = model
            meta["alias"] = alias
            return bundle, meta
        except Exception:
            pass  # Fall through to filesystem

    # Fallback to filesystem
    return _load_minutes_from_filesystem()


def _load_minutes_from_filesystem() -> tuple[Any, dict[str, Any]]:
    """Load minutes bundle from filesystem."""
    from projections.minutes_v1.production import load_production_minutes_bundle

    bundle = load_production_minutes_bundle()
    run_id = bundle.get("run_id", "unknown") if isinstance(bundle, dict) else "unknown"

    metadata = {
        "source": "filesystem",
        "run_id": run_id,
        "alias": "production",
        "model_name": "minutes_v1",
    }
    return bundle, metadata


@lru_cache(maxsize=4)
def resolve_rates_bundle(
    model: str = "rates_v1",
    alias: str = "production",
    *,
    tracking_uri: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Resolve rates bundle artifacts.

    Returns:
        Tuple of (bundle_data, metadata) where metadata includes:
        - source: "mlflow_registry" or "filesystem"
        - run_id: the resolved run ID
        - alias: the requested alias
        - model_name: the model name
    """
    # Check filesystem fallback
    if _use_rates_filesystem_fallback():
        return _load_rates_from_filesystem()

    # Try MLflow registry
    uri = tracking_uri or _get_tracking_uri()
    if uri and MLFLOW_AVAILABLE:
        try:
            from projections.registry.model_resolver import resolve_model

            bundle, meta = resolve_model(model, alias, tracking_uri=uri)
            meta["model_name"] = model
            meta["alias"] = alias
            return bundle, meta
        except Exception:
            pass  # Fall through to filesystem

    # Fallback to filesystem
    return _load_rates_from_filesystem()


def _load_rates_from_filesystem() -> tuple[Any, dict[str, Any]]:
    """Load rates bundle from filesystem."""
    from projections.rates_v1.production import load_production_rates_bundle

    bundle = load_production_rates_bundle()
    run_id = getattr(bundle, "run_id", None) or "unknown"

    metadata = {
        "source": "filesystem",
        "run_id": run_id,
        "alias": "production",
        "model_name": "rates_v1",
    }
    return bundle, metadata


def clear_bundle_caches() -> None:
    """Clear all bundle resolver caches."""
    resolve_minutes_bundle.cache_clear()
    resolve_rates_bundle.cache_clear()


def build_provenance(
    *,
    minutes_meta: dict[str, Any] | None = None,
    rates_meta: dict[str, Any] | None = None,
    sim_profile: str | None = None,
    seed: int | str | None = None,
    n_worlds: int | None = None,
) -> dict[str, Any]:
    """Build provenance dict for run manifest.

    Args:
        minutes_meta: Metadata from resolve_minutes_bundle()
        rates_meta: Metadata from resolve_rates_bundle()
        sim_profile: Simulation profile name
        seed: Random seed used
        n_worlds: Number of worlds simulated

    Returns:
        Provenance dict suitable for inclusion in latest_run.json
    """
    provenance: dict[str, Any] = {}

    if minutes_meta:
        provenance["minutes_v1"] = {
            "alias": minutes_meta.get("alias", "production"),
            "run_id": minutes_meta.get("run_id"),
            "source": minutes_meta.get("source"),
        }

    if rates_meta:
        provenance["rates_v1"] = {
            "alias": rates_meta.get("alias", "production"),
            "run_id": rates_meta.get("run_id"),
            "source": rates_meta.get("source"),
        }

    if sim_profile:
        provenance["sim_profile"] = sim_profile

    if seed is not None:
        provenance["seed"] = seed if isinstance(seed, int) else str(seed)
    else:
        provenance["seed"] = "random"

    if n_worlds:
        provenance["n_worlds"] = n_worlds

    git_sha = _get_git_sha()
    if git_sha:
        provenance["git_sha"] = git_sha

    return provenance


__all__ = [
    "resolve_minutes_bundle",
    "resolve_rates_bundle",
    "clear_bundle_caches",
    "build_provenance",
]
