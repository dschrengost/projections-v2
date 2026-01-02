"""Model resolver for loading models from MLflow registry.

Provides a unified interface for loading production models by alias,
with fallback to filesystem bundles for development/backward compatibility.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    from mlflow.tracking import MlflowClient
    import mlflow.pyfunc

    MLFLOW_AVAILABLE = True
except ImportError:
    MlflowClient = None  # type: ignore
    mlflow = None  # type: ignore
    MLFLOW_AVAILABLE = False

from projections.mlflow_utils import get_tracking_uri

# Environment variable to force filesystem bundle loading
ENV_USE_FILESYSTEM = "MINUTES_USE_FILESYSTEM_BUNDLE"


def _use_filesystem_fallback() -> bool:
    """Check if filesystem fallback is enabled via environment."""
    return os.environ.get(ENV_USE_FILESYSTEM, "").lower() in ("1", "true", "yes")


@lru_cache(maxsize=4)
def resolve_model(
    model_name: str,
    alias: str = "production",
    *,
    tracking_uri: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Load a model from MLflow registry by alias.

    Args:
        model_name: Registered model name (e.g., "minutes_v1")
        alias: Model alias (e.g., "production", "staging")
        tracking_uri: Optional tracking URI override

    Returns:
        Tuple of (model_object, metadata_dict)
        - model_object: The loaded model (type depends on how it was logged)
        - metadata_dict: Dictionary with run_id, version, alias, params, etc.

    Raises:
        RuntimeError: If model cannot be loaded and no fallback available
    """
    if _use_filesystem_fallback():
        # Import here to avoid circular imports
        from projections.minutes_v1.production import load_production_minutes_bundle

        bundle = load_production_minutes_bundle()
        metadata = {
            "source": "filesystem",
            "run_id": bundle.get("run_id"),
            "bundle_kind": bundle.get("bundle_kind"),
            "run_dir": bundle.get("run_dir"),
        }
        return bundle, metadata

    if not MLFLOW_AVAILABLE:
        raise RuntimeError(
            "MLflow is not available. Install with 'pip install mlflow' "
            f"or set {ENV_USE_FILESYSTEM}=1 to use filesystem bundles."
        )

    uri = tracking_uri or get_tracking_uri()
    if not uri:
        raise RuntimeError(
            f"MLFLOW_TRACKING_URI not set. Set the environment variable "
            f"or use {ENV_USE_FILESYSTEM}=1 for filesystem bundles."
        )

    try:
        client = MlflowClient(tracking_uri=uri)

        # Get model version by alias
        model_version = client.get_model_version_by_alias(model_name, alias)
        version = model_version.version
        run_id = model_version.run_id

        # Load the model
        model_uri = f"models:/{model_name}@{alias}"
        model = mlflow.pyfunc.load_model(model_uri)

        # Get run metadata
        run = client.get_run(run_id)

        metadata = {
            "source": "mlflow_registry",
            "model_name": model_name,
            "version": version,
            "alias": alias,
            "run_id": run_id,
            "run_name": run.data.tags.get("mlflow.runName"),
            "experiment_id": run.info.experiment_id,
            "git_sha": run.data.tags.get("git_sha"),
            "params": dict(run.data.params),
            "metrics": dict(run.data.metrics),
            "promotion_time": model_version.creation_timestamp,
        }

        return model, metadata

    except Exception as e:
        raise RuntimeError(
            f"Failed to load model '{model_name}' with alias '{alias}': {e}. "
            f"Set {ENV_USE_FILESYSTEM}=1 to fall back to filesystem bundles."
        ) from e


def resolve_model_version(
    model_name: str,
    version: str,
    *,
    tracking_uri: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Load a specific model version from MLflow registry.

    Args:
        model_name: Registered model name
        version: Specific version number (as string)
        tracking_uri: Optional tracking URI override

    Returns:
        Tuple of (model_object, metadata_dict)
    """
    if not MLFLOW_AVAILABLE:
        raise RuntimeError("MLflow is not available")

    uri = tracking_uri or get_tracking_uri()
    client = MlflowClient(tracking_uri=uri)

    model_version_info = client.get_model_version(model_name, version)
    run_id = model_version_info.run_id

    model_uri = f"models:/{model_name}/{version}"
    model = mlflow.pyfunc.load_model(model_uri)

    run = client.get_run(run_id)

    metadata = {
        "source": "mlflow_registry",
        "model_name": model_name,
        "version": version,
        "run_id": run_id,
        "run_name": run.data.tags.get("mlflow.runName"),
        "params": dict(run.data.params),
        "metrics": dict(run.data.metrics),
    }

    return model, metadata


def get_production_model_info(
    model_name: str,
    alias: str = "production",
    *,
    tracking_uri: str | None = None,
) -> dict[str, Any] | None:
    """Get metadata about the current production model without loading it.

    Args:
        model_name: Registered model name
        alias: Model alias to check
        tracking_uri: Optional tracking URI override

    Returns:
        Dictionary with model info, or None if not found
    """
    if _use_filesystem_fallback() or not MLFLOW_AVAILABLE:
        return None

    try:
        uri = tracking_uri or get_tracking_uri()
        client = MlflowClient(tracking_uri=uri)
        model_version = client.get_model_version_by_alias(model_name, alias)

        return {
            "model_name": model_name,
            "alias": alias,
            "version": model_version.version,
            "run_id": model_version.run_id,
            "creation_timestamp": model_version.creation_timestamp,
            "description": model_version.description,
        }
    except Exception:
        return None


def clear_model_cache() -> None:
    """Clear the model cache (useful for testing or forced reloads)."""
    resolve_model.cache_clear()
