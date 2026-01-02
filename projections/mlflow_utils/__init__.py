"""Centralized MLflow utilities for experiment tracking and model registry.

All MLflow configuration is read from environment variables:
- MLFLOW_TRACKING_URI: MLflow tracking server URI (required for production)
- MLFLOW_EXPERIMENT_NAME: Experiment name (default: projections-v2)

Example:
    from projections.mlflow_utils import start_run, log_params, log_metrics

    with start_run("my_training_run", tags={"model": "minutes_v1"}):
        log_params({"learning_rate": 0.01, "num_leaves": 31})
        log_metrics({"mae": 3.5, "rmse": 5.2})
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

try:
    import mlflow
    from mlflow.tracking import MlflowClient

    MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None  # type: ignore
    MlflowClient = None  # type: ignore
    MLFLOW_AVAILABLE = False

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    pd = None  # type: ignore
    PANDAS_AVAILABLE = False

__all__ = [
    "MLFLOW_AVAILABLE",
    "get_tracking_uri",
    "get_experiment_name",
    "get_client",
    "start_run",
    "log_params",
    "log_metrics",
    "log_artifact_text",
    "log_dataframe",
    "log_dataset_manifest",
    "log_schema",
    "register_model",
    "get_git_sha",
]

DEFAULT_EXPERIMENT_NAME = "projections-v2"
ENV_TRACKING_URI = "MLFLOW_TRACKING_URI"
ENV_EXPERIMENT_NAME = "MLFLOW_EXPERIMENT_NAME"


def get_tracking_uri() -> str | None:
    """Get MLflow tracking URI from environment."""
    return os.environ.get(ENV_TRACKING_URI)


def get_experiment_name() -> str:
    """Get MLflow experiment name from environment (default: projections-v2)."""
    return os.environ.get(ENV_EXPERIMENT_NAME, DEFAULT_EXPERIMENT_NAME)


def get_git_sha() -> str | None:
    """Get current git commit SHA, or None if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()[:12]
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return None


def get_client() -> Any:
    """Get MLflow client instance."""
    if not MLFLOW_AVAILABLE:
        raise RuntimeError("MLflow is not installed. Install with: pip install mlflow")
    tracking_uri = get_tracking_uri()
    if tracking_uri:
        return MlflowClient(tracking_uri=tracking_uri)
    return MlflowClient()


def _ensure_tracking_configured() -> None:
    """Configure MLflow tracking if URI is set."""
    if not MLFLOW_AVAILABLE:
        return
    tracking_uri = get_tracking_uri()
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    experiment_name = get_experiment_name()
    mlflow.set_experiment(experiment_name)


@contextmanager
def start_run(
    run_name: str,
    *,
    tags: dict[str, str] | None = None,
    git_sha: str | None = None,
    nested: bool = False,
) -> Iterator[Any]:
    """Start an MLflow run with the given name and tags.

    Args:
        run_name: Human-readable run name
        tags: Optional tags to attach to the run
        git_sha: Git commit SHA (auto-detected if None)
        nested: Whether this is a nested run

    Yields:
        The active MLflow run object

    Example:
        with start_run("training_v1", tags={"model": "minutes"}):
            log_params({"lr": 0.01})
            log_metrics({"mae": 3.5})
    """
    if not MLFLOW_AVAILABLE:
        # Yield a dummy context when MLflow is not available
        yield None
        return

    _ensure_tracking_configured()

    # Build tags
    all_tags = dict(tags) if tags else {}
    if git_sha is None:
        git_sha = get_git_sha()
    if git_sha:
        all_tags["git_sha"] = git_sha
    all_tags["run_started_at"] = datetime.now(timezone.utc).isoformat()

    with mlflow.start_run(run_name=run_name, tags=all_tags, nested=nested) as run:
        yield run


def log_params(params: dict[str, Any]) -> None:
    """Log parameters to the active MLflow run.

    Handles nested dicts by flattening with underscore separators.
    Skips None values and complex types (lists, dicts).
    """
    if not MLFLOW_AVAILABLE:
        return

    flat_params: dict[str, Any] = {}

    def flatten(obj: dict[str, Any], prefix: str = "") -> None:
        for key, value in obj.items():
            full_key = f"{prefix}{key}" if prefix else key
            if value is None:
                continue
            if isinstance(value, dict):
                flatten(value, f"{full_key}_")
            elif isinstance(value, (str, int, float, bool)):
                flat_params[full_key] = value
            # Skip lists and other complex types

    flatten(params)
    if flat_params:
        mlflow.log_params(flat_params)


def log_metrics(metrics: dict[str, Any], step: int | None = None) -> None:
    """Log metrics to the active MLflow run.

    Handles nested dicts by flattening with underscore separators.
    Only logs numeric values.
    """
    if not MLFLOW_AVAILABLE:
        return

    flat_metrics: dict[str, float] = {}

    def sanitize_name(name: str) -> str:
        """Sanitize metric name for MLflow compatibility."""
        return (
            name.replace("|", "_")
            .replace("<", "lt")
            .replace(">", "gt")
            .replace("(", "")
            .replace(")", "")
            .replace(" ", "_")
        )

    def flatten(obj: dict[str, Any], prefix: str = "") -> None:
        for key, value in obj.items():
            full_key = sanitize_name(f"{prefix}{key}" if prefix else key)
            if value is None:
                continue
            if isinstance(value, dict):
                flatten(value, f"{full_key}_")
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                flat_metrics[full_key] = float(value)

    flatten(metrics)
    if flat_metrics:
        if step is not None:
            for key, value in flat_metrics.items():
                mlflow.log_metric(key, value, step=step)
        else:
            mlflow.log_metrics(flat_metrics)


def log_artifact_text(name: str, text: str) -> None:
    """Log a text blob as an artifact.

    Args:
        name: Artifact filename (e.g., "config.json")
        text: Text content to log
    """
    if not MLFLOW_AVAILABLE:
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / name
        path.write_text(text, encoding="utf-8")
        mlflow.log_artifact(str(path))


def log_dataframe(name: str, df: Any) -> None:
    """Log a DataFrame as a parquet artifact.

    Args:
        name: Artifact filename (without extension, .parquet will be added)
        df: pandas DataFrame to log
    """
    if not MLFLOW_AVAILABLE or not PANDAS_AVAILABLE:
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        filename = name if name.endswith(".parquet") else f"{name}.parquet"
        path = Path(tmpdir) / filename
        df.to_parquet(path, index=False)
        mlflow.log_artifact(str(path))


def log_dataset_manifest(manifest: dict[str, Any]) -> None:
    """Log dataset manifest with hashes, paths, and time windows.

    The manifest should include:
    - data_hash: Hash of the training data
    - data_path: Path to the training data
    - train_start, train_end: Training window
    - val_start, val_end: Validation window (optional)
    - feature_count: Number of features
    - row_count: Number of rows

    Args:
        manifest: Dictionary containing dataset metadata
    """
    if not MLFLOW_AVAILABLE:
        return

    # Add timestamp
    manifest_with_ts = dict(manifest)
    manifest_with_ts["logged_at"] = datetime.now(timezone.utc).isoformat()

    log_artifact_text("dataset_manifest.json", json.dumps(manifest_with_ts, indent=2))

    # Also log key fields as params for searchability
    param_keys = ["data_hash", "train_start", "train_end", "val_start", "val_end", "feature_count", "row_count"]
    params = {k: manifest.get(k) for k in param_keys if manifest.get(k) is not None}
    if params:
        log_params({"dataset": params})


def log_schema(schema: dict[str, Any] | list[str]) -> None:
    """Log feature schema/contract.

    Args:
        schema: Either a dict with schema details or a list of feature names
    """
    if not MLFLOW_AVAILABLE:
        return

    if isinstance(schema, list):
        schema_dict = {"feature_names": schema, "feature_count": len(schema)}
    else:
        schema_dict = schema

    schema_dict["logged_at"] = datetime.now(timezone.utc).isoformat()
    log_artifact_text("feature_schema.json", json.dumps(schema_dict, indent=2))


def register_model(
    run_id: str,
    model_name: str,
    artifact_path: str = "model",
) -> str | None:
    """Register a model from a run to the MLflow Model Registry.

    Args:
        run_id: The MLflow run ID containing the model
        model_name: Name to register the model under (e.g., "minutes_v1")
        artifact_path: Path within the run's artifacts where the model is stored

    Returns:
        The model version string, or None if registration failed
    """
    if not MLFLOW_AVAILABLE:
        return None

    try:
        model_uri = f"runs:/{run_id}/{artifact_path}"
        result = mlflow.register_model(model_uri, model_name)
        return str(result.version)
    except Exception as e:
        print(f"[mlflow] Warning: Failed to register model: {e}")
        return None


def compute_data_hash(data: Any, sample_size: int = 10000) -> str:
    """Compute a hash of training data for reproducibility tracking.

    Args:
        data: A pandas DataFrame or path to data file
        sample_size: Number of rows to sample for hashing (for performance)

    Returns:
        SHA256 hash string (first 16 chars)
    """
    if PANDAS_AVAILABLE and hasattr(data, "sample"):
        # DataFrame
        if len(data) > sample_size:
            sample = data.sample(n=sample_size, random_state=42)
        else:
            sample = data
        content = sample.to_json(orient="records")
    elif isinstance(data, (str, Path)):
        # File path - hash file contents
        path = Path(data)
        if path.exists():
            content = path.read_bytes()[:1_000_000].decode("utf-8", errors="ignore")
        else:
            content = str(data)
    else:
        content = str(data)

    return hashlib.sha256(content.encode()).hexdigest()[:16]
