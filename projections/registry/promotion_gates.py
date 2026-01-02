"""Promotion gates for model registry.

Lightweight precheck functions that ensure required artifacts exist
before allowing model promotion to production.
"""

from __future__ import annotations

from typing import Any

try:
    from mlflow.tracking import MlflowClient

    MLFLOW_AVAILABLE = True
except ImportError:
    MlflowClient = None  # type: ignore
    MLFLOW_AVAILABLE = False


# Required artifacts for promotion
REQUIRED_ARTIFACTS = [
    "dataset_manifest.json",
    "feature_schema.json",
]

# Required metrics (at least one must be present)
REQUIRED_METRICS = [
    "mae",
    "rmse",
    "val_mae",
    "val_rmse",
    "train_mae",
    "train_rmse",
]


def check_promotion_readiness(
    run_id: str,
    *,
    tracking_uri: str | None = None,
    require_model_artifact: bool = True,
    model_artifact_path: str = "model",
) -> tuple[bool, list[str]]:
    """Check if an MLflow run is ready for promotion.

    Args:
        run_id: The MLflow run ID to check
        tracking_uri: Optional tracking URI (uses env default if None)
        require_model_artifact: Whether to require model artifact
        model_artifact_path: Path where model artifact should exist

    Returns:
        Tuple of (passed, list_of_issues).
        If passed is True, list_of_issues is empty.
    """
    if not MLFLOW_AVAILABLE:
        return False, ["MLflow is not installed"]

    issues: list[str] = []

    try:
        client = MlflowClient(tracking_uri=tracking_uri)
        run = client.get_run(run_id)
    except Exception as e:
        return False, [f"Failed to fetch run {run_id}: {e}"]

    # Check run status
    if run.info.status != "FINISHED":
        issues.append(f"Run status is '{run.info.status}', expected 'FINISHED'")

    # Get artifacts list
    try:
        artifacts = client.list_artifacts(run_id)
        artifact_names = {a.path for a in artifacts}
    except Exception as e:
        issues.append(f"Failed to list artifacts: {e}")
        artifact_names = set()

    # Check required artifacts
    for required in REQUIRED_ARTIFACTS:
        if required not in artifact_names:
            issues.append(f"Missing required artifact: {required}")

    # Check model artifact if required
    if require_model_artifact:
        # Model artifacts are typically directories, check for common patterns
        has_model = any(
            a.startswith(model_artifact_path) or a == f"{model_artifact_path}.joblib"
            for a in artifact_names
        )
        # Also check for joblib files at root (common pattern)
        if not has_model:
            has_model = any(a.endswith(".joblib") for a in artifact_names)
        if not has_model:
            issues.append(f"Missing model artifact (expected: {model_artifact_path})")

    # Check metrics
    metrics = run.data.metrics
    has_required_metric = any(
        any(m.lower().endswith(req) or m.lower() == req for req in REQUIRED_METRICS)
        for m in metrics.keys()
    )
    if not has_required_metric:
        issues.append(f"Missing required metrics (need at least one of: {REQUIRED_METRICS})")

    return len(issues) == 0, issues


def get_run_metadata(run_id: str, *, tracking_uri: str | None = None) -> dict[str, Any]:
    """Get metadata about an MLflow run for promotion records.

    Args:
        run_id: The MLflow run ID
        tracking_uri: Optional tracking URI

    Returns:
        Dictionary with run metadata
    """
    if not MLFLOW_AVAILABLE:
        return {"error": "MLflow not available"}

    try:
        client = MlflowClient(tracking_uri=tracking_uri)
        run = client.get_run(run_id)
    except Exception as e:
        return {"error": str(e)}

    return {
        "run_id": run_id,
        "run_name": run.data.tags.get("mlflow.runName", run_id),
        "experiment_id": run.info.experiment_id,
        "status": run.info.status,
        "start_time": run.info.start_time,
        "end_time": run.info.end_time,
        "git_sha": run.data.tags.get("git_sha"),
        "metrics": dict(run.data.metrics),
        "params_count": len(run.data.params),
        "tags": {k: v for k, v in run.data.tags.items() if not k.startswith("mlflow.")},
    }


def validate_model_exists(
    model_name: str,
    version: str | None = None,
    alias: str | None = None,
    *,
    tracking_uri: str | None = None,
) -> tuple[bool, str | None]:
    """Check if a model version or alias exists in the registry.

    Args:
        model_name: Name of the registered model
        version: Specific version to check (optional)
        alias: Alias to check (optional, e.g., "production")
        tracking_uri: Optional tracking URI

    Returns:
        Tuple of (exists, error_message)
    """
    if not MLFLOW_AVAILABLE:
        return False, "MLflow not available"

    try:
        client = MlflowClient(tracking_uri=tracking_uri)

        if alias:
            # Check for alias
            model_version = client.get_model_version_by_alias(model_name, alias)
            return True, None
        elif version:
            # Check for specific version
            model_version = client.get_model_version(model_name, version)
            return True, None
        else:
            # Just check if model exists at all
            model = client.get_registered_model(model_name)
            return True, None

    except Exception as e:
        return False, str(e)
