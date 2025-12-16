"""JSON-based model registry manifest and operations.

This module provides a lightweight model registry that tracks versions,
supports staging (dev/staging/prod), and enables rollback without
external dependencies like MLflow.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Any

Stage = Literal["dev", "staging", "prod"]


@dataclass
class ModelVersion:
    """Metadata for a single model version."""

    version: str
    run_id: str
    artifact_path: str
    stage: Stage = "dev"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    promoted_at: str | None = None

    # Training metadata
    training_start: str | None = None
    training_end: str | None = None
    feature_schema_version: str | None = None
    feature_hash: str | None = None

    # Validation metrics
    metrics: dict[str, float] = field(default_factory=dict)

    # Notes
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelVersion":
        """Create from dict (e.g., loaded from JSON)."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ModelEntry:
    """Entry for a named model with its versions."""

    name: str
    versions: list[ModelVersion] = field(default_factory=list)
    current_prod: str | None = None
    current_staging: str | None = None

    def get_version(self, version: str) -> ModelVersion | None:
        """Get a specific version by version string."""
        for v in self.versions:
            if v.version == version:
                return v
        return None

    def get_by_run_id(self, run_id: str) -> ModelVersion | None:
        """Get a version by run_id."""
        for v in self.versions:
            if v.run_id == run_id:
                return v
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "name": self.name,
            "versions": [v.to_dict() for v in self.versions],
            "current_prod": self.current_prod,
            "current_staging": self.current_staging,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelEntry":
        """Create from dict."""
        return cls(
            name=data["name"],
            versions=[ModelVersion.from_dict(v) for v in data.get("versions", [])],
            current_prod=data.get("current_prod"),
            current_staging=data.get("current_staging"),
        )


@dataclass
class ModelManifest:
    """Root manifest containing all registered models."""

    models: dict[str, ModelEntry] = field(default_factory=dict)
    schema_version: str = "1.0.0"
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def get_model(self, name: str) -> ModelEntry | None:
        """Get model entry by name."""
        return self.models.get(name)

    def ensure_model(self, name: str) -> ModelEntry:
        """Get or create model entry."""
        if name not in self.models:
            self.models[name] = ModelEntry(name=name)
        return self.models[name]

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "schema_version": self.schema_version,
            "updated_at": self.updated_at,
            "models": {name: entry.to_dict() for name, entry in self.models.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelManifest":
        """Create from dict."""
        models = {
            name: ModelEntry.from_dict(entry)
            for name, entry in data.get("models", {}).items()
        }
        return cls(
            models=models,
            schema_version=data.get("schema_version", "1.0.0"),
            updated_at=data.get("updated_at", datetime.now(timezone.utc).isoformat()),
        )


DEFAULT_MANIFEST_PATH = Path("artifacts/registry/manifest.json")


def load_manifest(path: Path | None = None) -> ModelManifest:
    """Load manifest from JSON file, creating empty manifest if not exists."""
    manifest_path = path or DEFAULT_MANIFEST_PATH
    if not manifest_path.exists():
        return ModelManifest()
    with manifest_path.open("r") as f:
        data = json.load(f)
    return ModelManifest.from_dict(data)


def save_manifest(manifest: ModelManifest, path: Path | None = None) -> None:
    """Save manifest to JSON file."""
    manifest_path = path or DEFAULT_MANIFEST_PATH
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.updated_at = datetime.now(timezone.utc).isoformat()
    with manifest_path.open("w") as f:
        json.dump(manifest.to_dict(), f, indent=2)


def register_model(
    manifest: ModelManifest,
    *,
    model_name: str,
    version: str,
    run_id: str,
    artifact_path: str,
    training_start: str | None = None,
    training_end: str | None = None,
    feature_schema_version: str | None = None,
    feature_hash: str | None = None,
    metrics: dict[str, float] | None = None,
    description: str = "",
) -> ModelVersion:
    """Register a new model version in the manifest."""
    entry = manifest.ensure_model(model_name)

    # Check for duplicate version
    if entry.get_version(version):
        raise ValueError(f"Version {version} already exists for {model_name}")

    model_version = ModelVersion(
        version=version,
        run_id=run_id,
        artifact_path=artifact_path,
        stage="dev",
        training_start=training_start,
        training_end=training_end,
        feature_schema_version=feature_schema_version,
        feature_hash=feature_hash,
        metrics=metrics or {},
        description=description,
    )
    entry.versions.append(model_version)
    return model_version


def promote_model(
    manifest: ModelManifest,
    *,
    model_name: str,
    version: str,
    stage: Stage,
) -> ModelVersion:
    """Promote a model version to a new stage."""
    entry = manifest.get_model(model_name)
    if entry is None:
        raise ValueError(f"Model {model_name} not found in registry")

    model_version = entry.get_version(version)
    if model_version is None:
        raise ValueError(f"Version {version} not found for {model_name}")

    # Update version stage
    model_version.stage = stage
    model_version.promoted_at = datetime.now(timezone.utc).isoformat()

    # Update current pointers
    if stage == "prod":
        entry.current_prod = version
    elif stage == "staging":
        entry.current_staging = version

    return model_version


def get_production_model(
    manifest: ModelManifest,
    model_name: str,
) -> ModelVersion | None:
    """Get the current production version of a model."""
    entry = manifest.get_model(model_name)
    if entry is None or entry.current_prod is None:
        return None
    return entry.get_version(entry.current_prod)


def list_versions(
    manifest: ModelManifest,
    model_name: str,
    *,
    stage: Stage | None = None,
) -> list[ModelVersion]:
    """List all versions for a model, optionally filtered by stage."""
    entry = manifest.get_model(model_name)
    if entry is None:
        return []
    if stage is None:
        return list(entry.versions)
    return [v for v in entry.versions if v.stage == stage]


__all__ = [
    "ModelVersion",
    "ModelEntry",
    "ModelManifest",
    "Stage",
    "DEFAULT_MANIFEST_PATH",
    "load_manifest",
    "save_manifest",
    "register_model",
    "promote_model",
    "get_production_model",
    "list_versions",
]
