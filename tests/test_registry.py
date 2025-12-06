"""Tests for model registry module."""

from __future__ import annotations

from pathlib import Path

import pytest

from projections.registry.manifest import (
    ModelVersion,
    ModelEntry,
    ModelManifest,
    load_manifest,
    save_manifest,
    register_model,
    promote_model,
    get_production_model,
    list_versions,
)


def test_model_version_to_from_dict() -> None:
    """ModelVersion should round-trip through dict."""
    version = ModelVersion(
        version="1.0.0",
        run_id="20241201T030000Z",
        artifact_path="/path/to/model",
        stage="dev",
        metrics={"mae": 3.5, "coverage": 0.8},
        description="Test model",
    )
    data = version.to_dict()
    restored = ModelVersion.from_dict(data)
    assert restored.version == version.version
    assert restored.run_id == version.run_id
    assert restored.metrics == version.metrics


def test_manifest_save_load(tmp_path: Path) -> None:
    """Manifest should persist and reload correctly."""
    manifest_path = tmp_path / "manifest.json"
    manifest = ModelManifest()
    entry = manifest.ensure_model("test_model")
    entry.versions.append(
        ModelVersion(
            version="1.0.0",
            run_id="run123",
            artifact_path="/test/path",
        )
    )
    save_manifest(manifest, manifest_path)
    loaded = load_manifest(manifest_path)
    assert "test_model" in loaded.models
    assert len(loaded.models["test_model"].versions) == 1


def test_register_model() -> None:
    """register_model should add new version to manifest."""
    manifest = ModelManifest()
    version = register_model(
        manifest,
        model_name="minutes_v1_lgbm",
        version="1.0.0",
        run_id="20241201T030000Z",
        artifact_path="artifacts/minutes_lgbm/run1",
        training_start="2024-10-01",
        training_end="2024-11-30",
        metrics={"val_mae": 3.2},
    )
    assert version.version == "1.0.0"
    assert manifest.get_model("minutes_v1_lgbm") is not None
    assert len(manifest.models["minutes_v1_lgbm"].versions) == 1


def test_register_duplicate_version_fails() -> None:
    """Registering duplicate version should fail."""
    manifest = ModelManifest()
    register_model(
        manifest,
        model_name="test",
        version="1.0.0",
        run_id="run1",
        artifact_path="/path1",
    )
    with pytest.raises(ValueError, match="already exists"):
        register_model(
            manifest,
            model_name="test",
            version="1.0.0",
            run_id="run2",
            artifact_path="/path2",
        )


def test_promote_model() -> None:
    """promote_model should update stage and pointers."""
    manifest = ModelManifest()
    register_model(
        manifest,
        model_name="test",
        version="1.0.0",
        run_id="run1",
        artifact_path="/path",
    )
    version = promote_model(manifest, model_name="test", version="1.0.0", stage="prod")
    assert version.stage == "prod"
    assert version.promoted_at is not None
    assert manifest.models["test"].current_prod == "1.0.0"


def test_promote_nonexistent_fails() -> None:
    """Promoting nonexistent model should fail."""
    manifest = ModelManifest()
    with pytest.raises(ValueError, match="not found"):
        promote_model(manifest, model_name="missing", version="1.0.0", stage="prod")


def test_get_production_model() -> None:
    """get_production_model should return current prod version."""
    manifest = ModelManifest()
    register_model(
        manifest,
        model_name="test",
        version="1.0.0",
        run_id="run1",
        artifact_path="/path",
    )
    promote_model(manifest, model_name="test", version="1.0.0", stage="prod")
    prod = get_production_model(manifest, "test")
    assert prod is not None
    assert prod.version == "1.0.0"


def test_get_production_model_none() -> None:
    """get_production_model should return None if no prod version."""
    manifest = ModelManifest()
    register_model(
        manifest,
        model_name="test",
        version="1.0.0",
        run_id="run1",
        artifact_path="/path",
    )
    # Not promoted yet
    prod = get_production_model(manifest, "test")
    assert prod is None


def test_list_versions_filter_by_stage() -> None:
    """list_versions should support stage filtering."""
    manifest = ModelManifest()
    register_model(manifest, model_name="test", version="1.0.0", run_id="r1", artifact_path="/p1")
    register_model(manifest, model_name="test", version="1.1.0", run_id="r2", artifact_path="/p2")
    promote_model(manifest, model_name="test", version="1.0.0", stage="prod")
    promote_model(manifest, model_name="test", version="1.1.0", stage="staging")

    all_versions = list_versions(manifest, "test")
    assert len(all_versions) == 2

    prod_only = list_versions(manifest, "test", stage="prod")
    assert len(prod_only) == 1
    assert prod_only[0].version == "1.0.0"

    staging_only = list_versions(manifest, "test", stage="staging")
    assert len(staging_only) == 1
    assert staging_only[0].version == "1.1.0"
