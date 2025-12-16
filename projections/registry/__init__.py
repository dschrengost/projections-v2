"""Model registry for versioned model management.

Provides a lightweight JSON-based registry for tracking model versions,
enabling promotion between stages, and rollback capabilities.
"""

from projections.registry.manifest import (
    ModelVersion,
    ModelManifest,
    load_manifest,
    save_manifest,
    register_model,
    promote_model,
    get_production_model,
    list_versions,
)

__all__ = [
    "ModelVersion",
    "ModelManifest",
    "load_manifest",
    "save_manifest",
    "register_model",
    "promote_model",
    "get_production_model",
    "list_versions",
]
