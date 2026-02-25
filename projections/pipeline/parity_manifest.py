"""Parity manifest helpers for GameTransformerV2 live inference.

This module defines a strict manifest contract used to enforce
training/inference parity at runtime.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

PARITY_MANIFEST_FILENAME = "parity_manifest.json"
PARITY_MANIFEST_VERSION = 1


class ParityManifestError(RuntimeError):
    """Raised when parity manifest payload is invalid."""


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def stable_json_sha256(payload: Any) -> str:
    """Return a stable SHA-256 hash for a JSON-serializable payload."""
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def sha256_file(path: Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    """Return SHA-256 hash for a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_bytes)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def hash_paths(paths: Iterable[Path]) -> str:
    """Return a deterministic hash over an ordered list of file hashes."""
    normalized: list[dict[str, str]] = []
    for p in sorted((Path(x).resolve() for x in paths), key=lambda x: str(x)):
        if not p.exists() or not p.is_file():
            continue
        normalized.append({"path": str(p), "sha256": sha256_file(p)})
    return stable_json_sha256(normalized)


def infer_feature_schema(
    features_df: pd.DataFrame,
    *,
    feature_columns: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Infer ordered feature schema from a feature frame."""
    cols = list(feature_columns) if feature_columns is not None else list(features_df.columns)
    schema: list[dict[str, Any]] = []
    for col in cols:
        if col not in features_df.columns:
            raise ParityManifestError(f"feature column missing from frame: {col}")
        series = features_df[col]
        schema.append(
            {
                "name": str(col),
                "dtype": str(series.dtype),
                "nullable": bool(series.isna().any()),
            }
        )
    return schema


def _assert_manifest_payload(payload: dict[str, Any]) -> None:
    required_keys = {
        "manifest_version",
        "model_id",
        "created_at",
        "feature_schema",
        "missing_value_policy",
        "transform_manifest",
        "output_manifest",
        "integrity",
    }
    missing = sorted(required_keys - set(payload.keys()))
    if missing:
        raise ParityManifestError(f"parity manifest missing required keys: {missing}")
    if int(payload.get("manifest_version", -1)) != PARITY_MANIFEST_VERSION:
        raise ParityManifestError(
            "unsupported parity manifest version "
            f"{payload.get('manifest_version')} (expected {PARITY_MANIFEST_VERSION})"
        )
    feature_schema = payload.get("feature_schema")
    if not isinstance(feature_schema, list) or len(feature_schema) <= 0:
        raise ParityManifestError("feature_schema must be a non-empty list")
    for idx, row in enumerate(feature_schema):
        if not isinstance(row, dict):
            raise ParityManifestError(f"feature_schema[{idx}] must be an object")
        for key in ("name", "dtype", "nullable"):
            if key not in row:
                raise ParityManifestError(f"feature_schema[{idx}] missing key: {key}")


def compute_manifest_hash(payload: dict[str, Any]) -> str:
    """Compute stable hash for a parity manifest payload."""
    cloned = json.loads(json.dumps(payload))
    integrity = cloned.get("integrity")
    if isinstance(integrity, dict):
        integrity.pop("parity_manifest_hash", None)
    return stable_json_sha256(cloned)


def build_parity_manifest(
    *,
    model_id: str,
    features_df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    missing_value_policy: dict[str, Any] | None = None,
    transform_manifest: dict[str, Any] | None = None,
    output_manifest: dict[str, Any] | None = None,
    integrity: dict[str, Any] | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build a parity manifest payload from runtime inputs."""
    payload: dict[str, Any] = {
        "manifest_version": PARITY_MANIFEST_VERSION,
        "model_id": str(model_id),
        "created_at": str(created_at or _utc_now_iso()),
        "feature_schema": infer_feature_schema(features_df, feature_columns=feature_columns),
        "missing_value_policy": dict(missing_value_policy or {}),
        "transform_manifest": dict(transform_manifest or {}),
        "output_manifest": dict(output_manifest or {}),
        "integrity": dict(integrity or {}),
    }
    payload["integrity"]["parity_manifest_hash"] = compute_manifest_hash(payload)
    _assert_manifest_payload(payload)
    return payload


def write_parity_manifest(path: Path, payload: dict[str, Any]) -> Path:
    """Write parity manifest to disk after validation."""
    _assert_manifest_payload(payload)
    manifest_hash = compute_manifest_hash(payload)
    existing_hash = str(payload.get("integrity", {}).get("parity_manifest_hash", ""))
    if existing_hash != manifest_hash:
        raise ParityManifestError(
            "integrity.parity_manifest_hash mismatch: "
            f"expected={manifest_hash} actual={existing_hash}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_parity_manifest(path: Path) -> dict[str, Any]:
    """Load parity manifest from disk and verify integrity hash."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ParityManifestError(f"parity manifest not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ParityManifestError(f"parity manifest is not valid JSON: {path}") from exc

    if not isinstance(payload, dict):
        raise ParityManifestError("parity manifest root must be an object")
    _assert_manifest_payload(payload)

    expected_hash = compute_manifest_hash(payload)
    actual_hash = str(payload.get("integrity", {}).get("parity_manifest_hash", ""))
    if expected_hash != actual_hash:
        raise ParityManifestError(
            "parity manifest hash validation failed: "
            f"expected={expected_hash} actual={actual_hash}"
        )
    return payload


def resolve_parity_manifest_path(bundle_dir: Path) -> Path:
    """Return the canonical parity manifest path for a bundle directory."""
    return Path(bundle_dir) / PARITY_MANIFEST_FILENAME
