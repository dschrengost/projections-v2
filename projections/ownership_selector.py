"""Resolve active ownership live-scoring selector config."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from projections import model_selectors

OwnershipSource = str
OwnershipModelFamily = str


@dataclass(frozen=True)
class OwnershipSelector:
    source: OwnershipSource
    model_family: OwnershipModelFamily
    model_run: str | None
    gtv2_features_path: str | None
    fallback_source: OwnershipSource | None
    fallback_model_family: OwnershipModelFamily | None
    fallback_model_run: str | None
    fallback_gtv2_features_path: str | None
    config_path: Path


def _normalize_source(value: object, *, field: str) -> str:
    source = str(value or "").strip().lower()
    if source not in {"linestar", "internal"}:
        raise RuntimeError(f"{field} must be one of linestar|internal, got {value!r}")
    return source


def _normalize_model_family(value: object, *, field: str) -> str:
    family = str(value or "").strip().lower()
    if family not in {"ownership_v1", "ownership_v2"}:
        raise RuntimeError(f"{field} must be one of ownership_v1|ownership_v2, got {value!r}")
    return family


def _normalize_optional_path(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def load_ownership_selector(
    *,
    config_path: Path | None = None,
    data_root: Path | None = None,
    project_root: Path | None = None,
) -> OwnershipSelector:
    """Load active ownership selector config for live orchestration."""

    resolved = (
        (config_path.expanduser().resolve() if config_path is not None else None)
        or model_selectors.active_ownership_selector_path(data_root=data_root, project_root=project_root)
    )
    if not resolved.exists():
        raise RuntimeError(f"ownership selector config not found at {resolved}")

    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in ownership selector {resolved}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"ownership selector must be a JSON object: {resolved}")

    source = _normalize_source(payload.get("source", "internal"), field="source")
    model_family = _normalize_model_family(
        payload.get("model_family", "ownership_v1"),
        field="model_family",
    )
    model_run_raw = payload.get("model_run")
    model_run = None if model_run_raw is None else str(model_run_raw).strip() or None
    gtv2_features_path = _normalize_optional_path(payload.get("gtv2_features_path"))

    fallback_source_raw = payload.get("fallback_source")
    if fallback_source_raw is None or str(fallback_source_raw).strip() == "":
        fallback_source = None
    else:
        fallback_source = _normalize_source(fallback_source_raw, field="fallback_source")

    fallback_family_raw = payload.get("fallback_model_family")
    if fallback_family_raw is None or str(fallback_family_raw).strip() == "":
        fallback_model_family = None
    else:
        fallback_model_family = _normalize_model_family(
            fallback_family_raw,
            field="fallback_model_family",
        )

    fallback_run_raw = payload.get("fallback_model_run")
    fallback_model_run = (
        None
        if fallback_run_raw is None
        else str(fallback_run_raw).strip() or None
    )
    fallback_gtv2_features_path = _normalize_optional_path(
        payload.get("fallback_gtv2_features_path")
    )

    # If fallback source is internal but family omitted, default rollback to v1.
    if fallback_source == "internal" and fallback_model_family is None:
        fallback_model_family = "ownership_v1"

    return OwnershipSelector(
        source=source,
        model_family=model_family,
        model_run=model_run,
        gtv2_features_path=gtv2_features_path,
        fallback_source=fallback_source,
        fallback_model_family=fallback_model_family,
        fallback_model_run=fallback_model_run,
        fallback_gtv2_features_path=fallback_gtv2_features_path,
        config_path=resolved,
    )

