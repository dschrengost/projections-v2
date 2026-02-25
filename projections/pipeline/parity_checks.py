"""Runtime parity checks for GameTransformerV2 live inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


class ParityCheckError(RuntimeError):
    """Raised when runtime parity validation fails."""


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _normalize_dtype_name(dtype: Any) -> str:
    return str(dtype)


def _require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def validate_feature_frame_against_manifest(
    features_df: pd.DataFrame,
    manifest: Mapping[str, Any],
    *,
    require_exact_order: bool = True,
) -> dict[str, Any]:
    """Validate feature schema/order/dtypes/null policy against manifest."""
    feature_schema = manifest.get("feature_schema")
    if not isinstance(feature_schema, list) or len(feature_schema) <= 0:
        raise ParityCheckError("manifest.feature_schema must be a non-empty list")

    expected_cols = [str(row.get("name")) for row in feature_schema]
    expected_dtype = {str(row.get("name")): str(row.get("dtype")) for row in feature_schema}
    expected_nullable = {str(row.get("name")): bool(row.get("nullable", True)) for row in feature_schema}
    actual_cols = list(features_df.columns)

    errors: list[str] = []

    expected_set = set(expected_cols)
    actual_set = set(actual_cols)
    missing = sorted(expected_set - actual_set)
    extra = sorted(actual_set - expected_set)
    if missing:
        errors.append(f"missing feature columns: {missing}")
    if extra:
        errors.append(f"unexpected feature columns: {extra}")

    if require_exact_order and not missing and not extra and expected_cols != actual_cols:
        errors.append(
            "feature column order mismatch: "
            f"expected={expected_cols[:12]} actual={actual_cols[:12]}"
        )

    shared_cols = [c for c in expected_cols if c in features_df.columns]
    dtype_mismatch: dict[str, dict[str, str]] = {}
    for col in shared_cols:
        actual_dtype = _normalize_dtype_name(features_df[col].dtype)
        expected = _normalize_dtype_name(expected_dtype[col])
        if actual_dtype != expected:
            dtype_mismatch[col] = {"expected": expected, "actual": actual_dtype}
    if dtype_mismatch:
        errors.append(f"dtype mismatch: {dtype_mismatch}")

    # Column-level nullable contract from schema.
    nullable_violations: list[str] = []
    for col in shared_cols:
        if not bool(expected_nullable.get(col, True)) and bool(features_df[col].isna().any()):
            nullable_violations.append(col)
    if nullable_violations:
        errors.append(f"non-nullable columns contain nulls: {sorted(nullable_violations)}")

    # Optional missing-value policy overlay.
    policy = manifest.get("missing_value_policy")
    if isinstance(policy, Mapping):
        disallow_null_cols = [str(c) for c in policy.get("disallow_null_columns", [])]
        bad = [c for c in disallow_null_cols if c in features_df.columns and bool(features_df[c].isna().any())]
        if bad:
            errors.append(f"disallow_null_columns violated: {sorted(bad)}")

    if errors:
        raise ParityCheckError("feature parity check failed: " + "; ".join(errors))

    return {
        "row_count": int(len(features_df)),
        "column_count": int(len(actual_cols)),
        "columns": list(actual_cols),
    }


def validate_transform_manifest(
    manifest: Mapping[str, Any],
    observed_transform_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate transform metadata exactly matches the training manifest."""
    expected = manifest.get("transform_manifest", {})
    if not isinstance(expected, Mapping):
        raise ParityCheckError("manifest.transform_manifest must be an object")
    if not isinstance(observed_transform_manifest, Mapping):
        raise ParityCheckError("observed_transform_manifest must be an object")

    if _canonical_json(expected) != _canonical_json(dict(observed_transform_manifest)):
        raise ParityCheckError(
            "transform parity check failed: observed transform metadata does not match manifest"
        )

    return {
        "transform_keys": sorted(dict(expected).keys()),
        "transform_hash": _canonical_json(expected),
    }


def validate_integrity_manifest(
    manifest: Mapping[str, Any],
    observed_integrity: Mapping[str, Any],
    *,
    keys: Iterable[str] = ("git_sha", "config_hash", "artifact_hash"),
) -> dict[str, Any]:
    """Validate integrity metadata keys exactly match expected values when present."""
    expected = manifest.get("integrity", {})
    if not isinstance(expected, Mapping):
        raise ParityCheckError("manifest.integrity must be an object")
    if not isinstance(observed_integrity, Mapping):
        raise ParityCheckError("observed_integrity must be an object")

    mismatches: dict[str, dict[str, str | None]] = {}
    for key in keys:
        if key not in expected:
            continue
        exp = expected.get(key)
        obs = observed_integrity.get(key)
        if str(exp) != str(obs):
            mismatches[str(key)] = {"expected": None if exp is None else str(exp), "actual": None if obs is None else str(obs)}

    if mismatches:
        raise ParityCheckError(f"integrity parity check failed: {mismatches}")

    return {"validated_keys": [k for k in keys if k in expected]}


def validate_projection_output_columns(
    projections_df: pd.DataFrame,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate projection output schema contract before publish."""
    output_manifest = manifest.get("output_manifest")
    if not isinstance(output_manifest, Mapping):
        raise ParityCheckError("manifest.output_manifest must be an object")

    expected_cols = output_manifest.get("projection_columns")
    if not isinstance(expected_cols, list) or len(expected_cols) <= 0:
        raise ParityCheckError("output manifest requires non-empty projection_columns")

    expected = [str(c) for c in expected_cols]
    actual = list(projections_df.columns)
    missing = sorted(set(expected) - set(actual))
    if missing:
        raise ParityCheckError(f"projection schema missing required columns: {missing}")

    # Fail on order mismatch for contract columns.
    actual_prefix = [c for c in actual if c in set(expected)]
    if actual_prefix[: len(expected)] != expected:
        raise ParityCheckError(
            "projection schema order mismatch for contract columns: "
            f"expected_prefix={expected} actual_prefix={actual_prefix[:len(expected)]}"
        )

    return {
        "projection_row_count": int(len(projections_df)),
        "projection_columns": list(actual),
    }


def _load_bundle_norm_from_config(config_path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ParityCheckError(f"bundle config not found for distribution check: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ParityCheckError(f"bundle config is not valid JSON: {config_path}") from exc

    feature_cols = payload.get("feature_columns")
    feature_mean = payload.get("feature_mean")
    feature_std = payload.get("feature_std")
    if not isinstance(feature_cols, list) or not isinstance(feature_mean, list) or not isinstance(feature_std, list):
        raise ParityCheckError("bundle config missing feature_columns/feature_mean/feature_std for distribution check")
    if len(feature_cols) != len(feature_mean) or len(feature_cols) != len(feature_std):
        raise ParityCheckError(
            "bundle config feature normalization lengths mismatch "
            f"(cols={len(feature_cols)} mean={len(feature_mean)} std={len(feature_std)})"
        )

    mean = np.asarray(feature_mean, dtype=np.float64)
    std = np.asarray(feature_std, dtype=np.float64)
    std = np.where(std <= 1e-6, 1.0, std)
    return [str(c) for c in feature_cols], mean, std


def validate_feature_distribution_contract(
    features_df: pd.DataFrame,
    manifest: Mapping[str, Any],
    *,
    bundle_config_path: Path | None = None,
) -> dict[str, Any]:
    """Validate high-impact feature distribution policy from manifest.

    Policy location:
      manifest.missing_value_policy.distribution_contract
    """
    mvp = manifest.get("missing_value_policy", {})
    if not isinstance(mvp, Mapping):
        return {"enabled": False}
    dist = mvp.get("distribution_contract")
    if not isinstance(dist, Mapping) or not bool(dist.get("enabled", False)):
        return {"enabled": False}

    cfg_hint = dist.get("bundle_config_path")
    cfg_path = (
        Path(str(cfg_hint)).expanduser().resolve()
        if isinstance(cfg_hint, str) and cfg_hint.strip()
        else (Path(bundle_config_path).expanduser().resolve() if bundle_config_path is not None else None)
    )
    if cfg_path is None:
        raise ParityCheckError("distribution contract enabled but no bundle_config_path provided")

    feature_cols, feature_mean, feature_std = _load_bundle_norm_from_config(cfg_path)
    feature_index = {c: i for i, c in enumerate(feature_cols)}

    feature_limits = dist.get("feature_limits", {})
    if not isinstance(feature_limits, Mapping):
        raise ParityCheckError("distribution_contract.feature_limits must be an object")

    errors: list[str] = []
    feature_report: dict[str, dict[str, float]] = {}
    for raw_name, raw_limits in feature_limits.items():
        name = str(raw_name)
        if name not in features_df.columns:
            errors.append(f"distribution feature missing from frame: {name}")
            continue
        if name not in feature_index:
            errors.append(f"distribution feature missing from bundle feature normalization: {name}")
            continue
        if not isinstance(raw_limits, Mapping):
            errors.append(f"distribution feature limits must be an object: {name}")
            continue

        idx = int(feature_index[name])
        series = pd.to_numeric(features_df[name], errors="coerce").to_numpy(dtype=np.float64, copy=False)
        series = np.nan_to_num(series, nan=float(feature_mean[idx]), posinf=float(feature_mean[idx]), neginf=float(feature_mean[idx]))
        z = np.abs((series - float(feature_mean[idx])) / float(feature_std[idx]))

        stats = {
            "mean_abs_z": float(np.mean(z)) if z.size > 0 else 0.0,
            "p95_abs_z": float(np.percentile(z, 95)) if z.size > 0 else 0.0,
            "max_abs_z": float(np.max(z)) if z.size > 0 else 0.0,
        }
        feature_report[name] = stats

        max_abs_mean_z = raw_limits.get("max_abs_mean_z")
        max_p95_abs_z = raw_limits.get("max_p95_abs_z")
        if max_abs_mean_z is not None and float(stats["mean_abs_z"]) > float(max_abs_mean_z):
            errors.append(
                f"distribution gate failed for {name}: "
                f"mean_abs_z={stats['mean_abs_z']:.3f} > max_abs_mean_z={float(max_abs_mean_z):.3f}"
            )
        if max_p95_abs_z is not None and float(stats["p95_abs_z"]) > float(max_p95_abs_z):
            errors.append(
                f"distribution gate failed for {name}: "
                f"p95_abs_z={stats['p95_abs_z']:.3f} > max_p95_abs_z={float(max_p95_abs_z):.3f}"
            )

    conditional_limits = dist.get("conditional_limits", [])
    conditional_report: list[dict[str, Any]] = []
    if conditional_limits is not None:
        if not isinstance(conditional_limits, list):
            errors.append("distribution_contract.conditional_limits must be a list")
        else:
            for entry in conditional_limits:
                if not isinstance(entry, Mapping):
                    errors.append("distribution_contract.conditional_limits entries must be objects")
                    continue
                name = str(entry.get("name", "unnamed"))
                cond_col = str(entry.get("condition_col", ""))
                cond_eq = entry.get("condition_eq", 1)
                metric_col = str(entry.get("metric_col", ""))
                max_rate = entry.get("max_rate")
                if not cond_col or not metric_col:
                    errors.append(f"distribution conditional gate missing columns: {name}")
                    continue
                if cond_col not in features_df.columns or metric_col not in features_df.columns:
                    errors.append(f"distribution conditional gate missing feature columns: {name}")
                    continue
                cond = pd.to_numeric(features_df[cond_col], errors="coerce").fillna(0)
                metric = pd.to_numeric(features_df[metric_col], errors="coerce").fillna(1)
                mask = cond.eq(cond_eq)
                if bool(mask.any()):
                    rate = float(metric.loc[mask].mean())
                    n = int(mask.sum())
                else:
                    rate = 0.0
                    n = 0
                conditional_report.append(
                    {
                        "name": name,
                        "condition_col": cond_col,
                        "condition_eq": cond_eq,
                        "metric_col": metric_col,
                        "n": n,
                        "rate": rate,
                        "max_rate": None if max_rate is None else float(max_rate),
                    }
                )
                if max_rate is not None and rate > float(max_rate):
                    errors.append(
                        f"distribution conditional gate failed for {name}: "
                        f"rate={rate:.3f} > max_rate={float(max_rate):.3f}"
                    )

    if errors:
        raise ParityCheckError("feature distribution contract check failed: " + "; ".join(errors))

    return {
        "enabled": True,
        "bundle_config_path": str(cfg_path),
        "feature_report": feature_report,
        "conditional_report": conditional_report,
    }
