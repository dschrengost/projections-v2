"""LightGBM compute-device helpers for training scripts."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable, Mapping

import lightgbm as lgb
import numpy as np

_VALID_DEVICE_TYPES = {"cpu", "cuda", "gpu"}


def normalize_lgbm_device_type(value: str) -> str:
    """Normalize user-provided device type (cpu/cuda/gpu/auto)."""
    normalized = str(value).strip().lower()
    if normalized in {"", "auto"}:
        return "auto"
    if normalized not in _VALID_DEVICE_TYPES:
        raise ValueError(f"Unsupported LightGBM device type {value!r}. Expected one of: auto, cpu, cuda, gpu")
    return normalized


@lru_cache(maxsize=8)
def _probe_lgbm_device_type(device_type: str) -> tuple[bool, str]:
    """Return whether a LightGBM device_type works in this runtime."""
    if device_type not in _VALID_DEVICE_TYPES:
        return False, f"invalid device_type={device_type!r}"
    rng = np.random.default_rng(0)
    x = rng.normal(size=(64, 8)).astype(np.float32)
    y = rng.normal(size=(64,)).astype(np.float32)
    dtrain = lgb.Dataset(x, label=y, free_raw_data=True)
    params: dict[str, Any] = {
        "objective": "regression",
        "metric": "l2",
        "verbosity": -1,
        "num_threads": 1,
        "device_type": device_type,
    }
    try:
        lgb.train(params=params, train_set=dtrain, num_boost_round=4)
    except Exception as exc:  # pragma: no cover - depends on local LightGBM build/runtime
        return False, f"{type(exc).__name__}: {exc}"
    return True, ""


def resolve_lgbm_device_type(
    requested: str,
    *,
    allow_fallback_to_cpu: bool = True,
    log_fn: Callable[[str], None] | None = None,
) -> str:
    """Resolve LightGBM device_type with optional GPU probing + CPU fallback."""
    log = log_fn or (lambda _: None)
    requested_type = normalize_lgbm_device_type(requested)
    if requested_type == "cpu":
        log("[lgbm] device_type=cpu (explicit)")
        return "cpu"

    # Default to CUDA-only probing for "auto" to keep behavior deterministic on
    # NVIDIA hosts and avoid silently selecting OpenCL GPU backends.
    probe_order = ["cuda"] if requested_type == "auto" else [requested_type]
    failures: dict[str, str] = {}
    for candidate in probe_order:
        ok, detail = _probe_lgbm_device_type(candidate)
        if ok:
            log(f"[lgbm] device_type={candidate} (probe passed)")
            return candidate
        failures[candidate] = detail
        log(f"[lgbm] device_type={candidate} unavailable: {detail}")

    if allow_fallback_to_cpu:
        log("[lgbm] falling back to device_type=cpu")
        return "cpu"

    failure_blob = "; ".join(f"{k}: {v}" for k, v in failures.items())
    raise RuntimeError(f"Unable to use requested LightGBM device_type={requested_type}. Probe errors: {failure_blob}")


def apply_lgbm_compute_params(
    params: Mapping[str, Any],
    *,
    device_type: str,
    num_threads: int | None = None,
) -> dict[str, Any]:
    """Inject compute-related LightGBM params in a backend-safe way."""
    out = dict(params)
    normalized_device = normalize_lgbm_device_type(device_type)
    if normalized_device == "auto":
        raise ValueError("apply_lgbm_compute_params requires a concrete device_type (cpu/cuda/gpu), not auto")
    out["device_type"] = normalized_device
    if num_threads is not None:
        if int(num_threads) > 0:
            threads = int(num_threads)
            out["num_threads"] = threads
            out["n_jobs"] = threads
        else:
            out.pop("num_threads", None)
            out.pop("n_jobs", None)
    if normalized_device != "cpu":
        # CPU-specific determinism knobs can be unsupported on GPU backends.
        out.pop("deterministic", None)
        out.pop("force_row_wise", None)
    return out
