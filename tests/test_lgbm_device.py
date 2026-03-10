from __future__ import annotations

import pytest

from projections.lgbm_device import (
    apply_lgbm_compute_params,
    normalize_lgbm_device_type,
    resolve_lgbm_device_type,
)


def test_normalize_lgbm_device_type() -> None:
    assert normalize_lgbm_device_type("auto") == "auto"
    assert normalize_lgbm_device_type("cpu") == "cpu"
    assert normalize_lgbm_device_type("cuda") == "cuda"
    assert normalize_lgbm_device_type("gpu") == "gpu"
    with pytest.raises(ValueError):
        normalize_lgbm_device_type("tpu")


def test_apply_lgbm_compute_params_cpu_threads() -> None:
    params = {"objective": "regression"}
    out = apply_lgbm_compute_params(params, device_type="cpu", num_threads=3)
    assert out["device_type"] == "cpu"
    assert out["num_threads"] == 3
    assert out["n_jobs"] == 3


def test_apply_lgbm_compute_params_gpu_strips_cpu_only_flags() -> None:
    params = {
        "objective": "regression",
        "deterministic": True,
        "force_row_wise": True,
    }
    out = apply_lgbm_compute_params(params, device_type="gpu", num_threads=2)
    assert out["device_type"] == "gpu"
    assert out["num_threads"] == 2
    assert out["n_jobs"] == 2
    assert "deterministic" not in out
    assert "force_row_wise" not in out


def test_resolve_auto_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "projections.lgbm_device._probe_lgbm_device_type",
        lambda device_type: (False, "probe-failed"),
    )
    assert resolve_lgbm_device_type("auto") == "cpu"


def test_resolve_explicit_device_without_fallback_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "projections.lgbm_device._probe_lgbm_device_type",
        lambda device_type: (False, "probe-failed"),
    )
    with pytest.raises(RuntimeError):
        resolve_lgbm_device_type("cuda", allow_fallback_to_cpu=False)
