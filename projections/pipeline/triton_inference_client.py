"""HTTP client helpers for internal Triton-backed GTV2 inference actions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import requests


class TritonInferenceError(RuntimeError):
    """Raised when Triton request/response handling fails."""


@dataclass(frozen=True)
class TritonEndpointConfig:
    endpoint: str
    model_name: str
    model_version: str | None = None
    timeout_seconds: float = 60.0


def _normalize_http_base(endpoint: str) -> str:
    value = str(endpoint).strip().rstrip("/")
    if value.startswith("http://") or value.startswith("https://"):
        return value
    if ":" in value:
        host, port = value.rsplit(":", 1)
        if port == "8001":
            # Common typo/mismatch: 8001 is Triton gRPC, HTTP default is 8000.
            return f"http://{host}:8000"
        return f"http://{value}"
    return f"http://{value}:8000"


def check_triton_health(endpoint: str, *, timeout_seconds: float = 2.0) -> tuple[bool, str]:
    base = _normalize_http_base(endpoint)
    url = f"{base}/v2/health/ready"
    try:
        response = requests.get(url, timeout=float(timeout_seconds))
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"
    if response.status_code != 200:
        return False, f"HTTP {response.status_code}: {response.text[:256]}"
    return True, "ok"


def _infer_url(*, cfg: TritonEndpointConfig) -> str:
    base = _normalize_http_base(cfg.endpoint)
    if cfg.model_version:
        return (
            f"{base}/v2/models/{cfg.model_name}/versions/{cfg.model_version}/infer"
        )
    return f"{base}/v2/models/{cfg.model_name}/infer"


def infer_json_action(
    *,
    cfg: TritonEndpointConfig,
    request_payload: dict[str, Any],
) -> dict[str, Any]:
    body = {
        "inputs": [
            {
                "name": "request_json",
                "shape": [1],
                "datatype": "BYTES",
                "data": [json.dumps(request_payload, separators=(",", ":"))],
            }
        ],
        "outputs": [{"name": "response_json"}],
    }
    url = _infer_url(cfg=cfg)
    try:
        response = requests.post(url, json=body, timeout=float(cfg.timeout_seconds))
    except Exception as exc:  # noqa: BLE001
        raise TritonInferenceError(f"Triton request failed: {type(exc).__name__}: {exc}") from exc
    if response.status_code != 200:
        raise TritonInferenceError(
            f"Triton infer failed: HTTP {response.status_code}: {response.text[:512]}"
        )

    try:
        payload = response.json()
    except Exception as exc:  # noqa: BLE001
        raise TritonInferenceError(
            f"Triton infer returned non-JSON response: {response.text[:256]}"
        ) from exc

    outputs = payload.get("outputs")
    if not isinstance(outputs, list):
        raise TritonInferenceError(f"Malformed Triton response: {payload}")

    response_json: str | None = None
    for output in outputs:
        if str(output.get("name")) != "response_json":
            continue
        data = output.get("data")
        if isinstance(data, list) and data:
            response_json = str(data[0])
            break
    if response_json is None:
        raise TritonInferenceError(f"response_json output missing: {payload}")

    try:
        result = json.loads(response_json)
    except Exception as exc:  # noqa: BLE001
        raise TritonInferenceError(
            f"response_json payload is not valid JSON: {response_json[:256]}"
        ) from exc
    if not isinstance(result, dict):
        raise TritonInferenceError(f"response_json payload must decode to object: {result!r}")
    return result
