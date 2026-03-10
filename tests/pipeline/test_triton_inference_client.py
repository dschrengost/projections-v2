from __future__ import annotations

import json

import pytest

from projections.pipeline.triton_inference_client import (
    TritonEndpointConfig,
    TritonInferenceError,
    _normalize_http_base,
    check_triton_health,
    infer_json_action,
)


class _Resp:
    def __init__(self, *, status_code: int, payload: dict | None = None, text: str = "") -> None:
        self.status_code = int(status_code)
        self._payload = payload
        self.text = text

    def json(self) -> dict:
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def test_normalize_http_base_converts_common_grpc_port() -> None:
    assert _normalize_http_base("localhost:8001") == "http://localhost:8000"
    assert _normalize_http_base("localhost:8000") == "http://localhost:8000"
    assert _normalize_http_base("http://host:9000") == "http://host:9000"


def test_check_triton_health_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "projections.pipeline.triton_inference_client.requests.get",
        lambda url, timeout: _Resp(status_code=200, payload={}, text="ok"),
    )
    ok, detail = check_triton_health("localhost:8000", timeout_seconds=1.0)
    assert ok is True
    assert detail == "ok"


def test_infer_json_action_success(monkeypatch: pytest.MonkeyPatch) -> None:
    response_payload = {
        "outputs": [
            {
                "name": "response_json",
                "data": [json.dumps({"ok": True, "rows": 123})],
            }
        ]
    }

    monkeypatch.setattr(
        "projections.pipeline.triton_inference_client.requests.post",
        lambda url, json, timeout: _Resp(
            status_code=200,
            payload=response_payload,
            text="ok",
        ),
    )

    result = infer_json_action(
        cfg=TritonEndpointConfig(
            endpoint="localhost:8000",
            model_name="gtv2_scorer",
            model_version="1",
            timeout_seconds=5.0,
        ),
        request_payload={"action": "score"},
    )
    assert result == {"ok": True, "rows": 123}


def test_infer_json_action_http_error_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "projections.pipeline.triton_inference_client.requests.post",
        lambda url, json, timeout: _Resp(
            status_code=503,
            payload=None,
            text="service unavailable",
        ),
    )
    with pytest.raises(TritonInferenceError):
        infer_json_action(
            cfg=TritonEndpointConfig(
                endpoint="localhost:8000",
                model_name="gtv2_scorer",
                timeout_seconds=5.0,
            ),
            request_payload={"action": "score"},
        )
