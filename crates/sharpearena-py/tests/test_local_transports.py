"""Loopback transport and concurrent lane-order regression tests."""

from __future__ import annotations

import json
import socket
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from sharpearena.local_agents import (
    InferenceResult,
    ModelHttpError,
    ModelIdentity,
    ModelResponseError,
    ModelRunConfig,
    ModelTransportError,
    OllamaClient,
    OpenAICompatibleClient,
    PromptRenderer,
)


class _Handler(BaseHTTPRequestHandler):
    hits: list[str] = []

    def do_GET(self):
        type(self).hits.append(self.path)
        if self.path == "/ok" or self.path == "/v1/ok":
            payload = json.dumps({"ok": True}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if self.path == "/bad-json" or self.path == "/v1/bad-json":
            payload = b"not-json"
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if self.path == "/redirect" or self.path == "/v1/redirect":
            self.send_response(302)
            self.send_header("Location", "/ok")
            self.end_headers()
            return
        self.send_response(503)
        self.end_headers()
        self.wfile.write(b"fixture failure")

    def log_message(self, format, *args):  # noqa: A002
        return


@contextmanager
def _server():
    _Handler.hits = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def _openai_identity():
    return ModelIdentity(
        model="fixture",
        digest="sha256:fixture",
        parameter_size="1B",
        quantization="Q4",
        offload="full GPU",
        server="fixture",
        server_version="1",
    )


@pytest.mark.parametrize("client_kind", ["ollama", "openai"])
def test_real_loopback_transports_map_http_json_and_redirect_faults(client_kind):
    with _server() as origin:
        if client_kind == "ollama":
            client = OllamaClient(origin)
            prefix = ""
        else:
            client = OpenAICompatibleClient((_openai_identity(),), base_url=origin)
            prefix = "/v1"
        assert client._request("GET", "/ok") == {"ok": True}
        with pytest.raises(ModelResponseError, match="invalid JSON"):
            client._request("GET", "/bad-json")
        with pytest.raises(ModelHttpError, match="HTTP 503"):
            client._request("GET", "/fail")
        with pytest.raises(ModelHttpError, match="HTTP 302"):
            client._request("GET", "/redirect")
        assert f"{prefix}/ok" in _Handler.hits
        assert _Handler.hits.count(f"{prefix}/ok") == 1


def test_decide_many_preserves_lane_order_and_captures_one_lane_fault():
    class RacingClient(OllamaClient):
        def decide(self, observation, model, renderer, *, sampling_seed=None):
            lane = observation["lane"]
            time.sleep((2 - lane) * 0.01)
            if lane == 1:
                raise RuntimeError("lane one failed")
            return InferenceResult(
                decision={"orders": [], "reasoning": str(lane)},
                raw_response_sha256=f"hash-{lane}",
                prompt_tokens=lane,
                output_tokens=1,
                reasoning_tokens=0,
                total_duration_ns=1,
                raw_response=str(lane),
            )

    outcomes = RacingClient().decide_many(
        [{"lane": 0}, {"lane": 1}, {"lane": 2}],
        ModelRunConfig("fixture"),
        PromptRenderer(),
        max_workers=3,
        sampling_seeds=[10, 11, 12],
    )
    assert outcomes[0].result is not None
    assert outcomes[0].result.decision["reasoning"] == "0"
    assert outcomes[1].error_type == "RuntimeError"
    assert outcomes[2].result is not None
    assert outcomes[2].result.decision["reasoning"] == "2"


def test_real_transport_refuses_an_unreachable_loopback_server():
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    client = OllamaClient(f"http://127.0.0.1:{port}", timeout_seconds=0.2)
    with pytest.raises(ModelTransportError):
        client._request("GET", "/unreachable")
