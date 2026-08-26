"""The OpenAI-compatible console adapter must preserve every declared axis."""

from __future__ import annotations

from sharpearena import openai_compatible_shim
from sharpearena.local_agents import ModelIdentity


def test_openai_shim_builds_the_declared_client_and_model_config(tmp_path, monkeypatch):
    identity_path = tmp_path / "identity.json"
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    identity = ModelIdentity(model="fixture", digest="sha256:fixture")
    captured = {}

    class Client:
        def __init__(self, identities, **kwargs):
            captured["identities"] = identities
            captured["client"] = kwargs

    def fake_run(client, config, stdin, stdout, stderr, *, identity_path):
        captured["config"] = config
        captured["identity_path"] = identity_path
        return 17

    monkeypatch.setattr(openai_compatible_shim, "load_identity_manifest", lambda _: (identity,))
    monkeypatch.setattr(openai_compatible_shim, "OpenAICompatibleClient", Client)
    monkeypatch.setattr(openai_compatible_shim, "run_stdio", fake_run)
    result = openai_compatible_shim.main(
        [
            "--model",
            "fixture",
            "--identity-manifest",
            str(manifest),
            "--base-url",
            "http://127.0.0.1:9000/v1",
            "--temperature",
            "0.3",
            "--seed",
            "19",
            "--max-tokens",
            "123",
            "--context-tokens",
            "4096",
            "--thinking",
            "--supports-thinking",
            "--thinking-budget-tokens",
            "512",
            "--supports-thinking-budget",
            "--identity-out",
            str(identity_path),
        ]
    )
    assert result == 17
    assert captured["identities"] == (identity,)
    assert captured["client"]["supports_thinking"] is True
    assert captured["client"]["supports_thinking_budget"] is True
    assert captured["config"].sampling.context_tokens == 4096
    assert captured["config"].sampling.thinking_budget_tokens == 512
    assert captured["identity_path"] == identity_path
