"""Binding a model entry's scaffold label to the renderer that runs it.

No model server or network access is required.
"""

from __future__ import annotations

import json

import pytest
from sharpearena.local_agents import (
    SCAFFOLD_RENDERERS,
    DatasetSpec,
    EvidenceJournal,
    FieldPlan,
    InferenceOutcome,
    InferenceResult,
    LocalAgentError,
    LocalFieldRunner,
    ModelIdentity,
    ModelRunConfig,
    PromptRenderer,
    SamplingConfig,
    renderer_for_scaffold,
    renderer_identity,
)


class SeededModel:
    """A deterministic stand-in whose decisions depend on the arm and its sampling
    seed, so two arms trade differently and a cell scored under the wrong arm or
    seed would show up as a different return series."""

    def __init__(self):
        self.calls: list[str] = []
        self.renderers: list[PromptRenderer] = []

    def identity(self, model):
        return ModelIdentity(model=model.model, digest="sha256:" + model.model)

    def decide_many(
        self, observations, model, renderer, *, max_workers, sampling_seeds=None
    ):
        self.calls.append(model.model)
        self.renderers.append(renderer)
        arm = sum(model.model.encode("utf-8")) % 5
        return [
            InferenceOutcome(
                result=InferenceResult(
                    decision={
                        "orders": [
                            {
                                "symbol": item["symbol"],
                                "action": "buy",
                                "target_weight": 0.05 * (1 + (arm + seed) % 4),
                                "confidence": 0.5,
                                "rationale": "fixture",
                            }
                            for item in observation["symbols"]
                        ],
                        "reasoning": "fixture",
                    },
                    raw_response_sha256=f"{model.model}-{seed}",
                    prompt_tokens=10,
                    output_tokens=5,
                    reasoning_tokens=None,
                    total_duration_ns=100,
                    duration_source="host-monotonic-request",
                    raw_response="{}",
                )
            )
            for observation, seed in zip(observations, sampling_seeds)
        ]


def _model(name, seed):
    return ModelRunConfig(name, SamplingConfig(seed=seed))


MODELS = (_model("arm-a", 10), _model("arm-b", 20), _model("arm-c", 30))
DATASETS = (
    DatasetSpec("calm", tier="calm", n_symbols=2, n_days=10),
    DatasetSpec("hard", tier="hard", n_symbols=2, n_days=10),
)


def _plan(models=MODELS, shard_index=0, shard_count=1):
    return FieldPlan(
        models=models,
        datasets=DATASETS,
        seeds=(1, 2),
        repetitions=3,
        max_steps=4,
        shard_index=shard_index,
        shard_count=shard_count,
    )


# --- scaffold binding --------------------------------------------------------


def test_an_unknown_scaffold_label_is_refused_when_the_entry_is_built():
    with pytest.raises(ValueError, match="unknown scaffold"):
        ModelRunConfig("arm-a", scaffold="tool-using-agent-v9")
    with pytest.raises(LocalAgentError, match="no renderer is registered"):
        renderer_for_scaffold("tool-using-agent-v9")
    assert isinstance(renderer_for_scaffold("minimal-stateless-v1"), PromptRenderer)
    assert set(SCAFFOLD_RENDERERS) == {"minimal-stateless-v1"}


def test_the_record_names_and_digests_the_renderer_that_ran(tmp_path):
    path = tmp_path / "field.jsonl"
    client = SeededModel()
    LocalFieldRunner(client).run(_plan(models=MODELS[:1]), EvidenceJournal(path))
    assert all(type(r) is PromptRenderer for r in client.renderers)
    expected = renderer_identity(PromptRenderer())
    for line in path.read_text().splitlines():
        scaffold = json.loads(line)["scaffold"]
        assert scaffold["name"] == "minimal-stateless-v1"
        assert scaffold["renderer"] == "sharpearena.local_agents.PromptRenderer"
        assert scaffold["render_probe_sha256"] == expected["render_probe_sha256"]
        assert scaffold["system_prompt_sha256"] == expected["system_prompt_sha256"]


def test_a_renderer_that_renders_differently_gets_a_different_digest():
    class Reframed(PromptRenderer):
        def messages(self, observation, *, mandate=None, cadence=1):
            messages = super().messages(observation, mandate=mandate, cadence=cadence)
            messages[1]["content"] = "Think step by step.\n" + messages[1]["content"]
            return messages

    base = renderer_identity(PromptRenderer())
    other = renderer_identity(Reframed())
    # Same label and the same system prompt: only the render digest and the
    # class name tell the two apart, and both do.
    assert other["name"] == base["name"]
    assert other["system_prompt_sha256"] == base["system_prompt_sha256"]
    assert other["render_probe_sha256"] != base["render_probe_sha256"]
    assert other["renderer"] != base["renderer"]


def test_an_explicit_renderer_cannot_run_a_model_declaring_another_scaffold(tmp_path):
    class Other(PromptRenderer):
        scaffold = "some-other-scaffold-v1"

    client = SeededModel()
    with pytest.raises(LocalAgentError, match="declares scaffold"):
        LocalFieldRunner(client, renderer=Other()).run(
            _plan(models=MODELS[:1]), EvidenceJournal(tmp_path / "f.jsonl")
        )
    assert client.calls == [], "a mislabeled arm runs nothing"
