"""The interleaved, counterbalanced cell order of the local field.

No model server or network access is required.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
from sharpearena.local_agents import (
    CELL_ORDER_RULE,
    DatasetSpec,
    EvidenceJournal,
    FieldPlan,
    InferenceOutcome,
    InferenceResult,
    LocalFieldRunner,
    ModelIdentity,
    ModelRunConfig,
    SamplingConfig,
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


def _plan(models=MODELS, order_seed=0, shard_index=0, shard_count=1):
    return FieldPlan(
        models=models,
        datasets=DATASETS,
        seeds=(1, 2),
        repetitions=3,
        max_steps=4,
        shard_index=shard_index,
        shard_count=shard_count,
        order_seed=order_seed,
    )


def _scored(path):
    """Each completed cell's scored outcome, keyed by what identifies the cell
    independently of plan digest and execution order."""

    out = {}
    for line in path.read_text().splitlines():
        record = json.loads(line)
        assert record["status"] == "completed"
        key = (
            record["model"]["model"],
            record["dataset_index"],
            record["seed_index"],
            record["inference_seed"],
        )
        assert key not in out
        out[key] = (record["returns"], record["score"])
    return out


# --- interleaved cell order --------------------------------------------------


def test_the_execution_order_interleaves_model_arms():
    plan = _plan()
    order = LocalFieldRunner.execution_order(plan)
    # Collapse to the sequence of model blocks the runner executes.
    blocks = []
    for cell in order:
        key = (cell.model_index, cell.dataset_index, cell.repetition)
        if not blocks or blocks[-1] != key:
            blocks.append(key)
    models_in_order = [model for model, _, _ in blocks]
    rounds = len(DATASETS) * plan.repetitions
    assert len(blocks) == rounds * len(MODELS)
    # Every round of len(MODELS) consecutive blocks holds every model once, so
    # no arm runs as one contiguous block.
    for start in range(0, len(models_in_order), len(MODELS)):
        assert sorted(models_in_order[start : start + len(MODELS)]) == [0, 1, 2]
    # And the permutation is drawn, not the plan order every round.
    assert len({tuple(models_in_order[i : i + 3]) for i in range(0, 18, 3)}) > 1
    # It is a function of the seed: reproducible, and another seed reorders.
    assert LocalFieldRunner.execution_order(plan) == order
    assert LocalFieldRunner.execution_order(_plan(order_seed=9)) != order


def test_the_order_rule_and_seed_are_bound_into_the_plan_digest():
    assert _plan(order_seed=1).plan_sha256 != _plan(order_seed=2).plan_sha256
    assert CELL_ORDER_RULE.startswith("sharpearena.cell-order.")
    with pytest.raises(ValueError, match="order_seed"):
        _plan(order_seed=-1)


@pytest.mark.parametrize("order_seed", [0, 5])
@pytest.mark.parametrize("shard_count", [1, 2, 3, 5])
def test_shards_cover_the_full_grid_exactly_once(order_seed, shard_count):
    full = LocalFieldRunner.execution_order(_plan(order_seed=order_seed))
    grid = {
        (m, d, s, r)
        for m in range(len(MODELS))
        for d in range(len(DATASETS))
        for s in (1, 2)
        for r in range(3)
    }
    assert {(c.model_index, c.dataset_index, c.seed, c.repetition) for c in full} == grid
    assert len(full) == len(grid)
    seen = []
    for shard_index in range(shard_count):
        shard = _plan(
            order_seed=order_seed, shard_index=shard_index, shard_count=shard_count
        )
        cells = LocalFieldRunner.cells(shard)
        seen.extend((c.model_index, c.dataset_index, c.seed, c.repetition) for c in cells)
        # A shard runs its cells in the same interleaved order as the full field.
        positions = [full.index(cell) for cell in cells]
        assert positions == sorted(positions)
    assert sorted(seen) == sorted(grid), "every cell in exactly one shard"


def test_scored_outcomes_do_not_depend_on_the_execution_order(tmp_path):
    interleaved = tmp_path / "interleaved.jsonl"
    reordered = tmp_path / "reordered.jsonl"
    LocalFieldRunner(SeededModel()).run(_plan(order_seed=0), EvidenceJournal(interleaved))
    LocalFieldRunner(SeededModel()).run(_plan(order_seed=77), EvidenceJournal(reordered))
    # Each arm alone, run as one sequential block: the order the field used to
    # have.
    alone = {}
    for index, model in enumerate(MODELS):
        path = tmp_path / f"alone-{index}.jsonl"
        LocalFieldRunner(SeededModel()).run(_plan(models=(model,)), EvidenceJournal(path))
        alone.update(_scored(path))
    # Shards of the interleaved field, run separately.
    sharded = {}
    for shard_index in range(2):
        path = tmp_path / f"shard-{shard_index}.jsonl"
        LocalFieldRunner(SeededModel()).run(
            _plan(shard_index=shard_index, shard_count=2), EvidenceJournal(path)
        )
        sharded.update(_scored(path))

    reference = _scored(interleaved)
    assert len(reference) == len(MODELS) * len(DATASETS) * 2 * 3
    assert _scored(reordered) == reference
    assert alone == reference
    assert sharded == reference
    # The arms really do trade differently, so the equality above is not vacuous.
    assert len({json.dumps(returns) for returns, _ in reference.values()}) > 1


def test_model_arms_are_called_in_interleaved_rounds(tmp_path):
    client = SeededModel()
    LocalFieldRunner(client).run(_plan(), EvidenceJournal(tmp_path / "f.jsonl"))
    # Collapse consecutive calls for one arm into one run of calls.
    runs = [name for i, name in enumerate(client.calls) if i == 0 or client.calls[i - 1] != name]
    assert len(runs) > len(MODELS), f"arms ran as contiguous blocks: {runs}"


def test_a_replaced_order_seed_is_a_different_plan_not_a_resume(tmp_path):
    path = tmp_path / "field.jsonl"
    journal = EvidenceJournal(path)
    plan = _plan(models=MODELS[:1])
    first = LocalFieldRunner(SeededModel()).run(plan, journal)
    assert first["completed"] == len(DATASETS) * 2 * 3
    again = LocalFieldRunner(SeededModel()).run(plan, journal)
    assert again == {"completed": 0, "failed": 0, "skipped": first["completed"]}
    other = LocalFieldRunner(SeededModel()).run(replace(plan, order_seed=3), journal)
    assert other["skipped"] == 0
