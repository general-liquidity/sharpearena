"""A stated confidence is paired with the reward of the step that realizes it.

The reward the shared engine books at step t is the price move on the holdings the
decision at step t - 1 chose; decision t adds only its own trading cost. The local
field runner therefore pairs each stated confidence with the next step's reward, and
a lane's final decision, whose outcome falls outside the run, adds no pair. No model
server or network access is required.
"""

from __future__ import annotations

import json

from sharpearena.local_agents import EvidenceJournal, LocalFieldRunner
from test_local_agents import FixedModel, _plan


class RisingConfidenceModel(FixedModel):
    """States 0.25 on its first inference round, 0.5 on the second, 0.75 on the third."""

    def __init__(self):
        self._rounds = 0

    def decide_many(
        self, observations, model, renderer, *, max_workers, sampling_seeds=None
    ):
        self._rounds += 1
        outcomes = super().decide_many(
            observations,
            model,
            renderer,
            max_workers=max_workers,
            sampling_seeds=sampling_seeds,
        )
        for outcome in outcomes:
            for order in outcome.result.decision["orders"]:
                order["confidence"] = 0.25 * self._rounds
        return outcomes


def test_each_confidence_is_paired_with_the_next_steps_reward(tmp_path):
    path = tmp_path / "pairing.jsonl"
    counts = LocalFieldRunner(RisingConfidenceModel()).run(
        _plan(repetitions=1), EvidenceJournal(path)
    )
    assert counts == {"completed": 2, "failed": 0, "skipped": 0}
    for line in path.read_text().splitlines():
        record = json.loads(line)
        returns = record["returns"]
        assert len(returns) == 5
        # Inference every second bar (steps 0, 2, 4), each decision held for the next
        # bar: the applied confidences are 0.25, 0.25, 0.5, 0.5, 0.75. The last one
        # has no outcome inside the run.
        assert record["confidences"] == [0.25, 0.25, 0.5, 0.5]
        assert record["outcomes"] == [reward > 0.0 for reward in returns[1:]]
