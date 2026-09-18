"""A stated confidence is paired with the reward of the step that realizes it.

The reward the shared engine books at step t is the price move on the holdings the
decision at step t - 1 chose; decision t adds only its own trading cost. The local
field runner therefore pairs each stated confidence with the next step's reward, and
a lane's final decision, whose outcome falls outside the run, adds no pair. No model
server or network access is required.
"""

from __future__ import annotations

import json

import pytest
from sharpearena.bench_bridge import (
    BenchBridgeError,
    _digest,
    compile_benchmark_evidence,
)
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


def _rewrite(path, mutate):
    """Rewrite each completed record through ``mutate`` and return the new journal path."""

    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    out = path.with_name(path.stem + "-edited.jsonl")
    with out.open("w", encoding="utf-8") as handle:
        for record in records:
            if record.get("status") == "completed":
                mutate(record)
                record["returns_sha256"] = _digest(record["returns"])
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return out


def _honest_journal(tmp_path, name="pairing.jsonl"):
    path = tmp_path / name
    LocalFieldRunner(RisingConfidenceModel()).run(_plan(repetitions=1), EvidenceJournal(path))
    return path


def test_the_bridge_rebuilds_the_pairs_from_the_recorded_decisions(tmp_path):
    """An honest record still compiles, and the emitted submission carries exactly the
    pairs the decisions and returns imply."""

    path = _honest_journal(tmp_path)
    compile_benchmark_evidence([path], tmp_path / "compiled")
    for line in path.read_text().splitlines():
        record = json.loads(line)
        assert record["confidences"] == [0.25, 0.25, 0.5, 0.5]
        assert record["outcomes"] == [reward > 0.0 for reward in record["returns"][1:]]


def test_the_bridge_refuses_fabricated_calibration_pairs(tmp_path):
    """The defect: `returns` was digest-bound while `confidences` and `outcomes` were
    checked only for length, so a record could claim confident correct calls on a run with
    no positive bar and compile into a fabricated Brier score. Recomputing the digest of
    the edited returns does not rescue it, because the pairs are rebuilt from evidence."""

    def fabricate(record):
        n = len(record["returns"])
        record["returns"] = [0.0] * n
        record["confidences"] = [1.0] * n
        record["outcomes"] = [True] * n

    edited = _rewrite(_honest_journal(tmp_path), fabricate)
    with pytest.raises(BenchBridgeError, match="disagree with the recorded decisions"):
        compile_benchmark_evidence([edited], tmp_path / "compiled-fabricated")


def test_the_bridge_refuses_cherry_picked_outcomes(tmp_path):
    """Stronger than a length or subsequence check: dropping only the losing pairs keeps
    an in-order subsequence of the true outcomes and a legal length, and is still refused."""

    def cherry_pick(record):
        kept = [
            (confidence, outcome)
            for confidence, outcome in zip(record["confidences"], record["outcomes"])
            if outcome
        ]
        record["confidences"] = [confidence for confidence, _ in kept]
        record["outcomes"] = [outcome for _, outcome in kept]

    path = _honest_journal(tmp_path, "cherry.jsonl")
    losing = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if json.loads(line).get("status") == "completed"
        and not all(json.loads(line)["outcomes"])
    ]
    assert losing, "the fixture must contain a cell with at least one losing pair"

    edited = _rewrite(path, cherry_pick)
    with pytest.raises(BenchBridgeError, match="disagree with the recorded decisions"):
        compile_benchmark_evidence([edited], tmp_path / "compiled-cherry")


def test_the_bridge_refuses_an_outcome_that_is_not_a_boolean(tmp_path):
    """`1` is an `int` that equals `True` in Python, so an unguarded comparison would
    accept a record whose outcomes were written as integers."""

    def as_integers(record):
        record["outcomes"] = [1 if outcome else 0 for outcome in record["outcomes"]]

    edited = _rewrite(_honest_journal(tmp_path, "ints.jsonl"), as_integers)
    with pytest.raises(BenchBridgeError, match="outcomes must be booleans"):
        compile_benchmark_evidence([edited], tmp_path / "compiled-ints")


def test_the_bridge_refuses_a_record_missing_its_decisions(tmp_path):
    def drop_decisions(record):
        record.pop("decisions")

    edited = _rewrite(_honest_journal(tmp_path, "nodec.jsonl"), drop_decisions)
    with pytest.raises(BenchBridgeError, match="one applied decision per return"):
        compile_benchmark_evidence([edited], tmp_path / "compiled-nodec")
