"""Strict trace promotion: reject rather than skip, and freeze minimal cases."""

from __future__ import annotations

import json
import socket

import pytest
from sharpearena.trace import load_trace
from sharpearena.trace_promotion import (
    GoldCase,
    OperatorDecision,
    PromotionError,
    SilverStore,
    TraceIntegrityError,
    blocking_failures,
    build_silver_candidate,
    evaluate_gold_case,
    load_gold_case,
    load_trace_strict,
    minimize_scenario,
    process_events,
    promote_to_gold,
    run_promotion_checks,
)


def _meta(**overrides):
    meta = {
        "kind": "meta",
        "schema_version": "sharpearena.trace/1.0.0",
        "environment_id": "sharpearena-lob-v0",
        "model_digest": "sha256:model",
        "scaffold_digest": "sha256:scaffold",
        "contract_version": "decision/1",
        "dataset_sha256": "sha256:data",
        "config": {"symbols": ["AAA"]},
        "n_steps": 0,
        "scenario_seeds": [7],
    }
    meta.update(overrides)
    return meta


def _step(index, *, reward=0.01, observation=None, decision=None, info=None):
    return {
        "kind": "step",
        "step": index,
        "observation": observation
        if observation is not None
        else {"close": 100.0 + index},
        "decision": decision
        if decision is not None
        else {"orders": [{"symbol": "AAA", "action": "buy", "target_weight": 0.1}]},
        "reward": reward,
        "info": info if info is not None else {"scenario_seed": 7},
    }


def _write(path, steps, meta_overrides=None):
    meta = _meta(n_steps=len(steps), **(meta_overrides or {}))
    lines = [json.dumps(step) for step in steps] + [json.dumps(meta)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _clean_trace(tmp_path, n=6):
    return _write(
        tmp_path / "clean.jsonl",
        [_step(index, reward=0.01 * (index + 1)) for index in range(n)],
    )


def test_permissive_reader_skips_what_the_strict_reader_refuses(tmp_path):
    path = _clean_trace(tmp_path)
    text = path.read_text(encoding="utf-8").splitlines()
    text.insert(2, "{ this is not json")
    path.write_text("\n".join(text) + "\n", encoding="utf-8")

    records, meta = load_trace(path.as_posix())
    assert len(records) == 6 and meta["environment_id"] == "sharpearena-lob-v0"

    with pytest.raises(TraceIntegrityError, match="not JSON"):
        load_trace_strict(path)


def test_a_missing_reward_is_not_read_as_zero(tmp_path):
    steps = [_step(index) for index in range(4)]
    del steps[2]["reward"]
    path = _write(tmp_path / "noreward.jsonl", steps)

    assert load_trace(path.as_posix())[0][2].get("reward") is None
    with pytest.raises(TraceIntegrityError, match="a missing reward is not zero"):
        load_trace_strict(path)


def test_out_of_sequence_steps_are_refused(tmp_path):
    steps = [_step(index) for index in range(4)]
    steps[2]["step"] = 9
    path = _write(tmp_path / "gap.jsonl", steps)
    with pytest.raises(TraceIntegrityError, match="breaks the sequence"):
        load_trace_strict(path)


def test_a_trace_without_provenance_meta_cannot_be_promoted(tmp_path):
    steps = [_step(index) for index in range(4)]
    meta = _meta(n_steps=4)
    del meta["model_digest"]
    path = tmp_path / "nomodel.jsonl"
    path.write_text(
        "\n".join([json.dumps(step) for step in steps] + [json.dumps(meta)]) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(TraceIntegrityError, match="missing required field 'model_digest'"):
        load_trace_strict(path)


def test_a_trace_with_no_meta_record_is_incomplete(tmp_path):
    path = tmp_path / "nometa.jsonl"
    path.write_text(
        "\n".join(json.dumps(_step(index)) for index in range(3)) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(TraceIntegrityError, match="no meta record"):
        load_trace_strict(path)


def test_meta_step_count_must_match_what_was_read(tmp_path):
    steps = [_step(index) for index in range(3)]
    meta = _meta()
    meta["n_steps"] = 99
    path = tmp_path / "count.jsonl"
    path.write_text(
        "\n".join([json.dumps(step) for step in steps] + [json.dumps(meta)]) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(TraceIntegrityError, match="but 3 step records were read"):
        load_trace_strict(path)


def test_fingerprint_separates_model_scaffold_data_and_process(tmp_path):
    base = load_trace_strict(_clean_trace(tmp_path)).fingerprint
    other_model = load_trace_strict(
        _write(
            tmp_path / "m.jsonl",
            [_step(index, reward=0.01 * (index + 1)) for index in range(6)],
            {"model_digest": "sha256:other-model"},
        )
    ).fingerprint
    other_data = load_trace_strict(
        _write(
            tmp_path / "d.jsonl",
            [_step(index, reward=0.01 * (index + 1)) for index in range(6)],
            {"dataset_sha256": "sha256:other-data"},
        )
    ).fingerprint

    assert base.model != other_model.model
    assert base.data == other_model.data
    assert base.data != other_data.data
    assert len({base.composite, other_model.composite, other_data.composite}) == 3


def test_identical_rewards_with_different_actions_fingerprint_differently(tmp_path):
    left = [_step(index, reward=0.02) for index in range(4)]
    right = [
        _step(
            index,
            reward=0.02,
            decision={"orders": [{"symbol": "AAA", "action": "sell"}]},
        )
        for index in range(4)
    ]
    left_print = load_trace_strict(_write(tmp_path / "l.jsonl", left)).fingerprint
    right_print = load_trace_strict(_write(tmp_path / "r.jsonl", right)).fingerprint
    assert left_print.environment == right_print.environment
    assert left_print.process != right_print.process


def test_process_events_ignore_reward_and_record_actions():
    events = process_events(
        [
            _step(0, reward=5.0),
            _step(1, reward=-5.0, decision={"orders": []}),
        ]
    )
    assert events[0]["actions"] == ["buy"]
    assert events[1]["n_orders"] == 0
    assert "reward" not in events[0]


def test_a_clean_trace_has_no_blocking_failures(tmp_path):
    trace = load_trace_strict(_clean_trace(tmp_path))
    results = run_promotion_checks(trace)
    assert blocking_failures(results) == ()
    assert {result.check_id for result in results} == {
        "rewards_finite",
        "no_lookahead_in_observation",
        "decision_is_structured",
        "seeds_declared_in_meta",
        "reward_series_varies",
    }


def test_a_lookahead_leak_is_a_blocking_failure(tmp_path):
    steps = [_step(index, reward=0.01 * (index + 1)) for index in range(8)]
    steps[5]["observation"] = {"close": 105.0, "next_close": 111.0}
    trace = load_trace_strict(_write(tmp_path / "leak.jsonl", steps))
    failures = blocking_failures(run_promotion_checks(trace))
    assert [failure.check_id for failure in failures] == ["no_lookahead_in_observation"]
    assert failures[0].implicated_steps == (5,)


def test_minimization_shrinks_the_scenario_while_the_failure_survives(tmp_path):
    steps = [_step(index, reward=0.01 * (index + 1)) for index in range(12)]
    steps[7]["observation"] = {"close": 107.0, "full_series": [1, 2, 3]}
    trace = load_trace_strict(_write(tmp_path / "leak.jsonl", steps))
    minimal = minimize_scenario(trace, "no_lookahead_in_observation")
    assert len(minimal.steps) == 2
    assert len(minimal.steps) < len(trace.steps)
    assert [step["step"] for step in minimal.steps] == [0, 1]
    assert any("full_series" in step["observation"] for step in minimal.steps)


def test_minimizing_a_passing_check_is_refused(tmp_path):
    trace = load_trace_strict(_clean_trace(tmp_path))
    with pytest.raises(PromotionError, match="nothing to minimize"):
        minimize_scenario(trace, "no_lookahead_in_observation")


def _leaky_candidate(tmp_path, name="leak.jsonl"):
    steps = [_step(index, reward=0.01 * (index + 1)) for index in range(10)]
    steps[4]["observation"] = {"close": 104.0, "label": 1}
    trace = load_trace_strict(_write(tmp_path / name, steps))
    failure = blocking_failures(run_promotion_checks(trace))[0]
    return trace, build_silver_candidate(trace, failure, now_unix_ns=1)


def test_a_silver_candidate_carries_its_trigger_and_source_hash(tmp_path):
    trace, candidate = _leaky_candidate(tmp_path)
    assert candidate.triggering_check == "no_lookahead_in_observation"
    assert candidate.severity == "block"
    assert candidate.source_trace_sha256 == trace.source_sha256
    assert candidate.fingerprint.composite == trace.fingerprint.composite
    assert candidate.expected_invariant == {
        "check_id": "no_lookahead_in_observation",
        "must_pass": True,
        "rationale": "the frozen scenario reproduces the failure; a fix is "
        "what makes this check pass on it",
    }


def test_a_passing_check_never_produces_a_candidate(tmp_path):
    trace = load_trace_strict(_clean_trace(tmp_path))
    passing = run_promotion_checks(trace)[0]
    with pytest.raises(PromotionError, match="only a failing check"):
        build_silver_candidate(trace, passing)


def test_silver_rows_are_immutable(tmp_path):
    _, candidate = _leaky_candidate(tmp_path)
    store = SilverStore(tmp_path / "silver.jsonl")
    assert store.append(candidate) is True
    assert store.append(candidate) is False
    assert len(store.read()) == 1

    mutated = type(candidate)(
        candidate_id=candidate.candidate_id,
        triggering_check="rewards_finite",
        severity=candidate.severity,
        detail="rewritten",
        source_trace_sha256=candidate.source_trace_sha256,
        fingerprint=candidate.fingerprint,
        scenario=candidate.scenario,
        expected_invariant=candidate.expected_invariant,
        created_at_unix_ns=candidate.created_at_unix_ns,
    )
    with pytest.raises(PromotionError, match="immutable"):
        store.append(mutated)


def test_silver_queue_reads_strictly(tmp_path):
    _, candidate = _leaky_candidate(tmp_path)
    store = SilverStore(tmp_path / "silver.jsonl")
    store.append(candidate)
    with store.path.open("a", encoding="utf-8") as handle:
        handle.write("{broken\n")
    with pytest.raises(TraceIntegrityError, match="not JSON"):
        store.read()


def test_nothing_becomes_gold_without_a_recorded_operator_decision(tmp_path):
    _, candidate = _leaky_candidate(tmp_path)
    with pytest.raises(PromotionError, match="rationale of at least 20"):
        OperatorDecision(candidate.candidate_id, "promote", "tt", "ok", 1)
    with pytest.raises(PromotionError, match="must name its operator"):
        OperatorDecision(
            candidate.candidate_id, "promote", "  ", "a" * 30, 1
        )
    rejected = OperatorDecision(
        candidate.candidate_id,
        "reject",
        "tt",
        "this leak is a fixture artifact, not a real environment defect",
        1,
    )
    with pytest.raises(PromotionError, match="was rejected, not promoted"):
        promote_to_gold(candidate, rejected)

    wrong_target = OperatorDecision(
        "some-other-candidate",
        "promote",
        "tt",
        "this leak is real and must never regress again",
        1,
    )
    with pytest.raises(PromotionError, match="names a different candidate"):
        promote_to_gold(candidate, wrong_target)


def _gold(tmp_path):
    trace, candidate = _leaky_candidate(tmp_path)
    decision = OperatorDecision(
        candidate.candidate_id,
        "promote",
        "tt",
        "the observation exposed a label column; freeze it as a regression case",
        1,
    )
    return trace, promote_to_gold(candidate, decision)


def test_a_gold_case_freezes_a_minimal_scenario_and_not_the_transcript(tmp_path):
    trace, gold = _gold(tmp_path)
    record = gold.as_record()
    assert len(gold.scenario["steps"]) < len(trace.steps)
    assert "transcript" not in record and "raw_response" not in record
    assert set(gold.scenario) == {"steps", "meta"}
    assert record["operator_decision"]["operator"] == "tt"
    assert record["offline"] == {"network": "forbidden", "model_calls": "none"}


def test_a_gold_case_fails_while_the_defect_is_present_and_passes_once_fixed(tmp_path):
    _, gold = _gold(tmp_path)
    assert evaluate_gold_case(gold).passed is False

    fixed_steps = [
        {
            **step,
            "observation": {
                key: value
                for key, value in step["observation"].items()
                if key != "label"
            },
        }
        for step in gold.scenario["steps"]
    ]
    fixed = GoldCase(
        case_id=gold.case_id,
        triggering_check=gold.triggering_check,
        scenario={"steps": fixed_steps, "meta": gold.scenario["meta"]},
        expected_invariant=gold.expected_invariant,
        source_trace_sha256=gold.source_trace_sha256,
        fingerprint=gold.fingerprint,
        decision=gold.decision,
    )
    outcome = evaluate_gold_case(fixed)
    assert outcome.passed is True
    assert outcome.check_id == "no_lookahead_in_observation"


def test_gold_evaluation_runs_with_sockets_disabled(tmp_path):
    _, gold = _gold(tmp_path)
    opened = []
    original = socket.socket

    def _spy(*args, **kwargs):
        opened.append(args)
        return original(*args, **kwargs)

    socket.socket = _spy
    try:
        evaluate_gold_case(gold)
    finally:
        socket.socket = original
    assert opened == []
    assert socket.socket is original


def test_a_gold_case_round_trips_through_disk(tmp_path):
    _, gold = _gold(tmp_path)
    path = gold.write(tmp_path / "gold" / f"{gold.case_id}.json")
    reloaded = load_gold_case(path)
    assert reloaded.as_record() == gold.as_record()
    assert evaluate_gold_case(reloaded).passed is evaluate_gold_case(gold).passed
