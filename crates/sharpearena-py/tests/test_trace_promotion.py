"""Strict trace promotion: reject rather than skip, and freeze minimal cases."""

from __future__ import annotations

import json
import socket
from copy import deepcopy
from dataclasses import replace

import pytest
from sharpearena.trace import load_trace
from sharpearena.gym import SharpeArenaEnv
from sharpearena.promotion_replay import gym_replay_inputs, replay_gym_inputs
from sharpearena.trace_promotion import (
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
        "observation": (
            observation if observation is not None else {"close": 100.0 + index}
        ),
        "decision": (
            decision
            if decision is not None
            else {"orders": [{"symbol": "AAA", "action": "buy", "target_weight": 0.1}]}
        ),
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
    with pytest.raises(
        TraceIntegrityError, match="missing required field 'model_digest'"
    ):
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
        "rationale": "the named check must pass on newly produced replay output",
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
    with pytest.raises(PromotionError, match="identity"):
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
        OperatorDecision(candidate.candidate_id, "promote", "  ", "a" * 30, 1)
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


def _replay_gold(tmp_path, monkeypatch):
    original = SharpeArenaEnv._decode_obs

    def leaking_decode(self, wire):
        observed = original(self, wire)
        if len(json.loads(wire)["symbols"][0]["close_history"]) == 5:
            observed["label"] = 1
        return observed

    monkeypatch.setattr(SharpeArenaEnv, "_decode_obs", leaking_decode)
    env = SharpeArenaEnv(n_symbols=1, n_days=16, seed=7)
    try:
        inputs = gym_replay_inputs(env, [[0.1]] * 10)
    finally:
        env.close()
    produced = replay_gym_inputs(inputs, _meta(environment_id="sharpearena-gym-v0"))
    path = tmp_path / "produced.jsonl"
    path.write_text(
        "\n".join(json.dumps(row) for row in (*produced.steps, produced.meta)) + "\n",
        encoding="utf-8",
    )
    trace = load_trace_strict(path)
    failure = blocking_failures(run_promotion_checks(trace))[0]
    candidate = build_silver_candidate(
        trace, failure, replay_inputs=inputs, now_unix_ns=1
    )
    decision = OperatorDecision(
        candidate.candidate_id,
        "promote",
        "fixture-operator",
        "preserve the action prefix that exposes the label",
        1,
    )
    return trace, candidate, promote_to_gold(candidate, decision), original


def test_gold_keeps_replay_history_not_minimized_bad_output(tmp_path, monkeypatch):
    trace, candidate, gold, _ = _replay_gold(tmp_path, monkeypatch)
    record = gold.as_record()
    assert len(candidate.scenario["steps"]) < len(trace.steps)
    assert len(gold.scenario["inputs"]["actions"]) == len(trace.steps) == 10
    assert "transcript" not in record and "raw_response" not in record
    assert set(gold.scenario) == {"inputs", "meta"}
    assert "observation" not in json.dumps(gold.scenario)
    assert record["operator_decision"]["operator"] == "fixture-operator"
    assert record["offline"] == {
        "network": "python_socket_guard",
        "model_calls": "none",
    }


def test_same_gold_case_fails_with_broken_producer_and_passes_with_fixed_producer(
    tmp_path, monkeypatch
):
    _, _, gold, original = _replay_gold(tmp_path, monkeypatch)
    before = gold.as_record()
    assert evaluate_gold_case(gold).passed is False
    calls = []

    def repaired(self, wire):
        calls.append(wire)
        return original(self, wire)

    monkeypatch.setattr(SharpeArenaEnv, "_decode_obs", repaired)
    outcome = evaluate_gold_case(gold)
    assert outcome.passed is True
    assert outcome.check_id == "no_lookahead_in_observation"
    assert len(calls) == 11  # reset plus every action actually reached the producer
    assert gold.as_record() == before  # never edit the fixture to make it pass


def test_gold_replay_actually_refuses_a_producer_socket_attempt(tmp_path, monkeypatch):
    _, _, gold, _ = _replay_gold(tmp_path, monkeypatch)
    original = socket.socket
    attempted = []

    def network_step(self, action):
        attempted.append(True)
        return socket.socket()

    monkeypatch.setattr(SharpeArenaEnv, "step", network_step)
    with pytest.raises(PromotionError, match="attempted a network call"):
        evaluate_gold_case(gold)
    assert attempted == [True]
    assert socket.socket is original


def test_a_gold_case_round_trips_through_disk(tmp_path, monkeypatch):
    _, _, gold, _ = _replay_gold(tmp_path, monkeypatch)
    path = gold.write(tmp_path / "gold" / f"{gold.case_id}.json")
    reloaded = load_gold_case(path)
    assert reloaded.as_record() == gold.as_record()
    assert evaluate_gold_case(reloaded).passed is evaluate_gold_case(gold).passed


def test_frozen_output_alone_is_not_an_executable_gold_case(tmp_path):
    with pytest.raises(PromotionError, match="replay inputs"):
        _gold(tmp_path)


def test_changed_gold_content_cannot_keep_its_identity(tmp_path, monkeypatch):
    _, _, gold, _ = _replay_gold(tmp_path, monkeypatch)
    gold.scenario["inputs"]["actions"][0] = [0.9]
    with pytest.raises((PromotionError, TraceIntegrityError), match="identity"):
        evaluate_gold_case(gold)


def test_replay_inputs_must_reproduce_the_source_before_candidate_approval(
    tmp_path, monkeypatch
):
    trace, candidate, _, _ = _replay_gold(tmp_path, monkeypatch)
    inputs = deepcopy(candidate.scenario["replay"]["inputs"])
    inputs["actions"][0] = [0.9]
    with pytest.raises(PromotionError, match="complete source trace"):
        build_silver_candidate(
            trace,
            blocking_failures(run_promotion_checks(trace))[0],
            replay_inputs=inputs,
        )


def test_shortened_replay_prefix_is_not_a_minimized_reproduction(tmp_path, monkeypatch):
    trace, candidate, _, _ = _replay_gold(tmp_path, monkeypatch)
    inputs = deepcopy(candidate.scenario["replay"]["inputs"])
    inputs["actions"] = inputs["actions"][4:6]
    with pytest.raises(PromotionError, match="complete source trace"):
        build_silver_candidate(
            trace,
            blocking_failures(run_promotion_checks(trace))[0],
            replay_inputs=inputs,
        )


@pytest.mark.parametrize(
    "field",
    [
        "scenario",
        "expected_invariant",
        "source_trace_sha256",
        "operator_decision",
        "fingerprint",
    ],
)
def test_changed_gold_record_is_refused_on_load(tmp_path, monkeypatch, field):
    _, _, gold, _ = _replay_gold(tmp_path, monkeypatch)
    record = gold.as_record()
    if field == "scenario":
        record[field]["inputs"]["params"]["seed"] += 1
    elif field == "expected_invariant":
        record[field]["must_pass"] = False
    elif field == "source_trace_sha256":
        record[field] = "0" * 64
    elif field == "operator_decision":
        record[field]["operator"] = "somebody-else"
    else:
        record[field]["model"] = "0" * 64
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(PromotionError, match="identity"):
        load_gold_case(path)


def test_legacy_output_only_gold_is_refused(tmp_path, monkeypatch):
    _, _, gold, _ = _replay_gold(tmp_path, monkeypatch)
    record = gold.as_record()
    record["schema_version"] = "sharpearena.promotion/1.0.0"
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(TraceIntegrityError, match="V1 output-only"):
        load_gold_case(path)


def test_unknown_producer_never_loads_code(tmp_path, monkeypatch):
    _, candidate, _, _ = _replay_gold(tmp_path, monkeypatch)
    inputs = deepcopy(candidate.scenario["replay"]["inputs"])
    inputs["producer"] = "untrusted.module:function"
    with pytest.raises(PromotionError, match="unsupported replay producer"):
        replay_gym_inputs(inputs, {})


def test_gold_and_record_do_not_alias_the_candidate(tmp_path, monkeypatch):
    _, candidate, gold, _ = _replay_gold(tmp_path, monkeypatch)
    frozen = gold.as_record()
    candidate.scenario["replay"]["inputs"]["actions"][0] = [0.9]
    exported = gold.as_record()
    exported["scenario"]["inputs"]["params"]["seed"] += 1
    assert gold.as_record() == frozen
    assert evaluate_gold_case(gold).passed is False


def test_candidate_identity_binds_source_and_expected_check(tmp_path):
    _, candidate = _leaky_candidate(tmp_path)
    for changed in (
        replace(candidate, source_trace_sha256="0" * 64),
        replace(candidate, expected_invariant={"must_pass": False}),
    ):
        with pytest.raises(PromotionError, match="identity"):
            changed.validate_identity()


def test_replay_recipe_covers_every_gym_constructor_parameter():
    import inspect
    from sharpearena.promotion_replay import GYM_PARAM_FIELDS

    assert GYM_PARAM_FIELDS == set(inspect.signature(SharpeArenaEnv).parameters)


@pytest.mark.parametrize(
    "actions",
    [[], [[0.1]], [[True], [0.1]], [[float("nan")], [0.1]], [[10**400], [0.1]]],
)
def test_invalid_action_prefix_is_refused_before_native_execution(
    tmp_path, monkeypatch, actions
):
    _, candidate, _, _ = _replay_gold(tmp_path, monkeypatch)
    inputs = deepcopy(candidate.scenario["replay"]["inputs"])
    inputs["actions"] = actions
    with pytest.raises(PromotionError, match="actions"):
        replay_gym_inputs(inputs, {})


def test_replay_refuses_early_terminal_and_always_closes(tmp_path, monkeypatch):
    _, candidate, _, _ = _replay_gold(tmp_path, monkeypatch)
    inputs = candidate.scenario["replay"]["inputs"]
    closed = []
    step = SharpeArenaEnv.step

    def premature_end(self, action):
        obs, reward, _, _, info = step(self, action)
        return obs, reward, False, True, info

    monkeypatch.setattr(SharpeArenaEnv, "step", premature_end)
    monkeypatch.setattr(SharpeArenaEnv, "close", lambda self: closed.append(True))
    with pytest.raises(PromotionError, match="ended before"):
        replay_gym_inputs(inputs, {})
    assert closed == [True]


def test_replay_exposes_nonfinite_producer_reward_to_the_check(tmp_path, monkeypatch):
    _, _, gold, _ = _replay_gold(tmp_path, monkeypatch)
    step = SharpeArenaEnv.step

    def invalid_reward(self, action):
        obs, _, terminated, truncated, info = step(self, action)
        return obs, float("nan"), terminated, truncated, info

    monkeypatch.setattr(SharpeArenaEnv, "step", invalid_reward)
    # We cannot relabel an approved invariant under the same identity.
    changed = replace(
        gold,
        triggering_check="rewards_finite",
        expected_invariant={"check_id": "rewards_finite", "must_pass": True},
    )
    with pytest.raises(PromotionError, match="identity"):
        evaluate_gold_case(changed)
    result = run_promotion_checks(
        replay_gym_inputs(gold.scenario["inputs"], gold.scenario["meta"])
    )
    assert result[0].check_id == "rewards_finite" and result[0].passed is False


def test_replay_owns_actions_and_captures_nondefault_controls():
    import numpy as np

    env = SharpeArenaEnv(
        n_symbols=1,
        n_days=16,
        seed=42,
        window_start=2,
        window_end=12,
        distribution_mode="extreme",
        vol_clustering=0.3,
        jump_burst_probability=0.2,
        jump_burst_persistence=0.4,
        jump_burst_size=0.05,
        mode="eval",
        env_kwargs={"fee_bps": 3.0},
    )
    actions = [np.array([0.1]), np.array([-0.2])]
    try:
        inputs = gym_replay_inputs(env, actions)
    finally:
        env.close()
    actions[0][0] = 0.9
    assert inputs["actions"] == [[0.1], [-0.2]]
    first = replay_gym_inputs(inputs, _meta())
    second = replay_gym_inputs(inputs, _meta())
    assert first == second
    assert first.meta["config"] == inputs["params"]
    assert first.meta["config"]["jump_burst_persistence"] == 0.4
    assert first.meta["config"]["env_kwargs"] == {"fee_bps": 3.0}


def test_gold_loader_refuses_duplicate_keys_and_omitted_fields(tmp_path, monkeypatch):
    _, _, gold, _ = _replay_gold(tmp_path, monkeypatch)
    record = gold.as_record()
    path = tmp_path / "ambiguous.json"
    encoded = json.dumps(record)
    path.write_text('{"case_id":"wrong",' + encoded[1:], encoding="utf-8")
    with pytest.raises(TraceIntegrityError, match="duplicate JSON key"):
        load_gold_case(path)
    del record["expected_invariant"]
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(TraceIntegrityError, match="expected_invariant"):
        load_gold_case(path)


def test_silver_load_revalidates_content_identity(tmp_path):
    _, candidate = _leaky_candidate(tmp_path)
    store = SilverStore(tmp_path / "silver.jsonl")
    store.append(candidate)
    record = candidate.as_record()
    record["scenario"]["steps"][0]["reward"] = 9000.0
    store.path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(PromotionError, match="identity"):
        store.read()
