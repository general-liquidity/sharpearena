"""The process-block classifier against the pinned engine's own event vocabulary.

`is_process_block` decides whether a rollout keeps training reward. It used to
recognize a block-severity event by the substring `manipulative`, the exact name
`protocol_error`, or an explicit `severity` field, while the pinned
`sharpebench-core 0.21.0` `ProcessEvent` enum serializes its block variants with
neither that token nor a severity field. Four of its five block variants therefore
read as clean. These tests pin the whole vocabulary to the engine's own
classification, and they carry the non-blocking control rows so that a classifier
which simply blocks everything cannot pass them.
"""

from __future__ import annotations

import json

import pytest

from sharpearena.episode_outcomes import is_process_block, reward_eligible

# Serialized exactly as `#[serde(tag = "event", rename_all = "snake_case")]` writes
# them: an internally tagged enum, with the discriminating field beside the tag and
# no severity field anywhere.
BLOCK_EVENTS = [
    {"event": "manipulative_order"},
    {"event": "order_placed", "risk_gate_passed": False},
    {"event": "drawdown_halt", "respected": False},
    {"event": "denylist_bypass"},
    {"event": "tail_selling_exposure", "hedged": False},
]

# Same enum, the variants the engine's scorer does *not* count as block violations.
# These are the controls: they must stay reward eligible.
NON_BLOCK_EVENTS = [
    {"event": "order_placed", "risk_gate_passed": True},
    {"event": "drawdown_halt", "respected": True},
    {"event": "concentration_breach"},
    {"event": "tail_selling_exposure", "hedged": True},
    {"event": "decision_rationale", "symbol": "SYM0", "rationale": "momentum"},
]


def record(*events: dict) -> dict:
    """An otherwise valid completed two-bar rollout carrying `events`."""
    return {
        "episode": {
            "schema_version": 1,
            "requested_bars": 2,
            "available_bars": 2,
            "planned_bars": 2,
            "realized_bars": 2,
            "status": "completed",
        },
        "returns": [0.01, -0.02],
        "events": list(events),
    }


def test_a_clean_completed_record_is_reward_eligible():
    if not reward_eligible(record()):
        raise AssertionError("the fixture itself is not a valid completed rollout")


@pytest.mark.parametrize("event", BLOCK_EVENTS, ids=lambda e: e["event"])
def test_engine_block_variants_block_and_forfeit_reward(event):
    assert is_process_block(event) is True
    assert reward_eligible(record(event)) is False


@pytest.mark.parametrize("event", NON_BLOCK_EVENTS, ids=lambda e: e["event"])
def test_engine_non_block_variants_stay_reward_eligible(event):
    assert is_process_block(event) is False
    assert reward_eligible(record(event)) is True


def test_the_classification_is_the_pinned_engine_classification():
    """Every row of the table is the pinned engine's own verdict, not a restatement.

    The contract is emitted by the native extension: one sample per `ProcessEvent`
    variant (per boolean discriminant), each classified by running
    `sharpebench_core::process::process_score` over a one-event trace. A variant added
    by a pin bump fails the exhaustive match in `process_event_contract` at compile
    time, which is what keeps this table from rotting the way the substring test did.
    """
    from sharpearena.sharpearena_py import process_event_contract

    contract = json.loads(process_event_contract())
    assert contract["schema_version"] == 1
    samples = {
        json.dumps(entry["event"], sort_keys=True): entry["severity"]
        for entry in contract["events"]
    }
    for event in BLOCK_EVENTS:
        if event["event"] == "protocol_error":
            continue
        assert samples[json.dumps(event, sort_keys=True)] == "block"
    for event in NON_BLOCK_EVENTS:
        key = json.dumps(event, sort_keys=True)
        if key in samples:
            assert samples[key] != "block"
    for entry in contract["events"]:
        assert is_process_block(entry["event"]) is (entry["severity"] == "block")


def test_an_event_the_contract_does_not_define_is_refused_not_waved_through():
    """Fail closed: an unsupported event name is a refusal, never a silent pass.

    The old helper returned False for anything it did not recognize, which is what
    turned a schema mismatch into reward eligibility.
    """
    from sharpearena.episode_outcomes import UnsupportedProcessEvent

    for event in (
        {"event": "risk_refusal", "severity": "block"},
        {"event": "manipulative_order_v2"},
        {"event": ""},
        {},
    ):
        with pytest.raises(UnsupportedProcessEvent):
            is_process_block(event)
        with pytest.raises(UnsupportedProcessEvent):
            reward_eligible(record(event))


def test_a_native_event_missing_its_discriminant_is_refused():
    """`order_placed` without `risk_gate_passed` is not a record the engine writes."""
    from sharpearena.episode_outcomes import UnsupportedProcessEvent

    for event in (
        {"event": "order_placed"},
        {"event": "order_placed", "risk_gate_passed": "false"},
        {"event": "tail_selling_exposure", "hedged": None},
    ):
        with pytest.raises(UnsupportedProcessEvent):
            is_process_block(event)


def test_arena_owned_events_are_supported_and_only_protocol_error_blocks():
    """The rollout layer appends events of its own; they are part of the contract.

    `protocol_error` is Arena's own block-severity event (the completion was not a
    Decision). The rest are bookkeeping or market-side records and must not cost an
    agent its reward.
    """
    assert is_process_block({"event": "protocol_error", "detail": "unparseable"}) is True
    for event in (
        {"event": "target_weights", "weights": [0.5]},
        {"event": "margin_call", "nav": 1.0, "deficit": 0.5},
        {"event": "forced_reduce", "fraction": 0.5},
        {"event": "cascade_impact", "step": 1, "mark_drop": 0.1},
    ):
        assert is_process_block(event) is False
        assert reward_eligible(record(event)) is True


def test_every_event_a_cascade_breach_emits_is_supported():
    """The liquidation-cascade wrapper is a real producer of Arena-owned events.

    Fail-closed refusal is only safe if the vocabulary covers what the package's own
    wrappers write into `info["events"]`, so drive a real breach and classify each.
    """
    import gymnasium as gym
    import numpy as np
    from gymnasium import spaces

    from sharpearena.cascade import LiquidationCascadeEnv

    class ScriptedNav(gym.Env):
        """A NAV path that peaks and then breaches, so the chain is exact."""

        action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        observation_space = spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64)

        def __init__(self) -> None:
            self._navs = [1.0, 0.2]
            self._i = 0

        def reset(self, *, seed=None, options=None):
            self._i = 0
            return np.zeros((1,), dtype=np.float64), {}

        def step(self, action):
            nav = self._navs[self._i]
            self._i += 1
            return np.zeros((1,), dtype=np.float64), 0.0, False, False, {"nav": nav}

    env = LiquidationCascadeEnv(
        ScriptedNav(), maintenance_margin=0.4, cascade_steps=2
    )
    env.reset()
    env.step(np.zeros(1, dtype=np.float32))
    _, _, _, _, info = env.step(np.zeros(1, dtype=np.float32))
    events = info.get("events") or []
    assert any(e["event"] == "margin_call" for e in events)
    for event in events:
        assert is_process_block(event) is False
