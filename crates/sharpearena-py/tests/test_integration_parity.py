"""The parity checker catches a broken adapter, and cannot pass without comparing.

Every later adapter ticket proves itself with this checker, so the checker's own
failure modes are what these tests pin: a deliberately broken mapping must be caught by
name, and the three ways a check could report parity without having compared anything
(no engine spec hash, a disagreeing spec hash, no steps) must raise instead.

Run from the crate dir after ``python -m maturin develop``::

    python -m pytest crates/sharpearena-py/tests/test_integration_parity.py -q
"""

import dataclasses

import numpy as np
import pytest

from sharpearena.gym import SharpeArenaEnv
from sharpearena.integrations.parity import (
    CORE_FIXTURES,
    ParityFixture,
    ParityMismatch,
    ParityUnavailable,
    adapter_rollout,
    check_adapter_parity,
    compare,
    engine_rollout,
)

FIXTURES = {fixture.name: fixture for fixture in CORE_FIXTURES}


class _Delegating:
    """A real ``SharpeArenaEnv`` behind a mapping a subclass breaks on purpose."""

    def __init__(self, fixture: ParityFixture) -> None:
        self.inner = SharpeArenaEnv(
            seed=fixture.seed,
            max_weight=fixture.max_weight,
            allow_short=fixture.allow_short,
            mode=fixture.mode,
            **fixture.scenario_kwargs(),
        )
        self._env = self.inner._env

    @property
    def symbols(self):
        return self.inner.symbols

    def reset(self):
        return self.inner.reset()

    def step(self, action):
        return self.inner.step(action)


class _SwapsTerminalFlags(_Delegating):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.inner.step(action)
        return obs, reward, truncated, terminated, info


class _ClipsInsteadOfRefusing(_Delegating):
    def step(self, action):
        low = self.inner.action_space.low
        high = self.inner.action_space.high
        return self.inner.step(np.clip(np.asarray(action, dtype=np.float32), low, high))


class _PermutesSymbols(_Delegating):
    def step(self, action):
        rolled = np.roll(np.asarray(action, dtype=np.float32), 1)
        return self.inner.step(rolled)


class _DropsLastBar(_Delegating):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.inner.step(action)
        if truncated:
            return obs, reward, terminated, False, info
        return obs, reward, terminated, truncated, info


@pytest.mark.parametrize("fixture", CORE_FIXTURES, ids=lambda f: f.name)
def test_the_shipped_adapter_reproduces_the_engine(fixture):
    report = check_adapter_parity(fixture)
    assert report.ok, report.mismatches
    assert report.steps_compared > 0


@pytest.mark.parametrize("fixture", CORE_FIXTURES, ids=lambda f: f.name)
def test_each_fixture_reaches_the_outcome_it_declares(fixture):
    record = engine_rollout(fixture)
    expected = {"complete": "truncated", "incomplete": "incomplete", "refused": "refused"}
    assert record.terminal == expected[fixture.expect]


@pytest.mark.parametrize(
    "broken, fixture_name, expected",
    [
        pytest.param(_SwapsTerminalFlags, "calm-complete", "terminated", id="swapped terminal flags"),
        pytest.param(
            _ClipsInsteadOfRefusing,
            "calm-refused-out-of-bounds",
            "accepted:out_of_bounds",
            id="clips instead of refusing",
        ),
        pytest.param(_PermutesSymbols, "calm-complete", "reward", id="permuted symbol order"),
        pytest.param(_DropsLastBar, "calm-complete", "truncated", id="hidden truncation"),
    ],
)
def test_a_broken_adapter_mapping_is_caught(broken, fixture_name, expected):
    report = check_adapter_parity(FIXTURES[fixture_name], broken, source=broken.__name__)
    assert not report.ok
    assert any(expected in mismatch for mismatch in report.mismatches), report.mismatches
    with pytest.raises(ParityMismatch):
        report.require_parity()


def test_an_adapter_that_cannot_be_built_is_unavailable_not_parity():
    def explodes(_fixture):
        raise ImportError("no such learner")

    with pytest.raises(ParityUnavailable, match="could not be constructed"):
        adapter_rollout(FIXTURES["calm-complete"], explodes, source="absent-adapter")


def test_a_record_without_an_engine_spec_hash_cannot_be_compared():
    fixture = FIXTURES["calm-complete"]
    reference = engine_rollout(fixture)
    candidate = dataclasses.replace(adapter_rollout(fixture), spec_hash=None)
    with pytest.raises(ParityUnavailable, match="no engine spec hash"):
        compare(reference, candidate)


def test_two_engine_builds_cannot_be_compared_to_each_other():
    fixture = FIXTURES["calm-complete"]
    reference = engine_rollout(fixture)
    candidate = dataclasses.replace(adapter_rollout(fixture), spec_hash="0000000000000000")
    with pytest.raises(ParityUnavailable, match="engine spec"):
        compare(reference, candidate)


def test_comparing_nothing_is_not_parity():
    fixture = FIXTURES["calm-complete"]
    empty = dataclasses.replace(engine_rollout(fixture), steps=())
    with pytest.raises(ParityUnavailable, match="nothing to compare"):
        compare(empty, dataclasses.replace(empty, source="candidate"))


def test_records_of_different_fixtures_cannot_be_compared():
    reference = engine_rollout(FIXTURES["calm-complete"])
    candidate = adapter_rollout(FIXTURES["calm-incomplete"])
    with pytest.raises(ParityUnavailable, match="different fixtures"):
        compare(reference, candidate)


class _UnrelatedStepFailure(_Delegating):
    def step(self, action):
        raise RuntimeError("unrelated adapter failure")


def test_unrelated_exception_cannot_impersonate_a_bounds_refusal():
    fixture = dataclasses.replace(
        FIXTURES["calm-refused-out-of-bounds"],
        actions=((0.125, 0.25, -0.125), (5.0, 0.0, 0.0)),
    )

    class FailsOnlyInvalidAction(_Delegating):
        def step(self, action):
            if abs(float(action[0])) > 0.5:
                raise RuntimeError("not an action validation refusal")
            return super().step(action)

    assert not check_adapter_parity(fixture, FailsOnlyInvalidAction).ok


def test_original_first_action_false_agreement_is_unavailable():
    fixture = ParityFixture(
        name="original-c1", n_symbols=3, n_days=40, seed=5,
        max_weight=0.5, actions=((5.0, 0.0, 0.0),), expect="complete",
    )
    with pytest.raises(ParityUnavailable, match="transition"):
        check_adapter_parity(fixture, _UnrelatedStepFailure)


def test_fixture_expectation_is_enforced_not_only_validated():
    fixture = dataclasses.replace(FIXTURES["calm-incomplete"], expect="complete")
    report = check_adapter_parity(fixture)
    assert not report.ok
    assert any("expected outcome" in mismatch for mismatch in report.mismatches)


def test_missing_config_readback_is_unavailable():
    reference = engine_rollout(FIXTURES["calm-complete"])
    candidate = dataclasses.replace(adapter_rollout(FIXTURES["calm-complete"]), effective_config={})
    with pytest.raises(ParityUnavailable, match="effective config"):
        compare(reference, candidate)


def test_missing_nav_is_not_favorable_default_equity():
    class MissingNav(_Delegating):
        def step(self, action):
            obs, reward, terminated, truncated, info = super().step(action)
            info.pop("nav", None)
            return obs, reward, terminated, truncated, info

    with pytest.raises(ParityUnavailable, match="nav"):
        adapter_rollout(FIXTURES["calm-incomplete"], MissingNav)


def test_missing_execution_seed_readback_is_unavailable():
    reference = engine_rollout(FIXTURES["calm-complete"])
    candidate = adapter_rollout(FIXTURES["calm-complete"])
    config = dict(candidate.effective_config)
    config.pop("exec_seed")
    with pytest.raises(ParityUnavailable, match="effective config"):
        compare(reference, dataclasses.replace(candidate, effective_config=config))


def test_changed_execution_seed_readback_is_a_mismatch():
    reference = engine_rollout(FIXTURES["calm-complete"])
    candidate = adapter_rollout(FIXTURES["calm-complete"])
    config = dict(candidate.effective_config)
    config["exec_seed"] += 1
    report = compare(reference, dataclasses.replace(candidate, effective_config=config))
    assert not report.ok
    assert any("exec_seed" in mismatch for mismatch in report.mismatches)


def test_imported_extension_does_not_attest_an_unrelated_adapter():
    class UnrelatedEngine(_Delegating):
        def __init__(self, fixture):
            super().__init__(fixture)
            self._env = object()

    with pytest.raises(ParityUnavailable, match="native engine"):
        adapter_rollout(FIXTURES["calm-incomplete"], UnrelatedEngine)
