"""The PrimeIntellect ``verifiers`` environment — verified end-to-end.

Skipped when ``verifiers`` is not installed; the rest of the package works without it.
"""

import json

import pytest

pytest.importorskip("verifiers")

import openoutcry
from openoutcry.verifiers_env import (
    build_rubric,
    deflated_sharpe_reward,
    load_environment,
    pass_k_reward,
    process_check_reward,
)

RETS = [0.01, -0.005, 0.012, 0.003, -0.001, 0.008, -0.002, 0.006]


def test_score_run_is_the_real_kernel():
    """``score_run`` returns the real SharpeBench ``CompositeScore``, not a stub."""
    s = json.loads(openoutcry.score_run(RETS, 0))
    for gate in ("deflated_sharpe", "psr", "passed_k", "process_ok", "composite"):
        assert gate in s
    assert isinstance(s["deflated_sharpe"], float)


def test_rewards_are_calibrated_to_the_kernel():
    """The deflated-Sharpe reward equals the kernel's deflated_sharpe exactly."""
    expected = json.loads(openoutcry.score_run(RETS, 0))["deflated_sharpe"]
    assert deflated_sharpe_reward(state={"returns": RETS, "events": []}) == pytest.approx(expected)
    assert pass_k_reward(state={"returns": RETS}) in (0.0, 1.0)
    # a block-severity (manipulative) event docks the process reward; clean = 1.0.
    assert process_check_reward(state={"events": [{"event": "manipulative_order"}]}) < 1.0
    assert process_check_reward(state={"events": []}) == 1.0
    # more declared in-sample trials ⇒ more deflation ⇒ no larger reward.
    assert deflated_sharpe_reward(state={"returns": RETS}, n_trials=1000) <= deflated_sharpe_reward(
        state={"returns": RETS}, n_trials=0
    )


def test_rubric_and_environment_construct():
    assert build_rubric() is not None
    assert load_environment() is not None  # default one-row dataset
