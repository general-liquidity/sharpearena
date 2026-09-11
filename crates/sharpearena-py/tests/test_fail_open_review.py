"""Python-surface regressions for the 2026-09-09 adversarial review of SharpeArena
(``docs/audits/2026-09-09/ARENA-REVIEW.md``), findings A5, A6 and A8.

The Rust twin is ``crates/sharpearena/tests/fail_open_review.rs``. A source-tree Rust test
does not establish that the installed package refuses, so every assertion here drives the
built extension or the Python layer that sits on it.

Isolation, per ``VERIFICATION.md`` ("Three tests that would have passed while the thing
they name was not what refused"): each grader below has several rules, so a fixture broken
in more than one way would be refused by whichever fired first and would prove nothing
about the rule under test. Every fixture is therefore broken in exactly one way, and each
case is paired with the otherwise identical well-formed fixture that still succeeds.

Run from the crate dir after ``python -m maturin develop``::

    python -m pytest tests/test_fail_open_review.py -q
"""

import math

import gymnasium as gym
import numpy as np
import pytest

from sharpearena.execution_noise import ExecutionNoiseWrapper
from sharpearena.mandate import Mandate, MandateError, mandate_breach, require_mandate
from sharpearena.preprocessing import ExecutionNoiseConfig
from sharpearena.sharpearena_py import InvalidArgument, perturb_action
from sharpearena.verifiers_env import mandate_reward

_CLEAN_BAR = {"event": "target_weights", "weights": [0.5, 0.3]}


# -- A5. A NaN book scored a clean mandate on all four breach sources ---------


def test_a5_non_finite_weight_is_refused_not_scored():
    """Isolation: the mandate carries no ``max_drawdown`` and no ``max_inventory``, so
    neither cap check can refuse, and ``returns`` is empty, so the return scan cannot
    either. The only rule left that can raise is the weight scan. The paired assertion is
    the same event list with the NaN replaced, which pins the NaN rather than the shape."""
    m = Mandate(style="long_only")
    nan_bar = {"event": "target_weights", "weights": [0.5, float("nan")]}
    with pytest.raises(InvalidArgument) as excinfo:
        mandate_breach(m, [], [nan_bar])
    assert "weights[0][1]" in str(excinfo.value), excinfo.value
    assert mandate_breach(m, [], [_CLEAN_BAR]) == 0.0


def test_a5_non_finite_return_is_refused_not_scored():
    """Isolation: no caps and a clean finite weight event, so both cap checks and the
    weight scan are unreachable and only the return scan can raise. Under the old kernel
    the leading NaN poisoned the equity curve and hid every later drawdown."""
    m = Mandate(style="long_only")
    with pytest.raises(InvalidArgument) as excinfo:
        mandate_breach(m, [float("nan"), -0.30, -0.30], [_CLEAN_BAR])
    assert "returns[0]" in str(excinfo.value), excinfo.value
    assert mandate_breach(m, [0.01, 0.01, 0.01], [_CLEAN_BAR]) == 0.0


def test_a5_an_unusable_cap_is_refused_one_cap_at_a_time():
    """Isolation: each fixture sets exactly one cap out of range and leaves the other
    absent, with empty series so neither non-finite scan can fire. Only the named cap
    check can refuse each one.

    A NaN cap is deliberately not in this list: ``json.dumps`` emits the non-standard
    ``NaN`` literal, so a NaN cap is refused one layer earlier by the wire boundary, with
    ``InvalidJson``. Asserting a refusal on it here would be an assertion satisfied by a
    cause other than the one the test names. The NaN cap is covered where it is actually
    the validator that rejects it, in the A6 cases below."""
    for bad in (0.0, -0.10, 1.5):
        with pytest.raises(InvalidArgument) as excinfo:
            mandate_breach(Mandate(style="unconstrained", max_drawdown=bad), [], [])
        assert "max_drawdown" in str(excinfo.value), excinfo.value
    for bad in (0.0, -1.0):
        with pytest.raises(InvalidArgument) as excinfo:
            mandate_breach(Mandate(style="unconstrained", max_inventory=bad), [], [])
        assert "max_inventory" in str(excinfo.value), excinfo.value
    # The declared ranges still score: max_inventory deliberately admits values above 1.
    assert mandate_breach(Mandate(style="unconstrained", max_drawdown=1.0), [], []) == 0.0
    assert mandate_breach(Mandate(style="unconstrained", max_inventory=2.0), [], []) == 0.0


# -- A6. A malformed mandate graded as no mandate, which is full credit -------


def test_a6_absent_mandate_still_earns_the_vacuous_full_credit():
    """The control for the two tests below: absence is a real state and keeps its 1.0, so
    a refusal there cannot be what makes them pass."""
    assert mandate_reward(state={}) == 1.0
    assert mandate_reward(state={"mandate": None, "info": {}}) == 1.0
    assert mandate_reward(state=None) == 1.0


@pytest.mark.parametrize(
    "malformed,rule",
    [
        ({"style": "momentum_v2"}, "unrecognized style"),
        ({"style": "long_only", "max_drawdown": 1.5}, "drawdown cap above 1"),
        ({"style": "long_only", "max_drawdown": 0.0}, "drawdown cap at 0"),
        ({"style": "long_only", "max_drawdown": float("nan")}, "NaN drawdown cap"),
        ({"style": "long_only", "max_inventory": 0.0}, "non-positive inventory cap"),
        ({"style": "long_only", "max_inventory": float("nan")}, "NaN inventory cap"),
        ({"no_style_key": 1}, "unparseable payload"),
    ],
    ids=["style", "dd_high", "dd_zero", "dd_nan", "inventory", "inventory_nan", "unparseable"],
)
def test_a6_a_present_but_malformed_mandate_is_refused(malformed, rule):
    """Isolation: each payload violates exactly one of ``validate_mandate``'s rules and
    satisfies the rest, so the refusal is attributable to ``rule``. Each is paired below
    with the same payload repaired, which then grades normally rather than raising."""
    with pytest.raises(MandateError):
        require_mandate(malformed)
    with pytest.raises(MandateError):
        mandate_reward(state={"mandate": malformed})
    # ...and the fallback path reads the same way: a malformed `info` mandate is not
    # absence either, so it cannot collect the vacuous 1.0 through the second lookup.
    with pytest.raises(MandateError):
        mandate_reward(state={"info": {"mandate": malformed}})


def test_a6_the_repaired_payloads_grade_normally():
    """The other half of the isolation: every rule violated above, repaired, scores."""
    repaired = [
        {"style": "momentum"},
        {"style": "long_only", "max_drawdown": 0.5},
        {"style": "long_only", "max_drawdown": 0.10},
        {"style": "long_only", "max_drawdown": 0.20},
        {"style": "long_only", "max_inventory": 2.0},
        {"style": "long_only", "max_inventory": 0.5},
        {"style": "long_only"},
    ]
    for payload in repaired:
        assert require_mandate(payload) is payload
        # Scored, not refused. The value itself is the grader's business: the 0.5 gross
        # cap is genuinely breached by the 0.8-gross clean bar, which is the point.
        reward = mandate_reward(state={"mandate": payload, "events": [_CLEAN_BAR]})
        assert 0.0 <= reward <= 1.0
    # A book that breaches nothing keeps the full 1.0, so "scored" is not "penalized".
    assert mandate_reward(state={"mandate": {"style": "long_only"}, "events": [_CLEAN_BAR]}) == 1.0
    # And a real breach still bites, so "no exception" is not the same as "full credit".
    short_bar = {"event": "target_weights", "weights": [-0.5, 0.2]}
    breached = mandate_reward(
        state={"mandate": {"style": "long_only"}, "events": [short_bar]}
    )
    assert breached < 1.0


# -- A8. Execution-noise knobs unvalidated on every surface -------------------
#
# Surfaces that accept these knobs: the pyo3 binding `perturb_action`, the gym wrapper
# `ExecutionNoiseWrapper`, and the `ExecutionNoiseConfig` dataclass. There is no WASM or
# npm export of either knob, so those three are the whole Python-visible set.


@pytest.mark.parametrize("bad", [-1.0, -0.1, 1.000_001, 4.0, float("nan"), math.inf])
def test_a8_binding_refuses_a_bad_delay_prob(bad):
    """Isolation: ``slippage_bps`` is held at the in-range ``0.0``, so only ``delay_prob``
    can be refused. A config broken in both knobs would be refused by either check."""
    with pytest.raises(InvalidArgument) as excinfo:
        perturb_action(5, 1, [0.2, -0.5], [9.9, 9.9], bad, 0.0)
    assert "delay_prob" in str(excinfo.value), excinfo.value


@pytest.mark.parametrize("bad", [-100.0, -1e-9, float("nan"), math.inf])
def test_a8_binding_refuses_a_bad_slippage_bps(bad):
    """Isolation: ``delay_prob`` is held at the in-range ``0.0``, so only ``slippage_bps``
    can be refused."""
    with pytest.raises(InvalidArgument) as excinfo:
        perturb_action(5, 1, [0.2, -0.5], [9.9, 9.9], 0.0, bad)
    assert "slippage_bps" in str(excinfo.value), excinfo.value


def test_a8_binding_still_runs_the_declared_ranges():
    """The paired success leg: both ends of ``delay_prob`` and a live slippage knob."""
    requested, previous = [0.2, -0.5], [9.9, 9.9]
    assert perturb_action(5, 1, requested, previous, 0.0, 0.0) == requested
    assert perturb_action(5, 1, requested, previous, 1.0, 0.0) == previous
    assert perturb_action(5, 1, requested, previous, 0.0, 100.0) != requested


@pytest.mark.parametrize(
    "kwargs,knob",
    [
        ({"delay_prob": -0.1}, "delay_prob"),
        ({"delay_prob": 4.0}, "delay_prob"),
        ({"delay_prob": float("nan")}, "delay_prob"),
        ({"slippage_bps": -25.0}, "slippage_bps"),
        ({"slippage_bps": float("nan")}, "slippage_bps"),
    ],
    ids=["negative", "above_one", "nan_delay", "negative_slip", "nan_slip"],
)
def test_a8_wrapper_refuses_at_construction(kwargs, knob):
    """Isolation: each case sets one knob and leaves the other at its in-range ``0.0``
    default, so the message names the knob the case is about. The wrapper is the surface a
    user configures, and it accepted these long before the core saw an action."""
    env = gym.make("Pendulum-v1")
    try:
        with pytest.raises(ValueError) as excinfo:
            ExecutionNoiseWrapper(env, seed=1, **kwargs)
        assert knob in str(excinfo.value), excinfo.value
        # The same wrapper with the knob in range constructs and steps.
        ok = ExecutionNoiseWrapper(env, seed=1, delay_prob=0.5, slippage_bps=25.0)
        ok.reset(seed=1)
        ok.step(np.zeros(env.action_space.shape, dtype=np.float32))
    finally:
        env.close()


@pytest.mark.parametrize(
    "kwargs,knob",
    [
        ({"delay_prob": -0.1}, "delay_prob"),
        ({"delay_prob": 1.5}, "delay_prob"),
        ({"delay_prob": float("nan")}, "delay_prob"),
        ({"slippage_bps": -1.0}, "slippage_bps"),
        ({"slippage_bps": float("nan")}, "slippage_bps"),
    ],
    ids=["negative", "above_one", "nan_delay", "negative_slip", "nan_slip"],
)
def test_a8_config_refuses_at_construction(kwargs, knob):
    """Isolation: one knob per case, the other left at its default. ``ExecutionNoiseConfig``
    had no ``__post_init__`` at all while its sibling ``PreprocessingConfig`` did."""
    with pytest.raises(ValueError) as excinfo:
        ExecutionNoiseConfig(**kwargs)
    assert knob in str(excinfo.value), excinfo.value


def test_a8_config_enabled_can_no_longer_report_a_nan_as_live():
    """``enabled`` is ``delay_prob != 0.0 or slippage_bps != 0.0`` and ``nan != 0.0`` is
    True, so a NaN knob reported the run as noise-enabled while the core treated it as
    configured-and-poisoned. The config cannot hold a NaN now, and the in-range values on
    either side of the old behaviour still read correctly."""
    with pytest.raises(ValueError):
        ExecutionNoiseConfig(delay_prob=float("nan"))
    assert ExecutionNoiseConfig().enabled is False
    assert ExecutionNoiseConfig(slippage_bps=25.0).enabled is True


# -- The generalization score manufactured out of no evidence ----------------
#
# Same shape as A5 and A6: a published property that quietly does not hold on some
# input, reported as a plausible number instead of a refusal. `evaluate_seeds` skipped
# `score_run` below two pooled observations and wrote `deflated_sharpe: 0.0`, which
# `kernel_score_difference` would then difference against a real score.


def _no_bar_env(_seed):
    """An env whose episode ends before it produces a second bar.

    Deliberately minimal: it has no way to fail other than by yielding too few
    observations, so a refusal downstream cannot come from a malformed observation, a
    non-finite reward, or an exception escaping the rollout."""

    class _Env:
        def reset(self):
            return {"closes": np.array([1.0, 1.0])}, {}

        def step(self, _action):
            return {"closes": np.array([1.0, 1.0])}, 0.01, True, False, {}

    return _Env()


def test_evaluate_seeds_reports_unavailability_not_a_zero_score():
    """Isolation: the only defect in this fixture is the number of observations.

    One seed, one bar, a finite reward and a well-formed observation, so the kernel's
    other typed errors are unreachable and ``at least 2 observations are required`` is
    the only refusal it can raise. The paired case below is the identical fixture with
    enough bars, which scores a real float. The two cases differ in nothing but the
    pooled length, which is the cause the test names."""
    from sharpearena.generalization import evaluate_seeds
    from sharpearena.kernel_score import (
        UNAVAILABLE_KERNEL_ERROR,
        is_kernel_score_unavailable,
    )

    row = evaluate_seeds(_no_bar_env, [0], max_steps=8)
    score = row["deflated_sharpe"]
    assert is_kernel_score_unavailable(score), score
    assert score.startswith(UNAVAILABLE_KERNEL_ERROR), score
    assert "at least 2 observations are required" in score, score
    assert row["n_seeds"] == 1
    # An empty seed list pools nothing at all and is refused the same way.
    empty = evaluate_seeds(_no_bar_env, [], max_steps=8)
    assert is_kernel_score_unavailable(empty["deflated_sharpe"]), empty


def test_evaluate_seeds_still_scores_a_series_the_kernel_can_read():
    """The other half: change only the bar count and a real float comes back."""
    from sharpearena.generalization import evaluate_seeds
    from sharpearena.kernel_score import is_kernel_score_unavailable

    class _Env:
        def __init__(self):
            self._i = 0

        def reset(self):
            self._i = 0
            return {"closes": np.array([1.0, 1.0])}, {}

        def step(self, _action):
            self._i += 1
            return {"closes": np.array([1.0, 1.0])}, 0.01 * self._i, self._i >= 4, False, {}

    row = evaluate_seeds(lambda _s: _Env(), [0], max_steps=8)
    assert not is_kernel_score_unavailable(row["deflated_sharpe"]), row
    assert isinstance(row["deflated_sharpe"], float)


def test_a_gap_against_an_unavailable_split_stays_unavailable():
    """The consequence the refusal buys: an unscored split can no longer be
    differenced against a scored one to produce a generalization gap. Isolation: the
    train split here is the well-formed env from the test above, so the only reason the
    gap is unavailable is the test split's observation count."""
    from sharpearena.generalization import evaluate_seeds
    from sharpearena.kernel_score import (
        is_kernel_score_unavailable,
        kernel_score_difference,
    )

    short = evaluate_seeds(_no_bar_env, [0], max_steps=8)
    gap = kernel_score_difference(2.5, short["deflated_sharpe"])
    assert is_kernel_score_unavailable(gap), gap
