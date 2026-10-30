"""The C06/C07 structures refuse the records that would misreport a run.

Run from the crate dir after ``python -m maturin develop``::

    python -m pytest crates/sharpearena-py/tests/test_integration_contracts.py -q
"""

import numpy as np
import pytest
from sharpearena import SharpeArenaEnv
from sharpearena.integrations.contracts import (
    ACTION_SPACE_DTYPE,
    AUTORESET_MODES_WITH_FINAL_OBS,
    ENGINE_ACTION_DTYPE,
    ENGINE_EXACTNESS,
    SUPPORTED_AUTORESET_MODES,
    AttemptCounts,
    ContractViolation,
    ExactnessDomain,
    TrustClass,
    is_float32_exact,
)
from sharpearena.vector import SharpeArenaVectorEnv


def _counts(**overrides) -> dict:
    base = {
        "expected": 4,
        "attempted": 4,
        "completed": 4,
        "refused": 0,
        "failed": 0,
        "retried": 0,
    }
    base.update(overrides)
    return base


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"completed": 3}, id="an attempt reached no disposition"),
        pytest.param(
            {"expected": 5, "attempted": 5, "completed": 5, "refused": 1},
            id="dispositions exceed attempts",
        ),
        pytest.param(
            {"expected": 2, "attempted": 4, "completed": 4},
            id="completed exceeds planned",
        ),
        pytest.param({"retried": 9}, id="more retries than attempts"),
        pytest.param({"refused": -1, "completed": 5}, id="a negative count"),
    ],
)
def test_counts_that_do_not_add_up_are_refused(overrides):
    with pytest.raises(ContractViolation):
        AttemptCounts(**_counts(**overrides))


def test_a_missing_count_is_refused_rather_than_read_as_zero():
    payload = _counts()
    del payload["refused"]
    with pytest.raises(ContractViolation, match="absent is not zero"):
        AttemptCounts.from_record(payload)


def test_a_run_with_refusals_is_not_complete_and_says_why():
    counts = AttemptCounts(
        expected=4, attempted=4, completed=2, refused=1, failed=1, retried=0
    )
    assert not counts.is_complete
    assert "1 refused" in counts.shortfall()
    assert "1 failed" in counts.shortfall()
    assert counts.as_record()["complete"] is False


def test_a_field_with_no_declared_rule_has_no_tolerance():
    domain = ExactnessDomain(
        exact=("reward",), tolerances={"nav": 1e-9}, rationale="test"
    )
    assert domain.is_exact("reward")
    assert domain.tolerance_for("nav") == pytest.approx(1e-9)
    with pytest.raises(ContractViolation, match="no declared comparison rule"):
        domain.tolerance_for("turnover")
    with pytest.raises(ContractViolation, match="declared exact"):
        domain.tolerance_for("reward")


def test_a_field_cannot_be_both_exact_and_tolerance_bounded():
    with pytest.raises(ContractViolation):
        ExactnessDomain(exact=("nav",), tolerances={"nav": 1e-9}, rationale="test")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), 10**400])
def test_exactness_tolerances_must_be_finite_and_representable(value):
    with pytest.raises(ContractViolation):
        ExactnessDomain(exact=(), tolerances={"nav": value}, rationale="test")


def test_exactness_domain_snapshots_caller_owned_tolerances():
    tolerances = {"nav": 1e-9}
    domain = ExactnessDomain(exact=(), tolerances=tolerances, rationale="test")
    tolerances["nav"] = float("nan")
    assert domain.tolerance_for("nav") == 1e-9
    with pytest.raises(TypeError):
        domain.tolerances["nav"] = float("inf")


def test_empty_workload_is_arithmetically_complete_but_plan_checked_explicitly():
    counts = AttemptCounts(
        expected=0, attempted=0, completed=0, refused=0, failed=0, retried=0
    )
    assert counts.is_complete
    with pytest.raises(ContractViolation, match="differs from planned"):
        counts.require_complete(expected=4)
    with pytest.raises(ContractViolation, match="explicit allow_empty"):
        counts.require_complete(expected=0)
    counts.require_complete(expected=0, allow_empty=True)


def test_plan_completeness_checks_counts_and_dispositions():
    AttemptCounts(**_counts()).require_complete(expected=4)
    incomplete = AttemptCounts(**_counts(completed=3, failed=1))
    with pytest.raises(ContractViolation, match="1 failed"):
        incomplete.require_complete(expected=4)


def test_the_engine_domain_admits_no_tolerances():
    assert ENGINE_EXACTNESS.tolerances == {}
    assert "reward" in ENGINE_EXACTNESS.exact


def test_a_digest_match_does_not_make_foreign_bytes_safe_to_load():
    assert TrustClass.TRUSTED_LOCAL.safe_to_deserialize_locally
    assert not TrustClass.DIGEST_VERIFIED.safe_to_deserialize_locally
    assert not TrustClass.UNTRUSTED.safe_to_deserialize_locally


@pytest.mark.parametrize("mode", SUPPORTED_AUTORESET_MODES)
def test_every_listed_autoreset_mode_is_one_the_vector_env_accepts(mode):
    # The list is a claim about this package, so it is checked against the constructor
    # rather than against the upstream enum.
    env = SharpeArenaVectorEnv(num_envs=2, n_symbols=2, n_days=20, autoreset_mode=mode)
    obs, _info = env.reset()
    assert obs["closes"].shape == (2, 2)


@pytest.mark.parametrize("mode", SUPPORTED_AUTORESET_MODES)
def test_final_observations_are_surfaced_only_under_the_listed_modes(mode):
    """Runs through the episode end, because that is the only step on which a mode could
    deliver a terminal payload. A lane is present in ``final_obs`` on that step alone."""
    n_days = 6
    env = SharpeArenaVectorEnv(
        num_envs=1, n_symbols=2, n_days=n_days, autoreset_mode=mode
    )
    obs, _info = env.reset()
    action = np.zeros((1, 2), dtype=np.float32)

    for bar in range(1, n_days + 1):
        previous = obs["closes"][0].copy()
        obs, _reward, _term, truncated, infos = env.step(action)
        ended = bar == n_days
        assert bool(truncated[0]) == ended
        if mode not in AUTORESET_MODES_WITH_FINAL_OBS:
            assert "final_obs" not in infos
        elif ended:
            # The terminal observation repeats the last in-window bar, while the
            # returned batch has already moved on to the new episode.
            assert np.array_equal(infos["final_obs"][0]["closes"], previous)
            assert infos["first"][0]
        else:
            assert infos["final_obs"][0] is None


# -- C02 action dtype boundary, stated in docs/integrations/contracts.md ---------


EXACT = np.array([0.123456789123, 0.234567891234], dtype=np.float64)


def _scalar_outcome(action) -> tuple[float, float]:
    env = SharpeArenaEnv(n_symbols=2, n_days=20, seed=1)
    env.reset()
    _obs, reward, _term, _trunc, info = env.step(action)
    return reward, info["nav"]


def _vector_outcome(action) -> tuple[float, float]:
    env = SharpeArenaVectorEnv(seeds=[1], n_symbols=2, n_days=20, autoreset_mode="disabled")
    env.reset()
    _obs, rewards, _term, _trunc, infos = env.step(action[None, :])
    return float(rewards[0]), float(infos["nav"][0])


def test_the_stated_dtypes_are_the_ones_the_environments_advertise():
    scalar = SharpeArenaEnv(n_symbols=2, n_days=20, seed=1)
    vector = SharpeArenaVectorEnv(num_envs=1, n_symbols=2, n_days=20)

    assert scalar.action_space.dtype == np.dtype(ACTION_SPACE_DTYPE)
    assert vector.single_action_space.dtype == np.dtype(ACTION_SPACE_DTYPE)
    assert ENGINE_ACTION_DTYPE == "float64"


@pytest.mark.parametrize("outcome", [_scalar_outcome, _vector_outcome])
def test_a_float32_action_and_its_widening_are_the_same_engine_input(outcome):
    narrow = EXACT.astype(np.float32)
    assert outcome(narrow) == outcome(narrow.astype(np.float64))


@pytest.mark.parametrize("outcome", [_scalar_outcome, _vector_outcome])
def test_a_float64_action_reaches_the_engine_unquantised(outcome):
    """The rule keeps checkpoint replay exact: `step` must not round to the advertised
    dtype. A cast anywhere on the way in would make these two outcomes equal."""
    assert not is_float32_exact(EXACT)
    assert outcome(EXACT) != outcome(EXACT.astype(np.float32))


@pytest.mark.parametrize(
    "action,exact",
    [
        pytest.param(np.array([0.1], dtype=np.float32), True, id="float32 value"),
        pytest.param([0.25, -0.5], True, id="dyadic float64"),
        pytest.param([0.1], False, id="decimal float64"),
        pytest.param([1e300], False, id="beyond the float32 range"),
        pytest.param([float("nan")], False, id="not a number"),
    ],
)
def test_float32_exactness_is_a_bit_for_bit_round_trip(action, exact):
    assert is_float32_exact(action) is exact
