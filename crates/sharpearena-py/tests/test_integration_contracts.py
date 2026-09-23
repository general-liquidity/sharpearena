"""The C06/C07 structures refuse the records that would misreport a run.

Run from the crate dir after ``python -m maturin develop``::

    python -m pytest crates/sharpearena-py/tests/test_integration_contracts.py -q
"""

import numpy as np
import pytest

from sharpearena.integrations.contracts import (
    AUTORESET_MODES_WITH_FINAL_OBS,
    ENGINE_EXACTNESS,
    SUPPORTED_AUTORESET_MODES,
    AttemptCounts,
    ContractViolation,
    ExactnessDomain,
    TrustClass,
)


def _counts(**overrides) -> dict:
    base = dict(expected=4, attempted=4, completed=4, refused=0, failed=0, retried=0)
    base.update(overrides)
    return base


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param(dict(completed=3), id="an attempt reached no disposition"),
        pytest.param(
            dict(expected=5, attempted=5, completed=5, refused=1),
            id="dispositions exceed attempts",
        ),
        pytest.param(dict(expected=2, attempted=4, completed=4), id="completed exceeds planned"),
        pytest.param(dict(retried=9), id="more retries than attempts"),
        pytest.param(dict(refused=-1, completed=5), id="a negative count"),
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
    counts = AttemptCounts(expected=4, attempted=4, completed=2, refused=1, failed=1, retried=0)
    assert not counts.is_complete
    assert "1 refused" in counts.shortfall()
    assert "1 failed" in counts.shortfall()
    assert counts.as_record()["complete"] is False


def test_a_field_with_no_declared_rule_has_no_tolerance():
    domain = ExactnessDomain(exact=("reward",), tolerances={"nav": 1e-9}, rationale="test")
    assert domain.is_exact("reward")
    assert domain.tolerance_for("nav") == pytest.approx(1e-9)
    with pytest.raises(ContractViolation, match="no declared comparison rule"):
        domain.tolerance_for("turnover")
    with pytest.raises(ContractViolation, match="declared exact"):
        domain.tolerance_for("reward")


def test_a_field_cannot_be_both_exact_and_tolerance_bounded():
    with pytest.raises(ContractViolation):
        ExactnessDomain(exact=("nav",), tolerances={"nav": 1e-9}, rationale="test")


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
    from sharpearena.vector import SharpeArenaVectorEnv

    env = SharpeArenaVectorEnv(num_envs=2, n_symbols=2, n_days=20, autoreset_mode=mode)
    obs, _info = env.reset()
    assert obs["closes"].shape == (2, 2)


def test_final_observations_are_surfaced_only_under_the_listed_modes():
    from sharpearena.vector import SharpeArenaVectorEnv

    for mode in SUPPORTED_AUTORESET_MODES:
        env = SharpeArenaVectorEnv(num_envs=1, n_symbols=2, n_days=20, autoreset_mode=mode)
        env.reset()
        _obs, _reward, _term, _trunc, infos = env.step(np.zeros((1, 2), dtype=np.float32))
        assert ("final_obs" in infos) == (mode in AUTORESET_MODES_WITH_FINAL_OBS)
