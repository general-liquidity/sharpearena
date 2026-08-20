"""Tests for the behaviorally-biased counterparty policies.

These policies exist to be predictably wrong, so the tests check the bias is actually
present and correctly signed, not that the policy performs well. They also check the two
non-negotiables for a published, replayable package: the ``Policy`` protocol is honored
(``obs -> float32 weight vector`` of the right shape), and the output is reproducible from
a fixed seed.
"""

import importlib

import pytest

try:
    import numpy as np

    from sharpearena.baselines import (
        BASELINE_POLICIES,
        BEHAVIORAL_POLICIES,
        DispositionEffectPolicy,
        OverconfidentPolicy,
    )

    _HAVE_BINDING = importlib.util.find_spec("sharpearena.sharpearena_py") is not None
except Exception:  # pragma: no cover - exercised only without the binding/numpy
    _HAVE_BINDING = False


requires_binding = pytest.mark.skipif(
    not _HAVE_BINDING, reason="native sharpearena binding not built"
)


# -- protocol conformance ---------------------------------------------------


@requires_binding
def test_behavioral_registry_lists_both_policies():
    names = [name for name, _ in BEHAVIORAL_POLICIES]
    assert names == ["disposition_effect", "overconfident"]
    for name, factory in BEHAVIORAL_POLICIES:
        assert factory().name == name


@requires_binding
def test_behavioral_policies_are_excluded_from_the_ranked_baselines():
    # The canonical table's n_trials deflation footprint must not move when a
    # counterparty is added.
    ranked = {name for name, _ in BASELINE_POLICIES}
    assert ranked.isdisjoint({name for name, _ in BEHAVIORAL_POLICIES})
    assert len(BASELINE_POLICIES) == 6


@requires_binding
@pytest.mark.parametrize("n", [1, 3, 7])
def test_behavioral_policies_conform_to_the_policy_protocol(n):
    obs = {"closes": np.linspace(100.0, 110.0, n)}
    for _name, factory in BEHAVIORAL_POLICIES:
        policy = factory()
        for step in range(6):
            action = policy({"closes": obs["closes"] * (1.0 + 0.01 * step)})
            assert isinstance(action, np.ndarray)
            assert action.dtype == np.float32
            assert action.shape == (n,)
            assert np.all(np.isfinite(action))
            assert float(np.abs(action).sum()) <= 1.0 + 1e-6


# -- determinism ------------------------------------------------------------


def _replay(factory, closes_path) -> list[list[float]]:
    policy = factory()
    return [policy({"closes": c}).tolist() for c in closes_path]


@requires_binding
@pytest.mark.parametrize("name", ["disposition_effect", "overconfident"])
def test_behavioral_policies_are_deterministic(name):
    factory = dict(BEHAVIORAL_POLICIES)[name]
    rng = np.random.default_rng(21)
    path = [100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.02, size=4)) for _ in range(25)]
    assert _replay(factory, path) == _replay(factory, path)


@requires_binding
def test_overconfident_seed_controls_the_churn():
    rng = np.random.default_rng(4)
    path = [100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, size=3)) for _ in range(12)]
    a = _replay(lambda: OverconfidentPolicy(seed=0), path)
    b = _replay(lambda: OverconfidentPolicy(seed=0), path)
    c = _replay(lambda: OverconfidentPolicy(seed=1), path)
    assert a == b
    assert a != c


# -- the biases themselves --------------------------------------------------


@requires_binding
def test_disposition_effect_warms_up_equal_weight_long():
    pol = DispositionEffectPolicy()
    first = pol({"closes": np.array([100.0, 100.0, 100.0, 100.0])})
    assert np.allclose(first, 0.25)


@requires_binding
def test_disposition_effect_trims_winners_and_rides_losers():
    pol = DispositionEffectPolicy(gain_threshold=0.01, loss_threshold=0.01)
    start = pol({"closes": np.array([100.0, 100.0])})
    # Symbol 0 gains 5%, symbol 1 loses 5%.
    after = pol({"closes": np.array([105.0, 95.0])})
    assert after[0] < start[0], "the winner must be cut"
    assert after[1] > start[1], "the loser must be ridden, not cut"


@requires_binding
def test_disposition_effect_keeps_the_stale_reference_on_losers():
    # The loser's entry reference is deliberately not re-anchored, so it keeps being read
    # as a loser and keeps being topped up. That persistence is the bias.
    pol = DispositionEffectPolicy(loss_threshold=0.01, add_fraction=0.25)
    pol({"closes": np.array([100.0, 100.0])})
    w1 = pol({"closes": np.array([100.0, 95.0])})
    w2 = pol({"closes": np.array([100.0, 94.0])})
    assert w2[1] > w1[1]


@requires_binding
def test_disposition_effect_respects_gross_exposure():
    pol = DispositionEffectPolicy(max_gross=1.0, add_fraction=0.9, loss_threshold=0.0)
    closes = np.array([100.0, 100.0, 100.0])
    pol({"closes": closes})
    for step in range(1, 12):
        w = pol({"closes": closes * (1.0 - 0.02 * step)})
        assert float(np.abs(w).sum()) <= 1.0 + 1e-6


@requires_binding
def test_overconfident_warms_up_flat_then_over_sizes_the_short_signal():
    pol = OverconfidentPolicy(overconfidence=4.0, churn=0.0, seed=0)
    assert np.allclose(pol({"closes": np.array([100.0, 100.0])}), 0.0)
    w = pol({"closes": np.array([101.0, 99.0])})
    # A 1% move, sized as if a 3-bar estimate justified 4x conviction.
    assert w[0] > 0.03
    assert w[1] < -0.03


@requires_binding
def test_overconfident_churns_more_than_a_rational_baseline():
    # Over-trading is the point: with a flat price path a rational policy would sit still,
    # while this one keeps rewriting its target weights for no informational reason.
    closes = np.array([100.0, 100.0, 100.0])
    pol = OverconfidentPolicy(seed=3)
    weights = [pol({"closes": closes}) for _ in range(20)]
    turnover = sum(
        float(np.abs(b - a).sum()) for a, b in zip(weights[1:], weights[2:])
    )
    assert turnover > 0.5
