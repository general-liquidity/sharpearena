"""Tests for the reward-misspecification negative-control track.

When the native binding and numpy are importable these exercise a real (small) proxy sweep
against the SharpeBench kernel. They do not validate a training intervention. The module
imports ``score_run`` at top, so the whole file is binding-gated like ``test_baselines``.
"""

import importlib

import pytest

try:
    import numpy as np

    from sharpearena.reward_misspecification import (
        MISSPECIFIED_PROXY_POLICIES,
        MISSPECIFIED_REWARDS,
        MomentumChasePolicy,
        RecencyChasePolicy,
        demonstrate_punishment,
        indicator_shaped,
        misspecification_gap,
        raw_pnl_unpenalized,
        recency_biased,
        win_rate,
    )
    from sharpearena.gym import SharpeArenaEnv

    _HAVE_BINDING = importlib.util.find_spec("sharpearena.sharpearena_py") is not None
except Exception:  # pragma: no cover - exercised only without the binding/numpy
    _HAVE_BINDING = False


requires_binding = pytest.mark.skipif(
    not _HAVE_BINDING, reason="native sharpearena binding not built"
)


def _make_env(seed: int):
    return SharpeArenaEnv(n_symbols=3, n_days=60, seed=seed)


def _hand_state() -> dict:
    return {
        "returns": [0.01, -0.02, 0.03, -0.005, 0.012, -0.001, 0.004],
        "events": [
            {"event": "target_weights", "weights": [0.3, 0.3, 0.3]} for _ in range(7)
        ],
    }


# -- rewards: bounded + computable from state -------------------------------


@requires_binding
def test_misspecified_rewards_are_bounded_and_computable():
    state = _hand_state()
    for name, fn in MISSPECIFIED_REWARDS.items():
        r = fn(state=state)
        assert isinstance(r, float)
        assert np.isfinite(r)
        assert -1.0 <= r <= 1.0, name
    # win_rate / indicator_shaped live in [0, 1].
    assert 0.0 <= win_rate(state=state) <= 1.0
    assert 0.0 <= indicator_shaped(state=state) <= 1.0


@requires_binding
def test_rewards_handle_empty_state():
    for fn in MISSPECIFIED_REWARDS.values():
        assert fn(state={}) == 0.0
        assert fn(state=None) == 0.0


@requires_binding
def test_raw_pnl_rewards_supplied_return_unpenalized():
    # No risk/cost term: a bigger summed return strictly scores higher.
    low = raw_pnl_unpenalized(state={"returns": [0.001, 0.001]})
    high = raw_pnl_unpenalized(state={"returns": [0.05, 0.05]})
    assert high > low


@requires_binding
def test_win_rate_ignores_magnitude():
    # Many tiny wins + one huge loss still scores high (the blow-up reward).
    blowup = win_rate(state={"returns": [0.001, 0.001, 0.001, -0.5]})
    assert blowup == pytest.approx(0.75)


# -- proxy policies produce valid actions -----------------------------------


@requires_binding
def test_proxy_policies_produce_valid_actions():
    obs = {"closes": np.array([100.0, 101.0, 99.0])}
    for factory in MISSPECIFIED_PROXY_POLICIES.values():
        policy = factory()
        first = policy(obs)
        second = policy({"closes": np.array([101.0, 100.0, 100.0])})
        for action in (first, second):
            assert action.shape == (3,)
            assert np.all(np.isfinite(action))
            assert np.all(np.abs(action) <= 1.0 + 1e-6)


# -- the falsifiable wedge --------------------------------------------------


@requires_binding
def test_demo_fixture_reports_proxy_diagnostics():
    table = demonstrate_punishment(_make_env, range(6), max_steps=128)
    assert set(table) == set(MISSPECIFIED_REWARDS)
    for row in table.values():
        assert set(row) == {"deflated_sharpe", "passed_k", "mean_return"}
        assert np.isfinite(row["deflated_sharpe"])
        assert 0.0 <= row["passed_k"] <= 1.0

    # Pin this synthetic fixture only. These two summary fields do not implement
    # full ranking eligibility or guarantee punishment on another seed population.
    for name, row in table.items():
        assert row["deflated_sharpe"] < 0.95 or row["passed_k"] < 1.0, name

    # This fixture has no training phase or in/out-of-sample split.
    best = max(table.values(), key=lambda r: r["mean_return"])
    assert best["mean_return"] > 0.0
    assert best["deflated_sharpe"] <= 0.5


@requires_binding
def test_misspecification_gap_reports_clean_vs_flawed():
    g = misspecification_gap(
        _make_env, range(6), flawed_reward="raw_pnl_unpenalized", max_steps=128
    )
    assert g["clean_reward"] == "differential_sharpe"
    assert g["flawed_reward"] == "raw_pnl_unpenalized"
    assert g["proxy_is_stand_in"] is True
    assert np.isfinite(g["gap_deflated_sharpe"])
    assert np.isfinite(g["gap_mean_return"])
    for side in ("clean", "flawed"):
        assert set(g[side]) == {"deflated_sharpe", "passed_k", "mean_return"}


@requires_binding
def test_misspecification_gap_rejects_unknown_reward():
    with pytest.raises(ValueError):
        misspecification_gap(_make_env, range(2), flawed_reward="not_a_reward")


# -- the critical invariant: never registered into production ----------------


@requires_binding
def test_negative_controls_are_not_in_production_registry():
    from sharpearena import rewards as production_rewards

    for name in (
        "raw_pnl_unpenalized",
        "win_rate",
        "indicator_shaped",
        "recency_biased",
    ):
        assert name not in production_rewards.REWARD_SCHEMES


@requires_binding
def test_indicator_proxy_uses_three_realized_portfolio_returns_not_last_price_move():
    momentum = MomentumChasePolicy()
    recency = RecencyChasePolicy()
    realized = [-0.2, 0.2, -0.1, 0.0]
    events = []
    for close, reward in zip([100.0, 80.0, 64.0, 70.4], realized):
        obs = {"closes": np.array([close])}
        action = momentum(obs)
        last_move = recency(obs)
        events.append({"event": "target_weights", "weights": action.tolist()})
        momentum.observe_return(reward)
    assert action.tolist() == [-1.0]
    assert last_move.tolist() == [1.0]
    assert indicator_shaped(state={"returns": realized, "events": events}) == 1.0
    events[-1]["weights"] = [1.0]
    assert indicator_shaped(state={"returns": realized, "events": events}) == 0.0


@requires_binding
@pytest.mark.parametrize(
    "history, direction",
    [
        ([-0.3, 0.05, 0.05], -1.0),
        ([0.3, -0.05, -0.05], 1.0),
        ([0.0, 0.0, 0.0], 0.0),
    ],
)
def test_indicator_proxy_obeys_its_window_and_zero_signal(history, direction):
    policy = MomentumChasePolicy(window=3)
    for reward in [
        10.0,
        *history,
    ]:  # The older observation must fall out of the window.
        policy.observe_return(reward)
    assert policy({"closes": np.array([100.0, 1.0])}).tolist() == [direction] * 2


@requires_binding
def test_proxy_rollout_delivers_realized_feedback_and_closes_on_failure():
    from sharpearena.reward_misspecification import _rollout_returns

    observed = []

    class Policy:
        def __call__(self, obs):
            return np.array([0.0])

        def observe_return(self, reward):
            observed.append(reward)

    class Market:
        closed = False
        count = 0

        def reset(self):
            return {}, {}

        def step(self, action):
            self.count += 1
            if self.count == 2:
                raise RuntimeError("synthetic market failure")
            return {}, 0.125, False, False, {}

        def close(self):
            self.closed = True

    env = Market()
    with pytest.raises(RuntimeError, match="synthetic market failure"):
        _rollout_returns(env, Policy(), 3)
    assert observed == [0.125]
    assert env.closed


@requires_binding
def test_recency_observation_is_an_owned_snapshot():
    policy = RecencyChasePolicy()
    closes = np.array([100.0])
    policy({"closes": closes})
    closes[0] = 90.0
    assert policy({"closes": closes}).tolist() == [-1.0]


@requires_binding
def test_gap_labels_are_not_reported_as_optimization_evidence():
    result = misspecification_gap(
        _make_env, [0], flawed_reward="indicator_shaped", max_steps=4
    )
    assert result["comparison_kind"] == "heuristic_policy_comparison"
    assert result["optimization_performed"] is False
    assert result["clean_reward_role"] == "label_only"


@requires_binding
@pytest.mark.parametrize("window", [0, -1, True, 1.5])
def test_indicator_reward_and_policy_reject_the_same_invalid_windows(window):
    with pytest.raises(ValueError, match="positive integer"):
        MomentumChasePolicy(window=window)
    with pytest.raises(ValueError, match="positive integer"):
        indicator_shaped(state=_hand_state(), window=window)


@requires_binding
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_indicator_feedback_refuses_nonfinite_returns(value):
    with pytest.raises(ValueError, match="finite realized return"):
        MomentumChasePolicy().observe_return(value)
