"""Persistence and the evaluation freeze of the causal normalizers.

``tests/test_conformance.py`` holds the training-mode causality of the scalar wrappers.
This file covers what a learner needs on top of it: statistics that survive a checkpoint,
a ``training=False`` mode in which no held-out bar moves them, and the direct coverage of
``VectorCausalNormalizeObservation`` that INT-03 still lacked.
"""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest
from gymnasium import spaces
from gymnasium.vector import VectorEnv

from sharpearena import SharpeArenaEnv
from sharpearena.vector import SharpeArenaVectorEnv
from sharpearena.wrappers import (
    CausalNormalizeObservation,
    CausalNormalizeReward,
    NormalizerNotFittedError,
    NormalizerStateError,
)
from sharpearena.wrappers_vector import VectorCausalNormalizeObservation

_EPS = 1e-8
N_DAYS = 12
WEIGHTS = np.array([0.4, -0.2], dtype=np.float32)


def _env(seed: int = 5, n_days: int = N_DAYS) -> SharpeArenaEnv:
    return SharpeArenaEnv(n_symbols=2, n_days=n_days, seed=seed)


def _vector(mode: str = "disabled", seeds=(3, 9), n_days: int = N_DAYS) -> SharpeArenaVectorEnv:
    return SharpeArenaVectorEnv(
        seeds=list(seeds), n_symbols=2, n_days=n_days, autoreset_mode=mode
    )


def _batch(env) -> np.ndarray:
    return np.tile(WEIGHTS, (env.num_envs, 1))


def _json_round_trip(state: dict) -> dict:
    return json.loads(json.dumps(state, allow_nan=False))


def _closes(wrapper, steps: int) -> list[np.ndarray]:
    """Emitted ``closes`` for ``steps`` bars from a fresh reset, stopping at an episode end."""
    obs, _info = wrapper.reset()
    out = [obs["closes"].copy()]
    for _ in range(steps):
        obs, _reward, terminated, truncated, _info = wrapper.step(WEIGHTS)
        out.append(obs["closes"].copy())
        if terminated or truncated:
            break
    return out


def _rewards(wrapper, steps: int) -> list[float]:
    out = []
    for _ in range(steps):
        _obs, reward, terminated, truncated, _info = wrapper.step(WEIGHTS)
        out.append(reward)
        if terminated or truncated:
            break
    return out


def _fitted_observation_state(seed: int = 5) -> dict:
    wrapper = CausalNormalizeObservation(_env(seed))
    _closes(wrapper, N_DAYS)
    return wrapper.state_dict()


def _fitted_reward_state(seed: int = 5) -> dict:
    wrapper = CausalNormalizeReward(_env(seed))
    wrapper.reset()
    _rewards(wrapper, N_DAYS)
    return wrapper.state_dict()


def _fitted_vector_state() -> dict:
    wrapper = VectorCausalNormalizeObservation(_vector())
    wrapper.reset()
    for _ in range(4):
        wrapper.step(_batch(wrapper))
    return wrapper.state_dict()


# -- round trips ---------------------------------------------------------------


def test_observation_state_resumes_bit_identically_through_json():
    control = CausalNormalizeObservation(_env())
    expected = _closes(control, 8)

    first = CausalNormalizeObservation(_env())
    _closes(first, 3)
    state = _json_round_trip(first.state_dict())
    resumed = CausalNormalizeObservation(first.env)
    resumed.load_state_dict(state)

    tail = [resumed.step(WEIGHTS)[0]["closes"].copy() for _ in range(5)]
    for got, want in zip(tail, expected[4:]):
        assert np.array_equal(got, want)


def test_reward_state_resumes_bit_identically_through_json():
    control = CausalNormalizeReward(_env())
    control.reset()
    expected = _rewards(control, 8)

    first = CausalNormalizeReward(_env())
    first.reset()
    _rewards(first, 3)
    resumed = CausalNormalizeReward(first.env)
    resumed.load_state_dict(_json_round_trip(first.state_dict()))

    assert _rewards(resumed, 5) == expected[3:]


@pytest.mark.parametrize("mode", ["next_step", "same_step"])
def test_vector_state_resumes_bit_identically_across_autoreset(mode):
    """The resumed tail crosses the lane reset, so the cleared lane has to match too."""
    control = VectorCausalNormalizeObservation(_vector(mode))
    control.reset()
    expected = [control.step(_batch(control))[0]["closes"].copy() for _ in range(N_DAYS + 4)]

    first = VectorCausalNormalizeObservation(_vector(mode))
    first.reset()
    for _ in range(3):
        first.step(_batch(first))
    resumed = VectorCausalNormalizeObservation(first.env)
    resumed.load_state_dict(_json_round_trip(first.state_dict()))

    tail = [resumed.step(_batch(resumed))[0]["closes"].copy() for _ in range(N_DAYS + 1)]
    for got, want in zip(tail, expected[3:]):
        assert np.array_equal(got, want)


@pytest.mark.parametrize(
    "make,fit",
    [
        pytest.param(
            lambda: CausalNormalizeObservation(_env()),
            lambda w: _closes(w, 3),
            id="observation",
        ),
        pytest.param(
            lambda: CausalNormalizeReward(_env()),
            lambda w: (w.reset(), _rewards(w, 3)),
            id="reward",
        ),
        pytest.param(
            lambda: VectorCausalNormalizeObservation(_vector()),
            lambda w: (w.reset(), [w.step(_batch(w)) for _ in range(3)]),
            id="vector",
        ),
    ],
)
def test_a_taken_state_is_a_snapshot_not_a_view(make, fit):
    """Further training must not rewrite a checkpoint, nor a loaded state its source."""
    wrapper = make()
    fit(wrapper)
    state = wrapper.state_dict()
    pristine = copy.deepcopy(state)
    fit(wrapper)
    assert state == pristine

    loaded = make()
    loaded.load_state_dict(state)
    fit(loaded)
    assert state == pristine


def test_a_loaded_array_is_owned_by_the_wrapper():
    state = _fitted_observation_state()
    pristine = copy.deepcopy(state)
    mean = np.asarray(state["features"]["closes"]["mean"])
    state["features"]["closes"]["mean"] = mean

    wrapper = CausalNormalizeObservation(_env())
    wrapper.load_state_dict(state)
    mean[0] = 0.0

    assert wrapper.state_dict() == pristine


# -- the freeze ----------------------------------------------------------------


def test_frozen_observation_is_the_stored_scale_applied_to_the_raw_bar():
    state = _fitted_observation_state(seed=5)
    wrapper = CausalNormalizeObservation(_env(seed=7), training=False)
    wrapper.load_state_dict(state)

    emitted = _closes(wrapper, 5)

    raw = _closes(_env(seed=7), 5)
    feature = state["features"]["closes"]
    mean = np.asarray(feature["mean"])
    std = np.sqrt(np.asarray(feature["m2"]) / feature["count"])
    for got, bar in zip(emitted, raw):
        assert np.array_equal(got, (bar - mean) / (std + _EPS))
    assert wrapper.state_dict() == state


def test_frozen_observation_is_prefix_stable_across_an_episode_boundary():
    state = _fitted_observation_state()

    def rollout(extra_steps: int) -> list[np.ndarray]:
        wrapper = CausalNormalizeObservation(_env(seed=7), training=False)
        wrapper.load_state_dict(state)
        out = _closes(wrapper, N_DAYS)
        out += _closes(wrapper, extra_steps)
        return out

    short, long = rollout(2), rollout(N_DAYS)
    assert len(long) > len(short)
    for t, (left, right) in enumerate(zip(short, long)):
        assert np.array_equal(left, right), f"bar {t} moved under the freeze"
    # A replayed episode has to replay its values, which a running update would not.
    assert np.array_equal(short[0], short[N_DAYS + 1])


def test_unfreezing_resumes_updates_from_the_frozen_statistics():
    wrapper = CausalNormalizeObservation(_env())
    _closes(wrapper, 4)
    state = wrapper.state_dict()

    wrapper.training = False
    _closes(wrapper, 2)
    assert wrapper.state_dict() == state

    wrapper.training = True
    wrapper.step(WEIGHTS)
    assert wrapper.state_dict()["features"]["closes"]["count"] == (
        state["features"]["closes"]["count"] + 1
    )


def test_frozen_reward_uses_the_scale_of_the_finished_training_episode():
    """Training clears the accumulator at the episode boundary, so a checkpoint taken
    there would have nothing to freeze without the retired accumulator. The expected
    scale is an independent ``np.std`` of the raw rewards of that episode."""
    raw = _env()
    raw.reset()
    training_rewards = _rewards(raw, N_DAYS)
    assert len(training_rewards) == N_DAYS

    trained = CausalNormalizeReward(_env())
    trained.reset()
    _rewards(trained, N_DAYS)
    trained.reset()  # An empty live accumulator must not displace the finished one.
    state = trained.state_dict()
    assert state["live"]["count"] == 0

    wrapper = CausalNormalizeReward(_env(seed=7), training=False)
    wrapper.load_state_dict(state)
    wrapper.reset()
    frozen = _rewards(wrapper, N_DAYS)
    wrapper.reset()
    frozen += _rewards(wrapper, N_DAYS)

    held_out = _env(seed=7)
    held_out.reset()
    expected = _rewards(held_out, N_DAYS)
    scale = float(np.std(training_rewards)) + _EPS
    assert frozen == pytest.approx([r / scale for r in expected * 2], rel=1e-9)
    assert wrapper.state_dict() == state


def test_frozen_reward_falls_back_to_the_live_episode_before_any_has_finished():
    raw = _env()
    raw.reset()
    raw_rewards = _rewards(raw, 5)

    wrapper = CausalNormalizeReward(_env())
    wrapper.reset()
    _rewards(wrapper, 4)
    assert wrapper.state_dict()["retired"]["count"] == 0

    wrapper.training = False
    scale = float(np.std(raw_rewards[:4])) + _EPS
    assert wrapper.step(WEIGHTS)[1] == pytest.approx(raw_rewards[4] / scale, rel=1e-9)


def test_retiring_an_episode_leaves_training_rewards_unscaled_at_the_next_start():
    wrapper = CausalNormalizeReward(_env())
    wrapper.reset()
    _rewards(wrapper, N_DAYS)

    wrapper.reset()
    raw = _env()
    raw.reset()
    # With no history in the live episode the first two rewards pass through. A retired
    # accumulator consulted in training would have scaled them.
    assert _rewards(wrapper, 2) == _rewards(raw, 2)


def test_loading_leaves_the_training_flag_alone():
    wrapper = CausalNormalizeObservation(_env())
    wrapper.load_state_dict(_fitted_observation_state())
    assert wrapper.training is True

    frozen = CausalNormalizeObservation(_env(), training=False)
    frozen.load_state_dict(_fitted_observation_state())
    assert frozen.training is False


@pytest.mark.parametrize(
    "make",
    [
        pytest.param(lambda: CausalNormalizeObservation(_env()), id="observation"),
        pytest.param(lambda: CausalNormalizeReward(_env()), id="reward"),
        pytest.param(lambda: VectorCausalNormalizeObservation(_vector()), id="vector"),
    ],
)
def test_training_must_be_a_real_boolean(make):
    with pytest.raises(TypeError, match="training must be a boolean"):
        make().training = "false"


# -- a frozen normalizer with no scale refuses ---------------------------------


def test_unfitted_frozen_observation_refuses_before_touching_the_engine():
    base = _env()
    wrapper = CausalNormalizeObservation(base, training=False)
    with pytest.raises(NormalizerNotFittedError, match="at least 2"):
        wrapper.reset()
    with pytest.raises(NormalizerNotFittedError, match="at least 2"):
        wrapper.step(WEIGHTS)
    base.reset()
    assert base.step(WEIGHTS)[4]["nav"] == pytest.approx(_first_step_nav())


def _first_step_nav() -> float:
    env = _env()
    env.reset()
    return env.step(WEIGHTS)[4]["nav"]


def test_one_sample_is_not_enough_to_freeze_an_observation_normalizer():
    wrapper = CausalNormalizeObservation(_env())
    wrapper.reset()
    wrapper.training = False
    with pytest.raises(NormalizerNotFittedError, match="1 samples"):
        wrapper.step(WEIGHTS)


def test_unfitted_frozen_reward_refuses_rather_than_passing_rewards_through():
    wrapper = CausalNormalizeReward(_env(), training=False)
    with pytest.raises(NormalizerNotFittedError, match="2 samples"):
        wrapper.reset()
    with pytest.raises(NormalizerNotFittedError, match="2 samples"):
        wrapper.step(WEIGHTS)


# -- refused states ------------------------------------------------------------


def _set(path, value):
    def apply(state):
        node = state
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
        return state

    return apply


def _drop(path):
    def apply(state):
        node = state
        for key in path[:-1]:
            node = node[key]
        del node[path[-1]]
        return state

    return apply


@pytest.mark.parametrize(
    "corrupt,message",
    [
        pytest.param(_set(["format"], "v0"), "unknown state format", id="format"),
        pytest.param(_set(["wrapper"], "CausalNormalizeReward"), "state is for", id="wrapper"),
        pytest.param(_drop(["features", "cash"]), "missing .'cash'.", id="missing feature"),
        pytest.param(_set(["features", "volume"], {}), "unexpected .'volume'.", id="extra feature"),
        pytest.param(_set(["extra"], 1), "unexpected .'extra'.", id="extra key"),
        pytest.param(_set(["features", "positions", "count"], True), "must be an integer", id="bool count"),
        pytest.param(_set(["features", "positions", "count"], -1), "must not be negative", id="negative count"),
        pytest.param(_set(["features", "positions", "mean"], None), "must be numeric", id="absent mean"),
        pytest.param(_set(["features", "positions", "mean"], ["1", "2"]), "must be numeric", id="text mean"),
        pytest.param(_set(["features", "positions", "mean"], [1.0]), "shape must be", id="wrong shape"),
        pytest.param(_set(["features", "positions", "mean"], [1.0, float("nan")]), "must be finite", id="nan mean"),
        pytest.param(_set(["features", "positions", "m2"], [-1.0, 0.0]), "must not be negative", id="negative m2"),
    ],
)
def test_observation_state_that_does_not_fit_is_refused_whole(corrupt, message):
    """``positions`` is the last feature loaded, because Dict spaces sort their keys, so
    a piecemeal load would already have replaced the others by the time it refused."""
    good = _fitted_observation_state()
    wrapper = CausalNormalizeObservation(_env())
    wrapper.load_state_dict(good)

    other = _fitted_observation_state(seed=7)
    with pytest.raises(NormalizerStateError, match=message):
        wrapper.load_state_dict(corrupt(other))
    assert wrapper.state_dict() == good


def test_reward_state_refused_partway_keeps_both_accumulators():
    good = _fitted_reward_state()
    wrapper = CausalNormalizeReward(_env())
    wrapper.load_state_dict(good)

    other = _fitted_reward_state(seed=7)
    assert other["live"]["count"] == 0
    other["retired"]["m2"] = [float("inf")]
    other["live"] = other["retired"] | {"m2": [1.0]}
    with pytest.raises(NormalizerStateError, match="must be finite"):
        wrapper.load_state_dict(other)
    assert wrapper.state_dict() == good


@pytest.mark.parametrize(
    "corrupt,message",
    [
        pytest.param(_set(["num_envs"], 3), "holds 3 lanes", id="lane count"),
        pytest.param(_drop(["features", "positions", "retired"]), "missing .'retired'.", id="no retired"),
        pytest.param(_set(["features", "positions", "live", "count"], [4]), "shape must be", id="count shape"),
        pytest.param(_set(["features", "positions", "live", "count"], [4.5, 4.0]), "must hold integers", id="float count"),
        pytest.param(_set(["features", "positions", "live", "count"], [-1, 4]), "must not be negative", id="negative count"),
        pytest.param(_set(["features", "positions", "live", "mean"], [[1.0, 2.0]]), "shape must be", id="mean shape"),
        pytest.param(_set(["features", "positions", "retired", "m2"], [[-1.0, 0.0], [0.0, 0.0]]), "must not be negative", id="negative m2"),
    ],
)
def test_vector_state_that_does_not_fit_is_refused_whole(corrupt, message):
    good = _fitted_vector_state()
    wrapper = VectorCausalNormalizeObservation(_vector())
    wrapper.load_state_dict(good)

    other = _fitted_vector_state()
    other["features"]["closes"]["live"]["count"] = [99, 99]
    with pytest.raises(NormalizerStateError, match=message):
        wrapper.load_state_dict(corrupt(other))
    assert wrapper.state_dict() == good


# -- the vector wrapper directly -----------------------------------------------


def test_vector_emission_matches_a_by_hand_causal_zscore_per_lane():
    """Bars 0 and 1 of a lane have no scale and emit zeros. From bar 2 each lane is scored
    by the mean and population std of its own earlier bars, and no other lane's."""
    raw = _vector()
    obs, _infos = raw.reset()
    bars = [obs["closes"].copy()]
    wrapper = VectorCausalNormalizeObservation(_vector())
    emitted = [wrapper.reset()[0]["closes"].copy()]
    for _ in range(7):
        bars.append(raw.step(_batch(raw))[0]["closes"].copy())
        emitted.append(wrapper.step(_batch(wrapper))[0]["closes"].copy())

    assert not np.array_equal(bars[0][0], bars[0][1]), "lanes must differ to prove isolation"
    history = np.stack(bars)
    for t, got in enumerate(emitted):
        if t < 2:
            assert np.array_equal(got, np.zeros_like(got))
            continue
        past = history[:t]
        expected = (history[t] - past.mean(axis=0)) / (past.std(axis=0) + _EPS)
        np.testing.assert_allclose(got, expected, rtol=1e-9)


class _ScriptedVectorEnv(VectorEnv):
    """A boundary stub: lanes in the real env end together, so a staggered ``first`` needs
    a scripted source. It feeds the wrapper known bars and known reset flags, nothing else.
    """

    def __init__(self, bars: np.ndarray, first: np.ndarray) -> None:
        self.num_envs = bars.shape[1]
        self._bars, self._first = bars, first
        self._t = 0
        self.single_observation_space = spaces.Dict(
            {"x": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64)}
        )
        self.single_action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = self.single_observation_space
        self.action_space = self.single_action_space

    def _frame(self) -> dict:
        return {"x": self._bars[self._t]}

    def reset(self, *, seed=None, options=None):
        self._t = 0
        return self._frame(), {}

    def step(self, actions):
        self._t += 1
        zeros = np.zeros(self.num_envs)
        done = zeros.astype(bool)
        return self._frame(), zeros, done, done, {"first": self._first[self._t]}


def _staggered() -> VectorCausalNormalizeObservation:
    """Lane 0 restarts at bar 4. Lane 1 runs straight through."""
    bars = np.array(
        [[[1.0], [10.0]], [[2.0], [20.0]], [[4.0], [40.0]], [[8.0], [80.0]],
         [[100.0], [160.0]], [[200.0], [320.0]], [[400.0], [640.0]]]
    )
    first = np.zeros((7, 2), dtype=bool)
    first[4, 0] = True
    return VectorCausalNormalizeObservation(_ScriptedVectorEnv(bars, first))


def test_a_lane_reset_clears_only_that_lane_and_keeps_its_scale_as_retired():
    wrapper = _staggered()
    wrapper.reset()
    action = np.zeros((2, 1))
    for _ in range(3):
        wrapper.step(action)

    closed_out = wrapper.state_dict()["features"]["x"]["live"]
    emitted = wrapper.step(action)[0]["x"]

    assert emitted[0, 0] == 0.0, "the restarted lane has no history"
    prior = np.array([10.0, 20.0, 40.0, 80.0])
    assert emitted[1, 0] == pytest.approx((160.0 - prior.mean()) / (prior.std() + _EPS))

    feature = wrapper.state_dict()["features"]["x"]
    assert feature["live"]["count"] == [1, 5]
    assert feature["retired"]["count"] == [4, 0]
    assert feature["retired"]["mean"][0] == closed_out["mean"][0]
    assert feature["retired"]["m2"][0] == closed_out["m2"][0]


def test_a_reset_with_too_little_history_keeps_the_retired_scale():
    wrapper = _staggered()
    wrapper.reset()
    action = np.zeros((2, 1))
    for _ in range(3):
        wrapper.step(action)
    wrapper.reset()
    retired = wrapper.state_dict()["features"]["x"]["retired"]
    assert retired["count"] == [4, 4]

    wrapper.reset()

    assert wrapper.state_dict()["features"]["x"]["retired"] == retired


def test_retired_lane_statistics_never_reach_a_training_emission():
    wrapper = _staggered()
    wrapper.reset()
    action = np.zeros((2, 1))
    for _ in range(4):
        wrapper.step(action)
    # Lane 0 has one bar since its restart, so it still has no scale of its own.
    assert wrapper.step(action)[0]["x"][0, 0] == 0.0


def test_frozen_lanes_use_retired_statistics_where_they_exist_and_live_otherwise():
    wrapper = _staggered()
    wrapper.reset()
    action = np.zeros((2, 1))
    for _ in range(5):
        wrapper.step(action)
    state = wrapper.state_dict()["features"]["x"]

    wrapper.training = False
    emitted = wrapper.step(action)[0]["x"]

    lane0 = np.array([1.0, 2.0, 4.0, 8.0])
    lane1 = np.array([10.0, 20.0, 40.0, 80.0, 160.0, 320.0])
    assert emitted[0, 0] == pytest.approx((400.0 - lane0.mean()) / (lane0.std() + _EPS))
    assert emitted[1, 0] == pytest.approx((640.0 - lane1.mean()) / (lane1.std() + _EPS))
    assert wrapper.state_dict()["features"]["x"] == state


@pytest.mark.parametrize("mode", ["next_step", "same_step"])
def test_frozen_vector_ignores_autoreset_and_is_prefix_stable(mode):
    state = _fitted_vector_state()

    def rollout(steps: int) -> list[np.ndarray]:
        wrapper = VectorCausalNormalizeObservation(_vector(mode), training=False)
        wrapper.load_state_dict(state)
        out = [wrapper.reset()[0]["closes"].copy()]
        for _ in range(steps):
            obs, _rewards, _terminated, _truncated, infos = wrapper.step(_batch(wrapper))
            out.append(obs["closes"].copy())
            seen_first[0] |= bool(infos["first"].any())
        assert wrapper.state_dict() == state
        return out

    seen_first = [False]
    short, long = rollout(N_DAYS + 2), rollout(N_DAYS + 6)
    assert seen_first[0], "the rollout must cross an autoreset for this to mean anything"
    for t, (left, right) in enumerate(zip(short, long)):
        assert np.array_equal(left, right), f"bar {t} moved under the freeze"
    assert not np.array_equal(short[0], np.zeros_like(short[0])), "a frozen t0 has a scale"


def test_unfitted_frozen_vector_names_the_lanes_and_leaves_the_engine_alone():
    wrapper = _staggered()
    wrapper.reset()
    wrapper.training = False
    with pytest.raises(NormalizerNotFittedError, match=r"'x' in lanes \[0, 1\]"):
        wrapper.step(np.zeros((2, 1)))
    assert wrapper.state_dict()["features"]["x"]["live"]["count"] == [1, 1]
    assert wrapper.env._t == 0
