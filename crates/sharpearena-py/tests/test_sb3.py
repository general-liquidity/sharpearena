"""The Stable-Baselines3 route, and specifically its autoreset mapping.

SB3's ``VecEnv`` is not Gymnasium's vector API, and the difference that matters is
*when* a finished lane resets and *which* observation the step that finished it returns.
Getting that backwards raises nothing and corrupts every return an SB3 agent computes,
so these tests pin it from three directions:

* :func:`test_autoreset_is_same_step_and_not_next_step` drives one SB3 env and one
  ``next_step`` Gymnasium vector env through the identical action sequence and asserts
  the exact one-step offset between the two streams. An inverted mapping (terminal
  observation returned as the step observation, reset observation filed as
  ``terminal_observation``) fails it on both assertions.
* :func:`test_timelimit_truncated_is_not_plain_truncated` pins the ``and not
  terminated`` term, including the both-flags-set step SB3's encoding cannot represent.
* the parity tests run the adapter against the native engine through
  ``integrations.parity``, reconstructing Gymnasium's flags from SB3's two fields, so a
  mapping error shows up as a mismatch against the engine itself rather than against
  another adapter.

Run from the crate dir after ``python -m maturin develop``::

    python -m pytest crates/sharpearena-py/tests/test_sb3.py -q
"""

import importlib.util
import inspect

import numpy as np
import pytest

import sharpearena  # noqa: F401  (the package must import without stable-baselines3)
import sharpearena.sb3_env as sb3_env
from sharpearena._seed_bands import EVAL_SEED_BASE
from sharpearena.integrations.parity import (
    CORE_FIXTURES,
    ParityFixture,
    check_adapter_parity,
    resolved_seeds,
)
from sharpearena.sb3_env import (
    SB3_AUTORESET_MODE,
    SB3ContractError,
    SharpeArenaSB3VecEnv,
    recover_gymnasium_flags,
    sb3_dones,
    sb3_reference_semantics,
    sb3_timelimit_truncated,
)
from sharpearena.vector import SharpeArenaVectorEnv

_HAS_SB3 = importlib.util.find_spec("stable_baselines3") is not None
needs_sb3 = pytest.mark.skipif(not _HAS_SB3, reason="stable-baselines3 not installed")

OBS_KEYS = ("closes", "positions", "cash")


def _lane(batch, index=0):
    return {key: np.asarray(batch[key][index]) for key in OBS_KEYS}


def _same_obs(left, right):
    return all(np.array_equal(np.asarray(left[k]), np.asarray(right[k])) for k in OBS_KEYS)


# ---------------------------------------------------------------------------
# Pure mapping functions: no stable-baselines3 needed
# ---------------------------------------------------------------------------


def test_dones_is_the_union_of_both_flags():
    terminated = np.array([False, True, False, True])
    truncated = np.array([False, False, True, True])
    assert sb3_dones(terminated, truncated).tolist() == [False, True, True, True]


def test_timelimit_truncated_is_not_plain_truncated():
    """``truncated and not terminated``, over all four combinations.

    The third row is the one that separates the correct mapping from the plausible
    wrong one: a step that both blew up and ran out of bars is **not** a timeout, and
    reporting it as one tells SB3 to bootstrap past an absorbing state.
    """
    terminated = np.array([False, True, True, False])
    truncated = np.array([False, False, True, True])
    assert sb3_timelimit_truncated(terminated, truncated).tolist() == [
        False,
        False,
        False,
        True,
    ]
    # And it is not simply `truncated`, which is what a careless mapping would emit.
    assert sb3_timelimit_truncated(terminated, truncated).tolist() != truncated.tolist()


@pytest.mark.parametrize(
    ("terminated", "truncated"),
    [(False, False), (True, False), (False, True)],
)
def test_sb3_encoding_round_trips_the_representable_cases(terminated, truncated):
    done = bool(terminated or truncated)
    timelimit = bool(sb3_timelimit_truncated(np.array([terminated]), np.array([truncated]))[0])
    assert recover_gymnasium_flags(done, timelimit) == (terminated, truncated)


def test_sb3_encoding_cannot_represent_both_flags_at_once():
    """The documented, unavoidable loss, asserted rather than left to be discovered.

    SB3 states that ``TimeLimit.truncated`` and ``terminated`` are mutually exclusive,
    so a step that is both comes back as termination alone. The bootstrapping decision
    survives (a terminated step is never bootstrapped past); the truncation bit does
    not, which is why the adapter also publishes the unreduced flags.
    """
    timelimit = bool(sb3_timelimit_truncated(np.array([True]), np.array([True]))[0])
    assert recover_gymnasium_flags(True, timelimit) == (True, False)


@needs_sb3
def test_reference_semantics_still_match_installed_sb3():
    """The three ``DummyVecEnv.step_wait`` lines this adapter reproduces.

    Restated in the module rather than imported, so this test is what couples the two.
    An SB3 release that changes the mapping fails here instead of silently changing what
    this route computes.
    """
    from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv

    source = inspect.getsource(DummyVecEnv.step_wait)
    for name, line in sb3_reference_semantics().items():
        assert line in source, f"{name}: SB3's DummyVecEnv no longer contains {line!r}"


# ---------------------------------------------------------------------------
# Optional-import contract (runs when stable-baselines3 is NOT installed)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(_HAS_SB3, reason="stable-baselines3 is installed")
def test_import_works_and_construction_raises_without_sb3():
    assert sb3_env._HAS_SB3 is False
    with pytest.raises(sb3_env.SB3Unavailable, match="stable-baselines3 is not installed"):
        SharpeArenaSB3VecEnv(2, n_symbols=3, n_days=24)


# ---------------------------------------------------------------------------
# Construction refuses the autoreset modes it cannot map
# ---------------------------------------------------------------------------


@needs_sb3
@pytest.mark.parametrize("mode", ["next_step", "disabled"])
def test_unmappable_autoreset_modes_are_refused(mode):
    with pytest.raises(SB3ContractError, match="autoreset_mode"):
        SharpeArenaSB3VecEnv(1, n_symbols=3, n_days=24, autoreset_mode=mode)


@needs_sb3
def test_underlying_env_runs_in_same_step_mode():
    env = SharpeArenaSB3VecEnv(1, n_symbols=3, n_days=24)
    assert env.vector_env._autoreset_mode == SB3_AUTORESET_MODE


# ---------------------------------------------------------------------------
# The autoreset and terminal-observation mapping
# ---------------------------------------------------------------------------


def _roll(env, actions, steps, sb3):
    """Collect ``steps`` transitions, as (obs, reward, terminated, truncated, info)."""
    out = []
    for _ in range(steps):
        if sb3:
            obs, rewards, dones, infos = env.step(actions)
            info = infos[0]
            terminated, truncated = recover_gymnasium_flags(
                dones[0], info["TimeLimit.truncated"]
            )
            out.append((_lane(obs), float(rewards[0]), terminated, truncated, info))
        else:
            obs, rewards, terminated, truncated, infos = env.step(actions)
            out.append(
                (_lane(obs), float(rewards[0]), bool(terminated[0]), bool(truncated[0]), infos)
            )
    return out


@needs_sb3
def test_autoreset_is_same_step_and_not_next_step():
    """One SB3 env and one ``next_step`` env on the same tape and the same actions.

    Under ``next_step`` the terminal bar is returned on the step it happens and the
    reset surfaces one step later. Under SB3's contract the reset surfaces on the same
    step and the terminal bar moves into ``terminal_observation``. So for a lane that
    finishes at index ``k`` the two streams must satisfy, exactly:

    * ``terminal_observation`` at ``k`` equals the ``next_step`` observation at ``k``;
    * the SB3 observation at ``k`` equals the ``next_step`` observation at ``k + 1``,
      which is that mode's recycling step;
    * and from there the SB3 stream stays one step ahead.

    Swap the two observations in the adapter and the first two assertions both fail.
    """
    kwargs = dict(seeds=[3], n_symbols=3, n_days=24)
    sb3 = SharpeArenaSB3VecEnv(**kwargs)
    reference = SharpeArenaVectorEnv(autoreset_mode="next_step", **kwargs)
    sb3.reset()
    reference.reset()

    actions = np.full((1, 3), 0.2, dtype=np.float32)
    horizon = 40
    sb3_steps = _roll(sb3, actions, horizon, sb3=True)
    ref_steps = _roll(reference, actions, horizon, sb3=False)

    ends = [i for i, step in enumerate(sb3_steps) if step[2] or step[3]]
    assert ends, "the lane never finished within the horizon; the fixture proves nothing"
    k = ends[0]

    assert (ref_steps[k][2], ref_steps[k][3]) == (sb3_steps[k][2], sb3_steps[k][3])
    assert _same_obs(sb3_steps[k][4]["terminal_observation"], ref_steps[k][0]), (
        "terminal_observation is not the finished lane's last bar"
    )
    assert not _same_obs(sb3_steps[k][0], ref_steps[k][0]), (
        "the SB3 step observation is the terminal bar; SB3 requires the reset observation"
    )
    assert _same_obs(sb3_steps[k][0], ref_steps[k + 1][0]), (
        "the SB3 step observation is not the new episode's first bar"
    )

    # The reference mode's step k+1 is its recycling transition: zero reward, both flags
    # clear. SB3 must never see one, so its stream stays one step ahead from here on.
    assert ref_steps[k + 1][1] == 0.0
    assert (ref_steps[k + 1][2], ref_steps[k + 1][3]) == (False, False)
    for offset in range(1, 6):
        assert _same_obs(sb3_steps[k + offset][0], ref_steps[k + offset + 1][0]), (
            f"streams re-synchronised at offset {offset}: a recycling transition leaked "
            "into the SB3 stream"
        )
        assert sb3_steps[k + offset][1] == ref_steps[k + offset + 1][1]


@needs_sb3
def test_terminal_observation_absent_while_the_lane_runs():
    env = SharpeArenaSB3VecEnv(1, n_symbols=3, n_days=24)
    env.reset()
    _obs, _rewards, dones, infos = env.step(np.zeros((1, 3), dtype=np.float32))
    assert not dones[0]
    assert "terminal_observation" not in infos[0]


@needs_sb3
def test_terminal_payloads_are_per_lane():
    """Two lanes on different tapes get their own terminal observation, not a shared one.

    The two lanes share a bar count, so they run out of bars together; what this pins
    is that the transposition into SB3's list-of-dicts keeps each lane's payload with
    that lane. A transposition that indexed the wrong axis would hand both lanes the
    same observation, and these tapes differ, so that fails here.
    """
    env = SharpeArenaSB3VecEnv(seeds=[3, 5], n_symbols=3, n_days=24)
    env.reset()
    actions = np.full((2, 3), 0.2, dtype=np.float32)
    ends = 0
    for _ in range(40):
        _obs, _rewards, dones, infos = env.step(actions)
        for index in range(2):
            assert ("terminal_observation" in infos[index]) == bool(dones[index])
            assert infos[index]["scenario_seed"] == env.scenario_seeds[index]
        if dones.all():
            ends += 1
            assert not _same_obs(
                infos[0]["terminal_observation"], infos[1]["terminal_observation"]
            )
    assert ends, "no lane finished within the horizon; the fixture proves nothing"


# ---------------------------------------------------------------------------
# Engine parity, through integrations.parity
# ---------------------------------------------------------------------------


class _ScalarSB3View:
    """A one-lane SB3 env re-presented in the scalar Gymnasium shape the checker drives.

    The flags are reconstructed with SB3's own documented inversion and the observation
    is taken from ``terminal_observation`` on a finished step, so what this view hands
    the checker is exactly what an SB3 agent can recover. A mapping error therefore
    shows up as a mismatch against the native engine.
    """

    def __init__(self, fixture: ParityFixture) -> None:
        # The vector constructor takes the resolved *scenario* seed per lane and one
        # global execution seed, while the fixture carries a single user seed. Both are
        # taken from `parity.resolved_seeds`, which restates the split, so the lane and
        # the engine reference wrap the same tape and the same execution stream.
        scenario_seed, exec_seed = resolved_seeds(fixture)
        self.venv = SharpeArenaSB3VecEnv(
            seeds=[scenario_seed],
            max_weight=fixture.max_weight,
            allow_short=fixture.allow_short,
            env_kwargs={"exec_seed": exec_seed},
            **fixture.scenario_kwargs(),
        )

    @property
    def symbols(self):
        return self.venv.symbols

    def reset(self):
        return _lane(self.venv.reset()), {}

    def step(self, action):
        obs, rewards, dones, infos = self.venv.step(
            np.asarray([action], dtype=np.float32)
        )
        info = infos[0]
        terminated, truncated = recover_gymnasium_flags(
            dones[0], info["TimeLimit.truncated"]
        )
        lane = info["terminal_observation"] if dones[0] else _lane(obs)
        return lane, float(rewards[0]), terminated, truncated, info


@needs_sb3
@pytest.mark.parametrize("fixture", CORE_FIXTURES, ids=lambda f: f.name)
def test_parity_with_the_native_engine(fixture):
    check_adapter_parity(
        fixture,
        _ScalarSB3View,
        source="sharpearena.sb3_env.SharpeArenaSB3VecEnv",
    ).require_parity()


# ---------------------------------------------------------------------------
# The rest of the VecEnv surface
# ---------------------------------------------------------------------------


@needs_sb3
def test_reset_returns_observation_only_and_files_info_separately():
    env = SharpeArenaSB3VecEnv(2, n_symbols=3, n_days=24)
    obs = env.reset()
    assert isinstance(obs, dict) and set(obs) == set(OBS_KEYS)
    assert len(env.reset_infos) == 2
    assert env.reset_infos[0]["scenario_seed"] == env.scenario_seeds[0]


@needs_sb3
def test_seed_warns_and_does_not_reseed_the_market():
    env = SharpeArenaSB3VecEnv(seeds=[7], n_symbols=3, n_days=24)
    before = _lane(env.reset())
    with pytest.warns(UserWarning, match="does not reseed the market"):
        assert env.seed(123) == [7]
    assert _same_obs(_lane(env.reset()), before)


@needs_sb3
def test_per_lane_attribute_write_is_refused():
    env = SharpeArenaSB3VecEnv(2, n_symbols=3, n_days=24)
    with pytest.raises(SB3ContractError, match="share one native engine"):
        env.set_attr("metadata", {}, indices=[0])


@needs_sb3
def test_step_wait_without_step_async_is_refused():
    env = SharpeArenaSB3VecEnv(1, n_symbols=3, n_days=24)
    env.reset()
    with pytest.raises(SB3ContractError, match="without a pending step_async"):
        env.step_wait()


# ---------------------------------------------------------------------------
# End to end: the route actually trains and evaluates
# ---------------------------------------------------------------------------


@needs_sb3
def test_learn_then_evaluate_on_a_tiny_budget():
    """PPO over the Dict observation, trained and then evaluated. CPU, seconds.

    The point is not the score, which is meaningless at this budget. It is that the
    whole SB3 path runs against this adapter: policy construction over a single-level
    ``Dict`` space via ``MultiInputPolicy``, rollout collection across the autoreset
    boundary, an update, and a separate evaluation pass on held-out lanes.
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.evaluation import evaluate_policy

    train = SharpeArenaSB3VecEnv(seeds=[0, 1], n_symbols=3, n_days=24)
    model = PPO(
        "MultiInputPolicy",
        train,
        n_steps=32,
        batch_size=16,
        n_epochs=1,
        policy_kwargs={"net_arch": [16]},
        device="cpu",
        verbose=0,
    )
    model.learn(total_timesteps=64)

    evaluation = SharpeArenaSB3VecEnv(
        seeds=[EVAL_SEED_BASE, EVAL_SEED_BASE + 1], n_symbols=3, n_days=24
    )
    mean_reward, std_reward = evaluate_policy(
        model, evaluation, n_eval_episodes=2, warn=False
    )
    assert np.isfinite(mean_reward) and np.isfinite(std_reward)
