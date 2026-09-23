"""RL-contract regressions for the declared Gymnasium semantics (ticket P17).

These cover the parts of the declared contract that the existing suite does not reach.
What is already covered stays where it is: ``test_gym.py`` holds reset/reseed and
same-seed determinism, ``test_equivalence.py`` holds scalar/vector trajectory equality
under bridged seeds, ``test_vector.py`` holds the per-mode autoreset behaviour,
``test_action_validation.py`` holds invalid actions, ``test_conformance.py`` holds the
space/wrapper/reward-scaling contract and ``test_checkpoint.py`` holds restore and
branch replay.

What is added here:

- the two episode endings come from different causes and carry different flags
  (`terminated` is a blow-up, `truncated` is the window running out), and the
  observation the truncating step hands a learner is the last in-window bar again,
  not a fresh one, while the step still earns a real reward;
- the scenario stream and the execution stream are separable: the execution seed moves
  fills without moving a single observed close;
- lanes hold independent state under per-lane actions, and lane order is respected;
- the autoreset mode the batch runs under is recorded in ``metadata`` as Gymnasium's
  own enum, for each of the three modes upstream defines;
- per-step reward, episode accounting and the reported evaluation metric are the same
  number chain;
- the adapter constructs, resets and steps with no optional dependency importable.

Gymnasium references: termination and truncation are distinct
(https://gymnasium.farama.org/introduction/basic_usage/); the three autoreset modes are
NEXT_STEP, SAME_STEP and DISABLED (https://farama.org/Vector-Autoreset-Mode).
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from sharpearena import SharpeArenaEnv
from sharpearena.metrics import RunMetrics
from sharpearena.vector import SharpeArenaVectorEnv

# One symbol held fully short under persistent large jumps. NAV crosses zero at bar 165
# of the 200-bar window, so the blow-up ending is reached well inside the horizon and is
# reproducible from the seed alone.
BLOWUP = dict(
    n_symbols=1,
    n_days=200,
    seed=2,
    distribution_mode="extreme",
    jump_burst_probability=0.9,
    jump_burst_persistence=0.9,
    jump_burst_size=0.5,
)


def _flat(env, weight: float) -> np.ndarray:
    return np.full(env.action_space.shape, weight, dtype=np.float32)


def _run_to_end(env, action, cap: int = 1000):
    """Step ``action`` to the episode boundary, returning the per-step records."""
    out = []
    for _ in range(cap):
        obs, reward, terminated, truncated, info = env.step(action)
        out.append((obs, float(reward), bool(terminated), bool(truncated), info))
        if terminated or truncated:
            return out
    pytest.fail("episode never reached a terminal or truncated step")


# -- terminated versus truncated --------------------------------------------


def test_insolvency_terminates_inside_the_horizon_without_truncating():
    env = SharpeArenaEnv(**BLOWUP)
    env.reset()
    steps = _run_to_end(env, _flat(env, -1.0))
    _obs, _reward, terminated, truncated, info = steps[-1]
    assert terminated, "NAV crossing zero is a task failure, so it must terminate"
    assert not truncated, "the window still had bars left, so nothing was cut short"
    assert info["nav"] <= 0.0
    assert len(steps) < BLOWUP["n_days"], "the blow-up must precede the window end"
    assert all(not t and not c for _o, _r, t, c, _i in steps[:-1])


@pytest.mark.parametrize("n_symbols,n_days,seed", [(2, 12, 4), (3, 40, 1), (1, 15, 3)])
def test_running_out_of_bars_truncates_and_repeats_the_last_bar(n_symbols, n_days, seed):
    """The terminal observation a learner sees on the truncating step is the final
    in-window bar again, because there is no further bar to advance to. The step is
    still a real accounting step, so bootstrapping past it is the correct treatment."""
    env = SharpeArenaEnv(n_symbols=n_symbols, n_days=n_days, seed=seed)
    env.reset()
    steps = _run_to_end(env, _flat(env, 0.2))

    obs, reward, terminated, truncated, info = steps[-1]
    assert truncated and not terminated
    assert info["nav"] > 0.0, "an exhausted window is not a blow-up"
    assert len(steps) == n_days, "the episode spans the window, one step per bar"
    assert np.array_equal(obs["closes"], steps[-2][0]["closes"])
    assert reward == pytest.approx(info["nav"] / steps[-2][4]["nav"] - 1.0, rel=1e-12)
    assert reward != 0.0, "the truncating step earns a return, it is not padding"


# -- scenario versus execution RNG ------------------------------------------


def _lane_rollout(scenario_seed: int, exec_seed: int, steps: int = 10):
    env = SharpeArenaVectorEnv(
        seeds=[scenario_seed],
        n_symbols=2,
        n_days=25,
        autoreset_mode="disabled",
        env_kwargs={"exec_seed": exec_seed},
    )
    obs, _infos = env.reset()
    closes = [obs["closes"][0].copy()]
    rewards = []
    action = np.full((1, 2), 0.3, dtype=np.float32)
    for _ in range(steps):
        obs, reward, _t, _c, _i = env.step(action)
        closes.append(obs["closes"][0].copy())
        rewards.append(float(reward[0]))
    return closes, rewards


def test_execution_seed_moves_fills_without_moving_the_price_path():
    """The two streams are independent: re-seeding execution perturbs what the agent
    gets filled at, and nothing about the point-in-time market it observes."""
    closes_a, rewards_a = _lane_rollout(5, 11)
    closes_b, rewards_b = _lane_rollout(5, 12)

    assert all(np.array_equal(a, b) for a, b in zip(closes_a, closes_b))
    assert rewards_a != rewards_b


def test_scenario_seed_moves_the_price_path():
    closes_a, _ = _lane_rollout(5, 11)
    closes_c, _ = _lane_rollout(6, 11)
    assert any(not np.array_equal(a, c) for a, c in zip(closes_a, closes_c))


# -- independent lane state --------------------------------------------------


def _solo_lane_rewards(seed: int, weights, steps: int):
    env = SharpeArenaVectorEnv(
        seeds=[seed], n_symbols=2, n_days=30, autoreset_mode="disabled"
    )
    env.reset()
    action = np.asarray([weights], dtype=np.float32)
    return [float(env.step(action)[1][0]) for _ in range(steps)]


def test_each_lane_matches_a_solo_env_under_per_lane_actions():
    """Lanes share no mutable state and each one is driven by its own row of the action
    batch. Uniform actions cannot see a lane/action misalignment; these differ per lane,
    and the seed order is permuted so a transposed dispatch fails too."""
    steps = 10
    weights = [[0.4, 0.1], [-0.3, 0.2]]
    solo = [_solo_lane_rewards(3, weights[0], steps), _solo_lane_rewards(9, weights[1], steps)]

    env = SharpeArenaVectorEnv(
        seeds=[3, 9], n_symbols=2, n_days=30, autoreset_mode="disabled"
    )
    env.reset()
    action = np.asarray(weights, dtype=np.float32)
    batched = [env.step(action)[1].copy() for _ in range(steps)]
    for lane in (0, 1):
        assert [float(r[lane]) for r in batched] == solo[lane]

    swapped = SharpeArenaVectorEnv(
        seeds=[9, 3], n_symbols=2, n_days=30, autoreset_mode="disabled"
    )
    swapped.reset()
    swapped_action = np.asarray(weights[::-1], dtype=np.float32)
    swapped_rewards = [swapped.step(swapped_action)[1].copy() for _ in range(steps)]
    for lane in (0, 1):
        assert [float(r[lane]) for r in swapped_rewards] == solo[1 - lane]


# -- autoreset mode is recorded ----------------------------------------------


@pytest.mark.parametrize(
    "label,attribute",
    [("next_step", "NEXT_STEP"), ("same_step", "SAME_STEP"), ("disabled", "DISABLED")],
)
def test_metadata_records_the_gymnasium_autoreset_mode(label, attribute):
    """A learner reads the mode off ``metadata`` rather than a private attribute, and
    upstream support is mode-dependent, so the recorded value has to be Gymnasium's own
    enum member for the mode the batch was built with."""
    autoreset = pytest.importorskip("gymnasium.vector").AutoresetMode
    env = SharpeArenaVectorEnv(
        seeds=[1], n_symbols=2, n_days=20, autoreset_mode=label
    )
    assert env.metadata["autoreset_mode"] is getattr(autoreset, attribute)


def test_only_the_three_upstream_modes_are_accepted():
    """Upstream defines three modes and this environment implements exactly those. The
    wire labels are the contract; the enum member names are not."""
    for label in ("next_step", "same_step", "disabled"):
        SharpeArenaVectorEnv(seeds=[1], n_symbols=2, n_days=20, autoreset_mode=label)
    for label in ("NEXT_STEP", "same-step", "next", ""):
        with pytest.raises(Exception):
            SharpeArenaVectorEnv(seeds=[1], n_symbols=2, n_days=20, autoreset_mode=label)


# -- reward, episode accounting and the reported metric ----------------------


def test_reward_is_the_nav_return_the_metrics_panel_reconstructs():
    """One number chain: the engine's per-step reward is the NAV return of that step, so
    feeding ``RunMetrics`` the rewards and feeding it the NAV series must agree."""
    env = SharpeArenaEnv(n_symbols=3, n_days=40, seed=12)
    env.reset()
    steps = _run_to_end(env, _flat(env, 0.3))

    from_reward, from_nav = RunMetrics(), RunMetrics()
    previous_nav = 1.0
    for _obs, reward, _t, _c, info in steps:
        nav = float(info["nav"])
        assert reward == pytest.approx(nav / previous_nav - 1.0, rel=1e-12)
        from_reward.record_step(reward=reward)
        from_nav.record_step(nav=nav)
        previous_nav = nav

    left, right = from_reward.to_dict(), from_nav.to_dict()
    for key in ("realized_return", "max_drawdown", "volatility", "sortino"):
        assert left[key] == pytest.approx(right[key], rel=1e-9, abs=1e-12)


def test_the_reported_eval_metric_scores_exactly_the_reward_series():
    """``evaluate_eval_set`` is the reported evaluation surface. Its numbers have to come
    from the same per-step rewards a learner optimizes, with nothing rescaled in between."""
    from sharpearena.baselines import EqualWeightLongPolicy
    from sharpearena.eval_seeds import EVAL_SEEDS, evaluate_eval_set
    from sharpearena.kernel_score import kernel_score_or_unavailable
    from sharpearena.sharpearena_py import score_run

    trials = len(EVAL_SEEDS)
    reported = evaluate_eval_set(n_symbols=2, n_days=30, n_trials=trials)

    for name, seed in EVAL_SEEDS.items():
        env = SharpeArenaEnv(n_symbols=2, n_days=30, seed=seed, mode="train")
        obs, _info = env.reset()
        policy = EqualWeightLongPolicy()
        rewards = []
        while True:
            obs, reward, terminated, truncated, _info = env.step(policy(obs))
            rewards.append(float(reward))
            if terminated or truncated:
                break

        assert reported[name]["mean_return"] == pytest.approx(
            float(np.mean(rewards)), rel=1e-12
        )
        composite = json.loads(score_run(rewards, trials))
        assert reported[name]["deflated_sharpe"] == kernel_score_or_unavailable(composite)


# -- usable without the optional dependencies --------------------------------

# Declared in pyproject as extras, plus the accelerator stacks a learner would bring.
# None of them may be needed to construct, reset or step the environment.
OPTIONAL_DEPENDENCIES = ("verifiers", "minari", "pettingzoo", "mcp", "torch", "jax")

_NO_OPTIONAL_DEPENDENCY_PROGRAM = textwrap.dedent(
    """
    import sys

    BLOCKED = {blocked!r}

    class Blocker:
        def find_module(self, name, path=None):
            return None

        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in BLOCKED:
                raise ImportError("blocked optional dependency: " + name)
            return None

    sys.meta_path.insert(0, Blocker())

    import numpy as np

    from sharpearena import SharpeArenaEnv
    from sharpearena.vector import SharpeArenaVectorEnv

    env = SharpeArenaEnv(n_symbols=2, n_days=20, seed=1)
    obs, info = env.reset()
    assert env.observation_space.contains(obs), obs
    obs, reward, terminated, truncated, info = env.step(
        np.full((2,), 0.25, dtype=np.float32)
    )
    assert isinstance(reward, float) and not terminated and not truncated

    vec = SharpeArenaVectorEnv(seeds=[1, 2], n_symbols=2, n_days=20)
    vec.reset()
    _o, rewards, _t, _c, _i = vec.step(np.full((2, 2), 0.25, dtype=np.float32))
    assert rewards.shape == (2,)

    for name in BLOCKED:
        assert name not in sys.modules, name

    print("ok")
    """
)


def test_the_adapter_runs_with_every_optional_dependency_blocked():
    """The extras are optional in the metadata; this proves they are optional at runtime.
    A subprocess, because the blocker has to be installed before sharpearena is imported,
    and because a test session that already imported an extra would mask a lazy import."""
    result = subprocess.run(
        [sys.executable, "-c", _NO_OPTIONAL_DEPENDENCY_PROGRAM.format(blocked=OPTIONAL_DEPENDENCIES)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("ok")
