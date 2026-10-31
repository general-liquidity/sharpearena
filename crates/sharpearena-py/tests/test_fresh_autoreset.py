"""The opt-in ``fresh_v1`` scenario stream for recycled vector lanes.

``replay`` keeps a run to as many scenarios as lanes. These tests pin what ``fresh_v1``
adds and what it must not change: draws are deterministic and per-lane, they stay in the
training band, each recycled episode is exactly the scalar scenario its recorded seed
names, and the default leaves the wire shape alone.
"""

import json

import numpy as np
import pytest

from sharpearena.sharpearena_py import TradingEnv, VecTradingEnv
from sharpearena.vector import SharpeArenaVectorEnv
from sharpearena._seed_bands import EVAL_SEED_BASE

PANEL = dict(n_symbols=3, n_days=30)
ACTIONS = np.full((2, 3), 0.25, dtype=np.float32)
RECYCLING_MODES = ("next_step", "same_step")


def fresh_env(autoreset_mode, seeds=(4, 9), **kwargs):
    return SharpeArenaVectorEnv(
        seeds=list(seeds),
        autoreset_mode=autoreset_mode,
        autoreset_scenarios="fresh_v1",
        **PANEL,
        **kwargs,
    )


def rollout(env, steps=200):
    env.reset()
    closes, drawn = [], []
    for _ in range(steps):
        obs, _rewards, _terminated, _truncated, infos = env.step(ACTIONS[: env.num_envs])
        closes.append(obs["closes"].copy())
        drawn.append(list(infos["drawn_scenario_seed"]))
    return np.stack(closes), drawn


@pytest.mark.parametrize("mode", RECYCLING_MODES)
def test_two_runs_draw_the_same_seeds_and_tape(mode):
    closes_a, drawn_a = rollout(fresh_env(mode))
    closes_b, drawn_b = rollout(fresh_env(mode))
    assert drawn_a == drawn_b
    assert closes_a.tobytes() == closes_b.tobytes()


@pytest.mark.parametrize("mode", RECYCLING_MODES)
def test_every_draw_stays_inside_the_training_band(mode):
    env = fresh_env(mode, max_episode_steps=2)
    _closes, drawn = rollout(env, steps=300)
    seeds = [seed for row in drawn for seed in row if seed is not None]
    assert len(seeds) >= 100
    assert all(0 <= seed < EVAL_SEED_BASE for seed in seeds)
    assert len(set(seeds)) > 90, "recycled episodes must see new scenarios"


def test_disabled_mode_stays_on_its_scenarios():
    env = fresh_env("disabled")
    _closes, drawn = rollout(env, steps=80)
    assert all(seed is None for row in drawn for seed in row)
    assert env.scenario_seeds == [4, 9]


def test_the_stream_is_frozen_across_runtimes():
    """Same values as the Rust pin, reached through the installed binding."""
    env = VecTradingEnv(
        seeds=[0],
        n_symbols=2,
        n_days=1,
        autoreset_mode="same_step",
        autoreset_scenarios="fresh_v1",
    )
    decision = json.dumps([{"orders": [], "reasoning": ""}])
    env.reset_batch()
    drawn = [json.loads(env.step_batch(decision))["drawn_seeds"][0] for _ in range(4)]
    assert drawn == [579153, 783213, 915425, 650791]


@pytest.mark.parametrize("mode", RECYCLING_MODES)
def test_each_recycled_episode_matches_a_scalar_env_on_its_recorded_seed(mode):
    """A recorded seed is enough to replay the episode, execution noise included."""
    env = fresh_env(mode, seeds=[4])
    action = ACTIONS[:1]
    env.reset()
    decision = json.dumps(
        {
            "orders": [
                {
                    "symbol": symbol,
                    "action": "buy",
                    "target_weight": float(action[0][0]),
                    "confidence": 0.5,
                }
                for symbol in env.symbols
            ],
            "reasoning": "",
        }
    )
    while True:
        _obs, _r, _te, _tr, infos = env.step(action)
        if infos["first"][0]:
            break
    seed = infos["drawn_scenario_seed"][0]
    scalar = TradingEnv(seed=seed, exec_seed=seed, **PANEL)
    scalar.reset()
    for _ in range(PANEL["n_days"]):
        obs, reward, terminated, truncated, infos = env.step(action)
        scalar_obs, scalar_reward, done, _info = scalar.step(decision)
        closes = [s["close_history"][-1] for s in json.loads(scalar_obs)["symbols"]]
        assert reward[0] == scalar_reward
        assert truncated[0] == done
        if done:
            if mode == "same_step":
                assert list(infos["final_obs"][0]["closes"]) == closes
                assert infos["final_info"][0]["scenario_seed"] == seed
            else:
                assert list(obs["closes"][0]) == closes
            break
        assert list(obs["closes"][0]) == closes
        assert infos["scenario_seed"][0] == seed


def test_readback_records_the_stream_and_the_live_tape():
    env = fresh_env("same_step", seeds=[4])
    _closes, drawn = rollout(env, steps=70)
    seeds = [row[0] for row in drawn if row[0] is not None]
    readback = env.effective_configs[0]
    assert readback["autoreset_scenarios"] == "fresh_v1"
    assert readback["drawn_seeds"] == seeds
    assert readback["seed_stream"] == {
        "root_seed": 4,
        "lane": 0,
        "band_start": 0,
        "band_end": EVAL_SEED_BASE,
    }
    assert readback["current_scenario_seed"] == seeds[-1] == env.scenario_seeds[0]

    live = SharpeArenaVectorEnv(seeds=[seeds[-1]], **PANEL).effective_configs[0]
    assert readback["current_dataset_fnv1a64"] == live["dataset_fnv1a64"]
    assert readback["dataset_fnv1a64"] != live["dataset_fnv1a64"]


def test_reset_without_a_seed_keeps_the_stream_position():
    env = fresh_env("same_step", seeds=[4])
    rollout(env, steps=40)
    before = env.scenario_seeds
    drawn_before = env.effective_configs[0]["drawn_seeds"]
    _obs, infos = env.reset()
    assert env.scenario_seeds == before
    assert list(infos["scenario_seed"]) == before
    assert env.effective_configs[0]["drawn_seeds"] == drawn_before


def test_a_seeded_reset_reopens_the_stream_from_the_new_seeds():
    env = fresh_env("same_step", seeds=[4])
    rollout(env, steps=40)
    env.reset(seed=[17])
    assert env.effective_configs[0]["seed_stream"]["root_seed"] == 17
    assert env.effective_configs[0]["drawn_seeds"] == []


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(
            lambda: SharpeArenaVectorEnv(
                num_envs=2, mode="eval", autoreset_scenarios="fresh_v1", **PANEL
            ),
            id="eval-mode",
        ),
        pytest.param(
            lambda: SharpeArenaVectorEnv(
                seeds=[1], mode="eval", autoreset_scenarios="fresh_v1", **PANEL
            ),
            id="eval-mode-with-train-seeds",
        ),
        pytest.param(lambda: fresh_env("next_step", seeds=[EVAL_SEED_BASE]), id="eval-seed"),
        pytest.param(
            lambda: fresh_env("next_step", seeds=[EVAL_SEED_BASE + 7, 1]), id="mixed-lanes"
        ),
    ],
)
def test_held_out_starts_are_refused(build):
    with pytest.raises(ValueError):
        build()


def test_a_held_out_reset_seed_is_refused_and_leaves_the_batch_intact():
    env = fresh_env("next_step", seeds=[4])
    with pytest.raises(ValueError, match="training band"):
        env.reset(seed=EVAL_SEED_BASE)
    assert env.scenario_seeds == [4]


def test_an_unversioned_label_is_refused():
    with pytest.raises(ValueError, match="autoreset_scenarios"):
        VecTradingEnv(seeds=[1], autoreset_scenarios="fresh")


def test_the_default_leaves_the_wire_shape_and_scenarios_alone():
    env = VecTradingEnv(seeds=[1], autoreset_mode="same_step", n_days=2, n_symbols=2)
    readback_before = env.effective_configs
    env.reset_batch()
    decision = json.dumps([{"orders": [], "reasoning": ""}])
    keys = set()
    final = None
    for _ in range(3):
        out = json.loads(env.step_batch(decision))
        keys |= set(out)
        final = next((f for f in out["final_info"] if f is not None), final)
    assert keys == {
        "n",
        "observations",
        "rewards",
        "terminated",
        "truncated",
        "first",
        "infos",
        "final_obs",
        "final_info",
    }
    assert set(final) == {"nav", "events"}
    assert env.effective_configs == readback_before
    assert set(json.loads(readback_before[0])) == {
        "n_symbols",
        "n_bars",
        "window_start",
        "window_end",
        "exec_seed",
        "dataset_fnv1a64",
    }
    assert env.scenario_seeds == [1]
    assert env.autoreset_scenarios == "replay"
