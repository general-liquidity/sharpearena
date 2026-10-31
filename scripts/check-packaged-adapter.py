#!/usr/bin/env python3
"""Exercise the packaged Gymnasium adapter, with no optional dependency installed.

The existing wheel consumer check imports the native `TradingEnv` and steps one bar.
That leaves the surface a learner actually binds against, `SharpeArenaEnv` and
`SharpeArenaVectorEnv`, proven only from the source tree. This script drives both from
whatever `sharpearena` the interpreter resolves, so it belongs in a venv holding the
installed wheel and its declared dependencies only.

It also refuses to run if any optional or accelerator dependency is importable, because
the property under test is that none of them is needed to construct, reset or step the
environment. Run it from outside the checkout so the import cannot fall back to the
source tree::

    python -m venv consumer
    ./consumer/bin/pip install sharpearena-*.whl
    ./consumer/bin/python scripts/check-packaged-adapter.py

The rollout loop here is an integration fixture, not a learning experiment: it selects
among fixed candidate weights by realized reward. It establishes that a training loop
can drive the packaged adapter, and nothing about what a learner would learn.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def optional_dependencies() -> tuple:
    """The names an installed extra would make importable, plus the accelerators.

    Read from the declared extras rather than restated here: a hand-kept list stops
    covering the next extra someone adds and nothing says so. `SHARPEARENA_OPTIONAL_IMPORTS`
    carries the same names for interpreters older than the 3.11 that `tomllib` needs; with
    neither available this refuses rather than checking a list it cannot vouch for.
    """
    from optional_extras import ACCELERATOR_IMPORTS, optional_import_names

    passed = os.environ.get("SHARPEARENA_OPTIONAL_IMPORTS")
    if passed:
        return tuple(passed.split()) + ACCELERATOR_IMPORTS
    return tuple(optional_import_names()) + ACCELERATOR_IMPORTS


def assert_no_optional_dependencies() -> None:
    present = [name for name in optional_dependencies() if importlib.util.find_spec(name)]
    if present:
        raise SystemExit(
            f"{', '.join(present)} is importable; this check must run in an environment "
            "holding only the wheel and its declared dependencies"
        )


def assert_import_is_the_installed_package(module) -> None:
    """A `maturin develop` checkout resolves the import to the crate's `python/`
    directory through a `.pth` entry, which would leave the packaged surface unchecked.
    An installed distribution lives under the interpreter's package directory."""
    location = Path(module.__file__).resolve()
    if not {"site-packages", "dist-packages"} & set(location.parts):
        raise SystemExit(f"imported {location}, which is not an installed distribution")


def check_binary_step_path(scalar_type, vector_type, np) -> None:
    """The installed extension's array path against its JSON path, byte for byte."""
    json_env, array_env = (scalar_type(n_symbols=3, n_days=40, seed=1) for _ in range(2))
    json_env.reset(seed=1)
    array_env.reset(seed=1)
    action = np.full(json_env.action_space.shape, 0.25, dtype=np.float32)
    for _ in range(40):
        expected = json_env.step(action)
        got = array_env.step_array(action)
        if expected[0]["closes"].tobytes() != got[0]["closes"].tobytes():
            raise SystemExit("step_array closes differ from the JSON path")
        if expected[1:4] != got[1:4]:
            raise SystemExit(f"step_array differs from the JSON path: {expected[1:4]}, {got[1:4]}")
    if json_env.clone_state() != array_env.clone_state():
        raise SystemExit("step_array left the engine in a different state")

    json_batch, array_batch = (
        vector_type(seeds=[1, 2], n_symbols=3, n_days=4, autoreset_mode="same_step")
        for _ in range(2)
    )
    json_batch.reset()
    array_batch.reset()
    actions = np.full((2, 3), 0.25, dtype=np.float32)
    for _ in range(9):
        expected = json_batch.step(actions)
        got = array_batch.step_array(actions)
        if (
            expected[0]["positions"].tobytes() != got[0]["positions"].tobytes()
            or expected[1].tobytes() != got[1].tobytes()
            or expected[3].tobytes() != got[3].tobytes()
        ):
            raise SystemExit("step_array batch differs from the JSON path")


def check_fresh_scenarios(vector_type, np) -> None:
    """The installed extension draws the frozen stream and keeps it in the train band."""
    from sharpearena._seed_bands import EVAL_SEED_BASE

    vector = vector_type(
        seeds=[0],
        n_symbols=2,
        n_days=1,
        autoreset_mode="same_step",
        autoreset_scenarios="fresh_v1",
    )
    vector.reset()
    action = np.zeros((1, 2), dtype=np.float32)
    drawn = [vector.step_array(action)[4]["drawn_scenario_seed"][0] for _ in range(4)]
    if drawn != [579153, 783213, 915425, 650791]:
        raise SystemExit(f"fresh_v1 drew an unexpected stream: {drawn}")
    try:
        vector_type(seeds=[EVAL_SEED_BASE], autoreset_scenarios="fresh_v1")
    except ValueError:
        return
    raise SystemExit("fresh_v1 accepted a held-out lane")


def main() -> int:
    assert_no_optional_dependencies()

    import numpy as np

    import sharpearena
    from sharpearena import SharpeArenaEnv
    from sharpearena.vector import SharpeArenaVectorEnv

    assert_import_is_the_installed_package(sharpearena)

    env = SharpeArenaEnv(n_symbols=3, n_days=40, seed=1)
    observation, info = env.reset()
    if not env.observation_space.contains(observation):
        raise SystemExit(f"reset observation outside the declared space: {observation}")
    if set(info) != {"scenario_seed", "seeds"}:
        raise SystemExit(f"unexpected reset info keys: {sorted(info)}")

    # A candidate-selection loop over the packaged adapter: fixed candidates, episodes
    # rolled to their own boundary, best mean reward kept. No gradient, no accelerator.
    best_weight, best_score = None, -float("inf")
    for weight in (0.0, 0.25, 0.5):
        env.reset(seed=1)
        action = np.full(env.action_space.shape, weight, dtype=np.float32)
        rewards = []
        while True:
            _observation, reward, terminated, truncated, _info = env.step(action)
            rewards.append(float(reward))
            if terminated or truncated:
                break
        if len(rewards) != 40:
            raise SystemExit(f"episode spanned {len(rewards)} bars, expected the window")
        score = sum(rewards) / len(rewards)
        if score > best_score:
            best_weight, best_score = weight, score

    vector = SharpeArenaVectorEnv(seeds=[1, 2, 3], n_symbols=3, n_days=40)
    vector.reset()
    batch, rewards, terminated, truncated, infos = vector.step(
        np.full((3, 3), 0.25, dtype=np.float32)
    )
    if batch["closes"].shape != (3, 3) or rewards.shape != (3,):
        raise SystemExit(f"unexpected batch shapes: {batch['closes'].shape}, {rewards.shape}")
    if terminated.any() or truncated.any():
        raise SystemExit("a 40-bar lane ended on its first step")
    if infos["nav"].shape != (3,):
        raise SystemExit(f"unexpected nav shape: {infos['nav'].shape}")

    check_binary_step_path(SharpeArenaEnv, SharpeArenaVectorEnv, np)
    check_fresh_scenarios(SharpeArenaVectorEnv, np)

    print(
        f"packaged adapter ok: scalar and vector rollouts ran with no optional "
        f"dependency installed (best candidate weight {best_weight})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
