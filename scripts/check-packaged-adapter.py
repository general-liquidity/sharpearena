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

    print(
        f"packaged adapter ok: scalar and vector rollouts ran with no optional "
        f"dependency installed (best candidate weight {best_weight})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
