"""The one place that says which optional extras exist and what proves each one.

`crates/sharpearena-py/pyproject.toml` declares the extras. This module declares the
coverage for them, and :func:`validate_coverage` refuses when the two disagree. An extra
added to the package without an entry here fails CI instead of going quietly uncovered,
which is the failure mode a hand-maintained list reproduces every time it goes stale.

The exercises run against whatever `sharpearena` the interpreter resolves, so they belong
in a virtual environment holding an installed wheel and nothing from the checkout. None of
them calls a model, reaches the network or opens a window.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "crates" / "sharpearena-py" / "pyproject.toml"

# Every distribution an extra may require, mapped to the names it is imported under.
# A requirement whose distribution is missing from this table fails validation, so
# widening an existing extra is as visible as adding a new one.
DIST_IMPORTS: Dict[str, Tuple[str, ...]] = {
    "verifiers": ("verifiers",),
    "minari": ("minari",),
    "pillow": ("PIL",),
    "pettingzoo": ("pettingzoo",),
    "mcp": ("mcp",),
    "stable-baselines3": ("stable_baselines3",),
    "torchrl": ("torchrl",),
    "ray": ("ray",),
}

# Asserted absent only where it is actually true: in the no-extras environment `run_none()`
# checks, where none of the seven declared extras, sb3/torchrl/ray included, are installed.
# It is not a claim about every cell this matrix runs, because it no longer can be: sb3 and
# torchrl each declare a real dependency on torch, and `ray[rllib]` can pull it too (see
# INSTALL_MATRIX_EXCLUDED below), so an sb3, torchrl or ray cell would find it importable by
# design. "No accelerator is needed" stays a tested claim for the bare wheel; it was never
# meant to hold once an extra legitimately wants one.
ACCELERATOR_IMPORTS: Tuple[str, ...] = ("torch", "jax")

# Extras whose real behavior is not exercised against an installed wheel in CI, with the
# guard that stops the gap from being silent: `INSTALL_MATRIX_EXCLUDED` is walked by
# `validate_coverage()` exactly like `EXERCISES`, so an extra can be listed here or in
# `EXERCISES` but never neither, and removing an entry without adding the other fails CI
# the same way an unlisted extra always has.
#
# Three extras, three different reasons, none of them "we didn't get to it":
#
# - sb3 and torchrl both pull torch, the heaviest single dependency any extra here
#   declares. `ci-requirements.txt` already excludes both for that reason, so
#   `tests/test_sb3.py` and `tests/test_torchrl.py` skip in the source-tree suite too.
# - ray[rllib]==2.58.0 pins gymnasium==1.2.2 exactly, which conflicts with the CI pin
#   of gymnasium==1.3.0 in a single `pip install --requirement` invocation
#   (`docs/integrations/inventory.md`'s version matrix). `ray` is excluded from
#   `ci-requirements.txt`, and `tests/test_ray_executor.py` / `tests/test_rllib.py`
#   skip in the source-tree suite, for that pin conflict rather than for size, though
#   `ray[rllib]` is also a large multi-package install in its own right.
#
# A `wheel-extras` cell for any of the three would either add a heavy torch install or
# force the wheel-extras job onto an incompatible gymnasium, to a matrix deliberately
# sized to stay near fifteen minutes. What is NOT skipped: `GUARDS["sb3"]`,
# `GUARDS["torchrl"]` and `GUARDS["ray"]` still run in `wheel-no-extras`, for free, since
# that job never installs any extra either way, and they are what prove
# `SharpeArenaSB3VecEnv(...)`, `SharpeArenaTorchRLEnv(...)`, `run_episodes(...,
# num_workers=1)` and `sharpearena_env_creator()` all refuse by name with the dependency
# absent. What IS unexercised against an installed wheel: the actual SB3 `VecEnv` /
# TorchRL `EnvBase` / Ray task / RLlib route with the extra present. Revisit if a
# CPU-only torch wheel becomes cheap enough to pin, if the gymnasium pin conflict is
# resolved upstream, or if the added CI minutes are explicitly accepted.
INSTALL_MATRIX_EXCLUDED: Dict[str, str] = {
    "sb3": "pulls torch; excluded from ci-requirements.txt for the same reason",
    "torchrl": "pulls torch; excluded from ci-requirements.txt for the same reason",
    "ray": (
        "ray[rllib]==2.58.0 pins gymnasium==1.2.2, conflicting with the CI gymnasium "
        "pin (1.3.0); excluded from ci-requirements.txt for the same reason"
    ),
}

_REQUIREMENT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*")


def read_extras() -> Dict[str, List[str]]:
    """`[project.optional-dependencies]` as declared, read with a real TOML parser."""
    try:
        import tomllib
    except ModuleNotFoundError as error:  # Python < 3.11
        raise SystemExit(
            "reading the declared extras needs tomllib (Python 3.11+); run this check "
            "on a newer interpreter, or pass the names in explicitly"
        ) from error
    with PYPROJECT.open("rb") as handle:
        document = tomllib.load(handle)
    return dict(document["project"].get("optional-dependencies", {}))


def requirement_distribution(requirement: str) -> str:
    """`minari[create,hdf5]` -> `minari`; `numpy>=1.21` -> `numpy`."""
    match = _REQUIREMENT_NAME.match(requirement.strip())
    if match is None:
        raise SystemExit(f"cannot read a distribution name out of {requirement!r}")
    return match.group(0).lower().replace("_", "-")


def optional_import_names() -> List[str]:
    """Every module name an installed extra would make importable."""
    names: List[str] = []
    for requirements in read_extras().values():
        for requirement in requirements:
            for name in DIST_IMPORTS[requirement_distribution(requirement)]:
                if name not in names:
                    names.append(name)
    return sorted(names)


def extra_import_names(extra: str) -> List[str]:
    requirements = read_extras()[extra]
    names: List[str] = []
    for requirement in requirements:
        for name in DIST_IMPORTS[requirement_distribution(requirement)]:
            if name not in names:
                names.append(name)
    return names


def validate_coverage() -> List[str]:
    """Refuse a declared extra with no exercise and no recorded install-matrix exclusion,
    an exercise or exclusion for no declared extra, an extra carrying both, and a
    requirement whose distribution this module cannot map to an import name."""
    declared = read_extras()
    covered = set(EXERCISES)
    excluded = set(INSTALL_MATRIX_EXCLUDED)
    both = sorted(covered & excluded)
    if both:
        raise SystemExit(
            f"{', '.join(both)} is in both EXERCISES and INSTALL_MATRIX_EXCLUDED; an "
            "extra is either exercised against an installed wheel or excluded with a "
            "reason, never both"
        )
    uncovered = sorted(set(declared) - covered - excluded)
    if uncovered:
        raise SystemExit(
            f"pyproject declares {', '.join(uncovered)} with no entry in EXERCISES and "
            "no recorded reason in INSTALL_MATRIX_EXCLUDED; add one so the extra is "
            "either exercised against the installed wheel or its absence from that "
            "matrix is a stated decision instead of a gap"
        )
    orphaned = sorted((covered | excluded) - set(declared))
    if orphaned:
        raise SystemExit(
            f"EXERCISES or INSTALL_MATRIX_EXCLUDED covers {', '.join(orphaned)}, which "
            "pyproject no longer declares"
        )
    for extra, requirements in declared.items():
        for requirement in requirements:
            distribution = requirement_distribution(requirement)
            if distribution not in DIST_IMPORTS:
                raise SystemExit(
                    f"the {extra} extra requires {distribution!r}, which DIST_IMPORTS "
                    "cannot map to an import name; add it so the absence check covers it"
                )
    missing_guards = sorted(set(declared) - set(GUARDS))
    if missing_guards:
        raise SystemExit(
            f"no guard check for {', '.join(missing_guards)}; the base package is supposed "
            "to refuse by name when an extra is absent, and that has to be asserted"
        )
    return sorted(declared)


def installable_extras() -> List[str]:
    """Declared extras that get their own `wheel-extras` cell: `EXERCISES` minus whatever
    `INSTALL_MATRIX_EXCLUDED` has recorded a reason to leave out. Call after
    `validate_coverage()`, which is what guarantees this partition is exhaustive."""
    return sorted(set(EXERCISES) - set(INSTALL_MATRIX_EXCLUDED))


# ---------------------------------------------------------------------------
# What each extra does when it is installed
# ---------------------------------------------------------------------------


def _exercise_pettingzoo() -> str:
    from pettingzoo import ParallelEnv

    from sharpearena.pettingzoo_env import MultiAgentSharpeArenaEnv

    env = MultiAgentSharpeArenaEnv(n_agents=2, n_symbols=4, n_days=30, seed=1)
    if not isinstance(env, ParallelEnv):
        raise SystemExit("the multi-agent env is not a pettingzoo ParallelEnv")
    env.reset(seed=1)
    for agent in env.agents:
        env.action_space(agent).seed(0)
    rewards: dict = {}
    for _ in range(5):
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        _observations, rewards, _terminations, _truncations, _infos = env.step(actions)
    if set(rewards) != set(env.possible_agents):
        raise SystemExit(f"rewards cover {sorted(rewards)}, not the roster")
    return f"stepped a {len(env.possible_agents)}-agent tournament: {sorted(rewards)}"


def _exercise_minari() -> str:
    import tempfile

    from sharpearena import SharpeArenaEnv
    from sharpearena.dataset import EVAL_SEED_BASE
    from sharpearena.minari_export import to_minari_train_test
    from sharpearena.trace import RolloutTraceWriter

    def record(seed: int, path: Path) -> SharpeArenaEnv:
        env = SharpeArenaEnv(n_symbols=4, n_days=30, seed=seed)
        observation, _info = env.reset(seed=seed)
        writer = RolloutTraceWriter(str(path), config={"n_symbols": 4, "n_days": 30})
        env.action_space.seed(seed)
        for step in range(10):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            writer.record_step(
                step=step,
                observation=observation,
                decision=action,
                reward=reward,
                info=info,
            )
            if terminated or truncated:
                break
        writer.finalize()
        writer.close()
        return env

    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        train_env = record(1, directory / "train.jsonl")
        record(EVAL_SEED_BASE, directory / "test.jsonl")
        # A unique id per run: Minari refuses to recreate an id already on disk, and a
        # rerun of this job on the same runner image would otherwise fail on that rather
        # than on anything about the package.
        dataset_id = f"sharpearena/ci-{directory.name.lower()}"
        train, test = to_minari_train_test(
            str(directory / "train.jsonl"),
            str(directory / "test.jsonl"),
            f"{dataset_id}-v0",
            observation_space=train_env.observation_space,
            action_space=train_env.action_space,
        )
    if train.total_episodes != 1 or test.total_episodes != 1:
        raise SystemExit(
            f"exported {train.total_episodes} train and {test.total_episodes} test "
            "episodes, expected one of each"
        )
    if not train.id.endswith("-train-v0") or not test.id.endswith("-test-v0"):
        raise SystemExit(f"unexpected dataset ids: {train.id}, {test.id}")
    return f"exported {train.id} and {test.id} over disjoint seed bands"


def _exercise_mcp() -> str:
    import asyncio

    from sharpearena.mcp_server import build_server

    server = build_server(env_kwargs={"n_symbols": 4, "n_days": 30, "seed": 1})
    tools = {tool.name for tool in asyncio.run(server.list_tools())}
    if tools != {"reset", "step", "spec"}:
        raise SystemExit(f"the MCP server exposes {sorted(tools)}, not reset/step/spec")
    return "built the MCP server and listed reset, step and spec with no transport"


def _exercise_verifiers() -> str:
    import verifiers as vf

    from sharpearena import verifiers_env

    if not verifiers_env._HAS_VERIFIERS:
        raise SystemExit("verifiers is installed but the module did not bind to it")
    if not issubclass(verifiers_env.SharpeArenaVerifiersEnv, vf.MultiTurnEnv):
        raise SystemExit(
            "SharpeArenaVerifiersEnv is the placeholder class, so the real one did not "
            "compile against the installed verifiers"
        )
    env = verifiers_env.load_environment(
        n_windows=2, n_symbols=4, n_days=30, max_episode_bars=10
    )
    rows = len(env.dataset)
    if rows != 2:
        raise SystemExit(f"the built dataset holds {rows} rows, expected 2")
    return f"loaded the multi-turn environment over {rows} scenario rows"


# ---------------------------------------------------------------------------
# What the base package does when the extra is absent
# ---------------------------------------------------------------------------


def _expect_refusal(call: Callable[[], object], fragment: str, what: str) -> str:
    try:
        call()
    except RuntimeError as error:
        if fragment not in str(error):
            raise SystemExit(
                f"{what} raised a RuntimeError that does not name the missing "
                f"dependency: {error}"
            ) from None
        return f"{what} refused with {str(error).split('.')[0]!r}"
    except BaseException as error:  # noqa: BLE001 - the type is the finding
        raise SystemExit(
            f"{what} raised {type(error).__name__} instead of the guarded RuntimeError: "
            f"{error}"
        ) from None
    raise SystemExit(f"{what} succeeded with the dependency absent; the guard is gone")


def _guard_pettingzoo() -> List[str]:
    from sharpearena.pettingzoo_env import MultiAgentSharpeArenaEnv, make_aec_env

    return [
        _expect_refusal(
            lambda: MultiAgentSharpeArenaEnv(n_agents=2),
            "pettingzoo is not installed",
            "MultiAgentSharpeArenaEnv(...)",
        ),
        _expect_refusal(
            lambda: make_aec_env(n_agents=2),
            "pettingzoo is not installed",
            "make_aec_env(...)",
        ),
    ]


def _guard_minari() -> List[str]:
    from sharpearena.minari_export import to_minari

    return [
        _expect_refusal(
            lambda: to_minari(
                [], "sharpearena/guard-v0", observation_space=None, action_space=None
            ),
            "minari is not installed",
            "to_minari(...)",
        )
    ]


def _guard_mcp() -> List[str]:
    from sharpearena.mcp_server import build_server

    return [
        _expect_refusal(
            build_server,
            "mcp is not installed",
            "build_server()",
        )
    ]


def _guard_verifiers() -> List[str]:
    from sharpearena.verifiers_env import SharpeArenaVerifiersEnv, load_environment

    return [
        _expect_refusal(
            load_environment,
            "verifiers is not installed",
            "load_environment()",
        ),
        _expect_refusal(
            SharpeArenaVerifiersEnv,
            "verifiers is not installed",
            "SharpeArenaVerifiersEnv(...)",
        ),
    ]


def _guard_sb3() -> List[str]:
    from sharpearena.sb3_env import SharpeArenaSB3VecEnv

    return [
        _expect_refusal(
            SharpeArenaSB3VecEnv,
            "stable-baselines3 is not installed",
            "SharpeArenaSB3VecEnv(...)",
        )
    ]


def _guard_torchrl() -> List[str]:
    from sharpearena.torchrl_env import SharpeArenaTorchRLEnv

    return [
        _expect_refusal(
            SharpeArenaTorchRLEnv,
            "torchrl is not installed",
            "SharpeArenaTorchRLEnv(...)",
        )
    ]


def _guard_ray() -> List[str]:
    from sharpearena.ray_executor import run_episodes
    from sharpearena.rllib_env import sharpearena_env_creator

    return [
        # num_workers=1 takes the Ray path; num_workers=0 (the default) stays local and
        # never imports Ray, which would prove nothing about this guard.
        _expect_refusal(
            lambda: run_episodes([], num_workers=1),
            "ray is not installed",
            "run_episodes([], num_workers=1)",
        ),
        _expect_refusal(
            sharpearena_env_creator,
            "ray[rllib] is not installed",
            "sharpearena_env_creator()",
        ),
    ]


EXERCISES: Dict[str, Callable[[], str]] = {
    "pettingzoo": _exercise_pettingzoo,
    "minari": _exercise_minari,
    "mcp": _exercise_mcp,
    "verifiers": _exercise_verifiers,
}

GUARDS: Dict[str, Callable[[], List[str]]] = {
    "pettingzoo": _guard_pettingzoo,
    "minari": _guard_minari,
    "mcp": _guard_mcp,
    "verifiers": _guard_verifiers,
    "sb3": _guard_sb3,
    "torchrl": _guard_torchrl,
    "ray": _guard_ray,
}

# The adapter modules that carry a guard. Importing each of them has to work with no
# extra installed, which is a separate claim from the guard refusing when called.
GUARDED_MODULES: Tuple[str, ...] = (
    "sharpearena.pettingzoo_env",
    "sharpearena.minari_export",
    "sharpearena.mcp_server",
    "sharpearena.verifiers_env",
    "sharpearena.sb3_env",
    "sharpearena.torchrl_env",
    "sharpearena.ray_executor",
    "sharpearena.rllib_env",
)
