"""A Stable-Baselines3 ``VecEnv`` over the native batched SharpeArena engine.

SB3 does not consume Gymnasium's vector API. It defines its own ``VecEnv``, and its
documentation says so in terms: "SB3 VecEnv API is not the same as Gym API. SB3 VecEnv
API is actually close to Gym 0.21 API but differs to Gym 0.26+ API"
(`guide/vec_envs.html <https://stable-baselines3.readthedocs.io/en/v2.9.0/guide/vec_envs.html>`_).
The two differ on step arity, on where truncation is reported, on the ``info`` container,
on the reset signature, and, most consequentially, on **when a finished lane resets and
which observation is returned on the step it finishes**. Getting that last point backwards
does not raise: it silently corrupts every return an SB3 agent computes. So the mapping is
written out here, checked by ``tests/test_sb3.py``, and refuses rather than guesses.

**Targeted version.** The semantics below were read from the installed
``stable_baselines3`` **2.9.0** source, specifically
``stable_baselines3/common/vec_env/base_vec_env.py`` and ``dummy_vec_env.py``, and match
the ``/en/v2.9.0/`` documentation. ``DummyVecEnv.step_wait`` is the reference
implementation this adapter reproduces; :func:`sb3_reference_semantics` restates the three
lines of it that matter, so a future SB3 release that changes them shows up as a failing
test rather than as a quiet return-corruption. There is no ``/en/stable/`` alias on that
documentation site; ``/en/master/`` tracks an unreleased ``2.9.2a0``.

**The autoreset mapping, which is the whole point of this module.**

SB3 resets a finished environment *on the same step it finished*, and returns the new
episode's first observation as that step's observation:

    "the observation returned for the i-th environment when ``done[i]`` is true will in
    fact be the first observation of the next episode, not the last observation of the
    episode that has just terminated"

That is gym3 same-step autoreset. :class:`~sharpearena.vector.SharpeArenaVectorEnv`
supports exactly that under ``autoreset_mode="same_step"``, and it is the **only** mode
this adapter accepts:

* ``next_step`` (the package default, and Gymnasium 1.x's) returns the terminal step
  verbatim and then emits a **shaped recycling transition** on the following step, with
  reward ``0.0`` and both flags clear. Fed to SB3 that recycling step is indistinguishable
  from a real transition, so it enters the rollout buffer as a genuine zero-reward step of
  the new episode and biases every advantage computed from it. It is refused.
* ``disabled`` never recycles a lane, so a finished lane would keep re-emitting its
  terminal bar forever while SB3 kept collecting. It is refused.

The full field mapping, engine to SB3:

+--------------------------+---------------------------------------------------------------+
| SB3 ``step_wait`` field  | Produced from                                                 |
+==========================+===============================================================+
| ``obs``                  | The same-step batch verbatim: already the new episode's t0 for |
|                          | any lane that finished, the ordinary next bar otherwise.       |
+--------------------------+---------------------------------------------------------------+
| ``rewards``              | ``rewards`` verbatim, in the engine's ``float64``. SB3's own   |
|                          | ``DummyVecEnv`` narrows to ``float32``; narrowing here would   |
|                          | break this package's exact-reward parity claim, and SB3's      |
|                          | buffers downcast on insert anyway.                             |
+--------------------------+---------------------------------------------------------------+
| ``dones``                | ``terminated | truncated`` (:func:`sb3_dones`).                |
+--------------------------+---------------------------------------------------------------+
| ``infos``                | The dict-of-arrays transposed into SB3's list-of-dicts.        |
+--------------------------+---------------------------------------------------------------+
| ``infos[i]["TimeLimit.   | ``truncated[i] and not terminated[i]``                         |
| truncated"]``            | (:func:`sb3_timelimit_truncated`). **Not** ``truncated[i]``.   |
+--------------------------+---------------------------------------------------------------+
| ``infos[i]["terminal_    | ``infos["final_obs"][i]``, the finished lane's last            |
| observation"]``          | observation. Present only on a lane that finished this step.   |
+--------------------------+---------------------------------------------------------------+

**Where SB3's encoding loses a bit, stated rather than hidden.** SB3 defines
``TimeLimit.truncated`` as ``truncated and not terminated``, and its own documentation
notes the consequence: "compared to Gym 0.26+ ``infos[env_idx]["TimeLimit.truncated"]``
and ``terminated`` are mutually exclusive." This environment can set both flags on one
step, because running out of bars is truncation and ``nav <= 0`` is termination, and a
blow-up on the final bar is both (``gym.py``). SB3's two-field encoding cannot represent
that, so :func:`recover_gymnasium_flags` returns ``(True, False)`` for it. The
bootstrapping decision is still right (a terminated step is never bootstrapped past), but
the truncation bit is gone. Nothing this adapter can do changes that, so instead of
pretending otherwise it also publishes the **unreduced** flags under
``infos[i]["sharpearena/terminated"]`` and ``infos[i]["sharpearena/truncated"]``, which are
lossless and are what a caller that needs the distinction should read.

**Seeding.** SB3's ``VecEnv.seed`` sets per-env seeds consumed at the next reset. This
environment's scenario tape is fixed at construction, by the ``seeds`` list, so that the
leak-free point-in-time window is reproducible; a reset replays the same tape rather than
drawing a new market. :meth:`SharpeArenaSB3VecEnv.seed` therefore seeds the action space
only, returns the fixed lane seeds, and warns once when handed a seed, rather than
accepting one and silently discarding it.

``stable_baselines3`` is an **optional** dependency, guarded like ``pettingzoo`` and
``mcp``: ``import sharpearena`` and ``import sharpearena.sb3_env`` both work without it,
and only *constructing* :class:`SharpeArenaSB3VecEnv` raises a named error when it is
absent. Install it with ``pip install "sharpearena[sb3]"``.

The observation is a single-level ``Dict``, which SB3 supports through
``MultiInputPolicy``; its "``Tuple`` observation spaces are not supported by any
environment, however, single-level ``Dict`` spaces are" is the governing sentence
(`guide/algos.html <https://stable-baselines3.readthedocs.io/en/v2.9.0/guide/algos.html>`_).
Pass ``"MultiInputPolicy"``, or wrap the flat route with
:class:`~sharpearena.spaces.FlattenObservation` if a plain ``MlpPolicy`` is wanted.
"""

from __future__ import annotations

import warnings
from typing import Any, Optional, Sequence

import numpy as np

from .vector import SharpeArenaVectorEnv

try:  # pragma: no cover - exercised only when stable-baselines3 is installed
    from stable_baselines3.common.vec_env.base_vec_env import VecEnv

    _HAS_SB3 = True
except Exception:  # noqa: BLE001 - any import failure means "not available"
    VecEnv = object  # type: ignore[assignment,misc]
    _HAS_SB3 = False


#: The one autoreset mode whose semantics match SB3's ``VecEnv``. See the module
#: docstring for why the other two are refused rather than adapted.
SB3_AUTORESET_MODE = "same_step"

#: The SB3 release whose ``DummyVecEnv.step_wait`` this adapter reproduces.
TARGET_SB3_VERSION = "2.9.0"


class SB3Unavailable(RuntimeError):
    """``stable_baselines3`` is not installed, so this route cannot be constructed."""


class SB3ContractError(ValueError):
    """A construction argument cannot be mapped onto SB3's ``VecEnv`` contract."""


def sb3_dones(terminated: np.ndarray, truncated: np.ndarray) -> np.ndarray:
    """SB3's ``dones``: the union of the two Gymnasium flags."""
    return np.asarray(terminated, dtype=bool) | np.asarray(truncated, dtype=bool)


def sb3_timelimit_truncated(
    terminated: np.ndarray, truncated: np.ndarray
) -> np.ndarray:
    """SB3's ``infos[i]["TimeLimit.truncated"]``: ``truncated and not terminated``.

    The ``and not terminated`` is the part that is easy to drop and impossible to notice
    afterwards. Passing ``truncated`` straight through makes a step that both blew up and
    ran out of bars read as a pure timeout, which tells SB3 to bootstrap past an absorbing
    state.
    """
    return np.asarray(truncated, dtype=bool) & ~np.asarray(terminated, dtype=bool)


def recover_gymnasium_flags(done: bool, timelimit_truncated: bool) -> tuple[bool, bool]:
    """Invert the SB3 encoding back to ``(terminated, truncated)``, as SB3 documents it.

    ``terminated = done and not infos[i]["TimeLimit.truncated"]``, from
    `guide/vec_envs.html <https://stable-baselines3.readthedocs.io/en/v2.9.0/guide/vec_envs.html>`_.
    Lossy for the both-flags-set step, by construction of SB3's encoding; the module
    docstring says which bit is lost and where the lossless one lives.
    """
    done = bool(done)
    timelimit_truncated = bool(timelimit_truncated)
    terminated = done and not timelimit_truncated
    truncated = done and timelimit_truncated
    return terminated, truncated


def sb3_reference_semantics() -> dict[str, str]:
    """The three lines of ``DummyVecEnv.step_wait`` this adapter reproduces, restated.

    Restated here rather than imported, for the reason ``integrations.parity`` restates
    the wire contract: a test that asks the adapter what the rule is cannot catch the
    adapter applying the wrong one. ``tests/test_sb3.py`` checks these strings against the
    installed ``DummyVecEnv`` source, so an upstream change to the mapping fails a test
    instead of quietly changing what this route computes.
    """
    return {
        "dones": "self.buf_dones[env_idx] = terminated or truncated",
        "timelimit_truncated": (
            'self.buf_infos[env_idx]["TimeLimit.truncated"] = truncated and not terminated'
        ),
        "terminal_observation": (
            'self.buf_infos[env_idx]["terminal_observation"] = obs'
        ),
    }


class SharpeArenaSB3VecEnv(VecEnv):  # type: ignore[misc]
    """``B`` leak-free SharpeArena lanes behind Stable-Baselines3's ``VecEnv`` API.

    Takes the same scenario arguments as
    :class:`~sharpearena.vector.SharpeArenaVectorEnv` and drives one native batched
    engine underneath, so the lanes are not separate Python environment objects. That is
    why :meth:`get_attr`, :meth:`set_attr` and :meth:`env_method` address the shared
    engine rather than ``B`` independent ones, and why :meth:`set_attr` refuses a
    per-lane write.

    ``autoreset_mode`` is fixed at ``"same_step"``; passing anything else raises
    :class:`SB3ContractError` rather than producing a route that miscomputes returns.
    """

    def __init__(
        self,
        num_envs: Optional[int] = None,
        *,
        seeds: Optional[Sequence[int]] = None,
        autoreset_mode: str = SB3_AUTORESET_MODE,
        **kwargs: Any,
    ) -> None:
        if not _HAS_SB3:
            raise SB3Unavailable(
                "stable-baselines3 is not installed; install it with "
                '`pip install "sharpearena[sb3]"` to use SharpeArenaSB3VecEnv'
            )
        if autoreset_mode != SB3_AUTORESET_MODE:
            raise SB3ContractError(
                f"SB3's VecEnv resets a finished env on the step it finishes, so this "
                f"route requires autoreset_mode={SB3_AUTORESET_MODE!r}; "
                f"{autoreset_mode!r} would feed SB3 transitions it reads as real ones "
                "and corrupt every return computed from them"
            )

        self._vec = SharpeArenaVectorEnv(
            num_envs, seeds=seeds, autoreset_mode=SB3_AUTORESET_MODE, **kwargs
        )
        self._seed_warned = False
        super().__init__(
            self._vec.num_envs,
            self._vec.single_observation_space,
            self._vec.single_action_space,
        )
        self._actions: Optional[np.ndarray] = None

    # -- introspection -----------------------------------------------------

    @property
    def symbols(self) -> list[str]:
        return self._vec.symbols

    @property
    def scenario_seeds(self) -> list[int]:
        return self._vec.scenario_seeds

    @property
    def vector_env(self) -> SharpeArenaVectorEnv:
        """The Gymnasium vector env underneath, for a caller that wants the 5-tuple."""
        return self._vec

    # -- the SB3 VecEnv contract -------------------------------------------

    def reset(self) -> dict[str, np.ndarray]:
        """SB3's reset: no arguments, observation only, info in ``reset_infos``."""
        obs, infos = self._vec.reset()
        self.reset_infos = self._transpose_infos(infos)
        self._reset_seeds()
        self._reset_options()
        self._actions = None
        return obs

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = actions

    def step_wait(self) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, list[dict]]:
        if self._actions is None:
            raise SB3ContractError("step_wait called without a pending step_async")
        actions = self._actions
        self._actions = None

        obs, rewards, terminated, truncated, infos = self._vec.step(actions)
        dones = sb3_dones(terminated, truncated)
        timelimit = sb3_timelimit_truncated(terminated, truncated)

        per_env = self._transpose_infos(infos)
        final_obs = infos.get("final_obs")
        final_info = infos.get("final_info")
        for index in range(self.num_envs):
            info = per_env[index]
            info["TimeLimit.truncated"] = bool(timelimit[index])
            # The lossless channel. SB3's own two fields cannot carry a step that both
            # terminated and truncated; these two always can.
            info["sharpearena/terminated"] = bool(terminated[index])
            info["sharpearena/truncated"] = bool(truncated[index])
            if dones[index]:
                if final_obs is None or final_obs[index] is None:
                    raise SB3ContractError(
                        f"lane {index} reported done with no final observation; the "
                        f"underlying env is not in {SB3_AUTORESET_MODE!r} autoreset mode"
                    )
                info["terminal_observation"] = final_obs[index]
                if final_info is not None and final_info[index] is not None:
                    info["sharpearena/terminal_info"] = final_info[index]

        return obs, np.asarray(rewards, dtype=np.float64), dones, per_env

    def close(self) -> None:
        self._vec.close()

    def seed(self, seed: Optional[int] = None) -> list[Optional[int]]:
        """Seed the action space and report the fixed lane seeds.

        The market tape is chosen at construction and a reset replays it, which is what
        makes an episode reproducible. Accepting a seed here and letting it change nothing
        would read as reseeding the market, so a seed is honoured where it can be (the
        action space) and warned about where it cannot.
        """
        if seed is not None:
            if not self._seed_warned:
                warnings.warn(
                    "SharpeArenaSB3VecEnv.seed() does not reseed the market: the scenario "
                    "tape is fixed by the `seeds` given at construction so that the "
                    "point-in-time window stays reproducible. The seed was applied to the "
                    "action space only. Construct a new env with different `seeds` to "
                    "change the market.",
                    UserWarning,
                    stacklevel=2,
                )
                self._seed_warned = True
            self.action_space.seed(int(seed))
        return list(self._vec.scenario_seeds)

    def get_attr(self, attr_name: str, indices: Any = None) -> list[Any]:
        """Read an attribute off the shared engine, replicated per addressed lane."""
        value = getattr(self._vec, attr_name)
        return [value for _ in self._get_indices(indices)]

    def set_attr(self, attr_name: str, value: Any, indices: Any = None) -> None:
        """Write an attribute on the shared engine.

        The lanes share one object, so a write addressed to a strict subset of them
        cannot be honoured and is refused instead of being applied to all.
        """
        addressed = list(self._get_indices(indices))
        if len(addressed) != self.num_envs:
            raise SB3ContractError(
                f"set_attr({attr_name!r}) addressed lanes {addressed}, but all "
                f"{self.num_envs} lanes share one native engine; a per-lane attribute "
                "write cannot be represented"
            )
        setattr(self._vec, attr_name, value)

    def env_method(
        self, method_name: str, *args: Any, indices: Any = None, **kwargs: Any
    ) -> list[Any]:
        """Call a method on the shared engine **once**, and replicate its result.

        Calling it once per addressed lane would run a batched method ``B`` times.
        """
        result = getattr(self._vec, method_name)(*args, **kwargs)
        return [result for _ in self._get_indices(indices)]

    def env_is_wrapped(self, wrapper_class: type, indices: Any = None) -> list[bool]:
        """Always ``False``: no per-lane ``gymnasium.Wrapper`` exists to inspect."""
        return [False for _ in self._get_indices(indices)]

    # -- internals ---------------------------------------------------------

    def _transpose_infos(self, infos: dict) -> list[dict[str, Any]]:
        """Gymnasium's dict-of-arrays into SB3's list-of-dicts.

        ``final_obs`` and ``final_info`` are dropped here: they are per-lane payloads
        that :meth:`step_wait` republishes under SB3's own ``terminal_observation`` key,
        and leaving the Gymnasium spelling in as well would give the same value two names.
        """
        per_env: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]
        for key, values in infos.items():
            if key in ("final_obs", "final_info"):
                continue
            for index in range(self.num_envs):
                per_env[index][key] = values[index]
        return per_env


__all__ = [
    "SB3_AUTORESET_MODE",
    "TARGET_SB3_VERSION",
    "SB3ContractError",
    "SB3Unavailable",
    "SharpeArenaSB3VecEnv",
    "recover_gymnasium_flags",
    "sb3_dones",
    "sb3_reference_semantics",
    "sb3_timelimit_truncated",
]
