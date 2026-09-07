"""Trusted replay/native checkpoints for :class:`~sharpearena.gym.SharpeArenaEnv`.

A restorable checkpoint contains construction params (including the full CSV, if supplied,
and the scenario seed), an action prefix and optionally a native snapshot. It is private
operator state: never give it, its pickle, or its explicit private export to an evaluated
agent. Rejecting live Dataset/env handles does not make reconstruction params future-free.

``state.to_dict()`` exports only a versioned step/action record. It omits construction
params, seeds and native state and cannot be restored. ``to_dict(include_private=True)``
opts into the full, trusted-only payload. This distinction does not protect information
that a caller deliberately encodes in actions and is not an in-process isolation boundary.

Replay rebuilds the environment and applies every recorded action. Native restore avoids
action replay but still rebuilds the environment and copies/parses the snapshot, including
accumulated trace data; neither path promises constant-time restoration.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import gymnasium as gym

from .gym import SharpeArenaEnv, _EVAL_SEED_BASE
from ._action_validation import validated_action

# Live handles are not reconstruction data. CSV/seed params, however, are intentional
# private inputs, not proof of leak-freedom. Match names plus the reset/step duck-type.
_LEAKY_TYPE_NAMES = frozenset({"TradingEnv", "Dataset", "SharpeArenaEnv"})
_PUBLIC_SCHEMA = "sharpearena.checkpoint-public.v1"
_PRIVATE_SCHEMA = "sharpearena.checkpoint-restore.v1"


def _is_leaky(value: Any) -> bool:
    if type(value).__name__ in _LEAKY_TYPE_NAMES:
        return True
    return callable(getattr(value, "reset", None)) and callable(getattr(value, "step", None))


def _assert_no_leak(params: dict) -> None:
    seen: set[int] = set()

    def visit(value: Any) -> None:
        if _is_leaky(value):
            raise TypeError(
                f"refusing to checkpoint a raw dataset/env handle ({type(value).__name__})"
            )
        if isinstance(value, (dict, list, tuple)) and id(value) not in seen:
            seen.add(id(value))
            for nested in (value.values() if isinstance(value, dict) else value):
                visit(nested)

    visit(params)


def _extract_params(env: SharpeArenaEnv) -> dict:
    """Read the construction params back off a wrapped :class:`SharpeArenaEnv`.

    ``max_weight`` / ``allow_short`` aren't stored as attributes, but they're fully recovered
    from the action-space bounds the constructor derived from them. ``mode`` is recovered
    from the seed offset (``eval`` lives in the disjoint ``_EVAL_SEED_BASE`` band).
    """
    act = env.action_space
    high = float(np.asarray(act.high).reshape(-1)[0])
    low = float(np.asarray(act.low).reshape(-1)[0])
    params = {
        "n_symbols": env._n_symbols,
        "n_days": env._n_days,
        "seed": int(env._seed),
        "window_start": env._window_start,
        "window_end": env._window_end,
        "csv_text": env._csv_text,
        "max_weight": high,
        "allow_short": low < 0.0,
        "distribution_mode": env._distribution_mode,
        "vol_clustering": env._vol_clustering,
        "jump_burst_probability": env._jump_burst_probability,
        "jump_burst_persistence": env._jump_burst_persistence,
        "jump_burst_size": env._jump_burst_size,
        "mode": "eval" if env._seed_offset == _EVAL_SEED_BASE else "train",
        "env_kwargs": dict(env._kwargs),
    }
    _assert_no_leak(params)
    return deepcopy(params)


def _build_env(params: dict) -> SharpeArenaEnv:
    """Construct a fresh :class:`SharpeArenaEnv` from captured params."""
    return SharpeArenaEnv(
        n_symbols=params["n_symbols"],
        n_days=params["n_days"],
        seed=params["seed"],
        window_start=params.get("window_start"),
        window_end=params.get("window_end"),
        csv_text=params.get("csv_text"),
        max_weight=params.get("max_weight", 1.0),
        allow_short=params.get("allow_short", True),
        distribution_mode=params.get("distribution_mode", "calm"),
        vol_clustering=params.get("vol_clustering", 0.0),
        jump_burst_probability=params.get("jump_burst_probability", 0.0),
        jump_burst_persistence=params.get("jump_burst_persistence", 0.0),
        jump_burst_size=params.get("jump_burst_size", 0.0),
        mode=params.get("mode", "train"),
        env_kwargs=params.get("env_kwargs") or None,
    )


@dataclass
class CheckpointState:
    """A private, restorable snapshot; the default dict export is not restorable.

    Plain data only — construction ``params``, the ordered ``actions`` replayed so far (as
    nested lists, JSON/pickle-native), and the ``step`` index. ``include_rng`` records the
    ALE include-RNG / not distinction: the env is fully seed-deterministic, so the RNG state
    is *implied* by the seed already inside ``params`` (``include_rng=True`` ⇒ exact replay).
    ``include_rng=False`` is reserved for a future stochastic-fill mode where execution noise
    would need an explicit RNG snapshot to reproduce; it does not change behavior today.
    """

    params: dict
    actions: list = field(default_factory=list)
    step: int = 0
    include_rng: bool = True
    native_state: Any = None  # private native snapshot JSON (set when native=True)

    def _validate(self) -> None:
        if not isinstance(self.params, dict):
            raise TypeError("checkpoint params must be a dict")
        _assert_no_leak(self.params)
        if type(self.step) is not int or self.step < 0:
            raise ValueError("checkpoint step must be a nonnegative integer")
        if not isinstance(self.actions, list) or len(self.actions) != self.step:
            raise ValueError("checkpoint step must equal the action-prefix length")
        if type(self.include_rng) is not bool:
            raise TypeError("checkpoint include_rng must be a bool")
        if self.native_state is not None and not isinstance(self.native_state, str):
            raise TypeError("checkpoint native_state must be a JSON string or None")
        for action in self.actions:
            values = np.asarray(action)
            if values.ndim != 1 or values.dtype.kind not in "fiu":
                raise ValueError("checkpoint actions must be one-dimensional real arrays")
            if not np.all(np.isfinite(values)):
                raise ValueError("checkpoint actions must be finite")

    def to_dict(self, *, include_private: bool = False) -> dict:
        """Export a public step/action record, or explicitly opt into private state.

        Private payloads and pickled CheckpointState objects contain future-reconstructing
        inputs. Keep them operator-only; never unpickle data from an untrusted sender.
        """
        if type(include_private) is not bool:
            raise TypeError("include_private must be a bool")
        self._validate()
        public = {
            "schema": _PUBLIC_SCHEMA,
            "actions": deepcopy(self.actions),
            "step": self.step,
        }
        if not include_private:
            return public
        _assert_no_leak(self.params)
        return {**public, "schema": _PRIVATE_SCHEMA, "params": deepcopy(self.params),
                "include_rng": self.include_rng, "native_state": deepcopy(self.native_state)}

    @classmethod
    def from_dict(cls, d: dict) -> "CheckpointState":
        """Read an explicit private payload (or a legacy private dict with params)."""
        if not isinstance(d, dict):
            raise TypeError("checkpoint payload must be a dict")
        if d.get("schema") == _PUBLIC_SCHEMA:
            raise ValueError("a public checkpoint export cannot restore an environment")
        if d.get("schema") not in (None, _PRIVATE_SCHEMA):
            raise ValueError("unsupported checkpoint schema")
        if not isinstance(d.get("params"), dict):
            raise ValueError("restoration requires private checkpoint params")
        _assert_no_leak(d["params"])
        state = cls(
            params=deepcopy(d["params"]),
            actions=deepcopy(d.get("actions", [])),
            step=d.get("step", 0),
            include_rng=d.get("include_rng", True),
            native_state=deepcopy(d.get("native_state")),
        )
        state._validate()
        return state


class CheckpointableEnv(gym.Wrapper):
    """Wrap an :class:`SharpeArenaEnv` and record every action passed to :meth:`step`.

    The recorded action list *is* the restorable state: combined with the wrapped env's
    construction params it replays the episode byte-identically. Use :meth:`clone_state` to
    snapshot, :meth:`restore_state` to rewind this env, and :meth:`branch` to fork an
    independent env (tree search / what-if) that shares no mutable state with the parent.
    """

    def __init__(self, env: SharpeArenaEnv) -> None:
        super().__init__(env)
        self._actions: list[np.ndarray] = []
        self._step: int = 0

    # -- gymnasium API -----------------------------------------------------

    def reset(self, *, seed=None, options=None):
        out = self.env.reset(seed=seed, options=options)
        # A reset starts a fresh episode (and, with an int seed, a fresh scenario whose seed
        # the wrapped env now stores), so the recorded prefix is cleared.
        self._actions = []
        self._step = 0
        return out

    def step(self, action):
        arr = validated_action(action, self.env.action_space)
        result = self.env.step(arr)
        self._actions.append(arr.copy())
        self._step += 1
        return result

    # -- checkpoint API ----------------------------------------------------

    def clone_state(self, *, include_rng: bool = True, native: bool = False) -> CheckpointState:
        """Capture the current env state as a serializable :class:`CheckpointState`.

        With ``native=False`` (default) the snapshot is the recorded action prefix
        (O(prefix length); engine-agnostic). With ``native=True`` it also contains the
        engine snapshot, whose copying/serialization cost grows with its state size.
        ``include_rng`` is documented on :class:`CheckpointState`.
        """
        if type(include_rng) is not bool or type(native) is not bool:
            raise TypeError("include_rng and native must be bools")
        return CheckpointState(
            params=_extract_params(self.env),
            actions=[a.tolist() for a in self._actions],
            step=self._step,
            include_rng=include_rng,
            native_state=self.env.clone_state() if native else None,
        )

    def restore_state(self, state: CheckpointState) -> None:
        """Rewind THIS env to ``state``. If ``state`` carries a ``native_state`` snapshot,
        rebuild the env and restore the engine without replay; otherwise rebuild and replay the
        recorded action prefix (O(prefix length)). Both are exact (the engine is
        deterministic), so the restored env reproduces the snapshot point byte-for-byte.
        """
        # Validation, reconstruction and replay may all fail. Keep the old engine and
        # prefix untouched until the candidate has successfully completed every step.
        candidate = self.branch(state)
        self.env = candidate.env
        self._actions = candidate._actions
        self._step = candidate._step

    def branch(self, state: CheckpointState) -> "CheckpointableEnv":
        """Return a NEW, independent :class:`CheckpointableEnv` restored to ``state``.

        The fork wraps its own freshly-built native env, so stepping it cannot touch this
        env (or any sibling branch) — the property tree search relies on. Two branches from
        the same ``state`` fed the same actions produce identical trajectories. Also
        O(prefix length) to materialize.
        """
        if not isinstance(state, CheckpointState):
            raise TypeError("restore requires a private CheckpointState")
        state._validate()
        fork = CheckpointableEnv(_build_env(deepcopy(state.params)))
        actions = [validated_action(a, fork.env.action_space) for a in state.actions]
        if state.native_state is not None:
            fork.env.reset()
            fork.env.restore_state(state.native_state)
            fork._actions = actions
            fork._step = state.step
        else:
            fork._replay(actions)
        return fork

    # -- internal ----------------------------------------------------------

    def _replay(self, actions: list[np.ndarray]) -> None:
        """Reset a freshly built candidate and replay its validated action prefix."""
        self.env.reset()
        self._actions = []
        for arr in actions:
            self.env.step(arr)
            self._actions.append(arr)
        self._step = len(actions)


__all__ = ["CheckpointableEnv", "CheckpointState"]
