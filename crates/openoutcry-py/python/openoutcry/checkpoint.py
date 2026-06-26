"""Replay-based state clone / restore for :class:`~openoutcry.gym.OpenOutcryEnv`.

OpenOutcry's whole philosophy is "recompute from raw decisions": a trajectory replays
byte-identically because the engine is deterministic given its construction params, the
user seed, and the ordered sequence of actions applied so far. This module turns that
property into a checkpoint primitive **without touching the engine**.

A checkpoint is therefore *not* a memory image of the native env. It is::

    (construction params) + (ordered action list) + (step index)

and ``restore`` is "build a fresh env from the params, ``reset(seed)``, and replay every
recorded action". Because the engine is seed-deterministic, the restored env is identical
to the snapshot point — same next observations, same next rewards. ``branch`` does the same
into an *independent* env, which is what tree search / MCTS / counterfactual rollouts need:
explore a subtree without perturbing the parent.

Cost model (be honest): ``restore_state`` / ``branch`` are **O(prefix length)** — they
replay every action up to the snapshot. For deep trees this is quadratic in the worst case.
An O(1) engine-level snapshot (copy the native simulator state) is a future
``sharpebench-sim`` enhancement; until then, replay is the leak-free, engine-agnostic way to
get exact restoration.

Leak-safety: the state carries only construction params + decisions — **never** the
underlying ``Dataset`` / native ``TradingEnv`` handle or a raw price series. Those would let
a deserialized checkpoint peek at future bars. Params are validated against that invariant
on capture (mirrors :mod:`openoutcry.trace`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import gymnasium as gym

from .gym import OpenOutcryEnv, _EVAL_SEED_BASE

# Mirrors trace.py: a checkpoint must never serialize a raw dataset / env handle — that
# carries the full (incl. future) series. Matched by class name + the reset/step duck-type.
_LEAKY_TYPE_NAMES = frozenset({"TradingEnv", "Dataset", "OpenOutcryEnv"})


def _is_leaky(value: Any) -> bool:
    if type(value).__name__ in _LEAKY_TYPE_NAMES:
        return True
    return callable(getattr(value, "reset", None)) and callable(getattr(value, "step", None))


def _assert_no_leak(params: dict) -> None:
    for key, value in params.items():
        if _is_leaky(value):
            raise TypeError(
                f"refusing to checkpoint param {key}={type(value).__name__!r}: a raw "
                "dataset/env handle would leak future bars; the state stores construction "
                "params + decisions only."
            )


def _extract_params(env: OpenOutcryEnv) -> dict:
    """Read the construction params back off a wrapped :class:`OpenOutcryEnv`.

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
        "mode": "eval" if env._seed_offset == _EVAL_SEED_BASE else "train",
        "env_kwargs": dict(env._kwargs),
    }
    _assert_no_leak(params)
    return params


def _build_env(params: dict) -> OpenOutcryEnv:
    """Construct a fresh :class:`OpenOutcryEnv` from captured params."""
    return OpenOutcryEnv(
        n_symbols=params["n_symbols"],
        n_days=params["n_days"],
        seed=params["seed"],
        window_start=params.get("window_start"),
        window_end=params.get("window_end"),
        csv_text=params.get("csv_text"),
        max_weight=params.get("max_weight", 1.0),
        allow_short=params.get("allow_short", True),
        distribution_mode=params.get("distribution_mode", "calm"),
        mode=params.get("mode", "train"),
        env_kwargs=params.get("env_kwargs") or None,
    )


@dataclass
class CheckpointState:
    """A serializable snapshot of an :class:`OpenOutcryEnv` at a point in an episode.

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

    def to_dict(self) -> dict:
        return {
            "params": dict(self.params),
            "actions": [list(a) for a in self.actions],
            "step": int(self.step),
            "include_rng": bool(self.include_rng),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CheckpointState":
        return cls(
            params=dict(d["params"]),
            actions=[list(a) for a in d.get("actions", [])],
            step=int(d.get("step", 0)),
            include_rng=bool(d.get("include_rng", True)),
        )


class CheckpointableEnv(gym.Wrapper):
    """Wrap an :class:`OpenOutcryEnv` and record every action passed to :meth:`step`.

    The recorded action list *is* the restorable state: combined with the wrapped env's
    construction params it replays the episode byte-identically. Use :meth:`clone_state` to
    snapshot, :meth:`restore_state` to rewind this env, and :meth:`branch` to fork an
    independent env (tree search / what-if) that shares no mutable state with the parent.
    """

    def __init__(self, env: OpenOutcryEnv) -> None:
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
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        result = self.env.step(arr)
        self._actions.append(arr.copy())
        self._step += 1
        return result

    # -- checkpoint API ----------------------------------------------------

    def clone_state(self, *, include_rng: bool = True) -> CheckpointState:
        """Capture the current env state as a serializable :class:`CheckpointState`.

        O(prefix length) in space (the action list). See :meth:`restore_state` for the
        replay cost. ``include_rng`` is documented on :class:`CheckpointState`.
        """
        return CheckpointState(
            params=_extract_params(self.env),
            actions=[a.tolist() for a in self._actions],
            step=self._step,
            include_rng=include_rng,
        )

    def restore_state(self, state: CheckpointState) -> None:
        """Rewind THIS env to ``state`` by rebuilding from params and replaying actions.

        Builds a brand-new native env from ``state.params`` (so no stale simulator state
        survives), ``reset``s it, and replays every recorded action. **O(prefix length)** —
        it re-executes the whole decision prefix. Exact because the engine is deterministic.
        """
        self.env = _build_env(state.params)
        self._replay(state)

    def branch(self, state: CheckpointState) -> "CheckpointableEnv":
        """Return a NEW, independent :class:`CheckpointableEnv` restored to ``state``.

        The fork wraps its own freshly-built native env, so stepping it cannot touch this
        env (or any sibling branch) — the property tree search relies on. Two branches from
        the same ``state`` fed the same actions produce identical trajectories. Also
        O(prefix length) to materialize.
        """
        fork = CheckpointableEnv(_build_env(state.params))
        fork._replay(state)
        return fork

    # -- internal ----------------------------------------------------------

    def _replay(self, state: CheckpointState) -> None:
        """Reset ``self.env`` (assumed freshly built from ``state.params``) and replay."""
        self.env.reset()
        self._actions = []
        for a in state.actions:
            arr = np.asarray(a, dtype=np.float32).reshape(-1)
            self.env.step(arr)
            self._actions.append(arr)
        self._step = int(state.step)


__all__ = ["CheckpointableEnv", "CheckpointState"]
