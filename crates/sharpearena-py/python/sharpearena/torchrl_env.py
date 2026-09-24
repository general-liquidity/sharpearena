"""A TorchRL :class:`~torchrl.envs.EnvBase` over the SharpeArena engine.

TorchRL does not consume the Gymnasium contract. Its environment methods read and
write :class:`~tensordict.TensorDict` instances, the step output lives under a
``"next"`` entry, and a custom environment implements ``_reset``, ``_step`` and
``_set_seed`` rather than the public methods. :class:`SharpeArenaTorchRLEnv` is that
environment: it drives a :class:`~sharpearena.gym.SharpeArenaEnv` underneath and
restates its outputs in the layout a TorchRL collector expects.

**Truncation is written explicitly.** TorchRL's ``"done"`` is the union of the
end-of-trajectory signals, and an environment that writes ``"done"`` alone has it read
as ``terminated``. For SharpeArena that would be wrong in the common case: running out
of bars at the end of the point-in-time window is truncation, and bankruptcy is the
only true termination. All three keys are written on every step, so a value estimate
can bootstrap past the window end and not past a blow-up.

**The dtype boundary, stated rather than assumed.** The engine works in ``float64``
and most torch pipelines default to ``float32``. The two sides convert in opposite
directions here, and each is deliberate:

* Observations and the reward cross as ``torch.float64``, the engine's own width. No
  precision is lost, and :func:`torch.equal` against the NumPy ``float64`` array the
  Gymnasium adapter decodes holds bit for bit. ``obs_dtype=torch.float32`` narrows
  them, which is lossy; it is a named argument precisely so that it cannot happen by
  accident, and Gymnasium's ``Box.contains`` would not catch it because it casts
  safely.
* Actions cross as ``torch.float32``, because that is the width of the ``Box`` this
  package advertises as its action space (``SharpeArenaEnv.action_space``), and the
  fixture bounds the parity checker applies are the bounds of that ``Box``. A policy
  emitting ``float64`` weights therefore loses precision at this spec, and that is the
  one place in this adapter where precision is lost. Nothing is lost after it:
  ``_action_validation.validated_action`` widens the incoming weights back to
  ``float64`` before they reach the engine.

``torchrl`` is an **optional** dependency, guarded like ``pettingzoo`` and ``minari``:
``import sharpearena`` and ``import sharpearena.torchrl_env`` both work without it, and
only constructing :class:`SharpeArenaTorchRLEnv` raises. The base package imports with
``torch`` blocked entirely, which ``tests/test_rl_contract.py`` enforces, so nothing
here is imported from ``sharpearena/__init__.py``.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .gym import SharpeArenaEnv

try:  # pragma: no cover - exercised only when torchrl is installed
    import torch
    from tensordict import TensorDict
    from torchrl.data import Binary, Bounded, Composite, Unbounded
    from torchrl.envs import EnvBase

    _HAS_TORCHRL = True
except Exception:  # noqa: BLE001 - any import failure means "not available"
    torch = None  # type: ignore[assignment]
    TensorDict = None  # type: ignore[assignment]
    Binary = Bounded = Composite = Unbounded = None  # type: ignore[assignment]
    EnvBase = object  # type: ignore[assignment,misc]
    _HAS_TORCHRL = False


#: The observation keys, in the order ``SharpeArenaEnv`` decodes them.
OBSERVATION_KEYS = ("closes", "positions", "cash")

#: The engine's own width. Observations and rewards cross the boundary at this dtype
#: unless a caller names the narrower one, and the parity digest is computed from it.
ENGINE_DTYPE_NAME = "float64"


class TorchRLUnavailable(RuntimeError):
    """``torchrl`` is not installed, so the TorchRL route cannot be constructed."""


def _require_torchrl() -> None:
    if not _HAS_TORCHRL:
        raise TorchRLUnavailable(
            "torchrl is not installed. Install the 'torchrl' extra "
            "(pip install 'sharpearena[torchrl]') to drive this environment from a "
            "TorchRL collector; the rest of the sharpearena package works without it."
        )


def _resolve_obs_dtype(obs_dtype: Any) -> Any:
    """The observation/reward dtype, refusing any width that is neither of the two.

    ``None`` means the engine's own ``float64``. ``float32`` is accepted because a
    caller may genuinely want it, and refused silently by nothing: it has to be named.
    Every other dtype is refused rather than coerced, because a coercion here is the
    narrowing this adapter exists to keep visible.
    """
    if obs_dtype is None:
        return torch.float64
    resolved = obs_dtype if isinstance(obs_dtype, torch.dtype) else getattr(torch, str(obs_dtype), None)
    if resolved not in (torch.float64, torch.float32):
        raise ValueError(
            f"obs_dtype must be torch.float64 (the engine's own width, lossless) or "
            f"torch.float32 (an explicit narrowing), got {obs_dtype!r}"
        )
    return resolved


class SharpeArenaTorchRLEnv(EnvBase):
    """A scalar (``batch_size == torch.Size([])``) TorchRL environment over SharpeArena.

    Constructor arguments other than ``obs_dtype`` and ``device`` are forwarded to
    :class:`~sharpearena.gym.SharpeArenaEnv` unchanged, so the scenario controls, the
    weight bound and the train/eval seed band are the ones that package already
    documents.
    """

    def __init__(
        self,
        *,
        obs_dtype: Any = None,
        device: Any = "cpu",
        **env_kwargs: Any,
    ) -> None:
        _require_torchrl()
        resolved_dtype = _resolve_obs_dtype(obs_dtype)
        super().__init__(device=device, batch_size=torch.Size([]))
        self._obs_dtype = resolved_dtype
        self._gym = SharpeArenaEnv(**env_kwargs)
        self._pending_seed: Optional[int] = None
        self._last_step_info: dict = {}

        n = len(self._gym.symbols)
        action_space = self._gym.action_space
        self.observation_spec = Composite(
            closes=Bounded(
                low=0.0, high=float("inf"), shape=(n,), dtype=resolved_dtype
            ),
            positions=Unbounded(shape=(n,), dtype=resolved_dtype),
            cash=Unbounded(shape=(1,), dtype=resolved_dtype),
            shape=torch.Size([]),
        )
        # float32, matching the advertised Box exactly: a spec wider than the Box would
        # accept weights the Gymnasium surface and the parity fixtures both refuse.
        self.action_spec = Bounded(
            low=torch.as_tensor(action_space.low),
            high=torch.as_tensor(action_space.high),
            shape=(n,),
            dtype=torch.float32,
        )
        self.reward_spec = Unbounded(shape=(1,), dtype=resolved_dtype)
        # All three, not just "done": see the module docstring on why emitting the union
        # alone would be read as termination.
        self.done_spec = Composite(
            done=Binary(n=1, shape=(1,), dtype=torch.bool),
            terminated=Binary(n=1, shape=(1,), dtype=torch.bool),
            truncated=Binary(n=1, shape=(1,), dtype=torch.bool),
            shape=torch.Size([]),
        )

    # -- accessors ---------------------------------------------------------

    @property
    def symbols(self) -> list[str]:
        return list(self._gym.symbols)

    @property
    def gymnasium_env(self) -> SharpeArenaEnv:
        """The :class:`~sharpearena.gym.SharpeArenaEnv` this environment drives."""
        return self._gym

    @property
    def last_step_info(self) -> dict:
        """The engine's own ``info`` from the most recent step, or ``{}`` before one.

        The TensorDict layout carries the observation keys, the reward and the three
        flags, and nothing else. NAV and the rest of the engine's step info have no
        spec key here, so they are surfaced separately rather than quietly folded into
        the observation, where a policy would consume them as features.
        """
        return dict(self._last_step_info)

    @property
    def obs_dtype(self) -> Any:
        """The dtype observations and rewards cross the boundary at."""
        return self._obs_dtype

    @property
    def narrows_observations(self) -> bool:
        """True when this instance was asked for the lossy ``float32`` observations."""
        return self._obs_dtype is not torch.float64

    # -- internal helpers --------------------------------------------------

    def _observation_tensors(self, obs: dict) -> dict:
        return {
            key: torch.as_tensor(obs[key]).to(self._obs_dtype) for key in OBSERVATION_KEYS
        }

    def _flags(self, terminated: bool, truncated: bool) -> dict:
        return {
            "terminated": torch.tensor([bool(terminated)], dtype=torch.bool),
            "truncated": torch.tensor([bool(truncated)], dtype=torch.bool),
            "done": torch.tensor([bool(terminated) or bool(truncated)], dtype=torch.bool),
        }

    # -- EnvBase API -------------------------------------------------------

    def _reset(self, tensordict: Any = None, **kwargs: Any) -> Any:
        """Reset, applying a seed handed to :meth:`set_seed` since the last reset.

        Applying it here rather than in ``_set_seed`` matches the underlying adapter,
        where ``reset(seed=k)`` rebuilds the scenario; the seed is consumed once and the
        scenario then persists across further resets, which is Gymnasium's "seed once"
        paradigm and what the collector relies on.
        """
        self._last_step_info = {}
        if self._pending_seed is None:
            obs, _info = self._gym.reset()
        else:
            obs, _info = self._gym.reset(seed=self._pending_seed)
            self._pending_seed = None
        return TensorDict(
            {**self._observation_tensors(obs), **self._flags(False, False)},
            batch_size=torch.Size([]),
            device=self.device,
        )

    def _step(self, tensordict: Any) -> Any:
        action = tensordict.get("action").detach().cpu().numpy()
        obs, reward, terminated, truncated, info = self._gym.step(action)
        self._last_step_info = dict(info)
        return TensorDict(
            {
                **self._observation_tensors(obs),
                "reward": torch.tensor([float(reward)], dtype=self._obs_dtype),
                **self._flags(terminated, truncated),
            },
            batch_size=torch.Size([]),
            device=self.device,
        )

    def _set_seed(self, seed: Optional[int]) -> None:
        # 0.14 signature: returns None. Older examples return the next seed, which is a
        # type error here.
        self._pending_seed = None if seed is None else int(seed)


def equal_weight_policy(env: SharpeArenaTorchRLEnv):
    """A deterministic policy writing the equal-weight action into the tensordict.

    Not a learner. It exists so a collector round trip has a policy whose output is a
    function of nothing but the symbol count, which is what lets a replay of the
    collected actions be compared against the engine exactly.
    """
    _require_torchrl()
    n = len(env.symbols)
    weights = torch.full((n,), 1.0 / n, dtype=torch.float32)

    def policy(tensordict: Any) -> Any:
        tensordict.set("action", weights.clone())
        return tensordict

    return policy


__all__ = [
    "ENGINE_DTYPE_NAME",
    "OBSERVATION_KEYS",
    "SharpeArenaTorchRLEnv",
    "TorchRLUnavailable",
    "equal_weight_policy",
]
