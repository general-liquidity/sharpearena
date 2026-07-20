"""Discrete-action adapter over :class:`~sharpearena.gym.SharpeArenaEnv`.

SharpeArena's canonical action is a continuous per-symbol target-weight ``Box``.
Value-based learners (DQN and the wider small-gym literature) need a discrete
head, so :class:`DiscreteAction` is a thin, deterministic ``Discrete``/
``MultiDiscrete`` -> target-weight adapter at the gym boundary. The canonical
contract is unchanged: every discrete action is mapped to the exact weight vector
the underlying env already consumes.

Multi-symbol default is **per-symbol** ``MultiDiscrete`` (one independent sub-action
per symbol); a single-symbol env collapses to a flat ``Discrete``.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

_SCHEMES = ("long_flat_short", "binned")


class DiscreteAction(gym.ActionWrapper):
    """Map a discrete action onto the env's target-weight ``Box``.

    Parameters
    ----------
    env:
        An env whose ``action_space`` is a per-symbol target-weight ``Box``.
    scheme:
        ``"long_flat_short"`` exposes 3 levels per symbol
        (``-max_weight`` / ``0`` / ``+max_weight``); when shorting is disabled the
        short level is dropped, leaving 2 (``0`` / ``+max_weight``).
        ``"binned"`` exposes ``n_bins`` levels per symbol spread monotonically
        across ``[low, high]``.
    n_bins:
        Required (>= 2) for ``scheme="binned"``; ignored otherwise.

    The exposed ``action_space`` is ``Discrete`` for a single-symbol env and
    ``MultiDiscrete`` (one entry per symbol) for multi-symbol.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        scheme: str = "long_flat_short",
        n_bins: int | None = None,
    ) -> None:
        super().__init__(env)
        if scheme not in _SCHEMES:
            raise ValueError(f"scheme must be one of {_SCHEMES}")
        base = env.action_space
        if not isinstance(base, spaces.Box):
            raise TypeError("DiscreteAction expects a Box target-weight action space")

        low = np.broadcast_to(base.low, base.shape).astype(np.float64)
        high = np.broadcast_to(base.high, base.shape).astype(np.float64)
        n = base.shape[0]
        allow_short = bool(np.any(low < 0.0))

        if scheme == "long_flat_short":
            levels = [
                ([low[i], 0.0, high[i]] if allow_short else [0.0, high[i]])
                for i in range(n)
            ]
        else:
            if n_bins is None or int(n_bins) < 2:
                raise ValueError("scheme='binned' requires n_bins >= 2")
            n_bins = int(n_bins)
            levels = [list(np.linspace(low[i], high[i], n_bins)) for i in range(n)]

        # (n_symbols, levels_per_symbol) lookup; symbols are homogeneous here so the
        # row count is constant, but the table stays per-symbol to honour ragged bounds.
        self._table = np.asarray(levels, dtype=np.float32)
        counts = [len(row) for row in levels]

        if n == 1:
            self.action_space: spaces.Space = spaces.Discrete(counts[0])
        else:
            self.action_space = spaces.MultiDiscrete(counts)

    def action(self, action) -> np.ndarray:
        idx = np.asarray(action, dtype=np.int64).reshape(-1)
        rows = np.arange(self._table.shape[0])
        return self._table[rows, idx].astype(np.float32)


__all__ = ["DiscreteAction"]
