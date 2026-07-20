"""An optimal-execution (VWAP/TWAP) scenario env for SharpeArena.

SharpeArena's core envs score **allocation**; this module covers the other half of a
trading-agent benchmark — **execution**. :class:`ExecutionEnv` is the canonical
optimal-execution MDP: liquidate a fixed parent order over a fixed window, minimizing
implementation shortfall against the period VWAP. This is where microstructure and
slippage bite, so it is a deliberately leak-free, deterministic-given-seed env.

**Private state.** ``leftover_order`` is the fraction of the parent order still to fill;
``leftover_time`` is the fraction of the execution window still open. Both are in the
observation alongside a trailing (causal) price window.

**Action.** A single child-order size in ``[0, 1]`` — the fraction of the *remaining*
order to execute this bar (``Box(0, 1, (1,))``). Sizing the remaining order (not the
parent) keeps the action bounded yet always able to finish: ``1.0`` clears whatever is
left.

**Reward (TradeMaster ``pd_environment`` shortfall form).** Per bar,
``reward = executed · (current_price / running_avg − 1)`` where ``executed`` is the
fraction of the *parent* order filled this bar and ``running_avg`` is the **point-in-time**
average price ``mean(prices[:t])`` — strictly causal, computed from prices *before* the
current bar, never the full-window mean. For a liquidation, selling above the running
average earns positive reward; the full-window VWAP is only the terminal scoring
benchmark (see :func:`execution_quality`), never fed back into the per-bar reward, so the
env stays leak-free.

**Completion process-check.** The window closing with the order unfilled is the execution
analogue of a tripped risk gate: any remainder is **force-liquidated at an unfavorable
price** and the terminal info flags ``completed = False``. A good policy fills the order
through its own child orders before the window ends.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import gymnasium as gym
from gymnasium import spaces

_EPS = 1e-9


class ExecutionEnv(gym.Env):
    """Single-asset optimal-execution MDP: liquidate a parent order over ``window`` bars.

    The price path is deterministic given ``seed`` (a seeded driftless log-price walk), or
    supply ``price_path`` to pin an exact path. ``reset(seed=k)`` is reproducible and
    distinct seeds give distinct paths. ``forced_penalty`` is the fractional price haircut
    applied to any remainder force-liquidated when the window closes.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        window: int = 20,
        *,
        seed: int = 0,
        history_len: int = 10,
        start_price: float = 100.0,
        sigma: float = 0.01,
        forced_penalty: float = 0.005,
        price_path: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__()
        if int(window) < 1:
            raise ValueError("window must be >= 1")
        self.window = int(window)
        self._seed = int(seed)
        self._history_len = int(history_len)
        self._start_price = float(start_price)
        self._sigma = float(sigma)
        self._forced_penalty = float(forced_penalty)
        self._fixed_path = (
            np.asarray(price_path, dtype=np.float64).reshape(-1)
            if price_path is not None
            else None
        )
        if self._fixed_path is not None and self._fixed_path.shape[0] != self.window:
            raise ValueError("price_path length must equal window")

        self._prices = self._build_path(self._seed)
        self._t = 0
        self._leftover_order = 1.0
        self._leftover_time = 1.0

        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "leftover_order": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float64),
                "leftover_time": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float64),
                "prices": spaces.Box(
                    0.0, np.inf, shape=(self._history_len,), dtype=np.float64
                ),
            }
        )

    # -- internal helpers --------------------------------------------------

    def _build_path(self, seed: int) -> np.ndarray:
        """The deterministic price path for ``seed``: a fixed override if supplied, else a
        seeded driftless log-price walk (a martingale, so no free directional edge)."""
        if self._fixed_path is not None:
            return self._fixed_path.copy()
        rng = np.random.default_rng(np.random.SeedSequence(int(seed)))
        shocks = rng.normal(0.0, self._sigma, size=self.window)
        return (self._start_price * np.exp(np.cumsum(shocks))).astype(np.float64)

    def _running_average(self, t: int) -> float:
        """Point-in-time average of prices *before* bar ``t`` (causal — uses only
        ``prices[:t]``, never the current or any future bar). Bar 0 has no prior price, so
        it anchors on its own price (a zero-shortfall reference for the first fill)."""
        if t == 0:
            return float(self._prices[0])
        return float(np.mean(self._prices[:t]))

    def _obs(self) -> dict[str, np.ndarray]:
        end = min(self._t, self.window - 1)
        idx = np.clip(np.arange(end - self._history_len + 1, end + 1), 0, self.window - 1)
        return {
            "leftover_order": np.array([self._leftover_order], dtype=np.float64),
            "leftover_time": np.array([self._leftover_time], dtype=np.float64),
            "prices": self._prices[idx].astype(np.float64),
        }

    # -- gymnasium API -----------------------------------------------------

    @property
    def prices(self) -> np.ndarray:
        return self._prices.copy()

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[dict[str, np.ndarray], dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._seed = int(seed)
            self._prices = self._build_path(self._seed)
        self._t = 0
        self._leftover_order = 1.0
        self._leftover_time = 1.0
        info = {"scenario_seed": self._seed, "window": self.window}
        return self._obs(), info

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict]:
        a = float(np.clip(np.asarray(action, dtype=np.float64).reshape(-1)[0], 0.0, 1.0))
        t = self._t
        price = float(self._prices[t])
        running_avg = self._running_average(t)

        child = a * self._leftover_order
        self._leftover_order = max(self._leftover_order - child, 0.0)
        reward = child * (price / running_avg - 1.0)

        is_last = t == self.window - 1
        forced = 0.0
        unfav = price * (1.0 - self._forced_penalty)
        if is_last and self._leftover_order > _EPS:
            forced = self._leftover_order
            reward += forced * (unfav / running_avg - 1.0)
            self._leftover_order = 0.0

        self._t += 1
        terminated = self._t >= self.window
        truncated = False
        self._leftover_time = max(self.window - self._t, 0) / self.window

        executed = child + forced
        exec_price = (child * price + forced * unfav) / executed if executed > _EPS else price
        info: dict[str, Any] = {
            "t": t,
            "price": price,
            "running_avg": running_avg,
            "child": child,
            "executed": executed,
            "exec_price": exec_price,
            "participation": executed,
            "forced_fraction": forced,
            "leftover_order": self._leftover_order,
            "scenario_seed": self._seed,
        }
        if terminated:
            info["completed"] = forced <= _EPS
            info["forced_remainder"] = forced
        return self._obs(), float(reward), terminated, truncated, info

    def render(self):  # pragma: no cover - no visual rendering
        return None

    def close(self):  # pragma: no cover
        return None


def execution_quality(trajectory: Sequence[dict]) -> dict:
    """Execution-quality diagnostics over a recorded :class:`ExecutionEnv` rollout.

    ``trajectory`` is the list of per-bar ``info`` dicts. Returns:

    * ``shortfall_bps`` — implementation shortfall (basis points) of the volume-weighted
      achieved fill price vs the **window VWAP** (the uniform-volume mean of bar prices,
      the terminal benchmark). Signed for a liquidation: negative means the order filled
      *above* VWAP (beat the benchmark), positive means it filled below.
    * ``participation_variance`` — variance of the per-bar participation rates (fraction of
      the parent filled each bar); ``0`` for a perfectly uniform (TWAP) schedule.
    * ``completion_fraction`` — fraction of the parent filled by the agent's own child
      orders (excludes any force-liquidated remainder); ``1.0`` iff the order completed
      before the window closed.
    """
    if not trajectory:
        return {
            "shortfall_bps": 0.0,
            "participation_variance": 0.0,
            "completion_fraction": 0.0,
        }
    bar_prices = np.array([float(r["price"]) for r in trajectory], dtype=np.float64)
    executed = np.array([float(r["executed"]) for r in trajectory], dtype=np.float64)
    exec_prices = np.array([float(r["exec_price"]) for r in trajectory], dtype=np.float64)
    child = np.array([float(r["child"]) for r in trajectory], dtype=np.float64)

    benchmark = float(np.mean(bar_prices))
    total = float(executed.sum())
    achieved = float((executed * exec_prices).sum() / total) if total > _EPS else benchmark
    shortfall_bps = (benchmark - achieved) / benchmark * 1e4 if benchmark > 0.0 else 0.0
    return {
        "shortfall_bps": float(shortfall_bps),
        "participation_variance": float(np.var(executed)),
        "completion_fraction": float(child.sum()),
    }


def twap_policy(obs: dict, window: int) -> np.ndarray:
    """Uniform child orders (TWAP): fill an equal slice of the *parent* each bar.

    Expressed as a fraction of the *remaining* order, the uniform slice is
    ``1 / remaining_bars``; ``remaining_bars`` is recovered from ``leftover_time`` and the
    horizon ``window``. On the final bar this is ``1.0`` (clear the remainder), so a TWAP
    rollout always completes.
    """
    leftover_time = float(np.asarray(obs["leftover_time"]).reshape(-1)[0])
    remaining = max(int(round(leftover_time * int(window))), 1)
    return np.array([1.0 / remaining], dtype=np.float32)


def immediate_policy(obs: dict) -> np.ndarray:
    """Liquidate the whole remaining order at once — fills entirely on bar 0."""
    return np.array([1.0], dtype=np.float32)


__all__ = [
    "ExecutionEnv",
    "execution_quality",
    "twap_policy",
    "immediate_policy",
]
