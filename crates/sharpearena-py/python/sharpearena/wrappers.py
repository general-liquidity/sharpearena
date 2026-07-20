"""Point-in-time-SAFE composable wrappers over :class:`~sharpearena.gym.SharpeArenaEnv`.

The stdlib :class:`gymnasium.wrappers.NormalizeObservation` / ``NormalizeReward``
maintain running statistics, but their *purpose* — better-conditioned features —
makes the leak easy to introduce: any normalization that lets information from bar
``t+k`` influence the value emitted at bar ``t`` silently contaminates a
point-in-time backtest. Offline z-scoring over the full series is the textbook
version of that bug. The causal variants here update their running statistics with
**strictly past** bars only (``count`` at emission time = bars ``0..t-1``); bar
``t`` is folded in *after* it has been emitted, so no future bar can ever reach
back. That is the property the conformance tests assert by prefix-equality.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

_EPS = 1e-8


class _Welford:
    """Elementwise online mean/variance (Welford), one accumulator per feature."""

    __slots__ = ("count", "mean", "m2")

    def __init__(self) -> None:
        self.count = 0
        self.mean: Optional[np.ndarray] = None
        self.m2: Optional[np.ndarray] = None

    def std(self) -> Optional[np.ndarray]:
        if self.count < 2 or self.m2 is None:
            return None
        return np.sqrt(self.m2 / self.count)

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        if self.mean is None:
            self.mean = np.zeros_like(x)
            self.m2 = np.zeros_like(x)
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (x - self.mean)


class TimeLimit(gym.Wrapper):
    """Truncate the episode at ``max_episode_steps`` steps.

    Composes with the env's own ``truncated``: either trigger truncates. The cap is
    counted from the most recent ``reset``.
    """

    def __init__(self, env: gym.Env, max_episode_steps: int) -> None:
        super().__init__(env)
        self._max = int(max_episode_steps)
        self._elapsed = 0

    def reset(self, **kwargs: Any):
        self._elapsed = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._elapsed += 1
        if self._elapsed >= self._max:
            truncated = True
        return obs, reward, bool(terminated), bool(truncated), info


class CausalNormalizeObservation(gym.Wrapper):
    """Z-score each observation feature using **only** bars at or before ``t``.

    Leak-free by construction: the value emitted for bar ``t`` is computed from the
    running mean/std of bars ``0..t-1`` (the accumulator is updated with bar ``t``
    *after* emission). The first observation of the run has no history and is
    emitted as zeros. Statistics persist across episodes (a longer history is a
    better estimator, and nothing about a *past* episode is a future leak).
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        assert isinstance(env.observation_space, spaces.Dict), (
            "CausalNormalizeObservation expects a Dict observation space"
        )
        self._stats: dict[str, _Welford] = {
            k: _Welford() for k in env.observation_space.spaces
        }
        self.observation_space = spaces.Dict(
            {
                k: spaces.Box(low=-np.inf, high=np.inf, shape=s.shape, dtype=np.float64)
                for k, s in env.observation_space.spaces.items()
            }
        )

    def _normalize(self, obs: dict) -> dict:
        out: dict[str, np.ndarray] = {}
        for k, v in obs.items():
            v = np.asarray(v, dtype=np.float64)
            st = self._stats[k]
            std = st.std()
            if st.count < 1 or std is None:
                out[k] = np.zeros_like(v)
            else:
                out[k] = (v - st.mean) / (std + _EPS)
            st.update(v)  # fold bar t in only AFTER it has been emitted
        return out

    def reset(self, **kwargs: Any):
        obs, info = self.env.reset(**kwargs)
        return self._normalize(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._normalize(obs), reward, terminated, truncated, info


class CausalNormalizeReward(gym.Wrapper):
    """Scale reward by the running std of **past** rewards in the current episode.

    Like the observation variant, the divisor at step ``t`` is computed from
    rewards ``0..t-1`` only; the current reward is folded in afterward. The
    accumulator resets on every episode boundary — ``terminated`` *or*
    ``truncated`` — so the scale reflects the live episode's reward distribution
    and never bleeds across the boundary.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self._stats = _Welford()

    def reset(self, **kwargs: Any):
        self._stats = _Welford()
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        r = np.float64(reward)
        std = self._stats.std()
        scaled = r if std is None else r / (float(np.ravel(std)[0]) + _EPS)
        self._stats.update(np.asarray([r]))
        if bool(terminated) or bool(truncated):
            self._stats = _Welford()
        return obs, float(scaled), terminated, truncated, info


class FrameStack(gym.Wrapper):
    """Stack the last ``num_stack`` observations along a new leading axis.

    Initial frames are padded by repeating the first observation of the episode, so
    the stacked observation has a fixed shape from step 0.
    """

    def __init__(self, env: gym.Env, num_stack: int) -> None:
        super().__init__(env)
        assert num_stack >= 1, "num_stack must be >= 1"
        assert isinstance(env.observation_space, spaces.Dict), (
            "FrameStack expects a Dict observation space"
        )
        self._k = int(num_stack)
        self._frames: dict[str, deque] = {
            k: deque(maxlen=self._k) for k in env.observation_space.spaces
        }
        self.observation_space = spaces.Dict(
            {
                k: spaces.Box(
                    low=np.broadcast_to(s.low, (self._k, *s.shape)).copy(),
                    high=np.broadcast_to(s.high, (self._k, *s.shape)).copy(),
                    shape=(self._k, *s.shape),
                    dtype=s.dtype,
                )
                for k, s in env.observation_space.spaces.items()
            }
        )

    def _stacked(self) -> dict:
        return {k: np.stack(list(d), axis=0) for k, d in self._frames.items()}

    def reset(self, **kwargs: Any):
        obs, info = self.env.reset(**kwargs)
        for k, v in obs.items():
            d = self._frames[k]
            d.clear()
            for _ in range(self._k):
                d.append(np.asarray(v))
        return self._stacked(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        for k, v in obs.items():
            self._frames[k].append(np.asarray(v))
        return self._stacked(), reward, terminated, truncated, info


class RecordEpisodeStatistics(gym.Wrapper):
    """Inject per-episode statistics into ``info["episode"]`` at episode end.

    Adds the gymnasium-standard ``r`` (return), ``l`` (length), ``t`` (wall-clock
    seconds) plus trading-specific ``sharpe`` and ``max_drawdown`` computed from the
    episode's per-step reward series, so SharpeBench / verifiers can read summary
    stats straight off ``info`` without re-deriving them from a return log.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self._rewards: list[float] = []
        self._t0 = 0.0

    def reset(self, **kwargs: Any):
        self._rewards = []
        self._t0 = time.perf_counter()
        return self.env.reset(**kwargs)

    @staticmethod
    def _sharpe(rewards: np.ndarray) -> float:
        if rewards.size < 2:
            return 0.0
        sd = float(np.std(rewards))
        return float(np.mean(rewards) / sd) if sd > 0.0 else 0.0

    @staticmethod
    def _max_drawdown(rewards: np.ndarray) -> float:
        if rewards.size == 0:
            return 0.0
        equity = np.cumprod(1.0 + rewards)
        peak = np.maximum.accumulate(equity)
        return float(np.max((peak - equity) / peak))

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._rewards.append(float(reward))
        if bool(terminated) or bool(truncated):
            r = np.asarray(self._rewards, dtype=np.float64)
            info = dict(info)
            info["episode"] = {
                "r": float(r.sum()),
                "l": int(r.size),
                "t": float(time.perf_counter() - self._t0),
                "sharpe": self._sharpe(r),
                "max_drawdown": self._max_drawdown(r),
            }
        return obs, reward, terminated, truncated, info


__all__ = [
    "TimeLimit",
    "CausalNormalizeObservation",
    "CausalNormalizeReward",
    "FrameStack",
    "RecordEpisodeStatistics",
]
