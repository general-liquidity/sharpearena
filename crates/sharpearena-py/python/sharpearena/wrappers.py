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

**Persistence and evaluation.** A learner checkpoint that omits normalizer statistics
restores a policy trained on one input scale and feeds it another, and a normalizer that
keeps updating during evaluation lets the held-out bars shape what the policy sees. So
both normalizers carry a ``training`` flag and a JSON-serialisable ``state_dict`` /
``load_state_dict`` pair. With ``training=False`` no statistic moves, and the value
emitted for a bar is a pure function of that bar and the frozen statistics. Loading never
changes the flag, so an evaluator sets it explicitly. A frozen normalizer with fewer than
two samples refuses with :class:`NormalizerNotFittedError` instead of emitting zeros.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Iterable, Mapping
from typing import Any, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

_EPS = 1e-8

# Names the layout of a normalizer ``state_dict``. A change to that layout gets a new
# value here, and ``load_state_dict`` refuses any value it does not know.
_STATE_FORMAT = "sharpearena.normalizer.v1"


class NormalizerStateError(ValueError):
    """A normalizer state that does not fit the wrapper it was loaded into."""


class NormalizerNotFittedError(RuntimeError):
    """A frozen normalizer holds too little history to produce a scale."""


class _TrainingFlag:
    """The ``training`` attribute: ``True`` updates statistics, ``False`` freezes them."""

    _training: bool = True

    @property
    def training(self) -> bool:
        return self._training

    @training.setter
    def training(self, value: bool) -> None:
        # Truthiness would read the string "false" as training and keep updating.
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError("training must be a boolean")
        self._training = bool(value)


def _state_mapping(value: Any, keys: Iterable[str], where: str) -> Mapping:
    """``value`` as a mapping holding exactly ``keys``."""
    if not isinstance(value, Mapping):
        raise NormalizerStateError(f"{where} must be a mapping")
    expected, actual = set(keys), set(value)
    if expected != actual:
        raise NormalizerStateError(
            f"{where} keys differ: missing {sorted(map(str, expected - actual))}, "
            f"unexpected {sorted(map(str, actual - expected))}"
        )
    return value


def _state_body(state: Any, wrapper: str, keys: Iterable[str]) -> Mapping:
    """Check a state's header and return it, holding exactly ``keys`` besides the header."""
    state = _state_mapping(state, ("format", "wrapper", *keys), "state")
    if state["format"] != _STATE_FORMAT:
        raise NormalizerStateError(f"unknown state format {state['format']!r}")
    if state["wrapper"] != wrapper:
        raise NormalizerStateError(f"state is for {state['wrapper']!r}, not {wrapper!r}")
    return state


def _state_array(value: Any, shape: tuple[int, ...], where: str) -> np.ndarray:
    """An owned, finite float64 array of exactly ``shape``."""
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise NormalizerStateError(f"{where} is not a rectangular array") from exc
    if raw.dtype.kind not in "fiu":
        raise NormalizerStateError(f"{where} must be numeric")
    if raw.shape != shape:
        raise NormalizerStateError(f"{where} shape must be {shape}, got {raw.shape}")
    out = raw.astype(np.float64, copy=True)
    if not np.isfinite(out).all():
        raise NormalizerStateError(f"{where} must be finite")
    return out


def _state_variation(value: Any, shape: tuple[int, ...], where: str) -> np.ndarray:
    """A sum of squared deviations, which is never negative."""
    out = _state_array(value, shape, where)
    if (out < 0.0).any():
        raise NormalizerStateError(f"{where} must not be negative")
    return out


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

    def to_state(self) -> dict:
        return {
            "count": self.count,
            "mean": None if self.mean is None else self.mean.tolist(),
            "m2": None if self.m2 is None else self.m2.tolist(),
        }

    @classmethod
    def from_state(cls, state: Any, shape: tuple[int, ...], where: str) -> "_Welford":
        state = _state_mapping(state, ("count", "mean", "m2"), where)
        count = state["count"]
        if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)):
            raise NormalizerStateError(f"{where}.count must be an integer")
        if count < 0:
            raise NormalizerStateError(f"{where}.count must not be negative")
        out = cls()
        out.count = int(count)
        if count == 0:
            if state["mean"] is not None or state["m2"] is not None:
                raise NormalizerStateError(f"{where} holds statistics for zero samples")
            return out
        out.mean = _state_array(state["mean"], shape, f"{where}.mean")
        out.m2 = _state_variation(state["m2"], shape, f"{where}.m2")
        return out


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


class CausalNormalizeObservation(_TrainingFlag, gym.Wrapper):
    """Z-score each observation feature using **only** bars at or before ``t``.

    Leak-free by construction: the value emitted for bar ``t`` is computed from the
    running mean/std of bars ``0..t-1`` (the accumulator is updated with bar ``t``
    *after* emission). The first observation of the run has no history and is
    emitted as zeros. Statistics persist across episodes (a longer history is a
    better estimator, and nothing about a *past* episode is a future leak).

    With ``training=False`` the statistics stay as they are, and every bar, including
    the first of a run, is scored against them. See the module docstring.
    """

    def __init__(self, env: gym.Env, *, training: bool = True) -> None:
        super().__init__(env)
        self.training = training
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
            if self._training:
                st.update(v)  # fold bar t in only AFTER it has been emitted
        return out

    def _require_fitted_when_frozen(self) -> None:
        if self._training:
            return
        for k, st in self._stats.items():
            if st.std() is None:
                raise NormalizerNotFittedError(
                    f"frozen CausalNormalizeObservation has {st.count} samples of "
                    f"{k!r}; it needs at least 2"
                )

    def state_dict(self) -> dict:
        """The running statistics as JSON-serialisable data, sharing no memory."""
        return {
            "format": _STATE_FORMAT,
            "wrapper": "CausalNormalizeObservation",
            "features": {k: st.to_state() for k, st in self._stats.items()},
        }

    def load_state_dict(self, state: Any) -> None:
        """Replace the statistics, or raise and leave them all untouched."""
        body = _state_body(state, "CausalNormalizeObservation", ("features",))
        features = _state_mapping(body["features"], self._stats, "features")
        spaces_by_key = self.observation_space.spaces
        self._stats = {
            k: _Welford.from_state(features[k], spaces_by_key[k].shape, f"features[{k!r}]")
            for k in self._stats
        }

    def reset(self, **kwargs: Any):
        self._require_fitted_when_frozen()
        obs, info = self.env.reset(**kwargs)
        return self._normalize(obs), info

    def step(self, action):
        self._require_fitted_when_frozen()
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._normalize(obs), reward, terminated, truncated, info


class CausalNormalizeReward(_TrainingFlag, gym.Wrapper):
    """Scale reward by the running std of **past** rewards in the current episode.

    Like the observation variant, the divisor at step ``t`` is computed from
    rewards ``0..t-1`` only; the current reward is folded in afterward. The
    accumulator resets on every episode boundary — ``terminated`` *or*
    ``truncated`` — so the scale reflects the live episode's reward distribution
    and never bleeds across the boundary.

    That boundary reset would leave a checkpoint taken between episodes with nothing to
    freeze, so the last accumulator it discards with at least two samples is kept as
    ``retired``. It never affects a training-mode reward. With ``training=False`` every
    reward is divided by one fixed scale: the std of ``retired`` if there is one,
    otherwise of the live accumulator. The scale crosses episode boundaries unchanged.
    """

    def __init__(self, env: gym.Env, *, training: bool = True) -> None:
        super().__init__(env)
        self.training = training
        self._stats = _Welford()
        self._retired = _Welford()

    def _retire_live_stats(self) -> None:
        if self._stats.std() is not None:
            self._retired = self._stats
        self._stats = _Welford()

    def _frozen_scale(self) -> float:
        std = self._retired.std()
        if std is None:
            std = self._stats.std()
        if std is None:
            raise NormalizerNotFittedError(
                "frozen CausalNormalizeReward has no accumulator with 2 samples"
            )
        return float(np.ravel(std)[0])

    def state_dict(self) -> dict:
        """The accumulators as JSON-serialisable data, sharing no memory."""
        return {
            "format": _STATE_FORMAT,
            "wrapper": "CausalNormalizeReward",
            "live": self._stats.to_state(),
            "retired": self._retired.to_state(),
        }

    def load_state_dict(self, state: Any) -> None:
        """Replace the accumulators, or raise and leave them both untouched."""
        body = _state_body(state, "CausalNormalizeReward", ("live", "retired"))
        live = _Welford.from_state(body["live"], (1,), "live")
        retired = _Welford.from_state(body["retired"], (1,), "retired")
        self._stats, self._retired = live, retired

    def reset(self, **kwargs: Any):
        if self._training:
            self._retire_live_stats()
        else:
            self._frozen_scale()
        return self.env.reset(**kwargs)

    def step(self, action):
        if not self._training:
            scale = self._frozen_scale()
            obs, reward, terminated, truncated, info = self.env.step(action)
            return obs, float(np.float64(reward) / (scale + _EPS)), terminated, truncated, info
        obs, reward, terminated, truncated, info = self.env.step(action)
        r = np.float64(reward)
        std = self._stats.std()
        scaled = r if std is None else r / (float(np.ravel(std)[0]) + _EPS)
        self._stats.update(np.asarray([r]))
        if bool(terminated) or bool(truncated):
            self._retire_live_stats()
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

    Drawdown compounds emitted rewards as simple returns from opening equity
    one; shaped rewards need not reproduce native NAV drawdown. These summaries
    are diagnostics, not SharpeBench scores or ranking inputs.
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
        equity = np.concatenate(([1.0], np.cumprod(1.0 + rewards)))
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
    "NormalizerNotFittedError",
    "NormalizerStateError",
    "FrameStack",
    "RecordEpisodeStatistics",
]
