"""Forecast-quality calibrated eval axis — the FinPILOT "cheating forecaster".

A novel benchmark axis: inject a synthetic forecaster of EXACTLY KNOWN quality and
sweep its skill to measure how much of an agent's score is forecast-driven versus
structural. The trick is to interpolate between a noisy prediction and the realized
(ground-truth) next return,

    p_c = (1 - c) * p_noisy + c * truth,

and solve the single scalar ``c`` analytically so the forecast's out-of-sample R²
(measured against a trailing-mean baseline, Campbell-Thompson) equals a target value.
Sweeping the target over a grid yields a "forecast-skill dependence" curve.

Closed form. The out-of-sample R² is

    R²(c) = 1 - SS_res(c) / SS_tot ,  SS_tot = Σ (truth - baseline)² ,

and because ``truth - p_c = (1 - c) * (truth - p_noisy)`` identically (for ANY
``p_noisy``), the residual sum is exactly quadratic in ``c``:

    SS_res(c) = (1 - c)² * Σ (truth - p_noisy)²  =  (1 - c)² * A * SS_tot ,
    A = Σ (truth - p_noisy)² / SS_tot ,
    R²(c) = 1 - (1 - c)² * A .

Setting ``R²(c) = target`` and taking the root in ``[0, 1]`` gives

    c = 1 - sqrt((1 - target) / A) ,   clamped to [0, 1].

So the realized R² equals the requested target exactly (up to clamping at the
extremes). ``A`` does not depend on ``c``; the noise realization only shapes what the
forecast *looks* like, never the calibration.

LEAKAGE / EVAL-ONLY — READ THIS. The "truth" here is the realized next return, so
``calibrated_forecast`` and :class:`ForecastChannelObservation` deliberately use
lookahead. This is a CALIBRATION / EVAL construct. It MUST NOT be fed into any
scored, leak-free training rollout or the default benchmark observation. It exists
solely for the forecast-skill eval sweep and is gated behind explicit opt-in
(``ForecastChannelObservation`` adds ``obs["forecast"]`` only when wrapped on
purpose). Treat ``obs["forecast"]`` as a known future-leak surface.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .generalization import evaluate_seeds

MakeEnv = Callable[[int], gym.Env]
Policy = Callable[[dict], np.ndarray]


def trailing_mean_baseline(returns: np.ndarray) -> np.ndarray:
    """The no-skill baseline forecast: ``baseline[t] = mean(returns[:t])`` (the
    expanding trailing mean, the Campbell-Thompson OOS-R² reference). ``baseline[0]``
    is 0 (no history) and is excluded from scoring."""
    r = np.asarray(returns, dtype=np.float64).reshape(-1)
    n = r.size
    b = np.zeros(n, dtype=np.float64)
    if n > 1:
        b[1:] = np.cumsum(r)[:-1] / np.arange(1, n, dtype=np.float64)
    return b


def oos_r2(returns: np.ndarray, forecast: np.ndarray, *, min_history: int = 1) -> float:
    """Out-of-sample R² of ``forecast`` against the trailing-mean baseline over the
    region ``t >= min_history`` (where the baseline is defined). 0 means no skill, 1
    means perfect; negative means worse than the trailing mean."""
    r = np.asarray(returns, dtype=np.float64).reshape(-1)
    f = np.asarray(forecast, dtype=np.float64).reshape(-1)
    b = trailing_mean_baseline(r)
    lo = max(int(min_history), 1)
    e = r[lo:]
    ss_tot = float(np.sum((e - b[lo:]) ** 2))
    if ss_tot == 0.0:
        return 0.0
    ss_res = float(np.sum((e - f[lo:]) ** 2))
    return 1.0 - ss_res / ss_tot


def calibrated_forecast(
    returns: np.ndarray,
    target_r2: float,
    *,
    seed: int,
    baseline: str = "trailing_mean",
    noise_scale: float = 1.0,
    min_history: int = 1,
) -> np.ndarray:
    """A per-bar forecast of the next return whose realized out-of-sample R² (vs the
    trailing-mean baseline) equals ``target_r2``.

    EVAL-ONLY: uses lookahead (the realized ``returns`` are the "truth"). Never feed
    the output into a scored leak-free rollout — see the module docstring.

    The forecast blends a seed-driven noisy prediction toward the realized return,
    ``p_c = (1 - c) * p_noisy + c * truth``, with ``c`` solved from the closed form
    ``c = 1 - sqrt((1 - target_r2) / A)`` (clamped to ``[0, 1]``). ``target_r2 = 0``
    gives a no-skill forecast (R² ≈ 0; with ``noise_scale = 0`` it is exactly the
    baseline); ``target_r2 = 1`` gives the truth. Deterministic given ``seed``.
    """
    if baseline != "trailing_mean":
        raise ValueError("only baseline='trailing_mean' is supported")
    r = np.asarray(returns, dtype=np.float64).reshape(-1)
    n = r.size
    b = trailing_mean_baseline(r)
    if n <= int(min_history):
        return b
    lo = max(int(min_history), 1)

    rng = np.random.default_rng(int(seed))
    sigma = float(np.std(r))
    p_noisy = b + float(noise_scale) * sigma * rng.standard_normal(n)

    e = r[lo:]
    ss_tot = float(np.sum((e - b[lo:]) ** 2))
    if ss_tot == 0.0:
        return b
    a = float(np.sum((e - p_noisy[lo:]) ** 2)) / ss_tot
    target = float(np.clip(target_r2, 0.0, 1.0))
    if a <= 0.0:
        return r.copy()
    c = 1.0 - np.sqrt((1.0 - target) / a)
    c = float(np.clip(c, 0.0, 1.0))
    return (1.0 - c) * p_noisy + c * r


class ForecastChannelObservation(gym.Wrapper):
    """Add an ``obs["forecast"]`` channel of calibrated, KNOWN quality, for the
    forecast-skill eval sweep ONLY.

    LEAKAGE SURFACE — opt-in, eval-only. ``obs["forecast"][j]`` is a noisy version of
    symbol ``j``'s realized NEXT return (calibrated to ``target_r2``), so it encodes
    the future. It is added only when an env is explicitly wrapped in this class and
    is NOT part of the default leak-free observation or the scored benchmark. Use it
    solely inside :func:`forecast_skill_curve` / forecast-dependence analysis.

    Assumes the wrapped env's price path is exogenous (independent of the agent's
    actions, as for :class:`OpenOutcryEnv`): on ``reset`` it pre-rolls the env with a
    hold action to record the realized close path, calibrates a per-symbol forecast,
    then re-resets the same scenario to replay. Do not wrap a price-impacting env.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        target_r2: float,
        seed: int,
        noise_scale: float = 1.0,
        max_probe_steps: int = 4096,
    ) -> None:
        super().__init__(env)
        assert isinstance(env.observation_space, spaces.Dict), (
            "ForecastChannelObservation expects a Dict observation space"
        )
        assert "closes" in env.observation_space.spaces, (
            "ForecastChannelObservation requires a 'closes' obs key"
        )
        self._target_r2 = float(target_r2)
        self._seed = int(seed)
        self._noise_scale = float(noise_scale)
        self._max_probe_steps = int(max_probe_steps)
        self._n = int(env.observation_space.spaces["closes"].shape[-1])
        self._forecasts: list[np.ndarray] = []
        self._t = 0
        self.observation_space = spaces.Dict(
            {
                **env.observation_space.spaces,
                "forecast": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self._n,), dtype=np.float64
                ),
            }
        )

    def _hold_action(self) -> np.ndarray:
        return np.zeros(self.env.action_space.shape, dtype=self.env.action_space.dtype)

    def _build_forecasts(self, first_obs: dict) -> None:
        closes = [np.asarray(first_obs["closes"], dtype=np.float64).reshape(-1)]
        hold = self._hold_action()
        for _ in range(self._max_probe_steps):
            obs, _r, terminated, truncated, _i = self.env.step(hold)
            closes.append(np.asarray(obs["closes"], dtype=np.float64).reshape(-1))
            if bool(terminated) or bool(truncated):
                break
        prices = np.stack(closes, axis=0)  # (T+1, n_symbols)
        rets = prices[1:] / prices[:-1] - 1.0  # (T, n_symbols)
        children = np.random.SeedSequence(self._seed).spawn(self._n)
        self._forecasts = [
            calibrated_forecast(
                rets[:, j],
                self._target_r2,
                seed=int(children[j].generate_state(1)[0]),
                noise_scale=self._noise_scale,
            )
            for j in range(self._n)
        ]

    def _forecast_row(self) -> np.ndarray:
        out = np.zeros((self._n,), dtype=np.float64)
        for j, f in enumerate(self._forecasts):
            if self._t < f.size:
                out[j] = f[self._t]
        return out

    def reset(self, **kwargs: Any):
        obs, _info = self.env.reset(**kwargs)
        self._build_forecasts(obs)
        obs, info = self.env.reset()  # replay the same (exogenous) scenario
        self._t = 0
        return {**obs, "forecast": self._forecast_row()}, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._t += 1
        return (
            {**obs, "forecast": self._forecast_row()},
            reward,
            terminated,
            truncated,
            info,
        )


def forecast_skill_curve(
    make_env_for_seed: MakeEnv,
    seeds: Sequence[int],
    policy: Optional[Policy] = None,
    *,
    r2_grid: Sequence[float] = (0.001, 0.01, 0.05, 0.1, 0.2, 0.4),
    seed: int = 0,
    noise_scale: float = 1.0,
    max_steps: int = 512,
    n_trials: int = 0,
) -> dict[float, float]:
    """The forecast-skill dependence curve ``{target_r2: deflated_sharpe}``.

    For each target R² in ``r2_grid``, wrap every seed's env with a
    :class:`ForecastChannelObservation` of that calibrated quality, evaluate ``policy``
    over the seeds, and score the pooled return series with the SharpeBench kernel.
    The shape reveals how much of the agent's score is forecast-driven: FinPILOT found
    a threshold-like curve where very low skill levels (R² = 0.001 vs 0.01) are nearly
    indistinguishable, the score lifting only once forecast quality clears a knee.

    EVAL-ONLY: the wrapped observation is a deliberate future-leak surface (see
    :class:`ForecastChannelObservation`); this sweep is the only sanctioned use.
    """
    curve: dict[float, float] = {}
    for r2 in r2_grid:

        def make_wrapped(s: int, _r2: float = float(r2)) -> gym.Env:
            return ForecastChannelObservation(
                make_env_for_seed(s),
                target_r2=_r2,
                seed=seed + int(s),
                noise_scale=noise_scale,
            )

        result = evaluate_seeds(
            make_wrapped, seeds, policy, max_steps, n_trials=n_trials
        )
        curve[float(r2)] = float(result["deflated_sharpe"])
    return curve


__all__ = [
    "calibrated_forecast",
    "ForecastChannelObservation",
    "forecast_skill_curve",
    "oos_r2",
    "trailing_mean_baseline",
]
