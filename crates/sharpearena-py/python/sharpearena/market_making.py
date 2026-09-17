"""Avellaneda-Stoikov market-making env with a closed-form reference baseline.

A canonical single-asset market-making environment: the agent quotes a bid and an ask
around a mid that follows a seeded arithmetic random walk, earns the spread on filled
market-order arrivals, and carries inventory risk (a squared running penalty plus a
forced terminal liquidation at an unfavorable price). Shipped alongside is the
Avellaneda-Stoikov *closed-form* quoting policy as a committed reference baseline, so a
learner can be scored on **regret versus a fixed analytical reference** rather than on
relative ranking alone. The closed form is the source model's asymptotic approximation,
and it is not proven optimal for this env's reward (which adds an inventory cap, a running
inventory penalty and a terminal liquidation charge the source model lacks), so the metric
is regret against a reference, not against a proven optimum.

The model (Avellaneda & Stoikov 2008):

* Mid follows arithmetic Brownian motion ``dS = sigma dW``.
* Per step, market orders arrive ``~Poisson(lambda)`` on each side; an arrival fills the
  maker's quote at depth ``delta`` with probability ``exp(-kappa * delta)``.
* The maker's reservation (indifference) price skews with inventory ``q``:
  ``r = s - q * gamma * sigma**2 * tau`` (``tau`` = remaining time).
* The model's total spread is ``gamma*sigma**2*tau + (2/gamma)*ln(1 + gamma/kappa)``,
  i.e. a half-spread ``delta* = gamma*sigma**2*tau/2 + (1/gamma)*ln(1+gamma/kappa)``
  quoted symmetrically around ``r`` — so the inventory-skewed quote depths are
  ``bid_depth = delta* + q*gamma*sigma**2*tau`` and ``ask_depth = delta* - q*gamma*sigma**2*tau``.

Pure Python (numpy): every draw comes from a seeded ``np.random.default_rng`` so an episode
is reproducible given its seed. This is a NEW training surface, deterministic-given-seed in
Python — NOT part of the cross-runtime byte-identical scored core.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, replace
from typing import Callable, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

Action = np.ndarray
Policy = Callable[[dict], Action]


@dataclass(frozen=True)
class MMParams:
    """Frozen Avellaneda-Stoikov market-making parameters.

    ``sigma``/``gamma``/``kappa`` are the A-S volatility, risk-aversion and order-book
    depth-decay constants; ``arrival_rate`` is the per-side Poisson market-order intensity
    per unit time; ``n_steps``/``dt`` set the horizon (total time ``n_steps*dt``). The
    closed-form reference policy is a pure function of these, so the same params instance
    feeds both the env and :func:`closed_form_reference_policy`.
    """

    sigma: float = 2.0
    gamma: float = 0.1
    kappa: float = 1.5
    arrival_rate: float = 140.0
    n_steps: int = 200
    dt: float = 0.005
    s0: float = 100.0
    inventory_cap: int = 50
    phi: float = 0.0015
    terminal_liq_penalty: float = 0.2
    max_depth: float = 5.0
    tick_size: float = 0.01
    initial_cash: float = 0.0

    @property
    def horizon(self) -> float:
        return self.n_steps * self.dt


def _reference_depths(q: float, tau: float, p: MMParams) -> tuple[float, float]:
    """Avellaneda-Stoikov closed-form ``(bid_depth, ask_depth)`` at inventory ``q``, remaining
    time ``tau``. Half-spread widens with time-to-go and skews by inventory; depths are
    clipped to the env's quotable ``[0, max_depth]`` band."""
    skew = q * p.gamma * p.sigma**2 * tau
    half = 0.5 * p.gamma * p.sigma**2 * tau + (1.0 / p.gamma) * math.log1p(p.gamma / p.kappa)
    bid = min(max(half + skew, 0.0), p.max_depth)
    ask = min(max(half - skew, 0.0), p.max_depth)
    return bid, ask


class MarketMakingEnv(gym.Env):
    """Single-asset Avellaneda-Stoikov market-making env.

    State: inventory ``q``, cash, mid price (a seeded arithmetic random walk) and time
    remaining. Action is ``(bid_depth, ask_depth)`` quote distances from mid (price units,
    a multiple of ``tick_size``). Each step draws Poisson market-order arrivals per side and
    fills the maker's quote with probability ``exp(-kappa*depth)``, updating inventory under
    a hard cap ``+/-inventory_cap``. Reward is the mark-to-mid value change minus a running
    squared-inventory penalty ``phi*q**2``; the terminal step force-liquidates remaining
    inventory at an unfavorable price.

    ``step`` also splits each reward into four components in ``info``, each signed as its
    contribution to the reward, so they sum to it up to floating-point rounding:

    * ``spread_capture``: ``ask_fills*ask_depth + bid_fills*bid_depth``, the quoted depth
      earned on this step's fills.
    * ``inventory_pnl``: ``q*(mid_after - mid_before)``, the post-fill inventory marked
      to the mid move.
    * ``inventory_penalty``: ``-phi*q**2`` on the post-fill inventory.
    * ``liquidation_cost``: ``-abs(q)*terminal_liq_penalty`` on the terminal step when
      inventory is left, ``0.0`` otherwise.

    The first two are an exact algebraic split of the mark-to-mid value change (equal to it
    up to floating-point rounding), because the fill cash
    ``ask_fills*(mid+ask_depth) - bid_fills*(mid-bid_depth)`` and the inventory change at the
    pre-move mid cancel. This follows the reward decomposition in Fernández Vicente's
    "Market Making Strategies with Reinforcement Learning" (dissertation, 2025, Eqs. 4.1 to
    4.3: spread earnings plus inventory times the price change). This env has no hedging cost
    (Eq. 4.4) and no adaptive inventory penalty (Eq. 4.5); its own running penalty and
    terminal liquidation charge take those places. The split is a diagnostic: nothing in
    SharpeArena or SharpeBench scores on it.
    """

    metadata = {"render_modes": []}

    def __init__(self, params: Optional[MMParams] = None, **overrides) -> None:
        super().__init__()
        self.params = replace(params or MMParams(), **overrides)
        p = self.params
        self.action_space = spaces.Box(
            low=0.0, high=p.max_depth, shape=(2,), dtype=np.float32
        )
        self.observation_space = spaces.Dict(
            {
                "inventory": spaces.Box(
                    low=-p.inventory_cap, high=p.inventory_cap, shape=(1,), dtype=np.float64
                ),
                "mid": spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float64),
                "time_remaining": spaces.Box(
                    low=0.0, high=p.horizon, shape=(1,), dtype=np.float64
                ),
                "cash": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64),
            }
        )
        self._rng = np.random.default_rng(0)
        self._q = 0
        self._cash = p.initial_cash
        self._mid = p.s0
        self._t = 0

    # -- internal helpers --------------------------------------------------

    def _obs(self) -> dict:
        p = self.params
        return {
            "inventory": np.array([float(self._q)], dtype=np.float64),
            "mid": np.array([self._mid], dtype=np.float64),
            "time_remaining": np.array([(p.n_steps - self._t) * p.dt], dtype=np.float64),
            "cash": np.array([self._cash], dtype=np.float64),
        }

    def _value(self) -> float:
        return self._cash + self._q * self._mid

    # -- gymnasium API -----------------------------------------------------

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[dict, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))
        p = self.params
        self._q = 0
        self._cash = p.initial_cash
        self._mid = p.s0
        self._t = 0
        info = {"value": self._value(), "inventory": self._q}
        return self._obs(), info

    def step(self, action: Action) -> tuple[dict, float, bool, bool, dict]:
        p = self.params
        bid_depth, ask_depth = (
            float(np.clip(action[0], 0.0, p.max_depth)),
            float(np.clip(action[1], 0.0, p.max_depth)),
        )
        value_before = self._value()

        # Market-order arrivals per side, then a fill draw per arrival at prob exp(-k*depth).
        lam = p.arrival_rate * p.dt
        n_buy_orders = int(self._rng.poisson(lam))   # market BUYs lift the maker's ask
        n_sell_orders = int(self._rng.poisson(lam))  # market SELLs hit the maker's bid
        ask_fills = int(self._rng.binomial(n_buy_orders, math.exp(-p.kappa * ask_depth)))
        bid_fills = int(self._rng.binomial(n_sell_orders, math.exp(-p.kappa * bid_depth)))

        # Hard inventory cap: cannot sell below -cap or buy above +cap.
        ask_fills = min(ask_fills, self._q + p.inventory_cap)
        bid_fills = min(bid_fills, p.inventory_cap - self._q)

        ask_price = self._mid + ask_depth
        bid_price = self._mid - bid_depth
        self._cash += ask_fills * ask_price - bid_fills * bid_price
        self._q += bid_fills - ask_fills

        # Mid advances as arithmetic Brownian motion.
        mid_before = self._mid
        self._mid += p.sigma * math.sqrt(p.dt) * float(self._rng.standard_normal())
        self._t += 1

        # The reward arithmetic is unchanged in value and order, so the frozen F2 regrets
        # reproduce bit for bit; the split below is computed beside it.
        value_after = self._value()
        running_penalty = p.phi * self._q**2
        reward = (value_after - value_before) - running_penalty
        spread_capture = ask_fills * ask_depth + bid_fills * bid_depth
        inventory_pnl = self._q * (self._mid - mid_before)

        terminated = self._t >= p.n_steps
        liquidated = 0.0
        liquidation_cost = 0.0
        if terminated and self._q != 0:
            # Forced liquidation crosses the spread at an unfavorable price.
            sign = 1.0 if self._q > 0 else -1.0
            liq_price = self._mid - sign * p.terminal_liq_penalty
            proceeds = self._q * liq_price
            liquidation_cost = proceeds - self._q * self._mid
            reward += liquidation_cost
            self._cash += proceeds
            liquidated = float(self._q)
            self._q = 0

        info = {
            "value": self._value(),
            "inventory": self._q,
            "bid_fills": bid_fills,
            "ask_fills": ask_fills,
            "mid": self._mid,
            "liquidated": liquidated,
            "spread_capture": spread_capture,
            "inventory_pnl": inventory_pnl,
            "inventory_penalty": -running_penalty,
            "liquidation_cost": liquidation_cost,
        }
        return self._obs(), float(reward), bool(terminated), False, info

    def render(self):  # pragma: no cover - no visual rendering
        return None

    def close(self):  # pragma: no cover
        return None


# -- policies ---------------------------------------------------------------


def closed_form_reference_policy(env_params: MMParams) -> Policy:
    """The Avellaneda-Stoikov closed-form reference quoting policy as a ``(obs)->action``
    callable, the fixed analytical reference this benchmark scores regret against. It is
    the source model's asymptotic closed form, not a proven optimum for this env's reward.

    Reservation price ``r = s - q*gamma*sigma**2*tau`` and the model half-spread
    ``delta* = gamma*sigma**2*tau/2 + (1/gamma)*ln(1+gamma/kappa)`` give inventory-skewed
    depths ``bid = delta*+q*gamma*sigma**2*tau``, ``ask = delta*-q*gamma*sigma**2*tau``.
    """
    p = env_params

    def policy(obs: dict) -> Action:
        q = float(np.asarray(obs["inventory"]).reshape(-1)[0])
        tau = float(np.asarray(obs["time_remaining"]).reshape(-1)[0])
        bid, ask = _reference_depths(q, tau, p)
        return np.array([bid, ask], dtype=np.float32)

    return policy


def analytically_optimal_policy(env_params: MMParams) -> Policy:
    """Deprecated alias for :func:`closed_form_reference_policy`.

    The old name asserted an optimality this policy does not have; it is kept so
    existing call sites keep working, and it emits a ``DeprecationWarning``.
    """
    warnings.warn(
        "analytically_optimal_policy is deprecated; the policy is a closed-form "
        "reference, not a proven optimum. Use closed_form_reference_policy.",
        DeprecationWarning,
        stacklevel=2,
    )
    return closed_form_reference_policy(env_params)


def fixed_spread_policy(half_spread: float) -> Policy:
    """A naive symmetric fixed-spread maker: quotes ``half_spread`` on both sides, ignoring
    inventory and time-to-go. The candidate policy whose regret against the closed-form
    reference the F2 sweep measures."""

    def policy(obs: dict) -> Action:
        return np.array([half_spread, half_spread], dtype=np.float32)

    return policy


# -- regret metric ----------------------------------------------------------


_PNL_COMPONENTS = ("spread_capture", "inventory_pnl", "inventory_penalty", "liquidation_cost")


class UnpairedMidPathError(ValueError):
    """:func:`mm_regret` refused because the two arms did not see the same mid path.

    ``seed`` is the first episode seed whose paths differ, ``step`` the zero-based index of
    the first ``env.step`` call after which the two mids differ, and ``reference_mid`` /
    ``candidate_mid`` the two mids at that step.
    """

    def __init__(
        self, seed: int, step: int, reference_mid: float, candidate_mid: float
    ) -> None:
        self.seed = seed
        self.step = step
        self.reference_mid = reference_mid
        self.candidate_mid = candidate_mid
        super().__init__(
            f"mid paths differ at seed {seed}, step {step}: reference mid "
            f"{reference_mid!r}, candidate mid {candidate_mid!r}; the regret would be "
            f"unpaired"
        )


def _rollout(
    env: MarketMakingEnv, policy: Policy, seed: int
) -> tuple[float, list[float], dict[str, float]]:
    """One seeded episode: its summed reward, its post-step mid path and the episode sum of
    each reward component."""
    obs, _ = env.reset(seed=seed)
    total = 0.0
    mids: list[float] = []
    split = dict.fromkeys(_PNL_COMPONENTS, 0.0)
    while True:
        obs, reward, terminated, truncated, info = env.step(policy(obs))
        total += reward
        mids.append(info["mid"])
        for key in _PNL_COMPONENTS:
            split[key] += info[key]
        if terminated or truncated:
            return total, mids, split


def _check_paired(seed: int, reference_mids: list[float], candidate_mids: list[float]) -> None:
    for step, (ref_mid, cand_mid) in enumerate(zip(reference_mids, candidate_mids)):
        if ref_mid != cand_mid:
            raise UnpairedMidPathError(seed, step, ref_mid, cand_mid)


def mm_regret(
    policy: Policy,
    *,
    params: Optional[MMParams] = None,
    n_episodes: int = 16,
    seed_base: int = 0,
) -> float:
    """Mean reward gap between :func:`closed_form_reference_policy` and ``policy`` over
    ``n_episodes`` seeded episodes, the regret-versus-reference metric. Both policies run
    on the *same* seeds, so the reference scores ~0 regret against itself and a worse
    policy scores a positive gap; the zero point is the reference, not a proven optimum.

    The pairing is checked, not assumed. The env draws arrivals, fills and the mid step from
    one generator per episode, and numpy's binomial sampler consumes a number of underlying
    draws that depends on the fill probability. Measured with numpy 2.5.1, it takes one draw
    while ``n*min(p, 1-p)`` is at most 30, a variable number above that, and none when the
    fill probability is exactly zero. Two quoting policies can therefore desynchronize the
    stream, after which their mid paths (and arrivals) differ and the gap mixes policy with
    luck. Each episode records both arms' post-step mids and raises
    :class:`UnpairedMidPathError` at the first episode whose paths differ, so an unpaired
    gap is never returned.

    Measured on seeds 0 to 15: at the default parameters every fixed-spread quoter in the F2
    grid shares the reference's path; at arrival rates of 16000, 20000, 30000 and 40000 (80
    to 200 arrivals per step) none of them does. A quote wide enough that
    ``exp(-kappa*depth)`` underflows to zero draws nothing where the reference draws, so it
    unpairs at the first order arriving on that side, even at the default rate (at the
    default ``kappa`` and ``max_depth`` no quote can underflow). The default random stream is left
    as it is so the committed F2 regrets reproduce exactly.
    """
    p = params or MMParams()
    reference = closed_form_reference_policy(p)
    env = MarketMakingEnv(p)
    gap = 0.0
    for i in range(n_episodes):
        seed = seed_base + i
        ref_r, ref_mids, _ = _rollout(env, reference, seed)
        pol_r, pol_mids, _ = _rollout(env, policy, seed)
        _check_paired(seed, ref_mids, pol_mids)
        gap += ref_r - pol_r
    return gap / n_episodes


@dataclass(frozen=True)
class MMPnLSplit:
    """A policy's mean episode reward split into the four ``step`` components.

    Each component field is the mean over episodes of that component's episode sum, signed
    as its contribution to the reward. ``reward`` is the mean episode reward as the env
    accumulated it, and :attr:`total` (the sum of the four components) matches it up to
    floating-point rounding. Diagnostic only: nothing scores on it.
    """

    spread_capture: float
    inventory_pnl: float
    inventory_penalty: float
    liquidation_cost: float
    reward: float

    @property
    def total(self) -> float:
        return (
            self.spread_capture
            + self.inventory_pnl
            + self.inventory_penalty
            + self.liquidation_cost
        )


def mm_pnl_split(
    policy: Policy,
    *,
    params: Optional[MMParams] = None,
    n_episodes: int = 16,
    seed_base: int = 0,
) -> MMPnLSplit:
    """Where ``policy``'s reward comes from, over the same seeded episodes :func:`mm_regret`
    uses: spread capture, inventory marked to the mid move, the running inventory penalty
    and the terminal liquidation cost, each averaged over episodes.

    It separates a quoter that earns by capturing spread from one that earns by holding
    inventory into a favourable mid path. Differencing two policies' splits attributes their
    regret component by component only when their mid paths are shared, which is the
    condition :func:`mm_regret` checks for the same ``params`` and seeds. Rank-neutral: no
    SharpeArena or SharpeBench score reads it.
    """
    p = params or MMParams()
    env = MarketMakingEnv(p)
    reward = 0.0
    sums = dict.fromkeys(_PNL_COMPONENTS, 0.0)
    for i in range(n_episodes):
        episode_reward, _, split = _rollout(env, policy, seed_base + i)
        reward += episode_reward
        for key in _PNL_COMPONENTS:
            sums[key] += split[key]
    return MMPnLSplit(
        **{key: value / n_episodes for key, value in sums.items()},
        reward=reward / n_episodes,
    )


__all__ = [
    "MMParams",
    "MMPnLSplit",
    "MarketMakingEnv",
    "UnpairedMidPathError",
    "analytically_optimal_policy",
    "closed_form_reference_policy",
    "fixed_spread_policy",
    "mm_pnl_split",
    "mm_regret",
]
