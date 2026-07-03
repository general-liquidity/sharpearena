"""A PettingZoo **endogenous shared-book market** over OpenOutcry (M2).

:class:`EndogenousMarketEnv` is a :class:`pettingzoo.ParallelEnv` in which ``n_agents``
agents trade **one shared book per symbol**: their aggregate order flow *moves the
price*. This is a real market with endogenous impact, **distinct from**
:class:`~openoutcry.pettingzoo_env.MultiAgentOpenOutcryEnv` — there every agent trades
its own private copy of a frozen path and no agent's order ever moves the price another
sees. Here the cleared price is the common consequence of everyone's flow.

**The model** (Kyle 1985 linear price impact + an Almgren-Chriss permanent/temporary
split), implemented in pure deterministic Rust (``openoutcry::market``):

* The frozen synthetic panel is the **exogenous** (fundamental) component.
* Each bar, each agent's target weight becomes a signed order size
  ``q_i = capital * (w_i - prev_w_i) / cleared_mid``; aggregate net flow ``Q = Σ_i q_i``
  is summed in **sorted agent order** for float determinism.
* **Permanent impact** accumulates a running multiplier ``M`` on the reference price:
  ``M_{t+1} = M_t * (1 + lambda * Q_t / V)`` — the cleared mid is ``exo_mid_t * M_t``.
* **Temporary impact** is what an agent pays this bar:
  ``fill_i = cleared_mid_t * (1 + (lambda * Q_t + eta * q_i) / V)`` — it pays for the
  crowd's flow plus its own size.
* **Reward** is each agent's own realized portfolio return, at its own fill prices.

**Leak-free.** An agent's observation at ``t`` reflects only cleared prices ``<= t`` and
its own fills — **never** another agent's *pending* order for ``t``. Two facts enforce
it: (1) the cleared reference mid and each agent's order *size* are functions of ``M_t``,
which embeds only flow strictly before ``t``; (2) the Parallel API collects **all**
bar-``t`` actions before producing any ``t+1`` observation, so no agent's bar-``t``
decision can see a peer's bar-``t`` intent. The realized fill *price* does reflect the
aggregate cleared flow — that is price impact, the point of the market, not a leak.

**Parallel, not AEC.** Order submission in a market is simultaneous, so the native API is
:class:`pettingzoo.ParallelEnv`. Agents are always iterated in sorted roster order
(``agent_0`` … ``agent_{n-1}``) and result dicts are built in that canonical order, so the
clearing is reproducible.

``pettingzoo`` is an **optional** dependency (guarded exactly like
:mod:`openoutcry.pettingzoo_env`): ``import openoutcry`` and ``import
openoutcry.market_env`` both work without it; only *constructing*
:class:`EndogenousMarketEnv` raises a clear ``RuntimeError`` when it is absent.

The Rust clearing engine is reached through the native ``PyMarketClearing`` pyclass, to a
small documented JSON interface:

* ``PyMarketClearing(n_symbols, n_days, seed, n_agents, capital, kyle_lambda, eta,
  volume_scale, distribution_mode, richness)``. ``richness`` (``data_poor`` | ``standard``
  | ``data_rich``) is the information-disclosure difficulty axis, orthogonal to
  ``distribution_mode``: it sets how much of the market each observation surfaces (trailing
  lookback + optional fundamentals/news), never revealing a future bar. ``standard`` is the
  historical default disclosure.
* ``reset_market() -> json``: ``{symbols, n_agents, n_bars, start_bar, cursor, capital,
  observations:[MarketObservation, ...]}`` (observations in canonical agent order).
* ``step_market(orders_json) -> json``: ``orders_json`` is a JSON array of shape
  ``n_agents × n_symbols`` of target weights (canonical agent order, sorted symbol
  order); returns ``{cleared_mids, net_flow, rewards, navs, fills, observations, done,
  cursor}``.
"""

from __future__ import annotations

import json
from typing import Any, Optional

import numpy as np

try:  # pragma: no cover - exercised only when pettingzoo is installed
    from pettingzoo import ParallelEnv
    from pettingzoo.utils.conversions import parallel_to_aec

    _HAS_PETTINGZOO = True
except Exception:  # noqa: BLE001 - any import failure means "not available"
    ParallelEnv = object  # type: ignore[assignment,misc]
    parallel_to_aec = None  # type: ignore[assignment]
    _HAS_PETTINGZOO = False

try:  # gymnasium ships with the package, but guard so import never hard-fails
    import gymnasium as gym
    from gymnasium import spaces

    _HAS_GYM = True
except Exception:  # noqa: BLE001
    gym = None  # type: ignore[assignment]
    spaces = None  # type: ignore[assignment]
    _HAS_GYM = False

from .openoutcry_py import PyMarketClearing


def _agent_ids(n_agents: int) -> list[str]:
    """The canonical, sorted-stable roster (``agent_0`` … ``agent_{n-1}``)."""
    return [f"agent_{i}" for i in range(int(n_agents))]


class EndogenousMarketEnv(ParallelEnv):
    """A shared-book, endogenous-impact :class:`pettingzoo.ParallelEnv` over OpenOutcry.

    ``n_agents`` agents submit target-weight vectors simultaneously each bar; the Rust
    clearing engine converts them to signed order sizes, moves the cleared price by their
    aggregate flow (Kyle permanent impact + Almgren-Chriss temporary impact), fills each
    agent at its own impacted price, and returns each agent's realized bar return as its
    reward. **This is a real shared market with endogenous price impact**, unlike the
    competition env where agents never interact.

    Determinism: agents are iterated in sorted roster order and every result dict is built
    in that canonical order, so a run is reproducible from ``(seed, params, actions)``.
    """

    metadata = {
        "render_modes": [],
        "name": "openoutcry_endogenous_market_v0",
        "is_parallelizable": True,
    }

    def __init__(
        self,
        n_agents: int = 2,
        *,
        n_symbols: int = 4,
        n_days: int = 120,
        seed: int = 0,
        capital: float = 1.0,
        kyle_lambda: float = 0.1,
        eta: float = 0.05,
        volume_scale: float = 1.0,
        vol_scale: float = 0.0,
        distribution_mode: str = "calm",
        richness: str = "standard",
        max_weight: float = 1.0,
        allow_short: bool = True,
    ) -> None:
        if not _HAS_PETTINGZOO:
            raise RuntimeError(
                "pettingzoo is not installed. Install 'pettingzoo' to use "
                "EndogenousMarketEnv; the rest of the openoutcry package works without it."
            )
        if int(n_agents) < 1:
            raise ValueError("n_agents must be >= 1")

        self._n_agents = int(n_agents)
        self._n_symbols = int(n_symbols)
        self._n_days = int(n_days)
        self._seed = int(seed)
        self._capital = float(capital)
        self._kyle_lambda = float(kyle_lambda)
        self._eta = float(eta)
        self._volume_scale = float(volume_scale)
        self._vol_scale = float(vol_scale)
        self._distribution_mode = str(distribution_mode)
        self._richness = str(richness)
        self._max_weight = float(max_weight)
        self._allow_short = bool(allow_short)

        self.possible_agents: list[str] = _agent_ids(self._n_agents)
        self.agents: list[str] = list(self.possible_agents)

        # Index of each agent in the canonical (sorted) order the Rust engine expects.
        self._agent_index = {a: i for i, a in enumerate(self.possible_agents)}

        self._market: Optional[PyMarketClearing] = None
        self._symbols: list[str] = []
        self._build_market()

        # Discover the symbol axis from the freshly built market and fix the spaces.
        meta = json.loads(self._market.reset_market())
        self._symbols = list(meta["symbols"])
        self._build_spaces()

    # -- internal helpers --------------------------------------------------

    def _build_market(self) -> None:
        """(Re)construct the native shared-book market at the current scenario seed."""
        kwargs = dict(
            n_symbols=self._n_symbols,
            n_days=self._n_days,
            seed=self._seed,
            n_agents=self._n_agents,
            capital=self._capital,
            kyle_lambda=self._kyle_lambda,
            eta=self._eta,
            volume_scale=self._volume_scale,
            vol_scale=self._vol_scale,
            distribution_mode=self._distribution_mode,
            richness=self._richness,
        )
        try:
            self._market = PyMarketClearing(**kwargs)
            return
        except TypeError:
            pass
        # An older native binding may predate the newest optional params (richness, then
        # vol_scale). Drop them only when they sit at their defaults, so default behavior is
        # unchanged; a non-default request for a missing param still surfaces the error.
        if self._richness != "standard":
            raise TypeError(
                "the native binding predates the 'richness' parameter (needs a rebuild)"
            )
        kwargs.pop("richness")
        try:
            self._market = PyMarketClearing(**kwargs)
            return
        except TypeError:
            pass
        if self._vol_scale != 0.0:
            raise TypeError(
                "the native binding predates the 'vol_scale' parameter (needs a rebuild)"
            )
        kwargs.pop("vol_scale")
        self._market = PyMarketClearing(**kwargs)

    def _build_spaces(self) -> None:
        if not _HAS_GYM:  # pragma: no cover - gymnasium is a hard dep of the package
            self._single_obs_space = None
            self._single_act_space = None
            return
        n = len(self._symbols)
        low = -self._max_weight if self._allow_short else 0.0
        self._single_act_space = spaces.Box(
            low=low, high=self._max_weight, shape=(n,), dtype=np.float32
        )
        # Mirrors openoutcry.gym.OpenOutcryEnv's decoded-observation Dict space.
        self._single_obs_space = spaces.Dict(
            {
                "closes": spaces.Box(low=0.0, high=np.inf, shape=(n,), dtype=np.float64),
                "positions": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(n,), dtype=np.float64
                ),
                "cash": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64
                ),
            }
        )

    def _decode_obs(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        """Decode one wire-format ``MarketObservation`` dict into the Dict-space arrays:
        the latest cleared close, the agent's share positions, and its cash."""
        by_symbol = {s["symbol"]: s for s in obs["symbols"]}
        closes = np.array(
            [by_symbol[sym]["close_history"][-1] for sym in self._symbols],
            dtype=np.float64,
        )
        pos = {p["symbol"]: p["shares"] for p in obs.get("portfolio", [])}
        positions = np.array(
            [pos.get(sym, 0.0) for sym in self._symbols], dtype=np.float64
        )
        return {
            "closes": closes,
            "positions": positions,
            "cash": np.array([obs["cash"]], dtype=np.float64),
        }

    def _orders_matrix(self, actions: dict[str, Any]) -> list[list[float]]:
        """Assemble the ``n_agents × n_symbols`` target-weight matrix in canonical agent
        order. Agents that have left the live roster (blown up / out of bars) submit a flat
        (all-zero) order so the shared book keeps its fixed dimensions."""
        matrix = [[0.0] * len(self._symbols) for _ in range(self._n_agents)]
        for agent, action in actions.items():
            idx = self._agent_index[agent]
            weights = np.asarray(action, dtype=np.float64).reshape(-1)
            matrix[idx] = [float(w) for w in weights]
        return matrix

    # -- PettingZoo (Parallel) API -----------------------------------------

    def observation_space(self, agent: str):
        """The per-agent observation space (shared across agents)."""
        return self._single_obs_space

    def action_space(self, agent: str):
        """The per-agent target-weight ``Box`` action space (shared across agents)."""
        return self._single_act_space

    @property
    def symbols(self) -> list[str]:
        return list(self._symbols)

    @property
    def scenario_seed(self) -> int:
        return self._seed

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[dict[str, Any], dict[str, dict]]:
        """Reset the shared market. Passing an int ``seed`` reselects the exogenous path
        (the market is rebuilt on it). Returns the standard Parallel ``(observations,
        infos)`` pair, keyed by live agent in canonical order."""
        if seed is not None:
            self._seed = int(seed)
            self._build_market()

        self.agents = list(self.possible_agents)
        meta = json.loads(self._market.reset_market())
        self._symbols = list(meta["symbols"])

        observations: dict[str, Any] = {}
        infos: dict[str, dict] = {}
        per_agent_obs = meta["observations"]
        for agent in self.agents:  # canonical order
            idx = self._agent_index[agent]
            observations[agent] = self._decode_obs(per_agent_obs[idx])
            infos[agent] = {"scenario_seed": self._seed}
        return observations, infos

    def step(
        self, actions: dict[str, Any]
    ) -> tuple[
        dict[str, Any],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict],
    ]:
        """Clear one bar of the shared book.

        Every live agent's target weight is fanned into the aggregate flow; the cleared
        price moves with that flow, each agent fills at its own impacted price, and each
        agent's realized bar return is its reward. ``truncated`` means the path ran out of
        bars (a window cut-off, shared by all agents); ``terminated`` means an agent blew
        up (``nav <= 0``), an absorbing state. Ended agents are dropped from
        ``self.agents`` (kept in ``possible_agents``), per the PettingZoo contract.
        """
        acting = list(self.agents)  # the agents that get result entries this step
        result = json.loads(self._market.step_market(json.dumps(self._orders_matrix(actions))))

        per_agent_obs = result["observations"]
        rewards_in = result["rewards"]
        navs_in = result["navs"]
        out_of_bars = bool(result["done"])

        observations: dict[str, Any] = {}
        rewards: dict[str, float] = {}
        terminations: dict[str, bool] = {}
        truncations: dict[str, bool] = {}
        infos: dict[str, dict] = {}

        for agent in acting:  # canonical order
            idx = self._agent_index[agent]
            nav = float(navs_in[idx])
            observations[agent] = self._decode_obs(per_agent_obs[idx])
            rewards[agent] = float(rewards_in[idx])
            terminations[agent] = nav <= 0.0
            truncations[agent] = out_of_bars
            infos[agent] = {
                "nav": nav,
                "cleared_mids": result["cleared_mids"],
                "net_flow": result["net_flow"],
                "fills": result["fills"][idx],
                "scenario_seed": self._seed,
            }

        # Drop agents whose episode ended; iterate canonical order so the survivors stay
        # sorted (PettingZoo keeps them in possible_agents).
        self.agents = [
            agent
            for agent in acting
            if not (terminations[agent] or truncations[agent])
        ]
        return observations, rewards, terminations, truncations, infos

    def render(self):  # pragma: no cover - no visual rendering
        return None

    def close(self):  # pragma: no cover
        self._market = None


def make_aec_env(*args: Any, **kwargs: Any):
    """The turn-based **AEC** view of the endogenous market, derived from the Parallel env
    via the Farama-recommended :func:`pettingzoo.utils.conversions.parallel_to_aec`.

    Accepts the same arguments as :class:`EndogenousMarketEnv`. Raises a clear
    ``RuntimeError`` when ``pettingzoo`` is not installed.
    """
    if not _HAS_PETTINGZOO:
        raise RuntimeError(
            "pettingzoo is not installed. Install 'pettingzoo' to build the AEC view; "
            "the rest of the openoutcry package works without it."
        )
    return parallel_to_aec(EndogenousMarketEnv(*args, **kwargs))


__all__ = [
    "EndogenousMarketEnv",
    "make_aec_env",
]
