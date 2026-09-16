"""A multi-agent limit-order-book market over the native M3 matching engine.

:class:`LOBMarketEnv` is a PettingZoo ``ParallelEnv`` where ``n_agents`` market makers
post bid/ask quotes into one shared, deterministic integer-tick order book (the native
``PyOrderBook``: price-time priority, real fills). A seeded noise trader sends market
orders each step so quotes actually fill. Distinct from the bar-level position env and the
M2 endogenous (batch-clearing) market: here orders match against a real resting book.

**Leak-free.** An agent's observation is the post-step public depth ladder plus its own
inventory/cash; it never sees other agents' pending same-step orders (all quotes are
collected, then the book clears, then the next observation is produced).

**Inventory mark.** Reward values inventory at a per-agent mark. The default
``mark="ex_own_mid"`` is the midpoint of the best bid and best ask among resting orders that
*other* agents own. While the others' orders lack a side, the previous mark is carried
forward, starting at the opening reference mid (1000 ticks). The noise trader never rests,
so a lone agent's mark stays at 1000 ticks. An agent's own resting quotes never enter its
mark. ``mark="book_mid"`` is the mark this environment used before 2026-09-16, the mid of
the whole book with the agent's own quotes included, kept only to replay earlier runs.
Under it an agent moves its own valuation without a counterparty: with the noise trader
off, a lone agent quoting ``(1, 20)`` walks the book mid 1000, 1010, 1014, 1016, 1018 ticks
over four steps with no fill, and with the noise trader on its equity changes on steps
where it trades with nobody.

**Same-bar priority.** The native book folds a bar's orders sorted by their ``agent``
field, so under the default ``priority="agent_index"`` seat 0 queues ahead of seat 1 at a
shared tick on every step (two identical ``(3, 3)`` quoters over seeds 0 to 31: seat 0 was
filled more on all 32, 14,222 units against 13,900). ``priority="seeded_shuffle"`` draws a
uniform permutation of the seats for each step from ``(seed, step)`` (SplitMix64 with
Fisher-Yates, on a stream separate from the noise trader's, so both rules see the same
noise-trader orders) and submits each quote under a code that sorts by its seat's rank.
Every pair of seats is then ordered each way with probability 1/2. A cyclic rotation with a
uniform offset was not used: for three or more seats it queues seat ``i`` ahead of seat
``j`` with probability ``(n - d) / n``, where ``d = (j - i) mod n``. The rule lives
entirely in the codes this environment submits, so the native engine, ``SPEC_HASH`` and the
golden fill tape are unchanged, and the default path is byte-identical to the one before
the rule existed.

**Self-trades.** The book has no self-trade prevention. A new quote here crosses a resting
order only after the previous step left one side of the book empty and the reference mid
took a seeded step, so self-trades are rare (2 of 27,552 fills over 400 seeded random
configurations). Both legs belong to one agent at one price, so its cash and inventory do
not change, and under the default mark its own orders are not part of its valuation.
"""

from __future__ import annotations

import json
from typing import Any, Optional

import numpy as np

try:  # pragma: no cover - exercised only when pettingzoo is installed
    from pettingzoo import ParallelEnv

    _HAS_PZ = True
except Exception:  # noqa: BLE001
    ParallelEnv = object  # type: ignore[assignment,misc]
    _HAS_PZ = False

from .sharpearena_py import PyOrderBook

_MID_TICK = 1000  # the reference mid starts here (in ticks)
_MASK64 = 0xFFFFFFFFFFFFFFFF
_SEAT_KEY = 0x5EA70D3E2B1CA6F9
_SEAT_STEP = 0xD1B54A32D192ED03

MARKS = ("ex_own_mid", "book_mid")
PRIORITIES = ("agent_index", "seeded_shuffle")


def _splitmix_bits(state: int) -> tuple[int, int]:
    """One SplitMix64 draw -> (new_state, 64-bit output); deterministic, no numpy RNG."""
    state = (state + 0x9E3779B97F4A7C15) & _MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    z ^= z >> 31
    return state, z


def _splitmix(state: int) -> tuple[int, float]:
    """One SplitMix64 draw -> (new_state, unit in [0, 1)); deterministic, no numpy RNG."""
    state, z = _splitmix_bits(state)
    return state, (z >> 11) / float(1 << 53)


class LOBMarketEnv(ParallelEnv):  # type: ignore[misc]
    """N market makers quoting into one shared limit-order book.

    Each agent's action is a 2-vector ``[bid_offset, ask_offset]`` of ticks from the
    reference mid (clamped to ``[1, max_offset]``); it posts a buy at ``mid - bid_offset``
    and a sell at ``mid + ask_offset``, each of size ``quote_qty``. A seeded noise trader
    then sends a market order, the book clears, and reward is the change in
    ``cash + inventory * mark`` minus a squared-inventory penalty.

    ``mark`` (one of :data:`MARKS`, default ``"ex_own_mid"``) and ``priority`` (one of
    :data:`PRIORITIES`, default ``"agent_index"``) are described in the module docstring.
    """

    metadata = {"render_modes": [], "name": "sharpearena_lob_v0"}

    def __init__(
        self,
        n_agents: int = 2,
        *,
        n_steps: int = 120,
        seed: int = 0,
        tick_size: float = 0.01,
        levels: int = 5,
        quote_qty: int = 10,
        max_offset: int = 20,
        inventory_penalty: float = 0.001,
        noise_intensity: float = 2.0,
        mark: str = "ex_own_mid",
        priority: str = "agent_index",
    ) -> None:
        if not _HAS_PZ:
            raise RuntimeError(
                "pettingzoo is not installed. Install the 'pettingzoo' extra to use "
                "LOBMarketEnv; the rest of the sharpearena package works without it."
            )
        if n_agents < 1:
            raise ValueError("n_agents must be >= 1")
        if mark not in MARKS:
            raise ValueError(f"mark must be one of {MARKS}, got {mark!r}")
        if priority not in PRIORITIES:
            raise ValueError(f"priority must be one of {PRIORITIES}, got {priority!r}")
        self._mark_rule = mark
        self._priority = priority
        self._n_agents = int(n_agents)
        self._n_steps = int(n_steps)
        self._seed = int(seed)
        self._tick_size = float(tick_size)
        self._levels = int(levels)
        self._quote_qty = int(quote_qty)
        self._max_offset = int(max_offset)
        self._inv_pen = float(inventory_penalty)
        self._noise = float(noise_intensity)
        self.possible_agents = [f"agent_{i}" for i in range(self._n_agents)]

        from gymnasium import spaces

        self._obs_dim = 4 * self._levels + 5  # ladder (bids+asks) + mid/micro/imb + inv/cash
        self._obs_space = spaces.Box(-np.inf, np.inf, shape=(self._obs_dim,), dtype=np.float64)
        self._act_space = spaces.Box(
            low=1.0, high=float(self._max_offset), shape=(2,), dtype=np.float32
        )

    @property
    def mark(self) -> str:
        """The inventory mark rule this environment was built with."""
        return self._mark_rule

    @property
    def priority(self) -> str:
        """The same-bar seat-priority rule this environment was built with."""
        return self._priority

    # -- PettingZoo API ----------------------------------------------------

    def observation_space(self, agent):  # noqa: D401
        return self._obs_space

    def action_space(self, agent):  # noqa: D401
        return self._act_space

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._seed = int(seed)
        self.agents = list(self.possible_agents)
        self._book = PyOrderBook(tick_size=self._tick_size, levels=self._levels)
        self._book.reset_book()
        self._mid = _MID_TICK
        self._rng = self._seed ^ 0x1234_5678_9ABC_DEF0
        self._step = 0
        self._inventory = {a: 0 for a in self.agents}
        self._cash = {a: 0.0 for a in self.agents}
        self._prev_equity = {a: 0.0 for a in self.agents}
        # order id -> [agent index, side, price_tick, resting qty]; only agents ever rest.
        self._resting: dict[int, list] = {}
        self._next_order_id = 0
        self._marks = {a: float(_MID_TICK) for a in self.agents}
        ladder = json.loads(self._book.ladder())
        obs = {a: self._obs(a, ladder) for a in self.agents}
        infos = {a: {} for a in self.agents}
        return obs, infos

    def step(self, actions: dict):
        # 1. every live agent posts a two-sided quote (collected before any clear). The
        #    `agent` field is the seat code the book's canonical sort orders the bar by.
        codes = self._seat_codes()
        orders: list[dict] = []
        for i, a in enumerate(self.possible_agents):
            if a not in actions:
                continue
            bid_off, ask_off = (int(round(float(x))) for x in np.asarray(actions[a]).reshape(-1)[:2])
            bid_off = max(1, min(self._max_offset, bid_off))
            ask_off = max(1, min(self._max_offset, ask_off))
            orders.append({"agent": codes[i], "kind": "limit", "side": "buy",
                           "price_tick": self._mid - bid_off, "qty": self._quote_qty})
            orders.append({"agent": codes[i], "kind": "limit", "side": "sell",
                           "price_tick": self._mid + ask_off, "qty": self._quote_qty})

        # 2. a seeded noise trader sends one market order under the exogenous code, which
        #    sorts after every agent seat.
        self._rng, u = _splitmix(self._rng)
        if u < 0.5 + 0.1 * self._noise:
            self._rng, u2 = _splitmix(self._rng)
            side = "buy" if u2 < 0.5 else "sell"
            self._rng, u3 = _splitmix(self._rng)
            qty = 1 + int(u3 * self._noise * self._quote_qty)
            orders.append({"agent": self._exogenous_code(), "kind": "market",
                           "side": side, "qty": qty})

        out = json.loads(self._book.step_book(json.dumps(orders)))
        ladder = out["ladder"]
        self._track_resting(orders, out["fills"])
        self._apply_fills(out["fills"], ladder)
        self._update_marks()
        self._mid = self._next_mid(ladder)
        self._step += 1

        done = self._step >= self._n_steps
        obs, rewards, terms, truncs, infos = {}, {}, {}, {}, {}
        for a in self.agents:
            obs[a] = self._obs(a, ladder)
            rewards[a] = self._reward(a, ladder)
            terms[a] = False
            truncs[a] = done
            infos[a] = {"inventory": self._inventory[a], "cash": self._cash[a]}
        if done:
            self.agents = []
        return obs, rewards, terms, truncs, infos

    # -- internals ---------------------------------------------------------

    def _exogenous_code(self) -> int:
        """The noise trader's book code: past every seat code of the active rule."""
        n = self._n_agents
        return n if self._priority == "agent_index" else n * n

    def _owner(self, code: int) -> Optional[int]:
        """The agent index behind a book code, or ``None`` for the noise trader."""
        if code >= self._exogenous_code():
            return None
        return code % self._n_agents

    def _seat_codes(self) -> list[int]:
        """Book code per agent index for this step (see the module docstring).

        ``seeded_shuffle`` gives agent ``i`` the code ``rank * n + i``: the book sorts by
        rank, and ``code % n`` recovers the agent from any fill, including fills against a
        quote that rested under an earlier step's permutation.
        """
        n = self._n_agents
        if self._priority == "agent_index":
            return list(range(n))
        state = (self._seed ^ _SEAT_KEY ^ (self._step * _SEAT_STEP)) & _MASK64
        order = list(range(n))
        for k in range(n - 1, 0, -1):
            state, z = _splitmix_bits(state)
            j = ((z >> 11) * (k + 1)) >> 53  # exact integer draw in [0, k]
            order[k], order[j] = order[j], order[k]
        codes = [0] * n
        for rank, i in enumerate(order):
            codes[i] = rank * n + i
        return codes

    def _track_resting(self, orders: list[dict], fills: list[dict]) -> None:
        """Mirror every agent's resting quotes, which the ex-own mark reads.

        Replays the engine's id rule: the book folds the batch by ``(agent code,
        submission index)`` and every limit consumes one id. A code posts at most one
        limit per side per step, so a fill's ``(taker code, taker side)`` names the order
        that crossed.
        """
        crossed: dict[tuple[int, str], int] = {}
        for f in fills:
            key = (f["taker_agent"], f["taker_side"])
            crossed[key] = crossed.get(key, 0) + f["qty"]
        for j in sorted(range(len(orders)), key=lambda j: (orders[j]["agent"], j)):
            o = orders[j]
            if o["kind"] != "limit":
                continue
            order_id = self._next_order_id
            self._next_order_id += 1
            left = o["qty"] - crossed.get((o["agent"], o["side"]), 0)
            if left > 0:
                owner = self._owner(o["agent"])
                self._resting[order_id] = [owner, o["side"], o["price_tick"], left]
        for f in fills:
            entry = self._resting[f["maker_id"]]
            entry[3] -= f["qty"]
            if entry[3] == 0:
                del self._resting[f["maker_id"]]

    def _update_marks(self) -> None:
        """Move each agent's ex-own mark to the others' mid when they quote both sides."""
        if self._mark_rule != "ex_own_mid":
            return
        levels: dict[str, dict[int, set]] = {"buy": {}, "sell": {}}
        for owner, side, price, _qty in self._resting.values():
            levels[side].setdefault(price, set()).add(owner)
        bids = sorted(levels["buy"].items(), reverse=True)
        asks = sorted(levels["sell"].items())
        for i, a in enumerate(self.possible_agents):
            bid = next((p for p, owners in bids if owners - {i}), None)
            ask = next((p for p, owners in asks if owners - {i}), None)
            if bid is not None and ask is not None:
                self._marks[a] = (bid + ask) / 2.0

    def _mark_price(self, agent: str, ladder) -> float:
        if self._mark_rule == "book_mid":
            return ladder["mid"] or float(self._mid)
        return self._marks[agent]

    def _apply_fills(self, fills, ladder) -> None:
        mid = ladder["mid"] or float(self._mid)
        for f in fills:
            price = f["price_tick"]
            qty = f["qty"]
            maker_index = self._owner(f["maker_agent"])
            taker_index = self._owner(f["taker_agent"])
            maker = self.possible_agents[maker_index] if maker_index is not None else None
            taker = self.possible_agents[taker_index] if taker_index is not None else None
            # maker side is the opposite of the taker side.
            if maker is not None:
                if f["taker_side"] == "buy":  # maker sold
                    self._inventory[maker] -= qty
                    self._cash[maker] += price * qty
                else:
                    self._inventory[maker] += qty
                    self._cash[maker] -= price * qty
            if taker is not None:
                if f["taker_side"] == "buy":
                    self._inventory[taker] += qty
                    self._cash[taker] -= price * qty
                else:
                    self._inventory[taker] -= qty
                    self._cash[taker] += price * qty

    def _equity(self, agent: str, mid: float) -> float:
        return self._cash[agent] + self._inventory[agent] * mid

    def _reward(self, agent: str, ladder) -> float:
        mid = self._mark_price(agent, ladder)
        eq = self._equity(agent, mid)
        prev = getattr(self, "_prev_equity", {}).get(agent, 0.0)
        if not hasattr(self, "_prev_equity"):
            self._prev_equity = {}
        self._prev_equity[agent] = eq
        return float(eq - prev - self._inv_pen * self._inventory[agent] ** 2)

    def _next_mid(self, ladder) -> int:
        # the reference mid follows the cleared microprice when available, else a seeded walk.
        if ladder["bids"] and ladder["asks"]:
            return int(round((ladder["bids"][0][0] + ladder["asks"][0][0]) / 2))
        self._rng, u = _splitmix(self._rng)
        return self._mid + (1 if u < 0.5 else -1)

    def _obs(self, agent: str, ladder) -> np.ndarray:
        bids = ladder["bids"][: self._levels]
        asks = ladder["asks"][: self._levels]
        vec = np.zeros(self._obs_dim, dtype=np.float64)
        for j, lvl in enumerate(bids):
            vec[2 * j] = lvl[0]
            vec[2 * j + 1] = lvl[1]
        base = 2 * self._levels
        for j, lvl in enumerate(asks):
            vec[base + 2 * j] = lvl[0]
            vec[base + 2 * j + 1] = lvl[1]
        tail = 4 * self._levels
        vec[tail] = ladder["mid"]
        vec[tail + 1] = ladder["microprice"]
        vec[tail + 2] = ladder["queue_imbalance"]
        vec[tail + 3] = self._inventory[agent]
        vec[tail + 4] = self._cash[agent]
        return vec


def symmetric_quote_policy(observation: Any = None, *, offset: int = 3) -> np.ndarray:
    """A fixed symmetric two-sided quote `offset` ticks from mid (the MM reference)."""
    return np.array([offset, offset], dtype=np.float32)


def noise_trader_policy(observation: Any = None, *, max_offset: int = 20) -> np.ndarray:
    """A wide, passive quote that rarely fills (a near-inactive reference)."""
    return np.array([max_offset, max_offset], dtype=np.float32)


__all__ = ["LOBMarketEnv", "symmetric_quote_policy", "noise_trader_policy"]
