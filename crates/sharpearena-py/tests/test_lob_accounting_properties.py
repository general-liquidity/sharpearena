"""Seeded accounting properties for the Python LOB market (``lob_env.py``).

The Rust ``OrderBook`` carries no accounts: cash and inventory live only in
``LOBMarketEnv`` (``lob_env.py``, ``_apply_fills`` / ``_equity`` / ``_reward``). These
properties drive the shipped environment with seeded ``random.Random`` order streams and
check the interface-level accounting after every step, exactly as the environment
computes it (integer tick prices, so cash sums are exact in float).

The seeded-reset property pins the repaired defect from
``docs/audits/2026-09-07/arena-reviewer.md`` item 7 (LOB reset retained ``_prev_equity``),
closed under batch G in ``VERIFICATION-LOG.md``; it is a regression guard, not a reopening.

The default ``ex_own_mid`` mark is checked against the native engine rather than against
the environment's own bookkeeping: every batch the environment submitted is replayed into a
fresh ``PyOrderBook``, the agent's own resting orders are cancelled there, and the mark must
equal that book's mid whenever both sides remain, and otherwise the price of the step's last
fill between two different owners.
"""

from __future__ import annotations

import json
import random

import pytest

pytest.importorskip("pettingzoo")

from sharpearena.lob_env import LOBMarketEnv  # noqa: E402
from sharpearena.sharpearena_py import PyOrderBook  # noqa: E402

CASES = 24
RULES = [
    ("ex_own_mid", "agent_index"),
    ("book_mid", "agent_index"),
    ("ex_own_mid", "seeded_shuffle"),
    ("book_mid", "seeded_shuffle"),
]
_DEEP = 1_000_000  # ladder depth that shows every level of a replayed book


class _Recorder:
    """Delegates to the environment's book and keeps every submitted batch."""

    def __init__(self, book) -> None:
        self._book = book
        self.batches: list[str] = []

    def step_book(self, orders_json: str) -> str:
        self.batches.append(orders_json)
        return self._book.step_book(orders_json)

    def __getattr__(self, name):
        return getattr(self._book, name)


def _levels(entries, side: str) -> list[list[int]]:
    """Aggregate ``[price, qty]`` levels for one side, best first."""
    agg: dict[int, int] = {}
    for _owner, entry_side, price, qty in entries:
        if entry_side == side:
            agg[price] = agg.get(price, 0) + qty
    return [[p, agg[p]] for p in sorted(agg, reverse=side == "buy")]


def _engine_ex_own_ladder(env: LOBMarketEnv, batches: list[str], agent_index: int) -> dict:
    """The native book after every batch, with ``agent_index``'s resting orders cancelled."""
    book = PyOrderBook(tick_size=env._tick_size, levels=_DEEP)
    book.reset_book()
    for batch in batches:
        book.step_book(batch)
    everything = json.loads(book.ladder())
    assert everything["bids"] == _levels(env._resting.values(), "buy")
    assert everything["asks"] == _levels(env._resting.values(), "sell")
    own = [oid for oid, entry in env._resting.items() if entry[0] == agent_index]
    cancels = [{"agent": 0, "kind": "cancel", "id": oid} for oid in own]
    ladder = json.loads(book.step_book(json.dumps(cancels)))["ladder"]
    others = [entry for entry in env._resting.values() if entry[0] != agent_index]
    assert ladder["bids"] == _levels(others, "buy")
    assert ladder["asks"] == _levels(others, "sell")
    return ladder


def _actions(rng: random.Random, env: LOBMarketEnv) -> dict:
    return {
        a: [rng.randint(1, env._max_offset), rng.randint(1, env._max_offset)]
        for a in env.agents
    }


def _capture_fills(env: LOBMarketEnv) -> list:
    """Record every fill tape the environment books, without changing what it books."""
    captured: list = []
    original = env._apply_fills

    def wrapped(fills, ladder):
        captured.append(fills)
        original(fills, ladder)

    env._apply_fills = wrapped
    return captured


@pytest.mark.parametrize("mark, priority", RULES)
def test_equity_reward_identity_and_flow_conservation_every_step(mark, priority):
    """Binds ``_equity`` / ``_reward`` / ``_mark_price`` / ``_update_marks`` and
    ``_apply_fills`` in ``lob_env.py``.

    After each step, for every agent: ``cash + inventory * mark`` is the stored equity
    baseline, and ``reward == equity - prev_equity - penalty * inventory**2``. The
    ``book_mid`` mark is the full book's mid; the ``ex_own_mid`` mark is the native book's
    mid with the agent's own resting orders cancelled; while that book lacks a side it is
    the price of the step's last fill between two different owners, and it carries
    forward from 1000 ticks on a step with neither. Across agents, inventory and cash are conserved against
    the exogenous noise trader's fills; with the noise trader disabled they sum to exactly
    zero.
    """
    for case in range(CASES):
        rng = random.Random(case)
        n_agents = rng.randint(1, 3)
        noise = rng.choice([-5.0, 2.0, 4.0])  # -5.0 makes the noise trader never fire
        env = LOBMarketEnv(
            n_agents,
            n_steps=rng.randint(5, 40),
            seed=case,
            noise_intensity=noise,
            mark=mark,
            priority=priority,
        )
        captured = _capture_fills(env)
        env.reset(seed=case)
        recorder = _Recorder(env._book)
        env._book = recorder
        expected_mark = {agent: 1000.0 for agent in env.possible_agents}
        noise_inventory = 0
        noise_cash = 0
        while env.agents:
            prev_equity = dict(env._prev_equity)
            _obs, rewards, _terms, _truncs, infos = env.step(_actions(rng, env))
            ladder = json.loads(env._book.ladder())
            for index, agent in enumerate(env.possible_agents):
                if mark == "book_mid":
                    price = ladder["mid"] or float(env._mid)
                else:
                    ex_own = _engine_ex_own_ladder(env, recorder.batches, index)
                    if ex_own["bids"] and ex_own["asks"]:
                        expected_mark[agent] = ex_own["mid"]
                    else:
                        traded = [
                            fill["price_tick"]
                            for fill in captured[-1]
                            if env._owner(fill["maker_agent"]) != env._owner(fill["taker_agent"])
                        ]
                        if traded:
                            expected_mark[agent] = float(traded[-1])
                    price = expected_mark[agent]
                    assert env._marks[agent] == price
                inventory = env._inventory[agent]
                cash = env._cash[agent]
                equity = cash + inventory * price
                assert env._prev_equity[agent] == equity
                expected = float(equity - prev_equity[agent] - env._inv_pen * inventory**2)
                assert rewards[agent] == expected
                assert infos[agent] == {"inventory": inventory, "cash": cash}
            for fill in captured[-1]:
                assert env._owner(fill["maker_agent"]) is not None, "the noise trader never rests"
                if env._owner(fill["taker_agent"]) is None:
                    sign = 1 if fill["taker_side"] == "buy" else -1
                    noise_inventory += sign * fill["qty"]
                    noise_cash -= sign * fill["price_tick"] * fill["qty"]
            assert sum(env._inventory.values()) + noise_inventory == 0
            assert sum(env._cash.values()) + noise_cash == 0
            if noise == -5.0:
                assert noise_inventory == 0 and noise_cash == 0
                assert sum(env._inventory.values()) == 0


def test_seeded_reset_reproduces_the_first_rewards():
    """Binds ``reset`` (lob_env.py:103-118): ``reset(seed)`` twice on one instance gives
    identical reward sequences, and the equity baseline restarts at zero."""
    for case in range(CASES):
        rng = random.Random(1000 + case)
        env = LOBMarketEnv(rng.randint(1, 3), n_steps=8, seed=case)
        episodes = []
        for _ in range(2):
            env.reset(seed=case)
            assert all(value == 0.0 for value in env._prev_equity.values())
            assert all(value == 0 for value in env._inventory.values())
            assert all(value == 0.0 for value in env._cash.values())
            stream = random.Random(case)
            rewards = [env.step(_actions(stream, env))[1] for _ in range(4)]
            episodes.append(rewards)
        assert episodes[0] == episodes[1]
