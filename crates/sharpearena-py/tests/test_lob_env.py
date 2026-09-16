"""Tests for the M3 limit-order-book env + the native PyOrderBook engine boundary."""

import hashlib
import json
import random

import numpy as np
import pytest


def _book():
    from sharpearena.sharpearena_py import PyOrderBook

    return PyOrderBook(tick_size=0.01, levels=5)


def test_orderbook_resting_and_ladder():
    b = _book()
    b.reset_book()
    r = json.loads(
        b.step_book(
            json.dumps(
                [
                    {"agent": 0, "kind": "limit", "side": "buy", "price_tick": 99, "qty": 10},
                    {"agent": 1, "kind": "limit", "side": "sell", "price_tick": 101, "qty": 10},
                ]
            )
        )
    )
    assert r["ladder"]["bids"] == [[99, 10]]
    assert r["ladder"]["asks"] == [[101, 10]]
    assert r["ladder"]["mid"] == 100.0
    assert -1.0 <= r["ladder"]["queue_imbalance"] <= 1.0


def test_orderbook_market_order_crosses():
    b = _book()
    b.reset_book()
    b.step_book(json.dumps([{"agent": 1, "kind": "limit", "side": "sell", "price_tick": 101, "qty": 10}]))
    r = json.loads(b.step_book(json.dumps([{"agent": 2, "kind": "market", "side": "buy", "qty": 4}])))
    assert len(r["fills"]) == 1
    f = r["fills"][0]
    assert f["price_tick"] == 101 and f["qty"] == 4 and f["taker_side"] == "buy" and f["maker_agent"] == 1
    # 4 of the 10 resting were consumed.
    assert r["ladder"]["asks"] == [[101, 6]]


def test_orderbook_canonical_order_deterministic():
    def run(orders):
        b = _book()
        b.reset_book()
        return b.step_book(json.dumps(orders))

    o = [
        {"agent": 0, "kind": "limit", "side": "buy", "price_tick": 99, "qty": 5},
        {"agent": 1, "kind": "limit", "side": "sell", "price_tick": 101, "qty": 5},
    ]
    assert run(o) == run(list(reversed(o)))  # reorder input -> identical (canonical sort)


def test_orderbook_rejects_bad_order():
    b = _book()
    with pytest.raises(Exception):
        b.step_book(json.dumps([{"agent": 0, "kind": "limit", "side": "buy"}]))  # missing price/qty


def test_orderbook_sweep_cost_is_read_only():
    b = _book()
    b.reset_book()
    b.step_book(
        json.dumps(
            [
                {"agent": 0, "kind": "limit", "side": "sell", "price_tick": 100, "qty": 2},
                {"agent": 1, "kind": "limit", "side": "sell", "price_tick": 101, "qty": 2},
                {"agent": 2, "kind": "limit", "side": "sell", "price_tick": 102, "qty": 2},
            ]
        )
    )
    before = json.loads(b.ladder())
    c = json.loads(b.sweep_cost("buy", 5))
    assert c["filled_qty"] == 5
    assert c["avg_px_tick"] == 504.0 / 5.0  # (2*100 + 2*101 + 1*102) / 5
    assert c["slippage_ticks"] == abs(504.0 / 5.0 - 100.0)
    # Read-only: the book is byte-identical after the query.
    assert json.loads(b.ladder()) == before


def test_orderbook_sweep_cost_partial_and_empty():
    b = _book()
    b.reset_book()
    b.step_book(json.dumps([{"agent": 0, "kind": "limit", "side": "sell", "price_tick": 101, "qty": 6}]))
    c = json.loads(b.sweep_cost("buy", 100))
    assert c["filled_qty"] == 6 and c["avg_px_tick"] == 101.0 and c["slippage_ticks"] == 0.0
    b.reset_book()
    empty = json.loads(b.sweep_cost("buy", 10))
    assert empty["filled_qty"] == 0 and empty["avg_px_tick"] == 0.0


def test_orderbook_uncross_none_on_uncrossed_book():
    b = _book()
    b.reset_book()
    b.step_book(
        json.dumps(
            [
                {"agent": 0, "kind": "limit", "side": "buy", "price_tick": 99, "qty": 10},
                {"agent": 1, "kind": "limit", "side": "sell", "price_tick": 101, "qty": 10},
            ]
        )
    )
    assert json.loads(b.uncross()) is None


pettingzoo = pytest.importorskip("pettingzoo")


def test_lob_env_constructs_and_steps():
    from sharpearena.lob_env import LOBMarketEnv

    env = LOBMarketEnv(n_agents=2, n_steps=10, seed=1)
    obs, infos = env.reset(seed=1)
    assert set(obs) == {"agent_0", "agent_1"}
    assert obs["agent_0"].shape == env.observation_space("agent_0").shape
    actions = {a: env.action_space(a).sample() for a in env.agents}
    obs, rewards, terms, truncs, infos = env.step(actions)
    assert set(rewards) == {"agent_0", "agent_1"}
    assert all(np.isfinite(v) for v in rewards.values())


def test_lob_env_deterministic():
    from sharpearena.lob_env import LOBMarketEnv, symmetric_quote_policy

    def rollout(seed):
        env = LOBMarketEnv(n_agents=2, n_steps=12, seed=seed)
        env.reset(seed=seed)
        out = []
        done = False
        while not done:
            acts = {a: symmetric_quote_policy(offset=3) for a in env.agents}
            _o, r, _t, tr, _i = env.step(acts)
            out.append(tuple(round(x, 6) for x in r.values()))
            done = any(tr.values())
        return out

    assert rollout(7) == rollout(7)
    assert rollout(7) != rollout(8)


def test_lob_env_parallel_api():
    from pettingzoo.test import parallel_api_test
    from sharpearena.lob_env import LOBMarketEnv

    parallel_api_test(LOBMarketEnv(n_agents=2, n_steps=20, seed=2), num_cycles=10)


@pytest.mark.parametrize("first_steps", [3, 12])
def test_reset_after_partial_or_terminal_episode_replays_rewards(first_steps):
    from sharpearena.lob_env import LOBMarketEnv, symmetric_quote_policy

    env = LOBMarketEnv(n_agents=2, n_steps=12, seed=7)
    env.reset()
    for _ in range(first_steps):
        env.step({a: symmetric_quote_policy(offset=3) for a in env.agents})
    # Ensure the old state could contaminate the reward, not merely a flat reset.
    assert any(value != 0 for value in env._prev_equity.values())
    actual, _ = env.reset()
    fresh = LOBMarketEnv(n_agents=2, n_steps=12, seed=7)
    expected, _ = fresh.reset()
    for agent in env.agents:
        np.testing.assert_array_equal(actual[agent], expected[agent])
    while env.agents:
        actions = {a: symmetric_quote_policy(offset=3) for a in env.agents}
        actual, expected = env.step(actions), fresh.step(actions)
        assert actual[1:] == expected[1:]
        for agent in actual[0]:
            np.testing.assert_array_equal(actual[0][agent], expected[0][agent])


# ---------------------------------------------------------------------------
# Inventory mark (A5): the agent's own resting quotes must not value its inventory.
# ---------------------------------------------------------------------------


def _fill_log(env):
    """Record each step's fill tape without changing what the environment books."""
    log = []
    original = env._apply_fills

    def wrapped(fills, ladder):
        log.append(fills)
        original(fills, ladder)

    env._apply_fills = wrapped
    return log


def _quote(bid_offset, ask_offset):
    return np.array([bid_offset, ask_offset], dtype=np.float32)


@pytest.mark.parametrize("quote", [(1, 20), (20, 1), (3, 3)])
def test_lone_agent_equity_moves_only_when_it_trades(quote):
    """A lone agent's only counterparty is the noise trader. On a step with no fill its
    cash and inventory are unchanged, so its equity must be too, and the reward is the
    inventory penalty alone. Under the book-mid mark used before 2026-09-16 an asymmetric
    quote moved the book mid and booked that walk as P&L (83 no-fill steps over these
    seeds for ``(1, 20)``)."""
    from sharpearena.lob_env import LOBMarketEnv

    for seed in range(8):
        env = LOBMarketEnv(1, n_steps=120, seed=seed)
        log = _fill_log(env)
        env.reset(seed=seed)
        no_fill_steps = 0
        while env.agents:
            _o, rewards, _t, _tr, _i = env.step({"agent_0": _quote(*quote)})
            if not log[-1]:
                no_fill_steps += 1
                penalty = env._inv_pen * env._inventory["agent_0"] ** 2
                assert rewards["agent_0"] == -penalty
        assert no_fill_steps > 0


def test_hand_trace_quote_walk_moves_the_book_mid_but_not_the_mark():
    """The reader's hand trace, executed. Noise trader off, one agent quoting ``(1, 20)``:
    each new bid sits below the stale ask, so nothing fills while the reference mid walks
    1000, 1010, 1014, 1016, 1018, 1018. The default mark stays at the opening 1000."""
    from sharpearena.lob_env import LOBMarketEnv

    walks = {}
    for quote in [(1, 20), (3, 3)]:
        env = LOBMarketEnv(1, n_steps=6, seed=0, noise_intensity=-5.0)
        log = _fill_log(env)
        env.reset(seed=0)
        mids = [env._mid]
        while env.agents:
            env.step({"agent_0": _quote(*quote)})
            mids.append(env._mid)
            assert env._marks["agent_0"] == 1000.0
        assert not any(log)
        walks[quote] = mids
    assert walks[(1, 20)] == [1000, 1010, 1014, 1016, 1018, 1018, 1018]
    assert walks[(3, 3)] == [1000] * 7


def test_self_trade_needs_an_emptied_side_and_nets_to_zero():
    """The book has no self-trade prevention. A quote crosses a resting order only after a
    step left one side of the book empty, and a self-trade leaves the agent's cash and
    inventory exactly where its fills with the noise trader put them. Seed 327 with
    ``(1, 3)`` quotes self-trades twice, at step 76."""
    from sharpearena.lob_env import LOBMarketEnv

    env = LOBMarketEnv(1, n_steps=80, seed=327, noise_intensity=4.0, max_offset=3)
    log = _fill_log(env)
    env.reset(seed=327)
    self_trades = 0
    step = 0
    while env.agents:
        before = json.loads(env._book.ladder())
        inventory, cash = env._inventory["agent_0"], env._cash["agent_0"]
        env.step({"agent_0": _quote(1, 3)})
        external = [f for f in log[-1] if env._owner(f["taker_agent"]) is None]
        own = [f for f in log[-1] if env._owner(f["taker_agent"]) is not None]
        if own:
            assert step > 0 and not (before["bids"] and before["asks"])
        self_trades += sum(f["maker_agent"] == f["taker_agent"] for f in own)
        units = sum((-1 if f["taker_side"] == "buy" else 1) * f["qty"] for f in external)
        cash_in = sum(
            (1 if f["taker_side"] == "buy" else -1) * f["price_tick"] * f["qty"] for f in external
        )
        assert env._inventory["agent_0"] == inventory + units
        assert env._cash["agent_0"] == cash + cash_in
        assert env._marks["agent_0"] == 1000.0
        step += 1
    assert self_trades == 2


def test_agent_quotes_cross_only_after_a_side_empties():
    """Why self-trades are rare: while both sides of the book exist, the reference mid is
    their rounded midpoint, so no new quote is marketable."""
    from sharpearena.lob_env import LOBMarketEnv

    for case in range(60):
        rng = random.Random(case)
        env = LOBMarketEnv(
            rng.randint(1, 3),
            n_steps=60,
            seed=case,
            noise_intensity=rng.choice([0.0, 2.0, 4.0]),
            max_offset=rng.choice([3, 20]),
            priority=rng.choice(["agent_index", "seeded_shuffle"]),
        )
        log = _fill_log(env)
        env.reset(seed=case)
        step = 0
        while env.agents:
            before = json.loads(env._book.ladder())
            top = env._max_offset
            env.step({a: [rng.randint(1, top), rng.randint(1, top)] for a in env.agents})
            if any(env._owner(f["taker_agent"]) is not None for f in log[-1]):
                assert step > 0 and not (before["bids"] and before["asks"])
            step += 1


def test_unknown_mark_or_priority_is_refused():
    from sharpearena.lob_env import LOBMarketEnv

    with pytest.raises(ValueError, match="mark"):
        LOBMarketEnv(2, mark="last_trade")
    with pytest.raises(ValueError, match="priority"):
        LOBMarketEnv(2, priority="pro_rata")
    env = LOBMarketEnv(2)
    assert (env.mark, env.priority) == ("ex_own_mid", "agent_index")


# ---------------------------------------------------------------------------
# Byte identity with the environment before the mark and priority rules existed.
# ---------------------------------------------------------------------------


class _BookRecorder:
    """Delegates to the environment's book and keeps every batch and engine reply."""

    def __init__(self, book, sink):
        self._book, self._sink = book, sink

    def step_book(self, orders_json):
        self._sink.append(orders_json.encode())
        out = self._book.step_book(orders_json)
        self._sink.append(out.encode())
        return out

    def __getattr__(self, name):
        return getattr(self._book, name)


def _fingerprint(**kwargs):
    """SHA-256 of every submitted batch, engine reply, observation, info and done flag
    (``tape``), and of every reward (``rewards``), over twelve seeded random episodes."""
    from sharpearena.lob_env import LOBMarketEnv

    tape, rew = hashlib.sha256(), hashlib.sha256()
    for case in range(12):
        rng = random.Random(case)
        n = 1 + case % 3
        noise = (-5.0, 2.0, 4.0)[case % 3 if case < 9 else (case + 1) % 3]
        max_offset = (3, 20)[case % 2]
        env = LOBMarketEnv(
            n, n_steps=40, seed=case, noise_intensity=noise, max_offset=max_offset, **kwargs
        )
        env.reset(seed=case)
        sink = []
        env._book = _BookRecorder(env._book, sink)
        while env.agents:
            acts = {a: [rng.randint(1, max_offset), rng.randint(1, max_offset)] for a in env.agents}
            obs, rewards, terms, truncs, infos = env.step(acts)
            for chunk in sink:
                tape.update(chunk)
            sink.clear()
            for a in sorted(obs):
                tape.update(a.encode())
                tape.update(obs[a].tobytes())
                tape.update(json.dumps(infos[a], sort_keys=True).encode())
                tape.update(json.dumps([terms[a], truncs[a]]).encode())
                rew.update(a.encode())
                rew.update(float(rewards[a]).hex().encode())
    return tape.hexdigest(), rew.hexdigest()


# `_fingerprint()` on the unmodified tree at origin/main c179190 (2026-09-16), before
# `mark` and `priority` existed.
_PRE_RULE_TAPE = "bfa73717f0e576b0c166dc77a52f9e95f557f1b921d7ff1c967d528cb60a4330"
_PRE_RULE_REWARDS = "81c810ffd3fbe6ce1f58d6a844b9f51efeef1adfacf8947e01d26ff74866e75c"


def test_default_priority_book_tape_is_byte_identical_to_the_pre_rule_env():
    """The default ``agent_index`` path submits the same batches and gets the same fills,
    ladders, observations and infos as before. Only the default mark changes rewards."""
    tape, rewards = _fingerprint()
    assert tape == _PRE_RULE_TAPE
    assert rewards != _PRE_RULE_REWARDS


def test_book_mid_mark_replays_the_pre_fix_environment_exactly():
    expected = (_PRE_RULE_TAPE, _PRE_RULE_REWARDS)
    assert _fingerprint(mark="book_mid") == expected
    assert _fingerprint(mark="book_mid", priority="agent_index") == expected


# ---------------------------------------------------------------------------
# Same-bar seat priority (A7).
# ---------------------------------------------------------------------------


def _maker_units(seed, n, priority):
    """Units each seat sold or bought as a resting maker, identical ``(3, 3)`` quoters."""
    from sharpearena.lob_env import LOBMarketEnv

    env = LOBMarketEnv(n, n_steps=120, seed=seed, priority=priority)
    log = _fill_log(env)
    env.reset(seed=seed)
    while env.agents:
        env.step({a: _quote(3, 3) for a in env.agents})
    units = [0] * n
    for fills in log:
        for f in fills:
            units[env._owner(f["maker_agent"])] += f["qty"]
    return units


def test_default_priority_always_queues_seat_zero_first():
    """The measured gap: two identical quoters, seat 0 filled more on 32 of 32 seeds."""
    units = [_maker_units(seed, 2, "agent_index") for seed in range(32)]
    assert sum(a > b for a, b in units) == 32
    assert [sum(col) for col in zip(*units)] == [14222, 13900]


def test_seeded_shuffle_makes_identical_quoters_exchangeable():
    """Under the opt-in rule seat 0 leads on a count of seeds inside the two-sided
    binomial(32, 1/2) acceptance band 9 to 23 (about 99%), and the totals are close."""
    units = [_maker_units(seed, 2, "seeded_shuffle") for seed in range(32)]
    ahead = sum(a > b for a, b in units)
    assert 9 <= ahead <= 23
    assert ahead == 15
    assert [sum(col) for col in zip(*units)] == [14081, 14041]


def test_seeded_shuffle_orders_every_pair_each_way_about_half_the_time():
    from sharpearena.lob_env import LOBMarketEnv

    env = LOBMarketEnv(3, seed=0, priority="seeded_shuffle")
    first = [0, 0, 0]
    ahead = {(0, 1): 0, (0, 2): 0, (1, 2): 0}
    draws = 0
    for seed in range(64):
        env._seed = seed
        for step in range(200):
            env._step = step
            codes = env._seat_codes()
            assert sorted(code // 3 for code in codes) == [0, 1, 2]
            assert [env._owner(code) for code in codes] == [0, 1, 2]
            order = sorted(range(3), key=codes.__getitem__)
            first[order[0]] += 1
            for i, j in ahead:
                ahead[(i, j)] += order.index(i) < order.index(j)
            draws += 1
    assert all(abs(count / draws - 1 / 3) < 0.03 for count in first)
    assert all(abs(count / draws - 1 / 2) < 0.03 for count in ahead.values())


@pytest.mark.parametrize("priority", ["agent_index", "seeded_shuffle"])
def test_first_fill_at_a_shared_tick_goes_to_the_first_seat_of_the_bar(priority):
    """Both agents rest a quote at the same ticks on the opening step, and the noise
    trader's market order fills the seat the bar ranked first."""
    from sharpearena.lob_env import LOBMarketEnv

    first_makers = set()
    for seed in range(32):
        env = LOBMarketEnv(2, n_steps=4, seed=seed, priority=priority)
        log = _fill_log(env)
        env.reset(seed=seed)
        codes = env._seat_codes()
        env.step({a: _quote(3, 3) for a in env.agents})
        if not log[-1]:
            continue
        leader = min(range(2), key=codes.__getitem__)
        assert env._owner(log[-1][0]["maker_agent"]) == leader
        first_makers.add(leader)
    assert first_makers == ({0} if priority == "agent_index" else {0, 1})


def test_seeded_shuffle_keeps_the_noise_trader_stream():
    """The permutation draws from its own stream, so both rules see the same market
    orders on a seed."""
    from sharpearena.lob_env import LOBMarketEnv

    def market_orders(priority):
        env = LOBMarketEnv(3, n_steps=50, seed=11, priority=priority)
        env.reset(seed=11)
        sink = []
        env._book = _BookRecorder(env._book, sink)
        while env.agents:
            env.step({a: _quote(2, 2) for a in env.agents})
        batches = [json.loads(chunk) for chunk in sink[::2]]
        return [(o["side"], o["qty"]) for batch in batches for o in batch if o["kind"] == "market"]

    assert market_orders("agent_index")
    assert market_orders("seeded_shuffle") == market_orders("agent_index")
