"""The PettingZoo **endogenous shared-book market** env (M2).

Behavior tests skip when ``pettingzoo`` is not installed; the optional-import test runs
*only* when pettingzoo is absent and asserts a clean import + a clear ``RuntimeError`` on
construction. The endogenous market is reached through the native ``PyMarketClearing``
pyclass; these tests exercise the real shared book (aggregate flow moves the price),
distinct from the competition env in ``test_pettingzoo.py``.
"""

import importlib.util

import numpy as np
import pytest

import openoutcry  # noqa: F401  (ensures the package imports without pettingzoo)
import openoutcry.market_env as market_env
from openoutcry.market_env import EndogenousMarketEnv, make_aec_env

_HAS_PETTINGZOO = importlib.util.find_spec("pettingzoo") is not None
needs_pz = pytest.mark.skipif(not _HAS_PETTINGZOO, reason="pettingzoo not installed")


def _make(n_agents=3, n_symbols=2, n_days=40, seed=0, **kwargs):
    return EndogenousMarketEnv(
        n_agents=n_agents, n_symbols=n_symbols, n_days=n_days, seed=seed, **kwargs
    )


def _flat(env):
    n = len(env.symbols)
    return {a: np.zeros(n, dtype=np.float32) for a in env.agents}


# ---------------------------------------------------------------------------
# Optional-import contract (runs when pettingzoo is NOT installed)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(_HAS_PETTINGZOO, reason="pettingzoo is installed")
def test_import_works_and_construction_raises_without_pettingzoo():
    """`import openoutcry.market_env` succeeds without pettingzoo; constructing the env
    (or the AEC view) raises a clear RuntimeError rather than an ImportError."""
    assert market_env._HAS_PETTINGZOO is False
    with pytest.raises(RuntimeError, match="pettingzoo is not installed"):
        EndogenousMarketEnv(n_agents=2)
    with pytest.raises(RuntimeError, match="pettingzoo is not installed"):
        make_aec_env(n_agents=2)


# ---------------------------------------------------------------------------
# Full env behavior (runs when pettingzoo IS installed)
# ---------------------------------------------------------------------------

@needs_pz
def test_construct_reset_and_step_shapes():
    env = _make()
    assert env.possible_agents == ["agent_0", "agent_1", "agent_2"]
    obs, infos = env.reset(seed=0)

    assert set(obs.keys()) == set(env.agents) == set(env.possible_agents)
    assert set(infos.keys()) == set(env.agents)
    for agent in env.agents:
        assert env.observation_space(agent).contains(obs[agent])

    for _ in range(3):
        actions = {a: env.action_space(a).sample() for a in env.agents}
        obs, rewards, terms, truncs, infos = env.step(actions)
        for d in (obs, rewards, terms, truncs, infos):
            assert set(d.keys()) == set(actions.keys())
        for a in rewards:
            assert isinstance(rewards[a], float)
            assert isinstance(terms[a], bool) and isinstance(truncs[a], bool)
    env.close()


@needs_pz
def test_all_flat_reproduces_the_frozen_exogenous_path():
    """N agents all flat == the frozen path: with zero aggregate flow the cleared price is
    the exogenous path, independent of how many agents are in the book. So a 3-agent and a
    1-agent market (same seed) clear an identical tape, and every reward is ~0."""
    many = _make(n_agents=3, n_symbols=2, n_days=30, seed=11)
    one = _make(n_agents=1, n_symbols=2, n_days=30, seed=11)
    many.reset(seed=11)
    one.reset(seed=11)

    for _ in range(8):
        _, r_many, _, tr_many, i_many = many.step(_flat(many))
        _, r_one, _, _, i_one = one.step(_flat(one))
        # zero net flow on every symbol
        assert all(abs(f) < 1e-12 for f in i_many["agent_0"]["net_flow"])
        # the cleared tape is identical regardless of agent count (nobody moved it)
        np.testing.assert_array_equal(
            np.array(i_many["agent_0"]["cleared_mids"]),
            np.array(i_one["agent_0"]["cleared_mids"]),
        )
        # flat books earn exactly zero
        assert all(abs(v) < 1e-12 for v in r_many.values())
        if all(tr_many.values()):
            break
    many.close()
    one.close()


@needs_pz
def test_a_coordinated_buy_moves_the_cleared_price_up():
    """A coordinated buy moves the price: against an identical-seed flat market, the
    buying market's cleared price sits strictly above the (exogenous) flat path once the
    permanent impact of the first bar's flow has accumulated."""
    buy_env = _make(n_agents=3, n_symbols=2, n_days=30, seed=6)
    flat_env = _make(n_agents=3, n_symbols=2, n_days=30, seed=6)
    buy_env.reset(seed=6)
    flat_env.reset(seed=6)

    n = len(buy_env.symbols)
    buy = {a: np.full(n, 0.8, dtype=np.float32) for a in buy_env.agents}

    # Bar 1: establish the position (positive flow); impact lands on the next bar.
    _, _, _, _, i1 = buy_env.step(buy)
    flat_env.step(_flat(flat_env))
    assert all(f > 0.0 for f in i1["agent_0"]["net_flow"])

    # Bar 2: hold the target (no fresh flow) — the permanent impact persists, so the
    # cleared price is now strictly above the exogenous flat path.
    _, _, _, _, ib = buy_env.step(buy)
    _, _, _, _, ifl = flat_env.step(_flat(flat_env))
    cm_buy = ib["agent_0"]["cleared_mids"]
    cm_flat = ifl["agent_0"]["cleared_mids"]
    assert all(b > f for b, f in zip(cm_buy, cm_flat)), (cm_buy, cm_flat)
    buy_env.close()
    flat_env.close()


@needs_pz
def test_deterministic_under_fixed_seed_and_actions():
    """Same seed + same action sequence -> byte-identical observations and rewards."""
    def rollout():
        env = _make(n_agents=3, n_symbols=2, n_days=30, seed=5)
        env.reset(seed=5)
        acts = {f"agent_{i}": np.full(2, 0.1 * (i + 1), dtype=np.float32) for i in range(3)}
        log = []
        for _ in range(8):
            obs, rewards, _, truncs, _ = env.step(acts)
            log.append((obs, rewards))
            if all(truncs.values()):
                break
        env.close()
        return log

    a, b = rollout(), rollout()
    assert len(a) == len(b)
    for (oa, ra), (ob, rb) in zip(a, b):
        assert ra == rb
        assert set(oa) == set(ob)
        for agent in oa:
            for key in oa[agent]:
                np.testing.assert_array_equal(oa[agent][key], ob[agent][key])


@needs_pz
def test_vol_scale_zero_matches_default_behavior():
    """vol_scale=0 (explicit) clears a byte-identical tape and fills to the default env, so
    volatility-scaled costs are strictly opt-in."""
    default_env = _make(n_agents=2, n_symbols=2, n_days=30, seed=7)
    zero_env = _make(n_agents=2, n_symbols=2, n_days=30, seed=7, vol_scale=0.0)
    default_env.reset(seed=7)
    zero_env.reset(seed=7)

    def buy(env):
        return {a: np.full(len(env.symbols), 0.5, dtype=np.float32) for a in env.agents}

    for _ in range(8):
        _, rd, _, td, idd = default_env.step(buy(default_env))
        _, rz, _, _, iz = zero_env.step(buy(zero_env))
        assert rd == rz
        np.testing.assert_array_equal(
            np.array(idd["agent_0"]["cleared_mids"]),
            np.array(iz["agent_0"]["cleared_mids"]),
        )
        for agent in idd:
            f_default = [f["fill_price"] for f in idd[agent]["fills"]]
            f_zero = [f["fill_price"] for f in iz[agent]["fills"]]
            np.testing.assert_array_equal(np.array(f_default), np.array(f_zero))
        if all(td.values()):
            break
    default_env.close()
    zero_env.close()


@needs_pz
def test_vol_scale_positive_widens_fills_on_a_volatile_path():
    """A positive vol_scale is accepted by the rebuilt binding and widens execution costs
    on a volatile path: with the same path and orders only the fill price moves, so the
    impact away from the cleared mid is at least as large and strictly larger somewhere.
    Skips when the installed binding predates the parameter (needs a rebuild)."""
    try:
        vol_env = _make(
            n_agents=2, n_symbols=2, n_days=40, seed=3,
            distribution_mode="extreme", vol_scale=8.0,
        )
    except TypeError:
        pytest.skip("native binding lacks the vol_scale parameter (needs rebuild)")
    base_env = _make(
        n_agents=2, n_symbols=2, n_days=40, seed=3,
        distribution_mode="extreme", vol_scale=0.0,
    )
    vol_env.reset(seed=3)
    base_env.reset(seed=3)

    def buy(env):
        return {a: np.full(len(env.symbols), 0.6, dtype=np.float32) for a in env.agents}

    # The entry bar establishes the position (large flow); vol scaling widens that fill.
    _, _, _, _, iv = vol_env.step(buy(vol_env))
    _, _, _, _, ib = base_env.step(buy(base_env))
    agent = "agent_0"
    mids = ib[agent]["cleared_mids"]
    np.testing.assert_allclose(iv[agent]["cleared_mids"], mids)

    wider = 0
    for s, (fv, fb) in enumerate(zip(iv[agent]["fills"], ib[agent]["fills"])):
        base_impact = abs(fb["fill_price"] - mids[s])
        vol_impact = abs(fv["fill_price"] - mids[s])
        assert vol_impact >= base_impact - 1e-12
        if vol_impact > base_impact + 1e-9:
            wider += 1
    assert wider > 0, "vol_scale>0 must widen at least one fill on a volatile path"
    vol_env.close()
    base_env.close()


@needs_pz
def test_parallel_api_test():
    from pettingzoo.test import parallel_api_test

    env = _make(n_agents=2, n_symbols=3, n_days=24, seed=0)
    parallel_api_test(env, num_cycles=30)


@needs_pz
def test_aec_view_constructs_and_steps():
    """The AEC view derived via parallel_to_aec is a usable PettingZoo AEC env."""
    aec = make_aec_env(n_agents=2, n_symbols=2, n_days=24, seed=0)
    aec.reset(seed=0)
    assert set(aec.possible_agents) == {"agent_0", "agent_1"}
    for agent in aec.agent_iter(max_iter=8):
        _, _, termination, truncation, _ = aec.last()
        action = None if (termination or truncation) else aec.action_space(agent).sample()
        aec.step(action)
    aec.close()
