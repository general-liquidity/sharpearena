"""Paired-path check and PnL split for the market-making regret.

``mm_regret`` claims the candidate and the reference share their luck because they run on
the same seeds. The env draws arrivals, fills and the mid step from one generator, and the
binomial fill draw consumes a policy-dependent number of underlying draws, so that claim is
checked: an unpaired comparison raises ``UnpairedMidPathError`` instead of returning a
number. The default stream is unchanged, pinned here against the committed F2 regrets.

``step`` also splits each reward into spread capture, inventory mark-to-market, the running
inventory penalty and the terminal liquidation cost, and ``mm_pnl_split`` aggregates that
split per policy. The split is diagnostic and feeds no score.

The module is imported whole so each test fails on its own (rather than the file failing to
collect) where a symbol is missing.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

import sharpearena.market_making as mm

COMPONENTS = ("spread_capture", "inventory_pnl", "inventory_penalty", "liquidation_cost")

# The committed F2 regrets (paper/evidence/f2-regret.json: optimal_regret and
# fixed_spread_regret) at default MMParams, 16 episodes, seed_base 0, as exact doubles.
# origin/main (c179190) reproduces every one bit for bit.
F2_OPTIMAL_REGRET = 0.0
F2_FIXED_SPREAD_REGRET = {
    "0.05": float.fromhex("0x1.528b11f412cfap+6"),
    "0.1": float.fromhex("0x1.17415e4cb277ep+6"),
    "0.25": float.fromhex("0x1.2222f27972c0fp+5"),
    "0.5": float.fromhex("0x1.322edb5e274c0p-5"),
    "1.0": float.fromhex("0x1.bcbfc5930353dp+1"),
    "2.0": float.fromhex("0x1.2510ffabbdc10p+5"),
    "4.0": float.fromhex("0x1.d66f8e8f5bbe4p+5"),
}
F2_EVIDENCE = Path(__file__).resolve().parents[3] / "paper" / "evidence" / "f2-regret.json"


def _steps(params, policy, seed):
    env = mm.MarketMakingEnv(params)
    obs, _ = env.reset(seed=seed)
    out = []
    while True:
        action = policy(obs)
        q_before = int(obs["inventory"][0])
        obs, reward, terminated, truncated, info = env.step(action)
        out.append((action, q_before, reward, info))
        if terminated or truncated:
            return out


# -- A1: pairing is checked, default stream unchanged ------------------------


def test_default_regret_is_bit_identical_to_committed_f2():
    p = mm.MMParams()
    optimal = mm.mm_regret(mm.closed_form_reference_policy(p), params=p, n_episodes=16)
    assert optimal == F2_OPTIMAL_REGRET
    got = {
        key: mm.mm_regret(mm.fixed_spread_policy(float(key)), params=p, n_episodes=16)
        for key in F2_FIXED_SPREAD_REGRET
    }
    assert {k: v.hex() for k, v in got.items()} == {
        k: v.hex() for k, v in F2_FIXED_SPREAD_REGRET.items()
    }
    if F2_EVIDENCE.is_file():
        committed = json.loads(F2_EVIDENCE.read_text())
        assert committed["optimal_regret"] == optimal
        assert committed["fixed_spread_regret"] == got


def test_mm_regret_refuses_when_arrival_rate_unpairs_the_mid_path():
    params = mm.MMParams(arrival_rate=40000.0)  # 200 arrivals per step
    with pytest.raises(mm.UnpairedMidPathError) as excinfo:
        mm.mm_regret(mm.fixed_spread_policy(0.05), params=params)
    err = excinfo.value
    assert isinstance(err, ValueError)
    assert (err.seed, err.step) == (0, 0)
    assert err.reference_mid != err.candidate_mid
    message = str(err)
    assert "seed 0" in message and "step 0" in message
    assert message[0].islower() and not message.endswith(".")


def test_refusal_names_the_first_unpaired_episode_not_only_the_first_episode():
    # At 50 arrivals per step, seeds 0 and 1 still share the path and seed 2 is the
    # first that does not, 63 steps in.
    params = mm.MMParams(arrival_rate=10000.0)
    candidate = mm.fixed_spread_policy(0.05)
    assert math.isfinite(mm.mm_regret(candidate, params=params, n_episodes=2))
    with pytest.raises(mm.UnpairedMidPathError) as excinfo:
        mm.mm_regret(candidate, params=params, n_episodes=16)
    assert (excinfo.value.seed, excinfo.value.step) == (2, 63)
    with pytest.raises(mm.UnpairedMidPathError) as excinfo:
        mm.mm_regret(candidate, params=params, n_episodes=1, seed_base=2)
    assert (excinfo.value.seed, excinfo.value.step) == (2, 63)


def test_zero_fill_probability_quote_unpairs_at_the_default_rate():
    # exp(-kappa * 1e9) underflows to 0.0, and numpy draws nothing for a zero-probability
    # binomial while the reference's draw consumes the stream.
    params = mm.MMParams(max_depth=1e9)
    with pytest.raises(mm.UnpairedMidPathError):
        mm.mm_regret(mm.fixed_spread_policy(1e9), params=params)


def test_paired_rate_above_default_still_returns_a_number():
    # 40 arrivals per step keeps every fill draw on numpy's one-draw sampler for this pair.
    regret = mm.mm_regret(mm.fixed_spread_policy(0.05), params=mm.MMParams(arrival_rate=8000.0))
    assert math.isfinite(regret)


# -- A2: exact per-step split -------------------------------------------------


@pytest.mark.parametrize(
    "params",
    [
        mm.MMParams(),
        mm.MMParams(inventory_cap=5),
        mm.MMParams(arrival_rate=40000.0),
        mm.MMParams(n_steps=30, s0=10000.0),
    ],
    ids=["default", "tight_cap", "high_rate", "high_price"],
)
@pytest.mark.parametrize("policy_name", ["reference", "tight", "wide"])
def test_step_split_sums_to_reward_and_matches_its_definitions(params, policy_name):
    policy = {
        "reference": mm.closed_form_reference_policy(params),
        "tight": mm.fixed_spread_policy(0.05),
        "wide": mm.fixed_spread_policy(4.0),
    }[policy_name]
    for seed in range(4):
        steps = _steps(params, policy, seed)
        mid_before = params.s0
        for index, (action, q_before, reward, info) in enumerate(steps):
            assert set(COMPONENTS) <= set(info)
            total = sum(info[key] for key in COMPONENTS)
            assert total == pytest.approx(reward, rel=0.0, abs=1e-9)

            bid_depth, ask_depth = float(action[0]), float(action[1])
            assert info["spread_capture"] == (
                info["ask_fills"] * ask_depth + info["bid_fills"] * bid_depth
            )
            q_after_fills = q_before + info["bid_fills"] - info["ask_fills"]
            assert info["inventory_pnl"] == q_after_fills * (info["mid"] - mid_before)
            assert info["inventory_penalty"] == -params.phi * q_after_fills**2

            terminal = index == len(steps) - 1
            if terminal:
                assert info["liquidated"] == float(q_after_fills)
                assert info["liquidation_cost"] == pytest.approx(
                    -abs(q_after_fills) * params.terminal_liq_penalty, rel=0.0, abs=1e-9
                )
            else:
                assert info["liquidated"] == 0.0
                assert info["liquidation_cost"] == 0.0
            mid_before = info["mid"]


def test_existing_info_keys_keep_their_meaning():
    params = mm.MMParams(n_steps=20)
    steps = _steps(params, mm.fixed_spread_policy(0.05), 3)
    for action, q_before, reward, info in steps:
        assert {"value", "inventory", "bid_fills", "ask_fills", "mid", "liquidated"} <= set(info)
    last = steps[-1][3]
    assert last["inventory"] == 0
    assert isinstance(last["bid_fills"], int) and isinstance(last["ask_fills"], int)


def test_split_isolates_inventory_penalty_on_a_frozen_mid():
    params = mm.MMParams(n_steps=4, phi=0.5, sigma=0.0, max_depth=1e9)
    env = mm.MarketMakingEnv(params)
    env.reset(seed=0)
    env._q = 3
    _, reward, _, _, info = env.step(np.array([1e9, 1e9], dtype=np.float32))
    assert info["spread_capture"] == 0.0
    assert info["inventory_pnl"] == 0.0
    assert info["liquidation_cost"] == 0.0
    assert info["inventory_penalty"] == reward == -params.phi * 3**2


def test_mm_pnl_split_reconciles_to_mean_reward_and_to_regret():
    p = mm.MMParams()
    reference = mm.mm_pnl_split(mm.closed_form_reference_policy(p), params=p)
    tight = mm.mm_pnl_split(mm.fixed_spread_policy(0.05), params=p)
    for split in (reference, tight):
        assert isinstance(split, mm.MMPnLSplit)
        assert split.total == pytest.approx(split.reward, rel=0.0, abs=1e-9)

    env = mm.MarketMakingEnv(p)
    rewards = []
    for seed in range(16):
        obs, _ = env.reset(seed=seed)
        episode = 0.0
        while True:
            obs, r, terminated, truncated, _ = env.step(mm.fixed_spread_policy(0.05)(obs))
            episode += r
            if terminated or truncated:
                break
        rewards.append(episode)
    assert tight.reward == pytest.approx(sum(rewards) / 16, rel=0.0, abs=1e-12)

    regret = mm.mm_regret(mm.fixed_spread_policy(0.05), params=p)
    assert reference.reward - tight.reward == pytest.approx(regret, rel=0.0, abs=1e-9)
    by_component = sum(getattr(reference, k) - getattr(tight, k) for k in COMPONENTS)
    assert by_component == pytest.approx(regret, rel=0.0, abs=1e-9)
    # The single regret number hides two sources that the split separates: the tight
    # quoter captures less spread and pays a larger running inventory penalty.
    assert tight.spread_capture < reference.spread_capture
    assert tight.inventory_penalty < reference.inventory_penalty


def test_mm_pnl_split_is_seed_deterministic_and_seed_sensitive():
    p = mm.MMParams(n_steps=40)
    policy = mm.fixed_spread_policy(0.25)
    a = mm.mm_pnl_split(policy, params=p, n_episodes=4, seed_base=5)
    b = mm.mm_pnl_split(policy, params=p, n_episodes=4, seed_base=5)
    c = mm.mm_pnl_split(policy, params=p, n_episodes=4, seed_base=9)
    assert a == b
    assert a != c
