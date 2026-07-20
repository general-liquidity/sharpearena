"""Tests for the forced-liquidation cascade wrapper (Stream W3-CASCADE).

Run from the crate dir::

    python -m pytest crates/sharpearena-py/tests/test_cascade.py -q

The logic tests drive a deterministic NAV-scripted stub env so the cascade chain is an exact
function of a known NAV path (no binding needed). The live-binding test is skipped when the
native ``sharpearena`` module is unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest
import gymnasium as gym
from gymnasium import spaces

from sharpearena.cascade import (
    LiquidationCascadeEnv,
    cascade_survived,
    cascade_summary,
)


class _NavEnv(gym.Env):
    """Stub env replaying a scripted NAV path, surfacing ``info["nav"]`` and recording the
    action actually executed so an action override can be observed."""

    def __init__(self, navs, n: int = 2) -> None:
        super().__init__()
        self._navs = [float(x) for x in navs]
        self.action_space = spaces.Box(-1.0, 1.0, shape=(n,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64)
        self._i = 0
        self.executed: np.ndarray | None = None

    def _obs(self):
        return np.zeros((1,), dtype=np.float64)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._i = 0
        self.executed = None
        return self._obs(), {}

    def step(self, action):
        self.executed = np.asarray(action).copy()
        nav = self._navs[self._i]
        self._i += 1
        out_of_bars = self._i >= len(self._navs)
        terminated = nav <= 0.0
        truncated = out_of_bars and not terminated
        return self._obs(), 0.0, terminated, truncated, {"nav": nav}


def _act(env) -> np.ndarray:
    n = env.action_space.shape[0]
    return np.full((n,), 0.5, dtype=np.float32)


def _events_of(info, name):
    return [e for e in info.get("events", []) if e.get("event") == name]


# -- breach fires a bounded, ordered cascade --------------------------------

def test_cascade_fires_and_emits_ordered_bounded_chain():
    # peak=1.0; maintenance_margin=0.4 -> breach when nav <= 0.4. nav drops to 0.3.
    env = LiquidationCascadeEnv(
        _NavEnv([1.0, 1.0, 0.3]),
        maintenance_margin=0.4,
        cascade_steps=3,
        impact_per_step=0.02,
    )
    env.reset()
    a = _act(env)
    env.step(a)
    env.step(a)
    _o, _r, terminated, truncated, info = env.step(a)

    chain = info["cascade"]["events"]
    names = [e["event"] for e in chain]
    # margin_call once, then (forced_reduce, cascade_impact) per step, bounded by 3.
    assert names == [
        "margin_call",
        "forced_reduce", "cascade_impact",
        "forced_reduce", "cascade_impact",
        "forced_reduce", "cascade_impact",
    ]
    assert info["cascade"]["triggered"] is True
    assert len(_events_of(info, "cascade_impact")) == 3
    assert [e["step"] for e in _events_of(info, "cascade_impact")] == [1, 2, 3]
    # chain is also merged into info["events"] for the verifiers consumers.
    assert _events_of(info, "margin_call")[0]["deficit"] == pytest.approx(0.4 - 0.3)
    # survived (equity not wiped) -> truncated, not terminated.
    assert truncated is True and terminated is False
    assert cascade_survived(info) is True


def test_cascade_summary_counts():
    env = LiquidationCascadeEnv(_NavEnv([1.0, 0.2]), maintenance_margin=0.5, cascade_steps=2)
    env.reset()
    a = _act(env)
    env.step(a)
    _o, _r, _t, _tr, info = env.step(a)
    s = cascade_summary(info)
    assert s["triggered"] is True
    assert s["margin_calls"] == 1
    assert s["forced_reduces"] == 2
    assert s["cascade_impacts"] == 2
    assert s["survived"] is True


# -- no breach -> pass-through ----------------------------------------------

def test_no_breach_no_cascade_events():
    # nav dips to 0.7 of a peak of 1.0; margin 0.4 must NOT fire (0.7 > 0.4).
    env = LiquidationCascadeEnv(_NavEnv([1.0, 0.7, 0.6]), maintenance_margin=0.4)
    env.reset()
    a = _act(env)
    rows = [env.step(a) for _ in range(2)]
    for _o, _r, terminated, truncated, info in rows:
        assert info["cascade"]["triggered"] is False
        assert _events_of(info, "margin_call") == []
        assert terminated is False and truncated is False
        assert cascade_survived(info) is True


def test_passthrough_action_until_breach():
    base = _NavEnv([1.0, 0.9, 0.3, 0.3])
    env = LiquidationCascadeEnv(base, maintenance_margin=0.4)
    env.reset()
    a = _act(env)
    env.step(a)  # nav 1.0
    env.step(a)  # nav 0.9, no breach
    assert np.allclose(base.executed, a)  # still passing the agent's action through


# -- forced-flat override during the cascade --------------------------------

def test_position_forced_toward_flat_while_liquidating():
    # nav breaches at step 3 (latch set), step 4 stays breached -> executed forced flat.
    base = _NavEnv([1.0, 1.0, 0.3, 0.3])
    env = LiquidationCascadeEnv(base, maintenance_margin=0.4)
    env.reset()
    a = _act(env)
    env.step(a)
    env.step(a)
    env.step(a)  # breach -> latch
    assert np.allclose(base.executed, a)  # the breach step itself executed the agent action
    env.step(a)  # liquidating -> override flat
    assert np.all(base.executed == 0.0)


def test_latch_releases_on_recovery():
    base = _NavEnv([1.0, 0.3, 0.9, 0.9])
    env = LiquidationCascadeEnv(base, maintenance_margin=0.4)
    env.reset()
    a = _act(env)
    env.step(a)              # nav 1.0
    env.step(a)              # nav 0.3 -> breach, latch
    env.step(a)              # forced flat; nav 0.9 recovers above 0.4 -> latch clears
    assert np.all(base.executed == 0.0)
    env.step(a)              # recovered -> agent action restored
    assert np.allclose(base.executed, a)


# -- wipe -> terminated, not truncated --------------------------------------

def test_cascade_wipe_sets_terminated():
    # large impact so the chain drives equity below zero -> terminated.
    env = LiquidationCascadeEnv(
        _NavEnv([1.0, 0.01]),
        maintenance_margin=0.5,
        cascade_steps=5,
        impact_per_step=0.9,
    )
    env.reset()
    a = _act(env)
    env.step(a)
    _o, _r, terminated, truncated, info = env.step(a)
    assert info["cascade"]["final_nav"] <= 0.0
    assert terminated is True
    assert cascade_survived(info) is False
    # bounded: at most cascade_steps impacts even when wiping.
    assert len(_events_of(info, "cascade_impact")) <= 5


def test_base_termination_preserved():
    # base env goes bankrupt (nav <= 0) -> terminated stays true regardless of cascade.
    env = LiquidationCascadeEnv(_NavEnv([1.0, -0.1]), maintenance_margin=0.4)
    env.reset()
    a = _act(env)
    env.step(a)
    _o, _r, terminated, _tr, _info = env.step(a)
    assert terminated is True


# -- determinism ------------------------------------------------------------

def test_cascade_deterministic_event_sequence():
    def run():
        env = LiquidationCascadeEnv(
            _NavEnv([1.0, 1.2, 1.0, 0.35, 0.3, 0.25]),
            maintenance_margin=0.4,
            cascade_steps=3,
            impact_per_step=0.05,
        )
        env.reset()
        a = _act(env)
        seq = []
        for _ in range(5):
            _o, _r, _t, _tr, info = env.step(a)
            seq.append(
                [
                    (e["event"], round(float(e.get("mark_drop", e.get("deficit", e.get("fraction", 0.0)))), 10))
                    for e in info["cascade"]["events"]
                ]
            )
        return seq

    assert run() == run()


def test_five_tuple_shape_preserved():
    env = LiquidationCascadeEnv(_NavEnv([1.0, 1.0]), maintenance_margin=0.4)
    env.reset()
    out = env.step(_act(env))
    assert len(out) == 5
    _o, reward, terminated, truncated, info = out
    assert np.isfinite(reward)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    assert isinstance(info, dict)


# -- live binding (skipped when the native module is absent) -----------------

sharpearena = pytest.importorskip("sharpearena")


def test_live_cascade_extreme_scenario():
    # Drive a live env, then wrap with an extreme maintenance margin so any drawdown
    # breaches and fires a cascade; assert the 5-tuple and event plumbing hold up.
    base = sharpearena.SharpeArenaEnv(n_symbols=3, n_days=50, seed=1, distribution_mode="extreme")
    env = LiquidationCascadeEnv(base, maintenance_margin=0.999, cascade_steps=3)
    env.reset(seed=1)
    n = env.action_space.shape[0]
    a = np.full((n,), 1.0 / n, dtype=np.float32)
    fired = False
    for _ in range(40):
        _o, _r, terminated, truncated, info = env.step(a)
        assert "cascade" in info and "events" in info
        if info["cascade"]["triggered"]:
            fired = True
            assert any(e["event"] == "margin_call" for e in info["cascade"]["events"])
            assert isinstance(cascade_survived(info), bool)
        if terminated or truncated:
            break
    assert fired
