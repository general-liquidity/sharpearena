"""Reward eligibility uses the episode, not just a favorable return prefix.

These run the actual verifiers rubric with synthetic market steps. Merely checking
the zero-weight process metric would pass while the agent still receives reward.
"""

import asyncio
import copy

import pytest

vf = pytest.importorskip("verifiers")

from sharpearena.rewards import build_scheme_rubric, list_reward_schemes
from sharpearena.verifiers_env import load_environment

ACTION = '<action>{"orders": [], "reasoning": "hold"}</action>'


def respond(env, state, text=ACTION):
    return asyncio.run(env.env_response([vf.AssistantMessage(content=text)], state))


def score(state, scheme="default", mandate=True):
    state["prompt"] = []
    state["completion"] = [vf.AssistantMessage(content=ACTION)]
    asyncio.run(build_scheme_rubric(scheme, mandate=mandate).score_rollout(state))
    return state["reward"]


@pytest.fixture
def market(monkeypatch):
    env = load_environment(n_symbols=1, n_days=4, n_windows=2, max_episode_bars=3)
    state = {"info": {"seed": 0, "mandate": {"style": "unconstrained"}}}
    asyncio.run(env.setup_state(state))
    native = state["_oo_env"]
    obs = state["_oo_last_obs"]
    calls = {"step": 0, "close": 0}

    def step(action):
        calls["step"] += 1
        return obs, 0.2, False, False, {"events": []}

    def close():
        calls["close"] += 1

    monkeypatch.setattr(native, "step", step)
    monkeypatch.setattr(native, "close", close)
    yield env, state, native, calls
    if state.get("_oo_env") is not None:
        env._close_env(state)


@pytest.mark.parametrize("scheme", list_reward_schemes())
@pytest.mark.parametrize("mandate", [False, True])
def test_profitable_prefix_then_protocol_failure_receives_composite_floor(
    market, scheme, mandate
):
    env, state, _, calls = market
    respond(env, state)
    respond(env, state, "malformed completion")
    assert state["returns"] == [0.2]  # Retain evidence; do not fabricate hold bars.
    assert score(state, scheme, mandate) == -1.0
    assert state["episode"] == {
        "schema_version": 1,
        "requested_bars": 3,
        "available_bars": 4,
        "planned_bars": 3,
        "realized_bars": 1,
        "status": "failed",
        "reason": "protocol_error",
    }
    assert calls == {"step": 1, "close": 1}


def test_terminal_response_cannot_reset_or_erase_failure(market):
    env, state, _, calls = market
    respond(env, state, "malformed")
    before = copy.deepcopy(state)
    with pytest.raises(RuntimeError, match="terminal"):
        respond(env, state)
    assert state == before
    assert calls == {"step": 0, "close": 1}


@pytest.mark.parametrize("scheme", list_reward_schemes())
def test_completed_horizon_remains_reward_eligible(market, scheme):
    env, state, _, calls = market
    for _ in range(3):
        response = respond(env, state)
    assert score(state, scheme) > -1.0
    assert state["episode"]["status"] == "completed"
    assert state["episode"]["realized_bars"] == 3
    assert state["final_env_response"] == response  # No extra model turn after close.
    assert calls == {"step": 3, "close": 1}


@pytest.mark.parametrize("stop", ["max_turns_reached", "prompt_too_long", "timeout"])
def test_framework_cutoff_closes_market_without_crediting_prefix(market, stop):
    env, state, _, calls = market
    respond(env, state)
    state["stop_condition"] = stop
    # Check registration as well as the handler: otherwise real rollouts can leak.
    assert env.finalize_episode in env._cleanup_handlers
    asyncio.run(env.finalize_episode(state))
    asyncio.run(env.finalize_episode(state))
    assert state["episode"]["status"] == "incomplete"
    assert state["episode"]["reason"] == stop
    assert state["episode"]["realized_bars"] == 1
    assert score(state) == -1.0
    assert calls == {"step": 1, "close": 1}


@pytest.mark.parametrize("terminal", ["bankruptcy", "early_truncation", "engine_error"])
def test_market_failure_never_scores_a_profitable_prefix(market, monkeypatch, terminal):
    env, state, native, _ = market
    respond(env, state)

    def failure(action):
        if terminal == "engine_error":
            raise OSError("synthetic market failure")
        return state["_oo_last_obs"], 0.2, terminal == "bankruptcy", True, {}

    monkeypatch.setattr(native, "step", failure)
    if terminal == "engine_error":
        with pytest.raises(vf.InfraError, match="market step"):
            respond(env, state)
    else:
        respond(env, state)
    assert score(state) == -1.0
    assert state["episode"]["reason"] == terminal
    assert "_oo_env" not in state


def test_missing_outcome_cannot_be_submitted_as_a_completed_rollout():
    assert score({"returns": [0.2] * 4, "events": []}) == -1.0


@pytest.mark.parametrize(
    "event",
    [
        {"event": "manipulative_order"},
        {"event": "risk_refusal", "severity": "block"},
    ],
)
def test_process_block_is_a_reward_gate_not_only_a_metric(market, event):
    env, state, _, _ = market
    for _ in range(3):
        respond(env, state)
    state["events"].append(event)
    assert score(state) == -1.0


@pytest.mark.parametrize("mutation", ["count", "horizon", "nan", "error", "timeout"])
def test_inconsistent_completed_record_cannot_keep_credit(market, mutation):
    env, state, _, _ = market
    for _ in range(3):
        respond(env, state)
    if mutation == "count":
        state["episode"]["realized_bars"] = 2
    elif mutation == "horizon":
        state["episode"]["planned_bars"] = 2
    elif mutation == "nan":
        state["returns"][0] = float("nan")
    elif mutation == "timeout":
        state["timed_out"] = True
    else:
        state["error"] = vf.ModelError("synthetic failure")
    assert score(state) == -1.0


def test_native_window_end_is_a_complete_horizon_not_an_abort():
    env = load_environment(n_symbols=1, n_days=4, n_windows=2, max_episode_bars=8)
    state = {"info": {"seed": 0}}
    for _ in range(4):
        respond(env, state)
    assert state["episode"]["available_bars"] == 4
    assert state["episode"]["planned_bars"] == 4
    assert state["episode"]["status"] == "completed"
    assert score(state) > -1.0


@pytest.mark.parametrize("bars", [0, -1])
def test_nonpositive_episode_cap_is_rejected(bars):
    with pytest.raises(ValueError, match="max_episode_bars"):
        load_environment(n_windows=2, max_episode_bars=bars)


def test_time_aversion_reads_the_planned_horizon_even_for_a_partial_trace():
    from sharpearena.rewards import time_inhomogeneous_vol_aversion

    returns = [0.1, -0.02, 0.03, -0.01, 0.1, -0.02]
    state = {"returns": returns, "episode": {"planned_bars": 12}}
    assert time_inhomogeneous_vol_aversion(state=state) == pytest.approx(
        time_inhomogeneous_vol_aversion(state={"returns": returns}, horizon=12)
    )
    assert time_inhomogeneous_vol_aversion(state=state) != pytest.approx(
        time_inhomogeneous_vol_aversion(state={"returns": returns})
    )


@pytest.mark.parametrize("mode", ["complete", "turn_limit", "model_error", "timeout"])
def test_real_framework_rollout_finalizes_and_scores_without_a_model(monkeypatch, mode):
    from verifiers.envs import environment as framework
    from verifiers.types import Response, ResponseMessage
    from sharpearena.gym import SharpeArenaEnv

    # Only the client construction and response are synthetic. Keep the actual
    # framework loop, trajectory, stop handlers, cleanup and reward routing.
    monkeypatch.setattr(framework, "resolve_client", lambda client: None)
    env = load_environment(
        n_symbols=1,
        n_days=4,
        n_windows=2,
        max_episode_bars=2,
        max_turns=2 if mode == "turn_limit" else 10,
        timeout_seconds=0.1 if mode == "timeout" else None,
    )
    calls = {"responses": 0, "close": 0}
    original_close = SharpeArenaEnv.close

    def close(market):
        calls["close"] += 1
        original_close(market)

    async def synthetic_response(state, messages):
        calls["responses"] += 1
        if calls["responses"] == 2:
            if mode == "model_error":
                raise vf.ModelError("synthetic response failure")
            if mode == "timeout":
                await asyncio.Future()  # Framework timeout cancels this, no network.
        return Response(
            id="fixture",
            created=0,
            model="fixture",
            message=ResponseMessage(
                content=ACTION, finish_reason="stop", is_truncated=False
            ),
        )

    monkeypatch.setattr(SharpeArenaEnv, "close", close)
    monkeypatch.setattr(env, "get_model_response", synthetic_response)
    state = asyncio.run(
        env.rollout(
            {"prompt": [vf.UserMessage(content="fixture")], "info": {"seed": 0}},
            client=None,
            model="fixture",
        )
    )
    assert calls == {"responses": 2, "close": 1}
    assert "_oo_env" not in state
    asyncio.run(env.rubric.score_rollout(state))
    if mode == "complete":
        assert state["episode"]["status"] == "completed"
        assert len(state["returns"]) == 2
        assert state["reward"] > -1.0
    else:
        assert state["episode"]["status"] in ("failed", "incomplete")
        assert len(state["returns"]) == 1
        assert state["reward"] == -1.0


def test_nonfinite_market_return_is_a_failure_not_a_truncated_numeric_trace(
    market, monkeypatch
):
    env, state, native, _ = market
    monkeypatch.setattr(
        native,
        "step",
        lambda action: (state["_oo_last_obs"], float("nan"), False, False, {}),
    )
    with pytest.raises(vf.InfraError, match="nonfinite"):
        respond(env, state)
    assert state["episode"]["reason"] == "nonfinite_return"
    assert state["returns"] == []
    assert score(state) == -1.0
