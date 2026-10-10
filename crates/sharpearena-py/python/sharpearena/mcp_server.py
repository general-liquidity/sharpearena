"""An MCP server exposing SharpeArena's reset/step/spec over streamable HTTP.

A thin :class:`~mcp.server.fastmcp.FastMCP` server so any MCP-capable harness — e.g.
PrimeIntellect ``rlm``, which reads a standard ``mcpServers`` URL map over streamable
HTTP — can drive an :class:`~sharpearena.gym.SharpeArenaEnv` episode with zero glue.

``mcp`` is an OPTIONAL dependency (guarded exactly like ``verifiers_env`` guards
``verifiers``): ``import sharpearena.mcp_server`` succeeds without it; :func:`build_server`
/ :func:`main` raise a clear error if called without it. The rest of the package is
unaffected.

Malformed decisions and typed input refusals are returned as JSON error objects
in the tool's string result. This is not the MCP ``structuredContent`` field.
Internal runtime/serialization failures still raise; callers must not treat them
as a valid agent refusal or a scored transition.
"""

from __future__ import annotations

import json
from threading import RLock
from typing import Any

import numpy as np

from ._action_validation import ActionValidationError
from .decision_parser import DecisionParseError, parse_decision, portfolio_weights
from .gym import SharpeArenaEnv
from .sharpearena_py import InvalidArgument, InvalidJson

try:  # pragma: no cover - exercised only when mcp is installed
    from mcp.server.fastmcp import FastMCP

    _HAS_MCP = True
except Exception:  # noqa: BLE001 - any import failure means "not available"
    FastMCP = None  # type: ignore[assignment, misc]
    _HAS_MCP = False


def _obs_payload(obs: dict, symbols: list[str]) -> dict:
    """A compact, JSON-safe observation (it re-enters the agent's context every step)."""
    return {
        "symbols": list(symbols),
        "closes": [float(x) for x in obs["closes"]],
        "positions": [float(x) for x in obs["positions"]],
        "cash": float(obs["cash"][0]),
    }


def _decision_to_weights(
    decision_json: str,
    symbols: list[str],
    observation: dict,
    *,
    max_abs_weight: float = 1.0,
) -> np.ndarray:
    """Strictly parse one canonical Decision over the episode's symbol axis."""

    return parse_decision(
        decision_json,
        symbols,
        current_weights=portfolio_weights(
            observation["positions"], observation["closes"], observation["cash"][0]
        ),
        max_abs_weight=max_abs_weight,
    )


def build_server(env_kwargs: dict | None = None) -> FastMCP:
    """Build a ``FastMCP('sharpearena')`` exposing ``reset`` / ``step`` / ``spec``.

    ``env_kwargs`` is forwarded to :class:`~sharpearena.gym.SharpeArenaEnv` (e.g.
    ``n_symbols``, ``n_days``, ``seed``, ``allow_short``). The episode env is created
    lazily and held for the lifetime of the server. Raises ``RuntimeError`` if ``mcp`` is
    not installed.

    One server owns one episode shared by all callers, not one episode per transport
    session. Tool calls are serialized; reset discards that shared episode's progress.
    Use a separate server per independent rollout. Transport access control is the
    deployment's responsibility and is not configured by this builder.
    """
    if not _HAS_MCP:
        raise RuntimeError(
            "mcp is not installed. Install the 'mcp' SDK to run the SharpeArena MCP "
            "server; the rest of the sharpearena package works without it."
        )

    server = FastMCP("sharpearena")
    lock = RLock()
    state: dict[str, Any] = {
        "env": None,
        "last_observation": None,
        "kwargs": dict(env_kwargs or {}),
        "done": False,
    }

    def _env() -> SharpeArenaEnv:
        if state["env"] is None:
            state["env"] = SharpeArenaEnv(**state["kwargs"])
        return state["env"]

    @server.tool()
    def reset() -> str:
        """(Re)create and reset the episode; return the initial observation JSON."""
        with lock:
            env = _env()
            obs, _info = env.reset()
            state["last_observation"] = obs
            state["done"] = False
            return json.dumps(_obs_payload(obs, env.symbols))

    @server.tool()
    def step(decision_json: str) -> str:
        """Advance one bar with a wire-contract ``Decision`` JSON.

        Returns ``{observation, reward, terminated, truncated, info}``; a malformed
        decision returns ``{"error": ...}`` in a JSON string, not an exception.
        """
        with lock:
            return _step(decision_json)

    def _step(decision_json: str) -> str:
        env = _env()
        if state["last_observation"] is None:
            return json.dumps(
                {
                    "error": "episode_not_reset",
                    "detail": "call reset before step",
                    "environment_advanced": False,
                }
            )
        if state["done"]:
            return json.dumps(
                {
                    "error": "episode_done",
                    "detail": "call reset before another step",
                    "environment_advanced": False,
                }
            )
        try:
            weights = _decision_to_weights(
                decision_json,
                env.symbols,
                state["last_observation"],
                max_abs_weight=float(np.max(env.action_space.high)),
            )
        except DecisionParseError as error:
            return json.dumps(
                {
                    "error": "invalid_decision",
                    "detail": str(error),
                    "environment_advanced": False,
                }
            )
        try:
            obs, reward, terminated, truncated, info = env.step(weights)
        except (ActionValidationError, InvalidArgument, InvalidJson) as error:
            # These typed input refusals happen before a kernel transition. Do not
            # turn an unrelated runtime/serialization failure into an agent refusal.
            return json.dumps(
                {
                    "error": "invalid_decision",
                    "detail": str(error),
                    "reason_code": getattr(error, "code", type(error).__name__),
                    "environment_advanced": False,
                }
            )
        state["last_observation"] = obs
        state["done"] = bool(terminated or truncated)
        return json.dumps(
            {
                "observation": _obs_payload(obs, env.symbols),
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "info": info,
            }
        )

    @server.tool()
    def spec() -> str:
        """The decision/observation wire schema plus the episode's symbol axis."""
        with lock:
            env = _env()
            return json.dumps(
                {
                    "decision_space": env.decision_space(),
                    "observation_space": env.observation_space_schema(),
                    "symbols": env.symbols,
                    "session_contract": {
                        "scope": "single_episode_per_server",
                        "shared_by_callers": True,
                        "tool_calls": "serialized",
                        "reset": "discards_shared_episode_progress",
                        "after_done": "reset_required",
                    },
                }
            )

    return server


def main() -> None:
    """Run the server over streamable HTTP (``python -m sharpearena.mcp_server``)."""
    build_server().run(transport="streamable-http")


if __name__ == "__main__":  # pragma: no cover
    main()
