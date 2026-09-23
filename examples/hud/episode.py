"""Private episode state and the decision-time MCP service for the HUD fixture.

This is the trusted worker side of the INT-08 feasibility check. The engine, the
scenario seed, the mandate and every future bar live here; the agent reaches this
module only through the three MCP tools published below. Nothing in this file is a
package API: it is an example fixture, and the adapter itself waits on the INT-01
contracts.

The refusal shapes (`invalid_decision`, `out_of_order`, `duplicate_submission`,
`episode_closed`) are returned as structured content rather than raised, so a
harness that converts a tool exception into an aborted rollout still records the
attempt. Every refusal carries `environment_advanced`, which is the field a grader
reads to tell a refused attempt from a consumed bar.
"""

from __future__ import annotations

import asyncio
import json
import socket
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from sharpearena.decision_parser import DecisionParseError, parse_decision, portfolio_weights
from sharpearena.effective_config import env_effective_config, scenario_fingerprint
from sharpearena.gym import SharpeArenaEnv
from sharpearena.mandate import mandate_text, sample_mandate

#: Opaque scenario labels the task row carries, mapped to their private seeds here.
#: The label is what an entrant sees; the seed never leaves this process. A public
#: label keeps the task row reproducible without publishing the generator input.
SCENARIO_SEEDS: dict[str, dict[str, Any]] = {
    "calm-a": {"seed": 4101, "n_symbols": 3, "n_days": 64, "distribution_mode": "calm"},
    "calm-b": {"seed": 4102, "n_symbols": 3, "n_days": 64, "distribution_mode": "calm"},
    "fat-a": {"seed": 4201, "n_symbols": 3, "n_days": 64, "distribution_mode": "hard"},
}

#: Versioned task identity. Bump when the mandate, permitted information, action
#: schema, horizon rule or grading inputs change; a stored result names this string.
TASK_IDENTITY = "sharpearena/bounded_episode@1"


@dataclass
class Attempt:
    """One submitted decision, accepted or refused, in submission order."""

    bar_index: int
    accepted: bool
    error: Optional[str] = None


@dataclass
class Episode:
    """One bounded episode: private engine state plus the evidence it produced."""

    scenario: str
    horizon: int
    env: SharpeArenaEnv
    mandate: Any
    observation: dict[str, np.ndarray]
    bars_advanced: int = 0
    closed: bool = False
    returns: list[float] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    attempts: list[Attempt] = field(default_factory=list)

    @property
    def bars_remaining(self) -> int:
        return max(0, self.horizon - self.bars_advanced)

    def observation_payload(self) -> dict[str, Any]:
        """The permitted information at the current bar. No future bar, no seed."""
        return {
            "bar_index": self.bars_advanced,
            "bars_remaining": self.bars_remaining,
            "symbols": list(self.env.symbols),
            "closes": [float(x) for x in self.observation["closes"]],
            "positions": [float(x) for x in self.observation["positions"]],
            "cash": float(self.observation["cash"][0]),
        }

    def submit(self, decision_json: str, bar_index: int) -> dict[str, Any]:
        """Advance exactly one bar, or refuse without advancing."""
        if self.closed or self.bars_remaining == 0:
            return self._refuse(bar_index, "episode_closed", "the horizon is exhausted")
        if bar_index != self.bars_advanced:
            reason = "duplicate_submission" if bar_index < self.bars_advanced else "out_of_order"
            return self._refuse(
                bar_index,
                reason,
                f"expected bar_index {self.bars_advanced}, received {bar_index}",
            )
        try:
            weights = parse_decision(
                decision_json,
                self.env.symbols,
                current_weights=portfolio_weights(
                    self.observation["positions"],
                    self.observation["closes"],
                    self.observation["cash"][0],
                ),
            )
        except DecisionParseError as error:
            return self._refuse(bar_index, "invalid_decision", str(error))

        obs, reward, terminated, truncated, info = self.env.step(weights)
        self.observation = obs
        self.bars_advanced += 1
        self.returns.append(float(reward))
        self.events.extend(dict(event) for event in info.get("events", []) or [])
        self.attempts.append(Attempt(bar_index=bar_index, accepted=True))
        if terminated or truncated:
            self.closed = True
        return {
            "environment_advanced": True,
            "observation": self.observation_payload(),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }

    def _refuse(self, bar_index: int, error: str, detail: str) -> dict[str, Any]:
        self.attempts.append(Attempt(bar_index=bar_index, accepted=False, error=error))
        return {
            "environment_advanced": False,
            "error": error,
            "detail": detail,
            "bar_index_expected": self.bars_advanced,
        }

    def attempt_counts(self) -> dict[str, int]:
        """Expected, attempted, accepted and refused counts, kept distinct.

        A grader that only sees `len(returns)` cannot tell a clean run from one
        that burned attempts, so the refusal classes are counted separately.
        """
        counts = {
            "expected_bars": self.horizon,
            "attempted": len(self.attempts),
            "accepted": sum(1 for a in self.attempts if a.accepted),
            "refused": sum(1 for a in self.attempts if not a.accepted),
        }
        for reason in ("invalid_decision", "duplicate_submission", "out_of_order", "episode_closed"):
            counts[reason] = sum(1 for a in self.attempts if a.error == reason)
        return counts


def open_episode(scenario: str, horizon: int) -> Episode:
    """Build one bounded episode from an opaque scenario label."""
    if scenario not in SCENARIO_SEEDS:
        raise KeyError(f"unknown scenario label: {scenario!r}")
    config = dict(SCENARIO_SEEDS[scenario])
    if horizon < 1 or horizon > config["n_days"] - 2:
        raise ValueError(f"horizon {horizon} outside the scenario's usable range")
    env = SharpeArenaEnv(mode="eval", **config)
    observation, _info = env.reset()
    mandate = sample_mandate(config["seed"], n_symbols=config["n_symbols"])
    return Episode(
        scenario=scenario,
        horizon=horizon,
        env=env,
        mandate=mandate,
        observation=observation,
    )


def episode_identity(episode: Episode) -> dict[str, Any]:
    """The C01 identity block: what was requested and what the engine read back."""
    config = SCENARIO_SEEDS[episode.scenario]
    return {
        "task_identity": TASK_IDENTITY,
        "scenario": episode.scenario,
        "horizon": episode.horizon,
        # The fingerprint binds the generator inputs without publishing the seed.
        "scenario_fingerprint": scenario_fingerprint(
            seed=config["seed"],
            n_symbols=config["n_symbols"],
            n_days=config["n_days"],
            distribution_mode=config["distribution_mode"],
        ),
        "effective_config": env_effective_config(episode.env),
    }


def prompt_text(episode: Episode) -> str:
    """Mandate, permitted information, action schema and horizon, as one prompt."""
    identity = episode_identity(episode)
    return json.dumps(
        {
            "task_identity": identity["task_identity"],
            "scenario": episode.scenario,
            "scenario_fingerprint": identity["scenario_fingerprint"],
            "mandate": mandate_text(episode.mandate),
            "horizon_bars": episode.horizon,
            "decision_space": episode.env.decision_space(),
            "permitted_information": (
                "Call observe() for the current bar. No future bar, scenario seed or "
                "engine state is available through any tool."
            ),
            "capability": "market",
            "tools": ["spec", "observe", "submit_decision"],
        },
        indent=2,
        sort_keys=True,
    )


# ─── decision-time MCP service ────────────────────────────────────────────────


def build_market_server(get_episode) -> Any:
    """A FastMCP server publishing the three decision-time tools.

    `get_episode` returns the episode the current rollout owns, or None. Holding the
    episode behind a callable rather than binding it at build time is what lets one
    served environment run successive rollouts without carrying state between them.
    """
    from mcp.server.fastmcp import FastMCP

    server = FastMCP("sharpearena-market")

    def _closed() -> str:
        return json.dumps({"error": "episode_closed", "detail": "no episode in progress"})

    @server.tool()
    def spec() -> str:
        """The action schema, symbol axis and remaining horizon."""
        episode = get_episode()
        if episode is None:
            return _closed()
        return json.dumps(
            {
                "task_identity": TASK_IDENTITY,
                "decision_space": episode.env.decision_space(),
                "symbols": list(episode.env.symbols),
                "horizon_bars": episode.horizon,
                "bars_remaining": episode.bars_remaining,
            }
        )

    @server.tool()
    def observe() -> str:
        """The current bar's permitted information."""
        episode = get_episode()
        if episode is None:
            return _closed()
        return json.dumps(episode.observation_payload())

    @server.tool()
    def submit_decision(decision_json: str, bar_index: int) -> str:
        """Advance exactly one bar with a canonical Decision, or refuse."""
        episode = get_episode()
        if episode is None:
            return _closed()
        return json.dumps(episode.submit(decision_json, bar_index))

    return server


class MarketService:
    """A running decision-time service: its URL, and a graceful shutdown.

    Loopback only. Anything else on this host can reach the port, which is the
    isolation limit recorded in the feasibility report rather than a containment
    claim.
    """

    def __init__(self, url: str, server: Any, task: "asyncio.Task") -> None:
        self.url = url
        self._server = server
        self._task = task

    async def stop(self) -> None:
        # Graceful exit rather than cancellation: uvicorn's lifespan receive()
        # raises CancelledError through Starlette and prints a traceback that
        # looks like a fixture failure.
        self._server.should_exit = True
        await asyncio.gather(self._task, return_exceptions=True)


async def serve_market(server: Any) -> MarketService:
    """Serve `server` over streamable HTTP on an ephemeral loopback port."""
    import uvicorn

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    config = uvicorn.Config(
        server.streamable_http_app(),
        log_level="warning",
        lifespan="on",
    )
    http = uvicorn.Server(config)
    task = asyncio.create_task(http.serve(sockets=[sock]))
    while not http.started:
        if task.done():
            await task
            raise RuntimeError("market service exited before serving")
        await asyncio.sleep(0.01)
    return MarketService(f"http://127.0.0.1:{port}/mcp", http, task)
