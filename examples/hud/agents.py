"""Deterministic agent doubles for the INT-08 fixture. No model is called.

Each double drives the published `market` capability with a fixed policy so the
fixture exercises one named failure mode per run. They are doubles, not baselines:
their decisions are the cheapest thing that reaches the engine, and no score they
produce is evidence about trading quality.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from hud.agents.base import Agent
from hud.types import Step


def equal_weight_decision(symbols: list[str]) -> str:
    """One canonical Decision holding an equal long weight across every symbol."""
    weight = round(1.0 / max(1, len(symbols)), 6)
    return json.dumps(
        {
            "orders": [
                {"symbol": symbol, "action": "buy", "target_weight": weight}
                for symbol in symbols
            ],
            "reasoning": "fixture double: equal weight",
        }
    )


class _MarketAgent(Agent):
    """Shared plumbing: open the capability, loop, record steps, answer."""

    async def __call__(self, run: Any) -> None:
        client = await run.client.open("market")
        report = await self.drive(run, client)
        run.trace.content = json.dumps(report)

    async def _call(self, run: Any, client: Any, tool: str, **arguments: Any) -> dict:
        result = await client.call_tool(tool, arguments)
        text = "".join(
            block.text for block in result.content if getattr(block, "type", None) == "text"
        )
        payload = json.loads(text)
        run.record(
            Step(
                source="agent",
                extra={"tool": tool, "arguments": arguments, "result": payload},
            )
        )
        return payload

    async def drive(self, run: Any, client: Any) -> dict:
        raise NotImplementedError


class CompletingAgent(_MarketAgent):
    """Submits one valid decision per bar until the horizon is exhausted."""

    async def drive(self, run: Any, client: Any) -> dict:
        spec = await self._call(run, client, "spec")
        symbols = spec["symbols"]
        decision = equal_weight_decision(symbols)
        advanced = 0
        for bar in range(spec["horizon_bars"]):
            result = await self._call(
                run, client, "submit_decision", decision_json=decision, bar_index=bar
            )
            if not result.get("environment_advanced"):
                break
            advanced += 1
        return {"double": "completing", "bars_submitted": advanced}


class MalformedThenValidAgent(_MarketAgent):
    """Sends an unparseable decision first, then completes the horizon.

    The refused attempt must not consume a bar, and the run must still be able to
    finish: a malformed decision is a refused attempt, not a destroyed episode.
    """

    async def drive(self, run: Any, client: Any) -> dict:
        spec = await self._call(run, client, "spec")
        symbols = spec["symbols"]
        refused = await self._call(
            run, client, "submit_decision", decision_json="{not json", bar_index=0
        )
        decision = equal_weight_decision(symbols)
        advanced = 0
        for bar in range(spec["horizon_bars"]):
            result = await self._call(
                run, client, "submit_decision", decision_json=decision, bar_index=bar
            )
            if not result.get("environment_advanced"):
                break
            advanced += 1
        return {
            "double": "malformed_then_valid",
            "first_error": refused.get("error"),
            "bars_submitted": advanced,
        }


class DuplicateSubmissionAgent(_MarketAgent):
    """Replays bar 0 after it was already consumed."""

    async def drive(self, run: Any, client: Any) -> dict:
        spec = await self._call(run, client, "spec")
        decision = equal_weight_decision(spec["symbols"])
        await self._call(run, client, "submit_decision", decision_json=decision, bar_index=0)
        replay = await self._call(
            run, client, "submit_decision", decision_json=decision, bar_index=0
        )
        advanced = 1
        for bar in range(1, spec["horizon_bars"]):
            result = await self._call(
                run, client, "submit_decision", decision_json=decision, bar_index=bar
            )
            if not result.get("environment_advanced"):
                break
            advanced += 1
        return {
            "double": "duplicate_submission",
            "replay_error": replay.get("error"),
            "bars_submitted": advanced,
        }


class EarlyExitAgent(_MarketAgent):
    """Answers after `stop_after` bars, leaving the horizon unfinished."""

    def __init__(self, stop_after: int = 2) -> None:
        self.stop_after = stop_after

    async def drive(self, run: Any, client: Any) -> dict:
        spec = await self._call(run, client, "spec")
        decision = equal_weight_decision(spec["symbols"])
        for bar in range(self.stop_after):
            await self._call(
                run, client, "submit_decision", decision_json=decision, bar_index=bar
            )
        return {"double": "early_exit", "bars_submitted": self.stop_after}


class StallingAgent(_MarketAgent):
    """Submits one bar and then blocks, so the rollout timeout is the terminator."""

    def __init__(self, stall_seconds: float = 30.0) -> None:
        self.stall_seconds = stall_seconds

    async def drive(self, run: Any, client: Any) -> dict:
        spec = await self._call(run, client, "spec")
        decision = equal_weight_decision(spec["symbols"])
        await self._call(run, client, "submit_decision", decision_json=decision, bar_index=0)
        await asyncio.sleep(self.stall_seconds)
        return {"double": "stalling", "bars_submitted": 1}
