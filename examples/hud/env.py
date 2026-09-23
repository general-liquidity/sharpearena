"""The HUD environment served by the INT-08 feasibility fixture.

One bounded SharpeArena episode expressed as a HUD task: the template yields the
prompt (mandate, permitted information, action schema, horizon, versioned task
identity), the agent drives the episode through the published `market` capability,
and the template grades from canonical engine evidence.

Serve it with any runtime that accepts a source path:

    LocalRuntime(Path(__file__))        # same process as the agent
    SubprocessRuntime(Path(__file__))   # separate OS process, same host
    DockerRuntime(image=...)            # container per rollout, see the report

`python -m hud.environment.server env.py --env sharpearena-feasibility` is the
serving entry point a container CMD would run.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any, Optional

from hud import Environment
from hud.capabilities import Capability

from episode import Episode, build_market_server, open_episode, prompt_text, serve_market
from grading import grade_episode

env = Environment("sharpearena-feasibility", version="0.31.0")

#: The episode the current rollout owns. One served environment runs one rollout at
#: a time (the runtime serializes acquisitions of the same instance), so a single
#: slot is the whole lifecycle; teardown clears it so no state crosses rollouts.
_current: dict[str, Optional[Episode]] = {"episode": None}
_service: dict[str, Any] = {"task": None}


@env.initialize
async def _start_market() -> None:
    service = await serve_market(build_market_server(lambda: _current["episode"]))
    _service["market"] = service
    env.add_capability(Capability(name="market", protocol="mcp/2025-11-25", url=service.url))


@env.shutdown
async def _stop_market() -> None:
    service = _service.pop("market", None)
    if service is not None:
        await service.stop()
    _current["episode"] = None


@env.template(
    id="bounded_episode",
    description="Trade one bounded SharpeArena episode under a sampled mandate.",
)
async def bounded_episode(scenario: str, horizon: int) -> AsyncGenerator[Any, Any]:
    episode = open_episode(scenario, horizon)
    _current["episode"] = episode
    try:
        answer = yield prompt_text(episode)
        yield grade_episode(episode, answer if isinstance(answer, str) else None)
    finally:
        _current["episode"] = None
