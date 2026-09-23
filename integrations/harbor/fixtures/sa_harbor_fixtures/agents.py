"""Deterministic Harbor agents that run one fixture script and stop.

No model is called and no network is used, so a fixture result is a property of
the Harbor runtime and the task package rather than of a sampled policy. One
class per case because Harbor derives trial names from the agent name.
"""

from __future__ import annotations

from typing import override

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from sa_harbor_fixtures.scripts import CASE_SCRIPTS


class TamperAgent(BaseAgent):
    """Runs ``CASE_SCRIPTS[self.case]`` once inside the agent container."""

    case: str = "honest"

    def __init__(self, **kwargs) -> None:
        kwargs.pop("task_dir", None)
        kwargs.pop("trial_paths", None)
        kwargs.pop("agent_timeout_sec", None)
        super().__init__(**kwargs)

    @staticmethod
    @override
    def name() -> str:
        return "sa-tamper"

    @override
    def version(self) -> str:
        return "1.0.0"

    @override
    async def setup(self, environment: BaseEnvironment) -> None:
        return

    @override
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        script = CASE_SCRIPTS[self.case]
        await environment.exec(command=script)


class HonestAgent(TamperAgent):
    case = "honest"

    @staticmethod
    @override
    def name() -> str:
        return "sa-honest"


class ForgedRewardAgent(TamperAgent):
    case = "forged_reward"

    @staticmethod
    @override
    def name() -> str:
        return "sa-forged-reward"


class WritableGraderPathAgent(TamperAgent):
    case = "writable_grader_path"

    @staticmethod
    @override
    def name() -> str:
        return "sa-writable-grader-path"


class MissingFinalActionAgent(TamperAgent):
    case = "missing_final_action"

    @staticmethod
    @override
    def name() -> str:
        return "sa-missing-final-action"


class PathTraversalAgent(TamperAgent):
    case = "path_traversal"

    @staticmethod
    @override
    def name() -> str:
        return "sa-path-traversal"


class StaleOutputsAgent(TamperAgent):
    case = "stale_outputs"

    @staticmethod
    @override
    def name() -> str:
        return "sa-stale-outputs"


class CrossTaskStateAgent(TamperAgent):
    case = "cross_task_state"

    @staticmethod
    @override
    def name() -> str:
        return "sa-cross-task-state"


class PrivateEvaluatorReadAgent(TamperAgent):
    case = "private_evaluator_read"

    @staticmethod
    @override
    def name() -> str:
        return "sa-private-evaluator-read"
