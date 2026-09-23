"""Shared contract structures and the engine-versus-adapter parity checker.

Nothing here is imported by ``sharpearena/__init__.py``: the base package must still
import, reset and step with no learner, harness or framework installed, so an adapter
pulls these in by name (``from sharpearena.integrations import parity``). The modules
themselves depend only on the base package and NumPy.
"""

from .contracts import (
    AUTORESET_MODES_WITH_FINAL_OBS,
    ENGINE_EXACTNESS,
    SUPPORTED_AUTORESET_MODES,
    AttemptCounts,
    ContractViolation,
    ExactnessDomain,
    TrustClass,
)
from .parity import (
    CORE_FIXTURES,
    ParityFixture,
    ParityMismatch,
    ParityRecord,
    ParityReport,
    ParityUnavailable,
    StepRecord,
    adapter_rollout,
    check_adapter_parity,
    compare,
    decision_json,
    deterministic_actions,
    engine_rollout,
    observation_digest,
)

__all__ = [
    "AUTORESET_MODES_WITH_FINAL_OBS",
    "CORE_FIXTURES",
    "ENGINE_EXACTNESS",
    "SUPPORTED_AUTORESET_MODES",
    "AttemptCounts",
    "ContractViolation",
    "ExactnessDomain",
    "ParityFixture",
    "ParityMismatch",
    "ParityRecord",
    "ParityReport",
    "ParityUnavailable",
    "StepRecord",
    "TrustClass",
    "adapter_rollout",
    "check_adapter_parity",
    "compare",
    "decision_json",
    "deterministic_actions",
    "engine_rollout",
    "observation_digest",
]
