"""The C01-C07 structures that had no existing home in the package.

Most of the shared integration contract is already carried by types this package
ships: ``effective_config`` reads C01 back out of the environment that consumed it,
``_action_validation`` and ``spaces`` carry C02, ``gym``/``vector`` carry C03,
``rewards`` and ``kernel_score`` keep C04's two sides apart, ``eval_seeds`` and
``_seed_bands`` carry C05, and ``trace``/``bench_bridge``/``edge_manifest`` carry
C06's evidence path. ``docs/integrations/contracts.md`` maps each clause to the
module that owns it.

Three things had no owner, and they are what this module adds:

* C06 attempt accounting at the level a framework adapter reports it. ``bench_bridge``
  counts attempts for the local-model field and refuses a journal that disagrees with
  itself, but nothing lets an RL or harness adapter state expected/attempted/completed/
  refused/failed/retried for a rollout batch and have the arithmetic checked.
* C07's exactness domain. ``check_spec_hash`` pins the engine build and the goldens pin
  chosen values, but no type says which comparison is exact and which carries a
  tolerance, so a comparison could quietly be made under a rule nobody declared.
* C07's trust class, which is a property of where a checkpoint or agent came from and
  is not implied by a digest matching.

These are plain typed records. They do not re-implement scoring, refusal semantics or
the audit path; an adapter fills them from the evidence it already produces.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

#: The autoreset modes ``SharpeArenaVectorEnv`` accepts, in the package's own spelling.
#: Gymnasium's ``AutoresetMode`` enum names the same three (``NEXT_STEP``, ``SAME_STEP``,
#: ``DISABLED``); ``vector.py`` maps these strings onto it when the enum is importable.
SUPPORTED_AUTORESET_MODES = ("next_step", "same_step", "disabled")

#: Modes under which a finished lane's terminal observation is surfaced in ``infos``
#: as ``final_obs``/``final_info``. Under ``next_step`` the terminal observation is the
#: step return itself, and under ``disabled`` the lane stays at its terminal bar, so
#: neither carries a separate final-observation payload.
AUTORESET_MODES_WITH_FINAL_OBS = ("same_step",)


class ContractViolation(ValueError):
    """A value that cannot be reconciled with the C01-C07 contract it claims to satisfy."""


_COUNT_FIELDS = ("expected", "attempted", "completed", "refused", "failed", "retried")


@dataclass(frozen=True)
class AttemptCounts:
    """C06 attempt accounting for one run, batch or taskset.

    ``expected`` counts logical trials the caller planned. ``attempted`` counts
    attempts started, which is larger than the number of logical trials whenever an
    attempt was retried. Every started attempt reaches exactly one disposition, so
    ``completed + refused + failed == attempted``; a record that does not satisfy that
    is refused rather than reported, because the failure this guards against is a
    partial or duplicated run that reads as a complete one.

    ``refused`` and ``failed`` are distinct: a refusal is the engine or the adapter
    declining a decision it will not represent, and a failure is an attempt that broke.
    Collapsing either into a zero-valued completion is the C04/C06 error this type
    exists to prevent, so :meth:`is_complete` requires both to be zero.
    """

    expected: int
    attempted: int
    completed: int
    refused: int
    failed: int
    retried: int

    def __post_init__(self) -> None:
        for field in _COUNT_FIELDS:
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ContractViolation(f"{field} must be a non-negative integer, got {value!r}")
        dispositions = self.completed + self.refused + self.failed
        if dispositions != self.attempted:
            raise ContractViolation(
                f"attempted={self.attempted} but completed+refused+failed={dispositions}: "
                "every started attempt must reach exactly one disposition"
            )
        if self.retried > self.attempted:
            raise ContractViolation(
                f"retried={self.retried} exceeds attempted={self.attempted}: a retry is an attempt"
            )
        if self.completed > self.expected:
            raise ContractViolation(
                f"completed={self.completed} exceeds expected={self.expected}: more logical "
                "trials completed than were planned"
            )

    @property
    def is_complete(self) -> bool:
        """Every planned trial completed, with nothing refused or failed."""
        return self.completed == self.expected and self.refused == 0 and self.failed == 0

    def shortfall(self) -> str:
        """Why the run is not complete, or ``""`` when it is.

        A caller that prints this instead of a bare count keeps the reason in the
        evidence rather than leaving a reader to infer it from a missing row.
        """
        if self.is_complete:
            return ""
        parts = []
        if self.completed < self.expected:
            parts.append(f"{self.expected - self.completed} of {self.expected} trials not completed")
        if self.refused:
            parts.append(f"{self.refused} refused")
        if self.failed:
            parts.append(f"{self.failed} failed")
        return "; ".join(parts)

    def as_record(self) -> dict[str, Any]:
        """The evidence block, including the derived completeness so a reader of the
        artifact alone does not have to recompute it."""
        return {
            **{field: getattr(self, field) for field in _COUNT_FIELDS},
            "complete": self.is_complete,
            "shortfall": self.shortfall(),
        }

    @classmethod
    def from_record(cls, payload: Mapping[str, Any]) -> "AttemptCounts":
        """Parse a record. A missing count is refused, never defaulted to zero."""
        missing = [field for field in _COUNT_FIELDS if field not in payload]
        if missing:
            raise ContractViolation(f"attempt counts are missing {missing}; absent is not zero")
        return cls(**{field: payload[field] for field in _COUNT_FIELDS})


@dataclass(frozen=True)
class ExactnessDomain:
    """C07's declaration of which comparisons are exact and which carry a tolerance.

    The engine is deterministic for a fixed effective configuration and action
    sequence, so its own outputs compare exactly; a trained policy's parameters, a GPU
    reduction and a distributed optimiser state do not. A field that appears in neither
    ``exact`` nor ``tolerances`` has no declared comparison rule, and
    :meth:`tolerance_for` refuses it rather than picking one, so a comparison cannot be
    made under a rule nobody wrote down.
    """

    exact: tuple[str, ...]
    tolerances: Mapping[str, float]
    rationale: str

    def __post_init__(self) -> None:
        overlap = sorted(set(self.exact) & set(self.tolerances))
        if overlap:
            raise ContractViolation(f"{overlap} are declared both exact and tolerance-bounded")
        for field, value in self.tolerances.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0.0:
                raise ContractViolation(f"tolerance for {field} must be a positive number")
        if not self.rationale.strip():
            raise ContractViolation("an exactness domain must state why its fields are exact")

    def is_exact(self, field: str) -> bool:
        return field in self.exact

    def tolerance_for(self, field: str) -> float:
        if field in self.exact:
            raise ContractViolation(f"{field} is declared exact; it has no tolerance")
        try:
            return float(self.tolerances[field])
        except KeyError:
            raise ContractViolation(
                f"{field} has no declared comparison rule in this exactness domain"
            ) from None

    def as_record(self) -> dict[str, Any]:
        return {
            "exact": list(self.exact),
            "tolerances": {k: float(v) for k, v in self.tolerances.items()},
            "rationale": self.rationale,
        }


#: The domain every engine-versus-adapter comparison in this package runs under. The
#: engine produces these values from the same tape and the same decisions on both sides
#: of the comparison, so a difference of one ULP is a mapping bug, not float noise.
ENGINE_EXACTNESS = ExactnessDomain(
    exact=("reward", "nav", "observation", "terminated", "truncated", "outcome", "step_count"),
    tolerances={},
    rationale=(
        "one deterministic engine build replays one tape under one action sequence on "
        "both sides, so every compared value is produced by the same code path"
    ),
)


class TrustClass(str, Enum):
    """Where a checkpoint, policy or agent came from, for C07.

    A digest establishes that bytes did not change in transit. It does not make
    deserialising them safe, and it says nothing about who produced them, so the trust
    class is recorded separately from any digest.
    """

    #: Produced in this repository or by the operator, and loaded without a sandbox.
    TRUSTED_LOCAL = "trusted_local"
    #: Bytes match a recorded digest, but the producer is not the operator. Still
    #: requires containment before deserialisation.
    DIGEST_VERIFIED = "digest_verified"
    #: Third-party or entrant-supplied. Not loaded outside containment.
    UNTRUSTED = "untrusted"

    @property
    def safe_to_deserialize_locally(self) -> bool:
        return self is TrustClass.TRUSTED_LOCAL


__all__ = [
    "AUTORESET_MODES_WITH_FINAL_OBS",
    "ENGINE_EXACTNESS",
    "SUPPORTED_AUTORESET_MODES",
    "AttemptCounts",
    "ContractViolation",
    "ExactnessDomain",
    "TrustClass",
]
