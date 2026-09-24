"""A bounded Ray executor for SharpeArena episodes (INT-06).

This is a rollout and evaluation executor, not an adoption of Ray as a runtime. The
base package still imports, resets and steps with no Ray installed; ``ray`` is an
optional extra, the import here is guarded, and :func:`run_episodes` raises the named
:class:`RayUnavailable` rather than failing somewhere inside a scheduler.

The same executor runs with no Ray at all (``num_workers=0``), and that local route is
not a lesser fallback: it calls the identical :func:`execute_episode` on the identical
:class:`EpisodeSpec`, so the determinism test can compare a local run against a
three-worker run and expect the same bytes.

What crosses a process boundary
-------------------------------

Only plain data. An :class:`EpisodeSpec` holds the scenario parameters, the seed and
either a literal action tape or the dotted import path of a policy factory. The native
environment is built *inside* the worker from that specification and never leaves it.
This is the pattern Ray's own serialization page prescribes for objects it cannot
pickle, and it is structural here rather than advisory: :class:`EpisodeSpec` has no
field that could hold a live handle, an open file, or a closure over one. A caller who
wants a custom policy passes ``policy="my_pkg.my_module:make_policy"``, which the
worker imports by name, so a lambda cannot be smuggled across in the first place.

At-least-once semantics, handled rather than hoped about
--------------------------------------------------------

Ray retries a failed task (``max_retries`` defaults to 3) and may re-execute a task to
reconstruct a lost object. Ray does not promise exactly-once. Two things make that safe
here, and neither is a comment asking a future caller to be careful:

1. :func:`execute_episode` is a pure function of its spec. It opens no file, appends to
   no journal, consumes no budget and draws from no RNG the spec does not name, so
   running it twice produces the same value twice. There is no side effect for a
   duplicate to double-count.
2. Identity is assigned before scheduling, not after completion. Every result carries
   its ``episode_id``, and :func:`reduce_results` stores results into a dict keyed by
   that id. A second result for an id already present is compared against the first: if
   it is identical it is dropped as the duplicate it is, and if it *differs* the run is
   refused with :class:`ExecutorViolation`, because two different answers for one
   identity means the purity assumption above is false and no aggregate over them means
   anything.

Determinism does not come from Ray
----------------------------------

Ray's documentation makes no claim that a reduction over asynchronously completing
tasks is stable, so the executor does not rely on one. Results are placed by
``episode_id`` and the report is emitted in the order the caller supplied the specs.
Completion order is not observable in the output, and :func:`report_digest` over a run
is therefore identical across worker counts.

Nested parallelism
------------------

Ray's resources are logical: ``num_cpus=1`` on a task is a scheduling hint and does not
cap what the process actually uses. Each task here drives one scalar
:class:`~sharpearena.gym.SharpeArenaEnv`, which steps the engine on the calling thread
and starts no Rayon pool, so there is no nested pool to cap in this route. The native
vector engine does use Rayon, and it is deliberately not used here; a future executor
that batches lanes inside a worker would have to cap that itself, because Ray will not.
"""

from __future__ import annotations

import importlib
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np

from .canonical_json import canonical_sha256_v1
from .gym import SharpeArenaEnv
from .integrations.contracts import AttemptCounts, ContractViolation
from .integrations.parity import observation_digest

try:  # pragma: no cover - exercised only when ray is installed
    import ray

    _HAS_RAY = True
except Exception:  # noqa: BLE001 - any import failure means "not available"
    ray = None  # type: ignore[assignment]
    _HAS_RAY = False


class RayUnavailable(RuntimeError):
    """``ray`` was asked for and is not importable."""


class ExecutorViolation(ContractViolation):
    """A run whose results cannot be reconciled with the specs that were scheduled."""


def _require_ray() -> None:
    if not _HAS_RAY:
        raise RayUnavailable(
            "ray is not installed. Install 'sharpearena[ray]' to run episodes across "
            "Ray workers, or pass num_workers=0 to run the same episodes locally; the "
            "rest of the sharpearena package works without ray."
        )


# ---------------------------------------------------------------------------
# What crosses the boundary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EpisodeSpec:
    """One scheduled episode, as plain data.

    ``episode_id`` is the identity a retry reuses. It is assigned here, before
    scheduling, so a retried attempt lands on the same identity instead of reading as a
    fresh trial. Exactly one of ``actions`` and ``policy`` says what to do each bar:
    ``actions`` is a literal tape, and ``policy`` is the dotted ``module:attribute``
    path of a zero-argument factory returning a ``Callable[[dict], np.ndarray]``. The
    path is a string rather than a callable on purpose: a closure over a live handle
    cannot be expressed here, so it cannot fail at pickling time in a worker.
    """

    episode_id: str
    seed: int = 0
    n_symbols: int = 3
    n_days: int = 40
    distribution_mode: str = "calm"
    max_weight: float = 1.0
    allow_short: bool = True
    mode: str = "train"
    max_steps: int = 1_000
    actions: Optional[tuple[tuple[float, ...], ...]] = None
    policy: Optional[str] = None
    record_steps: bool = True

    def __post_init__(self) -> None:
        if not str(self.episode_id).strip():
            raise ExecutorViolation("episode_id must be a non-empty string")
        if (self.actions is None) == (self.policy is None):
            raise ExecutorViolation(
                f"{self.episode_id}: pass exactly one of actions (a literal tape) or "
                "policy (a 'module:attribute' path); passing neither leaves the episode "
                "undefined and passing both leaves it ambiguous"
            )
        if self.policy is not None and ":" not in self.policy:
            raise ExecutorViolation(
                f"{self.episode_id}: policy must be 'module:attribute', got {self.policy!r}"
            )
        if self.actions is not None:
            widths = {len(row) for row in self.actions}
            if widths and widths != {self.n_symbols}:
                raise ExecutorViolation(
                    f"{self.episode_id}: action tape is {sorted(widths)} wide for "
                    f"n_symbols={self.n_symbols}"
                )
        if int(self.max_steps) <= 0:
            raise ExecutorViolation(f"{self.episode_id}: max_steps must be positive")

    def as_record(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "seed": int(self.seed),
            "n_symbols": int(self.n_symbols),
            "n_days": int(self.n_days),
            "distribution_mode": self.distribution_mode,
            "max_weight": float(self.max_weight),
            "allow_short": bool(self.allow_short),
            "mode": self.mode,
            "max_steps": int(self.max_steps),
            "actions": None if self.actions is None else [list(row) for row in self.actions],
            "policy": self.policy,
        }


@dataclass(frozen=True)
class StepEvidence:
    """One bar, as the worker observed it.

    ``observation`` is the canonical digest of the decoded observation, produced by the
    same :func:`~sharpearena.integrations.parity.observation_digest` the engine-side
    reference path uses, so a remote episode can be compared with a direct engine replay
    field by field rather than only on its total.
    """

    reward: float
    nav: float
    terminated: bool
    truncated: bool
    observation: str

    def as_record(self) -> dict[str, Any]:
        return {
            "reward": self.reward,
            "nav": self.nav,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "observation": self.observation,
        }


@dataclass(frozen=True)
class EpisodeResult:
    """What one episode produced, keyed by the identity it was scheduled under.

    ``status`` is ``"completed"`` when the episode ran to a terminal flag or to
    ``max_steps``, ``"refused"`` when the engine or the declared action bounds declined
    a decision, and ``"failed"`` when the attempt broke. The three are kept apart
    because a refusal folded into a zero-reward completion is the accounting error the
    C06 contract exists to prevent.
    """

    episode_id: str
    status: str
    steps: int
    total_reward: float
    final_nav: float
    terminated: bool
    truncated: bool
    spec_hash: Optional[str]
    detail: str = ""
    evidence: tuple[StepEvidence, ...] = ()

    def as_record(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "status": self.status,
            "steps": self.steps,
            "total_reward": self.total_reward,
            "final_nav": self.final_nav,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "spec_hash": self.spec_hash,
            "detail": self.detail,
            "evidence": [step.as_record() for step in self.evidence],
        }


@dataclass(frozen=True)
class ExecutorReport:
    """The reduced run, in the caller's own spec order."""

    results: tuple[EpisodeResult, ...]
    counts: AttemptCounts
    unscheduled: tuple[str, ...] = ()
    num_workers: int = 0
    duplicates_dropped: int = 0

    def as_record(self) -> dict[str, Any]:
        """The evidence block. ``num_workers`` is deliberately absent: it is a property
        of how the run was executed, not of what it produced, and including it would
        make :func:`report_digest` differ across worker counts by construction."""
        return {
            "results": [result.as_record() for result in self.results],
            "counts": self.counts.as_record(),
            "unscheduled": list(self.unscheduled),
        }


def report_digest(report: ExecutorReport) -> str:
    """A canonical digest of what a run produced, independent of how it was executed."""
    return canonical_sha256_v1(report.as_record())


# ---------------------------------------------------------------------------
# The unit of work: a pure function of its spec
# ---------------------------------------------------------------------------


def resolve_policy(path: str) -> Callable[[dict], np.ndarray]:
    """Import ``module:attribute`` and call it to get a per-bar policy.

    Resolution happens in the process that will use it, so nothing about the policy
    crosses a boundary except its name.
    """
    module_name, _, attribute = path.partition(":")
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise ExecutorViolation(f"policy module {module_name!r} is not importable: {exc}") from exc
    try:
        factory = getattr(module, attribute)
    except AttributeError as exc:
        raise ExecutorViolation(
            f"policy module {module_name!r} has no attribute {attribute!r}"
        ) from exc
    policy = factory()
    if not callable(policy):
        raise ExecutorViolation(f"{path} returned {type(policy).__name__}, which is not callable")
    return policy


def _out_of_bounds(action: Sequence[float], spec: EpisodeSpec) -> bool:
    low = -spec.max_weight if spec.allow_short else 0.0
    values = [float(value) for value in action]
    if any(not np.isfinite(value) for value in values):
        return True
    return any(value < low or value > spec.max_weight for value in values)


def execute_episode(spec: EpisodeSpec) -> EpisodeResult:
    """Build the environment from ``spec``, replay it, and return plain data.

    Pure with respect to everything outside the returned value: no file is opened, no
    counter is incremented, and every random draw comes from the seed in the spec. That
    is what makes a Ray retry harmless rather than a double count.
    """
    from ._spec_hash import engine_spec_hash

    try:
        from . import sharpearena_py

        spec_hash = engine_spec_hash(sharpearena_py)
    except Exception as exc:  # pragma: no cover - exercised only without the binding
        return EpisodeResult(
            episode_id=spec.episode_id,
            status="failed",
            steps=0,
            total_reward=0.0,
            final_nav=0.0,
            terminated=False,
            truncated=False,
            spec_hash=None,
            detail=f"no native engine: {exc}",
        )

    try:
        env = SharpeArenaEnv(
            n_symbols=spec.n_symbols,
            n_days=spec.n_days,
            seed=spec.seed,
            max_weight=spec.max_weight,
            allow_short=spec.allow_short,
            distribution_mode=spec.distribution_mode,
            mode=spec.mode,
        )
        policy = None if spec.policy is None else resolve_policy(spec.policy)
        obs, _info = env.reset()
    except Exception as exc:
        return EpisodeResult(
            episode_id=spec.episode_id,
            status="failed",
            steps=0,
            total_reward=0.0,
            final_nav=0.0,
            terminated=False,
            truncated=False,
            spec_hash=spec_hash,
            detail=f"{type(exc).__name__}: {exc}",
        )

    tape = spec.actions
    horizon = spec.max_steps if tape is None else min(spec.max_steps, len(tape))

    evidence: list[StepEvidence] = []
    total = 0.0
    nav = 1.0
    terminated = False
    truncated = False
    for index in range(horizon):
        action = policy(obs) if tape is None else np.asarray(tape[index], dtype=np.float32)
        if _out_of_bounds(action, spec):
            return EpisodeResult(
                episode_id=spec.episode_id,
                status="refused",
                steps=len(evidence),
                total_reward=total,
                final_nav=nav,
                terminated=False,
                truncated=False,
                spec_hash=spec_hash,
                detail=f"step {index}: action outside the declared bounds",
                evidence=tuple(evidence),
            )
        try:
            obs, reward, terminated, truncated, info = env.step(np.asarray(action, dtype=np.float32))
        except Exception as exc:
            return EpisodeResult(
                episode_id=spec.episode_id,
                status="refused",
                steps=len(evidence),
                total_reward=total,
                final_nav=nav,
                terminated=False,
                truncated=False,
                spec_hash=spec_hash,
                detail=f"step {index}: {type(exc).__name__}: {exc}",
                evidence=tuple(evidence),
            )
        nav = float(info.get("nav", 1.0))
        total += float(reward)
        if spec.record_steps:
            evidence.append(
                StepEvidence(
                    reward=float(reward),
                    nav=nav,
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    observation=observation_digest(
                        obs["closes"], obs["positions"], float(obs["cash"][0])
                    ),
                )
            )
        if terminated or truncated:
            break

    return EpisodeResult(
        episode_id=spec.episode_id,
        status="completed",
        steps=len(evidence) if spec.record_steps else horizon,
        total_reward=total,
        final_nav=nav,
        terminated=bool(terminated),
        truncated=bool(truncated),
        spec_hash=spec_hash,
        evidence=tuple(evidence),
    )


# ---------------------------------------------------------------------------
# The reduction
# ---------------------------------------------------------------------------


def reduce_results(
    specs: Sequence[EpisodeSpec],
    results: Iterable[EpisodeResult],
    *,
    retried: int = 0,
    num_workers: int = 0,
) -> ExecutorReport:
    """Place results by identity and emit them in the caller's spec order.

    Completion order never reaches the output. A duplicate result for an identity
    already held is dropped when it is identical and refused when it is not, because
    two different answers under one identity falsify the purity the retry policy rests
    on and no aggregate over them would mean anything.
    """
    scheduled = [spec.episode_id for spec in specs]
    if len(set(scheduled)) != len(scheduled):
        duplicated = sorted({eid for eid in scheduled if scheduled.count(eid) > 1})
        raise ExecutorViolation(f"episode ids must be unique; {duplicated} appear more than once")

    known = set(scheduled)
    placed: dict[str, EpisodeResult] = {}
    duplicates = 0
    for result in results:
        if result.episode_id not in known:
            raise ExecutorViolation(
                f"result for {result.episode_id!r}, which was never scheduled"
            )
        previous = placed.get(result.episode_id)
        if previous is None:
            placed[result.episode_id] = result
            continue
        if previous == result:
            duplicates += 1
            continue
        raise ExecutorViolation(
            f"two different results for episode {result.episode_id!r}: an at-least-once "
            "retry produced a different answer, so the episode is not a pure function of "
            "its spec and the run cannot be reduced"
        )

    ordered = tuple(placed[eid] for eid in scheduled if eid in placed)
    unscheduled = tuple(eid for eid in scheduled if eid not in placed)
    attempted = len(ordered)
    counts = AttemptCounts(
        expected=len(scheduled),
        attempted=attempted,
        completed=sum(1 for r in ordered if r.status == "completed"),
        refused=sum(1 for r in ordered if r.status == "refused"),
        failed=sum(1 for r in ordered if r.status == "failed"),
        retried=int(retried),
    )
    return ExecutorReport(
        results=ordered,
        counts=counts,
        unscheduled=unscheduled,
        num_workers=int(num_workers),
        duplicates_dropped=duplicates,
    )


# ---------------------------------------------------------------------------
# The two routes
# ---------------------------------------------------------------------------


def run_episodes_local(specs: Sequence[EpisodeSpec]) -> ExecutorReport:
    """Run every spec in this process. The documented route when Ray is absent."""
    return reduce_results(specs, [execute_episode(spec) for spec in specs], num_workers=0)


def run_episodes(
    specs: Sequence[EpisodeSpec],
    *,
    num_workers: int = 0,
    max_in_flight: Optional[int] = None,
    max_retries: int = 3,
    deadline_s: Optional[float] = None,
    num_cpus_per_task: float = 1.0,
) -> ExecutorReport:
    """Run ``specs`` across ``num_workers`` Ray tasks and reduce them canonically.

    ``num_workers=0`` takes the local route and never imports Ray. Otherwise Ray must
    already be initialised or initialisable; this function does not start a cluster, and
    it does not touch dashboard, tracking or credential configuration, which are
    operational settings rather than executor inputs.

    ``max_in_flight`` bounds pending tasks, defaulting to twice the worker count, so a
    long spec list does not put every episode's result in the object store at once.
    ``deadline_s`` stops scheduling and cancels what is pending when the budget is
    spent; episodes that had not started are reported in ``unscheduled`` rather than
    silently dropped, and the partial evidence already collected is kept.
    """
    if num_workers <= 0:
        return run_episodes_local(specs)
    _require_ray()
    if not specs:
        return reduce_results(specs, [], num_workers=num_workers)

    window = int(max_in_flight) if max_in_flight else max(1, 2 * int(num_workers))
    remote = ray.remote(num_cpus=num_cpus_per_task, max_retries=int(max_retries))(execute_episode)

    started = time.monotonic()
    pending: dict[Any, EpisodeSpec] = {}
    collected: list[EpisodeResult] = []
    queue = list(specs)

    def _expired() -> bool:
        return deadline_s is not None and (time.monotonic() - started) >= float(deadline_s)

    index = 0
    while (index < len(queue) or pending) and not _expired():
        while index < len(queue) and len(pending) < window and not _expired():
            spec = queue[index]
            pending[remote.remote(spec)] = spec
            index += 1
        if not pending:
            break
        ready, _rest = ray.wait(list(pending), num_returns=1, timeout=1.0)
        for ref in ready:
            spec = pending.pop(ref)
            try:
                collected.append(ray.get(ref))
            except Exception as exc:  # noqa: BLE001 - a lost worker is a failed attempt
                collected.append(
                    EpisodeResult(
                        episode_id=spec.episode_id,
                        status="failed",
                        steps=0,
                        total_reward=0.0,
                        final_nav=0.0,
                        terminated=False,
                        truncated=False,
                        spec_hash=None,
                        detail=f"{type(exc).__name__}: {exc}",
                    )
                )

    for ref in pending:
        # force=False so a worker mid-step is interrupted cooperatively; Ray does not
        # retry a cancelled task, which is what "stop cleanly" has to mean here.
        ray.cancel(ref, force=False)

    return reduce_results(specs, collected, num_workers=num_workers)


__all__ = [
    "EpisodeResult",
    "EpisodeSpec",
    "ExecutorReport",
    "ExecutorViolation",
    "RayUnavailable",
    "StepEvidence",
    "execute_episode",
    "reduce_results",
    "report_digest",
    "resolve_policy",
    "run_episodes",
    "run_episodes_local",
]
