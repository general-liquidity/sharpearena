"""Engine-versus-adapter parity: one fixture, two paths, one comparison.

Every framework adapter added under this package claims the same thing: for one
effective configuration and one action sequence, what the framework sees is what the
engine produced. This module is how that claim is checked, and it is deliberately not
built out of the adapter's own helpers.

The reference path drives the native ``TradingEnv`` directly and encodes decisions with
:func:`decision_json`, a restatement of the wire contract written here rather than
imported from ``sharpearena.gym``. Sharing the encoder would make the comparison
circular: a wrapper that mislabels a short as a buy, drops the last symbol or swaps
termination for truncation would produce matching records on both sides. The
restatement is the same technique ``effective_config`` uses for the seed split, and for
the same reason.

Three ways a parity check could pass without having compared anything are refused
rather than reported:

* the compiled extension is missing, stale or reports no ``spec_hash``, so the two
  paths are not the same engine (:class:`ParityUnavailable`);
* the adapter could not be imported or constructed (:class:`ParityUnavailable`, never a
  clean report);
* no step was compared, because a fixture refused on its first action or asked for zero
  steps (:class:`ParityUnavailable`).

A refusal and an incomplete episode are first-class fixture outcomes, not errors: both
sides must refuse the same action with the same typed error, and both must stop at the
same step when a fixture ends before the horizon.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

from .._seed_bands import EVAL_SEED_BASE
from .._spec_hash import EXPECTED_SPEC_HASH, engine_spec_hash
from ..canonical_json import canonical_sha256_v1
from .contracts import ENGINE_EXACTNESS, AttemptCounts, ExactnessDomain

_HOLD_EPS = 0.0


class ParityUnavailable(RuntimeError):
    """The comparison could not be made, so nothing may be reported as parity."""


class ParityMismatch(AssertionError):
    """The adapter did not reproduce the engine's outputs for this fixture."""


def decision_json(symbols: Sequence[str], weights: Sequence[float]) -> str:
    """The wire-contract ``Decision`` for one bar, restated from the schema.

    ``action`` is the descriptive label the engine scores for calibration and
    ``target_weight`` is the instruction; a label that disagrees with the sign of its
    weight is exactly the mapping bug this restatement exists to catch.
    """
    if len(symbols) != len(weights):
        raise ParityUnavailable(
            f"{len(weights)} weights for {len(symbols)} symbols: the fixture does not fit the tape"
        )
    orders = []
    for symbol, weight in zip(symbols, weights):
        weight = float(weight)
        if weight > _HOLD_EPS:
            label = "buy"
        elif weight < _HOLD_EPS:
            label = "sell"
        else:
            label = "hold"
        orders.append(
            {
                "symbol": symbol,
                "action": label,
                "target_weight": weight,
                "confidence": 0.5,
            }
        )
    return json.dumps({"orders": orders, "reasoning": "integrations.parity"})


def observation_digest(closes: Sequence[float], positions: Sequence[float], cash: float) -> str:
    """A canonical digest of the decoded observation, stable across both paths.

    Uses ``sharpebench/canonical-json/v1`` so the digest is the same bytes whichever
    language produced the numbers, and so a float that differs by one ULP changes it.
    """
    return canonical_sha256_v1(
        {
            "cash": float(cash),
            "closes": [float(c) for c in closes],
            "positions": [float(p) for p in positions],
        }
    )


def _decode_wire_observation(obs_json: str, symbols: Sequence[str]) -> tuple[list, list, float]:
    """Decode the engine's ``MarketObservation`` the way the schema documents it."""
    obs = json.loads(obs_json)
    by_symbol = {entry["symbol"]: entry for entry in obs["symbols"]}
    closes = [by_symbol[symbol]["close_history"][-1] for symbol in symbols]
    held = {entry["symbol"]: entry["shares"] for entry in obs.get("portfolio", [])}
    positions = [held.get(symbol, 0.0) for symbol in symbols]
    return closes, positions, float(obs["cash"])


@dataclass(frozen=True)
class StepRecord:
    """One compared step. ``outcome`` is ``"ok"`` or ``"refused:<code>"``."""

    step: int
    action: tuple[float, ...]
    outcome: str
    reward: Optional[float]
    terminated: Optional[bool]
    truncated: Optional[bool]
    nav: Optional[float]
    observation: Optional[str]

    def as_record(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "action": list(self.action),
            "outcome": self.outcome,
            "reward": self.reward,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "nav": self.nav,
            "observation": self.observation,
        }


@dataclass(frozen=True)
class ParityFixture:
    """One deterministic scenario plus the exact action sequence both paths replay.

    The actions are stored literally rather than regenerated per path, so the two
    rollouts cannot diverge because a generator was called in a different order.
    """

    name: str
    actions: tuple[tuple[float, ...], ...]
    seed: int = 0
    n_symbols: int = 3
    n_days: int = 40
    distribution_mode: str = "calm"
    max_weight: float = 1.0
    allow_short: bool = True
    mode: str = "train"
    expect: str = "complete"

    def __post_init__(self) -> None:
        if self.expect not in ("complete", "incomplete", "refused"):
            raise ParityUnavailable(
                f"{self.name}: expect must be complete, incomplete or refused, got {self.expect!r}"
            )
        if not self.actions:
            raise ParityUnavailable(f"{self.name}: a fixture with no actions compares nothing")
        widths = {len(action) for action in self.actions}
        if widths != {self.n_symbols}:
            raise ParityUnavailable(
                f"{self.name}: actions are {sorted(widths)} wide for n_symbols={self.n_symbols}"
            )

    def refusal_reason(self, action: Sequence[float]) -> Optional[str]:
        """Why this fixture's declared action bounds reject ``action``, or ``None``.

        The bound is part of the effective configuration the adapter advertises (C02),
        not something the engine enforces, so the reference path applies it here rather
        than discovering it from the adapter. An adapter that clips, projects or rounds
        instead of refusing produces ``accepted:<reason>`` and fails the comparison.
        """
        low = -self.max_weight if self.allow_short else 0.0
        if any(not math.isfinite(float(value)) for value in action):
            return "not_finite"
        if any(float(value) < low or float(value) > self.max_weight for value in action):
            return "out_of_bounds"
        return None

    def scenario_kwargs(self) -> dict[str, Any]:
        return {
            "n_symbols": self.n_symbols,
            "n_days": self.n_days,
            "distribution_mode": self.distribution_mode,
        }


@dataclass(frozen=True)
class ParityRecord:
    """What one path did with one fixture."""

    fixture: str
    source: str
    spec_hash: Optional[str]
    effective_config: dict
    steps: tuple[StepRecord, ...]
    counts: AttemptCounts
    terminal: str

    def as_record(self) -> dict[str, Any]:
        return {
            "fixture": self.fixture,
            "source": self.source,
            "spec_hash": self.spec_hash,
            "effective_config": self.effective_config,
            "counts": self.counts.as_record(),
            "terminal": self.terminal,
            "steps": [step.as_record() for step in self.steps],
        }


@dataclass(frozen=True)
class ParityReport:
    """The comparison. ``ok`` is only true when steps were actually compared."""

    fixture: str
    reference: str
    candidate: str
    steps_compared: int
    exactness: ExactnessDomain
    mismatches: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.mismatches and self.steps_compared > 0

    def require_parity(self) -> "ParityReport":
        if not self.ok:
            raise ParityMismatch(
                f"{self.candidate} does not reproduce {self.reference} on fixture "
                f"{self.fixture} ({self.steps_compared} steps compared): "
                + "; ".join(self.mismatches)
            )
        return self

    def as_record(self) -> dict[str, Any]:
        return {
            "fixture": self.fixture,
            "reference": self.reference,
            "candidate": self.candidate,
            "steps_compared": self.steps_compared,
            "ok": self.ok,
            "exactness": self.exactness.as_record(),
            "mismatches": list(self.mismatches),
        }


def _refusal_code(exc: BaseException) -> str:
    """The engine's ``[CODE] `` prefix when there is one, else the exception class."""
    message = str(exc)
    if message.startswith("[") and "] " in message:
        return message[1 : message.index("] ")]
    return type(exc).__name__


def resolved_seeds(fixture: ParityFixture) -> tuple[int, int]:
    """The (scenario, execution) seeds a fixture's user seed lands on.

    The band base is read from ``_seed_bands`` (its one definition); the split itself is
    restated here rather than imported from ``sharpearena.gym``, for the same reason
    ``effective_config`` restates it: a reference path that asks the adapter how it split
    the seed cannot detect the adapter splitting it wrongly.
    """
    import numpy as np

    offset = EVAL_SEED_BASE if fixture.mode == "eval" else 0
    state = np.random.SeedSequence(int(fixture.seed) + offset).generate_state(2)
    return int(state[0]), int(state[1])


def _refused_step(index: int, action: Sequence[float], reason: str) -> StepRecord:
    return StepRecord(
        step=index,
        action=tuple(float(value) for value in action),
        outcome=f"refused:{reason}",
        reward=None,
        terminated=None,
        truncated=None,
        nav=None,
        observation=None,
    )


def _counts(steps: Sequence[StepRecord], expected_steps: int) -> AttemptCounts:
    refused = sum(1 for step in steps if step.outcome.startswith("refused:"))
    failed = sum(1 for step in steps if step.outcome.startswith("accepted:"))
    return AttemptCounts(
        expected=expected_steps,
        attempted=len(steps),
        completed=len(steps) - refused - failed,
        refused=refused,
        failed=failed,
        retried=0,
    )


def _terminal_label(steps: Sequence[StepRecord]) -> str:
    if not steps:
        return "empty"
    last = steps[-1]
    if last.outcome.startswith("accepted:"):
        return "contract_violation"
    if last.outcome.startswith("refused:"):
        return "refused"
    if last.terminated:
        return "terminated"
    if last.truncated:
        return "truncated"
    return "incomplete"


def engine_rollout(fixture: ParityFixture) -> ParityRecord:
    """Replay the fixture against the native engine, with no adapter in the path."""
    try:
        from ..sharpearena_py import TradingEnv
        from .. import sharpearena_py
    except Exception as exc:  # pragma: no cover - exercised only without the binding
        raise ParityUnavailable(f"the native sharpearena binding is not importable: {exc}") from exc

    reported = engine_spec_hash(sharpearena_py)
    if reported != EXPECTED_SPEC_HASH:
        raise ParityUnavailable(
            f"engine spec hash {reported!r} is not the pinned {EXPECTED_SPEC_HASH!r}; "
            "the two paths would not be the same engine"
        )

    scenario_seed, exec_seed = resolved_seeds(fixture)
    env = TradingEnv(seed=scenario_seed, exec_seed=exec_seed, **fixture.scenario_kwargs())
    obs_json = env.reset()
    symbols = [entry["symbol"] for entry in json.loads(obs_json)["symbols"]]

    steps: list[StepRecord] = []
    for index, action in enumerate(fixture.actions):
        reason = fixture.refusal_reason(action)
        if reason is not None:
            steps.append(_refused_step(index, action, reason))
            break
        try:
            obs_json, reward, done, info_json = env.step(decision_json(symbols, action))
        except Exception as exc:
            steps.append(_refused_step(index, action, f"engine:{_refusal_code(exc)}"))
            break
        info = json.loads(info_json)
        nav = float(info.get("nav", 1.0))
        closes, positions, cash = _decode_wire_observation(obs_json, symbols)
        steps.append(
            StepRecord(
                step=index,
                action=action,
                outcome="ok",
                reward=float(reward),
                # C03 as this package defines it: running out of bars is truncation,
                # and bankruptcy is the absorbing terminal state.
                terminated=nav <= 0.0,
                truncated=bool(done),
                nav=nav,
                observation=observation_digest(closes, positions, cash),
            )
        )
        if done or nav <= 0.0:
            break

    return ParityRecord(
        fixture=fixture.name,
        source="engine",
        spec_hash=reported,
        effective_config=json.loads(env.effective_config),
        steps=tuple(steps),
        counts=_counts(steps, len(fixture.actions)),
        terminal=_terminal_label(steps),
    )


def _default_adapter(fixture: ParityFixture):
    from ..gym import SharpeArenaEnv

    return SharpeArenaEnv(
        seed=fixture.seed,
        max_weight=fixture.max_weight,
        allow_short=fixture.allow_short,
        mode=fixture.mode,
        **fixture.scenario_kwargs(),
    )


def adapter_rollout(
    fixture: ParityFixture,
    make_env: Optional[Callable[[ParityFixture], Any]] = None,
    *,
    source: str = "sharpearena.gym.SharpeArenaEnv",
) -> ParityRecord:
    """Replay the fixture against an adapter exposing the Gymnasium scalar API.

    ``make_env`` receives the fixture and returns an environment with ``reset``,
    ``step`` and ``symbols``. A construction or import failure is raised as
    :class:`ParityUnavailable`; it never becomes an empty record that compares equal.
    """
    import numpy as np

    builder = _default_adapter if make_env is None else make_env
    try:
        env = builder(fixture)
    except Exception as exc:
        raise ParityUnavailable(f"{source} could not be constructed: {exc}") from exc

    spec_hash = None
    try:
        from .. import sharpearena_py

        spec_hash = engine_spec_hash(sharpearena_py)
    except Exception as exc:  # pragma: no cover - exercised only without the binding
        raise ParityUnavailable(f"{source} has no native engine underneath: {exc}") from exc

    obs, _info = env.reset()
    symbols = list(env.symbols)

    steps: list[StepRecord] = []
    for index, action in enumerate(fixture.actions):
        reason = fixture.refusal_reason(action)
        try:
            obs, reward, terminated, truncated, info = env.step(
                np.asarray(action, dtype=np.float32)
            )
        except Exception as exc:
            code = reason if reason is not None else f"engine:{_refusal_code(exc)}"
            steps.append(_refused_step(index, action, code))
            break
        if reason is not None:
            # The adapter took an action its own declared bounds reject. Recorded as a
            # distinct outcome so a silent clip or projection cannot read as agreement.
            steps.append(
                StepRecord(
                    step=index,
                    action=action,
                    outcome=f"accepted:{reason}",
                    reward=float(reward),
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    nav=float(info.get("nav", 1.0)),
                    observation=None,
                )
            )
            break
        nav = float(info.get("nav", 1.0))
        steps.append(
            StepRecord(
                step=index,
                action=action,
                outcome="ok",
                reward=float(reward),
                terminated=bool(terminated),
                truncated=bool(truncated),
                nav=nav,
                observation=observation_digest(
                    obs["closes"], obs["positions"], float(obs["cash"][0])
                ),
            )
        )
        if terminated or truncated:
            break

    effective = {}
    native = getattr(env, "_env", None)
    if native is not None and getattr(native, "effective_config", None) is not None:
        effective = json.loads(native.effective_config)

    return ParityRecord(
        fixture=fixture.name,
        source=source,
        spec_hash=spec_hash,
        effective_config=effective,
        steps=tuple(steps),
        counts=_counts(steps, len(fixture.actions)),
        terminal=_terminal_label(steps),
    )


_STEP_FIELDS = ("outcome", "reward", "terminated", "truncated", "nav", "observation")
_CONFIG_FIELDS = ("n_symbols", "n_bars", "window_start", "window_end", "dataset_fnv1a64")


def compare(
    reference: ParityRecord,
    candidate: ParityRecord,
    *,
    exactness: ExactnessDomain = ENGINE_EXACTNESS,
) -> ParityReport:
    """Compare two records of the same fixture under a declared exactness domain.

    Raises :class:`ParityUnavailable` when the comparison itself is not meaningful:
    different fixtures, a missing or disagreeing spec hash, or no step on either side.
    Those are the states in which an ``ok`` report would be a false claim.
    """
    if reference.fixture != candidate.fixture:
        raise ParityUnavailable(
            f"records are for different fixtures: {reference.fixture} and {candidate.fixture}"
        )
    for record in (reference, candidate):
        if record.spec_hash is None:
            raise ParityUnavailable(f"{record.source} reports no engine spec hash")
    if reference.spec_hash != candidate.spec_hash:
        raise ParityUnavailable(
            f"{reference.source} ran engine spec {reference.spec_hash} and "
            f"{candidate.source} ran {candidate.spec_hash}"
        )
    if not reference.steps and not candidate.steps:
        raise ParityUnavailable(
            f"fixture {reference.fixture} produced no steps on either path; there is nothing to compare"
        )

    mismatches: list[str] = []
    if len(reference.steps) != len(candidate.steps):
        mismatches.append(
            f"step count: engine {len(reference.steps)}, {candidate.source} {len(candidate.steps)}"
        )
    if reference.terminal != candidate.terminal:
        mismatches.append(
            f"terminal: engine {reference.terminal}, {candidate.source} {candidate.terminal}"
        )
    if reference.counts != candidate.counts:
        mismatches.append(
            f"attempt counts: engine {reference.counts.as_record()}, "
            f"{candidate.source} {candidate.counts.as_record()}"
        )
    for field in _CONFIG_FIELDS:
        if field in reference.effective_config and field in candidate.effective_config:
            if reference.effective_config[field] != candidate.effective_config[field]:
                mismatches.append(
                    f"effective config {field}: engine {reference.effective_config[field]}, "
                    f"{candidate.source} {candidate.effective_config[field]}"
                )

    paired = list(zip(reference.steps, candidate.steps))
    for left, right in paired:
        for field in _STEP_FIELDS:
            expected = getattr(left, field)
            actual = getattr(right, field)
            if exactness.is_exact(_domain_field(field)):
                if expected != actual:
                    mismatches.append(f"step {left.step} {field}: engine {expected!r}, got {actual!r}")
                continue
            tolerance = exactness.tolerance_for(_domain_field(field))
            if expected is None or actual is None:
                if expected is not actual:
                    mismatches.append(f"step {left.step} {field}: engine {expected!r}, got {actual!r}")
            elif abs(float(expected) - float(actual)) > tolerance:
                mismatches.append(
                    f"step {left.step} {field}: engine {expected!r}, got {actual!r} "
                    f"(tolerance {tolerance})"
                )

    return ParityReport(
        fixture=reference.fixture,
        reference=reference.source,
        candidate=candidate.source,
        steps_compared=len(paired),
        exactness=exactness,
        mismatches=tuple(mismatches),
    )


def _domain_field(field: str) -> str:
    """Map a step field onto the exactness domain's vocabulary."""
    return "observation" if field == "observation" else field


def check_adapter_parity(
    fixture: ParityFixture,
    make_env: Optional[Callable[[ParityFixture], Any]] = None,
    *,
    source: str = "sharpearena.gym.SharpeArenaEnv",
    exactness: ExactnessDomain = ENGINE_EXACTNESS,
) -> ParityReport:
    """Run both paths on one fixture and compare them."""
    return compare(
        engine_rollout(fixture),
        adapter_rollout(fixture, make_env, source=source),
        exactness=exactness,
    )


def deterministic_actions(
    *,
    n_symbols: int,
    n_steps: int,
    seed: int,
    max_weight: float = 1.0,
    allow_short: bool = True,
) -> tuple[tuple[float, ...], ...]:
    """A reproducible action sequence, generated once and then stored literally.

    Generated from its own ``numpy`` generator so the sequence does not depend on the
    environment's RNG or on how many times either path drew from it. Values are
    quantised to ``float32`` because the Gymnasium ``Box`` is ``float32``: without that,
    the adapter would send the engine a rounded weight while the reference path sent the
    unrounded one, and the resulting difference would look like a mapping bug.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    low = -max_weight if allow_short else 0.0
    raw = rng.uniform(low, max_weight, size=(n_steps, n_symbols))
    scale = max(1.0, float(np.abs(raw).sum(axis=1).max()) / max_weight)
    quantised = (raw / scale).astype(np.float32)
    return tuple(tuple(float(value) for value in row) for row in quantised)


#: The fixtures every adapter ticket runs. ``complete`` reaches the window end,
#: ``incomplete`` stops before it, and ``refused`` hands over an action outside the
#: declared bounds, which both paths must decline.
CORE_FIXTURES = (
    ParityFixture(
        name="calm-complete",
        n_symbols=3,
        n_days=24,
        seed=7,
        actions=deterministic_actions(n_symbols=3, n_steps=64, seed=7),
        expect="complete",
    ),
    ParityFixture(
        name="calm-incomplete",
        n_symbols=3,
        n_days=40,
        seed=11,
        actions=deterministic_actions(n_symbols=3, n_steps=5, seed=11),
        expect="incomplete",
    ),
    ParityFixture(
        name="calm-refused-out-of-bounds",
        n_symbols=3,
        n_days=40,
        seed=13,
        max_weight=0.5,
        # Literal weights are float32-exact, for the reason deterministic_actions quantises.
        actions=(
            (0.125, 0.25, -0.125),
            (5.0, 0.0, 0.0),
        ),
        expect="refused",
    ),
)


__all__ = [
    "CORE_FIXTURES",
    "ParityFixture",
    "ParityMismatch",
    "ParityRecord",
    "ParityReport",
    "ParityUnavailable",
    "StepRecord",
    "adapter_rollout",
    "check_adapter_parity",
    "compare",
    "decision_json",
    "deterministic_actions",
    "engine_rollout",
    "observation_digest",
]
